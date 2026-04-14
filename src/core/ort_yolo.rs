/// ORT-based object detection backend — YOLOv8n.
///
/// Model contract (opset 17, Ultralytics YOLOv8n):
///   Input  "images"  : [1, 3, 640, 640] f32, RGB values 0.0–1.0
///   Output "output0" : [1, 84, 8400]   f32
///     rows 0-3   : cx, cy, w, h  (640-pixel space)
///     rows 4-83  : 80 COCO class scores (raw logits, no sigmoid applied)
///     8400 = 80×80 + 40×40 + 20×20 grid cells
///
/// No built-in NMS — implemented manually here.
use anyhow::{Context, Result};
use image::{DynamicImage, ImageBuffer, Rgb};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::core::yolo::{BoundingBox, Detection, YoloResults};
use crate::tensor_pool::{TensorPool, TensorShape};

/// Module-level output buffer pool for YOLO inference.
/// Pools the [84 × 8400] f32 output tensor to avoid ~2.8 MB allocation per request.
static OUTPUT_POOL: OnceLock<TensorPool> = OnceLock::new();

fn output_pool() -> &'static TensorPool {
    OUTPUT_POOL.get_or_init(|| TensorPool::new(16))
}

pub const MODEL_INPUT_SIZE: u32 = 640;
const NUM_CLASSES: usize = 80;
const NUM_ANCHORS: usize = 8400;

struct BoxCandidate {
    x1: f32, y1: f32, x2: f32, y2: f32,
    score: f32, class_id: usize,
}

pub struct OrtYoloDetector {
    session: Mutex<Session>,
    class_names: Vec<String>,
    conf_threshold: f32,
    iou_threshold: f32,
}

impl OrtYoloDetector {
    pub fn new(model_path: &Path, class_names: Vec<String>) -> Result<Self> {
        let physical_cpus = num_cpus::get_physical().max(1);
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(physical_cpus)?
            .with_inter_threads(1)?
            .with_memory_pattern(true)?;

        #[cfg(target_os = "macos")]
        {
            builder = builder.with_execution_providers([
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
                ort::execution_providers::CPUExecutionProvider::default().build(),
            ])?;
        }
        #[cfg(not(target_os = "macos"))]
        {
            builder = builder.with_execution_providers([
                ort::execution_providers::CPUExecutionProvider::default().build(),
            ])?;
        }

        let session = builder
            .commit_from_file(model_path)
            .with_context(|| format!("loading YOLOv8n ONNX model from {:?}", model_path))?;

        tracing::info!(model = ?model_path, "OrtYoloDetector (YOLOv8n) loaded");

        Ok(Self {
            session: Mutex::new(session),
            class_names,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
        })
    }

    pub fn set_conf_threshold(&mut self, t: f32) { self.conf_threshold = t.clamp(0.0, 1.0); }
    pub fn set_iou_threshold(&mut self, t: f32)  { self.iou_threshold  = t.clamp(0.0, 1.0); }

    pub fn detect_bytes(&self, image_bytes: &[u8]) -> Result<YoloResults> {
        let img = image::load_from_memory(image_bytes).context("decode image")?;
        self.run(&img)
    }

    #[allow(dead_code)]
    pub fn detect_file(&self, path: &Path) -> Result<YoloResults> {
        let img = image::open(path).context("open image file")?;
        self.run(&img)
    }

    fn run(&self, img: &DynamicImage) -> Result<YoloResults> {
        let t_start = Instant::now();

        // ── Preprocess ────────────────────────────────────────────────────────
        let t_pre = Instant::now();
        let orig_w = img.width() as f32;
        let orig_h = img.height() as f32;
        let size = MODEL_INPUT_SIZE;

        let resized = img.resize_exact(size, size, image::imageops::FilterType::Lanczos3);
        let rgb = resized.to_rgb8();
        let input_data = Self::to_chw_f32_norm(&rgb);
        let preprocessing_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

        // ── Inference ─────────────────────────────────────────────────────────
        let t_infer = Instant::now();
        let image_tensor = Tensor::<f32>::from_array((
            [1usize, 3, size as usize, size as usize],
            input_data,
        ))?;

        let mut sess = self.session.lock().unwrap();
        let outputs = sess.run(ort::inputs!["images" => image_tensor])?;
        let inference_ms = t_infer.elapsed().as_secs_f64() * 1000.0;

        // ── Postprocess ───────────────────────────────────────────────────────
        let t_post = Instant::now();

        // output0: [1, 84, 8400]
        let (_out_shape, out_view) = outputs[0].try_extract_tensor::<f32>()?;
        let raw_len = out_view.len();
        let raw_shape = TensorShape::new(vec![raw_len]);
        let mut raw = output_pool().acquire(raw_shape.clone());
        raw.clear();
        raw.extend(out_view.iter().copied());
        // raw is row-major [84][8400] → index: row * 8400 + anchor

        let scale_x = orig_w / size as f32;
        let scale_y = orig_h / size as f32;

        // Collect candidates that pass confidence threshold.
        // Pre-allocate for typical detection count; Vec grows if needed.
        let mut candidates: Vec<(f32, f32, f32, f32, f32, usize)> = Vec::with_capacity(512); // cx,cy,w,h,score,class_id
        for a in 0..NUM_ANCHORS {
            // Find best class
            let mut best_class = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for c in 0..NUM_CLASSES {
                let s = raw[(4 + c) * NUM_ANCHORS + a];
                if s > best_score {
                    best_score = s;
                    best_class = c;
                }
            }

            // sigmoid of raw score
            let confidence = 1.0 / (1.0 + (-best_score).exp());
            if confidence < self.conf_threshold {
                continue;
            }

            let cx = raw[a];
            let cy = raw[NUM_ANCHORS + a];
            let bw = raw[2 * NUM_ANCHORS + a];
            let bh = raw[3 * NUM_ANCHORS + a];

            candidates.push((cx, cy, bw, bh, confidence, best_class));
        }

        // Return the pooled raw buffer now — candidates already extracted all needed values.
        output_pool().release(raw_shape, raw);

        // Convert cxcywh → x1y1x2y2, then scale to original image size
        let mut boxes: Vec<BoxCandidate> = candidates.into_iter().map(|(cx, cy, bw, bh, score, class_id)| {
            let half_w = bw / 2.0;
            let half_h = bh / 2.0;
            BoxCandidate {
                x1: (cx - half_w) * scale_x,
                y1: (cy - half_h) * scale_y,
                x2: (cx + half_w) * scale_x,
                y2: (cy + half_h) * scale_y,
                score,
                class_id,
            }
        }).collect();

        // NMS per class
        boxes.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let mut kept = vec![true; boxes.len()];
        for i in 0..boxes.len() {
            if !kept[i] { continue; }
            for j in (i + 1)..boxes.len() {
                if !kept[j] { continue; }
                if boxes[i].class_id != boxes[j].class_id { continue; }
                if Self::iou(&boxes[i], &boxes[j]) > self.iou_threshold {
                    kept[j] = false;
                }
            }
        }

        let detections: Vec<Detection> = boxes.into_iter().enumerate()
            .filter(|(idx, _)| kept[*idx])
            .map(|(_, b)| Detection {
                class_id: b.class_id,
                class_name: self.class_names.get(b.class_id)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{}", b.class_id)),
                confidence: b.score,
                bbox: BoundingBox {
                    x1: b.x1.max(0.0),
                    y1: b.y1.max(0.0),
                    x2: b.x2.min(orig_w),
                    y2: b.y2.min(orig_h),
                },
            })
            .collect();

        let postprocessing_ms = t_post.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        Ok(YoloResults {
            detections,
            inference_time_ms: inference_ms,
            preprocessing_time_ms: preprocessing_ms,
            postprocessing_time_ms: postprocessing_ms,
            total_time_ms: total_ms,
        })
    }

    /// Convert RGB image to CHW float32, normalised to [0, 1].
    fn to_chw_f32_norm(img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Vec<f32> {
        let (w, h) = img.dimensions();
        let (w, h) = (w as usize, h as usize);
        let mut data = vec![0f32; 3 * h * w];
        for (x, y, px) in img.enumerate_pixels() {
            let (x, y) = (x as usize, y as usize);
            data[y * w + x] = px[0] as f32 / 255.0;
            data[h * w + y * w + x] = px[1] as f32 / 255.0;
            data[2 * h * w + y * w + x] = px[2] as f32 / 255.0;
        }
        data
    }

    fn iou(a: &BoxCandidate, b: &BoxCandidate) -> f32 {
        let ix1 = a.x1.max(b.x1);
        let iy1 = a.y1.max(b.y1);
        let ix2 = a.x2.min(b.x2);
        let iy2 = a.y2.min(b.y2);

        let inter_w = (ix2 - ix1).max(0.0);
        let inter_h = (iy2 - iy1).max(0.0);
        let inter = inter_w * inter_h;
        if inter == 0.0 { return 0.0; }

        let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        inter / (area_a + area_b - inter)
    }
}

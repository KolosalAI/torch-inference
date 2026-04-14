// YOLO Object Detection Module
// Supports YOLOv5, YOLOv8, YOLOv10, YOLOv11, and YOLOv12 (YOLO11)

#![allow(dead_code)]
#[cfg(feature = "torch")]
use anyhow::Context;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg(feature = "torch")]
use tch::{vision, Device, Kind, Tensor};

#[cfg(not(feature = "torch"))]
type Device = ();

#[cfg(feature = "torch")]
use crate::torch_optimization::{
    OptimizedTorchModel, TorchOptimizationConfig, TorchOptimizationConfigBuilder,
};

#[cfg(feature = "torch")]
use crate::core::model_cache::cache_key;
use crate::core::model_cache::{CacheStats, ModelCache};

/// YOLO Model Version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum YoloVersion {
    V5,
    V8,
    V10,
    V11, // Also known as YOLO11
    V12, // Latest YOLO (December 2024)
}

impl YoloVersion {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "v5" | "yolov5" => Some(Self::V5),
            "v8" | "yolov8" => Some(Self::V8),
            "v10" | "yolov10" => Some(Self::V10),
            "v11" | "yolov11" | "yolo11" => Some(Self::V11),
            "v12" | "yolov12" | "yolo12" => Some(Self::V12),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::V5 => "YOLOv5",
            Self::V8 => "YOLOv8",
            Self::V10 => "YOLOv10",
            Self::V11 => "YOLOv11",
            Self::V12 => "YOLOv12",
        }
    }
}

/// YOLO Model Size
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum YoloSize {
    Nano,   // n - fastest, least accurate
    Small,  // s
    Medium, // m
    Large,  // l
    XLarge, // x - slowest, most accurate
}

impl YoloSize {
    pub fn suffix(&self) -> &'static str {
        match self {
            Self::Nano => "n",
            Self::Small => "s",
            Self::Medium => "m",
            Self::Large => "l",
            Self::XLarge => "x",
        }
    }

    pub fn from_suffix(s: &str) -> Option<Self> {
        match s {
            "n" => Some(Self::Nano),
            "s" => Some(Self::Small),
            "m" => Some(Self::Medium),
            "l" => Some(Self::Large),
            "x" => Some(Self::XLarge),
            _ => None,
        }
    }
}

/// Detection result for a single object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BoundingBox {
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    pub fn center_x(&self) -> f32 {
        (self.x1 + self.x2) / 2.0
    }

    pub fn center_y(&self) -> f32 {
        (self.y1 + self.y2) / 2.0
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let x1 = self.x1.max(other.x1);
        let y1 = self.y1.max(other.y1);
        let x2 = self.x2.min(other.x2);
        let y2 = self.y2.min(other.y2);

        if x2 < x1 || y2 < y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let union = self.area() + other.area() - intersection;

        intersection / union
    }
}

/// YOLO detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YoloResults {
    pub detections: Vec<Detection>,
    pub inference_time_ms: f64,
    pub preprocessing_time_ms: f64,
    pub postprocessing_time_ms: f64,
    pub total_time_ms: f64,
}

/// YOLO Object Detector
pub struct YoloDetector {
    #[cfg(feature = "torch")]
    optimizer: OptimizedTorchModel,

    #[cfg(not(feature = "torch"))]
    _optimizer: (),

    version: YoloVersion,
    size: YoloSize,
    class_names: Vec<String>,
    input_size: (i64, i64),
    conf_threshold: f32,
    iou_threshold: f32,
    cache: ModelCache,
    model_id: String,
}

impl YoloDetector {
    /// Create a new YOLO detector
    #[cfg(feature = "torch")]
    pub fn new(
        model_path: &Path,
        version: YoloVersion,
        size: YoloSize,
        class_names: Vec<String>,
        device: Option<Device>,
        opt_config: Option<TorchOptimizationConfig>,
    ) -> Result<Self> {
        let device = device.unwrap_or(Device::Cpu);
        let model_id = model_path.to_string_lossy().to_string();

        let config = opt_config.unwrap_or_else(|| {
            let is_accel = device != Device::Cpu;
            TorchOptimizationConfigBuilder::new()
                .device(device)
                .cudnn_benchmark(true)
                .warmup_iterations(3)
                .autocast(is_accel)
                .fp16(is_accel)
                .warmup_shape(vec![1, 3, 640, 640])
                .build()
        });

        log::info!(
            "Loading {} ({}) model from {:?}",
            version.as_str(),
            size.suffix(),
            model_path
        );
        let model_path_str = model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?;
        let mut optimizer = OptimizedTorchModel::new(model_path_str, config)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        optimizer.warmup().map_err(|e| anyhow::anyhow!("{}", e))?;

        Ok(Self {
            optimizer,
            version,
            size,
            class_names,
            input_size: (640, 640),
            conf_threshold: 0.25,
            iou_threshold: 0.45,
            cache: ModelCache::new(128),
            model_id,
        })
    }

    #[cfg(not(feature = "torch"))]
    pub fn new(
        _model_path: &Path,
        _version: YoloVersion,
        _size: YoloSize,
        _class_names: Vec<String>,
        _device: Option<()>,
        _opt_config: Option<()>,
    ) -> Result<Self> {
        bail!("PyTorch feature not enabled. Compile with --features torch");
    }

    /// Set confidence threshold
    pub fn set_conf_threshold(&mut self, threshold: f32) {
        self.conf_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Set IoU threshold for NMS
    pub fn set_iou_threshold(&mut self, threshold: f32) {
        self.iou_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Set custom input size
    pub fn set_input_size(&mut self, width: i64, height: i64) {
        self.input_size = (width, height);
    }

    /// Preprocess image for detection
    #[cfg(feature = "torch")]
    pub fn preprocess_image(&self, image_path: &Path) -> Result<Tensor> {
        log::debug!("Preprocessing image: {:?}", image_path);

        // Load image
        let image = vision::image::load(image_path).context("Failed to load image")?;

        // Resize to model input size
        let resized = vision::image::resize(&image, self.input_size.0, self.input_size.1)?;

        // Convert to float and normalize to [0, 1]
        let mut tensor = resized.to_kind(Kind::Float) / 255.0;

        // Add batch dimension
        tensor = tensor.unsqueeze(0);

        Ok(tensor.to_device(self.optimizer.device()))
    }

    #[cfg(not(feature = "torch"))]
    pub fn preprocess_image(&self, _image_path: &Path) -> Result<()> {
        bail!("PyTorch feature not enabled");
    }

    /// Perform object detection
    #[cfg(feature = "torch")]
    pub fn detect(&self, image_path: &Path) -> Result<YoloResults> {
        let start = std::time::Instant::now();

        // Preprocess
        let preprocess_start = std::time::Instant::now();
        let input = self.preprocess_image(image_path)?;
        let preprocessing_time_ms = preprocess_start.elapsed().as_secs_f64() * 1000.0;

        // Inference
        let inference_start = std::time::Instant::now();
        let output = self
            .optimizer
            .infer(&input)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let inference_time_ms = inference_start.elapsed().as_secs_f64() * 1000.0;

        // Postprocess
        let postprocess_start = std::time::Instant::now();
        let detections = self.postprocess(output)?;
        let postprocessing_time_ms = postprocess_start.elapsed().as_secs_f64() * 1000.0;

        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(YoloResults {
            detections,
            inference_time_ms,
            preprocessing_time_ms,
            postprocessing_time_ms,
            total_time_ms,
        })
    }

    #[cfg(not(feature = "torch"))]
    pub fn detect(&self, _image_path: &Path) -> Result<YoloResults> {
        bail!("PyTorch feature not enabled");
    }

    /// Detect objects from raw image bytes. Result is cached by (model_id, bytes, conf+iou thresholds).
    #[cfg(feature = "torch")]
    pub fn detect_bytes(&self, image_bytes: &[u8]) -> Result<YoloResults> {
        let mut params = Vec::with_capacity(8);
        params.extend_from_slice(&self.conf_threshold.to_le_bytes());
        params.extend_from_slice(&self.iou_threshold.to_le_bytes());
        let key = cache_key(&self.model_id, image_bytes, &params);

        self.cache.get_or_run(key, || {
            // Write temp file for image loading.
            let temp_path = std::env::temp_dir().join(format!("yolo_{}.jpg", uuid::Uuid::new_v4()));
            std::fs::write(&temp_path, image_bytes)?;
            let result = self.detect(&temp_path);
            let _ = std::fs::remove_file(&temp_path);
            result
        })
    }

    #[cfg(not(feature = "torch"))]
    pub fn detect_bytes(&self, _image_bytes: &[u8]) -> Result<YoloResults> {
        bail!("PyTorch feature not enabled");
    }

    /// Postprocess YOLO output
    #[cfg(feature = "torch")]
    fn postprocess(&self, output: Tensor) -> Result<Vec<Detection>> {
        let mut detections = Vec::new();

        // Convert to CPU for processing
        let output = output.to_device(Device::Cpu);

        // Different postprocessing based on YOLO version
        match self.version {
            YoloVersion::V5 => self.postprocess_v5(output, &mut detections)?,
            YoloVersion::V8 => self.postprocess_v8(output, &mut detections)?,
            YoloVersion::V10 => self.postprocess_v10(output, &mut detections)?,
            YoloVersion::V11 => self.postprocess_v11(output, &mut detections)?,
            YoloVersion::V12 => self.postprocess_v12(output, &mut detections)?,
        }

        // Apply NMS
        let detections = self.non_maximum_suppression(detections);

        Ok(detections)
    }

    /// Postprocess YOLOv5 output
    #[cfg(feature = "torch")]
    fn postprocess_v5(&self, output: Tensor, detections: &mut Vec<Detection>) -> Result<()> {
        // YOLOv5 output shape: [batch, num_predictions, 85] (for 80 classes)
        // Each prediction: [x, y, w, h, objectness, class_0_prob, ..., class_79_prob]

        let output_size = output.size();
        let num_predictions = output_size[1];

        for i in 0..num_predictions {
            let prediction = output.get(0).get(i);

            let objectness: f32 = prediction.double_value(&[4]) as f32;

            if objectness < self.conf_threshold {
                continue;
            }

            // Get class probabilities
            let class_probs = prediction.narrow(0, 5, self.class_names.len() as i64);
            let (max_prob, class_id) = class_probs.max_dim(0, false);
            let max_prob: f32 = max_prob.double_value(&[]) as f32;
            let class_id: usize = class_id.int64_value(&[]) as usize;

            let confidence = objectness * max_prob;

            if confidence < self.conf_threshold {
                continue;
            }

            // Get bounding box
            let x: f32 = prediction.double_value(&[0]) as f32;
            let y: f32 = prediction.double_value(&[1]) as f32;
            let w: f32 = prediction.double_value(&[2]) as f32;
            let h: f32 = prediction.double_value(&[3]) as f32;

            let bbox = BoundingBox {
                x1: x - w / 2.0,
                y1: y - h / 2.0,
                x2: x + w / 2.0,
                y2: y + h / 2.0,
            };

            detections.push(Detection {
                class_id,
                class_name: self
                    .class_names
                    .get(class_id)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{}", class_id)),
                confidence,
                bbox,
            });
        }

        Ok(())
    }

    /// Postprocess YOLOv8 output
    #[cfg(feature = "torch")]
    fn postprocess_v8(&self, output: Tensor, detections: &mut Vec<Detection>) -> Result<()> {
        // YOLOv8 output shape: [batch, 84, 8400] (for 80 classes)
        // Transpose to [batch, 8400, 84]
        let output = output.transpose(1, 2);

        let output_size = output.size();
        let num_predictions = output_size[1];

        for i in 0..num_predictions {
            let prediction = output.get(0).get(i);

            // First 4 elements are bbox coordinates
            let x: f32 = prediction.double_value(&[0]) as f32;
            let y: f32 = prediction.double_value(&[1]) as f32;
            let w: f32 = prediction.double_value(&[2]) as f32;
            let h: f32 = prediction.double_value(&[3]) as f32;

            // Remaining elements are class probabilities
            let class_probs = prediction.narrow(0, 4, self.class_names.len() as i64);
            let (max_prob, class_id) = class_probs.max_dim(0, false);
            let confidence: f32 = max_prob.double_value(&[]) as f32;
            let class_id: usize = class_id.int64_value(&[]) as usize;

            if confidence < self.conf_threshold {
                continue;
            }

            let bbox = BoundingBox {
                x1: x - w / 2.0,
                y1: y - h / 2.0,
                x2: x + w / 2.0,
                y2: y + h / 2.0,
            };

            detections.push(Detection {
                class_id,
                class_name: self
                    .class_names
                    .get(class_id)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{}", class_id)),
                confidence,
                bbox,
            });
        }

        Ok(())
    }

    /// Postprocess YOLOv10 output
    #[cfg(feature = "torch")]
    fn postprocess_v10(&self, output: Tensor, detections: &mut Vec<Detection>) -> Result<()> {
        // YOLOv10 uses similar format to YOLOv8 but with improved architecture
        self.postprocess_v8(output, detections)
    }

    /// Postprocess YOLOv11 output
    #[cfg(feature = "torch")]
    fn postprocess_v11(&self, output: Tensor, detections: &mut Vec<Detection>) -> Result<()> {
        // YOLOv11 (YOLO11) maintains YOLOv8 output format with architecture improvements
        self.postprocess_v8(output, detections)
    }

    /// Postprocess YOLOv12 output (Latest - December 2024)
    #[cfg(feature = "torch")]
    fn postprocess_v12(&self, output: Tensor, detections: &mut Vec<Detection>) -> Result<()> {
        // YOLOv12 uses enhanced output format with better feature extraction
        // Still compatible with YOLOv8 format
        self.postprocess_v8(output, detections)
    }

    /// Non-Maximum Suppression
    fn non_maximum_suppression(&self, mut detections: Vec<Detection>) -> Vec<Detection> {
        if detections.is_empty() {
            return detections;
        }

        // Sort by confidence (descending)
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut keep = Vec::new();

        while !detections.is_empty() {
            let current = detections.remove(0);
            keep.push(current.clone());

            detections.retain(|det| {
                // Keep detections of different classes
                if det.class_id != current.class_id {
                    return true;
                }

                // Remove detections with high IoU overlap
                det.bbox.iou(&current.bbox) < self.iou_threshold
            });
        }

        keep
    }

    /// Return cache hit/miss statistics.
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    #[cfg(all(test, not(feature = "torch")))]
    pub fn new_stub(version: YoloVersion, size: YoloSize, class_names: Vec<String>) -> Self {
        Self {
            #[cfg(not(feature = "torch"))]
            _optimizer: (),
            version,
            size,
            class_names,
            input_size: (640, 640),
            conf_threshold: 0.25,
            iou_threshold: 0.45,
            cache: ModelCache::new(128),
            model_id: "stub".to_string(),
        }
    }

    /// Get model information
    pub fn info(&self) -> String {
        format!(
            "{} ({}) - {} classes, input: {}x{}, conf: {}, iou: {}",
            self.version.as_str(),
            self.size.suffix(),
            self.class_names.len(),
            self.input_size.0,
            self.input_size.1,
            self.conf_threshold,
            self.iou_threshold
        )
    }
}

/// Load COCO class names (80 classes)
pub fn load_coco_names() -> Vec<String> {
    vec![
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yolo_version_parsing() {
        assert_eq!(YoloVersion::from_str("v5"), Some(YoloVersion::V5));
        assert_eq!(YoloVersion::from_str("YOLOv8"), Some(YoloVersion::V8));
        assert_eq!(YoloVersion::from_str("yolo11"), Some(YoloVersion::V11));
        assert_eq!(YoloVersion::from_str("v12"), Some(YoloVersion::V12));
    }

    #[test]
    fn test_bbox_iou() {
        let bbox1 = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
        };
        let bbox2 = BoundingBox {
            x1: 5.0,
            y1: 5.0,
            x2: 15.0,
            y2: 15.0,
        };

        let iou = bbox1.iou(&bbox2);
        assert!(iou > 0.0 && iou < 1.0);
    }

    #[test]
    fn test_coco_names() {
        let names = load_coco_names();
        assert_eq!(names.len(), 80);
        assert_eq!(names[0], "person");
        assert_eq!(names[79], "toothbrush");
    }

    // ─── YoloVersion ────────────────────────────────────────────────────────

    #[test]
    fn test_yolo_version_all_variants_from_str() {
        // Case-insensitive aliases
        assert_eq!(YoloVersion::from_str("yolov5"), Some(YoloVersion::V5));
        assert_eq!(YoloVersion::from_str("V5"), Some(YoloVersion::V5));
        assert_eq!(YoloVersion::from_str("v8"), Some(YoloVersion::V8));
        assert_eq!(YoloVersion::from_str("yolov8"), Some(YoloVersion::V8));
        assert_eq!(YoloVersion::from_str("v10"), Some(YoloVersion::V10));
        assert_eq!(YoloVersion::from_str("yolov10"), Some(YoloVersion::V10));
        assert_eq!(YoloVersion::from_str("v11"), Some(YoloVersion::V11));
        assert_eq!(YoloVersion::from_str("yolov11"), Some(YoloVersion::V11));
        assert_eq!(YoloVersion::from_str("yolo11"), Some(YoloVersion::V11));
        assert_eq!(YoloVersion::from_str("v12"), Some(YoloVersion::V12));
        assert_eq!(YoloVersion::from_str("yolov12"), Some(YoloVersion::V12));
        assert_eq!(YoloVersion::from_str("yolo12"), Some(YoloVersion::V12));
    }

    #[test]
    fn test_yolo_version_from_str_unknown_returns_none() {
        assert_eq!(YoloVersion::from_str(""), None);
        assert_eq!(YoloVersion::from_str("v99"), None);
        assert_eq!(YoloVersion::from_str("yolov3"), None);
        assert_eq!(YoloVersion::from_str("unknown"), None);
    }

    #[test]
    fn test_yolo_version_as_str() {
        assert_eq!(YoloVersion::V5.as_str(), "YOLOv5");
        assert_eq!(YoloVersion::V8.as_str(), "YOLOv8");
        assert_eq!(YoloVersion::V10.as_str(), "YOLOv10");
        assert_eq!(YoloVersion::V11.as_str(), "YOLOv11");
        assert_eq!(YoloVersion::V12.as_str(), "YOLOv12");
    }

    #[test]
    fn test_yolo_version_serde_roundtrip() {
        let v = YoloVersion::V12;
        let json = serde_json::to_string(&v).unwrap();
        let back: YoloVersion = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn test_yolo_version_equality() {
        assert_eq!(YoloVersion::V5, YoloVersion::V5);
        assert_ne!(YoloVersion::V5, YoloVersion::V8);
        assert_ne!(YoloVersion::V10, YoloVersion::V11);
    }

    // ─── YoloSize ───────────────────────────────────────────────────────────

    #[test]
    fn test_yolo_size_suffix() {
        assert_eq!(YoloSize::Nano.suffix(), "n");
        assert_eq!(YoloSize::Small.suffix(), "s");
        assert_eq!(YoloSize::Medium.suffix(), "m");
        assert_eq!(YoloSize::Large.suffix(), "l");
        assert_eq!(YoloSize::XLarge.suffix(), "x");
    }

    #[test]
    fn test_yolo_size_from_suffix_all_variants() {
        assert_eq!(YoloSize::from_suffix("n"), Some(YoloSize::Nano));
        assert_eq!(YoloSize::from_suffix("s"), Some(YoloSize::Small));
        assert_eq!(YoloSize::from_suffix("m"), Some(YoloSize::Medium));
        assert_eq!(YoloSize::from_suffix("l"), Some(YoloSize::Large));
        assert_eq!(YoloSize::from_suffix("x"), Some(YoloSize::XLarge));
    }

    #[test]
    fn test_yolo_size_from_suffix_unknown_returns_none() {
        assert_eq!(YoloSize::from_suffix(""), None);
        assert_eq!(YoloSize::from_suffix("X"), None);
        assert_eq!(YoloSize::from_suffix("xl"), None);
        assert_eq!(YoloSize::from_suffix("nano"), None);
    }

    #[test]
    fn test_yolo_size_serde_roundtrip() {
        for size in [
            YoloSize::Nano,
            YoloSize::Small,
            YoloSize::Medium,
            YoloSize::Large,
            YoloSize::XLarge,
        ] {
            let json = serde_json::to_string(&size).unwrap();
            let back: YoloSize = serde_json::from_str(&json).unwrap();
            assert_eq!(size, back);
        }
    }

    #[test]
    fn test_yolo_size_equality() {
        assert_eq!(YoloSize::Nano, YoloSize::Nano);
        assert_ne!(YoloSize::Nano, YoloSize::XLarge);
    }

    // ─── BoundingBox ────────────────────────────────────────────────────────

    #[test]
    fn test_bbox_dimensions() {
        let bbox = BoundingBox {
            x1: 10.0,
            y1: 20.0,
            x2: 30.0,
            y2: 50.0,
        };
        assert!((bbox.width() - 20.0).abs() < 1e-6);
        assert!((bbox.height() - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_center() {
        let bbox = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 20.0,
        };
        assert!((bbox.center_x() - 5.0).abs() < 1e-6);
        assert!((bbox.center_y() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_area() {
        let bbox = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 4.0,
            y2: 5.0,
        };
        assert!((bbox.area() - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_area_zero_width() {
        let bbox = BoundingBox {
            x1: 3.0,
            y1: 0.0,
            x2: 3.0,
            y2: 5.0,
        };
        assert!((bbox.area() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_iou_no_overlap() {
        // Completely separate boxes should have IoU = 0
        let bbox1 = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
        };
        let bbox2 = BoundingBox {
            x1: 2.0,
            y1: 2.0,
            x2: 3.0,
            y2: 3.0,
        };
        assert!((bbox1.iou(&bbox2) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_iou_perfect_overlap() {
        // Identical boxes should have IoU = 1
        let bbox = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
        };
        assert!((bbox.iou(&bbox) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_iou_partial_overlap() {
        let bbox1 = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 2.0,
            y2: 2.0,
        };
        let bbox2 = BoundingBox {
            x1: 1.0,
            y1: 1.0,
            x2: 3.0,
            y2: 3.0,
        };
        let iou = bbox1.iou(&bbox2);
        // Intersection = 1x1 = 1; Union = 4 + 4 - 1 = 7
        let expected = 1.0 / 7.0;
        assert!(
            (iou - expected).abs() < 1e-5,
            "Expected ~{:.4}, got {:.4}",
            expected,
            iou
        );
    }

    #[test]
    fn test_bbox_iou_touching_edge_returns_zero() {
        // Boxes that share only an edge have 0 area intersection
        let bbox1 = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
        };
        let bbox2 = BoundingBox {
            x1: 1.0,
            y1: 0.0,
            x2: 2.0,
            y2: 1.0,
        };
        assert!((bbox1.iou(&bbox2) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_serde_roundtrip() {
        let bbox = BoundingBox {
            x1: 1.5,
            y1: 2.5,
            x2: 10.0,
            y2: 20.0,
        };
        let json = serde_json::to_string(&bbox).unwrap();
        let back: BoundingBox = serde_json::from_str(&json).unwrap();
        assert!((bbox.x1 - back.x1).abs() < 1e-6);
        assert!((bbox.y2 - back.y2).abs() < 1e-6);
    }

    // ─── Detection ──────────────────────────────────────────────────────────

    #[test]
    fn test_detection_construction_and_serde() {
        let det = Detection {
            class_id: 0,
            class_name: "person".to_string(),
            confidence: 0.95,
            bbox: BoundingBox {
                x1: 0.0,
                y1: 0.0,
                x2: 100.0,
                y2: 200.0,
            },
        };
        let json = serde_json::to_string(&det).unwrap();
        let back: Detection = serde_json::from_str(&json).unwrap();
        assert_eq!(back.class_id, 0);
        assert_eq!(back.class_name, "person");
        assert!((back.confidence - 0.95).abs() < 1e-5);
    }

    // ─── YoloResults ────────────────────────────────────────────────────────

    #[test]
    fn test_yolo_results_serde_roundtrip() {
        let results = YoloResults {
            detections: vec![Detection {
                class_id: 1,
                class_name: "car".to_string(),
                confidence: 0.8,
                bbox: BoundingBox {
                    x1: 10.0,
                    y1: 10.0,
                    x2: 50.0,
                    y2: 50.0,
                },
            }],
            inference_time_ms: 12.5,
            preprocessing_time_ms: 3.0,
            postprocessing_time_ms: 1.5,
            total_time_ms: 17.0,
        };

        let json = serde_json::to_string(&results).unwrap();
        let back: YoloResults = serde_json::from_str(&json).unwrap();
        assert_eq!(back.detections.len(), 1);
        assert_eq!(back.detections[0].class_name, "car");
        assert!((back.inference_time_ms - 12.5).abs() < 1e-6);
    }

    // ─── YoloDetector::new (no-torch path) ──────────────────────────────────

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_yolo_detector_new_errors_without_torch() {
        let result = YoloDetector::new(
            std::path::Path::new("/nonexistent/model.pt"),
            YoloVersion::V8,
            YoloSize::Nano,
            vec!["person".to_string()],
            None,
            None,
        );
        assert!(
            result.is_err(),
            "new() should error when torch feature is disabled"
        );
        if let Err(e) = result {
            let msg = format!("{}", e);
            assert!(
                msg.contains("PyTorch") || msg.contains("torch"),
                "Error should mention torch: {}",
                msg
            );
        }
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_yolo_cache_stats_initially_zero() {
        let detector =
            YoloDetector::new_stub(YoloVersion::V8, YoloSize::Nano, vec!["person".to_string()]);
        let stats = detector.cache_stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    // ─── NMS via YoloDetector helper ────────────────────────────────────────

    #[cfg(not(feature = "torch"))]
    fn make_detector() -> YoloDetector {
        YoloDetector::new_stub(YoloVersion::V8, YoloSize::Nano, load_coco_names())
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_nms_empty_input_returns_empty() {
        let detector = make_detector();
        let result = detector.non_maximum_suppression(vec![]);
        assert!(result.is_empty());
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_nms_single_detection_is_kept() {
        let detector = make_detector();
        let dets = vec![Detection {
            class_id: 0,
            class_name: "person".to_string(),
            confidence: 0.9,
            bbox: BoundingBox {
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
            },
        }];
        let result = detector.non_maximum_suppression(dets);
        assert_eq!(result.len(), 1);
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_nms_suppresses_high_iou_same_class() {
        let detector = make_detector();
        // Two nearly identical boxes — lower confidence should be suppressed.
        let dets = vec![
            Detection {
                class_id: 0,
                class_name: "person".to_string(),
                confidence: 0.9,
                bbox: BoundingBox {
                    x1: 0.0,
                    y1: 0.0,
                    x2: 1.0,
                    y2: 1.0,
                },
            },
            Detection {
                class_id: 0,
                class_name: "person".to_string(),
                confidence: 0.8,
                bbox: BoundingBox {
                    x1: 0.05,
                    y1: 0.05,
                    x2: 1.05,
                    y2: 1.05,
                },
            },
        ];
        let result = detector.non_maximum_suppression(dets);
        assert_eq!(result.len(), 1, "high-IoU duplicate should be suppressed");
        assert!((result[0].confidence - 0.9).abs() < 1e-5);
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_nms_keeps_different_class_same_location() {
        let detector = make_detector();
        // Same location but different class — both should be kept.
        let dets = vec![
            Detection {
                class_id: 0,
                class_name: "person".to_string(),
                confidence: 0.9,
                bbox: BoundingBox {
                    x1: 0.0,
                    y1: 0.0,
                    x2: 1.0,
                    y2: 1.0,
                },
            },
            Detection {
                class_id: 1,
                class_name: "bicycle".to_string(),
                confidence: 0.85,
                bbox: BoundingBox {
                    x1: 0.0,
                    y1: 0.0,
                    x2: 1.0,
                    y2: 1.0,
                },
            },
        ];
        let result = detector.non_maximum_suppression(dets);
        assert_eq!(
            result.len(),
            2,
            "different classes at same location should both be kept"
        );
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_nms_keeps_non_overlapping_same_class() {
        let detector = make_detector();
        // Two boxes that do not overlap — both should survive NMS.
        let dets = vec![
            Detection {
                class_id: 0,
                class_name: "person".to_string(),
                confidence: 0.9,
                bbox: BoundingBox {
                    x1: 0.0,
                    y1: 0.0,
                    x2: 1.0,
                    y2: 1.0,
                },
            },
            Detection {
                class_id: 0,
                class_name: "person".to_string(),
                confidence: 0.7,
                bbox: BoundingBox {
                    x1: 5.0,
                    y1: 5.0,
                    x2: 6.0,
                    y2: 6.0,
                },
            },
        ];
        let result = detector.non_maximum_suppression(dets);
        assert_eq!(result.len(), 2, "non-overlapping boxes should both be kept");
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_nms_output_sorted_by_confidence_descending() {
        let detector = make_detector();
        let dets = vec![
            Detection {
                class_id: 1,
                class_name: "bicycle".to_string(),
                confidence: 0.5,
                bbox: BoundingBox {
                    x1: 10.0,
                    y1: 10.0,
                    x2: 20.0,
                    y2: 20.0,
                },
            },
            Detection {
                class_id: 0,
                class_name: "person".to_string(),
                confidence: 0.95,
                bbox: BoundingBox {
                    x1: 0.0,
                    y1: 0.0,
                    x2: 1.0,
                    y2: 1.0,
                },
            },
        ];
        let result = detector.non_maximum_suppression(dets);
        assert_eq!(result.len(), 2);
        assert!(
            result[0].confidence >= result[1].confidence,
            "results should be sorted by confidence"
        );
    }

    // ─── YoloDetector methods that do not require a model ───────────────────

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_set_conf_threshold_clamps() {
        let mut detector = make_detector();
        detector.set_conf_threshold(1.5);
        assert!((detector.conf_threshold - 1.0).abs() < 1e-6);
        detector.set_conf_threshold(-0.5);
        assert!((detector.conf_threshold - 0.0).abs() < 1e-6);
        detector.set_conf_threshold(0.5);
        assert!((detector.conf_threshold - 0.5).abs() < 1e-6);
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_set_iou_threshold_clamps() {
        let mut detector = make_detector();
        detector.set_iou_threshold(2.0);
        assert!((detector.iou_threshold - 1.0).abs() < 1e-6);
        detector.set_iou_threshold(-1.0);
        assert!((detector.iou_threshold - 0.0).abs() < 1e-6);
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_set_input_size() {
        let mut detector = make_detector();
        detector.set_input_size(1280, 720);
        assert_eq!(detector.input_size, (1280, 720));
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_info_string_contains_expected_fields() {
        let detector = make_detector();
        let info = detector.info();
        assert!(info.contains("YOLOv8"), "info should contain version");
        assert!(info.contains("n"), "info should contain size suffix");
        assert!(info.contains("80"), "info should contain class count");
    }

    // ─── load_coco_names ────────────────────────────────────────────────────

    #[test]
    fn test_coco_names_all_strings_non_empty() {
        let names = load_coco_names();
        for name in &names {
            assert!(!name.is_empty(), "COCO class name should not be empty");
        }
    }

    #[test]
    fn test_coco_names_spot_checks() {
        let names = load_coco_names();
        assert_eq!(names[1], "bicycle");
        assert_eq!(names[2], "car");
        assert_eq!(names[15], "cat");
    }

    // ── Lines 253-254, 289-290: #[cfg(not(feature = "torch"))] stubs ─────────

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_preprocess_image_errors_without_torch() {
        let detector = make_detector();
        let result = detector.preprocess_image(std::path::Path::new("/img.jpg"));
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("feature") || msg.contains("torch") || msg.contains("PyTorch"),
            "{msg}"
        );
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_detect_errors_without_torch() {
        let detector = make_detector();
        let result = detector.detect(std::path::Path::new("/img.jpg"));
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("feature") || msg.contains("torch") || msg.contains("PyTorch"),
            "{msg}"
        );
    }
}

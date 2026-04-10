//! YOLO / object-detection preprocessing benchmarks.
//!
//! Covers:
//! - Letterbox padding + bilinear resize to YOLO input sizes (640×640, 1280×1280)
//! - Pixel normalization (u8 → f32 ÷ 255)
//! - NCHW tensor packing
//! - Non-maximum suppression (NMS) over raw detections
//! - IoU computation between bounding boxes
//! - Coordinate rescaling from model-space back to original image dimensions
//!
//! Run: cargo bench --bench detection_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct BBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    class_id: usize,
}

impl BBox {
    fn area(&self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }

    fn iou(&self, other: &BBox) -> f32 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);
        let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
        let union = self.area() + other.area() - inter;
        if union > 0.0 { inter / union } else { 0.0 }
    }
}

// ---------------------------------------------------------------------------
// Image preprocessing helpers
// ---------------------------------------------------------------------------

fn synthetic_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            v.push(((x * 5  + y * 3) % 256) as u8);
            v.push(((x * 3  + y * 7) % 256) as u8);
            v.push(((x * 11 + y * 5) % 256) as u8);
        }
    }
    v
}

/// Letterbox an image to `target × target` with padding value `pad`.
/// Returns (padded_image: Vec<u8>, scale, pad_x, pad_y).
fn letterbox(
    src: &[u8],
    sw: usize,
    sh: usize,
    target: usize,
    pad: u8,
) -> (Vec<u8>, f32, usize, usize) {
    let scale = (target as f32 / sw as f32).min(target as f32 / sh as f32);
    let new_w = (sw as f32 * scale) as usize;
    let new_h = (sh as f32 * scale) as usize;
    let pad_x = (target - new_w) / 2;
    let pad_y = (target - new_h) / 2;

    let mut dst = vec![pad; target * target * 3];

    let x_ratio = sw as f32 / new_w as f32;
    let y_ratio = sh as f32 / new_h as f32;

    for dy in 0..new_h {
        for dx in 0..new_w {
            let sx = ((dx as f32 * x_ratio) as usize).min(sw - 1);
            let sy = ((dy as f32 * y_ratio) as usize).min(sh - 1);
            let out_y = dy + pad_y;
            let out_x = dx + pad_x;
            for c in 0..3 {
                dst[(out_y * target + out_x) * 3 + c] = src[(sy * sw + sx) * 3 + c];
            }
        }
    }
    (dst, scale, pad_x, pad_y)
}

/// Convert HWC u8 to NCHW f32 (÷255 normalization).
fn hwc_to_nchw_f32(hwc: &[u8], w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    let mut nchw = vec![0.0f32; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            nchw[c * n + i] = hwc[i * 3 + c] as f32 / 255.0;
        }
    }
    nchw
}

// ---------------------------------------------------------------------------
// Detection post-processing
// ---------------------------------------------------------------------------

/// Greedy NMS: suppress boxes with IoU > threshold.
fn nms(mut boxes: Vec<BBox>, iou_thresh: f32) -> Vec<BBox> {
    // Sort descending by score.
    boxes.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut kept = Vec::with_capacity(boxes.len());
    let mut suppressed = vec![false; boxes.len()];

    for i in 0..boxes.len() {
        if suppressed[i] {
            continue;
        }
        kept.push(boxes[i]);
        for j in (i + 1)..boxes.len() {
            if !suppressed[j] && boxes[i].class_id == boxes[j].class_id {
                if boxes[i].iou(&boxes[j]) > iou_thresh {
                    suppressed[j] = true;
                }
            }
        }
    }
    kept
}

/// Rescale boxes from model-space (640×640) back to original image dimensions.
fn rescale_boxes(boxes: &mut [BBox], scale: f32, pad_x: usize, pad_y: usize) {
    for b in boxes.iter_mut() {
        b.x1 = ((b.x1 - pad_x as f32) / scale).max(0.0);
        b.y1 = ((b.y1 - pad_y as f32) / scale).max(0.0);
        b.x2 = ((b.x2 - pad_x as f32) / scale).max(0.0);
        b.y2 = ((b.y2 - pad_y as f32) / scale).max(0.0);
    }
}

/// Generate `n` synthetic detection boxes.
fn synthetic_boxes(n: usize, img_size: f32) -> Vec<BBox> {
    (0..n)
        .map(|i| {
            let x1 = (i as f32 * 7.3) % (img_size - 50.0);
            let y1 = (i as f32 * 11.7) % (img_size - 50.0);
            BBox {
                x1,
                y1,
                x2: x1 + 30.0 + (i as f32 * 3.1) % 200.0,
                y2: y1 + 30.0 + (i as f32 * 4.7) % 200.0,
                score: 0.9 - (i as f32 * 0.003) % 0.85,
                class_id: i % 80,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_letterbox(c: &mut Criterion) {
    let mut group = c.benchmark_group("det_letterbox");
    group.measurement_time(Duration::from_secs(10));

    let cases: &[(&str, usize, usize, usize)] = &[
        ("vga_to_640",     640,  480,  640),
        ("hd_to_640",     1280,  720,  640),
        ("full_hd_to_640",1920, 1080,  640),
        ("full_hd_to_1280",1920,1080, 1280),
    ];
    for &(name, sw, sh, target) in cases {
        let src = synthetic_rgb(sw, sh);
        group.throughput(Throughput::Bytes((sw * sh * 3) as u64));
        group.bench_function(name, |b| {
            b.iter(|| black_box(letterbox(black_box(&src), sw, sh, target, 114)));
        });
    }
    group.finish();
}

fn bench_hwc_to_nchw(c: &mut Criterion) {
    let sizes: &[(usize, usize)] = &[(640, 640), (1280, 1280)];
    let mut group = c.benchmark_group("det_hwc_to_nchw");

    for &(w, h) in sizes {
        let hwc = synthetic_rgb(w, h);
        group.throughput(Throughput::Bytes((w * h * 3) as u64));
        group.bench_with_input(BenchmarkId::new("size", format!("{w}x{h}")), &hwc, |b, hwc| {
            b.iter(|| black_box(hwc_to_nchw_f32(black_box(hwc), w, h)));
        });
    }
    group.finish();
}

fn bench_iou(c: &mut Criterion) {
    let a = BBox { x1: 10.0, y1: 20.0, x2: 100.0, y2: 200.0, score: 0.9, class_id: 0 };
    let b = BBox { x1: 50.0, y1: 80.0, x2: 150.0, y2: 250.0, score: 0.8, class_id: 0 };
    c.bench_function("det_iou_single", |b_fn| {
        b_fn.iter(|| black_box(a.iou(black_box(&b))));
    });
}

fn bench_nms(c: &mut Criterion) {
    let detection_counts = [10usize, 100, 1000, 10_000];
    let iou_thresholds = [0.45f32, 0.5];
    let mut group = c.benchmark_group("det_nms");
    group.measurement_time(Duration::from_secs(10));

    for &n in &detection_counts {
        for &thresh in &iou_thresholds {
            let label = format!("{n}boxes_iou{}", (thresh * 100.0) as u32);
            let boxes = synthetic_boxes(n, 640.0);
            group.throughput(Throughput::Elements(n as u64));
            group.bench_function(&label, |b| {
                b.iter(|| black_box(nms(black_box(boxes.clone()), thresh)));
            });
        }
    }
    group.finish();
}

fn bench_rescale_boxes(c: &mut Criterion) {
    let counts = [10usize, 100, 1000];
    let mut group = c.benchmark_group("det_rescale_boxes");

    for &n in &counts {
        let boxes = synthetic_boxes(n, 640.0);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("n", n), &boxes, |b, boxes| {
            b.iter(|| { let mut v = boxes.clone(); rescale_boxes(&mut v, 0.5625, 0, 80); black_box(v) });
        });
    }
    group.finish();
}

fn bench_full_preprocess_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("det_full_preprocess");
    group.measurement_time(Duration::from_secs(10));

    let cases: &[(&str, usize, usize, usize)] = &[
        ("vga",      640,  480, 640),
        ("hd",      1280,  720, 640),
        ("full_hd", 1920, 1080, 640),
    ];
    for &(name, sw, sh, target) in cases {
        let src = synthetic_rgb(sw, sh);
        group.throughput(Throughput::Bytes((sw * sh * 3) as u64));
        group.bench_function(name, |b| {
            b.iter(|| {
                let (padded, _scale, _px, _py) = letterbox(black_box(&src), sw, sh, target, 114);
                black_box(hwc_to_nchw_f32(&padded, target, target))
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_letterbox,
    bench_hwc_to_nchw,
    bench_iou,
    bench_nms,
    bench_rescale_boxes,
    bench_full_preprocess_pipeline,
);
criterion_main!(benches);

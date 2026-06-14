//! Measures the cost of building an `OrtYoloDetector` per request vs
//! reusing a single cached `Arc<OrtYoloDetector>` across requests.
//!
//! `cold_per_request` mirrors the production code path before bucket
//! "YOLO #4" lands: every `/yolo/detect` call ran
//! `OrtYoloDetector::new(...)` (Session::builder + EP setup +
//! `commit_from_file`) and then a single inference. `cached_arc`
//! mirrors the post-fix path: build once, reuse forever. Same image,
//! same number of iterations.
//!
//! Run: `cargo bench --bench yolo_detector_cache_bench`
//!
//! Skips itself with a clear message if `models/yolo/yolov8n.onnx` is
//! absent so CI without weights doesn't fail.

use criterion::{criterion_group, criterion_main, Criterion};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use torch_inference::core::ort_yolo::OrtYoloDetector;

const MODEL_REL_PATH: &str = "models/yolo/yolov8n.onnx";
const CONF: f32 = 0.25;
const IOU: f32 = 0.45;

fn coco_class_names() -> Vec<String> {
    // 80 COCO classes — content doesn't matter for timing.
    (0..80).map(|i| format!("class_{i}")).collect()
}

/// A small synthetic JPEG so we don't depend on a fixture.
fn synth_jpeg() -> Vec<u8> {
    use image::codecs::jpeg::JpegEncoder;
    let w = 640u32;
    let h = 640u32;
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            rgb.push(((x * 5 + y * 3) % 256) as u8);
            rgb.push(((x * 3 + y * 7) % 256) as u8);
            rgb.push(((x * 11 + y * 5) % 256) as u8);
        }
    }
    let mut buf = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut buf, 85);
        enc.encode(&rgb, w, h, image::ColorType::Rgb8.into()).unwrap();
    }
    buf
}

fn model_path() -> Option<PathBuf> {
    let p = PathBuf::from(MODEL_REL_PATH);
    if p.exists() { Some(p) } else { None }
}

fn cold_per_request(c: &mut Criterion) {
    let Some(model) = model_path() else {
        eprintln!(
            "skip: yolo_detector_cache_bench/cold_per_request — \
             {MODEL_REL_PATH} missing"
        );
        return;
    };
    let bytes = synth_jpeg();
    let class_names = coco_class_names();

    // Confirm the detector can be built and a single inference works
    // before we start timing. Failures should fail fast, not pollute
    // the benchmark output with errors.
    let probe = OrtYoloDetector::new(&model, class_names.clone())
        .expect("smoke: OrtYoloDetector::new");
    let _ = probe
        .detect_bytes(&bytes, CONF, IOU)
        .expect("smoke: detect_bytes");
    drop(probe);

    let mut group = c.benchmark_group("yolo_detector");
    group.sample_size(10).measurement_time(Duration::from_secs(20));

    group.bench_function("cold_per_request", |b| {
        b.iter(|| {
            // The unfixed production behaviour: build a fresh detector
            // for every request, run one inference, drop.
            let det = OrtYoloDetector::new(&model, class_names.clone()).unwrap();
            let _ = det.detect_bytes(&bytes, CONF, IOU).unwrap();
        });
    });

    group.finish();
}

fn cached_arc(c: &mut Criterion) {
    let Some(model) = model_path() else {
        eprintln!(
            "skip: yolo_detector_cache_bench/cached_arc — \
             {MODEL_REL_PATH} missing"
        );
        return;
    };
    let bytes = synth_jpeg();
    let class_names = coco_class_names();

    let det = Arc::new(
        OrtYoloDetector::new(&model, class_names).expect("OrtYoloDetector::new"),
    );

    // Warm the model so the first iteration doesn't pay first-run
    // cache misses (matches steady-state production after first hit).
    let _ = det.detect_bytes(&bytes, CONF, IOU).unwrap();

    let mut group = c.benchmark_group("yolo_detector");
    group.sample_size(20).measurement_time(Duration::from_secs(15));

    group.bench_function("cached_arc", |b| {
        b.iter(|| {
            let det = det.clone();
            let _ = det.detect_bytes(&bytes, CONF, IOU).unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, cold_per_request, cached_arc);
criterion_main!(benches);

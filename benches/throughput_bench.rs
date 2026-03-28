/// throughput_bench.rs — Measures torch-inference serving throughput across
/// three modalities: text, TTS, and image preprocessing.
///
/// Run with:
///   cargo bench --bench throughput_bench
///   cargo bench --bench throughput_bench -- text_request_throughput
///
/// Results let you compare the serving-layer overhead against external cloud
/// providers (see src/bin/provider_comparison.rs for the live HTTP comparison).
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use image::{DynamicImage, ImageBuffer, Rgb};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tokio::runtime::Runtime;
use torch_inference::batch::{BatchProcessor, BatchRequest};
use torch_inference::cache::Cache;

fn make_rt() -> Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

// ── 1. Text Request Queue Throughput ────────────────────────────────────────
// How many chars/sec can the batch-queue intake at different payload sizes.
// This is the gating overhead before the model does any work.
fn bench_text_request_throughput(c: &mut Criterion) {
    let rt = make_rt();
    let mut group = c.benchmark_group("text_request_throughput");
    group.measurement_time(std::time::Duration::from_secs(5));

    for (label, chars) in [("64_chars", 64usize), ("256_chars", 256), ("1024_chars", 1024)] {
        let text = "x".repeat(chars);
        group.throughput(Throughput::Bytes(chars as u64));
        group.bench_with_input(BenchmarkId::new("batch_queue", label), &text, |b, text| {
            // Large max-batch so the queue never blocks during the measurement.
            let processor = BatchProcessor::new(4096, 50);
            let mut id = 0u64;
            b.iter(|| {
                id += 1;
                let req = BatchRequest {
                    id: id.to_string(),
                    model_name: "text-model".to_string(),
                    inputs: vec![json!({"text": text})],
                    priority: 0,
                    timestamp: Instant::now(),
                };
                rt.block_on(black_box(processor.add_request(req))).ok();
                // Drain periodically to prevent OOM.
                if id % 2048 == 0 {
                    rt.block_on(processor.clear_batch());
                }
            });
        });
    }
    group.finish();
}

// ── 2. TTS Queue Throughput ──────────────────────────────────────────────────
// Characters-per-second queued for TTS inference.
// Real-time factor (RTF) = synthesis_time / audio_duration; this benchmark
// measures only the queue-entry step so you can isolate serving overhead from
// model cost.
fn bench_tts_throughput(c: &mut Criterion) {
    let rt = make_rt();
    let mut group = c.benchmark_group("tts_throughput");
    group.measurement_time(std::time::Duration::from_secs(5));

    let cases: &[(&str, &str)] = &[
        (
            "short_54",
            "The quick brown fox jumps over the lazy dog.",
        ),
        (
            "sentence_121",
            "Hello, this is a benchmark of the TTS synthesis system. \
             We measure characters per second through the queue.",
        ),
        (
            "paragraph_400",
            "Welcome to the torch inference server benchmark suite. \
             This paragraph tests how many characters per second can be queued \
             for TTS synthesis. The system uses dynamic batching and adaptive \
             timeouts to maximize throughput while minimising latency. \
             Longer texts are split into sentences internally before being \
             dispatched to the chosen synthesis engine.",
        ),
    ];

    for (label, text) in cases {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::new("tts_queue", label), text, |b, text| {
            let processor = BatchProcessor::new(64, 50);
            let mut id = 0u64;
            b.iter(|| {
                id += 1;
                let req = BatchRequest {
                    id: id.to_string(),
                    model_name: "kokoro-tts".to_string(),
                    inputs: vec![json!({"text": text, "voice": "af_bella", "speed": 1.0})],
                    priority: 0,
                    timestamp: Instant::now(),
                };
                rt.block_on(black_box(processor.add_request(req))).ok();
                if id % 32 == 0 {
                    rt.block_on(processor.clear_batch());
                }
            });
        });
    }
    group.finish();
}

// ── 3. Image Preprocessing Throughput ───────────────────────────────────────
// Pixels-per-second through the resize step that runs before model inference.
// Sizes match common model input requirements:
//   224×224 — ResNet, ViT, EfficientNet
//   448×448 — MobileNetV4-Hybrid-Large
//   640×640 — YOLOv8 default
fn bench_image_preprocessing_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_preprocessing_throughput");
    group.measurement_time(std::time::Duration::from_secs(5));

    let sizes: &[(&str, u32, u32)] = &[
        ("thumbnail_64x64", 64, 64),
        ("resnet_224x224", 224, 224),
        ("mobilenetv4_448x448", 448, 448),
        ("yolo_640x640", 640, 640),
    ];

    for (label, w, h) in sizes {
        // Pixels × 3 channels (RGB).
        let pixel_bytes = (*w * *h * 3) as u64;
        group.throughput(Throughput::Bytes(pixel_bytes));

        // Source image is 4× the target so resize always does real work.
        let src: DynamicImage = DynamicImage::ImageRgb8(ImageBuffer::<Rgb<u8>, _>::from_fn(
            w * 4,
            h * 4,
            |x, y| Rgb([(x % 255) as u8, (y % 255) as u8, ((x + y) % 255) as u8]),
        ));

        group.bench_with_input(BenchmarkId::new("resize_to", label), &src, |b, src| {
            b.iter(|| {
                black_box(src.resize_exact(*w, *h, image::imageops::FilterType::Lanczos3))
            });
        });
    }
    group.finish();
}

// ── 4. Cache Round-trip Throughput ──────────────────────────────────────────
// Bytes/sec through the full cache write + read cycle that executes once per
// request (write result after inference, read on cache-hit path).
fn bench_cache_roundtrip_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_roundtrip_throughput");
    group.measurement_time(std::time::Duration::from_secs(5));

    let payloads: &[(&str, serde_json::Value)] = &[
        (
            "text_small_200B",
            json!({"tokens": 50, "text": "This is a short answer.", "model": "gpt-4o-mini"}),
        ),
        (
            "text_medium_800B",
            json!({
                "tokens": 200,
                "text": "x".repeat(800),
                "model": "gpt-4o",
                "finish_reason": "stop"
            }),
        ),
        (
            "image_result_300B",
            json!({
                "class": "tabby_cat",
                "confidence": 0.973,
                "top5": [
                    {"label": "tabby_cat",     "score": 0.973},
                    {"label": "tiger_cat",     "score": 0.018},
                    {"label": "persian_cat",   "score": 0.004},
                    {"label": "egyptian_cat",  "score": 0.003},
                    {"label": "lynx",          "score": 0.001}
                ]
            }),
        ),
    ];

    for (label, payload) in payloads {
        let bytes = serde_json::to_vec(payload).unwrap().len() as u64;
        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(
            BenchmarkId::new("set_then_get", label),
            payload,
            |b, payload| {
                let cache = Cache::new(100_000);
                let mut id = 0u64;
                b.iter(|| {
                    id += 1;
                    let key = format!("model:v1:{}", id % 50_000);
                    cache
                        .set(key.clone(), black_box(payload.clone()), 300)
                        .ok();
                    black_box(cache.get(&key))
                });
            },
        );
    }
    group.finish();
}

// ── 5. Concurrent Cache Read Throughput ─────────────────────────────────────
// DashMap is lock-free for reads. This bench shows throughput under contention
// from 1 → 16 parallel inference worker threads.
fn bench_concurrent_cache_reads(c: &mut Criterion) {
    use std::thread;

    let mut group = c.benchmark_group("concurrent_cache_reads");
    group.measurement_time(std::time::Duration::from_secs(5));

    // Shared pre-populated cache.
    let cache = Arc::new(Cache::new(100_000));
    for i in 0..50_000 {
        cache
            .set(
                format!("key_{}", i),
                json!({"result": i, "confidence": 0.99}),
                3600,
            )
            .ok();
    }

    for threads in [1usize, 2, 4, 8, 16] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("parallel_readers", threads),
            &threads,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let cache = Arc::clone(&cache);
                    let per_thread = (iters / threads as u64).max(1);
                    let start = Instant::now();
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let c = Arc::clone(&cache);
                            thread::spawn(move || {
                                for i in 0..per_thread {
                                    black_box(c.get(&format!(
                                        "key_{}",
                                        (i + t as u64 * per_thread) % 50_000
                                    )));
                                }
                            })
                        })
                        .collect();
                    for h in handles {
                        h.join().unwrap();
                    }
                    start.elapsed()
                });
            },
        );
    }
    group.finish();
}

// ── 6. Batch Latency vs Batch Size ──────────────────────────────────────────
// Shows how average per-request overhead changes as batch size grows.
// Larger batches amortise fixed costs; this bench helps tune max_batch_size.
fn bench_batch_latency_vs_size(c: &mut Criterion) {
    let rt = make_rt();
    let mut group = c.benchmark_group("batch_latency_vs_size");
    group.measurement_time(std::time::Duration::from_secs(5));

    for batch_size in [1usize, 4, 8, 16, 32] {
        group.bench_with_input(
            BenchmarkId::new("fill_batch", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::ZERO;
                    for run in 0..iters {
                        let processor = BatchProcessor::new(batch_size, 50);
                        let start = Instant::now();
                        for i in 0..batch_size {
                            let req = BatchRequest {
                                id: format!("{}-{}", run, i),
                                model_name: "model".to_string(),
                                inputs: vec![json!({"text": "bench payload"})],
                                priority: 0,
                                timestamp: Instant::now(),
                            };
                            rt.block_on(processor.add_request(req)).ok();
                        }
                        total += start.elapsed();
                        rt.block_on(processor.clear_batch());
                    }
                    total
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_text_request_throughput,
    bench_tts_throughput,
    bench_image_preprocessing_throughput,
    bench_cache_roundtrip_throughput,
    bench_concurrent_cache_reads,
    bench_batch_latency_vs_size,
);
criterion_main!(benches);

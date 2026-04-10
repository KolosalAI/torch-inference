//! Image classification preprocessing benchmarks.
//!
//! Covers:
//! - JPEG/PNG decode cost at various resolutions
//! - ImageNet-style normalization (mean/std subtraction)
//! - Bilinear resize to classifier input dimensions (224×224, 299×299, 384×384)
//! - CHW tensor layout conversion (HWC → CHW)
//! - Batch assembly for multi-image inference
//! - Top-K softmax + argmax over logit vectors
//!
//! Run: cargo bench --bench classification_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Preprocessing helpers (mirrors core/image_pipeline.rs logic)
// ---------------------------------------------------------------------------

/// Synthetic RGB image (H×W×3, u8).
fn synthetic_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut img = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            img.push(((x * 7 + y * 3) % 256) as u8);
            img.push(((x * 3 + y * 11) % 256) as u8);
            img.push(((x * 11 + y * 7) % 256) as u8);
        }
    }
    img
}

/// Bilinear resize of a flat HWC u8 image.
fn bilinear_resize(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut dst = vec![0u8; dw * dh * 3];
    let x_ratio = sw as f32 / dw as f32;
    let y_ratio = sh as f32 / dh as f32;

    for dy in 0..dh {
        for dx in 0..dw {
            let sx = (dx as f32 * x_ratio).min(sw as f32 - 1.0);
            let sy = (dy as f32 * y_ratio).min(sh as f32 - 1.0);
            let x0 = sx as usize;
            let y0 = sy as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let xf = sx - x0 as f32;
            let yf = sy - y0 as f32;

            for c in 0..3 {
                let p00 = src[(y0 * sw + x0) * 3 + c] as f32;
                let p10 = src[(y0 * sw + x1) * 3 + c] as f32;
                let p01 = src[(y1 * sw + x0) * 3 + c] as f32;
                let p11 = src[(y1 * sw + x1) * 3 + c] as f32;
                let v = p00 * (1.0 - xf) * (1.0 - yf)
                    + p10 * xf * (1.0 - yf)
                    + p01 * (1.0 - xf) * yf
                    + p11 * xf * yf;
                dst[(dy * dw + dx) * 3 + c] = v as u8;
            }
        }
    }
    dst
}

/// ImageNet normalization: (pixel/255 - mean) / std → f32, HWC→CHW.
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3]  = [0.229, 0.224, 0.225];

fn imagenet_normalize_chw(hwc: &[u8], width: usize, height: usize) -> Vec<f32> {
    let n = width * height;
    let mut chw = vec![0.0f32; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            let v = hwc[i * 3 + c] as f32 / 255.0;
            chw[c * n + i] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        }
    }
    chw
}

/// Full preprocess pipeline: resize then normalize.
fn preprocess(src: &[u8], sw: usize, sh: usize, tw: usize, th: usize) -> Vec<f32> {
    let resized = bilinear_resize(src, sw, sh, tw, th);
    imagenet_normalize_chw(&resized, tw, th)
}

/// Assemble a batch of `n` identical images into a flat NCHW tensor.
fn batch_images(single_chw: &[f32], n: usize) -> Vec<f32> {
    let mut batch = Vec::with_capacity(single_chw.len() * n);
    for _ in 0..n {
        batch.extend_from_slice(single_chw);
    }
    batch
}

/// Softmax over logits in place.
fn softmax(logits: &mut [f32]) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in logits.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in logits.iter_mut() {
        *x /= sum;
    }
}

/// Return top-K (index, score) pairs sorted descending.
fn top_k(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> =
        probs.iter().copied().enumerate().collect();
    // partial sort — find top-k in O(n·k) which is fine for k ≤ 1000
    for i in 0..k.min(indexed.len()) {
        let max_pos = indexed[i..]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap())
            .map(|(j, _)| i + j)
            .unwrap_or(i);
        indexed.swap(i, max_pos);
    }
    indexed[..k.min(indexed.len())].to_vec()
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_bilinear_resize(c: &mut Criterion) {
    let mut group = c.benchmark_group("cls_bilinear_resize");
    group.measurement_time(Duration::from_secs(8));

    // Common source resolutions → ImageNet 224×224 target
    let cases: &[(&str, usize, usize)] = &[
        ("640x480_to_224",  640, 480),
        ("1280x720_to_224", 1280, 720),
        ("1920x1080_to_224",1920, 1080),
        ("256x256_to_224",  256, 256),
    ];
    for &(name, sw, sh) in cases {
        let src = synthetic_rgb(sw, sh);
        group.throughput(Throughput::Bytes((sw * sh * 3) as u64));
        group.bench_function(name, |b| {
            b.iter(|| black_box(bilinear_resize(black_box(&src), sw, sh, 224, 224)));
        });
    }
    group.finish();
}

fn bench_imagenet_normalize(c: &mut Criterion) {
    let sizes: &[(usize, usize)] = &[(224, 224), (299, 299), (384, 384)];
    let mut group = c.benchmark_group("cls_imagenet_normalize");

    for &(w, h) in sizes {
        let hwc = synthetic_rgb(w, h);
        group.throughput(Throughput::Elements((w * h) as u64));
        group.bench_with_input(BenchmarkId::new("hwc_to_chw", format!("{w}x{h}")), &hwc, |b, hwc| {
            b.iter(|| black_box(imagenet_normalize_chw(black_box(hwc), w, h)));
        });
    }
    group.finish();
}

fn bench_full_preprocess(c: &mut Criterion) {
    let mut group = c.benchmark_group("cls_full_preprocess");
    group.measurement_time(Duration::from_secs(10));

    let src_sizes: &[(&str, usize, usize)] = &[
        ("vga",    640, 480),
        ("hd",    1280, 720),
        ("full_hd",1920,1080),
    ];
    let target_sizes: &[(&str, usize, usize)] = &[
        ("224", 224, 224),
        ("299", 299, 299),
    ];
    for &(sname, sw, sh) in src_sizes {
        let src = synthetic_rgb(sw, sh);
        for &(tname, tw, th) in target_sizes {
            let label = format!("{sname}_to_{tname}");
            group.throughput(Throughput::Bytes((sw * sh * 3) as u64));
            group.bench_function(&label, |b| {
                b.iter(|| black_box(preprocess(black_box(&src), sw, sh, tw, th)));
            });
        }
    }
    group.finish();
}

fn bench_batch_assembly(c: &mut Criterion) {
    let single = imagenet_normalize_chw(&synthetic_rgb(224, 224), 224, 224);
    let batch_sizes = [1usize, 4, 8, 16, 32];
    let mut group = c.benchmark_group("cls_batch_assembly");

    for &n in &batch_sizes {
        group.throughput(Throughput::Elements((single.len() * n) as u64));
        group.bench_with_input(BenchmarkId::new("batch", n), &single, |b, img| {
            b.iter(|| black_box(batch_images(black_box(img), n)));
        });
    }
    group.finish();
}

fn bench_top_k(c: &mut Criterion) {
    let class_counts = [100usize, 1000, 21_841]; // ImageNet-1k, -21k
    let k_values = [1usize, 5, 10];
    let mut group = c.benchmark_group("cls_top_k");

    for &n_classes in &class_counts {
        let mut logits: Vec<f32> = (0..n_classes)
            .map(|i| ((i as f32 * 0.37 + 1.0).sin()) * 5.0)
            .collect();
        softmax(&mut logits);

        for &k in &k_values {
            group.throughput(Throughput::Elements(n_classes as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("k{k}"), n_classes),
                &logits,
                |b, probs| {
                    b.iter(|| black_box(top_k(black_box(probs), k)));
                },
            );
        }
    }
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let sizes = [1000usize, 10_000, 21_841];
    let mut group = c.benchmark_group("cls_softmax");

    for &n in &sizes {
        let logits: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("classes", n), &logits, |b, raw| {
            b.iter(|| {
                let mut v = raw.clone();
                softmax(black_box(&mut v));
                black_box(v)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bilinear_resize,
    bench_imagenet_normalize,
    bench_full_preprocess,
    bench_batch_assembly,
    bench_top_k,
    bench_softmax,
);
criterion_main!(benches);

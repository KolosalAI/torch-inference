use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use image::{RgbImage, ImageBuffer, imageops};
use std::time::Instant;

fn create_test_image(width: u32, height: u32) -> RgbImage {
    ImageBuffer::from_fn(width, height, |x, y| {
        image::Rgb([
            ((x * 255) / width) as u8,
            ((y * 255) / height) as u8,
            128,
        ])
    })
}

#[inline]
fn preprocess_optimized(img: &RgbImage, target_size: (u32, u32)) -> Vec<f32> {
    let resized = imageops::resize(
        img,
        target_size.0,
        target_size.1,
        imageops::FilterType::Triangle,
    );
    
    let mut tensor = Vec::with_capacity((target_size.0 * target_size.1 * 3) as usize);
    let pixels = resized.as_raw();
    const INV_SCALE: f32 = 1.0 / 127.5;
    
    for chunk in pixels.chunks_exact(3) {
        unsafe {
            tensor.push((*chunk.get_unchecked(0) as f32 - 127.5) * INV_SCALE);
            tensor.push((*chunk.get_unchecked(1) as f32 - 127.5) * INV_SCALE);
            tensor.push((*chunk.get_unchecked(2) as f32 - 127.5) * INV_SCALE);
        }
    }
    
    tensor
}

fn preprocess_baseline(img: &RgbImage, target_size: (u32, u32)) -> Vec<f32> {
    let resized = imageops::resize(
        img,
        target_size.0,
        target_size.1,
        imageops::FilterType::Lanczos3,
    );
    
    let mut tensor = Vec::with_capacity((target_size.0 * target_size.1 * 3) as usize);
    
    for pixel in resized.pixels() {
        tensor.push((pixel[0] as f32 / 255.0 - 0.5) * 2.0);
        tensor.push((pixel[1] as f32 / 255.0 - 0.5) * 2.0);
        tensor.push((pixel[2] as f32 / 255.0 - 0.5) * 2.0);
    }
    
    tensor
}

fn benchmark_single_image(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_image");
    let img = create_test_image(1920, 1080);
    let target_size = (224, 224);
    
    group.bench_function("baseline", |b| {
        b.iter(|| black_box(preprocess_baseline(&img, target_size)))
    });
    
    group.bench_function("optimized", |b| {
        b.iter(|| black_box(preprocess_optimized(&img, target_size)))
    });
    
    group.finish();
}

fn benchmark_batch_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_throughput");
    
    let batch_sizes = vec![1, 8, 32, 128, 512, 1024];
    let target_size = (224, 224);
    
    for batch_size in batch_sizes {
        let images: Vec<_> = (0..batch_size)
            .map(|_| create_test_image(1920, 1080))
            .collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("batch_{}", batch_size)),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let results: Vec<_> = images.iter()
                        .map(|img| preprocess_optimized(img, target_size))
                        .collect();
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_single_image, benchmark_batch_throughput);
criterion_main!(benches);

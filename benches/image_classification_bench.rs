use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use image::{RgbImage, ImageBuffer};

// Simple image classification benchmark without requiring actual model loading
// This measures the preprocessing and data handling pipeline

fn create_test_image(width: u32, height: u32) -> RgbImage {
    ImageBuffer::from_fn(width, height, |x, y| {
        image::Rgb([
            ((x * 255) / width) as u8,
            ((y * 255) / height) as u8,
            128,
        ])
    })
}

fn preprocess_image(img: &RgbImage, target_size: (u32, u32)) -> Vec<f32> {
    let resized = image::imageops::resize(
        img,
        target_size.0,
        target_size.1,
        image::imageops::FilterType::Lanczos3,
    );
    
    let mut tensor = Vec::with_capacity((target_size.0 * target_size.1 * 3) as usize);
    
    // Normalize to [-1, 1] range (ImageNet preprocessing)
    for pixel in resized.pixels() {
        tensor.push((pixel[0] as f32 / 255.0 - 0.5) * 2.0);
        tensor.push((pixel[1] as f32 / 255.0 - 0.5) * 2.0);
        tensor.push((pixel[2] as f32 / 255.0 - 0.5) * 2.0);
    }
    
    tensor
}

fn benchmark_image_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_preprocessing");
    group.measurement_time(Duration::from_secs(10));
    
    let image_sizes = vec![
        (224, 224, "ResNet/MobileNet"),
        (384, 384, "ViT-Large"),
        (448, 448, "EVA02-Large"),
        (512, 512, "ConvNeXt"),
    ];
    
    for (width, height, name) in image_sizes {
        let img = create_test_image(1920, 1080); // Full HD source image
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}-{}", width, height, name)),
            &(width, height),
            |b, &(w, h)| {
                b.iter(|| {
                    black_box(preprocess_image(&img, (w, h)))
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_batch_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_preprocessing");
    group.measurement_time(Duration::from_secs(10));
    
    let batch_sizes = vec![1, 4, 8, 16, 32];
    let img = create_test_image(1920, 1080);
    
    for batch_size in batch_sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("batch_{}", batch_size)),
            &batch_size,
            |b, &size| {
                b.iter(|| {
                    let mut batch = Vec::with_capacity(size);
                    for _ in 0..size {
                        batch.push(preprocess_image(&img, (224, 224)));
                    }
                    black_box(batch)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_operations");
    
    let sizes = vec![224, 384, 448, 512];
    
    for size in sizes {
        let tensor_size = (size * size * 3) as usize;
        let tensor: Vec<f32> = (0..tensor_size).map(|i| i as f32).collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("size_{}", size)),
            &tensor,
            |b, t| {
                b.iter(|| {
                    // Simulate common tensor operations
                    let result: Vec<f32> = t.iter()
                        .map(|&x| x * 0.5 + 0.25)
                        .collect();
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_image_preprocessing,
    benchmark_batch_preprocessing,
    benchmark_tensor_operations
);
criterion_main!(benches);

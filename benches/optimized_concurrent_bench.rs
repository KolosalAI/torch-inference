use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;
use image::{RgbImage, ImageBuffer};
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use rayon::iter::{IntoParallelRefIterator, IntoParallelIterator};

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
    
    for pixel in resized.pixels() {
        tensor.push((pixel[0] as f32 / 255.0 - 0.5) * 2.0);
        tensor.push((pixel[1] as f32 / 255.0 - 0.5) * 2.0);
        tensor.push((pixel[2] as f32 / 255.0 - 0.5) * 2.0);
    }
    
    tensor
}

// Strategy 1: Rayon thread pool (CPU-bound work)
fn benchmark_rayon_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("rayon_parallel");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(30);
    
    let concurrency_levels = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    let img = Arc::new(create_test_image(1920, 1080));
    let target_size = (224, 224);
    
    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("rayon_{}", concurrency)),
            &concurrency,
            |b, &conc| {
                b.iter(|| {
                    let images: Vec<_> = (0..conc).map(|_| Arc::clone(&img)).collect();
                    let results: Vec<_> = images.par_iter()
                        .map(|img| preprocess_image(img, target_size))
                        .collect();
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

// Strategy 2: Custom worker pool with bounded queue
fn benchmark_bounded_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("bounded_pool");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(30);
    
    let concurrency_levels = vec![1, 4, 16, 64, 256];
    let img = Arc::new(create_test_image(1920, 1080));
    let target_size = (224, 224);
    
    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("bounded_{}", concurrency)),
            &concurrency,
            |b, &conc| {
                let rt = Runtime::new().unwrap();
                
                b.iter(|| {
                    rt.block_on(async {
                        // Limit concurrent spawn_blocking to reasonable number
                        let max_workers = std::cmp::min(conc, 64);
                        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_workers));
                        let mut tasks = Vec::with_capacity(conc);
                        
                        for _ in 0..conc {
                            let permit = semaphore.clone().acquire_owned().await.unwrap();
                            let img_clone = Arc::clone(&img);
                            
                            tasks.push(tokio::task::spawn_blocking(move || {
                                let result = preprocess_image(&img_clone, target_size);
                                drop(permit);
                                result
                            }));
                        }
                        
                        let results = futures::future::join_all(tasks).await;
                        black_box(results)
                    })
                });
            },
        );
    }
    
    group.finish();
}

// Strategy 3: Batch processing with optimal chunk size
fn benchmark_batched_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batched_processing");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(30);
    
    let total_images = 1000;
    let chunk_sizes = vec![8, 16, 32, 64];
    let img = Arc::new(create_test_image(1920, 1080));
    let target_size = (224, 224);
    
    for chunk_size in chunk_sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("chunk_{}", chunk_size)),
            &chunk_size,
            |b, &chunk| {
                b.iter(|| {
                    let images: Vec<_> = (0..total_images).map(|_| Arc::clone(&img)).collect();
                    
                    let results: Vec<_> = images.par_chunks(chunk)
                        .flat_map(|chunk_imgs| {
                            chunk_imgs.iter()
                                .map(|img| preprocess_image(img, target_size))
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

// Strategy 4: Pipeline with stages
fn benchmark_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);
    
    let total_images = 100;
    let img = Arc::new(create_test_image(1920, 1080));
    let target_size = (224, 224);
    
    group.bench_function("pipeline_parallel", |b| {
        b.iter(|| {
            // Stage 1: Resize (parallel)
            let images: Vec<_> = (0..total_images).map(|_| Arc::clone(&img)).collect();
            let resized: Vec<_> = images.par_iter()
                .map(|img| {
                    image::imageops::resize(
                        img.as_ref(),
                        target_size.0,
                        target_size.1,
                        image::imageops::FilterType::Lanczos3,
                    )
                })
                .collect();
            
            // Stage 2: Tensorize (parallel)
            let tensors: Vec<_> = resized.par_iter()
                .map(|resized_img| {
                    let mut tensor = Vec::with_capacity((target_size.0 * target_size.1 * 3) as usize);
                    for pixel in resized_img.pixels() {
                        tensor.push((pixel[0] as f32 / 255.0 - 0.5) * 2.0);
                        tensor.push((pixel[1] as f32 / 255.0 - 0.5) * 2.0);
                        tensor.push((pixel[2] as f32 / 255.0 - 0.5) * 2.0);
                    }
                    tensor
                })
                .collect();
            
            black_box(tensors)
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_rayon_parallel,
    benchmark_bounded_pool,
    benchmark_batched_processing,
    benchmark_pipeline
);
criterion_main!(benches);

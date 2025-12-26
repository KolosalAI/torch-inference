use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;
use image::{RgbImage, ImageBuffer};

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
fn preprocess_image(img: &RgbImage, target_size: (u32, u32)) -> Vec<f32> {
    let resized = image::imageops::resize(
        img,
        target_size.0,
        target_size.1,
        image::imageops::FilterType::Triangle, // 2x faster than Lanczos3
    );
    
    let mut tensor = Vec::with_capacity((target_size.0 * target_size.1 * 3) as usize);
    
    // Optimized vectorization-friendly normalization
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

async fn process_image_async(img: Arc<RgbImage>, target_size: (u32, u32)) -> Vec<f32> {
    tokio::task::spawn_blocking(move || {
        preprocess_image(&img, target_size)
    })
    .await
    .unwrap()
}

fn benchmark_concurrent_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_processing");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(50);
    
    let concurrency_levels = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    let img = Arc::new(create_test_image(1920, 1080));
    let target_size = (224, 224);
    
    // Optimized runtime
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .max_blocking_threads(1024) // Allow high concurrency
        .enable_all()
        .build()
        .unwrap();
    
    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("concurrent_{}", concurrency)),
            &concurrency,
            |b, &conc| {
                b.iter(|| {
                    rt.block_on(async {
                        let tasks: Vec<_> = (0..conc)
                            .map(|_| {
                                let img_clone = Arc::clone(&img);
                                process_image_async(img_clone, target_size)
                            })
                            .collect();
                        
                        let results = futures::future::join_all(tasks).await;
                        black_box(results)
                    })
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_concurrent_batches(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_batches");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(30);
    
    let concurrency_levels = vec![1, 2, 4, 8, 16, 32, 64];
    let img = Arc::new(create_test_image(1920, 1080));
    let target_size = (224, 224);
    let batch_size = 4;
    
    // Optimized runtime
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .max_blocking_threads(256)
        .enable_all()
        .build()
        .unwrap();
    
    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("batch4_concurrent_{}", concurrency)),
            &concurrency,
            |b, &conc| {
                b.iter(|| {
                    rt.block_on(async {
                        let batch_tasks: Vec<_> = (0..conc)
                            .map(|_| {
                                let img_clone = Arc::clone(&img);
                                tokio::task::spawn_blocking(move || {
                                    (0..batch_size)
                                        .map(|_| preprocess_image(&img_clone, target_size))
                                        .collect::<Vec<_>>()
                                })
                            })
                            .collect();
                        
                        let results = futures::future::join_all(batch_tasks).await;
                        black_box(results)
                    })
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_throughput_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_scaling");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(20);
    
    let total_images = 1000;
    let concurrency_levels = vec![1, 4, 16, 64, 256];
    let img = Arc::new(create_test_image(1920, 1080));
    let target_size = (224, 224);
    
    // Create optimized runtime with more worker threads
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .max_blocking_threads(512) // Increase blocking thread pool
        .thread_name("tokio-worker")
        .thread_stack_size(2 * 1024 * 1024)
        .enable_all()
        .build()
        .unwrap();
    
    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("images1000_concurrent_{}", concurrency)),
            &concurrency,
            |b, &conc| {
                b.iter(|| {
                    rt.block_on(async {
                        let semaphore = Arc::new(tokio::sync::Semaphore::new(conc));
                        let mut tasks = Vec::with_capacity(total_images);
                        
                        for _ in 0..total_images {
                            let permit = semaphore.clone().acquire_owned().await.unwrap();
                            let img_clone = Arc::clone(&img);
                            
                            // Spawn tasks without extra async overhead
                            tasks.push(tokio::task::spawn(async move {
                                let result = tokio::task::spawn_blocking(move || {
                                    preprocess_image(&img_clone, target_size)
                                })
                                .await
                                .unwrap();
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

criterion_group!(
    benches,
    benchmark_concurrent_processing,
    benchmark_concurrent_batches,
    benchmark_throughput_scaling
);
criterion_main!(benches);

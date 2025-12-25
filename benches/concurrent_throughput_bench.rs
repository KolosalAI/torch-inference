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
    let rt = Runtime::new().unwrap();
    
    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("concurrent_{}", concurrency)),
            &concurrency,
            |b, &conc| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut tasks = Vec::with_capacity(conc);
                        
                        for _ in 0..conc {
                            let img_clone = Arc::clone(&img);
                            tasks.push(process_image_async(img_clone, target_size));
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

fn benchmark_concurrent_batches(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_batches");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(30);
    
    let concurrency_levels = vec![1, 2, 4, 8, 16, 32, 64];
    let img = Arc::new(create_test_image(1920, 1080));
    let target_size = (224, 224);
    let batch_size = 4;
    let rt = Runtime::new().unwrap();
    
    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("batch4_concurrent_{}", concurrency)),
            &concurrency,
            |b, &conc| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut batch_tasks = Vec::with_capacity(conc);
                        
                        for _ in 0..conc {
                            let img_clone = Arc::clone(&img);
                            
                            batch_tasks.push(tokio::task::spawn_blocking(move || {
                                let mut batch = Vec::with_capacity(batch_size);
                                for _ in 0..batch_size {
                                    batch.push(preprocess_image(&img_clone, target_size));
                                }
                                batch
                            }));
                        }
                        
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
    let rt = Runtime::new().unwrap();
    
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
                            
                            tasks.push(tokio::task::spawn(async move {
                                let result = process_image_async(img_clone, target_size).await;
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

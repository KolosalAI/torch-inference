use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::Arc;
use std::time::Duration;
use torch_inference::image_processor::{ImageProcessor, create_test_image};
use tokio::runtime::Runtime;

fn benchmark_optimized_concurrent(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimized_concurrent");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(30);
    
    let concurrency_levels = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    let img = Arc::new(create_test_image(1920, 1080));
    let target_size = (224, 224);
    
    for concurrency in concurrency_levels {
        let processor = Arc::new(ImageProcessor::new(64)); // Bounded at 64
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("bounded_{}", concurrency)),
            &concurrency,
            |b, &conc| {
                let rt = Runtime::new().unwrap();
                
                b.iter(|| {
                    rt.block_on(async {
                        let mut tasks = Vec::with_capacity(conc);
                        
                        for _ in 0..conc {
                            let processor = Arc::clone(&processor);
                            let img_clone = img.clone();
                            
                            tasks.push(tokio::spawn(async move {
                                processor.preprocess_async((*img_clone).clone(), target_size).await
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

fn benchmark_rayon_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("rayon_parallel");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(30);
    
    let batch_sizes = vec![1, 4, 8, 16, 32, 64, 128, 256];
    let img = create_test_image(1920, 1080);
    let target_size = (224, 224);
    
    for batch_size in batch_sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("batch_{}", batch_size)),
            &batch_size,
            |b, &size| {
                let images: Vec<_> = (0..size).map(|_| img.clone()).collect();
                
                b.iter(|| {
                    let results = ImageProcessor::preprocess_batch_parallel(&images, target_size);
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_optimized_concurrent,
    benchmark_rayon_parallel
);
criterion_main!(benches);

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use torch_inference::ultra_optimized_processor::{UltraOptimizedProcessor, create_test_image_fast};

fn benchmark_ultra_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("ultra_batch");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(30);
    
    let processor = UltraOptimizedProcessor::new(None); // Use all cores
    let batch_sizes = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    let target_size = (224, 224);
    
    for batch_size in batch_sizes {
        let images: Vec<_> = (0..batch_size)
            .map(|_| create_test_image_fast(1920, 1080))
            .collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("batch_{}", batch_size)),
            &images,
            |b, imgs| {
                b.iter(|| {
                    let results = processor.process_batch_ultra(imgs, target_size);
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_ultra_chunked(c: &mut Criterion) {
    let mut group = c.benchmark_group("ultra_chunked");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);
    
    let processor = UltraOptimizedProcessor::new(None);
    let batch_sizes = vec![64, 128, 256, 512, 1024];
    let target_size = (224, 224);
    
    for batch_size in batch_sizes {
        let images: Vec<_> = (0..batch_size)
            .map(|_| create_test_image_fast(1920, 1080))
            .collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("chunked_{}", batch_size)),
            &images,
            |b, imgs| {
                b.iter(|| {
                    let results = processor.process_batch_chunked(imgs, target_size);
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_single_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("ultra_single");
    group.measurement_time(Duration::from_secs(10));
    
    let processor = UltraOptimizedProcessor::new(None);
    let img = create_test_image_fast(1920, 1080);
    
    group.bench_function("single_224", |b| {
        b.iter(|| {
            let result = processor.preprocess_optimized(&img, (224, 224));
            black_box(result)
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_ultra_batch,
    benchmark_ultra_chunked,
    benchmark_single_optimized
);
criterion_main!(benches);

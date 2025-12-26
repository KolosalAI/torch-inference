use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use torch_inference::image_processor::{ImageProcessor, create_test_image};

fn benchmark_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(30);
    
    let processor = ImageProcessor::new(64);
    let batch_sizes = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    let target_size = (224, 224);
    
    for batch_size in batch_sizes {
        let images: Vec<_> = (0..batch_size)
            .map(|_| create_test_image(1920, 1080))
            .collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("batch_{}", batch_size)),
            &images,
            |b, imgs| {
                b.iter(|| {
                    let results = processor.preprocess_batch(imgs, target_size);
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_chunked_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunked_processing");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);
    
    let batch_sizes = vec![64, 128, 256, 512, 1024];
    let target_size = (224, 224);
    
    for batch_size in batch_sizes {
        let images: Vec<_> = (0..batch_size)
            .map(|_| create_test_image(1920, 1080))
            .collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("chunked_{}", batch_size)),
            &images,
            |b, imgs| {
                b.iter(|| {
                    let results = ImageProcessor::preprocess_batch_chunked(imgs, target_size, 8);
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_single_image(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_image");
    group.measurement_time(Duration::from_secs(10));
    
    let img = create_test_image(1920, 1080);
    
    group.bench_function("single_224", |b| {
        b.iter(|| {
            let result = ImageProcessor::preprocess_sync(&img, (224, 224));
            black_box(result)
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_batch_processing,
    benchmark_chunked_processing,
    benchmark_single_image
);
criterion_main!(benches);

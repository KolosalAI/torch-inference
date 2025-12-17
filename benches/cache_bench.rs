use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use torch_inference::cache::Cache;
use serde_json::json;

fn cache_set_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_set");
    
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let cache = Cache::new(size);
            let mut counter = 0;
            
            b.iter(|| {
                let key = format!("key_{}", counter);
                let value = json!({"data": counter});
                counter += 1;
                black_box(cache.set(key, value, 60))
            });
        });
    }
    
    group.finish();
}

fn cache_get_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_get");
    
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let cache = Cache::new(size);
            
            // Pre-populate cache
            for i in 0..size {
                cache.set(format!("key_{}", i), json!(i), 60).ok();
            }
            
            let mut counter = 0;
            b.iter(|| {
                let key = format!("key_{}", counter % size);
                counter += 1;
                black_box(cache.get(&key))
            });
        });
    }
    
    group.finish();
}

fn cache_cleanup_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_cleanup");
    
    for size in [100, 1000, 5000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_with_setup(
                || {
                    let cache = Cache::new(size * 2);
                    // Half expired, half not
                    for i in 0..size {
                        cache.set(format!("expire_{}", i), json!(i), 1).ok();
                    }
                    for i in 0..size {
                        cache.set(format!("keep_{}", i), json!(i), 3600).ok();
                    }
                    std::thread::sleep(std::time::Duration::from_secs(2));
                    cache
                },
                |cache| black_box(cache.cleanup_expired()),
            );
        });
    }
    
    group.finish();
}

criterion_group!(benches, cache_set_benchmark, cache_get_benchmark, cache_cleanup_benchmark);
criterion_main!(benches);

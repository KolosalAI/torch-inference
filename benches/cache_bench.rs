use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Bencher, Criterion};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use torch_inference::cache::Cache;
use torch_inference::core::model_cache::{cache_key, ModelCache};

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

// ── ModelCache Arc<dyn Any> vs serde hot-path comparison ─────────────────────

#[derive(Clone, Serialize, Deserialize)]
struct ClassificationResult {
    score: f32,
    label: String,
}

fn any_hit(b: &mut Bencher) {
    // Pre-populate the cache, then measure warm-cache hit cost.
    let cache = ModelCache::new(4);
    let key = cache_key("model", b"input", b"params");
    let _: ClassificationResult = cache
        .get_or_run(key, || {
            Ok(ClassificationResult {
                score: 0.95,
                label: "cat".to_string(),
            })
        })
        .unwrap();

    b.iter(|| {
        let v: ClassificationResult = cache
            .get_or_run(black_box(key), || {
                Ok(ClassificationResult {
                    score: 0.0,
                    label: String::new(),
                })
            })
            .unwrap();
        black_box(v)
    })
}

fn serde_hit(b: &mut Bencher) {
    // Simulate the old hot path: serialize once, then measure each cache-hit
    // deserialization.
    let value = ClassificationResult {
        score: 0.95,
        label: "cat".to_string(),
    };
    let bytes = Arc::new(serde_json::to_vec(&value).unwrap());
    b.iter(|| {
        let _: ClassificationResult =
            serde_json::from_slice(black_box(&bytes)).unwrap();
        black_box(())
    })
}

fn model_cache_any_vs_serde(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_cache_any_vs_serde");
    group.bench_function("any_hit", any_hit);
    group.bench_function("serde_hit", serde_hit);
    group.finish();
}

criterion_group!(
    benches,
    cache_set_benchmark,
    cache_get_benchmark,
    cache_cleanup_benchmark,
    model_cache_any_vs_serde
);
criterion_main!(benches);

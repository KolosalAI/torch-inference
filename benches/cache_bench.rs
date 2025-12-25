use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, measurement::WallTime, BenchmarkGroup};
use torch_inference::cache::Cache;
use serde_json::json;
use std::sync::{Arc, Mutex};
use std::time::Duration;

mod benchmark_reporter;
use benchmark_reporter::{BenchmarkReporter, BenchmarkResult, SystemInfo};

lazy_static::lazy_static! {
    static ref REPORTER: Arc<Mutex<BenchmarkReporter>> = Arc::new(Mutex::new(
        BenchmarkReporter::new("benches/data")
    ));
}

fn record_result(bench_name: &str, param: &str, elapsed: Duration, iterations: u64) {
    let nanos = elapsed.as_nanos() as f64 / iterations as f64;
    
    let result = BenchmarkResult {
        timestamp: chrono::Utc::now().to_rfc3339(),
        benchmark_name: bench_name.to_string(),
        model_name: None,
        parameter: Some(param.to_string()),
        mean_time_ns: nanos,
        mean_time_ms: nanos / 1_000_000.0,
        std_dev_ns: 0.0,
        median_ns: nanos,
        min_ns: nanos,
        max_ns: nanos,
        sample_count: 100,
        iterations,
        throughput_ops_per_sec: Some(1_000_000_000.0 / nanos),
        system_info: SystemInfo::new(),
    };
    
    if let Ok(mut reporter) = REPORTER.lock() {
        reporter.add_result(result);
    }
}

fn cache_set_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_set");
    
    for size in [100, 1000, 10000].iter() {
        let size_val = *size;
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let cache = Cache::new(size);
            let mut counter = 0;
            
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let key = format!("key_{}", counter);
                    let value = json!({"data": counter});
                    counter += 1;
                    black_box(cache.set(key, value, 60));
                }
                let elapsed = start.elapsed();
                
                // Record result
                record_result("cache_set", &format!("size_{}", size_val), elapsed, iters);
                
                elapsed
            });
        });
    }
    
    group.finish();
}

fn cache_get_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_get");
    
    for size in [100, 1000, 10000].iter() {
        let size_val = *size;
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let cache = Cache::new(size);
            
            // Pre-populate cache
            for i in 0..size {
                cache.set(format!("key_{}", i), json!(i), 60).ok();
            }
            
            let mut counter = 0;
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let key = format!("key_{}", counter % size);
                    counter += 1;
                    black_box(cache.get(&key));
                }
                let elapsed = start.elapsed();
                
                // Record result
                record_result("cache_get", &format!("size_{}", size_val), elapsed, iters);
                
                elapsed
            });
        });
    }
    
    group.finish();
}

fn cache_cleanup_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_cleanup");
    
    for size in [100, 1000, 5000].iter() {
        let size_val = *size;
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut first_run = true;
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
                |cache| {
                    let start = std::time::Instant::now();
                    let result = black_box(cache.cleanup_expired());
                    let elapsed = start.elapsed();
                    
                    // Record only first run to avoid duplicates
                    if first_run {
                        record_result("cache_cleanup", &format!("size_{}", size_val), elapsed, 1);
                        first_run = false;
                    }
                    
                    result
                },
            );
        });
    }
    
    group.finish();
}

fn save_reports(_: &mut Criterion) {
    if let Ok(reporter) = REPORTER.lock() {
        reporter.print_summary();
        if let Err(e) = reporter.save_all("cache_benchmark") {
            eprintln!("Failed to save benchmark reports: {}", e);
        }
    }
}

criterion_group!(benches, cache_set_benchmark, cache_get_benchmark, cache_cleanup_benchmark, save_reports);
criterion_main!(benches);

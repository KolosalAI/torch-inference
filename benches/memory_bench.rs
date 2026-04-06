//! Memory allocation benchmarks: BufferPool reuse vs raw Vec<u8> allocation.
//! Measures allocation rate at 1KB, 64KB, 1MB sizes plus sustained 100-request load.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

/// Minimal buffer pool using thread_local storage to avoid DashMap overhead in benchmarks.
/// Mirrors the production BufferPool bucket strategy.
mod local_pool {
    use std::cell::RefCell;
    use std::collections::HashMap;

    const BUCKETS: &[usize] = &[1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576];

    fn bucket_for(size: usize) -> usize {
        *BUCKETS.iter().find(|&&b| b >= size).unwrap_or(&size)
    }

    thread_local! {
        static POOL: RefCell<HashMap<usize, Vec<Vec<u8>>>> = RefCell::new(HashMap::new());
    }

    pub fn acquire(min_size: usize) -> Vec<u8> {
        let bucket = bucket_for(min_size);
        POOL.with(|p| {
            p.borrow_mut()
                .get_mut(&bucket)
                .and_then(|v| v.pop())
                .unwrap_or_else(|| vec![0u8; bucket])
        })
    }

    pub fn release(mut buf: Vec<u8>) {
        let bucket = bucket_for(buf.len());
        if buf.capacity() < bucket {
            return; // wrong size, drop
        }
        buf.clear();
        POOL.with(|p| {
            let mut pool = p.borrow_mut();
            let slot = pool.entry(bucket).or_default();
            if slot.len() < 32 {
                slot.push(buf);
            }
            // otherwise drop
        });
    }
}

fn bench_pooled_vs_raw(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    group.measurement_time(Duration::from_secs(5));

    for (label, size) in [("1kb", 1_024usize), ("64kb", 65_536), ("1mb", 1_048_576)] {
        group.bench_with_input(BenchmarkId::new("pooled", label), &size, |b, &sz| {
            b.iter(|| {
                let buf = local_pool::acquire(sz);
                let len = buf.len();
                local_pool::release(buf);
                len
            });
        });

        group.bench_with_input(BenchmarkId::new("raw_alloc", label), &size, |b, &sz| {
            b.iter(|| {
                let buf = vec![0u8; sz];
                buf.len()
            });
        });
    }
    group.finish();
}

fn bench_sustained_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_sustained_load");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(50);

    // Simulate 100 sequential image requests, each needing a 64KB scratch buffer
    group.bench_function("pooled_100_requests", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for _ in 0..100 {
                let buf = local_pool::acquire(65_536);
                total += buf.len();
                local_pool::release(buf);
            }
            total
        });
    });

    group.bench_function("raw_alloc_100_requests", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for _ in 0..100 {
                let buf = vec![0u8; 65_536];
                total += buf.len();
                drop(buf);
            }
            total
        });
    });

    group.finish();
}

criterion_group!(memory_benches, bench_pooled_vs_raw, bench_sustained_load);
criterion_main!(memory_benches);

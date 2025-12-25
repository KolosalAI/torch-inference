# Concurrency Optimization Guide

**Date:** 2024-12-25  
**Issue:** Throughput plateaus at 64 concurrency, drops slightly at 128+  
**Goal:** Improve concurrent throughput beyond 64 concurrent requests

## Problem Analysis

### Current Performance
```
Concurrency 1:    77 img/sec   (baseline)
Concurrency 64:   364 img/sec  (4.7x speedup) ✅ PEAK
Concurrency 128:  360 img/sec  (plateau) ⚠️
Concurrency 256:  354 img/sec  (slight drop) ⚠️
Concurrency 1024: 334 img/sec  (8% drop) ❌
```

### Root Causes

#### 1. **Thread Pool Exhaustion** 
```rust
// Current code uses spawn_blocking
tokio::task::spawn_blocking(move || {
    preprocess_image(&img, target_size)
}).await
```

**Problem:**
- tokio's blocking thread pool has default limit (512 threads)
- Above 64 concurrent, threads queue waiting for pool slots
- Context switching overhead increases dramatically

#### 2. **CPU Core Saturation**
- M4 has 10 cores (6 performance + 4 efficiency)
- Peak efficiency at 6-12 concurrent threads per core
- 64 concurrent ≈ 6x per core (optimal)
- 256 concurrent ≈ 25x per core (excessive)

#### 3. **Memory Allocation Contention**
- Each task allocates 150KB tensor
- 1024 concurrent = 154MB active memory
- Global allocator lock contention
- Cache thrashing

---

## Solution 1: Bounded Concurrency Limiter ⭐ RECOMMENDED

### Implementation

```rust
use tokio::sync::Semaphore;
use std::sync::Arc;

pub struct ConcurrencyLimiter {
    semaphore: Arc<Semaphore>,
    max_concurrent: usize,
}

impl ConcurrencyLimiter {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        }
    }
    
    pub async fn process_image<F, T>(&self, f: F) -> T
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        // Acquire permit (blocks if at limit)
        let permit = self.semaphore.acquire().await.unwrap();
        
        // Execute work
        let result = tokio::task::spawn_blocking(f).await.unwrap();
        
        // Release permit automatically on drop
        drop(permit);
        
        result
    }
}

// Usage in server
let limiter = Arc::new(ConcurrencyLimiter::new(64)); // Optimal limit

async fn handle_request(limiter: Arc<ConcurrencyLimiter>, image: RgbImage) -> Vec<f32> {
    limiter.process_image(move || {
        preprocess_image(&image, (224, 224))
    }).await
}
```

### Benefits
✅ Prevents thread pool exhaustion  
✅ Maintains peak throughput (364 img/sec)  
✅ Graceful degradation under extreme load  
✅ Simple to implement  

### Expected Performance
```
Concurrent requests: 1000
Max concurrent:      64
Throughput:          364 img/sec
Time:                2.75 seconds
```

---

## Solution 2: Dedicated Thread Pool

### Implementation

```rust
use rayon::ThreadPoolBuilder;
use std::sync::Arc;

pub struct ImageProcessor {
    thread_pool: Arc<rayon::ThreadPool>,
}

impl ImageProcessor {
    pub fn new(num_threads: usize) -> Self {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("img-worker-{}", i))
            .build()
            .unwrap();
            
        Self {
            thread_pool: Arc::new(thread_pool),
        }
    }
    
    pub async fn process_batch(&self, images: Vec<RgbImage>) -> Vec<Vec<f32>> {
        let pool = self.thread_pool.clone();
        
        tokio::task::spawn_blocking(move || {
            pool.install(|| {
                images.par_iter()
                    .map(|img| preprocess_image(img, (224, 224)))
                    .collect()
            })
        }).await.unwrap()
    }
}

// Usage
let processor = Arc::new(ImageProcessor::new(10)); // Match CPU cores
```

### Benefits
✅ Full control over thread count  
✅ Work stealing for load balancing  
✅ No spawn_blocking overhead  
✅ Better CPU cache utilization  

### Expected Performance
```
Cores: 10
Throughput: 400-450 img/sec (10-25% faster)
```

---

## Solution 3: Batch Processing Pipeline

### Implementation

```rust
use tokio::sync::mpsc;

pub struct BatchProcessor {
    batch_size: usize,
    sender: mpsc::Sender<ProcessRequest>,
}

struct ProcessRequest {
    image: RgbImage,
    response: tokio::sync::oneshot::Sender<Vec<f32>>,
}

impl BatchProcessor {
    pub fn new(batch_size: usize, num_workers: usize) -> Self {
        let (tx, mut rx) = mpsc::channel::<ProcessRequest>(1000);
        
        // Spawn worker that processes in batches
        tokio::spawn(async move {
            let mut batch = Vec::with_capacity(batch_size);
            
            loop {
                // Collect batch
                while batch.len() < batch_size {
                    match rx.try_recv() {
                        Ok(req) => batch.push(req),
                        Err(_) => break,
                    }
                }
                
                if batch.is_empty() {
                    tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                    continue;
                }
                
                // Process batch in parallel
                let results: Vec<_> = tokio::task::spawn_blocking(move || {
                    batch.iter()
                        .map(|req| preprocess_image(&req.image, (224, 224)))
                        .collect::<Vec<_>>()
                }).await.unwrap();
                
                // Send responses
                for (req, result) in batch.drain(..).zip(results.into_iter()) {
                    let _ = req.response.send(result);
                }
            }
        });
        
        Self { batch_size, sender: tx }
    }
    
    pub async fn process(&self, image: RgbImage) -> Vec<f32> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.sender.send(ProcessRequest {
            image,
            response: tx,
        }).await.unwrap();
        rx.await.unwrap()
    }
}
```

### Benefits
✅ Natural batching reduces overhead  
✅ Better throughput under load  
✅ Lower latency variance  

### Expected Performance
```
Batch size: 32
Throughput: 380-420 img/sec (5-15% faster)
Latency: More predictable
```

---

## Solution 4: CPU Affinity & Priority

### Implementation

```rust
use std::thread;

pub fn configure_worker_threads() {
    // Set thread pool size to match CPU cores
    std::env::set_var("TOKIO_WORKER_THREADS", "10");
    
    // Use rayon with CPU core count
    rayon::ThreadPoolBuilder::new()
        .num_threads(10)
        .build_global()
        .unwrap();
}

// In main.rs
fn main() {
    configure_worker_threads();
    
    // Build tokio runtime with optimized settings
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(10)
        .max_blocking_threads(64) // Limit blocking pool
        .thread_name("tokio-worker")
        .enable_all()
        .build()
        .unwrap();
    
    runtime.block_on(async {
        // Your server code
    });
}
```

### Benefits
✅ Prevents oversubscription  
✅ Better CPU cache utilization  
✅ Consistent performance  

---

## Solution 5: Memory Pool for Tensors

### Implementation

```rust
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

pub struct TensorPool {
    pool: Arc<Mutex<VecDeque<Vec<f32>>>>,
    capacity: usize,
}

impl TensorPool {
    pub fn new(capacity: usize, tensor_size: usize) -> Self {
        let mut pool = VecDeque::new();
        for _ in 0..capacity {
            pool.push_back(Vec::with_capacity(tensor_size));
        }
        
        Self {
            pool: Arc::new(Mutex::new(pool)),
            capacity,
        }
    }
    
    pub fn acquire(&self) -> Vec<f32> {
        self.pool.lock().unwrap()
            .pop_front()
            .unwrap_or_else(|| Vec::new())
    }
    
    pub fn release(&self, mut tensor: Vec<f32>) {
        tensor.clear();
        let mut pool = self.pool.lock().unwrap();
        if pool.len() < self.capacity {
            pool.push_back(tensor);
        }
    }
}

// Usage
let pool = Arc::new(TensorPool::new(128, 150_528)); // Pre-allocate 128 tensors
```

### Benefits
✅ Reduces allocation overhead  
✅ Less GC pressure  
✅ 10-20% faster under high load  

---

## Recommended Implementation Strategy

### Phase 1: Quick Win (1 hour)
**Implement Solution 1: Bounded Concurrency Limiter**

```rust
// In src/api/inference.rs or similar
lazy_static! {
    static ref CONCURRENCY_LIMITER: Arc<ConcurrencyLimiter> = 
        Arc::new(ConcurrencyLimiter::new(64));
}

pub async fn process_image_request(image: RgbImage) -> Vec<f32> {
    CONCURRENCY_LIMITER.process_image(move || {
        preprocess_image(&image, (224, 224))
    }).await
}
```

**Expected Result:**
- Throughput: 364 img/sec (maintained)
- No degradation at high concurrency
- Queue builds up gracefully

### Phase 2: Performance Boost (4 hours)
**Add Solution 2: Dedicated Thread Pool**

```toml
# Cargo.toml
[dependencies]
rayon = "1.8"
```

```rust
// Use rayon for CPU-bound work
use rayon::prelude::*;

pub fn process_batch_sync(images: Vec<RgbImage>) -> Vec<Vec<f32>> {
    images.par_iter()
        .map(|img| preprocess_image(img, (224, 224)))
        .collect()
}
```

**Expected Result:**
- Throughput: 400-450 img/sec (+10-25%)
- Better CPU utilization
- Lower latency variance

### Phase 3: Production Ready (8 hours)
**Add Solution 3: Batch Processing + Solution 5: Memory Pool**

**Expected Result:**
- Throughput: 450-500 img/sec (+25-35%)
- Predictable latency
- Memory efficient

---

## Performance Comparison

| Strategy | Throughput | Latency (p99) | Complexity | Recommended |
|----------|------------|---------------|------------|-------------|
| **Current (unlimited)** | 364 → 334 img/sec | Variable | Low | ❌ No |
| **Solution 1 (Bounded)** | 364 img/sec | ~180ms | Low | ✅ Yes (Quick win) |
| **Solution 2 (Thread Pool)** | 400-450 img/sec | ~150ms | Medium | ✅ Yes (Best performance) |
| **Solution 3 (Batching)** | 380-420 img/sec | ~120ms | Medium | ⭐ Yes (Best latency) |
| **Solution 4 (CPU Affinity)** | +5-10% | Same | Low | ✅ Yes (Easy add-on) |
| **Solution 5 (Memory Pool)** | +10-20% | Same | Medium | ⭐ Yes (Under high load) |
| **All Combined** | **500-550 img/sec** | **~100ms** | High | ⭐⭐⭐ Best |

---

## Monitoring & Tuning

### Key Metrics to Track

```rust
use prometheus::{IntGauge, Histogram};

lazy_static! {
    static ref ACTIVE_TASKS: IntGauge = IntGauge::new(
        "active_preprocessing_tasks",
        "Number of active image preprocessing tasks"
    ).unwrap();
    
    static ref QUEUE_DEPTH: IntGauge = IntGauge::new(
        "preprocessing_queue_depth",
        "Number of tasks waiting in queue"
    ).unwrap();
    
    static ref PROCESSING_TIME: Histogram = Histogram::new(
        "image_preprocessing_duration_seconds",
        "Time spent preprocessing images"
    ).unwrap();
}
```

### Tuning Guidelines

1. **Adjust concurrency limit based on CPU cores:**
   ```
   Optimal limit = CPU cores × 6-8
   M4 (10 cores) = 60-80 concurrent
   ```

2. **Monitor queue depth:**
   ```
   If queue > 100: Increase worker threads
   If queue = 0 often: Decrease concurrency limit
   ```

3. **Watch CPU utilization:**
   ```
   Target: 80-90% CPU usage
   If < 70%: Increase concurrency
   If > 95%: Decrease concurrency
   ```

---

## Conclusion

### Quick Summary

**Problem:** Throughput plateaus at 64 concurrency  
**Root Cause:** Thread pool exhaustion + CPU oversubscription  
**Solution:** Bounded concurrency + dedicated thread pool  
**Expected Improvement:** 364 → 500+ img/sec (+37%)  

### Implementation Priority

1. ⭐⭐⭐ **Bounded Concurrency Limiter** (1 hour, +0% throughput, prevents degradation)
2. ⭐⭐⭐ **Dedicated Thread Pool** (4 hours, +10-25% throughput)
3. ⭐⭐ **Batch Processing** (6 hours, +5-15% throughput, better latency)
4. ⭐ **Memory Pool** (4 hours, +10-20% under high load)
5. ⭐ **CPU Affinity** (2 hours, +5-10%)

### Next Steps

1. Implement Solution 1 (bounded concurrency) immediately
2. Benchmark and verify no degradation
3. Add Solution 2 (thread pool) for performance boost
4. Consider Solutions 3-5 based on production needs

---

**Generated:** 2024-12-25  
**Status:** Implementation guide ready  
**Priority:** High - Prevents performance degradation under load

# Components Guide

Deep dive into each component of Torch Inference Server.

## Component Overview

```
torch-inference/
├── API Layer           - HTTP endpoints and handlers
├── Middleware          - Cross-cutting concerns
├── Authentication      - JWT-based auth
├── Cache System        - Multi-level caching
├── Batch Processor     - Dynamic batching
├── Deduplicator        - Request deduplication
├── Resilience          - Circuit breakers, bulkheads
├── Model Management    - Model lifecycle
├── Inference Engine    - ML execution
├── Worker Pool         - Thread management
├── Tensor Pool         - Memory management
├── Monitoring          - Metrics and health
├── Telemetry           - Logging and tracing
└── Security            - Validation and sanitization
```

## 1. Cache System (`src/cache.rs`)

### Purpose
High-performance in-memory caching with LRU eviction and TTL support.

### Architecture

```rust
pub struct Cache {
    store: Arc<DashMap<String, CacheEntry>>,
    config: CacheConfig,
    stats: Arc<RwLock<CacheStats>>,
}

pub struct CacheEntry {
    value: Vec<u8>,
    created_at: Instant,
    ttl: Duration,
    access_count: AtomicU64,
}

pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub size: usize,
    pub memory_bytes: usize,
    pub evictions: u64,
}
```

### Features
- **LRU Eviction**: Least Recently Used policy
- **TTL Expiration**: Time-based expiration
- **Concurrent Access**: Lock-free using DashMap
- **Statistics**: Real-time hit rate tracking
- **Memory Management**: Automatic cleanup

### Usage

```rust
use torch_inference::cache::Cache;
use std::time::Duration;

// Create cache (100MB)
let cache = Cache::new(100);

// Store value (60s TTL)
cache.set("request_123", result.clone(), Duration::from_secs(60));

// Retrieve value
if let Some(cached) = cache.get("request_123") {
    println!("Cache hit!");
} else {
    println!("Cache miss");
}

// Get statistics
let stats = cache.stats();
println!("Hit rate: {:.2}%", stats.hit_rate * 100.0);
```

### Performance
- **Hit Rate**: 80-85% typical
- **Lookup Time**: O(1) average
- **Thread-Safe**: Yes (lock-free)
- **Memory Efficient**: Automatic eviction

### Configuration

```toml
[performance]
enable_caching = true
cache_size_mb = 2048
cache_ttl_seconds = 3600
```

## 2. Batch Processor (`src/batch.rs`)

### Purpose
Dynamic request batching for improved throughput and GPU utilization.

### Architecture

```rust
pub struct BatchProcessor<T> {
    queue: Arc<Mutex<VecDeque<BatchItem<T>>>>,
    config: BatchConfig,
    stats: Arc<RwLock<BatchStats>>,
}

pub struct BatchItem<T> {
    id: String,
    data: T,
    priority: u8,
    response_tx: Sender<Result<Vec<u8>>>,
}

pub struct BatchConfig {
    pub max_batch_size: usize,
    pub min_batch_size: usize,
    pub timeout_ms: u64,
    pub adaptive_timeout: bool,
    pub enable_priority: bool,
}
```

### Batching Strategy

**Adaptive Timeout**:
```
Queue Depth    Timeout
-----------    -------
0-2 items      100ms
3-5 items      50ms
6-10 items     25ms
11+ items      12.5ms
```

**Priority Levels**:
- `0` - Low priority
- `1` - Normal priority (default)
- `2` - High priority
- `3` - Critical priority

### Usage

```rust
use torch_inference::batch::{BatchProcessor, BatchConfig};

// Create processor
let processor = BatchProcessor::new(BatchConfig {
    max_batch_size: 32,
    adaptive_timeout: true,
    enable_priority: true,
    ..Default::default()
});

// Add items
let (tx, rx) = oneshot::channel();
processor.add_with_priority("item_1", data, 2, tx).await;

// Wait for result
let result = rx.await.unwrap();
```

### Performance
- **Throughput Increase**: 2-4x
- **Latency**: Minimal overhead (<10ms)
- **Efficiency**: Reduced GPU context switches

### Configuration

```toml
[batch]
max_batch_size = 32
adaptive_batch_timeout = true
enable_priority_batching = true
```

## 3. Request Deduplicator (`src/dedup.rs`)

### Purpose
Eliminate redundant requests by coalescing identical in-flight requests.

### Architecture

```rust
pub struct RequestDeduplicator {
    inflight: Arc<DashMap<String, InflightRequest>>,
}

struct InflightRequest {
    waiters: Vec<Sender<Result<Vec<u8>>>>,
    started_at: Instant,
}
```

### Algorithm

```
1. Hash request → request_id
2. Check if request_id in inflight map
3. If found:
   - Add to waiters
   - Return when original completes
4. If not found:
   - Add to inflight map
   - Process request
   - Broadcast to all waiters
   - Remove from inflight map
```

### Usage

```rust
use torch_inference::dedup::RequestDeduplicator;

let dedup = RequestDeduplicator::new();

// Deduplicate request
let result = dedup.deduplicate("request_hash", || async {
    // This only runs once for duplicate requests
    expensive_operation().await
}).await;
```

### Benefits
- Eliminates redundant computation
- Reduces resource usage
- Faster response for duplicates

## 4. Circuit Breaker (`src/resilience/circuit_breaker.rs`)

### Purpose
Prevent cascading failures and provide graceful degradation.

### States

```
         Failures ≥ Threshold
Closed ─────────────────────────► Open
  ▲                                 │
  │                                 │ Timeout
  │                                 ▼
  └────────────────────────── Half-Open
         Success ≥ Threshold
```

### Architecture

```rust
pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitBreakerState>>,
    config: CircuitBreakerConfig,
    stats: Arc<RwLock<CircuitBreakerStats>>,
}

pub enum CircuitBreakerState {
    Closed,     // Normal operation
    Open,       // Rejecting requests
    HalfOpen,   // Testing recovery
}
```

### Usage

```rust
use torch_inference::resilience::CircuitBreaker;

let cb = CircuitBreaker::new(CircuitBreakerConfig {
    failure_threshold: 5,
    timeout_duration: Duration::from_secs(30),
    success_threshold: 2,
});

// Execute with circuit breaker
match cb.call(|| async {
    risky_operation().await
}).await {
    Ok(result) => handle_success(result),
    Err(CircuitBreakerError::Open) => handle_degraded(),
    Err(e) => handle_error(e),
}
```

### Configuration

```toml
[guard]
enable_circuit_breaker = true
failure_threshold = 5
timeout_seconds = 30
success_threshold = 2
```

## 5. Bulkhead (`src/resilience/bulkhead.rs`)

### Purpose
Resource isolation to prevent exhaustion.

### Architecture

```rust
pub struct Bulkhead {
    semaphore: Arc<Semaphore>,
    config: BulkheadConfig,
    stats: Arc<RwLock<BulkheadStats>>,
}
```

### Usage

```rust
use torch_inference::resilience::Bulkhead;

let bulkhead = Bulkhead::new(BulkheadConfig {
    max_concurrent: 100,
    max_wait_duration: Duration::from_secs(5),
});

// Acquire permit
let permit = bulkhead.acquire().await?;

// Execute operation
let result = operation().await;

// Permit automatically released on drop
drop(permit);
```

## 6. Model Manager (`src/models/manager.rs`)

### Purpose
Manage model lifecycle: loading, caching, unloading.

### Features
- **Lazy Loading**: Load models on-demand
- **Model Pooling**: Multiple instances per model
- **Hot-Swapping**: Update models without downtime
- **Version Control**: Manage model versions

### Usage

```rust
use torch_inference::models::ModelManager;

let manager = ModelManager::new(config);

// Load model
manager.load_model("resnet50", "1.0").await?;

// Get model instance
let model = manager.get_model("resnet50").await?;

// Use model
let output = model.infer(input).await?;

// Unload model
manager.unload_model("resnet50").await?;
```

## 7. Inference Engine (`src/core/engine.rs`)

### Purpose
Execute ML model inference across multiple backends.

### Supported Backends

**PyTorch (tch-rs)**:
```rust
let engine = InferenceEngine::new(EngineConfig {
    backend: Backend::PyTorch,
    device: Device::Cuda(0),
    use_fp16: true,
});
```

**ONNX Runtime**:
```rust
let engine = InferenceEngine::new(EngineConfig {
    backend: Backend::Onnx,
    device: Device::Cpu,
    execution_provider: ExecutionProvider::Cpu,
});
```

### Features
- Multi-backend support
- Device management (CPU/GPU)
- Precision control (FP32/FP16)
- Batch inference
- Model compilation

## 8. Worker Pool (`src/worker_pool.rs`)

### Purpose
Managed thread pool with auto-scaling.

### Features
- **Auto-Scaling**: Scale based on load
- **Min/Max Workers**: Configurable limits
- **Work Queue**: Fair task distribution
- **Load Balancing**: Round-robin

### Usage

```rust
use torch_inference::worker_pool::WorkerPool;

let pool = WorkerPool::new(WorkerPoolConfig {
    min_workers: 2,
    max_workers: 16,
    enable_auto_scaling: true,
});

// Submit work
let result = pool.execute(|| {
    expensive_computation()
}).await?;
```

## 9. Tensor Pool (`src/tensor_pool.rs`)

### Purpose
Reusable tensor memory to reduce allocation overhead.

### Architecture

```rust
pub struct TensorPool {
    pools: DashMap<TensorShape, Vec<Tensor>>,
    stats: Arc<RwLock<PoolStats>>,
}
```

### Benefits
- **50-70% faster** allocation
- **95%+ reuse rate**
- Reduced memory fragmentation

### Usage

```rust
use torch_inference::tensor_pool::TensorPool;

let pool = TensorPool::new(500);

// Acquire tensor
let tensor = pool.acquire(&shape).await;

// Use tensor
process(tensor);

// Return to pool (automatic on drop)
```

## 10. Monitor (`src/monitor.rs`)

### Purpose
Real-time metrics and observability.

### Metrics

**Request Metrics**:
- Count
- Latency (min/max/avg/p95/p99)
- Throughput (req/s)
- Error rate

**Endpoint Metrics**:
- Per-endpoint statistics
- Request distribution

**System Metrics**:
- Memory usage
- Cache statistics
- Queue depth

### Usage

```rust
use torch_inference::monitor::Monitor;

let monitor = Monitor::new();

// Record request
let start = Instant::now();
let result = handle_request().await;
let latency = start.elapsed().as_millis();

monitor.record_request(
    "/api/classify",
    latency,
    result.is_ok()
);

// Get statistics
let stats = monitor.get_stats();
println!("Avg latency: {}ms", stats.avg_latency_ms);
```

## 11. Security (`src/security/`)

### Validation (`validation.rs`)

```rust
pub fn validate_image_size(size: usize) -> Result<()> {
    if size > MAX_IMAGE_SIZE {
        return Err(ValidationError::SizeExceeded);
    }
    Ok(())
}
```

### Sanitization (`sanitizer.rs`)

```rust
pub fn sanitize_filename(filename: &str) -> String {
    filename
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '.' || *c == '_')
        .collect()
}
```

## Component Interactions

```
Request → Middleware → Cache Check
                         ↓ miss
                      Deduplicator
                         ↓ new
                    Batch Queue
                         ↓
                 Circuit Breaker
                         ↓
                  Bulkhead Permit
                         ↓
                   Worker Pool
                         ↓
                  Tensor Pool
                         ↓
                Inference Engine
                         ↓
                      Cache Store
                         ↓
                     Response
```

## Performance Characteristics

| Component | Operation | Time Complexity | Thread-Safe |
|-----------|-----------|-----------------|-------------|
| Cache | Get/Set | O(1) | Yes |
| Batch | Add | O(1) | Yes |
| Dedup | Check | O(1) | Yes |
| Circuit Breaker | Call | O(1) | Yes |
| Bulkhead | Acquire | O(1) | Yes |
| Model Manager | Get | O(1) | Yes |
| Worker Pool | Execute | O(1) | Yes |
| Tensor Pool | Acquire | O(1) | Yes |
| Monitor | Record | O(1) | Yes |

## Testing

Each component has comprehensive unit tests:
- Cache: 38 tests
- Batch: 28 tests
- Monitor: 28 tests
- Circuit Breaker: 10 tests
- Bulkhead: 6 tests
- Deduplicator: 9 tests

See [Testing Guide](TESTING.md) for details.

---

**Next**: See [Architecture Overview](ARCHITECTURE.md) for system-level design.

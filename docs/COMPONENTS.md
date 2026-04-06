# Components

Per-module developer reference. Each section covers the public API, internal design, and a Mermaid diagram.

---

## `cache` — `src/cache.rs`

Dual-store in-memory cache: a JSON value store and a zero-copy byte-slice store, both backed by `DashMap` with approximate-LRU eviction.

### Key Structs

```rust
pub struct Cache {
    data: DashMap<String, CacheEntry>,
    bytes_data: DashMap<String, BytesCacheEntry>,
    max_size: usize,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    insertion_counter: AtomicU64,
}
pub struct BytesCacheEntry { pub data: Arc<Bytes>, pub timestamp: u64, pub ttl: u64 }
pub struct CacheEntry { pub data: Value, pub timestamp: u64, pub ttl: u64,
                        pub last_access: u64, pub access_count: u64, pub insertion_order: u64 }
```

### Diagram

```mermaid
classDiagram
    class Cache {
        -DashMap~String,CacheEntry~ data
        -DashMap~String,BytesCacheEntry~ bytes_data
        -usize max_size
        -AtomicU64 hits
        -AtomicU64 misses
        -AtomicU64 evictions
        -AtomicU64 insertion_counter
        +new(max_size) Cache
        +get(key) Option~CacheEntry~
        +set(key, value, ttl)
        +get_bytes(key) Option~BytesCacheEntry~
        +set_bytes(key, data, ttl)
        +evict_if_needed()
        +hit_rate() f64
    }
    class CacheEntry {
        +Value data
        +u64 timestamp
        +u64 ttl
        +u64 last_access
        +u64 access_count
        +u64 insertion_order
        +is_expired() bool
        +is_expired_at(now) bool
    }
    class BytesCacheEntry {
        +Arc~Bytes~ data
        +u64 timestamp
        +u64 ttl
        +is_expired() bool
        +is_expired_at(now) bool
    }
    Cache "1" --> "*" CacheEntry : data
    Cache "1" --> "*" BytesCacheEntry : bytes_data
```

### Usage

```rust
let cache = Cache::new(2048); // max 2048 entries

// Store inference result (zero-copy path)
let bytes = Arc::new(Bytes::from(serde_json::to_vec(&result)?));
cache.set_bytes("model:sha256:abc", bytes, 3600);

// Read — Arc pointer increment only (~5 ns)
if let Some(entry) = cache.get_bytes("model:sha256:abc") {
    return Ok(HttpResponse::Ok().body(entry.data.as_ref().clone()));
}
```

**Eviction**: approximate LRU — random sample → evict expired first, then oldest `insertion_order`. No global lock required.

---

## `batch` — `src/batch.rs`

Adaptive request batcher. Groups concurrent requests into GPU-efficient batches; timeout shrinks as queue depth grows.

### Key Structs

```rust
pub struct BatchProcessor {
    max_batch_size: usize,
    min_batch_size: usize,
    batch_timeout_ms: u64,
    adaptive_timeout_enabled: bool,
    current_batch: Arc<Mutex<Vec<BatchRequest>>>,   // parking_lot
    processed_batches: AtomicU64,
    total_latency_ms: AtomicU64,
    queue_depth: AtomicUsize,
}
pub struct BatchRequest {
    pub id: String,
    pub model_name: String,
    pub inputs: Vec<Value>,
    pub priority: i32,
    pub timestamp: Instant,
}
```

### Diagram

```mermaid
flowchart TD
    In["add_request(BatchRequest)"] --> Lock["parking_lot::Mutex::lock()"]
    Lock --> Append["push to current_batch"]
    Append --> Size{"len ≥ max_batch_size?"}
    Size -->|yes| Full["return Ok(true) — trigger immediate flush"]
    Size -->|no| Depth["queue_depth.store(len, Relaxed)"]
    Depth --> Adaptive{"adaptive_timeout_enabled?"}
    Adaptive -->|yes| Calc["timeout = base_ms / (1 + queue_depth)"]
    Adaptive -->|no| Fixed["timeout = batch_timeout_ms"]
    Calc --> Wait["should_process_batch() polls age of oldest entry"]
    Fixed --> Wait
    Wait -->|"age ≥ timeout OR full"| Drain["drain Vec, release Mutex"]
    Drain --> Dispatch["send batch to InferenceEngine"]
    Dispatch --> Stats["processed_batches += 1\ntotal_latency_ms += elapsed"]
```

### Usage

```rust
let bp = BatchProcessor::new(32, 50)   // max 32, 50 ms base timeout
    .with_adaptive_batching(true)
    .with_min_batch_size(1);

let req = BatchRequest {
    id: uuid::Uuid::new_v4().to_string(),
    model_name: "resnet50".into(),
    inputs: vec![json!({"image": base64_data})],
    priority: 1,
    timestamp: Instant::now(),
};
let is_full = bp.add_request(req)?;
if is_full || bp.should_process_batch() {
    let batch = bp.take_batch();
    // dispatch batch to InferenceEngine
}
```

---

## `dedup` — `src/dedup.rs`

Deduplicates identical in-flight requests using an LRU cache keyed on FNV-1a hashes. Results stored behind `Arc<Value>` for O(1) clone on cache hit.

### Key Structs

```rust
pub struct RequestDeduplicator {
    cache: Mutex<LruCache<String, DeduplicationEntry>>,  // parking_lot
}
pub struct DeduplicationEntry {
    pub result: Arc<Value>,
    pub timestamp: u64,
    pub ttl: u64,
}
```

### Diagram

```mermaid
flowchart LR
    Req["infer(model, inputs)"] --> Key["generate_key(model, inputs)<br/>FNV-1a(canonical_json) + epoch_window"]
    Key --> Lookup["LruCache::get(key)"]
    Lookup -->|"HIT + not expired"| Return["return Arc::clone(result) — O(1)"]
    Lookup -->|"MISS or expired"| Exec["execute inference"]
    Exec --> Store["LruCache::put(key, DeduplicationEntry{Arc::new(result)})"]
    Store --> Return2["return result"]

    note["FNV-1a replaces SHA-256<br/>~20× faster for short strings<br/>No cryptographic properties needed"]
```

### Usage

```rust
let dedup = RequestDeduplicator::new(1024); // LRU capacity
let key = dedup.generate_key("resnet50", &inputs_json);
if let Some(entry) = dedup.get(&key) {
    return Ok((*entry.result).clone());
}
let result = engine.infer("resnet50", &inputs_json).await?;
dedup.insert(key, Arc::new(result.clone()));
```

---

## `resilience/circuit_breaker` — `src/resilience/circuit_breaker.rs`

Prevents cascading failures. State machine: `Closed → Open → HalfOpen → Closed`. State lock released before calling `f()` so concurrent callers are not serialised.

### Key Structs

```rust
pub struct CircuitBreaker {
    state: parking_lot::Mutex<CircuitState>,
    failure_count: AtomicU32,
    success_count: AtomicU32,
    last_failure_time: parking_lot::Mutex<Option<Instant>>,
    config: CircuitBreakerConfig,
}
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,   // default 5
    pub success_threshold: u32,   // default 2
    pub timeout: Duration,        // default 60 s
}
pub enum CircuitState { Closed, Open, HalfOpen }
```

### Diagram

```mermaid
classDiagram
    class CircuitBreaker {
        -Mutex~CircuitState~ state
        -AtomicU32 failure_count
        -AtomicU32 success_count
        -Mutex~Option~Instant~~ last_failure_time
        -CircuitBreakerConfig config
        +new(config) CircuitBreaker
        +call(f) Result~T,String~
        -on_success()
        -on_failure()
    }
    class CircuitBreakerConfig {
        +u32 failure_threshold
        +u32 success_threshold
        +Duration timeout
    }
    class CircuitState {
        <<enumeration>>
        Closed
        Open
        HalfOpen
    }
    CircuitBreaker --> CircuitBreakerConfig
    CircuitBreaker --> CircuitState
```

### Usage

```rust
let cb = CircuitBreaker::new(CircuitBreakerConfig {
    failure_threshold: 5,
    success_threshold: 2,
    timeout: Duration::from_secs(30),
});

match cb.call(|| backend.run(input)) {
    Ok(result) => Ok(result),
    Err(e) if e == "Circuit breaker is open" => Err(InferenceError::InternalError(e)),
    Err(e) => Err(InferenceError::InferenceFailed(e)),
}
```

---

## `resilience/bulkhead` — `src/resilience/bulkhead.rs`

Caps concurrent in-flight operations via a Tokio `Semaphore`. Non-blocking: returns `Err` immediately when at capacity (no queue).

### Key Structs

```rust
pub struct Bulkhead {
    semaphore: Arc<Semaphore>,
    config: BulkheadConfig,
}
pub struct BulkheadConfig {
    pub max_concurrent: usize,  // default 100
    pub queue_size: usize,      // default 1000 (informational)
}
```

### Diagram

```mermaid
flowchart LR
    Req["acquire_permit()"] --> Try["Semaphore::try_acquire_owned()"]
    Try -->|"Ok(permit)"| Hold["OwnedSemaphorePermit held by caller"]
    Try -->|"Err(_)"| Reject["Err('Bulkhead at capacity')"]
    Hold -->|"drop(permit)"| Release["permit returns to Semaphore"]
```

### Usage

```rust
let bulkhead = Bulkhead::new(BulkheadConfig { max_concurrent: 50, queue_size: 500 });

let permit = bulkhead.acquire_permit().await
    .map_err(|_| InferenceError::InternalError("overloaded".into()))?;
let result = engine.infer(model, input).await;
drop(permit); // explicit or via scope
```

---

## `resilience/retry` — `src/resilience/retry.rs`

Exponential backoff with optional jitter. Multiplier doubles base delay each attempt up to `max_delay`.

### Key Struct

```rust
pub struct RetryPolicy {
    pub max_retries: usize,      // default 3
    pub base_delay: Duration,    // default 100 ms
    pub max_delay: Duration,     // default 30 s
    pub multiplier: f64,         // default 2.0
    pub jitter: bool,            // default true
}
```

### Diagram

```mermaid
flowchart TD
    Start["execute(f)"] --> Call["f().await"]
    Call -->|"Ok(result)"| Done["return Ok(result)"]
    Call -->|"Err(e)"| Check{"attempt < max_retries?"}
    Check -->|"no"| Fail["return Err(e)"]
    Check -->|"yes"| Delay["delay = min(base * multiplier^attempt, max_delay)"]
    Delay --> Jitter{"jitter?"}
    Jitter -->|"yes"| Rand["delay *= rand(0.5..1.5)"]
    Jitter -->|"no"| Sleep["tokio::time::sleep(delay)"]
    Rand --> Sleep
    Sleep --> Retry["attempt += 1"]
    Retry --> Call
```

### Usage

```rust
let policy = RetryPolicy::with_delays(3,
    Duration::from_millis(100), Duration::from_secs(5));
let result = policy.execute(|| async { risky_call().await }).await?;
```

---

## `resilience/token_bucket` — `src/resilience/token_bucket.rs`

Leaky-bucket rate limiter. `parking_lot::Mutex` (sync) replaces async mutex — lock is never held across `.await`, eliminating future overhead.

### Key Structs

```rust
pub struct TokenBucket {
    capacity: f64,
    refill_rate: f64,          // tokens per second
    state: Mutex<BucketState>, // parking_lot — sync
}
pub struct KeyedRateLimiter {
    buckets: DashMap<String, Arc<TokenBucket>>,
    capacity: usize,
    refill_rate: f64,
}
```

### Diagram

```mermaid
flowchart LR
    Req["try_acquire(tokens_needed)"] --> Lock["parking_lot::Mutex::lock()"]
    Lock --> Refill["do_refill: tokens += elapsed * refill_rate (capped at capacity)"]
    Refill --> Check{"tokens >= tokens_needed?"}
    Check -->|"yes"| Sub["tokens -= tokens_needed → true"]
    Check -->|"no"| Deny["false — drop lock"]
    Sub --> Drop["drop lock (sync)"]
```

---

## `resilience/per_model_breaker` — `src/resilience/per_model_breaker.rs`

Per-model circuit breaker registry. Each model name gets its own `CircuitBreaker` instance; `DashMap` allows lock-free creation and lookup.

### Key Struct

```rust
pub struct CircuitBreakerRegistry {
    breakers: DashMap<String, Arc<CircuitBreaker>>,
    config: CircuitBreakerConfig,
}
```

### Diagram

```mermaid
flowchart LR
    Call["call(model_name, f)"] --> Lookup["DashMap::entry(model_name)"]
    Lookup -->|"existing"| CB["Arc&lt;CircuitBreaker&gt;"]
    Lookup -->|"new"| Create["CircuitBreaker::new(config)"]
    Create --> CB
    CB --> Exec["CircuitBreaker::call(f)"]
    Exec --> Result["Ok(T) or Err(CircuitBreakerError)"]
```

---

## `core/engine` — `src/core/engine.rs`

Central inference coordinator. Sanitizes inputs, dispatches to `ModelManager`, records metrics via `MetricsCollector`.

### Key Struct

```rust
pub struct InferenceEngine {
    pub model_manager: Arc<ModelManager>,
    metrics: MetricsCollector,
    config: Config,
    sanitizer: Sanitizer,
}
```

### Diagram

```mermaid
flowchart TD
    Req["infer(model_name, inputs)"] --> San["sanitizer.sanitize_input(inputs)"]
    San -->|"Err"| ErrII["InferenceError::InvalidInput"]
    San -->|"Ok(sanitized)"| Reg{"model_manager<br/>.get_model_metadata()?"}
    Reg -->|"Ok — registered model"| RI["model_manager.infer_registered()<br/>.instrument(span)"]
    Reg -->|"Err — legacy model"| LM["model_manager.get_model()?"]
    LM --> LI["model.forward(inputs)<br/>.instrument(span)"]
    RI --> Metrics["metrics.record_inference(model, latency_ms)"]
    LI --> Metrics
    Metrics --> Return["Result&lt;Value&gt;"]
```

### Usage

```rust
let engine = InferenceEngine::new(Arc::clone(&model_manager), &config);
engine.warmup(&config).await?; // fires dummy inference per auto_load model

let result = engine.infer("yolov8n", &json!({"image": b64})).await?;
```

---

## `models/manager` — `src/models/manager.rs`

Model lifecycle hub: loads PyTorch (`.pt`) or ONNX (`.onnx`) models, maintains a `DashMap` registry, and owns a `TensorPool`.

### Key Struct

```rust
pub struct ModelManager {
    models: DashMap<String, Arc<BaseModel>>,
    registry: Arc<ModelRegistry>,
    onnx_loader: OnnxModelLoader,
    pytorch_loader: PyTorchModelLoader,
    tensor_pool: Arc<TensorPool>,
    config: Config,
}
```

### Diagram

```mermaid
classDiagram
    class ModelManager {
        -DashMap models
        -Arc~ModelRegistry~ registry
        -OnnxModelLoader onnx_loader
        -PyTorchModelLoader pytorch_loader
        -Arc~TensorPool~ tensor_pool
        +load_model(name, path, format)
        +get_model(name) Arc~BaseModel~
        +get_model_metadata(name) ModelMetadata
        +infer_registered(name, inputs) Value
        +unload_model(name)
        +list_models() Vec~String~
    }
    class ModelRegistry {
        -DashMap~String,ModelMetadata~ metadata
        +register(name, metadata)
        +get(name) ModelMetadata
    }
    class OnnxModelLoader {
        +load(path) LoadedOnnxModel
    }
    class PyTorchModelLoader {
        +load(path, device) LoadedPyTorchModel
    }
    ModelManager --> ModelRegistry
    ModelManager --> OnnxModelLoader
    ModelManager --> PyTorchModelLoader
    ModelManager --> TensorPool
```

---

## `worker_pool` — `src/worker_pool.rs`

Managed async worker pool. Each `Worker` exposes its state as an `AtomicU8` (0=Idle, 1=Processing, 2=Paused, 3=Stopping, 4=Stopped) — no lock needed for state reads.

### Key Structs

```rust
pub struct Worker {
    pub id: usize,
    state: Arc<AtomicU8>,
    tasks_processed: AtomicU64,
    total_processing_time_ms: AtomicU64,
    start_time: Instant,
}
pub enum WorkerState { Idle, Processing, Paused, Stopping, Stopped }
```

### Diagram

```mermaid
stateDiagram-v2
    [*] --> Idle : Worker::spawn()
    Idle --> Processing : task received via channel
    Processing --> Idle : task complete
    Idle --> Paused : pool.pause()
    Paused --> Idle : pool.resume()
    Processing --> Stopping : pool.shutdown()
    Idle --> Stopping : pool.shutdown()
    Stopping --> Stopped : JoinHandle completes
```

### Usage

```rust
let pool = WorkerPool::new(WorkerPoolConfig {
    min_workers: 2,
    max_workers: num_cpus::get(),
});
let result = pool.execute(|| heavy_inference_task()).await?;
let stats = pool.stats(); // per-worker tasks_processed, avg_ms
```

---

## `tensor_pool` — `src/tensor_pool.rs`

Object pool for `Vec<f32>` tensors keyed by `TensorShape`. Reduces heap allocation churn for repeated same-shape inference.

### Key Structs

```rust
pub struct TensorPool {
    pools: DashMap<TensorShape, Vec<Vec<f32>>>,
    max_pooled_tensors: usize,
    allocations: AtomicUsize,
    reuses: AtomicUsize,
}
pub struct TensorShape { pub dims: Vec<usize>, pub total_size: usize }
```

### Diagram

```mermaid
flowchart LR
    Acq["acquire(shape)"] --> Check{"pool contains shape\nwith entries?"}
    Check -->|"yes"| Pop["pool.pop() → reuses += 1"]
    Check -->|"no"| Alloc["vec![0.0; total_size] → allocations += 1"]
    Pop --> Use["caller uses Vec&lt;f32&gt;"]
    Alloc --> Use
    Use --> Rel["release(shape, tensor)"]
    Rel --> Fill["tensor.fill(0.0)  ← zero before reuse"]
    Fill --> Len{"pool len < max_pooled_tensors?"}
    Len -->|"yes"| Push["pool.push(tensor)"]
    Len -->|"no"| Drop["drop tensor"]
```

### Usage

```rust
let pool = TensorPool::new(500);
let shape = TensorShape::new(vec![1, 3, 224, 224]);
let mut tensor = pool.acquire(shape.clone());
// fill tensor with input data, run inference ...
pool.release(shape, tensor); // zero-filled and returned to pool

let reuse_rate = pool.reuses() as f64 / (pool.reuses() + pool.allocations()) as f64;
```

---

## `telemetry` — `src/telemetry/`

Metrics collection, Prometheus export, and structured JSON logging.

### Diagram

```mermaid
classDiagram
    class MetricsCollector {
        -AtomicU64 request_count
        -AtomicU64 error_count
        -DashMap~String,ModelMetrics~ model_metrics
        +record_request()
        +record_error()
        +record_inference(model, latency_ms)
        +get_model_metrics(model) ModelMetrics
    }
    class ModelMetrics {
        +String model_name
        +u64 inference_count
        +f64 avg_inference_time_ms
        +DateTime~Utc~ last_used
    }
    MetricsCollector "1" --> "*" ModelMetrics
```

Files: `metrics.rs` · `prometheus.rs` · `logger.rs` · `structured_logging.rs` · `mod.rs`

---

## `security` — `src/security/`

Input sanitization and validation. `Sanitizer` is owned by `InferenceEngine`; called before every inference.

### Diagram

```mermaid
flowchart TD
    Input["Raw serde_json::Value"] --> San["sanitizer.sanitize_input()"]
    San --> Valid["validation.rs: size checks, type checks, path traversal"]
    Valid -->|"Err"| Reject["InferenceError::InvalidInput"]
    Valid -->|"Ok"| Sanitized["sanitizer.rs: XSS strip, null-byte removal, length truncation"]
    Sanitized --> Out["Sanitized Value → InferenceEngine"]
```

Files: `sanitizer.rs` · `validation.rs` · `mod.rs`

---

## `middleware` — `src/middleware/`

Three Actix-Web middleware components applied in order.

### Diagram

```mermaid
flowchart LR
    Req["HTTP Request"] --> CorrID["correlation_id.rs<br/>inject X-Correlation-ID header"]
    CorrID --> RL["rate_limit.rs<br/>TokenBucket::try_acquire()"]
    RL -->|"false"| R429["429 Too Many Requests"]
    RL -->|"true"| Log["request_logger.rs<br/>tracing::info_span!"]
    Log --> Next["next handler"]
```

---

## `monitor` — `src/monitor.rs`

Real-time system health aggregator. Polls worker stats, cache hit rates, circuit breaker states, and queue depths. Exposed via `GET /health` and `GET /metrics`.

### Diagram

```mermaid
flowchart TD
    Poll["monitor tick (tokio::time::interval)"]
    Poll --> W["WorkerPool::stats()"]
    Poll --> C["Cache::hit_rate()"]
    Poll --> CB2["CircuitBreakerRegistry::states()"]
    Poll --> Q["BatchProcessor::queue_depth"]
    W & C & CB2 & Q --> Agg["SystemHealth snapshot"]
    Agg --> Health["GET /health → 200 / 503"]
    Agg --> Prom["GET /metrics → Prometheus text format"]
```

---

**See also**: [Architecture](../ARCHITECTURE.md) · [Modules](modules/README.md)

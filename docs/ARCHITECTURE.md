# Architecture Overview

This document provides a comprehensive overview of the Torch Inference system architecture.

## System Architecture

### Layered Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Client Layer                            │
│  HTTP Clients, SDKs, Web Applications, Mobile Apps         │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│                 API Gateway Layer                          │
│  • REST API Endpoints (Actix-Web)                         │
│  • Authentication (JWT)                                    │
│  • Rate Limiting                                          │
│  • Request Logging                                        │
│  • Correlation IDs                                        │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│              Business Logic Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │   Caching    │  │   Batching   │  │  Deduplication  │ │
│  │   System     │  │  Processor   │  │                 │ │
│  └──────────────┘  └──────────────┘  └─────────────────┘ │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│               Resilience Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │   Circuit    │  │   Bulkhead   │  │     Retry       │ │
│  │   Breaker    │  │   Pattern    │  │    Logic        │ │
│  └──────────────┘  └──────────────┘  └─────────────────┘ │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│              Inference Engine Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │   PyTorch    │  │     ONNX     │  │     Candle      │ │
│  │   (tch-rs)   │  │   Runtime    │  │                 │ │
│  └──────────────┘  └──────────────┘  └─────────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │    Model     │  │    Worker    │  │     Tensor      │ │
│  │   Manager    │  │     Pool     │  │      Pool       │ │
│  └──────────────┘  └──────────────┘  └─────────────────┘ │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│           Infrastructure Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │  Monitoring  │  │  Telemetry   │  │    Security     │ │
│  │   & Metrics  │  │   & Logs     │  │   Validation    │ │
│  └──────────────┘  └──────────────┘  └─────────────────┘ │
└────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│                Hardware Layer                              │
│  CPU, GPU (CUDA/Metal), Memory, Storage                   │
└────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. API Layer (`src/api/`)

**Purpose**: Handle HTTP requests and routing

**Components**:
- `handlers.rs` - Request handlers and routing
- `health.rs` - Health check endpoints
- `inference.rs` - Inference endpoints
- `image.rs` - Image processing endpoints
- `tts.rs` - Text-to-Speech endpoints
- `yolo.rs` - Object detection endpoints
- `classification.rs` - Classification endpoints
- `models.rs` - Model management endpoints
- `system.rs` - System information endpoints
- `metrics_endpoint.rs` - Metrics exposition

**Key Features**:
- RESTful API design
- JSON request/response
- Multipart file uploads
- Error handling
- Request validation

### 2. Middleware Layer (`src/middleware/`)

**Purpose**: Cross-cutting concerns

**Components**:
- `rate_limit.rs` - Rate limiting per IP/user
- `correlation_id.rs` - Request tracking
- `request_logger.rs` - Access logging

**Patterns**:
- Actix-Web middleware integration
- Request/response transformation
- Error propagation

### 3. Authentication (`src/auth/`)

**Purpose**: Secure access control

**Features**:
- JWT-based authentication
- Token generation and validation
- User management
- Role-based access control (RBAC)

### 4. Cache System (`src/cache.rs`)

**Purpose**: Multi-level caching for performance

**Architecture**:
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
```

**Features**:
- LRU eviction policy
- TTL-based expiration
- Concurrent access (DashMap)
- Hit rate tracking
- Memory-efficient storage

**Performance**:
- 80-85% hit rate (typical)
- O(1) lookup time
- Thread-safe operations
- Automatic cleanup

### 5. Batch Processor (`src/batch.rs`)

**Purpose**: Dynamic request batching

**Architecture**:
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
```

**Batching Strategy**:
- Adaptive timeout based on queue depth
- Priority-based ordering
- Min/max batch size constraints
- Concurrent batch processing

**Performance Improvement**:
- 2-4x throughput increase
- Reduced GPU context switching
- Better resource utilization

### 6. Request Deduplicator (`src/dedup.rs`)

**Purpose**: Eliminate redundant requests

**Algorithm**:
```rust
pub struct RequestDeduplicator {
    inflight: Arc<DashMap<String, InflightRequest>>,
}

struct InflightRequest {
    waiters: Vec<Sender<Result<Vec<u8>>>>,
    started_at: Instant,
}
```

**Flow**:
1. Hash incoming request
2. Check if identical request is in-flight
3. If yes, subscribe to existing request
4. If no, process new request
5. Broadcast result to all waiters

**Benefits**:
- Reduces redundant computation
- Lower resource usage
- Faster response for duplicate requests

### 7. Resilience Layer (`src/resilience/`)

#### Circuit Breaker (`circuit_breaker.rs`)

**Purpose**: Prevent cascading failures

**States**:
```
         Failures > Threshold
Closed ──────────────────────► Open
  ▲                              │
  │                              │ Timeout
  │                              ▼
  └────────────────────── Half-Open
         Success
```

**Configuration**:
- Failure threshold (e.g., 5 failures)
- Timeout duration (e.g., 30 seconds)
- Success threshold for recovery

#### Bulkhead (`bulkhead.rs`)

**Purpose**: Resource isolation

**Pattern**:
- Limit concurrent operations
- Prevent resource exhaustion
- Fair resource allocation

#### Retry Logic (`retry.rs`)

**Strategy**:
- Exponential backoff
- Jitter to prevent thundering herd
- Max retry attempts

### 8. Model Management (`src/models/`)

**Components**:
- `manager.rs` - Model lifecycle management
- `registry.rs` - Model catalog
- `download.rs` - Model download and caching
- `onnx_loader.rs` - ONNX model loading
- `pytorch_loader.rs` - PyTorch model loading

**Features**:
- Lazy loading
- Model versioning
- Hot-swapping
- Multi-model support

### 9. Inference Engine (`src/core/engine.rs`)

**Purpose**: ML model execution

**Supported Backends**:
- **PyTorch** (tch-rs) - Primary backend
- **ONNX Runtime** - Cross-platform inference
- **Candle** - Rust-native ML framework

**Optimizations**:
- FP16 inference
- Batch inference
- CUDA/Metal acceleration
- Model compilation (PyTorch 2.0)

### 10. Worker Pool (`src/worker_pool.rs`)

**Purpose**: Managed worker threads

**Features**:
- Auto-scaling based on load
- Min/max worker limits
- Work queue management
- Load balancing

### 11. Tensor Pool (`src/tensor_pool.rs`)

**Purpose**: Reusable tensor memory

**Benefits**:
- 50-70% faster allocation
- Reduced memory fragmentation
- 95%+ reuse rate

**Implementation**:
```rust
pub struct TensorPool {
    pools: DashMap<TensorShape, Vec<Tensor>>,
    stats: Arc<RwLock<PoolStats>>,
}
```

### 12. Monitoring (`src/monitor.rs`)

**Purpose**: Real-time metrics and observability

**Metrics**:
- Request count
- Latency (min/max/avg/p95/p99)
- Throughput (req/s)
- Error rate
- Cache hit rate
- Batch statistics

**Integration**:
- Prometheus metrics (optional)
- Structured logging
- Health checks

### 13. Telemetry (`src/telemetry/`)

**Components**:
- `logger.rs` - Structured logging
- `metrics.rs` - Metrics collection
- `prometheus.rs` - Prometheus exporter
- `structured_logging.rs` - JSON logging

### 14. Security (`src/security/`)

**Components**:
- `validation.rs` - Input validation
- `sanitizer.rs` - Input sanitization

**Features**:
- XSS prevention
- SQL injection prevention
- Path traversal protection
- Size limit enforcement

## Data Flow

### Typical Request Flow

```
1. HTTP Request
       ↓
2. Middleware (Auth, Rate Limit, Logging)
       ↓
3. Request Validation & Sanitization
       ↓
4. Cache Lookup
       ↓
   ┌─ Cache Hit ──────► Return Cached Result
   │
   └─ Cache Miss
       ↓
5. Request Deduplication Check
       ↓
   ┌─ Duplicate ──────► Wait for In-Flight Request
   │
   └─ New Request
       ↓
6. Add to Batch Queue
       ↓
7. Batch Formation (Adaptive Timeout)
       ↓
8. Circuit Breaker Check
       ↓
9. Bulkhead Acquire Permit
       ↓
10. Model Inference (Worker Pool)
       ↓
11. Store in Cache
       ↓
12. Broadcast to Waiters
       ↓
13. Return Response
```

### Batch Processing Flow

```
Requests → Queue → Batch Formation → Inference → Results Distribution
              ↑            ↓
              └─── Adaptive Timeout
                   (Based on Queue Depth)
```

## Concurrency Model

### Thread Safety

**Lock-Free Structures**:
- `DashMap` for caches and registries
- `AtomicU64` for counters
- `Arc` for shared ownership

**Locks**:
- `RwLock` for read-heavy structures (stats)
- `Mutex` for write-heavy structures (queues)

### Async Runtime

**Tokio**:
- Multi-threaded runtime
- Work-stealing scheduler
- Async I/O
- Timers and delays

## Performance Optimizations

### Memory

1. **Tensor Pooling**: Reuse tensor allocations
2. **Zero-Copy**: Minimize data copying
3. **Memory Limits**: Configurable memory caps
4. **Compression**: Gzip for large responses

### CPU

1. **Multi-Threading**: Parallel request processing
2. **SIMD**: Vectorized operations (via libraries)
3. **Thread Pool**: Reusable worker threads

### GPU

1. **Batch Inference**: Maximize GPU utilization
2. **FP16**: 2x faster on compatible GPUs
3. **CUDA Streams**: Concurrent operations
4. **Metal Performance Shaders**: Apple Silicon optimization

### Network

1. **Connection Pooling**: Reuse HTTP connections
2. **Compression**: Reduce bandwidth
3. **Keep-Alive**: Persistent connections

## Scalability

### Horizontal Scaling

- **Stateless Design**: No local state dependencies
- **External Cache**: Redis/Memcached (future)
- **Load Balancer**: Distribute across instances

### Vertical Scaling

- **Worker Pool**: Scale workers with CPUs
- **Model Pool**: Multiple model instances
- **Batch Size**: Larger batches for more memory

## Error Handling

### Error Types

```rust
pub enum InferenceError {
    ModelNotFound,
    InvalidInput,
    InferenceTimeout,
    ResourceExhausted,
    CircuitBreakerOpen,
    InternalError,
}
```

### Error Propagation

1. Typed errors with `thiserror`
2. Error context with `anyhow`
3. HTTP status code mapping
4. Structured error logging

## Configuration

### Configuration Hierarchy

```
1. Default values (compile-time)
2. config.toml file
3. Environment variables
4. Command-line arguments
```

### Hot-Reloading

- Model loading/unloading
- Cache size adjustment
- Worker pool scaling

## Monitoring & Observability

### Metrics

- Request rate
- Response time
- Error rate
- Resource usage

### Logging

- Structured JSON logging
- Log levels (DEBUG, INFO, WARN, ERROR)
- Correlation IDs for request tracking

### Health Checks

- Liveness probe
- Readiness probe
- Dependency checks

## Security Architecture

### Defense in Depth

1. **Network**: Rate limiting, firewall
2. **Application**: Authentication, authorization
3. **Data**: Validation, sanitization
4. **Infrastructure**: Secure defaults, least privilege

### Authentication Flow

```
Client → Login → JWT Token → Request + Token → Validate → Process
```

## Deployment Architecture

### Single Instance

```
┌──────────────────────┐
│  Load Balancer       │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  Torch Inference     │
│  (Single Instance)   │
└──────────────────────┘
```

### Multi-Instance

```
┌──────────────────────┐
│  Load Balancer       │
└──────┬───────────────┘
       │
   ┌───┴───┬───────┬───────┐
   ▼       ▼       ▼       ▼
┌────┐  ┌────┐  ┌────┐  ┌────┐
│Ins1│  │Ins2│  │Ins3│  │Ins4│
└────┘  └────┘  └────┘  └────┘
```

## Design Principles

1. **Performance First**: Optimize critical paths
2. **Type Safety**: Leverage Rust's type system
3. **Fail Fast**: Early error detection
4. **Observable**: Comprehensive metrics
5. **Testable**: Unit and integration tests
6. **Maintainable**: Clean code, documentation
7. **Extensible**: Plugin architecture for backends

## Future Architecture

### Planned Enhancements

- [ ] Distributed caching (Redis)
- [ ] Model versioning and A/B testing
- [ ] Streaming inference
- [ ] WebSocket support
- [ ] gRPC API
- [ ] Kubernetes Operators
- [ ] Auto-scaling policies
- [ ] Multi-GPU support
- [ ] Model quantization

---

**Next**: [Component Guide](COMPONENTS.md) for detailed component documentation.

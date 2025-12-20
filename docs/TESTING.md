# Testing Guide

Comprehensive testing guide for Torch Inference Server.

## Test Overview

### Test Statistics

- **Total Tests**: 147+
- **Unit Tests**: 91
- **Integration Tests**: 6
- **Benchmarks**: 3 suites
- **Test Runtime**: <30 seconds
- **Success Rate**: 100%

### Test Structure

```
torch-inference/
├── src/
│   ├── cache.rs              (38 tests)
│   ├── batch.rs              (28 tests)
│   ├── monitor.rs            (28 tests)
│   ├── dedup.rs              (9 tests)
│   ├── error.rs              (11 tests)
│   ├── config.rs             (7 tests)
│   └── resilience/
│       ├── circuit_breaker.rs (10 tests)
│       └── bulkhead.rs        (6 tests)
├── tests/
│   ├── integration_test.rs    (6 tests)
│   ├── benchmark_test.rs
│   └── yolo_test.rs
└── benches/
    ├── cache_bench.rs
    ├── model_inference_bench.rs
    └── model_inference_bench_optimized.rs
```

## Running Tests

### All Tests

```bash
# Run all tests (unit + integration)
cargo test

# Run with output
cargo test -- --nocapture

# Run in release mode (faster)
cargo test --release

# Parallel execution
cargo test -- --test-threads=8

# Sequential execution (for debugging)
cargo test -- --test-threads=1
```

### Specific Test Suites

```bash
# Cache system tests (38 tests)
cargo test cache::tests

# Batch processing tests (28 tests)
cargo test batch::tests

# Monitoring tests (28 tests)
cargo test monitor::tests

# Circuit breaker tests (10 tests)
cargo test circuit_breaker::tests

# Bulkhead tests (6 tests)
cargo test bulkhead::tests

# Deduplication tests (9 tests)
cargo test dedup::tests

# Error handling tests (11 tests)
cargo test error::tests

# Config tests (7 tests)
cargo test config::tests
```

### Integration Tests

```bash
# All integration tests
cargo test --test integration_test

# Specific integration test
cargo test --test integration_test test_end_to_end_inference

# YOLO tests
cargo test --test yolo_test
```

### Benchmarks

```bash
# Run all benchmarks
cargo bench

# Specific benchmark suite
cargo bench --bench cache_bench
cargo bench --bench model_inference_bench

# Generate HTML reports (in target/criterion/)
cargo bench -- --save-baseline my-baseline

# Compare with baseline
cargo bench -- --baseline my-baseline
```

## Test Categories

### Unit Tests

#### Cache Tests (38 tests)

**Basic Operations**:
- `test_cache_basic_operations` - Set, get, remove
- `test_cache_get_nonexistent` - Missing key handling
- `test_cache_overwrite` - Update existing keys
- `test_cache_clear` - Clear all entries

**TTL & Expiration**:
- `test_cache_expiration` - TTL-based expiration
- `test_cache_get_expired` - Expired entry removal
- `test_cache_expiration_cleanup` - Automatic cleanup

**Concurrency**:
- `test_cache_concurrent_set` - Parallel writes (10 threads)
- `test_cache_concurrent_get` - Parallel reads
- `test_cache_concurrent_mixed` - Mixed operations
- `test_cache_stress` - Stress test (20 threads × 100 ops)

**Edge Cases**:
- `test_cache_unicode_keys` - Unicode key support
- `test_cache_large_values` - Large value handling
- `test_cache_empty_values` - Empty value handling
- `test_cache_boundary_sizes` - Size boundaries

**Statistics**:
- `test_cache_stats_tracking` - Hit/miss tracking
- `test_cache_hit_rate_calculation` - Hit rate computation
- `test_cache_memory_tracking` - Memory usage tracking

**Example**:
```rust
#[test]
fn test_cache_basic_operations() {
    let cache = Cache::new(100);
    cache.set("key1", vec![1, 2, 3], Duration::from_secs(60));
    
    let value = cache.get("key1");
    assert!(value.is_some());
    assert_eq!(value.unwrap(), vec![1, 2, 3]);
    
    cache.remove("key1");
    assert!(cache.get("key1").is_none());
}
```

#### Batch Processing Tests (28 tests)

**Basic Batching**:
- `test_batch_creation` - Batch creation
- `test_batch_item_addition` - Add items to batch
- `test_batch_processing` - Process batch
- `test_batch_empty` - Empty batch handling

**Dynamic Batching**:
- `test_dynamic_batch_formation` - Dynamic size adjustment
- `test_adaptive_timeout` - Timeout adaptation
- `test_batch_priority` - Priority ordering

**Concurrency**:
- `test_batch_concurrent_addition` - Parallel additions
- `test_batch_stress` - Stress test (20 producers × 100 items)

**Edge Cases**:
- `test_batch_max_size` - Maximum size enforcement
- `test_batch_timeout_handling` - Timeout behavior
- `test_batch_large_items` - Large item handling

**Example**:
```rust
#[tokio::test]
async fn test_dynamic_batch_formation() {
    let processor = BatchProcessor::new(BatchConfig {
        max_batch_size: 10,
        adaptive_timeout: true,
        ..Default::default()
    });
    
    // Add items
    for i in 0..5 {
        processor.add(format!("item_{}", i), 0).await;
    }
    
    // Process batch
    let batch = processor.get_batch().await;
    assert_eq!(batch.len(), 5);
}
```

#### Monitoring Tests (28 tests)

**Request Tracking**:
- `test_monitor_request_tracking` - Track requests
- `test_monitor_endpoint_stats` - Per-endpoint statistics
- `test_monitor_latency_tracking` - Latency metrics

**Metrics**:
- `test_monitor_min_max_latency` - Min/max calculation
- `test_monitor_avg_latency` - Average calculation
- `test_monitor_percentiles` - P95/P99 calculation
- `test_monitor_throughput` - Throughput calculation

**Concurrency**:
- `test_monitor_concurrent_recording` - Parallel recording (10 threads)
- `test_monitor_high_frequency` - High-frequency updates (10k ops/sec)

**Example**:
```rust
#[test]
fn test_monitor_latency_tracking() {
    let monitor = Monitor::new();
    
    monitor.record_request("/api/classify", 15, true);
    monitor.record_request("/api/classify", 20, true);
    monitor.record_request("/api/classify", 25, true);
    
    let stats = monitor.get_stats();
    assert_eq!(stats.min_latency_ms, 15);
    assert_eq!(stats.max_latency_ms, 25);
    assert_eq!(stats.avg_latency_ms, 20);
}
```

#### Circuit Breaker Tests (10 tests)

**State Transitions**:
- `test_circuit_breaker_closed_to_open` - Failure threshold
- `test_circuit_breaker_open_to_half_open` - Timeout
- `test_circuit_breaker_half_open_to_closed` - Recovery
- `test_circuit_breaker_half_open_to_open` - Re-failure

**Operations**:
- `test_circuit_breaker_success` - Success handling
- `test_circuit_breaker_failure` - Failure handling
- `test_circuit_breaker_reset` - Manual reset

**Example**:
```rust
#[test]
fn test_circuit_breaker_closed_to_open() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 5,
        timeout_duration: Duration::from_secs(30),
        success_threshold: 2,
    });
    
    // Record failures
    for _ in 0..5 {
        cb.record_failure();
    }
    
    assert_eq!(cb.state(), CircuitBreakerState::Open);
}
```

### Integration Tests

#### End-to-End Tests

```rust
#[tokio::test]
async fn test_end_to_end_inference() {
    // Start server
    let server = start_test_server().await;
    
    // Upload model
    let model = load_test_model();
    
    // Make inference request
    let response = client.post("/api/inference")
        .json(&request)
        .send()
        .await
        .unwrap();
    
    assert_eq!(response.status(), 200);
    
    let result: InferenceResponse = response.json().await.unwrap();
    assert!(result.inference_time_ms < 100);
}
```

#### Concurrent Load Test

```rust
#[tokio::test]
async fn test_concurrent_load() {
    let server = start_test_server().await;
    
    // Spawn 100 concurrent requests
    let tasks: Vec<_> = (0..100)
        .map(|i| {
            tokio::spawn(async move {
                make_inference_request(i).await
            })
        })
        .collect();
    
    // Wait for all
    let results = futures::future::join_all(tasks).await;
    
    // All should succeed
    assert_eq!(results.len(), 100);
    assert!(results.iter().all(|r| r.is_ok()));
}
```

### Benchmark Tests

#### Cache Benchmark

```rust
fn bench_cache_set(c: &mut Criterion) {
    let cache = Cache::new(1000);
    
    c.bench_function("cache_set", |b| {
        b.iter(|| {
            cache.set(
                black_box("key"),
                black_box(vec![1, 2, 3, 4, 5]),
                Duration::from_secs(60)
            );
        });
    });
}
```

## Test Data

### Fixtures

```
tests/fixtures/
├── images/
│   ├── cat.jpg
│   ├── dog.jpg
│   └── test_image.png
├── audio/
│   ├── speech.wav
│   └── test_audio.mp3
└── models/
    └── test_model.pt
```

### Test Helpers

```rust
// tests/common/mod.rs
pub fn create_test_config() -> Config {
    Config {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0, // Random port
            workers: 1,
        },
        ..Default::default()
    }
}

pub async fn start_test_server() -> TestServer {
    let config = create_test_config();
    TestServer::new(config).await
}

pub fn load_test_image() -> Vec<u8> {
    std::fs::read("tests/fixtures/images/cat.jpg").unwrap()
}
```

## Test Patterns

### Async Tests

```rust
#[tokio::test]
async fn test_async_operation() {
    let result = async_function().await;
    assert!(result.is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_operations() {
    // Test with multiple threads
}
```

### Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_cache_any_key(key in "\\PC*") {
        let cache = Cache::new(100);
        cache.set(&key, vec![1, 2, 3], Duration::from_secs(60));
        assert!(cache.get(&key).is_some());
    }
}
```

### Parameterized Tests

```rust
use rstest::rstest;

#[rstest]
#[case(1, 2, 3)]
#[case(10, 20, 30)]
#[case(100, 200, 300)]
fn test_addition(#[case] a: i32, #[case] b: i32, #[case] expected: i32) {
    assert_eq!(a + b, expected);
}
```

## Test Coverage

### Generate Coverage Report

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage
cargo tarpaulin --out Html

# Open report
open tarpaulin-report.html
```

### Coverage Goals

- **Overall**: >80%
- **Core Logic**: >90%
- **Error Handling**: >85%
- **API Handlers**: >75%

## Continuous Integration

### GitHub Actions

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      
      - name: Run tests
        run: cargo test --all-features
      
      - name: Run benchmarks
        run: cargo bench --no-run
```

## Test Best Practices

1. **Isolation**: Each test is independent
2. **Determinism**: Tests produce consistent results
3. **Fast Execution**: Tests complete in <30 seconds
4. **Clear Names**: Descriptive test names
5. **Good Coverage**: Test critical paths
6. **Documentation**: Comment complex test logic

## Debugging Tests

### Debug Single Test

```bash
# Run with debug output
cargo test test_name -- --nocapture

# Run with backtrace
RUST_BACKTRACE=1 cargo test test_name

# Run with logging
RUST_LOG=debug cargo test test_name
```

### Test in IDE

**VS Code**:
1. Install Rust Analyzer extension
2. Click "Run Test" above test function
3. View output in terminal

**IntelliJ IDEA**:
1. Install Rust plugin
2. Right-click test function
3. Select "Run 'test_name'"

## Performance Testing

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:8000/api/health

# Using wrk
wrk -t4 -c100 -d30s http://localhost:8000/api/health
```

### Stress Testing

```rust
#[tokio::test]
async fn stress_test_concurrent_requests() {
    let tasks: Vec<_> = (0..1000)
        .map(|_| tokio::spawn(make_request()))
        .collect();
    
    let results = futures::future::join_all(tasks).await;
    assert!(results.iter().all(|r| r.is_ok()));
}
```

## Test Maintenance

### Update Test Data

```bash
# Refresh fixtures
./scripts/update_test_fixtures.sh

# Re-record snapshots
cargo test -- --ignored
```

### Flaky Tests

```bash
# Run test multiple times
cargo test test_name --release -- --test-threads=1 --nocapture
for i in {1..10}; do cargo test test_name; done
```

---

**Next**: See [Development Guide](DEVELOPMENT.md) for development workflow.

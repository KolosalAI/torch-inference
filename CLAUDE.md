# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-performance PyTorch inference server written in Rust with enterprise-grade testing, resilience patterns, and comprehensive ML model support. The system supports multiple inference backends (PyTorch, ONNX, TensorRT) and includes specialized TTS/STT engines, YOLO object detection, and image classification.

## Essential Commands

### Building
```bash
# Development build (fastest, no ML backends)
cargo build

# Production release build (recommended)
cargo build --release

# Build with specific backends
cargo build --release --features torch      # PyTorch support
cargo build --release --features cuda       # CUDA GPU support
cargo build --release --features onnx       # ONNX Runtime support
cargo build --release --features all-backends  # All ML backends
```

### Testing
```bash
# Run all tests (147+ unit + integration tests)
cargo test

# Run specific test suites
cargo test cache::tests           # Cache system (38 tests)
cargo test batch::tests           # Batch processing (28 tests)
cargo test monitor::tests         # Monitoring (28 tests)
cargo test resilience::           # Resilience patterns (16 tests)

# Run integration tests only
cargo test --test integration_test

# Run with verbose output
cargo test -- --nocapture

# Run single test
cargo test specific_test_name -- --exact --nocapture
```

### Running
```bash
# Run development server
cargo run

# Run release server
cargo run --release

# Run with specific features
cargo run --release --features cuda
```

### Benchmarking
```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suite
cargo bench cache_bench
cargo bench image_classification_benchmark

# Generate HTML reports (output in target/criterion/)
cargo bench -- --save-baseline my_baseline
```

### Cargo Aliases (in .cargo/config.toml)
```bash
cargo run-server        # Optimized release build without ML dependencies
cargo build-release     # Build release binary
cargo build-torch       # Build with PyTorch
cargo build-all         # Build with all backends
cargo test-no-ml        # Test core functionality without ML
```

## Architecture

### Core System Components

The system follows a layered architecture with clear separation of concerns:

**Entry Point (src/main.rs)**
- Initializes all components in a specific order
- Auto-detects GPU backend (CUDA/Metal/CPU) if device_type="auto"
- Sets up graceful shutdown handling
- Configures Actix-Web HTTP server with middleware

**Inference Pipeline Flow:**
```
HTTP Request → Middleware (RateLimiter, CorrelationID)
  → API Handler (api/)
  → InferenceEngine (core/engine.rs)
  → ModelManager (models/manager.rs)
  → Backend Loader (pytorch_loader/onnx_loader/tensorrt_auto)
  → Resilience Layer (circuit_breaker, bulkhead, retry)
  → Cache/Dedup (cache.rs, dedup.rs)
  → Response
```

**Key Subsystems:**

1. **Resilience Layer (src/resilience/)**
   - Circuit breaker with state machine (Closed → Open → HalfOpen)
   - Bulkhead pattern for resource isolation
   - Per-model circuit breakers (per_model_breaker.rs)
   - Token bucket rate limiting
   - Retry policies with exponential backoff

2. **Model Management (src/models/)**
   - ModelManager: Centralized model lifecycle management
   - ModelDownloadManager: Fetch models from HuggingFace/remote URLs
   - Registry: Model metadata and versioning (model_registry.json)
   - Backend-specific loaders (PyTorch, ONNX, TensorRT auto-detection)

3. **Caching & Optimization (root level)**
   - cache.rs: LRU cache with TTL expiration
   - dedup.rs: Request deduplication to prevent redundant inference
   - batch.rs: Dynamic batching with configurable timeouts
   - tensor_pool.rs: Memory pooling for tensors
   - worker_pool.rs: Auto-scaling worker pool (zero-scaling support)

4. **TTS/STT Engines (src/core/)**
   - tts_manager.rs: Unified TTS engine manager
   - Multiple engines: Kokoro (ONNX), Piper, VITS, StyleTTS2, Bark, XTTS
   - whisper_stt.rs: Speech-to-text via Whisper
   - phoneme_converter.rs: Text → phoneme conversion

5. **ML Models (src/core/)**
   - yolo.rs: YOLO object detection (v5, v8, v10, v11, v12)
   - image_classifier.rs: Image classification models
   - neural_network.rs: Generic neural network interface

### Important Implementation Patterns

**Auto-Detection:**
- Device detection happens in src/main.rs:78-134
- PyTorch initialization in src/main.rs:138-161
- GPU manager provides unified CUDA/Metal/CPU abstraction (src/core/gpu.rs)

**Graceful Degradation:**
- System continues if optional components fail (TTS engines, GPU detection)
- Circuit breakers prevent cascade failures
- Bulkhead isolates resource failures

**Memory Management:**
- Use jemalloc on Linux (enabled by default, feature flag: `jemalloc`)
- Tensor pooling reduces allocation overhead (feature: `enable_tensor_pooling`)
- Result compression for large outputs (feature: `enable_result_compression`)

**Backend Priority:**
- TensorRT auto-detection tries: TensorRT → CUDA → CPU (models/tensorrt_auto.rs)
- Device selection: auto → detected backend in config

## Configuration

Primary config file: `config.toml` (or `config/production.toml`)

Key sections:
- `[device]`: GPU/CPU settings, device_type ("auto", "cuda", "mps", "cpu")
- `[performance]`: Batching, caching, worker pool settings
- `[models]`: auto_load list, model paths
- `[server]`: host, port, workers

Environment variables:
- `LOG_JSON=true`: Enable JSON structured logging
- `LOG_DIR=/path`: Log file directory
- `MODEL_CACHE_DIR=/path`: Model storage location
- `AUDIO_MODEL_DIR=/path`: Audio model storage

## Development Guidelines

**Testing Standards:**
- All tests are co-located with implementation using `#[cfg(test)]`
- Use `#[tokio::test]` for async tests
- Tests must be isolated and parallelizable
- Aim for >90% coverage on core modules

**Adding New Models:**
1. Add metadata to `model_registry.json`
2. Implement loader in appropriate backend (pytorch_loader.rs, onnx_loader.rs)
3. Register in ModelManager (models/manager.rs)
4. Add API endpoint in api/ if needed
5. Add integration test

**Adding TTS Engines:**
1. Implement `TTSEngine` trait in src/core/tts_engine.rs
2. Create engine file in src/core/ (e.g., new_tts.rs)
3. Register in TTSManager::initialize_defaults() (src/core/tts_manager.rs)
4. Add API route in src/api/tts.rs

**Feature Flags:**
- Use `#[cfg(feature = "torch")]` for PyTorch-dependent code
- Use `#[cfg(feature = "cuda")]` for CUDA-specific code
- Default features should allow basic server functionality

## Common Patterns

**Error Handling:**
```rust
// Use custom error types from src/error.rs
use crate::error::{Result, InferenceError};

// Return Results, let errors propagate
pub async fn my_function() -> Result<Output> {
    let data = fetch_data()?;  // Propagate errors
    Ok(process(data))
}
```

**Async Operations:**
```rust
// Use tokio::spawn for background tasks
tokio::spawn(async move {
    // Long-running task
});

// Use Arc for shared state
let shared = Arc::new(MyState::new());
```

**Metrics & Monitoring:**
```rust
// Record metrics via Monitor
monitor.record_request("endpoint", latency_ms);

// Use structured logging
log::info!("Processing request", request_id = %id);
```

## Troubleshooting

**Build Issues:**
- LibTorch not found: Set `LIBTORCH` env var or use `--no-default-features`
- CUDA errors: Verify CUDA toolkit installed, check `nvml-wrapper` feature
- macOS linking: Ensure deployment target matches (MACOSX_DEPLOYMENT_TARGET=15.0)

**Runtime Issues:**
- Model not loading: Check model_registry.json and model file paths
- GPU not detected: Verify CUDA/Metal installation, check auto-detection logs
- OOM errors: Reduce batch_size, enable tensor pooling, adjust cache_size_mb

**Testing:**
- Flaky tests: Use `--test-threads=1` to run sequentially
- Timeout in tests: Increase timeout or mock slow operations
- Integration test failures: Ensure server components initialized correctly

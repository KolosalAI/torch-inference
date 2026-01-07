# CUDA and TensorRT Optimization Guide

This document describes the comprehensive CUDA and TensorRT optimizations implemented in the torch-inference framework for maximum inference performance.

## Overview

The framework provides state-of-the-art GPU acceleration through:

1. **Advanced CUDA Optimizer** - Memory pools, streams, graphs, and precision control
2. **TensorRT Auto-Integration** - Automatic detection and optimal configuration
3. **ONNX Runtime Execution Providers** - Optimized CUDA and TensorRT backends

## Quick Start

### For Maximum Throughput

```rust
use torch_inference::models::onnx_loader::OnnxModelLoader;

// Create loader optimized for throughput (batch processing)
let loader = OnnxModelLoader::for_throughput(0); // device_id = 0
```

### For Minimum Latency

```rust
// Create loader optimized for latency (real-time inference)
let loader = OnnxModelLoader::for_latency(0);
```

### For INT8 Inference (Maximum Speed)

```rust
// Create loader with INT8 quantization (requires calibration)
let loader = OnnxModelLoader::for_int8(0);
```

## CUDA Optimizer

The `CudaOptimizer` provides fine-grained control over GPU execution:

### Optimization Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `Minimal` | Basic optimizations | Compatibility-focused workloads |
| `Standard` | Balanced approach | General-purpose inference |
| `Aggressive` | More optimizations | Production inference |
| `Maximum` | All optimizations | Maximum performance |

### Configuration Options

```rust
use torch_inference::core::cuda_optimizer::{
    CudaOptimizerBuilder, CudaOptimizationLevel, ComputePrecision, CudnnAlgorithmStrategy
};

let optimizer = CudaOptimizerBuilder::new()
    .optimization_level(CudaOptimizationLevel::Maximum)
    .device_id(0)
    
    // Memory Configuration
    .memory_pool_size(2048, 16384)  // Initial: 2GB, Max: 16GB
    .memory_fraction(0.9)           // Use 90% of GPU memory
    
    // Stream Configuration
    .compute_streams(8)             // 8 parallel compute streams
    .copy_streams(4)                // 4 host<->device copy streams
    
    // CUDA Graphs
    .enable_cuda_graphs(true)
    .graph_warmup_iterations(20)
    
    // Precision
    .precision(ComputePrecision::FP16)
    .enable_amp(true)               // Automatic Mixed Precision
    .enable_tf32(true)              // TensorFloat-32 on Ampere GPUs
    
    // cuDNN
    .cudnn_benchmark(true)          // Auto-tune algorithms
    .cudnn_algorithm(CudnnAlgorithmStrategy::Exhaustive)
    
    // Hardware Features
    .enable_persistent_l2(true)     // L2 cache optimization
    
    .build();
```

### Memory Pool Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_size_mb` | 1024 | Initial pool allocation |
| `max_size_mb` | 8192 | Maximum pool size |
| `memory_fraction` | 0.9 | Fraction of GPU memory to use |
| `alignment` | 256 | Memory alignment (bytes) |
| `enable_async` | true | Async memory operations |
| `enable_defrag` | true | Memory defragmentation |

### Stream Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_compute_streams` | 4 | Parallel compute streams |
| `num_copy_streams` | 2 | Host<->Device copy streams |
| `enable_priorities` | true | Stream priorities |
| `default_priority` | -1 (high) | Default stream priority |

### CUDA Graph Configuration

CUDA graphs reduce kernel launch overhead by capturing and replaying GPU operations.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | true | Enable graph capture |
| `warmup_iterations` | 10 | Warmup before capture |
| `optimize_graph` | true | Apply graph optimizations |
| `cache_graphs` | true | Cache captured graphs |
| `max_cached_graphs` | 100 | Maximum cached graphs |

## TensorRT Configuration

### Default Configuration

```rust
use torch_inference::models::onnx_loader::TensorRTConfig;

let config = TensorRTConfig {
    enabled: true,
    precision: TensorRTPrecision::FP16,
    workspace_size_mb: 4096,           // 4GB workspace
    max_batch_size: 64,
    optimization_level: 5,             // Maximum optimization
    cache_dir: Some("./tensorrt_cache".to_string()),
    use_dynamic_shapes: true,
    builder_optimization_level: 5,
    auxiliary_streams: 4,
    enable_layer_norm_plugin: true,
    enable_gelu_plugin: true,
    enable_sparse_weights: false,
};
```

### Precision Modes

| Mode | Speedup | Memory | Use Case |
|------|---------|--------|----------|
| FP32 | 1x (baseline) | High | Accuracy-critical |
| FP16 | ~2-3x | 50% | Best balance |
| INT8 | ~3-5x | 25% | Maximum throughput |

### TensorRT Auto-Detection

The framework automatically detects TensorRT availability:

```rust
use torch_inference::models::tensorrt_auto::TensorRTAutoManager;

let manager = TensorRTAutoManager::new();

// Check status
match manager.status() {
    TensorRTStatus::Available { version, cuda_version } => {
        println!("TensorRT {} with CUDA {}", version, cuda_version);
    }
    TensorRTStatus::CudaOnly { cuda_version } => {
        println!("CUDA {} (TensorRT not installed)", cuda_version);
    }
    TensorRTStatus::CpuOnly => {
        println!("CPU-only mode");
    }
}

// Get recommended precision
let precision = manager.recommended_precision();

// Get configuration summary
println!("{}", manager.get_summary());
```

## ONNX Runtime Integration

### Execution Provider Priority

The framework configures execution providers in priority order:

1. **TensorRT** (if enabled and available)
   - FP16/INT8 precision
   - Engine caching for fast load times
   - Timing cache for optimization
   
2. **CUDA** (fallback)
   - Max workspace for cuDNN
   - Optimized stream configuration
   
3. **CoreML** (macOS only)
   - Apple Silicon optimization
   - Neural Engine utilization
   
4. **CPU** (always available)
   - Multi-threaded execution

### Thread Configuration

The loader automatically optimizes thread counts:

- `intra_threads`: Number of CPUs (for parallelism within operators)
- `inter_threads`: Half of CPUs (for parallelism between operators)

## Performance Benchmarks

Run the benchmark suite:

```bash
# Release build with CUDA
cargo run --example onnx_benchmark --features cuda --no-default-features --release

# Python comparison
python benchmark_onnx_inference.py --cuda
```

### Actual Inference Results (RTX 3060 Laptop GPU)

#### Final Results: Rust vs Python (ResNet-18)

| Batch | Rust IoBinding | Rust Baseline | Python (1.23.2) | Winner |
|-------|----------------|---------------|-----------------|--------|
| 1 | **697 img/s** | 667 img/s | 663 img/s | Rust +5% |
| 4 | **1,265 img/s** | 1,023 img/s | 1,226 img/s | Rust +3% |
| 8 | **1,581 img/s** | 1,208 img/s | 1,465 img/s | Rust +8% |
| 16 | **1,670 img/s** | 1,285 img/s | 1,558 img/s | Rust +7% |
| 32 | **1,798 img/s** | 1,352 img/s | 1,646 img/s | Rust +9% |
| 64 | **1,871 img/s** | 1,403 img/s | 1,710 img/s | Rust +9% |

**Rust with IoBinding beats Python at ALL batch sizes!**

#### Key Optimizations Applied

1. **IoBinding** - Zero-copy inference with pre-bound GPU tensors
2. **Pinned Memory** - CUDA_PINNED allocation for faster host-to-device transfers
3. **TF32** - TensorFloat-32 for Tensor Core acceleration
4. **Level3 Graph Optimization** - Maximum ONNX graph optimizations
5. **Exhaustive cuDNN Conv Search** - Optimal convolution algorithm selection

#### Latency Comparison (batch=64)

| Implementation | Batch Latency | Per-Image Latency |
|----------------|---------------|-------------------|
| Rust IoBinding | 34.2 ms | 0.534 ms |
| Python | 37.4 ms | 0.585 ms |

**Key findings:**
- IoBinding provides ~30% boost over baseline for batched inference
- Pinned memory helps with larger batches (~10-15% improvement)
- TF32 gives modest 1-2% improvement on RTX 3060
- NHWC layout actually hurts performance on this GPU (-10%)

### Configuration Benchmark Results

| Operation | Time |
|-----------|------|
| CUDA Optimizer Creation | ~180 ns |
| Stats Recording | ~13 ns |
| Batch Size Recommendation | ~1.4 ns |
| ORT Options Generation | ~600 ns |
| TensorRT Options Generation | ~1.1 µs |

## Setup Guide

### Prerequisites

Install cuDNN 9.x and CUDA 12.x:

```bash
# Via pip (recommended)
pip install nvidia-cudnn-cu12

# Or download from NVIDIA Developer site
```

### Environment Setup

```powershell
# Windows PowerShell
$pythonPath = "$env:LOCALAPPDATA\Programs\Python\Python311\Lib\site-packages"
$env:Path = "$pythonPath\nvidia\cudnn\bin;$pythonPath\nvidia\cublas\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;$env:Path"
```

### Rust Setup

For the `ort` crate with CUDA support:

```powershell
# Set ONNX Runtime path (use version 1.22.x)
$env:ORT_DYLIB_PATH = "path\to\onnxruntime.dll"
```

## Best Practices

### 1. Enable Engine Caching

TensorRT engine building is expensive. Always enable caching:

```rust
let config = TensorRTConfig {
    cache_dir: Some("./tensorrt_cache".to_string()),
    ..Default::default()
};
```

### 2. Use Appropriate Precision

- **FP16**: Best for most inference workloads (2-3x speedup)
- **INT8**: Maximum throughput but requires calibration
- **FP32**: Only for accuracy-critical applications

### 3. Configure Dynamic Shapes

For variable batch sizes:

```rust
let config = TensorRTConfig {
    use_dynamic_shapes: true,
    max_batch_size: 64,
    ..Default::default()
};
```

For fixed batch sizes (lower latency):

```rust
let config = TensorRTConfig {
    use_dynamic_shapes: false,
    max_batch_size: 1,
    ..Default::default()
};
```

### 4. Warm Up Models

Always warm up models before production:

```rust
// Perform warmup iterations
for _ in 0..10 {
    let _ = model.infer(&dummy_input);
}
```

### 5. Monitor Performance

Use the CUDA optimizer stats:

```rust
let optimizer = CudaOptimizer::new();

// After processing
let stats = optimizer.stats();
println!("Average batch time: {:.2} ms", stats.avg_batch_time_ms);
println!("Throughput: {:.2} inferences/sec", stats.throughput);
println!("Peak memory: {} MB", stats.peak_memory_usage / (1024 * 1024));
```

## Troubleshooting

### TensorRT Not Detected

1. Verify TensorRT installation:
   - Windows: Check `C:\Program Files\NVIDIA\TensorRT` or `TENSORRT_ROOT` env var
   - Linux: Check `/usr/lib/x86_64-linux-gnu/libnvinfer.so`

2. Verify ONNX Runtime has TensorRT support:
   ```bash
   pip install onnxruntime-gpu
   ```

### Slow First Inference

This is normal for TensorRT - engine building happens on first inference. Enable caching to avoid rebuilding:

```rust
let config = TensorRTConfig {
    cache_dir: Some("./tensorrt_cache".to_string()),
    ..Default::default()
};
```

### Out of Memory

Reduce memory pool size or batch size:

```rust
let config = CudaOptimizerBuilder::new()
    .memory_pool_size(512, 4096)  // Smaller pool
    .memory_fraction(0.7)         // Use less GPU memory
    .build();
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TENSORRT_PRECISION` | Override precision (fp32/fp16/int8) |
| `TENSORRT_WORKSPACE_SIZE_MB` | Workspace size override |
| `TENSORRT_MAX_BATCH_SIZE` | Max batch size override |
| `USE_FP16` | Enable FP16 (1/true) |
| `TENSORRT_ROOT` | TensorRT installation path |
| `CUDA_PATH` | CUDA installation path |

## Summary

The torch-inference framework provides world-class CUDA and TensorRT optimization through:

- **Automatic detection** of hardware capabilities
- **Optimal default configurations** for common use cases
- **Fine-grained control** for advanced tuning
- **Comprehensive benchmarking** for validation

For maximum performance:
1. Use TensorRT with FP16 precision
2. Enable engine and timing caches
3. Use appropriate batch sizes for your workload
4. Warm up models before production use

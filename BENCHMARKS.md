# Benchmark Documentation

## Overview

This document describes the benchmarking infrastructure for the PyTorch Inference Framework.

## Running Benchmarks

### Model Benchmark

Benchmarks individual model load times and inference latency:

```bash
cargo bench --bench comprehensive_api_benchmark
```

### Concurrent Throughput Benchmark

Measures throughput scaling across different concurrency levels:

```bash
cargo bench --bench concurrent_throughput_benchmark
```

## Output Files

Benchmarks generate the following files in `benchmark_results/`:

| File | Description |
|------|-------------|
| `*.csv` | Raw benchmark data in CSV format |
| `*.json` | Structured data with system info |
| `*.md` | Human-readable markdown report |
| `*_throughput.png` | Throughput vs concurrency chart |
| `*_latency.png` | Latency percentiles chart |
| `*_scaling.png` | Scaling efficiency chart |

## Metrics Collected

### Model Benchmark
- Load time (ms)
- Inference latency (avg, min, max, std dev)
- Throughput (req/s)
- File size
- Input shape
- Device (CPU/CUDA/MPS)

### Concurrent Benchmark
- Throughput at each concurrency level
- Latency percentiles (P50, P75, P90, P95, P99)
- Scaling efficiency
- Success/failure counts

## Configuration

Benchmark settings can be modified in `benches/*.rs`:

```rust
struct BenchmarkConfig {
    concurrency_levels: vec![1, 2, 4, 8, 16, 32, 64],
    requests_per_level: 100,
    warmup_requests: 10,
    output_dir: "benchmark_results".to_string(),
}
```

## Interpreting Results

### Throughput
Higher is better. Measures requests processed per second.

### Latency
Lower is better. P95 and P99 indicate tail latency.

### Scaling Efficiency
100% = perfect linear scaling. Lower values indicate bottlenecks.

## Hardware Recommendations

For optimal benchmarks:
- Use release builds: `cargo bench --release`
- Close other applications
- Run multiple iterations for consistency
- Monitor system resources during benchmark

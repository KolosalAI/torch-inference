# Benchmark Quick Reference

## Quick Start

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench comprehensive_api_benchmark
cargo bench --bench concurrent_throughput_benchmark
```

## Results Location

All results are saved to `benchmark_results/` directory.

## Available Benchmarks

| Benchmark | Description |
|-----------|-------------|
| `comprehensive_api_benchmark` | Model load time and inference latency |
| `concurrent_throughput_benchmark` | Throughput scaling with concurrency |

## Example Output

```
╔══════════════════════════════════════════════════════════╗
║     Concurrent Throughput Benchmark                      ║
╚══════════════════════════════════════════════════════════╝

═══ Testing: example ═══
✓ Model loaded
✓ Working input shape (float): image_224 [1, 3, 224, 224]
  Warming up (10 requests)...
  Concurrency  1:   100.00 req/s | avg:  10.00ms | p95:  12.00ms | p99:  15.00ms
  Concurrency  2:   180.00 req/s | avg:  11.00ms | p95:  14.00ms | p99:  18.00ms
  Concurrency  4:   320.00 req/s | avg:  12.50ms | p95:  16.00ms | p99:  20.00ms
```

## See Also

- [BENCHMARKS.md](BENCHMARKS.md) - Full documentation
- [README.md](README.md) - Project overview

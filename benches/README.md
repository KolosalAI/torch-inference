# Benchmarks

This directory contains benchmark suites for the torch-inference framework with comprehensive reporting capabilities.

## Available Benchmarks

### 1. Cache Benchmarks (`cache_bench.rs`)
Tests caching performance with different operations and sizes:
- Cache set operations
- Cache get operations
- Cache cleanup/expiration

### 2. Model Inference Benchmarks (`model_inference_bench_with_report.rs`)
Comprehensive benchmarks for model inference (excluding TTS):
- Model loading performance
- Image preprocessing
- Image classification (full pipeline and inference-only)
- Batch processing
- Memory operations

### 3. Legacy Benchmarks
- `model_inference_bench.rs` - Original benchmarks including TTS
- `model_inference_bench_optimized.rs` - Optimized variants

### 4. Batch and Concurrent Scaling Benchmarks (`batch_concurrent_bench.rs`)
**NEW!** Comprehensive scaling tests for production scenarios:
- **Batch Inference Scaling**: Tests batch sizes 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
- **Concurrent Request Scaling**: Tests concurrent requests 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
- Tests all image classification models (TTS excluded)
- Measures throughput, latency, and scaling efficiency
- Identifies optimal batch size and concurrency levels

## Running Benchmarks

### Quick Start
```bash
# Run cache benchmarks
cargo bench --bench cache_bench

# Run model inference benchmarks (requires torch feature)
cargo bench --bench model_inference_bench_with_report --features torch

# Run batch and concurrent scaling benchmarks (requires torch)
cargo bench --bench batch_concurrent_bench --features torch

# Run all benchmarks
cargo bench --features torch
```

### Advanced Options
```bash
# Run specific benchmark by name
cargo bench --bench cache_bench cache_set

# Run batch scaling benchmarks only
cargo bench --bench batch_concurrent_bench batch_inference_scaling --features torch

# Run concurrent scaling benchmarks only
cargo bench --bench batch_concurrent_bench concurrent_inference_scaling --features torch

# Save results with custom Criterion settings
cargo bench --bench cache_bench -- --save-baseline my_baseline

# Compare with previous baseline
cargo bench --bench cache_bench -- --baseline my_baseline
```

## Report Generation

All benchmarks automatically generate reports in two formats:

### CSV Format (`benches/data/*.csv`)
- Easy to analyze with Excel, LibreOffice, pandas
- Contains: timestamps, benchmark names, execution times, system info
- Appends to existing files for historical tracking
- **Not ignored by git** for version control tracking

### JSON Format (`benches/data/*.json`)
- Structured data for programmatic analysis
- Contains same data as CSV with full metadata
- One file per benchmark run with timestamp

## Analyzing Results

### Using the Analysis Tools (Rust binaries)

**For standard benchmarks:**
```bash
# Analyze latest benchmark results
cargo run --bin analyze-benchmarks

# Analyze specific CSV file
cargo run --bin analyze-benchmarks -- cache_benchmark
```

**For batch and concurrent scaling benchmarks:**
```bash
# Analyze scaling characteristics
cargo run --bin analyze-scaling

# Analyze specific scaling results
cargo run --bin analyze-scaling -- batch_concurrent
```

**For model inference tables:**
```bash
# Generate comprehensive model inference tables
cargo run --bin generate-inference-table

# Output to file
cargo run --bin generate-inference-table > MODEL_INFERENCE_RESULTS.md
```

The scaling analysis provides:
- Scaling efficiency metrics (100% = perfect linear scaling)
- Parallel efficiency metrics (100% = no concurrency overhead)
- Optimal batch size and concurrency recommendations
- Throughput analysis across different loads

### Using Python/Pandas
```python
import pandas as pd

# Load results
df = pd.read_csv('benches/data/cache_benchmark_20231220_120000.csv')

# Group by benchmark and show statistics
summary = df.groupby('benchmark_name')['mean_time_ms'].describe()
print(summary)

# Compare different parameters
pivot = df.pivot_table(
    values='mean_time_ms',
    index='benchmark_name',
    columns='parameter'
)
print(pivot)
```

### Using Excel
1. Open CSV file in Excel or LibreOffice Calc
2. Create pivot tables for comparisons
3. Generate charts for visualization
4. Track performance over time

## Understanding Results

### CSV Columns
- `timestamp`: When the benchmark was run (ISO 8601)
- `benchmark_name`: Name of the benchmark
- `model_name`: Model being tested (if applicable)
- `parameter`: Additional parameters (e.g., `batch_64`, `concurrent_128`)
- `mean_time_ms`: Average execution time in milliseconds
- `std_dev_ms`: Standard deviation
- `median_ms`: Median execution time
- `min_ms`/`max_ms`: Min and max times observed
- `sample_count`: Number of samples collected
- `iterations`: Iterations per sample
- `throughput_ops_per_sec`: Operations per second
- System info: `os`, `cpu_model`, `cpu_count`, `total_memory_mb`, `hostname`

### Interpreting Results
- **Lower times = better performance**
- **Lower std_dev = more consistent**
- **Higher throughput = better**
- Compare `mean_time_ms` across different models/parameters
- Check `throughput_ops_per_sec` for operations-based metrics

### Scaling Metrics (for batch_concurrent_bench)
- **Scaling Efficiency**: How well performance improves with batch size
  - 100% = perfect linear scaling (2x batch = 2x throughput)
  - >100% = super-linear scaling (unlikely, check cache effects)
  - <100% = sub-linear scaling (overhead from larger batches)
  
- **Parallel Efficiency**: How well concurrent requests are handled
  - 100% = no overhead from concurrency
  - <100% = concurrency overhead (contention, context switching)
  
- **Optimal Batch Size**: The batch size with highest throughput
- **Optimal Concurrency**: The concurrency level with best efficiency/throughput balance

## Customizing Benchmarks

### Adding New Benchmarks
1. Create a new benchmark function in the bench file
2. Use the `REPORTER` to track results
3. Add to `criterion_group!` macro
4. Ensure `save_reports` is last in the group

Example:
```rust
fn my_new_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_benchmark");
    
    group.bench_function("my_test", |b| {
        b.iter(|| {
            // Your benchmark code here
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    my_new_benchmark,
    save_reports  // Always last
);
```

### Configuring Criterion
Modify group settings for specific needs:
```rust
let mut group = c.benchmark_group("my_benchmark");
group.sample_size(100);                      // Number of samples
group.measurement_time(Duration::from_secs(60)); // Total time
group.warm_up_time(Duration::from_secs(10));     // Warmup period
```

## Best Practices

1. **Consistent Environment**
   - Close unnecessary applications
   - Disable CPU frequency scaling if possible
   - Run on same machine for comparisons

2. **Multiple Runs**
   - Run benchmarks multiple times
   - Check for consistency across runs
   - Look for outliers in the data

3. **Version Control**
   - Commit CSV/JSON results to track performance over time
   - Tag important benchmark runs
   - Document significant changes

4. **Analysis**
   - Use the provided analysis script
   - Create visualizations for trends
   - Compare before/after optimization

## Troubleshooting

### Benchmark Fails to Compile
```bash
# Check if torch feature is needed
cargo check --bench <bench_name> --features torch

# Verify all dependencies
cargo update
```

### No Data Generated
- Check that `save_reports` function is in the criterion_group
- Verify `benches/data` directory exists
- Check file permissions

### Inconsistent Results
- Close background applications
- Run multiple times and average
- Check CPU frequency scaling settings
- Monitor system temperature

## Output Locations

- **CSV Reports**: `benches/data/*_benchmark_*.csv`
- **JSON Reports**: `benches/data/*_benchmark_*.json`
- **Criterion HTML**: `target/criterion/` (detailed Criterion reports)
- **Analysis Summary**: `benches/data/latest_summary.json`
- **Scaling Analysis**: `benches/data/scaling_analysis.json` (from batch/concurrent benchmarks)

## Dependencies

- `criterion` - Benchmarking framework
- `chrono` - Timestamp generation
- `sysinfo` - System information
- `lazy_static` - Static reporter instance
- `serde` / `serde_json` - Serialization for JSON
- Built-in Rust binaries for analysis (no Python required)

## Notes

- TTS benchmarks are excluded in `model_inference_bench_with_report.rs` as requested
- CSV files automatically append to allow historical tracking
- System information is captured for reproducibility
- All times are stored in nanoseconds internally, converted to ms for display
- Benchmarks run with `--release` profile optimizations

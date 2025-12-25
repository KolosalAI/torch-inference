# Benchmark Quick Start Guide

## 🚀 Running Benchmarks

### Cache Benchmarks (No torch required)
```bash
cargo bench --bench cache_bench
```

Results appear in:
- Console: Summary table with system info
- CSV: `benches/data/cache_benchmark_YYYYMMDD_HHMMSS.csv`
- JSON: `benches/data/cache_benchmark_YYYYMMDD_HHMMSS.json`

### Model Inference Benchmarks (Requires torch)
```bash
cargo bench --bench model_inference_bench_with_report --features torch
```

### Batch and Concurrent Scaling Benchmarks (Requires torch)
**NEW!** Test scaling from 1 to 1024 batch/concurrent requests:
```bash
cargo bench --bench batch_concurrent_bench --features torch
```

This benchmark tests:
- **Batch sizes**: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
- **Concurrent requests**: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
- All image classification models
- Identifies optimal batch size and concurrency levels

### All Benchmarks
```bash
cargo bench --features torch
```

### Quick Test (Faster, fewer samples)
```bash
cargo bench --bench cache_bench -- --quick
```

## 📊 Analyzing Results

### Option 1: Rust Binaries (Built-in, No Dependencies)

**Standard Benchmarks:**
```bash
cargo run --bin analyze-benchmarks
```

**Batch/Concurrent Scaling Benchmarks:**
```bash
cargo run --bin analyze-scaling
```

**Model Inference Tables:**
```bash
cargo run --bin generate-inference-table > MODEL_INFERENCE_RESULTS.md
```

Output includes:
- Console summary with system info
- Grouped results by benchmark
- Model comparisons
- **Scaling efficiency metrics** (for batch/concurrent benchmarks)
- **Optimal batch size and concurrency recommendations**
- Generates JSON summaries

### Option 2: Excel/LibreOffice
```bash
open benches/data/cache_benchmark_*.csv  # macOS
# or
libreoffice benches/data/cache_benchmark_*.csv  # Linux
```

### Option 3: Python/Pandas
```python
import pandas as pd

# Load data
df = pd.read_csv('benches/data/cache_benchmark_20231220_120000.csv')

# Quick summary
print(df.groupby('benchmark_name')['mean_time_ms'].describe())

# Compare parameters
pivot = df.pivot_table(
    values='mean_time_ms',
    index='benchmark_name',
    columns='parameter'
)
print(pivot)

# Plot results
import matplotlib.pyplot as plt
df.groupby('parameter')['mean_time_ms'].mean().plot(kind='bar')
plt.ylabel('Mean Time (ms)')
plt.title('Benchmark Performance')
plt.show()
```

## 📁 File Locations

```
benches/data/
├── cache_benchmark_20251221_051700.csv      # CSV results
├── cache_benchmark_20251221_051700.json     # JSON results
├── latest_summary.json                       # Latest analysis
└── README.md                                 # Format documentation
```

## 🔍 Understanding Results

### CSV Columns
- `mean_time_ms`: Average execution time ⬇️ Lower is better
- `median_ms`: Middle value (less affected by outliers)
- `std_dev_ms`: Consistency ⬇️ Lower = more consistent
- `throughput_ops_per_sec`: Operations per second ⬆️ Higher is better
- `parameter`: Batch size (e.g., `batch_64`) or concurrency (e.g., `concurrent_128`)

### Scaling Metrics (from analyze_scaling.py)
- **Scaling Efficiency**: 100% = perfect linear scaling
  - Higher batch size should increase throughput proportionally
  - <100% indicates overhead from larger batches
  
- **Parallel Efficiency**: 100% = no concurrency overhead
  - Measures how well concurrent requests are handled
  - <100% indicates contention or context switching overhead

### Example Output
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━cache_set━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model/Parameter               Mean (ms)   Median (ms)  Std Dev (ms)
────────────────────────────────────────────────────────────────────────────
size_100                         0.0024       0.0003       0.0055
size_1000                        0.0006       0.0003       0.0013
size_10000                       0.0005       0.0002       0.0011
```

**Batch Scaling Example:**
```
Model: mobilenetv4-hybrid-large
Batch Size      Mean Time (ms)    Throughput (ops/s)    Scaling Efficiency
────────────────────────────────────────────────────────────────────────────
1                       12.5000              80.00                    100.0%
2                       23.0000              86.96                     91.3%
4                       44.0000              90.91                     89.8%
8                       85.0000              94.12                     88.2%
```

This shows that throughput increases with batch size but with diminishing returns (sub-linear scaling).

## 🎯 Best Practices

1. **Consistent Environment**
   ```bash
   # Close unnecessary apps
   # Run multiple times
   cargo bench --bench cache_bench
   cargo bench --bench cache_bench
   cargo bench --bench cache_bench
   ```

2. **Track Over Time**
   ```bash
   git add benches/data/*.csv
   git commit -m "Benchmark results: <description>"
   ```

3. **Compare Branches**
   ```bash
   # On main branch
   cargo bench --bench cache_bench
   mv benches/data/cache_benchmark_*.csv results_main.csv
   
   # On feature branch
   git checkout feature
   cargo bench --bench cache_bench
   
   # Compare
   cargo run --bin analyze-benchmarks -- cache_benchmark_main
   cargo run --bin analyze-benchmarks -- cache_benchmark
   ```

4. **Finding Optimal Settings**
   ```bash
   # Run scaling benchmarks
   cargo bench --bench batch_concurrent_bench --features torch
   
   # Analyze to find optimal batch size and concurrency
   cargo run --bin analyze-scaling
   
   # Look for:
   # - Highest throughput batch size
   # - Best parallel efficiency concurrency level
   # - Point where scaling efficiency drops below 80%
   ```

## 🛠️ Troubleshooting

### "torch feature not enabled"
```bash
cargo bench --features torch
```

### "No CSV files found"
```bash
# Run benchmarks first
cargo bench --bench cache_bench
```

### "Benchmark takes too long"
```bash
# Use quick mode
cargo bench --bench cache_bench -- --quick
```

## 📚 More Information

- Full docs: `benches/README.md`
- Data format: `benches/data/README.md`
- Implementation: `BENCHMARK_IMPLEMENTATION.md`
- Analysis tools: Rust binaries (no Python needed)
  - `cargo run --bin analyze-benchmarks`
  - `cargo run --bin analyze-scaling`  
  - `cargo run --bin generate-inference-table`

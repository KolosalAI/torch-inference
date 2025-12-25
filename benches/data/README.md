# Benchmark Data Directory

This directory contains benchmark results in both CSV and JSON formats for easy analysis and tracking over time.

## File Format

### CSV Format
The CSV files contain the following columns:
- `timestamp`: ISO 8601 timestamp when the benchmark was run
- `benchmark_name`: Name of the benchmark (e.g., "cache_set", "image_preprocessing")
- `model_name`: Name of the model being benchmarked (if applicable)
- `parameter`: Additional parameter information (e.g., "size_1000")
- `mean_time_ms`: Mean execution time in milliseconds
- `std_dev_ms`: Standard deviation in milliseconds
- `median_ms`: Median execution time in milliseconds
- `min_ms`: Minimum execution time in milliseconds
- `max_ms`: Maximum execution time in milliseconds
- `sample_count`: Number of samples collected
- `iterations`: Number of iterations per sample
- `throughput_ops_per_sec`: Operations per second (if applicable)
- `os`: Operating system
- `cpu_model`: CPU model name
- `cpu_count`: Number of CPU cores
- `total_memory_mb`: Total system memory in MB
- `hostname`: Machine hostname

### JSON Format
JSON files contain the same data in a structured format, allowing for easy programmatic analysis.

## Running Benchmarks

### Cache Benchmarks
```bash
cargo bench --bench cache_bench
```

### Model Inference Benchmarks (with reporting)
```bash
cargo bench --bench model_inference_bench_with_report --features torch
```

### All Benchmarks
```bash
cargo bench --features torch
```

## Analyzing Results

### Using Python/Pandas
```python
import pandas as pd

# Load benchmark results
df = pd.read_csv('benches/data/cache_benchmark_20231220_120000.csv')

# Group by benchmark name and calculate statistics
summary = df.groupby('benchmark_name').agg({
    'mean_time_ms': ['mean', 'std', 'min', 'max']
})
print(summary)

# Compare different models
model_comparison = df[df['model_name'].notna()].pivot_table(
    values='mean_time_ms',
    index='benchmark_name',
    columns='model_name'
)
print(model_comparison)
```

### Using Excel/LibreOffice
Simply open the CSV files in Excel or LibreOffice Calc for visual analysis, charting, and comparison.

## File Naming Convention

Files are named with the following pattern:
```
{benchmark_name}_{timestamp}.{extension}
```

Example:
- `cache_benchmark_20231220_143052.csv`
- `model_inference_benchmark_20231220_143052.json`

## Notes

- These files are **NOT** ignored by git, allowing you to track benchmark performance over time
- Results include system information for reproducibility
- All times are in nanoseconds internally but converted to milliseconds for CSV output
- The reporter automatically appends to existing CSV files with matching names

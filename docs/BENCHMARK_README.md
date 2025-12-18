# Complete Benchmark Testing Suite - Documentation

This directory contains comprehensive benchmark testing tools for evaluating image classification model performance.

---

## Available Benchmark Tools

### 1. **benchmark_models.sh** (Shell Script)
Fast, lightweight benchmark using bash and curl.

**Features:**
- Inference speed measurement
- Throughput testing
- Memory usage tracking
- Multi-image testing
- Markdown report generation

**Usage:**
```bash
chmod +x benchmark_models.sh
./benchmark_models.sh
```

**Best for:** Quick benchmarks, CI/CD integration, minimal dependencies

---

### 2. **benchmark_advanced.py** (Python Script)
Advanced benchmark with statistical analysis.

**Features:**
- Detailed statistical analysis (mean, median, stdev)
- Per-image performance breakdown
- JSON and Markdown reports
- Configurable iterations
- Better error handling

**Usage:**
```bash
# Default (10 iterations per test)
python3 benchmark_advanced.py

# Custom iterations
python3 benchmark_advanced.py 20
```

**Requirements:**
```bash
pip install requests
```

**Best for:** Detailed analysis, research, publication-quality results

---

## Quick Start

### Option 1: Shell Script (Recommended)
```bash
# 1. Start server
./target/release/torch-inference-server &

# 2. Run benchmark
./benchmark_models.sh
```

### Option 2: Python Script
```bash
# 1. Start server
./target/release/torch-inference-server &

# 2. Run benchmark
python3 benchmark_advanced.py
```

---

## Output Files

### Directory Structure
```
benchmark_results/
├── benchmark_20251218_031000.json    # Raw data
├── benchmark_20251218_031000.md      # Human-readable report
└── ...

benchmark_images/
├── cat.jpg                            # Test images
├── dog.jpg
├── bird.jpg
└── car.jpg
```

### JSON Report Format
```json
{
  "timestamp": "20251218_031000",
  "date": "2025-12-18T03:10:00",
  "models_tested": 5,
  "iterations_per_test": 10,
  "results": [
    {
      "model_id": "mobilenetv4-hybrid-large",
      "size": "140MB",
      "accuracy": "84.36%",
      "overall": {
        "mean": 42.5,
        "median": 41.8,
        "stdev": 3.2,
        "min": 38.2,
        "max": 49.5,
        "fps": 23.53
      },
      "images": {
        "cat": {...},
        "dog": {...}
      }
    }
  ]
}
```

### Markdown Report Sections
1. Performance comparison table
2. Detailed per-model results
3. Per-image breakdown
4. Recommendations
5. System information

---

## Metrics Explained

### Inference Time (ms)
Time to process one image through the model.
- **Lower is better**
- Includes preprocessing + inference + postprocessing
- **Target:** < 100ms for real-time

### FPS (Frames Per Second)
How many images can be processed per second.
- **Higher is better**
- Calculated as: 1000 / avg_inference_time_ms
- **Target:** > 10 FPS for interactive

### Mean vs Median
- **Mean:** Average of all measurements
- **Median:** Middle value (less affected by outliers)
- Use median for more stable metric

### Standard Deviation
Measure of variance in timing.
- **Lower is better** (more consistent)
- High stdev indicates unstable performance

### Min/Max
- **Min:** Best case performance
- **Max:** Worst case performance
- Large range indicates inconsistency

---

## Benchmark Configuration

### Models Tested
By default benchmarks these models (if downloaded):

| Model | Size | Accuracy | Category |
|-------|------|----------|----------|
| MobileNetV4 | 140 MB | 84.36% | Fast |
| CoAtNet-3 | 700 MB | 86.0% | Efficient |
| Swin Large | 790 MB | 87.3% | Balanced |
| EfficientNetV2 | 850 MB | 87.3% | Balanced |
| EVA-02 Large | 1.2 GB | 90.054% | Accurate |

### Test Images
- **Cat** - Domestic animal classification
- **Dog** - Domestic animal classification  
- **Bird** - Wildlife classification
- **Car** - Vehicle classification

### Test Parameters
- **Warmup iterations:** 3 (to warm up caches)
- **Test iterations:** 5-10 per image
- **Top-K predictions:** 5
- **Timeout:** 30 seconds per request

---

## Customization

### Adding Models

**Shell script (`benchmark_models.sh`):**
```bash
declare -a MODELS=(
    "your-model-id:size"
    "mobilenetv4-hybrid-large:140MB"
)
```

**Python script (`benchmark_advanced.py`):**
```python
self.models = [
    {"id": "your-model-id", "size": "XXX", "accuracy": "XX.X%"},
]
```

### Adding Test Images

**Shell script:**
```bash
declare -a TEST_IMAGES=(
    "name:https://url-to-image.jpg"
)
```

**Python script:**
```python
self.test_images = [
    {"name": "test", "url": "https://url-to-image.jpg"},
]
```

### Changing Iterations

**Shell script:**
```bash
# Line ~140
for i in {1..10}; do  # Change 10 to desired value
```

**Python script:**
```bash
python3 benchmark_advanced.py 20  # Command line argument
```

---

## Performance Targets

### Real-time Applications
- **Target:** < 50ms inference time
- **FPS:** > 20
- **Models:** MobileNetV4, CoAtNet-3

### Interactive Applications  
- **Target:** 50-100ms inference time
- **FPS:** 10-20
- **Models:** EfficientNetV2, Swin Transformer

### Batch Processing
- **Target:** < 500ms inference time
- **FPS:** > 2
- **Models:** EVA-02 Large, ConvNeXt V2 Huge

### Production Recommendations
- **CPU:** MobileNetV4 or EfficientNetV2
- **GPU:** EVA-02 Large or ConvNeXt V2 Huge
- **Edge:** MobileNetV4 only

---

## GPU Benchmarking

To benchmark with GPU acceleration:

### 1. Build with CUDA
```bash
cargo build --release --features "torch cuda"
```

### 2. Set GPU Device
```bash
export CUDA_VISIBLE_DEVICES=0
```

### 3. Run Benchmark
```bash
./benchmark_models.sh
# or
python3 benchmark_advanced.py
```

### Expected Results
- **CPU to GPU speedup:** 5-10x
- **Example:** 100ms → 10-20ms
- **FPS improvement:** 10 → 50-100

---

## Continuous Benchmarking

### Daily Benchmarks
```bash
# Add to crontab
0 2 * * * cd /path/to/project && ./benchmark_models.sh
```

### CI/CD Integration
```yaml
# .github/workflows/benchmark.yml
- name: Run Benchmarks
  run: |
    ./target/release/torch-inference-server &
    sleep 10
    ./benchmark_models.sh
    
- name: Upload Results
  uses: actions/upload-artifact@v2
  with:
    name: benchmark-results
    path: benchmark_results/
```

---

## Analyzing Results

### Compare Models
```bash
# View comparison table
cat benchmark_results/benchmark_YYYYMMDD_HHMMSS.md | grep "^|"
```

### Find Fastest Model
```bash
# Sort by mean time
jq '.results | sort_by(.overall.mean) | .[0]' \
   benchmark_results/benchmark_YYYYMMDD_HHMMSS.json
```

### Track Performance Over Time
```bash
# Compare two benchmark runs
diff benchmark_results/benchmark_20251218.json \
     benchmark_results/benchmark_20251219.json
```

---

## Troubleshooting

### "Server is not running"
```bash
./target/release/torch-inference-server &
sleep 5
```

### "Model not found"
Download the model first:
```bash
curl -X POST http://localhost:8000/models/download \
  -H "Content-Type: application/json" \
  -d '{"model_name":"mobilenetv4-hybrid-large",...}'
```

### Slow performance
- Use GPU if available
- Reduce image size
- Use smaller model
- Check system resources

### Inconsistent results
- Run more iterations
- Close background apps
- Use dedicated hardware
- Enable CPU affinity

---

## Best Practices

1. **Warmup First**
   - Always run 3-5 warmup iterations
   - Ensures caches are populated
   - More accurate results

2. **Multiple Iterations**
   - Run at least 10 iterations per test
   - Calculate statistical metrics
   - Identify outliers

3. **Consistent Environment**
   - Close unnecessary apps
   - Same test images
   - Same server config

4. **Document Changes**
   - Track code changes
   - Note configuration differences
   - Compare results over time

5. **Test Multiple Images**
   - Different image categories
   - Various image sizes
   - Edge cases

---

## Next Steps

After benchmarking:

1. **Choose Model**
   - Based on speed/accuracy tradeoff
   - Consider deployment constraints

2. **Optimize**
   - Enable GPU if available
   - Tune preprocessing
   - Batch processing

3. **Deploy**
   - Set up monitoring
   - Load testing
   - Production deployment

4. **Monitor**
   - Track real-world performance
   - Compare to benchmarks
   - Identify bottlenecks

---

## Example Benchmark Results

### Typical CPU Performance (Apple M4)

| Model | Inference Time | FPS | Use Case |
|-------|----------------|-----|----------|
| MobileNetV4 | 40-50ms | ~22 | ⚡ Real-time |
| EfficientNetV2 | 70-90ms | ~13 | Interactive |
| EVA-02 Large | 110-130ms | ~8 | Batch |

### Expected GPU Performance (NVIDIA RTX 4090)

| Model | Inference Time | FPS | Speedup |
|-------|----------------|-----|---------|
| MobileNetV4 | 5-8ms | ~150 | 8x |
| EfficientNetV2 | 10-15ms | ~80 | 6x |
| EVA-02 Large | 15-20ms | ~55 | 7x |

---

**Ready to benchmark! Start with `./benchmark_models.sh` for quick results.**

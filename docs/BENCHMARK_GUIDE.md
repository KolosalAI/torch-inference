# Image Classification Benchmark Testing Guide

This guide explains how to run comprehensive benchmarks on all SOTA image classification models.

---

## Quick Start

```bash
# 1. Ensure server is running with torch support
./target/release/torch-inference-server &

# 2. Run benchmarks
chmod +x benchmark_models.sh
./benchmark_models.sh
```

---

## What Gets Benchmarked

### 1. **Inference Speed**
- Average inference time per image (ms)
- FPS (frames per second)
- Min/max inference times
- Tested across multiple image categories

### 2. **Throughput**
- Concurrent request handling
- Requests per second
- Server scalability

### 3. **Memory Usage**
- Memory before/after inference
- Memory delta per model
- Peak memory usage

### 4. **Model Comparison**
- Performance across different architectures
- Size vs speed tradeoff
- Accuracy vs speed tradeoff

---

## Models Benchmarked

By default, tests these models (if downloaded):

1. **MobileNetV4 Hybrid Large** (140 MB) - Fastest
2. **CoAtNet-3** (700 MB) - Efficient hybrid
3. **Swin Transformer Large** (790 MB) - Hierarchical
4. **EfficientNetV2 XL** (850 MB) - Balanced
5. **EVA-02 Large** (1.2 GB) - Highest accuracy

---

## Test Images

Benchmarks use 4 different image categories:
- **Cat** - Domestic animal
- **Dog** - Domestic animal
- **Bird** - Wildlife
- **Car** - Vehicle

Each model is tested with all images, 5 iterations each.

---

## Output Files

### 1. JSON Results
```
benchmark_results/benchmark_YYYYMMDD_HHMMSS.json
```

Contains raw timing data:
```json
[
  {
    "model": "mobilenetv4-hybrid-large",
    "size": "140MB",
    "avg_time_ms": 45.2,
    "fps": 22.12
  },
  ...
]
```

### 2. Markdown Report
```
benchmark_results/benchmark_YYYYMMDD_HHMMSS.md
```

Contains:
- System information
- Performance comparison table
- Recommendations
- Detailed analysis

### 3. Downloaded Images
```
benchmark_images/
  ├── cat.jpg
  ├── dog.jpg
  ├── bird.jpg
  └── car.jpg
```

---

## Benchmark Process

For each model:

1. **Check Availability** - Verify model is downloaded
2. **Warmup** - 3 iterations to warm up caches
3. **Benchmark** - 5 iterations per test image
4. **Statistics** - Calculate avg, min, max
5. **Additional Tests** - Throughput and memory (first model only)

---

## Interpreting Results

### Inference Time
- **< 50ms** - Excellent for real-time
- **50-100ms** - Good for interactive
- **100-200ms** - Acceptable for batch
- **> 200ms** - Consider GPU or smaller model

### FPS (Frames Per Second)
- **> 20 FPS** - Real-time capable
- **10-20 FPS** - Interactive
- **< 10 FPS** - Batch processing only

### Memory Usage
- Baseline: Server memory usage
- Delta: Additional memory per inference
- Helps determine server capacity

---

## Advanced Usage

### Customize Models to Test

Edit `benchmark_models.sh`:

```bash
declare -a MODELS=(
    "mobilenetv4-hybrid-large:140MB"
    "efficientnetv2-xl:850MB"
    "eva02-large-patch14-448:1.2GB"
    # Add more models here
)
```

### Add More Test Images

```bash
declare -a TEST_IMAGES=(
    "cat:https://example.com/cat.jpg"
    "custom:https://example.com/custom.jpg"
)
```

### Change Iterations

```bash
# Warmup iterations (line ~120)
for i in {1..3}; do  # Change 3 to your value

# Test iterations (line ~140)
for i in {1..5}; do  # Change 5 to your value
```

---

## Example Output

```
╔═══════════════════════════════════════════════════════════════╗
║     Image Classification Benchmark Suite                     ║
╚═══════════════════════════════════════════════════════════════╝

Checking server status...
✓ Server is running

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Downloading Test Images
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ cat.jpg already exists
Downloading dog.jpg... ✓
Downloading bird.jpg... ✓
Downloading car.jpg... ✓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Benchmarking: mobilenetv4-hybrid-large (140MB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Checking if model is downloaded...
✓ Model available

Running warmup (3 iterations)...
... done

Benchmarking inference times:

  cat.jpg: 42.5ms (min: 38.2ms, max: 47.1ms)
  dog.jpg: 43.1ms (min: 39.5ms, max: 48.3ms)
  bird.jpg: 41.8ms (min: 37.9ms, max: 46.2ms)
  car.jpg: 44.2ms (min: 40.1ms, max: 49.5ms)

Summary:
  Average inference time: 42.9ms
  Images per second: 23.31

Testing throughput with 10 concurrent requests...
  Completed 10 requests in 2s
  Throughput: 5.00 requests/sec

Testing memory usage...
  Memory before: 245.32MB
  Memory after: 267.18MB
  Difference: 21.86MB

...

╔═══════════════════════════════════════════════════════════════╗
║                  Benchmark Complete                          ║
╚═══════════════════════════════════════════════════════════════╝

Results saved to:
  - JSON: benchmark_results/benchmark_20251218_031000.json
  - Report: benchmark_results/benchmark_20251218_031000.md
```

---

## Benchmark Report Example

The generated report includes:

### Performance Table
| Model | Size | Avg Time (ms) | FPS | Images/sec |
|-------|------|---------------|-----|------------|
| MobileNetV4 | 140 MB | 42.9 | 23.31 | 23.31 |
| CoAtNet-3 | 700 MB | 67.5 | 14.81 | 14.81 |
| EfficientNetV2 | 850 MB | 78.2 | 12.79 | 12.79 |
| EVA-02 Large | 1.2 GB | 112.4 | 8.90 | 8.90 |

### Recommendations
- **Real-time:** Use MobileNetV4 (fastest)
- **Accuracy:** Use EVA-02 Large (best accuracy)
- **Balance:** Use EfficientNetV2 XL

---

## GPU Benchmarking

To enable GPU acceleration:

1. **Build with CUDA:**
   ```bash
   cargo build --release --features "torch cuda"
   ```

2. **Set CUDA device:**
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

3. **Run benchmarks:**
   ```bash
   ./benchmark_models.sh
   ```

Expected GPU speedup: **5-10x faster** than CPU

---

## Continuous Benchmarking

### Run Daily Benchmarks
```bash
# Add to crontab
0 2 * * * cd /path/to/project && ./benchmark_models.sh
```

### Track Performance Over Time
```bash
# Compare results
diff benchmark_results/benchmark_20251218.json \
     benchmark_results/benchmark_20251219.json
```

---

## Troubleshooting

### Server not running
```bash
./target/release/torch-inference-server &
sleep 5
```

### Model not downloaded
```bash
curl -X POST http://localhost:8000/models/download \
  -H "Content-Type: application/json" \
  -d '{"model_name":"mobilenetv4-hybrid-large",...}'
```

### Out of memory
- Reduce number of concurrent requests
- Use smaller models
- Add more RAM or use GPU

---

## Performance Optimization Tips

### 1. Use GPU
- 5-10x faster than CPU
- Required for production

### 2. Batch Processing
- Process multiple images together
- Better GPU utilization

### 3. Model Selection
- Smaller models for real-time
- Larger models for accuracy

### 4. Image Preprocessing
- Resize before sending
- Use proper normalization

### 5. Server Configuration
- Increase worker threads
- Enable model caching
- Use connection pooling

---

## Next Steps

After benchmarking:

1. **Analyze Results** - Review the generated report
2. **Choose Model** - Based on your requirements
3. **Optimize** - Tune for your specific use case
4. **Deploy** - Move to production
5. **Monitor** - Track performance metrics

---

**Ready to benchmark! Run `./benchmark_models.sh` to get started.**

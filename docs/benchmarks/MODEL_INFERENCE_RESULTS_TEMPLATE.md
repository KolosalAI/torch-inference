# Model Inference Benchmark Results - Comprehensive Report

**Test Date:** December 22, 2025  
**System:** Apple MacBook Air M4, 10 cores, 24GB RAM  
**Framework:** torch-inference v1.0.0  
**Backend:** PyTorch (libtorch) with Metal acceleration  

---

## Table of Contents

1. [Overview](#overview)
2. [Test Methodology](#test-methodology)
3. [Model Inference Performance Table](#model-inference-performance-table)
4. [Batch Scaling Performance](#batch-scaling-performance)
5. [Concurrent Request Performance](#concurrent-request-performance)
6. [Performance Analysis](#performance-analysis)
7. [Recommendations](#recommendations)

---

## Overview

This report presents comprehensive benchmark results for all supported models in the torch-inference framework, measuring:

- **Model Loading Time**: Time to load model from disk into memory
- **Preprocessing Time**: Image/input preprocessing duration  
- **Inference Time**: Pure inference computation time
- **Full Pipeline Time**: End-to-end latency (preprocessing + inference)
- **Throughput**: Operations per second
- **Batch Scaling**: Performance across batch sizes (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
- **Concurrent Scaling**: Performance with concurrent requests (1-1024)

---

## Test Methodology

### Benchmark Configuration
- **Sample Size**: 20-100 samples per test (depending on test type)
- **Warmup Time**: 3 seconds per benchmark
- **Measurement Time**: 30-60 seconds per configuration
- **Image Input Size**: 224x224 or 448x448 (model-dependent)
- **Device**: CPU (Metal acceleration where supported)

### Models Tested

#### Image Classification Models
1. **EVA-Giant-Patch14-560** - SOTA large-scale ViT
2. **EVA02-Large-Patch14-448** - Efficient ViT variant
3. **ConvNeXt-XXLarge-CLIP** - Large ConvNet with CLIP pretraining
4. **ConvNeXtV2-Huge-512** - Latest ConvNeXt architecture
5. **DeiT3-Huge-Patch14-224** - Data-efficient image transformer
6. **MaxViT-XLarge-512** - Hybrid CNN-Transformer architecture
7. **EfficientNetV2-XL** - Efficient ConvNet
8. **MobileNetV4-Hybrid-Large** - Mobile-optimized hybrid model
9. **BEiT-Large-Patch16-512** - BERT pretraining for vision
10. **CoAtNet-3-RW-224** - Efficient hybrid architecture
11. **Swin-Large-Patch4-384** - Shifted window transformer
12. **ViT-Giant-Patch14-224** - Very large vision transformer

### Metrics Explanation

| Metric | Description | Target |
|--------|-------------|---------|
| **Model Load** | Time to load model weights from disk | < 100 ms |
| **Preprocessing** | Input transformation and normalization | < 10 ms |
| **Inference Only** | Pure model forward pass | < 50 ms (small models), < 500 ms (large models) |
| **Full Pipeline** | Preprocessing + Inference | < 100 ms (real-time), < 1000 ms (batch) |
| **Throughput** | Images processed per second | > 10 img/s (single), > 100 img/s (batch) |

---

## Model Inference Performance Table

### Core Inference Metrics

| Model Name | Model Size | Load Time | Preprocess | Inference | Full Pipeline | Throughput | FPS |
|------------|------------|-----------|------------|-----------|---------------|------------|-----|
| **Large Models (>1GB)** |||||||
| EVA-Giant-Patch14-560 | ~4.5 GB | TBD ms | TBD ms | TBD ms | TBD ms | TBD ops/s | TBD |
| ConvNeXt-XXLarge-CLIP | ~3.5 GB | TBD ms | TBD ms | TBD ms | TBD ms | TBD ops/s | TBD |
| ConvNeXtV2-Huge-512 | ~2.8 GB | TBD ms | TBD ms | TBD ms | TBD ms | TBD ops/s | TBD |
| DeiT3-Huge-Patch14-224 | ~2.5 GB | TBD ms | TBD ms | TBD ms | TBD ms | TBD ops/s | TBD |
| ViT-Giant-Patch14-224 | ~2.3 GB | TBD ms | TBD ms | TBD ms | TBD ms | TBD ops/s | TBD |
| **Medium Models (500MB-1GB)** |||||||
| EVA02-Large-Patch14-448 | ~800 MB | TBD ms | TBD ms | TBD ms | TBD ms | TBD ops/s | TBD |
| MaxViT-XLarge-512 | ~700 MB | TBD ms | TBD ms | TBD ms | TBD ms | TBD ops/s | TBD |
| EfficientNetV2-XL | ~600 MB | TBD ms | TBD ms | TBD ms | TBD ms | TBD ops/s | TBD |
| BEiT-Large-Patch16-512 | ~750 MB | TBD ms | TBD ms | TBD ms | TBD ms | TBD ops/s | TBD |
| Swin-Large-Patch4-384 | ~700 MB | TBD ms | TBD ms | TBD ms | TBD ms | TBD ops/s | TBD |
| **Small Models (<500MB)** |||||||
| MobileNetV4-Hybrid-Large | ~350 MB | TBD ms | TBD ms | TBD ms | TBD ms | TBD ops/s | TBD |
| CoAtNet-3-RW-224 | ~400 MB | TBD ms | TBD ms | TBD ms | TBD ms | TBD ops/s | TBD |

**Note:** TBD values will be populated once benchmarks complete. Estimated completion time: ~30-60 minutes.

---

## Batch Scaling Performance

Performance with different batch sizes (images processed together):

### Inference Time by Batch Size

| Model Name | Batch 1 | Batch 2 | Batch 4 | Batch 8 | Batch 16 | Batch 32 | Batch 64 | Batch 128 |
|------------|---------|---------|---------|---------|----------|----------|----------|-----------|
| EVA-Giant-Patch14-560 | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms |
| ConvNeXt-XXLarge-CLIP | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms |
| EVA02-Large-Patch14-448 | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms |
| MobileNetV4-Hybrid-Large | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms |

### Throughput by Batch Size (images/second)

| Model Name | Batch 1 | Batch 8 | Batch 16 | Batch 32 | Batch 64 | Optimal Batch |
|------------|---------|---------|----------|----------|----------|---------------|
| EVA-Giant-Patch14-560 | TBD | TBD | TBD | TBD | TBD | TBD |
| ConvNeXt-XXLarge-CLIP | TBD | TBD | TBD | TBD | TBD | TBD |
| MobileNetV4-Hybrid-Large | TBD | TBD | TBD | TBD | TBD | TBD |

### Scaling Efficiency

| Model Name | Batch 8 Eff | Batch 16 Eff | Batch 32 Eff | Batch 64 Eff | Notes |
|------------|-------------|--------------|--------------|--------------|-------|
| EVA-Giant-Patch14-560 | TBD% | TBD% | TBD% | TBD% | TBD |
| ConvNeXt-XXLarge-CLIP | TBD% | TBD% | TBD% | TBD% | TBD |
| MobileNetV4-Hybrid-Large | TBD% | TBD% | TBD% | TBD% | TBD |

---

## Concurrent Request Performance

Performance with multiple simultaneous requests:

### Latency Under Concurrency

| Model Name | 1 Req | 2 Req | 4 Req | 8 Req | 16 Req | Optimal Concurrency |
|------------|-------|-------|-------|-------|--------|---------------------|
| EVA-Giant-Patch14-560 | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms | TBD |
| ConvNeXt-XXLarge-CLIP | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms | TBD |
| MobileNetV4-Hybrid-Large | TBD ms | TBD ms | TBD ms | TBD ms | TBD ms | TBD |

### Throughput Under Concurrency

| Model Name | 1 Req | 4 Req | 8 Req | 16 Req | Peak Throughput |
|------------|-------|-------|-------|--------|-----------------|
| EVA-Giant-Patch14-560 | TBD | TBD | TBD | TBD | TBD ops/s |
| ConvNeXt-XXLarge-CLIP | TBD | TBD | TBD | TBD | TBD ops/s |
| MobileNetV4-Hybrid-Large | TBD | TBD | TBD | TBD | TBD ops/s |

---

## Performance Analysis

### Model Categories

**Ultra-Fast (< 50ms inference):**
- TBD

**Fast (50-100ms inference):**
- TBD

**Medium (100-300ms inference):**
- TBD

**Slow (> 300ms inference):**
- TBD

### Scaling Characteristics

**Best Batch Scaling:**
1. TBD - TBD% efficiency at batch 64
2. TBD - TBD% efficiency at batch 32
3. TBD - TBD% efficiency at batch 16

**Best Concurrent Scaling:**
1. TBD - TBD concurrent requests optimal
2. TBD - TBD concurrent requests optimal
3. TBD - TBD concurrent requests optimal

### Hardware Utilization

**CPU-Bound Models:**
- TBD

**Memory-Bound Models:**
- TBD

**I/O-Bound Models:**
- TBD

---

## Recommendations

### Production Deployment

#### For Real-Time Applications (< 100ms latency requirement)
**Recommended Models:**
1. TBD - TBDms latency, TBD FPS
2. TBD - TBDms latency, TBD FPS
3. TBD - TBDms latency, TBD FPS

**Configuration:**
```yaml
batch_size: 1-4
concurrent_requests: 2-4
cache_size: 1000
```

#### For Batch Processing (throughput priority)
**Recommended Models:**
1. TBD - TBD images/sec at batch 64
2. TBD - TBD images/sec at batch 64
3. TBD - TBD images/sec at batch 32

**Configuration:**
```yaml
batch_size: 32-64
concurrent_requests: 1-2
cache_size: 10000
```

#### For Balanced (latency + throughput)
**Recommended Models:**
1. TBD - TBDms latency, TBD FPS
2. TBD - TBDms latency, TBD FPS

**Configuration:**
```yaml
batch_size: 8-16
concurrent_requests: 4-8
cache_size: 5000
```

### Model Selection Guide

| Use Case | Recommended Model | Batch Size | Expected Latency | Expected Throughput |
|----------|-------------------|------------|------------------|---------------------|
| Mobile/Edge | MobileNetV4-Hybrid-Large | 1 | TBD ms | TBD FPS |
| Real-time API | TBD | 1-4 | TBD ms | TBD FPS |
| Batch Processing | TBD | 64 | TBD ms/img | TBD img/s |
| High Accuracy | EVA-Giant-Patch14-560 | TBD | TBD ms | TBD FPS |
| Balanced | TBD | TBD | TBD ms | TBD FPS |

### Optimization Tips

1. **For Latency-Critical Applications:**
   - Use smaller models (MobileNetV4, CoAtNet)
   - Batch size = 1
   - Enable model caching
   - Use connection pooling

2. **For Throughput-Critical Applications:**
   - Use batch sizes 32-64
   - Process multiple requests in parallel
   - Use larger models for better accuracy/throughput tradeoff

3. **For Resource-Constrained Environments:**
   - Use MobileNetV4 or EfficientNetV2
   - Batch size = 1-8
   - Limit concurrent requests to 2-4

---

## Appendix: Benchmark Commands

### Running Benchmarks

```bash
# Full model inference benchmarks
cargo bench --bench model_inference_bench_with_report --features torch

# Batch scaling benchmarks
cargo bench --bench batch_concurrent_bench batch_inference_scaling --features torch

# Concurrent scaling benchmarks
cargo bench --bench batch_concurrent_bench concurrent_inference_scaling --features torch

# Analyze results
python3 benches/analyze_benchmarks.py
python3 benches/analyze_scaling.py
python3 benches/generate_inference_table.py
```

### Data Files
- CSV Results: `benches/data/model_inference_benchmark_*.csv`
- JSON Results: `benches/data/model_inference_benchmark_*.json`
- Scaling Analysis: `benches/data/scaling_analysis.json`

---

## Status

**Benchmark Status:** 🔄 IN PROGRESS  
**Estimated Completion:** ~30-60 minutes from start  
**Last Updated:** December 22, 2025 04:38 UTC  

**To update this report with actual results:**
```bash
# Wait for benchmarks to complete, then run:
python3 benches/generate_inference_table.py > MODEL_INFERENCE_RESULTS.md
```

---

*This is a template document. Actual benchmark results will be populated once the benchmark suite completes execution.*

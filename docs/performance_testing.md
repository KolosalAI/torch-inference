# Comprehensive Performance Testing Suite

This enhanced performance testing tool provides detailed analysis of PyTorch inference server performance across multiple dimensions: latency, throughput, scalability, and memory usage.

## 🚀 New Test Types

### 1. Latency Test (`--test-type latency`)
**Measures per-inference latency with cold start vs warm start analysis.**

```bash
python tools/performance_test.py --model resnet50 --test-type latency
```

**What it measures:**
- Cold start times (first inference after model load)
- Warm start times (subsequent inferences)
- Cold start overhead calculation
- 95th percentile latencies for both scenarios

**Use cases:**
- Understanding model initialization cost
- Optimizing for real-time applications
- Deciding on model preloading strategies

### 2. Throughput Test (`--test-type throughput`)
**Tests inference throughput at different batch sizes.**

```bash
python tools/performance_test.py --model bert --test-type throughput
```

**What it measures:**
- Throughput (predictions/second) for batch sizes [1, 2, 4, 8, 16, 32]
- Latency per batch size
- Efficiency scores (throughput per unit latency)
- Optimal batch size identification

**Use cases:**
- Maximizing throughput for batch processing
- Finding the sweet spot between latency and throughput
- Capacity planning for production loads

### 3. Scalability Test (`--test-type scalability`)
**Runs multiple concurrent requests to check performance degradation.**

```bash
python tools/performance_test.py --model mobilenet --test-type scalability
```

**What it measures:**
- Performance at concurrency levels [1, 2, 4, 8, 16, 32]
- Throughput degradation with increased load
- Error rates at different concurrency levels
- Optimal concurrency identification

**Use cases:**
- Understanding system limits
- Load balancing configuration
- Identifying bottlenecks in concurrent processing

### 4. Memory Test (`--test-type memory`)
**Measures peak GPU/CPU RAM usage and detects fragmentation.**

```bash
python tools/performance_test.py --model resnet50 --test-type memory
```

**What it measures:**
- Baseline vs peak memory usage (CPU and GPU)
- Model memory footprint
- Cache growth over time
- GPU memory fragmentation detection
- Memory efficiency (requests per MB)

**Use cases:**
- Resource planning and allocation
- Detecting memory leaks
- Optimizing for memory-constrained environments
- GPU memory fragmentation monitoring

### 5. Comprehensive Test (`--test-type comprehensive`)
**Runs all test types and provides optimization recommendations.**

```bash
python tools/performance_test.py --model bert --test-type comprehensive
```

**What it provides:**
- Complete performance profile
- Optimal configuration recommendations
- Performance bottleneck identification
- Detailed analysis across all dimensions

## 📊 Sample Outputs

### Latency Test Output
```
🥶🔥 LATENCY TEST RESULTS
=======================
Cold Start Times: 5 samples
• Average: 2.345s
• 95th percentile: 2.678s

Warm Start Times: 20 samples
• Average: 0.123s  
• 95th percentile: 0.156s

Cold Start Overhead: +2.222s (1806% slower)
```

### Throughput Test Output
```
📦 THROUGHPUT TEST RESULTS
==========================
Optimal Batch Size: 8
Maximum Throughput: 145.67 pred/s

Batch Size Performance:
   Batch  1:    32.45 pred/s, 0.031s latency
   Batch  2:    58.23 pred/s, 0.034s latency
   Batch  4:    95.67 pred/s, 0.042s latency
⭐ Batch  8:   145.67 pred/s, 0.055s latency
   Batch 16:   142.34 pred/s, 0.112s latency
```

### Comprehensive Analysis Output
```
🎯 COMPREHENSIVE PERFORMANCE ANALYSIS
====================================

📊 BASIC PERFORMANCE SUMMARY:
   • Model: resnet50
   • Throughput: 45.67 req/s, 145.32 pred/s
   • Latency: 0.022s avg, 0.045s p95
   • Success Rate: 100.0%

💡 PERFORMANCE RECOMMENDATIONS:
   • Use batch size 8 for maximum throughput
   • Use 4 concurrent threads for optimal performance
   • Consider model preloading to avoid cold start delays
   • ✅ No GPU memory fragmentation detected
```

## 🛠 Advanced Usage

### Export Results to JSON
```bash
python tools/performance_test.py --model bert --test-type comprehensive --format json --output results.json
```

### Memory Monitoring for Large Models
```bash
python tools/performance_test.py --model roberta-large --test-type memory --verbose
```

### Production Load Testing
```bash
python tools/performance_test.py --model resnet50 --test-type scalability --duration 300 --verbose
```

### Quick Model Comparison
```bash
# Test multiple models
for model in resnet50 bert mobilenet; do
    python tools/performance_test.py --model $model --test-type throughput --format json --output ${model}_throughput.json
done
```

## 📈 Performance Optimization Guide

### Based on Test Results:

#### If Cold Start Overhead > 1s:
- Implement model preloading
- Use model caching strategies
- Consider keeping models warm with periodic requests

#### If Throughput Plateaus at Low Batch Sizes:
- Check GPU memory limits
- Optimize model architecture
- Consider model quantization

#### If Scalability Degrades Rapidly:
- Look for resource contention
- Check I/O bottlenecks
- Optimize thread/process management

#### If Memory Fragmentation Detected:
- Implement periodic model reloading
- Use memory pooling strategies
- Monitor for memory leaks

## 🔧 Dependencies

The enhanced performance testing requires:
- `psutil>=7.0.0` - For CPU memory monitoring
- `GPUtil>=1.4.0` - For GPU memory and utilization tracking
- `numpy>=1.21.0` - For statistical calculations

These are already included in the project's requirements.txt.

## 🚀 Quick Start Demo

Run the demonstration script to see all test types in action:

```bash
python tools/test_performance_suite.py
```

This will run through all test types with a sample model and show you what each test provides.

## 📋 Command Reference

### Test Type Options
- `basic` - Standard performance test (default)
- `latency` - Cold vs warm start analysis
- `throughput` - Batch size optimization
- `scalability` - Concurrency testing
- `memory` - Memory usage analysis
- `comprehensive` - All tests with recommendations

### Key Arguments
- `--test-type {basic,latency,throughput,scalability,memory,comprehensive}` - Type of test to run
- `--model MODEL_NAME` - Model to test (auto-downloads if needed)
- `--duration SECONDS` - Test duration in seconds
- `--concurrency THREADS` - Number of concurrent threads
- `--batch-size SIZE` - Batch size for requests
- `--format {text,json,csv}` - Output format
- `--output FILE` - Save results to file
- `--verbose` - Detailed output

## 🎯 Production Recommendations

1. **Always run comprehensive tests** before production deployment
2. **Use the optimal batch size and concurrency** identified by tests
3. **Monitor memory usage** regularly in production
4. **Set up alerts** for performance degradation
5. **Rerun tests** when changing models or infrastructure

The enhanced performance testing suite gives you the insights needed to optimize your PyTorch inference server for production workloads.

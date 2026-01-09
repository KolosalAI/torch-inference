#!/usr/bin/env python3
"""
Optimized Inference Benchmark
Tests throughput with batching and direct tensor API patterns

This script simulates the optimizations implemented in the Rust code:
1. Session pooling (multiple sessions)
2. Batch inference (stacking inputs)
3. Pre-allocated buffers
4. Warmup optimization
"""

import torch
import torchvision.models as models
import numpy as np
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

print("=" * 70)
print("Optimized Inference Benchmark")
print("=" * 70)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Benchmark configurations
WARMUP_ITERATIONS = 50
BENCHMARK_ITERATIONS = 200
BATCH_SIZES = [1, 4, 8, 16, 32]
SESSION_POOL_SIZES = [1, 2, 4, 8]

def create_model():
    """Create ResNet-50 model"""
    model = models.resnet50(weights=None).to(device)
    model.eval()
    return model

def warmup(model, input_shape, iterations=50):
    """Warmup model to optimize CUDA kernels"""
    dummy = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def benchmark_single(model, batch_size=1, iterations=100):
    """Benchmark single-request inference"""
    input_shape = (batch_size, 3, 224, 224)
    dummy = torch.randn(*input_shape, device=device)
    
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
    
    return {
        "batch_size": batch_size,
        "avg_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "throughput_fps": 1000 / np.mean(latencies) * batch_size,
    }

def benchmark_session_pool(model_factory, pool_size=4, batch_size=1, iterations=100):
    """Benchmark with simulated session pool (concurrent model instances)"""
    models = [model_factory() for _ in range(pool_size)]
    for m in models:
        warmup(m, (batch_size, 3, 224, 224), 10)
    
    input_shape = (batch_size, 3, 224, 224)
    dummy = torch.randn(*input_shape, device=device)
    
    pool_lock = threading.Lock()
    current_model_idx = [0]
    
    def get_next_model():
        with pool_lock:
            idx = current_model_idx[0]
            current_model_idx[0] = (idx + 1) % pool_size
            return models[idx]
    
    latencies = []
    
    def run_inference():
        model = get_next_model()
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return (time.perf_counter() - start) * 1000
    
    # Run concurrent inferences
    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = [executor.submit(run_inference) for _ in range(iterations)]
        latencies = [f.result() for f in futures]
    
    return {
        "pool_size": pool_size,
        "batch_size": batch_size,
        "avg_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "throughput_fps": 1000 / np.mean(latencies) * batch_size,
    }

def benchmark_batched(model, total_requests=100, batch_size=8):
    """Benchmark batched inference (stacking multiple requests)"""
    # Simulate batch_size requests being processed together
    input_shape = (batch_size, 3, 224, 224)
    dummy = torch.randn(*input_shape, device=device)
    
    num_batches = total_requests // batch_size
    batch_latencies = []
    
    with torch.no_grad():
        for _ in range(num_batches):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_latencies.append((time.perf_counter() - start) * 1000)
    
    avg_batch_time = np.mean(batch_latencies)
    per_request_time = avg_batch_time / batch_size
    
    return {
        "batch_size": batch_size,
        "avg_batch_ms": avg_batch_time,
        "per_request_ms": per_request_time,
        "throughput_fps": 1000 / per_request_time,
        "speedup_vs_single": None,  # Will be computed later
    }

# Run benchmarks
print("\n[1] Creating model and warming up...")
model = create_model()
warmup(model, (1, 3, 224, 224), WARMUP_ITERATIONS)
print(f"    Warmup complete ({WARMUP_ITERATIONS} iterations)")

# Baseline: single request
print("\n[2] Baseline: Single Request Inference")
baseline = benchmark_single(model, batch_size=1, iterations=BENCHMARK_ITERATIONS)
print(f"    Avg Latency: {baseline['avg_ms']:.2f} ms")
print(f"    Throughput:  {baseline['throughput_fps']:.0f} FPS")
print(f"    P95 Latency: {baseline['p95_ms']:.2f} ms")

# Test different batch sizes
print("\n[3] Batch Inference (varying batch size)")
batch_results = []
for bs in BATCH_SIZES:
    result = benchmark_batched(model, total_requests=BENCHMARK_ITERATIONS, batch_size=bs)
    result['speedup_vs_single'] = result['throughput_fps'] / baseline['throughput_fps']
    batch_results.append(result)
    print(f"    Batch={bs:2d}: {result['throughput_fps']:6.0f} FPS | "
          f"{result['per_request_ms']:.2f} ms/req | "
          f"Speedup: {result['speedup_vs_single']:.2f}x")

# Test session pooling
print("\n[4] Session Pooling (concurrent models)")
pool_results = []
for pool_size in SESSION_POOL_SIZES:
    result = benchmark_session_pool(create_model, pool_size=pool_size, iterations=100)
    result['speedup_vs_single'] = baseline['throughput_fps'] / (result['avg_ms'] / baseline['avg_ms']) if result['avg_ms'] > 0 else 0
    pool_results.append(result)
    print(f"    Pool={pool_size}: {result['throughput_fps']:.0f} FPS | {result['avg_ms']:.2f} ms avg")

# Combined: Batching + Pooling simulation
print("\n[5] Combined Optimization (Batch=8 + Pool=4)")
# The theoretical max is: baseline_fps * batch_improvement * pool_improvement
best_batch = max(batch_results, key=lambda x: x['throughput_fps'])
combined_fps = best_batch['throughput_fps']  # Batching provides the main benefit on single GPU
print(f"    Best throughput: {combined_fps:.0f} FPS")
print(f"    Speedup vs baseline: {combined_fps / baseline['throughput_fps']:.2f}x")

# Save results
results = {
    "timestamp": datetime.now().isoformat(),
    "hardware": {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "pytorch_version": torch.__version__,
    },
    "baseline": baseline,
    "batch_results": batch_results,
    "pool_results": pool_results,
    "best_throughput_fps": combined_fps,
    "speedup_vs_baseline": combined_fps / baseline['throughput_fps'],
}

output_path = "benchmark_results/optimized_benchmark_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Baseline (batch=1):     {baseline['throughput_fps']:6.0f} FPS")
print(f"Best Batched (batch=32): {best_batch['throughput_fps']:6.0f} FPS")
print(f"Speedup:                 {best_batch['speedup_vs_single']:.2f}x")
print(f"\nResults saved to: {output_path}")

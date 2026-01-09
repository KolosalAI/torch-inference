#!/usr/bin/env python3
"""
Comprehensive Framework Benchmark
Tests all frameworks that can be measured on this hardware:
1. PyTorch CUDA (baseline)
2. PyTorch CPU
3. ONNX Runtime CUDA
4. ONNX Runtime CPU
5. Model load times

This generates REAL data for the research paper.
"""

import torch
import torchvision.models as models
import numpy as np
import time
import json
import os
from datetime import datetime
from pathlib import Path

# Try ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available")

print("=" * 70)
print("Comprehensive Framework Benchmark")
print("=" * 70)
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU Memory: {props.total_memory / 1024**3:.2f} GB")
if ONNX_AVAILABLE:
    print(f"ONNX Runtime Version: {ort.__version__}")
    print(f"ONNX Providers: {ort.get_available_providers()}")
print("=" * 70)

# Configuration
WARMUP_ITERATIONS = 50
BENCHMARK_ITERATIONS = 100
BATCH_SIZES = [1, 8, 16, 32]

results = {
    "timestamp": datetime.now().isoformat(),
    "system": {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None,
        "onnx_available": ONNX_AVAILABLE,
        "onnx_version": ort.__version__ if ONNX_AVAILABLE else None,
        "onnx_providers": ort.get_available_providers() if ONNX_AVAILABLE else None,
    },
    "benchmarks": {}
}

def benchmark_pytorch(model, device, batch_size, iterations, warmup=50):
    """Benchmark PyTorch model"""
    model = model.to(device)
    model.eval()
    
    dummy = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
    
    return {
        "avg_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_fps": float(batch_size * 1000 / np.mean(latencies)),
        "per_request_ms": float(np.mean(latencies) / batch_size),
    }

def benchmark_load_time(model_factory, device, iterations=5):
    """Benchmark model load time"""
    load_times = []
    
    for _ in range(iterations):
        # Clear cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        start = time.perf_counter()
        model = model_factory()
        model = model.to(device)
        model.eval()
        # Run one inference to ensure model is fully loaded
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            _ = model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        load_times.append(time.perf_counter() - start)
        
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return {
        "avg_s": float(np.mean(load_times)),
        "std_s": float(np.std(load_times)),
        "min_s": float(np.min(load_times)),
        "max_s": float(np.max(load_times)),
    }

def export_to_onnx(model, path, input_shape=(1, 3, 224, 224)):
    """Export PyTorch model to ONNX"""
    model.eval()
    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        model, dummy, path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=17,
    )
    return path

def benchmark_onnx(onnx_path, provider, batch_size, iterations, warmup=50):
    """Benchmark ONNX Runtime"""
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(onnx_path, sess_options, providers=[provider])
        
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
        
        # Warmup
        for _ in range(warmup):
            _ = session.run(None, {input_name: dummy})
        
        # Benchmark
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = session.run(None, {input_name: dummy})
            latencies.append((time.perf_counter() - start) * 1000)
        
        return {
            "avg_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "throughput_fps": float(batch_size * 1000 / np.mean(latencies)),
            "per_request_ms": float(np.mean(latencies) / batch_size),
            "provider": provider,
        }
    except Exception as e:
        return {"error": str(e), "provider": provider}

def benchmark_onnx_load_time(onnx_path, provider, iterations=5):
    """Benchmark ONNX model load time"""
    load_times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(onnx_path, sess_options, providers=[provider])
        # Run one inference
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        _ = session.run(None, {input_name: dummy})
        load_times.append(time.perf_counter() - start)
        del session
    
    return {
        "avg_s": float(np.mean(load_times)),
        "std_s": float(np.std(load_times)),
        "min_s": float(np.min(load_times)),
        "max_s": float(np.max(load_times)),
        "provider": provider,
    }

# Create output directory
os.makedirs("benchmark_results/comprehensive", exist_ok=True)

# ============================================================================
# RESNET-50 BENCHMARKS
# ============================================================================
print("\n" + "=" * 70)
print("ResNet-50 Benchmarks")
print("=" * 70)

model_factory = lambda: models.resnet50(weights=None)
results["benchmarks"]["resnet50"] = {}

# 1. PyTorch CUDA
if torch.cuda.is_available():
    print("\n[1] PyTorch CUDA")
    device = torch.device("cuda")
    model = model_factory()
    
    results["benchmarks"]["resnet50"]["pytorch_cuda"] = {}
    for bs in BATCH_SIZES:
        result = benchmark_pytorch(model, device, bs, BENCHMARK_ITERATIONS, WARMUP_ITERATIONS)
        results["benchmarks"]["resnet50"]["pytorch_cuda"][f"batch_{bs}"] = result
        print(f"    Batch={bs:2d}: {result['throughput_fps']:6.0f} FPS | {result['avg_ms']:.2f} ms | P95: {result['p95_ms']:.2f} ms")
    
    # Load time
    print("    Load time...")
    load_result = benchmark_load_time(model_factory, device)
    results["benchmarks"]["resnet50"]["pytorch_cuda"]["load_time"] = load_result
    print(f"    Load: {load_result['avg_s']:.3f}s (±{load_result['std_s']:.3f}s)")
    
    del model
    torch.cuda.empty_cache()

# 2. PyTorch CPU
print("\n[2] PyTorch CPU")
device = torch.device("cpu")
model = model_factory()

results["benchmarks"]["resnet50"]["pytorch_cpu"] = {}
for bs in [1]:  # Only batch=1 for CPU (too slow otherwise)
    result = benchmark_pytorch(model, device, bs, 20, 5)  # Fewer iterations for CPU
    results["benchmarks"]["resnet50"]["pytorch_cpu"][f"batch_{bs}"] = result
    print(f"    Batch={bs:2d}: {result['throughput_fps']:6.0f} FPS | {result['avg_ms']:.2f} ms | P95: {result['p95_ms']:.2f} ms")

# Load time
print("    Load time...")
load_result = benchmark_load_time(model_factory, device, iterations=3)
results["benchmarks"]["resnet50"]["pytorch_cpu"]["load_time"] = load_result
print(f"    Load: {load_result['avg_s']:.3f}s (±{load_result['std_s']:.3f}s)")

del model

# 3. ONNX Runtime
if ONNX_AVAILABLE:
    print("\n[3] ONNX Runtime")
    
    # Export model to ONNX
    onnx_path = "benchmark_results/comprehensive/resnet50.onnx"
    if not os.path.exists(onnx_path):
        print("    Exporting to ONNX...")
        model = model_factory()
        export_to_onnx(model, onnx_path)
        del model
        print(f"    Exported to {onnx_path}")
    
    results["benchmarks"]["resnet50"]["onnx"] = {}
    
    # ONNX CPU
    print("    ONNX CPU:")
    results["benchmarks"]["resnet50"]["onnx"]["cpu"] = {}
    for bs in [1]:
        result = benchmark_onnx(onnx_path, "CPUExecutionProvider", bs, 20, 5)
        if "error" not in result:
            results["benchmarks"]["resnet50"]["onnx"]["cpu"][f"batch_{bs}"] = result
            print(f"      Batch={bs:2d}: {result['throughput_fps']:6.0f} FPS | {result['avg_ms']:.2f} ms")
        else:
            print(f"      Error: {result['error']}")
    
    # ONNX CPU load time
    load_result = benchmark_onnx_load_time(onnx_path, "CPUExecutionProvider", iterations=3)
    results["benchmarks"]["resnet50"]["onnx"]["cpu"]["load_time"] = load_result
    print(f"      Load: {load_result['avg_s']:.3f}s")
    
    # Try ONNX CUDA
    if "CUDAExecutionProvider" in ort.get_available_providers():
        print("    ONNX CUDA:")
        results["benchmarks"]["resnet50"]["onnx"]["cuda"] = {}
        for bs in BATCH_SIZES:
            result = benchmark_onnx(onnx_path, "CUDAExecutionProvider", bs, BENCHMARK_ITERATIONS, WARMUP_ITERATIONS)
            if "error" not in result:
                results["benchmarks"]["resnet50"]["onnx"]["cuda"][f"batch_{bs}"] = result
                print(f"      Batch={bs:2d}: {result['throughput_fps']:6.0f} FPS | {result['avg_ms']:.2f} ms")
            else:
                print(f"      Error: {result['error']}")
        
        # Load time
        load_result = benchmark_onnx_load_time(onnx_path, "CUDAExecutionProvider", iterations=3)
        results["benchmarks"]["resnet50"]["onnx"]["cuda"]["load_time"] = load_result
        print(f"      Load: {load_result['avg_s']:.3f}s")
    else:
        print("    ONNX CUDA: Not available")
        results["benchmarks"]["resnet50"]["onnx"]["cuda"] = {"error": "CUDAExecutionProvider not available"}

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY - ResNet-50 (batch=1)")
print("=" * 70)

summary = []

# PyTorch CUDA
if "pytorch_cuda" in results["benchmarks"]["resnet50"]:
    r = results["benchmarks"]["resnet50"]["pytorch_cuda"]["batch_1"]
    load = results["benchmarks"]["resnet50"]["pytorch_cuda"]["load_time"]["avg_s"]
    summary.append(("PyTorch CUDA", r["avg_ms"], r["std_ms"], r["p95_ms"], r["throughput_fps"], load))

# PyTorch CPU
if "pytorch_cpu" in results["benchmarks"]["resnet50"]:
    r = results["benchmarks"]["resnet50"]["pytorch_cpu"]["batch_1"]
    load = results["benchmarks"]["resnet50"]["pytorch_cpu"]["load_time"]["avg_s"]
    summary.append(("PyTorch CPU", r["avg_ms"], r["std_ms"], r["p95_ms"], r["throughput_fps"], load))

# ONNX CPU
if "onnx" in results["benchmarks"]["resnet50"] and "cpu" in results["benchmarks"]["resnet50"]["onnx"]:
    if "batch_1" in results["benchmarks"]["resnet50"]["onnx"]["cpu"]:
        r = results["benchmarks"]["resnet50"]["onnx"]["cpu"]["batch_1"]
        load = results["benchmarks"]["resnet50"]["onnx"]["cpu"]["load_time"]["avg_s"]
        summary.append(("ONNX Runtime CPU", r["avg_ms"], r["std_ms"], r["p95_ms"], r["throughput_fps"], load))

# ONNX CUDA
if "onnx" in results["benchmarks"]["resnet50"] and "cuda" in results["benchmarks"]["resnet50"]["onnx"]:
    if "batch_1" in results["benchmarks"]["resnet50"]["onnx"]["cuda"]:
        r = results["benchmarks"]["resnet50"]["onnx"]["cuda"]["batch_1"]
        load = results["benchmarks"]["resnet50"]["onnx"]["cuda"]["load_time"]["avg_s"]
        summary.append(("ONNX Runtime CUDA", r["avg_ms"], r["std_ms"], r["p95_ms"], r["throughput_fps"], load))

# Print summary table
print(f"\n{'Framework':<25} {'Avg (ms)':>10} {'±SD':>8} {'P95 (ms)':>10} {'FPS':>8} {'Load (s)':>10}")
print("-" * 75)
for name, avg, std, p95, fps, load in sorted(summary, key=lambda x: x[1]):
    print(f"{name:<25} {avg:>10.2f} {std:>8.2f} {p95:>10.2f} {fps:>8.0f} {load:>10.3f}")

# Batched summary
print("\n" + "=" * 70)
print("SUMMARY - ResNet-50 (Batched - PyTorch CUDA)")
print("=" * 70)
if "pytorch_cuda" in results["benchmarks"]["resnet50"]:
    baseline_fps = results["benchmarks"]["resnet50"]["pytorch_cuda"]["batch_1"]["throughput_fps"]
    print(f"\n{'Batch Size':<12} {'Avg (ms)':>10} {'Per-req (ms)':>12} {'FPS':>10} {'Speedup':>10}")
    print("-" * 60)
    for bs in BATCH_SIZES:
        key = f"batch_{bs}"
        if key in results["benchmarks"]["resnet50"]["pytorch_cuda"]:
            r = results["benchmarks"]["resnet50"]["pytorch_cuda"][key]
            speedup = r["throughput_fps"] / baseline_fps
            print(f"{bs:<12} {r['avg_ms']:>10.2f} {r['per_request_ms']:>12.2f} {r['throughput_fps']:>10.0f} {speedup:>10.2f}x")

# Save results
output_path = "benchmark_results/comprehensive/framework_comparison.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_path}")

# Generate LaTeX table
print("\n" + "=" * 70)
print("LaTeX Table (copy to paper)")
print("=" * 70)
print("""
\\begin{table}[t]
\\centering
\\caption{ResNet-50 Inference Latency Comparison (NVIDIA RTX 3060 Laptop GPU)}
\\label{tab:latency_gpu}
\\scriptsize
\\setlength{\\tabcolsep}{3pt}
\\begin{tabular}{@{}lrrrrr@{}}
\\toprule
\\textbf{Framework} & \\textbf{Avg (ms)} & \\textbf{$\\pm$SD} & \\textbf{P95 (ms)} & \\textbf{FPS} & \\textbf{Load (s)} \\\\
\\midrule""")

for name, avg, std, p95, fps, load in sorted(summary, key=lambda x: x[1]):
    print(f"{name} & {avg:.2f} & {std:.2f} & {p95:.2f} & {fps:.0f} & {load:.2f} \\\\")

print("""\\bottomrule
\\end{tabular}
\\end{table}
""")

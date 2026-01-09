#!/usr/bin/env python3
"""
Cross-Framework Benchmark Suite for Mac (MPS) and CPU
Generates real benchmark data for BENCHMARKS.md and LaTeX paper
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import json
import time
import os
import gc
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available")

OUTPUT_DIR = Path("benchmark_results/framework_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Models to benchmark
MODELS_CONFIG = {
    "ResNet-18": {"fn": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT), "input_size": (1, 3, 224, 224)},
    "ResNet-50": {"fn": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT), "input_size": (1, 3, 224, 224)},
    "ResNet-101": {"fn": lambda: models.resnet101(weights=models.ResNet101_Weights.DEFAULT), "input_size": (1, 3, 224, 224)},
    "MobileNetV3-L": {"fn": lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT), "input_size": (1, 3, 224, 224)},
    "EfficientNet-B0": {"fn": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT), "input_size": (1, 3, 224, 224)},
    "ViT-B/16": {"fn": lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT), "input_size": (1, 3, 224, 224)},
}

BATCH_SIZES = [1, 8, 16, 32]
WARMUP_ITERATIONS = 50
BENCHMARK_ITERATIONS = 100
LOAD_TIME_ITERATIONS = 5


def get_system_info() -> Dict:
    """Get comprehensive system information"""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "cuda_available": torch.cuda.is_available(),
    }
    
    # Get Mac-specific info
    try:
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                              capture_output=True, text=True)
        info["cpu_brand"] = result.stdout.strip()
    except:
        info["cpu_brand"] = platform.processor()
    
    try:
        result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                              capture_output=True, text=True)
        info["memory_gb"] = int(result.stdout.strip()) / (1024**3)
    except:
        info["memory_gb"] = "unknown"
    
    if ONNX_AVAILABLE:
        info["onnx_version"] = ort.__version__
        info["onnx_providers"] = ort.get_available_providers()
    
    return info


def benchmark_pytorch_inference(model, input_tensor, iterations: int, device: str) -> Dict:
    """Run PyTorch inference benchmark and collect statistics"""
    model.eval()
    latencies = []
    
    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            _ = model(input_tensor)
            if device == "mps":
                torch.mps.synchronize()
    
    # Benchmark
    with torch.no_grad():
        for _ in range(iterations):
            if device == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            if device == "mps":
                torch.mps.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    batch_size = input_tensor.shape[0]
    
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
        "batch_size": batch_size,
        "iterations": iterations,
    }


def benchmark_load_time(model_fn, device: str, iterations: int = LOAD_TIME_ITERATIONS) -> Dict:
    """Measure model load time"""
    load_times = []
    
    for _ in range(iterations):
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        
        start = time.perf_counter()
        model = model_fn()
        model = model.to(device)
        model.eval()
        if device == "mps":
            torch.mps.synchronize()
        end = time.perf_counter()
        
        load_times.append(end - start)
        del model
    
    return {
        "avg_s": float(np.mean(load_times)),
        "std_s": float(np.std(load_times)),
        "min_s": float(np.min(load_times)),
        "max_s": float(np.max(load_times)),
    }


def benchmark_onnx_inference(session, input_name: str, input_data: np.ndarray, iterations: int) -> Dict:
    """Run ONNX Runtime inference benchmark"""
    latencies = []
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        _ = session.run(None, {input_name: input_data})
    
    # Benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        _ = session.run(None, {input_name: input_data})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    latencies = np.array(latencies)
    batch_size = input_data.shape[0]
    
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
        "batch_size": batch_size,
        "iterations": iterations,
    }


def export_to_onnx(model, input_shape: Tuple, output_path: str) -> bool:
    """Export PyTorch model to ONNX"""
    try:
        model.eval()
        dummy_input = torch.randn(*input_shape)
        torch.onnx.export(
            model.cpu(),
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        return True
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return False


def benchmark_onnx_load_time(onnx_path: str, providers: List[str], iterations: int = LOAD_TIME_ITERATIONS) -> Dict:
    """Measure ONNX model load time"""
    load_times = []
    
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        session = ort.InferenceSession(onnx_path, providers=providers)
        end = time.perf_counter()
        load_times.append(end - start)
        del session
    
    return {
        "avg_s": float(np.mean(load_times)),
        "std_s": float(np.std(load_times)),
        "min_s": float(np.min(load_times)),
        "max_s": float(np.max(load_times)),
    }


def run_full_benchmark():
    """Run comprehensive benchmark across all models and frameworks"""
    print("=" * 70)
    print("CROSS-FRAMEWORK BENCHMARK SUITE (Mac/MPS)")
    print("=" * 70)
    
    system_info = get_system_info()
    print(f"Timestamp: {system_info['timestamp']}")
    print(f"Platform: {system_info['platform']}")
    print(f"CPU: {system_info.get('cpu_brand', 'unknown')}")
    print(f"Memory: {system_info.get('memory_gb', 'unknown')} GB")
    print(f"PyTorch: {system_info['pytorch_version']}")
    print(f"MPS Available: {system_info['mps_available']}")
    if ONNX_AVAILABLE:
        print(f"ONNX Runtime: {system_info['onnx_version']}")
        print(f"ONNX Providers: {system_info['onnx_providers']}")
    print("=" * 70)
    
    results = {
        "system": system_info,
        "benchmark_config": {
            "warmup_iterations": WARMUP_ITERATIONS,
            "benchmark_iterations": BENCHMARK_ITERATIONS,
            "load_time_iterations": LOAD_TIME_ITERATIONS,
            "batch_sizes": BATCH_SIZES,
        },
        "models": {}
    }
    
    for model_name, config in MODELS_CONFIG.items():
        print(f"\n{'='*70}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*70}")
        
        model_results = {
            "pytorch_mps": {},
            "pytorch_cpu": {},
            "onnx_cpu": {},
        }
        
        # ===== PyTorch MPS =====
        if torch.backends.mps.is_available():
            print(f"\n[PyTorch MPS]")
            try:
                model = config["fn"]().to("mps").eval()
                
                for batch_size in BATCH_SIZES:
                    input_shape = (batch_size,) + config["input_size"][1:]
                    input_tensor = torch.randn(*input_shape).to("mps")
                    
                    result = benchmark_pytorch_inference(model, input_tensor, BENCHMARK_ITERATIONS, "mps")
                    model_results["pytorch_mps"][f"batch_{batch_size}"] = result
                    
                    print(f"  Batch={batch_size:2d}: {result['throughput_fps']:7.1f} FPS | "
                          f"{result['per_request_ms']:.2f} ms/req | "
                          f"P50: {result['p50_ms']:.2f} ms | P95: {result['p95_ms']:.2f} ms | P99: {result['p99_ms']:.2f} ms")
                    
                    del input_tensor
                    torch.mps.empty_cache()
                
                # Load time
                del model
                torch.mps.empty_cache()
                gc.collect()
                load_result = benchmark_load_time(config["fn"], "mps")
                model_results["pytorch_mps"]["load_time"] = load_result
                print(f"  Load: {load_result['avg_s']:.3f}s (±{load_result['std_s']:.3f}s)")
                
            except Exception as e:
                print(f"  Error: {e}")
        
        # ===== PyTorch CPU =====
        print(f"\n[PyTorch CPU]")
        try:
            model = config["fn"]().cpu().eval()
            
            for batch_size in BATCH_SIZES:
                input_shape = (batch_size,) + config["input_size"][1:]
                input_tensor = torch.randn(*input_shape).cpu()
                
                result = benchmark_pytorch_inference(model, input_tensor, BENCHMARK_ITERATIONS, "cpu")
                model_results["pytorch_cpu"][f"batch_{batch_size}"] = result
                
                print(f"  Batch={batch_size:2d}: {result['throughput_fps']:7.1f} FPS | "
                      f"{result['per_request_ms']:.2f} ms/req | "
                      f"P50: {result['p50_ms']:.2f} ms | P95: {result['p95_ms']:.2f} ms | P99: {result['p99_ms']:.2f} ms")
                
                del input_tensor
            
            # Load time
            del model
            gc.collect()
            load_result = benchmark_load_time(config["fn"], "cpu")
            model_results["pytorch_cpu"]["load_time"] = load_result
            print(f"  Load: {load_result['avg_s']:.3f}s (±{load_result['std_s']:.3f}s)")
            
        except Exception as e:
            print(f"  Error: {e}")
        
        # ===== ONNX Runtime CPU =====
        if ONNX_AVAILABLE:
            onnx_dir = OUTPUT_DIR / "onnx_models"
            onnx_dir.mkdir(exist_ok=True)
            onnx_path = str(onnx_dir / f"{model_name.lower().replace('/', '_').replace('-', '_')}.onnx")
            
            # Export model
            print(f"\n[ONNX Export]")
            model = config["fn"]().cpu().eval()
            if export_to_onnx(model, config["input_size"], onnx_path):
                print(f"  Exported to {onnx_path}")
                file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
                print(f"  File size: {file_size_mb:.1f} MB")
                del model
                gc.collect()
                
                # ONNX CPU
                print(f"\n[ONNX Runtime CPU]")
                try:
                    sess_opts = ort.SessionOptions()
                    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    sess_opts.intra_op_num_threads = os.cpu_count()
                    
                    session = ort.InferenceSession(onnx_path, sess_opts, providers=['CPUExecutionProvider'])
                    input_name = session.get_inputs()[0].name
                    
                    for batch_size in BATCH_SIZES:
                        input_shape = (batch_size,) + config["input_size"][1:]
                        input_data = np.random.randn(*input_shape).astype(np.float32)
                        
                        result = benchmark_onnx_inference(session, input_name, input_data, BENCHMARK_ITERATIONS)
                        result["provider"] = "CPUExecutionProvider"
                        model_results["onnx_cpu"][f"batch_{batch_size}"] = result
                        
                        print(f"  Batch={batch_size:2d}: {result['throughput_fps']:7.1f} FPS | "
                              f"{result['per_request_ms']:.2f} ms/req | "
                              f"P50: {result['p50_ms']:.2f} ms | P95: {result['p95_ms']:.2f} ms | P99: {result['p99_ms']:.2f} ms")
                    
                    del session
                    gc.collect()
                    load_result = benchmark_onnx_load_time(onnx_path, ['CPUExecutionProvider'])
                    load_result["provider"] = "CPUExecutionProvider"
                    model_results["onnx_cpu"]["load_time"] = load_result
                    print(f"  Load: {load_result['avg_s']:.3f}s")
                    
                except Exception as e:
                    print(f"  Error: {e}")
            else:
                del model
        
        results["models"][model_name] = model_results
        
        # Cleanup
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"benchmark_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")
    
    # Print summary tables
    print_summary_tables(results)
    generate_markdown_report(results, OUTPUT_DIR / f"benchmark_report_{timestamp}.md")
    
    return results


def print_summary_tables(results: Dict):
    """Print summary tables"""
    print("\n" + "=" * 80)
    print("SUMMARY TABLES")
    print("=" * 80)
    
    # Table: MPS Performance by Model (batch=1)
    print("\n### PyTorch MPS Performance by Model (batch=1)")
    print("-" * 80)
    print(f"{'Model':<18} {'Latency (ms)':<14} {'±SD':<10} {'P50':<10} {'P95':<10} {'P99':<10} {'FPS':<10} {'Load (s)':<10}")
    print("-" * 80)
    
    for model_name, data in results["models"].items():
        if "pytorch_mps" in data and "batch_1" in data["pytorch_mps"]:
            b1 = data["pytorch_mps"]["batch_1"]
            lt = data["pytorch_mps"].get("load_time", {})
            print(f"{model_name:<18} {b1['avg_ms']:<14.2f} {b1['std_ms']:<10.2f} {b1['p50_ms']:<10.2f} "
                  f"{b1['p95_ms']:<10.2f} {b1['p99_ms']:<10.2f} {b1['throughput_fps']:<10.1f} {lt.get('avg_s', 0):<10.3f}")
    
    # Table: CPU Performance by Model (batch=1)
    print("\n### PyTorch CPU Performance by Model (batch=1)")
    print("-" * 80)
    print(f"{'Model':<18} {'Latency (ms)':<14} {'±SD':<10} {'P50':<10} {'P95':<10} {'P99':<10} {'FPS':<10} {'Load (s)':<10}")
    print("-" * 80)
    
    for model_name, data in results["models"].items():
        if "pytorch_cpu" in data and "batch_1" in data["pytorch_cpu"]:
            b1 = data["pytorch_cpu"]["batch_1"]
            lt = data["pytorch_cpu"].get("load_time", {})
            print(f"{model_name:<18} {b1['avg_ms']:<14.2f} {b1['std_ms']:<10.2f} {b1['p50_ms']:<10.2f} "
                  f"{b1['p95_ms']:<10.2f} {b1['p99_ms']:<10.2f} {b1['throughput_fps']:<10.1f} {lt.get('avg_s', 0):<10.3f}")
    
    # Table: ONNX CPU Performance by Model (batch=1)
    print("\n### ONNX Runtime CPU Performance by Model (batch=1)")
    print("-" * 80)
    print(f"{'Model':<18} {'Latency (ms)':<14} {'±SD':<10} {'P50':<10} {'P95':<10} {'P99':<10} {'FPS':<10} {'Load (s)':<10}")
    print("-" * 80)
    
    for model_name, data in results["models"].items():
        if "onnx_cpu" in data and "batch_1" in data["onnx_cpu"]:
            b1 = data["onnx_cpu"]["batch_1"]
            lt = data["onnx_cpu"].get("load_time", {})
            print(f"{model_name:<18} {b1['avg_ms']:<14.2f} {b1['std_ms']:<10.2f} {b1['p50_ms']:<10.2f} "
                  f"{b1['p95_ms']:<10.2f} {b1['p99_ms']:<10.2f} {b1['throughput_fps']:<10.1f} {lt.get('avg_s', 0):<10.3f}")
    
    # Table: MPS vs CPU Speedup
    print("\n### MPS vs CPU Speedup (batch=1)")
    print("-" * 60)
    print(f"{'Model':<18} {'MPS (ms)':<12} {'CPU (ms)':<12} {'Speedup':<12}")
    print("-" * 60)
    
    for model_name, data in results["models"].items():
        if "pytorch_mps" in data and "pytorch_cpu" in data:
            if "batch_1" in data["pytorch_mps"] and "batch_1" in data["pytorch_cpu"]:
                mps = data["pytorch_mps"]["batch_1"]["avg_ms"]
                cpu = data["pytorch_cpu"]["batch_1"]["avg_ms"]
                speedup = cpu / mps
                print(f"{model_name:<18} {mps:<12.2f} {cpu:<12.2f} {speedup:<12.2f}x")
    
    # Table: Batch Scaling (ResNet-50)
    print("\n### Batch Scaling (ResNet-50, MPS)")
    print("-" * 70)
    print(f"{'Batch':<10} {'Total (ms)':<14} {'Per-req (ms)':<14} {'FPS':<12} {'Speedup':<12}")
    print("-" * 70)
    
    if "ResNet-50" in results["models"]:
        resnet_data = results["models"]["ResNet-50"].get("pytorch_mps", {})
        base_fps = resnet_data.get("batch_1", {}).get("throughput_fps", 1)
        
        for batch_size in BATCH_SIZES:
            key = f"batch_{batch_size}"
            if key in resnet_data:
                b = resnet_data[key]
                speedup = b["throughput_fps"] / base_fps if base_fps > 0 else 0
                print(f"{batch_size:<10} {b['avg_ms']:<14.2f} {b['per_request_ms']:<14.2f} "
                      f"{b['throughput_fps']:<12.1f} {speedup:<12.2f}x")
    
    # Table: Framework Comparison (ResNet-50, batch=1)
    print("\n### Framework Comparison (ResNet-50, batch=1)")
    print("-" * 90)
    print(f"{'Framework':<25} {'Latency (ms)':<14} {'±SD':<10} {'P95 (ms)':<12} {'P99 (ms)':<12} {'FPS':<10} {'Load (s)':<10}")
    print("-" * 90)
    
    if "ResNet-50" in results["models"]:
        r50 = results["models"]["ResNet-50"]
        
        frameworks = [
            ("PyTorch MPS", r50.get("pytorch_mps", {})),
            ("PyTorch CPU", r50.get("pytorch_cpu", {})),
            ("ONNX Runtime CPU", r50.get("onnx_cpu", {})),
        ]
        
        for name, data in frameworks:
            if "batch_1" in data:
                b1 = data["batch_1"]
                lt = data.get("load_time", {})
                print(f"{name:<25} {b1['avg_ms']:<14.2f} {b1['std_ms']:<10.2f} {b1['p95_ms']:<12.2f} "
                      f"{b1['p99_ms']:<12.2f} {b1['throughput_fps']:<10.1f} {lt.get('avg_s', 0):<10.3f}")


def generate_markdown_report(results: Dict, output_path: Path):
    """Generate markdown report"""
    with open(output_path, 'w') as f:
        f.write("# Cross-Framework Benchmark Report\n\n")
        f.write(f"**Generated:** {results['system']['timestamp']}\n\n")
        
        f.write("## System Information\n\n")
        f.write("| Property | Value |\n")
        f.write("|----------|-------|\n")
        for key, value in results['system'].items():
            f.write(f"| {key} | {value} |\n")
        
        f.write("\n## Benchmark Configuration\n\n")
        f.write("| Setting | Value |\n")
        f.write("|---------|-------|\n")
        for key, value in results['benchmark_config'].items():
            f.write(f"| {key} | {value} |\n")
        
        # MPS Results
        f.write("\n## PyTorch MPS Performance (batch=1)\n\n")
        f.write("| Model | Latency (ms) | ±SD | P50 | P95 | P99 | FPS | Load (s) |\n")
        f.write("|-------|--------------|-----|-----|-----|-----|-----|----------|\n")
        
        for model_name, data in results["models"].items():
            if "pytorch_mps" in data and "batch_1" in data["pytorch_mps"]:
                b1 = data["pytorch_mps"]["batch_1"]
                lt = data["pytorch_mps"].get("load_time", {})
                f.write(f"| {model_name} | {b1['avg_ms']:.2f} | {b1['std_ms']:.2f} | {b1['p50_ms']:.2f} | "
                       f"{b1['p95_ms']:.2f} | {b1['p99_ms']:.2f} | {b1['throughput_fps']:.1f} | {lt.get('avg_s', 0):.3f} |\n")
        
        # CPU Results
        f.write("\n## PyTorch CPU Performance (batch=1)\n\n")
        f.write("| Model | Latency (ms) | ±SD | P50 | P95 | P99 | FPS | Load (s) |\n")
        f.write("|-------|--------------|-----|-----|-----|-----|-----|----------|\n")
        
        for model_name, data in results["models"].items():
            if "pytorch_cpu" in data and "batch_1" in data["pytorch_cpu"]:
                b1 = data["pytorch_cpu"]["batch_1"]
                lt = data["pytorch_cpu"].get("load_time", {})
                f.write(f"| {model_name} | {b1['avg_ms']:.2f} | {b1['std_ms']:.2f} | {b1['p50_ms']:.2f} | "
                       f"{b1['p95_ms']:.2f} | {b1['p99_ms']:.2f} | {b1['throughput_fps']:.1f} | {lt.get('avg_s', 0):.3f} |\n")
        
        # ONNX Results
        f.write("\n## ONNX Runtime CPU Performance (batch=1)\n\n")
        f.write("| Model | Latency (ms) | ±SD | P50 | P95 | P99 | FPS | Load (s) |\n")
        f.write("|-------|--------------|-----|-----|-----|-----|-----|----------|\n")
        
        for model_name, data in results["models"].items():
            if "onnx_cpu" in data and "batch_1" in data["onnx_cpu"]:
                b1 = data["onnx_cpu"]["batch_1"]
                lt = data["onnx_cpu"].get("load_time", {})
                f.write(f"| {model_name} | {b1['avg_ms']:.2f} | {b1['std_ms']:.2f} | {b1['p50_ms']:.2f} | "
                       f"{b1['p95_ms']:.2f} | {b1['p99_ms']:.2f} | {b1['throughput_fps']:.1f} | {lt.get('avg_s', 0):.3f} |\n")
        
        # MPS vs CPU Speedup
        f.write("\n## MPS vs CPU Speedup\n\n")
        f.write("| Model | MPS (ms) | CPU (ms) | Speedup |\n")
        f.write("|-------|----------|----------|--------|\n")
        
        for model_name, data in results["models"].items():
            if "pytorch_mps" in data and "pytorch_cpu" in data:
                if "batch_1" in data["pytorch_mps"] and "batch_1" in data["pytorch_cpu"]:
                    mps = data["pytorch_mps"]["batch_1"]["avg_ms"]
                    cpu = data["pytorch_cpu"]["batch_1"]["avg_ms"]
                    speedup = cpu / mps
                    f.write(f"| {model_name} | {mps:.2f} | {cpu:.2f} | {speedup:.2f}x |\n")
        
        # Batch Scaling
        f.write("\n## Batch Scaling (ResNet-50, MPS)\n\n")
        f.write("| Batch | Total (ms) | Per-req (ms) | FPS | Speedup |\n")
        f.write("|-------|------------|--------------|-----|--------|\n")
        
        if "ResNet-50" in results["models"]:
            resnet_data = results["models"]["ResNet-50"].get("pytorch_mps", {})
            base_fps = resnet_data.get("batch_1", {}).get("throughput_fps", 1)
            
            for batch_size in BATCH_SIZES:
                key = f"batch_{batch_size}"
                if key in resnet_data:
                    b = resnet_data[key]
                    speedup = b["throughput_fps"] / base_fps if base_fps > 0 else 0
                    f.write(f"| {batch_size} | {b['avg_ms']:.2f} | {b['per_request_ms']:.2f} | "
                           f"{b['throughput_fps']:.1f} | {speedup:.2f}x |\n")
    
    print(f"\nMarkdown report saved to: {output_path}")


if __name__ == "__main__":
    run_full_benchmark()

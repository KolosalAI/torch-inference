#!/usr/bin/env python3
"""
Comprehensive Framework Benchmark Suite
Tests multiple models across PyTorch CUDA/CPU and ONNX Runtime CUDA/CPU
Generates real data for all paper tables
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import json
import time
import os
import gc
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

OUTPUT_DIR = Path("benchmark_results/comprehensive")
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
WARMUP_ITERATIONS = 20
BENCHMARK_ITERATIONS = 100
LOAD_TIME_ITERATIONS = 5


def get_system_info() -> Dict:
    """Get system information"""
    info = {
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if ONNX_AVAILABLE:
        info["onnx_version"] = ort.__version__
        info["onnx_providers"] = ort.get_available_providers()
    return info


def benchmark_inference(model, input_tensor, iterations: int, device: str) -> Dict:
    """Run inference benchmark and collect statistics"""
    model.eval()
    latencies = []
    
    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            _ = model(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
    
    # Benchmark
    with torch.no_grad():
        for _ in range(iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
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
    }


def benchmark_load_time(model_fn, device: str, iterations: int = LOAD_TIME_ITERATIONS) -> Dict:
    """Measure model load time"""
    load_times = []
    
    for _ in range(iterations):
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        start = time.perf_counter()
        model = model_fn()
        model = model.to(device)
        model.eval()
        if device == "cuda":
            torch.cuda.synchronize()
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
    print("COMPREHENSIVE FRAMEWORK BENCHMARK")
    print("=" * 70)
    
    system_info = get_system_info()
    print(f"Timestamp: {system_info['timestamp']}")
    print(f"PyTorch: {system_info['pytorch_version']}")
    print(f"CUDA: {system_info.get('cuda_available', False)}")
    if system_info.get('cuda_available'):
        print(f"GPU: {system_info['gpu_name']}")
    if ONNX_AVAILABLE:
        print(f"ONNX Runtime: {system_info['onnx_version']}")
    print("=" * 70)
    
    results = {
        "system": system_info,
        "models": {}
    }
    
    for model_name, config in MODELS_CONFIG.items():
        print(f"\n{'='*70}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*70}")
        
        model_results = {
            "pytorch_cuda": {},
            "pytorch_cpu": {},
            "onnx_cpu": {},
            "onnx_cuda": {},
        }
        
        # ===== PyTorch CUDA =====
        if torch.cuda.is_available():
            print(f"\n[PyTorch CUDA]")
            try:
                model = config["fn"]().cuda().eval()
                
                for batch_size in BATCH_SIZES:
                    input_shape = (batch_size,) + config["input_size"][1:]
                    input_tensor = torch.randn(*input_shape).cuda()
                    
                    result = benchmark_inference(model, input_tensor, BENCHMARK_ITERATIONS, "cuda")
                    model_results["pytorch_cuda"][f"batch_{batch_size}"] = result
                    
                    print(f"  Batch={batch_size:2d}: {result['throughput_fps']:7.1f} FPS | {result['per_request_ms']:.2f} ms/req | P95: {result['p95_ms']:.2f} ms")
                    
                    del input_tensor
                    torch.cuda.empty_cache()
                
                # Load time
                del model
                torch.cuda.empty_cache()
                load_result = benchmark_load_time(config["fn"], "cuda")
                model_results["pytorch_cuda"]["load_time"] = load_result
                print(f"  Load: {load_result['avg_s']:.3f}s (±{load_result['std_s']:.3f}s)")
                
            except Exception as e:
                print(f"  Error: {e}")
        
        # ===== PyTorch CPU =====
        print(f"\n[PyTorch CPU]")
        try:
            model = config["fn"]().cpu().eval()
            input_tensor = torch.randn(*config["input_size"]).cpu()
            
            result = benchmark_inference(model, input_tensor, BENCHMARK_ITERATIONS, "cpu")
            model_results["pytorch_cpu"]["batch_1"] = result
            print(f"  Batch=1: {result['throughput_fps']:7.1f} FPS | {result['avg_ms']:.2f} ms | P95: {result['p95_ms']:.2f} ms")
            
            # Load time
            del model
            load_result = benchmark_load_time(config["fn"], "cpu")
            model_results["pytorch_cpu"]["load_time"] = load_result
            print(f"  Load: {load_result['avg_s']:.3f}s (±{load_result['std_s']:.3f}s)")
            
        except Exception as e:
            print(f"  Error: {e}")
        
        # ===== ONNX Runtime =====
        if ONNX_AVAILABLE:
            onnx_path = str(OUTPUT_DIR / f"{model_name.lower().replace('/', '_').replace('-', '_')}.onnx")
            
            # Export model
            print(f"\n[ONNX Export]")
            model = config["fn"]().cpu().eval()
            if export_to_onnx(model, config["input_size"], onnx_path):
                print(f"  Exported to {onnx_path}")
                del model
                gc.collect()
                
                # ONNX CPU
                print(f"\n[ONNX Runtime CPU]")
                try:
                    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                    input_name = session.get_inputs()[0].name
                    
                    for batch_size in [1]:  # CPU only batch=1
                        input_shape = (batch_size,) + config["input_size"][1:]
                        input_data = np.random.randn(*input_shape).astype(np.float32)
                        
                        result = benchmark_onnx_inference(session, input_name, input_data, BENCHMARK_ITERATIONS)
                        result["provider"] = "CPUExecutionProvider"
                        model_results["onnx_cpu"][f"batch_{batch_size}"] = result
                        
                        print(f"  Batch={batch_size}: {result['throughput_fps']:7.1f} FPS | {result['avg_ms']:.2f} ms")
                    
                    del session
                    load_result = benchmark_onnx_load_time(onnx_path, ['CPUExecutionProvider'])
                    load_result["provider"] = "CPUExecutionProvider"
                    model_results["onnx_cpu"]["load_time"] = load_result
                    print(f"  Load: {load_result['avg_s']:.3f}s")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                
                # ONNX CUDA
                if torch.cuda.is_available() and 'CUDAExecutionProvider' in ort.get_available_providers():
                    print(f"\n[ONNX Runtime CUDA]")
                    try:
                        session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                        actual_provider = session.get_providers()[0]
                        input_name = session.get_inputs()[0].name
                        
                        for batch_size in [1]:
                            input_shape = (batch_size,) + config["input_size"][1:]
                            input_data = np.random.randn(*input_shape).astype(np.float32)
                            
                            result = benchmark_onnx_inference(session, input_name, input_data, BENCHMARK_ITERATIONS)
                            result["provider"] = actual_provider
                            model_results["onnx_cuda"][f"batch_{batch_size}"] = result
                            
                            print(f"  Batch={batch_size}: {result['throughput_fps']:7.1f} FPS | {result['avg_ms']:.2f} ms (provider: {actual_provider})")
                        
                        del session
                        load_result = benchmark_onnx_load_time(onnx_path, ['CUDAExecutionProvider', 'CPUExecutionProvider'])
                        load_result["provider"] = actual_provider
                        model_results["onnx_cuda"]["load_time"] = load_result
                        print(f"  Load: {load_result['avg_s']:.3f}s")
                        
                    except Exception as e:
                        print(f"  Error: {e}")
            else:
                del model
        
        results["models"][model_name] = model_results
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results
    output_file = OUTPUT_DIR / "all_models_benchmark.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")
    
    # Print summary tables
    print_summary_tables(results)
    
    return results


def print_summary_tables(results: Dict):
    """Print summary tables for the paper"""
    print("\n" + "=" * 70)
    print("SUMMARY TABLES FOR PAPER")
    print("=" * 70)
    
    # Table: GPU Performance by Model (batch=1)
    print("\n### Table: GPU Performance by Model (PyTorch CUDA, batch=1)")
    print("-" * 70)
    print(f"{'Model':<20} {'Latency (ms)':<15} {'P95 (ms)':<12} {'FPS':<10} {'Load (s)':<10}")
    print("-" * 70)
    
    for model_name, data in results["models"].items():
        if "pytorch_cuda" in data and "batch_1" in data["pytorch_cuda"]:
            b1 = data["pytorch_cuda"]["batch_1"]
            lt = data["pytorch_cuda"].get("load_time", {})
            print(f"{model_name:<20} {b1['avg_ms']:<15.2f} {b1['p95_ms']:<12.2f} {b1['throughput_fps']:<10.1f} {lt.get('avg_s', 0):<10.3f}")
    
    # Table: CPU Performance by Model
    print("\n### Table: CPU Performance by Model (PyTorch CPU, batch=1)")
    print("-" * 70)
    print(f"{'Model':<20} {'Latency (ms)':<15} {'P95 (ms)':<12} {'FPS':<10} {'Load (s)':<10}")
    print("-" * 70)
    
    for model_name, data in results["models"].items():
        if "pytorch_cpu" in data and "batch_1" in data["pytorch_cpu"]:
            b1 = data["pytorch_cpu"]["batch_1"]
            lt = data["pytorch_cpu"].get("load_time", {})
            print(f"{model_name:<20} {b1['avg_ms']:<15.2f} {b1['p95_ms']:<12.2f} {b1['throughput_fps']:<10.1f} {lt.get('avg_s', 0):<10.3f}")
    
    # Table: Batch Scaling (ResNet-50)
    print("\n### Table: Batch Scaling (ResNet-50, PyTorch CUDA)")
    print("-" * 70)
    print(f"{'Batch':<10} {'Total (ms)':<12} {'Per-req (ms)':<15} {'FPS':<10} {'Speedup':<10}")
    print("-" * 70)
    
    if "ResNet-50" in results["models"]:
        resnet_data = results["models"]["ResNet-50"]["pytorch_cuda"]
        base_fps = resnet_data.get("batch_1", {}).get("throughput_fps", 1)
        
        for batch_size in BATCH_SIZES:
            key = f"batch_{batch_size}"
            if key in resnet_data:
                b = resnet_data[key]
                speedup = b["throughput_fps"] / base_fps if base_fps > 0 else 0
                print(f"{batch_size:<10} {b['avg_ms']:<12.2f} {b['per_request_ms']:<15.2f} {b['throughput_fps']:<10.1f} {speedup:<10.2f}x")
    
    # Table: Framework Comparison (ResNet-50)
    print("\n### Table: Framework Comparison (ResNet-50, batch=1)")
    print("-" * 70)
    print(f"{'Framework':<25} {'Latency (ms)':<15} {'±SD':<10} {'P95 (ms)':<12} {'FPS':<10} {'Load (s)':<10}")
    print("-" * 70)
    
    if "ResNet-50" in results["models"]:
        r50 = results["models"]["ResNet-50"]
        
        frameworks = [
            ("PyTorch CUDA", r50.get("pytorch_cuda", {})),
            ("PyTorch CPU", r50.get("pytorch_cpu", {})),
            ("ONNX Runtime CPU", r50.get("onnx_cpu", {})),
            ("ONNX Runtime CUDA", r50.get("onnx_cuda", {})),
        ]
        
        for name, data in frameworks:
            if "batch_1" in data:
                b1 = data["batch_1"]
                lt = data.get("load_time", {})
                print(f"{name:<25} {b1['avg_ms']:<15.2f} {b1['std_ms']:<10.2f} {b1['p95_ms']:<12.2f} {b1['throughput_fps']:<10.1f} {lt.get('avg_s', 0):<10.3f}")


if __name__ == "__main__":
    run_full_benchmark()

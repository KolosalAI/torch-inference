#!/usr/bin/env python3
"""
Comprehensive Quantization Benchmark Script

Compares inference performance across different precisions:
1. FP32 (baseline)
2. FP16 (half precision)
3. INT8 (dynamic quantization - PyTorch)
4. ONNX Runtime optimizations

Hardware: NVIDIA RTX 3060 Laptop GPU
"""

import torch
import torchvision.models as models
import numpy as np
import time
import json
import os
import statistics
from datetime import datetime
from pathlib import Path

# Try ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Try torch-tensorrt for FP16/INT8 compilation
try:
    import torch_tensorrt
    TORCH_TRT_AVAILABLE = True
except ImportError:
    TORCH_TRT_AVAILABLE = False

# Try TensorRT directly
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
    TRT_VERSION = trt.__version__
except ImportError:
    TRT_AVAILABLE = False
    TRT_VERSION = None

def print_header():
    """Print benchmark header with system info."""
    print("=" * 80)
    print("COMPREHENSIVE QUANTIZATION BENCHMARK")
    print("FP32 vs FP16 vs INT8 Comparison")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")
    
    print(f"\nONNX Runtime Available: {ONNX_AVAILABLE}")
    if ONNX_AVAILABLE:
        print(f"ONNX Runtime Version: {ort.__version__}")
        print(f"Available EPs: {ort.get_available_providers()}")
    
    print(f"TensorRT Available: {TRT_AVAILABLE}")
    if TRT_AVAILABLE:
        print(f"TensorRT Version: {TRT_VERSION}")
    print(f"Torch-TensorRT Available: {TORCH_TRT_AVAILABLE}")
    print("=" * 80)


def compute_stats(latencies):
    """Compute statistics from latency measurements."""
    n = len(latencies)
    sorted_latencies = sorted(latencies)
    
    return {
        'avg_ms': statistics.mean(latencies),
        'min_ms': min(latencies),
        'max_ms': max(latencies),
        'std_ms': statistics.stdev(latencies) if n > 1 else 0,
        'p50_ms': statistics.median(latencies),
        'p95_ms': sorted_latencies[int(n * 0.95)] if n > 20 else sorted_latencies[-1],
        'p99_ms': sorted_latencies[int(n * 0.99)] if n > 100 else sorted_latencies[-1],
        'fps': 1000 / statistics.mean(latencies),
        'n_samples': n,
    }


def benchmark_pytorch_fp32(model, input_shape, warmup=50, iterations=200):
    """Benchmark PyTorch model in FP32 on CUDA."""
    device = torch.device('cuda:0')
    model = model.to(device).float()
    model.eval()
    
    dummy_input = torch.randn(*input_shape, device=device, dtype=torch.float32)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    
    return compute_stats(latencies)


def benchmark_pytorch_fp16(model, input_shape, warmup=50, iterations=200):
    """Benchmark PyTorch model in FP16 (half precision) on CUDA."""
    device = torch.device('cuda:0')
    model = model.to(device).half()  # Convert to FP16
    model.eval()
    
    dummy_input = torch.randn(*input_shape, device=device, dtype=torch.float16)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    
    return compute_stats(latencies)


def benchmark_pytorch_amp(model, input_shape, warmup=50, iterations=200):
    """Benchmark PyTorch with Automatic Mixed Precision (AMP)."""
    device = torch.device('cuda:0')
    model = model.to(device).float()
    model.eval()
    
    dummy_input = torch.randn(*input_shape, device=device, dtype=torch.float32)
    
    # Warmup with AMP
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for _ in range(warmup):
                _ = model(dummy_input)
    torch.cuda.synchronize()
    
    # Benchmark with AMP
    latencies = []
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for _ in range(iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(dummy_input)
                torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
    
    return compute_stats(latencies)


def benchmark_pytorch_int8_dynamic(model, input_shape, warmup=50, iterations=200):
    """Benchmark PyTorch model with dynamic INT8 quantization (CPU only)."""
    # Dynamic quantization only works on CPU for most operations
    model_cpu = model.cpu().float()
    model_cpu.eval()
    
    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = quantized_model(dummy_input)
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = quantized_model(dummy_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    
    return compute_stats(latencies)


def benchmark_pytorch_compile(model, input_shape, warmup=50, iterations=200):
    """Benchmark PyTorch 2.0 torch.compile() with CUDA."""
    device = torch.device('cuda:0')
    model = model.to(device).float()
    model.eval()
    
    # Compile with torch.compile (PyTorch 2.0+)
    try:
        # Use default mode which is more compatible
        compiled_model = torch.compile(model, mode="default", backend="eager")
    except Exception as e:
        print(f"    torch.compile failed: {e}")
        return None
    
    dummy_input = torch.randn(*input_shape, device=device, dtype=torch.float32)
    
    # Warmup (important for torch.compile)
    with torch.no_grad():
        for _ in range(warmup * 2):  # More warmup for compilation
            _ = compiled_model(dummy_input)
    torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = compiled_model(dummy_input)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    
    return compute_stats(latencies)


def export_onnx(model, model_name, input_shape, output_dir="./onnx_models"):
    """Export PyTorch model to ONNX."""
    if not ONNX_AVAILABLE:
        return None
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    onnx_path = f"{output_dir}/{model_name}.onnx"
    
    if os.path.exists(onnx_path):
        print(f"    Using cached ONNX: {onnx_path}")
        return onnx_path
    
    model_cpu = model.cpu().float()
    model_cpu.eval()
    dummy_input = torch.randn(*input_shape)
    
    try:
        torch.onnx.export(
            model_cpu,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        print(f"    Exported ONNX: {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"    ONNX export failed: {e}")
        return None


def benchmark_onnx_cuda(onnx_path, input_shape, warmup=50, iterations=200):
    """Benchmark ONNX Runtime with CUDA EP."""
    if not ONNX_AVAILABLE or onnx_path is None:
        return None
    
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        print("    CUDA EP not available")
        return None
    
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            onnx_path, 
            sess_options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup):
            _ = session.run(None, {input_name: dummy_input})
        
        # Benchmark
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = session.run(None, {input_name: dummy_input})
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        return compute_stats(latencies)
    except Exception as e:
        print(f"    ONNX CUDA benchmark failed: {e}")
        return None


def benchmark_onnx_tensorrt(onnx_path, input_shape, warmup=50, iterations=200, fp16=False):
    """Benchmark ONNX Runtime with TensorRT EP."""
    if not ONNX_AVAILABLE or onnx_path is None:
        return None
    
    if 'TensorrtExecutionProvider' not in ort.get_available_providers():
        print("    TensorRT EP not available")
        return None
    
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        trt_options = {
            'trt_fp16_enable': fp16,
            'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': './tensorrt_cache',
        }
        
        providers = [
            ('TensorrtExecutionProvider', trt_options),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        print(f"    Building TensorRT engine ({'FP16' if fp16 else 'FP32'})...")
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup (longer for TensorRT engine building)
        for _ in range(warmup * 2):
            _ = session.run(None, {input_name: dummy_input})
        
        # Benchmark
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = session.run(None, {input_name: dummy_input})
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        return compute_stats(latencies)
    except Exception as e:
        print(f"    ONNX TensorRT benchmark failed: {e}")
        return None


def benchmark_torch_tensorrt_fp16(model, input_shape, warmup=50, iterations=200):
    """Benchmark with torch-tensorrt FP16 compilation."""
    if not TORCH_TRT_AVAILABLE:
        return None
    
    try:
        device = torch.device("cuda:0")
        model = model.to(device).float()
        model.eval()
        
        print("    Compiling with torch-tensorrt (FP16)...")
        dummy_input = torch.randn(*input_shape, device=device)
        
        compiled_model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input(
                shape=list(input_shape),
                dtype=torch.float32,
            )],
            enabled_precisions={torch.float16, torch.float32},
            truncate_long_and_double=True,
        )
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = compiled_model(dummy_input)
        torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = compiled_model(dummy_input)
                torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
        
        return compute_stats(latencies)
    except Exception as e:
        print(f"    torch-tensorrt FP16 failed: {e}")
        return None


def format_result(name, stats):
    """Format benchmark result."""
    if stats is None:
        return f"{name:35s}: FAILED/UNAVAILABLE"
    return f"{name:35s}: {stats['fps']:8.1f} FPS | {stats['avg_ms']:6.2f} ms | P95: {stats['p95_ms']:6.2f} ms"


def run_benchmarks(model_name, model_fn, batch_size=1):
    """Run all benchmarks for a model."""
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name} (batch_size={batch_size})")
    print("=" * 80)
    
    input_shape = (batch_size, 3, 224, 224)
    results = {
        'model': model_name,
        'batch_size': batch_size,
        'input_shape': list(input_shape),
        'timestamp': datetime.now().isoformat(),
        'benchmarks': {}
    }
    
    # Load model
    print("Loading model...")
    model = model_fn(weights='DEFAULT')
    model.eval()
    
    # 1. FP32 Baseline (CUDA)
    print("\n[1] PyTorch FP32 (CUDA)")
    fp32_stats = benchmark_pytorch_fp32(model, input_shape)
    results['benchmarks']['pytorch_fp32_cuda'] = fp32_stats
    print(f"    {format_result('PyTorch FP32 CUDA', fp32_stats)}")
    baseline_fps = fp32_stats['fps']
    
    # 2. FP16 (CUDA)
    print("\n[2] PyTorch FP16 (CUDA)")
    fp16_stats = benchmark_pytorch_fp16(model, input_shape)
    results['benchmarks']['pytorch_fp16_cuda'] = fp16_stats
    print(f"    {format_result('PyTorch FP16 CUDA', fp16_stats)}")
    if fp16_stats:
        print(f"    Speedup vs FP32: {fp16_stats['fps']/baseline_fps:.2f}x")
    
    # 3. AMP (Automatic Mixed Precision)
    print("\n[3] PyTorch AMP (CUDA)")
    amp_stats = benchmark_pytorch_amp(model, input_shape)
    results['benchmarks']['pytorch_amp_cuda'] = amp_stats
    print(f"    {format_result('PyTorch AMP CUDA', amp_stats)}")
    if amp_stats:
        print(f"    Speedup vs FP32: {amp_stats['fps']/baseline_fps:.2f}x")
    
    # 4. INT8 Dynamic Quantization (CPU)
    print("\n[4] PyTorch INT8 Dynamic (CPU)")
    try:
        int8_stats = benchmark_pytorch_int8_dynamic(model, input_shape)
        results['benchmarks']['pytorch_int8_cpu'] = int8_stats
        print(f"    {format_result('PyTorch INT8 CPU', int8_stats)}")
    except Exception as e:
        print(f"    INT8 quantization failed: {e}")
        results['benchmarks']['pytorch_int8_cpu'] = None
    
    # 5. ONNX Runtime CUDA
    if ONNX_AVAILABLE:
        print("\n[5] ONNX Runtime CUDA")
        onnx_path = export_onnx(model, model_name, input_shape)
        onnx_cuda_stats = benchmark_onnx_cuda(onnx_path, input_shape)
        results['benchmarks']['onnx_cuda'] = onnx_cuda_stats
        print(f"    {format_result('ONNX Runtime CUDA', onnx_cuda_stats)}")
        if onnx_cuda_stats:
            print(f"    Speedup vs FP32: {onnx_cuda_stats['fps']/baseline_fps:.2f}x")
        
        # 6. ONNX TensorRT FP32
        if TRT_AVAILABLE:
            print("\n[6] ONNX + TensorRT FP32")
            try:
                trt_fp32_stats = benchmark_onnx_tensorrt(onnx_path, input_shape, fp16=False)
                results['benchmarks']['onnx_tensorrt_fp32'] = trt_fp32_stats
                print(f"    {format_result('ONNX TensorRT FP32', trt_fp32_stats)}")
                if trt_fp32_stats:
                    print(f"    Speedup vs FP32: {trt_fp32_stats['fps']/baseline_fps:.2f}x")
            except Exception as e:
                print(f"    TensorRT FP32 failed: {e}")
                results['benchmarks']['onnx_tensorrt_fp32'] = None
            
            # 7. ONNX TensorRT FP16
            print("\n[7] ONNX + TensorRT FP16")
            try:
                trt_fp16_stats = benchmark_onnx_tensorrt(onnx_path, input_shape, fp16=True)
                results['benchmarks']['onnx_tensorrt_fp16'] = trt_fp16_stats
                print(f"    {format_result('ONNX TensorRT FP16', trt_fp16_stats)}")
                if trt_fp16_stats:
                    print(f"    Speedup vs FP32: {trt_fp16_stats['fps']/baseline_fps:.2f}x")
            except Exception as e:
                print(f"    TensorRT FP16 failed: {e}")
                results['benchmarks']['onnx_tensorrt_fp16'] = None
    
    # 8. torch-tensorrt FP16
    if TORCH_TRT_AVAILABLE:
        print("\n[8] torch-tensorrt FP16")
        try:
            torch_trt_stats = benchmark_torch_tensorrt_fp16(model, input_shape)
            results['benchmarks']['torch_tensorrt_fp16'] = torch_trt_stats
            print(f"    {format_result('torch-tensorrt FP16', torch_trt_stats)}")
            if torch_trt_stats:
                print(f"    Speedup vs FP32: {torch_trt_stats['fps']/baseline_fps:.2f}x")
        except Exception as e:
            print(f"    torch-tensorrt failed: {e}")
            results['benchmarks']['torch_tensorrt_fp16'] = None
    
    return results


def main():
    print_header()
    
    # Models to benchmark
    models_to_test = [
        ('ResNet-18', models.resnet18),
        ('ResNet-50', models.resnet50),
        ('ResNet-101', models.resnet101),
        ('MobileNetV3-Large', models.mobilenet_v3_large),
        ('EfficientNet-B0', models.efficientnet_b0),
        ('ViT-B/16', models.vit_b_16),
    ]
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'onnx_available': ONNX_AVAILABLE,
            'tensorrt_available': TRT_AVAILABLE,
            'tensorrt_version': TRT_VERSION,
            'torch_tensorrt_available': TORCH_TRT_AVAILABLE,
        },
        'models': {}
    }
    
    for model_name, model_fn in models_to_test:
        try:
            results = run_benchmarks(model_name, model_fn, batch_size=1)
            all_results['models'][model_name] = results
        except Exception as e:
            print(f"ERROR benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_dir = Path("benchmark_results/quantization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"quantization_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print("=" * 80)
    
    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (FPS)")
    print("=" * 100)
    print(f"{'Model':<20} {'FP32':>10} {'FP16':>10} {'AMP':>10} {'INT8(CPU)':>10} {'ONNX CUDA':>12} {'TRT FP32':>12} {'TRT FP16':>12}")
    print("-" * 100)
    
    for model_name in all_results['models']:
        benchmarks = all_results['models'][model_name]['benchmarks']
        fp32 = benchmarks.get('pytorch_fp32_cuda', {}).get('fps', 0) if benchmarks.get('pytorch_fp32_cuda') else 0
        fp16 = benchmarks.get('pytorch_fp16_cuda', {}).get('fps', 0) if benchmarks.get('pytorch_fp16_cuda') else 0
        amp = benchmarks.get('pytorch_amp_cuda', {}).get('fps', 0) if benchmarks.get('pytorch_amp_cuda') else 0
        int8 = benchmarks.get('pytorch_int8_cpu', {}).get('fps', 0) if benchmarks.get('pytorch_int8_cpu') else 0
        onnx = benchmarks.get('onnx_cuda', {}).get('fps', 0) if benchmarks.get('onnx_cuda') else 0
        trt_fp32 = benchmarks.get('onnx_tensorrt_fp32', {}).get('fps', 0) if benchmarks.get('onnx_tensorrt_fp32') else 0
        trt_fp16 = benchmarks.get('onnx_tensorrt_fp16', {}).get('fps', 0) if benchmarks.get('onnx_tensorrt_fp16') else 0
        
        print(f"{model_name:<20} {fp32:>10.1f} {fp16:>10.1f} {amp:>10.1f} {int8:>10.1f} {onnx:>12.1f} {trt_fp32:>12.1f} {trt_fp16:>12.1f}")


if __name__ == "__main__":
    main()

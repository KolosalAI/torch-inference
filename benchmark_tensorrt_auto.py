"""
TensorRT Auto-Integration Benchmark Script

This script benchmarks PyTorch models with different execution providers:
1. CPU baseline
2. CUDA GPU
3. CUDA with TensorRT (if available)
4. ONNX Runtime with TensorRT EP

Results are saved for comparison with the Rust torch-inference framework.
"""

import torch
import torchvision.models as models
import time
import json
import os
from datetime import datetime
import statistics
from pathlib import Path

# Try to import ONNX-related packages
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX/ONNXRuntime not available. Install with: pip install onnx onnxruntime-gpu")

# Try to import TensorRT
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
    TRT_VERSION = trt.__version__
except ImportError:
    TRT_AVAILABLE = False
    TRT_VERSION = None

# Check torch-tensorrt
try:
    import torch_tensorrt
    TORCH_TRT_AVAILABLE = True
except ImportError:
    TORCH_TRT_AVAILABLE = False


def print_header():
    """Print benchmark header with system info."""
    print("=" * 70)
    print("TensorRT Auto-Integration Benchmark")
    print("=" * 70)
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
    
    print(f"\nTensorRT Available: {TRT_AVAILABLE}")
    if TRT_AVAILABLE:
        print(f"TensorRT Version: {TRT_VERSION}")
    print(f"Torch-TensorRT Available: {TORCH_TRT_AVAILABLE}")
    print("=" * 70)


def export_to_onnx(model, model_name, input_shape, output_dir):
    """Export PyTorch model to ONNX format."""
    if not ONNX_AVAILABLE:
        return None
    
    onnx_path = Path(output_dir) / f"{model_name}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    if onnx_path.exists():
        print(f"  Using cached ONNX model: {onnx_path}")
        return str(onnx_path)
    
    print(f"  Exporting to ONNX: {onnx_path}")
    dummy_input = torch.randn(*input_shape)
    
    try:
        torch.onnx.export(
            model.cpu(),
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        return str(onnx_path)
    except Exception as e:
        print(f"  Failed to export ONNX: {e}")
        return None


def benchmark_pytorch(model, device, input_shape, warmup=20, iterations=100):
    """Benchmark PyTorch model on specified device."""
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
    
    return compute_stats(latencies)


def benchmark_onnx_runtime(onnx_path, execution_provider, input_shape, warmup=20, iterations=100):
    """Benchmark ONNX model with specified execution provider."""
    if not ONNX_AVAILABLE or onnx_path is None:
        return None
    
    try:
        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Configure TensorRT EP if requested
        providers = []
        if execution_provider == 'tensorrt':
            if 'TensorrtExecutionProvider' not in ort.get_available_providers():
                print("  TensorRT EP not available in ONNX Runtime")
                return None
            
            trt_options = {
                'trt_fp16_enable': True,
                'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,  # 2GB
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': './tensorrt_cache',
            }
            providers = [('TensorrtExecutionProvider', trt_options), 'CUDAExecutionProvider', 'CPUExecutionProvider']
        elif execution_provider == 'cuda':
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                print("  CUDA EP not available in ONNX Runtime")
                return None
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Create session
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        
        # Get input name
        input_name = session.get_inputs()[0].name
        dummy_input = torch.randn(*input_shape).numpy()
        
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
        print(f"  ONNX Runtime benchmark failed: {e}")
        return None


def benchmark_torch_tensorrt(model, input_shape, warmup=20, iterations=100):
    """Benchmark with torch-tensorrt compilation."""
    if not TORCH_TRT_AVAILABLE or not torch.cuda.is_available():
        return None
    
    try:
        device = torch.device("cuda:0")
        model = model.to(device)
        model.eval()
        
        # Compile with torch-tensorrt
        print("  Compiling with torch-tensorrt (FP16)...")
        dummy_input = torch.randn(*input_shape).to(device)
        
        compiled_model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input(
                min_shape=[1, 3, 224, 224],
                opt_shape=[1, 3, 224, 224],
                max_shape=[32, 3, 224, 224],
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
        print(f"  torch-tensorrt compilation failed: {e}")
        return None


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
        'p95_ms': sorted_latencies[int(n * 0.95)],
        'p99_ms': sorted_latencies[int(n * 0.99)],
        'fps': 1000 / statistics.mean(latencies),
    }


def format_result(name, stats):
    """Format benchmark result for display."""
    return f"{name:30s}: {stats['fps']:8.2f} FPS | {stats['avg_ms']:7.2f} ms avg | P95: {stats['p95_ms']:7.2f} ms"


def run_model_benchmarks(model_name, model_fn, input_shape, output_dir):
    """Run all benchmarks for a model."""
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    
    results = {
        'model': model_name,
        'input_shape': list(input_shape),
        'timestamp': datetime.now().isoformat(),
        'benchmarks': {}
    }
    
    # Load model
    print("Loading model...")
    model = model_fn(weights=None)
    model.eval()
    
    # 1. CPU Baseline
    print("\n[1] CPU Baseline")
    cpu_stats = benchmark_pytorch(model, torch.device('cpu'), input_shape)
    results['benchmarks']['cpu'] = cpu_stats
    print(f"    {format_result('PyTorch CPU', cpu_stats)}")
    
    # 2. CUDA GPU
    if torch.cuda.is_available():
        print("\n[2] CUDA GPU")
        cuda_stats = benchmark_pytorch(model, torch.device('cuda:0'), input_shape)
        results['benchmarks']['cuda'] = cuda_stats
        print(f"    {format_result('PyTorch CUDA', cuda_stats)}")
        
        speedup = cuda_stats['fps'] / cpu_stats['fps']
        print(f"    Speedup vs CPU: {speedup:.2f}x")
    
    # 3. Export to ONNX and benchmark
    if ONNX_AVAILABLE:
        print("\n[3] ONNX Runtime Benchmarks")
        onnx_path = export_to_onnx(model, model_name, input_shape, output_dir)
        
        if onnx_path:
            # ONNX CPU
            ort_cpu_stats = benchmark_onnx_runtime(onnx_path, 'cpu', input_shape)
            if ort_cpu_stats:
                results['benchmarks']['onnx_cpu'] = ort_cpu_stats
                print(f"    {format_result('ONNX Runtime CPU', ort_cpu_stats)}")
            
            # ONNX CUDA
            if torch.cuda.is_available():
                ort_cuda_stats = benchmark_onnx_runtime(onnx_path, 'cuda', input_shape)
                if ort_cuda_stats:
                    results['benchmarks']['onnx_cuda'] = ort_cuda_stats
                    print(f"    {format_result('ONNX Runtime CUDA', ort_cuda_stats)}")
            
            # ONNX TensorRT
            if TRT_AVAILABLE:
                print("\n[4] ONNX Runtime + TensorRT")
                ort_trt_stats = benchmark_onnx_runtime(onnx_path, 'tensorrt', input_shape, warmup=50, iterations=100)
                if ort_trt_stats:
                    results['benchmarks']['onnx_tensorrt'] = ort_trt_stats
                    print(f"    {format_result('ONNX + TensorRT', ort_trt_stats)}")
                    if 'cuda' in results['benchmarks']:
                        speedup = ort_trt_stats['fps'] / results['benchmarks']['cuda']['fps']
                        print(f"    Speedup vs CUDA: {speedup:.2f}x")
    
    # 4. torch-tensorrt
    if TORCH_TRT_AVAILABLE:
        print("\n[5] torch-tensorrt")
        trt_stats = benchmark_torch_tensorrt(model, input_shape)
        if trt_stats:
            results['benchmarks']['torch_tensorrt'] = trt_stats
            print(f"    {format_result('torch-tensorrt FP16', trt_stats)}")
            if 'cuda' in results['benchmarks']:
                speedup = trt_stats['fps'] / results['benchmarks']['cuda']['fps']
                print(f"    Speedup vs CUDA: {speedup:.2f}x")
    
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def main():
    print_header()
    
    # Output directory
    output_dir = Path("benchmark_results/tensorrt_auto")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Models to benchmark
    models_to_test = [
        ("resnet18", models.resnet18, (1, 3, 224, 224)),
        ("resnet50", models.resnet50, (1, 3, 224, 224)),
        ("mobilenet_v3_large", models.mobilenet_v3_large, (1, 3, 224, 224)),
        ("efficientnet_b0", models.efficientnet_b0, (1, 3, 224, 224)),
        ("vit_b_16", models.vit_b_16, (1, 3, 224, 224)),
    ]
    
    all_results = []
    
    for model_name, model_fn, input_shape in models_to_test:
        try:
            results = run_model_benchmarks(model_name, model_fn, input_shape, output_dir / "onnx_models")
            all_results.append(results)
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"tensorrt_benchmark_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'system': {
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'onnx_runtime_available': ONNX_AVAILABLE,
                'tensorrt_available': TRT_AVAILABLE,
                'tensorrt_version': TRT_VERSION,
                'torch_tensorrt_available': TORCH_TRT_AVAILABLE,
            },
            'results': all_results,
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Benchmark Complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Model':<25} {'CPU (FPS)':<12} {'CUDA (FPS)':<12} {'TensorRT (FPS)':<15} {'Speedup':<10}")
    print("-"*100)
    
    for result in all_results:
        model = result['model']
        cpu_fps = result['benchmarks'].get('cpu', {}).get('fps', 0)
        cuda_fps = result['benchmarks'].get('cuda', {}).get('fps', 0)
        
        # Get best TensorRT result
        trt_fps = 0
        for key in ['onnx_tensorrt', 'torch_tensorrt']:
            if key in result['benchmarks']:
                trt_fps = max(trt_fps, result['benchmarks'][key].get('fps', 0))
        
        speedup = trt_fps / cpu_fps if cpu_fps > 0 and trt_fps > 0 else 0
        
        print(f"{model:<25} {cpu_fps:<12.2f} {cuda_fps:<12.2f} {trt_fps:<15.2f} {speedup:<10.2f}x")
    
    print("="*100)


if __name__ == "__main__":
    main()

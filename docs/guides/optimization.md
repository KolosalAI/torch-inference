# Optimization Guide

Complete guide to optimizing the PyTorch Inference Framework for maximum performance, efficiency, and throughput.

## 📚 Overview

This guide covers comprehensive optimization strategies:
- **Hardware optimization** - GPU, CPU, and memory tuning
- **Model optimization** - TensorRT, ONNX, quantization, compilation
- **Batch optimization** - Dynamic batching and throughput tuning
- **Memory optimization** - Memory pooling and cache management
- **System optimization** - OS-level and deployment optimizations

## 🎯 Optimization Levels

### Quick Start Optimization

```python
# Fastest way to get optimized performance
from framework import TorchInferenceFramework

# Auto-optimized configuration
framework = TorchInferenceFramework(
    auto_optimize=True,
    optimization_level="aggressive"
)

# Load and optimize model automatically
model = framework.load_model(
    "path/to/model.pth",
    optimize=True,
    optimization_level="aggressive"
)

# The framework automatically applies:
# - Best device detection (CUDA > MPS > CPU)
# - Optimal precision (FP16 on GPU, FP32 on CPU)
# - Model compilation (TorchScript/TorchCompile)
# - Memory pooling and CUDA graphs
# - Dynamic batching
```

### Custom Optimization Pipeline

```python
from framework.core.config import OptimizationConfig, DeviceConfig

# Create custom optimization configuration
device_config = DeviceConfig(
    device_type="cuda",
    device_id=0,
    use_fp16=True,
    use_tensorrt=True,
    use_torch_compile=True,
    compile_mode="max-autotune"
)

optimization_config = OptimizationConfig(
    auto_optimize=True,
    benchmark_all=True,
    select_best=True,
    aggressive_optimizations=True,
    fallback_on_error=True,
    
    # Specific optimizers
    optimizers={
        'tensorrt': {
            'enabled': True,
            'precision': 'fp16',
            'max_batch_size': 32,
            'workspace_size_gb': 2
        },
        'onnx': {
            'enabled': True,
            'optimization_level': 'all',
            'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider']
        },
        'quantization': {
            'enabled': True,
            'dynamic': True,
            'calibration_samples': 100
        }
    }
)

framework = TorchInferenceFramework(
    device_config=device_config,
    optimization_config=optimization_config
)
```

## ⚡ Hardware Optimization

### GPU Optimization

```python
# GPU-specific optimizations
def optimize_for_gpu():
    """Comprehensive GPU optimization setup."""
    
    # Check GPU capabilities
    gpu_info = framework.get_gpu_info()
    print(f"GPU: {gpu_info['name']}")
    print(f"Compute capability: {gpu_info['compute_capability']}")
    print(f"Memory: {gpu_info['total_memory_gb']:.1f} GB")
    print(f"CUDA version: {gpu_info['cuda_version']}")
    
    # Configure for specific GPU
    if gpu_info['compute_capability'] >= (8, 0):  # Ampere or newer
        optimization_config = {
            'use_fp16': True,
            'use_tensorrt': True,
            'enable_cuda_graphs': True,
            'use_tensor_cores': True,
            'memory_format': 'channels_last',
            'compile_mode': 'max-autotune'
        }
    elif gpu_info['compute_capability'] >= (7, 0):  # Volta or newer
        optimization_config = {
            'use_fp16': True,
            'use_tensorrt': False,  # May not be optimal on older cards
            'enable_cuda_graphs': True,
            'compile_mode': 'reduce-overhead'
        }
    else:  # Older GPUs
        optimization_config = {
            'use_fp16': False,
            'use_tensorrt': False,
            'enable_cuda_graphs': False,
            'compile_mode': 'default'
        }
    
    # Apply GPU-specific optimizations
    framework.configure_gpu_optimizations(**optimization_config)
    
    # GPU memory optimization
    framework.configure_memory_optimization(
        enable_memory_pooling=True,
        pool_size_gb=min(4, gpu_info['total_memory_gb'] * 0.8),
        enable_garbage_collection=True,
        gc_threshold=0.8
    )
    
    return optimization_config

gpu_config = optimize_for_gpu()
```

### CPU Optimization

```python
# CPU-specific optimizations
def optimize_for_cpu():
    """Comprehensive CPU optimization setup."""
    
    import psutil
    
    # Get CPU information
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores
    logical_count = psutil.cpu_count(logical=True)  # Logical cores
    
    print(f"CPU cores: {cpu_count} physical, {logical_count} logical")
    
    # Configure PyTorch for CPU
    import torch
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(cpu_count)
    
    # Enable CPU optimizations
    cpu_config = {
        'num_threads': cpu_count,
        'use_mkldnn': True,
        'use_openmp': True,
        'enable_jit': True,
        'optimization_level': 'O2'
    }
    
    # Configure ONNX for CPU
    onnx_config = {
        'providers': [
            ('CPUExecutionProvider', {
                'intra_op_num_threads': cpu_count,
                'inter_op_num_threads': cpu_count,
                'enable_cpu_mem_arena': True,
                'enable_memory_pattern': True
            })
        ]
    }
    
    framework.configure_cpu_optimizations(**cpu_config)
    framework.configure_onnx_optimization(**onnx_config)
    
    return cpu_config

cpu_config = optimize_for_cpu()
```

### Mixed Precision Optimization

```python
# Mixed precision for optimal performance
def configure_mixed_precision():
    """Configure mixed precision for different scenarios."""
    
    # Automatic mixed precision based on hardware
    if framework.device.type == 'cuda':
        # NVIDIA GPU - use FP16
        precision_config = {
            'use_fp16': True,
            'use_autocast': True,
            'loss_scaling': True,
            'enable_tensor_cores': True
        }
    elif framework.device.type == 'mps':
        # Apple Silicon - use FP16
        precision_config = {
            'use_fp16': True,
            'use_autocast': True,
            'loss_scaling': False  # Not needed on MPS
        }
    else:
        # CPU - use FP32
        precision_config = {
            'use_fp16': False,
            'use_autocast': False,
            'enable_bf16': True  # BFloat16 on modern CPUs
        }
    
    framework.configure_mixed_precision(**precision_config)
    
    # Model-specific precision tuning
    def optimize_model_precision(model_id):
        """Optimize precision for specific model."""
        
        model_info = framework.get_model_info(model_id)
        
        if model_info['model_type'] == 'transformer':
            # Transformers work well with FP16
            framework.set_model_precision(model_id, 'fp16')
        elif model_info['model_type'] == 'cnn':
            # CNNs are robust to lower precision
            framework.set_model_precision(model_id, 'fp16')
        elif model_info['model_type'] == 'rnn':
            # RNNs may need higher precision
            framework.set_model_precision(model_id, 'fp32')
    
    return precision_config

precision_config = configure_mixed_precision()
```

## 🏗️ Model Optimization

### TensorRT Optimization

```python
# TensorRT optimization for NVIDIA GPUs
def tensorrt_optimization():
    """Comprehensive TensorRT optimization."""
    
    # Check TensorRT availability
    if not framework.is_tensorrt_available():
        print("❌ TensorRT not available")
        return None
    
    # TensorRT configuration
    tensorrt_config = {
        'precision': 'fp16',  # fp32, fp16, int8
        'max_batch_size': 32,
        'workspace_size_gb': 2,
        'enable_dynamic_shapes': True,
        'optimization_level': 5,  # 0-5, higher is more aggressive
        'enable_tactic_timing': True,
        'enable_layer_norm_opt': True,
        'enable_attention_opt': True
    }
    
    # Load model with TensorRT optimization
    model = framework.load_model(
        "path/to/model.pth",
        optimization_type="tensorrt",
        optimization_config=tensorrt_config
    )
    
    print(f"✅ TensorRT model loaded: {model.model_id}")
    
    # Benchmark TensorRT vs standard
    standard_model = framework.load_model(
        "path/to/model.pth",
        optimization_type="none"
    )
    
    # Performance comparison
    test_input = torch.randn(8, 3, 224, 224).cuda()
    
    # Benchmark standard model
    standard_times = []
    for _ in range(100):
        start = time.time()
        _ = framework.predict(standard_model.model_id, test_input)
        standard_times.append(time.time() - start)
    
    # Benchmark TensorRT model
    tensorrt_times = []
    for _ in range(100):
        start = time.time()
        _ = framework.predict(model.model_id, test_input)
        tensorrt_times.append(time.time() - start)
    
    # Print results
    print(f"Standard model: {np.mean(standard_times)*1000:.2f} ± {np.std(standard_times)*1000:.2f} ms")
    print(f"TensorRT model: {np.mean(tensorrt_times)*1000:.2f} ± {np.std(tensorrt_times)*1000:.2f} ms")
    print(f"Speedup: {np.mean(standard_times)/np.mean(tensorrt_times):.2f}x")
    
    return model

# tensorrt_model = tensorrt_optimization()
```

### ONNX Optimization

```python
# ONNX optimization for cross-platform deployment
def onnx_optimization():
    """Comprehensive ONNX optimization."""
    
    # Convert PyTorch model to ONNX
    pytorch_model_path = "path/to/model.pth"
    onnx_model_path = "models/optimized_model.onnx"
    
    # Export with optimization
    framework.export_to_onnx(
        pytorch_model_path=pytorch_model_path,
        onnx_model_path=onnx_model_path,
        input_shape=(1, 3, 224, 224),
        opset_version=17,  # Latest stable
        optimization_level='all',
        enable_onnx_runtime_optimization=True
    )
    
    # ONNX Runtime configuration
    onnx_config = {
        'providers': [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            ('CPUExecutionProvider', {
                'intra_op_num_threads': 8,
                'inter_op_num_threads': 8,
                'enable_cpu_mem_arena': True,
                'enable_memory_pattern': True,
            })
        ],
        'session_options': {
            'enable_profiling': False,
            'enable_cpu_mem_arena': True,
            'enable_mem_pattern': True,
            'graph_optimization_level': 'ORT_ENABLE_ALL'
        }
    }
    
    # Load optimized ONNX model
    onnx_model = framework.load_onnx_model(
        onnx_model_path,
        **onnx_config
    )
    
    # Compare with PyTorch model
    pytorch_model = framework.load_model(pytorch_model_path)
    
    # Benchmark comparison
    test_input = torch.randn(1, 3, 224, 224)
    
    # Test accuracy
    pytorch_output = framework.predict(pytorch_model.model_id, test_input)
    onnx_output = framework.predict(onnx_model.model_id, test_input)
    
    # Calculate accuracy difference
    accuracy_diff = torch.mean(torch.abs(
        pytorch_output['output'] - onnx_output['output']
    )).item()
    
    print(f"Accuracy difference: {accuracy_diff:.6f}")
    print(f"PyTorch time: {pytorch_output['inference_time_ms']:.2f} ms")
    print(f"ONNX time: {onnx_output['inference_time_ms']:.2f} ms")
    print(f"ONNX speedup: {pytorch_output['inference_time_ms']/onnx_output['inference_time_ms']:.2f}x")
    
    return onnx_model

onnx_model = onnx_optimization()
```

### Model Quantization

```python
# Model quantization for reduced memory and faster inference
def quantization_optimization():
    """Comprehensive model quantization."""
    
    original_model_path = "path/to/model.pth"
    
    # Load original model
    original_model = framework.load_model(original_model_path)
    
    # Dynamic quantization (easiest)
    dynamic_quantized = framework.quantize_model(
        model_id=original_model.model_id,
        quantization_type="dynamic",
        backend="fbgemm",  # fbgemm for x86, qnnpack for ARM
        dtype=torch.qint8
    )
    
    # Static quantization (better performance)
    # Requires calibration data
    calibration_data = []
    for i in range(100):  # 100 calibration samples
        calibration_data.append(torch.randn(1, 3, 224, 224))
    
    static_quantized = framework.quantize_model(
        model_id=original_model.model_id,
        quantization_type="static",
        calibration_data=calibration_data,
        backend="fbgemm",
        dtype=torch.qint8
    )
    
    # QAT (Quantization Aware Training) - best quality
    # Note: Requires retraining
    qat_quantized = framework.quantize_model(
        model_id=original_model.model_id,
        quantization_type="qat",
        training_data=calibration_data,  # Use training data here
        epochs=5,
        backend="fbgemm"
    )
    
    # Benchmark all quantization methods
    test_input = torch.randn(1, 3, 224, 224)
    
    models_to_test = [
        ("Original", original_model.model_id),
        ("Dynamic Quantized", dynamic_quantized.model_id),
        ("Static Quantized", static_quantized.model_id),
        ("QAT Quantized", qat_quantized.model_id)
    ]
    
    results = {}
    
    for name, model_id in models_to_test:
        # Performance test
        times = []
        for _ in range(50):
            start = time.time()
            result = framework.predict(model_id, test_input)
            times.append(time.time() - start)
        
        # Memory usage
        memory_usage = framework.get_model_memory_usage(model_id)
        
        # Model size
        model_size = framework.get_model_size(model_id)
        
        results[name] = {
            'avg_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'memory_mb': memory_usage,
            'model_size_mb': model_size,
            'output': result['output']
        }
    
    # Print results
    print("📊 Quantization Comparison:")
    print(f"{'Method':<20} {'Time (ms)':<12} {'Memory (MB)':<12} {'Size (MB)':<12} {'Accuracy':<10}")
    print("-" * 75)
    
    original_output = results["Original"]['output']
    
    for name, data in results.items():
        if name != "Original":
            # Calculate accuracy loss
            accuracy = 1.0 - torch.mean(torch.abs(
                original_output - data['output']
            )).item()
        else:
            accuracy = 1.0
        
        print(f"{name:<20} {data['avg_time_ms']:<12.2f} {data['memory_mb']:<12.1f} "
              f"{data['model_size_mb']:<12.1f} {accuracy:<10.4f}")
    
    return {
        'dynamic': dynamic_quantized,
        'static': static_quantized,
        'qat': qat_quantized,
        'results': results
    }

quantization_results = quantization_optimization()
```

### PyTorch Compilation

```python
# PyTorch 2.0 compilation optimization
def torch_compile_optimization():
    """Optimize using PyTorch 2.0 torch.compile."""
    
    # Load model for compilation
    model_path = "path/to/model.pth"
    
    # Different compilation modes
    compile_modes = {
        'default': 'default',
        'reduce-overhead': 'reduce-overhead',
        'max-autotune': 'max-autotune'
    }
    
    compiled_models = {}
    
    for mode_name, mode in compile_modes.items():
        print(f"Compiling with mode: {mode_name}")
        
        compiled_model = framework.load_model(
            model_path,
            optimization_type="torch_compile",
            compile_mode=mode,
            dynamic=True,  # Enable dynamic shapes
            fullgraph=False,  # Allow graph breaks
            backend="inductor"  # Default backend
        )
        
        compiled_models[mode_name] = compiled_model
    
    # Benchmark compilation modes
    test_inputs = [
        torch.randn(1, 3, 224, 224),
        torch.randn(4, 3, 224, 224),
        torch.randn(8, 3, 224, 224),
        torch.randn(16, 3, 224, 224)
    ]
    
    results = {}
    
    for mode_name, model in compiled_models.items():
        mode_results = {}
        
        for batch_size in [1, 4, 8, 16]:
            test_input = test_inputs[batch_size//4 if batch_size > 1 else 0]
            
            # Warmup
            for _ in range(10):
                _ = framework.predict(model.model_id, test_input)
            
            # Benchmark
            times = []
            for _ in range(50):
                start = time.time()
                _ = framework.predict(model.model_id, test_input)
                times.append(time.time() - start)
            
            mode_results[f'batch_{batch_size}'] = {
                'avg_time_ms': np.mean(times) * 1000,
                'throughput_fps': batch_size / np.mean(times)
            }
        
        results[mode_name] = mode_results
    
    # Print results
    print("\n📊 Torch Compile Comparison:")
    print(f"{'Mode':<15} {'Batch 1':<12} {'Batch 4':<12} {'Batch 8':<12} {'Batch 16':<12}")
    print("-" * 70)
    
    for mode_name, mode_results in results.items():
        times = [mode_results[f'batch_{bs}']['avg_time_ms'] for bs in [1, 4, 8, 16]]
        print(f"{mode_name:<15} {times[0]:<12.2f} {times[1]:<12.2f} {times[2]:<12.2f} {times[3]:<12.2f}")
    
    return compiled_models, results

compiled_models, compile_results = torch_compile_optimization()
```

## 🎯 Batch Optimization

### Dynamic Batching

```python
# Dynamic batching for optimal throughput
def dynamic_batching_optimization():
    """Configure and optimize dynamic batching."""
    
    # Configure dynamic batching
    batching_config = {
        'enable_dynamic_batching': True,
        'max_batch_size': 32,
        'batch_timeout_ms': 50,
        'enable_inflight_batching': True,
        'queue_size': 100,
        'priority_queue': True
    }
    
    framework.configure_dynamic_batching(**batching_config)
    
    # Test different batch configurations
    batch_configs = [
        {'max_batch_size': 8, 'timeout_ms': 10},   # Low latency
        {'max_batch_size': 16, 'timeout_ms': 25},  # Balanced
        {'max_batch_size': 32, 'timeout_ms': 50},  # High throughput
        {'max_batch_size': 64, 'timeout_ms': 100}  # Maximum throughput
    ]
    
    model_id = "test_model"
    results = {}
    
    for i, config in enumerate(batch_configs):
        config_name = f"config_{i+1}"
        print(f"Testing {config_name}: {config}")
        
        # Apply configuration
        framework.update_batching_config(config)
        
        # Simulate concurrent requests
        import asyncio
        import random
        
        async def simulate_requests():
            tasks = []
            request_times = []
            
            # Generate 100 concurrent requests
            for j in range(100):
                # Random delay to simulate real traffic
                await asyncio.sleep(random.uniform(0.001, 0.01))
                
                start_time = time.time()
                test_input = torch.randn(1, 3, 224, 224)
                
                task = framework.predict_async(model_id, test_input)
                tasks.append((task, start_time))
            
            # Wait for all requests
            results = []
            for task, start_time in tasks:
                result = await task
                end_time = time.time()
                latency = (end_time - start_time) * 1000
                results.append(latency)
            
            return results
        
        # Run simulation
        latencies = asyncio.run(simulate_requests())
        
        results[config_name] = {
            'config': config,
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_rps': 100 / (max(latencies) / 1000)
        }
    
    # Print results
    print("\n📊 Dynamic Batching Optimization:")
    print(f"{'Config':<10} {'Avg Latency':<12} {'P95 Latency':<12} {'P99 Latency':<12} {'Throughput':<12}")
    print("-" * 70)
    
    for config_name, data in results.items():
        print(f"{config_name:<10} {data['avg_latency_ms']:<12.2f} "
              f"{data['p95_latency_ms']:<12.2f} {data['p99_latency_ms']:<12.2f} "
              f"{data['throughput_rps']:<12.1f}")
    
    return results

batching_results = dynamic_batching_optimization()
```

### Batch Size Optimization

```python
# Find optimal batch size for your model
def find_optimal_batch_size():
    """Find the optimal batch size for maximum throughput."""
    
    model_id = "test_model"
    batch_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        try:
            # Create test input
            test_input = torch.randn(batch_size, 3, 224, 224)
            
            # Move to appropriate device
            if framework.device.type == 'cuda':
                test_input = test_input.cuda()
            
            # Warmup
            for _ in range(10):
                _ = framework.predict(model_id, test_input)
            
            # Clear cache
            if framework.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Memory before inference
            if framework.device.type == 'cuda':
                memory_before = torch.cuda.memory_allocated()
            
            # Benchmark
            times = []
            for _ in range(20):
                start = time.time()
                result = framework.predict(model_id, test_input)
                times.append(time.time() - start)
            
            # Memory after inference
            if framework.device.type == 'cuda':
                memory_after = torch.cuda.memory_allocated()
                memory_used = (memory_after - memory_before) / 1024 / 1024  # MB
            else:
                memory_used = 0
            
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            
            results[batch_size] = {
                'avg_time_s': avg_time,
                'throughput_fps': throughput,
                'memory_mb': memory_used,
                'efficiency': throughput / memory_used if memory_used > 0 else throughput
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ❌ OOM at batch size {batch_size}")
                break
            else:
                print(f"  ❌ Error: {e}")
                continue
    
    # Find optimal batch size
    if results:
        # Best throughput
        best_throughput = max(results.items(), key=lambda x: x[1]['throughput_fps'])
        
        # Best efficiency (throughput/memory)
        best_efficiency = max(results.items(), key=lambda x: x[1]['efficiency'])
        
        print("\n📊 Batch Size Optimization Results:")
        print(f"{'Batch Size':<12} {'Time (s)':<10} {'Throughput':<12} {'Memory (MB)':<12} {'Efficiency':<12}")
        print("-" * 70)
        
        for batch_size, data in results.items():
            marker = ""
            if batch_size == best_throughput[0]:
                marker += " (Best Throughput)"
            if batch_size == best_efficiency[0]:
                marker += " (Best Efficiency)"
                
            print(f"{batch_size:<12} {data['avg_time_s']:<10.4f} "
                  f"{data['throughput_fps']:<12.1f} {data['memory_mb']:<12.1f} "
                  f"{data['efficiency']:<12.1f}{marker}")
        
        print(f"\n🎯 Recommendations:")
        print(f"  Maximum throughput: batch size {best_throughput[0]} "
              f"({best_throughput[1]['throughput_fps']:.1f} FPS)")
        print(f"  Best efficiency: batch size {best_efficiency[0]} "
              f"({best_efficiency[1]['efficiency']:.1f} FPS/MB)")
    
    return results

optimal_batch_results = find_optimal_batch_size()
```

## 💾 Memory Optimization

### Memory Pooling

```python
# Memory pooling for reduced allocation overhead
def memory_pooling_optimization():
    """Configure and optimize memory pooling."""
    
    # Configure memory pooling
    memory_config = {
        'enable_memory_pooling': True,
        'pool_size_gb': 4,
        'cleanup_threshold': 0.8,
        'auto_gc': True,
        'gc_frequency': 100,  # Run GC every 100 inferences
        'enable_cudnn_benchmark': True,
        'use_memory_mapping': True
    }
    
    framework.configure_memory_optimization(**memory_config)
    
    # Test memory usage patterns
    def test_memory_pattern(name, test_func):
        """Test memory usage pattern."""
        print(f"\n🧪 Testing {name}...")
        
        # Clear memory
        if framework.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Initial memory
        initial_memory = framework.get_memory_usage()
        
        # Run test
        start_time = time.time()
        test_func()
        end_time = time.time()
        
        # Final memory
        final_memory = framework.get_memory_usage()
        
        # Peak memory
        if framework.device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            peak_memory = final_memory['used_mb']
        
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Initial memory: {initial_memory['used_mb']:.1f} MB")
        print(f"  Final memory: {final_memory['used_mb']:.1f} MB")
        print(f"  Peak memory: {peak_memory:.1f} MB")
        print(f"  Memory growth: {final_memory['used_mb'] - initial_memory['used_mb']:.1f} MB")
        
        return {
            'time': end_time - start_time,
            'initial_memory': initial_memory['used_mb'],
            'final_memory': final_memory['used_mb'],
            'peak_memory': peak_memory,
            'memory_growth': final_memory['used_mb'] - initial_memory['used_mb']
        }
    
    # Test scenarios
    model_id = "test_model"
    
    # Scenario 1: Sequential inference
    def sequential_inference():
        for i in range(100):
            test_input = torch.randn(4, 3, 224, 224)
            _ = framework.predict(model_id, test_input)
    
    # Scenario 2: Variable batch sizes
    def variable_batches():
        batch_sizes = [1, 4, 8, 16] * 25
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, 3, 224, 224)
            _ = framework.predict(model_id, test_input)
    
    # Scenario 3: Large allocations
    def large_allocations():
        for i in range(20):
            test_input = torch.randn(32, 3, 224, 224)
            _ = framework.predict(model_id, test_input)
    
    # Run tests
    results = {
        'sequential': test_memory_pattern("Sequential Inference", sequential_inference),
        'variable': test_memory_pattern("Variable Batches", variable_batches),
        'large': test_memory_pattern("Large Allocations", large_allocations)
    }
    
    # Memory pool statistics
    pool_stats = framework.get_memory_pool_stats()
    print(f"\n📊 Memory Pool Statistics:")
    print(f"  Pool size: {pool_stats['pool_size_gb']:.1f} GB")
    print(f"  Pool usage: {pool_stats['usage_percent']:.1f}%")
    print(f"  Cache hits: {pool_stats['cache_hit_rate']:.3f}")
    print(f"  Allocations: {pool_stats['total_allocations']}")
    print(f"  Deallocations: {pool_stats['total_deallocations']}")
    
    return results, pool_stats

memory_results, pool_stats = memory_pooling_optimization()
```

### CUDA Graphs

```python
# CUDA Graphs for ultra-low latency
def cuda_graphs_optimization():
    """Optimize using CUDA Graphs for consistent workloads."""
    
    if framework.device.type != 'cuda':
        print("❌ CUDA Graphs only available on CUDA devices")
        return None
    
    model_id = "test_model"
    
    # Configure CUDA Graphs
    graph_config = {
        'enable_cuda_graphs': True,
        'capture_warmup_iterations': 10,
        'enable_memory_pooling': True,
        'static_shapes_only': True  # Required for CUDA Graphs
    }
    
    framework.configure_cuda_graphs(**graph_config)
    
    # Create static input for graph capture
    static_input = torch.randn(8, 3, 224, 224, device='cuda')
    
    # Capture CUDA Graph
    print("📸 Capturing CUDA Graph...")
    graph_model = framework.create_cuda_graph(
        model_id=model_id,
        static_input=static_input,
        warmup_iterations=20
    )
    
    print(f"✅ CUDA Graph captured for model {graph_model.model_id}")
    
    # Compare performance
    test_input = torch.randn(8, 3, 224, 224, device='cuda')
    
    # Regular inference
    regular_times = []
    for _ in range(100):
        start = time.time()
        _ = framework.predict(model_id, test_input)
        regular_times.append(time.time() - start)
    
    # CUDA Graph inference
    graph_times = []
    for _ in range(100):
        start = time.time()
        _ = framework.predict(graph_model.model_id, test_input)
        graph_times.append(time.time() - start)
    
    # Results
    regular_avg = np.mean(regular_times) * 1000
    regular_std = np.std(regular_times) * 1000
    graph_avg = np.mean(graph_times) * 1000
    graph_std = np.std(graph_times) * 1000
    
    print(f"\n📊 CUDA Graphs Performance:")
    print(f"Regular inference: {regular_avg:.2f} ± {regular_std:.2f} ms")
    print(f"CUDA Graph inference: {graph_avg:.2f} ± {graph_std:.2f} ms")
    print(f"Speedup: {regular_avg/graph_avg:.2f}x")
    print(f"Latency reduction: {((regular_avg - graph_avg)/regular_avg)*100:.1f}%")
    print(f"Std dev reduction: {((regular_std - graph_std)/regular_std)*100:.1f}%")
    
    return {
        'regular_avg_ms': regular_avg,
        'regular_std_ms': regular_std,
        'graph_avg_ms': graph_avg,
        'graph_std_ms': graph_std,
        'speedup': regular_avg/graph_avg,
        'graph_model': graph_model
    }

# cuda_graph_results = cuda_graphs_optimization()
```

## 🌊 Streaming and Pipeline Optimization

### Streaming Inference

```python
# Streaming inference for real-time applications
def streaming_optimization():
    """Optimize for streaming inference workloads."""
    
    # Configure streaming
    streaming_config = {
        'enable_streaming': True,
        'stream_buffer_size': 1024,
        'prefetch_batches': 2,
        'enable_pipelining': True,
        'pipeline_stages': 3,
        'async_execution': True
    }
    
    framework.configure_streaming(**streaming_config)
    
    # Create streaming pipeline
    streaming_pipeline = framework.create_streaming_pipeline(
        model_id="test_model",
        input_queue_size=100,
        output_queue_size=100,
        worker_threads=4
    )
    
    # Test streaming performance
    import asyncio
    import queue
    
    async def streaming_test():
        """Test streaming inference performance."""
        
        input_queue = asyncio.Queue(maxsize=100)
        output_queue = asyncio.Queue(maxsize=100)
        
        # Producer: Generate inputs
        async def producer():
            for i in range(1000):
                test_input = torch.randn(1, 3, 224, 224)
                await input_queue.put((i, test_input))
                await asyncio.sleep(0.001)  # Simulate real-time input
        
        # Consumer: Process outputs
        async def consumer():
            results = []
            for i in range(1000):
                request_id, result, timestamp = await output_queue.get()
                latency = time.time() - timestamp
                results.append(latency)
            return results
        
        # Streaming processor
        async def processor():
            while True:
                try:
                    request_id, input_data = await asyncio.wait_for(
                        input_queue.get(), timeout=1.0
                    )
                    
                    timestamp = time.time()
                    result = await framework.predict_async("test_model", input_data)
                    
                    await output_queue.put((request_id, result, timestamp))
                    
                except asyncio.TimeoutError:
                    break
        
        # Run all tasks
        producer_task = asyncio.create_task(producer())
        processor_task = asyncio.create_task(processor())
        consumer_task = asyncio.create_task(consumer())
        
        await producer_task
        await processor_task
        latencies = await consumer_task
        
        return latencies
    
    # Run streaming test
    latencies = asyncio.run(streaming_test())
    
    # Analyze results
    avg_latency = np.mean(latencies) * 1000
    p95_latency = np.percentile(latencies, 95) * 1000
    p99_latency = np.percentile(latencies, 99) * 1000
    throughput = 1000 / (max(latencies))
    
    print(f"📊 Streaming Performance:")
    print(f"  Average latency: {avg_latency:.2f} ms")
    print(f"  P95 latency: {p95_latency:.2f} ms")
    print(f"  P99 latency: {p99_latency:.2f} ms")
    print(f"  Throughput: {throughput:.1f} RPS")
    
    return {
        'avg_latency_ms': avg_latency,
        'p95_latency_ms': p95_latency,
        'p99_latency_ms': p99_latency,
        'throughput_rps': throughput,
        'pipeline': streaming_pipeline
    }

streaming_results = streaming_optimization()
```

## 🔧 System-Level Optimization

### CPU Affinity and Threading

```python
# System-level optimizations
def system_optimization():
    """Apply system-level optimizations."""
    
    import os
    import psutil
    
    # CPU affinity optimization
    def optimize_cpu_affinity():
        """Optimize CPU affinity for better performance."""
        
        # Get CPU information
        cpu_count = psutil.cpu_count(logical=False)
        logical_count = psutil.cpu_count(logical=True)
        
        # Set CPU affinity to physical cores only
        if cpu_count < logical_count:  # Hyperthreading enabled
            physical_cores = list(range(0, logical_count, 2))
            os.sched_setaffinity(0, physical_cores)
            print(f"✅ CPU affinity set to physical cores: {physical_cores}")
        
        # Optimize thread scheduling
        torch.set_num_threads(cpu_count)
        torch.set_num_interop_threads(cpu_count)
        
        # Set thread priorities (Linux only)
        if hasattr(os, 'nice'):
            os.nice(-10)  # Higher priority
            print("✅ Process priority increased")
    
    # Memory optimization
    def optimize_memory():
        """Optimize memory settings."""
        
        # Disable swap if possible (requires sudo)
        # os.system("sudo swapoff -a")
        
        # Set memory allocation strategies
        os.environ['MALLOC_ARENA_MAX'] = '4'
        os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'
        
        # PyTorch memory settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        if framework.device.type == 'cuda':
            # CUDA memory settings
            torch.cuda.empty_cache()
            
            # Set memory fraction
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_fraction = 0.9  # Use 90% of GPU memory
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            
            print(f"✅ CUDA memory fraction set to {memory_fraction}")
    
    # I/O optimization
    def optimize_io():
        """Optimize I/O performance."""
        
        # Set I/O scheduling (Linux only)
        try:
            with open('/proc/sys/vm/swappiness', 'w') as f:
                f.write('10')  # Reduce swappiness
            print("✅ Swappiness optimized")
        except:
            pass
        
        # Optimize file system cache
        try:
            with open('/proc/sys/vm/vfs_cache_pressure', 'w') as f:
                f.write('50')  # Keep file system cache longer
            print("✅ VFS cache pressure optimized")
        except:
            pass
    
    # Network optimization (for distributed inference)
    def optimize_network():
        """Optimize network settings."""
        
        # TCP optimization
        network_settings = {
            'net.core.rmem_max': '134217728',
            'net.core.wmem_max': '134217728',
            'net.ipv4.tcp_rmem': '4096 65536 134217728',
            'net.ipv4.tcp_wmem': '4096 65536 134217728',
            'net.core.netdev_max_backlog': '5000'
        }
        
        for setting, value in network_settings.items():
            try:
                os.system(f"sudo sysctl -w {setting}={value}")
            except:
                pass
        
        print("✅ Network settings optimized")
    
    # Apply all optimizations
    try:
        optimize_cpu_affinity()
        optimize_memory()
        optimize_io()
        optimize_network()
        
        print("✅ System optimizations applied successfully")
        
    except Exception as e:
        print(f"⚠️  Some optimizations failed: {e}")
        print("Note: Some optimizations require root privileges")

system_optimization()
```

## 📊 Comprehensive Benchmark

```python
# Complete optimization benchmark
def comprehensive_benchmark():
    """Run comprehensive benchmark across all optimizations."""
    
    # Test configurations
    configurations = {
        'baseline': {
            'optimization_level': 'none',
            'use_fp16': False,
            'use_tensorrt': False,
            'use_torch_compile': False,
            'enable_dynamic_batching': False,
            'enable_memory_pooling': False
        },
        'basic': {
            'optimization_level': 'basic',
            'use_fp16': True,
            'use_tensorrt': False,
            'use_torch_compile': True,
            'compile_mode': 'default',
            'enable_dynamic_batching': True,
            'enable_memory_pooling': True
        },
        'aggressive': {
            'optimization_level': 'aggressive',
            'use_fp16': True,
            'use_tensorrt': True,
            'use_torch_compile': True,
            'compile_mode': 'max-autotune',
            'enable_dynamic_batching': True,
            'enable_memory_pooling': True,
            'enable_cuda_graphs': True
        }
    }
    
    # Test scenarios
    test_scenarios = [
        {'name': 'Single Inference', 'batch_size': 1, 'iterations': 100},
        {'name': 'Small Batch', 'batch_size': 4, 'iterations': 50},
        {'name': 'Medium Batch', 'batch_size': 16, 'iterations': 25},
        {'name': 'Large Batch', 'batch_size': 32, 'iterations': 10}
    ]
    
    results = {}
    
    for config_name, config in configurations.items():
        print(f"\n🧪 Testing configuration: {config_name}")
        
        # Apply configuration
        framework.reset_optimizations()
        framework.apply_optimization_config(config)
        
        # Load model with this configuration
        model = framework.load_model(
            "path/to/test_model.pth",
            **config
        )
        
        config_results = {}
        
        for scenario in test_scenarios:
            scenario_name = scenario['name']
            batch_size = scenario['batch_size']
            iterations = scenario['iterations']
            
            print(f"  📋 {scenario_name} (batch_size={batch_size})")
            
            # Create test input
            test_input = torch.randn(batch_size, 3, 224, 224)
            if framework.device.type == 'cuda':
                test_input = test_input.cuda()
            
            # Warmup
            for _ in range(10):
                _ = framework.predict(model.model_id, test_input)
            
            # Clear memory
            if framework.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Benchmark
            times = []
            memory_usage = []
            
            for _ in range(iterations):
                # Memory before
                if framework.device.type == 'cuda':
                    mem_before = torch.cuda.memory_allocated()
                
                # Inference
                start = time.time()
                result = framework.predict(model.model_id, test_input)
                end = time.time()
                
                times.append(end - start)
                
                # Memory after
                if framework.device.type == 'cuda':
                    mem_after = torch.cuda.memory_allocated()
                    memory_usage.append((mem_after - mem_before) / 1024 / 1024)
            
            # Calculate metrics
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = batch_size / avg_time
            
            if framework.device.type == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                avg_memory = np.mean(memory_usage)
            else:
                peak_memory = 0
                avg_memory = 0
            
            config_results[scenario_name] = {
                'avg_time_ms': avg_time * 1000,
                'std_time_ms': std_time * 1000,
                'throughput_fps': throughput,
                'peak_memory_mb': peak_memory,
                'avg_memory_mb': avg_memory,
                'batch_size': batch_size
            }
            
            print(f"    ⏱️  Time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
            print(f"    🚀 Throughput: {throughput:.1f} FPS")
            print(f"    💾 Memory: {peak_memory:.1f} MB")
        
        results[config_name] = config_results
    
    # Print comprehensive comparison
    print(f"\n📊 Comprehensive Benchmark Results:")
    print(f"{'Config':<12} {'Scenario':<15} {'Time (ms)':<12} {'Throughput':<12} {'Memory (MB)':<12} {'Speedup':<10}")
    print("-" * 85)
    
    baseline_results = results.get('baseline', {})
    
    for config_name, config_results in results.items():
        for scenario_name, data in config_results.items():
            # Calculate speedup vs baseline
            if baseline_results and scenario_name in baseline_results:
                baseline_time = baseline_results[scenario_name]['avg_time_ms']
                speedup = baseline_time / data['avg_time_ms']
            else:
                speedup = 1.0
            
            print(f"{config_name:<12} {scenario_name:<15} {data['avg_time_ms']:<12.2f} "
                  f"{data['throughput_fps']:<12.1f} {data['peak_memory_mb']:<12.1f} "
                  f"{speedup:<10.2f}x")
    
    return results

benchmark_results = comprehensive_benchmark()
```

## 🎯 Best Practices Summary

### Quick Optimization Checklist

```python
# Production-ready optimization setup
def production_optimization_setup():
    """Apply production-ready optimizations."""
    
    print("🚀 Applying production optimizations...")
    
    # 1. Auto-detect optimal configuration
    optimal_config = framework.auto_detect_optimal_config()
    
    # 2. Apply hardware-specific optimizations
    if framework.device.type == 'cuda':
        # GPU optimizations
        framework.apply_gpu_optimizations(
            use_fp16=True,
            use_tensorrt=True,
            enable_cuda_graphs=True,
            memory_pooling=True
        )
    else:
        # CPU optimizations
        framework.apply_cpu_optimizations(
            use_mkldnn=True,
            use_quantization=True,
            enable_jit=True
        )
    
    # 3. Configure dynamic batching
    framework.configure_dynamic_batching(
        max_batch_size=32,
        timeout_ms=50,
        enable_inflight_batching=True
    )
    
    # 4. Enable monitoring
    framework.enable_performance_monitoring()
    
    # 5. Apply system optimizations
    framework.apply_system_optimizations()
    
    print("✅ Production optimizations applied")
    
    return optimal_config

production_config = production_optimization_setup()
```

### Optimization Decision Tree

```python
def optimization_decision_tree():
    """Guide for choosing the right optimizations."""
    
    # Get system information
    system_info = framework.get_system_info()
    
    recommendations = {
        'device_optimizations': [],
        'model_optimizations': [],
        'batch_optimizations': [],
        'memory_optimizations': []
    }
    
    # Device-specific recommendations
    if system_info['device_type'] == 'cuda':
        if system_info['gpu_memory_gb'] >= 8:
            recommendations['device_optimizations'].extend([
                'Use FP16 precision',
                'Enable TensorRT',
                'Use CUDA Graphs',
                'Enable tensor cores'
            ])
        else:
            recommendations['device_optimizations'].extend([
                'Use FP16 precision',
                'Enable memory pooling',
                'Use smaller batch sizes'
            ])
    elif system_info['device_type'] == 'cpu':
        recommendations['device_optimizations'].extend([
            'Use MKLDNN',
            'Enable dynamic quantization',
            'Use ONNX runtime',
            'Optimize thread count'
        ])
    
    # Model-specific recommendations
    model_info = framework.get_model_info("target_model")
    
    if model_info['model_type'] == 'transformer':
        recommendations['model_optimizations'].extend([
            'Use torch.compile with max-autotune',
            'Enable attention optimization',
            'Use dynamic shapes carefully'
        ])
    elif model_info['model_type'] == 'cnn':
        recommendations['model_optimizations'].extend([
            'Use TensorRT for CNN layers',
            'Enable CUDNN benchmarking',
            'Use static shapes'
        ])
    
    # Batch recommendations based on use case
    if system_info['use_case'] == 'real_time':
        recommendations['batch_optimizations'].extend([
            'Use small batch sizes (1-4)',
            'Enable CUDA Graphs',
            'Minimize batching timeout'
        ])
    elif system_info['use_case'] == 'throughput':
        recommendations['batch_optimizations'].extend([
            'Use large batch sizes (16-32)',
            'Enable dynamic batching',
            'Optimize for GPU utilization'
        ])
    
    # Memory recommendations
    if system_info['memory_gb'] < 16:
        recommendations['memory_optimizations'].extend([
            'Use aggressive memory pooling',
            'Enable automatic garbage collection',
            'Use gradient checkpointing',
            'Consider model quantization'
        ])
    
    # Print recommendations
    print("🎯 Optimization Recommendations:")
    for category, recs in recommendations.items():
        if recs:
            print(f"\n{category.replace('_', ' ').title()}:")
            for rec in recs:
                print(f"  • {rec}")
    
    return recommendations

optimization_recommendations = optimization_decision_tree()
```

---

This optimization guide provides comprehensive strategies for maximizing the performance of your PyTorch Inference Framework deployment. Choose the optimizations that best fit your hardware, model, and use case requirements.

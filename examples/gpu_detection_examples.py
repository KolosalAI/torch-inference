"""
GPU Detection Integration Examples

This module provides examples of how to use the GPU detection system
in different scenarios within the PyTorch inference framework.
"""

import torch
import time
from pathlib import Path
import sys

# Add project root to path for standalone execution
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from framework.core.gpu_detection import GPUDetector, print_gpu_report
from framework.core.gpu_manager import GPUManager, auto_configure_device
from framework.core.config import InferenceConfig, DeviceConfig


def example_basic_detection():
    """Example: Basic GPU detection and reporting."""
    print("=" * 60)
    print("EXAMPLE 1: Basic GPU Detection")
    print("=" * 60)
    
    # Simple detection
    detector = GPUDetector(enable_benchmarks=False)
    gpus = detector.detect_all_gpus()
    
    print(f"Found {len(gpus)} GPU(s)")
    
    for gpu in gpus:
        print(f"\nGPU {gpu.id}: {gpu.name}")
        print(f"  Vendor: {gpu.vendor.value}")
        print(f"  Memory: {gpu.memory.total_mb:.0f} MB")
        print(f"  PyTorch Support: {gpu.pytorch_support}")
        print(f"  Suitable for Inference: {gpu.is_suitable_for_inference()}")


def example_detailed_detection():
    """Example: Detailed GPU detection with benchmarks."""
    print("=" * 60)
    print("EXAMPLE 2: Detailed GPU Detection with Benchmarks")
    print("=" * 60)
    
    # Detailed detection with benchmarks
    print("Running comprehensive GPU detection (this may take a moment)...")
    print_gpu_report(enable_benchmarks=True)


def example_auto_configuration():
    """Example: Automatic device configuration."""
    print("=" * 60)
    print("EXAMPLE 3: Automatic Device Configuration")
    print("=" * 60)
    
    # Automatic configuration
    device_config = auto_configure_device()
    
    print("Auto-configured device settings:")
    print(f"  Device Type: {device_config.device_type.value}")
    print(f"  Device ID: {device_config.device_id}")
    print(f"  FP16 Enabled: {device_config.use_fp16}")
    print(f"  INT8 Enabled: {device_config.use_int8}")
    print(f"  TensorRT Enabled: {device_config.use_tensorrt}")
    print(f"  Torch Compile Enabled: {device_config.use_torch_compile}")
    
    # Create inference configuration
    inference_config = InferenceConfig()
    inference_config.device = device_config
    
    print(f"\nActual PyTorch device: {device_config.get_torch_device()}")


def example_memory_optimization():
    """Example: Memory-aware optimization recommendations."""
    print("=" * 60)
    print("EXAMPLE 4: Memory Optimization Recommendations")
    print("=" * 60)
    
    manager = GPUManager()
    gpus, device_config = manager.detect_and_configure()
    
    # Get memory recommendations
    memory_rec = manager.get_memory_recommendations()
    
    print("Memory Analysis:")
    print(f"  Total Memory: {memory_rec.get('total_memory_mb', 0):.0f} MB")
    print(f"  Available Memory: {memory_rec.get('available_memory_mb', 0):.0f} MB")
    print(f"  Estimated Max Batch Size: {memory_rec.get('estimated_max_batch_size', 1)}")
    
    print("\nMemory Optimization Recommendations:")
    for rec in memory_rec.get("recommendations", []):
        print(f"  - {rec}")


def example_best_gpu_selection():
    """Example: Selecting the best GPU for inference."""
    print("=" * 60)
    print("EXAMPLE 5: Best GPU Selection")
    print("=" * 60)
    
    detector = GPUDetector(enable_benchmarks=False)
    gpus = detector.detect_all_gpus()
    
    if not gpus:
        print("No GPUs detected.")
        return
    
    best_gpu = detector.get_best_gpu(gpus)
    
    if best_gpu:
        print(f"Best GPU for inference: {best_gpu.name}")
        print(f"  Device ID: {best_gpu.device_id}")
        print(f"  Memory: {best_gpu.memory.total_mb:.0f} MB")
        print(f"  Architecture: {best_gpu.architecture.value}")
        
        if best_gpu.compute_capability:
            print(f"  Compute Capability: {best_gpu.compute_capability.version}")
            print(f"  Tensor Cores: {best_gpu.compute_capability.supports_tensor_cores}")
        
        print(f"  Recommended Precisions: {', '.join(best_gpu.get_recommended_precision())}")
    else:
        print("No suitable GPU found for inference.")


def example_model_specific_optimization():
    """Example: Model-specific GPU optimization."""
    print("=" * 60)
    print("EXAMPLE 6: Model-Specific Optimization")
    print("=" * 60)
    
    # Simulate different model scenarios
    model_scenarios = [
        {"name": "Small CNN", "size_mb": 50, "type": "computer_vision"},
        {"name": "BERT-Base", "size_mb": 500, "type": "nlp"},
        {"name": "Large Transformer", "size_mb": 2000, "type": "llm"},
        {"name": "Huge Model", "size_mb": 10000, "type": "multimodal"}
    ]
    
    manager = GPUManager()
    best_gpu = manager.get_best_gpu_info()
    
    if not best_gpu:
        print("No suitable GPU found for model optimization examples.")
        return
    
    print(f"Using GPU: {best_gpu.name} ({best_gpu.memory.total_mb:.0f} MB)")
    print()
    
    for scenario in model_scenarios:
        print(f"Model: {scenario['name']} ({scenario['size_mb']} MB)")
        
        # Estimate batch size
        max_batch_size = best_gpu.estimate_max_batch_size(scenario['size_mb'])
        print(f"  Estimated Max Batch Size: {max_batch_size}")
        
        # Memory utilization
        memory_usage_percent = (scenario['size_mb'] / best_gpu.memory.total_mb) * 100
        print(f"  Model Memory Usage: {memory_usage_percent:.1f}%")
        
        # Recommendations based on model size
        if memory_usage_percent > 80:
            print("  Recommendation: Consider model quantization or smaller batch sizes")
        elif memory_usage_percent > 50:
            print("  Recommendation: Use moderate batch sizes, enable mixed precision")
        else:
            print("  Recommendation: Can use large batch sizes, full precision OK")
        
        print()


def example_performance_monitoring():
    """Example: GPU performance monitoring during inference."""
    print("=" * 60)
    print("EXAMPLE 7: Performance Monitoring")
    print("=" * 60)
    
    detector = GPUDetector(enable_benchmarks=True)
    gpus = detector.detect_all_gpus()
    
    if not gpus:
        print("No GPUs detected for performance monitoring.")
        return
    
    gpu = gpus[0]  # Use first GPU
    
    if not gpu.pytorch_support:
        print("GPU does not support PyTorch for performance monitoring.")
        return
    
    print(f"Monitoring GPU: {gpu.name}")
    
    # Show current performance metrics
    if gpu.performance.gpu_utilization_percent > 0:
        print(f"  GPU Utilization: {gpu.performance.gpu_utilization_percent:.1f}%")
    
    if gpu.performance.memory_utilization_percent > 0:
        print(f"  Memory Utilization: {gpu.performance.memory_utilization_percent:.1f}%")
    
    if gpu.performance.temperature_celsius:
        print(f"  Temperature: {gpu.performance.temperature_celsius:.1f}°C")
    
    if gpu.performance.power_draw_watts:
        print(f"  Power Draw: {gpu.performance.power_draw_watts:.1f}W")
    
    # Show benchmark results
    if gpu.benchmark_results:
        print("\nBenchmark Results:")
        
        if "memory_bandwidth" in gpu.benchmark_results:
            bw = gpu.benchmark_results["memory_bandwidth"]["bandwidth_gb_s"]
            print(f"  Memory Bandwidth: {bw:.1f} GB/s")
        
        if "fp32_performance" in gpu.benchmark_results:
            ops = gpu.benchmark_results["fp32_performance"]["ops_per_second"]
            print(f"  FP32 Performance: {ops:.0f} ops/sec")
        
        if "matmul_performance" in gpu.benchmark_results:
            matmul_results = gpu.benchmark_results["matmul_performance"]
            for key, result in matmul_results.items():
                if isinstance(result, dict) and "gflops" in result:
                    print(f"  {key}: {result['gflops']:.1f} GFLOPS")


def example_error_handling():
    """Example: Error handling and fallback strategies."""
    print("=" * 60)
    print("EXAMPLE 8: Error Handling and Fallback")
    print("=" * 60)
    
    try:
        # Try to detect GPUs
        manager = GPUManager()
        gpus, device_config = manager.detect_and_configure()
        
        if not gpus:
            print("No GPUs detected - falling back to CPU")
            print(f"CPU Configuration: {device_config.device_type.value}")
        else:
            print(f"Successfully configured for: {device_config.device_type.value}")
            
        # Test device creation
        device = device_config.get_torch_device()
        print(f"PyTorch device: {device}")
        
        # Test tensor creation
        test_tensor = torch.randn(10, 10, device=device)
        print(f"Successfully created tensor on {device}: {test_tensor.shape}")
        
    except Exception as e:
        print(f"Error during GPU detection: {e}")
        print("Falling back to CPU configuration")
        
        # Fallback configuration
        fallback_config = DeviceConfig(device_type="cpu")
        device = fallback_config.get_torch_device()
        test_tensor = torch.randn(10, 10, device=device)
        print(f"Fallback successful: {test_tensor.shape} on {device}")


def run_all_examples():
    """Run all examples."""
    examples = [
        example_basic_detection,
        example_auto_configuration,
        example_memory_optimization,
        example_best_gpu_selection,
        example_model_specific_optimization,
        example_performance_monitoring,
        example_error_handling,
        # Note: example_detailed_detection is excluded by default as it's slow
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
            if i < len(examples):
                print("\n" + "—" * 60 + "\n")
        except Exception as e:
            print(f"Error in example {i}: {e}")
            print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Detection Examples")
    parser.add_argument("--example", "-e", type=int, choices=range(1, 9),
                       help="Run specific example (1-8)")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Run all examples")
    parser.add_argument("--detailed", "-d", action="store_true",
                       help="Include detailed detection example (slow)")
    
    args = parser.parse_args()
    
    if args.example:
        examples = [
            example_basic_detection,
            example_detailed_detection,
            example_auto_configuration,
            example_memory_optimization,
            example_best_gpu_selection,
            example_model_specific_optimization,
            example_performance_monitoring,
            example_error_handling,
        ]
        examples[args.example - 1]()
    elif args.all:
        if args.detailed:
            example_detailed_detection()
            print("\n" + "—" * 60 + "\n")
        run_all_examples()
    else:
        print("GPU Detection System Examples")
        print("=" * 40)
        print("1. Basic GPU Detection")
        print("2. Detailed Detection with Benchmarks")
        print("3. Automatic Device Configuration")
        print("4. Memory Optimization")
        print("5. Best GPU Selection")
        print("6. Model-Specific Optimization")
        print("7. Performance Monitoring")
        print("8. Error Handling")
        print()
        print("Use --help for options")
        print("Example: python examples/gpu_detection_examples.py --example 1")
        print("Example: python examples/gpu_detection_examples.py --all")

"""
Comprehensive Examples for Vulkan, Numba, and Enhanced JIT Integration

This module demonstrates how to use the enhanced optimization capabilities
including Vulkan compute acceleration, Numba JIT compilation, and multi-backend
optimization strategies.
"""

import logging
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional

# Framework imports
from framework.core.config import InferenceConfig, DeviceConfig, DeviceType
from framework.optimizers.performance_optimizer import PerformanceOptimizer
from framework.optimizers.jit_optimizer import EnhancedJITOptimizer
from framework.optimizers.vulkan_optimizer import VulkanOptimizer, VULKAN_AVAILABLE
from framework.optimizers.numba_optimizer import NumbaOptimizer, NUMBA_AVAILABLE


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTestModel(nn.Module):
    """Simple model for testing optimizations."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 256, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ConvTestModel(nn.Module):
    """Convolutional model for testing compute-intensive optimizations."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


def example_basic_optimization():
    """Example 1: Basic optimization with automatic backend selection."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Optimization with Automatic Backend Selection")
    print("=" * 80)
    
    # Create configuration
    config = InferenceConfig()
    config.device.jit_strategy = "auto"  # Let the system choose the best strategy
    
    # Create model and example inputs
    model = SimpleTestModel()
    example_inputs = torch.randn(1, 28, 28)
    
    # Initialize performance optimizer
    optimizer = PerformanceOptimizer(config)
    
    # Optimize device configuration
    optimized_device_config = optimizer.optimize_device_config()
    print(f"Optimized device: {optimized_device_config.device_type.value}")
    print(f"JIT strategy: {optimized_device_config.jit_strategy}")
    print(f"Vulkan enabled: {optimized_device_config.use_vulkan}")
    print(f"Numba enabled: {optimized_device_config.use_numba}")
    
    # Get torch device
    device = optimized_device_config.get_torch_device()
    
    # Optimize model
    optimized_model = optimizer.optimize_model(model, device, example_inputs)
    
    print(f"Applied optimizations: {', '.join(optimizer.optimizations_applied)}")
    print()


def example_vulkan_acceleration():
    """Example 2: Vulkan compute acceleration."""
    print("=" * 80)
    print("EXAMPLE 2: Vulkan Compute Acceleration")
    print("=" * 80)
    
    if not VULKAN_AVAILABLE:
        print("Vulkan not available - skipping example")
        print("Install with: pip install vulkan")
        return
    
    # Initialize Vulkan optimizer
    vulkan_optimizer = VulkanOptimizer()
    
    # Check availability
    print(f"Vulkan available: {vulkan_optimizer.is_available()}")
    
    if not vulkan_optimizer.is_available():
        print("No suitable Vulkan devices found")
        return
    
    # Get device information
    device_info = vulkan_optimizer.get_device_info()
    print(f"Vulkan devices detected: {device_info['devices_detected']}")
    print(f"Compute-capable devices: {device_info['compute_capable_devices']}")
    print(f"Active device: {device_info['active_device']}")
    
    # Benchmark Vulkan performance
    print("\nBenchmarking Vulkan performance...")
    test_sizes = [(1024, 1024), (2048, 2048), (4096, 4096)]
    
    for size in test_sizes:
        results = vulkan_optimizer.benchmark_vulkan_performance(size, iterations=50)
        if "error" not in results:
            print(f"Size {size}: Vulkan {results['vulkan_time_ms']:.2f}ms, "
                  f"PyTorch {results['pytorch_time_ms']:.2f}ms, "
                  f"Speedup: {results['speedup']:.2f}x")
        else:
            print(f"Size {size}: {results['error']}")
    
    # Demonstrate tensor optimization
    print("\nTensor optimization example...")
    test_tensor = torch.randn(1000, 1000)
    
    # Original operation
    start_time = time.perf_counter()
    result_original = torch.sin(test_tensor) + torch.cos(test_tensor)
    original_time = (time.perf_counter() - start_time) * 1000
    
    # Vulkan-optimized operation
    start_time = time.perf_counter()
    result_vulkan = vulkan_optimizer.optimize_tensor_operations(test_tensor, "elementwise")
    vulkan_time = (time.perf_counter() - start_time) * 1000
    
    print(f"Original operation: {original_time:.2f}ms")
    print(f"Vulkan operation: {vulkan_time:.2f}ms")
    print()


def example_numba_jit_acceleration():
    """Example 3: Numba JIT acceleration."""
    print("=" * 80)
    print("EXAMPLE 3: Numba JIT Acceleration")
    print("=" * 80)
    
    if not NUMBA_AVAILABLE:
        print("Numba not available - skipping example")
        print("Install with: pip install numba")
        return
    
    # Initialize Numba optimizer
    numba_optimizer = NumbaOptimizer()
    
    # Check availability
    print(f"Numba available: {numba_optimizer.is_available()}")
    print(f"Numba CUDA available: {numba_optimizer.is_cuda_available()}")
    
    # Get optimization stats
    stats = numba_optimizer.get_optimization_stats()
    print(f"Numba version: {stats.get('numba_version', 'N/A')}")
    
    # Create optimized operations
    print("\nCreating optimized operations...")
    ops = numba_optimizer.create_optimized_operations()
    print(f"Created {len(ops)} optimized operations")
    
    # Benchmark operations
    print("\nBenchmarking Numba operations...")
    benchmark_configs = [
        {"size": (1000, 1000), "operation": "relu"},
        {"size": (2000, 2000), "operation": "sigmoid"},
        {"size": (1000, 1000), "operation": "elementwise_add"}
    ]
    
    for config in benchmark_configs:
        results = numba_optimizer.benchmark_numba_performance(
            config["size"], config["operation"], iterations=50
        )
        
        if "error" not in results:
            print(f"{config['operation']} {config['size']}: "
                  f"NumPy {results['numpy_time_ms']:.2f}ms, "
                  f"Numba {results['numba_time_ms']:.2f}ms, "
                  f"Speedup: {results['numba_speedup']:.2f}x")
            
            if results['cuda_speedup'] > 1.1:
                print(f"  CUDA: {results['cuda_time_ms']:.2f}ms, "
                      f"Speedup: {results['cuda_speedup']:.2f}x")
        else:
            print(f"{config['operation']}: {results['error']}")
    
    # Model optimization example
    print("\nModel optimization with Numba...")
    model = SimpleTestModel()
    optimized_model = numba_optimizer.wrap_model_with_numba(model)
    print("Model wrapped with Numba optimizations")
    print()


def example_enhanced_jit_strategies():
    """Example 4: Enhanced JIT optimization strategies."""
    print("=" * 80)
    print("EXAMPLE 4: Enhanced JIT Optimization Strategies")
    print("=" * 80)
    
    # Create different models for testing
    models = {
        "simple": SimpleTestModel(),
        "conv": ConvTestModel()
    }
    
    example_inputs = {
        "simple": torch.randn(4, 28, 28),
        "conv": torch.randn(4, 3, 32, 32)
    }
    
    # Test different optimization strategies
    strategies = ["auto", "torch_jit", "vulkan", "numba", "multi"]
    
    for model_name, model in models.items():
        print(f"\nTesting model: {model_name}")
        print("-" * 40)
        
        # Initialize enhanced JIT optimizer
        config = InferenceConfig()
        try:
            jit_optimizer = EnhancedJITOptimizer(config)
            
            # Get optimization capabilities
            capabilities = jit_optimizer.get_optimization_capabilities()
            print("Available optimizations:")
            for backend, info in capabilities.items():
                print(f"  {backend}: {info['available']} - {info['features']}")
            
            # Benchmark different strategies
            inputs = example_inputs[model_name]
            results = jit_optimizer.benchmark_optimization_strategies(
                model, inputs, strategies=strategies, iterations=20
            )
            
            print(f"\nBenchmark results (input shape: {list(inputs.shape)}):")
            for strategy, result in results["results"].items():
                if "error" not in result:
                    time_ms = result["avg_inference_time_ms"]
                    improvement = result.get("improvement_percent", 0)
                    print(f"  {strategy:12}: {time_ms:6.2f}ms ({improvement:+5.1f}%)")
                else:
                    print(f"  {strategy:12}: ERROR - {result['error']}")
                    
        except Exception as e:
            print(f"Enhanced JIT not available: {e}")
            print("Using basic JIT optimization")


def example_multi_backend_optimization():
    """Example 5: Multi-backend optimization pipeline."""
    print("=" * 80)
    print("EXAMPLE 5: Multi-Backend Optimization Pipeline")
    print("=" * 80)
    
    # Create comprehensive configuration
    config = InferenceConfig()
    config.device.jit_strategy = "multi"
    config.device.use_vulkan = VULKAN_AVAILABLE
    config.device.use_numba = NUMBA_AVAILABLE
    config.device.use_torch_compile = True
    
    # Create model and inputs
    model = ConvTestModel()
    example_inputs = torch.randn(1, 3, 32, 32)
    
    print("Configuration:")
    print(f"  JIT strategy: {config.device.jit_strategy}")
    print(f"  Vulkan enabled: {config.device.use_vulkan}")
    print(f"  Numba enabled: {config.device.use_numba}")
    print(f"  Torch compile enabled: {config.device.use_torch_compile}")
    
    # Initialize performance optimizer
    optimizer = PerformanceOptimizer(config)
    
    # Optimize device configuration
    device_config = optimizer.optimize_device_config()
    device = device_config.get_torch_device()
    
    print(f"\nOptimized device configuration:")
    print(f"  Device: {device}")
    print(f"  Optimizations applied: {optimizer.optimizations_applied}")
    
    # Benchmark original vs optimized model
    print(f"\nBenchmarking original vs optimized model...")
    
    # Original model performance
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(example_inputs)
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(50):
            _ = model(example_inputs)
        original_time = (time.perf_counter() - start_time) * 1000
    
    # Optimized model performance
    optimized_model = optimizer.optimize_model(model, device, example_inputs)
    optimized_model.eval()
    
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = optimized_model(example_inputs)
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(50):
            _ = optimized_model(example_inputs)
        optimized_time = (time.perf_counter() - start_time) * 1000
    
    speedup = original_time / optimized_time if optimized_time > 0 else 1.0
    improvement = ((original_time - optimized_time) / original_time) * 100
    
    print(f"Original model: {original_time:.2f}ms")
    print(f"Optimized model: {optimized_time:.2f}ms")
    print(f"Speedup: {speedup:.2f}x ({improvement:+.1f}%)")
    print()


def example_optimization_analysis():
    """Example 6: Optimization analysis and recommendations."""
    print("=" * 80)
    print("EXAMPLE 6: Optimization Analysis and Recommendations")
    print("=" * 80)
    
    # Create different model types for analysis
    models = {
        "lightweight": SimpleTestModel(input_size=128, hidden_size=64),
        "standard": SimpleTestModel(input_size=784, hidden_size=256),
        "large": SimpleTestModel(input_size=1024, hidden_size=512),
        "conv_small": ConvTestModel(),
    }
    
    config = InferenceConfig()
    optimizer = PerformanceOptimizer(config)
    
    try:
        jit_optimizer = EnhancedJITOptimizer(config)
        
        print("Model Analysis and Optimization Recommendations:")
        print("-" * 60)
        
        for model_name, model in models.items():
            print(f"\nModel: {model_name}")
            
            # Analyze model characteristics
            if hasattr(jit_optimizer, '_analyze_model'):
                analysis = jit_optimizer._analyze_model(model)
                print(f"  Parameters: {analysis['num_parameters']:,}")
                print(f"  Layers: {analysis['num_layers']}")
                print(f"  Compute intensive: {analysis['is_compute_intensive']}")
                print(f"  Simple operations: {analysis['has_simple_ops']}")
                
                # Recommend optimization strategy
                if analysis['is_compute_intensive'] and VULKAN_AVAILABLE:
                    print("  Recommendation: Use Vulkan for compute acceleration")
                elif analysis['has_simple_ops'] and NUMBA_AVAILABLE:
                    print("  Recommendation: Use Numba for JIT acceleration")
                else:
                    print("  Recommendation: Use TorchScript optimization")
                    
        # System capabilities summary
        print(f"\nSystem Optimization Capabilities:")
        capabilities = jit_optimizer.get_optimization_capabilities()
        for backend, info in capabilities.items():
            status = "✓" if info['available'] else "✗"
            print(f"  {status} {backend.upper()}: {info['features']}")
            
    except Exception as e:
        print(f"Enhanced analysis not available: {e}")
    
    print()


def run_all_examples():
    """Run all optimization examples."""
    print("🚀 PyTorch Inference Framework - Enhanced Optimization Examples")
    print("=" * 80)
    print("This demo showcases Vulkan, Numba, and Enhanced JIT integration")
    print()
    
    examples = [
        example_basic_optimization,
        example_vulkan_acceleration,
        example_numba_jit_acceleration,
        example_enhanced_jit_strategies,
        example_multi_backend_optimization,
        example_optimization_analysis
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"Example {i} failed: {e}")
            print()
    
    print("=" * 80)
    print("✅ All examples completed!")
    print()
    print("Next steps:")
    print("1. Install missing dependencies (Vulkan, Numba) for full functionality")
    print("2. Configure your models to use the enhanced optimization strategies")
    print("3. Benchmark your specific workloads to find optimal settings")
    print("4. Monitor performance improvements in production")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Optimization Examples")
    parser.add_argument("--example", "-e", type=int, choices=range(1, 7),
                       help="Run specific example (1-6)")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Run all examples")
    
    args = parser.parse_args()
    
    if args.example:
        examples = [
            example_basic_optimization,
            example_vulkan_acceleration,
            example_numba_jit_acceleration,
            example_enhanced_jit_strategies,
            example_multi_backend_optimization,
            example_optimization_analysis
        ]
        examples[args.example - 1]()
    elif args.all:
        run_all_examples()
    else:
        print("Enhanced Optimization Examples Available:")
        print("1. Basic Optimization with Automatic Backend Selection")
        print("2. Vulkan Compute Acceleration")
        print("3. Numba JIT Acceleration")
        print("4. Enhanced JIT Optimization Strategies")
        print("5. Multi-Backend Optimization Pipeline")
        print("6. Optimization Analysis and Recommendations")
        print()
        print("Run with: python examples_enhanced_optimization.py --all")
        print("Or run specific example: python examples_enhanced_optimization.py -e 1")

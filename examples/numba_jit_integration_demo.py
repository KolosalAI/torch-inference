"""
Example script demonstrating Numba JIT integration in the PyTorch Inference Framework

This script shows how Numba JIT compilation is seamlessly integrated into the
existing codebase to provide performance optimizations without changing any
class names or core structure.
"""

import sys
import os
import time
import numpy as np
import torch
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from framework.core.jit_integration import (
        initialize_jit_integration, 
        apply_tensor_jit, 
        apply_array_jit,
        fast_matrix_multiply,
        fast_activation_functions
    )
    from framework.optimizers.numba_optimizer import NumbaOptimizer
    from framework.core.inference_engine import InferenceEngine
    from framework.core.config import InferenceConfig
    from tests.models.simple_model import SimpleModel  # Use test model instead
    
    print("✅ All framework imports successful")
    
except ImportError as e:
    print(f"❌ Framework import failed: {e}")
    print("Please ensure the framework is properly installed")
    
    # Try alternative import for SimpleModel
    try:
        from tests.models.test_model import TestModel as SimpleModel
        print("✅ Using test model as fallback")
    except ImportError:
        print("❌ No suitable model found, will skip model tests")
        SimpleModel = None


async def demonstrate_jit_integration():
    """Demonstrate JIT integration across the framework."""
    
    print("\n" + "="*60)
    print("🚀 NUMBA JIT INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Initialize JIT integration
    print("\n1. Initializing JIT Integration...")
    jit_manager = initialize_jit_integration(enable_jit=True)
    
    if jit_manager.is_available():
        print("✅ JIT integration available")
        stats = jit_manager.get_performance_stats()
        print(f"   - Numba version: {stats.get('numba_version', 'N/A')}")
        print(f"   - CUDA available: {stats.get('cuda_available', False)}")
        print(f"   - Optimized functions: {stats.get('optimized_functions', 0)}")
    else:
        print("❌ JIT integration not available")
        return
    
    # Demonstrate tensor optimization
    print("\n2. Tensor Optimization Example...")
    
    # Create test tensor
    test_tensor = torch.randn(1000, 1000)
    print(f"   Test tensor shape: {test_tensor.shape}")
    
    # Standard PyTorch operation
    start_time = time.perf_counter()
    standard_result = torch.relu(test_tensor)
    pytorch_time = (time.perf_counter() - start_time) * 1000
    
    # JIT-optimized operation
    start_time = time.perf_counter()
    jit_result = apply_tensor_jit(test_tensor, "relu")
    jit_time = (time.perf_counter() - start_time) * 1000
    
    print(f"   PyTorch time: {pytorch_time:.2f}ms")
    print(f"   JIT time: {jit_time:.2f}ms")
    print(f"   Speedup: {pytorch_time/jit_time:.2f}x" if jit_time > 0 else "   Speedup: N/A")
    print(f"   Results match: {torch.allclose(standard_result, jit_result, atol=1e-6)}")
    
    # Demonstrate numpy array optimization
    print("\n3. NumPy Array Optimization Example...")
    
    # Create test array
    test_array = np.random.randn(1000, 1000).astype(np.float32)
    print(f"   Test array shape: {test_array.shape}")
    
    # Standard NumPy operation
    start_time = time.perf_counter()
    standard_numpy = np.maximum(0, test_array)
    numpy_time = (time.perf_counter() - start_time) * 1000
    
    # JIT-optimized operation
    start_time = time.perf_counter()
    jit_numpy = apply_array_jit(test_array, "relu")
    jit_numpy_time = (time.perf_counter() - start_time) * 1000
    
    print(f"   NumPy time: {numpy_time:.2f}ms")
    print(f"   JIT time: {jit_numpy_time:.2f}ms")
    print(f"   Speedup: {numpy_time/jit_numpy_time:.2f}x" if jit_numpy_time > 0 else "   Speedup: N/A")
    print(f"   Results match: {np.allclose(standard_numpy, jit_numpy, atol=1e-6)}")
    
    # Demonstrate matrix multiplication optimization
    print("\n4. Matrix Multiplication Optimization...")
    
    # Create test matrices
    A = np.random.randn(500, 500).astype(np.float32)
    B = np.random.randn(500, 500).astype(np.float32)
    
    # Standard NumPy operation
    start_time = time.perf_counter()
    standard_matmul = np.dot(A, B)
    numpy_matmul_time = (time.perf_counter() - start_time) * 1000
    
    # JIT-optimized operation
    start_time = time.perf_counter()
    jit_matmul = fast_matrix_multiply(A, B)
    jit_matmul_time = (time.perf_counter() - start_time) * 1000
    
    print(f"   NumPy matmul time: {numpy_matmul_time:.2f}ms")
    print(f"   JIT matmul time: {jit_matmul_time:.2f}ms")
    print(f"   Speedup: {numpy_matmul_time/jit_matmul_time:.2f}x" if jit_matmul_time > 0 else "   Speedup: N/A")
    print(f"   Results match: {np.allclose(standard_matmul, jit_matmul, atol=1e-5)}")
    
    # Demonstrate activation function optimization
    print("\n5. Activation Function Optimization...")
    
    test_activations = np.random.randn(1000, 1000).astype(np.float32)
    
    activations = ["relu", "sigmoid", "tanh"]
    for activation in activations:
        print(f"\n   Testing {activation.upper()}:")
        
        # Standard operation
        start_time = time.perf_counter()
        if activation == "relu":
            standard_act = np.maximum(0, test_activations)
        elif activation == "sigmoid":
            standard_act = 1.0 / (1.0 + np.exp(-test_activations))
        elif activation == "tanh":
            standard_act = np.tanh(test_activations)
        standard_act_time = (time.perf_counter() - start_time) * 1000
        
        # JIT-optimized operation
        start_time = time.perf_counter()
        jit_act = fast_activation_functions(test_activations, activation)
        jit_act_time = (time.perf_counter() - start_time) * 1000
        
        print(f"     Standard time: {standard_act_time:.2f}ms")
        print(f"     JIT time: {jit_act_time:.2f}ms")
        print(f"     Speedup: {standard_act_time/jit_act_time:.2f}x" if jit_act_time > 0 else "     Speedup: N/A")
        print(f"     Results match: {np.allclose(standard_act, jit_act, atol=1e-5)}")
    
    # Demonstrate framework integration
    print("\n6. Framework Integration Example...")
    
    if SimpleModel is not None:
        try:
            # Create a simple inference setup
            config = InferenceConfig()
            model = SimpleModel(config)
            
            # Create inference engine (should automatically have JIT integration)
            engine = InferenceEngine(model, config)
            
            print("✅ Inference engine created with JIT integration")
            print(f"   JIT enabled in engine: {hasattr(engine, '_numba_enabled') and engine._numba_enabled}")
            
            # Test inference with JIT optimization
            test_input = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            start_time = time.perf_counter()
            result = await engine.predict_async(test_input)
            inference_time = (time.perf_counter() - start_time) * 1000
            
            print(f"   Inference time: {inference_time:.2f}ms")
            print(f"   Result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            
        except Exception as e:
            print(f"   Framework integration test failed: {e}")
    else:
        print("   Skipping framework integration test (no model available)")
    
    print("\n" + "="*60)
    print("🎉 JIT INTEGRATION DEMONSTRATION COMPLETE")
    print("="*60)


def benchmark_performance():
    """Benchmark JIT performance improvements."""
    
    print("\n" + "="*60)
    print("📊 PERFORMANCE BENCHMARKING")
    print("="*60)
    
    # Initialize JIT
    jit_manager = initialize_jit_integration(enable_jit=True)
    
    if not jit_manager.is_available():
        print("❌ JIT not available for benchmarking")
        return
    
    # Test different array sizes
    sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    operations = ["relu", "sigmoid"]
    
    print("\nBenchmarking different array sizes and operations:")
    print("-" * 60)
    
    for operation in operations:
        print(f"\n{operation.upper()} Operation:")
        print("Size\t\tNumPy(ms)\tJIT(ms)\t\tSpeedup")
        print("-" * 50)
        
        for size in sizes:
            try:
                # Run benchmark
                stats = jit_manager.benchmark_optimization(size, operation, iterations=50)
                
                numpy_time = stats.get("numpy_time_ms", 0)
                jit_time = stats.get("numba_time_ms", 0)
                speedup = stats.get("numba_speedup", 1.0)
                
                print(f"{size[0]}x{size[1]}\t\t{numpy_time:.2f}\t\t{jit_time:.2f}\t\t{speedup:.2f}x")
                
            except Exception as e:
                print(f"{size[0]}x{size[1]}\t\tError: {e}")
    
    print("\n" + "="*60)


async def main():
    """Main demonstration function."""
    
    print("🔥 PyTorch Inference Framework - Numba JIT Integration Demo")
    print("This demonstration shows how Numba JIT is integrated seamlessly")
    print("into the existing codebase without changing any class names or structure.")
    
    # Demonstrate JIT integration
    await demonstrate_jit_integration()
    
    # Benchmark performance
    benchmark_performance()
    
    print("\n✨ Integration Benefits:")
    print("  • Automatic JIT compilation for numerical operations")
    print("  • No changes to existing class names or structure")
    print("  • Graceful fallback when Numba is not available")
    print("  • Seamless integration across the entire framework")
    print("  • Performance improvements for CPU-intensive operations")
    print("  • Optional CUDA acceleration for GPU operations")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

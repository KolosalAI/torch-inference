"""
Basic Usage Example for PyTorch Inference Framework

This example demonstrates simple synchronous inference patterns
using the optimized PyTorch inference framework.
"""

import torch
import numpy as np
from pathlib import Path

# Import framework (placeholder - adjust based on actual implementation)
# from framework import create_pytorch_framework

def basic_inference_example():
    """
    Basic synchronous inference example
    """
    print("🚀 Basic Inference Example")
    print("=" * 50)
    
    # Example model path (placeholder)
    model_path = "path/to/your/model.pt"
    
    # Create basic framework
    print("📦 Creating PyTorch framework...")
    # framework = create_pytorch_framework(
    #     model_path=model_path,
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # )
    
    # Example input data
    input_data = torch.randn(1, 3, 224, 224)  # Example image tensor
    
    print(f"📊 Input shape: {input_data.shape}")
    print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Single prediction
    print("\n🔮 Single Prediction:")
    # result = framework.predict(input_data)
    # print(f"Result: {result}")
    
    # Batch prediction
    batch_data = [
        torch.randn(1, 3, 224, 224),
        torch.randn(1, 3, 224, 224), 
        torch.randn(1, 3, 224, 224)
    ]
    
    print(f"\n📦 Batch Prediction (batch size: {len(batch_data)}):")
    # results = framework.predict_batch(batch_data)
    # for i, result in enumerate(results):
    #     print(f"  Image {i+1}: {result}")
    
    print("\n✅ Basic inference example completed!")

def optimized_inference_example():
    """
    Example with automatic optimization enabled
    """
    print("\n⚡ Optimized Inference Example")
    print("=" * 50)
    
    # Enable automatic optimization
    # framework = create_pytorch_framework(
    #     model_path="path/to/your/model.pt",
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    #     enable_optimization=True,  # Automatic TensorRT/ONNX optimization
    #     optimization_level="balanced"  # Options: conservative, balanced, aggressive
    # )
    
    # Example input for optimization
    input_data = torch.randn(1, 3, 224, 224)
    
    print("🔧 Framework will automatically:")
    print("  - Detect optimal optimization method")
    print("  - Apply TensorRT if available")
    print("  - Fallback to ONNX or JIT compilation")
    print("  - Benchmark all methods and select best")
    
    # The framework handles optimization automatically
    # result = framework.predict(input_data)
    
    # Get optimization report
    # report = framework.get_optimization_report()
    # print(f"\n📊 Optimization Report:")
    # print(f"  Active optimization: {report.get('active_optimization', 'none')}")
    # print(f"  Speedup achieved: {report.get('speedup', 1.0):.2f}x")
    # print(f"  Memory saved: {report.get('memory_reduction', 0):.1%}")
    
    print("\n✅ Optimized inference example completed!")

def configuration_example():
    """
    Example showing different configuration options
    """
    print("\n🔧 Configuration Example")
    print("=" * 50)
    
    # from framework.core.config import InferenceConfig, DeviceConfig
    
    # # Create custom configuration
    # config = InferenceConfig(
    #     model_path="path/to/model.pt",
    #     device=DeviceConfig(
    #         device_type="cuda",
    #         gpu_id=0,
    #         memory_fraction=0.8,
    #         use_fp16=True  # Half precision for 2x speedup
    #     ),
    #     batch_size=8,
    #     enable_monitoring=True
    # )
    
    # # Create framework with configuration
    # framework = create_framework(config)
    
    print("⚙️  Configuration options available:")
    print("  - Device selection (CPU/CUDA/Auto)")
    print("  - Memory management")
    print("  - Precision settings (FP32/FP16)")
    print("  - Batch processing")
    print("  - Performance monitoring")
    print("  - Optimization preferences")
    
    print("\n✅ Configuration example completed!")

if __name__ == "__main__":
    """
    Run basic usage examples
    
    To run this example:
        uv run python examples/basic_usage.py
    """
    
    print("🎯 PyTorch Inference Framework - Basic Usage Examples")
    print("=" * 60)
    
    # Check PyTorch installation
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🚀 GPU device: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    try:
        # Run examples
        basic_inference_example()
        optimized_inference_example()
        configuration_example()
        
        print("\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("  - Try async_processing.py for high-throughput processing")
        print("  - See fastapi_server.py for REST API integration")
        print("  - Run ../optimization_demo.py for complete optimization showcase")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("Make sure to adjust model paths and install required dependencies")

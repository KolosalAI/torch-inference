# 🚀 PyTorch Inference Framework

> **Production-ready PyTorch inference framework with TensorRT, ONNX, quantization, and advanced acceleration techniques**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.12%2B-orange)](https://developer.nvidia.com/tensorrt)
[![uv](https://img.shields.io/badge/Package%20Manager-uv-purple)](https://github.com/astral-sh/uv)

A comprehensive, production-ready PyTorch inference framework that delivers **2-10x performance improvements** through advanced optimization techniques including TensorRT, ONNX Runtime, quantization, JIT compilation, and CUDA optimizations.

## 📚 Documentation

**Complete documentation is available in the [`docs/`](docs/) directory:**

- **[📖 Documentation Overview](docs/README.md)** - Complete documentation guide
- **[🚀 Quick Start](docs/quickstart.md)** - Get started in minutes  
- **[📦 Installation](docs/installation.md)** - Complete setup instructions
- **[⚙️ Configuration](docs/configuration.md)** - Configuration management
- **[📊 Examples](docs/examples.md)** - Code examples and tutorials
- **[🧪 Testing](docs/testing.md)** - Test suite documentation

## 🌟 Key Features

### 🚀 **Performance Optimizations**
- **TensorRT Integration**: 2-5x GPU speedup with automatic optimization
- **ONNX Runtime**: Cross-platform optimization with 1.5-3x performance gains  
- **Dynamic Quantization**: 2-4x memory reduction with minimal accuracy loss
- **JIT Compilation**: PyTorch native optimization with 20-50% speedup
- **CUDA Graphs**: Advanced GPU optimization for consistent low latency
- **Memory Pooling**: 30-50% memory usage reduction

### ⚡ **Production-Ready Features**
- **Async Processing**: High-throughput async inference with dynamic batching
- **FastAPI Integration**: Production-ready REST API with automatic documentation
- **Performance Monitoring**: Real-time metrics and profiling capabilities
- **Multi-Framework Support**: PyTorch, ONNX, TensorRT, HuggingFace models
- **Device Auto-Detection**: Automatic GPU/CPU optimization selection
- **Graceful Fallbacks**: Robust error handling with optimization fallbacks

### 🔧 **Developer Experience**
- **Modern Package Manager**: Powered by `uv` for 10-100x faster dependency resolution
- **Comprehensive Documentation**: Detailed guides, examples, and API reference
- **Type Safety**: Full type annotations with mypy validation
- **Code Quality**: Black formatting, Ruff linting, pre-commit hooks
- **Testing Suite**: Comprehensive unit tests with pytest
- **Docker Support**: Production-ready containerization

## ⚡ Quick Start

### Installation
```bash
# Install uv package manager
pip install uv

# Clone and setup the framework
git clone https://github.com/Evintkoo/torch-inference.git
cd torch-inference

# Run automated setup
uv sync && uv run python test_installation.py
```

### Basic Usage
```python
from framework import create_pytorch_framework

# Initialize framework with automatic optimization
framework = create_pytorch_framework(
    model_path="path/to/your/model.pt",
    device="cuda" if torch.cuda.is_available() else "cpu",
    enable_optimization=True  # Automatic TensorRT/ONNX optimization
)

# Single prediction
result = framework.predict(input_data)
print(f"Prediction: {result}")
```

### Async High-Performance Processing
```python
import asyncio
from framework import create_async_framework

async def async_example():
    framework = await create_async_framework(
        model_path="path/to/your/model.pt",
        batch_size=16,              # Dynamic batching
        enable_tensorrt=True        # TensorRT optimization
    )
    
    # Concurrent predictions
    tasks = [framework.predict_async(data) for data in batch_inputs]
    results = await asyncio.gather(*tasks)
    
    await framework.close()

asyncio.run(async_example())
```

## 🎯 Use Cases

- **🖼️ Image Classification**: High-performance image inference with CNNs
- **📝 Text Processing**: NLP models with BERT, GPT, and transformers
- **🔍 Object Detection**: Real-time object detection with YOLO, R-CNN
- **🌐 Production APIs**: REST APIs with FastAPI integration
- **📊 Batch Processing**: Large-scale batch inference workloads
- **⚡ Real-time Systems**: Low-latency real-time inference

## 📊 Performance Benchmarks

| Model Type | Baseline | Optimized | Speedup | Memory Saved |
|------------|----------|-----------|---------|--------------|
| **ResNet-50** | 100ms | **20ms** | **5x** | 81% |
| **BERT-Base** | 50ms | **12ms** | **4.2x** | 75% |
| **YOLOv8** | 80ms | **18ms** | **4.4x** | 71% |

*See [benchmarks documentation](docs/benchmarks.md) for detailed performance analysis.*

---

## 🛠️ Optimization Techniques

### 1. TensorRT Optimization (Recommended for NVIDIA GPUs)

```python
from framework.optimizers import TensorRTOptimizer

# Create TensorRT optimizer
trt_optimizer = TensorRTOptimizer(
    precision="fp16",        # fp32, fp16, or int8
    max_batch_size=32,       # Maximum batch size
    workspace_size=1 << 30   # 1GB workspace
)

# Optimize model
optimized_model = trt_optimizer.optimize_model(model, example_inputs)

# Benchmark optimization
benchmark = trt_optimizer.benchmark_optimization(model, optimized_model, inputs)
print(f"TensorRT speedup: {benchmark['speedup']:.2f}x")
```

**Expected Results:**
- 2-5x speedup on modern GPUs (RTX 30/40 series, A100, H100)
- 50-80% memory reduction with INT8 quantization
- Best for inference-only workloads

### 2. ONNX Runtime Optimization

```python
from framework.optimizers import ONNXOptimizer

# Export and optimize with ONNX
onnx_optimizer = ONNXOptimizer(
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    optimization_level='all'
)

optimized_model = onnx_optimizer.optimize_model(model, example_inputs)
```

**Expected Results:**
- 1.5-3x speedup on CPU, 1.2-2x on GPU
- Better cross-platform compatibility
- Excellent for edge deployment

### 3. Dynamic Quantization

```python
from framework.optimizers import QuantizationOptimizer

# Dynamic quantization (easiest setup)
quantized_model = QuantizationOptimizer.quantize_dynamic(
    model, dtype=torch.qint8
)

# Static quantization (better performance)
quantized_model = QuantizationOptimizer.quantize_static(
    model, calibration_dataloader
)
```

**Expected Results:**
- 2-4x speedup on CPU
- 50-75% memory reduction
- <1% typical accuracy loss

### 4. Complete Optimization Pipeline

```python
from framework.core.optimized_model import create_optimized_model

# Automatic optimization selection
config = InferenceConfig()
config.optimization.auto_optimize = True     # Automatic optimization
config.optimization.benchmark_all = True    # Benchmark all methods
config.optimization.select_best = True      # Auto-select best performer

## 🐳 Docker Deployment

### Quick Setup
```bash
# Build and run with GPU support
docker build -t torch-inference .
docker run --gpus all -p 8000:8000 torch-inference

# Or use docker compose
docker compose up --build
```

See [Deployment Guide](docs/deployment.md) for production deployment.

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=framework --cov-report=html
```

See [Testing Documentation](docs/testing.md) for comprehensive test information.

## � More Documentation

- **[🏗️ Framework Architecture](docs/framework.md)** - Core framework concepts
- **[🔧 Optimization Guide](docs/optimization.md)** - Performance optimization
- **[🚀 Deployment Guide](docs/deployment.md)** - Production deployment  
- **[📊 Monitoring Guide](docs/monitoring.md)** - Performance monitoring
- **[🔒 Security Guide](docs/security.md)** - Security features
- **[� API Reference](docs/api.md)** - Complete API documentation
- **[🚨 Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

## 🤝 Contributing

We welcome contributions! See the [Contributing Guide](docs/contributing.md) for development setup and guidelines.

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/Evintkoo/torch-inference/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Evintkoo/torch-inference/discussions)
- 📧 **Email**: [support@torch-inference.dev](mailto:support@torch-inference.dev)

---

<div align="center">

**⭐ Star this repository if it helped you!**

*Built with ❤️ for the PyTorch community*

</div>
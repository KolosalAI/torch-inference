# 📚 PyTorch Inference Framework Documentation

Welcome to the comprehensive documentation for the PyTorch Inference Framework - a production-ready inference solution with advanced optimization capabilities.

## 🚀 Quick Navigation

### 🏁 Getting Started
- **[Installation Guide](installation.md)** - Complete setup instructions
- **[Quick Start](quickstart.md)** - Basic usage examples
- **[Configuration](configuration.md)** - Configuration management

### 🔧 Core Features  
- **[Framework Overview](framework.md)** - Architecture and components
- **[Model Management](models.md)** - Loading and managing models
- **[Model Download Guide](model_download.md)** - Download models from various sources
- **[Inference Engine](inference.md)** - Sync and async inference
- **[Optimization](optimization.md)** - TensorRT, ONNX, quantization
- **[Audio Processing](audio.md)** - Text-to-Speech and Speech-to-Text
- **[TTS Models Guide](tts_models.md)** - Comprehensive Text-to-Speech models

### 🏭 Production Use
- **[Deployment](deployment.md)** - Docker and production deployment
- **[Monitoring](monitoring.md)** - Performance monitoring and metrics
- **[Security](security.md)** - Security features and best practices

### 🧪 Development
- **[API Reference](api.md)** - Complete API documentation
- **[Testing](testing.md)** - Test suite and guidelines
- **[Examples](examples.md)** - Code examples and tutorials
- **[Contributing](contributing.md)** - Development guidelines

### 📊 Performance
- **[Benchmarks](benchmarks.md)** - Performance comparisons
- **[Optimization Guide](optimization-guide.md)** - Advanced performance tuning
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## 🌟 Key Features

### ⚡ Performance Optimizations
- **2-10x speedup** with TensorRT, ONNX Runtime, and quantization
- **CUDA graphs** for consistent low latency
- **Memory pooling** for 30-50% memory reduction
- **JIT compilation** with 20-50% performance gains

### 🚀 Production Ready
- **Async processing** with dynamic batching
- **FastAPI integration** with automatic documentation  
- **Real-time monitoring** and profiling
- **Multi-framework support** (PyTorch, ONNX, TensorRT)

### 🔧 Developer Experience
- **Modern package management** with `uv`
- **Type safety** with full annotations
- **Comprehensive testing** with pytest
- **Docker support** for easy deployment

## 📖 Documentation Structure

```
docs/
├── README.md              # This overview
├── installation.md        # Setup and installation
├── quickstart.md         # Getting started guide
├── configuration.md      # Configuration management
├── framework.md          # Core framework concepts
├── models.md             # Model management
├── model_download.md     # Model downloading and sources
├── inference.md          # Inference capabilities
├── optimization.md       # Performance optimization
├── audio.md              # Audio processing features
├── tts_models.md         # Text-to-Speech models guide
├── deployment.md         # Production deployment
├── monitoring.md         # Monitoring and metrics
├── security.md           # Security features
├── api.md                # API reference
├── testing.md            # Testing documentation
├── examples.md           # Examples and tutorials
├── benchmarks.md         # Performance benchmarks
├── optimization-guide.md # Advanced optimization
├── troubleshooting.md    # Common issues
└── contributing.md       # Development guidelines
```

## 🎯 Use Cases

### 🖼️ Image Classification
```python
from framework import create_pytorch_framework

framework = create_pytorch_framework(
    model_path="models/resnet50.pt",
    enable_optimization=True
)

result = framework.predict(image_tensor)
```

### 📝 Text Processing
```python
from framework import create_async_framework

framework = await create_async_framework(
    model_path="models/bert.pt",
    batch_size=16
)

results = await framework.predict_batch(text_samples)
```

### 🚀 High-Performance API
```python
from fastapi import FastAPI
from framework import create_optimized_framework

framework = create_optimized_framework(
    optimization_level="aggressive"
)

app = FastAPI()

@app.post("/predict")
async def predict(data: InputData):
    return await framework.predict_async(data)
```

## 🔗 External Resources

- **[GitHub Repository](https://github.com/Evintkoo/torch-inference)**
- **[PyPI Package](https://pypi.org/project/torch-inference/)**
- **[Docker Images](https://hub.docker.com/r/evintkoo/torch-inference)**
- **[Documentation Site](https://evintkoo.github.io/torch-inference/)**

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/Evintkoo/torch-inference/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Evintkoo/torch-inference/discussions)
- 📧 **Email**: [support@torch-inference.dev](mailto:support@torch-inference.dev)

---

*Built with ❤️ for the PyTorch community*

# 📚 Examples Directory

This directory contains comprehensive examples for using the PyTorch Inference Framework.

## Available Examples

### Core Examples
- **`basic_usage.py`** - Simple synchronous inference patterns
- **`async_processing.py`** - High-throughput async inference with batching  
- **`fastapi_server.py`** - Production REST API integration
- **`custom_models.py`** - Integrating your own PyTorch models

### Optimization Examples
- **`tensorrt_optimization.py`** - TensorRT optimization examples
- **`onnx_optimization.py`** - ONNX Runtime optimization
- **`quantization_examples.py`** - Model quantization techniques
- **`performance_tuning.py`** - Advanced performance optimization

### Deployment Examples
- **`docker_deployment.py`** - Docker containerization
- **`kubernetes_deployment.yaml`** - Kubernetes deployment manifests
- **`monitoring_setup.py`** - Production monitoring configuration

## Quick Start

Run any example with:
```bash
uv run python examples/basic_usage.py
```

For examples requiring specific models, download test models:
```bash
uv run python examples/download_test_models.py
```

## Example Categories

### 🚀 **Performance-Focused**
Examples demonstrating different optimization techniques and their performance characteristics.

### 🏭 **Production-Ready**  
Real-world deployment patterns with proper error handling, monitoring, and scaling.

### 🔧 **Development**
Development workflow examples including debugging, profiling, and testing patterns.

See the main [README.md](../README.md) for detailed documentation and API reference.

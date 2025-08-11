# 🚀 Quick Start Guide

Get up and running with the PyTorch Inference Framework in minutes. This guide covers basic usage patterns and common scenarios.

## 🎯 Prerequisites

Before starting, ensure you have:
- Python 3.10+ installed
- Basic familiarity with PyTorch
- A PyTorch model (we'll help you create one if needed)

## 📦 Installation

### Quick Installation
```bash
# Install uv package manager
pip install uv

# Clone and setup the framework
git clone https://github.com/Evintkoo/torch-inference.git
cd torch-inference

# Run automated setup
uv sync && uv run python test_installation.py
```

For detailed installation instructions, see the [Installation Guide](installation.md).

## 🏁 Your First Inference

### 1. Basic Synchronous Inference

```python
from framework import create_pytorch_framework
import torch

# Create a simple test model (or use your own)
model = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 5)
)

# Save the model
torch.save(model.state_dict(), "simple_model.pt")

# Initialize framework
framework = create_pytorch_framework(
    model_path="simple_model.pt",
    device="cpu"  # or "cuda" if you have GPU
)

# Run inference
input_data = torch.randn(1, 10)
result = framework.predict(input_data)
print(f"Prediction shape: {result.shape}")
print(f"Prediction: {result}")
```

### 2. Async High-Performance Inference

```python
import asyncio
from framework import create_async_framework

async def async_inference_example():
    # Initialize async framework
    framework = await create_async_framework(
        model_path="simple_model.pt",
        batch_size=4,           # Enable batching
        max_batch_delay=0.05   # 50ms max batching delay
    )
    
    # Single async prediction
    input_data = torch.randn(1, 10)
    result = await framework.predict_async(input_data)
    print(f"Async result: {result.shape}")
    
    # Batch prediction
    batch_inputs = [torch.randn(1, 10) for _ in range(8)]
    batch_results = await framework.predict_batch_async(batch_inputs)
    print(f"Batch results: {len(batch_results)} predictions")
    
    # Concurrent predictions (automatically batched)
    concurrent_inputs = [torch.randn(1, 10) for _ in range(10)]
    tasks = [framework.predict_async(inp) for inp in concurrent_inputs]
    concurrent_results = await asyncio.gather(*tasks)
    print(f"Concurrent results: {len(concurrent_results)} predictions")
    
    await framework.close()

# Run async example
asyncio.run(async_inference_example())
```

### 3. Optimized Inference (Automatic)

```python
from framework import create_optimized_framework

# Framework automatically selects best optimizations
framework = create_optimized_framework(
    model_path="simple_model.pt",
    optimization_level="aggressive"  # auto, balanced, or aggressive
)

# The framework will:
# - Auto-detect available optimizations (TensorRT, ONNX, etc.)
# - Benchmark different optimization methods
# - Select the fastest configuration
# - Provide fallbacks if optimizations fail

input_data = torch.randn(4, 10)  # Batch input
result = framework.predict(input_data)

# Get optimization report
report = framework.get_optimization_report()
print(f"Selected optimization: {report['best_optimization']}")
print(f"Performance improvement: {report['speedup']:.1f}x")
print(f"Memory reduction: {report['memory_reduction']:.1%}")
```

## 🖼️ Image Classification Example

### Working with Real Models

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
from framework import create_pytorch_framework

# Load a pre-trained model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# Save for framework usage
torch.save(model.state_dict(), "resnet18.pt")

# Initialize framework with optimization
framework = create_pytorch_framework(
    model_path="resnet18.pt",
    device="cuda" if torch.cuda.is_available() else "cpu",
    enable_optimization=True  # Enable automatic optimization
)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
image = Image.open("path/to/your/image.jpg")
input_tensor = transform(image).unsqueeze(0)

# Run inference
with torch.no_grad():
    prediction = framework.predict(input_tensor)
    probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
    
# Get top 5 predictions
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(f"Class {top5_catid[i]}: {top5_prob[i]:.4f}")
```

## 🌐 REST API Server

### FastAPI Integration

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from framework import create_optimized_framework
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

# Initialize optimized framework
framework = create_optimized_framework(
    model_path="resnet18.pt",
    optimization_level="balanced"
)

# Create FastAPI app
app = FastAPI(
    title="PyTorch Inference API",
    description="High-performance image classification API",
    version="1.0.0"
)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Classify an uploaded image"""
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        # Run inference
        prediction = await framework.predict_async(input_tensor)
        probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
        
        # Get top prediction
        top_prob, top_class = torch.max(probabilities, 0)
        
        return {
            "predicted_class": int(top_class.item()),
            "confidence": float(top_prob.item()),
            "filename": file.filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stats = await framework.get_health_status()
    return {
        "status": "healthy",
        "model_loaded": stats["model_loaded"],
        "predictions_served": stats["prediction_count"],
        "average_latency_ms": stats["avg_latency_ms"]
    }

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "PyTorch Inference API",
        "version": "1.0.0",
        "optimization": framework.get_optimization_info(),
        "endpoints": {
            "POST /predict": "Upload image for classification",
            "GET /health": "API health status",
            "GET /docs": "Interactive API documentation"
        }
    }

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

### Running the API Server

```bash
# Install FastAPI dependencies
uv add fastapi uvicorn python-multipart

# Run the server
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Test the API
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/image.jpg"

# View interactive docs
open http://localhost:8000/docs
```

## 🔧 Configuration Basics

### Environment Variables (.env file)

Create a `.env` file in your project root:

```bash
# Device Configuration
DEVICE=cuda              # auto, cpu, cuda, mps
USE_FP16=true           # Enable half precision for speed

# Performance Settings
BATCH_SIZE=8            # Default batch size
MAX_BATCH_SIZE=32       # Maximum batch size for batching
WARMUP_ITERATIONS=5     # Model warmup iterations

# Optimization Settings
ENABLE_TENSORRT=true    # Enable TensorRT (requires NVIDIA GPU)
ENABLE_QUANTIZATION=true # Enable quantization
ENABLE_JIT=true         # Enable JIT compilation

# Server Settings
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
```

### YAML Configuration (config.yaml)

```yaml
device:
  type: "cuda"
  use_fp16: true
  memory_fraction: 0.8

batch:
  batch_size: 8
  max_batch_size: 32
  adaptive_batching: true
  timeout_seconds: 0.1

optimization:
  enable_tensorrt: true
  enable_quantization: true
  enable_jit: true
  optimization_level: "balanced"

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "INFO"
```

### Using Configuration

```python
from framework.core.config_manager import get_config_manager
from framework import TorchInferenceFramework

# Load configuration
config_manager = get_config_manager()
inference_config = config_manager.get_inference_config()

# Create framework with configuration
framework = TorchInferenceFramework(config=inference_config)
framework.load_model("path/to/model.pt", "my_model")

# Configuration is automatically applied
result = framework.predict(input_data)
```

## ⚡ Performance Optimization

### Quick Performance Boost

```python
from framework import create_pytorch_framework

# Automatic optimization (easiest)
framework = create_pytorch_framework(
    model_path="your_model.pt",
    device="cuda",              # Use GPU
    enable_optimization=True    # Auto-optimize
)

# Manual optimization control
from framework.core.config import InferenceConfig, OptimizationConfig

config = InferenceConfig(
    model_path="your_model.pt",
    optimization=OptimizationConfig(
        enable_tensorrt=True,      # 2-5x GPU speedup
        enable_quantization=True,  # 2x memory reduction
        enable_jit=True,          # 20-50% speedup
        enable_cuda_graphs=True   # Consistent low latency
    )
)

framework = TorchInferenceFramework(config=config)
```

### Benchmark Your Performance

```python
import time
from framework import create_optimized_framework

# Create optimized framework
framework = create_optimized_framework(
    model_path="your_model.pt",
    optimization_level="aggressive"
)

# Benchmark inference
test_input = torch.randn(16, 3, 224, 224)  # Batch of 16 images

# Warmup
for _ in range(10):
    _ = framework.predict(test_input)

# Benchmark
num_runs = 100
start_time = time.time()
for _ in range(num_runs):
    result = framework.predict(test_input)
end_time = time.time()

# Calculate metrics
total_time = end_time - start_time
avg_latency = (total_time / num_runs) * 1000  # ms
throughput = (num_runs * test_input.shape[0]) / total_time  # samples/sec

print(f"Average latency: {avg_latency:.1f}ms")
print(f"Throughput: {throughput:.1f} samples/sec")
print(f"Batch size: {test_input.shape[0]}")

# Get optimization report
report = framework.get_optimization_report()
print(f"Optimization: {report['best_optimization']}")
print(f"Speedup: {report['speedup']:.1f}x")
```

## 🐳 Docker Deployment

### Quick Docker Setup

```bash
# Build container
docker build -t my-inference-api .

# Run with GPU support
docker run --gpus all -p 8000:8000 \
    -e DEVICE=cuda \
    -e ENABLE_TENSORRT=true \
    my-inference-api

# Or use docker-compose
docker-compose up --build
```

### Docker Compose (docker-compose.yml)

```yaml
version: '3.8'

services:
  inference-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEVICE=cuda
      - BATCH_SIZE=16
      - ENABLE_TENSORRT=true
      - LOG_LEVEL=INFO
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/app/models
      - ./data:/app/data
```

## 🔍 Monitoring and Debugging

### Built-in Monitoring

```python
from framework import create_monitored_framework

# Framework with monitoring
framework = create_monitored_framework(
    model_path="your_model.pt",
    enable_detailed_metrics=True
)

# Run inference
result = framework.predict(input_data)

# Get performance metrics
metrics = framework.get_metrics()
print(f"Latency: {metrics['latency']['avg_ms']:.1f}ms")
print(f"Throughput: {metrics['throughput']['requests_per_second']:.1f} req/s")
print(f"Memory usage: {metrics['memory']['gpu_used_gb']:.1f}GB")

# Health check
health = framework.get_health_status()
print(f"Status: {health['status']}")
print(f"Predictions served: {health['prediction_count']}")
```

### Debug Mode

```python
from framework import create_pytorch_framework
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Framework with debug info
framework = create_pytorch_framework(
    model_path="your_model.pt",
    debug=True,                 # Enable debug mode
    enable_profiling=True       # Enable profiling
)

# Detailed prediction info
result = framework.predict_with_info(input_data)
print(f"Prediction: {result['prediction']}")
print(f"Latency: {result['latency_ms']:.1f}ms")
print(f"Memory used: {result['memory_mb']:.1f}MB")
print(f"Optimization: {result['optimization_used']}")
```

## 🚨 Common Issues and Solutions

### Issue: CUDA out of memory
```python
# Solution: Reduce batch size or use CPU
framework = create_pytorch_framework(
    model_path="your_model.pt",
    device="cpu",  # Use CPU instead
    # Or reduce batch size
    batch_size=4   # Instead of 16
)
```

### Issue: Slow first inference
```python
# Solution: Enable warmup
framework = create_pytorch_framework(
    model_path="your_model.pt",
    warmup_iterations=10  # Warmup model
)
```

### Issue: TensorRT optimization fails
```python
# Solution: Use fallback optimization
from framework.core.config import InferenceConfig, OptimizationConfig

config = InferenceConfig(
    optimization=OptimizationConfig(
        enable_tensorrt=False,    # Disable TensorRT
        enable_quantization=True, # Use quantization instead
        enable_jit=True          # Enable JIT compilation
    )
)
```

## 📚 Next Steps

Now that you've got the basics, explore more advanced features:

1. **[Configuration Guide](configuration.md)** - Advanced configuration options
2. **[Optimization Guide](optimization.md)** - Detailed performance tuning
3. **[API Reference](api.md)** - Complete API documentation
4. **[Examples](examples.md)** - More complex use cases
5. **[Deployment Guide](deployment.md)** - Production deployment

## 🆘 Getting Help

- 📖 **Documentation**: Full documentation at [docs/](.)
- 🐛 **Issues**: Report issues on [GitHub](https://github.com/Evintkoo/torch-inference/issues)
- 💬 **Discussions**: Ask questions on [GitHub Discussions](https://github.com/Evintkoo/torch-inference/discussions)
- 📧 **Email**: Contact us at [support@torch-inference.dev](mailto:support@torch-inference.dev)

---

*Ready for production? Check out the [Deployment Guide](deployment.md) for scaling and production best practices.*

# Quick Start Guide

Get the PyTorch Inference Framework running in just 5 minutes! This guide will walk you through installation, basic setup, and your first prediction.

## 🚀 Prerequisites

- **Python 3.10+** (3.11+ recommended)
- **PyTorch 2.0+** 
- **CUDA 12.0+** (optional, for GPU acceleration)
- **8GB RAM minimum** (16GB+ recommended)

## ⚡ 1-Minute Installation

### Option A: Quick Setup with uv (Recommended)

```bash
# Install uv package manager (10-100x faster than pip)
pip install uv

# Clone the repository
git clone https://github.com/Evintkoo/torch-inference.git
cd torch-inference

# One-command setup (installs dependencies and validates installation)
uv sync && uv run python -c "print('✅ PyTorch Inference Framework ready!')"
```

### Option B: Traditional pip Installation

```bash
# Clone and install
git clone https://github.com/Evintkoo/torch-inference.git
cd torch-inference

# Install dependencies
pip install -r requirements.txt

# Validate installation
python -c "import framework; print('✅ Framework installed!')"
```

## 🎯 First Inference (2 minutes)

### Option 1: Using the Python API

Create a file called `quick_test.py`:

```python
# quick_test.py
import asyncio
from framework import TorchInferenceFramework

async def main():
    # Initialize framework
    framework = TorchInferenceFramework()
    
    # Download and load a pre-trained model (ResNet-18)
    framework.download_and_load_model(
        source="torchvision",
        model_id="resnet18",
        model_name="resnet18",
        pretrained=True
    )
    
    # Start the async engine
    async with framework.async_context():
        # Make a prediction (dummy input for demo)
        import torch
        dummy_image = torch.randn(1, 3, 224, 224)
        
        result = await framework.predict_async(dummy_image)
        print(f"🎉 Prediction result: {result}")
        print(f"📊 Model info: {framework.get_model_info()}")

# Run the example
asyncio.run(main())
```

Run it:
```bash
uv run python quick_test.py
# or: python quick_test.py
```

### Option 2: Using the REST API

Start the server:
```bash
# Start the API server
uv run python main.py
# or: python main.py
```

In another terminal, test the API:
```bash
# Test with curl
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"inputs": [1, 2, 3, 4, 5]}'

# Expected response:
# {
#   "success": true,
#   "result": 0.75,
#   "processing_time": 0.025,
#   "model_info": {...}
# }
```

## 🎵 TTS Quick Start (3 minutes)

### Download and Use a TTS Model

```python
# tts_demo.py
import asyncio
import base64
from framework import TorchInferenceFramework

async def tts_demo():
    framework = TorchInferenceFramework()
    
    # Download SpeechT5 TTS model
    framework.download_and_load_model(
        source="huggingface",
        model_id="microsoft/speecht5_tts", 
        model_name="speecht5_tts"
    )
    
    # Generate speech
    text = "Hello! This is the PyTorch Inference Framework."
    result = framework.predict(text)
    
    # Save audio (result contains base64 encoded audio)
    if result.get("success"):
        audio_data = base64.b64decode(result["audio_data"])
        with open("output.wav", "wb") as f:
            f.write(audio_data)
        print("🎵 Audio saved to output.wav")
        print(f"⏱️  Processing time: {result['processing_time']:.2f}s")

asyncio.run(tts_demo())
```

### Using TTS via REST API

```bash
# Start server (if not already running)
uv run python main.py &

# Synthesize speech
curl -X POST "http://localhost:8000/tts/synthesize" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Hello from PyTorch Inference Framework!",
       "model_name": "speecht5_tts",
       "speed": 1.0,
       "language": "en"
     }'
```

## 📊 Performance Test (1 minute)

Test the framework performance:

```python
# benchmark.py
from framework import TorchInferenceFramework
import torch
import time

# Initialize framework
framework = TorchInferenceFramework()

# Load a model (example model for demo)
framework.load_model("dummy_path")  # Uses built-in example model

# Create test input
test_input = [1, 2, 3, 4, 5]

# Run benchmark
results = framework.benchmark(test_input, iterations=100, warmup=10)

print("🏆 Performance Results:")
print(f"   Average time: {results['mean_time_ms']:.2f}ms")
print(f"   Throughput: {results['throughput_fps']:.1f} FPS")
print(f"   Device: {results['device']}")
```

## 🔧 GPU Acceleration Setup

### Check GPU Availability

```python
# gpu_check.py
import torch
from framework.core.gpu_manager import GPUManager

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name()}")

# Use framework GPU manager
gpu_manager = GPUManager()
gpus, config = gpu_manager.detect_and_configure()

print(f"Detected {len(gpus)} suitable GPU(s)")
print(f"Recommended config: {config.device_type} with FP16: {config.use_fp16}")
```

### Enable GPU Optimizations

```python
# gpu_optimized.py
from framework import TorchInferenceFramework
from framework.core.config import InferenceConfig, DeviceConfig, DeviceType

# Create GPU-optimized configuration
config = InferenceConfig()
config.device = DeviceConfig(
    device_type=DeviceType.CUDA,
    device_id=0,
    use_fp16=True,           # Half precision for speed
    use_tensorrt=True,       # TensorRT optimization
    use_torch_compile=True   # PyTorch 2.0 compilation
)

# Initialize with optimized config
framework = TorchInferenceFramework(config)
framework.load_model("dummy_path")

# Apply automatic optimizations
optimizations = framework.apply_automatic_optimizations(aggressive=True)
print(f"Applied optimizations: {optimizations}")

# Test performance
results = framework.benchmark([1,2,3,4,5], iterations=50)
print(f"Optimized performance: {results['throughput_fps']:.1f} FPS")
```

## 📚 Next Steps

### Essential Guides
1. **[Complete Installation Guide](installation.md)** - Detailed setup including Docker
2. **[Configuration Guide](configuration.md)** - Customize your setup
3. **[Model Loading Guide](model-loading.md)** - Load different model types
4. **[Optimization Guide](optimization.md)** - Maximize performance

### Popular Tutorials
1. **[Image Classification Tutorial](../tutorials/basic-classification.md)**
2. **[Text Processing with BERT](../tutorials/bert-classification.md)**
3. **[Audio Processing Tutorial](../tutorials/tts-synthesis.md)**

### Advanced Features
1. **[Autoscaling Setup](autoscaling.md)** - Dynamic scaling
2. **[Production Deployment](../tutorials/production-api.md)** - Deploy to production
3. **[Custom Models](../tutorials/custom-models.md)** - Integrate your models

## 🆘 Troubleshooting

### Common Issues

**1. Import Error: "No module named 'framework'"**
```bash
# Make sure you're in the project directory
cd torch-inference

# Verify installation
uv run python -c "import framework; print('OK')"
```

**2. CUDA Out of Memory**
```python
# Reduce batch size in config
config.batch.batch_size = 1
config.batch.max_batch_size = 4
```

**3. Model Download Fails**
```bash
# Check internet connection and retry
# Use smaller models for testing:
framework.download_and_load_model("torchvision", "mobilenet_v2", "mobile_net")
```

**4. TTS Audio Generation Fails**
```bash
# Install audio dependencies
pip install librosa soundfile torchaudio

# Test with simple text
text = "Hello"  # Keep it short for testing
```

### Getting Help

- **📖 Documentation**: [Full Documentation](../README.md)
- **🐛 Issues**: [GitHub Issues](https://github.com/Evintkoo/torch-inference/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/Evintkoo/torch-inference/discussions)
- **📧 Support**: support@torch-inference.dev

## ✅ Verification Checklist

After following this guide, you should be able to:

- [ ] ✅ Install the framework successfully
- [ ] ✅ Run a basic prediction
- [ ] ✅ Start the REST API server
- [ ] ✅ Make API requests
- [ ] ✅ Test TTS functionality (if needed)
- [ ] ✅ Check GPU acceleration (if available)
- [ ] ✅ Run performance benchmarks

## 🎉 Success!

Congratulations! You now have a working PyTorch Inference Framework setup. The framework provides:

- **High-performance inference** with automatic optimizations
- **Production-ready REST API** with comprehensive endpoints
- **Advanced features** like TTS, autoscaling, and GPU acceleration
- **Comprehensive monitoring** and logging

**Ready for more?** Check out the [tutorials](../tutorials/) for specific use cases, or dive into the [API reference](../api/) for detailed documentation.

---

*⭐ **Pro Tip**: Bookmark the [API documentation](../api/rest-api.md) - it contains all endpoints with examples!*

# Frequently Asked Questions (FAQ)

Common questions and solutions for the PyTorch Inference Framework.

## 🚀 Getting Started

### Q: How do I quickly get started with the framework?

**A:** Follow these steps:

1. **Install the framework:**
   ```bash
   pip install torch-inference-framework
   ```

2. **Basic setup:**
   ```python
   from framework import TorchInferenceFramework
   
   # Initialize with auto-optimization
   framework = TorchInferenceFramework(auto_optimize=True)
   
   # Load your model
   model = framework.load_model("path/to/your/model.pth", optimize=True)
   
   # Run inference
   result = framework.predict(model.model_id, your_input_data)
   ```

3. **Check the [Quick Start Guide](guides/quickstart.md) for detailed examples.**

### Q: What models are supported?

**A:** The framework supports:
- **PyTorch models** (.pth, .pt files)
- **ONNX models** (.onnx files) 
- **TensorRT engines** (.trt files)
- **Hugging Face models** (via transformers library)
- **Custom architectures** (with proper model loading)

See the [model compatibility guide](guides/model-compatibility.md) for details.

### Q: What are the system requirements?

**A:** 
- **Python:** 3.8+ (3.10+ recommended)
- **PyTorch:** 2.0+ (2.8+ recommended)
- **CUDA:** 11.8+ for GPU acceleration (optional)
- **Memory:** 8GB+ RAM (16GB+ recommended)
- **Storage:** 2GB+ free space for models and cache

Full requirements in [Installation Guide](guides/installation.md).

## 🔧 Configuration & Setup

### Q: How do I configure the framework for my specific hardware?

**A:** Use automatic detection or manual configuration:

```python
# Automatic (recommended)
framework = TorchInferenceFramework(auto_optimize=True)

# Manual GPU configuration
from framework.core.config import DeviceConfig
device_config = DeviceConfig(
    device_type="cuda",
    device_id=0,
    use_fp16=True,
    use_tensorrt=True
)
framework = TorchInferenceFramework(device_config=device_config)

# Manual CPU configuration
device_config = DeviceConfig(
    device_type="cpu",
    use_mkldnn=True,
    num_threads=8
)
```

See [Configuration Guide](guides/configuration.md) for all options.

### Q: How do I optimize for maximum performance?

**A:** Follow the optimization hierarchy:

1. **Enable auto-optimization:**
   ```python
   framework = TorchInferenceFramework(
       auto_optimize=True,
       optimization_level="aggressive"
   )
   ```

2. **Use appropriate precision:**
   - FP16 for modern GPUs
   - FP32 for older GPUs or CPU
   - INT8 for edge deployment

3. **Enable hardware-specific optimizations:**
   - TensorRT for NVIDIA GPUs
   - MKLDNN for Intel CPUs
   - ONNX Runtime for cross-platform

4. **Configure batching:**
   ```python
   framework.configure_dynamic_batching(
       max_batch_size=32,
       timeout_ms=50
   )
   ```

See [Optimization Guide](guides/optimization.md) for comprehensive strategies.

## 🏃‍♂️ Performance & Troubleshooting

### Q: My inference is slow. How can I improve performance?

**A:** Try these solutions in order:

1. **Check GPU utilization:**
   ```python
   gpu_info = framework.get_gpu_info()
   print(f"GPU utilization: {gpu_info['utilization_percent']}%")
   ```

2. **Enable optimizations:**
   ```python
   # Enable all optimizations
   model = framework.load_model("model.pth", optimize=True)
   
   # Or specific optimizations
   framework.optimize_model(model.model_id, ["torch_compile", "tensorrt", "fp16"])
   ```

3. **Increase batch size:**
   ```python
   # Test different batch sizes
   optimal_batch = framework.find_optimal_batch_size(model.model_id)
   ```

4. **Use memory pooling:**
   ```python
   framework.enable_memory_pooling(pool_size_gb=4)
   ```

5. **Profile your code:**
   ```python
   framework.enable_profiling()
   result = framework.predict(model.model_id, input_data)
   profiling_results = framework.get_profiling_results()
   ```

### Q: I'm getting CUDA out of memory errors. What should I do?

**A:** Follow these steps:

1. **Reduce batch size:**
   ```python
   framework.update_config({"batch": {"batch_size": 1}})
   ```

2. **Enable memory optimization:**
   ```python
   framework.configure_memory_optimization(
       enable_memory_pooling=True,
       auto_gc=True,
       cleanup_threshold=0.8
   )
   ```

3. **Use gradient checkpointing:**
   ```python
   framework.enable_gradient_checkpointing(model.model_id)
   ```

4. **Switch to CPU or reduce precision:**
   ```python
   # Use CPU
   framework.move_model_to_device(model.model_id, "cpu")
   
   # Or use FP16
   framework.set_model_precision(model.model_id, "fp16")
   ```

### Q: How do I monitor performance in production?

**A:** Enable comprehensive monitoring:

```python
# Enable monitoring
framework.enable_monitoring()

# Configure metrics collection
framework.configure_metrics(
    enable_prometheus=True,
    metrics_port=9090,
    enable_logging=True
)

# Get real-time metrics
metrics = framework.get_metrics()
print(f"Throughput: {metrics['throughput_rps']} RPS")
print(f"Average latency: {metrics['avg_latency_ms']} ms")
print(f"Error rate: {metrics['error_rate']}%")

# Set up alerts
framework.configure_alerts(
    webhook_url="https://your-webhook.com",
    thresholds={
        'latency_ms': 1000,
        'error_rate': 0.05,
        'memory_usage': 0.9
    }
)
```

## 🎵 Audio Processing

### Q: How do I use text-to-speech (TTS)?

**A:** Basic TTS usage:

```python
# Initialize framework with audio support
framework = TorchInferenceFramework(enable_audio=True)

# Generate speech
audio_result = framework.synthesize_speech(
    text="Hello, this is a test of text-to-speech!",
    model="speecht5_tts",
    voice="default",
    language="en"
)

# Save audio file
framework.save_audio(
    audio_data=audio_result['audio_data'],
    file_path="output.wav",
    sample_rate=audio_result['sample_rate']
)
```

Available TTS models:
- `speecht5_tts` - Microsoft SpeechT5 (default)
- `bark_tts` - Suno Bark (voice cloning support)
- `tacotron2` - NVIDIA Tacotron2
- `fastpitch` - NVIDIA FastPitch

### Q: How do I use speech-to-text (STT)?

**A:** Basic STT usage:

```python
# Transcribe audio file
result = framework.transcribe_audio(
    audio_path="speech.wav",
    model="whisper-base",
    language="auto"
)

print(f"Transcription: {result['text']}")
print(f"Confidence: {result['confidence']}")
print(f"Language: {result['language']}")

# Real-time transcription
transcription_stream = framework.create_transcription_stream(
    model="whisper-base",
    language="en"
)

# Feed audio chunks
for audio_chunk in audio_stream:
    partial_result = transcription_stream.process_chunk(audio_chunk)
    print(f"Partial: {partial_result['text']}")
```

Available STT models:
- `whisper-tiny` - Fastest, basic accuracy
- `whisper-base` - Good balance (default)
- `whisper-small` - Better accuracy
- `whisper-medium` - High accuracy
- `whisper-large` - Best accuracy

### Q: Can I use custom voices for TTS?

**A:** Yes, with voice cloning:

```python
# Clone voice from reference audio
cloned_voice = framework.clone_voice(
    reference_audio_path="reference_speaker.wav",
    voice_name="custom_voice_1",
    model="bark_tts"  # Supports voice cloning
)

# Use cloned voice
audio_result = framework.synthesize_speech(
    text="This is my cloned voice speaking!",
    model="bark_tts",
    voice=cloned_voice['voice_id']
)
```

## 🌐 API & Deployment

### Q: How do I deploy the framework as a REST API?

**A:** Use the built-in FastAPI server:

```python
# server.py
from framework.api import create_app
from framework import TorchInferenceFramework

# Initialize framework
framework = TorchInferenceFramework(auto_optimize=True)

# Load your models
framework.load_model("models/classifier.pth", model_id="classifier")
framework.load_model("models/tts_model.pth", model_id="tts")

# Create API app
app = create_app(framework)

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Then start the server:
```bash
python server.py
```

The API will be available at `http://localhost:8000` with automatic documentation at `/docs`.

### Q: What API endpoints are available?

**A:** The framework provides 30+ REST API endpoints:

**Core Inference:**
- `POST /predict` - General model inference
- `POST /predict/batch` - Batch inference
- `POST /predict/async` - Async inference

**Audio Processing:**
- `POST /tts/synthesize` - Text-to-speech
- `POST /transcribe` - Speech-to-text
- `POST /audio/enhance` - Audio enhancement

**Model Management:**
- `GET /models` - List loaded models
- `POST /models/load` - Load new model
- `DELETE /models/{model_id}` - Unload model

**System Monitoring:**
- `GET /health` - Health check
- `GET /metrics` - Performance metrics
- `GET /gpu/info` - GPU information

See [REST API Reference](api/rest-api.md) for complete documentation.

### Q: How do I deploy with Docker?

**A:** Use the provided Docker setup:

1. **Build Docker image:**
   ```bash
   docker build -t torch-inference .
   ```

2. **Run container:**
   ```bash
   docker run -d \
     --name torch-inference \
     --gpus all \
     -p 8000:8000 \
     -v $(pwd)/models:/app/models \
     torch-inference
   ```

3. **Docker Compose for production:**
   ```yaml
   # docker-compose.yml
   version: '3.8'
   services:
     torch-inference:
       build: .
       ports:
         - "8000:8000"
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
       environment:
         - DEVICE_TYPE=cuda
         - OPTIMIZATION_LEVEL=aggressive
         - ENABLE_MONITORING=true
   ```

### Q: How do I handle authentication?

**A:** Configure API authentication:

```python
# Enable authentication
framework.configure_security(
    enable_auth=True,
    api_key="your-secret-api-key",
    jwt_secret="jwt-secret-key"
)

# Use API key in requests
import requests

headers = {
    "Authorization": "Bearer your-secret-api-key",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/predict",
    headers=headers,
    json={"model_id": "classifier", "input_data": [...]}
)
```

## 🐛 Common Issues & Solutions

### Q: ImportError: No module named 'framework'

**A:** Install the framework properly:

```bash
# Install from PyPI
pip install torch-inference-framework

# Or install from source
git clone https://github.com/your-org/torch-inference.git
cd torch-inference
pip install -e .
```

### Q: Model loading fails with "File not found"

**A:** Check file paths and permissions:

```python
import os

# Check if file exists
model_path = "path/to/model.pth"
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")

# Check file permissions
if not os.access(model_path, os.R_OK):
    print(f"No read permission: {model_path}")

# Use absolute paths
import os.path
abs_path = os.path.abspath(model_path)
model = framework.load_model(abs_path)
```

### Q: "CUDA driver version is insufficient" error

**A:** Update CUDA drivers:

1. **Check CUDA version:**
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. **Install compatible PyTorch:**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Fall back to CPU if needed:**
   ```python
   framework = TorchInferenceFramework(device_type="cpu")
   ```

### Q: High memory usage that doesn't decrease

**A:** Enable memory management:

```python
# Enable automatic garbage collection
framework.enable_auto_gc()

# Clear cache manually
framework.clear_cache()

# Use memory pooling
framework.configure_memory_optimization(
    enable_memory_pooling=True,
    auto_cleanup=True,
    cleanup_threshold=0.8
)

# Monitor memory usage
memory_stats = framework.get_memory_stats()
print(f"Memory usage: {memory_stats}")
```

### Q: Inconsistent inference results

**A:** Ensure deterministic behavior:

```python
import torch
import numpy as np
import random

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Use deterministic algorithms
torch.use_deterministic_algorithms(True)

# Disable CUDNN benchmarking for determinism
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Configure framework for determinism
framework.configure_deterministic_mode(True)
```

## 📚 Advanced Usage

### Q: How do I implement custom optimizations?

**A:** Create custom optimization plugins:

```python
from framework.core.optimization import OptimizationPlugin

class CustomOptimization(OptimizationPlugin):
    def __init__(self):
        super().__init__("custom_opt", "Custom Optimization")
    
    def can_optimize(self, model):
        # Check if model is compatible
        return True
    
    def optimize(self, model, config):
        # Implement your optimization
        optimized_model = self.apply_custom_optimization(model)
        return optimized_model

# Register custom optimization
framework.register_optimization_plugin(CustomOptimization())

# Use custom optimization
model = framework.load_model(
    "model.pth",
    optimizations=["custom_opt"]
)
```

### Q: How do I add custom models or architectures?

**A:** Implement custom model loaders:

```python
from framework.core.models import ModelLoader

class CustomModelLoader(ModelLoader):
    def __init__(self):
        super().__init__("custom_model", ["custom"])
    
    def can_load(self, model_path):
        return model_path.endswith('.custom')
    
    def load_model(self, model_path, config):
        # Implement your model loading logic
        model = self.load_custom_model(model_path)
        return self.wrap_model(model, config)

# Register custom loader
framework.register_model_loader(CustomModelLoader())

# Load custom model
model = framework.load_model("model.custom")
```

### Q: How do I implement custom preprocessing/postprocessing?

**A:** Use processing pipelines:

```python
from framework.core.processing import ProcessingPipeline

# Custom preprocessing
def custom_preprocess(input_data):
    # Your preprocessing logic
    processed = normalize_and_transform(input_data)
    return processed

# Custom postprocessing
def custom_postprocess(model_output):
    # Your postprocessing logic
    result = apply_custom_transforms(model_output)
    return result

# Create pipeline
pipeline = ProcessingPipeline(
    preprocessor=custom_preprocess,
    postprocessor=custom_postprocess
)

# Use with model
model = framework.load_model(
    "model.pth",
    processing_pipeline=pipeline
)
```

## 🔗 Integration & Compatibility

### Q: How do I integrate with existing ML pipelines?

**A:** The framework provides multiple integration options:

```python
# 1. As a library
from framework import TorchInferenceFramework
framework = TorchInferenceFramework()

# 2. As a service
import requests
response = requests.post("http://inference-service/predict", json=data)

# 3. With MLflow
import mlflow
model_uri = "models:/my-model/production"
model = framework.load_model(model_uri, source="mlflow")

# 4. With Kubernetes
# See deployment examples in docs/examples/kubernetes/
```

### Q: Can I use this with Jupyter notebooks?

**A:** Yes, the framework is Jupyter-friendly:

```python
# Enable Jupyter integration
framework.enable_jupyter_integration()

# Use interactive widgets
from framework.jupyter import InferenceWidget
widget = InferenceWidget(framework)
widget.display()

# Automatic progress bars
framework.enable_progress_bars()

# Rich output formatting
framework.enable_rich_output()
```

### Q: How do I contribute to the project?

**A:** We welcome contributions:

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes and add tests**
4. **Run the test suite:**
   ```bash
   python -m pytest tests/
   ```
5. **Submit a pull request**

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

## 📞 Support & Community

### Q: Where can I get help?

**A:** Multiple support channels:

- **Documentation:** [docs/](README.md)
- **GitHub Issues:** Report bugs and feature requests
- **Discussions:** Community Q&A and discussions
- **Discord:** Real-time chat support
- **Stack Overflow:** Tag questions with `torch-inference`

### Q: How do I report bugs?

**A:** Create a detailed bug report:

1. **Check existing issues** to avoid duplicates
2. **Include system information:**
   ```python
   print(framework.get_system_info())
   ```
3. **Provide minimal reproduction code**
4. **Include error messages and logs**
5. **Describe expected vs actual behavior**

### Q: How do I request features?

**A:** Feature requests are welcome:

1. **Search existing feature requests**
2. **Describe the use case and benefits**
3. **Provide implementation suggestions if possible**
4. **Consider contributing the feature yourself**

---

## 🏷️ Tags

`pytorch` `inference` `optimization` `tensorrt` `onnx` `gpu` `cpu` `performance` `api` `docker` `kubernetes` `tts` `stt` `audio` `ml` `ai` `deep-learning`

---

*Can't find your question? Check our [complete documentation](README.md) or [ask the community](https://github.com/your-org/torch-inference/discussions).*

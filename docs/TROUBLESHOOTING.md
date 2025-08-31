# Troubleshooting Guide

Comprehensive troubleshooting guide for common issues with the PyTorch Inference Framework.

## 🚨 Quick Diagnostics

First, run our built-in diagnostic tool:

```python
from framework.diagnostics import run_diagnostics

# Run comprehensive system check
diagnostic_results = run_diagnostics()

# Print summary
print("🔍 System Diagnostics:")
for category, results in diagnostic_results.items():
    status = "✅" if results['status'] == 'ok' else "❌"
    print(f"{status} {category}: {results['message']}")

# Get detailed report
detailed_report = diagnostic_results.get_detailed_report()
print(detailed_report)
```

## 🔧 Installation Issues

### Package Installation Failures

**Problem:** `pip install` fails with dependency errors

**Solutions:**

1. **Update pip and setuptools:**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

2. **Use clean environment:**
   ```bash
   # Create new virtual environment
   python -m venv torch_inference_env
   
   # Windows
   torch_inference_env\Scripts\activate
   
   # Linux/Mac
   source torch_inference_env/bin/activate
   
   # Install framework
   pip install torch-inference-framework
   ```

3. **Install dependencies manually:**
   ```bash
   # Install PyTorch first
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Then install framework
   pip install torch-inference-framework
   ```

4. **Build from source:**
   ```bash
   git clone https://github.com/your-org/torch-inference.git
   cd torch-inference
   pip install -e .
   ```

### CUDA/GPU Setup Issues

**Problem:** CUDA not detected or version mismatch

**Diagnostic commands:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

**Solutions:**

1. **Install compatible PyTorch:**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # For CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Update NVIDIA drivers:**
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install nvidia-driver-525
   
   # CentOS/RHEL
   sudo dnf install nvidia-driver cuda-toolkit
   ```

3. **Verify installation:**
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU count: {torch.cuda.device_count()}")
       print(f"Current GPU: {torch.cuda.get_device_name(0)}")
   ```

### Audio Dependencies Issues

**Problem:** Audio processing not working

**Solutions:**

1. **Install audio dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libsndfile1 ffmpeg
   
   # CentOS/RHEL
   sudo yum install libsndfile ffmpeg
   
   # macOS
   brew install libsndfile ffmpeg
   
   # Windows (using conda)
   conda install libsndfile ffmpeg -c conda-forge
   ```

2. **Install Python audio packages:**
   ```bash
   pip install soundfile librosa torchaudio
   ```

3. **Test audio functionality:**
   ```python
   try:
       import soundfile as sf
       import librosa
       import torchaudio
       print("✅ Audio dependencies installed correctly")
   except ImportError as e:
       print(f"❌ Missing audio dependency: {e}")
   ```

## 🏃‍♂️ Performance Issues

### Slow Inference Speed

**Problem:** Inference is slower than expected

**Diagnostic steps:**

1. **Check system utilization:**
   ```python
   # Get performance metrics
   metrics = framework.get_performance_metrics()
   print(f"GPU utilization: {metrics['gpu_utilization']}%")
   print(f"Memory usage: {metrics['memory_usage_percent']}%")
   print(f"CPU usage: {metrics['cpu_usage']}%")
   ```

2. **Profile inference:**
   ```python
   # Enable profiling
   framework.enable_profiling()
   
   # Run inference
   result = framework.predict(model_id, input_data)
   
   # Get profiling results
   profile = framework.get_profiling_results()
   print("Bottlenecks:")
   for step, time_ms in profile['step_times'].items():
       print(f"  {step}: {time_ms:.2f} ms")
   ```

**Solutions:**

1. **Enable optimizations:**
   ```python
   # Auto-optimize
   framework = TorchInferenceFramework(
       auto_optimize=True,
       optimization_level="aggressive"
   )
   
   # Or specific optimizations
   model = framework.load_model(
       "model.pth",
       optimize=True,
       optimizations=["torch_compile", "tensorrt", "fp16"]
   )
   ```

2. **Increase batch size:**
   ```python
   # Find optimal batch size
   optimal_batch = framework.find_optimal_batch_size(model_id)
   framework.set_default_batch_size(optimal_batch)
   ```

3. **Use appropriate device:**
   ```python
   # Force GPU usage
   framework.move_model_to_device(model_id, "cuda")
   
   # Or optimize for CPU
   framework.optimize_for_cpu(model_id)
   ```

4. **Enable dynamic batching:**
   ```python
   framework.configure_dynamic_batching(
       max_batch_size=32,
       timeout_ms=50
   )
   ```

### High Memory Usage

**Problem:** Excessive memory consumption

**Diagnostic:**
```python
# Monitor memory usage
memory_stats = framework.get_memory_stats()
print(f"Total memory: {memory_stats['total_gb']:.1f} GB")
print(f"Used memory: {memory_stats['used_gb']:.1f} GB")
print(f"Model memory: {memory_stats['model_memory_gb']:.1f} GB")
print(f"Cache memory: {memory_stats['cache_memory_gb']:.1f} GB")

# Get per-model memory usage
for model_id in framework.list_models():
    model_memory = framework.get_model_memory_usage(model_id)
    print(f"Model {model_id}: {model_memory:.1f} MB")
```

**Solutions:**

1. **Enable memory optimization:**
   ```python
   framework.configure_memory_optimization(
       enable_memory_pooling=True,
       pool_size_gb=4,
       auto_gc=True,
       cleanup_threshold=0.8
   )
   ```

2. **Reduce precision:**
   ```python
   # Use FP16 instead of FP32
   framework.set_model_precision(model_id, "fp16")
   
   # Or use quantization
   quantized_model = framework.quantize_model(model_id, "int8")
   ```

3. **Clear caches periodically:**
   ```python
   # Clear all caches
   framework.clear_cache()
   
   # Or configure automatic cleanup
   framework.configure_cache_cleanup(
       max_cache_size_gb=10,
       cleanup_interval_hours=1
   )
   ```

4. **Unload unused models:**
   ```python
   # Unload specific model
   framework.unload_model(model_id)
   
   # Auto-unload based on usage
   framework.enable_auto_model_unloading(
       max_inactive_minutes=30
   )
   ```

### GPU Out of Memory (OOM)

**Problem:** CUDA out of memory errors

**Solutions:**

1. **Reduce batch size:**
   ```python
   # Start with batch size 1
   framework.set_default_batch_size(1)
   
   # Gradually increase
   for batch_size in [2, 4, 8]:
       try:
           framework.set_default_batch_size(batch_size)
           result = framework.predict(model_id, test_input)
           print(f"✅ Batch size {batch_size} works")
       except RuntimeError as e:
           if "out of memory" in str(e):
               print(f"❌ OOM at batch size {batch_size}")
               break
   ```

2. **Enable gradient checkpointing:**
   ```python
   framework.enable_gradient_checkpointing(model_id)
   ```

3. **Use memory-efficient optimizations:**
   ```python
   framework.apply_memory_efficient_optimizations(model_id)
   ```

4. **Move to CPU if necessary:**
   ```python
   try:
       # Try GPU first
       result = framework.predict(model_id, input_data, device="cuda")
   except RuntimeError as e:
       if "out of memory" in str(e):
           print("⚠️  GPU OOM, falling back to CPU")
           result = framework.predict(model_id, input_data, device="cpu")
   ```

## 🔌 Model Loading Issues

### Model File Not Found

**Problem:** Cannot load model file

**Solutions:**

1. **Verify file path:**
   ```python
   import os
   
   model_path = "path/to/model.pth"
   
   # Check if file exists
   if not os.path.exists(model_path):
       print(f"❌ File not found: {model_path}")
       
       # List directory contents
       dir_path = os.path.dirname(model_path)
       if os.path.exists(dir_path):
           files = os.listdir(dir_path)
           print(f"Files in {dir_path}:")
           for file in files:
               print(f"  {file}")
   
   # Use absolute path
   abs_path = os.path.abspath(model_path)
   print(f"Absolute path: {abs_path}")
   ```

2. **Check file permissions:**
   ```python
   import stat
   
   if os.path.exists(model_path):
       file_stat = os.stat(model_path)
       permissions = stat.filemode(file_stat.st_mode)
       print(f"File permissions: {permissions}")
       
       if not os.access(model_path, os.R_OK):
           print("❌ No read permission")
   ```

3. **Download model if needed:**
   ```python
   # Download from URL
   framework.download_model(
       url="https://example.com/model.pth",
       local_path="models/model.pth"
   )
   
   # Or from Hugging Face
   framework.download_from_huggingface(
       model_name="microsoft/DialoGPT-medium",
       local_path="models/dialogpt"
   )
   ```

### Model Architecture Mismatch

**Problem:** Model fails to load due to architecture issues

**Solutions:**

1. **Check model compatibility:**
   ```python
   # Inspect model file
   model_info = framework.inspect_model_file("model.pth")
   print(f"Model type: {model_info['type']}")
   print(f"Framework: {model_info['framework']}")
   print(f"Version: {model_info['version']}")
   print(f"Architecture: {model_info['architecture']}")
   ```

2. **Use compatibility mode:**
   ```python
   # Load with compatibility settings
   model = framework.load_model(
       "model.pth",
       strict_loading=False,
       map_location="cpu",
       compatibility_mode=True
   )
   ```

3. **Convert model format:**
   ```python
   # Convert PyTorch to ONNX
   framework.convert_pytorch_to_onnx(
       pytorch_model_path="model.pth",
       onnx_model_path="model.onnx",
       input_shape=(1, 3, 224, 224)
   )
   
   # Load ONNX model
   model = framework.load_onnx_model("model.onnx")
   ```

### Unsupported Model Format

**Problem:** Framework doesn't recognize model format

**Solutions:**

1. **Check supported formats:**
   ```python
   supported_formats = framework.get_supported_model_formats()
   print("Supported formats:")
   for format_name, extensions in supported_formats.items():
       print(f"  {format_name}: {extensions}")
   ```

2. **Convert to supported format:**
   ```python
   # Convert TensorFlow to ONNX
   framework.convert_tensorflow_to_onnx(
       tf_model_path="model.pb",
       onnx_model_path="model.onnx"
   )
   
   # Convert Keras to PyTorch
   framework.convert_keras_to_pytorch(
       keras_model_path="model.h5",
       pytorch_model_path="model.pth"
   )
   ```

3. **Implement custom loader:**
   ```python
   from framework.core.models import ModelLoader
   
   class CustomModelLoader(ModelLoader):
       def can_load(self, model_path):
           return model_path.endswith('.custom')
       
       def load_model(self, model_path, config):
           # Your custom loading logic
           return loaded_model
   
   # Register custom loader
   framework.register_model_loader(CustomModelLoader())
   ```

## 🌐 API & Network Issues

### API Server Not Starting

**Problem:** FastAPI server fails to start

**Solutions:**

1. **Check port availability:**
   ```python
   import socket
   
   def check_port(port):
       sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
       result = sock.connect_ex(('localhost', port))
       sock.close()
       return result == 0
   
   port = 8000
   if check_port(port):
       print(f"❌ Port {port} is already in use")
   else:
       print(f"✅ Port {port} is available")
   ```

2. **Use different port:**
   ```python
   # Start on different port
   framework.start_api_server(port=8001)
   
   # Or find available port automatically
   port = framework.find_available_port()
   framework.start_api_server(port=port)
   ```

3. **Check firewall settings:**
   ```bash
   # Linux: Check if port is blocked
   sudo iptables -L | grep 8000
   
   # Windows: Check Windows Firewall
   netsh advfirewall firewall show rule name="Port 8000"
   ```

4. **Run with different host:**
   ```python
   # Listen on all interfaces
   framework.start_api_server(host="0.0.0.0", port=8000)
   
   # Or localhost only
   framework.start_api_server(host="127.0.0.1", port=8000)
   ```

### API Request Failures

**Problem:** API requests failing or timing out

**Solutions:**

1. **Check API health:**
   ```python
   import requests
   
   try:
       response = requests.get("http://localhost:8000/health")
       print(f"API Status: {response.status_code}")
       print(f"Response: {response.json()}")
   except requests.exceptions.ConnectionError:
       print("❌ Cannot connect to API server")
   except requests.exceptions.Timeout:
       print("❌ Request timed out")
   ```

2. **Increase timeout:**
   ```python
   # Client-side timeout
   response = requests.post(
       "http://localhost:8000/predict",
       json=data,
       timeout=60  # 60 seconds
   )
   
   # Server-side timeout
   framework.configure_api_timeouts(
       request_timeout=120,
       inference_timeout=60
   )
   ```

3. **Check request format:**
   ```python
   # Correct request format
   data = {
       "model_id": "classifier",
       "input_data": [[1.0, 2.0, 3.0]],  # List of lists for batch
       "options": {
           "return_probabilities": True
       }
   }
   
   response = requests.post(
       "http://localhost:8000/predict",
       json=data,
       headers={"Content-Type": "application/json"}
   )
   ```

4. **Enable debug logging:**
   ```python
   # Enable API debug logging
   framework.configure_api_logging(level="DEBUG")
   
   # Check API logs
   logs = framework.get_api_logs()
   for log_entry in logs:
       print(log_entry)
   ```

### Authentication Issues

**Problem:** API authentication failing

**Solutions:**

1. **Verify API key:**
   ```python
   # Check if API key is required
   api_config = framework.get_api_config()
   if api_config['auth_enabled']:
       print("🔒 Authentication required")
       
   # Test with API key
   headers = {
       "Authorization": "Bearer your-api-key",
       "Content-Type": "application/json"
   }
   
   response = requests.post(
       "http://localhost:8000/predict",
       json=data,
       headers=headers
   )
   ```

2. **Reset authentication:**
   ```python
   # Disable authentication for testing
   framework.configure_security(enable_auth=False)
   
   # Or generate new API key
   new_key = framework.generate_api_key()
   print(f"New API key: {new_key}")
   ```

## 🎵 Audio Processing Issues

### TTS Not Working

**Problem:** Text-to-speech synthesis fails

**Solutions:**

1. **Check audio setup:**
   ```python
   # Verify audio capabilities
   audio_info = framework.get_audio_info()
   print(f"TTS enabled: {audio_info['tts_enabled']}")
   print(f"Available TTS models: {audio_info['tts_models']}")
   ```

2. **Install missing dependencies:**
   ```bash
   pip install torchaudio soundfile librosa
   ```

3. **Test with simple text:**
   ```python
   try:
       # Simple test
       result = framework.synthesize_speech(
           text="Hello world",
           model="speecht5_tts"
       )
       print("✅ TTS working")
   except Exception as e:
       print(f"❌ TTS failed: {e}")
   ```

4. **Download TTS models:**
   ```python
   # Download required models
   framework.download_tts_model("speecht5_tts")
   framework.download_tts_model("bark_tts")
   ```

### STT Not Working

**Problem:** Speech-to-text transcription fails

**Solutions:**

1. **Check audio file format:**
   ```python
   import soundfile as sf
   
   try:
       data, samplerate = sf.read("audio.wav")
       print(f"✅ Audio file valid: {len(data)} samples at {samplerate} Hz")
   except Exception as e:
       print(f"❌ Invalid audio file: {e}")
   ```

2. **Convert audio format:**
   ```python
   # Convert to supported format
   framework.convert_audio(
       input_path="audio.mp3",
       output_path="audio.wav",
       target_sample_rate=16000
   )
   ```

3. **Test STT:**
   ```python
   try:
       result = framework.transcribe_audio(
           audio_path="audio.wav",
           model="whisper-base"
       )
       print(f"✅ STT result: {result['text']}")
   except Exception as e:
       print(f"❌ STT failed: {e}")
   ```

### Audio Quality Issues

**Problem:** Poor audio quality or incorrect results

**Solutions:**

1. **Analyze audio quality:**
   ```python
   # Check audio properties
   audio_analysis = framework.analyze_audio_quality("audio.wav")
   print(f"Signal-to-noise ratio: {audio_analysis['snr']:.2f} dB")
   print(f"Dynamic range: {audio_analysis['dynamic_range']:.2f} dB")
   print(f"Frequency response: {audio_analysis['frequency_score']:.3f}")
   ```

2. **Enhance audio:**
   ```python
   # Apply audio enhancement
   enhanced_audio = framework.enhance_audio(
       audio_path="noisy_audio.wav",
       noise_reduction=True,
       normalize=True,
       output_path="enhanced_audio.wav"
   )
   ```

3. **Use appropriate sample rate:**
   ```python
   # Resample audio
   framework.resample_audio(
       input_path="audio.wav",
       output_path="resampled_audio.wav",
       target_sample_rate=16000  # Standard for STT
   )
   ```

## 🐳 Docker & Deployment Issues

### Docker Build Failures

**Problem:** Docker image build fails

**Solutions:**

1. **Check Docker installation:**
   ```bash
   docker --version
   docker info
   ```

2. **Update base image:**
   ```dockerfile
   # Use specific tag instead of latest
   FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
   
   # Or use official Python image
   FROM python:3.10-slim
   ```

3. **Fix dependency conflicts:**
   ```dockerfile
   # Install system dependencies first
   RUN apt-get update && apt-get install -y \
       build-essential \
       libsndfile1 \
       ffmpeg \
       && rm -rf /var/lib/apt/lists/*
   
   # Install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   ```

4. **Use multi-stage build:**
   ```dockerfile
   # Build stage
   FROM python:3.10 as builder
   COPY requirements.txt .
   RUN pip install --user -r requirements.txt
   
   # Runtime stage
   FROM python:3.10-slim
   COPY --from=builder /root/.local /root/.local
   ```

### Container Runtime Issues

**Problem:** Docker container fails at runtime

**Solutions:**

1. **Check container logs:**
   ```bash
   # View logs
   docker logs container_name
   
   # Follow logs
   docker logs -f container_name
   
   # Debug mode
   docker run -it --entrypoint /bin/bash your_image
   ```

2. **Fix GPU access:**
   ```bash
   # Install nvidia-container-runtime
   sudo apt-get install nvidia-container-runtime
   
   # Run with GPU support
   docker run --gpus all your_image
   
   # Or specific GPU
   docker run --gpus device=0 your_image
   ```

3. **Mount volumes correctly:**
   ```bash
   # Mount models directory
   docker run -v $(pwd)/models:/app/models your_image
   
   # Mount cache directory
   docker run -v ~/.cache:/root/.cache your_image
   ```

### Kubernetes Deployment Issues

**Problem:** Kubernetes deployment fails

**Solutions:**

1. **Check resource limits:**
   ```yaml
   # deployment.yaml
   resources:
     requests:
       memory: "4Gi"
       cpu: "2"
       nvidia.com/gpu: 1
     limits:
       memory: "8Gi"
       cpu: "4"
       nvidia.com/gpu: 1
   ```

2. **Fix image pull issues:**
   ```yaml
   # Use image pull secrets
   imagePullSecrets:
   - name: registry-secret
   
   # Or use always pull policy
   imagePullPolicy: Always
   ```

3. **Check node selectors:**
   ```yaml
   # Ensure GPU nodes
   nodeSelector:
     accelerator: nvidia-tesla-k80
   
   # Or use node affinity
   affinity:
     nodeAffinity:
       requiredDuringSchedulingIgnoredDuringExecution:
         nodeSelectorTerms:
         - matchExpressions:
           - key: nvidia.com/gpu
             operator: Exists
   ```

## 🔍 Debug Mode & Logging

### Enable Debug Mode

```python
# Enable comprehensive debugging
framework.enable_debug_mode()

# Configure detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable framework-specific debug logging
framework.configure_logging(
    level="DEBUG",
    enable_performance_logging=True,
    enable_memory_logging=True,
    enable_model_logging=True
)
```

### Collect System Information

```python
# Generate comprehensive diagnostic report
def generate_debug_report():
    report = {
        'system_info': framework.get_system_info(),
        'gpu_info': framework.get_gpu_info(),
        'memory_stats': framework.get_memory_stats(),
        'model_info': {
            model_id: framework.get_model_info(model_id)
            for model_id in framework.list_models()
        },
        'performance_metrics': framework.get_performance_metrics(),
        'config': framework.get_config(),
        'logs': framework.get_recent_logs(limit=100)
    }
    
    # Save report
    import json
    with open('debug_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("Debug report saved to debug_report.json")
    return report

debug_report = generate_debug_report()
```

## 📞 Getting Help

### When to Contact Support

Contact support if you've tried the solutions above and still experience:
- Persistent crashes or errors
- Significant performance degradation
- Data corruption or incorrect results
- Security concerns

### Information to Include

When reporting issues, include:

1. **System information:**
   ```python
   print(framework.get_system_info())
   ```

2. **Error details:**
   - Full error message and stack trace
   - Steps to reproduce
   - Expected vs actual behavior

3. **Environment details:**
   - Framework version
   - PyTorch version
   - CUDA version
   - Operating system

4. **Minimal reproduction code:**
   ```python
   # Provide minimal code that reproduces the issue
   from framework import TorchInferenceFramework
   
   framework = TorchInferenceFramework()
   # ... minimal reproduction steps
   ```

### Support Channels

- **GitHub Issues:** For bugs and feature requests
- **Discussions:** For general questions and community help
- **Documentation:** Check [FAQ](FAQ.md) and guides
- **Discord/Slack:** For real-time community support

---

## 🏷️ Common Error Codes

| Error Code | Description | Solution |
|------------|-------------|----------|
| `CUDA_ERROR_001` | CUDA out of memory | Reduce batch size or enable memory optimization |
| `MODEL_ERROR_002` | Model loading failed | Check file path and format |
| `API_ERROR_003` | Authentication failed | Verify API key or disable auth |
| `AUDIO_ERROR_004` | Audio processing failed | Install audio dependencies |
| `OPTIMIZATION_ERROR_005` | Optimization failed | Disable specific optimization or use fallback |

---

*For additional help, see our [FAQ](FAQ.md) or [contact support](https://github.com/your-org/torch-inference/issues).*

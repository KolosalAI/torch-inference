# 🎵 Text-to-Speech Models Guide

> **Comprehensive guide for downloading, configuring, and using Text-to-Speech models with the PyTorch Inference Framework**

[![TTS](https://img.shields.io/badge/Task-Text%20to%20Speech-green)](https://en.wikipedia.org/wiki/Speech_synthesis)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20Inference-red)](../README.md)
[![Models](https://img.shields.io/badge/Models-Multiple%20Supported-blue)](#supported-models)

This guide provides comprehensive instructions for working with Text-to-Speech (TTS) models in the PyTorch Inference Framework, including model downloads, configuration, and usage examples.

## 📑 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🤖 Supported Models](#-supported-models)
- [⬇️ Model Downloads](#️-model-downloads)
- [🔧 Configuration](#-configuration)
- [🎯 Usage Examples](#-usage-examples)
- [📊 Performance Benchmarks](#-performance-benchmarks)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [📚 API Reference](#-api-reference)

## 🚀 Quick Start

### Prerequisites
1. **PyTorch Inference Framework** running on `http://localhost:8000`
2. **Python 3.8+** with required dependencies
3. **Internet connection** for model downloads
4. **Sufficient disk space** (models range from 500MB to 5GB)

### Start the Framework
```bash
# Navigate to framework directory
cd /path/to/torch-inference

# Start the inference server
python main.py

# Verify server is running
curl http://localhost:8000/health
```

## 🤖 Supported Models

The framework supports various state-of-the-art TTS models:

### Core TTS Models

| Model | Provider | Quality | Speed | Size | Use Case |
|-------|----------|---------|-------|------|----------|
| **SpeechT5** | Microsoft | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ~2.5GB | High-quality general TTS |
| **Bark** | Suno AI | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ~4GB | Multi-lingual, emotional TTS |
| **Tacotron2** | NVIDIA | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ~300MB | Fast, lightweight TTS |
| **VALL-E X** | Microsoft | ⭐⭐⭐⭐⭐ | ⭐⭐ | ~3GB | Voice cloning, advanced TTS |

### Foundation Models for TTS

| Model | Provider | Size | TTS Application |
|-------|----------|------|-----------------|
| **BART-Base** | Facebook | ~500MB | Lightweight TTS foundation |
| **BART-Large** | Facebook | ~1.6GB | High-quality TTS foundation |
| **T5** | Google | ~800MB - 3GB | Text-to-text TTS adaptation |

## ⬇️ Model Downloads

### Framework-Managed Downloads

The framework provides automatic model download and optimization:

#### 1. Download SpeechT5 (Recommended)

```bash
# Using CLI
python -m framework.tools.model_downloader \
    --source huggingface \
    --model-id microsoft/speecht5_tts \
    --name speecht5_tts \
    --task text-to-speech \
    --include-vocoder
```

```python
# Using Python API
from framework.core.model_downloader import download_model

model_path, info = download_model(
    source="huggingface",
    model_id="microsoft/speecht5_tts",
    name="speecht5_tts",
    task="text-to-speech",
    include_vocoder=True,
    vocoder_model="microsoft/speecht5_hifigan"
)
```

#### 2. Download Bark TTS

```bash
# CLI download
python -m framework.tools.model_downloader \
    --source huggingface \
    --model-id suno/bark \
    --name bark_tts \
    --task text-to-speech \
    --enable-large-model
```

#### 3. Download BART Models for TTS

```bash
# BART-Large for TTS foundation
python -m framework.tools.model_downloader \
    --source huggingface \
    --model-id facebook/bart-large \
    --name bart_large_tts \
    --task text-generation \
    --auto-convert-tts

# BART-Base for lightweight TTS
python -m framework.tools.model_downloader \
    --source huggingface \
    --model-id facebook/bart-base \
    --name bart_base_tts \
    --task text-generation \
    --auto-convert-tts
```

### REST API Downloads

Download models via HTTP API:

```bash
# Download SpeechT5 via API
curl -X POST "http://localhost:8000/models/download" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "huggingface",
    "model_id": "microsoft/speecht5_tts",
    "name": "speecht5_tts",
    "task": "text-to-speech",
    "include_vocoder": true,
    "vocoder_model": "microsoft/speecht5_hifigan"
  }'
```

```bash
# Download Bark TTS via API
curl -X POST "http://localhost:8000/models/download" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "huggingface",
    "model_id": "suno/bark",
    "name": "bark_tts",
    "task": "text-to-speech",
    "enable_large_model": true
  }'
```

### PowerShell Downloads (Windows)

For Windows users with PowerShell, detailed Windows-specific commands and examples are available in this guide's PowerShell sections.

## 🔧 Configuration

### Basic TTS Configuration

Create or update `config.yaml`:

```yaml
# TTS Configuration
tts:
  enabled: true
  default_model: "speecht5_tts"
  cache_enabled: true
  optimization_level: "high"
  
  # Model-specific settings
  models:
    speecht5_tts:
      vocoder: "microsoft/speecht5_hifigan"
      sample_rate: 16000
      max_length: 1024
    
    bark_tts:
      enable_large_model: true
      voice_preset: "v2/en_speaker_6"
      generation_temperature: 0.8
    
    bart_large_tts:
      auto_convert: true
      max_sequence_length: 1024
      batch_size: 4

# Server optimization
server:
  enable_model_caching: true
  auto_optimization: true
  memory_management: "aggressive"
```

### Advanced Configuration

```yaml
tts:
  # Quality settings
  quality:
    sample_rate: 22050
    bit_depth: 16
    channels: 1
  
  # Performance settings
  performance:
    batch_processing: true
    parallel_synthesis: true
    cache_strategy: "adaptive"
    real_time_factor_target: 0.5
  
  # Output settings
  output:
    format: "wav"
    compression: false
    normalize_audio: true
    trim_silence: true
```

## 🎯 Usage Examples

### 1. Basic TTS Synthesis

```python
import asyncio
import aiohttp

async def synthesize_speech():
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/tts/synthesize", json={
            "text": "Hello, this is a test of the text-to-speech system.",
            "model_name": "speecht5_tts",
            "voice": "default",
            "speed": 1.0,
            "language": "en"
        }) as response:
            result = await response.json()
            
            if result["success"]:
                # Save audio file
                import base64
                audio_data = base64.b64decode(result["audio_data"])
                with open("output.wav", "wb") as f:
                    f.write(audio_data)
                print("✅ TTS synthesis completed!")
            else:
                print(f"❌ TTS failed: {result['error']}")

asyncio.run(synthesize_speech())
```

### 2. Advanced TTS with Bark

```python
async def advanced_bark_tts():
    tts_request = {
        "text": "This is an advanced text-to-speech demonstration with emotional tone and custom voice settings.",
        "model_name": "bark_tts",
        "voice": "v2/en_speaker_6",
        "speed": 1.1,
        "emotion": "happy",
        "temperature": 0.8,
        "output_format": "wav",
        "server_features": {
            "enable_caching": True,
            "optimization_level": "high",
            "quality_preset": "maximum"
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/tts/synthesize", json=tts_request) as response:
            result = await response.json()
            
            if result["success"]:
                # Process and save audio
                audio_data = base64.b64decode(result["audio_data"])
                with open("bark_advanced_output.wav", "wb") as f:
                    f.write(audio_data)
                
                print(f"✅ Advanced Bark TTS completed!")
                print(f"   Duration: {result['duration']} seconds")
                print(f"   Processing time: {result['processing_time']}s")
                print(f"   Optimizations: {', '.join(result.get('optimizations_applied', []))}")
```

### 3. BART-based TTS

```python
async def bart_tts_generation():
    bart_request = {
        "text": "Generate high-quality speech from this text using the BART foundation model.",
        "model_name": "bart_large_tts",
        "priority": 1,
        "timeout": 30.0,
        "enable_batching": True,
        "tts_config": {
            "voice": "default",
            "speed": 1.0,
            "pitch": 1.0,
            "output_format": "wav"
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/tts/generate", json=bart_request) as response:
            result = await response.json()
            
            if result["success"]:
                audio_data = base64.b64decode(result["audio_data"])
                with open("bart_tts_output.wav", "wb") as f:
                    f.write(audio_data)
                print("✅ BART TTS generation completed!")
```

### 4. Batch TTS Processing

```python
async def batch_tts_processing():
    texts = [
        "Welcome to the PyTorch inference framework.",
        "This system supports multiple text-to-speech models.",
        "Including SpeechT5, Bark, and BART-based models.",
        "Thank you for using our TTS system!"
    ]
    
    batch_request = {
        "texts": texts,
        "model_name": "speecht5_tts",
        "batch_config": {
            "parallel_processing": True,
            "enable_caching": True,
            "optimization_level": "high",
            "output_format": "wav"
        },
        "voice_config": {
            "speed": 1.0,
            "language": "en",
            "voice": "default"
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/tts/batch", json=batch_request) as response:
            result = await response.json()
            
            if result["success"]:
                batch_id = result["batch_id"]
                print(f"✅ Batch TTS initiated! Batch ID: {batch_id}")
                
                # Monitor progress
                while True:
                    async with session.get(f"http://localhost:8000/tts/batch/status/{batch_id}") as status_response:
                        status = await status_response.json()
                        print(f"   Progress: {status['completed']}/{status['total']} completed")
                        
                        if status["status"] == "completed":
                            break
                        
                    await asyncio.sleep(2)
                
                # Download results
                async with session.get(f"http://localhost:8000/tts/batch/results/{batch_id}") as results_response:
                    results = await results_response.json()
                    
                    for i, audio_file in enumerate(results["audio_files"]):
                        audio_data = base64.b64decode(audio_file["audio_data"])
                        filename = f"batch_tts_{i+1}.wav"
                        with open(filename, "wb") as f:
                            f.write(audio_data)
                        print(f"   Saved: {filename}")
```

## 📊 Performance Benchmarks

### Benchmark Different Models

```python
import time
import asyncio

async def benchmark_tts_models():
    benchmark_text = "This is a standardized text for performance benchmarking of text-to-speech models."
    
    models = [
        {"name": "speecht5_tts", "optimization": "high"},
        {"name": "bark_tts", "optimization": "balanced"},
        {"name": "bart_large_tts", "optimization": "high"}
    ]
    
    results = []
    
    for model in models:
        print(f"🔄 Benchmarking {model['name']}...")
        
        request = {
            "text": benchmark_text,
            "model_name": model["name"],
            "speed": 1.0,
            "language": "en",
            "output_format": "wav",
            "server_config": {
                "optimization_level": model["optimization"],
                "enable_caching": False  # Disable for accurate benchmarking
            }
        }
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:8000/tts/synthesize", json=request) as response:
                result = await response.json()
                
                total_time = time.time() - start_time
                
                if result["success"]:
                    results.append({
                        "model": model["name"],
                        "processing_time": result["processing_time"],
                        "audio_duration": result["duration"],
                        "total_time": total_time,
                        "real_time_factor": result["processing_time"] / result["duration"],
                        "memory_usage": result.get("memory_usage_mb", 0)
                    })
                    
                    print(f"✅ {model['name']} completed")
    
    # Display results
    print("\n📊 Performance Benchmark Results:")
    print(f"Text length: {len(benchmark_text)} characters")
    
    for result in results:
        print(f"\n🎤 {result['model']}:")
        print(f"   Processing Time: {result['processing_time']:.3f}s")
        print(f"   Audio Duration: {result['audio_duration']:.3f}s")
        print(f"   Total Time: {result['total_time']:.3f}s")
        print(f"   Real-time Factor: {result['real_time_factor']:.3f}x")
        print(f"   Memory Usage: {result['memory_usage']:.1f} MB")

# Run benchmark
asyncio.run(benchmark_tts_models())
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Model Download Failures

```bash
# Check download status
curl http://localhost:8000/models/status/speecht5_tts

# Retry failed download
curl -X POST "http://localhost:8000/models/retry" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "speecht5_tts"}'
```

#### 2. TTS Synthesis Errors

```python
# Test TTS endpoint health
async def test_tts_health():
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/tts/health") as response:
            health = await response.json()
            print(f"TTS Health: {health['status']}")
            print(f"Available models: {health['available_models']}")
```

#### 3. Performance Issues

```bash
# Check system resources
curl http://localhost:8000/stats

# Optimize server performance
curl -X POST "http://localhost:8000/server/optimize" \
  -H "Content-Type: application/json" \
  -d '{"action": "optimize_memory", "clear_unused_cache": true}'
```

### Diagnostic Script

```python
async def run_tts_diagnostics():
    """Comprehensive TTS system diagnostics"""
    
    print("🔧 Running TTS Diagnostics...")
    
    # 1. Check framework health
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/health") as response:
                health = await response.json()
                print(f"✅ Framework: {'Online' if health['healthy'] else 'Issues detected'}")
    except:
        print("❌ Framework: Offline or unreachable")
        return
    
    # 2. Check available models
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/models") as response:
                models = await response.json()
                tts_models = [m for m in models if m.get('task') == 'text-to-speech']
                print(f"📊 TTS Models: {len(tts_models)} available")
                
                for model in tts_models:
                    status = "✅" if model.get('loaded') else "⬇️"
                    print(f"   {status} {model['name']}")
    except:
        print("⚠️ Unable to retrieve model information")
    
    # 3. Test TTS endpoint
    try:
        test_request = {
            "text": "Test",
            "model_name": "default",
            "output_format": "wav"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:8000/tts/synthesize", json=test_request) as response:
                result = await response.json()
                print(f"✅ TTS Endpoint: {'Working' if result.get('success') else 'Failed'}")
    except:
        print("❌ TTS Endpoint: Not working")
    
    print("🎉 Diagnostics completed!")

# Run diagnostics
asyncio.run(run_tts_diagnostics())
```

## 📚 API Reference

### TTS Synthesis Endpoint

**POST** `/tts/synthesize`

Generate speech from text using specified TTS model.

#### Request Body

```json
{
  "text": "string",           // Text to synthesize (required)
  "model_name": "string",    // TTS model name (required)
  "voice": "string",         // Voice preset (optional)
  "speed": 1.0,              // Speech speed (0.5-2.0)
  "pitch": 1.0,              // Pitch adjustment (0.5-2.0)
  "language": "en",          // Language code
  "output_format": "wav",    // Audio format
  "server_config": {         // Server optimization settings
    "optimization_level": "high",
    "enable_caching": true,
    "quality_preset": "maximum"
  }
}
```

#### Response

```json
{
  "success": true,
  "audio_data": "base64_encoded_audio",
  "duration": 3.5,
  "sample_rate": 22050,
  "processing_time": 0.8,
  "cache_hit": false,
  "optimizations_applied": ["memory_optimization", "batch_processing"]
}
```

### Batch TTS Endpoint

**POST** `/tts/batch`

Process multiple texts in batch for efficient synthesis.

#### Request Body

```json
{
  "texts": ["string array"],
  "model_name": "string",
  "batch_config": {
    "parallel_processing": true,
    "enable_caching": true,
    "optimization_level": "high"
  },
  "voice_config": {
    "speed": 1.0,
    "language": "en"
  }
}
```

### Model Management

**GET** `/models/tts`
List available TTS models.

**POST** `/models/download`
Download new TTS model.

**GET** `/models/status/{model_name}`
Check model download/load status.

---

## Related Documentation

- **[Audio Processing Guide](audio.md)** - Complete audio features
- **[Model Download Guide](model_download.md)** - Model management
- **[API Reference](api/)** - Complete API documentation
- **[Configuration Guide](configuration.md)** - Framework configuration

## 🤝 Contributing

Found an issue or want to add more TTS models? Please contribute!

1. Fork the repository
2. Add new model support
3. Test with the framework
4. Submit a pull request

## 📄 License

This documentation is part of the PyTorch Inference Framework project, licensed under the **MIT License**.

---

<div align="center">

**🎵 Happy Text-to-Speech Generation! 🎵**

*Built with ❤️ for the PyTorch and AI community*

</div>

# PyTorch Inference Framework - API Documentation

This directory contains comprehensive API documentation for all endpoints, authentication, and usage examples.

## 📚 Documentation Structure

```
docs/api/
├── README.md                    # This overview
├── rest-api.md                 # Complete REST API reference (existing)
├── authentication.md           # Authentication and security
├── endpoints/                  # Individual endpoint documentation
│   ├── core.md                # Core inference endpoints
│   ├── models.md              # Model management
│   ├── audio.md               # Audio/TTS/STT processing
│   ├── gpu.md                 # GPU detection and management
│   ├── autoscaler.md          # Autoscaling endpoints
│   ├── server.md              # Server management
│   └── logging.md             # Logging endpoints
├── examples/                   # Code examples and tutorials
│   ├── basic-usage.md         # Basic API usage examples
│   ├── tts-examples.md        # Text-to-speech examples
│   ├── model-management.md    # Model download and management
│   ├── authentication.md      # Authentication examples
│   └── advanced-usage.md      # Advanced features and optimizations
└── schemas/                    # Request/Response schemas
    ├── inference.md           # Inference request/response schemas
    ├── audio.md               # Audio processing schemas
    ├── models.md              # Model management schemas
    └── common.md              # Common response formats
```

## 🚀 Quick Start

### 1. Basic Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "example",
    "inputs": [1, 2, 3, 4, 5]
  }'
```

### 2. Text-to-Speech
```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "speecht5_tts",
    "inputs": "Hello, this is a test of the text-to-speech system."
  }'
```

### 3. Model Download
```bash
curl -X POST http://localhost:8000/models/download \
  -H "Content-Type: application/json" \
  -d '{
    "source": "huggingface",
    "model_id": "microsoft/speecht5_tts",
    "name": "speecht5_tts",
    "task": "text-to-speech",
    "include_vocoder": true
  }'
```

### 4. System Health Check
```bash
curl http://localhost:8000/health
```

## 📋 API Categories

### Core Inference
- **`/predict`** - Unified prediction endpoint for all models
- **`/health`** - System health check with autoscaler info
- **`/info`** - Comprehensive system information
- **`/stats`** - Performance statistics
- **`/config`** - Configuration information

### Model Management
- **`/models`** - List available models
- **`/models/download`** - Download and load models
- **`/models/available`** - List downloadable models
- **`/models/managed`** - Server-managed model info
- **`/models/cache/info`** - Model cache information

### Audio Processing
- **`/synthesize`** - Text-to-speech synthesis
- **`/transcribe`** - Speech-to-text transcription
- **`/audio/health`** - Audio processing health
- **`/tts/health`** - TTS service health check

### GPU Management
- **`/gpu/detect`** - Detect available GPUs
- **`/gpu/best`** - Get best GPU for inference
- **`/gpu/config`** - GPU-optimized configuration
- **`/gpu/report`** - Comprehensive GPU report

### Autoscaling
- **`/autoscaler/stats`** - Autoscaler statistics
- **`/autoscaler/health`** - Autoscaler health check
- **`/autoscaler/scale`** - Scale model instances
- **`/autoscaler/metrics`** - Detailed metrics

### Server Management
- **`/server/config`** - Server configuration
- **`/server/optimize`** - Performance optimization
- **`/metrics/server`** - Server performance metrics
- **`/metrics/tts`** - TTS-specific metrics

### Logging
- **`/logs`** - Logging information
- **`/logs/{log_file}`** - View/download log files
- **`DELETE /logs/{log_file}`** - Clear log files

### Authentication (if enabled)
- **`/auth/register`** - User registration
- **`/auth/login`** - User login
- **`/auth/logout`** - User logout
- **`/auth/profile`** - User profile
- **`/auth/generate-key`** - Generate API keys

## 🔧 Configuration

The API server supports various configuration options:

### Environment Variables
- `DEVICE` - Device type (cuda, cpu, auto)
- `BATCH_SIZE` - Default batch size
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `ENABLE_AUTH` - Enable authentication (true/false)

### Configuration Files
- `config.yaml` - Main configuration
- `models.json` - Available models configuration
- `auth.yaml` - Authentication settings (if enabled)

## 📊 Response Formats

All API endpoints return consistent JSON responses:

### Success Response
```json
{
  "success": true,
  "result": "...",
  "processing_time": 0.025,
  "model_info": {
    "model": "example",
    "device": "cuda:0"
  }
}
```

### Error Response
```json
{
  "success": false,
  "error": "Error description",
  "detail": "Detailed error information"
}
```

## 🛡️ Security Features

- **Optional Authentication** - JWT-based authentication with API keys
- **Rate Limiting** - Configurable request rate limits
- **Input Validation** - Comprehensive request validation
- **Security Headers** - CORS and security headers
- **Audit Logging** - Request/response logging

## 📈 Performance Features

- **Batch Processing** - Efficient batch inference
- **Model Caching** - Automatic model caching
- **GPU Acceleration** - CUDA, MPS, and multi-GPU support
- **Autoscaling** - Dynamic model scaling
- **Optimization** - TensorRT, FP16, INT8 support

## 🎵 TTS/Audio Features

- **Multiple TTS Models** - BART, SpeechT5, Bark, VALL-E X, Tacotron2
- **Auto Model Download** - Automatic model downloading
- **Multiple Formats** - WAV, MP3, FLAC support
- **Voice Control** - Speed, pitch, volume adjustment
- **Multi-language** - Support for multiple languages

## 🔗 External Links

- [Interactive API Documentation](http://localhost:8000/docs) - Swagger UI
- [Alternative API Documentation](http://localhost:8000/redoc) - ReDoc
- [Health Check](http://localhost:8000/health) - System health
- [System Info](http://localhost:8000/info) - Detailed system information

## 📝 Examples

See the `examples/` directory for comprehensive code examples in multiple programming languages:
- Python (requests, httpx, aiohttp)
- JavaScript (fetch, axios)
- cURL commands
- Postman collections

## 🐛 Troubleshooting

Common issues and solutions:

1. **Model not found** - Check `/models/available` for downloadable models
2. **CUDA out of memory** - Reduce batch size or use smaller models
3. **Audio processing errors** - Install audio dependencies (`librosa`, `soundfile`)
4. **Authentication errors** - Ensure valid token or API key
5. **Performance issues** - Check GPU utilization and enable optimizations

## 📞 Support

For issues and questions:
- Check the troubleshooting guide
- Review log files via `/logs` endpoint
- Monitor system health via `/health` endpoint
- Check performance metrics via `/stats` endpoint

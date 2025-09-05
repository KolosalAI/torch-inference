# REST API Reference

The PyTorch Inference Framework provides a comprehensive REST API for model inference, management, and monitoring. This document details all available endpoints with examples and response formats.

## Base URL

```
http://localhost:8000
```

## Quick Reference

| Category | Endpoint | Method | Description |
|----------|----------|--------|-------------|
| **Core** | `/` | GET | API information |
| **Core** | `/predict` | POST | Unified prediction endpoint |
| **Audio** | `/synthesize` | POST | Text-to-speech synthesis |
| **Health** | `/health` | GET | System health check |
| **Stats** | `/stats` | GET | Performance statistics |
| **Config** | `/config` | GET | Configuration info |
| **Models** | `/models` | GET | List loaded models |
| **Audio** | `/stt/transcribe` | POST | Speech-to-text |
| **GPU** | `/gpu/detect` | GET | GPU detection |
| **Autoscaling** | `/autoscaler/stats` | GET | Autoscaler statistics |

## Authentication

Currently, the API doesn't require authentication for local development. For production deployment, consider implementing API keys or OAuth2.

## Content Types

- **Request Content-Type**: `application/json` (for most endpoints)
- **File Upload Content-Type**: `multipart/form-data` (for audio files)
- **Response Content-Type**: `application/json`

## Error Handling

All endpoints return consistent error responses:

```json
{
  "detail": "Error description",
  "status_code": 400
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (model/resource not found)
- `422`: Validation Error (request validation failed)
- `500`: Internal Server Error
- `503`: Service Unavailable (engine not ready)

---

## Core Inference Endpoints

### Root Endpoint

Get API information and available endpoints.

**Request:**
```http
GET /
```

**Response:**
```json
{
  "message": "PyTorch Inference Framework API - Enhanced with TTS Support",
  "version": "1.0.0-TTS-Enhanced",
  "status": "running",
  "timestamp": "2024-01-15T10:30:00Z",
  "environment": "development",
  "tts_support": {
    "enabled": true,
    "supported_models": [
      "facebook/bart-large",
      "microsoft/speecht5_tts",
      "suno/bark",
      "tacotron2"
    ],
    "features": [
      "Auto-download popular TTS models",
      "GPU acceleration",
      "Voice synthesis",
      "Multiple audio formats"
    ]
  },
  "endpoints": {
    "inference": {
      "predict": "/predict",
      "model_specific": "/{model_name}/predict",
      "batch": "/predict/batch"
    },
    "audio": {
      "synthesize": "/tts/synthesize",
      "transcribe": "/stt/transcribe"
    }
  }
}
```

### Unified Prediction

Perform inference using any torch model or deep learning model with ultra-optimized processing.

**Request:**
```http
POST /predict
Content-Type: application/json

{
  "model_name": "my_model",
  "inputs": [1, 2, 3, 4, 5],
  "token": "optional_auth_token",
  "priority": 0,
  "timeout": 30.0,
  "enable_batching": true
}
```

**Batch Request:**
```http
POST /predict
Content-Type: application/json

{
  "model_name": "my_model",
  "inputs": [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15]
  ],
  "token": "optional_auth_token",
  "priority": 1,
  "timeout": 30.0,
  "enable_batching": true
}
```

**Response:**
```json
{
  "success": true,
  "result": 0.75,
  "processing_time": 0.025,
  "model_info": {
    "model": "my_model",
    "device": "cuda:0",
    "input_type": "single",
    "input_count": 1,
    "processing_path": "optimized"
  },
  "batch_info": {
    "inflight_batching_enabled": true,
    "processed_as_batch": false,
    "concurrent_optimization": false
  }
}
```json
{
  "success": true,
  "result": "Processed: Hello world",
  "processing_time": 0.018,
  "model_info": {
    "model": "my_model",
    "device": "cuda:0",
    "input_type": "single",
    "input_count": 1,
    "processing_path": "optimized"
  }
}
```

### Batch Prediction

Process multiple inputs efficiently.

**Request:**
```http
POST /predict/batch
Content-Type: application/json

{
  "inputs": [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
  ],
  "priority": 0,
  "timeout": 60.0,
  "enable_batching": true
}
```

**Response:**
```json
{
  "success": true,
  "result": [0.45, 0.67, 0.89],
  "processing_time": 0.055,
  "model_info": {
    "model": "example",
    "device": "cuda:0",
    "input_type": "batch",
    "input_count": 3,
    "processing_path": "optimized"
  },
  "batch_info": {
    "inflight_batching_enabled": true,
    "processed_as_batch": true,
    "concurrent_optimization": false
  }
}
```

---

## Health and Monitoring Endpoints

### Health Check

Check system health and component status.

**Request:**
```http
GET /health
```

**Response:**
```json
{
  "healthy": true,
  "checks": {
    "inference_engine": true,
    "model": true,
    "device": true
  },
  "timestamp": 1705315800.123,
  "engine_stats": {
    "requests_processed": 1234,
    "avg_processing_time": 0.025,
    "active_requests": 2
  }
}
```

### Performance Statistics

Get detailed performance metrics.

**Request:**
```http
GET /stats
```

**Response:**
```json
{
  "stats": {
    "requests_processed": 1234,
    "total_processing_time": 30.75,
    "avg_processing_time": 0.025,
    "active_requests": 2,
    "queue_size": 0,
    "memory_usage_mb": 2048
  },
  "performance_report": {
    "throughput_fps": 40.0,
    "latency_p50": 0.020,
    "latency_p95": 0.045,
    "latency_p99": 0.080,
    "device_utilization": 0.75
  }
}
```

### Configuration Information

Get current system configuration.

**Request:**
```http
GET /config
```

**Response:**
```json
{
  "configuration": {
    "device": {
      "device_type": "cuda",
      "device_id": 0,
      "use_fp16": true,
      "use_tensorrt": false
    },
    "batch": {
      "batch_size": 4,
      "max_batch_size": 16
    }
  },
  "inference_config": {
    "device_type": "cuda",
    "batch_size": 4,
    "use_fp16": true,
    "enable_profiling": false
  },
  "server_config": {
    "host": "0.0.0.0",
    "port": 8000,
    "log_level": "INFO"
  },
  "logging_config": {
    "log_level": "INFO",
    "log_directory": "/app/logs",
    "log_files": {
      "server.log": {
        "size_mb": 15.3,
        "last_modified": "2024-01-15T10:25:00Z"
      }
    }
  }
}
```

---

## Model Management Endpoints

### List Models

Get information about loaded models.

**Request:**
```http
GET /models
```

**Response:**
```json
{
  "models": ["example", "bert_model", "resnet50"],
  "model_info": {
    "example": {
      "model_name": "ExampleModel",
      "device": "cuda:0",
      "loaded": true,
      "optimized": true
    },
    "bert_model": {
      "model_name": "BERT",
      "device": "cuda:0",
      "loaded": true,
      "optimized": false
    }
  },
  "total_models": 3
}
```

### Enhanced Model Download

Download models with comprehensive TTS support.

**Request:**
```http
POST /models/download
Content-Type: application/json

{
  "source": "huggingface",
  "model_id": "microsoft/speecht5_tts",
  "name": "speecht5_tts",
  "task": "text-to-speech",
  "auto_convert_tts": true,
  "include_vocoder": true,
  "vocoder_model": "microsoft/speecht5_hifigan",
  "enable_large_model": false,
  "experimental": false
}
```

**Response:**
```json
{
  "success": true,
  "download_id": "abc12345",
  "message": "Started downloading TTS model 'speecht5_tts'",
  "model_name": "speecht5_tts",
  "source": "huggingface",
  "model_id": "microsoft/speecht5_tts",
  "status": "downloading",
  "estimated_time": "8-15 minutes",
  "download_info": {
    "download_id": "abc12345",
    "model_type": "tts",
    "description": "Microsoft SpeechT5 TTS model",
    "estimated_size_mb": 2500,
    "supports_voice_cloning": false,
    "vocoder_included": true,
    "vocoder_models": ["microsoft/speecht5_hifigan"]
  }
}
```

### Available Models for Download

List models available for download with TTS focus.

**Request:**
```http
GET /models/available
```

**Response:**
```json
{
  "available_models": {
    "speecht5_tts": {
      "name": "speecht5_tts",
      "source": "huggingface",
      "model_id": "microsoft/speecht5_tts",
      "task": "text-to-speech",
      "description": "Microsoft SpeechT5 TTS model",
      "size_mb": 2500,
      "tags": ["speecht5", "microsoft", "tts"],
      "tts_features": {
        "supports_tts": true,
        "quality": "very-high",
        "speed": "medium",
        "vocoder_required": true
      }
    },
    "bark_tts": {
      "name": "bark_tts",
      "source": "huggingface", 
      "model_id": "suno/bark",
      "task": "text-to-speech",
      "description": "Suno Bark TTS model with voice cloning",
      "size_mb": 4000,
      "tags": ["bark", "suno", "tts", "voice-cloning"],
      "tts_features": {
        "supports_tts": true,
        "quality": "very-high",
        "speed": "slow",
        "supports_voice_cloning": true,
        "supports_emotions": true
      }
    }
  },
  "total_available": 6,
  "categories": {
    "tts_models": ["speecht5_tts", "bark_tts", "bart_large_tts"],
    "general_models": ["resnet50", "bert_base"]
  },
  "popular_tts_models": [
    "speecht5_tts",
    "bark_tts",
    "bart_large_tts"
  ],
  "download_recommendations": {
    "beginners": ["speecht5_tts", "tacotron2_tts"],
    "advanced": ["bark_tts", "vall_e_x"],
    "fast_setup": ["bart_base_tts", "tacotron2_tts"],
    "highest_quality": ["bark_tts", "speecht5_tts"]
  }
}
```

### Download Status

Check the status of a model download.

**Request:**
```http
GET /models/download/status/{download_id}
```

**Response:**
```json
{
  "download_id": "abc12345",
  "status": "completed",
  "progress": 100,
  "eta": null,
  "message": "Download completed successfully"
}
```

### Managed Models

Get information about server-managed models.

**Request:**
```http
GET /models/managed
```

**Response:**
```json
{
  "total_models": 3,
  "downloaded_models": [
    {
      "name": "speecht5_tts",
      "loaded": true,
      "device": "cuda:0",
      "is_tts": true,
      "optimized": true,
      "memory_usage": {
        "allocated_mb": 512,
        "reserved_mb": 768
      },
      "parameters": 144000000
    }
  ],
  "categories": {
    "tts_models": [
      {
        "name": "speecht5_tts",
        "loaded": true,
        "is_tts": true
      }
    ],
    "other_models": [
      {
        "name": "example",
        "loaded": true,
        "is_tts": false
      }
    ]
  },
  "summary": {
    "total_loaded": 3,
    "tts_models_count": 1,
    "optimized_count": 2,
    "total_parameters": 150000000
  }
}
```

### Model Cache Information

Get detailed cache information with TTS-specific details.

**Request:**
```http
GET /models/cache/info
```

**Response:**
```json
{
  "cache_directory": "/app/models_cache",
  "total_models": 5,
  "total_size_mb": 8500,
  "models": ["speecht5_tts", "bark_tts", "example"],
  "tts_specific": {
    "tts_models_count": 2,
    "tts_models": [
      {
        "name": "speecht5_tts",
        "size_mb": 2500,
        "source": "huggingface",
        "task": "text-to-speech",
        "tags": ["speecht5", "microsoft", "tts"]
      }
    ],
    "tts_total_size_mb": 6500,
    "tts_percentage": 76.5
  },
  "cache_optimization": {
    "cache_hit_rate": 0.85,
    "optimization_enabled": true,
    "auto_cleanup": true
  }
}
```

### Model Management

Manage models (retry downloads, optimize, etc.).

**Request:**
```http
POST /models/manage?action=optimize&model_name=speecht5_tts
```

**Response:**
```json
{
  "success": true,
  "message": "Model 'speecht5_tts' optimized successfully",
  "action": "optimize",
  "model_name": "speecht5_tts"
}
```

---

## Audio Processing Endpoints

### Text-to-Speech Synthesis

Convert text to speech using TTS models.

**Request:**
```http
POST /synthesize
Content-Type: application/json

{
  "model_name": "speecht5_tts",
  "inputs": "Hello, this is a test of the text-to-speech system.",
  "token": "optional_auth_token",
  "voice": "default",
  "speed": 1.0,
  "pitch": 1.0,
  "volume": 1.0,
  "language": "en",
  "emotion": null,
  "output_format": "wav"
}
```

**Response:**
```json
{
  "success": true,
  "audio_data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
  "audio_format": "wav",
  "duration": 4.2,
  "sample_rate": 16000,
  "processing_time": 0.85,
  "model_info": {
    "model_name": "speecht5_tts",
    "voice": "default",
    "language": "en",
    "actual_model": "microsoft/speecht5_tts"
  }
}
```

### Speech-to-Text Transcription

Transcribe audio to text using STT models.

**Request:**
```http
POST /stt/transcribe
Content-Type: multipart/form-data

form-data:
- file: [audio file]
- model_name: "whisper-base"
- language: "auto"
- enable_timestamps: true
- beam_size: 5
- temperature: 0.0
- suppress_blank: true
```

**Response:**
```json
{
  "success": true,
  "text": "Hello, this is a test of the speech-to-text system.",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, this is a test",
      "confidence": 0.95
    },
    {
      "start": 2.5,
      "end": 4.2,
      "text": "of the speech-to-text system.",
      "confidence": 0.92
    }
  ],
  "language": "en",
  "confidence": 0.94,
  "processing_time": 1.25,
  "model_info": {
    "model_name": "whisper-base",
    "language": "en",
    "file_name": "audio.wav"
  }
}
```

### Audio Models

List available audio models.

**Request:**
```http
GET /audio/models
```

**Response:**
```json
{
  "tts_models": {
    "speecht5_tts": {
      "type": "huggingface",
      "model_name": "microsoft/speecht5_tts",
      "description": "Microsoft SpeechT5 TTS model"
    },
    "bark": {
      "type": "huggingface",
      "model_name": "suno/bark",
      "description": "Suno Bark TTS model"
    }
  },
  "stt_models": {
    "whisper-base": {
      "type": "whisper",
      "model_size": "base",
      "description": "OpenAI Whisper Base model"
    }
  },
  "loaded_models": ["speecht5_tts"],
  "supported_tts_types": ["huggingface", "torchaudio", "custom"],
  "supported_stt_types": ["whisper", "wav2vec2", "custom"],
  "examples": {
    "tts_request": {
      "model_name": "speecht5_tts",
      "alternatives": ["speecht5", "bark", "tacotron2", "default"]
    },
    "stt_request": {
      "model_name": "whisper-base",
      "alternatives": ["whisper-small", "whisper-medium"]
    }
  }
}
```

### Audio Health Check

Check audio processing capabilities.

**Request:**
```http
GET /audio/health
```

**Response:**
```json
{
  "audio_available": true,
  "tts_available": true,
  "stt_available": true,
  "dependencies": {
    "librosa": {
      "available": true,
      "description": "Audio processing"
    },
    "transformers": {
      "available": true,
      "description": "HuggingFace models"
    }
  },
  "errors": []
}
```

### TTS Health Check

Check TTS service health with available voices.

**Request:**
```http
GET /tts/health
```

**Response:**
```json
{
  "status": "healthy",
  "available_voices": ["default", "female", "male"],
  "supported_languages": ["en", "es", "fr", "de", "it"],
  "optimizations_enabled": [
    "model_caching",
    "gpu_acceleration",
    "batch_synthesis",
    "audio_optimization"
  ],
  "loaded_tts_models": ["speecht5_tts"],
  "capabilities": {
    "text_to_speech": true,
    "voice_cloning": false,
    "emotion_synthesis": false,
    "streaming": false,
    "real_time": true
  }
}
```

### Audio Validation

Validate audio file integrity.

**Request:**
```http
POST /audio/validate?file_path=/path/to/audio.wav&validate_format=true&check_integrity=true
```

**Response:**
```json
{
  "valid": true,
  "file_path": "/path/to/audio.wav",
  "size_bytes": 352844,
  "format": "wav",
  "format_valid": true,
  "duration": 4.2,
  "sample_rate": 16000,
  "estimated": false
}
```

---

## GPU and Hardware Endpoints

### GPU Detection

Detect available GPUs and their capabilities.

**Request:**
```http
GET /gpu/detect?include_benchmarks=false
```

**Response:**
```json
{
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA GeForce RTX 4080",
      "vendor": "nvidia",
      "architecture": "ada_lovelace",
      "device_id": 0,
      "memory_mb": 16384,
      "available_memory_mb": 14500,
      "pytorch_support": true,
      "suitable_for_inference": true,
      "recommended_precisions": ["fp16", "int8"],
      "supported_accelerators": ["tensorrt", "cuda"],
      "compute_capability": {
        "major": 8,
        "minor": 9,
        "version": "8.9",
        "supports_fp16": true,
        "supports_int8": true,
        "supports_tensor_cores": true,
        "supports_tf32": true
      }
    }
  ],
  "total_gpus": 1,
  "suitable_gpus": 1,
  "include_benchmarks": false
}
```

### Best GPU

Get the best GPU for inference.

**Request:**
```http
GET /gpu/best
```

**Response:**
```json
{
  "best_gpu": {
    "id": 0,
    "name": "NVIDIA GeForce RTX 4080",
    "vendor": "nvidia",
    "architecture": "ada_lovelace",
    "device_id": 0,
    "memory_mb": 16384,
    "available_memory_mb": 14500,
    "pytorch_support": true,
    "recommended_precisions": ["fp16", "int8"],
    "estimated_max_batch_size": 32,
    "compute_capability": {
      "version": "8.9",
      "supports_tensor_cores": true,
      "supports_fp16": true
    }
  },
  "message": "Best GPU for inference: NVIDIA GeForce RTX 4080"
}
```

### GPU Configuration

Get GPU-optimized configuration.

**Request:**
```http
GET /gpu/config
```

**Response:**
```json
{
  "device_config": {
    "device_type": "cuda",
    "device_id": 0,
    "use_fp16": true,
    "use_int8": false,
    "use_tensorrt": true,
    "use_torch_compile": true,
    "compile_mode": "max-autotune"
  },
  "memory_recommendations": {
    "batch_size": 16,
    "max_batch_size": 32,
    "memory_fraction": 0.9
  },
  "optimization_recommendations": {
    "recommended_optimizers": ["tensorrt", "torch_compile", "fp16"],
    "aggressive_optimizations": ["int8_quantization"]
  },
  "pytorch_device": "cuda:0"
}
```

### GPU Report

Get comprehensive GPU report.

**Request:**
```http
GET /gpu/report?format=json
```

**Response:**
```json
{
  "summary": {
    "total_gpus": 1,
    "suitable_gpus": 1,
    "best_gpu": "NVIDIA GeForce RTX 4080"
  },
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA GeForce RTX 4080",
      "vendor": "nvidia",
      "architecture": "ada_lovelace",
      "device_id": 0,
      "memory_mb": 16384,
      "available_memory_mb": 14500,
      "pytorch_support": true,
      "suitable_for_inference": true,
      "recommended_precisions": ["fp16", "int8"],
      "estimated_max_batch_size": 32
    }
  ],
  "device_config": {
    "device_type": "cuda",
    "device_id": 0,
    "use_fp16": true,
    "use_tensorrt": true
  },
  "recommendations": {
    "memory": {
      "batch_size": 16,
      "max_batch_size": 32
    },
    "optimization": {
      "recommended_optimizers": ["tensorrt", "fp16"]
    }
  }
}
```

---

## Autoscaling Endpoints

### Autoscaler Statistics

Get autoscaler performance statistics.

**Request:**
```http
GET /autoscaler/stats
```

**Response:**
```json
{
  "zero_scaler": {
    "enabled": true,
    "models_scaled_to_zero": 2,
    "scale_to_zero_events": 15,
    "avg_scale_up_time": 2.5,
    "preloaded_models": ["popular_model_1"]
  },
  "model_loader": {
    "total_load_requests": 150,
    "successful_loads": 148,
    "failed_loads": 2,
    "avg_load_time": 1.8,
    "load_balancing_strategy": "least_connections",
    "active_instances": {
      "model_a": 2,
      "model_b": 1
    }
  },
  "performance": {
    "predictions_per_second": 450,
    "avg_response_time": 0.025,
    "queue_depth": 3,
    "memory_usage_mb": 2048
  },
  "health": {
    "healthy": true,
    "last_health_check": 1705315800.123
  }
}
```

### Autoscaler Health

Check autoscaler health status.

**Request:**
```http
GET /autoscaler/health
```

**Response:**
```json
{
  "healthy": true,
  "components": {
    "zero_scaler": {
      "status": "healthy",
      "last_activity": 1705315800.123
    },
    "model_loader": {
      "status": "healthy",
      "active_loaders": 2
    },
    "metrics_collector": {
      "status": "healthy",
      "events_processed": 1234
    }
  },
  "timestamp": 1705315800.123
}
```

### Scale Model

Scale a model to target number of instances.

**Request:**
```http
POST /autoscaler/scale?model_name=my_model&target_instances=3
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully scaled my_model to 3 instances",
  "model_name": "my_model",
  "target_instances": 3
}
```

### Load Model (Autoscaler)

Load a model through the autoscaler.

**Request:**
```http
POST /autoscaler/load?model_name=new_model&version=v1
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully loaded new_model:v1",
  "model_name": "new_model",
  "version": "v1"
}
```

### Unload Model (Autoscaler)

Unload a model through the autoscaler.

**Request:**
```http
DELETE /autoscaler/unload?model_name=old_model&version=v1
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully unloaded old_model:v1",
  "model_name": "old_model",
  "version": "v1"
}
```

### Autoscaler Metrics

Get detailed autoscaling metrics.

**Request:**
```http
GET /autoscaler/metrics?window_seconds=300
```

**Response:**
```json
{
  "metrics": {
    "zero_scaler": {
      "enabled": true,
      "models_scaled_to_zero": 2,
      "scale_events": 15
    },
    "model_loader": {
      "total_loads": 150,
      "success_rate": 0.987,
      "avg_load_time": 1.8
    },
    "detailed_metrics": {
      "prediction_rate": [450, 445, 460, 455],
      "response_times": [0.025, 0.028, 0.022, 0.026],
      "memory_usage": [2048, 2056, 2051, 2049]
    }
  },
  "window_seconds": 300,
  "timestamp": 1705315800.123
}
```

---

## Server Management Endpoints

### Server Configuration

Get server configuration including TTS settings.

**Request:**
```http
GET /server/config
```

**Response:**
```json
{
  "optimization_level": "high",
  "caching_strategy": "aggressive",
  "tts_backend": "huggingface_transformers",
  "auto_optimization": true,
  "server_features": [
    "model_caching",
    "auto_optimization",
    "tts_synthesis",
    "batch_processing",
    "gpu_acceleration"
  ],
  "tts_configuration": {
    "default_models": {
      "tts": "speecht5_tts",
      "vocoder": "microsoft/speecht5_hifigan"
    },
    "supported_formats": ["wav", "mp3", "flac"],
    "max_text_length": 5000,
    "default_sample_rate": 16000,
    "auto_model_download": true
  },
  "performance_settings": {
    "enable_model_compilation": true,
    "enable_fp16": true,
    "batch_optimization": true,
    "memory_management": "auto"
  }
}
```

### Server Optimization

Optimize server performance and memory usage.

**Request:**
```http
POST /server/optimize
```

**Response:**
```json
{
  "success": true,
  "memory_freed_mb": 256,
  "models_optimized": 3,
  "optimizations_applied": [
    "gpu_cache_clear",
    "model_optimization",
    "memory_cleanup",
    "cache_optimization"
  ]
}
```

### Server Metrics

Get server performance metrics.

**Request:**
```http
GET /metrics/server
```

**Response:**
```json
{
  "cache_hit_rate": 0.87,
  "active_optimizations": [
    "model_compilation",
    "memory_pooling",
    "batch_processing"
  ],
  "models_in_memory": 5,
  "system_metrics": {
    "cpu_percent": 25.5,
    "memory_percent": 45.2,
    "memory_available_gb": 8.5,
    "memory_total_gb": 16.0
  },
  "gpu_metrics": {
    "gpu_available": true,
    "gpu_count": 1,
    "current_device": 0,
    "memory_allocated_mb": 2048,
    "memory_reserved_mb": 2560,
    "gpu_utilization": 85.0
  },
  "tts_metrics": {
    "tts_models_loaded": 2,
    "tts_models": ["speecht5_tts", "bark_tts"],
    "avg_synthesis_time_ms": 850,
    "synthesis_requests_total": 125
  }
}
```

### TTS Metrics

Get TTS-specific performance metrics.

**Request:**
```http
GET /metrics/tts
```

**Response:**
```json
{
  "requests_processed": 125,
  "avg_processing_time": 0.85,
  "success_rate": 0.95,
  "models_performance": {
    "speecht5_tts": {
      "avg_time_ms": 800,
      "success_rate": 0.98,
      "quality_score": 4.5
    },
    "bark_tts": {
      "avg_time_ms": 2500,
      "success_rate": 0.92,
      "quality_score": 4.8
    }
  },
  "audio_stats": {
    "total_audio_generated_minutes": 45.2,
    "avg_audio_length_seconds": 5.2,
    "formats_used": {
      "wav": 0.8,
      "mp3": 0.15,
      "flac": 0.05
    }
  },
  "optimization_stats": {
    "cache_hits": 45,
    "gpu_accelerated_requests": 120,
    "batch_processed_requests": 15
  }
}
```

---

## Logging Endpoints

### Logging Information

Get logging configuration and statistics.

**Request:**
```http
GET /logs
```

**Response:**
```json
{
  "log_directory": "/app/logs",
  "log_level": "INFO",
  "available_log_files": [
    {
      "name": "server.log",
      "path": "/app/logs/server.log",
      "size_bytes": 16058624,
      "size_mb": 15.3,
      "line_count": 45230,
      "last_modified": "2024-01-15T10:25:00Z",
      "created": "2024-01-15T08:00:00Z"
    },
    {
      "name": "api_requests.log",
      "path": "/app/logs/api_requests.log",
      "size_bytes": 5242880,
      "size_mb": 5.0,
      "line_count": 12450,
      "last_modified": "2024-01-15T10:30:00Z",
      "created": "2024-01-15T08:00:00Z"
    }
  ],
  "total_log_size_mb": 25.5
}
```

### Get Log File

View or download specific log file.

**Request:**
```http
GET /logs/server.log?lines=100&from_end=true
```

**Response:**
```
2024-01-15 10:30:15,123 - main - INFO - [API REQUEST] GET /health - Client: 127.0.0.1
2024-01-15 10:30:15,125 - main - INFO - [ENDPOINT] Health check requested
2024-01-15 10:30:15,128 - main - INFO - [ENDPOINT] Health check completed - Healthy: True
...
```

**Headers:**
```
Content-Type: text/plain
Content-Disposition: inline; filename=server.log
X-Total-Lines: 100
X-File-Size: 16058624
```

### Clear Log File

Clear specific log file.

**Request:**
```http
DELETE /logs/server.log
```

**Response:**
```json
{
  "success": true,
  "message": "Log file server.log cleared successfully",
  "original_size_bytes": 16058624,
  "original_size_mb": 15.3
}
```

---

## Request/Response Models

### Common Request Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `inputs` | Any | Input data for inference | Required |
| `priority` | int | Request priority (0-10) | 0 |
| `timeout` | float | Timeout in seconds | 30.0 |
| `enable_batching` | bool | Enable batch optimization | true |

### Common Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Operation success status |
| `result` | Any | Operation result data |
| `error` | string | Error message (if failed) |
| `processing_time` | float | Processing time in seconds |
| `timestamp` | float | Unix timestamp |

### Error Response Format

```json
{
  "detail": "Error description",
  "status_code": 400,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider implementing rate limiting based on:
- Requests per second per client
- Concurrent requests per client
- Model-specific rate limits
- Resource-based throttling

## WebSocket Support

WebSocket endpoints are planned for future releases to support:
- Real-time inference streaming
- Live audio processing
- Real-time model metrics
- Interactive model management

## API Versioning

Current API version: `v1` (implied in base URL)
Future versions will use path-based versioning: `/v2/predict`

---

*For more examples and advanced usage, see the [Examples](../examples/) section.*

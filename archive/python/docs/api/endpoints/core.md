# Core Inference Endpoints

The core inference endpoints provide the primary functionality for model prediction and system monitoring.

## Overview

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|----------------|
| `/` | GET | API information and capabilities | None |
| `/predict` | POST | Unified prediction for all models | Optional |
| `/health` | GET | System health with autoscaler info | None |
| `/info` | GET | Comprehensive system information | None |
| `/stats` | GET | Performance statistics | Required |
| `/config` | GET | Configuration information | None |

---

## Root Endpoint

Get API information, available endpoints, and system capabilities.

### Request
```http
GET /
```

### Response
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
      "predict": "/predict"
    },
    "audio": {
      "synthesize": "/synthesize",
      "transcribe": "/transcribe"
    },
    "models": {
      "list": "/models",
      "download": "/models/download"
    }
  }
}
```

### Example
```bash
curl http://localhost:8000/
```

---

## Unified Prediction

Perform inference using any torch model with optimized processing.

### Request
```http
POST /predict
Content-Type: application/json

{
  "model_name": "example",
  "inputs": [1, 2, 3, 4, 5],
  "token": "optional_auth_token",
  "priority": 0,
  "timeout": 30.0,
  "enable_batching": true
}
```

### Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_name` | string | Yes | - | Name of model to use |
| `inputs` | any/array | Yes | - | Input data (single or batch) |
| `token` | string | No | null | Authentication token |
| `priority` | integer | No | 0 | Request priority (0-10) |
| `timeout` | float | No | 30.0 | Timeout in seconds |
| `enable_batching` | boolean | No | true | Enable batch optimization |

### Response
```json
{
  "success": true,
  "result": 0.75,
  "processing_time": 0.025,
  "model_info": {
    "model": "example",
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
```

### Batch Request Example
```json
{
  "model_name": "example",
  "inputs": [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15]
  ],
  "priority": 1,
  "timeout": 30.0
}
```

### Examples

#### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "example",
    "inputs": [1, 2, 3, 4, 5]
  }'
```

#### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "example",
    "inputs": [[1,2,3], [4,5,6], [7,8,9]]
  }'
```

#### With Authentication
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "model_name": "example",
    "inputs": "Hello world",
    "token": "optional_token_override"
  }'
```

---

## Health Check

Check system health and component status, including autoscaler information.

### Request
```http
GET /health
```

### Response
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
  },
  "autoscaler": {
    "healthy": true,
    "state": "running",
    "components": {
      "zero_scaler": {
        "status": "healthy",
        "last_activity": 1705315800.123
      },
      "model_loader": {
        "status": "healthy",
        "active_loaders": 2
      }
    }
  }
}
```

### Example
```bash
curl http://localhost:8000/health
```

---

## System Information

Get comprehensive system information including performance metrics and TTS capabilities.

### Request
```http
GET /info
```

### Response
```json
{
  "timestamp": "2025-09-05T13:34:20.123456",
  "server_config": {
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
    ]
  },
  "performance_metrics": {
    "cache_hit_rate": 0.87,
    "models_in_memory": 3,
    "system_metrics": {
      "cpu_percent": 45.2,
      "memory_percent": 68.5,
      "memory_available_gb": 12.3
    },
    "gpu_metrics": {
      "gpu_available": true,
      "gpu_count": 1,
      "memory_allocated_mb": 2048.5,
      "gpu_utilization": 85.0
    },
    "tts_metrics": {
      "tts_models_loaded": 2,
      "avg_synthesis_time_ms": 850,
      "synthesis_requests_total": 42
    }
  },
  "tts_service": {
    "status": "healthy",
    "available_voices": ["default", "female", "male"],
    "supported_languages": ["en", "es", "fr", "de", "it"],
    "loaded_tts_models": ["speecht5_tts", "bark_tts"],
    "capabilities": {
      "text_to_speech": true,
      "voice_cloning": true,
      "emotion_synthesis": true,
      "real_time": true
    }
  }
}
```

### Example
```bash
curl http://localhost:8000/info
```

---

## Performance Statistics

Get detailed performance metrics and statistics. Requires authentication.

### Request
```http
GET /stats
Authorization: Bearer YOUR_JWT_TOKEN
```

### Response
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

### Example
```bash
curl http://localhost:8000/stats \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

---

## Configuration Information

Get current system configuration including logging and device settings.

### Request
```http
GET /config
```

### Response
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

### Example
```bash
curl http://localhost:8000/config
```

---

## Error Handling

All endpoints return consistent error responses:

### Error Response Format
```json
{
  "success": false,
  "error": "Error description",
  "detail": "Detailed error information",
  "status_code": 400
}
```

### Common HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (authentication required)
- `404`: Not Found (model/resource not found)
- `422`: Validation Error (request validation failed)
- `500`: Internal Server Error
- `503`: Service Unavailable (engine not ready)

### Example Error Responses

#### Model Not Found
```json
{
  "success": false,
  "error": "Model 'unknown_model' is not available. Check models.json configuration.",
  "model_info": {
    "model": "unknown_model",
    "available": false
  }
}
```

#### Timeout Error
```json
{
  "success": false,
  "error": "Request timed out",
  "processing_time": 30.0
}
```

#### Authentication Error
```json
{
  "detail": "Not authenticated",
  "status_code": 401
}
```

---

## Performance Optimization

### Request Optimization
1. **Batch Processing**: Send multiple inputs in a single request
2. **Enable Batching**: Set `enable_batching: true` for optimal throughput
3. **Priority Queuing**: Use `priority` parameter for important requests
4. **Timeout Management**: Set appropriate `timeout` values

### Response Caching
The system automatically caches frequently accessed endpoints to improve response times.

### Device Optimization
The system automatically selects the best available device (GPU/CPU) and applies optimizations like:
- FP16 precision for compatible GPUs
- TensorRT acceleration for NVIDIA GPUs
- Torch compilation for modern GPUs
- Batch processing optimization

---

## Rate Limiting

Rate limiting may be applied based on configuration:
- Requests per second per client
- Concurrent requests per client
- Model-specific rate limits
- Resource-based throttling

Check the `/config` endpoint for current rate limiting settings.

---

## Monitoring and Debugging

### Health Monitoring
Use `/health` endpoint to monitor system status and component health.

### Performance Monitoring
Use `/stats` endpoint to track performance metrics and identify bottlenecks.

### System Information
Use `/info` endpoint to get comprehensive system information including resource usage.

### Configuration Debugging
Use `/config` endpoint to verify current configuration settings.

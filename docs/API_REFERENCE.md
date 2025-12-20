# API Reference

Complete REST API reference for Torch Inference Server.

## Base URL

```
http://localhost:8000/api
```

## Authentication

Most endpoints require JWT authentication.

### Login

**Endpoint**: `POST /api/auth/login`

**Request**:
```json
{
  "username": "admin",
  "password": "admin"
}
```

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

### Using Auth Token

Include in header:
```
Authorization: Bearer YOUR_TOKEN_HERE
```

## Health & System

### Health Check

**Endpoint**: `GET /api/health`

**Auth**: Not required

**Response**:
```json
{
  "status": "healthy",
  "uptime": "3h 45m 12s",
  "memory_used_mb": 1024,
  "cache_size": 1500,
  "device_type": "cuda",
  "model_count": 3
}
```

### System Information

**Endpoint**: `GET /api/system/info`

**Auth**: Not required

**Response**:
```json
{
  "version": "1.0.0",
  "rust_version": "1.75.0",
  "os": "Linux",
  "arch": "x86_64",
  "cpu_count": 16,
  "total_memory_mb": 32768,
  "gpu_info": {
    "backend": "CUDA",
    "device_count": 2,
    "devices": [
      {
        "id": 0,
        "name": "NVIDIA A100",
        "memory_mb": 40960
      }
    ]
  }
}
```

### GPU Information

**Endpoint**: `GET /api/system/gpu`

**Auth**: Not required

**Response**:
```json
{
  "backend": "CUDA",
  "available": true,
  "device_count": 2,
  "devices": [
    {
      "id": 0,
      "name": "NVIDIA A100",
      "compute_capability": "8.0",
      "total_memory_mb": 40960,
      "free_memory_mb": 38912,
      "temperature": 42,
      "utilization": 15
    }
  ]
}
```

## Models

### List Models

**Endpoint**: `GET /api/models`

**Auth**: Not required

**Response**:
```json
{
  "models": [
    {
      "name": "resnet50",
      "version": "1.0",
      "backend": "PyTorch",
      "loaded": true,
      "memory_mb": 102,
      "last_used": "2024-12-20T10:30:00Z"
    }
  ],
  "total": 1,
  "loaded": 1
}
```

### Load Model

**Endpoint**: `POST /api/models/load`

**Auth**: Required

**Request**:
```json
{
  "model_name": "resnet50",
  "version": "1.0"
}
```

**Response**:
```json
{
  "success": true,
  "model_name": "resnet50",
  "memory_mb": 102,
  "load_time_ms": 1523
}
```

### Unload Model

**Endpoint**: `POST /api/models/unload`

**Auth**: Required

**Request**:
```json
{
  "model_name": "resnet50"
}
```

**Response**:
```json
{
  "success": true,
  "model_name": "resnet50",
  "memory_freed_mb": 102
}
```

## Model Registry

### List Available Models

**Endpoint**: `GET /api/registry/models`

**Auth**: Not required

**Response**:
```json
{
  "models": [
    {
      "name": "resnet50",
      "description": "ResNet-50 image classification",
      "task": "image-classification",
      "size_mb": 102,
      "supported_backends": ["PyTorch", "ONNX"],
      "source": "torchvision"
    }
  ],
  "total": 1
}
```

### Download Model

**Endpoint**: `POST /api/models/download`

**Auth**: Required

**Request**:
```json
{
  "model_name": "resnet50",
  "backend": "PyTorch"
}
```

**Response**:
```json
{
  "success": true,
  "model_name": "resnet50",
  "download_size_mb": 102,
  "download_time_ms": 5421
}
```

## Inference

### Generic Inference

**Endpoint**: `POST /api/inference`

**Auth**: Required

**Request** (JSON):
```json
{
  "model_name": "resnet50",
  "input": {
    "image": "base64_encoded_image_data"
  },
  "options": {
    "batch_size": 1,
    "use_cache": true
  }
}
```

**Response**:
```json
{
  "output": [...],
  "inference_time_ms": 15,
  "from_cache": false
}
```

## Image Classification

### Classify Image

**Endpoint**: `POST /api/classify`

**Auth**: Required

**Request** (Multipart Form):
```
image: <file>
top_k: 5 (optional)
```

**cURL Example**:
```bash
curl -X POST http://localhost:8000/api/classify \
  -H "Authorization: Bearer TOKEN" \
  -F "image=@cat.jpg" \
  -F "top_k=5"
```

**Response**:
```json
{
  "predictions": [
    {
      "class": "tabby_cat",
      "confidence": 0.92,
      "class_id": 281
    },
    {
      "class": "egyptian_cat",
      "confidence": 0.05,
      "class_id": 285
    }
  ],
  "inference_time_ms": 12,
  "from_cache": false
}
```

## Object Detection (YOLO)

### Detect Objects

**Endpoint**: `POST /api/yolo/detect`

**Auth**: Required

**Request** (Multipart Form):
```
image: <file>
confidence_threshold: 0.5 (optional)
iou_threshold: 0.45 (optional)
```

**Response**:
```json
{
  "detections": [
    {
      "class": "person",
      "confidence": 0.89,
      "bbox": {
        "x": 120,
        "y": 50,
        "width": 200,
        "height": 400
      }
    },
    {
      "class": "dog",
      "confidence": 0.76,
      "bbox": {
        "x": 350,
        "y": 200,
        "width": 150,
        "height": 180
      }
    }
  ],
  "inference_time_ms": 28,
  "from_cache": false
}
```

## Text-to-Speech

### Synthesize Speech

**Endpoint**: `POST /api/tts/synthesize`

**Auth**: Required

**Request**:
```json
{
  "text": "Hello, world!",
  "engine": "piper",
  "voice": "en_US-amy-medium",
  "speed": 1.0,
  "pitch": 1.0
}
```

**Response**: Audio file (WAV format)

**Response Headers**:
```
Content-Type: audio/wav
Content-Disposition: attachment; filename="speech.wav"
```

### List TTS Engines

**Endpoint**: `GET /api/tts/engines`

**Auth**: Not required

**Response**:
```json
{
  "engines": [
    {
      "name": "piper",
      "description": "Piper TTS - Fast neural TTS",
      "available": true,
      "voices": [
        "en_US-amy-medium",
        "en_US-ryan-high"
      ]
    },
    {
      "name": "kokoro",
      "description": "Kokoro ONNX TTS",
      "available": true,
      "voices": ["default"]
    }
  ]
}
```

## Audio Processing

### Speech Recognition

**Endpoint**: `POST /api/audio/transcribe`

**Auth**: Required

**Request** (Multipart Form):
```
audio: <file>
language: en (optional)
```

**Response**:
```json
{
  "text": "This is the transcribed text",
  "language": "en",
  "confidence": 0.94,
  "processing_time_ms": 1250
}
```

## Monitoring & Metrics

### Cache Statistics

**Endpoint**: `GET /api/stats/cache`

**Auth**: Not required

**Response**:
```json
{
  "size": 1500,
  "hits": 12450,
  "misses": 2340,
  "hit_rate": 0.842,
  "memory_used_mb": 256,
  "evictions": 45
}
```

### Batch Statistics

**Endpoint**: `GET /api/stats/batch`

**Auth**: Not required

**Response**:
```json
{
  "total_batches": 523,
  "total_requests": 4821,
  "avg_batch_size": 9.2,
  "avg_wait_time_ms": 15,
  "queue_depth": 3
}
```

### Tensor Pool Statistics

**Endpoint**: `GET /api/stats/tensor_pool`

**Auth**: Not required

**Response**:
```json
{
  "total_tensors": 450,
  "active_tensors": 23,
  "reuse_rate": 0.96,
  "memory_saved_mb": 1250
}
```

### Monitor Statistics

**Endpoint**: `GET /api/stats/monitor`

**Auth**: Not required

**Response**:
```json
{
  "request_count": 15234,
  "avg_latency_ms": 18.5,
  "min_latency_ms": 5,
  "max_latency_ms": 142,
  "p95_latency_ms": 45,
  "p99_latency_ms": 89,
  "error_count": 12,
  "error_rate": 0.0008,
  "throughput_rps": 125.3
}
```

### Endpoint Statistics

**Endpoint**: `GET /api/stats/endpoints`

**Auth**: Not required

**Response**:
```json
{
  "endpoints": [
    {
      "path": "/api/classify",
      "request_count": 5420,
      "avg_latency_ms": 15.2,
      "error_rate": 0.001
    },
    {
      "path": "/api/yolo/detect",
      "request_count": 3241,
      "avg_latency_ms": 28.7,
      "error_rate": 0.002
    }
  ]
}
```

## Prometheus Metrics

**Endpoint**: `GET /metrics`

**Auth**: Not required

**Format**: Prometheus text format

**Example**:
```
# HELP torch_inference_requests_total Total number of requests
# TYPE torch_inference_requests_total counter
torch_inference_requests_total 15234

# HELP torch_inference_latency_seconds Request latency in seconds
# TYPE torch_inference_latency_seconds histogram
torch_inference_latency_seconds_bucket{le="0.01"} 8523
torch_inference_latency_seconds_bucket{le="0.05"} 14231
torch_inference_latency_seconds_bucket{le="0.1"} 15102
```

## Configuration

### Get Configuration

**Endpoint**: `GET /api/config`

**Auth**: Required

**Response**:
```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 8
  },
  "device": {
    "device_type": "cuda",
    "use_fp16": true
  },
  "performance": {
    "enable_caching": true,
    "cache_size_mb": 2048,
    "max_batch_size": 32
  }
}
```

### Update Configuration (Hot-reload)

**Endpoint**: `POST /api/config/update`

**Auth**: Required

**Request**:
```json
{
  "performance": {
    "cache_size_mb": 4096,
    "max_batch_size": 64
  }
}
```

**Response**:
```json
{
  "success": true,
  "updated_fields": ["cache_size_mb", "max_batch_size"],
  "restart_required": false
}
```

## Logging

### Get Logs

**Endpoint**: `GET /api/logs`

**Auth**: Required

**Query Parameters**:
- `level`: debug, info, warn, error
- `limit`: Number of entries (default: 100)
- `since`: ISO 8601 timestamp

**Response**:
```json
{
  "logs": [
    {
      "timestamp": "2024-12-20T10:30:15Z",
      "level": "info",
      "message": "Request processed",
      "fields": {
        "latency_ms": 15,
        "endpoint": "/api/classify"
      }
    }
  ],
  "total": 1523,
  "returned": 100
}
```

### Set Log Level

**Endpoint**: `POST /api/logs/level`

**Auth**: Required

**Request**:
```json
{
  "level": "debug"
}
```

**Response**:
```json
{
  "success": true,
  "previous_level": "info",
  "new_level": "debug"
}
```

## Performance

### Warmup

**Endpoint**: `POST /api/performance/warmup`

**Auth**: Required

**Request**:
```json
{
  "model_name": "resnet50",
  "iterations": 10
}
```

**Response**:
```json
{
  "success": true,
  "iterations": 10,
  "avg_time_ms": 12.5,
  "min_time_ms": 11.2,
  "max_time_ms": 18.7
}
```

### Benchmark

**Endpoint**: `POST /api/performance/benchmark`

**Auth**: Required

**Request**:
```json
{
  "model_name": "resnet50",
  "duration_seconds": 30,
  "concurrency": 10
}
```

**Response**:
```json
{
  "total_requests": 15234,
  "duration_seconds": 30,
  "throughput_rps": 507.8,
  "avg_latency_ms": 19.7,
  "p95_latency_ms": 45.2,
  "p99_latency_ms": 89.3,
  "error_rate": 0.0005
}
```

## Error Responses

### Standard Error Format

```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model 'invalid_model' not found",
    "details": {
      "model_name": "invalid_model",
      "available_models": ["resnet50", "yolo"]
    }
  },
  "request_id": "req_abc123",
  "timestamp": "2024-12-20T10:30:00Z"
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created |
| 400 | Bad Request | Invalid input |
| 401 | Unauthorized | Missing/invalid auth |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Circuit breaker open |

### Error Codes

| Code | Description |
|------|-------------|
| `MODEL_NOT_FOUND` | Requested model not found |
| `INVALID_INPUT` | Invalid request parameters |
| `INFERENCE_TIMEOUT` | Inference took too long |
| `RESOURCE_EXHAUSTED` | Out of memory/resources |
| `CIRCUIT_BREAKER_OPEN` | Service temporarily unavailable |
| `UNAUTHORIZED` | Authentication required |
| `RATE_LIMIT_EXCEEDED` | Too many requests |

## Rate Limiting

**Headers**:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 945
X-RateLimit-Reset: 1640000000
```

**Default Limits**:
- 1000 requests per minute per IP
- 10000 requests per minute per authenticated user

## Pagination

For list endpoints:

**Query Parameters**:
```
?page=1&per_page=20
```

**Response**:
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total_pages": 5,
    "total_items": 100
  }
}
```

## Request IDs

All requests receive a unique correlation ID:

**Response Header**:
```
X-Request-ID: req_abc123def456
```

Use this ID for debugging and log correlation.

## Compression

Responses are automatically compressed when:
- Response size > 1KB
- Client sends `Accept-Encoding: gzip`

**Response Header**:
```
Content-Encoding: gzip
```

## CORS

CORS is enabled by default for development.

**Production Configuration**:
```toml
[server]
cors_origins = ["https://app.example.com"]
cors_methods = ["GET", "POST"]
```

## WebSocket (Future)

Coming soon: Real-time inference updates

```
ws://localhost:8000/ws/inference
```

---

**Next**: See [Testing Guide](TESTING.md) for API testing examples.

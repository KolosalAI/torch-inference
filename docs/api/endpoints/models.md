# Model Management Endpoints

The model management endpoints provide functionality for downloading, loading, and managing machine learning models.

## Overview

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|----------------|
| `/models` | GET | List available models | Optional |
| `/models/download` | POST | Download and load models | None |
| `/models/available` | GET | List downloadable models | None |
| `/models/managed` | GET | Server-managed model info | None |
| `/models/cache/info` | GET | Model cache information | None |
| `/models/download/status/{id}` | GET | Check download status | None |
| `/models/manage` | POST | Manage models (retry, optimize) | None |
| `/models/download/{model_name}` | DELETE | Remove model from cache | None |

---

## List Available Models

Get list of currently available and configured models from models.json configuration.

### Request
```http
GET /models
```

### Response
```json
{
  "success": true,
  "timestamp": "2024-01-15T10:30:00Z",
  "total_models": 15,
  "available_models": [
    "example",
    "speecht5_tts",
    "bark_tts",
    "resnet50"
  ],
  "models_info": {
    "speecht5_tts": {
      "display_name": "SpeechT5 TTS Model",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "tts_enabled": true
    }
  },
  "specialized_models": {
    "tts_models": ["speecht5_tts", "bark_tts"],
    "stt_models": ["whisper-base"],
    "image_classification": ["resnet50"],
    "text_classification": ["bert-base"]
  },
  "audio_models": {
    "tts_detailed": {
      "speecht5_tts": {
        "type": "huggingface",
        "model_name": "microsoft/speecht5_tts",
        "description": "Microsoft SpeechT5 TTS model"
      }
    },
    "loaded_audio_models": ["speecht5_tts"]
  }
}
```

### Example
```bash
curl http://localhost:8000/models
```

---

## Download Model

Download and load a model with enhanced TTS support and automatic configuration.

### Request
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

### Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source` | string | Yes | - | Model source (huggingface, pytorch_hub, torchvision, url, tts_auto) |
| `model_id` | string | Yes | - | Model identifier/path |
| `name` | string | Yes | - | Custom name for the model |
| `task` | string | No | "text-generation" | Task type (text-to-speech, text-generation, etc.) |
| `auto_convert_tts` | boolean | No | false | Auto-convert to TTS if applicable |
| `include_vocoder` | boolean | No | false | Include vocoder for TTS models |
| `vocoder_model` | string | No | null | Specific vocoder model |
| `enable_large_model` | boolean | No | false | Enable large model variants |
| `experimental` | boolean | No | false | Allow experimental models |

### Response
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

### Examples

#### Download SpeechT5 TTS Model
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

#### Download Bark TTS Model
```bash
curl -X POST http://localhost:8000/models/download \
  -H "Content-Type: application/json" \
  -d '{
    "source": "huggingface",
    "model_id": "suno/bark",
    "name": "bark_tts",
    "task": "text-to-speech"
  }'
```

#### Download BART Model for TTS
```bash
curl -X POST http://localhost:8000/models/download \
  -H "Content-Type: application/json" \
  -d '{
    "source": "huggingface",
    "model_id": "facebook/bart-large",
    "name": "bart_large_tts",
    "task": "text-generation",
    "auto_convert_tts": true
  }'
```

---

## List Available Downloads

Get enhanced list of models available for download with TTS focus.

### Request
```http
GET /models/available
```

### Response
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
    "tts_models": ["speecht5_tts", "bark_tts"],
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

### Example
```bash
curl http://localhost:8000/models/available
```

---

## Server-Managed Models

Get information about models currently managed by the server.

### Request
```http
GET /models/managed
```

### Response
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

### Example
```bash
curl http://localhost:8000/models/managed
```

---

## Model Cache Information

Get detailed cache information with TTS-specific details.

### Request
```http
GET /models/cache/info
```

### Response
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

### Example
```bash
curl http://localhost:8000/models/cache/info
```

---

## Download Status

Check the status of a model download by ID.

### Request
```http
GET /models/download/status/{download_id}
```

### Response
```json
{
  "download_id": "abc12345",
  "status": "completed",
  "progress": 100,
  "eta": null,
  "message": "Download completed successfully"
}
```

### Status Values
- `pending`: Download queued
- `downloading`: Download in progress
- `completed`: Download finished successfully
- `failed`: Download failed
- `cancelled`: Download was cancelled

### Example
```bash
curl http://localhost:8000/models/download/status/abc12345
```

---

## Model Management

Manage models (retry downloads, optimization, etc.).

### Request
```http
POST /models/manage?action=optimize&model_name=speecht5_tts
```

### Parameters
- `action`: Action to perform (optimize, retry_download, update)
- `model_name`: Name of the model to manage
- `force_redownload`: Force redownload if action is retry_download

### Response
```json
{
  "success": true,
  "message": "Model 'speecht5_tts' optimized successfully",
  "action": "optimize",
  "model_name": "speecht5_tts"
}
```

### Available Actions
- `optimize`: Optimize model for inference
- `retry_download`: Retry failed download
- `update`: Update model to latest version

### Examples

#### Optimize Model
```bash
curl -X POST "http://localhost:8000/models/manage?action=optimize&model_name=speecht5_tts"
```

#### Retry Download
```bash
curl -X POST "http://localhost:8000/models/manage?action=retry_download&model_name=failed_model&force_redownload=true"
```

---

## Remove Model

Remove a downloaded model from cache.

### Request
```http
DELETE /models/download/{model_name}
```

### Response
```json
{
  "message": "Successfully removed model from cache: old_model",
  "model_name": "old_model"
}
```

### Example
```bash
curl -X DELETE http://localhost:8000/models/download/old_model
```

---

## Supported Model Sources

### HuggingFace
- **Source**: `huggingface`
- **Models**: BERT, GPT, SpeechT5, Bark, VALL-E X
- **Authentication**: Optional (for private models)

### PyTorch Hub
- **Source**: `pytorch_hub`
- **Models**: ResNet, EfficientNet, MobileNet
- **Format**: Standard PyTorch models

### TorchVision
- **Source**: `torchvision`
- **Models**: Pre-trained vision models
- **Format**: TorchVision model zoo

### Custom URL
- **Source**: `url`
- **Format**: Direct download links
- **Supported**: `.pth`, `.pt`, `.bin` files

### TTS Auto
- **Source**: `tts_auto`
- **Models**: Automatic TTS model detection
- **Features**: Auto-configuration for TTS

---

## TTS Model Categories

### Popular TTS Models

#### SpeechT5
- **Model ID**: `microsoft/speecht5_tts`
- **Quality**: Very High
- **Speed**: Medium
- **Features**: Vocoder required, multi-language

#### Bark
- **Model ID**: `suno/bark`
- **Quality**: Very High
- **Speed**: Slow
- **Features**: Voice cloning, emotions, no vocoder needed

#### BART (Adapted for TTS)
- **Model ID**: `facebook/bart-large`
- **Quality**: High
- **Speed**: Medium
- **Features**: Requires TTS adaptation

#### Tacotron2
- **Model ID**: `tacotron2`
- **Source**: `torchaudio`
- **Quality**: High
- **Speed**: Fast
- **Features**: Requires WaveGlow vocoder

#### VALL-E X
- **Model ID**: `Plachtaa/VALL-E-X`
- **Quality**: Very High
- **Speed**: Slow
- **Features**: Zero-shot voice cloning, experimental

---

## Error Handling

### Common Errors

#### Model Already Exists
```json
{
  "success": true,
  "download_id": "abc12345",
  "message": "Model 'speecht5_tts' already exists and is ready to use",
  "status": "already_exists"
}
```

#### Invalid Source
```json
{
  "success": false,
  "message": "Invalid source. Must be one of: [pytorch_hub, torchvision, huggingface, url, tts_auto]",
  "error": "Invalid source: invalid_source"
}
```

#### Model Not Found
```json
{
  "success": false,
  "message": "Model not found in cache: unknown_model",
  "error": "Model does not exist"
}
```

#### Download Failed
```json
{
  "success": false,
  "message": "Failed to download model 'failed_model'",
  "status": "failed",
  "error": "Download task failed"
}
```

---

## Best Practices

### Model Selection
1. **Use recommended models** from `/models/available`
2. **Check TTS features** for audio applications
3. **Consider model size** vs quality trade-offs
4. **Verify GPU memory requirements**

### Download Optimization
1. **Enable vocoder inclusion** for TTS models
2. **Use background downloads** for large models
3. **Monitor download status** via status endpoint
4. **Check cache before downloading**

### Cache Management
1. **Monitor cache size** via `/models/cache/info`
2. **Remove unused models** to free space
3. **Use optimization** for better performance
4. **Regular cache cleanup**

### Performance Tips
1. **Optimize models** after download
2. **Use GPU acceleration** when available
3. **Enable batching** for multiple requests
4. **Monitor memory usage**

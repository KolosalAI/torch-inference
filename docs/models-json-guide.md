# Models.json Configuration Guide

The PyTorch Inference Framework now uses `data/models.json` to determine model availability and configuration. This guide explains how to configure and use this system.

## Overview

The `models.json` file controls:
- Which models are available for inference
- Model metadata and requirements
- TTS/STT capabilities
- Hardware compatibility
- Auto-loading behavior
- Model grouping and organization

## File Location

The configuration file should be located at:
```
data/models.json
```

## Configuration Structure

### Available Models

Each model in the `available_models` section defines:

```json
{
  "model_name": {
    "name": "model_name",
    "display_name": "Human-readable name",
    "description": "Model description",
    "source": "huggingface|torchvision|builtin|custom",
    "model_id": "actual_model_identifier",
    "task": "text-to-speech|image-classification|etc",
    "category": "audio|nlp|computer-vision|demo",
    "enabled": true,
    "auto_load": false,
    "priority": 1,
    "hardware_requirements": {
      "min_memory_mb": 512,
      "recommended_memory_mb": 1024,
      "gpu_required": false,
      "min_gpu_memory_mb": 0
    },
    "inference_config": {
      "batch_size": 1,
      "max_batch_size": 8,
      "timeout_seconds": 30,
      "warmup_iterations": 3
    },
    "metadata": {
      "parameters": 1000000,
      "size_mb": 100.0,
      "architecture": "ModelArchitecture",
      "framework": "pytorch|transformers",
      "version": "1.0.0"
    }
  }
}
```

### TTS Models

For text-to-speech models, add:

```json
{
  "tts_features": {
    "supports_tts": true,
    "quality": "high|medium|low",
    "speed": "fast|medium|slow",
    "supports_voice_cloning": true,
    "supports_emotions": true,
    "max_text_length": 1000
  }
}
```

### STT Models

For speech-to-text models, add:

```json
{
  "stt_features": {
    "supports_stt": true,
    "quality": "high|medium|low",
    "speed": "fast|medium|slow",
    "supports_timestamps": true,
    "supports_language_detection": true,
    "max_audio_length_seconds": 3600
  }
}
```

## Model Groups

Organize related models:

```json
{
  "model_groups": {
    "text_to_speech": {
      "name": "Text-to-Speech Models",
      "description": "Models for converting text to speech",
      "models": ["speecht5_tts", "bark_tts"],
      "default_model": "speecht5_tts",
      "enabled": true
    }
  }
}
```

## Hardware Profiles

Define compatibility rules:

```json
{
  "hardware_profiles": {
    "cpu_only": {
      "name": "CPU Only",
      "description": "Profile for CPU-only inference",
      "allowed_models": ["example", "distilbert_sentiment"],
      "blocked_models": ["bark_tts"],
      "max_memory_usage_mb": 8192,
      "optimization_level": "medium"
    }
  }
}
```

## Deployment Profiles

Control which models are loaded:

```json
{
  "deployment_profiles": {
    "development": {
      "name": "Development Profile",
      "auto_load_models": ["example"],
      "preload_models": [],
      "max_models_in_memory": 3,
      "prefer_cpu": true,
      "enable_model_caching": true
    }
  }
}
```

## API Usage

### Check Available Models

```bash
GET /models
```

Returns all available models with their configurations.

### Model Validation

The system automatically validates:
- Model availability before inference
- TTS capability for synthesis endpoints
- Hardware requirements
- Model enablement status

### Prediction with Model Validation

```bash
POST /predict
{
  "model_name": "distilbert_sentiment",
  "inputs": "This is a test"
}
```

The system will:
1. Check if the model exists in `models.json`
2. Verify the model is enabled
3. Validate hardware requirements
4. Proceed with inference

### TTS with Model Validation

```bash
POST /synthesize
{
  "model_name": "speecht5_tts",
  "inputs": "Hello world"
}
```

The system will:
1. Check if the model exists and is enabled
2. Verify TTS capability
3. Validate text length limits
4. Proceed with synthesis

## Error Handling

Common error messages:

- `Model 'model_name' is not available. Check models.json configuration.`
- `Model 'model_name' is disabled in configuration`
- `Model 'model_name' does not support Text-to-Speech. Available TTS models: ...`

## Configuration Management

### Enable/Disable Models

Set `"enabled": false` to disable a model without removing its configuration.

### Hardware Optimization

Use hardware profiles to automatically filter models based on system capabilities:

```python
from framework.core.model_config import get_model_config_manager

manager = get_model_config_manager()
cpu_models = manager.get_compatible_models("cpu_only")
gpu_models = manager.get_compatible_models("gpu_advanced")
```

### Priority System

Models with lower priority numbers are loaded first. Use this to prioritize important models.

## Best Practices

1. **Enable Only Needed Models**: Disable models not needed for your use case
2. **Set Appropriate Hardware Requirements**: Ensure requirements match actual needs
3. **Use Model Groups**: Organize related models for easier management
4. **Configure Auto-load Carefully**: Only auto-load essential models
5. **Update Metadata**: Keep size and parameter counts accurate for resource planning

## Validation

Use the test script to validate your configuration:

```bash
python test_models_json.py
```

This will check:
- JSON syntax validity
- Required fields presence
- Model configuration correctness
- Hardware profile compatibility

## Integration Points

The models.json configuration integrates with:

- **Model Manager**: Filters available models
- **Inference Engine**: Validates models before processing
- **TTS Endpoints**: Checks TTS capabilities
- **API Endpoints**: Returns configuration-based model lists
- **Auto-scaling**: Respects model priorities and requirements
- **Health Checks**: Reports on configured vs. loaded models

## Example Configurations

See the provided `data/models.json` for a complete example with:
- 9 different model types
- TTS and STT models
- Multiple hardware profiles
- Deployment configurations
- Model groups and metadata

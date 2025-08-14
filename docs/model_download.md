# Model Download Feature

The PyTorch Inference Framework now includes a comprehensive model download feature that allows you to easily download models from various sources and integrate them into your inference pipeline.

## Features

- **Multiple Sources**: Download models from PyTorch Hub, Torchvision, Hugging Face, and custom URLs
- **Automatic Caching**: Models are cached locally to avoid re-downloading
- **Registry Management**: Track downloaded models with metadata
- **CLI Interface**: Command-line tool for model management
- **REST API**: HTTP endpoints for downloading and managing models
- **Framework Integration**: Seamlessly load downloaded models into the inference framework

## Supported Sources

### 1. PyTorch Hub
Download models from the PyTorch Hub ecosystem:
```python
from framework.core.model_downloader import download_model

model_path, info = download_model(
    source="pytorch_hub",
    model_id="pytorch/vision/resnet50",  # repo/model format
    pretrained=True
)
```

### 2. Torchvision
Download pre-trained models from torchvision:
```python
model_path, info = download_model(
    source="torchvision", 
    model_id="resnet18",
    pretrained=True
)
```

### 3. Hugging Face
Download transformer models from Hugging Face Hub:
```python
model_path, info = download_model(
    source="huggingface",
    model_id="bert-base-uncased", 
    task="text-classification"
)
```

### 4. Custom URLs
Download models from any URL:
```python
model_path, info = download_model(
    source="url",
    model_id="https://example.com/model.pt",
    model_name="custom_model"
)
```

## Usage Examples

### Basic Download
```python
from framework.core.model_downloader import get_model_downloader

# Get downloader instance
downloader = get_model_downloader()

# Download a torchvision model
model_path, model_info = downloader.download_torchvision_model(
    model_name="resnet18",
    pretrained=True
)

print(f"Downloaded {model_info.name} ({model_info.size_mb:.1f} MB)")
```

### Direct Framework Integration
```python
from framework.core.base_model import get_model_manager

# Get model manager
model_manager = get_model_manager()

# Download and load model in one step
model_manager.download_and_load_model(
    source="torchvision",
    model_id="mobilenet_v2", 
    name="mobile_classifier"
)

# Use the model
model = model_manager.get_model("mobile_classifier")
prediction = model.predict(your_input)
```

### List Available Models
```python
from framework.core.model_downloader import list_available_models

# List all cached models
models = list_available_models()
for name, info in models.items():
    print(f"{name}: {info.source} - {info.size_mb:.1f} MB")
```

## Command Line Interface

The framework includes a CLI tool for model management:

### Download Models
```bash
# Download a torchvision model
python -m framework.scripts.download_models download torchvision resnet18

# Download a PyTorch Hub model  
python -m framework.scripts.download_models download pytorch_hub "pytorch/vision:v0.10.0/resnet50"

# Download a Hugging Face model
python -m framework.scripts.download_models download huggingface bert-base-uncased --task text-classification

# Download from URL
python -m framework.scripts.download_models download url "https://example.com/model.pt" --name my_model
```

### List Models
```bash
# List all cached models
python -m framework.scripts.download_models list

# Get info about a specific model
python -m framework.scripts.download_models info resnet18
```

### Manage Cache
```bash
# Show cache information
python -m framework.scripts.download_models cache

# Remove a model from cache
python -m framework.scripts.download_models remove resnet18

# Remove all models
python -m framework.scripts.download_models remove all
```

## REST API Endpoints

The framework provides HTTP endpoints for model downloading:

### Download a Model
```http
POST /models/download
Content-Type: application/json

{
    "source": "torchvision",
    "model_id": "resnet18", 
    "name": "my_resnet18",
    "task": "classification",
    "pretrained": true
}
```

### List Available Downloads
```http
GET /models/available
```

### Get Download Info
```http
GET /models/download/{model_name}/info
```

### Remove Downloaded Model
```http
DELETE /models/download/{model_name}
```

### Get Cache Info
```http
GET /models/cache/info
```

## Configuration

Models are cached in `~/.torch_inference/models/` by default. You can configure the cache directory:

```python
from framework.core.model_downloader import ModelDownloader

# Custom cache directory
downloader = ModelDownloader(cache_dir="/path/to/cache")
```

## Model Registry

Downloaded models are tracked in a JSON registry that stores:
- Model metadata (name, source, task, size, etc.)
- File paths and checksums
- Download timestamps
- Tags and descriptions

The registry enables:
- Avoiding duplicate downloads
- Fast model lookup
- Metadata tracking
- Cache management

## Integration with Framework

Downloaded models seamlessly integrate with the existing framework:

1. **Automatic Adapter Selection**: The framework automatically selects the appropriate adapter (PyTorch, ONNX, etc.) based on the model format
2. **Optimization**: Downloaded models are automatically optimized for inference
3. **Unified API**: Use the same prediction API regardless of model source
4. **Configuration**: Apply the same configuration system to all models

## Error Handling

The download system includes robust error handling:
- Network connectivity issues
- Corrupted downloads (with hash verification)
- Disk space limitations  
- Invalid model formats
- Missing dependencies

## Best Practices

1. **Check Cache First**: Always check if a model is already cached before downloading
2. **Use Appropriate Names**: Give descriptive names to your models for easy identification
3. **Monitor Cache Size**: Regularly clean up unused models to save disk space
4. **Verify Checksums**: Use hash verification for critical models
5. **Handle Errors**: Implement proper error handling in your applications

## Examples

See `examples/model_download_example.py` for a comprehensive demonstration of all features.

## Dependencies

### Required
- `torch`
- `requests`

### Optional
- `torchvision` (for torchvision models)
- `transformers` (for Hugging Face models)
- `onnxruntime` (for ONNX models)
- `tensorrt` (for TensorRT models)

## Troubleshooting

### Common Issues

1. **Import Errors**: Install missing optional dependencies
2. **Network Errors**: Check internet connectivity and proxy settings
3. **Disk Space**: Ensure sufficient disk space for model downloads
4. **Permissions**: Check write permissions for cache directory
5. **Model Loading**: Verify model format compatibility

### Debug Mode

Enable debug logging to troubleshoot issues:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

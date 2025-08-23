# Model Auto-Download Feature

This document describes the enhanced model downloading functionality that automatically downloads models to the `models/` directory in your project.

## Overview

The model downloader has been updated to:
- ✅ Automatically download models to the project's `models/` directory
- ✅ Support configuration via `config.yaml`
- ✅ Provide automatic source detection
- ✅ Include convenient CLI tools and Python APIs
- ✅ Maintain a registry of downloaded models

## Configuration

The model download configuration is now part of `config.yaml`:

```yaml
models:
  model_path: "models/"
  
  # Model Download Configuration
  download:
    cache_dir: "models/"
    auto_download: true
    sources:
      pytorch_hub:
        enabled: true
      torchvision:
        enabled: true
      huggingface:
        enabled: true
      url:
        enabled: true
        verify_ssl: true
        timeout_seconds: 300
    registry_file: "model_registry.json"
```

## Directory Structure

Downloaded models are organized in the following structure:

```
models/
├── model_registry.json                 # Registry of all downloaded models
├── torchvision_resnet18/               # Model directory
│   ├── model.pt                        # Full model file
│   ├── state_dict.pt                   # State dictionary only
│   └── model_info.json                 # Model metadata
└── huggingface_bert-base-uncased/      # Another model
    ├── model.pt
    ├── pytorch_model.pt
    ├── config.json
    └── tokenizer files...
```

## Usage

### 1. CLI Commands

#### Basic Download (existing functionality)
```bash
# Download with explicit source
python -m framework.scripts.download_models download torchvision resnet18
python -m framework.scripts.download_models download huggingface bert-base-uncased
```

#### Auto Download (new functionality)
```bash
# Auto-download with source detection
python -m framework.scripts.download_models auto torchvision:resnet18
python -m framework.scripts.download_models auto huggingface:bert-base-uncased
python -m framework.scripts.download_models auto resnet18  # Auto-detects torchvision
```

#### Dedicated Auto-Download Script
```bash
# Download individual models
python -m framework.scripts.auto_download torchvision:resnet18
python -m framework.scripts.auto_download huggingface:bert-base-uncased

# Download preset collections
python -m framework.scripts.auto_download --preset basic    # resnet18, mobilenet_v2
python -m framework.scripts.auto_download --preset vision   # Multiple vision models
python -m framework.scripts.auto_download --preset nlp      # NLP models

# Download multiple models at once
python -m framework.scripts.auto_download torchvision:resnet18 torchvision:mobilenet_v2

# Force re-download
python -m framework.scripts.auto_download torchvision:resnet18 --force

# Show configuration
python -m framework.scripts.auto_download --list-config
```

### 2. Python API

#### Basic Usage
```python
from framework.core.model_downloader import auto_download_model, ensure_model_available

# Auto-download with explicit source
model_path, model_info = auto_download_model("torchvision:resnet18")

# Auto-download with source detection
model_path, model_info = auto_download_model("resnet18")

# Ensure model is available (downloads if not cached)
model_path, model_info = ensure_model_available("torchvision:mobilenet_v2")

# Download from URL
model_path, model_info = auto_download_model(
    "https://example.com/model.pt", 
    model_name="custom_model"
)
```

#### Advanced Usage
```python
from framework.core.model_downloader import get_model_downloader

# Get downloader instance
downloader = get_model_downloader()

# Check configuration
config = downloader.get_config()
print(f"Auto-download enabled: {config['auto_download']}")

# List available models
models = downloader.list_available_models()
for name, info in models.items():
    print(f"{name}: {info.size_mb:.1f} MB from {info.source}")

# Check if model is cached
if downloader.is_model_cached("torchvision_resnet18"):
    model_path = downloader.get_model_path("torchvision_resnet18")
    print(f"Model available at: {model_path}")
```

### 3. Batch Downloads
```python
from framework.scripts.auto_download import download_multiple_models, download_preset_models

# Download multiple models
results = download_multiple_models([
    "torchvision:resnet18",
    "torchvision:mobilenet_v2",
    "huggingface:bert-base-uncased"
])

# Download preset collections
results = download_preset_models("basic")  # Downloads basic model set
```

## Model Identifier Formats

The auto-download feature supports several identifier formats:

### Explicit Source Format
- `torchvision:model_name` - e.g., `torchvision:resnet18`
- `huggingface:model_id` - e.g., `huggingface:bert-base-uncased`
- `pytorch_hub:repo/model` - e.g., `pytorch_hub:pytorch/vision/resnet50`
- `url:https://...` - Direct URL downloads

### Auto-Detection Format
- `resnet18` - Tries torchvision first, then PyTorch Hub
- `bert-base-uncased` - Tries Hugging Face if other sources fail

### URL Format
- `https://example.com/model.pt` - Direct download from URL

## Preset Collections

Several preset collections are available for quick setup:

- **basic**: `resnet18`, `mobilenet_v2`
- **vision**: `resnet18`, `resnet50`, `mobilenet_v2`, `efficientnet_b0`, `vgg16`
- **nlp**: `bert-base-uncased`, `distilbert-base-uncased`, `roberta-base`
- **all**: Combination of vision and nlp models

## Integration with Existing Framework

Downloaded models automatically integrate with the existing inference framework:

```python
# The model manager can now automatically download models
from framework.core.base_model import get_model_manager

model_manager = get_model_manager()

# This will auto-download if not available
model_manager.load_model("torchvision:resnet18", name="my_classifier")

# Use the model
prediction = model_manager.predict("my_classifier", input_data)
```

## Features

### ✅ Implemented Features
- [x] Auto-download to project `models/` directory
- [x] Configuration via `config.yaml`
- [x] Multiple source support (torchvision, PyTorch Hub, Hugging Face, URLs)
- [x] Automatic source detection
- [x] Model registry and caching
- [x] CLI tools for management
- [x] Python API for programmatic use
- [x] Batch download capabilities
- [x] Preset model collections
- [x] Progress tracking and error handling

### 🔄 Configuration Options
- Cache directory location
- Enable/disable specific sources
- SSL verification for URL downloads
- Download timeouts
- Registry file location

### 📊 Management Features
- List downloaded models
- Check cache size
- Remove individual models
- Clear entire cache
- Model metadata tracking

## Examples

### Quick Start
```bash
# Download some basic models
python -m framework.scripts.auto_download --preset basic

# List what's available
python -m framework.scripts.download_models list

# Use in your code
python -c "
from framework.core.model_downloader import ensure_model_available
path, info = ensure_model_available('torchvision:resnet18')
print(f'Model ready at: {path}')
"
```

### Custom Configuration
To modify the download behavior, edit `config.yaml`:

```yaml
models:
  download:
    cache_dir: "custom_models/"  # Use different directory
    sources:
      huggingface:
        enabled: false  # Disable Hugging Face downloads
      url:
        timeout_seconds: 600  # Increase timeout for large files
```

## Troubleshooting

### Common Issues

1. **Models downloading to wrong location**: Check `cache_dir` in `config.yaml`
2. **Source not available**: Ensure required packages are installed (`transformers` for Hugging Face)
3. **Permission errors**: Ensure write access to the models directory
4. **Network timeouts**: Increase `timeout_seconds` in configuration

### Debug Commands
```bash
# Check configuration
python -m framework.scripts.auto_download --list-config

# Verify model locations
python -m framework.scripts.download_models list

# Test download with verbose output
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from framework.core.model_downloader import auto_download_model
auto_download_model('torchvision:resnet18')
"
```

This enhanced model downloading system provides a robust, configurable way to manage PyTorch models in your inference framework while maintaining backward compatibility with existing functionality.

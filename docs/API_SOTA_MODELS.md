# SOTA Model Download API

## Overview
The torch-inference server now supports downloading state-of-the-art (SOTA) image classification models directly from the model registry through REST API endpoints.

## API Endpoints

### 1. List All SOTA Models
```http
GET /models/sota
```

Returns a list of all SOTA image classification models available in the registry, sorted by rank.

**Response:**
```json
{
  "models": [
    {
      "id": "eva02-large-patch14-448",
      "name": "EVA-02 Large",
      "architecture": "Vision Transformer (ViT)",
      "accuracy": "90.054% Top-1",
      "rank": 1,
      "size": "~1.2 GB",
      "url": "https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
      "dataset": "ImageNet-1K (with ImageNet-22K pretraining)",
      "status": "Available"
    },
    ...
  ],
  "total": 12,
  "message": "SOTA image classification models. Use POST /models/sota/{model_id} to download",
  "documentation": "See SOTA_MODELS.md for details"
}
```

### 2. Download a SOTA Model
```http
POST /models/sota/{model_id}
```

Downloads a specific SOTA model from the registry.

**Parameters:**
- `model_id` (path): The model ID from the registry (e.g., `eva02-large-patch14-448`)

**Example:**
```bash
curl -X POST http://localhost:8080/models/sota/eva02-large-patch14-448
```

**Response:**
```json
{
  "task_id": "abc123...",
  "status": "started",
  "model_id": "eva02-large-patch14-448",
  "model_name": "EVA-02 Large",
  "message": "Download task abc123... created for model EVA-02 Large"
}
```

### 3. List All Available Models (Including TTS & SOTA)
```http
GET /models/available
```

Returns all models from the registry including TTS, image classification (SOTA), and other models.

**Response:**
```json
{
  "models": [...],
  "total": 22,
  "categories": {
    "tts": 10,
    "image-classification": 12,
    "speech-recognition": 0,
    "multimodal": 0
  },
  "message": "Use POST /models/download to download any model",
  "source": "model_registry.json"
}
```

### 4. Check Download Status
```http
GET /models/download/status/{task_id}
```

Check the progress of a download task.

**Example:**
```bash
curl http://localhost:8080/models/download/status/abc123...
```

**Response:**
```json
{
  "task_id": "abc123...",
  "model_name": "EVA-02 Large",
  "status": "downloading",
  "progress": 45.2,
  "bytes_downloaded": 542398464,
  "total_bytes": 1200000000,
  "speed_mbps": 12.5,
  "eta_seconds": 52
}
```

### 5. List Downloaded Models
```http
GET /models/managed
```

Lists all models that have been downloaded and are available locally.

**Response:**
```json
{
  "models": [
    {
      "name": "EVA-02 Large",
      "source": "HuggingFace(...)",
      "size_bytes": 1200000000,
      "size_human": "1.12 GB",
      "downloaded_at": "2025-12-17T23:00:00Z"
    }
  ],
  "total": 1
}
```

### 6. Delete a Downloaded Model
```http
DELETE /models/download/{model_name}
```

Removes a downloaded model from the cache.

**Example:**
```bash
curl -X DELETE http://localhost:8080/models/download/eva02-large-patch14-448
```

## Usage Examples

### Download Top-3 SOTA Models

```bash
#!/bin/bash

# 1. List SOTA models
curl http://localhost:8080/models/sota | jq '.models[:3]'

# 2. Download EVA-02 Large (#1 accuracy)
curl -X POST http://localhost:8080/models/sota/eva02-large-patch14-448

# 3. Download EVA Giant (#2 accuracy)
curl -X POST http://localhost:8080/models/sota/eva-giant-patch14-560

# 4. Download ConvNeXt V2 Huge (#3 accuracy)
curl -X POST http://localhost:8080/models/sota/convnextv2-huge-512

# 5. Check download status
TASK_ID=$(curl -X POST http://localhost:8080/models/sota/eva02-large-patch14-448 | jq -r '.task_id')
curl http://localhost:8080/models/download/status/$TASK_ID
```

### Monitor Download Progress

```bash
#!/bin/bash

TASK_ID="abc123..."

while true; do
  STATUS=$(curl -s http://localhost:8080/models/download/status/$TASK_ID)
  PROGRESS=$(echo $STATUS | jq -r '.progress')
  STATE=$(echo $STATUS | jq -r '.status')
  
  echo "Progress: $PROGRESS% - Status: $STATE"
  
  if [ "$STATE" = "completed" ] || [ "$STATE" = "failed" ]; then
    break
  fi
  
  sleep 2
done
```

### Download Efficient Models for Production

```bash
# MobileNetV4 (edge-optimized, 84.36% accuracy)
curl -X POST http://localhost:8080/models/sota/mobilenetv4-hybrid-large

# EfficientNetV2 XL (best efficiency, 87.3% accuracy)
curl -X POST http://localhost:8080/models/sota/efficientnetv2-xl

# CoAtNet-3 (efficient hybrid, ~86% accuracy)
curl -X POST http://localhost:8080/models/sota/coatnet-3-rw-224
```

## Python Client Example

```python
import requests
import time

BASE_URL = "http://localhost:8080"

def list_sota_models():
    """List all SOTA models"""
    response = requests.get(f"{BASE_URL}/models/sota")
    return response.json()

def download_model(model_id):
    """Download a SOTA model"""
    response = requests.post(f"{BASE_URL}/models/sota/{model_id}")
    return response.json()

def check_status(task_id):
    """Check download status"""
    response = requests.get(f"{BASE_URL}/models/download/status/{task_id}")
    return response.json()

def wait_for_download(task_id, timeout=3600):
    """Wait for download to complete"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status = check_status(task_id)
        state = status.get('status')
        progress = status.get('progress', 0)
        
        print(f"Progress: {progress:.1f}% - Status: {state}")
        
        if state == 'completed':
            print("Download completed!")
            return True
        elif state == 'failed':
            print(f"Download failed: {status.get('error')}")
            return False
        
        time.sleep(2)
    
    print("Download timeout")
    return False

# Example usage
if __name__ == "__main__":
    # List models
    models = list_sota_models()
    print(f"Found {models['total']} SOTA models")
    
    # Download top model
    top_model = models['models'][0]
    print(f"Downloading {top_model['name']}...")
    
    result = download_model(top_model['id'])
    task_id = result['task_id']
    
    # Wait for completion
    wait_for_download(task_id)
```

## Model Registry Format

The endpoints read from `model_registry.json` with the following structure:

```json
{
  "updated": "2025-12-17 23:06:00",
  "version": "1.1",
  "models": {
    "eva02-large-patch14-448": {
      "status": "Available",
      "architecture": "Vision Transformer (ViT)",
      "task": "Image Classification",
      "accuracy": "90.054% Top-1",
      "rank": 1,
      "size": "~1.2 GB",
      "url": "https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
      "dataset": "ImageNet-1K (with ImageNet-22K pretraining)",
      "name": "EVA-02 Large"
    }
  }
}
```

## Model Categories

### By Accuracy (ImageNet-1K Top-1)
- **Tier 1**: >89% - EVA-02, EVA Giant
- **Tier 2**: 88-89% - ConvNeXt V2, MaxViT
- **Tier 3**: 87-88% - EfficientNetV2, Swin, DeiT
- **Tier 4**: 84-86% - MobileNetV4, CoAtNet

### By Use Case
- **Maximum Accuracy**: eva02-large-patch14-448, eva-giant-patch14-560
- **Transfer Learning**: convnext-xxlarge-clip, vit-giant-patch14-224
- **Production**: efficientnetv2-xl, maxvit-xlarge-512
- **Edge/Mobile**: mobilenetv4-hybrid-large
- **Dense Prediction**: swin-large-patch4-384, convnextv2-huge-512

## Error Handling

### Model Not Found
```json
{
  "error": "NotFound",
  "message": "Model invalid-model-id not found in registry",
  "status": 404
}
```

### Download Failed
```json
{
  "task_id": "abc123...",
  "status": "failed",
  "error": "Connection timeout",
  "retry_count": 3
}
```

### Registry Not Available
```json
{
  "models": [],
  "total": 0,
  "message": "Model registry not found",
  "source": "hardcoded"
}
```

## Performance Notes

- Downloads are processed asynchronously in the background
- Multiple downloads can run concurrently
- Models are cached in `models/` directory
- Duplicate downloads are prevented automatically
- Resume support for interrupted downloads (if supported by source)

## See Also

- [SOTA_MODELS.md](SOTA_MODELS.md) - Detailed model information and benchmarks
- [model_registry.json](model_registry.json) - Complete model registry
- [README.md](README.md) - General server documentation

---

## Complete Model Catalog

### Top Tier Models (>88% Accuracy)

#### 1. EVA-02 Large (90.054% Top-1) 🥇
- **Best for:** Highest accuracy requirements
- **Size:** 1.2 GB (efficient for SOTA)
- **Recommended:** Production use with high accuracy needs

#### 2. EVA Giant (89.792% Top-1) 🥈  
- **Best for:** Research and benchmarking
- **Size:** 4.0 GB (largest ViT)
- **Note:** Requires significant compute resources

#### 3. ConvNeXt V2 Huge (88.848% Top-1) 🥉
- **Best for:** Pure CNN enthusiasts
- **Size:** 2.6 GB
- **Note:** Best performing ConvNet architecture

### Efficient Models

#### MobileNetV4 Hybrid Large (84.36% Top-1) ⚡
- **Best for:** Mobile and edge deployment  
- **Size:** 140 MB (Smallest!)
- **Recommended:** Real-time inference, resource-constrained environments

---

## Testing the API

### Step 1: Start the Server
```bash
./target/release/torch-inference-server
```

### Step 2: List Available SOTA Models
```bash
curl http://localhost:8000/models/sota | jq '.models[] | {id, name, accuracy, size}'
```

### Step 3: Download a Model (Example: MobileNetV4)
```bash
curl -X POST http://localhost:8000/models/sota/mobilenetv4-hybrid-large
```

**Response:**
```json
{
  "task_id": "uuid-here",
  "status": "started",
  "message": "Download task created"
}
```

### Step 4: Check Download Progress
```bash
curl http://localhost:8000/models/download/list
```

---

## Model Comparison

| Model | Top-1 Acc | Size | Type | Speed | Memory |
|-------|-----------|------|------|-------|--------|
| EVA-02 Large | 90.1% | 1.2GB | ViT | Medium | High |
| MobileNetV4 | 84.4% | 140MB | Hybrid | Fast | Low |
| EfficientNetV2 XL | 87.3% | 850MB | CNN | Fast | Medium |
| ConvNeXt V2 Huge | 88.8% | 2.6GB | CNN | Medium | High |

---

## Important Notes

⚠️ **Torch Feature Required**

The current build does not include PyTorch support. To enable inference:

```bash
# Rebuild with torch feature
cargo build --release --features torch

# Ensure LibTorch is available
export LIBTORCH=/path/to/libtorch
```

✅ **Currently Working:**
- Model download API
- Download progress tracking
- Model registry queries

❌ **Requires Rebuild:**
- Actual inference
- GPU acceleration
- Model loading for prediction


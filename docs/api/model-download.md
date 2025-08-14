# POST /model-download - Download Model

**URL**: `/model-download`  
**Method**: `POST`  
**Authentication**: Not required  
**Content-Type**: `application/json`

## Description

Downloads and installs a model from a configured model repository or URL. This endpoint supports downloading models from various sources and automatically handles model validation, caching, and installation.

## Request

### URL Parameters
None

### Query Parameters
None

### Request Body

#### Basic Model Download
```json
{
  "model_name": "resnet50",
  "version": "v1.0"
}
```

#### Download from URL
```json
{
  "model_name": "custom-model",
  "download_url": "https://models.example.com/custom-model-v2.pth",
  "version": "v2.0",
  "checksum": "sha256:abc123def456...",
  "force_download": false
}
```

#### Advanced Download Options
```json
{
  "model_name": "bert-large",
  "version": "v3.1",
  "source": "huggingface",
  "options": {
    "chunk_size": 8192,
    "timeout": 300,
    "retry_count": 3,
    "verify_ssl": true,
    "extract_after_download": true,
    "cleanup_temp_files": true
  },
  "metadata": {
    "description": "BERT Large model for advanced text processing",
    "tags": ["nlp", "transformer", "large"],
    "author": "Custom Organization"
  }
}
```

#### Request Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_name` | string | Yes | - | Unique identifier for the model |
| `version` | string | No | "latest" | Model version to download |
| `download_url` | string | No | - | Direct download URL (overrides repository lookup) |
| `source` | string | No | "default" | Model source/repository name |
| `checksum` | string | No | - | Expected file checksum for verification |
| `force_download` | boolean | No | false | Force re-download even if model exists |
| `options` | object | No | - | Advanced download options |
| `metadata` | object | No | - | Additional model metadata |

#### Options Object Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chunk_size` | integer | 8192 | Download chunk size in bytes |
| `timeout` | integer | 300 | Download timeout in seconds |
| `retry_count` | integer | 3 | Number of retry attempts |
| `verify_ssl` | boolean | true | Whether to verify SSL certificates |
| `extract_after_download` | boolean | true | Auto-extract compressed archives |
| `cleanup_temp_files` | boolean | true | Remove temporary files after installation |

## Response

### Success Response (200 OK)

#### Download Started
```json
{
  "status": "started",
  "message": "Model download started",
  "download_id": "dwnld_12345",
  "model_name": "resnet50",
  "version": "v1.0",
  "estimated_size_mb": 97.8,
  "estimated_duration_seconds": 45,
  "download_url": "https://models.example.com/resnet50-v1.0.pth",
  "progress_endpoint": "/model-download/dwnld_12345/progress"
}
```

#### Download Completed (Synchronous)
```json
{
  "status": "completed",
  "message": "Model downloaded and installed successfully",
  "model_name": "resnet50",
  "version": "v1.0",
  "file_path": "/models/resnet50-v1.0.pth",
  "file_size_mb": 97.8,
  "download_duration_seconds": 42.3,
  "checksum_verified": true,
  "installation": {
    "status": "success",
    "installed_at": "2025-08-14T10:35:22.000Z",
    "model_id": "resnet50_v1.0",
    "ready_for_inference": true
  }
}
```

#### Download in Progress (Asynchronous)
```json
{
  "status": "downloading",
  "message": "Download in progress",
  "download_id": "dwnld_12345",
  "progress": {
    "percentage": 67.5,
    "downloaded_mb": 66.0,
    "total_mb": 97.8,
    "speed_mbps": 2.1,
    "eta_seconds": 15,
    "current_stage": "downloading"
  }
}
```

#### Response Fields

##### Download Response
| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Download status: "started", "downloading", "completed", "failed" |
| `message` | string | Human-readable status message |
| `download_id` | string | Unique download identifier for tracking |
| `model_name` | string | Name of the model being downloaded |
| `version` | string | Model version |
| `estimated_size_mb` | number | Expected download size |
| `estimated_duration_seconds` | number | Expected download duration |
| `download_url` | string | Actual download URL being used |
| `progress_endpoint` | string | Endpoint to check download progress |

##### Progress Object
| Field | Type | Description |
|-------|------|-------------|
| `percentage` | number | Download completion percentage (0-100) |
| `downloaded_mb` | number | Amount downloaded in MB |
| `total_mb` | number | Total file size in MB |
| `speed_mbps` | number | Current download speed in MB/s |
| `eta_seconds` | number | Estimated time to completion |
| `current_stage` | string | Current operation: "downloading", "verifying", "installing" |

##### Installation Object
| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Installation status: "success", "failed", "pending" |
| `installed_at` | string | Installation timestamp (ISO format) |
| `model_id` | string | Internal model identifier |
| `ready_for_inference` | boolean | Whether model is ready to use |

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success - Download started or completed |
| 400 | Bad request - Invalid parameters |
| 404 | Model not found in repository |
| 409 | Conflict - Model already exists (use force_download=true) |
| 413 | Model too large for available storage |
| 500 | Internal server error |

## Examples

### Basic Model Download

**Request:**
```bash
curl -X POST http://localhost:8000/model-download \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "resnet50",
    "version": "v1.0"
  }'
```

**Response:**
```json
{
  "status": "completed",
  "message": "Model downloaded and installed successfully",
  "model_name": "resnet50",
  "version": "v1.0",
  "file_path": "/models/resnet50-v1.0.pth",
  "file_size_mb": 97.8,
  "download_duration_seconds": 42.3,
  "installation": {
    "status": "success",
    "installed_at": "2025-08-14T10:35:22.000Z",
    "model_id": "resnet50_v1.0",
    "ready_for_inference": true
  }
}
```

### Download from Custom URL

**Request:**
```bash
curl -X POST http://localhost:8000/model-download \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "custom-classifier",
    "download_url": "https://storage.example.com/models/classifier-v2.1.pth",
    "version": "v2.1",
    "checksum": "sha256:a1b2c3d4e5f6...",
    "force_download": false
  }'
```

### Force Re-download Existing Model

**Request:**
```bash
curl -X POST http://localhost:8000/model-download \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "bert-base",
    "version": "v1.5",
    "force_download": true
  }'
```

### Python Model Downloader

```python
import requests
import time
import json
from typing import Dict, Optional, Callable

class ModelDownloader:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def download_model(
        self,
        model_name: str,
        version: str = "latest",
        download_url: Optional[str] = None,
        source: str = "default",
        force_download: bool = False,
        progress_callback: Optional[Callable] = None,
        **options
    ) -> Dict:
        """
        Download a model with progress monitoring
        """
        payload = {
            "model_name": model_name,
            "version": version,
            "source": source,
            "force_download": force_download
        }
        
        if download_url:
            payload["download_url"] = download_url
        
        if options:
            payload["options"] = options
        
        try:
            print(f"Starting download of {model_name} v{version}...")
            response = requests.post(
                f"{self.base_url}/model-download",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            # Handle asynchronous download
            if result.get('status') == 'started' and 'download_id' in result:
                return self._monitor_download(result['download_id'], progress_callback)
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _monitor_download(self, download_id: str, progress_callback: Optional[Callable] = None) -> Dict:
        """Monitor asynchronous download progress"""
        progress_url = f"{self.base_url}/model-download/{download_id}/progress"
        
        while True:
            try:
                response = requests.get(progress_url)
                response.raise_for_status()
                progress_data = response.json()
                
                status = progress_data.get('status')
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(progress_data)
                else:
                    self._print_progress(progress_data)
                
                if status in ['completed', 'failed']:
                    return progress_data
                
                time.sleep(2)  # Poll every 2 seconds
                
            except requests.exceptions.RequestException as e:
                print(f"Error checking progress: {e}")
                return {"status": "failed", "error": str(e)}
    
    def _print_progress(self, progress_data: Dict):
        """Default progress printer"""
        if 'progress' in progress_data:
            progress = progress_data['progress']
            percentage = progress.get('percentage', 0)
            speed = progress.get('speed_mbps', 0)
            eta = progress.get('eta_seconds', 0)
            stage = progress.get('current_stage', 'unknown')
            
            bar_length = 40
            filled_length = int(bar_length * percentage / 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            print(f"\r[{bar}] {percentage:.1f}% | "
                  f"{speed:.1f} MB/s | ETA: {eta}s | Stage: {stage}", end='', flush=True)
            
            if progress_data.get('status') in ['completed', 'failed']:
                print()  # New line when done
    
    def download_batch(self, models: list, max_concurrent: int = 2) -> Dict:
        """
        Download multiple models with concurrency control
        """
        import concurrent.futures
        import threading
        
        results = {}
        semaphore = threading.Semaphore(max_concurrent)
        
        def download_with_semaphore(model_config):
            with semaphore:
                model_name = model_config.get('model_name')
                try:
                    result = self.download_model(**model_config)
                    return model_name, result
                except Exception as e:
                    return model_name, {"status": "failed", "error": str(e)}
        
        print(f"Downloading {len(models)} models with {max_concurrent} concurrent downloads...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_model = {
                executor.submit(download_with_semaphore, model_config): model_config['model_name']
                for model_config in models
            }
            
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    name, result = future.result()
                    results[name] = result
                    status = result.get('status', 'unknown')
                    print(f"✅ {name}: {status}")
                except Exception as e:
                    results[model_name] = {"status": "failed", "error": str(e)}
                    print(f"❌ {model_name}: failed - {e}")
        
        return results
    
    def verify_download(self, model_name: str, version: str = "latest") -> bool:
        """Verify that a downloaded model is working"""
        try:
            # Check if model is available
            models_response = requests.get(f"{self.base_url}/models")
            models_response.raise_for_status()
            models_data = models_response.json()
            
            for model in models_data.get('models', []):
                if (model['name'] == model_name and 
                    model['version'] == version and 
                    model['status'] in ['loaded', 'available']):
                    print(f"✅ Model {model_name} v{version} verified successfully")
                    return True
            
            print(f"❌ Model {model_name} v{version} not found or not ready")
            return False
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    def get_download_status(self, download_id: str) -> Dict:
        """Get status of a specific download"""
        try:
            response = requests.get(f"{self.base_url}/model-download/{download_id}/progress")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "failed", "error": str(e)}

# Usage Examples
downloader = ModelDownloader()

# Simple download
result = downloader.download_model("resnet50", "v1.0")
print(f"Download result: {result['status']}")

# Download with custom progress callback
def custom_progress(data):
    if 'progress' in data:
        progress = data['progress']
        print(f"Custom: {progress['percentage']:.1f}% complete")

result = downloader.download_model(
    "bert-base",
    "v2.1",
    progress_callback=custom_progress
)

# Batch download
models_to_download = [
    {"model_name": "resnet18", "version": "v1.0"},
    {"model_name": "mobilenet", "version": "v2.0"},
    {"model_name": "efficientnet", "version": "v1.5"}
]

batch_results = downloader.download_batch(models_to_download, max_concurrent=2)

# Verify downloads
for model_config in models_to_download:
    downloader.verify_download(
        model_config["model_name"], 
        model_config["version"]
    )

# Download from custom URL with options
result = downloader.download_model(
    model_name="custom-transformer",
    version="v3.0",
    download_url="https://storage.example.com/models/transformer-v3.0.tar.gz",
    force_download=True,
    chunk_size=16384,
    timeout=600,
    retry_count=5
)
```

### JavaScript Download Manager

```javascript
class DownloadManager {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.activeDownloads = new Map();
  }
  
  async downloadModel(config) {
    try {
      const response = await fetch(`${this.baseUrl}/model-download`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(config)
      });
      
      const result = await response.json();
      
      if (result.status === 'started' && result.download_id) {
        return this.monitorDownload(result.download_id);
      }
      
      return result;
    } catch (error) {
      return { status: 'failed', error: error.message };
    }
  }
  
  async monitorDownload(downloadId) {
    return new Promise((resolve) => {
      const progressUrl = `${this.baseUrl}/model-download/${downloadId}/progress`;
      
      const checkProgress = async () => {
        try {
          const response = await fetch(progressUrl);
          const data = await response.json();
          
          this.updateProgressUI(downloadId, data);
          
          if (data.status === 'completed' || data.status === 'failed') {
            this.activeDownloads.delete(downloadId);
            resolve(data);
          } else {
            setTimeout(checkProgress, 2000);
          }
        } catch (error) {
          resolve({ status: 'failed', error: error.message });
        }
      };
      
      this.activeDownloads.set(downloadId, { status: 'downloading' });
      checkProgress();
    });
  }
  
  updateProgressUI(downloadId, data) {
    const progressElement = document.getElementById(`progress-${downloadId}`);
    if (!progressElement) return;
    
    if (data.progress) {
      const { percentage, speed_mbps, eta_seconds, current_stage } = data.progress;
      
      progressElement.innerHTML = `
        <div class="progress-bar">
          <div class="progress-fill" style="width: ${percentage}%"></div>
        </div>
        <div class="progress-info">
          <span>${percentage.toFixed(1)}%</span>
          <span>${speed_mbps.toFixed(1)} MB/s</span>
          <span>ETA: ${eta_seconds}s</span>
          <span>${current_stage}</span>
        </div>
      `;
    }
  }
  
  async downloadMultiple(models, maxConcurrent = 2) {
    const results = {};
    const downloading = [];
    
    for (let i = 0; i < models.length; i += maxConcurrent) {
      const batch = models.slice(i, i + maxConcurrent);
      
      const batchPromises = batch.map(async (modelConfig) => {
        const result = await this.downloadModel(modelConfig);
        results[modelConfig.model_name] = result;
        return result;
      });
      
      await Promise.all(batchPromises);
    }
    
    return results;
  }
  
  renderDownloadForm(containerId) {
    const container = document.getElementById(containerId);
    
    container.innerHTML = `
      <form id="download-form" class="download-form">
        <div class="form-group">
          <label for="model-name">Model Name:</label>
          <input type="text" id="model-name" required placeholder="e.g., resnet50">
        </div>
        
        <div class="form-group">
          <label for="model-version">Version:</label>
          <input type="text" id="model-version" placeholder="v1.0 or latest">
        </div>
        
        <div class="form-group">
          <label for="download-url">Custom URL (optional):</label>
          <input type="url" id="download-url" placeholder="https://...">
        </div>
        
        <div class="form-group">
          <label>
            <input type="checkbox" id="force-download">
            Force re-download
          </label>
        </div>
        
        <div class="form-group">
          <button type="submit" class="btn-primary">Download Model</button>
        </div>
      </form>
      
      <div id="downloads-list" class="downloads-list"></div>
    `;
    
    document.getElementById('download-form').addEventListener('submit', (e) => {
      e.preventDefault();
      this.handleFormSubmit();
    });
  }
  
  async handleFormSubmit() {
    const modelName = document.getElementById('model-name').value;
    const version = document.getElementById('model-version').value || 'latest';
    const downloadUrl = document.getElementById('download-url').value;
    const forceDownload = document.getElementById('force-download').checked;
    
    const config = {
      model_name: modelName,
      version: version,
      force_download: forceDownload
    };
    
    if (downloadUrl) {
      config.download_url = downloadUrl;
    }
    
    const downloadId = `download-${Date.now()}`;
    this.addDownloadToUI(downloadId, modelName, version);
    
    const result = await this.downloadModel(config);
    this.updateDownloadResult(downloadId, result);
  }
  
  addDownloadToUI(downloadId, modelName, version) {
    const downloadsList = document.getElementById('downloads-list');
    
    const downloadElement = document.createElement('div');
    downloadElement.id = `download-${downloadId}`;
    downloadElement.className = 'download-item';
    downloadElement.innerHTML = `
      <div class="download-header">
        <h4>${modelName} v${version}</h4>
        <span class="download-status">Starting...</span>
      </div>
      <div id="progress-${downloadId}" class="download-progress"></div>
    `;
    
    downloadsList.appendChild(downloadElement);
  }
  
  updateDownloadResult(downloadId, result) {
    const statusElement = document.querySelector(`#download-${downloadId} .download-status`);
    if (statusElement) {
      statusElement.textContent = result.status.toUpperCase();
      statusElement.className = `download-status status-${result.status}`;
    }
  }
}

// Initialize download manager
const downloadManager = new DownloadManager();
downloadManager.renderDownloadForm('download-container');
```

### Monitor Downloads Script

```bash
#!/bin/bash
# Monitor all active downloads

monitor_downloads() {
    echo "Monitoring active downloads..."
    
    while true; do
        # Get all models that might be downloading
        response=$(curl -s "http://localhost:8000/models")
        
        loading_models=$(echo "$response" | jq -r '.models[] | select(.status == "loading") | .name')
        
        if [[ -n "$loading_models" ]]; then
            echo "$(date): Active downloads:"
            for model in $loading_models; do
                echo "  - $model"
            done
        else
            echo "$(date): No active downloads"
        fi
        
        sleep 10
    done
}

# Check download by ID
check_download() {
    local download_id="$1"
    
    if [[ -z "$download_id" ]]; then
        echo "Usage: check_download <download_id>"
        return 1
    fi
    
    curl -s "http://localhost:8000/model-download/${download_id}/progress" | jq '.'
}

# Usage
monitor_downloads
```

## Error Handling

### Model Not Found (404)
```json
{
  "detail": "Model 'unknown-model' not found in repository"
}
```

### Model Already Exists (409)
```json
{
  "detail": "Model 'resnet50' version 'v1.0' already exists. Use force_download=true to override."
}
```

### Storage Full (413)
```json
{
  "detail": "Insufficient storage space. Required: 500MB, Available: 200MB"
}
```

### Download Failed (500)
```json
{
  "detail": "Download failed: Connection timeout"
}
```

## Related Endpoints

- [Models List](./models.md) - View available and downloaded models  
- [Model Info](./model-info.md) - Get detailed model information
- [Model Remove](./model-remove.md) - Remove downloaded models
- [Cache Info](./cache-info.md) - View download cache status

## Download Sources

The API supports downloading from multiple sources:

1. **Default Repository**: Configured model repository
2. **Hugging Face**: `source: "huggingface"`
3. **Custom URLs**: Direct download links
4. **Local Registry**: Internal model registry

## Best Practices

1. **Verify Checksums**: Always provide checksums for security
2. **Monitor Progress**: Use progress endpoints for large downloads
3. **Batch Downloads**: Use concurrent downloads efficiently
4. **Storage Management**: Check available space before downloads
5. **Error Handling**: Implement retry logic for network issues
6. **Cleanup**: Remove failed downloads to save space

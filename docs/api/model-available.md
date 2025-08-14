# GET /model-available/{model_name} - Check Model Availability

**URL**: `/model-available/{model_name}`  
**Method**: `GET`  
**Authentication**: Not required  
**Content-Type**: `application/json`

## Description

Checks whether a specific model is available for download or use. This endpoint provides detailed information about model availability, current status, and download options without actually downloading the model.

## Request

### URL Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | string | Yes | Name of the model to check |

### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `version` | string | No | "latest" | Specific version to check |
| `include_alternatives` | boolean | No | false | Include similar or alternative models |
| `check_sources` | boolean | No | true | Check all configured model sources |

### Request Body
None (GET request)

## Response

### Success Response (200 OK)

#### Model Available
```json
{
  "model_name": "resnet50",
  "requested_version": "v1.0",
  "availability": {
    "status": "available",
    "current_status": "not_loaded",
    "can_download": true,
    "can_load": false,
    "sources": [
      {
        "name": "official_repository",
        "available": true,
        "url": "https://models.example.com/resnet50-v1.0.pth",
        "size_mb": 97.8,
        "checksum": "sha256:abc123def456...",
        "last_updated": "2025-08-10T14:30:00.000Z"
      },
      {
        "name": "mirror_repository", 
        "available": true,
        "url": "https://mirror.example.com/resnet50-v1.0.pth",
        "size_mb": 97.8,
        "checksum": "sha256:abc123def456...",
        "last_updated": "2025-08-10T14:30:00.000Z"
      }
    ]
  },
  "metadata": {
    "description": "ResNet-50 deep residual network for image classification",
    "type": "image_classification",
    "framework": "pytorch",
    "precision": "fp32",
    "input_shape": [3, 224, 224],
    "output_shape": [1000],
    "parameters_count": 25557032,
    "memory_requirements_mb": 125.4,
    "supported_devices": ["cpu", "cuda"],
    "supported_formats": ["tensor", "numpy", "pil"],
    "tags": ["computer_vision", "classification", "imagenet"]
  },
  "versions": {
    "available": ["v1.0", "v1.1", "v2.0"],
    "latest": "v2.0",
    "recommended": "v1.1"
  },
  "requirements": {
    "minimum_python": "3.7",
    "dependencies": ["torch>=1.8.0", "torchvision>=0.9.0", "pillow>=8.0.0"],
    "minimum_memory_mb": 100,
    "minimum_storage_mb": 150
  }
}
```

#### Model Already Downloaded
```json
{
  "model_name": "bert-base",
  "requested_version": "v2.1",
  "availability": {
    "status": "already_available",
    "current_status": "loaded",
    "can_download": true,
    "can_load": true,
    "local_path": "/models/bert-base-v2.1.pth",
    "installed_at": "2025-08-12T09:15:00.000Z",
    "file_size_mb": 438.2,
    "sources": [
      {
        "name": "local_cache",
        "available": true,
        "url": "file:///models/bert-base-v2.1.pth",
        "size_mb": 438.2
      }
    ]
  },
  "metadata": {
    "description": "BERT base model for natural language understanding",
    "type": "text_processing",
    "framework": "pytorch",
    "precision": "fp16",
    "max_sequence_length": 512,
    "vocabulary_size": 30522,
    "memory_requirements_mb": 1200,
    "supported_tasks": ["classification", "question_answering", "feature_extraction"]
  }
}
```

#### Model Not Available
```json
{
  "model_name": "unknown-model",
  "requested_version": "latest",
  "availability": {
    "status": "not_available",
    "current_status": "not_found",
    "can_download": false,
    "can_load": false,
    "sources": [],
    "reason": "Model not found in any configured repository"
  },
  "alternatives": [
    {
      "name": "similar-model",
      "similarity_score": 0.85,
      "description": "Similar functionality to requested model",
      "available": true
    }
  ],
  "suggestions": [
    "Check model name spelling",
    "Try searching available models with /models endpoint",
    "Contact administrator to add model to repository"
  ]
}
```

#### Response Fields

##### Availability Object
| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Overall availability: "available", "already_available", "not_available", "partially_available" |
| `current_status` | string | Current model state: "loaded", "available", "not_loaded", "not_found", "error" |
| `can_download` | boolean | Whether model can be downloaded |
| `can_load` | boolean | Whether model can be loaded for inference |
| `local_path` | string | Local file path (if already downloaded) |
| `installed_at` | string | When model was installed locally |
| `file_size_mb` | number | Local file size in MB |
| `sources` | array | Available download sources |

##### Source Object
| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Source repository name |
| `available` | boolean | Whether source is accessible |
| `url` | string | Download URL |
| `size_mb` | number | File size in MB |
| `checksum` | string | File checksum for verification |
| `last_updated` | string | When source was last updated |

##### Metadata Object
| Field | Type | Description |
|-------|------|-------------|
| `description` | string | Model description |
| `type` | string | Model category/type |
| `framework` | string | ML framework (pytorch, tensorflow, onnx) |
| `precision` | string | Model precision (fp16, fp32, int8) |
| `input_shape` | array | Expected input tensor shape |
| `output_shape` | array | Output tensor shape |
| `parameters_count` | integer | Total number of parameters |
| `memory_requirements_mb` | number | Memory needed for inference |
| `supported_devices` | array | Compatible devices |
| `supported_formats` | array | Input format support |
| `tags` | array | Descriptive tags |

##### Versions Object
| Field | Type | Description |
|-------|------|-------------|
| `available` | array | All available versions |
| `latest` | string | Most recent version |
| `recommended` | string | Recommended stable version |

##### Requirements Object
| Field | Type | Description |
|-------|------|-------------|
| `minimum_python` | string | Minimum Python version |
| `dependencies` | array | Required package dependencies |
| `minimum_memory_mb` | number | Minimum RAM required |
| `minimum_storage_mb` | number | Minimum storage space needed |

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success - Availability information returned |
| 400 | Bad request - Invalid model name or parameters |
| 404 | Model not found in any repository |
| 500 | Internal server error |

## Examples

### Basic Availability Check

**Request:**
```bash
curl -X GET http://localhost:8000/model-available/resnet50
```

**Response:**
```json
{
  "model_name": "resnet50",
  "requested_version": "latest",
  "availability": {
    "status": "available",
    "current_status": "not_loaded",
    "can_download": true,
    "can_load": false
  }
}
```

### Check Specific Version

**Request:**
```bash
curl -X GET "http://localhost:8000/model-available/bert-base?version=v2.1"
```

### Check with Alternatives

**Request:**
```bash
curl -X GET "http://localhost:8000/model-available/unknown-model?include_alternatives=true"
```

### Python Model Checker

```python
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelSource:
    name: str
    available: bool
    url: str
    size_mb: float
    checksum: Optional[str] = None
    last_updated: Optional[str] = None

@dataclass
class ModelAvailability:
    model_name: str
    version: str
    status: str
    current_status: str
    can_download: bool
    can_load: bool
    sources: List[ModelSource]
    local_path: Optional[str] = None
    installed_at: Optional[str] = None

class ModelAvailabilityChecker:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def check_model_availability(
        self,
        model_name: str,
        version: str = "latest",
        include_alternatives: bool = False,
        check_sources: bool = True
    ) -> Dict:
        """Check if a model is available for download or use"""
        params = {
            'version': version,
            'include_alternatives': str(include_alternatives).lower(),
            'check_sources': str(check_sources).lower()
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/model-available/{model_name}",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "error"}
    
    def is_model_ready(self, model_name: str, version: str = "latest") -> bool:
        """Simple boolean check if model is ready to use"""
        result = self.check_model_availability(model_name, version)
        
        if "error" in result:
            return False
        
        availability = result.get('availability', {})
        return (availability.get('status') in ['available', 'already_available'] and 
                availability.get('can_load', False))
    
    def get_download_info(self, model_name: str, version: str = "latest") -> Dict:
        """Get download information for a model"""
        result = self.check_model_availability(model_name, version)
        
        if "error" in result:
            return {"error": result["error"]}
        
        availability = result.get('availability', {})
        
        if not availability.get('can_download', False):
            return {"error": "Model cannot be downloaded"}
        
        # Find best source (prefer official repository)
        sources = availability.get('sources', [])
        best_source = None
        
        for source in sources:
            if source['available']:
                if 'official' in source['name'].lower():
                    best_source = source
                    break
                elif not best_source:
                    best_source = source
        
        if not best_source:
            return {"error": "No available download sources"}
        
        return {
            "model_name": model_name,
            "version": version,
            "download_url": best_source['url'],
            "size_mb": best_source['size_mb'],
            "checksum": best_source.get('checksum'),
            "source": best_source['name'],
            "requirements": result.get('requirements', {}),
            "estimated_download_time": self._estimate_download_time(best_source['size_mb'])
        }
    
    def _estimate_download_time(self, size_mb: float, speed_mbps: float = 5.0) -> int:
        """Estimate download time in seconds"""
        return int(size_mb / speed_mbps)
    
    def check_system_compatibility(self, model_name: str, version: str = "latest") -> Dict:
        """Check if system meets model requirements"""
        result = self.check_model_availability(model_name, version)
        
        if "error" in result:
            return {"compatible": False, "error": result["error"]}
        
        requirements = result.get('requirements', {})
        metadata = result.get('metadata', {})
        
        compatibility_issues = []
        warnings = []
        
        # Check Python version
        min_python = requirements.get('minimum_python', '3.7')
        import sys
        current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        if current_python < min_python:
            compatibility_issues.append(
                f"Python {min_python}+ required, current: {current_python}"
            )
        
        # Check memory requirements
        min_memory = requirements.get('minimum_memory_mb', 0)
        if min_memory > 0:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024)
            
            if available_memory < min_memory:
                compatibility_issues.append(
                    f"Insufficient memory: {min_memory}MB required, "
                    f"{available_memory:.0f}MB available"
                )
            elif available_memory < min_memory * 1.5:
                warnings.append(
                    f"Low memory: {min_memory}MB required, "
                    f"{available_memory:.0f}MB available (recommend 1.5x requirement)"
                )
        
        # Check storage requirements
        min_storage = requirements.get('minimum_storage_mb', 0)
        if min_storage > 0:
            import shutil
            free_space = shutil.disk_usage('/').free / (1024 * 1024)
            
            if free_space < min_storage:
                compatibility_issues.append(
                    f"Insufficient storage: {min_storage}MB required, "
                    f"{free_space:.0f}MB available"
                )
        
        return {
            "compatible": len(compatibility_issues) == 0,
            "issues": compatibility_issues,
            "warnings": warnings,
            "requirements": requirements,
            "system_info": {
                "python_version": current_python,
                "available_memory_mb": int(psutil.virtual_memory().available / (1024 * 1024)),
                "free_storage_mb": int(shutil.disk_usage('/').free / (1024 * 1024))
            }
        }
    
    def find_similar_models(self, model_name: str) -> List[Dict]:
        """Find models similar to the requested one"""
        result = self.check_model_availability(
            model_name, 
            include_alternatives=True
        )
        
        return result.get('alternatives', [])
    
    def bulk_check_availability(self, model_list: List[Dict]) -> Dict:
        """Check availability for multiple models"""
        results = {}
        summary = {
            "total": len(model_list),
            "available": 0,
            "already_downloaded": 0,
            "not_available": 0,
            "errors": 0
        }
        
        for model_config in model_list:
            model_name = model_config.get('model_name')
            version = model_config.get('version', 'latest')
            
            try:
                result = self.check_model_availability(model_name, version)
                results[f"{model_name}_{version}"] = result
                
                status = result.get('availability', {}).get('status', 'error')
                
                if status == 'available':
                    summary['available'] += 1
                elif status == 'already_available':
                    summary['already_downloaded'] += 1
                elif status == 'not_available':
                    summary['not_available'] += 1
                else:
                    summary['errors'] += 1
                    
            except Exception as e:
                results[f"{model_name}_{version}"] = {"error": str(e)}
                summary['errors'] += 1
        
        return {
            "results": results,
            "summary": summary
        }

# Usage Examples
checker = ModelAvailabilityChecker()

# Basic availability check
availability = checker.check_model_availability("resnet50", "v1.0")
print(f"ResNet-50 v1.0 status: {availability['availability']['status']}")

# Simple ready check
is_ready = checker.is_model_ready("bert-base", "v2.1")
print(f"BERT base ready: {is_ready}")

# Get download information
download_info = checker.get_download_info("efficientnet", "latest")
if "error" not in download_info:
    print(f"Download size: {download_info['size_mb']} MB")
    print(f"Estimated time: {download_info['estimated_download_time']} seconds")

# Check system compatibility
compatibility = checker.check_system_compatibility("llama-7b", "v1.0")
if compatibility['compatible']:
    print("✅ System is compatible")
else:
    print("❌ Compatibility issues:")
    for issue in compatibility['issues']:
        print(f"  - {issue}")

if compatibility['warnings']:
    print("⚠️ Warnings:")
    for warning in compatibility['warnings']:
        print(f"  - {warning}")

# Bulk check
models_to_check = [
    {"model_name": "resnet50", "version": "v1.0"},
    {"model_name": "bert-base", "version": "v2.1"},
    {"model_name": "gpt2-small", "version": "latest"}
]

bulk_results = checker.bulk_check_availability(models_to_check)
print(f"\nBulk check summary:")
print(f"Total: {bulk_results['summary']['total']}")
print(f"Available: {bulk_results['summary']['available']}")
print(f"Already downloaded: {bulk_results['summary']['already_downloaded']}")
print(f"Not available: {bulk_results['summary']['not_available']}")
```

### JavaScript Availability Widget

```javascript
class ModelAvailabilityWidget {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }
  
  async checkAvailability(modelName, version = 'latest', options = {}) {
    const params = new URLSearchParams({
      version,
      include_alternatives: options.includeAlternatives || false,
      check_sources: options.checkSources !== false
    });
    
    try {
      const response = await fetch(
        `${this.baseUrl}/model-available/${modelName}?${params}`
      );
      return await response.json();
    } catch (error) {
      return { error: error.message };
    }
  }
  
  renderAvailabilityCard(data) {
    const availability = data.availability || {};
    const statusIcons = {
      'available': '🔵',
      'already_available': '🟢', 
      'not_available': '🔴',
      'partially_available': '🟡'
    };
    
    const statusColors = {
      'available': '#3b82f6',
      'already_available': '#22c55e',
      'not_available': '#ef4444', 
      'partially_available': '#f59e0b'
    };
    
    const status = availability.status || 'unknown';
    const icon = statusIcons[status] || '⚪';
    const color = statusColors[status] || '#6b7280';
    
    return `
      <div class="availability-card" data-status="${status}">
        <div class="card-header">
          <h3 class="model-name">
            <span class="status-icon" style="color: ${color}">${icon}</span>
            ${data.model_name}
          </h3>
          <span class="version-badge">${data.requested_version}</span>
        </div>
        
        <div class="status-section">
          <div class="status-item">
            <label>Status:</label>
            <span class="status-value" style="color: ${color}">
              ${status.replace('_', ' ').toUpperCase()}
            </span>
          </div>
          <div class="status-item">
            <label>Can Download:</label>
            <span class="${availability.can_download ? 'yes' : 'no'}">
              ${availability.can_download ? '✅' : '❌'}
            </span>
          </div>
          <div class="status-item">
            <label>Can Load:</label>
            <span class="${availability.can_load ? 'yes' : 'no'}">
              ${availability.can_load ? '✅' : '❌'}
            </span>
          </div>
        </div>
        
        ${this.renderSources(availability.sources || [])}
        ${this.renderMetadata(data.metadata)}
        ${this.renderVersions(data.versions)}
        ${this.renderRequirements(data.requirements)}
        ${this.renderAlternatives(data.alternatives)}
        ${this.renderActions(data)}
      </div>
    `;
  }
  
  renderSources(sources) {
    if (!sources.length) return '';
    
    return `
      <div class="sources-section">
        <h4>Available Sources</h4>
        <div class="sources-list">
          ${sources.map(source => `
            <div class="source-item">
              <div class="source-name">${source.name}</div>
              <div class="source-details">
                <span class="source-size">${source.size_mb} MB</span>
                ${source.checksum ? `<span class="source-checksum">✓ Verified</span>` : ''}
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }
  
  renderMetadata(metadata) {
    if (!metadata) return '';
    
    return `
      <div class="metadata-section">
        <h4>Model Information</h4>
        ${metadata.description ? `<p class="description">${metadata.description}</p>` : ''}
        <div class="metadata-grid">
          ${metadata.type ? `
            <div class="metadata-item">
              <label>Type:</label>
              <span>${metadata.type}</span>
            </div>
          ` : ''}
          ${metadata.framework ? `
            <div class="metadata-item">
              <label>Framework:</label>
              <span>${metadata.framework}</span>
            </div>
          ` : ''}
          ${metadata.parameters_count ? `
            <div class="metadata-item">
              <label>Parameters:</label>
              <span>${metadata.parameters_count.toLocaleString()}</span>
            </div>
          ` : ''}
          ${metadata.memory_requirements_mb ? `
            <div class="metadata-item">
              <label>Memory:</label>
              <span>${metadata.memory_requirements_mb} MB</span>
            </div>
          ` : ''}
        </div>
      </div>
    `;
  }
  
  renderVersions(versions) {
    if (!versions) return '';
    
    return `
      <div class="versions-section">
        <h4>Available Versions</h4>
        <div class="versions-info">
          <div class="version-item">
            <label>Latest:</label>
            <span class="version-tag latest">${versions.latest}</span>
          </div>
          ${versions.recommended ? `
            <div class="version-item">
              <label>Recommended:</label>
              <span class="version-tag recommended">${versions.recommended}</span>
            </div>
          ` : ''}
        </div>
        <div class="all-versions">
          ${versions.available.map(v => 
            `<span class="version-tag">${v}</span>`
          ).join('')}
        </div>
      </div>
    `;
  }
  
  renderRequirements(requirements) {
    if (!requirements) return '';
    
    return `
      <div class="requirements-section">
        <h4>System Requirements</h4>
        <div class="requirements-grid">
          ${requirements.minimum_python ? `
            <div class="req-item">
              <label>Python:</label>
              <span>${requirements.minimum_python}+</span>
            </div>
          ` : ''}
          ${requirements.minimum_memory_mb ? `
            <div class="req-item">
              <label>Memory:</label>
              <span>${requirements.minimum_memory_mb} MB</span>
            </div>
          ` : ''}
          ${requirements.minimum_storage_mb ? `
            <div class="req-item">
              <label>Storage:</label>
              <span>${requirements.minimum_storage_mb} MB</span>
            </div>
          ` : ''}
        </div>
        ${requirements.dependencies ? `
          <div class="dependencies">
            <label>Dependencies:</label>
            <div class="dep-list">
              ${requirements.dependencies.map(dep => 
                `<code class="dependency">${dep}</code>`
              ).join('')}
            </div>
          </div>
        ` : ''}
      </div>
    `;
  }
  
  renderAlternatives(alternatives) {
    if (!alternatives || !alternatives.length) return '';
    
    return `
      <div class="alternatives-section">
        <h4>Alternative Models</h4>
        <div class="alternatives-list">
          ${alternatives.map(alt => `
            <div class="alternative-item">
              <div class="alt-name">${alt.name}</div>
              <div class="alt-similarity">
                Similarity: ${(alt.similarity_score * 100).toFixed(0)}%
              </div>
              <div class="alt-description">${alt.description}</div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }
  
  renderActions(data) {
    const availability = data.availability || {};
    const actions = [];
    
    if (availability.can_download && availability.status === 'available') {
      actions.push(`
        <button class="btn-primary" onclick="downloadModel('${data.model_name}', '${data.requested_version}')">
          Download Model
        </button>
      `);
    }
    
    if (availability.can_load) {
      actions.push(`
        <button class="btn-secondary" onclick="loadModel('${data.model_name}', '${data.requested_version}')">
          Load for Inference
        </button>
      `);
    }
    
    actions.push(`
      <button class="btn-info" onclick="showDetailedInfo('${data.model_name}', '${data.requested_version}')">
        Detailed Info
      </button>
    `);
    
    return actions.length ? `
      <div class="actions-section">
        ${actions.join('')}
      </div>
    ` : '';
  }
  
  async renderWidget(containerId, modelName, version = 'latest') {
    const container = document.getElementById(containerId);
    
    // Show loading state
    container.innerHTML = `
      <div class="loading-card">
        <div class="loading-spinner"></div>
        <p>Checking availability for ${modelName}...</p>
      </div>
    `;
    
    // Fetch availability data
    const data = await this.checkAvailability(modelName, version, {
      includeAlternatives: true,
      checkSources: true
    });
    
    if (data.error) {
      container.innerHTML = `
        <div class="error-card">
          <h3>Error</h3>
          <p>${data.error}</p>
        </div>
      `;
    } else {
      container.innerHTML = this.renderAvailabilityCard(data);
    }
  }
}

// Global functions for button actions
async function downloadModel(modelName, version) {
  // Implementation would call model-download endpoint
  console.log(`Downloading ${modelName} v${version}`);
}

async function loadModel(modelName, version) {
  // Implementation would load model for inference
  console.log(`Loading ${modelName} v${version}`);
}

async function showDetailedInfo(modelName, version) {
  // Implementation would show detailed model info
  console.log(`Showing info for ${modelName} v${version}`);
}

// Usage
const widget = new ModelAvailabilityWidget();

// Render widget for a specific model
widget.renderWidget('availability-container', 'resnet50', 'v1.0');

// Multiple models
const modelsToCheck = ['resnet50', 'bert-base', 'gpt2-small'];
modelsToCheck.forEach((model, index) => {
  widget.renderWidget(`model-${index}`, model, 'latest');
});
```

### Batch Availability Checker

```bash
#!/bin/bash
# Check availability for multiple models

check_models_availability() {
    local models=("$@")
    
    echo "Checking availability for ${#models[@]} models..."
    echo "="*60
    
    for model in "${models[@]}"; do
        echo -n "Checking $model... "
        
        response=$(curl -s "http://localhost:8000/model-available/$model")
        status=$(echo "$response" | jq -r '.availability.status // "error"')
        
        case "$status" in
            "available")
                echo "🔵 Available for download"
                ;;
            "already_available") 
                echo "🟢 Already downloaded"
                ;;
            "not_available")
                echo "🔴 Not available"
                ;;
            *)
                echo "⚪ Unknown status: $status"
                ;;
        esac
    done
}

# Usage
models=(
    "resnet50"
    "bert-base" 
    "gpt2-small"
    "efficientnet"
    "mobilenet"
)

check_models_availability "${models[@]}"
```

## Error Handling

### Model Not Found (404)
```json
{
  "detail": "Model 'unknown-model' not found in any configured repository"
}
```

### Invalid Version (400)
```json
{
  "detail": "Invalid version format 'invalid-version'"
}
```

## Related Endpoints

- [Models List](./models.md) - List all available models
- [Model Download](./model-download.md) - Download models
- [Model Info](./model-info.md) - Detailed model information
- [Model Remove](./model-remove.md) - Remove models

## Best Practices

1. **Check Before Download**: Always check availability before attempting downloads
2. **Version Management**: Specify versions explicitly for production use
3. **System Requirements**: Verify compatibility before downloading large models
4. **Alternative Models**: Use alternatives when primary models are unavailable
5. **Caching**: Cache availability results to reduce API calls
6. **Error Handling**: Handle network timeouts and repository unavailability

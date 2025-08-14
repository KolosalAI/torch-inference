# GET /models - List Available Models

**URL**: `/models`  
**Method**: `GET`  
**Authentication**: Not required  
**Content-Type**: `application/json`

## Description

Retrieves a list of all available models that can be used for inference. This includes both loaded models and models available for download from configured sources.

## Request

### URL Parameters
None

### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `loaded_only` | boolean | No | false | Return only currently loaded models |
| `include_metadata` | boolean | No | false | Include detailed model metadata |
| `category` | string | No | - | Filter by model category/type |

### Request Body
None (GET request)

## Response

### Success Response (200 OK)

#### Basic Response (default)
```json
{
  "models": [
    {
      "name": "resnet50",
      "status": "loaded",
      "version": "v1.0",
      "type": "image_classification"
    },
    {
      "name": "bert-base",
      "status": "available",
      "version": "v2.1",
      "type": "text_processing"
    },
    {
      "name": "custom-transformer",
      "status": "loading",
      "version": "v1.5",
      "type": "text_generation"
    }
  ],
  "total_count": 3,
  "loaded_count": 1,
  "available_count": 2
}
```

#### Extended Response (include_metadata=true)
```json
{
  "models": [
    {
      "name": "resnet50",
      "status": "loaded",
      "version": "v1.0",
      "type": "image_classification",
      "metadata": {
        "size_mb": 97.8,
        "description": "ResNet-50 deep residual network for image classification",
        "input_shape": [3, 224, 224],
        "output_shape": [1000],
        "framework": "pytorch",
        "precision": "fp32",
        "device": "cpu",
        "loaded_at": "2025-08-14T10:30:00.000Z",
        "memory_usage_mb": 125.4,
        "parameters_count": 25557032,
        "supported_formats": ["tensor", "numpy", "pil"],
        "preprocessing": {
          "normalize": true,
          "mean": [0.485, 0.456, 0.406],
          "std": [0.229, 0.224, 0.225]
        }
      }
    },
    {
      "name": "bert-base",
      "status": "available",
      "version": "v2.1",
      "type": "text_processing",
      "metadata": {
        "size_mb": 438.2,
        "description": "BERT base model for text understanding and classification",
        "max_sequence_length": 512,
        "vocabulary_size": 30522,
        "framework": "pytorch",
        "precision": "fp16",
        "download_url": "https://models.example.com/bert-base-v2.1.pth",
        "checksum": "sha256:abc123...",
        "requirements": ["transformers>=4.0.0", "torch>=1.8.0"],
        "supported_tasks": ["classification", "question_answering", "feature_extraction"]
      }
    }
  ],
  "total_count": 2,
  "loaded_count": 1,
  "available_count": 1,
  "categories": ["image_classification", "text_processing"],
  "total_memory_usage_mb": 125.4,
  "repository_info": {
    "last_updated": "2025-08-14T09:15:00.000Z",
    "source": "https://models.example.com/registry"
  }
}
```

#### Response Fields

##### Model Object
| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique model identifier |
| `status` | string | Model status: "loaded", "available", "loading", "error" |
| `version` | string | Model version identifier |
| `type` | string | Model category/type |
| `metadata` | object | Detailed model information (if requested) |

##### Model Status Values
- **loaded**: Model is currently loaded in memory and ready for inference
- **available**: Model is available for download/loading but not currently loaded
- **loading**: Model is currently being loaded into memory
- **error**: Model failed to load or has encountered an error

##### Metadata Object (when include_metadata=true)
| Field | Type | Description |
|-------|------|-------------|
| `size_mb` | number | Model file size in megabytes |
| `description` | string | Human-readable model description |
| `input_shape` | array | Expected input tensor shape |
| `output_shape` | array | Output tensor shape |
| `framework` | string | Framework used (pytorch, tensorflow, onnx) |
| `precision` | string | Model precision (fp16, fp32, int8) |
| `device` | string | Device where model is loaded (cpu, cuda:0, etc.) |
| `loaded_at` | string | When model was loaded (ISO format) |
| `memory_usage_mb` | number | Current memory usage in MB |
| `parameters_count` | integer | Total number of model parameters |
| `download_url` | string | URL for downloading model (if available) |
| `checksum` | string | Model file checksum for verification |
| `requirements` | array | Required dependencies |
| `supported_tasks` | array | Tasks the model can perform |
| `preprocessing` | object | Preprocessing requirements |

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success - Models list returned |
| 400 | Bad request - Invalid query parameters |
| 500 | Internal server error |

## Examples

### Basic Model List Request

**Request:**
```bash
curl -X GET http://localhost:8000/models
```

**Response:**
```json
{
  "models": [
    {
      "name": "resnet50",
      "status": "loaded",
      "version": "v1.0",
      "type": "image_classification"
    },
    {
      "name": "bert-base",
      "status": "available",
      "version": "v2.1",
      "type": "text_processing"
    }
  ],
  "total_count": 2,
  "loaded_count": 1,
  "available_count": 1
}
```

### Loaded Models Only

**Request:**
```bash
curl -X GET "http://localhost:8000/models?loaded_only=true"
```

**Response:**
```json
{
  "models": [
    {
      "name": "resnet50",
      "status": "loaded",
      "version": "v1.0",
      "type": "image_classification"
    }
  ],
  "total_count": 1,
  "loaded_count": 1,
  "available_count": 0
}
```

### Models with Metadata

**Request:**
```bash
curl -X GET "http://localhost:8000/models?include_metadata=true"
```

### Filter by Category

**Request:**
```bash
curl -X GET "http://localhost:8000/models?category=text_processing"
```

**Response:**
```json
{
  "models": [
    {
      "name": "bert-base",
      "status": "available",
      "version": "v2.1",
      "type": "text_processing"
    },
    {
      "name": "gpt2-small",
      "status": "loaded",
      "version": "v1.0",
      "type": "text_processing"
    }
  ],
  "total_count": 2,
  "loaded_count": 1,
  "available_count": 1
}
```

### Python Model Explorer

```python
import requests
import json
from typing import List, Dict, Optional
from datetime import datetime

class ModelExplorer:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def get_models(self, loaded_only=False, include_metadata=False, category=None):
        params = {}
        if loaded_only:
            params['loaded_only'] = 'true'
        if include_metadata:
            params['include_metadata'] = 'true'
        if category:
            params['category'] = category
        
        try:
            response = requests.get(f"{self.base_url}/models", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching models: {e}")
            return None
    
    def list_models(self, detailed=False):
        """List all models with optional details"""
        models_data = self.get_models(include_metadata=detailed)
        
        if not models_data:
            return
        
        print("="*80)
        print("AVAILABLE MODELS")
        print("="*80)
        print(f"Total: {models_data['total_count']} | "
              f"Loaded: {models_data['loaded_count']} | "
              f"Available: {models_data['available_count']}")
        print("-" * 80)
        
        for model in models_data['models']:
            status_emoji = {
                'loaded': '🟢',
                'available': '🔵',
                'loading': '🟡',
                'error': '🔴'
            }.get(model['status'], '⚪')
            
            print(f"{status_emoji} {model['name']} ({model['version']})")
            print(f"   Type: {model['type']}")
            print(f"   Status: {model['status'].upper()}")
            
            if detailed and 'metadata' in model:
                meta = model['metadata']
                print(f"   Size: {meta.get('size_mb', 'Unknown')} MB")
                print(f"   Framework: {meta.get('framework', 'Unknown')}")
                print(f"   Precision: {meta.get('precision', 'Unknown')}")
                
                if 'memory_usage_mb' in meta:
                    print(f"   Memory Usage: {meta['memory_usage_mb']} MB")
                
                if 'device' in meta:
                    print(f"   Device: {meta['device']}")
            
            print()
    
    def get_loaded_models(self):
        """Get only currently loaded models"""
        return self.get_models(loaded_only=True, include_metadata=True)
    
    def get_model_by_name(self, name: str):
        """Find specific model by name"""
        models_data = self.get_models(include_metadata=True)
        
        if not models_data:
            return None
        
        for model in models_data['models']:
            if model['name'] == name:
                return model
        
        return None
    
    def get_models_by_type(self, model_type: str):
        """Get models filtered by type"""
        return self.get_models(category=model_type, include_metadata=True)
    
    def analyze_memory_usage(self):
        """Analyze memory usage of loaded models"""
        models_data = self.get_models(loaded_only=True, include_metadata=True)
        
        if not models_data or not models_data['models']:
            print("No loaded models found")
            return
        
        print("="*60)
        print("MEMORY USAGE ANALYSIS")
        print("="*60)
        
        total_memory = 0
        model_sizes = []
        
        for model in models_data['models']:
            if 'metadata' in model and 'memory_usage_mb' in model['metadata']:
                memory_mb = model['metadata']['memory_usage_mb']
                total_memory += memory_mb
                model_sizes.append((model['name'], memory_mb))
                
                print(f"{model['name']:<20} {memory_mb:>8.1f} MB")
        
        print("-" * 60)
        print(f"{'Total Memory Usage':<20} {total_memory:>8.1f} MB")
        print(f"{'Average per Model':<20} {total_memory/len(model_sizes):>8.1f} MB")
        
        # Show largest model
        if model_sizes:
            largest = max(model_sizes, key=lambda x: x[1])
            print(f"{'Largest Model':<20} {largest[0]} ({largest[1]:.1f} MB)")
    
    def check_model_compatibility(self, model_name: str):
        """Check if a model is compatible with current setup"""
        model = self.get_model_by_name(model_name)
        
        if not model:
            print(f"Model '{model_name}' not found")
            return
        
        print(f"COMPATIBILITY CHECK: {model_name}")
        print("="*50)
        
        if model['status'] == 'loaded':
            print("✅ Model is currently loaded and ready")
        elif model['status'] == 'available':
            print("🔵 Model is available for loading")
            
            if 'metadata' in model:
                meta = model['metadata']
                
                # Check size
                if 'size_mb' in meta:
                    size_mb = meta['size_mb']
                    if size_mb > 1000:
                        print(f"⚠️  Large model: {size_mb} MB")
                    else:
                        print(f"✅ Reasonable size: {size_mb} MB")
                
                # Check requirements
                if 'requirements' in meta:
                    print("📋 Requirements:")
                    for req in meta['requirements']:
                        print(f"   - {req}")
        
        elif model['status'] == 'error':
            print("❌ Model has errors and cannot be loaded")
        
        elif model['status'] == 'loading':
            print("🟡 Model is currently loading...")

# Usage Examples
explorer = ModelExplorer()

# List all models
explorer.list_models()

# List models with detailed information
print("\n" + "="*80)
print("DETAILED MODEL INFORMATION")
explorer.list_models(detailed=True)

# Analyze memory usage
print("\n")
explorer.analyze_memory_usage()

# Check specific model
print("\n")
explorer.check_model_compatibility("resnet50")

# Get models by type
text_models = explorer.get_models_by_type("text_processing")
if text_models:
    print(f"\nFound {text_models['total_count']} text processing models")
```

### JavaScript Model Dashboard

```javascript
class ModelDashboard {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.models = [];
    this.categories = new Set();
  }
  
  async fetchModels(options = {}) {
    const params = new URLSearchParams();
    
    if (options.loadedOnly) params.append('loaded_only', 'true');
    if (options.includeMetadata) params.append('include_metadata', 'true');
    if (options.category) params.append('category', options.category);
    
    try {
      const response = await fetch(`${this.baseUrl}/models?${params}`);
      const data = await response.json();
      
      this.models = data.models;
      data.models.forEach(model => this.categories.add(model.type));
      
      return data;
    } catch (error) {
      console.error('Error fetching models:', error);
      return null;
    }
  }
  
  renderModelCard(model) {
    const statusColors = {
      'loaded': '#22c55e',
      'available': '#3b82f6',
      'loading': '#f59e0b',
      'error': '#ef4444'
    };
    
    const statusIcons = {
      'loaded': '●',
      'available': '○',
      'loading': '◐',
      'error': '✕'
    };
    
    return `
      <div class="model-card" data-status="${model.status}" data-type="${model.type}">
        <div class="model-header">
          <h3 class="model-name">
            <span class="status-indicator" style="color: ${statusColors[model.status]}">
              ${statusIcons[model.status]}
            </span>
            ${model.name}
          </h3>
          <span class="model-version">${model.version}</span>
        </div>
        
        <div class="model-info">
          <div class="info-item">
            <label>Type:</label>
            <span class="model-type">${model.type}</span>
          </div>
          <div class="info-item">
            <label>Status:</label>
            <span class="model-status status-${model.status}">${model.status.toUpperCase()}</span>
          </div>
          
          ${model.metadata ? this.renderMetadata(model.metadata) : ''}
        </div>
        
        <div class="model-actions">
          ${this.renderModelActions(model)}
        </div>
      </div>
    `;
  }
  
  renderMetadata(metadata) {
    return `
      <div class="metadata-section">
        ${metadata.description ? `<p class="model-description">${metadata.description}</p>` : ''}
        
        <div class="metadata-grid">
          ${metadata.size_mb ? `
            <div class="metadata-item">
              <label>Size:</label>
              <span>${metadata.size_mb} MB</span>
            </div>
          ` : ''}
          
          ${metadata.framework ? `
            <div class="metadata-item">
              <label>Framework:</label>
              <span>${metadata.framework}</span>
            </div>
          ` : ''}
          
          ${metadata.precision ? `
            <div class="metadata-item">
              <label>Precision:</label>
              <span>${metadata.precision}</span>
            </div>
          ` : ''}
          
          ${metadata.device ? `
            <div class="metadata-item">
              <label>Device:</label>
              <span>${metadata.device}</span>
            </div>
          ` : ''}
          
          ${metadata.memory_usage_mb ? `
            <div class="metadata-item">
              <label>Memory:</label>
              <span>${metadata.memory_usage_mb} MB</span>
            </div>
          ` : ''}
          
          ${metadata.parameters_count ? `
            <div class="metadata-item">
              <label>Parameters:</label>
              <span>${metadata.parameters_count.toLocaleString()}</span>
            </div>
          ` : ''}
        </div>
        
        ${metadata.supported_tasks ? `
          <div class="supported-tasks">
            <label>Supported Tasks:</label>
            <div class="task-tags">
              ${metadata.supported_tasks.map(task => 
                `<span class="task-tag">${task}</span>`
              ).join('')}
            </div>
          </div>
        ` : ''}
      </div>
    `;
  }
  
  renderModelActions(model) {
    switch (model.status) {
      case 'loaded':
        return `
          <button class="btn-primary" onclick="dashboard.testModel('${model.name}')">
            Test Model
          </button>
          <button class="btn-secondary" onclick="dashboard.unloadModel('${model.name}')">
            Unload
          </button>
        `;
      case 'available':
        return `
          <button class="btn-primary" onclick="dashboard.loadModel('${model.name}')">
            Load Model
          </button>
          <button class="btn-info" onclick="dashboard.showModelInfo('${model.name}')">
            Info
          </button>
        `;
      case 'loading':
        return `
          <button class="btn-disabled" disabled>
            Loading...
          </button>
        `;
      case 'error':
        return `
          <button class="btn-warning" onclick="dashboard.retryModel('${model.name}')">
            Retry
          </button>
          <button class="btn-info" onclick="dashboard.showError('${model.name}')">
            Show Error
          </button>
        `;
      default:
        return '';
    }
  }
  
  renderDashboard(container, data) {
    const html = `
      <div class="dashboard-header">
        <h2>Model Dashboard</h2>
        <div class="dashboard-stats">
          <div class="stat-item">
            <span class="stat-value">${data.total_count}</span>
            <span class="stat-label">Total</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">${data.loaded_count}</span>
            <span class="stat-label">Loaded</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">${data.available_count}</span>
            <span class="stat-label">Available</span>
          </div>
        </div>
      </div>
      
      <div class="dashboard-filters">
        <select id="category-filter" onchange="dashboard.filterByCategory(this.value)">
          <option value="">All Categories</option>
          ${Array.from(this.categories).map(cat => 
            `<option value="${cat}">${cat.replace('_', ' ').toUpperCase()}</option>`
          ).join('')}
        </select>
        
        <select id="status-filter" onchange="dashboard.filterByStatus(this.value)">
          <option value="">All Status</option>
          <option value="loaded">Loaded</option>
          <option value="available">Available</option>
          <option value="loading">Loading</option>
          <option value="error">Error</option>
        </select>
        
        <button onclick="dashboard.refreshModels()" class="btn-refresh">
          🔄 Refresh
        </button>
      </div>
      
      <div class="models-grid">
        ${data.models.map(model => this.renderModelCard(model)).join('')}
      </div>
      
      ${data.total_memory_usage_mb ? `
        <div class="memory-usage">
          <h3>Memory Usage: ${data.total_memory_usage_mb} MB</h3>
          <div class="memory-bar">
            <div class="memory-used" style="width: ${Math.min(100, (data.total_memory_usage_mb / 1000) * 100)}%"></div>
          </div>
        </div>
      ` : ''}
    `;
    
    container.innerHTML = html;
  }
  
  async loadAndDisplay(containerId) {
    const container = document.getElementById(containerId);
    const data = await this.fetchModels({ includeMetadata: true });
    
    if (data) {
      this.renderDashboard(container, data);
    }
  }
  
  async filterByCategory(category) {
    const container = document.getElementById('models-container');
    const options = category ? { category, includeMetadata: true } : { includeMetadata: true };
    const data = await this.fetchModels(options);
    
    if (data) {
      this.renderDashboard(container, data);
    }
  }
  
  async filterByStatus(status) {
    const container = document.getElementById('models-container');
    const loadedOnly = status === 'loaded';
    const data = await this.fetchModels({ loadedOnly, includeMetadata: true });
    
    if (data) {
      // Further filter by status if needed
      if (status && status !== 'loaded') {
        data.models = data.models.filter(model => model.status === status);
        data.total_count = data.models.length;
      }
      
      this.renderDashboard(container, data);
    }
  }
  
  async refreshModels() {
    await this.loadAndDisplay('models-container');
  }
}

// Initialize dashboard
const dashboard = new ModelDashboard();
dashboard.loadAndDisplay('models-container');
```

### Model Status Monitoring

```bash
#!/bin/bash
# Monitor model loading status

monitor_models() {
    echo "Monitoring model status..."
    
    while true; do
        # Get models with status
        response=$(curl -s "http://localhost:8000/models?include_metadata=true")
        
        # Extract loading models
        loading_models=$(echo "$response" | jq -r '.models[] | select(.status == "loading") | .name')
        
        if [[ -n "$loading_models" ]]; then
            echo "$(date): Models loading: $loading_models"
        fi
        
        # Check for errors
        error_models=$(echo "$response" | jq -r '.models[] | select(.status == "error") | .name')
        
        if [[ -n "$error_models" ]]; then
            echo "$(date): Models with errors: $error_models"
        fi
        
        sleep 5
    done
}

# Usage
monitor_models
```

## Error Handling

### Bad Request (400)
```json
{
  "detail": "Invalid category filter"
}
```

### Internal Server Error (500)
```json
{
  "detail": "Failed to retrieve models list"
}
```

## Related Endpoints

- [Model Download](./model-download.md) - Download and install models
- [Model Info](./model-info.md) - Get detailed information about specific model
- [Model Remove](./model-remove.md) - Remove models from system
- [Predict](./predict.md) - Use loaded models for inference

## Best Practices

1. **Regular Updates**: Check for new models periodically
2. **Memory Management**: Monitor loaded models memory usage
3. **Filtering**: Use query parameters to reduce response size
4. **Caching**: Cache model lists on client side with appropriate TTL
5. **Status Monitoring**: Monitor loading/error status for critical models

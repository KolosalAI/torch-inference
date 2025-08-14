# GET /model-info/{model_name} - Get Model Information

**URL**: `/model-info/{model_name}`  
**Method**: `GET`  
**Authentication**: Not required  
**Content-Type**: `application/json`

## Description

Retrieves detailed information about a specific model, including metadata, configuration, performance metrics, and usage statistics. This endpoint provides comprehensive model documentation and technical specifications.

## Request

### URL Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | string | Yes | Name of the model to get information for |

### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `version` | string | No | "latest" | Specific version to get information for |
| `include_performance` | boolean | No | false | Include performance metrics and benchmarks |
| `include_usage_stats` | boolean | No | false | Include usage statistics |
| `include_examples` | boolean | No | false | Include usage examples and code samples |

### Request Body
None (GET request)

## Response

### Success Response (200 OK)

#### Basic Model Information
```json
{
  "model_name": "resnet50",
  "version": "v1.0",
  "status": "loaded",
  "basic_info": {
    "display_name": "ResNet-50",
    "description": "Deep residual network with 50 layers for image classification",
    "author": "Microsoft Research",
    "license": "MIT",
    "created_at": "2024-01-15T10:30:00.000Z",
    "last_updated": "2024-08-10T14:20:00.000Z",
    "homepage": "https://github.com/microsoft/ResNet",
    "paper_url": "https://arxiv.org/abs/1512.03385",
    "model_type": "image_classification",
    "framework": "pytorch",
    "tags": ["computer_vision", "classification", "imagenet", "deep_learning"]
  },
  "technical_specs": {
    "architecture": "Residual Neural Network",
    "layers_count": 50,
    "parameters_count": 25557032,
    "model_size_mb": 97.8,
    "memory_usage_mb": 125.4,
    "precision": "fp32",
    "supported_devices": ["cpu", "cuda"],
    "framework_version": "pytorch==1.12.0",
    "input_format": {
      "type": "tensor",
      "shape": [3, 224, 224],
      "dtype": "float32",
      "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
      }
    },
    "output_format": {
      "type": "tensor", 
      "shape": [1000],
      "dtype": "float32",
      "description": "Class probabilities for 1000 ImageNet classes"
    }
  },
  "deployment_info": {
    "file_path": "/models/resnet50-v1.0.pth",
    "file_size_mb": 97.8,
    "checksum": "sha256:abc123def456789...",
    "loaded_at": "2025-08-14T10:30:00.000Z",
    "device": "cpu",
    "optimization_level": "none",
    "ready_for_inference": true
  }
}
```

#### Extended Model Information (with performance and usage stats)
```json
{
  "model_name": "resnet50",
  "version": "v1.0",
  "status": "loaded",
  "basic_info": {
    "display_name": "ResNet-50",
    "description": "Deep residual network with 50 layers for image classification",
    "author": "Microsoft Research",
    "license": "MIT",
    "model_type": "image_classification",
    "framework": "pytorch",
    "tags": ["computer_vision", "classification", "imagenet"]
  },
  "technical_specs": {
    "architecture": "Residual Neural Network",
    "layers_count": 50,
    "parameters_count": 25557032,
    "model_size_mb": 97.8,
    "precision": "fp32",
    "input_format": {
      "type": "tensor",
      "shape": [3, 224, 224],
      "dtype": "float32"
    },
    "output_format": {
      "type": "tensor",
      "shape": [1000],
      "dtype": "float32"
    }
  },
  "performance_metrics": {
    "accuracy": {
      "top1": 76.15,
      "top5": 92.87,
      "dataset": "ImageNet-1K validation"
    },
    "speed_benchmarks": {
      "inference_time_ms": {
        "cpu": {
          "single": 45.2,
          "batch_4": 156.8,
          "batch_8": 302.1
        },
        "gpu": {
          "single": 2.1,
          "batch_4": 6.8,
          "batch_8": 12.4
        }
      },
      "throughput_fps": {
        "cpu": {
          "single": 22.1,
          "batch_4": 25.5,
          "batch_8": 26.5
        },
        "gpu": {
          "single": 476.2,
          "batch_4": 588.2,
          "batch_8": 645.1
        }
      }
    },
    "memory_usage": {
      "peak_memory_mb": {
        "cpu": 125.4,
        "gpu": 892.3
      },
      "memory_per_sample_mb": {
        "cpu": 15.7,
        "gpu": 111.5
      }
    },
    "benchmark_date": "2025-08-10T09:00:00.000Z",
    "benchmark_environment": {
      "cpu": "Intel Xeon E5-2686 v4",
      "gpu": "NVIDIA Tesla V100",
      "memory": "32GB",
      "pytorch_version": "1.12.0"
    }
  },
  "usage_statistics": {
    "total_requests": 15420,
    "successful_predictions": 15398,
    "failed_predictions": 22,
    "success_rate": 99.86,
    "avg_response_time_ms": 48.5,
    "requests_per_day": {
      "last_7_days": [2150, 2340, 1980, 2100, 2250, 1890, 2710],
      "avg": 2203
    },
    "most_common_errors": [
      {
        "error_type": "InvalidInputShape",
        "count": 15,
        "percentage": 68.2
      },
      {
        "error_type": "DeviceMemoryError", 
        "count": 7,
        "percentage": 31.8
      }
    ],
    "usage_patterns": {
      "peak_hours": ["09:00-11:00", "14:00-16:00"],
      "avg_batch_size": 2.8,
      "most_common_input_sizes": [
        {"size": [3, 224, 224], "percentage": 87.4},
        {"size": [3, 256, 256], "percentage": 8.9},
        {"size": [3, 384, 384], "percentage": 3.7}
      ]
    },
    "statistics_period": "last_30_days",
    "last_updated": "2025-08-14T08:00:00.000Z"
  },
  "examples": {
    "basic_usage": {
      "description": "Simple image classification example",
      "code": "import requests\nimport base64\n\nwith open('image.jpg', 'rb') as f:\n    image_data = base64.b64encode(f.read()).decode()\n\nresponse = requests.post(\n    'http://localhost:8000/predict',\n    json={\n        'model_name': 'resnet50',\n        'input_data': {'image': image_data}\n    }\n)\n\nresult = response.json()\nprint(f\"Predicted class: {result['predictions'][0]['class']}\")\nprint(f\"Confidence: {result['predictions'][0]['confidence']:.3f}\")"
    },
    "batch_processing": {
      "description": "Processing multiple images at once",
      "code": "import requests\nimport base64\n\n# Encode multiple images\nimage_data = []\nfor img_path in ['img1.jpg', 'img2.jpg', 'img3.jpg']:\n    with open(img_path, 'rb') as f:\n        image_data.append(base64.b64encode(f.read()).decode())\n\nresponse = requests.post(\n    'http://localhost:8000/batch-predict',\n    json={\n        'model_name': 'resnet50',\n        'batch_data': [{'image': img} for img in image_data]\n    }\n)\n\nresults = response.json()\nfor i, result in enumerate(results['predictions']):\n    print(f\"Image {i+1}: {result[0]['class']} ({result[0]['confidence']:.3f})\")"
    },
    "preprocessing": {
      "description": "Custom preprocessing for better accuracy",
      "code": "import torch\nimport torchvision.transforms as transforms\nfrom PIL import Image\n\n# Define preprocessing pipeline\ntransform = transforms.Compose([\n    transforms.Resize(256),\n    transforms.CenterCrop(224),\n    transforms.ToTensor(),\n    transforms.Normalize(\n        mean=[0.485, 0.456, 0.406],\n        std=[0.229, 0.224, 0.225]\n    )\n])\n\n# Preprocess image\nimage = Image.open('image.jpg')\ntensor = transform(image).unsqueeze(0)\n\n# Convert to base64 for API\nimport base64\nimport io\nbuffer = io.BytesIO()\ntorch.save(tensor, buffer)\nbuffer.seek(0)\ntensor_b64 = base64.b64encode(buffer.read()).decode()\n\nresponse = requests.post(\n    'http://localhost:8000/predict',\n    json={\n        'model_name': 'resnet50',\n        'input_data': {'tensor': tensor_b64},\n        'input_format': 'tensor'\n    }\n)"
    }
  }
}
```

#### Response Fields

##### Basic Info Object
| Field | Type | Description |
|-------|------|-------------|
| `display_name` | string | Human-readable model name |
| `description` | string | Detailed model description |
| `author` | string | Model creator/organization |
| `license` | string | License type (MIT, Apache, etc.) |
| `created_at` | string | Model creation timestamp |
| `last_updated` | string | Last update timestamp |
| `homepage` | string | Model homepage/repository URL |
| `paper_url` | string | Academic paper URL |
| `model_type` | string | Model category/type |
| `framework` | string | ML framework used |
| `tags` | array | Descriptive tags |

##### Technical Specs Object
| Field | Type | Description |
|-------|------|-------------|
| `architecture` | string | Model architecture type |
| `layers_count` | integer | Number of layers |
| `parameters_count` | integer | Total trainable parameters |
| `model_size_mb` | number | Model file size in MB |
| `memory_usage_mb` | number | Runtime memory usage |
| `precision` | string | Model precision (fp16, fp32, int8) |
| `supported_devices` | array | Compatible devices |
| `framework_version` | string | Framework version requirement |
| `input_format` | object | Input specification |
| `output_format` | object | Output specification |

##### Performance Metrics Object
| Field | Type | Description |
|-------|------|-------------|
| `accuracy` | object | Model accuracy metrics |
| `speed_benchmarks` | object | Performance benchmarks |
| `memory_usage` | object | Memory usage statistics |
| `benchmark_date` | string | When benchmarks were run |
| `benchmark_environment` | object | Hardware used for benchmarking |

##### Usage Statistics Object
| Field | Type | Description |
|-------|------|-------------|
| `total_requests` | integer | Total API requests |
| `successful_predictions` | integer | Successful prediction count |
| `failed_predictions` | integer | Failed prediction count |
| `success_rate` | number | Success rate percentage |
| `avg_response_time_ms` | number | Average response time |
| `requests_per_day` | object | Daily request patterns |
| `most_common_errors` | array | Error frequency breakdown |
| `usage_patterns` | object | Usage pattern analysis |

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success - Model information returned |
| 404 | Model not found |
| 400 | Bad request - Invalid parameters |
| 500 | Internal server error |

## Examples

### Basic Model Information

**Request:**
```bash
curl -X GET http://localhost:8000/model-info/resnet50
```

**Response:**
```json
{
  "model_name": "resnet50",
  "version": "latest",
  "status": "loaded",
  "basic_info": {
    "display_name": "ResNet-50",
    "description": "Deep residual network with 50 layers for image classification",
    "model_type": "image_classification",
    "framework": "pytorch"
  },
  "technical_specs": {
    "parameters_count": 25557032,
    "model_size_mb": 97.8,
    "precision": "fp32"
  }
}
```

### Information with Performance Metrics

**Request:**
```bash
curl -X GET "http://localhost:8000/model-info/resnet50?include_performance=true"
```

### Information with Usage Statistics

**Request:**
```bash
curl -X GET "http://localhost:8000/model-info/bert-base?version=v2.1&include_usage_stats=true"
```

### Complete Information with Examples

**Request:**
```bash
curl -X GET "http://localhost:8000/model-info/resnet50?include_performance=true&include_usage_stats=true&include_examples=true"
```

### Python Model Information Inspector

```python
import requests
import json
from typing import Dict, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class ModelInformationInspector:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def get_model_info(
        self,
        model_name: str,
        version: str = "latest",
        include_performance: bool = False,
        include_usage_stats: bool = False,
        include_examples: bool = False
    ) -> Dict:
        """Get comprehensive model information"""
        params = {
            'version': version,
            'include_performance': str(include_performance).lower(),
            'include_usage_stats': str(include_usage_stats).lower(),
            'include_examples': str(include_examples).lower()
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/model-info/{model_name}",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def print_model_summary(self, model_info: Dict):
        """Print a formatted model summary"""
        if "error" in model_info:
            print(f"❌ Error: {model_info['error']}")
            return
        
        print("="*80)
        print(f"MODEL INFORMATION: {model_info['model_name'].upper()}")
        print("="*80)
        
        # Basic Information
        basic = model_info.get('basic_info', {})
        print(f"Display Name: {basic.get('display_name', 'N/A')}")
        print(f"Version: {model_info.get('version', 'N/A')}")
        print(f"Status: {model_info.get('status', 'N/A').upper()}")
        print(f"Type: {basic.get('model_type', 'N/A')}")
        print(f"Framework: {basic.get('framework', 'N/A')}")
        print(f"Author: {basic.get('author', 'N/A')}")
        print(f"License: {basic.get('license', 'N/A')}")
        
        if basic.get('description'):
            print(f"\nDescription:")
            print(f"  {basic['description']}")
        
        # Technical Specifications
        specs = model_info.get('technical_specs', {})
        if specs:
            print(f"\nTECHNICAL SPECIFICATIONS:")
            print(f"  Architecture: {specs.get('architecture', 'N/A')}")
            print(f"  Parameters: {specs.get('parameters_count', 0):,}")
            print(f"  Model Size: {specs.get('model_size_mb', 0)} MB")
            print(f"  Memory Usage: {specs.get('memory_usage_mb', 0)} MB")
            print(f"  Precision: {specs.get('precision', 'N/A')}")
            
            if 'supported_devices' in specs:
                print(f"  Supported Devices: {', '.join(specs['supported_devices'])}")
            
            # Input/Output Format
            if 'input_format' in specs:
                inp = specs['input_format']
                print(f"  Input: {inp.get('type', 'N/A')} {inp.get('shape', [])}")
            
            if 'output_format' in specs:
                out = specs['output_format']
                print(f"  Output: {out.get('type', 'N/A')} {out.get('shape', [])}")
        
        # Performance Metrics
        if 'performance_metrics' in model_info:
            self._print_performance_metrics(model_info['performance_metrics'])
        
        # Usage Statistics
        if 'usage_statistics' in model_info:
            self._print_usage_statistics(model_info['usage_statistics'])
        
        # Examples
        if 'examples' in model_info:
            self._print_examples(model_info['examples'])
        
        print("="*80)
    
    def _print_performance_metrics(self, metrics: Dict):
        """Print performance metrics section"""
        print(f"\nPERFORMANCE METRICS:")
        
        # Accuracy
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            print(f"  Accuracy (on {acc.get('dataset', 'test set')}):")
            if 'top1' in acc:
                print(f"    Top-1: {acc['top1']:.2f}%")
            if 'top5' in acc:
                print(f"    Top-5: {acc['top5']:.2f}%")
        
        # Speed Benchmarks
        if 'speed_benchmarks' in metrics:
            speed = metrics['speed_benchmarks']
            print(f"  Speed Benchmarks:")
            
            if 'inference_time_ms' in speed:
                print(f"    Inference Time (ms):")
                for device, times in speed['inference_time_ms'].items():
                    print(f"      {device.upper()}:")
                    for batch, time in times.items():
                        print(f"        {batch}: {time} ms")
            
            if 'throughput_fps' in speed:
                print(f"    Throughput (FPS):")
                for device, fps in speed['throughput_fps'].items():
                    print(f"      {device.upper()}:")
                    for batch, rate in fps.items():
                        print(f"        {batch}: {rate} FPS")
        
        # Memory Usage
        if 'memory_usage' in metrics:
            mem = metrics['memory_usage']
            print(f"  Memory Usage:")
            if 'peak_memory_mb' in mem:
                for device, usage in mem['peak_memory_mb'].items():
                    print(f"    Peak {device.upper()}: {usage} MB")
    
    def _print_usage_statistics(self, stats: Dict):
        """Print usage statistics section"""
        print(f"\nUSAGE STATISTICS ({stats.get('statistics_period', 'N/A')}):")
        print(f"  Total Requests: {stats.get('total_requests', 0):,}")
        print(f"  Success Rate: {stats.get('success_rate', 0):.2f}%")
        print(f"  Avg Response Time: {stats.get('avg_response_time_ms', 0)} ms")
        
        # Daily requests
        if 'requests_per_day' in stats:
            rpd = stats['requests_per_day']
            if 'last_7_days' in rpd:
                print(f"  Daily Requests (last 7 days): {rpd['last_7_days']}")
                print(f"  Daily Average: {rpd.get('avg', 0):,}")
        
        # Common errors
        if 'most_common_errors' in stats:
            print(f"  Most Common Errors:")
            for error in stats['most_common_errors'][:3]:  # Top 3
                print(f"    {error['error_type']}: {error['count']} ({error['percentage']:.1f}%)")
        
        # Usage patterns
        if 'usage_patterns' in stats:
            patterns = stats['usage_patterns']
            if 'peak_hours' in patterns:
                print(f"  Peak Hours: {', '.join(patterns['peak_hours'])}")
            if 'avg_batch_size' in patterns:
                print(f"  Average Batch Size: {patterns['avg_batch_size']:.1f}")
    
    def _print_examples(self, examples: Dict):
        """Print usage examples section"""
        print(f"\nUSAGE EXAMPLES:")
        
        for example_name, example_data in examples.items():
            print(f"\n  {example_name.replace('_', ' ').title()}:")
            print(f"    {example_data.get('description', 'No description')}")
            
            # Don't print full code here, just mention it's available
            if 'code' in example_data:
                lines = example_data['code'].count('\n') + 1
                print(f"    Code Example: {lines} lines available")
    
    def compare_models(self, model_names: list) -> Dict:
        """Compare multiple models side by side"""
        models_info = {}
        
        for model_name in model_names:
            info = self.get_model_info(
                model_name, 
                include_performance=True,
                include_usage_stats=True
            )
            if "error" not in info:
                models_info[model_name] = info
        
        return self._generate_comparison_report(models_info)
    
    def _generate_comparison_report(self, models_info: Dict) -> Dict:
        """Generate a comparison report for multiple models"""
        comparison = {
            "models_compared": list(models_info.keys()),
            "comparison_date": datetime.now().isoformat(),
            "metrics": {}
        }
        
        # Compare parameters
        params = {}
        sizes = {}
        accuracies = {}
        response_times = {}
        
        for name, info in models_info.items():
            specs = info.get('technical_specs', {})
            params[name] = specs.get('parameters_count', 0)
            sizes[name] = specs.get('model_size_mb', 0)
            
            perf = info.get('performance_metrics', {})
            acc = perf.get('accuracy', {})
            if 'top1' in acc:
                accuracies[name] = acc['top1']
            
            stats = info.get('usage_statistics', {})
            response_times[name] = stats.get('avg_response_time_ms', 0)
        
        comparison['metrics'] = {
            'parameters_count': params,
            'model_size_mb': sizes,
            'accuracy_top1': accuracies,
            'avg_response_time_ms': response_times
        }
        
        # Find best in each category
        if params:
            comparison['largest_model'] = max(params, key=params.get)
            comparison['smallest_model'] = min(params, key=params.get)
        
        if accuracies:
            comparison['most_accurate'] = max(accuracies, key=accuracies.get)
        
        if response_times:
            comparison['fastest_response'] = min(
                {k: v for k, v in response_times.items() if v > 0},
                key=response_times.get
            )
        
        return comparison
    
    def visualize_performance(self, model_name: str, version: str = "latest"):
        """Create visualizations for model performance"""
        info = self.get_model_info(
            model_name, 
            version, 
            include_performance=True,
            include_usage_stats=True
        )
        
        if "error" in info:
            print(f"Error getting model info: {info['error']}")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Performance Analysis: {model_name}', fontsize=16)
        
        # Plot 1: Inference Time by Device
        perf = info.get('performance_metrics', {})
        if 'speed_benchmarks' in perf:
            speed = perf['speed_benchmarks']
            if 'inference_time_ms' in speed:
                devices = []
                single_times = []
                batch4_times = []
                
                for device, times in speed['inference_time_ms'].items():
                    devices.append(device.upper())
                    single_times.append(times.get('single', 0))
                    batch4_times.append(times.get('batch_4', 0))
                
                x = range(len(devices))
                width = 0.35
                ax1.bar([i - width/2 for i in x], single_times, width, label='Single')
                ax1.bar([i + width/2 for i in x], batch4_times, width, label='Batch 4')
                ax1.set_xlabel('Device')
                ax1.set_ylabel('Inference Time (ms)')
                ax1.set_title('Inference Time by Device')
                ax1.set_xticks(x)
                ax1.set_xticklabels(devices)
                ax1.legend()
        
        # Plot 2: Daily Usage Pattern
        stats = info.get('usage_statistics', {})
        if 'requests_per_day' in stats and 'last_7_days' in stats['requests_per_day']:
            days = list(range(1, 8))
            requests = stats['requests_per_day']['last_7_days']
            ax2.plot(days, requests, marker='o', linewidth=2)
            ax2.set_xlabel('Days Ago')
            ax2.set_ylabel('Requests')
            ax2.set_title('Daily Request Pattern (Last 7 Days)')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Error Distribution
        if 'most_common_errors' in stats:
            errors = stats['most_common_errors']
            error_types = [e['error_type'] for e in errors]
            error_counts = [e['count'] for e in errors]
            
            ax3.pie(error_counts, labels=error_types, autopct='%1.1f%%')
            ax3.set_title('Error Distribution')
        
        # Plot 4: Memory Usage
        if 'memory_usage' in perf and 'peak_memory_mb' in perf['memory_usage']:
            devices = list(perf['memory_usage']['peak_memory_mb'].keys())
            memory = list(perf['memory_usage']['peak_memory_mb'].values())
            
            colors = ['skyblue', 'lightcoral']
            bars = ax4.bar(devices, memory, color=colors[:len(devices)])
            ax4.set_xlabel('Device')
            ax4.set_ylabel('Memory Usage (MB)')
            ax4.set_title('Peak Memory Usage')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 10,
                        f'{height:.0f}MB', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# Usage Examples
inspector = ModelInformationInspector()

# Get basic model information
resnet_info = inspector.get_model_info("resnet50")
inspector.print_model_summary(resnet_info)

# Get complete information with performance and usage stats
bert_info = inspector.get_model_info(
    "bert-base", 
    version="v2.1",
    include_performance=True,
    include_usage_stats=True,
    include_examples=True
)
inspector.print_model_summary(bert_info)

# Compare multiple models
comparison = inspector.compare_models(["resnet50", "bert-base", "gpt2-small"])
print(f"\nMODEL COMPARISON:")
print(f"Most Accurate: {comparison.get('most_accurate', 'N/A')}")
print(f"Fastest Response: {comparison.get('fastest_response', 'N/A')}")
print(f"Largest Model: {comparison.get('largest_model', 'N/A')}")

# Visualize performance (requires matplotlib)
try:
    inspector.visualize_performance("resnet50")
except ImportError:
    print("Matplotlib not available for visualization")
```

### JavaScript Model Information Panel

```javascript
class ModelInfoPanel {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }
  
  async getModelInfo(modelName, options = {}) {
    const params = new URLSearchParams({
      version: options.version || 'latest',
      include_performance: options.includePerformance || false,
      include_usage_stats: options.includeUsageStats || false,
      include_examples: options.includeExamples || false
    });
    
    try {
      const response = await fetch(
        `${this.baseUrl}/model-info/${modelName}?${params}`
      );
      return await response.json();
    } catch (error) {
      return { error: error.message };
    }
  }
  
  renderModelInfo(data, containerId) {
    const container = document.getElementById(containerId);
    
    if (data.error) {
      container.innerHTML = `
        <div class="error-panel">
          <h3>Error Loading Model Information</h3>
          <p>${data.error}</p>
        </div>
      `;
      return;
    }
    
    container.innerHTML = `
      <div class="model-info-panel">
        ${this.renderHeader(data)}
        ${this.renderBasicInfo(data.basic_info)}
        ${this.renderTechnicalSpecs(data.technical_specs)}
        ${this.renderDeploymentInfo(data.deployment_info)}
        ${data.performance_metrics ? this.renderPerformance(data.performance_metrics) : ''}
        ${data.usage_statistics ? this.renderUsageStats(data.usage_statistics) : ''}
        ${data.examples ? this.renderExamples(data.examples) : ''}
      </div>
    `;
  }
  
  renderHeader(data) {
    const statusColors = {
      'loaded': '#22c55e',
      'available': '#3b82f6',
      'error': '#ef4444'
    };
    
    const status = data.status || 'unknown';
    const color = statusColors[status] || '#6b7280';
    
    return `
      <div class="model-header">
        <div class="title-section">
          <h1>${data.basic_info?.display_name || data.model_name}</h1>
          <div class="model-meta">
            <span class="version-badge">${data.version}</span>
            <span class="status-badge" style="background-color: ${color}">
              ${status.toUpperCase()}
            </span>
          </div>
        </div>
        ${data.basic_info?.description ? `
          <p class="model-description">${data.basic_info.description}</p>
        ` : ''}
      </div>
    `;
  }
  
  renderBasicInfo(basicInfo) {
    if (!basicInfo) return '';
    
    return `
      <div class="info-section">
        <h3>Basic Information</h3>
        <div class="info-grid">
          ${this.renderInfoItem('Author', basicInfo.author)}
          ${this.renderInfoItem('License', basicInfo.license)}
          ${this.renderInfoItem('Framework', basicInfo.framework)}
          ${this.renderInfoItem('Type', basicInfo.model_type)}
          ${this.renderInfoItem('Created', basicInfo.created_at ? new Date(basicInfo.created_at).toLocaleDateString() : null)}
          ${this.renderInfoItem('Updated', basicInfo.last_updated ? new Date(basicInfo.last_updated).toLocaleDateString() : null)}
        </div>
        
        ${basicInfo.tags ? `
          <div class="tags-section">
            <label>Tags:</label>
            <div class="tag-list">
              ${basicInfo.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
            </div>
          </div>
        ` : ''}
        
        ${basicInfo.homepage || basicInfo.paper_url ? `
          <div class="links-section">
            ${basicInfo.homepage ? `<a href="${basicInfo.homepage}" target="_blank" class="external-link">Homepage</a>` : ''}
            ${basicInfo.paper_url ? `<a href="${basicInfo.paper_url}" target="_blank" class="external-link">Paper</a>` : ''}
          </div>
        ` : ''}
      </div>
    `;
  }
  
  renderTechnicalSpecs(specs) {
    if (!specs) return '';
    
    return `
      <div class="info-section">
        <h3>Technical Specifications</h3>
        <div class="specs-grid">
          ${this.renderSpecItem('Architecture', specs.architecture)}
          ${this.renderSpecItem('Parameters', specs.parameters_count ? specs.parameters_count.toLocaleString() : null)}
          ${this.renderSpecItem('Model Size', specs.model_size_mb ? `${specs.model_size_mb} MB` : null)}
          ${this.renderSpecItem('Memory Usage', specs.memory_usage_mb ? `${specs.memory_usage_mb} MB` : null)}
          ${this.renderSpecItem('Precision', specs.precision)}
          ${this.renderSpecItem('Layers', specs.layers_count)}
        </div>
        
        ${specs.supported_devices ? `
          <div class="devices-section">
            <label>Supported Devices:</label>
            <div class="device-list">
              ${specs.supported_devices.map(device => `<span class="device-badge">${device.toUpperCase()}</span>`).join('')}
            </div>
          </div>
        ` : ''}
        
        <div class="format-section">
          ${specs.input_format ? `
            <div class="format-item">
              <h4>Input Format</h4>
              <div class="format-details">
                <div>Type: <code>${specs.input_format.type}</code></div>
                <div>Shape: <code>${JSON.stringify(specs.input_format.shape)}</code></div>
                <div>Data Type: <code>${specs.input_format.dtype}</code></div>
              </div>
            </div>
          ` : ''}
          
          ${specs.output_format ? `
            <div class="format-item">
              <h4>Output Format</h4>
              <div class="format-details">
                <div>Type: <code>${specs.output_format.type}</code></div>
                <div>Shape: <code>${JSON.stringify(specs.output_format.shape)}</code></div>
                <div>Data Type: <code>${specs.output_format.dtype}</code></div>
                ${specs.output_format.description ? `<div>Description: ${specs.output_format.description}</div>` : ''}
              </div>
            </div>
          ` : ''}
        </div>
      </div>
    `;
  }
  
  renderPerformance(performance) {
    return `
      <div class="info-section">
        <h3>Performance Metrics</h3>
        
        ${performance.accuracy ? `
          <div class="accuracy-section">
            <h4>Accuracy (${performance.accuracy.dataset || 'Test Set'})</h4>
            <div class="metrics-grid">
              ${performance.accuracy.top1 ? `<div class="metric-item"><span>Top-1:</span><strong>${performance.accuracy.top1}%</strong></div>` : ''}
              ${performance.accuracy.top5 ? `<div class="metric-item"><span>Top-5:</span><strong>${performance.accuracy.top5}%</strong></div>` : ''}
            </div>
          </div>
        ` : ''}
        
        ${performance.speed_benchmarks ? this.renderSpeedBenchmarks(performance.speed_benchmarks) : ''}
        
        ${performance.memory_usage ? `
          <div class="memory-section">
            <h4>Memory Usage</h4>
            <div class="memory-grid">
              ${Object.entries(performance.memory_usage.peak_memory_mb || {}).map(([device, memory]) => `
                <div class="memory-item">
                  <span>${device.toUpperCase()}:</span>
                  <strong>${memory} MB</strong>
                </div>
              `).join('')}
            </div>
          </div>
        ` : ''}
        
        ${performance.benchmark_date ? `
          <div class="benchmark-info">
            <small>Benchmarks run on: ${new Date(performance.benchmark_date).toLocaleDateString()}</small>
          </div>
        ` : ''}
      </div>
    `;
  }
  
  renderSpeedBenchmarks(benchmarks) {
    return `
      <div class="speed-section">
        <h4>Speed Benchmarks</h4>
        ${benchmarks.inference_time_ms ? `
          <div class="benchmark-table">
            <h5>Inference Time (ms)</h5>
            <table class="performance-table">
              <thead>
                <tr>
                  <th>Device</th>
                  <th>Single</th>
                  <th>Batch 4</th>
                  <th>Batch 8</th>
                </tr>
              </thead>
              <tbody>
                ${Object.entries(benchmarks.inference_time_ms).map(([device, times]) => `
                  <tr>
                    <td>${device.toUpperCase()}</td>
                    <td>${times.single || '-'}</td>
                    <td>${times.batch_4 || '-'}</td>
                    <td>${times.batch_8 || '-'}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        ` : ''}
        
        ${benchmarks.throughput_fps ? `
          <div class="benchmark-table">
            <h5>Throughput (FPS)</h5>
            <table class="performance-table">
              <thead>
                <tr>
                  <th>Device</th>
                  <th>Single</th>
                  <th>Batch 4</th>
                  <th>Batch 8</th>
                </tr>
              </thead>
              <tbody>
                ${Object.entries(benchmarks.throughput_fps).map(([device, fps]) => `
                  <tr>
                    <td>${device.toUpperCase()}</td>
                    <td>${fps.single || '-'}</td>
                    <td>${fps.batch_4 || '-'}</td>
                    <td>${fps.batch_8 || '-'}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        ` : ''}
      </div>
    `;
  }
  
  renderUsageStats(stats) {
    return `
      <div class="info-section">
        <h3>Usage Statistics</h3>
        <div class="stats-overview">
          <div class="stat-item">
            <span class="stat-value">${stats.total_requests?.toLocaleString() || 0}</span>
            <span class="stat-label">Total Requests</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">${stats.success_rate || 0}%</span>
            <span class="stat-label">Success Rate</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">${stats.avg_response_time_ms || 0}ms</span>
            <span class="stat-label">Avg Response Time</span>
          </div>
        </div>
        
        ${stats.requests_per_day?.last_7_days ? `
          <div class="usage-chart">
            <h4>Daily Requests (Last 7 Days)</h4>
            <div class="chart-placeholder">
              <canvas id="usage-chart" width="400" height="200"></canvas>
            </div>
          </div>
        ` : ''}
        
        ${stats.most_common_errors ? `
          <div class="errors-section">
            <h4>Most Common Errors</h4>
            <div class="error-list">
              ${stats.most_common_errors.slice(0, 3).map(error => `
                <div class="error-item">
                  <span class="error-type">${error.error_type}</span>
                  <span class="error-count">${error.count} (${error.percentage}%)</span>
                </div>
              `).join('')}
            </div>
          </div>
        ` : ''}
      </div>
    `;
  }
  
  renderExamples(examples) {
    return `
      <div class="info-section">
        <h3>Usage Examples</h3>
        <div class="examples-container">
          ${Object.entries(examples).map(([name, example]) => `
            <div class="example-item">
              <div class="example-header">
                <h4>${name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</h4>
                <p>${example.description}</p>
              </div>
              <div class="code-container">
                <button class="copy-btn" onclick="copyToClipboard('${name}-code')">Copy</button>
                <pre><code id="${name}-code" class="language-python">${example.code}</code></pre>
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }
  
  renderInfoItem(label, value) {
    if (!value) return '';
    return `
      <div class="info-item">
        <label>${label}:</label>
        <span>${value}</span>
      </div>
    `;
  }
  
  renderSpecItem(label, value) {
    if (!value) return '';
    return `
      <div class="spec-item">
        <span class="spec-label">${label}:</span>
        <span class="spec-value">${value}</span>
      </div>
    `;
  }
}

// Usage
const infoPanel = new ModelInfoPanel();

// Load and display model information
async function loadModelInfo(modelName) {
  const data = await infoPanel.getModelInfo(modelName, {
    includePerformance: true,
    includeUsageStats: true,
    includeExamples: true
  });
  
  infoPanel.renderModelInfo(data, 'model-info-container');
}

// Copy to clipboard function
function copyToClipboard(elementId) {
  const element = document.getElementById(elementId);
  const text = element.textContent;
  
  navigator.clipboard.writeText(text).then(() => {
    // Show feedback
    const btn = element.parentElement.querySelector('.copy-btn');
    const originalText = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => {
      btn.textContent = originalText;
    }, 2000);
  });
}

// Initialize
loadModelInfo('resnet50');
```

## Error Handling

### Model Not Found (404)
```json
{
  "detail": "Model 'unknown-model' not found"
}
```

### Invalid Version (400)
```json
{
  "detail": "Version 'invalid-version' not found for model 'resnet50'"
}
```

## Related Endpoints

- [Models List](./models.md) - List all available models
- [Model Available](./model-available.md) - Check model availability
- [Model Download](./model-download.md) - Download models
- [Model Remove](./model-remove.md) - Remove models
- [Predict](./predict.md) - Use models for inference

## Information Categories

The endpoint provides information in several categories:

1. **Basic Information**: General model metadata and description
2. **Technical Specifications**: Architecture, parameters, input/output formats
3. **Performance Metrics**: Benchmarks, accuracy scores, speed tests
4. **Usage Statistics**: Request patterns, error rates, response times
5. **Code Examples**: Implementation samples and best practices
6. **Deployment Information**: Current status, file paths, optimization settings

## Best Practices

1. **Version Specificity**: Always specify versions in production queries
2. **Selective Loading**: Only request additional information when needed
3. **Caching**: Cache model information to reduce API calls
4. **Performance Monitoring**: Use statistics to monitor model performance
5. **Documentation**: Use examples and specifications for implementation guidance

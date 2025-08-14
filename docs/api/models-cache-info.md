# GET /models/cache/info - Model Cache Information

**URL**: `/models/cache/info`  
**Method**: `GET`  
**Authentication**: Not required  
**Content-Type**: `application/json`

## Description

Retrieves detailed information about the model cache, including cached models, storage usage, and cache directory details. This endpoint provides specific information about model storage and management within the PyTorch Inference Framework.

## Request

### URL Parameters
None

### Query Parameters
None

### Request Body
None (GET request)

## Response

### Success Response (200 OK)

#### Model Cache Information
```json
{
  "cache_directory": "/cache/models",
  "total_models": 3,
  "total_size_mb": 1248.7,
  "models": [
    "resnet50",
    "bert-base",
    "efficientnet-b0"
  ]
}
```

#### Empty Cache Response
```json
{
  "cache_directory": "/cache/models",
  "total_models": 0,
  "total_size_mb": 0.0,
  "models": []
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `cache_directory` | string | Absolute path to the model cache directory |
| `total_models` | integer | Number of models currently cached |
| `total_size_mb` | number | Total storage used by cached models in megabytes |
| `models` | array | List of model names currently in cache |

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success - Cache information returned |
| 500 | Internal server error |

## Examples

### Basic Cache Information

**Request:**
```bash
curl -X GET http://localhost:8000/models/cache/info
```

**Response:**
```json
{
  "cache_directory": "/cache/models",
  "total_models": 3,
  "total_size_mb": 1248.7,
  "models": [
    "resnet50",
    "bert-base", 
    "efficientnet-b0"
  ]
}
```

### Python Cache Monitor

```python
import requests
import json
from typing import Dict, List
import os
from datetime import datetime

class ModelCacheMonitor:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def get_cache_info(self) -> Dict:
        """Get model cache information"""
        try:
            response = requests.get(f"{self.base_url}/models/cache/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def print_cache_summary(self, cache_data: Dict):
        """Print formatted cache summary"""
        if "error" in cache_data:
            print(f"❌ Error: {cache_data['error']}")
            return
        
        cache_dir = cache_data.get("cache_directory", "Unknown")
        total_models = cache_data.get("total_models", 0)
        total_size_mb = cache_data.get("total_size_mb", 0)
        models = cache_data.get("models", [])
        
        print("="*60)
        print("MODEL CACHE INFORMATION")
        print("="*60)
        print(f"Cache Directory: {cache_dir}")
        print(f"Total Models: {total_models}")
        print(f"Total Size: {total_size_mb:.1f} MB")
        
        if total_size_mb > 1024:
            print(f"             {total_size_mb/1024:.1f} GB")
        
        print(f"\nCached Models:")
        if models:
            for i, model in enumerate(models, 1):
                print(f"  {i:2d}. {model}")
        else:
            print("  No models currently cached")
        
        print("="*60)
    
    def analyze_cache_usage(self) -> Dict:
        """Analyze cache usage and provide insights"""
        cache_data = self.get_cache_info()
        
        if "error" in cache_data:
            return cache_data
        
        analysis = {
            "cache_status": "empty" if cache_data["total_models"] == 0 else "populated",
            "storage_usage": cache_data["total_size_mb"],
            "model_count": cache_data["total_models"],
            "average_model_size_mb": 0,
            "recommendations": []
        }
        
        # Calculate average model size
        if analysis["model_count"] > 0:
            analysis["average_model_size_mb"] = analysis["storage_usage"] / analysis["model_count"]
        
        # Generate recommendations
        if analysis["model_count"] == 0:
            analysis["recommendations"].append("ℹ️ No models cached - consider downloading commonly used models")
        elif analysis["model_count"] > 10:
            analysis["recommendations"].append("📦 Many models cached - consider cleanup if storage is limited")
        
        if analysis["storage_usage"] > 5000:  # > 5GB
            analysis["recommendations"].append("💾 High storage usage - monitor disk space")
        elif analysis["storage_usage"] > 2000:  # > 2GB
            analysis["recommendations"].append("📊 Moderate storage usage - regular monitoring recommended")
        
        if analysis["average_model_size_mb"] > 500:
            analysis["recommendations"].append("🔍 Large average model size - consider compression or optimization")
        
        return analysis
    
    def get_detailed_model_info(self) -> List[Dict]:
        """Get detailed information about cached models"""
        cache_data = self.get_cache_info()
        
        if "error" in cache_data:
            return [{"error": cache_data["error"]}]
        
        cache_dir = cache_data.get("cache_directory", "")
        models = cache_data.get("models", [])
        
        detailed_info = []
        
        for model in models:
            model_info = {
                "name": model,
                "cached": True,
                "cache_path": os.path.join(cache_dir, model) if cache_dir else "Unknown"
            }
            
            # Try to get additional file information (this would require file system access)
            # In a real implementation, you might call additional endpoints or check file stats
            try:
                # This is a placeholder for additional model information
                # In practice, you might integrate with the model info endpoint
                additional_info_response = requests.get(
                    f"{self.base_url}/models/download/{model}/info",
                    timeout=5
                )
                if additional_info_response.status_code == 200:
                    additional_data = additional_info_response.json()
                    model_info.update({
                        "version": additional_data.get("version", "Unknown"),
                        "size_mb": additional_data.get("size_mb"),
                        "status": additional_data.get("status")
                    })
            except:
                # Ignore errors when trying to get additional info
                pass
            
            detailed_info.append(model_info)
        
        return detailed_info
    
    def monitor_cache_changes(self, interval_seconds=30, duration_minutes=60):
        """Monitor cache changes over time"""
        print(f"📊 Monitoring model cache for {duration_minutes} minutes...")
        print("Time      | Models | Size (MB) | Changes")
        print("-" * 45)
        
        start_time = datetime.now()
        end_time = start_time.timestamp() + (duration_minutes * 60)
        
        previous_models = set()
        previous_size = 0
        
        while datetime.now().timestamp() < end_time:
            cache_data = self.get_cache_info()
            
            if "error" not in cache_data:
                current_models = set(cache_data.get("models", []))
                current_size = cache_data.get("total_size_mb", 0)
                
                # Detect changes
                added_models = current_models - previous_models
                removed_models = previous_models - current_models
                size_change = current_size - previous_size
                
                changes = []
                if added_models:
                    changes.append(f"+{len(added_models)} models")
                if removed_models:
                    changes.append(f"-{len(removed_models)} models")
                if abs(size_change) > 0.1:
                    sign = "+" if size_change > 0 else ""
                    changes.append(f"{sign}{size_change:.1f} MB")
                
                change_text = ", ".join(changes) if changes else "No changes"
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                model_count = len(current_models)
                
                print(f"{timestamp} | {model_count:6d} | {current_size:9.1f} | {change_text}")
                
                previous_models = current_models
                previous_size = current_size
            
            time.sleep(interval_seconds)
        
        print(f"\n📊 Monitoring completed")
    
    def export_cache_report(self, filename=None) -> str:
        """Export detailed cache report"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_cache_report_{timestamp}.json"
        
        cache_data = self.get_cache_info()
        detailed_models = self.get_detailed_model_info()
        analysis = self.analyze_cache_usage()
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "cache_info": cache_data,
            "detailed_models": detailed_models,
            "analysis": analysis
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✅ Model cache report exported to {filename}")
            return filename
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return ""

# Usage Examples
monitor = ModelCacheMonitor()

# Get and display basic cache information
cache_info = monitor.get_cache_info()
monitor.print_cache_summary(cache_info)

# Analyze cache usage
analysis = monitor.analyze_cache_usage()
print(f"\nCache Analysis:")
print(f"Status: {analysis.get('cache_status', 'unknown')}")
print(f"Model Count: {analysis.get('model_count', 0)}")
print(f"Storage Usage: {analysis.get('storage_usage', 0):.1f} MB")
print(f"Average Model Size: {analysis.get('average_model_size_mb', 0):.1f} MB")

if analysis.get('recommendations'):
    print(f"\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  {rec}")

# Get detailed model information
detailed_info = monitor.get_detailed_model_info()
print(f"\nDetailed Model Information:")
for model in detailed_info:
    if "error" not in model:
        print(f"  - {model['name']}: {model.get('cache_path', 'Unknown path')}")

# Export cache report
report_file = monitor.export_cache_report()
print(f"Report saved to: {report_file}")

# Monitor cache changes (uncomment to run)
# monitor.monitor_cache_changes(interval_seconds=30, duration_minutes=5)
```

### JavaScript Cache Dashboard

```javascript
class ModelCacheDashboard {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.updateInterval = null;
  }
  
  async getCacheInfo() {
    try {
      const response = await fetch(`${this.baseUrl}/models/cache/info`);
      return await response.json();
    } catch (error) {
      return { error: error.message };
    }
  }
  
  renderCacheDashboard(containerId) {
    const container = document.getElementById(containerId);
    
    container.innerHTML = `
      <div class="model-cache-dashboard">
        <div class="dashboard-header">
          <h2>Model Cache Dashboard</h2>
          <div class="controls">
            <button id="refresh-btn" class="btn-primary" onclick="cacheBoard.refreshData()">
              🔄 Refresh
            </button>
            <button id="auto-refresh-btn" class="btn-secondary" onclick="cacheBoard.toggleAutoRefresh()">
              ⏱️ Auto Refresh
            </button>
            <button id="export-btn" class="btn-secondary" onclick="cacheBoard.exportData()">
              📊 Export
            </button>
          </div>
        </div>
        
        <div id="cache-overview" class="cache-overview"></div>
        <div id="model-list" class="model-list"></div>
        <div id="cache-analysis" class="cache-analysis"></div>
      </div>
    `;
    
    this.loadCacheData();
  }
  
  async loadCacheData() {
    const data = await this.getCacheInfo();
    
    if (data.error) {
      this.renderError(data.error);
    } else {
      this.renderCacheOverview(data);
      this.renderModelList(data);
      this.renderCacheAnalysis(data);
    }
  }
  
  renderCacheOverview(data) {
    const container = document.getElementById('cache-overview');
    const totalModels = data.total_models || 0;
    const totalSizeMB = data.total_size_mb || 0;
    const cacheDir = data.cache_directory || 'Unknown';
    
    const sizeGB = totalSizeMB / 1024;
    const avgSizeMB = totalModels > 0 ? totalSizeMB / totalModels : 0;
    
    container.innerHTML = `
      <div class="overview-card">
        <h3>Cache Overview</h3>
        <div class="overview-stats">
          <div class="stat-item">
            <span class="stat-value">${totalModels}</span>
            <span class="stat-label">Total Models</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">${totalSizeMB.toFixed(1)}</span>
            <span class="stat-label">Size (MB)</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">${sizeGB.toFixed(2)}</span>
            <span class="stat-label">Size (GB)</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">${avgSizeMB.toFixed(1)}</span>
            <span class="stat-label">Avg Size (MB)</span>
          </div>
        </div>
        
        <div class="cache-location">
          <strong>Cache Directory:</strong> <code>${cacheDir}</code>
        </div>
      </div>
    `;
  }
  
  renderModelList(data) {
    const container = document.getElementById('model-list');
    const models = data.models || [];
    
    if (models.length === 0) {
      container.innerHTML = `
        <div class="model-list-card">
          <h3>Cached Models</h3>
          <div class="empty-state">
            <p>No models currently cached</p>
            <p>Download models to see them appear here</p>
          </div>
        </div>
      `;
      return;
    }
    
    const modelItems = models.map((model, index) => `
      <div class="model-item">
        <div class="model-index">${index + 1}</div>
        <div class="model-name">${model}</div>
        <div class="model-actions">
          <button class="btn-small" onclick="cacheBoard.viewModelDetails('${model}')">
            ℹ️ Info
          </button>
          <button class="btn-small btn-danger" onclick="cacheBoard.removeModel('${model}')">
            🗑️ Remove
          </button>
        </div>
      </div>
    `).join('');
    
    container.innerHTML = `
      <div class="model-list-card">
        <h3>Cached Models (${models.length})</h3>
        <div class="model-items">
          ${modelItems}
        </div>
      </div>
    `;
  }
  
  renderCacheAnalysis(data) {
    const container = document.getElementById('cache-analysis');
    const totalModels = data.total_models || 0;
    const totalSizeMB = data.total_size_mb || 0;
    
    const analysis = this.analyzeCacheData(data);
    
    let statusColor = '#22c55e'; // green
    let statusIcon = '✅';
    
    if (totalModels === 0) {
      statusColor = '#6b7280'; // gray
      statusIcon = 'ℹ️';
    } else if (totalSizeMB > 5000) {
      statusColor = '#f59e0b'; // orange
      statusIcon = '⚠️';
    } else if (totalSizeMB > 10000) {
      statusColor = '#ef4444'; // red
      statusIcon = '🚨';
    }
    
    container.innerHTML = `
      <div class="analysis-card">
        <h3>Cache Analysis</h3>
        <div class="analysis-status" style="color: ${statusColor}">
          <span class="status-icon">${statusIcon}</span>
          <span class="status-text">${analysis.status}</span>
        </div>
        
        <div class="analysis-metrics">
          <div class="metric-item">
            <span class="metric-label">Cache Health:</span>
            <span class="metric-value">${analysis.health}</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">Storage Level:</span>
            <span class="metric-value">${analysis.storageLevel}</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">Utilization:</span>
            <span class="metric-value">${analysis.utilization}</span>
          </div>
        </div>
        
        ${analysis.recommendations.length > 0 ? `
          <div class="recommendations">
            <h4>Recommendations</h4>
            <ul>
              ${analysis.recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
          </div>
        ` : ''}
      </div>
    `;
  }
  
  analyzeCacheData(data) {
    const totalModels = data.total_models || 0;
    const totalSizeMB = data.total_size_mb || 0;
    const avgSizeMB = totalModels > 0 ? totalSizeMB / totalModels : 0;
    
    const analysis = {
      status: 'Unknown',
      health: 'Unknown',
      storageLevel: 'Unknown',
      utilization: 'Unknown',
      recommendations: []
    };
    
    // Determine status
    if (totalModels === 0) {
      analysis.status = 'Empty Cache';
      analysis.health = 'Idle';
      analysis.utilization = 'None';
    } else if (totalModels <= 5) {
      analysis.status = 'Light Usage';
      analysis.health = 'Good';
      analysis.utilization = 'Low';
    } else if (totalModels <= 15) {
      analysis.status = 'Moderate Usage';
      analysis.health = 'Good';
      analysis.utilization = 'Moderate';
    } else {
      analysis.status = 'Heavy Usage';
      analysis.health = 'Monitor';
      analysis.utilization = 'High';
    }
    
    // Determine storage level
    if (totalSizeMB < 500) {
      analysis.storageLevel = 'Low';
    } else if (totalSizeMB < 2000) {
      analysis.storageLevel = 'Moderate';
    } else if (totalSizeMB < 5000) {
      analysis.storageLevel = 'High';
    } else {
      analysis.storageLevel = 'Very High';
      analysis.health = 'Warning';
    }
    
    // Generate recommendations
    if (totalModels === 0) {
      analysis.recommendations.push('Consider downloading commonly used models for better performance');
    }
    
    if (totalModels > 20) {
      analysis.recommendations.push('Large number of cached models - consider cleanup if needed');
    }
    
    if (totalSizeMB > 5000) {
      analysis.recommendations.push('High storage usage - monitor available disk space');
    }
    
    if (avgSizeMB > 1000) {
      analysis.recommendations.push('Large average model size - consider model optimization');
    }
    
    if (analysis.recommendations.length === 0) {
      analysis.recommendations.push('Cache is operating normally');
    }
    
    return analysis;
  }
  
  renderError(error) {
    const container = document.getElementById('cache-overview');
    container.innerHTML = `
      <div class="error-card">
        <h3>Error Loading Cache Information</h3>
        <p>${error}</p>
        <button onclick="cacheBoard.loadCacheData()" class="btn-primary">Retry</button>
      </div>
    `;
  }
  
  async refreshData() {
    const button = document.getElementById('refresh-btn');
    button.disabled = true;
    button.textContent = '🔄 Refreshing...';
    
    await this.loadCacheData();
    
    button.disabled = false;
    button.textContent = '🔄 Refresh';
  }
  
  toggleAutoRefresh() {
    const button = document.getElementById('auto-refresh-btn');
    
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
      button.textContent = '⏱️ Auto Refresh';
      button.classList.remove('active');
    } else {
      this.updateInterval = setInterval(() => {
        this.loadCacheData();
      }, 30000); // Refresh every 30 seconds
      button.textContent = '⏱️ Stop Auto';
      button.classList.add('active');
    }
  }
  
  async exportData() {
    const data = await this.getCacheInfo();
    
    if (data.error) {
      alert(`Export failed: ${data.error}`);
      return;
    }
    
    const exportData = {
      exported_at: new Date().toISOString(),
      cache_info: data,
      analysis: this.analyzeCacheData(data)
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `model_cache_export_${new Date().toISOString().slice(0,10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
  
  async viewModelDetails(modelName) {
    try {
      const response = await fetch(`${this.baseUrl}/models/download/${modelName}/info`);
      const details = await response.json();
      
      alert(`Model: ${modelName}\n\nDetails:\n${JSON.stringify(details, null, 2)}`);
    } catch (error) {
      alert(`Could not load details for ${modelName}: ${error.message}`);
    }
  }
  
  async removeModel(modelName) {
    const confirm = window.confirm(
      `Remove model "${modelName}" from cache?\n\n` +
      'This action cannot be undone and the model will need to be downloaded again.'
    );
    
    if (!confirm) return;
    
    try {
      const response = await fetch(`${this.baseUrl}/models/download/${modelName}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        alert(`Model "${modelName}" removed successfully`);
        this.loadCacheData(); // Refresh the data
      } else {
        const error = await response.json();
        alert(`Failed to remove model: ${error.detail || 'Unknown error'}`);
      }
    } catch (error) {
      alert(`Error removing model: ${error.message}`);
    }
  }
}

// Global instance
const cacheBoard = new ModelCacheDashboard();

// Initialize dashboard
// cacheBoard.renderCacheDashboard('model-cache-container');
```

## Use Cases

### Development Monitoring
- Check which models are cached during development
- Monitor cache growth and storage usage
- Identify unused models for cleanup

### Production Monitoring
- Track cache efficiency and model availability
- Monitor storage usage for capacity planning
- Validate model deployment status

### Maintenance Operations
- Identify models for cleanup or optimization
- Verify model cache state after operations
- Generate reports for storage management

## Related Endpoints

- [Models List](./models.md) - List all available models
- [Model Info](./model-info.md) - Detailed model information
- [Model Download](./model-download.md) - Download and cache models
- [Model Remove](./model-remove.md) - Remove models from cache

## Best Practices

1. **Regular Monitoring**: Check cache usage regularly to prevent storage issues
2. **Storage Planning**: Monitor cache growth for capacity planning
3. **Cleanup Strategy**: Remove unused models to optimize storage
4. **Performance Tracking**: Correlate cache status with inference performance
5. **Backup Considerations**: Plan backup strategies for critical cached models

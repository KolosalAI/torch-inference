# DELETE /model-remove/{model_name} - Remove Model

**URL**: `/model-remove/{model_name}`  
**Method**: `DELETE`  
**Authentication**: Not required  
**Content-Type**: `application/json`

## Description

Removes a model from the system, including unloading it from memory and optionally deleting model files from storage. This endpoint provides various removal options to suit different cleanup scenarios.

## Request

### URL Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | string | Yes | Name of the model to remove |

### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `version` | string | No | "all" | Specific version to remove ("all", "latest", or version number) |
| `remove_files` | boolean | No | false | Whether to delete model files from disk |
| `force` | boolean | No | false | Force removal even if model is currently in use |

### Request Body

#### Basic Removal
```json
{
  "confirm": true
}
```

#### Advanced Removal Options
```json
{
  "confirm": true,
  "remove_files": true,
  "remove_cache": true,
  "remove_logs": true,
  "backup_before_removal": false,
  "force": false
}
```

#### Request Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `confirm` | boolean | Yes | - | Confirmation that you want to remove the model |
| `remove_files` | boolean | No | false | Delete model files from storage |
| `remove_cache` | boolean | No | true | Remove cached inference results |
| `remove_logs` | boolean | No | false | Remove model-specific log files |
| `backup_before_removal` | boolean | No | false | Create backup before removal |
| `force` | boolean | No | false | Force removal ignoring active usage |

## Response

### Success Response (200 OK)

#### Model Removed Successfully
```json
{
  "status": "removed",
  "message": "Model 'resnet50' v1.0 removed successfully",
  "model_name": "resnet50",
  "version": "v1.0",
  "removal_details": {
    "unloaded_from_memory": true,
    "files_deleted": false,
    "cache_cleared": true,
    "logs_removed": false,
    "backup_created": false,
    "removed_at": "2025-08-14T10:45:00.000Z",
    "disk_space_freed_mb": 0
  },
  "remaining_versions": ["v1.1", "v2.0"],
  "cleanup_summary": {
    "memory_freed_mb": 125.4,
    "cache_entries_removed": 47,
    "temporary_files_cleaned": 12
  }
}
```

#### Complete Removal with Files
```json
{
  "status": "completely_removed",
  "message": "Model 'bert-base' completely removed from system",
  "model_name": "bert-base",
  "version": "all",
  "removal_details": {
    "unloaded_from_memory": true,
    "files_deleted": true,
    "cache_cleared": true,
    "logs_removed": true,
    "backup_created": false,
    "removed_at": "2025-08-14T10:45:00.000Z",
    "disk_space_freed_mb": 438.2,
    "removed_files": [
      "/models/bert-base-v2.1.pth",
      "/models/bert-base-v2.0.pth",
      "/cache/bert-base/",
      "/logs/bert-base.log"
    ]
  },
  "remaining_versions": [],
  "cleanup_summary": {
    "memory_freed_mb": 1200.0,
    "cache_entries_removed": 234,
    "temporary_files_cleaned": 45,
    "log_files_removed": 3
  }
}
```

#### Partial Removal (Version Specific)
```json
{
  "status": "version_removed",
  "message": "Version 'v1.0' of model 'efficientnet' removed",
  "model_name": "efficientnet",
  "version": "v1.0",
  "removal_details": {
    "unloaded_from_memory": false,
    "files_deleted": true,
    "cache_cleared": true,
    "logs_removed": false,
    "backup_created": true,
    "removed_at": "2025-08-14T10:45:00.000Z",
    "disk_space_freed_mb": 52.3,
    "backup_location": "/backups/efficientnet-v1.0-20250814.tar.gz"
  },
  "remaining_versions": ["v1.1", "v1.2", "latest"],
  "cleanup_summary": {
    "memory_freed_mb": 0,
    "cache_entries_removed": 18,
    "temporary_files_cleaned": 5
  }
}
```

#### Response Fields

##### Removal Details Object
| Field | Type | Description |
|-------|------|-------------|
| `unloaded_from_memory` | boolean | Whether model was unloaded from memory |
| `files_deleted` | boolean | Whether model files were deleted from disk |
| `cache_cleared` | boolean | Whether model cache was cleared |
| `logs_removed` | boolean | Whether model logs were removed |
| `backup_created` | boolean | Whether backup was created before removal |
| `removed_at` | string | When removal was completed (ISO format) |
| `disk_space_freed_mb` | number | Disk space freed in MB |
| `removed_files` | array | List of files that were deleted |
| `backup_location` | string | Path to backup file (if created) |

##### Cleanup Summary Object
| Field | Type | Description |
|-------|------|-------------|
| `memory_freed_mb` | number | Memory freed in MB |
| `cache_entries_removed` | integer | Number of cache entries cleared |
| `temporary_files_cleaned` | integer | Number of temporary files removed |
| `log_files_removed` | integer | Number of log files removed |

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success - Model removed |
| 400 | Bad request - Invalid parameters or missing confirmation |
| 404 | Model not found |
| 409 | Conflict - Model is currently in use (use force=true to override) |
| 423 | Locked - Model removal is locked by system policy |
| 500 | Internal server error |

## Examples

### Basic Model Removal (Memory Only)

**Request:**
```bash
curl -X DELETE "http://localhost:8000/model-remove/resnet50" \
  -H "Content-Type: application/json" \
  -d '{"confirm": true}'
```

**Response:**
```json
{
  "status": "removed",
  "message": "Model 'resnet50' v1.0 removed successfully",
  "model_name": "resnet50",
  "version": "latest",
  "removal_details": {
    "unloaded_from_memory": true,
    "files_deleted": false,
    "cache_cleared": true
  },
  "cleanup_summary": {
    "memory_freed_mb": 125.4,
    "cache_entries_removed": 47
  }
}
```

### Complete Removal with Files

**Request:**
```bash
curl -X DELETE "http://localhost:8000/model-remove/bert-base?remove_files=true" \
  -H "Content-Type: application/json" \
  -d '{
    "confirm": true,
    "remove_files": true,
    "remove_cache": true,
    "remove_logs": true
  }'
```

### Remove Specific Version

**Request:**
```bash
curl -X DELETE "http://localhost:8000/model-remove/gpt2-small?version=v1.0&remove_files=true" \
  -H "Content-Type: application/json" \
  -d '{
    "confirm": true,
    "backup_before_removal": true
  }'
```

### Force Removal of Active Model

**Request:**
```bash
curl -X DELETE "http://localhost:8000/model-remove/active-model?force=true" \
  -H "Content-Type: application/json" \
  -d '{
    "confirm": true,
    "force": true
  }'
```

### Python Model Remover

```python
import requests
import json
from typing import Dict, Optional, List
from datetime import datetime

class ModelRemover:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def remove_model(
        self,
        model_name: str,
        version: str = "latest",
        remove_files: bool = False,
        remove_cache: bool = True,
        remove_logs: bool = False,
        backup_before_removal: bool = False,
        force: bool = False,
        confirm: bool = False
    ) -> Dict:
        """Remove a model from the system"""
        
        if not confirm:
            # Interactive confirmation
            print(f"⚠️  You are about to remove model '{model_name}' version '{version}'")
            if remove_files:
                print("   🗑️  This will DELETE model files from disk")
            if remove_logs:
                print("   📋 This will DELETE log files")
            if force:
                print("   ⚡ This will FORCE removal even if model is in use")
            
            user_confirm = input("\n❓ Are you sure you want to proceed? (yes/no): ").lower()
            if user_confirm not in ['yes', 'y']:
                return {"status": "cancelled", "message": "Removal cancelled by user"}
            
            confirm = True
        
        params = {}
        if version != "latest":
            params['version'] = version
        if remove_files:
            params['remove_files'] = 'true'
        if force:
            params['force'] = 'true'
        
        payload = {
            "confirm": confirm,
            "remove_files": remove_files,
            "remove_cache": remove_cache,
            "remove_logs": remove_logs,
            "backup_before_removal": backup_before_removal,
            "force": force
        }
        
        try:
            response = requests.delete(
                f"{self.base_url}/model-remove/{model_name}",
                params=params,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            return {"status": "error", "error": str(e)}
    
    def remove_multiple_models(
        self,
        models: List[Dict],
        confirm_all: bool = False
    ) -> Dict:
        """Remove multiple models"""
        results = {}
        
        if not confirm_all:
            print(f"⚠️  About to remove {len(models)} models:")
            for model_config in models:
                model_name = model_config.get('model_name')
                version = model_config.get('version', 'latest')
                remove_files = model_config.get('remove_files', False)
                print(f"   - {model_name} v{version}" + 
                      (" [FILES WILL BE DELETED]" if remove_files else ""))
            
            user_confirm = input(f"\n❓ Remove all {len(models)} models? (yes/no): ").lower()
            if user_confirm not in ['yes', 'y']:
                return {"status": "cancelled", "message": "Batch removal cancelled by user"}
        
        print("\n🗑️  Starting batch removal...")
        
        for i, model_config in enumerate(models, 1):
            model_name = model_config.get('model_name')
            print(f"[{i}/{len(models)}] Removing {model_name}...")
            
            result = self.remove_model(
                confirm=True,  # Skip individual confirmations
                **model_config
            )
            
            results[model_name] = result
            
            if result.get('status') in ['removed', 'completely_removed', 'version_removed']:
                print(f"   ✅ {model_name}: {result.get('message', 'Removed')}")
            else:
                print(f"   ❌ {model_name}: {result.get('message', 'Failed')}")
        
        return {
            "batch_removal": True,
            "total_models": len(models),
            "results": results,
            "summary": self._generate_batch_summary(results)
        }
    
    def _generate_batch_summary(self, results: Dict) -> Dict:
        """Generate summary of batch removal"""
        summary = {
            "successful": 0,
            "failed": 0,
            "total_memory_freed_mb": 0,
            "total_disk_freed_mb": 0,
            "total_cache_entries_removed": 0
        }
        
        for model_name, result in results.items():
            if result.get('status') in ['removed', 'completely_removed', 'version_removed']:
                summary['successful'] += 1
                
                cleanup = result.get('cleanup_summary', {})
                summary['total_memory_freed_mb'] += cleanup.get('memory_freed_mb', 0)
                summary['total_cache_entries_removed'] += cleanup.get('cache_entries_removed', 0)
                
                removal_details = result.get('removal_details', {})
                summary['total_disk_freed_mb'] += removal_details.get('disk_space_freed_mb', 0)
            else:
                summary['failed'] += 1
        
        return summary
    
    def safe_remove_unused_models(self, dry_run: bool = True) -> Dict:
        """Remove models that haven't been used recently"""
        # First, get all models
        try:
            models_response = requests.get(f"{self.base_url}/models?include_metadata=true")
            models_response.raise_for_status()
            models_data = models_response.json()
        except Exception as e:
            return {"error": f"Failed to get models list: {e}"}
        
        unused_models = []
        cutoff_days = 30  # Models unused for 30+ days
        
        for model in models_data.get('models', []):
            if model['status'] != 'loaded':
                continue
            
            # Get model usage statistics
            try:
                info_response = requests.get(
                    f"{self.base_url}/model-info/{model['name']}?include_usage_stats=true"
                )
                info_data = info_response.json()
                
                usage_stats = info_data.get('usage_statistics', {})
                last_request_days = usage_stats.get('days_since_last_request', 999)
                
                if last_request_days >= cutoff_days:
                    unused_models.append({
                        'model_name': model['name'],
                        'version': model['version'],
                        'days_unused': last_request_days,
                        'memory_usage_mb': info_data.get('technical_specs', {}).get('memory_usage_mb', 0)
                    })
            except:
                continue
        
        if dry_run:
            print(f"🔍 Found {len(unused_models)} unused models (>{cutoff_days} days):")
            total_memory = 0
            for model in unused_models:
                print(f"   - {model['model_name']} v{model['version']}: "
                      f"{model['days_unused']} days, "
                      f"{model['memory_usage_mb']} MB")
                total_memory += model['memory_usage_mb']
            
            print(f"\n📊 Total memory that could be freed: {total_memory} MB")
            print("💡 Run with dry_run=False to actually remove these models")
            
            return {
                "dry_run": True,
                "unused_models": unused_models,
                "total_memory_mb": total_memory
            }
        else:
            # Actually remove unused models
            removal_configs = [
                {
                    'model_name': model['model_name'],
                    'version': model['version'],
                    'remove_files': False,  # Keep files, just unload from memory
                    'remove_cache': True
                }
                for model in unused_models
            ]
            
            return self.remove_multiple_models(removal_configs, confirm_all=True)
    
    def cleanup_old_versions(self, keep_latest: int = 2) -> Dict:
        """Remove old versions of models, keeping only the latest N versions"""
        try:
            models_response = requests.get(f"{self.base_url}/models")
            models_response.raise_for_status()
            models_data = models_response.json()
        except Exception as e:
            return {"error": f"Failed to get models list: {e}"}
        
        # Group models by name
        model_groups = {}
        for model in models_data.get('models', []):
            name = model['name']
            if name not in model_groups:
                model_groups[name] = []
            model_groups[name].append(model)
        
        to_remove = []
        
        for model_name, versions in model_groups.items():
            if len(versions) <= keep_latest:
                continue  # Skip if we don't have more versions than we want to keep
            
            # Sort by version (simple string sort, may need improvement)
            versions_sorted = sorted(versions, key=lambda x: x['version'], reverse=True)
            old_versions = versions_sorted[keep_latest:]  # Keep only latest N
            
            for old_version in old_versions:
                to_remove.append({
                    'model_name': model_name,
                    'version': old_version['version'],
                    'remove_files': True,  # Remove old version files
                    'remove_cache': True
                })
        
        if not to_remove:
            return {"message": "No old versions found to remove"}
        
        print(f"🧹 Found {len(to_remove)} old model versions to remove:")
        for item in to_remove:
            print(f"   - {item['model_name']} v{item['version']}")
        
        return self.remove_multiple_models(to_remove)
    
    def emergency_cleanup(self) -> Dict:
        """Emergency cleanup: remove all non-essential models to free memory"""
        print("🚨 EMERGENCY CLEANUP: Removing all models except essential ones")
        
        # Define essential models (customize as needed)
        essential_models = ['resnet50', 'bert-base']
        
        try:
            models_response = requests.get(f"{self.base_url}/models?loaded_only=true")
            models_response.raise_for_status()
            models_data = models_response.json()
        except Exception as e:
            return {"error": f"Failed to get loaded models: {e}"}
        
        to_remove = []
        for model in models_data.get('models', []):
            if model['name'] not in essential_models:
                to_remove.append({
                    'model_name': model['name'],
                    'version': model['version'],
                    'remove_files': False,  # Keep files, just unload
                    'remove_cache': True,
                    'force': True  # Force removal even if in use
                })
        
        if not to_remove:
            return {"message": "No non-essential models to remove"}
        
        return self.remove_multiple_models(to_remove, confirm_all=True)

# Usage Examples
remover = ModelRemover()

# Basic model removal
result = remover.remove_model("old-model", version="v1.0")
print(f"Removal result: {result.get('message')}")

# Complete removal with files
result = remover.remove_model(
    "bert-large",
    remove_files=True,
    remove_logs=True,
    backup_before_removal=True
)

# Batch removal
models_to_remove = [
    {"model_name": "old-resnet", "version": "v1.0", "remove_files": True},
    {"model_name": "deprecated-bert", "version": "v1.5", "remove_files": True},
    {"model_name": "unused-gpt", "version": "v2.0", "remove_files": False}
]

batch_result = remover.remove_multiple_models(models_to_remove)
print(f"Batch removal: {batch_result['summary']['successful']} successful, "
      f"{batch_result['summary']['failed']} failed")

# Safe cleanup of unused models (dry run first)
unused_result = remover.safe_remove_unused_models(dry_run=True)
if unused_result.get('total_memory_mb', 0) > 500:  # If more than 500MB can be freed
    remover.safe_remove_unused_models(dry_run=False)

# Cleanup old versions
cleanup_result = remover.cleanup_old_versions(keep_latest=2)
print(f"Old versions cleanup: {cleanup_result.get('message', 'Completed')}")

# Emergency cleanup (use with caution)
if input("Perform emergency cleanup? (yes/no): ").lower() == 'yes':
    emergency_result = remover.emergency_cleanup()
    print(f"Emergency cleanup: {emergency_result.get('message', 'Completed')}")
```

### JavaScript Model Removal Manager

```javascript
class ModelRemovalManager {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }
  
  async removeModel(modelName, options = {}) {
    const params = new URLSearchParams();
    if (options.version) params.append('version', options.version);
    if (options.removeFiles) params.append('remove_files', 'true');
    if (options.force) params.append('force', 'true');
    
    const payload = {
      confirm: true,
      remove_files: options.removeFiles || false,
      remove_cache: options.removeCache !== false,
      remove_logs: options.removeLogs || false,
      backup_before_removal: options.backupBeforeRemoval || false,
      force: options.force || false
    };
    
    try {
      const response = await fetch(
        `${this.baseUrl}/model-remove/${modelName}?${params}`,
        {
          method: 'DELETE',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        }
      );
      
      return await response.json();
    } catch (error) {
      return { status: 'error', error: error.message };
    }
  }
  
  renderRemovalInterface(containerId) {
    const container = document.getElementById(containerId);
    
    container.innerHTML = `
      <div class="removal-interface">
        <div class="interface-header">
          <h3>Model Removal Manager</h3>
          <p>Remove models from memory and optionally delete files</p>
        </div>
        
        <div class="removal-form">
          <div class="form-group">
            <label for="model-select">Select Model:</label>
            <select id="model-select" onchange="this.updateModelInfo()">
              <option value="">Loading models...</option>
            </select>
          </div>
          
          <div class="form-group">
            <label for="version-select">Version:</label>
            <select id="version-select">
              <option value="latest">Latest</option>
              <option value="all">All Versions</option>
            </select>
          </div>
          
          <div class="options-group">
            <h4>Removal Options</h4>
            
            <label class="checkbox-label">
              <input type="checkbox" id="remove-files">
              <span class="checkbox-text">
                <strong>Delete Files</strong> - Remove model files from disk (cannot be undone)
              </span>
            </label>
            
            <label class="checkbox-label">
              <input type="checkbox" id="remove-cache" checked>
              <span class="checkbox-text">
                <strong>Clear Cache</strong> - Remove cached inference results
              </span>
            </label>
            
            <label class="checkbox-label">
              <input type="checkbox" id="remove-logs">
              <span class="checkbox-text">
                <strong>Remove Logs</strong> - Delete model-specific log files
              </span>
            </label>
            
            <label class="checkbox-label">
              <input type="checkbox" id="backup-before">
              <span class="checkbox-text">
                <strong>Backup First</strong> - Create backup before removal
              </span>
            </label>
            
            <label class="checkbox-label">
              <input type="checkbox" id="force-removal">
              <span class="checkbox-text">
                <strong>Force Removal</strong> - Remove even if model is in use
              </span>
            </label>
          </div>
          
          <div class="warning-section">
            <div class="warning-box">
              <h4>⚠️ Warning</h4>
              <p>Model removal cannot be easily undone. Make sure you have backups if needed.</p>
            </div>
          </div>
          
          <div class="action-buttons">
            <button id="preview-btn" class="btn-secondary" onclick="this.previewRemoval()">
              Preview Removal
            </button>
            <button id="remove-btn" class="btn-danger" onclick="this.confirmAndRemove()">
              Remove Model
            </button>
          </div>
        </div>
        
        <div id="model-info" class="model-info-section"></div>
        <div id="removal-preview" class="preview-section"></div>
        <div id="removal-status" class="status-section"></div>
      </div>
    `;
    
    this.loadModels();
  }
  
  async loadModels() {
    try {
      const response = await fetch(`${this.baseUrl}/models`);
      const data = await response.json();
      
      const select = document.getElementById('model-select');
      select.innerHTML = '<option value="">Select a model...</option>';
      
      data.models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.name;
        option.textContent = `${model.name} (${model.status})`;
        select.appendChild(option);
      });
    } catch (error) {
      console.error('Error loading models:', error);
    }
  }
  
  async updateModelInfo() {
    const modelName = document.getElementById('model-select').value;
    const infoDiv = document.getElementById('model-info');
    
    if (!modelName) {
      infoDiv.innerHTML = '';
      return;
    }
    
    try {
      const response = await fetch(`${this.baseUrl}/model-info/${modelName}?include_usage_stats=true`);
      const info = await response.json();
      
      const specs = info.technical_specs || {};
      const usage = info.usage_statistics || {};
      
      infoDiv.innerHTML = `
        <div class="model-info-card">
          <h4>${info.model_name} - ${info.version}</h4>
          <div class="info-grid">
            <div class="info-item">
              <label>Status:</label>
              <span class="status-${info.status}">${info.status.toUpperCase()}</span>
            </div>
            <div class="info-item">
              <label>Size:</label>
              <span>${specs.model_size_mb || 'N/A'} MB</span>
            </div>
            <div class="info-item">
              <label>Memory Usage:</label>
              <span>${specs.memory_usage_mb || 'N/A'} MB</span>
            </div>
            <div class="info-item">
              <label>Total Requests:</label>
              <span>${usage.total_requests?.toLocaleString() || 'N/A'}</span>
            </div>
            <div class="info-item">
              <label>Success Rate:</label>
              <span>${usage.success_rate || 'N/A'}%</span>
            </div>
            <div class="info-item">
              <label>Last Used:</label>
              <span>${usage.last_request_time ? new Date(usage.last_request_time).toLocaleDateString() : 'N/A'}</span>
            </div>
          </div>
        </div>
      `;
    } catch (error) {
      infoDiv.innerHTML = `<div class="error">Error loading model info: ${error.message}</div>`;
    }
  }
  
  previewRemoval() {
    const modelName = document.getElementById('model-select').value;
    const version = document.getElementById('version-select').value;
    const removeFiles = document.getElementById('remove-files').checked;
    const removeCache = document.getElementById('remove-cache').checked;
    const removeLogs = document.getElementById('remove-logs').checked;
    const backupBefore = document.getElementById('backup-before').checked;
    const force = document.getElementById('force-removal').checked;
    
    if (!modelName) {
      alert('Please select a model first');
      return;
    }
    
    const previewDiv = document.getElementById('removal-preview');
    
    const actions = [];
    if (removeFiles) actions.push('🗑️ Delete model files from disk');
    if (removeCache) actions.push('🧹 Clear inference cache');
    if (removeLogs) actions.push('📋 Remove log files');
    if (backupBefore) actions.push('💾 Create backup before removal');
    if (force) actions.push('⚡ Force removal (ignore active usage)');
    
    previewDiv.innerHTML = `
      <div class="preview-card">
        <h4>Removal Preview</h4>
        <div class="preview-details">
          <div class="preview-item">
            <strong>Model:</strong> ${modelName}
          </div>
          <div class="preview-item">
            <strong>Version:</strong> ${version}
          </div>
          <div class="preview-item">
            <strong>Actions:</strong>
            <ul class="action-list">
              <li>🚫 Unload from memory</li>
              ${actions.map(action => `<li>${action}</li>`).join('')}
            </ul>
          </div>
        </div>
        
        ${removeFiles ? `
          <div class="warning-box">
            <strong>⚠️ WARNING:</strong> This will permanently delete model files. 
            This action cannot be undone unless you create a backup.
          </div>
        ` : ''}
      </div>
    `;
  }
  
  async confirmAndRemove() {
    const modelName = document.getElementById('model-select').value;
    const version = document.getElementById('version-select').value;
    
    if (!modelName) {
      alert('Please select a model first');
      return;
    }
    
    const removeFiles = document.getElementById('remove-files').checked;
    const confirmMessage = removeFiles 
      ? `This will permanently delete ${modelName} files. Are you sure?`
      : `Remove ${modelName} from memory?`;
    
    if (!confirm(confirmMessage)) {
      return;
    }
    
    const statusDiv = document.getElementById('removal-status');
    statusDiv.innerHTML = `
      <div class="status-loading">
        <div class="spinner"></div>
        <p>Removing ${modelName}...</p>
      </div>
    `;
    
    const options = {
      version: version === 'latest' ? undefined : version,
      removeFiles: document.getElementById('remove-files').checked,
      removeCache: document.getElementById('remove-cache').checked,
      removeLogs: document.getElementById('remove-logs').checked,
      backupBeforeRemoval: document.getElementById('backup-before').checked,
      force: document.getElementById('force-removal').checked
    };
    
    const result = await this.removeModel(modelName, options);
    
    if (result.status.includes('removed')) {
      statusDiv.innerHTML = `
        <div class="status-success">
          <h4>✅ Success</h4>
          <p>${result.message}</p>
          <div class="cleanup-summary">
            <h5>Cleanup Summary:</h5>
            <ul>
              <li>Memory freed: ${result.cleanup_summary?.memory_freed_mb || 0} MB</li>
              <li>Cache entries removed: ${result.cleanup_summary?.cache_entries_removed || 0}</li>
              <li>Disk space freed: ${result.removal_details?.disk_space_freed_mb || 0} MB</li>
            </ul>
          </div>
          ${result.removal_details?.backup_location ? `
            <div class="backup-info">
              <strong>Backup created:</strong> ${result.removal_details.backup_location}
            </div>
          ` : ''}
        </div>
      `;
      
      // Refresh the models list
      this.loadModels();
    } else {
      statusDiv.innerHTML = `
        <div class="status-error">
          <h4>❌ Error</h4>
          <p>${result.message || result.error}</p>
        </div>
      `;
    }
  }
}

// Utility functions for the interface
ModelRemovalManager.prototype.updateModelInfo = function() {
  this.updateModelInfo();
};

ModelRemovalManager.prototype.previewRemoval = function() {
  this.previewRemoval();
};

ModelRemovalManager.prototype.confirmAndRemove = function() {
  this.confirmAndRemove();
};

// Initialize the removal manager
const removalManager = new ModelRemovalManager();
removalManager.renderRemovalInterface('removal-container');
```

### Bash Cleanup Script

```bash
#!/bin/bash
# Model cleanup utility script

BASE_URL="http://localhost:8000"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to remove a single model
remove_model() {
    local model_name="$1"
    local version="${2:-latest}"
    local remove_files="${3:-false}"
    local force="${4:-false}"
    
    echo -e "${BLUE}Removing model: ${model_name} v${version}${NC}"
    
    local params=""
    if [[ "$version" != "latest" ]]; then
        params="?version=${version}"
    fi
    if [[ "$remove_files" == "true" ]]; then
        params="${params:+$params&}remove_files=true"
        params="${params:+?}${params#&}"
    fi
    if [[ "$force" == "true" ]]; then
        params="${params:+$params&}force=true"
        params="${params:+?}${params#&}"
    fi
    
    local payload='{"confirm": true'
    if [[ "$remove_files" == "true" ]]; then
        payload="${payload}, \"remove_files\": true"
    fi
    if [[ "$force" == "true" ]]; then
        payload="${payload}, \"force\": true"
    fi
    payload="${payload}}"
    
    local response=$(curl -s -X DELETE \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "${BASE_URL}/model-remove/${model_name}${params}")
    
    local status=$(echo "$response" | jq -r '.status // "error"')
    local message=$(echo "$response" | jq -r '.message // .error // "Unknown error"')
    
    case "$status" in
        *"removed"*)
            echo -e "${GREEN}✓ Success: ${message}${NC}"
            
            # Show cleanup summary if available
            local memory_freed=$(echo "$response" | jq -r '.cleanup_summary.memory_freed_mb // 0')
            local disk_freed=$(echo "$response" | jq -r '.removal_details.disk_space_freed_mb // 0')
            
            if [[ "$memory_freed" != "0" ]] || [[ "$disk_freed" != "0" ]]; then
                echo "  Memory freed: ${memory_freed} MB, Disk freed: ${disk_freed} MB"
            fi
            ;;
        "error")
            echo -e "${RED}✗ Error: ${message}${NC}"
            return 1
            ;;
        *)
            echo -e "${YELLOW}? Unknown status: ${status} - ${message}${NC}"
            return 1
            ;;
    esac
}

# Function to list models for removal selection
list_models_for_removal() {
    echo -e "${BLUE}Available models for removal:${NC}"
    echo "================================"
    
    local models=$(curl -s "${BASE_URL}/models")
    local model_names=$(echo "$models" | jq -r '.models[].name' | sort -u)
    
    local i=1
    while read -r model_name; do
        if [[ -n "$model_name" ]]; then
            local model_info=$(echo "$models" | jq -r ".models[] | select(.name == \"$model_name\")")
            local status=$(echo "$model_info" | jq -r '.status')
            local version=$(echo "$model_info" | jq -r '.version')
            
            printf "%2d) %-20s %-10s %s\n" "$i" "$model_name" "v$version" "($status)"
            ((i++))
        fi
    done <<< "$model_names"
}

# Function for interactive model removal
interactive_removal() {
    list_models_for_removal
    
    echo -e "\n${YELLOW}Enter model name to remove (or 'quit' to exit):${NC}"
    read -r model_name
    
    if [[ "$model_name" == "quit" ]]; then
        exit 0
    fi
    
    echo -e "${YELLOW}Remove files from disk? (y/n):${NC}"
    read -r remove_files_input
    local remove_files="false"
    if [[ "$remove_files_input" =~ ^[Yy] ]]; then
        remove_files="true"
        echo -e "${RED}WARNING: This will permanently delete model files!${NC}"
        echo -e "${YELLOW}Are you sure? (yes/no):${NC}"
        read -r confirm
        if [[ "$confirm" != "yes" ]]; then
            echo "Removal cancelled."
            return
        fi
    fi
    
    remove_model "$model_name" "latest" "$remove_files"
}

# Function to remove unused models
remove_unused_models() {
    local dry_run="${1:-true}"
    local cutoff_days="${2:-30}"
    
    echo -e "${BLUE}Finding models unused for more than ${cutoff_days} days...${NC}"
    
    local models=$(curl -s "${BASE_URL}/models?loaded_only=true")
    local model_names=$(echo "$models" | jq -r '.models[].name')
    
    local unused_models=()
    
    while read -r model_name; do
        if [[ -n "$model_name" ]]; then
            local info=$(curl -s "${BASE_URL}/model-info/${model_name}?include_usage_stats=true")
            local days_unused=$(echo "$info" | jq -r '.usage_statistics.days_since_last_request // 999')
            
            if [[ "$days_unused" -ge "$cutoff_days" ]]; then
                unused_models+=("$model_name:$days_unused")
            fi
        fi
    done <<< "$model_names"
    
    if [[ ${#unused_models[@]} -eq 0 ]]; then
        echo -e "${GREEN}No unused models found.${NC}"
        return
    fi
    
    echo -e "${YELLOW}Found ${#unused_models[@]} unused models:${NC}"
    for model_info in "${unused_models[@]}"; do
        IFS=':' read -r name days <<< "$model_info"
        echo "  - $name (unused for $days days)"
    done
    
    if [[ "$dry_run" == "true" ]]; then
        echo -e "\n${BLUE}This was a dry run. Use --execute to actually remove these models.${NC}"
        return
    fi
    
    echo -e "\n${YELLOW}Remove these unused models? (yes/no):${NC}"
    read -r confirm
    if [[ "$confirm" == "yes" ]]; then
        for model_info in "${unused_models[@]}"; do
            IFS=':' read -r name days <<< "$model_info"
            remove_model "$name" "latest" "false"
        done
    fi
}

# Function to remove old versions
cleanup_old_versions() {
    local keep_latest="${1:-2}"
    
    echo -e "${BLUE}Cleaning up old versions, keeping latest ${keep_latest} versions...${NC}"
    
    # This is a simplified version - in practice you'd need more sophisticated version parsing
    local models=$(curl -s "${BASE_URL}/models")
    local all_models=$(echo "$models" | jq -r '.models[] | "\(.name):\(.version)"')
    
    # Group by model name and find old versions
    declare -A model_versions
    
    while read -r model_info; do
        if [[ -n "$model_info" ]]; then
            IFS=':' read -r name version <<< "$model_info"
            if [[ -n "${model_versions[$name]}" ]]; then
                model_versions[$name]+=" $version"
            else
                model_versions[$name]="$version"
            fi
        fi
    done <<< "$all_models"
    
    # For each model, remove older versions
    for model_name in "${!model_versions[@]}"; do
        local versions=(${model_versions[$model_name]})
        local version_count=${#versions[@]}
        
        if [[ $version_count -gt $keep_latest ]]; then
            echo "Model $model_name has $version_count versions, removing oldest ones..."
            
            # Sort versions (simple string sort - may need improvement)
            IFS=$'\n' sorted_versions=($(sort <<< "${versions[*]}"))
            
            # Remove old versions (keep latest N)
            local to_remove=$((version_count - keep_latest))
            for ((i=0; i<to_remove; i++)); do
                echo "  Removing old version: ${sorted_versions[i]}"
                remove_model "$model_name" "${sorted_versions[i]}" "true"
            done
        fi
    done
}

# Main script logic
case "${1:-}" in
    "--interactive" | "-i")
        interactive_removal
        ;;
    "--unused" | "-u")
        remove_unused_models "${2:-true}"
        ;;
    "--unused-execute")
        remove_unused_models "false" "${2:-30}"
        ;;
    "--cleanup-versions" | "-c")
        cleanup_old_versions "${2:-2}"
        ;;
    "--remove" | "-r")
        if [[ -z "$2" ]]; then
            echo "Usage: $0 --remove <model_name> [version] [remove_files] [force]"
            exit 1
        fi
        remove_model "$2" "${3:-latest}" "${4:-false}" "${5:-false}"
        ;;
    "--help" | "-h" | "")
        echo "Model Removal Utility"
        echo "====================="
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  -i, --interactive        Interactive model removal"
        echo "  -u, --unused            List unused models (dry run)"
        echo "  --unused-execute [days] Remove unused models (default: 30 days)"
        echo "  -c, --cleanup-versions  Clean up old model versions"
        echo "  -r, --remove <name>     Remove specific model"
        echo "  -h, --help              Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 --interactive"
        echo "  $0 --unused"
        echo "  $0 --unused-execute 45"
        echo "  $0 --cleanup-versions 3"
        echo "  $0 --remove resnet50 v1.0 true false"
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
```

## Error Handling

### Model Not Found (404)
```json
{
  "detail": "Model 'unknown-model' not found"
}
```

### Missing Confirmation (400)
```json
{
  "detail": "Removal confirmation required. Set 'confirm': true in request body"
}
```

### Model In Use (409)
```json
{
  "detail": "Model 'resnet50' is currently processing requests. Use force=true to override"
}
```

### System Lock (423)
```json
{
  "detail": "Model removal is locked by system policy"
}
```

## Related Endpoints

- [Models List](./models.md) - View available models
- [Model Info](./model-info.md) - Get model details before removal
- [Cache Info](./cache-info.md) - Check cache status
- [Health Check](./health.md) - System health after cleanup

## Removal Strategies

### Memory-Only Removal
- Unloads model from memory
- Keeps files on disk for future use
- Fastest option for temporary cleanup

### Complete Removal
- Removes from memory and deletes files
- Frees maximum space
- Irreversible without backups

### Version-Specific Removal
- Removes only specific versions
- Useful for cleanup while keeping latest
- Gradual cleanup strategy

## Best Practices

1. **Backup Important Models**: Always backup before complete removal
2. **Check Usage**: Review usage statistics before removing
3. **Gradual Cleanup**: Remove unused models incrementally
4. **Monitor Memory**: Check memory usage before and after removal
5. **Version Management**: Keep only necessary versions to save space
6. **Emergency Procedures**: Have procedures for emergency cleanup
7. **Confirmation**: Always require explicit confirmation for destructive operations

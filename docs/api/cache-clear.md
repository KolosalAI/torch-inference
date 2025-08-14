# POST /cache-clear - Clear Cache

**URL**: `/cache-clear`  
**Method**: `POST`  
**Authentication**: Not required  
**Content-Type**: `application/json`

## Description

Clears cache data based on specified criteria. This endpoint allows selective or complete cache clearing for maintenance, troubleshooting, or memory management purposes. Use with caution in production environments.

## Request

### Request Body

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `cache_types` | array | No | ["all"] | Types of cache to clear: "inference", "model", "download", "all" |
| `models` | array | No | [] | Specific models to clear from cache |
| `age_threshold_hours` | number | No | null | Clear entries older than specified hours |
| `force` | boolean | No | false | Force clear even if cache is actively used |
| `confirmation` | string | No | null | Confirmation string "CONFIRM_CLEAR" for safety |
| `preserve_active` | boolean | No | true | Whether to preserve currently active cache entries |

### Request Examples

#### Clear All Cache
```json
{
  "cache_types": ["all"],
  "confirmation": "CONFIRM_CLEAR"
}
```

#### Clear Specific Cache Types
```json
{
  "cache_types": ["inference", "download"],
  "confirmation": "CONFIRM_CLEAR"
}
```

#### Clear Specific Models
```json
{
  "cache_types": ["model", "inference"],
  "models": ["resnet50", "bert-base"],
  "confirmation": "CONFIRM_CLEAR"
}
```

#### Clear Old Entries
```json
{
  "cache_types": ["inference"],
  "age_threshold_hours": 24,
  "preserve_active": false
}
```

#### Force Clear Everything
```json
{
  "cache_types": ["all"],
  "force": true,
  "preserve_active": false,
  "confirmation": "CONFIRM_CLEAR"
}
```

## Response

### Success Response (200 OK)

#### Successful Cache Clear
```json
{
  "status": "success",
  "message": "Cache cleared successfully",
  "cleared": {
    "inference_cache": {
      "entries_removed": 2450,
      "space_freed_mb": 456.7,
      "clearing_time_ms": 1250
    },
    "model_cache": {
      "models_unloaded": 3,
      "space_freed_mb": 234.5,
      "clearing_time_ms": 890
    },
    "download_cache": {
      "files_removed": 12,
      "space_freed_mb": 187.3,
      "clearing_time_ms": 340
    }
  },
  "summary": {
    "total_entries_removed": 2465,
    "total_space_freed_mb": 878.5,
    "total_clearing_time_ms": 2480,
    "cache_types_cleared": ["inference", "model", "download"],
    "cleared_at": "2025-08-14T10:30:00.000Z"
  },
  "cache_state_after": {
    "total_cache_size_mb": 1024.0,
    "used_cache_size_mb": 145.5,
    "available_cache_size_mb": 878.5,
    "cache_usage_percentage": 14.2
  }
}
```

#### Partial Cache Clear
```json
{
  "status": "partial_success",
  "message": "Cache partially cleared - some entries preserved",
  "cleared": {
    "inference_cache": {
      "entries_removed": 1200,
      "entries_preserved": 450,
      "space_freed_mb": 287.3,
      "clearing_time_ms": 890
    }
  },
  "warnings": [
    "Some active inference entries were preserved",
    "Model 'resnet50' is currently in use and was not cleared"
  ],
  "summary": {
    "total_entries_removed": 1200,
    "total_entries_preserved": 450,
    "total_space_freed_mb": 287.3,
    "preservation_reason": "preserve_active=true"
  }
}
```

#### Cache Clear with Age Filter
```json
{
  "status": "success",
  "message": "Old cache entries cleared successfully",
  "cleared": {
    "inference_cache": {
      "entries_removed": 567,
      "oldest_entry_age_hours": 48.5,
      "newest_entry_age_hours": 24.1,
      "space_freed_mb": 134.2
    }
  },
  "filter_applied": {
    "age_threshold_hours": 24,
    "entries_evaluated": 2450,
    "entries_matched": 567,
    "entries_skipped": 1883
  }
}
```

#### Dry Run Response (when using force=false with active cache)
```json
{
  "status": "preview",
  "message": "Cache clear preview - no changes made",
  "would_clear": {
    "inference_cache": {
      "entries_to_remove": 2450,
      "space_to_free_mb": 456.7,
      "active_entries_affected": 12
    },
    "model_cache": {
      "models_to_unload": 3,
      "space_to_free_mb": 234.5,
      "active_models_affected": 1
    }
  },
  "warnings": [
    "1 model is currently being used for inference",
    "12 inference cache entries are for active requests"
  ],
  "recommendation": "Use 'force=true' or 'preserve_active=true' to proceed"
}
```

### Error Responses

#### Missing Confirmation (400 Bad Request)
```json
{
  "error": "confirmation_required",
  "message": "Cache clearing requires confirmation",
  "required_confirmation": "CONFIRM_CLEAR",
  "details": "To clear cache, include 'confirmation': 'CONFIRM_CLEAR' in request body"
}
```

#### Invalid Cache Type (400 Bad Request)
```json
{
  "error": "invalid_cache_type",
  "message": "Invalid cache type specified",
  "invalid_types": ["invalid_type"],
  "valid_types": ["inference", "model", "download", "all"]
}
```

#### Cache Operation Failed (500 Internal Server Error)
```json
{
  "error": "cache_clear_failed",
  "message": "Failed to clear cache",
  "details": "Unable to acquire cache lock for clearing",
  "failed_operations": ["inference_cache"],
  "successful_operations": ["model_cache"]
}
```

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success - Cache cleared successfully |
| 400 | Bad Request - Invalid parameters or missing confirmation |
| 409 | Conflict - Cache is locked or actively being used |
| 500 | Internal server error |

## Examples

### Clear All Cache Types

**Request:**
```bash
curl -X POST http://localhost:8000/cache-clear \
  -H "Content-Type: application/json" \
  -d '{
    "cache_types": ["all"],
    "confirmation": "CONFIRM_CLEAR"
  }'
```

**Response:**
```json
{
  "status": "success",
  "message": "Cache cleared successfully",
  "summary": {
    "total_entries_removed": 2465,
    "total_space_freed_mb": 878.5,
    "cache_types_cleared": ["inference", "model", "download"]
  }
}
```

### Clear Specific Model Cache

**Request:**
```bash
curl -X POST http://localhost:8000/cache-clear \
  -H "Content-Type: application/json" \
  -d '{
    "cache_types": ["model", "inference"],
    "models": ["resnet50", "old-model"],
    "confirmation": "CONFIRM_CLEAR"
  }'
```

### Clear Old Entries Only

**Request:**
```bash
curl -X POST http://localhost:8000/cache-clear \
  -H "Content-Type: application/json" \
  -d '{
    "cache_types": ["inference"],
    "age_threshold_hours": 48,
    "preserve_active": true
  }'
```

### Preview Clear Operation

**Request:**
```bash
curl -X POST http://localhost:8000/cache-clear \
  -H "Content-Type: application/json" \
  -d '{
    "cache_types": ["all"],
    "force": false
  }'
```

### Python Cache Clear Manager

```python
import requests
import json
from typing import List, Optional, Dict
from datetime import datetime
import time

class CacheClearManager:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def clear_cache(
        self,
        cache_types: List[str] = None,
        models: List[str] = None,
        age_threshold_hours: Optional[float] = None,
        force: bool = False,
        preserve_active: bool = True,
        confirmation: bool = False
    ) -> Dict:
        """Clear cache with specified parameters"""
        
        if cache_types is None:
            cache_types = ["all"]
        
        payload = {
            "cache_types": cache_types,
            "force": force,
            "preserve_active": preserve_active
        }
        
        if models:
            payload["models"] = models
        
        if age_threshold_hours is not None:
            payload["age_threshold_hours"] = age_threshold_hours
        
        # Add confirmation for destructive operations
        if confirmation and (force or "all" in cache_types or not preserve_active):
            payload["confirmation"] = "CONFIRM_CLEAR"
        
        try:
            response = requests.post(f"{self.base_url}/cache-clear", json=payload)
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def preview_clear(self, cache_types: List[str] = None, models: List[str] = None) -> Dict:
        """Preview what would be cleared without actually clearing"""
        return self.clear_cache(
            cache_types=cache_types,
            models=models,
            force=False,
            confirmation=False
        )
    
    def emergency_clear_all(self) -> Dict:
        """Emergency clear all cache - use with extreme caution"""
        print("🚨 WARNING: This will clear ALL cache data immediately!")
        confirmation = input("Type 'EMERGENCY_CLEAR' to confirm: ")
        
        if confirmation != "EMERGENCY_CLEAR":
            return {"error": "Operation cancelled by user"}
        
        return self.clear_cache(
            cache_types=["all"],
            force=True,
            preserve_active=False,
            confirmation=True
        )
    
    def clear_old_entries(self, hours: float = 24.0) -> Dict:
        """Clear entries older than specified hours"""
        return self.clear_cache(
            age_threshold_hours=hours,
            preserve_active=True
        )
    
    def clear_specific_models(self, models: List[str]) -> Dict:
        """Clear cache for specific models"""
        if not models:
            return {"error": "No models specified"}
        
        print(f"Clearing cache for models: {', '.join(models)}")
        confirm = input("Proceed? (y/N): ").lower()
        
        if confirm != 'y':
            return {"error": "Operation cancelled by user"}
        
        return self.clear_cache(
            cache_types=["model", "inference"],
            models=models,
            confirmation=True
        )
    
    def maintenance_cleanup(
        self,
        max_age_hours: float = 72.0,
        preserve_popular: bool = True
    ) -> Dict:
        """Perform maintenance cleanup of cache"""
        print(f"🧹 Starting maintenance cleanup...")
        print(f"   - Clearing entries older than {max_age_hours} hours")
        print(f"   - Preserve active entries: {preserve_popular}")
        
        result = self.clear_cache(
            cache_types=["inference", "download"],
            age_threshold_hours=max_age_hours,
            preserve_active=preserve_popular
        )
        
        if "error" not in result:
            print("✅ Maintenance cleanup completed successfully")
            if "summary" in result:
                summary = result["summary"]
                print(f"   - Removed {summary.get('total_entries_removed', 0):,} entries")
                print(f"   - Freed {summary.get('total_space_freed_mb', 0):.1f} MB")
        else:
            print(f"❌ Cleanup failed: {result['error']}")
        
        return result
    
    def intelligent_cleanup(self) -> Dict:
        """Perform intelligent cleanup based on cache analysis"""
        print("🤖 Analyzing cache for intelligent cleanup...")
        
        # Get current cache info
        try:
            cache_response = requests.get(f"{self.base_url}/cache-info")
            cache_data = cache_response.json()
        except:
            return {"error": "Unable to analyze cache"}
        
        recommendations = []
        cleanup_actions = []
        
        # Analyze cache usage
        cache_info = cache_data.get("cache_info", {})
        usage_pct = cache_info.get("cache_usage_percentage", 0)
        
        if usage_pct > 85:
            recommendations.append("🚨 High cache usage - aggressive cleanup recommended")
            cleanup_actions.append({
                "cache_types": ["inference"],
                "age_threshold_hours": 12.0,
                "preserve_active": True
            })
        elif usage_pct > 70:
            recommendations.append("⚠️ Moderate cache usage - standard cleanup recommended")
            cleanup_actions.append({
                "cache_types": ["inference"],
                "age_threshold_hours": 24.0,
                "preserve_active": True
            })
        
        # Analyze inference cache performance
        inf_cache = cache_data.get("inference_cache", {})
        if inf_cache.get("enabled"):
            hit_rate = inf_cache.get("hit_rate_percentage", 0)
            if hit_rate < 50:
                recommendations.append("📊 Low hit rate - consider clearing ineffective entries")
                cleanup_actions.append({
                    "cache_types": ["inference"],
                    "age_threshold_hours": 6.0,
                    "preserve_active": True
                })
        
        # Check for partial downloads
        dl_cache = cache_data.get("download_cache", {})
        if dl_cache.get("partial_downloads", 0) > 0:
            recommendations.append("🗂️ Partial downloads found - cleanup recommended")
            cleanup_actions.append({
                "cache_types": ["download"],
                "age_threshold_hours": 48.0,
                "preserve_active": False
            })
        
        if not cleanup_actions:
            return {
                "status": "no_action_needed",
                "message": "Cache is healthy - no cleanup needed",
                "cache_usage_percentage": usage_pct
            }
        
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  {rec}")
        
        print(f"\nProposed cleanup actions:")
        for i, action in enumerate(cleanup_actions):
            print(f"  {i+1}. Clear {', '.join(action['cache_types'])} cache")
            if action.get("age_threshold_hours"):
                print(f"     - Entries older than {action['age_threshold_hours']} hours")
            if action.get("preserve_active"):
                print(f"     - Preserve active entries")
        
        proceed = input(f"\nProceed with cleanup? (y/N): ").lower()
        if proceed != 'y':
            return {"error": "Operation cancelled by user"}
        
        # Execute cleanup actions
        results = []
        for action in cleanup_actions:
            result = self.clear_cache(**action)
            results.append(result)
            time.sleep(1)  # Brief pause between operations
        
        return {
            "status": "intelligent_cleanup_completed",
            "actions_taken": len(cleanup_actions),
            "results": results,
            "recommendations": recommendations
        }
    
    def schedule_cleanup(self, interval_hours: float = 24.0):
        """Schedule periodic cache cleanup"""
        print(f"🕐 Starting scheduled cleanup every {interval_hours} hours...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running scheduled cleanup...")
                
                result = self.maintenance_cleanup()
                
                if "error" not in result:
                    print("✅ Scheduled cleanup completed")
                else:
                    print(f"❌ Scheduled cleanup failed: {result['error']}")
                
                print(f"💤 Sleeping for {interval_hours} hours...")
                time.sleep(interval_hours * 3600)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Scheduled cleanup stopped by user")
    
    def print_clear_result(self, result: Dict):
        """Print formatted clear result"""
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        status = result.get("status", "unknown")
        message = result.get("message", "")
        
        print(f"Status: {status}")
        print(f"Message: {message}")
        
        if "cleared" in result:
            print(f"\nClearing Results:")
            cleared = result["cleared"]
            
            for cache_type, details in cleared.items():
                print(f"  {cache_type.replace('_', ' ').title()}:")
                
                if "entries_removed" in details:
                    print(f"    - Entries removed: {details['entries_removed']:,}")
                if "models_unloaded" in details:
                    print(f"    - Models unloaded: {details['models_unloaded']}")
                if "files_removed" in details:
                    print(f"    - Files removed: {details['files_removed']}")
                if "space_freed_mb" in details:
                    print(f"    - Space freed: {details['space_freed_mb']:.1f} MB")
                if "clearing_time_ms" in details:
                    print(f"    - Time taken: {details['clearing_time_ms']} ms")
        
        if "summary" in result:
            summary = result["summary"]
            print(f"\nSummary:")
            print(f"  - Total entries removed: {summary.get('total_entries_removed', 0):,}")
            print(f"  - Total space freed: {summary.get('total_space_freed_mb', 0):.1f} MB")
            print(f"  - Total time: {summary.get('total_clearing_time_ms', 0)} ms")
        
        if "warnings" in result:
            print(f"\nWarnings:")
            for warning in result["warnings"]:
                print(f"  ⚠️  {warning}")
        
        if "cache_state_after" in result:
            state = result["cache_state_after"]
            usage_pct = state.get("cache_usage_percentage", 0)
            used_mb = state.get("used_cache_size_mb", 0)
            total_mb = state.get("total_cache_size_mb", 0)
            
            print(f"\nCache State After Clearing:")
            print(f"  - Usage: {usage_pct:.1f}% ({used_mb:.1f} MB / {total_mb:.1f} MB)")

# Usage Examples
manager = CacheClearManager()

# Preview what would be cleared
print("Previewing cache clear operation...")
preview = manager.preview_clear(cache_types=["inference"])
manager.print_clear_result(preview)

# Clear old entries
print("\nClearing entries older than 24 hours...")
result = manager.clear_old_entries(hours=24.0)
manager.print_clear_result(result)

# Clear specific models
print("\nClearing cache for specific models...")
models_result = manager.clear_specific_models(["old-model", "unused-model"])
manager.print_clear_result(models_result)

# Perform maintenance cleanup
print("\nPerforming maintenance cleanup...")
maintenance_result = manager.maintenance_cleanup(max_age_hours=48.0)

# Intelligent cleanup based on analysis
print("\nRunning intelligent cleanup...")
intelligent_result = manager.intelligent_cleanup()
manager.print_clear_result(intelligent_result)

# Schedule periodic cleanup (run in background)
# manager.schedule_cleanup(interval_hours=6.0)
```

### JavaScript Cache Clear Utility

```javascript
class CacheClearUtility {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }
  
  async clearCache(options = {}) {
    const payload = {
      cache_types: options.cacheTypes || ['all'],
      force: options.force || false,
      preserve_active: options.preserveActive !== false,
      ...options
    };
    
    // Add confirmation for destructive operations
    if (options.confirmation && (payload.force || payload.cache_types.includes('all') || !payload.preserve_active)) {
      payload.confirmation = 'CONFIRM_CLEAR';
    }
    
    try {
      const response = await fetch(`${this.baseUrl}/cache-clear`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      return await response.json();
    } catch (error) {
      return { error: error.message };
    }
  }
  
  async previewClear(cacheTypes = ['all']) {
    return this.clearCache({
      cacheTypes,
      force: false,
      confirmation: false
    });
  }
  
  async emergencyClear() {
    const confirm = window.confirm(
      '🚨 WARNING: This will clear ALL cache data immediately!\n\n' +
      'This action cannot be undone and may impact performance.\n\n' +
      'Click OK to proceed with emergency cache clear.'
    );
    
    if (!confirm) {
      return { error: 'Operation cancelled by user' };
    }
    
    return this.clearCache({
      cacheTypes: ['all'],
      force: true,
      preserveActive: false,
      confirmation: true
    });
  }
  
  async clearOldEntries(hours = 24) {
    return this.clearCache({
      ageThresholdHours: hours,
      preserveActive: true
    });
  }
  
  async clearSpecificModels(models) {
    if (!models || models.length === 0) {
      return { error: 'No models specified' };
    }
    
    const confirm = window.confirm(
      `Clear cache for models: ${models.join(', ')}?\n\n` +
      'This will remove cached data for these models.'
    );
    
    if (!confirm) {
      return { error: 'Operation cancelled by user' };
    }
    
    return this.clearCache({
      cacheTypes: ['model', 'inference'],
      models: models,
      confirmation: true
    });
  }
  
  async maintenanceCleanup(maxAgeHours = 72) {
    console.log('🧹 Starting maintenance cleanup...');
    console.log(`   - Clearing entries older than ${maxAgeHours} hours`);
    
    const result = await this.clearCache({
      cacheTypes: ['inference', 'download'],
      ageThresholdHours: maxAgeHours,
      preserveActive: true
    });
    
    if (!result.error) {
      console.log('✅ Maintenance cleanup completed successfully');
      if (result.summary) {
        const summary = result.summary;
        console.log(`   - Removed ${summary.total_entries_removed?.toLocaleString() || 0} entries`);
        console.log(`   - Freed ${summary.total_space_freed_mb?.toFixed(1) || 0} MB`);
      }
    } else {
      console.log(`❌ Cleanup failed: ${result.error}`);
    }
    
    return result;
  }
  
  renderClearControls(containerId) {
    const container = document.getElementById(containerId);
    
    container.innerHTML = `
      <div class="cache-clear-controls">
        <h3>Cache Clear Controls</h3>
        
        <div class="control-section">
          <h4>Quick Actions</h4>
          <div class="button-group">
            <button class="btn-primary" onclick="cacheUtil.handlePreviewClear()">
              👁️ Preview Clear
            </button>
            <button class="btn-secondary" onclick="cacheUtil.handleClearOld()">
              🗓️ Clear Old (24h)
            </button>
            <button class="btn-secondary" onclick="cacheUtil.handleMaintenance()">
              🧹 Maintenance
            </button>
            <button class="btn-danger" onclick="cacheUtil.handleEmergencyClear()">
              🚨 Emergency Clear All
            </button>
          </div>
        </div>
        
        <div class="control-section">
          <h4>Custom Clear</h4>
          <div class="form-group">
            <label>Cache Types:</label>
            <div class="checkbox-group">
              <label><input type="checkbox" value="inference" checked> Inference</label>
              <label><input type="checkbox" value="model"> Model</label>
              <label><input type="checkbox" value="download"> Download</label>
            </div>
          </div>
          
          <div class="form-group">
            <label>Models (comma-separated):</label>
            <input type="text" id="models-input" placeholder="resnet50, bert-base">
          </div>
          
          <div class="form-group">
            <label>Age Threshold (hours):</label>
            <input type="number" id="age-input" min="1" placeholder="24">
          </div>
          
          <div class="form-group">
            <label>
              <input type="checkbox" id="preserve-active" checked> 
              Preserve Active Entries
            </label>
          </div>
          
          <div class="form-group">
            <label>
              <input type="checkbox" id="force-clear"> 
              Force Clear (ignore locks)
            </label>
          </div>
          
          <button class="btn-primary" onclick="cacheUtil.handleCustomClear()">
            Clear Cache
          </button>
        </div>
        
        <div id="clear-results" class="results-section"></div>
      </div>
    `;
  }
  
  async handlePreviewClear() {
    const cacheTypes = this.getSelectedCacheTypes();
    const result = await this.previewClear(cacheTypes);
    this.displayResult(result, 'Preview Results');
  }
  
  async handleClearOld() {
    const result = await this.clearOldEntries(24);
    this.displayResult(result, 'Clear Old Entries');
  }
  
  async handleMaintenance() {
    const result = await this.maintenanceCleanup(72);
    this.displayResult(result, 'Maintenance Cleanup');
  }
  
  async handleEmergencyClear() {
    const result = await this.emergencyClear();
    this.displayResult(result, 'Emergency Clear');
  }
  
  async handleCustomClear() {
    const options = this.getCustomClearOptions();
    
    // Confirm destructive operations
    if (options.force || !options.preserveActive || options.cacheTypes.includes('all')) {
      const confirm = window.confirm(
        '⚠️ Warning: This operation may be destructive.\n\n' +
        'Proceeding may impact system performance.\n\n' +
        'Are you sure you want to continue?'
      );
      
      if (!confirm) {
        this.displayResult({ error: 'Operation cancelled by user' }, 'Custom Clear');
        return;
      }
      
      options.confirmation = true;
    }
    
    const result = await this.clearCache(options);
    this.displayResult(result, 'Custom Clear');
  }
  
  getSelectedCacheTypes() {
    const checkboxes = document.querySelectorAll('.checkbox-group input[type="checkbox"]:checked');
    const types = Array.from(checkboxes).map(cb => cb.value);
    return types.length > 0 ? types : ['all'];
  }
  
  getCustomClearOptions() {
    const cacheTypes = this.getSelectedCacheTypes();
    const modelsInput = document.getElementById('models-input').value.trim();
    const ageInput = document.getElementById('age-input').value;
    const preserveActive = document.getElementById('preserve-active').checked;
    const force = document.getElementById('force-clear').checked;
    
    const options = {
      cacheTypes,
      preserveActive,
      force
    };
    
    if (modelsInput) {
      options.models = modelsInput.split(',').map(m => m.trim()).filter(m => m);
    }
    
    if (ageInput && !isNaN(parseFloat(ageInput))) {
      options.ageThresholdHours = parseFloat(ageInput);
    }
    
    return options;
  }
  
  displayResult(result, title) {
    const container = document.getElementById('clear-results');
    
    let html = `<h4>${title}</h4>`;
    
    if (result.error) {
      html += `<div class="result-error">❌ Error: ${result.error}</div>`;
    } else {
      html += `<div class="result-success">✅ Status: ${result.status}</div>`;
      html += `<div class="result-message">${result.message}</div>`;
      
      if (result.cleared) {
        html += '<div class="result-details"><h5>Clearing Results:</h5>';
        
        for (const [cacheType, details] of Object.entries(result.cleared)) {
          html += `<div class="cache-type-result">`;
          html += `<strong>${cacheType.replace('_', ' ')}:</strong><br>`;
          
          if (details.entries_removed) {
            html += `&nbsp;&nbsp;Entries removed: ${details.entries_removed.toLocaleString()}<br>`;
          }
          if (details.models_unloaded) {
            html += `&nbsp;&nbsp;Models unloaded: ${details.models_unloaded}<br>`;
          }
          if (details.files_removed) {
            html += `&nbsp;&nbsp;Files removed: ${details.files_removed}<br>`;
          }
          if (details.space_freed_mb) {
            html += `&nbsp;&nbsp;Space freed: ${details.space_freed_mb.toFixed(1)} MB<br>`;
          }
          
          html += `</div>`;
        }
        
        html += '</div>';
      }
      
      if (result.summary) {
        const summary = result.summary;
        html += '<div class="result-summary"><h5>Summary:</h5>';
        html += `Total entries removed: ${(summary.total_entries_removed || 0).toLocaleString()}<br>`;
        html += `Total space freed: ${(summary.total_space_freed_mb || 0).toFixed(1)} MB<br>`;
        html += `Total time: ${summary.total_clearing_time_ms || 0} ms`;
        html += '</div>';
      }
      
      if (result.warnings && result.warnings.length > 0) {
        html += '<div class="result-warnings"><h5>Warnings:</h5>';
        result.warnings.forEach(warning => {
          html += `<div class="warning-item">⚠️ ${warning}</div>`;
        });
        html += '</div>';
      }
      
      if (result.would_clear) {
        html += '<div class="result-preview"><h5>Would Clear:</h5>';
        for (const [cacheType, details] of Object.entries(result.would_clear)) {
          html += `<div>${cacheType}: ${JSON.stringify(details)}</div>`;
        }
        html += '</div>';
      }
    }
    
    container.innerHTML = html;
  }
}

// Global instance
const cacheUtil = new CacheClearUtility();

// Initialize controls
// cacheUtil.renderClearControls('cache-clear-container');
```

## Safety Considerations

### Confirmation Requirements
- Destructive operations require explicit confirmation
- Preview mode available for testing
- Force operations have additional warnings

### Active Operation Protection
- `preserve_active=true` protects currently used cache entries
- Models in use are not unloaded by default
- Active inference requests are preserved

### Recovery Options
- Cache rebuilds automatically as needed
- Model cache reloads models on demand
- Download cache re-downloads as required

## Related Endpoints

- [Cache Info](./cache-info.md) - Monitor cache status before clearing
- [Stats](./stats.md) - Track system performance impact
- [Health](./health.md) - Verify system health after clearing

## Best Practices

1. **Always Preview First**: Use preview mode to understand impact
2. **Preserve Active**: Keep `preserve_active=true` in production
3. **Scheduled Maintenance**: Use age-based clearing for routine maintenance
4. **Monitor Impact**: Check system performance after clearing operations
5. **Confirmation Safety**: Always require confirmation for destructive operations
6. **Gradual Clearing**: Clear specific cache types rather than everything at once

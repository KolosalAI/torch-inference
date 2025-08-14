# GET /cache-info - Cache Information

**URL**: `/cache-info`  
**Method**: `GET`  
**Authentication**: Not required  
**Content-Type**: `application/json`

## Description

Retrieves detailed information about the system cache, including inference cache, model cache, download cache, and overall cache statistics. This endpoint helps monitor cache performance and manage storage usage.

## Request

### URL Parameters
None

### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `include_entries` | boolean | No | false | Include list of cache entries |
| `include_statistics` | boolean | No | true | Include cache performance statistics |
| `cache_type` | string | No | "all" | Specific cache type: "inference", "model", "download", or "all" |
| `details_level` | string | No | "summary" | Level of detail: "summary", "detailed", "full" |

### Request Body
None (GET request)

## Response

### Success Response (200 OK)

#### Summary Cache Information
```json
{
  "cache_info": {
    "total_cache_size_mb": 2048.7,
    "used_cache_size_mb": 1456.3,
    "available_cache_size_mb": 592.4,
    "cache_usage_percentage": 71.2,
    "last_updated": "2025-08-14T10:30:00.000Z"
  },
  "inference_cache": {
    "enabled": true,
    "size_limit_mb": 1024.0,
    "current_size_mb": 687.2,
    "entry_count": 3420,
    "hit_rate_percentage": 78.5,
    "miss_rate_percentage": 21.5,
    "eviction_count": 142,
    "last_cleanup": "2025-08-14T09:45:00.000Z"
  },
  "model_cache": {
    "enabled": true,
    "size_limit_mb": 512.0,
    "current_size_mb": 438.2,
    "entry_count": 5,
    "models_cached": [
      {
        "model_name": "resnet50",
        "version": "v1.0",
        "cache_size_mb": 97.8,
        "last_accessed": "2025-08-14T10:25:00.000Z"
      },
      {
        "model_name": "bert-base",
        "version": "v2.1", 
        "cache_size_mb": 210.4,
        "last_accessed": "2025-08-14T10:12:00.000Z"
      }
    ]
  },
  "download_cache": {
    "enabled": true,
    "size_limit_mb": 512.0,
    "current_size_mb": 331.0,
    "entry_count": 8,
    "partial_downloads": 2,
    "completed_downloads": 6,
    "oldest_entry": "2025-08-10T14:20:00.000Z"
  }
}
```

#### Detailed Cache Information (details_level=detailed)
```json
{
  "cache_info": {
    "total_cache_size_mb": 2048.7,
    "used_cache_size_mb": 1456.3,
    "available_cache_size_mb": 592.4,
    "cache_usage_percentage": 71.2,
    "cache_directories": [
      {
        "path": "/cache/inference",
        "size_mb": 687.2,
        "type": "inference"
      },
      {
        "path": "/cache/models",
        "size_mb": 438.2,
        "type": "model"
      },
      {
        "path": "/cache/downloads", 
        "size_mb": 331.0,
        "type": "download"
      }
    ],
    "cache_policies": {
      "inference_ttl_hours": 24,
      "model_ttl_hours": 168,
      "download_ttl_hours": 72,
      "max_entry_size_mb": 500,
      "cleanup_threshold_percentage": 85,
      "eviction_strategy": "lru"
    },
    "last_updated": "2025-08-14T10:30:00.000Z"
  },
  "inference_cache": {
    "enabled": true,
    "configuration": {
      "size_limit_mb": 1024.0,
      "max_entries": 10000,
      "ttl_hours": 24,
      "eviction_policy": "lru",
      "compression_enabled": true,
      "encryption_enabled": false
    },
    "current_status": {
      "current_size_mb": 687.2,
      "entry_count": 3420,
      "oldest_entry_age_hours": 22.5,
      "newest_entry_age_minutes": 2.1,
      "average_entry_size_kb": 205.8
    },
    "performance_metrics": {
      "hit_rate_percentage": 78.5,
      "miss_rate_percentage": 21.5,
      "total_requests": 15687,
      "cache_hits": 12314,
      "cache_misses": 3373,
      "eviction_count": 142,
      "average_lookup_time_ms": 0.8,
      "cache_saves": 3420,
      "save_failures": 12
    },
    "memory_usage": {
      "index_memory_mb": 45.2,
      "data_memory_mb": 642.0,
      "overhead_memory_mb": 31.7
    },
    "cleanup_history": [
      {
        "timestamp": "2025-08-14T09:45:00.000Z",
        "entries_removed": 87,
        "space_freed_mb": 125.4,
        "reason": "scheduled_cleanup"
      },
      {
        "timestamp": "2025-08-14T06:15:00.000Z",
        "entries_removed": 55,
        "space_freed_mb": 89.2,
        "reason": "size_limit_exceeded"
      }
    ]
  },
  "model_cache": {
    "enabled": true,
    "configuration": {
      "size_limit_mb": 512.0,
      "max_models": 10,
      "preload_popular": true,
      "lazy_loading": true,
      "compression_enabled": false
    },
    "current_status": {
      "current_size_mb": 438.2,
      "entry_count": 5,
      "loaded_models": 3,
      "cached_models": 2,
      "available_space_mb": 73.8
    },
    "models_cached": [
      {
        "model_name": "resnet50",
        "version": "v1.0",
        "cache_size_mb": 97.8,
        "status": "loaded",
        "last_accessed": "2025-08-14T10:25:00.000Z",
        "access_count": 1247,
        "load_time_ms": 2340,
        "memory_usage_mb": 125.4
      },
      {
        "model_name": "bert-base",
        "version": "v2.1",
        "cache_size_mb": 210.4,
        "status": "cached",
        "last_accessed": "2025-08-14T10:12:00.000Z",
        "access_count": 892,
        "load_time_ms": 4567,
        "memory_usage_mb": 0
      },
      {
        "model_name": "efficientnet",
        "version": "v1.2",
        "cache_size_mb": 52.3,
        "status": "loaded",
        "last_accessed": "2025-08-14T10:18:00.000Z",
        "access_count": 456,
        "load_time_ms": 1890,
        "memory_usage_mb": 68.7
      }
    ]
  },
  "download_cache": {
    "enabled": true,
    "configuration": {
      "size_limit_mb": 512.0,
      "max_files": 50,
      "ttl_hours": 72,
      "verify_checksums": true,
      "compress_files": true
    },
    "current_status": {
      "current_size_mb": 331.0,
      "entry_count": 8,
      "partial_downloads": 2,
      "completed_downloads": 6,
      "available_space_mb": 181.0
    },
    "download_entries": [
      {
        "file_name": "bert-large-v3.0.pth",
        "size_mb": 134.5,
        "status": "completed",
        "download_date": "2025-08-14T08:30:00.000Z",
        "checksum_verified": true,
        "compressed": true,
        "access_count": 3
      },
      {
        "file_name": "gpt2-medium-v2.1.pth",
        "size_mb": 78.9,
        "status": "partial",
        "download_progress": 67.3,
        "download_started": "2025-08-14T10:15:00.000Z",
        "estimated_completion": "2025-08-14T10:45:00.000Z"
      }
    ]
  },
  "statistics": {
    "cache_operations_per_hour": 4567,
    "cache_effectiveness_score": 85.2,
    "storage_efficiency_percentage": 73.4,
    "cache_maintenance_overhead_percentage": 2.1,
    "projected_cleanup_needed_hours": 12
  }
}
```

#### Cache Entries List (include_entries=true)
```json
{
  "cache_entries": {
    "inference_entries": [
      {
        "entry_id": "inf_12345",
        "model_name": "resnet50",
        "input_hash": "sha256:abc123...",
        "created_at": "2025-08-14T10:28:00.000Z",
        "last_accessed": "2025-08-14T10:29:45.000Z",
        "access_count": 5,
        "size_kb": 245.6,
        "ttl_remaining_hours": 23.5
      },
      {
        "entry_id": "inf_12346",
        "model_name": "bert-base",
        "input_hash": "sha256:def456...",
        "created_at": "2025-08-14T09:45:00.000Z",
        "last_accessed": "2025-08-14T10:12:00.000Z",
        "access_count": 12,
        "size_kb": 189.2,
        "ttl_remaining_hours": 22.8
      }
    ],
    "model_entries": [
      {
        "model_name": "resnet50",
        "version": "v1.0",
        "cache_path": "/cache/models/resnet50-v1.0.cached",
        "original_size_mb": 97.8,
        "cached_size_mb": 97.8,
        "compression_ratio": 1.0,
        "created_at": "2025-08-13T15:20:00.000Z",
        "last_accessed": "2025-08-14T10:25:00.000Z",
        "access_count": 1247
      }
    ],
    "download_entries": [
      {
        "download_id": "dl_67890",
        "original_url": "https://models.example.com/bert-large.pth",
        "local_path": "/cache/downloads/bert-large-v3.0.pth",
        "file_size_mb": 134.5,
        "downloaded_at": "2025-08-14T08:30:00.000Z",
        "checksum": "sha256:ghi789...",
        "verified": true,
        "reference_count": 3
      }
    ]
  }
}
```

#### Response Fields

##### Cache Info Object
| Field | Type | Description |
|-------|------|-------------|
| `total_cache_size_mb` | number | Total cache storage allocated |
| `used_cache_size_mb` | number | Currently used cache storage |
| `available_cache_size_mb` | number | Available cache storage |
| `cache_usage_percentage` | number | Percentage of cache storage used |
| `cache_directories` | array | Cache directory information |
| `cache_policies` | object | Cache configuration policies |
| `last_updated` | string | When cache info was last updated |

##### Inference Cache Object
| Field | Type | Description |
|-------|------|-------------|
| `enabled` | boolean | Whether inference caching is enabled |
| `configuration` | object | Cache configuration settings |
| `current_status` | object | Current cache status |
| `performance_metrics` | object | Cache performance statistics |
| `memory_usage` | object | Memory usage breakdown |
| `cleanup_history` | array | Recent cleanup operations |

##### Model Cache Object
| Field | Type | Description |
|-------|------|-------------|
| `enabled` | boolean | Whether model caching is enabled |
| `configuration` | object | Model cache settings |
| `current_status` | object | Current cache status |
| `models_cached` | array | List of cached models |

##### Download Cache Object
| Field | Type | Description |
|-------|------|-------------|
| `enabled` | boolean | Whether download caching is enabled |
| `configuration` | object | Download cache settings |
| `current_status` | object | Current cache status |
| `download_entries` | array | List of cached downloads |

##### Performance Metrics Object
| Field | Type | Description |
|-------|------|-------------|
| `hit_rate_percentage` | number | Cache hit rate |
| `miss_rate_percentage` | number | Cache miss rate |
| `total_requests` | integer | Total cache requests |
| `cache_hits` | integer | Successful cache hits |
| `cache_misses` | integer | Cache misses |
| `eviction_count` | integer | Number of cache evictions |
| `average_lookup_time_ms` | number | Average cache lookup time |

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success - Cache information returned |
| 500 | Internal server error |

## Examples

### Basic Cache Information

**Request:**
```bash
curl -X GET http://localhost:8000/cache-info
```

**Response:**
```json
{
  "cache_info": {
    "total_cache_size_mb": 2048.7,
    "used_cache_size_mb": 1456.3,
    "cache_usage_percentage": 71.2
  },
  "inference_cache": {
    "enabled": true,
    "current_size_mb": 687.2,
    "hit_rate_percentage": 78.5
  },
  "model_cache": {
    "enabled": true,
    "current_size_mb": 438.2,
    "entry_count": 5
  }
}
```

### Detailed Cache Information

**Request:**
```bash
curl -X GET "http://localhost:8000/cache-info?details_level=detailed&include_statistics=true"
```

### Specific Cache Type

**Request:**
```bash
curl -X GET "http://localhost:8000/cache-info?cache_type=inference&include_entries=true"
```

### Cache Entries List

**Request:**
```bash
curl -X GET "http://localhost:8000/cache-info?include_entries=true&details_level=summary"
```

### Python Cache Monitor

```python
import requests
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time

class CacheMonitor:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def get_cache_info(
        self,
        include_entries: bool = False,
        include_statistics: bool = True,
        cache_type: str = "all",
        details_level: str = "summary"
    ) -> Dict:
        """Get cache information"""
        params = {
            'include_entries': str(include_entries).lower(),
            'include_statistics': str(include_statistics).lower(),
            'cache_type': cache_type,
            'details_level': details_level
        }
        
        try:
            response = requests.get(f"{self.base_url}/cache-info", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def print_cache_summary(self, cache_data: Dict):
        """Print formatted cache summary"""
        if "error" in cache_data:
            print(f"❌ Error: {cache_data['error']}")
            return
        
        cache_info = cache_data.get('cache_info', {})
        
        print("="*80)
        print("CACHE INFORMATION SUMMARY")
        print("="*80)
        
        # Overall cache status
        total_mb = cache_info.get('total_cache_size_mb', 0)
        used_mb = cache_info.get('used_cache_size_mb', 0)
        available_mb = cache_info.get('available_cache_size_mb', 0)
        usage_pct = cache_info.get('cache_usage_percentage', 0)
        
        print(f"Total Cache Size: {total_mb:.1f} MB")
        print(f"Used: {used_mb:.1f} MB ({usage_pct:.1f}%)")
        print(f"Available: {available_mb:.1f} MB")
        
        # Create usage bar
        bar_length = 50
        used_length = int(bar_length * usage_pct / 100)
        bar = '█' * used_length + '░' * (bar_length - used_length)
        print(f"Usage: [{bar}] {usage_pct:.1f}%")
        
        print("\nCACHE BREAKDOWN:")
        print("-" * 40)
        
        # Inference cache
        inf_cache = cache_data.get('inference_cache', {})
        if inf_cache.get('enabled'):
            inf_size = inf_cache.get('current_size_mb', 0)
            inf_entries = inf_cache.get('entry_count', 0)
            hit_rate = inf_cache.get('hit_rate_percentage', 0)
            
            print(f"Inference Cache:")
            print(f"  Size: {inf_size:.1f} MB ({inf_entries:,} entries)")
            print(f"  Hit Rate: {hit_rate:.1f}%")
            print(f"  Status: {'✅ Enabled' if inf_cache.get('enabled') else '❌ Disabled'}")
        
        # Model cache  
        model_cache = cache_data.get('model_cache', {})
        if model_cache.get('enabled'):
            model_size = model_cache.get('current_size_mb', 0)
            model_count = model_cache.get('entry_count', 0)
            
            print(f"\nModel Cache:")
            print(f"  Size: {model_size:.1f} MB ({model_count} models)")
            print(f"  Status: {'✅ Enabled' if model_cache.get('enabled') else '❌ Disabled'}")
            
            # List cached models
            models_cached = model_cache.get('models_cached', [])
            if models_cached:
                print(f"  Cached Models:")
                for model in models_cached[:5]:  # Show top 5
                    name = model.get('model_name', 'Unknown')
                    version = model.get('version', 'N/A')
                    size = model.get('cache_size_mb', 0)
                    status = model.get('status', 'unknown')
                    print(f"    - {name} v{version}: {size:.1f} MB ({status})")
        
        # Download cache
        dl_cache = cache_data.get('download_cache', {})
        if dl_cache.get('enabled'):
            dl_size = dl_cache.get('current_size_mb', 0)
            dl_entries = dl_cache.get('entry_count', 0)
            
            print(f"\nDownload Cache:")
            print(f"  Size: {dl_size:.1f} MB ({dl_entries} files)")
            print(f"  Status: {'✅ Enabled' if dl_cache.get('enabled') else '❌ Disabled'}")
        
        # Cache statistics
        if 'statistics' in cache_data:
            stats = cache_data['statistics']
            print(f"\nCACHE STATISTICS:")
            print(f"  Operations/Hour: {stats.get('cache_operations_per_hour', 0):,}")
            print(f"  Effectiveness Score: {stats.get('cache_effectiveness_score', 0):.1f}")
            print(f"  Storage Efficiency: {stats.get('storage_efficiency_percentage', 0):.1f}%")
        
        print("="*80)
    
    def monitor_cache_performance(self, duration_minutes: int = 60, interval_seconds: int = 30):
        """Monitor cache performance over time"""
        print(f"🔍 Monitoring cache performance for {duration_minutes} minutes...")
        print("Time           | Hit Rate | Size (MB) | Entries | Operations/min")
        print("-" * 65)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        prev_requests = 0
        
        while time.time() < end_time:
            cache_data = self.get_cache_info(include_statistics=True)
            
            if "error" not in cache_data:
                inf_cache = cache_data.get('inference_cache', {})
                hit_rate = inf_cache.get('hit_rate_percentage', 0)
                size_mb = inf_cache.get('current_size_mb', 0)
                entries = inf_cache.get('entry_count', 0)
                
                perf = inf_cache.get('performance_metrics', {})
                total_requests = perf.get('total_requests', 0)
                
                # Calculate operations per minute
                ops_per_min = (total_requests - prev_requests) * (60 / interval_seconds)
                prev_requests = total_requests
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"{timestamp}      | {hit_rate:6.1f}% | {size_mb:7.1f} | {entries:7,} | {ops_per_min:10.0f}")
            
            time.sleep(interval_seconds)
    
    def analyze_cache_efficiency(self) -> Dict:
        """Analyze cache efficiency and provide recommendations"""
        cache_data = self.get_cache_info(
            include_entries=True, 
            include_statistics=True, 
            details_level="detailed"
        )
        
        if "error" in cache_data:
            return {"error": cache_data["error"]}
        
        analysis = {
            "overall_score": 0,
            "recommendations": [],
            "warnings": [],
            "metrics": {}
        }
        
        # Analyze inference cache
        inf_cache = cache_data.get('inference_cache', {})
        if inf_cache.get('enabled'):
            hit_rate = inf_cache.get('hit_rate_percentage', 0)
            usage_pct = cache_data.get('cache_info', {}).get('cache_usage_percentage', 0)
            
            # Hit rate analysis
            if hit_rate >= 80:
                analysis["recommendations"].append("✅ Excellent hit rate - cache is very effective")
            elif hit_rate >= 60:
                analysis["recommendations"].append("🟡 Good hit rate - consider tuning TTL settings")
            else:
                analysis["recommendations"].append("🔴 Low hit rate - review cache configuration")
            
            # Usage analysis
            if usage_pct >= 90:
                analysis["warnings"].append("⚠️ Cache nearly full - consider increasing size")
            elif usage_pct <= 30:
                analysis["recommendations"].append("💡 Cache underutilized - could reduce size")
            
            analysis["metrics"]["inference_hit_rate"] = hit_rate
            analysis["metrics"]["cache_usage_percentage"] = usage_pct
        
        # Analyze model cache
        model_cache = cache_data.get('model_cache', {})
        if model_cache.get('enabled'):
            models = model_cache.get('models_cached', [])
            loaded_models = [m for m in models if m.get('status') == 'loaded']
            
            if len(loaded_models) > 3:
                analysis["warnings"].append("⚠️ Many models loaded - high memory usage")
            
            # Check for unused models
            now = datetime.now()
            for model in models:
                last_accessed = model.get('last_accessed')
                if last_accessed:
                    last_access_time = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
                    hours_since_access = (now - last_access_time.replace(tzinfo=None)).total_seconds() / 3600
                    
                    if hours_since_access > 24:
                        analysis["recommendations"].append(
                            f"🧹 Model {model['model_name']} unused for {hours_since_access:.1f}h - consider unloading"
                        )
        
        # Calculate overall score
        score_components = []
        if inf_cache.get('enabled'):
            hit_rate = inf_cache.get('hit_rate_percentage', 0)
            score_components.append(min(hit_rate, 100))
        
        if score_components:
            analysis["overall_score"] = sum(score_components) / len(score_components)
        
        return analysis
    
    def cleanup_recommendations(self) -> List[str]:
        """Get cache cleanup recommendations"""
        cache_data = self.get_cache_info(include_entries=True, details_level="detailed")
        
        if "error" in cache_data:
            return [f"Error getting cache data: {cache_data['error']}"]
        
        recommendations = []
        
        # Check overall usage
        cache_info = cache_data.get('cache_info', {})
        usage_pct = cache_info.get('cache_usage_percentage', 0)
        
        if usage_pct > 85:
            recommendations.append("🚨 Urgent: Cache usage over 85% - immediate cleanup needed")
        elif usage_pct > 70:
            recommendations.append("⚠️ Warning: Cache usage over 70% - cleanup recommended")
        
        # Check inference cache
        inf_cache = cache_data.get('inference_cache', {})
        if inf_cache.get('enabled'):
            evictions = inf_cache.get('performance_metrics', {}).get('eviction_count', 0)
            if evictions > 100:
                recommendations.append(f"📊 High eviction count ({evictions}) - consider increasing cache size")
        
        # Check for old entries
        entries = cache_data.get('cache_entries', {})
        inf_entries = entries.get('inference_entries', [])
        
        old_entries = 0
        for entry in inf_entries:
            ttl_remaining = entry.get('ttl_remaining_hours', 24)
            if ttl_remaining < 1:  # Less than 1 hour remaining
                old_entries += 1
        
        if old_entries > 100:
            recommendations.append(f"🧹 {old_entries} inference entries expiring soon - cleanup will happen automatically")
        
        # Check download cache
        dl_cache = cache_data.get('download_cache', {})
        if dl_cache.get('enabled'):
            partial_downloads = dl_cache.get('partial_downloads', 0)
            if partial_downloads > 0:
                recommendations.append(f"📁 {partial_downloads} partial downloads - may need cleanup")
        
        if not recommendations:
            recommendations.append("✅ Cache is healthy - no immediate cleanup needed")
        
        return recommendations
    
    def export_cache_report(self, filename: Optional[str] = None) -> str:
        """Export detailed cache report"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cache_report_{timestamp}.json"
        
        cache_data = self.get_cache_info(
            include_entries=True,
            include_statistics=True,
            details_level="full"
        )
        
        # Add analysis
        cache_data["analysis"] = self.analyze_cache_efficiency()
        cache_data["recommendations"] = self.cleanup_recommendations()
        cache_data["report_generated"] = datetime.now().isoformat()
        
        try:
            with open(filename, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            print(f"✅ Cache report exported to {filename}")
            return filename
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return ""

# Usage Examples
monitor = CacheMonitor()

# Get and display basic cache information
cache_info = monitor.get_cache_info()
monitor.print_cache_summary(cache_info)

# Get detailed cache information
detailed_info = monitor.get_cache_info(
    include_entries=True,
    include_statistics=True,
    details_level="detailed"
)

# Analyze cache efficiency
analysis = monitor.analyze_cache_efficiency()
print(f"\nCache Efficiency Analysis:")
print(f"Overall Score: {analysis['overall_score']:.1f}/100")

if analysis['recommendations']:
    print("\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  {rec}")

if analysis['warnings']:
    print("\nWarnings:")
    for warning in analysis['warnings']:
        print(f"  {warning}")

# Get cleanup recommendations
cleanup_recs = monitor.cleanup_recommendations()
print(f"\nCleanup Recommendations:")
for rec in cleanup_recs:
    print(f"  {rec}")

# Monitor cache performance for 5 minutes
print(f"\nStarting cache performance monitoring...")
# monitor.monitor_cache_performance(duration_minutes=5, interval_seconds=30)

# Export detailed report
report_file = monitor.export_cache_report()
print(f"Report saved to: {report_file}")
```

### JavaScript Cache Dashboard

```javascript
class CacheDashboard {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.updateInterval = null;
  }
  
  async getCacheInfo(options = {}) {
    const params = new URLSearchParams({
      include_entries: options.includeEntries || false,
      include_statistics: options.includeStatistics !== false,
      cache_type: options.cacheType || 'all',
      details_level: options.detailsLevel || 'summary'
    });
    
    try {
      const response = await fetch(`${this.baseUrl}/cache-info?${params}`);
      return await response.json();
    } catch (error) {
      return { error: error.message };
    }
  }
  
  renderCacheDashboard(containerId) {
    const container = document.getElementById(containerId);
    
    container.innerHTML = `
      <div class="cache-dashboard">
        <div class="dashboard-header">
          <h2>Cache Dashboard</h2>
          <div class="controls">
            <button id="refresh-btn" class="btn-primary" onclick="dashboard.refreshData()">
              🔄 Refresh
            </button>
            <button id="auto-refresh-btn" class="btn-secondary" onclick="dashboard.toggleAutoRefresh()">
              ⏱️ Auto Refresh
            </button>
            <select id="detail-level" onchange="dashboard.changeDetailLevel(this.value)">
              <option value="summary">Summary</option>
              <option value="detailed">Detailed</option>
              <option value="full">Full Details</option>
            </select>
          </div>
        </div>
        
        <div id="cache-overview" class="cache-overview"></div>
        
        <div class="cache-sections">
          <div id="inference-cache" class="cache-section"></div>
          <div id="model-cache" class="cache-section"></div>
          <div id="download-cache" class="cache-section"></div>
        </div>
        
        <div id="cache-statistics" class="statistics-section"></div>
        <div id="recommendations" class="recommendations-section"></div>
      </div>
    `;
    
    this.loadCacheData();
  }
  
  async loadCacheData() {
    const detailLevel = document.getElementById('detail-level')?.value || 'summary';
    
    const data = await this.getCacheInfo({
      includeStatistics: true,
      detailsLevel: detailLevel
    });
    
    if (data.error) {
      this.renderError(data.error);
    } else {
      this.renderCacheOverview(data);
      this.renderInferenceCache(data.inference_cache);
      this.renderModelCache(data.model_cache);
      this.renderDownloadCache(data.download_cache);
      this.renderStatistics(data.statistics);
      this.renderRecommendations(data);
    }
  }
  
  renderCacheOverview(data) {
    const container = document.getElementById('cache-overview');
    const cacheInfo = data.cache_info || {};
    
    const totalMB = cacheInfo.total_cache_size_mb || 0;
    const usedMB = cacheInfo.used_cache_size_mb || 0;
    const availableMB = cacheInfo.available_cache_size_mb || 0;
    const usagePercent = cacheInfo.cache_usage_percentage || 0;
    
    const getUsageColor = (percent) => {
      if (percent < 50) return '#22c55e';
      if (percent < 75) return '#f59e0b';
      return '#ef4444';
    };
    
    container.innerHTML = `
      <div class="overview-card">
        <h3>Cache Overview</h3>
        <div class="overview-stats">
          <div class="stat-item">
            <span class="stat-value">${totalMB.toFixed(1)}</span>
            <span class="stat-label">Total MB</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">${usedMB.toFixed(1)}</span>
            <span class="stat-label">Used MB</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">${availableMB.toFixed(1)}</span>
            <span class="stat-label">Available MB</span>
          </div>
          <div class="stat-item">
            <span class="stat-value" style="color: ${getUsageColor(usagePercent)}">
              ${usagePercent.toFixed(1)}%
            </span>
            <span class="stat-label">Usage</span>
          </div>
        </div>
        
        <div class="usage-bar">
          <div class="usage-fill" style="width: ${usagePercent}%; background-color: ${getUsageColor(usagePercent)}"></div>
        </div>
        
        <div class="usage-text">
          Cache Usage: ${usedMB.toFixed(1)} MB / ${totalMB.toFixed(1)} MB (${usagePercent.toFixed(1)}%)
        </div>
      </div>
    `;
  }
  
  renderInferenceCache(cacheData) {
    const container = document.getElementById('inference-cache');
    
    if (!cacheData || !cacheData.enabled) {
      container.innerHTML = `
        <div class="cache-section-disabled">
          <h4>Inference Cache</h4>
          <p>Disabled</p>
        </div>
      `;
      return;
    }
    
    const hitRate = cacheData.hit_rate_percentage || 0;
    const currentSize = cacheData.current_size_mb || 0;
    const sizeLimit = cacheData.configuration?.size_limit_mb || cacheData.size_limit_mb || 0;
    const entryCount = cacheData.entry_count || 0;
    
    const hitRateColor = hitRate >= 70 ? '#22c55e' : hitRate >= 50 ? '#f59e0b' : '#ef4444';
    
    container.innerHTML = `
      <div class="cache-section-card">
        <h4>Inference Cache</h4>
        <div class="cache-metrics">
          <div class="metric-item">
            <span class="metric-label">Hit Rate:</span>
            <span class="metric-value" style="color: ${hitRateColor}">
              ${hitRate.toFixed(1)}%
            </span>
          </div>
          <div class="metric-item">
            <span class="metric-label">Size:</span>
            <span class="metric-value">${currentSize.toFixed(1)} / ${sizeLimit.toFixed(1)} MB</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">Entries:</span>
            <span class="metric-value">${entryCount.toLocaleString()}</span>
          </div>
        </div>
        
        ${cacheData.performance_metrics ? this.renderPerformanceMetrics(cacheData.performance_metrics) : ''}
      </div>
    `;
  }
  
  renderModelCache(cacheData) {
    const container = document.getElementById('model-cache');
    
    if (!cacheData || !cacheData.enabled) {
      container.innerHTML = `
        <div class="cache-section-disabled">
          <h4>Model Cache</h4>
          <p>Disabled</p>
        </div>
      `;
      return;
    }
    
    const currentSize = cacheData.current_size_mb || 0;
    const sizeLimit = cacheData.configuration?.size_limit_mb || cacheData.size_limit_mb || 0;
    const entryCount = cacheData.entry_count || 0;
    const models = cacheData.models_cached || [];
    
    container.innerHTML = `
      <div class="cache-section-card">
        <h4>Model Cache</h4>
        <div class="cache-metrics">
          <div class="metric-item">
            <span class="metric-label">Size:</span>
            <span class="metric-value">${currentSize.toFixed(1)} / ${sizeLimit.toFixed(1)} MB</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">Models:</span>
            <span class="metric-value">${entryCount}</span>
          </div>
        </div>
        
        ${models.length > 0 ? `
          <div class="model-list">
            <h5>Cached Models:</h5>
            <div class="model-items">
              ${models.slice(0, 5).map(model => `
                <div class="model-item">
                  <div class="model-name">${model.model_name} v${model.version}</div>
                  <div class="model-details">
                    <span class="model-size">${model.cache_size_mb?.toFixed(1) || 0} MB</span>
                    <span class="model-status status-${model.status}">${model.status}</span>
                  </div>
                </div>
              `).join('')}
              ${models.length > 5 ? `<div class="more-models">... and ${models.length - 5} more</div>` : ''}
            </div>
          </div>
        ` : ''}
      </div>
    `;
  }
  
  renderDownloadCache(cacheData) {
    const container = document.getElementById('download-cache');
    
    if (!cacheData || !cacheData.enabled) {
      container.innerHTML = `
        <div class="cache-section-disabled">
          <h4>Download Cache</h4>
          <p>Disabled</p>
        </div>
      `;
      return;
    }
    
    const currentSize = cacheData.current_size_mb || 0;
    const sizeLimit = cacheData.configuration?.size_limit_mb || cacheData.size_limit_mb || 0;
    const entryCount = cacheData.entry_count || 0;
    const partialDownloads = cacheData.partial_downloads || 0;
    
    container.innerHTML = `
      <div class="cache-section-card">
        <h4>Download Cache</h4>
        <div class="cache-metrics">
          <div class="metric-item">
            <span class="metric-label">Size:</span>
            <span class="metric-value">${currentSize.toFixed(1)} / ${sizeLimit.toFixed(1)} MB</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">Files:</span>
            <span class="metric-value">${entryCount}</span>
          </div>
          ${partialDownloads > 0 ? `
            <div class="metric-item">
              <span class="metric-label">Partial:</span>
              <span class="metric-value warning">${partialDownloads}</span>
            </div>
          ` : ''}
        </div>
      </div>
    `;
  }
  
  renderPerformanceMetrics(metrics) {
    return `
      <div class="performance-section">
        <h5>Performance Metrics</h5>
        <div class="perf-grid">
          <div class="perf-item">
            <span class="perf-label">Total Requests:</span>
            <span class="perf-value">${metrics.total_requests?.toLocaleString() || 0}</span>
          </div>
          <div class="perf-item">
            <span class="perf-label">Cache Hits:</span>
            <span class="perf-value">${metrics.cache_hits?.toLocaleString() || 0}</span>
          </div>
          <div class="perf-item">
            <span class="perf-label">Cache Misses:</span>
            <span class="perf-value">${metrics.cache_misses?.toLocaleString() || 0}</span>
          </div>
          <div class="perf-item">
            <span class="perf-label">Evictions:</span>
            <span class="perf-value">${metrics.eviction_count?.toLocaleString() || 0}</span>
          </div>
          <div class="perf-item">
            <span class="perf-label">Avg Lookup:</span>
            <span class="perf-value">${metrics.average_lookup_time_ms?.toFixed(2) || 0} ms</span>
          </div>
        </div>
      </div>
    `;
  }
  
  renderStatistics(stats) {
    const container = document.getElementById('cache-statistics');
    
    if (!stats) {
      container.innerHTML = '';
      return;
    }
    
    container.innerHTML = `
      <div class="statistics-card">
        <h4>Cache Statistics</h4>
        <div class="stats-grid">
          <div class="stat-item">
            <span class="stat-label">Operations/Hour:</span>
            <span class="stat-value">${stats.cache_operations_per_hour?.toLocaleString() || 0}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Effectiveness Score:</span>
            <span class="stat-value">${stats.cache_effectiveness_score?.toFixed(1) || 0}/100</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Storage Efficiency:</span>
            <span class="stat-value">${stats.storage_efficiency_percentage?.toFixed(1) || 0}%</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Maintenance Overhead:</span>
            <span class="stat-value">${stats.cache_maintenance_overhead_percentage?.toFixed(1) || 0}%</span>
          </div>
        </div>
      </div>
    `;
  }
  
  renderRecommendations(data) {
    const container = document.getElementById('recommendations');
    const recommendations = this.generateRecommendations(data);
    
    if (recommendations.length === 0) {
      container.innerHTML = `
        <div class="recommendations-card">
          <h4>Recommendations</h4>
          <div class="recommendation success">
            ✅ Cache is performing well - no immediate action needed
          </div>
        </div>
      `;
      return;
    }
    
    container.innerHTML = `
      <div class="recommendations-card">
        <h4>Recommendations</h4>
        <div class="recommendation-list">
          ${recommendations.map(rec => `
            <div class="recommendation ${rec.type}">
              ${rec.message}
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }
  
  generateRecommendations(data) {
    const recommendations = [];
    
    // Check cache usage
    const cacheInfo = data.cache_info || {};
    const usage = cacheInfo.cache_usage_percentage || 0;
    
    if (usage > 90) {
      recommendations.push({
        type: 'error',
        message: '🚨 Cache usage over 90% - immediate cleanup or expansion needed'
      });
    } else if (usage > 75) {
      recommendations.push({
        type: 'warning',
        message: '⚠️ Cache usage over 75% - consider cleanup or expansion'
      });
    } else if (usage < 30) {
      recommendations.push({
        type: 'info',
        message: '💡 Cache underutilized - consider reducing cache size'
      });
    }
    
    // Check inference cache performance
    const infCache = data.inference_cache || {};
    if (infCache.enabled) {
      const hitRate = infCache.hit_rate_percentage || 0;
      
      if (hitRate < 50) {
        recommendations.push({
          type: 'warning',
          message: '📊 Low cache hit rate - review TTL settings or cache size'
        });
      } else if (hitRate > 80) {
        recommendations.push({
          type: 'success',
          message: '✅ Excellent cache hit rate - cache is very effective'
        });
      }
      
      const evictions = infCache.performance_metrics?.eviction_count || 0;
      if (evictions > 100) {
        recommendations.push({
          type: 'info',
          message: `📈 High eviction count (${evictions}) - consider increasing cache size`
        });
      }
    }
    
    return recommendations;
  }
  
  renderError(error) {
    const container = document.getElementById('cache-overview');
    container.innerHTML = `
      <div class="error-card">
        <h3>Error Loading Cache Information</h3>
        <p>${error}</p>
        <button onclick="dashboard.loadCacheData()" class="btn-primary">Retry</button>
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
  
  changeDetailLevel(level) {
    this.loadCacheData();
  }
}

// Global functions for the dashboard
const dashboard = new CacheDashboard();

// Initialize dashboard
dashboard.renderCacheDashboard('cache-dashboard-container');

// Expose methods for button clicks
window.dashboard = dashboard;
```

## Cache Types

### Inference Cache
- **Purpose**: Cache inference results for repeated requests
- **Key Features**: LRU eviction, TTL-based expiration, compression support
- **Performance Impact**: Reduces inference latency for repeated inputs

### Model Cache
- **Purpose**: Cache loaded models in memory
- **Key Features**: Lazy loading, memory management, model versioning
- **Performance Impact**: Eliminates model loading time

### Download Cache
- **Purpose**: Cache downloaded model files
- **Key Features**: Resume capability, checksum verification, compression
- **Performance Impact**: Reduces download time for repeated model installs

## Cache Management

### Automatic Cleanup
- Scheduled cleanup based on TTL settings
- Size-based eviction when limits are reached
- Failed operation cleanup

### Manual Management
- Cache clearing via admin endpoints
- Selective cache invalidation
- Performance tuning options

## Related Endpoints

- [Models List](./models.md) - View cached models
- [Stats](./stats.md) - Overall system performance including cache
- [Health](./health.md) - Cache health status

## Best Practices

1. **Monitor Usage**: Regularly check cache usage percentages
2. **Tune TTL**: Adjust TTL settings based on usage patterns
3. **Size Optimization**: Balance cache size with available memory
4. **Hit Rate Monitoring**: Aim for >70% hit rates for effectiveness
5. **Regular Cleanup**: Implement automated cleanup procedures
6. **Performance Tracking**: Monitor cache impact on response times

# Autoscaler Endpoints

The autoscaler endpoints provide dynamic scaling functionality, zero-scaling capabilities, and performance metrics for automatic resource management.

## Overview

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|----------------|
| `/autoscaler/stats` | GET | Autoscaler statistics and metrics | None |
| `/autoscaler/health` | GET | Autoscaler health status | None |
| `/autoscaler/scale` | POST | Manual scaling operations | None |
| `/autoscaler/config` | GET | Autoscaler configuration | None |
| `/autoscaler/config` | POST | Update autoscaler settings | None |
| `/autoscaler/history` | GET | Scaling event history | None |

---

## Autoscaler Statistics

Get comprehensive autoscaler metrics and current status.

### Request
```http
GET /autoscaler/stats
```

### Response
```json
{
  "success": true,
  "timestamp": "2024-01-15T10:30:00Z",
  "autoscaler": {
    "enabled": true,
    "status": "active",
    "zero_scaling_enabled": true,
    "current_scale": 1,
    "target_scale": 1,
    "last_scale_event": "2024-01-15T10:25:00Z"
  },
  "models": {
    "total_models": 3,
    "loaded_models": 2,
    "unloaded_models": 1,
    "auto_managed_models": ["speecht5_tts", "bark_tts"],
    "always_loaded_models": ["example"]
  },
  "performance_metrics": {
    "requests_per_minute": 45,
    "average_response_time_ms": 850,
    "p95_response_time_ms": 1200,
    "p99_response_time_ms": 1800,
    "error_rate_percent": 0.2,
    "throughput_requests_per_second": 0.75
  },
  "resource_usage": {
    "cpu_usage_percent": 25.5,
    "memory_usage_percent": 42.3,
    "gpu_usage_percent": 35.8,
    "gpu_memory_usage_mb": 8192,
    "disk_usage_percent": 15.2
  },
  "scaling_metrics": {
    "scale_up_events_24h": 5,
    "scale_down_events_24h": 3,
    "zero_scale_events_24h": 2,
    "average_scale_up_time_seconds": 45,
    "average_scale_down_time_seconds": 15,
    "cold_start_time_seconds": 30
  },
  "queue_metrics": {
    "pending_requests": 0,
    "queue_length": 0,
    "max_queue_length": 100,
    "average_wait_time_ms": 0,
    "requests_dropped": 0
  },
  "efficiency_metrics": {
    "resource_utilization_score": 78.5,
    "cost_efficiency_score": 85.2,
    "availability_score": 99.8,
    "performance_stability": "excellent"
  },
  "thresholds": {
    "scale_up_threshold": {
      "cpu_percent": 70,
      "memory_percent": 80,
      "queue_length": 10,
      "response_time_ms": 2000
    },
    "scale_down_threshold": {
      "cpu_percent": 20,
      "memory_percent": 30,
      "idle_time_minutes": 5
    },
    "zero_scale_threshold": {
      "idle_time_minutes": 15,
      "no_requests_minutes": 10
    }
  }
}
```

### Key Metrics Explained

#### Performance Metrics
- **Requests per minute**: Current request rate
- **Response times**: P50, P95, P99 latencies
- **Error rate**: Percentage of failed requests
- **Throughput**: Successful requests per second

#### Resource Usage
- **CPU/Memory**: System resource utilization
- **GPU**: GPU utilization and memory consumption
- **Disk**: Storage utilization

#### Scaling Events
- **Scale events**: Recent scaling activity
- **Cold starts**: Time to initialize from zero
- **Efficiency scores**: Overall system performance

### Example
```bash
curl http://localhost:8000/autoscaler/stats
```

---

## Autoscaler Health Check

Check the health and status of the autoscaling system.

### Request
```http
GET /autoscaler/health
```

### Response
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "autoscaler": {
    "status": "operational",
    "enabled": true,
    "last_heartbeat": "2024-01-15T10:30:00Z",
    "uptime_seconds": 86400,
    "version": "2.1.0"
  },
  "components": {
    "metrics_collector": {
      "status": "healthy",
      "last_update": "2024-01-15T10:29:55Z"
    },
    "decision_engine": {
      "status": "healthy", 
      "decisions_per_minute": 12,
      "accuracy_percent": 96.5
    },
    "model_manager": {
      "status": "healthy",
      "managed_models": 3,
      "load_success_rate": 98.2
    },
    "resource_monitor": {
      "status": "healthy",
      "monitoring_interval_seconds": 5,
      "data_points_collected": 17280
    }
  },
  "configuration": {
    "zero_scaling_enabled": true,
    "min_replicas": 0,
    "max_replicas": 5,
    "target_cpu_utilization": 70,
    "scale_up_cooldown_seconds": 60,
    "scale_down_cooldown_seconds": 300
  },
  "recent_activity": {
    "last_scale_event": "2024-01-15T09:45:00Z",
    "last_scale_direction": "down", 
    "scale_events_1h": 2,
    "zero_scale_events_1h": 1
  },
  "system_checks": {
    "gpu_availability": true,
    "model_loading_capability": true,
    "memory_allocation": "sufficient",
    "network_connectivity": true,
    "disk_space": "sufficient"
  }
}
```

### Status Values
- **healthy**: All systems operational
- **degraded**: Some non-critical issues
- **unhealthy**: Critical issues detected
- **disabled**: Autoscaler is disabled

### Example
```bash
curl http://localhost:8000/autoscaler/health
```

---

## Manual Scaling Operations

Manually trigger scaling operations or override autoscaler decisions.

### Request
```http
POST /autoscaler/scale
Content-Type: application/json

{
  "action": "scale_up",
  "target_replicas": 2,
  "models": ["speecht5_tts", "bark_tts"],
  "override_cooldown": false,
  "reason": "Expected traffic spike"
}
```

### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | string | Yes | Scaling action (scale_up, scale_down, zero_scale, load_model, unload_model) |
| `target_replicas` | integer | No | Target number of replicas |
| `models` | array | No | Specific models to affect |
| `override_cooldown` | boolean | No | Override cooldown periods |
| `reason` | string | No | Reason for manual scaling |
| `force` | boolean | No | Force scaling even if conditions not met |

### Response
```json
{
  "success": true,
  "timestamp": "2024-01-15T10:30:00Z",
  "action": "scale_up",
  "previous_scale": 1,
  "target_scale": 2,
  "affected_models": ["speecht5_tts", "bark_tts"],
  "estimated_completion_time": "45 seconds",
  "scaling_id": "scale_20240115_103000",
  "status": "in_progress",
  "steps": [
    {
      "step": "resource_allocation",
      "status": "completed",
      "duration_ms": 150
    },
    {
      "step": "model_loading",
      "status": "in_progress",
      "estimated_duration_ms": 30000
    },
    {
      "step": "health_check",
      "status": "pending",
      "estimated_duration_ms": 5000
    }
  ],
  "resource_requirements": {
    "additional_memory_mb": 4096,
    "additional_gpu_memory_mb": 8192,
    "cpu_cores_required": 2
  }
}
```

### Scaling Actions

#### Scale Up
```bash
curl -X POST http://localhost:8000/autoscaler/scale \
  -H "Content-Type: application/json" \
  -d '{
    "action": "scale_up",
    "target_replicas": 3,
    "reason": "High traffic expected"
  }'
```

#### Scale Down
```bash
curl -X POST http://localhost:8000/autoscaler/scale \
  -H "Content-Type: application/json" \
  -d '{
    "action": "scale_down", 
    "target_replicas": 1,
    "reason": "Traffic decreased"
  }'
```

#### Zero Scale
```bash
curl -X POST http://localhost:8000/autoscaler/scale \
  -H "Content-Type: application/json" \
  -d '{
    "action": "zero_scale",
    "models": ["bark_tts"],
    "reason": "Maintenance period"
  }'
```

#### Load Specific Model
```bash
curl -X POST http://localhost:8000/autoscaler/scale \
  -H "Content-Type: application/json" \
  -d '{
    "action": "load_model",
    "models": ["speecht5_tts"],
    "reason": "Preload for expected usage"
  }'
```

#### Unload Model
```bash
curl -X POST http://localhost:8000/autoscaler/scale \
  -H "Content-Type: application/json" \
  -d '{
    "action": "unload_model",
    "models": ["unused_model"],
    "reason": "Free up memory"
  }'
```

---

## Autoscaler Configuration

Get or update autoscaler configuration settings.

### Get Configuration

#### Request
```http
GET /autoscaler/config
```

#### Response
```json
{
  "success": true,
  "configuration": {
    "enabled": true,
    "zero_scaling_enabled": true,
    "scaling_strategy": "reactive",
    "replicas": {
      "min_replicas": 0,
      "max_replicas": 5,
      "default_replicas": 1
    },
    "thresholds": {
      "cpu_threshold": 70,
      "memory_threshold": 80,
      "gpu_threshold": 85,
      "queue_length_threshold": 10,
      "response_time_threshold_ms": 2000
    },
    "cooldowns": {
      "scale_up_cooldown_seconds": 60,
      "scale_down_cooldown_seconds": 300,
      "zero_scale_cooldown_seconds": 900
    },
    "timing": {
      "metrics_collection_interval_seconds": 5,
      "decision_interval_seconds": 30,
      "health_check_interval_seconds": 60
    },
    "model_management": {
      "auto_unload_unused_models": true,
      "model_idle_timeout_minutes": 15,
      "preload_popular_models": true,
      "max_concurrent_model_loads": 2
    },
    "advanced": {
      "predictive_scaling": false,
      "custom_metrics_enabled": false,
      "external_metrics_webhook": null,
      "scaling_notifications_enabled": true
    }
  }
}
```

### Update Configuration

#### Request
```http
POST /autoscaler/config
Content-Type: application/json

{
  "enabled": true,
  "zero_scaling_enabled": true,
  "thresholds": {
    "cpu_threshold": 80,
    "memory_threshold": 85,
    "response_time_threshold_ms": 1500
  },
  "cooldowns": {
    "scale_up_cooldown_seconds": 45,
    "scale_down_cooldown_seconds": 240
  },
  "model_management": {
    "auto_unload_unused_models": true,
    "model_idle_timeout_minutes": 20
  }
}
```

#### Response
```json
{
  "success": true,
  "message": "Autoscaler configuration updated successfully",
  "updated_fields": [
    "thresholds.cpu_threshold",
    "thresholds.memory_threshold", 
    "thresholds.response_time_threshold_ms",
    "cooldowns.scale_up_cooldown_seconds",
    "cooldowns.scale_down_cooldown_seconds",
    "model_management.model_idle_timeout_minutes"
  ],
  "effective_timestamp": "2024-01-15T10:30:00Z",
  "restart_required": false
}
```

### Examples

#### Enable Aggressive Scaling
```bash
curl -X POST http://localhost:8000/autoscaler/config \
  -H "Content-Type: application/json" \
  -d '{
    "thresholds": {
      "cpu_threshold": 60,
      "memory_threshold": 70,
      "response_time_threshold_ms": 1000
    },
    "cooldowns": {
      "scale_up_cooldown_seconds": 30,
      "scale_down_cooldown_seconds": 180
    }
  }'
```

#### Enable Zero Scaling
```bash
curl -X POST http://localhost:8000/autoscaler/config \
  -H "Content-Type: application/json" \
  -d '{
    "zero_scaling_enabled": true,
    "model_management": {
      "auto_unload_unused_models": true,
      "model_idle_timeout_minutes": 10
    }
  }'
```

---

## Scaling History

Get historical scaling events and analysis.

### Request
```http
GET /autoscaler/history?hours=24&include_details=true
```

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hours` | integer | 24 | Hours of history to retrieve |
| `include_details` | boolean | false | Include detailed event information |
| `event_types` | array | null | Filter by event types |
| `limit` | integer | 100 | Maximum number of events |

### Response
```json
{
  "success": true,
  "period": {
    "start": "2024-01-14T10:30:00Z",
    "end": "2024-01-15T10:30:00Z",
    "hours": 24
  },
  "summary": {
    "total_events": 15,
    "scale_up_events": 8,
    "scale_down_events": 5,
    "zero_scale_events": 2,
    "average_scale_duration_seconds": 42,
    "efficiency_score": 87.5
  },
  "events": [
    {
      "id": "scale_20240115_102500",
      "timestamp": "2024-01-15T10:25:00Z",
      "type": "scale_up",
      "trigger": "high_cpu_usage",
      "previous_scale": 1,
      "target_scale": 2,
      "actual_scale": 2,
      "duration_seconds": 45,
      "success": true,
      "models_affected": ["speecht5_tts"],
      "trigger_metrics": {
        "cpu_usage_percent": 82,
        "memory_usage_percent": 75,
        "queue_length": 12,
        "response_time_ms": 1800
      },
      "outcome_metrics": {
        "cpu_usage_percent": 45,
        "memory_usage_percent": 65,
        "queue_length": 2,
        "response_time_ms": 650
      },
      "resource_impact": {
        "additional_memory_mb": 4096,
        "additional_gpu_memory_mb": 8192,
        "cost_increase_percent": 100
      }
    },
    {
      "id": "scale_20240115_094500",
      "timestamp": "2024-01-15T09:45:00Z", 
      "type": "scale_down",
      "trigger": "low_utilization",
      "previous_scale": 2,
      "target_scale": 1,
      "actual_scale": 1,
      "duration_seconds": 15,
      "success": true,
      "models_affected": ["bark_tts"],
      "trigger_metrics": {
        "cpu_usage_percent": 15,
        "memory_usage_percent": 25,
        "idle_time_minutes": 8
      },
      "resource_savings": {
        "memory_freed_mb": 4096,
        "gpu_memory_freed_mb": 8192,
        "cost_reduction_percent": 50
      }
    }
  ],
  "patterns": {
    "peak_hours": ["09:00-11:00", "14:00-16:00"],
    "low_traffic_hours": ["22:00-06:00"],
    "average_requests_per_hour": 120,
    "scaling_frequency": "moderate",
    "most_common_trigger": "cpu_usage"
  },
  "recommendations": [
    "Consider adjusting CPU threshold from 70% to 75%",
    "Peak traffic patterns are predictable - enable predictive scaling",
    "Zero-scaling working efficiently during low-traffic hours",
    "Model loading time could be optimized"
  ]
}
```

### Examples

#### Get Recent History
```bash
curl "http://localhost:8000/autoscaler/history?hours=12&include_details=true"
```

#### Filter by Event Type
```bash
curl "http://localhost:8000/autoscaler/history?event_types=[\"scale_up\",\"zero_scale\"]&limit=50"
```

---

## Zero-Scaling Feature

Zero-scaling allows the system to completely unload models when not in use, saving resources.

### How Zero-Scaling Works

1. **Idle Detection**: Monitor request patterns and resource usage
2. **Model Unloading**: Gracefully unload unused models from memory
3. **Cold Start**: Automatically reload models when new requests arrive
4. **Smart Preloading**: Preload frequently used models based on patterns

### Configuration
```json
{
  "zero_scaling_enabled": true,
  "model_idle_timeout_minutes": 15,
  "cold_start_timeout_seconds": 60,
  "preload_popular_models": true,
  "zero_scale_cooldown_seconds": 900
}
```

### Benefits
- **Resource Efficiency**: Up to 90% memory savings during idle periods
- **Cost Optimization**: Reduced cloud computing costs
- **Dynamic Scaling**: Automatic adaptation to traffic patterns
- **Multi-Model Support**: Independent scaling per model type

---

## Performance Optimization

### Scaling Strategies

#### Reactive Scaling (Default)
- Scale based on current metrics
- Fast response to load changes
- Conservative resource usage

#### Predictive Scaling
- Use historical patterns to predict load
- Preemptive scaling before traffic spikes
- Requires pattern learning period

#### Hybrid Scaling
- Combine reactive and predictive approaches
- Best of both strategies
- Most efficient resource utilization

### Optimization Tips

1. **Threshold Tuning**
   - Monitor false positive/negative rates
   - Adjust based on application requirements
   - Consider business impact of scaling delays

2. **Cooldown Configuration**
   - Prevent scaling oscillation
   - Balance responsiveness vs stability
   - Adjust based on model loading times

3. **Model Management**
   - Enable auto-unloading for memory efficiency
   - Use appropriate idle timeouts
   - Consider model loading times in decisions

4. **Resource Monitoring**
   - Track all relevant metrics (CPU, memory, GPU, latency)
   - Set appropriate thresholds for each metric
   - Use composite metrics for complex decisions

---

## Error Handling

### Common Autoscaler Errors

#### Scaling Failure
```json
{
  "success": false,
  "error": "scaling_failed",
  "message": "Failed to scale up: insufficient GPU memory",
  "scaling_id": "scale_20240115_103000",
  "requested_action": "scale_up",
  "target_scale": 3,
  "current_scale": 1,
  "failure_reason": "resource_constraint",
  "details": {
    "required_gpu_memory_mb": 16384,
    "available_gpu_memory_mb": 8192
  },
  "recommendations": [
    "Scale down other models first",
    "Use model with smaller memory footprint",
    "Add additional GPU capacity"
  ]
}
```

#### Model Loading Failure
```json
{
  "success": false,
  "error": "model_load_failed",
  "message": "Failed to load model 'large_model' during scale up",
  "model_name": "large_model",
  "failure_reason": "timeout",
  "timeout_seconds": 60,
  "suggestions": [
    "Increase model loading timeout",
    "Check model file integrity",
    "Verify sufficient system resources"
  ]
}
```

#### Configuration Error
```json
{
  "success": false,
  "error": "invalid_configuration",
  "message": "Invalid autoscaler configuration: min_replicas cannot be greater than max_replicas",
  "invalid_fields": [
    "min_replicas",
    "max_replicas"
  ],
  "provided_values": {
    "min_replicas": 3,
    "max_replicas": 2
  },
  "valid_ranges": {
    "min_replicas": "0-10",
    "max_replicas": "1-10"
  }
}
```

---

## Integration Examples

### Python Autoscaler Client
```python
import requests
import time
from typing import Dict, List, Optional

class AutoscalerClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def get_stats(self) -> Dict:
        """Get current autoscaler statistics"""
        response = requests.get(f"{self.base_url}/autoscaler/stats")
        return response.json()
    
    def get_health(self) -> Dict:
        """Check autoscaler health"""
        response = requests.get(f"{self.base_url}/autoscaler/health") 
        return response.json()
    
    def scale_up(self, target_replicas: int, reason: str = "") -> Dict:
        """Scale up to target replicas"""
        payload = {
            "action": "scale_up",
            "target_replicas": target_replicas,
            "reason": reason
        }
        response = requests.post(f"{self.base_url}/autoscaler/scale", json=payload)
        return response.json()
    
    def scale_down(self, target_replicas: int, reason: str = "") -> Dict:
        """Scale down to target replicas"""
        payload = {
            "action": "scale_down", 
            "target_replicas": target_replicas,
            "reason": reason
        }
        response = requests.post(f"{self.base_url}/autoscaler/scale", json=payload)
        return response.json()
    
    def zero_scale(self, models: Optional[List[str]] = None) -> Dict:
        """Zero-scale specific models or all"""
        payload = {"action": "zero_scale"}
        if models:
            payload["models"] = models
        response = requests.post(f"{self.base_url}/autoscaler/scale", json=payload)
        return response.json()
    
    def update_config(self, config: Dict) -> Dict:
        """Update autoscaler configuration"""
        response = requests.post(f"{self.base_url}/autoscaler/config", json=config)
        return response.json()
    
    def get_history(self, hours: int = 24, include_details: bool = True) -> Dict:
        """Get scaling history"""
        params = {"hours": hours, "include_details": include_details}
        response = requests.get(f"{self.base_url}/autoscaler/history", params=params)
        return response.json()
    
    def wait_for_scale_completion(self, scaling_id: str, timeout: int = 300) -> bool:
        """Wait for scaling operation to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            stats = self.get_stats()
            if stats.get("autoscaler", {}).get("status") == "active":
                return True
            time.sleep(5)
        
        return False
    
    def monitor_performance(self, duration: int = 60) -> List[Dict]:
        """Monitor autoscaler performance over time"""
        metrics = []
        end_time = time.time() + duration
        
        while time.time() < end_time:
            stats = self.get_stats()
            metrics.append({
                "timestamp": time.time(),
                "current_scale": stats["autoscaler"]["current_scale"],
                "cpu_usage": stats["resource_usage"]["cpu_usage_percent"],
                "memory_usage": stats["resource_usage"]["memory_usage_percent"],
                "response_time": stats["performance_metrics"]["average_response_time_ms"]
            })
            time.sleep(10)
        
        return metrics

# Usage example
autoscaler = AutoscalerClient()

# Check current status
stats = autoscaler.get_stats()
print(f"Current scale: {stats['autoscaler']['current_scale']}")
print(f"CPU usage: {stats['resource_usage']['cpu_usage_percent']}%")

# Scale up if needed
if stats["resource_usage"]["cpu_usage_percent"] > 80:
    result = autoscaler.scale_up(
        target_replicas=stats["autoscaler"]["current_scale"] + 1,
        reason="High CPU usage detected"
    )
    print(f"Scaling initiated: {result['scaling_id']}")

# Update configuration for more aggressive scaling
new_config = {
    "thresholds": {
        "cpu_threshold": 60,
        "response_time_threshold_ms": 1000
    },
    "cooldowns": {
        "scale_up_cooldown_seconds": 30
    }
}
autoscaler.update_config(new_config)
```

### Monitoring Dashboard Script
```bash
#!/bin/bash

# Autoscaler monitoring dashboard
BASE_URL="http://localhost:8000"

while true; do
    clear
    echo "=== Autoscaler Dashboard ==="
    echo "Timestamp: $(date)"
    echo
    
    # Get current stats
    STATS=$(curl -s "$BASE_URL/autoscaler/stats")
    
    # Current status
    echo "--- Current Status ---"
    echo "$STATS" | jq -r '"Scale: \(.autoscaler.current_scale)"'
    echo "$STATS" | jq -r '"Status: \(.autoscaler.status)"'
    echo "$STATS" | jq -r '"Zero Scaling: \(.autoscaler.zero_scaling_enabled)"'
    echo
    
    # Performance metrics
    echo "--- Performance ---"
    echo "$STATS" | jq -r '"Requests/min: \(.performance_metrics.requests_per_minute)"'
    echo "$STATS" | jq -r '"Avg Response: \(.performance_metrics.average_response_time_ms)ms"'
    echo "$STATS" | jq -r '"Error Rate: \(.performance_metrics.error_rate_percent)%"'
    echo
    
    # Resource usage
    echo "--- Resources ---"
    echo "$STATS" | jq -r '"CPU: \(.resource_usage.cpu_usage_percent)%"'
    echo "$STATS" | jq -r '"Memory: \(.resource_usage.memory_usage_percent)%"'
    echo "$STATS" | jq -r '"GPU: \(.resource_usage.gpu_usage_percent)%"'
    echo
    
    # Recent scaling events
    echo "--- Recent Events (24h) ---"
    echo "$STATS" | jq -r '"Scale Up: \(.scaling_metrics.scale_up_events_24h)"'
    echo "$STATS" | jq -r '"Scale Down: \(.scaling_metrics.scale_down_events_24h)"'
    echo "$STATS" | jq -r '"Zero Scale: \(.scaling_metrics.zero_scale_events_24h)"'
    echo
    
    sleep 10
done
```

### Load Testing with Autoscaler
```python
import concurrent.futures
import requests
import time
import matplotlib.pyplot as plt

def load_test_with_autoscaler():
    """Perform load test while monitoring autoscaler behavior"""
    
    base_url = "http://localhost:8000"
    autoscaler = AutoscalerClient(base_url)
    
    # Test configuration
    test_duration = 300  # 5 minutes
    ramp_up_time = 60    # 1 minute
    max_threads = 50
    
    # Metrics tracking
    metrics = {
        "timestamps": [],
        "response_times": [],
        "error_rates": [],
        "scale_levels": [],
        "cpu_usage": [],
        "memory_usage": []
    }
    
    def make_request():
        """Single request to the inference endpoint"""
        try:
            start_time = time.time()
            response = requests.post(f"{base_url}/predict", 
                json={"text": "Test inference request"}, timeout=30)
            end_time = time.time()
            
            return {
                "success": response.status_code == 200,
                "response_time": end_time - start_time,
                "status_code": response.status_code
            }
        except Exception as e:
            return {
                "success": False,
                "response_time": 30.0,  # timeout
                "error": str(e)
            }
    
    def monitor_autoscaler():
        """Monitor autoscaler metrics during test"""
        while True:
            try:
                stats = autoscaler.get_stats()
                metrics["timestamps"].append(time.time())
                metrics["scale_levels"].append(stats["autoscaler"]["current_scale"])
                metrics["cpu_usage"].append(stats["resource_usage"]["cpu_usage_percent"])
                metrics["memory_usage"].append(stats["resource_usage"]["memory_usage_percent"])
                time.sleep(5)
            except KeyboardInterrupt:
                break
    
    # Start monitoring in background
    import threading
    monitor_thread = threading.Thread(target=monitor_autoscaler)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    print("Starting load test...")
    print(f"Duration: {test_duration}s, Max threads: {max_threads}")
    
    start_time = time.time()
    
    # Gradually increase load
    for elapsed in range(test_duration):
        current_threads = min(max_threads, int((elapsed / ramp_up_time) * max_threads))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=current_threads) as executor:
            futures = [executor.submit(make_request) for _ in range(current_threads)]
            results = [f.result() for f in concurrent.futures.as_completed(futures, timeout=60)]
        
        # Calculate metrics for this interval
        successful_requests = sum(1 for r in results if r["success"])
        total_requests = len(results)
        avg_response_time = sum(r["response_time"] for r in results) / total_requests
        error_rate = ((total_requests - successful_requests) / total_requests) * 100
        
        metrics["response_times"].append(avg_response_time)
        metrics["error_rates"].append(error_rate)
        
        print(f"t={elapsed}s, threads={current_threads}, "
              f"avg_rt={avg_response_time:.2f}s, errors={error_rate:.1f}%")
        
        time.sleep(1)
    
    # Generate report
    print("\n=== Load Test Results ===")
    print(f"Test duration: {test_duration}s")
    print(f"Average response time: {sum(metrics['response_times'])/len(metrics['response_times']):.2f}s")
    print(f"Average error rate: {sum(metrics['error_rates'])/len(metrics['error_rates']):.2f}%")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Response times
    ax1.plot(metrics["response_times"])
    ax1.set_title("Response Times")
    ax1.set_ylabel("Seconds")
    
    # Error rates
    ax2.plot(metrics["error_rates"])
    ax2.set_title("Error Rates")
    ax2.set_ylabel("Percentage")
    
    # Scale levels
    if len(metrics["scale_levels"]) > 0:
        ax3.plot(metrics["scale_levels"])
        ax3.set_title("Autoscaler Scale Level")
        ax3.set_ylabel("Replicas")
    
    # Resource usage
    if len(metrics["cpu_usage"]) > 0:
        ax4.plot(metrics["cpu_usage"], label="CPU")
        ax4.plot(metrics["memory_usage"], label="Memory")
        ax4.set_title("Resource Usage")
        ax4.set_ylabel("Percentage")
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig("autoscaler_load_test_results.png")
    plt.show()

# Run the load test
if __name__ == "__main__":
    load_test_with_autoscaler()
```

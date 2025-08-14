# GET /health - Health Check

**URL**: `/health`  
**Method**: `GET`  
**Authentication**: Not required  
**Content-Type**: `application/json`

## Description

Provides comprehensive health status information for the API and inference engine. This endpoint is essential for monitoring, load balancing, and service discovery.

## Request

### URL Parameters
None

### Query Parameters
None

### Request Body
None (GET request)

## Response

### Healthy Response (200 OK)

```json
{
  "healthy": true,
  "checks": {
    "inference_engine": true,
    "model_loaded": true,
    "device_available": true,
    "memory_usage": "normal",
    "queue_status": "normal"
  },
  "timestamp": 1692013800.123,
  "engine_stats": {
    "requests_processed": 1234,
    "average_processing_time": 0.0156,
    "queue_length": 0,
    "active_workers": 2,
    "total_memory_mb": 2048.5,
    "available_memory_mb": 1536.7
  }
}
```

### Unhealthy Response (200 OK)

```json
{
  "healthy": false,
  "checks": {
    "inference_engine": false,
    "error": "Inference engine initialization failed"
  },
  "timestamp": 1692013800.123,
  "engine_stats": null
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `healthy` | boolean | Overall health status |
| `checks` | object | Detailed health check results |
| `timestamp` | float | Unix timestamp of health check |
| `engine_stats` | object\|null | Engine performance statistics |

#### Health Check Details

| Check | Description | Values |
|-------|-------------|---------|
| `inference_engine` | Engine availability | true/false |
| `model_loaded` | Model load status | true/false |
| `device_available` | Device (CPU/GPU) status | true/false |
| `memory_usage` | Memory utilization level | normal/high/critical |
| `queue_status` | Request queue status | normal/busy/overloaded |

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Health check completed (check `healthy` field) |
| 500 | Health check system failure |

## Examples

### Basic Health Check

**Request:**
```bash
curl -X GET http://localhost:8000/health
```

**Healthy Response:**
```json
{
  "healthy": true,
  "checks": {
    "inference_engine": true,
    "model_loaded": true,
    "device_available": true,
    "memory_usage": "normal",
    "queue_status": "normal"
  },
  "timestamp": 1692013800.123,
  "engine_stats": {
    "requests_processed": 1234,
    "average_processing_time": 0.0156,
    "queue_length": 0,
    "active_workers": 2,
    "total_memory_mb": 2048.5,
    "available_memory_mb": 1536.7
  }
}
```

### Python Health Monitor

```python
import requests
import time

def check_service_health():
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        data = response.json()
        
        print(f"Service Health: {'✅ Healthy' if data['healthy'] else '❌ Unhealthy'}")
        
        if data['healthy']:
            stats = data['engine_stats']
            print(f"Requests Processed: {stats['requests_processed']}")
            print(f"Average Processing Time: {stats['average_processing_time']:.3f}s")
            print(f"Queue Length: {stats['queue_length']}")
            print(f"Memory Usage: {stats['total_memory_mb']:.1f} MB")
        else:
            print(f"Health Issues: {data['checks']}")
            
        return data['healthy']
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Service Unavailable: {e}")
        return False

# Continuous monitoring
while True:
    healthy = check_service_health()
    if not healthy:
        print("⚠️ Service is unhealthy!")
    time.sleep(30)  # Check every 30 seconds
```

### JavaScript Health Dashboard

```javascript
async function checkHealth() {
  try {
    const response = await fetch('http://localhost:8000/health');
    const data = await response.json();
    
    updateHealthDisplay(data);
    
    return data.healthy;
  } catch (error) {
    console.error('Health check failed:', error);
    showServiceDown();
    return false;
  }
}

function updateHealthDisplay(healthData) {
  const statusElement = document.getElementById('service-status');
  const statsElement = document.getElementById('service-stats');
  
  if (healthData.healthy) {
    statusElement.innerHTML = '✅ Service Healthy';
    statusElement.className = 'status-healthy';
    
    const stats = healthData.engine_stats;
    statsElement.innerHTML = `
      <div>Requests: ${stats.requests_processed}</div>
      <div>Avg Time: ${stats.average_processing_time.toFixed(3)}s</div>
      <div>Queue: ${stats.queue_length}</div>
      <div>Memory: ${stats.total_memory_mb.toFixed(1)} MB</div>
    `;
  } else {
    statusElement.innerHTML = '❌ Service Unhealthy';
    statusElement.className = 'status-unhealthy';
    statsElement.innerHTML = 'Service diagnostics unavailable';
  }
}

// Check health every 10 seconds
setInterval(checkHealth, 10000);
checkHealth(); // Initial check
```

### Load Balancer Health Check

```bash
#!/bin/bash
# Simple health check script for load balancers

HEALTH_URL="http://localhost:8000/health"
TIMEOUT=5

response=$(curl -s -w "%{http_code}" --max-time $TIMEOUT "$HEALTH_URL")
http_code="${response: -3}"
body="${response%???}"

if [ "$http_code" -eq 200 ]; then
  healthy=$(echo "$body" | jq -r '.healthy')
  if [ "$healthy" = "true" ]; then
    echo "✅ Service healthy"
    exit 0
  else
    echo "❌ Service unhealthy"
    exit 1
  fi
else
  echo "❌ Service unavailable (HTTP $http_code)"
  exit 1
fi
```

## Health Check States

### Healthy State Indicators
- ✅ Inference engine running
- ✅ Model successfully loaded
- ✅ Device (CPU/GPU) accessible
- ✅ Memory usage under threshold
- ✅ Request queue processing normally

### Warning State Indicators
- ⚠️ High memory usage (>80%)
- ⚠️ Queue backlog building up
- ⚠️ Slower than normal processing times
- ⚠️ Device utilization near maximum

### Unhealthy State Indicators
- ❌ Inference engine not initialized
- ❌ Model loading failed
- ❌ Device unavailable or error
- ❌ Critical memory usage (>95%)
- ❌ Request queue overflow

## Monitoring Integration

### Prometheus Metrics
```yaml
# Example Prometheus scraping config
scrape_configs:
  - job_name: 'pytorch-inference-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/health'
    scrape_interval: 15s
```

### Docker Health Check
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

### Kubernetes Readiness Probe
```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

## Error Scenarios

### Engine Initialization Error
```json
{
  "healthy": false,
  "checks": {
    "inference_engine": false,
    "error": "Failed to initialize inference engine: CUDA device not available"
  },
  "timestamp": 1692013800.123,
  "engine_stats": null
}
```

### Memory Critical
```json
{
  "healthy": false,
  "checks": {
    "inference_engine": true,
    "model_loaded": true,
    "memory_usage": "critical",
    "available_memory_mb": 45.2
  },
  "timestamp": 1692013800.123,
  "engine_stats": {
    "total_memory_mb": 2048.0,
    "available_memory_mb": 45.2
  }
}
```

## Best Practices

1. **Regular Monitoring**: Check health every 10-30 seconds
2. **Timeout Settings**: Use 5-10 second timeouts
3. **Alerting**: Set up alerts for unhealthy states
4. **Load Balancing**: Remove unhealthy instances from rotation
5. **Graceful Degradation**: Have fallback strategies for unhealthy services
6. **Trend Analysis**: Monitor health trends over time

## Related Endpoints

- [Statistics](./stats.md) - Detailed performance metrics
- [Configuration](./config.md) - Service configuration
- [Root](./root.md) - Basic service information

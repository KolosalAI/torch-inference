# GET /stats - Statistics

**URL**: `/stats`  
**Method**: `GET`  
**Authentication**: Not required  
**Content-Type**: `application/json`

## Description

Returns detailed performance statistics and metrics for the inference engine. This endpoint provides insights into system performance, throughput, and resource utilization.

## Request

### URL Parameters
None

### Query Parameters
None

### Request Body
None (GET request)

## Response

### Success Response (200 OK)

```json
{
  "stats": {
    "requests_processed": 15420,
    "requests_failed": 23,
    "total_processing_time": 2341.567,
    "average_processing_time": 0.0152,
    "min_processing_time": 0.008,
    "max_processing_time": 0.234,
    "requests_per_second": 45.7,
    "queue_length": 0,
    "active_workers": 4,
    "total_workers": 4,
    "memory_usage_mb": 1024.5,
    "gpu_utilization_percent": 78.5,
    "uptime_seconds": 3600.0,
    "last_request_timestamp": 1692013800.123
  },
  "performance_report": {
    "throughput": {
      "current_rps": 45.7,
      "peak_rps": 89.2,
      "average_rps": 42.8
    },
    "latency": {
      "p50_ms": 12.5,
      "p90_ms": 18.7,
      "p95_ms": 22.4,
      "p99_ms": 45.6
    },
    "error_rate": {
      "current_error_rate": 0.0015,
      "total_error_rate": 0.0014
    },
    "resource_usage": {
      "cpu_percent": 45.2,
      "memory_percent": 32.1,
      "gpu_percent": 78.5
    }
  }
}
```

#### Response Fields

##### Stats Object
| Field | Type | Description |
|-------|------|-------------|
| `requests_processed` | integer | Total number of requests processed |
| `requests_failed` | integer | Total number of failed requests |
| `total_processing_time` | float | Cumulative processing time (seconds) |
| `average_processing_time` | float | Average processing time per request (seconds) |
| `min_processing_time` | float | Minimum processing time recorded (seconds) |
| `max_processing_time` | float | Maximum processing time recorded (seconds) |
| `requests_per_second` | float | Current requests per second |
| `queue_length` | integer | Current number of queued requests |
| `active_workers` | integer | Number of active worker threads |
| `total_workers` | integer | Total number of worker threads |
| `memory_usage_mb` | float | Current memory usage in MB |
| `gpu_utilization_percent` | float | GPU utilization percentage |
| `uptime_seconds` | float | Service uptime in seconds |
| `last_request_timestamp` | float | Timestamp of last processed request |

##### Performance Report Object
| Category | Field | Description |
|----------|-------|-------------|
| **Throughput** | `current_rps` | Current requests per second |
| | `peak_rps` | Peak requests per second |
| | `average_rps` | Average requests per second |
| **Latency** | `p50_ms` | 50th percentile latency (ms) |
| | `p90_ms` | 90th percentile latency (ms) |
| | `p95_ms` | 95th percentile latency (ms) |
| | `p99_ms` | 99th percentile latency (ms) |
| **Error Rate** | `current_error_rate` | Current error rate (0-1) |
| | `total_error_rate` | Total error rate (0-1) |
| **Resources** | `cpu_percent` | CPU utilization percentage |
| | `memory_percent` | Memory utilization percentage |
| | `gpu_percent` | GPU utilization percentage |

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success - Statistics returned |
| 503 | Service unavailable - Inference engine not available |
| 500 | Internal server error |

## Examples

### Basic Statistics Request

**Request:**
```bash
curl -X GET http://localhost:8000/stats
```

**Response:**
```json
{
  "stats": {
    "requests_processed": 15420,
    "requests_failed": 23,
    "total_processing_time": 2341.567,
    "average_processing_time": 0.0152,
    "requests_per_second": 45.7,
    "queue_length": 0,
    "active_workers": 4,
    "memory_usage_mb": 1024.5,
    "uptime_seconds": 3600.0
  },
  "performance_report": {
    "throughput": {
      "current_rps": 45.7,
      "peak_rps": 89.2
    },
    "latency": {
      "p50_ms": 12.5,
      "p90_ms": 18.7,
      "p95_ms": 22.4
    }
  }
}
```

### Python Performance Monitor

```python
import requests
import time
import json

class PerformanceMonitor:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.previous_stats = None
    
    def get_stats(self):
        try:
            response = requests.get(f"{self.base_url}/stats")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching stats: {e}")
            return None
    
    def print_performance_summary(self, stats_data):
        stats = stats_data['stats']
        perf = stats_data['performance_report']
        
        print("="*50)
        print("PYTORCH INFERENCE API - PERFORMANCE STATS")
        print("="*50)
        
        # Basic Stats
        print(f"Requests Processed: {stats['requests_processed']:,}")
        print(f"Success Rate: {(1 - stats['requests_failed']/max(stats['requests_processed'], 1))*100:.2f}%")
        print(f"Average Processing Time: {stats['average_processing_time']*1000:.2f} ms")
        print(f"Current RPS: {stats['requests_per_second']:.1f}")
        
        # Performance Metrics
        print(f"\nLatency Percentiles:")
        print(f"  P50: {perf['latency']['p50_ms']:.1f} ms")
        print(f"  P90: {perf['latency']['p90_ms']:.1f} ms")
        print(f"  P95: {perf['latency']['p95_ms']:.1f} ms")
        print(f"  P99: {perf['latency']['p99_ms']:.1f} ms")
        
        # Resource Usage
        print(f"\nResource Utilization:")
        print(f"  CPU: {perf['resource_usage']['cpu_percent']:.1f}%")
        print(f"  Memory: {perf['resource_usage']['memory_percent']:.1f}%")
        print(f"  GPU: {perf['resource_usage']['gpu_percent']:.1f}%")
        
        # Queue Status
        print(f"\nQueue Status:")
        print(f"  Current Queue Length: {stats['queue_length']}")
        print(f"  Active Workers: {stats['active_workers']}/{stats['total_workers']}")
        
        print("="*50)
    
    def monitor_continuous(self, interval=10):
        while True:
            stats = self.get_stats()
            if stats:
                self.print_performance_summary(stats)
                
                # Check for performance alerts
                self.check_alerts(stats)
            
            time.sleep(interval)
    
    def check_alerts(self, stats_data):
        perf = stats_data['performance_report']
        stats = stats_data['stats']
        
        alerts = []
        
        # High latency alert
        if perf['latency']['p95_ms'] > 100:
            alerts.append(f"⚠️ High P95 latency: {perf['latency']['p95_ms']:.1f}ms")
        
        # High error rate alert
        if perf['error_rate']['current_error_rate'] > 0.05:
            alerts.append(f"⚠️ High error rate: {perf['error_rate']['current_error_rate']*100:.2f}%")
        
        # Queue backlog alert
        if stats['queue_length'] > 10:
            alerts.append(f"⚠️ Queue backlog: {stats['queue_length']} requests")
        
        # Resource usage alerts
        if perf['resource_usage']['cpu_percent'] > 90:
            alerts.append(f"⚠️ High CPU usage: {perf['resource_usage']['cpu_percent']:.1f}%")
        
        if perf['resource_usage']['memory_percent'] > 90:
            alerts.append(f"⚠️ High memory usage: {perf['resource_usage']['memory_percent']:.1f}%")
        
        for alert in alerts:
            print(alert)

# Usage
monitor = PerformanceMonitor()
stats = monitor.get_stats()
if stats:
    monitor.print_performance_summary(stats)

# Continuous monitoring
# monitor.monitor_continuous(interval=30)
```

### JavaScript Real-time Dashboard

```javascript
class StatsMonitor {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.isMonitoring = false;
  }
  
  async fetchStats() {
    try {
      const response = await fetch(`${this.baseUrl}/stats`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching stats:', error);
      return null;
    }
  }
  
  updateDashboard(statsData) {
    const stats = statsData.stats;
    const perf = statsData.performance_report;
    
    // Update basic metrics
    document.getElementById('total-requests').textContent = 
      stats.requests_processed.toLocaleString();
    document.getElementById('success-rate').textContent = 
      `${((1 - stats.requests_failed / Math.max(stats.requests_processed, 1)) * 100).toFixed(2)}%`;
    document.getElementById('avg-time').textContent = 
      `${(stats.average_processing_time * 1000).toFixed(2)} ms`;
    document.getElementById('current-rps').textContent = 
      stats.requests_per_second.toFixed(1);
    
    // Update latency chart
    this.updateLatencyChart(perf.latency);
    
    // Update resource usage
    this.updateResourceGauges(perf.resource_usage);
    
    // Update queue status
    document.getElementById('queue-length').textContent = stats.queue_length;
    document.getElementById('active-workers').textContent = 
      `${stats.active_workers}/${stats.total_workers}`;
  }
  
  updateLatencyChart(latency) {
    // Example using Chart.js
    const ctx = document.getElementById('latency-chart').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['P50', 'P90', 'P95', 'P99'],
        datasets: [{
          label: 'Latency (ms)',
          data: [latency.p50_ms, latency.p90_ms, latency.p95_ms, latency.p99_ms],
          backgroundColor: ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Milliseconds'
            }
          }
        }
      }
    });
  }
  
  startMonitoring(interval = 5000) {
    this.isMonitoring = true;
    
    const monitor = async () => {
      if (!this.isMonitoring) return;
      
      const stats = await this.fetchStats();
      if (stats) {
        this.updateDashboard(stats);
      }
      
      setTimeout(monitor, interval);
    };
    
    monitor();
  }
  
  stopMonitoring() {
    this.isMonitoring = false;
  }
}

// Usage
const monitor = new StatsMonitor();
monitor.startMonitoring(5000); // Update every 5 seconds
```

## Error Handling

### Service Unavailable (503)
```json
{
  "detail": "Inference engine not available"
}
```

### Internal Server Error (500)
```json
{
  "detail": "Failed to retrieve statistics"
}
```

## Performance Metrics Interpretation

### Throughput Metrics
- **Current RPS**: Real-time requests per second
- **Peak RPS**: Highest sustained throughput achieved
- **Average RPS**: Mean throughput over service lifetime

### Latency Metrics
- **P50 (Median)**: 50% of requests complete faster
- **P90**: 90% of requests complete faster (good user experience)
- **P95**: 95% of requests complete faster (service level target)
- **P99**: 99% of requests complete faster (outlier detection)

### Resource Utilization
- **CPU**: Processor utilization (optimal: 60-80%)
- **Memory**: RAM usage (watch for memory leaks)
- **GPU**: Graphics processor utilization (model-dependent)

## Monitoring Best Practices

1. **Regular Polling**: Check stats every 5-30 seconds
2. **Alerting Thresholds**: Set alerts for P95 > 100ms, error rate > 5%
3. **Trend Analysis**: Monitor metrics over time for patterns
4. **Capacity Planning**: Use peak metrics for scaling decisions
5. **Performance Baselines**: Establish normal operating ranges

## Related Endpoints

- [Health Check](./health.md) - Service health status
- [Configuration](./config.md) - Current service configuration
- [Prediction](./predict.md) - Single prediction endpoint
- [Batch Prediction](./batch-predict.md) - Batch prediction endpoint

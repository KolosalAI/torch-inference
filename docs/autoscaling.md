# Autoscaling and Dynamic Model Loading

This document describes the advanced autoscaling and dynamic model loading features added to the PyTorch Inference Framework.

## Features Overview

### 🔄 Zero Scaling
- **Scale to Zero**: Automatically scale instances to zero when no requests are present
- **Cold Start Optimization**: Intelligent preloading and fast startup strategies  
- **Smart Preloading**: Preload frequently used models based on usage patterns
- **Resource Management**: Automatic cleanup of unused resources

### 📦 Dynamic Model Loading
- **On-Demand Loading**: Load models dynamically based on request patterns
- **Load Balancing**: Distribute requests across multiple model instances
- **Multi-Version Support**: Support multiple versions of the same model
- **Health Monitoring**: Continuous health checks with automatic failover

### 📊 Performance Monitoring
- **Real-time Metrics**: Comprehensive metrics collection and monitoring
- **Alerting System**: Configurable alerts based on performance thresholds
- **Historical Analysis**: Time-series data for performance analysis
- **Resource Tracking**: Monitor CPU, memory, and GPU utilization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Autoscaler                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   Zero Scaler   │    │   Dynamic Model Loader          │ │
│  │                 │    │                                 │ │
│  │ • Scale to zero │    │ • Multi-instance management     │ │
│  │ • Cold starts   │    │ • Load balancing                │ │
│  │ • Preloading    │    │ • Version control               │ │
│  │ • Cleanup       │    │ • Health monitoring             │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                   Metrics Collector                         │
│  • Performance tracking  • Alerting  • Historical data      │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Autoscaler Configuration

```python
from framework.autoscaling import AutoscalerConfig, ZeroScalingConfig, ModelLoaderConfig

config = AutoscalerConfig(
    enable_zero_scaling=True,
    enable_dynamic_loading=True,
    enable_metrics=True,
    
    # Zero scaling settings
    zero_scaling=ZeroScalingConfig(
        enabled=True,
        scale_to_zero_delay=300.0,  # 5 minutes
        max_loaded_models=5,
        preload_popular_models=True,
        popularity_threshold=10
    ),
    
    # Model loading settings
    model_loading=ModelLoaderConfig(
        max_instances_per_model=3,
        min_instances_per_model=1,
        enable_model_caching=True,
        prefetch_popular_models=True,
        load_timeout_seconds=300.0
    ),
    
    # Alert thresholds
    alert_thresholds={
        'memory_usage_percent': 85.0,
        'cpu_usage_percent': 80.0,
        'error_rate_percent': 5.0,
        'average_response_time_ms': 1000.0
    }
)
```

### Zero Scaling Configuration

```python
zero_config = ZeroScalingConfig(
    # Basic settings
    enabled=True,
    mode=ScalingMode.ZERO,
    cold_start_strategy=ColdStartStrategy.HYBRID,
    
    # Timing
    scale_to_zero_delay=300.0,      # Scale to zero after 5 minutes idle
    cold_start_timeout=30.0,        # Max cold start time
    warmup_timeout=60.0,            # Max warmup time
    
    # Resource limits
    max_loaded_models=5,            # Max models in memory
    model_ttl_seconds=1800.0,       # Model TTL (30 minutes)
    
    # Performance
    enable_predictive_scaling=True,
    learning_window_hours=24,
    prediction_horizon_minutes=30
)
```

### Model Loader Configuration

```python
loader_config = ModelLoaderConfig(
    # Loading strategy
    loading_strategy=ModelLoadingStrategy.LAZY,
    load_balancing_strategy=LoadBalancingStrategy.LEAST_CONNECTIONS,
    
    # Instance management
    max_instances_per_model=3,
    min_instances_per_model=1,
    default_instances_per_model=1,
    
    # Performance
    concurrent_loads=2,
    load_timeout_seconds=300.0,
    warmup_timeout_seconds=60.0,
    
    # Caching
    enable_model_caching=True,
    cache_size_gb=10.0,
    prefetch_popular_models=True,
    
    # Health monitoring
    health_check_interval_seconds=30.0,
    max_consecutive_failures=3,
    failure_cooldown_seconds=300.0
)
```

## Usage Examples

### Basic Autoscaler Setup

```python
from framework.autoscaling import Autoscaler, AutoscalerConfig
from framework.core.base_model import get_model_manager

# Create autoscaler
config = AutoscalerConfig()
model_manager = get_model_manager()
autoscaler = Autoscaler(config, model_manager)

# Start autoscaler
await autoscaler.start()

# Make predictions (automatically handles scaling)
result = await autoscaler.predict("my_model", input_data)

# Manual scaling
await autoscaler.scale_model("my_model", target_instances=3)

# Load/unload models
await autoscaler.load_model("new_model", version="v2")
await autoscaler.unload_model("old_model", version="v1")

# Get statistics
stats = autoscaler.get_stats()
health = autoscaler.get_health_status()

# Stop autoscaler
await autoscaler.stop()
```

### Zero Scaling with Context Manager

```python
from framework.autoscaling import ZeroScaler, ZeroScalingConfig

config = ZeroScalingConfig(scale_to_zero_delay=60.0)  # 1 minute
zero_scaler = ZeroScaler(config)

async with zero_scaler.scaler_context():
    # Make predictions - instances auto-created and removed
    for i in range(100):
        result = await zero_scaler.predict("model", f"input_{i}")
        await asyncio.sleep(0.1)
    
    # After 1 minute of inactivity, instances will be removed
```

### Dynamic Model Loading

```python
from framework.autoscaling import DynamicModelLoader, ModelLoaderConfig

config = ModelLoaderConfig(
    max_instances_per_model=5,
    load_balancing_strategy=LoadBalancingStrategy.LEAST_RESPONSE_TIME
)
loader = DynamicModelLoader(config)

await loader.start()

# Load multiple versions
await loader.load_model("my_model", "v1")
await loader.load_model("my_model", "v2")

# Make predictions - automatically load balanced
for i in range(50):
    result = await loader.predict("my_model", input_data, version="v1")

# Check loaded models
models = loader.list_loaded_models()
stats = loader.get_stats()

await loader.stop()
```

## API Endpoints

### Autoscaler Endpoints

```bash
# Get autoscaler statistics
GET /autoscaler/stats

# Get autoscaler health
GET /autoscaler/health

# Scale a model
POST /autoscaler/scale?model_name=example&target_instances=3

# Load a model
POST /autoscaler/load?model_name=example&version=v1

# Unload a model
DELETE /autoscaler/unload?model_name=example&version=v1

# Get detailed metrics
GET /autoscaler/metrics?window_seconds=3600
```

### Enhanced Prediction Endpoints

```bash
# Standard prediction (now with autoscaling)
POST /predict
{
    "inputs": [1, 2, 3],
    "priority": 5,
    "timeout": 30.0
}

# Batch prediction (now with load balancing)
POST /predict/batch
{
    "inputs": [[1, 2, 3], [4, 5, 6]],
    "priority": 1,
    "timeout": 60.0
}
```

## Monitoring and Metrics

### Key Metrics

- **Instance Metrics**: Active instances, loading instances, total instances
- **Request Metrics**: Request rate, response times, error rates  
- **Resource Metrics**: CPU, memory, GPU utilization
- **Scaling Metrics**: Scale up/down events, cold starts
- **Model Metrics**: Load times, cache hit rates, popularity

### Performance Dashboard

```python
# Get comprehensive metrics
async with aiohttp.ClientSession() as session:
    async with session.get("http://localhost:8000/autoscaler/metrics") as resp:
        metrics = await resp.json()
        
        print(f"Active instances: {metrics['active_instances']}")
        print(f"Request rate: {metrics['requests_per_second']}")
        print(f"Average response time: {metrics['average_response_time_ms']}ms")
        print(f"Error rate: {metrics['error_rate_percent']}%")
```

### Alerts Configuration

```python
config.alert_thresholds = {
    'memory_usage_percent': 85.0,      # Alert if memory > 85%
    'cpu_usage_percent': 80.0,         # Alert if CPU > 80%
    'error_rate_percent': 5.0,         # Alert if error rate > 5%
    'average_response_time_ms': 1000.0 # Alert if response time > 1s
}

# Add custom alert callback
def handle_alert(alert_key: str, message: str):
    print(f"ALERT: {alert_key} - {message}")
    # Send to Slack, email, etc.

autoscaler.add_alert_callback(handle_alert)
```

## Load Balancing Strategies

### Available Strategies

1. **Round Robin**: Distribute requests evenly across instances
2. **Least Connections**: Route to instance with fewest active requests
3. **Least Response Time**: Route to instance with lowest average response time
4. **Weighted Round Robin**: Route based on instance weights
5. **Consistent Hash**: Route based on input hash for sticky sessions

```python
# Configure load balancing strategy
config = ModelLoaderConfig(
    load_balancing_strategy=LoadBalancingStrategy.LEAST_RESPONSE_TIME
)
```

## Best Practices

### Configuration Tuning

1. **Scale-to-Zero Delay**: Set based on request patterns
   - High traffic: 300-600 seconds (5-10 minutes)  
   - Low traffic: 60-180 seconds (1-3 minutes)

2. **Instance Limits**: Balance resource usage vs. performance
   - Max instances: Based on available memory/GPU
   - Min instances: 1 for critical models, 0 for occasional models

3. **Health Checks**: Configure appropriate intervals
   - Production: 30-60 seconds
   - Development: 10-30 seconds

### Performance Optimization

1. **Model Preloading**: Enable for frequently used models
2. **Caching**: Configure appropriate cache sizes
3. **Warmup**: Set reasonable warmup timeouts
4. **Resource Limits**: Set memory and CPU limits per model

### Monitoring Setup

1. **Metrics Collection**: Enable comprehensive metrics
2. **Alerting**: Configure appropriate thresholds
3. **Historical Data**: Retain sufficient history for analysis
4. **Dashboard**: Create monitoring dashboards

## Troubleshooting

### Common Issues

1. **Models Not Scaling**: Check autoscaler configuration and logs
2. **High Response Times**: Review load balancing strategy and instance limits
3. **Memory Issues**: Adjust max_loaded_models and cache settings
4. **Failed Health Checks**: Review model loading and health check timeouts

### Debug Commands

```bash
# Check autoscaler status
curl http://localhost:8000/autoscaler/health

# Get detailed statistics  
curl http://localhost:8000/autoscaler/stats

# View metrics with time window
curl "http://localhost:8000/autoscaler/metrics?window_seconds=3600"

# Manual scaling for testing
curl -X POST "http://localhost:8000/autoscaler/scale?model_name=test&target_instances=2"
```

### Logging

Enable debug logging for troubleshooting:

```python
import logging
logging.getLogger("framework.autoscaling").setLevel(logging.DEBUG)
```

## Demo Script

Run the comprehensive demo to see all features in action:

```bash
python demo_autoscaling.py
```

The demo will show:
- Zero scaling behavior
- Dynamic model loading
- Load balancing across instances  
- Performance monitoring
- Metrics collection

## Integration with Existing Code

The autoscaling features are designed to be backward compatible. Existing prediction endpoints automatically benefit from autoscaling when enabled, with no code changes required.

To integrate autoscaling into existing applications:

1. Initialize the autoscaler instead of direct inference engines
2. Use autoscaler.predict() instead of engine.predict()  
3. Configure scaling parameters for your use case
4. Monitor performance and adjust settings as needed

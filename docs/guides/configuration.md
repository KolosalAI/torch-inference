# Configuration Guide

Comprehensive guide to configuring the PyTorch Inference Framework for optimal performance and functionality.

## 📋 Overview

The framework uses a hierarchical configuration system that supports:
- **Environment variables** - Runtime configuration
- **YAML files** - Structured configuration 
- **Python API** - Programmatic configuration
- **Auto-detection** - Intelligent defaults

## 🔧 Configuration Methods

### 1. Environment Variables (Recommended for Production)

```bash
# Device configuration
export DEVICE_TYPE=cuda                 # cuda, cpu, mps, auto
export DEVICE_ID=0                      # GPU device ID
export USE_FP16=true                    # Half precision
export USE_TENSORRT=false               # TensorRT optimization
export USE_TORCH_COMPILE=true           # PyTorch 2.0 compilation

# Batch configuration
export BATCH_SIZE=4                     # Default batch size
export MAX_BATCH_SIZE=16                # Maximum batch size
export ENABLE_DYNAMIC_BATCHING=true     # Dynamic batching

# Performance configuration
export WARMUP_ITERATIONS=10             # Model warmup
export ENABLE_PROFILING=false           # Performance profiling
export MAX_WORKERS=4                    # Async workers

# Server configuration
export HOST=0.0.0.0                     # Server host
export PORT=8000                        # Server port
export LOG_LEVEL=INFO                   # Logging level
export RELOAD=false                     # Auto-reload (dev only)

# Audio configuration
export ENABLE_TTS=true                  # TTS support
export ENABLE_STT=true                  # STT support
export DEFAULT_TTS_MODEL=speecht5_tts   # Default TTS model
export MAX_AUDIO_LENGTH=300             # Max audio length (seconds)

# Security configuration
export API_KEY=your_secret_key          # API authentication
export ALLOWED_ORIGINS=*                # CORS origins
export MAX_FILE_SIZE_MB=100             # Max upload size

# Cache configuration
export MODEL_CACHE_DIR=./models_cache   # Model cache directory
export CACHE_SIZE_GB=10                 # Cache size limit
export AUTO_CLEANUP=true                # Auto cache cleanup
```

### 2. YAML Configuration File

Create `config.yaml` in the project root:

```yaml
# config.yaml
environment: production  # development, staging, production

device:
  device_type: auto      # auto, cuda, cpu, mps
  device_id: 0
  use_fp16: true
  use_int8: false
  use_tensorrt: false
  use_torch_compile: true
  compile_mode: max-autotune  # default, reduce-overhead, max-autotune

batch:
  batch_size: 4
  max_batch_size: 16
  enable_dynamic_batching: true
  batch_timeout_ms: 50
  inflight_batching: true

performance:
  warmup_iterations: 10
  enable_profiling: false
  log_level: INFO
  max_workers: 4
  enable_async: true
  queue_size: 100

optimization:
  auto_optimize: true
  benchmark_all: false
  select_best: true
  aggressive_optimizations: false
  fallback_on_error: true
  
  # Specific optimizers
  optimizers:
    tensorrt:
      enabled: false
      precision: fp16        # fp32, fp16, int8
      max_batch_size: 32
      workspace_size_gb: 1
    
    onnx:
      enabled: true
      providers: [CUDAExecutionProvider, CPUExecutionProvider]
      optimization_level: all
    
    quantization:
      enabled: false
      dynamic: true
      static: false
      calibration_samples: 100
    
    jit:
      enabled: true
      script_mode: false
      trace_mode: true

memory:
  enable_memory_pooling: true
  pool_size_gb: 2
  cleanup_threshold: 0.8
  auto_gc: true
  cuda_graphs: true

server:
  host: 0.0.0.0
  port: 8000
  reload: false
  workers: 1
  log_level: INFO
  access_log: true
  
  # CORS settings
  cors:
    allow_origins: ["*"]
    allow_methods: ["*"]
    allow_headers: ["*"]
    allow_credentials: true
  
  # Rate limiting
  rate_limiting:
    enabled: false
    requests_per_minute: 60
    burst: 10

audio:
  enable_tts: true
  enable_stt: true
  
  tts:
    default_model: speecht5_tts
    default_voice: default
    default_language: en
    max_text_length: 5000
    auto_download_models: true
    supported_formats: [wav, mp3, flac]
    default_sample_rate: 16000
    
    models:
      speecht5_tts:
        model_id: microsoft/speecht5_tts
        vocoder: microsoft/speecht5_hifigan
        quality: high
      bark_tts:
        model_id: suno/bark
        quality: very_high
        supports_voice_cloning: true
  
  stt:
    default_model: whisper-base
    default_language: auto
    enable_timestamps: true
    beam_size: 5
    temperature: 0.0
    
    models:
      whisper-base:
        model_size: base
        quality: good
        speed: fast

security:
  enable_auth: false
  api_key: null
  allowed_ips: []
  max_file_size_mb: 100
  allowed_extensions: [.wav, .mp3, .flac, .m4a, .ogg]
  scan_uploads: false

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  handlers:
    console:
      enabled: true
      level: INFO
    
    file:
      enabled: true
      level: DEBUG
      filename: logs/framework.log
      max_size_mb: 50
      backup_count: 5
    
    api_requests:
      enabled: true
      filename: logs/api_requests.log
    
    errors:
      enabled: true
      filename: logs/errors.log
      level: ERROR

cache:
  model_cache_dir: ./models_cache
  cache_size_gb: 10
  auto_cleanup: true
  cleanup_threshold: 0.9
  ttl_hours: 168  # 1 week

monitoring:
  enable_metrics: true
  metrics_port: 9090
  enable_health_checks: true
  health_check_interval: 30
  
  alerts:
    enabled: false
    webhook_url: null
    email: null
    thresholds:
      memory_percent: 90
      cpu_percent: 80
      error_rate: 0.1
      response_time_ms: 1000

autoscaling:
  enable_zero_scaling: true
  enable_dynamic_loading: true
  
  zero_scaling:
    enabled: true
    scale_to_zero_delay: 300  # 5 minutes
    max_loaded_models: 5
    preload_popular_models: true
    enable_predictive_scaling: false
  
  model_loading:
    max_instances_per_model: 3
    min_instances_per_model: 0
    load_balancing_strategy: least_connections  # round_robin, least_connections, least_response_time
    enable_model_caching: true
    prefetch_popular_models: true
    health_check_interval: 60

development:
  enable_debug: false
  hot_reload: false
  enable_jupyter: false
  jupyter_port: 8888
  enable_tensorboard: false
  tensorboard_port: 6006
```

### 3. Python API Configuration

```python
# config_example.py
from framework.core.config import (
    InferenceConfig, DeviceConfig, BatchConfig, PerformanceConfig,
    ServerConfig, AudioConfig, SecurityConfig,
    DeviceType, OptimizationLevel
)

# Create device configuration
device_config = DeviceConfig(
    device_type=DeviceType.CUDA,
    device_id=0,
    use_fp16=True,
    use_tensorrt=False,
    use_torch_compile=True,
    compile_mode="max-autotune"
)

# Create batch configuration
batch_config = BatchConfig(
    batch_size=8,
    max_batch_size=32,
    enable_dynamic_batching=True,
    batch_timeout_ms=50
)

# Create performance configuration
performance_config = PerformanceConfig(
    warmup_iterations=15,
    enable_profiling=False,
    log_level="INFO",
    max_workers=6
)

# Create complete inference configuration
config = InferenceConfig(
    device=device_config,
    batch=batch_config,
    performance=performance_config
)

# Use with framework
from framework import TorchInferenceFramework
framework = TorchInferenceFramework(config)
```

### 4. Factory Configurations

Pre-configured setups for common use cases:

```python
from framework.core.config import ConfigFactory

# High-performance GPU configuration
gpu_config = ConfigFactory.create_gpu_optimized_config(
    device_id=0,
    aggressive_optimizations=True
)

# CPU-optimized configuration
cpu_config = ConfigFactory.create_cpu_optimized_config(
    num_threads=8,
    use_mkldnn=True
)

# Production server configuration
prod_config = ConfigFactory.create_production_config(
    enable_monitoring=True,
    enable_security=True,
    max_workers=8
)

# Development configuration
dev_config = ConfigFactory.create_development_config(
    enable_debug=True,
    hot_reload=True,
    enable_profiling=True
)

# Audio processing configuration
audio_config = ConfigFactory.create_audio_config(
    enable_tts=True,
    enable_stt=True,
    default_tts_model="speecht5_tts"
)
```

## 🎛️ Configuration Sections

### Device Configuration

```python
from framework.core.config import DeviceConfig, DeviceType

device_config = DeviceConfig(
    device_type=DeviceType.CUDA,    # AUTO, CUDA, CPU, MPS
    device_id=0,                    # GPU device ID
    use_fp16=True,                  # Half precision
    use_int8=False,                 # Int8 quantization
    use_tensorrt=False,             # TensorRT optimization
    use_torch_compile=True,         # PyTorch 2.0 compilation
    compile_mode="default"          # default, reduce-overhead, max-autotune
)
```

**Device Types:**
- `AUTO`: Automatically detect best device
- `CUDA`: NVIDIA GPU
- `CPU`: CPU processing
- `MPS`: Apple Metal Performance Shaders

**Compilation Modes:**
- `default`: Balanced performance and compilation time
- `reduce-overhead`: Faster compilation, good performance
- `max-autotune`: Slower compilation, maximum performance

### Batch Configuration

```python
from framework.core.config import BatchConfig

batch_config = BatchConfig(
    batch_size=4,                   # Default batch size
    max_batch_size=16,              # Maximum batch size
    enable_dynamic_batching=True,   # Dynamic batching
    batch_timeout_ms=50,            # Batch timeout
    inflight_batching=True          # In-flight batching
)
```

### Performance Configuration

```python
from framework.core.config import PerformanceConfig

performance_config = PerformanceConfig(
    warmup_iterations=10,           # Model warmup iterations
    enable_profiling=False,         # Performance profiling
    log_level="INFO",               # Logging level
    max_workers=4,                  # Async workers
    enable_async=True,              # Async processing
    queue_size=100                  # Request queue size
)
```

### Audio Configuration

```python
from framework.core.config import AudioConfig, TTSConfig, STTConfig

tts_config = TTSConfig(
    default_model="speecht5_tts",
    default_voice="default",
    default_language="en",
    max_text_length=5000,
    auto_download_models=True
)

stt_config = STTConfig(
    default_model="whisper-base",
    default_language="auto",
    enable_timestamps=True,
    beam_size=5
)

audio_config = AudioConfig(
    enable_tts=True,
    enable_stt=True,
    tts=tts_config,
    stt=stt_config
)
```

## 🚀 Optimization Configurations

### GPU Optimization

```yaml
# config.yaml - GPU optimized
device:
  device_type: cuda
  device_id: 0
  use_fp16: true
  use_tensorrt: true
  use_torch_compile: true
  compile_mode: max-autotune

optimization:
  auto_optimize: true
  aggressive_optimizations: true
  
  optimizers:
    tensorrt:
      enabled: true
      precision: fp16
      max_batch_size: 32
      workspace_size_gb: 2

memory:
  enable_memory_pooling: true
  pool_size_gb: 4
  cuda_graphs: true

batch:
  batch_size: 16
  max_batch_size: 64
  enable_dynamic_batching: true
```

### CPU Optimization

```yaml
# config.yaml - CPU optimized
device:
  device_type: cpu
  num_threads: 8
  use_mkldnn: true

optimization:
  optimizers:
    onnx:
      enabled: true
      providers: [CPUExecutionProvider]
      optimization_level: all
    
    quantization:
      enabled: true
      dynamic: true

batch:
  batch_size: 4
  max_batch_size: 16
```

### Memory-Constrained Environment

```yaml
# config.yaml - Low memory
device:
  use_fp16: true
  use_int8: true

batch:
  batch_size: 1
  max_batch_size: 4

memory:
  enable_memory_pooling: false
  auto_gc: true
  cleanup_threshold: 0.6

cache:
  cache_size_gb: 2
  auto_cleanup: true
```

## 🌍 Environment-Specific Configurations

### Development Environment

```yaml
# config.dev.yaml
environment: development

device:
  device_type: auto
  use_fp16: false  # Better debugging

performance:
  enable_profiling: true
  log_level: DEBUG

server:
  reload: true
  workers: 1

development:
  enable_debug: true
  hot_reload: true
  enable_jupyter: true
  enable_tensorboard: true

logging:
  level: DEBUG
  handlers:
    console:
      level: DEBUG
```

### Staging Environment

```yaml
# config.staging.yaml
environment: staging

device:
  device_type: cuda
  use_fp16: true

optimization:
  auto_optimize: true
  benchmark_all: true

server:
  workers: 2

monitoring:
  enable_metrics: true
  enable_health_checks: true

security:
  enable_auth: true
  allowed_ips: [10.0.0.0/8]
```

### Production Environment

```yaml
# config.prod.yaml
environment: production

device:
  device_type: cuda
  use_fp16: true
  use_tensorrt: true

optimization:
  auto_optimize: true
  aggressive_optimizations: true

server:
  workers: 4
  reload: false

security:
  enable_auth: true
  api_key: ${API_KEY}
  allowed_ips: ${ALLOWED_IPS}
  scan_uploads: true

monitoring:
  enable_metrics: true
  enable_health_checks: true
  alerts:
    enabled: true
    webhook_url: ${ALERT_WEBHOOK}

logging:
  level: INFO
  handlers:
    console:
      enabled: false
    file:
      enabled: true
```

## 🔒 Security Configuration

### API Authentication

```yaml
security:
  enable_auth: true
  api_key: "your-secret-api-key"
  
  # JWT authentication (future)
  jwt:
    secret_key: "jwt-secret"
    algorithm: HS256
    expire_minutes: 60

  # Rate limiting
  rate_limiting:
    enabled: true
    requests_per_minute: 100
    burst: 20
    
  # IP restrictions
  allowed_ips:
    - 192.168.1.0/24
    - 10.0.0.0/8
    
  # File upload security
  max_file_size_mb: 50
  allowed_extensions: [.wav, .mp3, .flac]
  scan_uploads: true
```

### CORS Configuration

```yaml
server:
  cors:
    allow_origins:
      - "https://yourapp.com"
      - "https://staging.yourapp.com"
    allow_methods: [GET, POST, PUT, DELETE]
    allow_headers: [Authorization, Content-Type]
    allow_credentials: true
    max_age: 86400
```

## 📊 Monitoring Configuration

### Metrics and Alerting

```yaml
monitoring:
  enable_metrics: true
  metrics_port: 9090
  metrics_path: /metrics
  
  # Prometheus configuration
  prometheus:
    enabled: true
    push_gateway: "http://prometheus:9091"
    job_name: "torch-inference"
    
  # Health checks
  enable_health_checks: true
  health_check_interval: 30
  health_check_timeout: 10
  
  # Alerting
  alerts:
    enabled: true
    channels:
      slack:
        webhook_url: "https://hooks.slack.com/..."
        channel: "#alerts"
      email:
        smtp_server: "smtp.gmail.com"
        smtp_port: 587
        username: "alerts@yourcompany.com"
        password: "${EMAIL_PASSWORD}"
        recipients: ["admin@yourcompany.com"]
    
    thresholds:
      memory_percent: 85
      cpu_percent: 80
      disk_percent: 90
      error_rate: 0.05
      response_time_p95_ms: 1000
      queue_depth: 50
```

## 🔄 Configuration Management

### Loading Configuration

```python
# Multiple configuration sources
from framework.core.config_manager import ConfigManager

# Load with priority: env vars > config file > defaults
config_manager = ConfigManager()

# Load specific config file
config_manager.load_config("config.prod.yaml")

# Override with environment variables
config_manager.load_env_vars()

# Get final configuration
config = config_manager.get_inference_config()
```

### Configuration Validation

```python
# Validate configuration
from framework.core.config import validate_config

try:
    validate_config(config)
    print("✅ Configuration valid")
except ValueError as e:
    print(f"❌ Configuration error: {e}")
```

### Dynamic Configuration Updates

```python
# Update configuration at runtime
config_manager.update_config({
    "batch": {"batch_size": 8},
    "device": {"use_fp16": True}
})

# Apply to running framework
framework.update_config(config_manager.get_inference_config())
```

## 🛠️ Configuration Templates

### High-Performance Template

```yaml
# config.high-performance.yaml
device:
  device_type: cuda
  use_fp16: true
  use_tensorrt: true
  use_torch_compile: true
  compile_mode: max-autotune

batch:
  batch_size: 16
  max_batch_size: 64
  enable_dynamic_batching: true

optimization:
  auto_optimize: true
  aggressive_optimizations: true
  
memory:
  enable_memory_pooling: true
  pool_size_gb: 4
  cuda_graphs: true
```

### Low-Latency Template

```yaml
# config.low-latency.yaml
device:
  device_type: cuda
  use_fp16: true
  use_torch_compile: true
  compile_mode: reduce-overhead

batch:
  batch_size: 1
  max_batch_size: 4
  batch_timeout_ms: 10

optimization:
  optimizers:
    jit:
      enabled: true
      
memory:
  cuda_graphs: true
```

### High-Throughput Template

```yaml
# config.high-throughput.yaml
batch:
  batch_size: 32
  max_batch_size: 128
  enable_dynamic_batching: true
  batch_timeout_ms: 100

performance:
  max_workers: 8
  queue_size: 500

server:
  workers: 4
```

## 📝 Configuration Best Practices

### 1. Environment-Specific Configs

```bash
# Use different configs per environment
config.dev.yaml     # Development
config.staging.yaml # Staging  
config.prod.yaml    # Production

# Load with environment variable
export CONFIG_FILE=config.${ENVIRONMENT}.yaml
```

### 2. Secrets Management

```yaml
# Use environment variables for secrets
security:
  api_key: ${API_KEY}
  jwt_secret: ${JWT_SECRET}

database:
  password: ${DB_PASSWORD}
  
monitoring:
  alerts:
    slack:
      webhook_url: ${SLACK_WEBHOOK}
```

### 3. Resource Limits

```yaml
# Set appropriate resource limits
memory:
  pool_size_gb: 2  # Don't exceed available memory

batch:
  max_batch_size: 16  # Based on GPU memory

cache:
  cache_size_gb: 5  # Leave room for system
```

### 4. Graceful Fallbacks

```yaml
optimization:
  fallback_on_error: true
  
  # Fallback chain: TensorRT -> ONNX -> JIT -> Standard
  optimizers:
    tensorrt:
      enabled: true
      fallback_to: onnx
    onnx:
      enabled: true
      fallback_to: jit
    jit:
      enabled: true
      fallback_to: standard
```

## 🧪 Testing Configuration

### Configuration Testing

```python
# test_config.py
import pytest
from framework.core.config import InferenceConfig, DeviceConfig, DeviceType

def test_config_creation():
    """Test configuration creation."""
    config = InferenceConfig()
    assert config.device.device_type == DeviceType.AUTO
    
def test_config_validation():
    """Test configuration validation."""
    config = InferenceConfig()
    config.batch.batch_size = -1  # Invalid
    
    with pytest.raises(ValueError):
        validate_config(config)

def test_config_from_yaml():
    """Test loading from YAML."""
    config = ConfigManager.from_yaml("test_config.yaml")
    assert config.device.use_fp16 == True
```

### Environment Testing

```bash
# Test different configurations
export CONFIG_TEST=true

# Test GPU config
export DEVICE_TYPE=cuda
python test_gpu_config.py

# Test CPU config  
export DEVICE_TYPE=cpu
python test_cpu_config.py
```

## 🔍 Configuration Debugging

### Debug Configuration

```python
# Debug configuration issues
from framework.core.config_manager import ConfigManager

config_manager = ConfigManager()
config_manager.set_debug(True)

# Print loaded configuration
print(config_manager.dump_config())

# Check configuration sources
print("Config sources:", config_manager.get_config_sources())

# Validate configuration
errors = config_manager.validate()
if errors:
    print("Configuration errors:", errors)
```

### Configuration Logging

```yaml
logging:
  level: DEBUG
  loggers:
    framework.core.config:
      level: DEBUG
      handlers: [console]
      
    framework.core.config_manager:
      level: DEBUG
      handlers: [file]
```

---

This configuration guide provides comprehensive coverage of all configuration options and best practices. For specific use cases, refer to the [optimization guide](optimization.md) and [production deployment guide](../tutorials/production-api.md).

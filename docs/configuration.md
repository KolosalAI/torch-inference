# ⚙️ Configuration Management

This guide covers the comprehensive configuration system for the PyTorch Inference Framework, supporting multiple configuration sources with clear precedence rules.

## 🔧 Configuration Overview

The framework supports multiple configuration sources:

1. **Environment Variables** (.env file) - Highest priority
2. **YAML Configuration** (config.yaml) - Environment-specific overrides  
3. **Default Values** - Built-in fallbacks

## 📁 Configuration Files

### Environment Variables (.env)
Primary source for environment-specific settings:

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Device Configuration  
DEVICE=cuda
USE_FP16=true

# Batch Configuration
BATCH_SIZE=8
MAX_BATCH_SIZE=32

# Performance Configuration
WARMUP_ITERATIONS=10
ENABLE_TENSORRT=true

# Security Configuration
MAX_FILE_SIZE_MB=100
VALIDATE_INPUTS=true
```

### YAML Configuration (config.yaml)
Structured configuration with environment overrides:

```yaml
# Base configuration
server:
  host: "0.0.0.0"
  port: 8000
  log_level: "INFO"
  workers: 1

device:
  type: "auto"
  use_fp16: false
  memory_fraction: 0.8

batch:
  batch_size: 4
  max_batch_size: 16
  adaptive_batching: true

optimization:
  enable_tensorrt: false
  enable_quantization: false
  enable_jit: true

# Environment-specific overrides
environments:
  development:
    server:
      reload: true
      log_level: "DEBUG"
    device:
      type: "cpu"
    optimization:
      enable_profiling: true
  
  staging:
    server:
      workers: 2
    device:
      use_fp16: true
    batch:
      batch_size: 8
  
  production:
    server:
      workers: 4
      log_level: "WARNING"
    device:
      use_fp16: true
      memory_fraction: 0.9
    batch:
      batch_size: 16
      max_batch_size: 64
    optimization:
      enable_tensorrt: true
      enable_quantization: true
```

## 🏗️ Configuration Architecture

### Configuration Precedence

Values are resolved in the following order (highest to lowest priority):

1. **Environment Variables** (from `.env` file or system)
2. **YAML Environment Overrides** (environment-specific section)
3. **YAML Base Configuration** (main configuration)
4. **Default Values** (hardcoded fallbacks)

### Using Configuration Manager

```python
from framework.core.config_manager import get_config_manager

# Get the global configuration manager
config_manager = get_config_manager()

# Get typed configuration objects
server_config = config_manager.get_server_config()
inference_config = config_manager.get_inference_config()

# Get individual values with fallbacks
batch_size = config_manager.get(
    'BATCH_SIZE', 
    default=4, 
    config_path='batch.batch_size'
)

# Environment-specific configuration
prod_config = ConfigManager(environment='production')
```

## 🔧 Configuration Categories

### Server Configuration

| Environment Variable | YAML Path | Default | Description |
|---------------------|-----------|---------|-------------|
| `HOST` | `server.host` | `"0.0.0.0"` | Server host address |
| `PORT` | `server.port` | `8000` | Server port number |
| `LOG_LEVEL` | `server.log_level` | `"INFO"` | Logging level |
| `RELOAD` | `server.reload` | `false` | Enable auto-reload |
| `WORKERS` | `server.workers` | `1` | Number of workers |

### Device Configuration

| Environment Variable | YAML Path | Default | Description |
|---------------------|-----------|---------|-------------|
| `DEVICE` | `device.type` | `"auto"` | Device type (auto/cpu/cuda/mps) |
| `DEVICE_ID` | `device.id` | `0` | GPU device ID |
| `USE_FP16` | `device.use_fp16` | `false` | Enable half precision |
| `MEMORY_FRACTION` | `device.memory_fraction` | `0.8` | GPU memory fraction |
| `USE_TORCH_COMPILE` | `device.use_torch_compile` | `false` | Enable torch.compile |

### Batch Configuration

| Environment Variable | YAML Path | Default | Description |
|---------------------|-----------|---------|-------------|
| `BATCH_SIZE` | `batch.batch_size` | `4` | Default batch size |
| `MIN_BATCH_SIZE` | `batch.min_batch_size` | `1` | Minimum batch size |
| `MAX_BATCH_SIZE` | `batch.max_batch_size` | `16` | Maximum batch size |
| `ADAPTIVE_BATCHING` | `batch.adaptive_batching` | `false` | Enable adaptive batching |
| `BATCH_TIMEOUT` | `batch.timeout_seconds` | `0.1` | Batch timeout |

### Optimization Configuration

| Environment Variable | YAML Path | Default | Description |
|---------------------|-----------|---------|-------------|
| `ENABLE_TENSORRT` | `optimization.enable_tensorrt` | `false` | Enable TensorRT |
| `ENABLE_QUANTIZATION` | `optimization.enable_quantization` | `false` | Enable quantization |
| `ENABLE_JIT` | `optimization.enable_jit` | `true` | Enable JIT compilation |
| `ENABLE_ONNX` | `optimization.enable_onnx` | `false` | Enable ONNX Runtime |
| `OPTIMIZATION_LEVEL` | `optimization.level` | `"balanced"` | Optimization level |

### Performance Configuration

| Environment Variable | YAML Path | Default | Description |
|---------------------|-----------|---------|-------------|
| `WARMUP_ITERATIONS` | `performance.warmup_iterations` | `5` | Model warmup iterations |
| `MAX_WORKERS` | `performance.max_workers` | `4` | Maximum worker threads |
| `ENABLE_PROFILING` | `performance.enable_profiling` | `false` | Enable profiling |
| `ENABLE_METRICS` | `performance.enable_metrics` | `true` | Enable metrics collection |

### Security Configuration

| Environment Variable | YAML Path | Default | Description |
|---------------------|-----------|---------|-------------|
| `MAX_FILE_SIZE_MB` | `security.max_file_size_mb` | `50` | Maximum file size |
| `ALLOWED_EXTENSIONS` | `security.allowed_extensions` | `[".jpg",".png"]` | Allowed file extensions |
| `VALIDATE_INPUTS` | `security.validate_inputs` | `true` | Enable input validation |
| `SANITIZE_OUTPUTS` | `security.sanitize_outputs` | `true` | Enable output sanitization |

## 🏢 Enterprise Configuration

Enable enterprise features:

```bash
ENTERPRISE_ENABLED=true
```

### Authentication Configuration

```yaml
enterprise:
  auth:
    enabled: true
    secret_key: "${JWT_SECRET_KEY}"
    algorithm: "HS256"
    expire_minutes: 30
    oauth2:
      client_id: "${OAUTH2_CLIENT_ID}"
      client_secret: "${OAUTH2_CLIENT_SECRET}"
```

### Security Configuration

```yaml
enterprise:
  security:
    enable_encryption_at_rest: true
    rate_limit_requests_per_minute: 100
    enable_audit_logging: true
    allowed_ips: ["192.168.1.0/24"]
```

### Integration Configuration

```yaml
enterprise:
  integration:
    database:
      url: "${DATABASE_URL}"
      pool_size: 10
    cache:
      url: "${CACHE_URL}"
      ttl_seconds: 3600
    message_broker:
      url: "${MESSAGE_BROKER_URL}"
      exchange: "inference"
```

## 🌍 Environment-Specific Configurations

### Development Environment
```bash
ENVIRONMENT=development
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=DEBUG
RELOAD=true
DEVICE=cpu
ENABLE_PROFILING=true
VALIDATE_INPUTS=false  # Relaxed validation
```

### Staging Environment
```bash
ENVIRONMENT=staging
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
DEVICE=cuda
USE_FP16=true
BATCH_SIZE=8
MAX_BATCH_SIZE=16
```

### Production Environment
```bash
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=WARNING
WORKERS=4
DEVICE=cuda
USE_FP16=true
BATCH_SIZE=16
MAX_BATCH_SIZE=64
ENTERPRISE_ENABLED=true
ENABLE_TENSORRT=true
ENABLE_QUANTIZATION=true
```

## 📖 Configuration Examples

### Basic Configuration Usage

```python
from framework.core.config_manager import get_config_manager
from framework import TorchInferenceFramework

# Initialize with configuration
config_manager = get_config_manager()
inference_config = config_manager.get_inference_config()

# Create framework with configuration
framework = TorchInferenceFramework(config=inference_config)
framework.load_model("path/to/model.pt")

# Use configuration values
batch_size = config_manager.get('BATCH_SIZE', default=4)
results = framework.predict_batch(inputs, batch_size=batch_size)
```

### Dynamic Configuration

```python
from framework.core.config_manager import ConfigManager

# Create configuration for specific environment
prod_manager = ConfigManager(environment='production')
dev_manager = ConfigManager(environment='development')

# Compare configurations
prod_config = prod_manager.get_inference_config()
dev_config = dev_manager.get_inference_config()

print(f"Production batch size: {prod_config.batch.batch_size}")
print(f"Development batch size: {dev_config.batch.batch_size}")
```

### Configuration Validation

```python
from framework.core.config_manager import get_config_manager

config_manager = get_config_manager()

# Validate configuration
if config_manager.validate_configuration():
    print("✅ Configuration is valid")
else:
    print("❌ Configuration has errors")
    for error in config_manager.get_validation_errors():
        print(f"  - {error}")
```

## 🔍 Configuration Debugging

### View Current Configuration

```python
from framework.core.config_manager import get_config_manager

config_manager = get_config_manager()

# Print all configuration values
config_manager.print_configuration()

# Get configuration as dictionary
config_dict = config_manager.to_dict()
print(json.dumps(config_dict, indent=2))

# Check configuration sources
sources = config_manager.get_configuration_sources()
for key, source in sources.items():
    print(f"{key}: {source}")
```

### API Endpoints

The framework provides configuration inspection endpoints:

- `GET /config` - View current configuration
- `GET /config/sources` - View configuration sources
- `GET /config/validate` - Validate configuration

### Environment Information

```bash
# View current environment
curl http://localhost:8000/

# View configuration
curl http://localhost:8000/config

# Validate configuration
curl http://localhost:8000/config/validate
```

## 📝 Configuration Testing

### Test Configuration Example

```python
# examples/config_example.py
from framework.core.config_manager import ConfigManager

def test_configuration_environments():
    """Test different environment configurations"""
    
    environments = ['development', 'staging', 'production']
    
    for env in environments:
        print(f"\n=== {env.upper()} ENVIRONMENT ===")
        
        config_manager = ConfigManager(environment=env)
        
        # Server configuration
        server_config = config_manager.get_server_config()
        print(f"Host: {server_config.host}")
        print(f"Port: {server_config.port}")
        print(f"Workers: {server_config.workers}")
        print(f"Log Level: {server_config.log_level}")
        
        # Device configuration
        device_config = config_manager.get_inference_config().device
        print(f"Device: {device_config.device_type}")
        print(f"FP16: {device_config.use_fp16}")
        
        # Batch configuration
        batch_config = config_manager.get_inference_config().batch
        print(f"Batch Size: {batch_config.batch_size}")
        print(f"Max Batch Size: {batch_config.max_batch_size}")

if __name__ == "__main__":
    test_configuration_environments()
```

Run the test:
```bash
uv run python examples/config_example.py
```

## 🚨 Configuration Best Practices

### Security
1. **Store secrets in .env**: Never commit sensitive values
2. **Use environment-specific configs**: Separate dev/staging/prod
3. **Validate inputs**: Always validate configuration values
4. **Audit configuration**: Log configuration changes

### Performance
1. **Cache configuration**: Avoid repeated parsing
2. **Use appropriate defaults**: Set sensible fallback values
3. **Profile configuration impact**: Monitor performance effects
4. **Optimize for your use case**: Tune batch sizes and workers

### Maintainability
1. **Document configuration**: Explain all options
2. **Use type hints**: Ensure type safety
3. **Version configuration**: Track configuration changes
4. **Test configuration**: Validate all environments

### Deployment
1. **Environment parity**: Keep environments consistent
2. **Configuration management**: Use proper config management tools
3. **Monitoring**: Track configuration-related issues
4. **Rollback capability**: Plan for configuration rollbacks

## 🔗 Related Documentation

- **[Installation Guide](installation.md)** - Setting up configuration files
- **[Deployment Guide](deployment.md)** - Production configuration
- **[Security Guide](security.md)** - Secure configuration practices
- **[API Reference](api.md)** - Configuration API documentation

---

*Need help with configuration? Check the [Troubleshooting Guide](troubleshooting.md) or [open an issue](https://github.com/Evintkoo/torch-inference/issues).*

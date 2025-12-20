# Configuration Guide

Complete configuration reference for Torch Inference Server.

## Configuration Files

### Priority Order

1. **Default values** - Built-in defaults
2. **config.toml** - Main configuration file
3. **Environment variables** - Override file settings
4. **Command-line arguments** - Highest priority

### Configuration File Locations

```bash
# Development
./config.toml

# Production
/etc/torch-inference/config.toml
~/.config/torch-inference/config.toml
./config.toml
```

## Complete Configuration Reference

### Server Configuration

```toml
[server]
# Network binding
host = "0.0.0.0"          # Bind address (0.0.0.0 = all interfaces)
port = 8000               # HTTP port

# Worker configuration
workers = 8               # Number of HTTP workers (default: num_cpus)

# Logging
log_level = "info"        # debug, info, warn, error

# Timeouts
request_timeout_seconds = 60
keep_alive_seconds = 75

# CORS (Cross-Origin Resource Sharing)
cors_enabled = true
cors_origins = ["*"]      # ["https://app.example.com"] in production
cors_methods = ["GET", "POST", "PUT", "DELETE"]
cors_headers = ["Authorization", "Content-Type"]
```

**Environment Variables**:
```bash
export SERVER_HOST=0.0.0.0
export SERVER_PORT=8000
export SERVER_WORKERS=16
export LOG_LEVEL=debug
```

### Device Configuration

```toml
[device]
# Device selection
device_type = "auto"      # auto, cuda, cpu, mps, metal

# GPU device IDs (for multi-GPU)
device_id = 0            # Primary device
device_ids = [0, 1]      # Multi-GPU (optional)

# Precision
use_fp16 = true          # Half-precision (2x faster on compatible GPUs)
use_bf16 = false         # BFloat16 (A100, H100)

# Backend-specific optimizations
use_tensorrt = false     # TensorRT (NVIDIA only)
use_torch_compile = true # PyTorch 2.0+ compilation
use_xla = false          # XLA compilation (experimental)

# Metal/Apple Silicon
metal_use_mlx = false               # Apple MLX framework
metal_cache_shaders = true          # Cache Metal shaders
metal_optimize_for_apple_silicon = true

# PyTorch threading
num_threads = 8          # Intra-op parallelism (recommend: num_cpus)
num_interop_threads = 1  # Inter-op parallelism (1 for serving)

# cuDNN settings (NVIDIA)
cudnn_benchmark = true   # Auto-select fastest algorithms
cudnn_deterministic = false

# Mixed precision
enable_autocast = true   # Automatic Mixed Precision (AMP)

# Warmup
torch_warmup_iterations = 5
```

**Auto-Detection**:
- `device_type = "auto"` detects CUDA → Metal → CPU
- Automatically configures `device_ids` for multi-GPU

**Performance Tips**:
```toml
# NVIDIA GPU
[device]
device_type = "cuda"
use_fp16 = true
cudnn_benchmark = true
use_torch_compile = true

# Apple Silicon
[device]
device_type = "mps"
use_fp16 = false         # Can be unstable on Metal
metal_optimize_for_apple_silicon = true

# CPU-only (Intel/AMD)
[device]
device_type = "cpu"
num_threads = 16         # Set to num_cpus
use_torch_compile = false
```

### Batch Configuration

```toml
[batch]
# Basic batching
batch_size = 1           # Default batch size
max_batch_size = 32      # Maximum batch size
min_batch_size = 1       # Minimum before processing

# Dynamic batching
enable_dynamic_batching = true
batch_timeout_ms = 50    # Wait time before processing partial batch

# Adaptive batching (adjusts timeout based on load)
adaptive_batch_timeout = true
# Adaptive timeouts:
# - 0-2 items:  100ms
# - 3-5 items:  50ms
# - 6-10 items: 25ms
# - 11+ items:  12.5ms

# Priority-based batching
enable_priority_batching = true
```

**Tuning Guide**:
```toml
# High throughput (batch processing)
max_batch_size = 64
adaptive_batch_timeout = true
min_batch_size = 8

# Low latency (real-time API)
max_batch_size = 8
batch_timeout_ms = 10
min_batch_size = 1

# Balanced
max_batch_size = 32
adaptive_batch_timeout = true
min_batch_size = 4
```

### Performance Configuration

```toml
[performance]
# === Caching ===
enable_caching = true
cache_size_mb = 2048     # LRU cache size
cache_ttl_seconds = 3600 # Time-to-live

# === Batching ===
enable_request_batching = true
adaptive_batch_timeout = true

# === Inflight Batching ===
enable_inflight_batching = false
max_inflight_batches = 4

# === Worker Pool ===
enable_worker_pool = true
min_workers = 2
max_workers = 16
enable_auto_scaling = true
enable_zero_scaling = false
worker_idle_timeout_seconds = 300

# === Tensor Pooling ===
enable_tensor_pooling = true
max_pooled_tensors = 500
tensor_pool_cleanup_interval_seconds = 60

# === Model Loading ===
enable_async_model_loading = true
preload_models_on_startup = false

# === Network ===
enable_result_compression = true
compression_level = 6    # 1=fast, 9=best, 6=balanced
compression_threshold_kb = 1

# === Advanced ===
enable_model_quantization = false
quantization_bits = 8    # 4, 8, or 16
enable_cuda_graphs = false

# === Profiling ===
enable_profiling = false
warmup_iterations = 3
```

**Memory Formula**:
```
Cache Memory = request_rate * avg_response_kb * ttl / 1024
Example: 100 req/s * 10 KB * 3600s / 1024 = ~3500 MB
```

### Authentication Configuration

```toml
[auth]
enabled = true
jwt_secret = "your-secret-key-change-in-production"
jwt_algorithm = "HS256"  # HS256, HS384, HS512, RS256

# Token expiration
access_token_expire_minutes = 60
refresh_token_expire_days = 7

# Password requirements
min_password_length = 8
require_uppercase = true
require_lowercase = true
require_digit = true
require_special_char = false
```

**Generate Secure Secret**:
```bash
openssl rand -base64 64
```

### Model Configuration

```toml
[models]
# Model directory
cache_dir = "models"     # Relative to working directory
# cache_dir = "/var/lib/torch-inference/models"  # Absolute path

# Model management
auto_load = ["resnet50", "yolo"]  # Load on startup
max_loaded_models = 5    # LRU eviction
model_timeout_seconds = 300

# Download settings
download_timeout_seconds = 600
download_retries = 3
verify_checksums = true

# Model pooling (multiple instances per model)
enable_model_pooling = true
models_per_pool = 2
```

### Guard Configuration

```toml
[guard]
enable_guards = true

# Resource limits
max_memory_mb = 8192     # Maximum memory usage
max_gpu_memory_mb = 16384
max_requests_per_second = 1000
max_queue_depth = 500

# Quality thresholds
min_cache_hit_rate = 60.0     # Percentage
max_error_rate = 5.0          # Percentage
max_avg_latency_ms = 100.0

# Circuit breaker
enable_circuit_breaker = true
failure_threshold = 5         # Failures before opening
timeout_seconds = 30          # Time before half-open
success_threshold = 2         # Successes to close

# Auto-mitigation
enable_auto_mitigation = true
mitigation_cooldown_seconds = 60

# Actions on threshold breach
on_memory_exceeded = "reject"     # reject, queue, scale
on_rate_exceeded = "throttle"     # reject, throttle
on_error_rate_high = "alert"      # alert, circuit_break
```

### Resilience Configuration

```toml
[resilience]
# Circuit breaker
circuit_breaker_enabled = true
failure_threshold = 5
success_threshold = 2
timeout_seconds = 30

# Bulkhead
bulkhead_enabled = true
max_concurrent_requests = 100
max_wait_duration_ms = 5000

# Retry policy
retry_enabled = true
max_retries = 3
initial_backoff_ms = 100
max_backoff_ms = 5000
backoff_multiplier = 2.0
jitter_factor = 0.1

# Timeout
default_timeout_seconds = 60
inference_timeout_seconds = 30
```

### Monitoring Configuration

```toml
[monitoring]
enable_metrics = true
enable_logging = true
enable_tracing = false

# Metrics
metrics_interval_seconds = 60
retain_metrics_hours = 24

# Health checks
health_check_interval_seconds = 30

# Prometheus
prometheus_enabled = true
prometheus_port = 9090

# OpenTelemetry (optional)
otlp_endpoint = "http://localhost:4317"
otlp_enabled = false
```

### Telemetry Configuration

```toml
[telemetry]
# Logging
log_format = "text"      # text, json
log_file = "logs/server.log"
log_rotation = "daily"   # hourly, daily, size
log_max_size_mb = 100
log_max_files = 7

# Structured logging
enable_structured_logs = true
include_timestamp = true
include_correlation_id = true
include_request_details = true

# Tracing
tracing_enabled = false
tracing_sample_rate = 0.1
```

### Security Configuration

```toml
[security]
# Input validation
max_request_size_mb = 10
max_image_size_mb = 5
max_audio_duration_seconds = 300
allowed_image_formats = ["jpg", "jpeg", "png", "webp"]
allowed_audio_formats = ["wav", "mp3", "flac"]

# Sanitization
enable_input_sanitization = true
enable_output_sanitization = false

# Rate limiting
rate_limit_per_ip = 1000          # Requests per minute
rate_limit_per_user = 10000       # Requests per minute
rate_limit_burst = 100            # Burst allowance

# Headers
enable_security_headers = true
enable_cors = true
enable_hsts = true
hsts_max_age_seconds = 31536000
```

## Environment Variables

### Server
```bash
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_WORKERS=8
LOG_LEVEL=info
```

### Device
```bash
TORCH_DEVICE=cuda
CUDA_VISIBLE_DEVICES=0,1
TORCH_NUM_THREADS=8
```

### Authentication
```bash
JWT_SECRET=your-secret-key
JWT_ALGORITHM=HS256
```

### Performance
```bash
CACHE_SIZE_MB=2048
MAX_BATCH_SIZE=32
ENABLE_TENSOR_POOLING=true
```

### LibTorch
```bash
LIBTORCH=/path/to/libtorch
LD_LIBRARY_PATH=$LIBTORCH/lib
LIBTORCH_USE_PYTORCH=1  # Use Python PyTorch's libtorch
```

## Configuration Profiles

### Development

```toml
[server]
host = "127.0.0.1"
workers = 4
log_level = "debug"

[device]
device_type = "cpu"
use_fp16 = false

[performance]
cache_size_mb = 512
max_batch_size = 8
enable_profiling = true

[auth]
enabled = false
```

### Production

```toml
[server]
host = "0.0.0.0"
workers = 16
log_level = "info"

[device]
device_type = "cuda"
use_fp16 = true
cudnn_benchmark = true

[performance]
cache_size_mb = 4096
max_batch_size = 64
enable_worker_pool = true
max_workers = 32

[auth]
enabled = true
jwt_secret = "production-secret-from-env"

[guard]
enable_guards = true
enable_auto_mitigation = true

[security]
rate_limit_per_ip = 1000
enable_security_headers = true
```

### Edge/Containers

```toml
[server]
workers = 2

[device]
device_type = "cpu"
num_threads = 4

[performance]
cache_size_mb = 256
max_batch_size = 8
enable_model_quantization = true
quantization_bits = 8

[models]
max_loaded_models = 2
```

## Command-Line Arguments

```bash
# Config file
./torch-inference-server --config production.toml

# Override port
./torch-inference-server --port 8080

# Override device
./torch-inference-server --device cuda

# Override workers
./torch-inference-server --workers 16

# Multiple overrides
./torch-inference-server \
  --config prod.toml \
  --port 8080 \
  --device cuda \
  --workers 16
```

## Hot-Reloading

Some settings can be updated without restart:

**Supported**:
- Cache size
- Batch size
- Worker pool size
- Log level

**Requires Restart**:
- Server port
- Device type
- Model directory

**Example**:
```bash
curl -X POST http://localhost:8000/api/config/update \
  -H "Authorization: Bearer TOKEN" \
  -d '{"performance": {"cache_size_mb": 4096}}'
```

## Validation

Validate configuration:

```bash
# Dry-run (validate only)
./torch-inference-server --config config.toml --validate

# Show effective config
./torch-inference-server --config config.toml --show-config
```

## Best Practices

1. **Use environment variables for secrets**
2. **Profile-specific configs** (dev, staging, prod)
3. **Version control configs** (without secrets)
4. **Document custom settings**
5. **Test configuration changes in staging**

## Troubleshooting

### Configuration Not Loading

```bash
# Check file exists
ls -l config.toml

# Validate TOML syntax
cargo run --bin torch-inference-server -- --validate
```

### Environment Variables Not Working

```bash
# Print all env vars
env | grep TORCH

# Test with explicit values
TORCH_DEVICE=cuda cargo run
```

### Performance Issues

```bash
# Enable profiling
[performance]
enable_profiling = true

# Check metrics
curl http://localhost:8000/api/stats/cache
curl http://localhost:8000/api/stats/batch
```

---

**Next**: See [Performance Tuning](TUNING.md) for optimization strategies.

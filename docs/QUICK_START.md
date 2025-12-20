# Quick Start Guide

Get Torch Inference up and running in 5 minutes!

## Prerequisites

- **Rust**: 1.70 or later
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4GB minimum, 8GB recommended
- **Optional**: CUDA 11.7+ (for GPU acceleration)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/torch-inference.git
cd torch-inference
```

### 2. Build the Server

#### CPU-Only Build (Fastest)
```bash
cargo build --release
```

#### With PyTorch Support
```bash
# Auto-downloads LibTorch
cargo build --release --features torch

# Or use existing PyTorch installation
export LIBTORCH=/path/to/libtorch
cargo build --release --features torch
```

#### With ONNX Support
```bash
cargo build --release --features onnx
```

#### All Features
```bash
cargo build --release --features all-backends
```

## First Run

### 1. Start the Server

```bash
cargo run --release --bin torch-inference-server
```

You should see:

```
═══════════════════════════════════════════════════════
  [START] PyTorch Inference Framework v1.0.0
═══════════════════════════════════════════════════════

[OK] Configuration loaded successfully
[AUTO] Detected CUDA - Setting device_type to 'cuda'
[OK] Server started at http://0.0.0.0:8000
```

### 2. Check Health

```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "uptime": "5s",
  "memory_used_mb": 245,
  "cache_size": 0,
  "device_type": "cuda"
}
```

### 3. Test Authentication

Generate an auth token:

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

## Your First Inference

### Example 1: Model Info

```bash
curl http://localhost:8000/api/models
```

### Example 2: System Stats

```bash
# Cache stats
curl http://localhost:8000/api/stats/cache

# Batch stats  
curl http://localhost:8000/api/stats/batch

# System info
curl http://localhost:8000/api/system/info
```

### Example 3: Load a Model

```bash
# List available models
curl http://localhost:8000/api/registry/models

# Download a model
curl -X POST http://localhost:8000/api/models/download \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"model_name": "resnet50"}'
```

### Example 4: Image Classification (if model loaded)

```bash
curl -X POST http://localhost:8000/api/classify \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image=@/path/to/image.jpg"
```

## Configuration

### Basic Configuration

Edit `config.toml`:

```toml
[server]
host = "0.0.0.0"
port = 8000
workers = 8

[device]
device_type = "auto"  # auto, cuda, cpu, mps
use_fp16 = true       # Enable FP16 for faster inference

[performance]
enable_caching = true
cache_size_mb = 2048
enable_request_batching = true
max_batch_size = 32
```

### Environment Variables

```bash
# Logging
export LOG_LEVEL=info
export LOG_JSON=false

# Device
export TORCH_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0,1

# Performance
export TORCH_NUM_THREADS=8
```

## Running Tests

```bash
# Run all tests
cargo test

# Run specific test suites
cargo test cache::tests
cargo test batch::tests
cargo test integration_test

# Run benchmarks
cargo bench
```

## Common Commands

### Development
```bash
# Run in dev mode with auto-reload
cargo watch -x run

# Run with debug logging
RUST_LOG=debug cargo run

# Format code
cargo fmt

# Lint
cargo clippy
```

### Production
```bash
# Build optimized binary
cargo build --release --features all-backends

# Run production server
./target/release/torch-inference-server

# Run with config file
./target/release/torch-inference-server --config production.toml
```

## Next Steps

1. **Configuration**: Read [Configuration Guide](CONFIGURATION.md) to optimize settings
2. **API**: Explore [API Reference](API_REFERENCE.md) for all endpoints
3. **Models**: Learn [Model Management](MODEL_MANAGEMENT_API.md) for model operations
4. **Deploy**: Follow [Deployment Guide](DEPLOYMENT.md) for production setup
5. **Performance**: Review [Performance Tuning](TUNING.md) for optimization

## Troubleshooting

### Server Won't Start

```bash
# Check port availability
lsof -i :8000

# Check logs
RUST_LOG=debug cargo run
```

### LibTorch Not Found

```bash
# Auto-download
export LIBTORCH_LOCAL=1
cargo build --features torch

# Or manual install
# Download from: https://pytorch.org/get-started/locally/
export LIBTORCH=/path/to/libtorch
cargo build --features torch
```

### CUDA Errors

```bash
# Check CUDA installation
nvidia-smi

# Force CPU mode
export TORCH_DEVICE=cpu
cargo run
```

### Build Errors

```bash
# Clean and rebuild
cargo clean
cargo build --release

# Update dependencies
cargo update
```

## Performance Tips

1. **Enable FP16**: 2x faster on compatible GPUs
   ```toml
   [device]
   use_fp16 = true
   ```

2. **Increase Cache**: Reduces redundant computation
   ```toml
   [performance]
   cache_size_mb = 4096
   ```

3. **Enable Batching**: 2-4x throughput improvement
   ```toml
   [performance]
   enable_request_batching = true
   adaptive_batch_timeout = true
   ```

4. **Use Multiple Workers**:
   ```toml
   [server]
   workers = 16  # Use num_cpus
   ```

## Getting Help

- **Documentation**: [docs/](.)
- **Issues**: [GitHub Issues](https://github.com/your-org/torch-inference/issues)
- **Discord**: [Community Server](https://discord.gg/your-server)

## What's Next?

- 📖 Read the [Architecture Overview](ARCHITECTURE.md)
- 🔧 Learn about [Components](COMPONENTS.md)
- 🚀 Explore [Performance Optimization](PERFORMANCE.md)
- 📊 Setup [Monitoring](MONITORING.md)
- 🐳 Deploy with [Docker](DOCKER.md)

---

**Congratulations!** You've successfully started Torch Inference. 🎉

# Quick Start Guide - PyTorch Inference Framework (Rust)

## 1-Minute Setup

### Option A: Pre-built Binary (Windows)

```bash
# Binary is ready at:
target/release/torch-inference-server.exe

# Run it:
.\target\release\torch-inference-server.exe

# Server starts on http://localhost:8000
```

### Option B: Build from Source

```bash
# Install Rust (if not already):
# https://rustup.rs/

# Build release binary (takes ~2 minutes):
cargo build --release

# Run the server:
.\target\release\torch-inference-server.exe
```

### Option C: Development Mode (with live reload)

```bash
# Install cargo-watch (optional):
cargo install cargo-watch

# Run with hot reload:
cargo watch -x run

# Or just run directly:
cargo run
```

## Verify Server is Running

```bash
# Health check (should return 200):
curl http://localhost:8000/health

# List models (should show example model):
curl http://localhost:8000/models

# Get system info:
curl http://localhost:8000/info
```

## Make Your First Inference Request

```bash
# Single inference:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "example",
    "inputs": [1.0, 2.0, 3.0],
    "priority": 0
  }'

# Expected response:
{
  "success": true,
  "result": 0.5234,
  "processing_time": 0.0045,
  "model_info": {
    "model": "example",
    "device": "cpu"
  }
}
```

## TTS Synthesis Example

```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "speecht5_tts",
    "text": "Hello world, this is a test",
    "voice": "default",
    "speed": 1.0,
    "pitch": 1.0,
    "volume": 1.0,
    "language": "en",
    "output_format": "wav"
  }'

# Returns base64 encoded WAV audio in response
```

## Configuration

### Using Default Configuration

```bash
# Server runs with defaults:
# - Host: 0.0.0.0
# - Port: 8000
# - Device: auto (CPU or GPU)
# - Batch size: 1
```

### Custom Configuration with config.toml

```toml
[server]
host = "0.0.0.0"
port = 8000
log_level = "info"
workers = 4

[device]
device_type = "auto"  # "auto", "cpu", or "cuda"
device_id = 0
use_fp16 = false

[batch]
batch_size = 1
max_batch_size = 8
enable_dynamic_batching = true

[performance]
warmup_iterations = 3
enable_caching = true
cache_size_mb = 1024

[auth]
enabled = false  # Set to true to enable JWT auth
jwt_secret = "your-secret-key"
```

## Enable Logging

```bash
# See debug logs:
set RUST_LOG=debug
cargo run --release

# Or on Linux/Mac:
RUST_LOG=debug cargo run --release

# Or for released binary:
set RUST_LOG=debug
.\target\release\torch-inference-server.exe
```

## Common Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Statistics
```bash
curl http://localhost:8000/stats
```

### List Available Models
```bash
curl http://localhost:8000/models
```

### System Information
```bash
curl http://localhost:8000/info
```

### Batch Inference
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "example",
    "inputs": [
      [1.0, 2.0],
      [3.0, 4.0],
      [5.0, 6.0]
    ]
  }'
```

## Deployment

### Docker

```dockerfile
# Build Docker image:
docker build -t torch-inference:latest .

# Run container:
docker run -p 8000:8000 torch-inference:latest

# Check if running:
curl http://localhost:8000/health
```

### Systemd (Linux)

```bash
# Create service file:
sudo nano /etc/systemd/system/torch-inference.service

# Add content:
[Unit]
Description=PyTorch Inference Framework
After=network.target

[Service]
Type=simple
User=inference
WorkingDirectory=/opt/torch-inference
ExecStart=/opt/torch-inference/torch-inference-server
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target

# Enable and start:
sudo systemctl enable torch-inference
sudo systemctl start torch-inference

# Check status:
sudo systemctl status torch-inference
```

## Performance Testing

### Simple Load Test (using Apache Bench)

```bash
# Install ab (if needed):
# Windows: https://httpd.apache.org/docs/2.4/programs/ab.html

# Run 1000 requests with 10 concurrent:
ab -n 1000 -c 10 http://localhost:8000/health

# Run inference load test:
ab -n 100 -c 5 -p request.json \
  -T application/json \
  http://localhost:8000/predict
```

### Using wrk (Modern Benchmarking)

```bash
# Install wrk:
# https://github.com/wg/wrk

# Simple load test:
wrk -t4 -c100 -d30s http://localhost:8000/health

# With custom script:
wrk -t4 -c100 -d30s -s script.lua http://localhost:8000/predict
```

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000:
netstat -ano | findstr :8000

# Kill process (if needed):
taskkill /PID <PID> /F

# Or use different port in config.toml
```

### CUDA/GPU Not Detected
```bash
# Check available device types:
cargo build --release
# Device type will default to CPU if GPU not available

# To force CPU:
set RUST_LOG=info
# Look for "Device detected: CPU" in logs
```

### High Memory Usage
```bash
# Check current memory:
# Task Manager -> Performance tab

# Reduce cache size in config.toml:
[performance]
cache_size_mb = 256  # Reduce from 1024
```

## Next Steps

1. **Read README.md** for full documentation
2. **Read ARCHITECTURE.md** for design details
3. **Explore API** via curl commands above
4. **Load custom models** (see documentation)
5. **Deploy to production** (Docker/Kubernetes)

## Performance Metrics

**Typical Performance** (on modern hardware):

- Startup Time: < 100ms
- Mean Response Time: 8-15ms (single request)
- Throughput: 2,000-5,000 req/s (depending on model)
- Memory Usage: 20-100MB
- CPU Usage: < 10% (idle)

## Getting Help

1. **Check logs**: `RUST_LOG=debug cargo run`
2. **Review README.md**: Full documentation
3. **Check ARCHITECTURE.md**: Design explanation
4. **See examples**: Above examples in this file

## Key Files

- `Cargo.toml` - Project configuration & dependencies
- `src/main.rs` - Server entry point
- `src/api/handlers.rs` - HTTP endpoints
- `src/core/engine.rs` - Inference engine
- `src/models/manager.rs` - Model management
- `config.toml` - Runtime configuration
- `README.md` - Full documentation
- `ARCHITECTURE.md` - Design documentation

## Server Endpoints Summary

| Endpoint | Method | Purpose | Example |
|----------|--------|---------|---------|
| `/` | GET | API info | `curl http://localhost:8000/` |
| `/health` | GET | Health check | `curl http://localhost:8000/health` |
| `/predict` | POST | Run inference | `curl -X POST http://localhost:8000/predict ...` |
| `/synthesize` | POST | TTS synthesis | `curl -X POST http://localhost:8000/synthesize ...` |
| `/models` | GET | List models | `curl http://localhost:8000/models` |
| `/stats` | GET | Statistics | `curl http://localhost:8000/stats` |
| `/info` | GET | System info | `curl http://localhost:8000/info` |

---

**Ready to go!** 🚀

Start the server with:
```bash
cargo build --release
./target/release/torch-inference-server
```

Then visit: http://localhost:8000/health

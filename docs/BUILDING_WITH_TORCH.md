# Building with PyTorch Support - Guide

**Date:** 2025-12-18  
**Purpose:** Enable image classification inference with SOTA models

## Current Status

The server is currently compiled **without** the `torch` feature, which means:
- ✅ Model download API works
- ✅ TTS synthesis works
- ❌ Image classification inference disabled

## Prerequisites

### 1. LibTorch Installation

**Option A: Auto-download during build**
The build script will attempt to auto-download LibTorch for your platform.

**Option B: Manual download**
```bash
# For macOS (Apple Silicon)
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.3.0.zip
unzip libtorch-macos-arm64-2.3.0.zip
export LIBTORCH=$(pwd)/libtorch

# For Linux (CPU)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.3.0+cpu.zip
export LIBTORCH=$(pwd)/libtorch

# For Linux (CUDA 11.8)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.3.0+cu118.zip
export LIBTORCH=$(pwd)/libtorch
```

### 2. System Dependencies

**macOS:**
```bash
# Xcode command line tools
xcode-select --install

# OpenMP (optional, for CPU acceleration)
brew install libomp
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake libssl-dev pkg-config

# Fedora/RHEL
sudo dnf install gcc-c++ cmake openssl-devel
```

## Building with PyTorch Support

### Method 1: Using the Build Script

```bash
chmod +x build_with_torch.sh
./build_with_torch.sh
```

### Method 2: Manual Build

```bash
# Set LibTorch path (if not auto-downloading)
export LIBTORCH=$(pwd)/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=${LIBTORCH}/lib:$DYLD_LIBRARY_PATH

# Build with torch feature
cargo build --release --features torch
```

### Method 3: Build with All Features

```bash
cargo build --release --features all-backends
```

This enables:
- PyTorch (torch)
- ONNX Runtime
- Candle
- CUDA (if available)

## Verifying the Build

### 1. Check Binary

```bash
ls -lh ./target/release/torch-inference-server

# Should show binary size (with torch: ~50-100MB larger)
```

### 2. Check Build Log

```bash
cat build_torch.log | grep -i "libtorch\|pytorch\|torch"
```

Expected output:
```
[OK] Found libtorch at: "/path/to/libtorch"
[OK] PyTorch initialized successfully
```

### 3. Run Server

```bash
./target/release/torch-inference-server
```

Look for in startup logs:
```
[OK] PyTorch initialized successfully
   ├─ Backend: CPU (or CUDA/Metal)
   ├─ Path: /path/to/libtorch
   └─ Version: 2.3.0
```

## Testing Image Classification

### 1. Start Server

```bash
./target/release/torch-inference-server
```

### 2. Download a Model

```bash
# Download smallest model for testing (MobileNetV4 - 140MB)
curl -X POST http://localhost:8000/models/sota/mobilenetv4-hybrid-large \
  -H "Content-Type: application/json"

# Response:
# {
#   "task_id": "uuid-here",
#   "status": "started",
#   "message": "Download task created"
# }
```

### 3. Check Download Progress

```bash
# List download tasks
curl http://localhost:8000/models/download/list | jq .

# Check specific task
TASK_ID="your-task-id-here"
curl http://localhost:8000/models/downloads/$TASK_ID | jq .
```

### 4. Run Inference (when implemented)

```bash
# Upload and classify an image
curl -X POST http://localhost:8000/classify \
  -F "image=@test_image.jpg" \
  -F "model=mobilenetv4-hybrid-large" \
  -F "top_k=5"
```

Expected response:
```json
{
  "predictions": [
    {"label": "tabby_cat", "confidence": 0.89, "class_id": 281},
    {"label": "egyptian_cat", "confidence": 0.07, "class_id": 285},
    ...
  ],
  "inference_time_ms": 45.2
}
```

## Troubleshooting

### Build Errors

#### Error: LibTorch not found
```
Solution: Download LibTorch manually and set LIBTORCH environment variable
```

#### Error: CUDA version mismatch
```
Solution: Download LibTorch version matching your CUDA version
```

#### Error: Linker errors
```
Solution on macOS:
export DYLD_LIBRARY_PATH=${LIBTORCH}/lib:$DYLD_LIBRARY_PATH

Solution on Linux:
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```

### Runtime Errors

#### Error: Library not found at runtime
```bash
# macOS
export DYLD_LIBRARY_PATH=$(pwd)/libtorch/lib:$DYLD_LIBRARY_PATH

# Linux
export LD_LIBRARY_PATH=$(pwd)/libtorch/lib:$LD_LIBRARY_PATH
```

#### Error: Model download fails
```
Check:
1. Internet connection
2. HuggingFace accessibility
3. Disk space
4. ./models/ directory permissions
```

## Performance Optimization

### CPU Optimization

```bash
# Set number of threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Run server
./target/release/torch-inference-server
```

### GPU Acceleration (CUDA)

```bash
# Build with CUDA
cargo build --release --features "torch cuda"

# Verify GPU usage in logs:
# [OK] GPU Manager initialized - 1 CUDA GPU(s) detected
```

### Metal Acceleration (macOS)

```bash
# Automatically detected on Apple Silicon
# Look for in logs:
# [OK] GPU Manager initialized - Metal GPU detected
```

## Expected Performance

### Model Download Times (on good connection)

| Model | Size | Download Time |
|-------|------|---------------|
| MobileNetV4 | 140 MB | ~30 sec |
| EfficientNetV2 XL | 850 MB | ~3 min |
| EVA-02 Large | 1.2 GB | ~4 min |
| ConvNeXt V2 Huge | 2.6 GB | ~8 min |
| EVA Giant | 4.0 GB | ~12 min |

### Inference Times (estimated, CPU)

| Model | Input Size | Inference Time |
|-------|-----------|----------------|
| MobileNetV4 | 224x224 | ~50-100ms |
| EfficientNetV2 | 224x224 | ~100-200ms |
| EVA-02 Large | 448x448 | ~300-500ms |
| ConvNeXt V2 | 512x512 | ~500-800ms |

*GPU inference will be 5-10x faster*

## Next Steps

After successful build:

1. ✅ Run comprehensive tests: `./test_final_report.sh`
2. ✅ Download a small model for testing
3. ✅ Test image classification endpoint
4. ✅ Monitor performance and memory usage
5. ✅ Deploy to production

## Additional Resources

- PyTorch Downloads: https://pytorch.org/get-started/locally/
- tch-rs Documentation: https://github.com/LaurentMazare/tch-rs
- timm Models: https://github.com/huggingface/pytorch-image-models
- Build Logs: `./build_torch.log`

---

**Note:** The current server binary (built without torch) is fully functional for TTS and model management. Rebuild is only needed for image classification inference.

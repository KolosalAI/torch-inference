# 📦 Installation Guide

This guide covers the complete installation process for the PyTorch Inference Framework.

## 🎯 Prerequisites

### System Requirements
- **Python**: 3.10+ (3.11+ recommended)
- **Operating System**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **Memory**: 8GB+ RAM (16GB+ recommended)

### GPU Requirements (Optional but Recommended)
- **NVIDIA GPU**: Compute capability 7.0+ (RTX 20/30/40 series, Tesla V100, A100, H100)
- **CUDA**: 12.4+ (for TensorRT optimization)
- **GPU Memory**: 6GB+ VRAM (8GB+ recommended for large models)

## 🚀 Quick Installation with uv

The framework uses `uv` for fast, reliable dependency management.

### 1. Install uv

**Option A: Using pip**
```bash
pip install uv
```

**Option B: Official installer (Linux/macOS)**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Option C: Official installer (Windows)**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Setup

**Windows (PowerShell)**
```powershell
# Clone repository
git clone https://github.com/Evintkoo/torch-inference.git
cd torch-inference

# Run automated setup
.\setup_uv.ps1

# Verify installation
uv run python test_installation.py
```

**Linux/macOS (Bash)**
```bash
# Clone repository
git clone https://github.com/Evintkoo/torch-inference.git
cd torch-inference

# Run automated setup
chmod +x setup_uv.sh
./setup_uv.sh

# Verify installation
uv run python test_installation.py
```

### 3. Manual Setup (Alternative)

```bash
# Install all dependencies
uv sync

# Install with GPU support
uv sync --extra cuda

# Install with all features
uv sync --extra all

# Verify installation
uv run python test_installation.py
```

## 🔧 Installation Options

### Base Installation
```bash
# Core framework only
uv sync

# This includes:
# - PyTorch CPU
# - FastAPI
# - Basic optimization
```

### GPU Support
```bash
# CUDA support
uv sync --extra cuda

# This adds:
# - PyTorch CUDA
# - CUDA optimizations
# - GPU memory management
```

### TensorRT Optimization
```bash
# TensorRT support
uv sync --extra tensorrt

# This adds:
# - TensorRT runtime
# - torch-tensorrt
# - Advanced GPU optimization
```

### Development Setup
```bash
# Development tools
uv sync --extra dev

# This adds:
# - Testing framework (pytest)
# - Code formatting (black, ruff)
# - Type checking (mypy)
# - Pre-commit hooks
```

### Complete Installation
```bash
# All features
uv sync --extra all

# Equivalent to:
uv sync --extra cuda,tensorrt,onnx,dev,docs
```

## 🐍 Python Environment Setup

### Using Conda (Recommended for GPU)
```bash
# Create environment with CUDA support
conda create -n torch-inference python=3.11
conda activate torch-inference

# Install CUDA toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install framework
cd torch-inference
uv sync --no-install-project  # Skip PyTorch reinstall
```

### Using Python Virtual Environment
```bash
# Create virtual environment
python -m venv torch-inference
source torch-inference/bin/activate  # Linux/macOS
# OR
torch-inference\Scripts\activate  # Windows

# Install framework
cd torch-inference
uv sync
```

### Using uv Managed Environment (Simplest)
```bash
# uv manages everything automatically
cd torch-inference
uv sync  # Creates and manages virtual environment
uv run python test_installation.py
```

## 🔍 Installation Verification

### Quick Test
```bash
uv run python -c "import framework; print('✅ Framework imported successfully')"
```

### Comprehensive Test
```bash
uv run python test_installation.py
```

Expected output:
```
✅ Python environment: OK (3.11.5)
✅ PyTorch installation: OK (2.8.0+cu124)
✅ CUDA available: OK (12.4)
✅ GPU memory: OK (24GB available)
✅ Framework import: OK
✅ Basic inference: OK (15.2ms)
✅ Optimized inference: OK (3.8ms, 4.0x speedup)
✅ TensorRT available: OK (10.12.0.36)
✅ ONNX Runtime: OK (1.22.1)
🎉 Installation verification complete!
```

## 🚨 Troubleshooting Installation

### Common Issues

#### uv command not found
```bash
# Add uv to PATH (Linux/macOS)
export PATH="$HOME/.local/bin:$PATH"

# Or reinstall uv
pip install --force-reinstall uv
```

#### CUDA out of memory during installation
```bash
# Install without CUDA first
uv sync

# Then install CUDA components separately
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### TensorRT installation fails
```bash
# Skip TensorRT for now
uv sync --extra cuda  # Without TensorRT

# Install TensorRT manually later
uv add tensorrt torch-tensorrt
```

#### Permission errors (Windows)
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then run setup
.\setup_uv.ps1
```

#### Import errors
```bash
# Verify environment activation
uv run which python

# Check installed packages
uv tree

# Reinstall if needed
uv sync --reinstall
```

### Platform-Specific Issues

#### Linux
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install build-essential

# For CUDA support
sudo apt-get install nvidia-cuda-toolkit
```

#### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# For Metal Performance Shaders (Apple Silicon)
uv sync --extra mps
```

#### Windows
```powershell
# Install Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or install Visual Studio with C++ support
```

### Environment Variables
```bash
# Set CUDA paths (if needed)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Performance optimization
export TORCH_CUDNN_BENCHMARK=true
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
```

## 📊 Installation Benchmarks

### Installation Speed Comparison

| Method | Time | Size | Features |
|--------|------|------|----------|
| **uv (base)** | ~30s | 1.2GB | Core only |
| **uv (cuda)** | ~60s | 3.8GB | GPU support |
| **uv (all)** | ~120s | 5.2GB | All features |
| **pip** | ~300s | 6.1GB | Traditional |
| **conda** | ~480s | 7.8GB | Kitchen sink |

### System Requirements by Usage

| Use Case | RAM | VRAM | Storage | Network |
|----------|-----|------|---------|---------|
| **CPU Only** | 4GB | - | 2GB | 1GB |
| **GPU Basic** | 8GB | 6GB | 5GB | 3GB |
| **Production** | 16GB | 12GB | 10GB | 5GB |
| **Development** | 16GB | 12GB | 15GB | 8GB |

## 🔄 Updating Installation

### Update Framework
```bash
# Update to latest version
git pull origin main
uv sync

# Update specific dependencies
uv add torch@latest
```

### Clean Installation
```bash
# Remove lock file and reinstall
rm uv.lock
uv sync

# Complete clean install
rm -rf .uv_cache/
uv sync --reinstall
```

### Migration from pip/conda
```bash
# If migrating from existing installation
pip uninstall torch-inference

# Clean install with uv
uv sync --reinstall
```

## 🐳 Docker Installation

### Pre-built Images
```bash
# CPU-only image
docker pull evintkoo/torch-inference:cpu

# GPU-enabled image
docker pull evintkoo/torch-inference:gpu

# Run container
docker run --gpus all -p 8000:8000 evintkoo/torch-inference:gpu
```

### Build from Source
```bash
# Build optimized image
docker build -t torch-inference-custom .

# Run with GPU support
docker run --gpus all -p 8000:8000 torch-inference-custom
```

## ⚙️ Configuration After Installation

### Set up environment variables
```bash
# Create .env file
cat > .env << EOF
DEVICE=cuda
BATCH_SIZE=16
LOG_LEVEL=INFO
ENABLE_TENSORRT=true
EOF
```

### Configure for your use case
```bash
# Copy example configuration
cp config.example.yaml config.yaml

# Edit for your needs
nano config.yaml
```

### Set up monitoring
```bash
# Install monitoring dependencies
uv sync --extra monitoring

# Start monitoring dashboard
uv run python -m framework.monitoring.dashboard
```

## 🎯 Next Steps

After successful installation:

1. **[Quick Start Guide](quickstart.md)** - Basic usage examples
2. **[Configuration Guide](configuration.md)** - Customize settings
3. **[Examples](examples.md)** - Explore use cases
4. **[API Reference](api.md)** - Detailed documentation

## 📞 Installation Support

If you encounter issues:

- 🔍 **Check**: [Troubleshooting Guide](troubleshooting.md)
- 🐛 **Report**: [GitHub Issues](https://github.com/Evintkoo/torch-inference/issues)
- 💬 **Ask**: [GitHub Discussions](https://github.com/Evintkoo/torch-inference/discussions)
- 📧 **Email**: [support@torch-inference.dev](mailto:support@torch-inference.dev)

---

*Installation completed? Check out the [Quick Start Guide](quickstart.md) to begin using the framework!*

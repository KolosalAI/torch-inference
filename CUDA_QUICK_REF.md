# 🚀 CUDA + TensorRT Quick Reference

## System Configuration
- **GPU:** NVIDIA RTX 3060 Laptop (6GB VRAM)
- **CPU:** Intel i7-12700H (14 cores, 20 threads)
- **RAM:** 64 GB DDR4
- **CUDA:** 11.8
- **Optimization:** TensorRT INT8 enabled

---

## 🎯 Quick Start (3 Commands)

### 1. Install PyTorch with CUDA
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Run Automated Setup
```powershell
cd D:\Works\Genta\codes\torch-inference
.\scripts\setup_cuda_benchmark.ps1
```

### 3. Or Manually Run Benchmark
```powershell
cargo bench --bench image_classification_benchmark --no-default-features --features cuda,torch
```

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| **CPU Baseline** | ~125 FPS |
| **CUDA FP32** | ~400 FPS (3× faster) |
| **CUDA FP16** | ~650 FPS (5× faster) |
| **TensorRT INT8** | ~1250 FPS (10× faster) ⭐ |

---

## 🔧 Configuration Files

| File | Purpose |
|------|---------|
| `config.toml` | Main config (CUDA enabled) |
| `.env` | Environment variables |
| `config/cuda_tensorrt.toml` | Full CUDA config |
| `scripts/setup_cuda_benchmark.ps1` | Automated setup |

---

## 📈 Monitor GPU

```powershell
# Real-time GPU monitoring
nvidia-smi -l 1

# Expected during benchmark:
# - GPU Utilization: 90-100%
# - Memory Usage: 3-5 GB
# - Temperature: 70-85°C
# - Power: 80-115W
```

---

## 📁 Results Location

```
benchmark_results/
├── windows_cuda/     # GPU results here
├── windows_cpu/      # CPU baseline
└── comparison/       # Side-by-side
```

---

## ⚡ Optimization Levels

Current configuration uses **maximum performance**:
- ✅ TensorRT INT8 quantization
- ✅ FP16 mixed precision
- ✅ cuDNN auto-tuner
- ✅ CUDA graphs
- ✅ Multi-stream execution
- ✅ Tensor pooling

---

## 🐛 Quick Troubleshooting

### CUDA not detected
```powershell
nvidia-smi  # Should show GPU
```

### PyTorch CUDA not working
```powershell
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### Build fails
```powershell
# Rebuild without jemalloc (Windows incompatible)
cargo build --release --no-default-features --features cuda,torch
```

---

## 📚 Full Documentation

- **Setup Guide:** `docs/CUDA_TENSORRT_SETUP.md`
- **Benchmark Plan:** `docs/WINDOWS_BENCHMARK_PLAN.md`
- **Quick Start:** `QUICK_START_BENCHMARKS.md`

---

## 🎯 One-Liner (After PyTorch Installed)

```powershell
.\scripts\setup_cuda_benchmark.ps1
```

This will:
1. ✅ Verify CUDA
2. ✅ Check PyTorch
3. ✅ Configure environment
4. ✅ Build with CUDA
5. ✅ Run benchmarks (optional)

---

**Status:** ✅ Ready to benchmark  
**Expected Speedup:** 10× vs CPU  
**First Run:** 2-4 hours (builds TensorRT engines)  
**Subsequent Runs:** 15-30 minutes

**Questions?** See `docs/CUDA_TENSORRT_SETUP.md`

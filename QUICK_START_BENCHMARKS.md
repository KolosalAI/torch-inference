# Quick Start: Running Benchmarks on Your Windows System

## Prerequisites Installation (Run Once)

### Step 1: Install Python
```powershell
winget install Python.Python.3.11
```

### Step 2: Install PyTorch with CUDA 11.8
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Set Environment Variable
```powershell
$env:LIBTORCH = (python -c "import torch; import os; print(os.path.dirname(torch.__file__))" | Out-String).Trim()
[System.Environment]::SetEnvironmentVariable('LIBTORCH', $env:LIBTORCH, 'User')
```

### Step 4: Verify Installation
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Running Benchmarks

### Quick Test (Single Model)
```powershell
cd D:\Works\Genta\codes\torch-inference
cargo run --release --no-default-features --features torch --example single_model_test
```

### Full CPU Benchmark Suite
```powershell
cargo bench --bench image_classification_benchmark --no-default-features --features torch
```

### GPU/CUDA Benchmarks
```powershell
cargo bench --bench image_classification_benchmark --no-default-features --features cuda,torch -- --device cuda
```

### Concurrent Throughput Test
```powershell
cargo bench --bench concurrent_throughput_benchmark --no-default-features --features torch
```

## Results Location
```
benchmark_results/
├── windows_cpu/
├── windows_cuda/
└── comparison/
```

## Viewing Results
```powershell
# View latest results
Get-Content benchmark_results\windows_cpu\summary.md

# View JSON data
Get-Content benchmark_results\windows_cpu\resnet50.json | ConvertFrom-Json
```

## Common Issues & Solutions

### Issue: "Cannot find libtorch"
**Solution:** Ensure LIBTORCH environment variable is set:
```powershell
echo $env:LIBTORCH
# Should output: C:\Users\Evint\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch
```

### Issue: "jemalloc not found"
**Solution:** Use `--no-default-features`:
```powershell
cargo bench --no-default-features --features torch
```

### Issue: CUDA not detected
**Solution:** Verify CUDA installation:
```powershell
nvidia-smi
nvcc --version
```

## System Info
- **CPU:** Intel i7-12700H (14 cores, 20 threads)
- **RAM:** 64 GB DDR4
- **GPU:** RTX 3060 Laptop (6GB VRAM)
- **CUDA:** 11.8

## Documentation
- Full Plan: `docs/WINDOWS_BENCHMARK_PLAN.md`
- Summary: `docs/BENCHMARK_UPDATE_SUMMARY.md`
- Paper: `docs/research/torch_inference_benchmark_paper.tex`

---
**Last Updated:** 2026-01-05

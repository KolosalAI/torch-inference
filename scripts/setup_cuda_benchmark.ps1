# CUDA + TensorRT Benchmark Setup and Execution Script
# For NVIDIA RTX 3060 Laptop GPU on Windows 11

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "CUDA + TensorRT Benchmark Setup" -ForegroundColor Cyan
Write-Host "System: LAPTOP-EVINT" -ForegroundColor Cyan
Write-Host "GPU: NVIDIA RTX 3060 Laptop (6GB)" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check CUDA availability
Write-Host "[1/8] Checking CUDA installation..." -ForegroundColor Yellow
try {
    $cudaVersion = (nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null)
    Write-Host "  checkmark NVIDIA Driver: $cudaVersion" -ForegroundColor Green
    
    $gpuName = (nvidia-smi --query-gpu=name --format=csv,noheader 2>$null)
    Write-Host "  checkmark GPU Detected: $gpuName" -ForegroundColor Green
    
    $gpuMemory = (nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>$null)
    Write-Host "  checkmark VRAM: $gpuMemory" -ForegroundColor Green
} catch {
    Write-Host "  X CUDA/NVIDIA drivers not found!" -ForegroundColor Red
    Write-Host "  Install CUDA Toolkit 11.8" -ForegroundColor Yellow
    exit 1
}

# Step 2: Check Python and PyTorch
Write-Host ""
Write-Host "[2/8] Checking PyTorch installation..." -ForegroundColor Yellow
try {
    $pythonVersion = (python --version 2>&1)
    Write-Host "  checkmark Python: $pythonVersion" -ForegroundColor Green
    
    $pytorchCheck = (python -c "import torch; print(f'{torch.__version__}'); print(torch.cuda.is_available())" 2>&1)
    if ($pytorchCheck -match "True") {
        Write-Host "  checkmark PyTorch with CUDA: Installed" -ForegroundColor Green
    } else {
        Write-Host "  X PyTorch CUDA not available!" -ForegroundColor Red
        Write-Host "  Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "  X Python or PyTorch not installed!" -ForegroundColor Red
    Write-Host "  Run: winget install Python.Python.3.11" -ForegroundColor Yellow
    exit 1
}

# Step 3: Set environment variables
Write-Host ""
Write-Host "[3/8] Setting environment variables..." -ForegroundColor Yellow

$libtorchPath = (python -c "import torch; import os; print(os.path.dirname(torch.__file__))" 2>&1)
if ($libtorchPath) {
    $env:LIBTORCH = $libtorchPath
    Write-Host "  checkmark LIBTORCH = $libtorchPath" -ForegroundColor Green
} else {
    Write-Host "  X Failed to detect LIBTORCH path!" -ForegroundColor Red
    exit 1
}

$env:TORCH_DEVICE = "cuda:0"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:CUDNN_BENCHMARK = "1"
$env:TENSORRT_ENABLED = "1"
$env:RUST_LOG = "info"

Write-Host "  checkmark CUDA environment configured" -ForegroundColor Green

# Step 4: Activate CUDA configuration
Write-Host ""
Write-Host "[4/8] Activating CUDA configuration..." -ForegroundColor Yellow
Copy-Item ".env.cuda" ".env" -Force
if (Test-Path "config\cuda_tensorrt.toml") {
    Copy-Item "config\cuda_tensorrt.toml" "config.toml" -Force
}
Write-Host "  checkmark CUDA config activated" -ForegroundColor Green

# Step 5: Build with CUDA support
Write-Host ""
Write-Host "[5/8] Building with CUDA support..." -ForegroundColor Yellow
Write-Host "  This may take 5-10 minutes..." -ForegroundColor Cyan

cargo build --release --no-default-features --features cuda,torch 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  checkmark Build successful" -ForegroundColor Green
} else {
    Write-Host "  X Build failed!" -ForegroundColor Red
    exit 1
}

# Step 6: GPU Information
Write-Host ""
Write-Host "[6/8] GPU Information:" -ForegroundColor Yellow
nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu,utilization.gpu --format=csv

# Step 7: Ready
Write-Host ""
Write-Host "[7/8] Ready to run benchmarks!" -ForegroundColor Green
Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Benchmark Commands:" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "CUDA GPU Benchmark:" -ForegroundColor Yellow
Write-Host "  cargo bench --bench image_classification_benchmark --no-default-features --features cuda,torch" -ForegroundColor White
Write-Host ""
Write-Host "Monitor GPU:" -ForegroundColor Yellow
Write-Host "  nvidia-smi -l 1" -ForegroundColor White
Write-Host ""

# Interactive prompt
$runNow = Read-Host "Run CUDA benchmark now? (y/n)"
if ($runNow -eq "y" -or $runNow -eq "Y") {
    Write-Host ""
    Write-Host "Starting CUDA benchmark..." -ForegroundColor Green
    cargo bench --bench image_classification_benchmark --no-default-features --features cuda,torch
} else {
    Write-Host "Setup complete! Run benchmarks when ready." -ForegroundColor Green
}

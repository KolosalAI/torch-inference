# Windows System Benchmark Plan

## System Configuration

### Hardware Specifications
- **System Name:** LAPTOP-EVINT
- **CPU:** Intel Core i7-12700H (12th Gen)
  - Architecture: Alder Lake (Hybrid)
  - P-cores: 6 (Performance cores, 12 threads)
  - E-cores: 8 (Efficiency cores, 8 threads)
  - Total: 14 cores, 20 logical processors
  - Base Clock: 2.3 GHz
  - Turbo Boost: Up to 4.7 GHz
- **Memory:** 64 GB DDR4
- **GPU (Discrete):** NVIDIA GeForce RTX 3060 Laptop GPU
  - VRAM: 6 GB GDDR6 (4,293,918,720 bytes)
  - CUDA Cores: 3840
  - Architecture: Ampere (GA106)
  - Compute Capability: 8.6
- **GPU (Integrated):** Intel Iris Xe Graphics
  - Shared Memory: 1 GB
- **Storage:** NVMe SSD
- **OS:** Microsoft Windows 11 Pro
  - Build: 26200
  - Architecture: x64

### Software Environment
- **CUDA:** CUDA Toolkit 11.8
- **Rust:** 1.75+ (stable-x86_64-pc-windows-msvc)
- **PyTorch:** 2.5.1 (to be installed with CUDA 11.8 support)
- **Python:** 3.11+ (to be installed)

## Benchmark Goals

### Primary Objectives
1. **Cross-Platform Validation**
   - Compare Rust FFI performance on Windows vs. macOS
   - Validate memory efficiency on x86_64 architecture
   - Measure cold-start performance on Windows

2. **CUDA Acceleration Benchmarks**
   - RTX 3060 inference performance
   - TensorRT optimization comparison
   - CUDA vs. CPU performance differential

3. **Hybrid Architecture Analysis**
   - P-core vs. E-core utilization
   - Thread scheduling efficiency
   - Power efficiency measurements

### Secondary Objectives
1. **Framework Comparison on Windows**
   - TorchServe on Windows
   - ONNX Runtime with DirectML
   - TensorRT on RTX 3060
   - OpenVINO on Intel Iris Xe

2. **Memory Consumption Analysis**
   - Windows vs. macOS memory overhead
   - VRAM utilization patterns
   - Unified memory vs. discrete VRAM comparison

## Prerequisites

### Software Installation Required
```powershell
# Install Python 3.11
winget install Python.Python.3.11

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies
pip install numpy pandas matplotlib seaborn tqdm

# Set LIBTORCH environment variable
$env:LIBTORCH = (python -c "import torch; import os; print(os.path.dirname(torch.__file__))" | Out-String).Trim()
[System.Environment]::SetEnvironmentVariable('LIBTORCH', $env:LIBTORCH, 'User')
```

### Rust Configuration
```powershell
# Build with CUDA support
cargo build --release --features cuda,torch

# Verify CUDA availability
cargo run --release --bin torch-inference-server -- --check-cuda
```

## Benchmark Execution Plan

### Phase 1: CPU Benchmarks (Intel i7-12700H)
```bash
# Run comprehensive CPU benchmarks
cargo bench --bench image_classification_benchmark --no-default-features --features torch

# Run concurrent throughput tests
cargo bench --bench concurrent_throughput_benchmark --no-default-features --features torch
```

**Expected Metrics:**
- Load time per model
- Inference latency (mean, P50, P95, P99)
- Throughput (FPS)
- Memory consumption
- CPU utilization (P-cores vs E-cores)

### Phase 2: CUDA Benchmarks (RTX 3060)
```bash
# Run GPU-accelerated benchmarks
cargo bench --bench image_classification_benchmark --features cuda,torch -- --device cuda

# Compare CUDA vs CPU
cargo bench --bench device_comparison_benchmark --features cuda,torch
```

**Expected Metrics:**
- GPU vs CPU speedup
- VRAM utilization
- TensorRT optimization gains
- Power consumption (if measurable)

### Phase 3: Cross-Framework Comparison
```bash
# TorchServe Windows benchmark
python benches/compare_torchserve.py --device cuda

# ONNX Runtime with DirectML
python benches/compare_onnxruntime.py --device dml

# TensorRT comparison
python benches/compare_tensorrt.py --device cuda
```

### Phase 4: Memory Efficiency Analysis
```bash
# Memory profiling
cargo bench --bench memory_benchmark --features torch

# Memory estimation accuracy
cargo test --test memory_estimation_tests --features torch -- --nocapture
```

## Expected Results

### CPU Performance (vs. M2 Pro)
- **Hypothesis:** i7-12700H should match or exceed M2 Pro on multi-threaded workloads
- **Rationale:** More cores (14 vs 12), higher clock speeds (4.7 GHz turbo vs 3.5 GHz)
- **Trade-offs:** Higher power consumption, potential thermal throttling

### GPU Performance
- **CUDA Acceleration:** Expected 5-10× speedup over CPU for CNN models
- **TensorRT:** Additional 2-3× improvement with INT8 quantization
- **Comparison:** RTX 3060 (6GB) vs M2 Pro GPU (unified memory)

### Memory Efficiency
- **Baseline:** Windows overhead typically 200-500 MB higher than macOS
- **Target:** Maintain <2× memory consumption vs bare libtorch
- **VRAM:** Efficient utilization of 6 GB VRAM for large models

## Success Criteria

### Minimum Requirements
- [x] System configuration documented
- [ ] PyTorch with CUDA successfully installed
- [ ] Rust FFI bindings compile on Windows
- [ ] At least 10 models benchmarked
- [ ] Results added to paper (Appendix B: Windows Validation)

### Optimal Achievements
- [ ] 50+ models benchmarked (matching macOS coverage)
- [ ] TensorRT integration functional
- [ ] Memory estimation accuracy <1.5% MAPE
- [ ] Statistical comparison published in paper

## Output Format

### Benchmark Results Structure
```
benchmark_results/
├── windows_cuda/
│   ├── resnet50_cuda.json
│   ├── efficientnet_b0_cuda.json
│   └── ...
├── windows_cpu/
│   ├── resnet50_cpu.json
│   └── ...
├── cross_platform_comparison.csv
└── windows_summary.md
```

### Paper Integration
Results will be added to:
1. **Table: Hardware Configuration** (Section 4.1) - ✅ Already added
2. **Table: Cross-Platform Performance** (New subsection)
3. **Figure: CPU Architecture Comparison** (i7 vs M2 Pro)
4. **Figure: GPU Acceleration** (RTX 3060 speedup)
5. **Appendix: Windows Validation Data**

## Timeline

### Week 1: Setup
- Install Python and PyTorch
- Configure CUDA environment
- Verify Rust builds

### Week 2: CPU Benchmarks
- Run classification benchmarks
- Collect throughput data
- Analyze P-core vs E-core usage

### Week 3: GPU Benchmarks
- CUDA acceleration tests
- TensorRT optimization
- Memory profiling

### Week 4: Analysis & Documentation
- Statistical analysis
- Cross-platform comparison
- Paper integration
- Supplementary material preparation

## Notes

### Known Issues
1. **jemalloc:** Not supported on Windows MSVC (requires Unix tools)
   - **Solution:** Use default system allocator or mimalloc
   - **Command:** Build with `--no-default-features --features torch`

2. **libtorch Path:** Must be set manually on Windows
   - **Solution:** Set `LIBTORCH` environment variable
   - **Verification:** `echo %LIBTORCH%` should point to PyTorch install

3. **CUDA Compatibility:** Ensure PyTorch CUDA version matches CUDA Toolkit
   - **Current:** CUDA 11.8 (pre-installed)
   - **Required:** PyTorch built for CUDA 11.8

### References
- [tch-rs Windows Setup](https://github.com/LaurentMazare/tch-rs#windows)
- [PyTorch Windows Installation](https://pytorch.org/get-started/locally/)
- [CUDA 11.8 Documentation](https://docs.nvidia.com/cuda/archive/11.8.0/)

## Contact
For questions or issues, contact the benchmark team or refer to project documentation.

---
**Document Version:** 1.0  
**Last Updated:** 2026-01-05  
**Status:** Setup Phase - Pending PyTorch Installation

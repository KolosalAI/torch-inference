# GPU Detection System

A comprehensive GPU detection and configuration system for the PyTorch Inference Framework that automatically detects available GPUs, analyzes their capabilities, and provides optimal configuration recommendations.

## Features

### 🔍 Comprehensive GPU Detection
- **NVIDIA CUDA GPUs**: Full support with compute capability analysis
- **AMD ROCm GPUs**: Detection and basic configuration
- **Apple Silicon (MPS)**: M1/M2/M3 chip detection and optimization
- **Intel GPUs**: XPU support detection
- **Multi-vendor support**: Automatic detection across different vendors

### 📊 Hardware Analysis
- **Memory Information**: Total, available, and usage statistics
- **Compute Capabilities**: FP16, INT8, Tensor Core support detection
- **Architecture Detection**: Automatic identification of GPU architectures
- **Performance Benchmarking**: Memory bandwidth and compute performance tests
- **Real-time Metrics**: Temperature, power consumption, utilization

### ⚙️ Automatic Configuration
- **Device Selection**: Automatically chooses the best available GPU
- **Precision Optimization**: Recommends optimal precision settings (FP32, FP16, INT8)
- **Memory Management**: Batch size recommendations based on available memory
- **Framework Integration**: Seamless PyTorch device configuration

### 🚀 Performance Optimization
- **CUDA Optimizations**: TensorRT, CUDA graphs, cuDNN optimizations
- **Memory Optimization**: Memory pool configuration and usage recommendations
- **Batch Size Estimation**: Automatic batch size recommendations
- **Benchmark Results**: Performance metrics for informed decisions

## Quick Start

### Basic GPU Detection

```python
from framework.core.gpu_detection import detect_gpus, print_gpu_report

# Simple detection
gpus = detect_gpus(enable_benchmarks=False)
print(f"Found {len(gpus)} GPU(s)")

# Detailed report
print_gpu_report(enable_benchmarks=True)
```

### Automatic Configuration

```python
from framework.core.gpu_manager import auto_configure_device
import torch

# Get optimized device configuration
device_config = auto_configure_device()
device = device_config.get_torch_device()

# Use the device
model = model.to(device)
data = data.to(device)
```

### GPU Manager

```python
from framework.core.gpu_manager import GPUManager

manager = GPUManager()
gpus, device_config = manager.detect_and_configure()

# Get recommendations
memory_rec = manager.get_memory_recommendations()
optimization_rec = manager.get_optimization_recommendations()
```

## Command Line Interface

### Basic Detection
```bash
python tools/gpu_detect.py
```

### Detailed Report with Recommendations
```bash
python tools/gpu_detect.py --detailed
```

### JSON Output
```bash
python tools/gpu_detect.py --json
```

### Configuration Only
```bash
python tools/gpu_detect.py --config-only
```

### With Benchmarks
```bash
python tools/gpu_detect.py --benchmark
```

## API Endpoints

When running the main server, the following GPU detection endpoints are available:

### Detect Available GPUs
```
GET /gpu/detect?include_benchmarks=false
```

### Get Best GPU
```
GET /gpu/best
```

### Get GPU Configuration
```
GET /gpu/config
```

### Get Comprehensive Report
```
GET /gpu/report?format=json
GET /gpu/report?format=text
```

## Examples

### Example 1: Basic Detection
```python
from framework.core.gpu_detection import GPUDetector

detector = GPUDetector(enable_benchmarks=False)
gpus = detector.detect_all_gpus()

for gpu in gpus:
    print(f"GPU: {gpu.name}")
    print(f"  Memory: {gpu.memory.total_mb:.0f} MB")
    print(f"  Suitable: {gpu.is_suitable_for_inference()}")
```

### Example 2: Model-Specific Optimization
```python
from framework.core.gpu_manager import GPUManager

manager = GPUManager()
best_gpu = manager.get_best_gpu_info()

if best_gpu:
    # Estimate batch size for a 500MB model
    max_batch_size = best_gpu.estimate_max_batch_size(model_size_mb=500)
    print(f"Recommended batch size: {max_batch_size}")
    
    # Get precision recommendations
    precisions = best_gpu.get_recommended_precision()
    print(f"Supported precisions: {precisions}")
```

### Example 3: Integration with Inference Config
```python
from framework.core.gpu_manager import GPUManager
from framework.core.config import InferenceConfig

manager = GPUManager()
_, device_config = manager.detect_and_configure()

# Create inference configuration
config = InferenceConfig()
config.device = device_config

# The device is now optimally configured
device = config.device.get_torch_device()
```

## GPU Information Structure

Each detected GPU provides comprehensive information:

```python
@dataclass
class GPUInfo:
    # Basic information
    id: int
    name: str
    vendor: GPUVendor  # NVIDIA, AMD, INTEL, APPLE, UNKNOWN
    architecture: GPUArchitecture  # AMPERE, TURING, VOLTA, etc.
    
    # Capabilities
    compute_capability: ComputeCapability  # CUDA compute capability
    supported_accelerators: List[AcceleratorType]  # CUDA, ROCm, MPS, etc.
    
    # Memory information
    memory: MemoryInfo  # Total, available, used memory
    
    # Performance metrics
    performance: PerformanceMetrics  # Utilization, temperature, power
    
    # Software support
    pytorch_support: bool
    tensorrt_support: bool
    
    # Benchmark results
    benchmark_results: Dict[str, Any]
```

## Supported GPU Architectures

### NVIDIA
- **Hopper** (H100, etc.) - Latest architecture with enhanced Transformer acceleration
- **Ada Lovelace** (RTX 40 series) - Latest gaming/workstation GPUs
- **Ampere** (RTX 30 series, A100) - Tensor Cores with sparsity support
- **Turing** (RTX 20 series) - First-gen RT and Tensor Cores
- **Volta** (V100) - First Tensor Core architecture
- **Pascal** (GTX 10 series) - Mainstream FP16 support
- **Maxwell/Kepler** - Legacy architectures

### AMD
- **RDNA3** (RX 7000 series) - Latest gaming architecture
- **RDNA2** (RX 6000 series) - Previous generation
- **CDNA2** (MI200 series) - Data center architecture

### Apple Silicon
- **M3** - Latest Apple Silicon with enhanced GPU
- **M2** - Second generation Apple Silicon
- **M1** - First generation Apple Silicon

### Intel
- **Xe HPG** (Arc series) - Gaming/workstation GPUs
- **Xe HPC** - Data center GPUs

## Optimization Recommendations

The system provides automatic optimization recommendations:

### Memory Optimization
- Batch size recommendations based on available memory
- Memory pool configuration suggestions
- Gradient checkpointing recommendations for large models

### Performance Optimization
- Precision format recommendations (FP32, FP16, INT8, TF32)
- CUDA graph usage suggestions
- cuDNN benchmark mode recommendations
- Tensor Core utilization advice

### Framework Integration
- PyTorch device configuration
- TensorRT optimization suggestions
- torch.compile recommendations

## Benchmarking

The system includes comprehensive benchmarking capabilities:

### Memory Bandwidth
- Sequential read/write performance
- Memory copy throughput

### Compute Performance
- FP32/FP16 arithmetic performance
- Matrix multiplication (GEMM) performance
- Element-wise operation throughput

### Real-world Workloads
- Convolution performance
- Transformer operation benchmarks
- Memory usage patterns

## Error Handling

The system gracefully handles various error conditions:

- **No GPU Available**: Falls back to CPU configuration
- **Driver Issues**: Provides diagnostic information
- **Memory Errors**: Suggests memory optimization strategies
- **Benchmark Failures**: Continues with basic detection

## Integration with Existing Framework

The GPU detection system integrates seamlessly with the existing PyTorch inference framework:

### Configuration System
```python
# The GPU manager automatically configures device settings
from framework.core.config_manager import get_config_manager
from framework.core.gpu_manager import get_gpu_manager

config_manager = get_config_manager()
gpu_manager = get_gpu_manager()

# GPU-optimized configuration is automatically applied
```

### Model Loading
```python
# Models automatically use the best available GPU
from framework.core.base_model import BaseModel

class MyModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Device is automatically configured based on GPU detection
```

### Inference Engine
```python
# Inference engine automatically uses optimal GPU configuration
from framework.core.inference_engine import create_inference_engine

engine = create_inference_engine(model, config)
# GPU optimization is automatically applied
```

## Testing

Run the test suite to verify GPU detection functionality:

```bash
# Basic functionality test
python test_gpu_detection.py

# Full test suite
python -m pytest tests/unit/test_gpu_detection.py -v

# Example usage
python examples/gpu_detection_examples.py --all
```

## Dependencies

### Required
- `torch` - PyTorch framework
- `dataclasses` - For GPU information structures (Python 3.7+)

### Optional
- `pynvml` - NVIDIA Management Library for detailed GPU metrics
- `psutil` - System information
- `GPUtil` - Additional GPU utilities

Install optional dependencies:
```bash
pip install pynvml psutil gputil
```

## Performance Impact

The GPU detection system is designed to be efficient:

- **Caching**: Results are cached for 5 minutes by default
- **Lazy Loading**: Benchmarks are only run when requested
- **Fast Detection**: Basic detection completes in under 1 second
- **Background Processing**: Benchmarks can run in background

## Configuration

The system can be configured through environment variables or configuration files:

```python
# Disable benchmarks for faster startup
detector = GPUDetector(enable_benchmarks=False)

# Custom benchmark duration
detector = GPUDetector(benchmark_duration=10.0)  # 10 seconds

# Force refresh cached results
gpus = detector.detect_all_gpus(force_refresh=True)
```

## Troubleshooting

### Common Issues

1. **CUDA Not Detected**
   - Verify NVIDIA drivers are installed
   - Check PyTorch CUDA installation: `torch.cuda.is_available()`

2. **Memory Information Missing**
   - Install `pynvml`: `pip install pynvml`
   - Check GPU driver version

3. **Benchmark Failures**
   - Reduce benchmark duration
   - Check available GPU memory
   - Verify PyTorch installation

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from framework.core.gpu_detection import GPUDetector
detector = GPUDetector()
```

## Contributing

To contribute to the GPU detection system:

1. Add support for new GPU vendors in `gpu_detection.py`
2. Extend architecture detection in `_detect_nvidia_architecture()`
3. Add new benchmark types in `_benchmark_gpu()`
4. Update tests in `tests/unit/test_gpu_detection.py`

## License

This GPU detection system is part of the PyTorch Inference Framework and follows the same license terms.

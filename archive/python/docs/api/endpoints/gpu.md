# GPU Management Endpoints

The GPU management endpoints provide comprehensive GPU detection, configuration, and optimization functionality for deep learning workloads.

## Overview

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|----------------|
| `/gpu/detect` | GET | Comprehensive GPU detection | None |
| `/gpu/best` | GET | Get best GPU recommendation | None |
| `/gpu/config` | GET | Current GPU configuration | None |
| `/gpu/report` | GET | Detailed GPU report | None |
| `/gpu/optimize` | POST | GPU optimization recommendations | None |
| `/gpu/memory` | GET | GPU memory usage | None |
| `/gpu/benchmark` | POST | GPU benchmark testing | None |

---

## GPU Detection

Comprehensive GPU detection with detailed hardware information and optimization recommendations.

### Request
```http
GET /gpu/detect
```

### Response
```json
{
  "success": true,
  "timestamp": "2024-01-15T10:30:00Z",
  "gpu_available": true,
  "cuda_available": true,
  "mps_available": false,
  "device_count": 2,
  "current_device": 0,
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "uuid": "GPU-12345678-1234-1234-1234-123456789abc",
      "architecture": "Ada Lovelace",
      "compute_capability": "8.9",
      "cuda_cores": 16384,
      "rt_cores": 128,
      "tensor_cores": 512,
      "memory": {
        "total_mb": 24576,
        "available_mb": 22528,
        "used_mb": 2048,
        "usage_percent": 8.3
      },
      "performance": {
        "base_clock_mhz": 1395,
        "boost_clock_mhz": 2520,
        "memory_clock_mhz": 10501,
        "memory_bandwidth_gbps": 1008,
        "fp32_tflops": 83.0,
        "tensor_tflops": 165.2
      },
      "power": {
        "tdp_watts": 450,
        "current_draw_watts": 125,
        "max_power_watts": 480,
        "power_efficiency": "excellent"
      },
      "temperature": {
        "current_c": 42,
        "max_c": 83,
        "fan_speed_percent": 35
      },
      "pcie": {
        "generation": "PCIe 4.0",
        "lanes": 16,
        "bandwidth_gbps": 64
      },
      "driver_version": "535.98",
      "cuda_version": "12.2",
      "status": "healthy",
      "recommended_for": ["training", "inference", "large_models"]
    }
  ],
  "system_info": {
    "cuda_runtime_version": "12.2",
    "cudnn_version": "8.9.2",
    "pytorch_version": "2.1.0+cu121",
    "driver_version": "535.98",
    "platform": "Windows-10-10.0.19041-SP0"
  },
  "optimization": {
    "recommended_device": 0,
    "multi_gpu_capable": true,
    "mixed_precision_supported": true,
    "gradient_checkpointing_recommended": true,
    "batch_size_recommendations": {
      "small_model": 64,
      "medium_model": 32,
      "large_model": 8,
      "xl_model": 2
    },
    "memory_optimization": {
      "enable_flash_attention": true,
      "use_gradient_accumulation": true,
      "recommended_accumulation_steps": 4
    }
  },
  "compatibility": {
    "huggingface_transformers": true,
    "pytorch_lightning": true,
    "deepspeed": true,
    "flash_attention": true,
    "xformers": true,
    "triton": true
  }
}
```

### Key Information Provided

#### Hardware Details
- **GPU Model**: Exact GPU name and specifications
- **Memory**: Total, available, and used VRAM
- **Compute Capability**: CUDA compute capability version
- **Performance Metrics**: Clock speeds, TFLOPS, bandwidth
- **Power Consumption**: Current draw, TDP, efficiency rating

#### Software Environment
- **CUDA Version**: Runtime and driver versions
- **PyTorch**: Version and CUDA compatibility
- **cuDNN**: Deep learning library version
- **Compatible Libraries**: Flash Attention, xFormers, etc.

#### Optimization Recommendations
- **Batch Size Suggestions**: Per model size category
- **Memory Optimization**: Gradient checkpointing, accumulation
- **Multi-GPU**: Support and recommendations
- **Mixed Precision**: FP16/BF16 capabilities

### Example
```bash
curl http://localhost:8000/gpu/detect
```

---

## Best GPU Recommendation

Get the optimal GPU selection for current workload.

### Request
```http
GET /gpu/best?task=inference&model_size=large&batch_size=16
```

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | string | "inference" | Task type (inference, training, both) |
| `model_size` | string | "medium" | Model size (small, medium, large, xl) |
| `batch_size` | integer | 1 | Target batch size |
| `precision` | string | "fp16" | Precision (fp32, fp16, bf16, int8) |
| `memory_usage` | string | "normal" | Memory usage (low, normal, high) |

### Response
```json
{
  "success": true,
  "recommended_gpu": 0,
  "recommendation_reason": "Best performance-to-memory ratio for large model inference",
  "gpu_info": {
    "id": 0,
    "name": "NVIDIA GeForce RTX 4090",
    "memory_available_mb": 22528,
    "compute_capability": "8.9",
    "performance_score": 95.2
  },
  "optimization_suggestions": {
    "use_mixed_precision": true,
    "recommended_precision": "fp16",
    "optimal_batch_size": 12,
    "enable_gradient_checkpointing": false,
    "use_flash_attention": true
  },
  "memory_estimate": {
    "model_memory_mb": 8192,
    "activation_memory_mb": 4096,
    "buffer_memory_mb": 2048,
    "total_required_mb": 14336,
    "safety_margin_mb": 8192,
    "memory_utilization_percent": 58.3
  },
  "performance_estimate": {
    "inference_time_ms": 125,
    "throughput_samples_per_second": 96,
    "tokens_per_second": 2400
  }
}
```

### Examples

#### Best GPU for Training
```bash
curl "http://localhost:8000/gpu/best?task=training&model_size=large&batch_size=8"
```

#### Best GPU for Inference
```bash
curl "http://localhost:8000/gpu/best?task=inference&model_size=xl&precision=fp16"
```

#### Best GPU for Memory-Constrained Task
```bash
curl "http://localhost:8000/gpu/best?memory_usage=low&model_size=medium"
```

---

## GPU Configuration

Get current GPU configuration and settings.

### Request
```http
GET /gpu/config
```

### Response
```json
{
  "success": true,
  "current_device": 0,
  "device_name": "NVIDIA GeForce RTX 4090",
  "configuration": {
    "mixed_precision_enabled": true,
    "precision_mode": "fp16",
    "gradient_checkpointing": false,
    "flash_attention_enabled": true,
    "memory_optimization": true,
    "multi_gpu_enabled": false,
    "dataloader_workers": 4
  },
  "memory_settings": {
    "memory_fraction": 0.9,
    "reserved_memory_mb": 2048,
    "cache_allocator": "native",
    "garbage_collection_threshold": 0.8
  },
  "compute_settings": {
    "allow_tf32": true,
    "cudnn_benchmark": true,
    "cudnn_deterministic": false,
    "compile_mode": "default"
  },
  "environment": {
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
    "CUDNN_BENCHMARK": "1"
  }
}
```

### Example
```bash
curl http://localhost:8000/gpu/config
```

---

## GPU Report

Generate a comprehensive GPU report with benchmarks and recommendations.

### Request
```http
GET /gpu/report?include_benchmarks=true&detailed=true
```

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_benchmarks` | boolean | false | Include performance benchmarks |
| `detailed` | boolean | true | Include detailed metrics |
| `format` | string | "json" | Report format (json, text) |

### Response
```json
{
  "success": true,
  "report_id": "gpu_report_20240115_103000",
  "timestamp": "2024-01-15T10:30:00Z",
  "summary": {
    "total_gpus": 2,
    "healthy_gpus": 2,
    "total_memory_gb": 48,
    "available_memory_gb": 44,
    "average_utilization": 15.2,
    "power_efficiency_rating": "excellent"
  },
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "health_status": "healthy",
      "performance_score": 95.2,
      "efficiency_score": 88.7,
      "memory_health": "good",
      "thermal_status": "normal",
      "benchmarks": {
        "matrix_multiply_gflops": 165200,
        "memory_bandwidth_gbps": 1008,
        "tensor_ops_tops": 1320,
        "fp16_performance_score": 98.5,
        "int8_performance_score": 96.8
      },
      "recommendations": [
        "Optimal for large model inference",
        "Consider enabling Flash Attention",
        "Suitable for mixed precision training",
        "Excellent for production workloads"
      ]
    }
  ],
  "system_recommendations": {
    "optimal_configuration": {
      "primary_gpu": 0,
      "enable_multi_gpu": true,
      "recommended_precision": "fp16",
      "batch_size_multiplier": 1.2
    },
    "performance_tips": [
      "Use mixed precision for 2x speedup",
      "Enable gradient checkpointing for memory efficiency",
      "Consider model sharding for very large models",
      "Use Flash Attention when possible"
    ],
    "memory_optimization": [
      "Enable gradient accumulation",
      "Use activation checkpointing",
      "Consider model parallelism",
      "Optimize batch sizes per GPU"
    ]
  },
  "compatibility_matrix": {
    "pytorch": "✅ Full support",
    "tensorflow": "✅ Full support", 
    "jax": "✅ Full support",
    "huggingface": "✅ Optimized",
    "deepspeed": "✅ Full support",
    "flash_attention": "✅ Native support"
  }
}
```

### Example
```bash
curl "http://localhost:8000/gpu/report?include_benchmarks=true&detailed=true"
```

---

## GPU Optimization

Get GPU optimization recommendations for specific workloads.

### Request
```http
POST /gpu/optimize
Content-Type: application/json

{
  "task": "training",
  "model_type": "transformer",
  "model_size": "7B",
  "sequence_length": 2048,
  "batch_size": 4,
  "precision": "fp16",
  "optimization_level": "aggressive"
}
```

### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task` | string | Yes | Task type (training, inference, fine-tuning) |
| `model_type` | string | Yes | Model architecture (transformer, cnn, rnn) |
| `model_size` | string | Yes | Model size (1B, 7B, 13B, 70B, etc.) |
| `sequence_length` | integer | No | Sequence length for transformers |
| `batch_size` | integer | No | Target batch size |
| `precision` | string | No | Desired precision (fp32, fp16, bf16, int8) |
| `optimization_level` | string | No | Optimization level (conservative, balanced, aggressive) |

### Response
```json
{
  "success": true,
  "optimization_recommendations": {
    "memory_optimizations": {
      "gradient_checkpointing": {
        "enabled": true,
        "strategy": "selective",
        "memory_savings_percent": 40,
        "speed_impact_percent": -15
      },
      "mixed_precision": {
        "enabled": true,
        "precision": "fp16",
        "autocast_enabled": true,
        "loss_scaling": "dynamic"
      },
      "gradient_accumulation": {
        "enabled": true,
        "steps": 8,
        "effective_batch_size": 32
      },
      "model_sharding": {
        "enabled": false,
        "reason": "Single GPU sufficient for 7B model"
      }
    },
    "performance_optimizations": {
      "flash_attention": {
        "enabled": true,
        "speedup_factor": 2.3,
        "memory_reduction_percent": 35
      },
      "torch_compile": {
        "enabled": true,
        "backend": "inductor",
        "expected_speedup": 1.8
      },
      "dataloader_optimization": {
        "num_workers": 8,
        "pin_memory": true,
        "prefetch_factor": 2
      }
    },
    "hardware_settings": {
      "gpu_selection": [0],
      "memory_fraction": 0.9,
      "allow_tf32": true,
      "cudnn_benchmark": true
    }
  },
  "estimated_improvements": {
    "memory_usage_reduction_percent": 45,
    "training_speed_improvement_percent": 25,
    "convergence_stability": "improved",
    "power_efficiency_gain_percent": 15
  },
  "implementation_code": {
    "pytorch": "# PyTorch optimization code\nimport torch\ntorch.backends.cudnn.benchmark = True\ntorch.backends.cuda.matmul.allow_tf32 = True",
    "transformers": "# HuggingFace Transformers optimization\nfrom transformers import TrainingArguments\nargs = TrainingArguments(\n    fp16=True,\n    gradient_checkpointing=True,\n    dataloader_num_workers=8\n)"
  }
}
```

### Examples

#### Optimize for Large Model Training
```bash
curl -X POST http://localhost:8000/gpu/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "task": "training",
    "model_type": "transformer",
    "model_size": "13B",
    "sequence_length": 4096,
    "batch_size": 2,
    "optimization_level": "aggressive"
  }'
```

#### Optimize for Inference
```bash
curl -X POST http://localhost:8000/gpu/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "task": "inference",
    "model_type": "transformer",
    "model_size": "7B",
    "precision": "int8",
    "optimization_level": "balanced"
  }'
```

---

## GPU Memory Usage

Monitor GPU memory usage and allocation.

### Request
```http
GET /gpu/memory?detailed=true
```

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detailed` | boolean | false | Include detailed memory breakdown |
| `gpu_id` | integer | null | Specific GPU ID (all if not specified) |

### Response
```json
{
  "success": true,
  "timestamp": "2024-01-15T10:30:00Z",
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "memory": {
        "total_mb": 24576,
        "allocated_mb": 8192,
        "reserved_mb": 10240,
        "free_mb": 14336,
        "utilization_percent": 33.3
      },
      "allocation_details": {
        "model_weights_mb": 6144,
        "activations_mb": 1536,
        "gradients_mb": 512,
        "optimizer_states_mb": 0,
        "cache_mb": 2048,
        "overhead_mb": 256
      },
      "memory_efficiency": {
        "fragmentation_percent": 5.2,
        "peak_usage_mb": 12288,
        "allocation_count": 1247,
        "deallocation_count": 1198
      }
    }
  ],
  "system_memory": {
    "total_system_ram_gb": 64,
    "available_system_ram_gb": 48,
    "swap_usage_gb": 0.5
  },
  "recommendations": [
    "Memory usage is within optimal range",
    "Consider enabling memory pooling",
    "Fragmentation is acceptable",
    "Peak usage suggests room for larger batches"
  ]
}
```

### Example
```bash
curl "http://localhost:8000/gpu/memory?detailed=true"
```

---

## GPU Benchmark

Run GPU benchmarks to assess performance.

### Request
```http
POST /gpu/benchmark
Content-Type: application/json

{
  "benchmark_type": "comprehensive",
  "duration_seconds": 30,
  "include_memory_test": true,
  "test_mixed_precision": true,
  "gpu_ids": [0]
}
```

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `benchmark_type` | string | "quick" | Type (quick, standard, comprehensive) |
| `duration_seconds` | integer | 10 | Benchmark duration |
| `include_memory_test` | boolean | true | Include memory bandwidth tests |
| `test_mixed_precision` | boolean | true | Test FP16/BF16 performance |
| `gpu_ids` | array | null | Specific GPUs to benchmark |

### Response
```json
{
  "success": true,
  "benchmark_id": "bench_20240115_103000",
  "duration_seconds": 30,
  "results": {
    "gpu_0": {
      "name": "NVIDIA GeForce RTX 4090",
      "compute_performance": {
        "fp32_gflops": 83000,
        "fp16_gflops": 165200,
        "int8_tops": 1320,
        "tensor_tflops": 165.2
      },
      "memory_performance": {
        "bandwidth_gbps": 1008,
        "latency_ns": 150,
        "random_access_mbps": 890000,
        "sequential_access_mbps": 1008000
      },
      "ml_workloads": {
        "matrix_multiply_ms": 12.5,
        "convolution_ms": 8.3,
        "transformer_attention_ms": 15.2,
        "batch_normalization_ms": 2.1
      },
      "temperature": {
        "initial_c": 35,
        "peak_c": 68,
        "final_c": 52
      },
      "power_consumption": {
        "average_watts": 380,
        "peak_watts": 425,
        "efficiency_gflops_per_watt": 218
      },
      "stability": {
        "error_rate": 0.0,
        "thermal_throttling": false,
        "memory_errors": 0
      }
    }
  },
  "comparison": {
    "performance_percentile": 95,
    "efficiency_rating": "excellent",
    "recommended_for": [
      "Large model training",
      "High-throughput inference",
      "Research workloads"
    ]
  }
}
```

### Examples

#### Quick Benchmark
```bash
curl -X POST http://localhost:8000/gpu/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "benchmark_type": "quick",
    "duration_seconds": 10
  }'
```

#### Comprehensive Benchmark
```bash
curl -X POST http://localhost:8000/gpu/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "benchmark_type": "comprehensive",
    "duration_seconds": 60,
    "include_memory_test": true,
    "test_mixed_precision": true
  }'
```

---

## GPU Architecture Support

### NVIDIA GPUs

#### Ada Lovelace (RTX 40 Series)
- **Models**: RTX 4090, RTX 4080, RTX 4070 Ti, RTX 4070, RTX 4060 Ti, RTX 4060
- **Compute Capability**: 8.9
- **Key Features**: 
  - 3rd gen RT cores
  - 4th gen Tensor cores
  - AV1 encoding
  - PCIe 4.0
- **Recommended For**: Latest AI workloads, highest performance

#### Ampere (RTX 30 Series, A100, H100)
- **Models**: RTX 3090, RTX 3080, A100, H100
- **Compute Capability**: 8.0-9.0
- **Key Features**:
  - 2nd/3rd gen RT cores
  - 3rd gen Tensor cores
  - Multi-Instance GPU (MIG)
- **Recommended For**: Training, professional workloads

#### Turing (RTX 20 Series)
- **Models**: RTX 2080 Ti, RTX 2080, RTX 2070
- **Compute Capability**: 7.5
- **Key Features**:
  - 1st gen RT cores
  - 2nd gen Tensor cores
- **Recommended For**: Inference, moderate training

#### Pascal (GTX 10 Series)
- **Models**: GTX 1080 Ti, GTX 1080, GTX 1070
- **Compute Capability**: 6.1
- **Key Features**:
  - FP16 support (limited)
  - High memory bandwidth
- **Recommended For**: Budget inference, legacy support

### AMD GPUs (ROCm Support)
- **MI250X**: Data center GPU with 128GB HBM2e
- **MI210**: 64GB HBM2e, excellent for training
- **RX 7900 XTX**: Consumer GPU with ROCm support
- **RX 6900 XT**: Older generation with basic support

### Intel GPUs (XPU Support)
- **Arc A770**: Consumer GPU with AI acceleration
- **Data Center GPU Max**: Professional AI workloads
- **Xe-HPG**: Gaming GPUs with compute capabilities

---

## Performance Optimization Guidelines

### Memory Optimization

#### Gradient Checkpointing
```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Or for specific modules
model.transformer.gradient_checkpointing = True
```

#### Mixed Precision Training
```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

# Training loop with mixed precision
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### Memory-Efficient Attention
```python
# Flash Attention
from flash_attn import flash_attn_func

# xFormers memory efficient attention
from xformers.ops import memory_efficient_attention
```

### Compute Optimization

#### Torch Compile
```python
# Compile model for faster inference
model = torch.compile(model, backend="inductor")

# For training
model = torch.compile(model, mode="max-autotune")
```

#### CUDA Optimization
```python
# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set memory allocation strategy
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

### Multi-GPU Setup

#### Data Parallel
```python
# Simple data parallel
model = torch.nn.DataParallel(model)

# Distributed data parallel
model = torch.nn.parallel.DistributedDataParallel(model)
```

#### Model Parallel
```python
# Manual model parallel
class ModelParallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000).cuda(0)
        self.layer2 = nn.Linear(1000, 1000).cuda(1)
    
    def forward(self, x):
        x = self.layer1(x.cuda(0))
        x = self.layer2(x.cuda(1))
        return x
```

---

## Error Handling

### Common GPU Errors

#### CUDA Out of Memory
```json
{
  "success": false,
  "error": "CUDA out of memory",
  "message": "GPU 0 ran out of memory during operation",
  "suggestions": [
    "Reduce batch size",
    "Enable gradient checkpointing", 
    "Use mixed precision training",
    "Clear GPU cache with torch.cuda.empty_cache()"
  ],
  "memory_info": {
    "allocated_mb": 24576,
    "total_mb": 24576,
    "reserved_mb": 24576
  }
}
```

#### No GPU Available
```json
{
  "success": false,
  "error": "No GPU available",
  "message": "No CUDA-capable GPU devices found",
  "suggestions": [
    "Check GPU drivers",
    "Verify CUDA installation",
    "Ensure GPU is not being used by another process",
    "Check CUDA_VISIBLE_DEVICES environment variable"
  ]
}
```

#### Incompatible GPU
```json
{
  "success": false,
  "error": "Incompatible GPU",
  "message": "GPU compute capability 3.5 is below minimum required 6.0",
  "gpu_info": {
    "name": "GeForce GTX 780",
    "compute_capability": "3.5",
    "minimum_required": "6.0"
  },
  "suggestions": [
    "Upgrade to a newer GPU",
    "Use CPU inference instead",
    "Check for legacy PyTorch versions"
  ]
}
```

---

## Integration Examples

### Python GPU Management
```python
import requests
import torch

class GPUManager:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def detect_gpus(self):
        """Get comprehensive GPU information"""
        response = requests.get(f"{self.base_url}/gpu/detect")
        return response.json()
    
    def get_best_gpu(self, task="inference", model_size="medium"):
        """Get best GPU recommendation"""
        params = {"task": task, "model_size": model_size}
        response = requests.get(f"{self.base_url}/gpu/best", params=params)
        return response.json()
    
    def optimize_for_workload(self, workload_config):
        """Get optimization recommendations"""
        response = requests.post(
            f"{self.base_url}/gpu/optimize",
            json=workload_config
        )
        return response.json()
    
    def monitor_memory(self):
        """Monitor GPU memory usage"""
        response = requests.get(f"{self.base_url}/gpu/memory?detailed=true")
        return response.json()
    
    def run_benchmark(self, benchmark_type="standard"):
        """Run GPU benchmark"""
        config = {"benchmark_type": benchmark_type, "duration_seconds": 30}
        response = requests.post(f"{self.base_url}/gpu/benchmark", json=config)
        return response.json()

# Usage example
gpu_manager = GPUManager()

# Detect GPUs
gpu_info = gpu_manager.detect_gpus()
print(f"Found {gpu_info['device_count']} GPUs")

# Get optimization recommendations
workload = {
    "task": "training",
    "model_type": "transformer", 
    "model_size": "7B",
    "precision": "fp16"
}
optimizations = gpu_manager.optimize_for_workload(workload)

# Apply PyTorch optimizations
if optimizations["success"]:
    opts = optimizations["optimization_recommendations"]
    
    # Configure mixed precision
    if opts["memory_optimizations"]["mixed_precision"]["enabled"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        
    # Configure memory settings
    if opts["hardware_settings"]["allow_tf32"]:
        torch.backends.cudnn.allow_tf32 = True
        
    # Set GPU
    device_id = opts["hardware_settings"]["gpu_selection"][0]
    torch.cuda.set_device(device_id)
```

### Monitoring Script
```bash
#!/bin/bash

# GPU Monitoring Script
BASE_URL="http://localhost:8000"

echo "=== GPU Status Report ==="
echo "Timestamp: $(date)"
echo

# Basic GPU detection
echo "--- GPU Detection ---"
curl -s "$BASE_URL/gpu/detect" | jq -r '.gpus[] | "GPU \(.id): \(.name) - \(.memory.available_mb)MB available"'
echo

# Memory usage
echo "--- Memory Usage ---"
curl -s "$BASE_URL/gpu/memory?detailed=true" | jq -r '.gpus[] | "GPU \(.id): \(.memory.utilization_percent)% used (\(.memory.allocated_mb)MB/\(.memory.total_mb)MB)"'
echo

# Health check
echo "--- Health Status ---"
curl -s "$BASE_URL/gpu/report" | jq -r '.gpus[] | "GPU \(.id): \(.health_status) (Score: \(.performance_score))"'
echo

echo "=== End Report ==="
```

### Configuration Management
```python
import json
import requests

def configure_gpu_environment():
    """Configure optimal GPU environment based on detection"""
    
    # Get GPU information
    gpu_info = requests.get("http://localhost:8000/gpu/detect").json()
    
    if not gpu_info["gpu_available"]:
        print("No GPU available, configuring CPU-only environment")
        return {"device": "cpu"}
    
    # Get best GPU
    best_gpu = requests.get(
        "http://localhost:8000/gpu/best?task=inference&model_size=large"
    ).json()
    
    # Configure environment variables
    gpu_id = best_gpu["recommended_gpu"]
    config = {
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
        "device": f"cuda:{gpu_id}",
        "mixed_precision": best_gpu["optimization_suggestions"]["use_mixed_precision"],
        "flash_attention": best_gpu["optimization_suggestions"]["use_flash_attention"]
    }
    
    # Apply environment variables
    import os
    for key, value in config.items():
        if key.isupper():
            os.environ[key] = str(value)
    
    print(f"Configured for GPU {gpu_id}: {gpu_info['gpus'][gpu_id]['name']}")
    return config

# Usage
config = configure_gpu_environment()
```

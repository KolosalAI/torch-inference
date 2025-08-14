"""
Optimization modules for PyTorch inference acceleration.

This package contains various optimization techniques including:
- TensorRT integration
- ONNX runtime support
- Model quantization
- Memory optimizations
- CUDA optimizations
"""

import logging

logger = logging.getLogger(__name__)

# Import optimizers with error handling
_available_optimizers = []
_unavailable_optimizers = []

# TensorRT optimizer
try:
    from .tensorrt_optimizer import TensorRTOptimizer, convert_to_tensorrt
    _available_optimizers.extend(['TensorRTOptimizer', 'convert_to_tensorrt'])
except ImportError as e:
    logger.info(f"TensorRT optimizer not available: {e}")
    TensorRTOptimizer = None
    convert_to_tensorrt = None
    _unavailable_optimizers.extend(['TensorRTOptimizer', 'convert_to_tensorrt'])

# ONNX optimizer
try:
    from .onnx_optimizer import ONNXOptimizer, convert_to_onnx
    _available_optimizers.extend(['ONNXOptimizer', 'convert_to_onnx'])
except ImportError as e:
    logger.info(f"ONNX optimizer not available: {e}")
    ONNXOptimizer = None
    convert_to_onnx = None
    _unavailable_optimizers.extend(['ONNXOptimizer', 'convert_to_onnx'])

# Quantization optimizer
try:
    from .quantization_optimizer import QuantizationOptimizer, quantize_model
    _available_optimizers.extend(['QuantizationOptimizer', 'quantize_model'])
except ImportError as e:
    logger.info(f"Quantization optimizer not available: {e}")
    QuantizationOptimizer = None
    quantize_model = None
    _unavailable_optimizers.extend(['QuantizationOptimizer', 'quantize_model'])

# Memory optimizer
try:
    from .memory_optimizer import MemoryPool, MemoryOptimizer
    _available_optimizers.extend(['MemoryPool', 'MemoryOptimizer'])
except ImportError as e:
    logger.info(f"Memory optimizer not available: {e}")
    MemoryPool = None
    MemoryOptimizer = None
    _unavailable_optimizers.extend(['MemoryPool', 'MemoryOptimizer'])

# CUDA optimizer
try:
    from .cuda_optimizer import CUDAOptimizer, enable_cuda_optimizations
    _available_optimizers.extend(['CUDAOptimizer', 'enable_cuda_optimizations'])
except ImportError as e:
    logger.info(f"CUDA optimizer not available: {e}")
    CUDAOptimizer = None
    enable_cuda_optimizations = None
    _unavailable_optimizers.extend(['CUDAOptimizer', 'enable_cuda_optimizations'])

# JIT optimizer
try:
    from .jit_optimizer import JITOptimizer, jit_compile_model
    _available_optimizers.extend(['JITOptimizer', 'jit_compile_model'])
except ImportError as e:
    logger.info(f"JIT optimizer not available: {e}")
    JITOptimizer = None
    jit_compile_model = None
    _unavailable_optimizers.extend(['JITOptimizer', 'jit_compile_model'])

# Log availability summary
if _available_optimizers:
    logger.info(f"Available optimizers: {', '.join(_available_optimizers)}")
if _unavailable_optimizers:
    logger.info(f"Unavailable optimizers: {', '.join(_unavailable_optimizers)}")

__all__ = [
    'TensorRTOptimizer',
    'convert_to_tensorrt',
    'ONNXOptimizer', 
    'convert_to_onnx',
    'QuantizationOptimizer',
    'quantize_model',
    'MemoryPool',
    'MemoryOptimizer',
    'CUDAOptimizer',
    'enable_cuda_optimizations',
    'JITOptimizer',
    'jit_compile_model'
]

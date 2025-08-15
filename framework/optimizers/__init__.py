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

# INT8 Calibration optimizer
try:
    from .int8_calibration import INT8CalibrationToolkit, CalibrationConfig, get_calibration_toolkit
    _available_optimizers.extend(['INT8CalibrationToolkit', 'CalibrationConfig', 'get_calibration_toolkit'])
except ImportError as e:
    logger.info(f"INT8 calibration toolkit not available: {e}")
    INT8CalibrationToolkit = None
    CalibrationConfig = None
    get_calibration_toolkit = None
    _unavailable_optimizers.extend(['INT8CalibrationToolkit', 'CalibrationConfig', 'get_calibration_toolkit'])

# Kernel auto-tuner
try:
    from .kernel_autotuner import KernelAutoTuner, HardwareProfiler, get_kernel_auto_tuner, auto_tune_model
    _available_optimizers.extend(['KernelAutoTuner', 'HardwareProfiler', 'get_kernel_auto_tuner', 'auto_tune_model'])
except ImportError as e:
    logger.info(f"Kernel auto-tuner not available: {e}")
    KernelAutoTuner = None
    HardwareProfiler = None
    get_kernel_auto_tuner = None
    auto_tune_model = None
    _unavailable_optimizers.extend(['KernelAutoTuner', 'HardwareProfiler', 'get_kernel_auto_tuner', 'auto_tune_model'])

# Advanced layer fusion
try:
    from .advanced_fusion import AdvancedLayerFusion, FusionPattern, get_advanced_layer_fusion, optimize_model_fusion
    _available_optimizers.extend(['AdvancedLayerFusion', 'FusionPattern', 'get_advanced_layer_fusion', 'optimize_model_fusion'])
except ImportError as e:
    logger.info(f"Advanced layer fusion not available: {e}")
    AdvancedLayerFusion = None
    FusionPattern = None
    get_advanced_layer_fusion = None
    optimize_model_fusion = None
    _unavailable_optimizers.extend(['AdvancedLayerFusion', 'FusionPattern', 'get_advanced_layer_fusion', 'optimize_model_fusion'])

# Memory optimizer (enhanced)
try:
    from .memory_optimizer import MemoryPool, MemoryOptimizer, AdvancedMemoryPool
    _available_optimizers.extend(['AdvancedMemoryPool'])
except ImportError as e:
    logger.info(f"Enhanced memory optimizer not available: {e}")
    AdvancedMemoryPool = None
    _unavailable_optimizers.extend(['AdvancedMemoryPool'])

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
    'AdvancedMemoryPool',
    'CUDAOptimizer',
    'enable_cuda_optimizations',
    'JITOptimizer',
    'jit_compile_model',
    'INT8CalibrationToolkit',
    'CalibrationConfig', 
    'get_calibration_toolkit',
    'KernelAutoTuner',
    'HardwareProfiler',
    'get_kernel_auto_tuner',
    'auto_tune_model',
    'AdvancedLayerFusion',
    'FusionPattern',
    'get_advanced_layer_fusion',
    'optimize_model_fusion'
]

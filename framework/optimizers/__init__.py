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

# Model Compression Suite (comprehensive HLRTF-inspired optimization)
try:
    from .model_compression_suite import ModelCompressionSuite, ModelCompressionConfig, CompressionMethod, CompressionTarget, KnowledgeDistillationTrainer, MultiObjectiveOptimizer, compress_model_comprehensive
    # Add aliases for convenience
    CompressionConfig = ModelCompressionConfig
    compress_model = compress_model_comprehensive
    _available_optimizers.extend(['ModelCompressionSuite', 'ModelCompressionConfig', 'CompressionMethod', 'CompressionTarget', 'KnowledgeDistillationTrainer', 'MultiObjectiveOptimizer', 'compress_model_comprehensive', 'CompressionConfig', 'compress_model'])
except ImportError as e:
    logger.info(f"Model compression suite not available: {e}")
    ModelCompressionSuite = None
    ModelCompressionConfig = None
    CompressionConfig = None
    CompressionMethod = None
    CompressionTarget = None
    KnowledgeDistillationTrainer = None
    MultiObjectiveOptimizer = None
    compress_model_comprehensive = None
    compress_model = None
    _unavailable_optimizers.extend(['ModelCompressionSuite', 'ModelCompressionConfig', 'CompressionMethod', 'CompressionTarget', 'KnowledgeDistillationTrainer', 'MultiObjectiveOptimizer', 'compress_model_comprehensive'])

# Tensor Factorization optimizer (HLRTF-inspired)
try:
    from .tensor_factorization_optimizer import TensorFactorizationOptimizer, TensorFactorizationConfig, HierarchicalTensorLayer, factorize_model
    # Add alias for convenience
    optimize_model_with_tensor_factorization = factorize_model
    _available_optimizers.extend(['TensorFactorizationOptimizer', 'TensorFactorizationConfig', 'HierarchicalTensorLayer', 'factorize_model', 'optimize_model_with_tensor_factorization'])
except ImportError as e:
    logger.info(f"Tensor factorization optimizer not available: {e}")
    TensorFactorizationOptimizer = None
    TensorFactorizationConfig = None
    HierarchicalTensorLayer = None
    factorize_model = None
    optimize_model_with_tensor_factorization = None
    _unavailable_optimizers.extend(['TensorFactorizationOptimizer', 'TensorFactorizationConfig', 'HierarchicalTensorLayer', 'factorize_model'])

# Memory optimizer (enhanced)
try:
    from .memory_optimizer import MemoryPool, MemoryOptimizer, AdvancedMemoryPool
    _available_optimizers.extend(['AdvancedMemoryPool'])
except ImportError as e:
    logger.info(f"Enhanced memory optimizer not available: {e}")
    AdvancedMemoryPool = None
    _unavailable_optimizers.extend(['AdvancedMemoryPool'])

# Mask-based structured pruning
try:
    from .mask_based_pruning import (
        MaskBasedStructuredPruning,
        MaskPruningConfig, 
        MaskedConv2d,
        MaskedLinear,
        prune_model_with_masks
    )
    _available_optimizers.extend([
        "MaskBasedStructuredPruning", "MaskPruningConfig", 
        "MaskedConv2d", "MaskedLinear", "prune_model_with_masks"
    ])
except ImportError as e:
    logger.warning(f"Could not import mask-based pruning optimizers: {e}")
    MaskBasedStructuredPruning = None
    MaskPruningConfig = None
    MaskedConv2d = None
    MaskedLinear = None
    prune_model_with_masks = None
    _unavailable_optimizers.extend(['MaskBasedStructuredPruning', 'MaskPruningConfig', 'MaskedConv2d', 'MaskedLinear', 'prune_model_with_masks'])

# Structured pruning optimizer (HLRTF-inspired)
try:
    from .structured_pruning_optimizer import (
        StructuredPruningOptimizer,
        StructuredPruningConfig,
        ChannelImportanceCalculator,
        LowRankRegularizer,
        prune_model
    )
    _available_optimizers.extend([
        "StructuredPruningOptimizer", "StructuredPruningConfig",
        "ChannelImportanceCalculator", "LowRankRegularizer", "prune_model"
    ])
except ImportError as e:
    logger.warning(f"Could not import structured pruning optimizers: {e}")
    StructuredPruningOptimizer = None
    StructuredPruningConfig = None
    ChannelImportanceCalculator = None
    LowRankRegularizer = None
    prune_model = None
    _unavailable_optimizers.extend(['StructuredPruningOptimizer', 'StructuredPruningConfig', 'ChannelImportanceCalculator', 'LowRankRegularizer', 'prune_model'])

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
    'optimize_model_fusion',
    # HLRTF-inspired optimizers
    'TensorFactorizationOptimizer',
    'TensorFactorizationConfig',
    'HierarchicalTensorLayer',
    'factorize_model',
    'optimize_model_with_tensor_factorization',
    'StructuredPruningOptimizer',
    'StructuredPruningConfig',
    'ChannelImportanceCalculator',
    'LowRankRegularizer',
    'prune_model',
    'ModelCompressionSuite',
    'ModelCompressionConfig',
    'CompressionConfig',
    'CompressionMethod',
    'CompressionTarget',
    'KnowledgeDistillationTrainer',
    'MultiObjectiveOptimizer',
    'compress_model_comprehensive',
    'compress_model',
    # Mask-based pruning
    'MaskBasedStructuredPruning',
    'MaskPruningConfig',
    'MaskedConv2d',
    'MaskedLinear',
    'prune_model_with_masks'
]

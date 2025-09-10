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

# Enhanced optimizers (Vulkan, Numba, Enhanced JIT)
try:
    from .vulkan_optimizer import VulkanOptimizer, VulkanDeviceInfo, VULKAN_AVAILABLE
    _available_optimizers.extend(['VulkanOptimizer', 'VulkanDeviceInfo'])
    __vulkan_available__ = True
except ImportError as e:
    logger.info(f"Vulkan optimizer not available: {e}")
    VulkanOptimizer = None
    VulkanDeviceInfo = None
    VULKAN_AVAILABLE = False
    __vulkan_available__ = False
    _unavailable_optimizers.extend(['VulkanOptimizer', 'VulkanDeviceInfo'])

try:
    from .numba_optimizer import NumbaOptimizer, NUMBA_AVAILABLE, NUMBA_CUDA_AVAILABLE
    _available_optimizers.extend(['NumbaOptimizer'])
    __numba_available__ = True
except ImportError as e:
    logger.info(f"Numba optimizer not available: {e}")
    NumbaOptimizer = None
    NUMBA_AVAILABLE = False
    NUMBA_CUDA_AVAILABLE = False
    __numba_available__ = False
    _unavailable_optimizers.extend(['NumbaOptimizer'])

try:
    from .jit_optimizer import EnhancedJITOptimizer
    _available_optimizers.extend(['EnhancedJITOptimizer'])
    __enhanced_jit_available__ = True
except ImportError as e:
    logger.info(f"Enhanced JIT optimizer not available: {e}")
    EnhancedJITOptimizer = None
    __enhanced_jit_available__ = False
    _unavailable_optimizers.extend(['EnhancedJITOptimizer'])

try:
    from .performance_optimizer import PerformanceOptimizer
    _available_optimizers.extend(['PerformanceOptimizer'])
except ImportError as e:
    logger.info(f"Performance optimizer not available: {e}")
    PerformanceOptimizer = None
    _unavailable_optimizers.extend(['PerformanceOptimizer'])

# JIT optimizer
try:
    from .jit_optimizer import JITOptimizer, jit_compile_model
    _available_optimizers.extend(['JITOptimizer', 'jit_compile_model'])
except ImportError as e:
    logger.info(f"JIT optimizer not available: {e}")
    JITOptimizer = None
    jit_compile_model = None
    _unavailable_optimizers.extend(['JITOptimizer', 'jit_compile_model'])

# Post-download optimizer
try:
    from .post_download_optimizer import PostDownloadOptimizer, create_post_download_optimizer, optimize_downloaded_model
    _available_optimizers.extend(['PostDownloadOptimizer', 'create_post_download_optimizer', 'optimize_downloaded_model'])
except ImportError as e:
    logger.info(f"Post-download optimizer not available: {e}")
    PostDownloadOptimizer = None
    create_post_download_optimizer = None
    optimize_downloaded_model = None
    _unavailable_optimizers.extend(['PostDownloadOptimizer', 'create_post_download_optimizer', 'optimize_downloaded_model'])

# Comprehensive optimizer suite (next_steps implementation)
try:
    from .comprehensive_optimizer import ComprehensiveOptimizationSuite, optimize_model_comprehensive
    _available_optimizers.extend(['ComprehensiveOptimizationSuite', 'optimize_model_comprehensive'])
except ImportError as e:
    logger.info(f"Comprehensive optimizer suite not available: {e}")
    ComprehensiveOptimizationSuite = None
    optimize_model_comprehensive = None
    _unavailable_optimizers.extend(['ComprehensiveOptimizationSuite', 'optimize_model_comprehensive'])

# Attention optimizer (Flash Attention implementation)
try:
    from .attention_optimizer import FlashAttentionOptimizer, OptimizedMultiheadAttention, optimize_attention_layers, create_optimized_attention
    _available_optimizers.extend(['FlashAttentionOptimizer', 'OptimizedMultiheadAttention', 'optimize_attention_layers', 'create_optimized_attention'])
except ImportError as e:
    logger.info(f"Attention optimizer not available: {e}")
    FlashAttentionOptimizer = None
    OptimizedMultiheadAttention = None
    optimize_attention_layers = None
    create_optimized_attention = None
    _unavailable_optimizers.extend(['FlashAttentionOptimizer', 'OptimizedMultiheadAttention', 'optimize_attention_layers', 'create_optimized_attention'])

# Universal optimization engine
try:
    from .performance_optimizer import UniversalOptimizationEngine
    _available_optimizers.extend(['UniversalOptimizationEngine'])
except ImportError as e:
    logger.info(f"Universal optimization engine not available: {e}")
    UniversalOptimizationEngine = None
    _unavailable_optimizers.extend(['UniversalOptimizationEngine'])

# Log availability summary
if _available_optimizers:
    logger.info(f"Available optimizers: {', '.join(_available_optimizers)}")
if _unavailable_optimizers:
    logger.info(f"Unavailable optimizers: {', '.join(_unavailable_optimizers)}")

def get_available_optimizers():
    """Get a dictionary of available optimizers and their status."""
    optimizers = {
        'performance': {'available': PerformanceOptimizer is not None, 'class': 'PerformanceOptimizer'},
        'comprehensive': {'available': ComprehensiveOptimizationSuite is not None, 'class': 'ComprehensiveOptimizationSuite'},
        'universal_engine': {'available': UniversalOptimizationEngine is not None, 'class': 'UniversalOptimizationEngine'},
        'attention': {'available': FlashAttentionOptimizer is not None, 'class': 'FlashAttentionOptimizer'},
        'tensorrt': {'available': TensorRTOptimizer is not None, 'class': 'TensorRTOptimizer'},
        'onnx': {'available': ONNXOptimizer is not None, 'class': 'ONNXOptimizer'},
        'quantization': {'available': QuantizationOptimizer is not None, 'class': 'QuantizationOptimizer'},
        'memory': {'available': MemoryOptimizer is not None, 'class': 'MemoryOptimizer'},
        'cuda': {'available': CUDAOptimizer is not None, 'class': 'CUDAOptimizer'},
        'jit': {'available': JITOptimizer is not None, 'class': 'JITOptimizer'},
        'enhanced_jit': {'available': __enhanced_jit_available__, 'class': 'EnhancedJITOptimizer'},
        'vulkan': {'available': __vulkan_available__, 'class': 'VulkanOptimizer'},
        'numba': {'available': __numba_available__, 'class': 'NumbaOptimizer'},
        'calibration': {'available': INT8CalibrationToolkit is not None, 'class': 'INT8CalibrationToolkit'},
        'auto_tuning': {'available': KernelAutoTuner is not None, 'class': 'KernelAutoTuner'},
        'layer_fusion': {'available': AdvancedLayerFusion is not None, 'class': 'AdvancedLayerFusion'},
        'tensor_factorization': {'available': TensorFactorizationOptimizer is not None, 'class': 'TensorFactorizationOptimizer'},
        'structured_pruning': {'available': StructuredPruningOptimizer is not None, 'class': 'StructuredPruningOptimizer'},
        'model_compression': {'available': ModelCompressionSuite is not None, 'class': 'ModelCompressionSuite'},
        'mask_pruning': {'available': MaskBasedStructuredPruning is not None, 'class': 'MaskBasedStructuredPruning'},
        'post_download': {'available': PostDownloadOptimizer is not None, 'class': 'PostDownloadOptimizer'}
    }
    return optimizers

def get_optimization_recommendations(device='auto', model_size='medium', target='inference'):
    """Get recommended optimizers based on device, model size, and target use case."""
    available = get_available_optimizers()
    recommendations = []
    
    # Always recommend post-download optimization for downloaded models
    if available['post_download']['available']:
        recommendations.append(('post_download', 'Automatic optimization after model download (quantization + tensor factorization)'))
    
    # Always recommend basic optimizations
    if available['memory']['available']:
        recommendations.append(('memory', 'Essential for memory management'))
    
    # Device-specific recommendations
    if device == 'auto' or 'cuda' in str(device).lower():
        if available['cuda']['available']:
            recommendations.append(('cuda', 'CUDA optimizations for GPU acceleration'))
        if available['tensorrt']['available']:
            recommendations.append(('tensorrt', 'TensorRT for NVIDIA GPU optimization'))
    
    # Enhanced JIT for all scenarios
    if available['enhanced_jit']['available']:
        recommendations.append(('enhanced_jit', 'Enhanced JIT compilation with multi-backend support'))
    elif available['jit']['available']:
        recommendations.append(('jit', 'Standard JIT compilation'))
    
    # Vulkan for cross-platform GPU acceleration
    if available['vulkan']['available']:
        recommendations.append(('vulkan', 'Cross-platform GPU compute acceleration'))
    
    # Numba for CPU-intensive workloads
    if available['numba']['available']:
        recommendations.append(('numba', 'JIT compilation for numerical operations'))
    
    # Model size specific
    if model_size in ['large', 'xlarge']:
        if available['quantization']['available']:
            recommendations.append(('quantization', 'Quantization for large models'))
        if available['model_compression']['available']:
            recommendations.append(('model_compression', 'Comprehensive model compression'))
    
    # Target specific
    if target == 'inference':
        if available['layer_fusion']['available']:
            recommendations.append(('layer_fusion', 'Layer fusion for inference optimization'))
    
    return recommendations

def create_optimizer_pipeline(optimizers_config):
    """Create a pipeline of optimizers based on configuration."""
    pipeline = []
    available = get_available_optimizers()
    
    for optimizer_name, config in optimizers_config.items():
        if optimizer_name in available and available[optimizer_name]['available']:
            optimizer_class = globals().get(available[optimizer_name]['class'])
            if optimizer_class:
                if isinstance(config, dict):
                    pipeline.append(optimizer_class(**config))
                else:
                    pipeline.append(optimizer_class())
    
    return pipeline

__all__ = [
    # Core optimizers
    'PerformanceOptimizer',
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
    
    # Enhanced optimizers
    'EnhancedJITOptimizer',
    'VulkanOptimizer',
    'VulkanDeviceInfo',
    'NumbaOptimizer',
    
    # Post-download optimizer
    'PostDownloadOptimizer',
    'create_post_download_optimizer',
    'optimize_downloaded_model',
    
    # Calibration and profiling
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
    'prune_model_with_masks',
    
    # Availability flags
    'VULKAN_AVAILABLE',
    'NUMBA_AVAILABLE', 
    'NUMBA_CUDA_AVAILABLE',
    
    # Utility functions
    'get_available_optimizers',
    'get_optimization_recommendations',
    'create_optimizer_pipeline'
]

"""
High-performance processors package for PyTorch inference framework.

This package provides optimized preprocessing and postprocessing pipelines
with advanced performance features enabled by default.
"""

# Import optimized processors
from .preprocessor import (
    PreprocessorPipeline,
    BasePreprocessor,
    ImagePreprocessor,
    TextPreprocessor,
    AudioPreprocessor,
    TensorPreprocessor,
    CustomPreprocessor,
    create_default_preprocessing_pipeline
)

from .postprocessor import (
    PostprocessorPipeline,
    BasePostprocessor,
    ClassificationPostprocessor,
    DetectionPostprocessor,
    SegmentationPostprocessor,
    AudioPostprocessor,
    CustomPostprocessor,
    create_default_postprocessing_pipeline
)

# Import performance configuration
from .performance_config import (
    ProcessorPerformanceConfig,
    ProcessorOptimizer,
    OptimizationMode,
    optimize_for_production_throughput,
    optimize_for_real_time_latency,
    optimize_for_memory_constrained_environment,
    auto_optimize_processors
)

# Import fast factory functions
from .fast_factory import (
    create_fast_preprocessor_pipeline,
    create_fast_postprocessor_pipeline,
    create_fast_processing_pipelines,
    create_optimized_config_for_processors,
    benchmark_processor_performance,
    FastProcessorFactory,
    # Convenience aliases
    create_fast_preprocessors,
    create_fast_postprocessors,
    create_fast_processors
)

# Default fast configurations
def create_fast_default_pipelines(config, class_names=None):
    """
    Create fast, optimized preprocessing and postprocessing pipelines with default settings.
    
    This is the recommended way to create processors for maximum performance.
    
    Args:
        config: InferenceConfig instance
        class_names: Optional list of class names for classification
    
    Returns:
        Tuple of (preprocessor_pipeline, postprocessor_pipeline)
    """
    return create_fast_processing_pipelines(
        config, 
        class_names, 
        optimization_mode="auto", 
        enable_all_optimizations=True
    )


# Convenience function for quick setup
def setup_fast_processors(config, optimization_mode="auto", class_names=None):
    """
    Quick setup function for creating optimized processors.
    
    Args:
        config: InferenceConfig instance
        optimization_mode: One of "auto", "throughput", "latency", "memory"
        class_names: Optional list of class names
    
    Returns:
        Dictionary with 'preprocessor' and 'postprocessor' keys
    """
    preprocessor, postprocessor = create_fast_processing_pipelines(
        config, class_names, optimization_mode, enable_all_optimizations=True
    )
    
    return {
        'preprocessor': preprocessor,
        'postprocessor': postprocessor
    }


__all__ = [
    # Core classes
    'PreprocessorPipeline',
    'PostprocessorPipeline',
    'BasePreprocessor',
    'BasePostprocessor',
    
    # Specific processors
    'ImagePreprocessor',
    'TextPreprocessor',
    'AudioPreprocessor',
    'TensorPreprocessor',
    'ClassificationPostprocessor',
    'DetectionPostprocessor',
    'SegmentationPostprocessor',
    'AudioPostprocessor',
    'CustomPreprocessor',
    'CustomPostprocessor',
    
    # Performance configuration
    'ProcessorPerformanceConfig',
    'ProcessorOptimizer',
    'OptimizationMode',
    
    # Fast factory functions
    'create_fast_preprocessor_pipeline',
    'create_fast_postprocessor_pipeline',
    'create_fast_processing_pipelines',
    'create_optimized_config_for_processors',
    'FastProcessorFactory',
    
    # Convenience functions
    'create_fast_default_pipelines',
    'setup_fast_processors',
    'benchmark_processor_performance',
    
    # Optimization functions
    'optimize_for_production_throughput',
    'optimize_for_real_time_latency',
    'optimize_for_memory_constrained_environment',
    'auto_optimize_processors',
    
    # Aliases
    'create_fast_preprocessors',
    'create_fast_postprocessors',
    'create_fast_processors',
    
    # Legacy functions
    'create_default_preprocessing_pipeline',
    'create_default_postprocessing_pipeline'
]

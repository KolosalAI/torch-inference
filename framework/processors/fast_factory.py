"""
Factory for creating optimized processors with default fast configurations.

This module provides convenient factory functions to create preprocessors and
postprocessors that are optimized for speed by default.
"""

from typing import Optional, List, Dict, Any
import logging

from ..core.config import InferenceConfig
from .preprocessor import PreprocessorPipeline, create_default_preprocessing_pipeline
from .postprocessor import PostprocessorPipeline, create_default_postprocessing_pipeline
from .performance_config import (
    ProcessorPerformanceConfig, 
    ProcessorOptimizer,
    optimize_for_production_throughput,
    optimize_for_real_time_latency,
    auto_optimize_processors
)


logger = logging.getLogger(__name__)


def create_fast_preprocessor_pipeline(
    config: InferenceConfig,
    optimization_mode: str = "auto",
    enable_all_optimizations: bool = True
) -> PreprocessorPipeline:
    """
    Create a fast, optimized preprocessor pipeline.
    
    Args:
        config: Inference configuration
        optimization_mode: One of "auto", "throughput", "latency", "memory"
        enable_all_optimizations: Whether to enable all available optimizations
    
    Returns:
        Optimized preprocessor pipeline
    """
    logger.info(f"Creating fast preprocessor pipeline with {optimization_mode} optimization")
    
    # Create base pipeline
    pipeline = create_default_preprocessing_pipeline(config)
    
    # Apply optimizations based on mode
    if optimization_mode == "throughput":
        optimizer = optimize_for_production_throughput(preprocessor_pipeline=pipeline)
    elif optimization_mode == "latency":
        optimizer = optimize_for_real_time_latency(preprocessor_pipeline=pipeline)
    elif optimization_mode == "memory":
        from .performance_config import optimize_for_memory_constrained_environment
        optimizer = optimize_for_memory_constrained_environment(preprocessor_pipeline=pipeline)
    else:  # auto
        optimizer = auto_optimize_processors(preprocessor_pipeline=pipeline)
    
    # Enable additional optimizations if requested
    if enable_all_optimizations:
        _enable_all_preprocessing_optimizations(pipeline)
    
    logger.info(f"Fast preprocessor pipeline created successfully")
    return pipeline


def create_fast_postprocessor_pipeline(
    config: InferenceConfig,
    class_names: Optional[List[str]] = None,
    optimization_mode: str = "auto",
    enable_all_optimizations: bool = True
) -> PostprocessorPipeline:
    """
    Create a fast, optimized postprocessor pipeline.
    
    Args:
        config: Inference configuration
        class_names: List of class names for classification
        optimization_mode: One of "auto", "throughput", "latency", "memory"
        enable_all_optimizations: Whether to enable all available optimizations
    
    Returns:
        Optimized postprocessor pipeline
    """
    logger.info(f"Creating fast postprocessor pipeline with {optimization_mode} optimization")
    
    # Create base pipeline
    pipeline = create_default_postprocessing_pipeline(config, class_names)
    
    # Apply optimizations based on mode
    if optimization_mode == "throughput":
        optimizer = optimize_for_production_throughput(postprocessor_pipeline=pipeline)
    elif optimization_mode == "latency":
        optimizer = optimize_for_real_time_latency(postprocessor_pipeline=pipeline)
    elif optimization_mode == "memory":
        from .performance_config import optimize_for_memory_constrained_environment
        optimizer = optimize_for_memory_constrained_environment(postprocessor_pipeline=pipeline)
    else:  # auto
        optimizer = auto_optimize_processors(postprocessor_pipeline=pipeline)
    
    # Enable additional optimizations if requested
    if enable_all_optimizations:
        _enable_all_postprocessing_optimizations(pipeline)
    
    logger.info(f"Fast postprocessor pipeline created successfully")
    return pipeline


def create_fast_processing_pipelines(
    config: InferenceConfig,
    class_names: Optional[List[str]] = None,
    optimization_mode: str = "auto",
    enable_all_optimizations: bool = True
) -> tuple[PreprocessorPipeline, PostprocessorPipeline]:
    """
    Create both fast preprocessor and postprocessor pipelines.
    
    Args:
        config: Inference configuration
        class_names: List of class names for classification
        optimization_mode: One of "auto", "throughput", "latency", "memory"
        enable_all_optimizations: Whether to enable all available optimizations
    
    Returns:
        Tuple of (preprocessor_pipeline, postprocessor_pipeline)
    """
    logger.info(f"Creating fast processing pipelines with {optimization_mode} optimization")
    
    preprocessor_pipeline = create_fast_preprocessor_pipeline(
        config, optimization_mode, enable_all_optimizations
    )
    
    postprocessor_pipeline = create_fast_postprocessor_pipeline(
        config, class_names, optimization_mode, enable_all_optimizations
    )
    
    return preprocessor_pipeline, postprocessor_pipeline


def create_optimized_config_for_processors(
    base_config: InferenceConfig,
    optimization_target: str = "balanced"
) -> InferenceConfig:
    """
    Create an optimized configuration specifically for fast processors.
    
    Args:
        base_config: Base inference configuration
        optimization_target: One of "speed", "memory", "balanced"
    
    Returns:
        Optimized configuration
    """
    # Create a copy of the base config
    import copy
    config = copy.deepcopy(base_config)
    
    # Optimize performance settings
    if optimization_target == "speed":
        config.performance.enable_async = True
        config.performance.max_workers = min(8, config.performance.max_workers * 2)
        config.batch.adaptive_batching = True
        config.batch.max_batch_size = min(64, config.batch.max_batch_size * 2)
        config.cache.enable_caching = True
        config.cache.cache_size = min(2000, config.cache.cache_size * 2)
        
        # Enable device optimizations
        if hasattr(config.device, 'use_fp16'):
            config.device.use_fp16 = True
        if hasattr(config.device, 'use_torch_compile'):
            config.device.use_torch_compile = False  # Keep disabled for stability
        
    elif optimization_target == "memory":
        config.performance.max_workers = max(1, config.performance.max_workers // 2)
        config.batch.max_batch_size = max(1, config.batch.max_batch_size // 2)
        config.cache.cache_size = max(50, config.cache.cache_size // 2)
        config.performance.enable_async = False
        
    else:  # balanced
        config.performance.enable_async = True
        config.batch.adaptive_batching = True
        config.cache.enable_caching = True
    
    logger.info(f"Created optimized config for {optimization_target} performance")
    return config


def _enable_all_preprocessing_optimizations(pipeline: PreprocessorPipeline):
    """Enable all available preprocessing optimizations."""
    try:
        # Enable pipeline-level optimizations
        pipeline._enable_parallel_processing = True
        pipeline._batch_processing_enabled = True
        
        # Enable optimizations on all preprocessors
        for preprocessor in pipeline.preprocessors:
            if hasattr(preprocessor, 'enable_optimizations'):
                preprocessor.enable_optimizations(True)
            
            # Enable specific optimizations based on preprocessor type
            if hasattr(preprocessor, '_use_opencv_optimizations'):
                preprocessor._use_opencv_optimizations = True
            if hasattr(preprocessor, '_use_pillow_simd'):
                preprocessor._use_pillow_simd = True
            if hasattr(preprocessor, '_numpy_ops_compiled'):
                try:
                    preprocessor._setup_numba_optimizations()
                except Exception:
                    pass  # Ignore if numba setup fails
        
        logger.debug("Enabled all preprocessing optimizations")
        
    except Exception as e:
        logger.warning(f"Could not enable all preprocessing optimizations: {e}")


def _enable_all_postprocessing_optimizations(pipeline: PostprocessorPipeline):
    """Enable all available postprocessing optimizations."""
    try:
        # Enable pipeline-level optimizations
        pipeline._enable_parallel_processing = True
        pipeline._batch_processing_enabled = True
        
        # Enable optimizations on all postprocessors
        for postprocessor in pipeline.postprocessors.values():
            if hasattr(postprocessor, 'enable_optimizations'):
                postprocessor.enable_optimizations(True)
        
        logger.debug("Enabled all postprocessing optimizations")
        
    except Exception as e:
        logger.warning(f"Could not enable all postprocessing optimizations: {e}")


def benchmark_processor_performance(
    pipeline: PreprocessorPipeline,
    test_inputs: List[Any],
    num_iterations: int = 10
) -> Dict[str, Any]:
    """
    Benchmark processor performance with test inputs.
    
    Args:
        pipeline: Pipeline to benchmark
        test_inputs: List of test inputs
        num_iterations: Number of iterations to run
    
    Returns:
        Performance benchmark results
    """
    import time
    import statistics
    
    logger.info(f"Benchmarking processor performance with {len(test_inputs)} inputs over {num_iterations} iterations")
    
    times = []
    errors = 0
    
    for iteration in range(num_iterations):
        start_time = time.perf_counter()
        
        try:
            for inputs in test_inputs:
                pipeline.preprocess(inputs)
        except Exception as e:
            errors += 1
            logger.warning(f"Error in iteration {iteration}: {e}")
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # Calculate statistics
    if times:
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        throughput = len(test_inputs) / avg_time if avg_time > 0 else 0
        
        # Get pipeline statistics
        pipeline_stats = pipeline.get_performance_stats() if hasattr(pipeline, 'get_performance_stats') else {}
    else:
        avg_time = min_time = max_time = throughput = 0
        pipeline_stats = {}
    
    results = {
        "benchmark_config": {
            "num_inputs": len(test_inputs),
            "num_iterations": num_iterations,
            "errors": errors
        },
        "timing": {
            "avg_time_seconds": avg_time,
            "min_time_seconds": min_time,
            "max_time_seconds": max_time,
            "throughput_items_per_second": throughput
        },
        "pipeline_stats": pipeline_stats
    }
    
    logger.info(f"Benchmark completed: {throughput:.2f} items/sec average throughput")
    return results


# Convenience aliases for common use cases
create_fast_preprocessors = create_fast_preprocessor_pipeline
create_fast_postprocessors = create_fast_postprocessor_pipeline
create_fast_processors = create_fast_processing_pipelines


class FastProcessorFactory:
    """Factory class for creating optimized processors with various presets."""
    
    @staticmethod
    def create_production_ready(config: InferenceConfig, class_names: Optional[List[str]] = None):
        """Create production-ready processors optimized for throughput."""
        return create_fast_processing_pipelines(
            config, class_names, optimization_mode="throughput", enable_all_optimizations=True
        )
    
    @staticmethod
    def create_real_time(config: InferenceConfig, class_names: Optional[List[str]] = None):
        """Create real-time processors optimized for latency."""
        return create_fast_processing_pipelines(
            config, class_names, optimization_mode="latency", enable_all_optimizations=True
        )
    
    @staticmethod
    def create_memory_efficient(config: InferenceConfig, class_names: Optional[List[str]] = None):
        """Create memory-efficient processors."""
        return create_fast_processing_pipelines(
            config, class_names, optimization_mode="memory", enable_all_optimizations=False
        )
    
    @staticmethod
    def create_auto_optimized(config: InferenceConfig, class_names: Optional[List[str]] = None):
        """Create auto-optimized processors based on system capabilities."""
        return create_fast_processing_pipelines(
            config, class_names, optimization_mode="auto", enable_all_optimizations=True
        )

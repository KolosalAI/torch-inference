"""
Performance configuration and optimization settings for processors.

This module provides centralized configuration for all performance optimizations
in the preprocessing and postprocessing pipelines.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import os
import torch
import psutil


class OptimizationMode(Enum):
    """Optimization modes for processors."""
    DISABLED = "disabled"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    AUTO = "auto"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY = "memory"


class ProcessorType(Enum):
    """Types of processors."""
    PREPROCESSOR = "preprocessor"
    POSTPROCESSOR = "postprocessor"
    BOTH = "both"


@dataclass
class ProcessorPerformanceConfig:
    """Performance configuration for processors."""
    
    # General optimization settings
    optimization_mode: OptimizationMode = OptimizationMode.AUTO
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # Parallel processing settings
    enable_parallel_processing: bool = True
    max_parallel_workers: int = 4
    enable_batch_processing: bool = True
    max_batch_size: int = 32
    adaptive_batching: bool = True
    
    # Memory optimization settings
    enable_memory_pooling: bool = True
    memory_pool_size: int = 50
    enable_memory_monitoring: bool = True
    memory_threshold: float = 0.8
    enable_garbage_collection: bool = True
    gc_frequency: int = 100  # Number of operations before GC
    
    # Compilation optimizations
    enable_torch_compile: bool = False  # Disabled by default due to compatibility
    torch_compile_mode: str = "reduce-overhead"
    enable_numba_jit: bool = True
    numba_target: str = "cpu"
    
    # Tensor optimizations
    enable_tensor_optimizations: bool = True
    prefer_channels_last: bool = True
    enable_non_blocking_transfer: bool = True
    optimize_tensor_memory_format: bool = True
    
    # I/O optimizations
    enable_async_io: bool = True
    enable_prefetching: bool = True
    prefetch_factor: int = 2
    
    # Profiling and monitoring
    enable_profiling: bool = False
    profile_frequency: int = 1000
    enable_performance_metrics: bool = True
    
    # Specific processor optimizations
    image_optimization_settings: Dict[str, Any] = field(default_factory=lambda: {
        "use_opencv_optimizations": True,
        "use_pillow_simd": True,
        "enable_fast_resize": True,
        "enable_vectorized_operations": True,
        "cache_transform_params": True
    })
    
    text_optimization_settings: Dict[str, Any] = field(default_factory=lambda: {
        "enable_fast_tokenization": True,
        "cache_tokenizer_outputs": True,
        "enable_batch_encoding": True,
        "use_fast_tokenizers": True
    })
    
    audio_optimization_settings: Dict[str, Any] = field(default_factory=lambda: {
        "enable_fast_resampling": True,
        "cache_audio_features": True,
        "enable_vectorized_audio_ops": True,
        "use_librosa_optimizations": True
    })
    
    @classmethod
    def create_auto_config(cls) -> "ProcessorPerformanceConfig":
        """Create auto-optimized configuration based on system capabilities."""
        config = cls()
        
        # Auto-detect system capabilities
        cpu_count = os.cpu_count() or 4
        memory = psutil.virtual_memory()
        has_cuda = torch.cuda.is_available()
        
        # Adjust settings based on system capabilities
        config.max_parallel_workers = min(cpu_count, 8)
        
        # Memory-based optimizations
        memory_gb = memory.total / (1024**3)
        if memory_gb > 16:
            config.cache_size = 2000
            config.memory_pool_size = 100
            config.max_batch_size = 64
        elif memory_gb > 8:
            config.cache_size = 1000
            config.memory_pool_size = 50
            config.max_batch_size = 32
        else:
            config.cache_size = 500
            config.memory_pool_size = 25
            config.max_batch_size = 16
        
        # GPU optimizations
        if has_cuda:
            config.enable_non_blocking_transfer = True
            config.prefer_channels_last = True
        
        # Disable heavy optimizations on limited systems
        if memory_gb < 4 or cpu_count < 4:
            config.optimization_mode = OptimizationMode.BASIC
            config.enable_parallel_processing = False
            config.enable_memory_pooling = False
        
        return config
    
    @classmethod
    def create_throughput_optimized_config(cls) -> "ProcessorPerformanceConfig":
        """Create configuration optimized for maximum throughput."""
        config = cls.create_auto_config()
        
        config.optimization_mode = OptimizationMode.THROUGHPUT
        config.enable_parallel_processing = True
        config.enable_batch_processing = True
        config.max_batch_size = 64
        config.adaptive_batching = True
        config.enable_memory_pooling = True
        config.enable_prefetching = True
        config.prefetch_factor = 4
        
        return config
    
    @classmethod
    def create_latency_optimized_config(cls) -> "ProcessorPerformanceConfig":
        """Create configuration optimized for minimum latency."""
        config = cls()
        
        config.optimization_mode = OptimizationMode.LATENCY
        config.enable_parallel_processing = False
        config.enable_batch_processing = False
        config.max_batch_size = 1
        config.cache_size = 100
        config.enable_memory_pooling = False
        config.enable_prefetching = False
        
        return config
    
    @classmethod
    def create_memory_optimized_config(cls) -> "ProcessorPerformanceConfig":
        """Create configuration optimized for low memory usage."""
        config = cls()
        
        config.optimization_mode = OptimizationMode.MEMORY
        config.cache_size = 50
        config.memory_pool_size = 10
        config.max_batch_size = 4
        config.enable_memory_monitoring = True
        config.memory_threshold = 0.7
        config.enable_garbage_collection = True
        config.gc_frequency = 50
        
        return config
    
    def apply_to_preprocessor_pipeline(self, pipeline):
        """Apply configuration to a preprocessor pipeline."""
        if hasattr(pipeline, '_enable_parallel_processing'):
            pipeline._enable_parallel_processing = self.enable_parallel_processing
        if hasattr(pipeline, '_batch_processing_enabled'):
            pipeline._batch_processing_enabled = self.enable_batch_processing
        if hasattr(pipeline, '_max_batch_size'):
            pipeline._max_batch_size = self.max_batch_size
        
        # Apply to individual preprocessors
        for preprocessor in getattr(pipeline, 'preprocessors', []):
            self._apply_to_processor(preprocessor)
    
    def apply_to_postprocessor_pipeline(self, pipeline):
        """Apply configuration to a postprocessor pipeline."""
        if hasattr(pipeline, '_enable_parallel_processing'):
            pipeline._enable_parallel_processing = self.enable_parallel_processing
        if hasattr(pipeline, '_batch_processing_enabled'):
            pipeline._batch_processing_enabled = self.enable_batch_processing
        if hasattr(pipeline, '_max_batch_size'):
            pipeline._max_batch_size = self.max_batch_size
        
        # Apply to individual postprocessors
        for postprocessor in getattr(pipeline, 'postprocessors', {}).values():
            self._apply_to_processor(postprocessor)
    
    def _apply_to_processor(self, processor):
        """Apply configuration to an individual processor."""
        if hasattr(processor, 'enable_optimizations'):
            enable_opts = self.optimization_mode != OptimizationMode.DISABLED
            processor.enable_optimizations(enable_opts)
        
        if hasattr(processor, '_max_cache_size'):
            processor._max_cache_size = self.cache_size
        
        # Apply specific optimizations based on processor type
        processor_class_name = processor.__class__.__name__.lower()
        
        if 'image' in processor_class_name:
            self._apply_image_optimizations(processor)
        elif 'text' in processor_class_name:
            self._apply_text_optimizations(processor)
        elif 'audio' in processor_class_name:
            self._apply_audio_optimizations(processor)
    
    def _apply_image_optimizations(self, processor):
        """Apply image-specific optimizations."""
        settings = self.image_optimization_settings
        
        if hasattr(processor, '_use_opencv_optimizations'):
            processor._use_opencv_optimizations = settings.get("use_opencv_optimizations", True)
        if hasattr(processor, '_use_pillow_simd'):
            processor._use_pillow_simd = settings.get("use_pillow_simd", True)
    
    def _apply_text_optimizations(self, processor):
        """Apply text-specific optimizations."""
        settings = self.text_optimization_settings
        
        if hasattr(processor, '_enable_fast_tokenization'):
            processor._enable_fast_tokenization = settings.get("enable_fast_tokenization", True)
    
    def _apply_audio_optimizations(self, processor):
        """Apply audio-specific optimizations."""
        settings = self.audio_optimization_settings
        
        if hasattr(processor, '_enable_fast_resampling'):
            processor._enable_fast_resampling = settings.get("enable_fast_resampling", True)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization settings."""
        return {
            "mode": self.optimization_mode.value,
            "optimization_mode": self.optimization_mode.value,
            "caching": {
                "enabled": self.enable_caching,
                "cache_size": self.cache_size,
                "ttl_seconds": self.cache_ttl_seconds
            },
            "parallel_processing": {
                "enabled": self.enable_parallel_processing,
                "max_workers": self.max_parallel_workers,
                "batch_processing": self.enable_batch_processing,
                "max_batch_size": self.max_batch_size
            },
            "memory_management": {
                "pooling_enabled": self.enable_memory_pooling,
                "monitoring_enabled": self.enable_memory_monitoring,
                "gc_enabled": self.enable_garbage_collection
            },
            "compilation": {
                "torch_compile": self.enable_torch_compile,
                "numba_jit": self.enable_numba_jit
            },
            "tensor_optimizations": {
                "enabled": self.enable_tensor_optimizations,
                "channels_last": self.prefer_channels_last,
                "non_blocking_transfer": self.enable_non_blocking_transfer
            }
        }


class ProcessorOptimizer:
    """Utility class for optimizing processors at runtime."""
    
    def __init__(self, config: ProcessorPerformanceConfig):
        self.config = config
        self._optimization_history = []
    
    def optimize_preprocessor_pipeline(self, pipeline):
        """Optimize a preprocessor pipeline based on configuration."""
        self.config.apply_to_preprocessor_pipeline(pipeline)
        
        # Apply mode-specific optimizations
        if self.config.optimization_mode in [OptimizationMode.AGGRESSIVE, OptimizationMode.THROUGHPUT]:
            pipeline.optimize_for_throughput()
        elif self.config.optimization_mode in [OptimizationMode.BASIC, OptimizationMode.LATENCY]:
            pipeline.optimize_for_latency()
        elif self.config.optimization_mode == OptimizationMode.MEMORY:
            pipeline.optimize_for_memory()
        
        self._record_optimization("preprocessor_pipeline", pipeline.__class__.__name__)
    
    def optimize_postprocessor_pipeline(self, pipeline):
        """Optimize a postprocessor pipeline based on configuration."""
        self.config.apply_to_postprocessor_pipeline(pipeline)
        
        # Apply mode-specific optimizations
        if self.config.optimization_mode in [OptimizationMode.AGGRESSIVE, OptimizationMode.THROUGHPUT]:
            pipeline.optimize_for_throughput()
        elif self.config.optimization_mode in [OptimizationMode.BASIC, OptimizationMode.LATENCY]:
            pipeline.optimize_for_latency()
        elif self.config.optimization_mode == OptimizationMode.MEMORY:
            pipeline.optimize_for_memory()
        
        self._record_optimization("postprocessor_pipeline", pipeline.__class__.__name__)
    
    def _record_optimization(self, component_type: str, component_name: str):
        """Record optimization for tracking."""
        import time
        self._optimization_history.append({
            "timestamp": time.time(),
            "component_type": component_type,
            "component_name": component_name,
            "optimization_mode": self.config.optimization_mode.value
        })
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization report."""
        return {
            "config_summary": self.config.get_optimization_summary(),
            "optimization_history": self._optimization_history,
            "total_optimizations": len(self._optimization_history)
        }


# Convenience functions for common optimization scenarios
def optimize_for_production_throughput(preprocessor_pipeline=None, postprocessor_pipeline=None):
    """Optimize processors for production throughput."""
    config = ProcessorPerformanceConfig.create_throughput_optimized_config()
    optimizer = ProcessorOptimizer(config)
    
    if preprocessor_pipeline:
        optimizer.optimize_preprocessor_pipeline(preprocessor_pipeline)
    if postprocessor_pipeline:
        optimizer.optimize_postprocessor_pipeline(postprocessor_pipeline)
    
    return optimizer


def optimize_for_real_time_latency(preprocessor_pipeline=None, postprocessor_pipeline=None):
    """Optimize processors for real-time latency."""
    config = ProcessorPerformanceConfig.create_latency_optimized_config()
    optimizer = ProcessorOptimizer(config)
    
    if preprocessor_pipeline:
        optimizer.optimize_preprocessor_pipeline(preprocessor_pipeline)
    if postprocessor_pipeline:
        optimizer.optimize_postprocessor_pipeline(postprocessor_pipeline)
    
    return optimizer


def optimize_for_memory_constrained_environment(preprocessor_pipeline=None, postprocessor_pipeline=None):
    """Optimize processors for memory-constrained environments."""
    config = ProcessorPerformanceConfig.create_memory_optimized_config()
    optimizer = ProcessorOptimizer(config)
    
    if preprocessor_pipeline:
        optimizer.optimize_preprocessor_pipeline(preprocessor_pipeline)
    if postprocessor_pipeline:
        optimizer.optimize_postprocessor_pipeline(postprocessor_pipeline)
    
    return optimizer


def auto_optimize_processors(preprocessor_pipeline=None, postprocessor_pipeline=None):
    """Auto-optimize processors based on system capabilities."""
    config = ProcessorPerformanceConfig.create_auto_config()
    optimizer = ProcessorOptimizer(config)
    
    if preprocessor_pipeline:
        optimizer.optimize_preprocessor_pipeline(preprocessor_pipeline)
    if postprocessor_pipeline:
        optimizer.optimize_postprocessor_pipeline(postprocessor_pipeline)
    
    return optimizer

# Alias for backward compatibility
PerformanceConfig = ProcessorPerformanceConfig

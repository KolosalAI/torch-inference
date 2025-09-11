"""
Main framework interface for PyTorch inference.

This module provides the main entry point for the inference framework,
combining all components into an easy-to-use API.

Production-Ready Components Added:
- Reliability: Circuit breakers, health checks, graceful shutdown, retry/DLQ
- Observability: Structured logging, metrics, alerting  
- Configuration: Environment-based config, feature flags, secret management
- Resource Management: Memory tracking, quotas, queue management
- Data Management: Model versioning, registry, caching
- Distributed Caching: Redis Cluster support
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import time
from contextlib import asynccontextmanager

# Core framework components
from .core.config import InferenceConfig, ModelType, ConfigFactory
from .core.base_model import BaseModel, get_model_manager
from .core.inference_engine import InferenceEngine, create_inference_engine
from .core.optimized_model import OptimizedModel, create_optimized_model
from .core.model_downloader import (
    ModelDownloader, get_model_downloader, download_model,
    list_available_models, ModelInfo
)
from .adapters.model_adapters import load_model
from .utils.monitoring import get_performance_monitor, get_metrics_collector

logger = logging.getLogger(__name__)

# Production-ready components
try:
    # Reliability components
    from .reliability.circuit_breaker import CircuitBreaker, CircuitBreakerState, get_circuit_breaker
    from .reliability.health_checks import (
        HealthCheckManager, SystemResourcesHealthCheck, GPUHealthCheck, 
        ModelHealthCheck, DependencyHealthCheck, get_health_check_manager
    )
    from .reliability.graceful_shutdown import GracefulShutdown, get_graceful_shutdown
    from .reliability.connection_pool import AsyncConnectionPool, ConnectionFactory, get_connection_pool
    from .reliability.retry_dlq import (
        RetryAndDLQManager, RetryConfig, RetryPolicy, ModelInferenceOperation,
        get_retry_dlq_manager, create_model_inference_operation
    )

    # Observability components  
    from .observability.structured_logging import (
        StructuredFormatter, CorrelationIDFilter, TraceContext,
        setup_structured_logging, get_correlation_id, set_correlation_id
    )
    from .observability.metrics import (
        MetricsManager, Counter, Gauge, Histogram, SLATracker,
        ResourceUtilizationTracker, get_metrics_manager
    )
    from .observability.alerting import (
        AlertManager, Alert, AlertSeverity, AlertRule, NotificationChannel,
        get_alert_manager, create_default_rules
    )

    # Configuration components
    from .config.advanced_config import (
        ConfigurationManager, FeatureFlagManager, SecretManager, 
        FeatureFlag, SecretConfig, get_config_manager,
        setup_default_configuration
    )

    # Resource management components
    from .resource.resource_manager import (
        ResourceManager, MemoryTracker, RequestQueue, ConnectionLimiter,
        ResourceQuotaManager, QueuedRequest, QueuePriority, ResourceType,
        get_resource_manager
    )

    # Data management components
    from .data.state_manager import (
        DataStateManager, ModelRegistry, IntelligentCache, ModelMetadata,
        ModelStatus, CachePolicy, get_data_state_manager
    )

    # Distributed caching components
    from .cache.distributed_cache import (
        DistributedCacheManager, CacheConfig, CacheBackend, SerializationFormat,
        CacheWarmingStrategy, get_distributed_cache_manager, create_cache_config
    )
    
    # Production features available
    PRODUCTION_FEATURES_AVAILABLE = True
    logger.info("Production-ready components loaded successfully")

except ImportError as e:
    logger.warning(f"Production components not available: {e}")
    # Create dummy references for missing components
    CircuitBreaker = None
    HealthCheckManager = None
    GracefulShutdown = None
    AsyncConnectionPool = None
    RetryAndDLQManager = None
    StructuredFormatter = None
    MetricsManager = None
    AlertManager = None
    ConfigurationManager = None
    ResourceManager = None
    DataStateManager = None
    DistributedCacheManager = None
    
    PRODUCTION_FEATURES_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import optimizers with error handling
try:
    from .optimizers import (
        # Core optimizers
        TensorRTOptimizer, ONNXOptimizer, QuantizationOptimizer,
        MemoryOptimizer, CUDAOptimizer, JITOptimizer,
        convert_to_tensorrt, convert_to_onnx, quantize_model,
        enable_cuda_optimizations, jit_compile_model,
        
        # Enhanced optimizers
        EnhancedJITOptimizer, VulkanOptimizer, NumbaOptimizer,
        PerformanceOptimizer,
        
        # Utility functions
        get_available_optimizers, get_optimization_recommendations,
        create_optimizer_pipeline,
        
        # Availability flags
        VULKAN_AVAILABLE, NUMBA_AVAILABLE, NUMBA_CUDA_AVAILABLE
    )
    
    # Track which optimizers are available
    _optimizer_availability = {
        'enhanced_jit': EnhancedJITOptimizer is not None,
        'vulkan': VulkanOptimizer is not None and VULKAN_AVAILABLE,
        'numba': NumbaOptimizer is not None and NUMBA_AVAILABLE,
        'numba_cuda': NUMBA_CUDA_AVAILABLE,
        'performance': PerformanceOptimizer is not None
    }
    
    logger.info(f"Enhanced optimizers available: {_optimizer_availability}")
    
except ImportError as e:
    logger.warning(f"Some optimizers not available: {e}")
    # Define dummy functions/classes for missing optimizers
    TensorRTOptimizer = None
    ONNXOptimizer = None
    QuantizationOptimizer = None
    MemoryOptimizer = None
    CUDAOptimizer = None
    JITOptimizer = None
    convert_to_tensorrt = None
    convert_to_onnx = None
    quantize_model = None
    enable_cuda_optimizations = None
    jit_compile_model = None
    
    # Enhanced optimizers
    EnhancedJITOptimizer = None
    VulkanOptimizer = None
    NumbaOptimizer = None
    PerformanceOptimizer = None
    
    # Utility functions
    get_available_optimizers = None
    get_optimization_recommendations = None
    create_optimizer_pipeline = None
    
    # Availability flags
    VULKAN_AVAILABLE = False
    NUMBA_AVAILABLE = False
    NUMBA_CUDA_AVAILABLE = False
    
    _optimizer_availability = {
        'enhanced_jit': False,
        'vulkan': False,
        'numba': False,
        'numba_cuda': False,
        'performance': False
    }


class TorchInferenceFramework:
    """
    Main framework class for PyTorch inference.
    
    This class provides a high-level interface for loading models,
    running inference, and managing the entire inference pipeline.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the framework.
        
        Args:
            config: Inference configuration. If None, will use global config.
            cache_dir: Optional cache directory path.
        """
        if config is None:
            from .core.config import get_global_config
            config = get_global_config()
        
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model: Optional[BaseModel] = None
        self.engine: Optional[InferenceEngine] = None
        self._model_manager = get_model_manager()  # Store the manager instance
        self.performance_monitor = get_performance_monitor()
        self.metrics_collector = get_metrics_collector()
        
        # State tracking
        self._initialized = False
        self._engine_running = False
        
        self.logger = logging.getLogger(f"{__name__}.TorchInferenceFramework")
        
        # Configure logging
        self._setup_logging()
        
        self.logger.info("TorchInferenceFramework initialized")
    
    @property
    def model_manager(self):
        """Backward compatibility property for model_manager."""
        return self._model_manager
    
    @property
    def is_loaded(self) -> bool:
        """Check if a model is loaded and ready for inference."""
        return self._initialized and self.model is not None and self.model.is_loaded
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(self.config.performance, 'log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_model(self, model_path: Union[str, Path], model_name: Optional[str] = None) -> None:
        """
        Load a model from file or identifier.
        
        Args:
            model_path: Path to model file or model identifier (e.g., HuggingFace model name)
            model_name: Optional name for the model (for model manager)
        """
        try:
            self.logger.info(f"Loading model from: {model_path}")
            
            # Load model using adapter factory
            self.model = load_model(model_path, self.config)
            
            # Register with model manager
            if model_name is None:
                model_name = Path(model_path).stem if isinstance(model_path, (str, Path)) else str(model_path)
            
            self._model_manager.register_model(model_name, self.model)
            
            # Create inference engine
            self.engine = create_inference_engine(self.model, self.config)
            
            self._initialized = True
            self.logger.info(f"Successfully loaded model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def download_and_load_model(
        self, 
        source: str,
        model_id: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Download and load a model from a source.
        
        Args:
            source: Model source ('pytorch_hub', 'torchvision', 'huggingface', 'url')
            model_id: Model identifier
            model_name: Name for the model. If None, will be generated.
            **kwargs: Additional arguments for download
        """
        try:
            # Generate model name if not provided
            if model_name is None:
                import re
                model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', str(model_id))
                model_name = f"{source}_{model_name}"
            
            self.logger.info(f"Downloading and loading model: {model_name}")
            
            # Download model
            model_path, model_info = download_model(
                source=source,
                model_id=model_id,
                model_name=model_name,
                **kwargs
            )
            
            # Load the downloaded model
            self.load_model(model_path, model_name)
            
            self.logger.info(f"Successfully downloaded and loaded: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to download and load model: {e}")
            raise
    
    def list_available_downloads(self) -> Dict[str, Any]:
        """List available models that can be downloaded."""
        return list_available_models()
    
    def get_model_downloader(self) -> ModelDownloader:
        """Get the model downloader instance."""
        return get_model_downloader()
    
    async def start_engine(self) -> None:
        """Start the inference engine for async processing."""
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not self.engine:
            raise RuntimeError("Inference engine not initialized.")
        
        await self.engine.start()
        self._engine_running = True
        self.logger.info("Inference engine started")
    
    async def stop_engine(self) -> None:
        """Stop the inference engine."""
        if self.engine and self._engine_running:
            await self.engine.stop()
            self._engine_running = False
            self.logger.info("Inference engine stopped")
    
    def predict(self, inputs: Any, **kwargs) -> Any:
        """
        Run inference on inputs (synchronous).
        
        Args:
            inputs: Input data (image path, tensor, text, etc.)
            **kwargs: Additional arguments passed to prediction
            
        Returns:
            Prediction results
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Track performance
        request_id = f"sync_{int(time.time() * 1000000)}"
        self.performance_monitor.start_request(request_id)
        try:
            result = self.model.predict(inputs)
            self.performance_monitor.end_request(request_id)
            return result
        except Exception as e:
            self.performance_monitor.end_request(request_id)
            raise
    
    async def predict_async(self, inputs: Any, priority: int = 0, 
                           timeout: Optional[float] = None, **kwargs) -> Any:
        """
        Run inference on inputs (asynchronous).
        
        Args:
            inputs: Input data
            priority: Request priority (higher = processed first)
            timeout: Timeout in seconds
            **kwargs: Additional arguments
            
        Returns:
            Prediction results
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not self._engine_running:
            raise RuntimeError("Engine not running. Call start_engine() first.")
        
        return await self.engine.predict(inputs, priority, timeout)
    
    def predict_batch(self, inputs_list: List[Any], **kwargs) -> List[Any]:
        """
        Run batch inference (synchronous).
        
        Args:
            inputs_list: List of input data
            **kwargs: Additional arguments
            
        Returns:
            List of prediction results
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use the model's predict_batch method if available
        if hasattr(self.model, 'predict_batch'):
            return self.model.predict_batch(inputs_list)
        
        # Fallback to individual predictions
        results = []
        for i, inputs in enumerate(inputs_list):
            request_id = f"batch_{int(time.time() * 1000000)}_{i}"
            self.performance_monitor.start_request(request_id)
            try:
                result = self.model.predict(inputs)
                results.append(result)
                self.performance_monitor.end_request(request_id)
            except Exception as e:
                self.performance_monitor.end_request(request_id)
                raise
        
        return results
    
    async def predict_batch_async(self, inputs_list: List[Any], priority: int = 0,
                                 timeout: Optional[float] = None, **kwargs) -> List[Any]:
        """
        Run batch inference (asynchronous).
        
        Args:
            inputs_list: List of input data
            priority: Request priority
            timeout: Timeout in seconds
            **kwargs: Additional arguments
            
        Returns:
            List of prediction results
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not self._engine_running:
            raise RuntimeError("Engine not running. Call start_engine() first.")
        
        return await self.engine.predict_batch(inputs_list, priority, timeout)
    
    def benchmark(self, inputs: Any, iterations: int = 100, warmup: int = 10) -> Dict[str, Any]:
        """
        Benchmark the model performance.
        
        Args:
            inputs: Sample input for benchmarking
            iterations: Number of benchmark iterations
            warmup: Number of warmup iterations
            
        Returns:
            Benchmark results
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.logger.info(f"Running benchmark: {warmup} warmup + {iterations} iterations")
        
        # Warmup
        for _ in range(warmup):
            _ = self.model.predict(inputs)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = self.model.predict(inputs)
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
        
        # Calculate statistics
        import statistics
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        
        results = {
            "iterations": iterations,
            "mean_time_ms": mean_time * 1000,
            "median_time_ms": median_time * 1000,
            "std_time_ms": std_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "throughput_fps": 1.0 / mean_time,
            "device": str(self.model.device),
            "model_info": self.model.model_info
        }
        
        self.logger.info(f"Benchmark complete: {results['throughput_fps']:.2f} FPS")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._initialized:
            return {"loaded": False}
        
        return self.model.model_info
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get inference engine statistics."""
        if not self.engine:
            return {"engine": "not_initialized"}
        
        return self.engine.get_stats()
    
    def get_optimization_recommendations(self, model_size: str = 'medium', target: str = 'inference', device: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Get optimization recommendations for the current setup.
        
        Args:
            model_size: Size of the model ('small', 'medium', 'large', 'xlarge')
            target: Target use case ('inference', 'training', 'serving')
            device: Device type to get recommendations for (if None, uses current config)
            
        Returns:
            List of (optimizer_name, description) tuples
        """
        if get_optimization_recommendations is None:
            return [('standard', 'Only standard optimizations available')]
        
        device_str = device if device is not None else str(self.config.device.device_type) if self.config else 'auto'
        return get_optimization_recommendations(device_str, model_size, target)
    
    def apply_automatic_optimizations(self, aggressive: bool = False) -> Dict[str, bool]:
        """
        Apply automatic optimizations based on system capabilities and model characteristics.
        
        Args:
            aggressive: Whether to apply aggressive optimizations that may reduce compatibility
            
        Returns:
            Dictionary showing which optimizations were applied
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        applied = {}
        
        # Get recommendations
        recommendations = self.get_optimization_recommendations()
        
        for optimizer_name, description in recommendations:
            try:
                if optimizer_name == 'enhanced_jit' and EnhancedJITOptimizer is not None:
                    optimizer = EnhancedJITOptimizer(self.config)
                    optimized_model = optimizer.optimize(self.model.model, self.model.example_inputs)
                    if optimized_model is not None:
                        self.model.model = optimized_model
                        applied['enhanced_jit'] = True
                        self.logger.info("Applied Enhanced JIT optimization")
                    else:
                        applied['enhanced_jit'] = False
                
                elif optimizer_name == 'vulkan' and VulkanOptimizer is not None and VULKAN_AVAILABLE:
                    optimizer = VulkanOptimizer(self.config)
                    # Note: Vulkan optimizer creates compute contexts for tensor operations
                    # Model stays the same, but operations are accelerated
                    result = optimizer.optimize(self.model.model, self.model.example_inputs)
                    if result is not None:
                        applied['vulkan'] = True
                        self.logger.info("Applied Vulkan optimization")
                    else:
                        applied['vulkan'] = False
                
                elif optimizer_name == 'numba' and NumbaOptimizer is not None and NUMBA_AVAILABLE:
                    optimizer = NumbaOptimizer(self.config)
                    # Numba optimizer wraps operations for JIT compilation
                    result = optimizer.optimize(self.model.model, self.model.example_inputs)
                    if result is not None:
                        applied['numba'] = True
                        self.logger.info("Applied Numba optimization")
                    else:
                        applied['numba'] = False
                
                elif optimizer_name == 'performance' and PerformanceOptimizer is not None:
                    optimizer = PerformanceOptimizer(self.config)
                    optimized_model = optimizer.optimize(self.model.model, self.model.example_inputs)
                    if optimized_model is not None:
                        self.model.model = optimized_model
                        applied['performance'] = True
                        self.logger.info("Applied Performance optimization")
                    else:
                        applied['performance'] = False
                
                else:
                    applied[optimizer_name] = False
                    
            except Exception as e:
                self.logger.warning(f"Failed to apply {optimizer_name} optimization: {e}")
                applied[optimizer_name] = False
        
        self.logger.info(f"Automatic optimizations applied: {applied}")
        return applied
    
    def get_available_optimizers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available optimizers.
        
        Returns:
            Dictionary with optimizer information
        """
        if get_available_optimizers is None:
            return {
                'standard': {'available': True, 'class': 'StandardOptimizer'},
                'enhanced': {'available': False, 'class': None}
            }
        
        return get_available_optimizers()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "framework_info": {
                "initialized": self._initialized,
                "engine_running": self._engine_running,
                "config": self.config
            },
            "model_info": self.get_model_info(),
            "performance_metrics": self.performance_monitor.get_performance_summary()
        }
        
        if self.engine:
            report["engine_stats"] = self.engine.get_stats()
            report["engine_performance"] = self.engine.get_performance_report()
        
        return report
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the framework."""
        health = {
            "healthy": True,
            "checks": {},
            "timestamp": time.time()
        }
        
        # Check framework initialization
        health["checks"]["framework_initialized"] = self._initialized
        if not self._initialized:
            health["healthy"] = False
        
        # Check model
        if self.model:
            health["checks"]["model_loaded"] = self.model.is_loaded
            if not self.model.is_loaded:
                health["healthy"] = False
        else:
            health["checks"]["model_loaded"] = False
            health["healthy"] = False
        
        # Check engine
        if self.engine:
            engine_health = await self.engine.health_check()
            health["checks"]["engine"] = engine_health
            if not engine_health["healthy"]:
                health["healthy"] = False
        else:
            health["checks"]["engine"] = {"healthy": False, "reason": "not_initialized"}
            # Engine not being initialized is okay for sync-only usage
        
        return health
    
    async def cleanup_async(self) -> None:
        """Cleanup all resources (async version)."""
        self.logger.info("Cleaning up framework resources")
        
        if self.engine and self._engine_running:
            await self.stop_engine()
        
        if self.model:
            self.model.cleanup()
        
        self._model_manager.cleanup_all()
        
        self.logger.info("Framework cleanup complete")
    
    def cleanup_sync(self) -> None:
        """Synchronous cleanup for backward compatibility."""
        self.logger.info("Cleaning up framework resources (sync)")
        
        if self.engine and self._engine_running:
            # For sync cleanup, we can't await, so just stop without awaiting
            self._engine_running = False
        
        if self.model:
            self.model.cleanup()
        
        self._model_manager.cleanup_all()
        
        self.logger.info("Framework cleanup complete (sync)")
    
    def cleanup(self) -> None:
        """Backward compatible cleanup method."""
        return self.cleanup_sync()
    
    @asynccontextmanager
    async def async_context(self):
        """Async context manager for automatic lifecycle management."""
        try:
            if self.engine and not self._engine_running:
                await self.start_engine()
            yield self
        finally:
            await self.cleanup_async()
    
    def __enter__(self):
        """Sync context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        if self.model:
            self.model.cleanup()


# Factory functions for common use cases

def create_classification_framework(model_path: Union[str, Path], 
                                   num_classes: int,
                                   class_names: Optional[List[str]] = None,
                                   input_size: Tuple[int, int] = (224, 224)) -> TorchInferenceFramework:
    """
    Create a framework configured for image classification.
    
    Args:
        model_path: Path to model file
        num_classes: Number of classification classes
        class_names: Optional list of class names
        input_size: Input image size
        
    Returns:
        Configured framework instance
    """
    config = ConfigFactory.create_classification_config(
        num_classes=num_classes,
        input_size=input_size,
        use_softmax=True
    )
    
    framework = TorchInferenceFramework(config)
    framework.load_model(model_path)
    
    return framework


def create_detection_framework(model_path: Union[str, Path],
                              class_names: Optional[List[str]] = None,
                              input_size: Tuple[int, int] = (640, 640),
                              confidence_threshold: float = 0.5) -> TorchInferenceFramework:
    """
    Create a framework configured for object detection.
    
    Args:
        model_path: Path to model file
        class_names: Optional list of class names
        input_size: Input image size
        confidence_threshold: Detection confidence threshold
        
    Returns:
        Configured framework instance
    """
    config = ConfigFactory.create_detection_config(
        input_size=input_size,
        confidence_threshold=confidence_threshold
    )
    
    framework = TorchInferenceFramework(config)
    framework.load_model(model_path)
    
    return framework


def create_segmentation_framework(model_path: Union[str, Path],
                                 input_size: Tuple[int, int] = (640, 640),
                                 threshold: float = 0.5) -> TorchInferenceFramework:
    """
    Create a framework configured for image segmentation.
    
    Args:
        model_path: Path to model file
        input_size: Input image size
        threshold: Segmentation threshold
        
    Returns:
        Configured framework instance
    """
    config = ConfigFactory.create_segmentation_config(
        input_size=input_size,
        threshold=threshold
    )
    
    framework = TorchInferenceFramework(config)
    framework.load_model(model_path)
    
    return framework


# Convenience functions for quick usage

def predict_image_classification(model_path: Union[str, Path], 
                                image_path: Union[str, Path],
                                num_classes: int,
                                class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Quick image classification prediction.
    
    Args:
        model_path: Path to model file
        image_path: Path to image file
        num_classes: Number of classes
        class_names: Optional class names
        
    Returns:
        Classification result
    """
    framework = create_classification_framework(model_path, num_classes, class_names)
    with framework:
        result = framework.predict(image_path)
    return result


def predict_object_detection(model_path: Union[str, Path],
                           image_path: Union[str, Path],
                           class_names: Optional[List[str]] = None,
                           confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Quick object detection prediction.
    
    Args:
        model_path: Path to model file
        image_path: Path to image file
        class_names: Optional class names
        confidence_threshold: Detection threshold
        
    Returns:
        Detection result
    """
    framework = create_detection_framework(
        model_path, class_names, confidence_threshold=confidence_threshold
    )
    with framework:
        result = framework.predict(image_path)
    return result


def predict_segmentation(model_path: Union[str, Path],
                        image_path: Union[str, Path],
                        threshold: float = 0.5) -> Dict[str, Any]:
    """
    Quick segmentation prediction.
    
    Args:
        model_path: Path to model file
        image_path: Path to image file
        threshold: Segmentation threshold
        
    Returns:
        Segmentation result
    """
    framework = create_segmentation_framework(model_path, threshold=threshold)
    with framework:
        result = framework.predict(image_path)
    return result


# Global framework instance for singleton usage
_global_framework: Optional[TorchInferenceFramework] = None


def get_global_framework() -> TorchInferenceFramework:
    """Get the global framework instance."""
    global _global_framework
    if _global_framework is None:
        _global_framework = TorchInferenceFramework()
    return _global_framework


def set_global_framework(framework: TorchInferenceFramework) -> None:
    """Set the global framework instance."""
    global _global_framework
    _global_framework = framework


# Convenience functions for downloading popular models

def download_torchvision_model(model_name: str, pretrained: bool = True) -> TorchInferenceFramework:
    """
    Download and create a framework with a torchvision model.
    
    Args:
        model_name: Name of the torchvision model (e.g., 'resnet18', 'vgg16')
        pretrained: Whether to download pretrained weights
        
    Returns:
        Framework instance with the downloaded model
    """
    framework = TorchInferenceFramework()
    framework.download_and_load_model(
        source="torchvision",
        model_id=model_name,
        pretrained=pretrained
    )
    return framework


def download_pytorch_hub_model(repo: str, model: str, pretrained: bool = True) -> TorchInferenceFramework:
    """
    Download and create a framework with a PyTorch Hub model.
    
    Args:
        repo: Repository name (e.g., 'pytorch/vision')
        model: Model name (e.g., 'resnet50')  
        pretrained: Whether to download pretrained weights
        
    Returns:
        Framework instance with the downloaded model
    """
    framework = TorchInferenceFramework()
    framework.download_and_load_model(
        source="pytorch_hub",
        model_id=f"{repo}/{model}",
        pretrained=pretrained
    )
    return framework


def download_huggingface_model(model_id: str, task: str = "feature-extraction") -> TorchInferenceFramework:
    """
    Download and create a framework with a Hugging Face model.
    
    Args:
        model_id: Hugging Face model identifier (e.g., 'bert-base-uncased')
        task: Task type for the model
        
    Returns:
        Framework instance with the downloaded model
    """
    framework = TorchInferenceFramework()
    framework.download_and_load_model(
        source="huggingface",
        model_id=model_id,
        task=task
    )
    return framework


# Popular model presets

def download_resnet18(pretrained: bool = True) -> TorchInferenceFramework:
    """Download ResNet-18 model."""
    return download_torchvision_model("resnet18", pretrained)


def download_resnet50(pretrained: bool = True) -> TorchInferenceFramework:
    """Download ResNet-50 model."""
    return download_torchvision_model("resnet50", pretrained)


def download_mobilenet_v2(pretrained: bool = True) -> TorchInferenceFramework:
    """Download MobileNet v2 model."""
    return download_torchvision_model("mobilenet_v2", pretrained)


def download_efficientnet_b0(pretrained: bool = True) -> TorchInferenceFramework:
    """Download EfficientNet B0 model."""
    return download_torchvision_model("efficientnet_b0", pretrained)


def download_bert_base(task: str = "feature-extraction") -> TorchInferenceFramework:
    """Download BERT base model."""
    return download_huggingface_model("bert-base-uncased", task)


def download_distilbert(task: str = "text-classification") -> TorchInferenceFramework:
    """Download DistilBERT model."""
    return download_huggingface_model("distilbert-base-uncased-finetuned-sst-2-english", task)
def create_optimized_framework(config: Optional[InferenceConfig] = None) -> TorchInferenceFramework:
    """
    Create an optimized framework with automatic optimization selection.
    
    Args:
        config: Inference configuration
        
    Returns:
        Optimized framework instance
    """
    class OptimizedFramework(TorchInferenceFramework):
        def load_model(self, model_path: Union[str, Path], model_name: Optional[str] = None) -> None:
            """Load model with automatic optimization."""
            # Use optimized model instead of regular model
            self.model = OptimizedModel(self.config)
            self.model.load_model(model_path)
            
            # Register with model manager
            if model_name is None:
                model_name = Path(model_path).stem if isinstance(model_path, (str, Path)) else str(model_path)
            
            self._model_manager.register_model(model_name, self.model)
            
            # Create inference engine
            self.engine = create_inference_engine(self.model, self.config)
            
            self._initialized = True
            self.logger.info(f"Successfully loaded optimized model: {model_name}")
    
    return OptimizedFramework(config)


# Production Framework Integration

def create_production_framework(config_dir: str = "config", 
                              model_storage_path: str = "models",
                              cache_size_mb: int = 1024,
                              redis_cluster_nodes: Optional[List[Dict]] = None,
                              inference_config: Optional[InferenceConfig] = None) -> Tuple[TorchInferenceFramework, Dict[str, Any]]:
    """
    Create a production-ready framework with all enterprise components.
    
    Args:
        config_dir: Directory for configuration files
        model_storage_path: Directory for model storage
        cache_size_mb: Cache size in megabytes
        redis_cluster_nodes: Redis cluster node configuration
        inference_config: Base inference configuration
    
    Returns:
        Tuple of (framework_instance, production_components)
    """
    if not PRODUCTION_FEATURES_AVAILABLE:
        logger.warning("Production features not available, creating standard framework")
        return TorchInferenceFramework(inference_config), {}
    
    # Initialize production components
    components = {
        'config': setup_default_configuration(config_dir),
        'circuit_breaker': get_circuit_breaker(),
        'health_checks': get_health_check_manager(), 
        'graceful_shutdown': get_graceful_shutdown(),
        'connection_pool': get_connection_pool(),
        'retry_dlq': get_retry_dlq_manager(),
        'metrics': get_metrics_manager(),
        'alerting': get_alert_manager(),
        'resource_manager': get_resource_manager(),
        'data_state_manager': get_data_state_manager(),
    }
    
    # Setup structured logging
    components['structured_logging'] = setup_structured_logging(
        level="INFO",
        format_json=True,
        include_trace_context=True
    )
    
    # Setup distributed cache if Redis cluster nodes provided
    if redis_cluster_nodes:
        config, nodes = create_cache_config(
            backend=CacheBackend.REDIS_CLUSTER,
            cluster_nodes=redis_cluster_nodes
        )
        components['distributed_cache'] = DistributedCacheManager(config, nodes)
    
    # Create enhanced framework with production components
    class ProductionTorchInferenceFramework(TorchInferenceFramework):
        def __init__(self, config: Optional[InferenceConfig] = None):
            super().__init__(config)
            self.production_components = components
            
        async def start_production_services(self):
            """Start all production services."""
            startup_order = [
                'config', 'metrics', 'resource_manager', 'data_state_manager', 
                'distributed_cache', 'retry_dlq', 'health_checks', 'alerting'
            ]
            
            for component_name in startup_order:
                component = self.production_components.get(component_name)
                if component and hasattr(component, 'start'):
                    try:
                        await component.start()
                        logger.info(f"Started production component: {component_name}")
                    except Exception as e:
                        logger.error(f"Failed to start {component_name}: {e}")
                        raise
        
        async def stop_production_services(self):
            """Stop all production services."""
            shutdown_order = [
                'alerting', 'health_checks', 'retry_dlq', 'distributed_cache',
                'data_state_manager', 'resource_manager', 'metrics'
            ]
            
            for component_name in shutdown_order:
                component = self.production_components.get(component_name)
                if component and hasattr(component, 'stop'):
                    try:
                        await component.stop()
                        logger.info(f"Stopped production component: {component_name}")
                    except Exception as e:
                        logger.error(f"Error stopping {component_name}: {e}")
        
        def get_production_stats(self) -> Dict[str, Any]:
            """Get comprehensive production statistics."""
            stats = {}
            
            for name, component in self.production_components.items():
                if hasattr(component, 'get_stats'):
                    try:
                        stats[name] = component.get_stats()
                    except Exception as e:
                        stats[name] = {'error': str(e)}
                elif hasattr(component, 'get_system_status'):
                    try:
                        stats[name] = component.get_system_status()
                    except Exception as e:
                        stats[name] = {'error': str(e)}
            
            return stats
        
        async def production_health_check(self) -> Dict[str, Any]:
            """Comprehensive production health check."""
            health = await super().health_check()
            
            # Add production component health checks
            production_health = {}
            
            for name, component in self.production_components.items():
                if hasattr(component, 'health_check'):
                    try:
                        component_health = await component.health_check()
                        production_health[name] = component_health
                    except Exception as e:
                        production_health[name] = {'healthy': False, 'error': str(e)}
                        health['healthy'] = False
                elif hasattr(component, 'get_stats'):
                    try:
                        # Use stats as basic health indicator
                        stats = component.get_stats()
                        production_health[name] = {'healthy': True, 'stats': stats}
                    except Exception as e:
                        production_health[name] = {'healthy': False, 'error': str(e)}
                        health['healthy'] = False
            
            health['production_components'] = production_health
            return health
        
        async def cleanup_async(self):
            """Enhanced cleanup with production services."""
            await self.stop_production_services()
            await super().cleanup_async()
    
    framework = ProductionTorchInferenceFramework(inference_config)
    return framework, components


async def start_production_framework(framework_and_components: Tuple[TorchInferenceFramework, Dict[str, Any]]):
    """Start production framework services."""
    framework, components = framework_and_components
    
    if hasattr(framework, 'start_production_services'):
        await framework.start_production_services()
    else:
        logger.warning("Framework does not support production services")


async def stop_production_framework(framework_and_components: Tuple[TorchInferenceFramework, Dict[str, Any]]):
    """Stop production framework services."""
    framework, components = framework_and_components
    
    if hasattr(framework, 'stop_production_services'):
        await framework.stop_production_services()
    else:
        logger.warning("Framework does not support production services")

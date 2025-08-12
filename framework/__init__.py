"""
Main framework interface for PyTorch inference.

This module provides the main entry point for the inference framework,
combining all components into an easy-to-use API.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import time
from contextlib import asynccontextmanager

from .core.config import InferenceConfig, ModelType, ConfigFactory
from .core.base_model import BaseModel, get_model_manager
from .core.inference_engine import InferenceEngine, create_inference_engine
from .core.optimized_model import OptimizedModel, create_optimized_model
from .adapters.model_adapters import load_model
from .utils.monitoring import get_performance_monitor, get_metrics_collector

logger = logging.getLogger(__name__)

# Import optimizers with error handling
try:
    from .optimizers import (
        TensorRTOptimizer, ONNXOptimizer, QuantizationOptimizer,
        MemoryOptimizer, CUDAOptimizer, JITOptimizer,
        convert_to_tensorrt, convert_to_onnx, quantize_model,
        enable_cuda_optimizations, jit_compile_model
    )
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


class TorchInferenceFramework:
    """
    Main framework class for PyTorch inference.
    
    This class provides a high-level interface for loading models,
    running inference, and managing the entire inference pipeline.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize the framework.
        
        Args:
            config: Inference configuration. If None, will use global config.
        """
        if config is None:
            from .core.config import get_global_config
            config = get_global_config()
        
        self.config = config
        self.model: Optional[BaseModel] = None
        self.engine: Optional[InferenceEngine] = None
        self.model_manager = get_model_manager  # Store the function, not call it
        self.performance_monitor = get_performance_monitor()
        self.metrics_collector = get_metrics_collector()
        
        # State tracking
        self._initialized = False
        self._engine_running = False
        
        self.logger = logging.getLogger(f"{__name__}.TorchInferenceFramework")
        
        # Configure logging
        self._setup_logging()
        
        self.logger.info("TorchInferenceFramework initialized")
    
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
            
            self.model_manager().register_model(model_name, self.model)
            
            # Create inference engine
            self.engine = create_inference_engine(self.model, self.config)
            
            self._initialized = True
            self.logger.info(f"Successfully loaded model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
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
    
    async def cleanup(self) -> None:
        """Cleanup all resources."""
        self.logger.info("Cleaning up framework resources")
        
        if self.engine and self._engine_running:
            await self.stop_engine()
        
        if self.model:
            self.model.cleanup()
        
        self.model_manager().cleanup_all()
        
        self.logger.info("Framework cleanup complete")
    
    @asynccontextmanager
    async def async_context(self):
        """Async context manager for automatic lifecycle management."""
        try:
            if self.engine and not self._engine_running:
                await self.start_engine()
            yield self
        finally:
            await self.cleanup()
    
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


# Convenience functions for optimization
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
            
            self.model_manager().register_model(model_name, self.model)
            
            # Create inference engine
            self.engine = create_inference_engine(self.model, self.config)
            
            self._initialized = True
            self.logger.info(f"Successfully loaded optimized model: {model_name}")
    
    return OptimizedFramework(config)

"""
Enhanced inference engine for PyTorch Inference Framework.

This module provides a robust, production-ready inference engine with 
comprehensive error handling, performance optimization, and monitoring.
"""

import asyncio
import time
import logging
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import traceback

import torch
import numpy as np

from .exceptions import (
    TorchInferenceError, PredictionError, DeviceError, 
    ResourceExhaustedError, TimeoutError, ConfigurationError
)
from .memory_manager import get_memory_manager
from .config_simple import InferenceConfig, AppConfig

logger = logging.getLogger(__name__)


class InferenceState(str, Enum):
    """Inference engine states."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class PredictionRequest:
    """Prediction request with metadata."""
    request_id: str
    inputs: Any
    priority: int = 0
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class PredictionResult:
    """Prediction result with timing and metadata."""
    request_id: str
    result: Any
    success: bool
    error: Optional[str] = None
    processing_time: float = 0.0
    queue_time: float = 0.0
    total_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineStats:
    """Engine performance statistics."""
    requests_processed: int = 0
    requests_failed: int = 0
    total_processing_time: float = 0.0
    total_queue_time: float = 0.0
    average_processing_time: float = 0.0
    average_queue_time: float = 0.0
    current_queue_size: int = 0
    peak_queue_size: int = 0
    memory_usage_mb: float = 0.0
    gpu_utilization_percent: float = 0.0


class InferenceEngine:
    """
    Enhanced inference engine with comprehensive error handling and optimization.
    
    Features:
    - Asynchronous prediction processing
    - Priority-based request queuing
    - Comprehensive error handling with fallbacks
    - Performance monitoring and metrics
    - Memory management integration
    - Graceful shutdown and cleanup
    """
    
    def __init__(self, model: Any, config: InferenceConfig):
        self.model = model
        self.config = config
        self.state = InferenceState.INITIALIZING
        
        # Queue management
        self._request_queue = asyncio.PriorityQueue()
        self._processing_lock = asyncio.Lock()
        
        # Statistics
        self._stats = EngineStats()
        self._stats_lock = threading.RLock()
        
        # Background tasks
        self._worker_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Memory management
        self._memory_manager = get_memory_manager()
        
        # Performance tracking
        self._last_prediction_time = time.time()
        self._prediction_times: List[float] = []
        self._error_count = 0
        self._consecutive_errors = 0
        
        # Device information
        self.device = self._get_device()
        
        logger.info(f"InferenceEngine initialized - Device: {self.device}, "
                   f"Model: {type(self.model).__name__}")
    
    async def start(self) -> None:
        """Start the inference engine and background workers."""
        try:
            logger.info("Starting inference engine...")
            
            # Validate configuration
            self._validate_configuration()
            
            # Initialize model
            await self._initialize_model()
            
            # Start worker tasks
            num_workers = self._get_optimal_worker_count()
            for i in range(num_workers):
                task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
                self._worker_tasks.append(task)
            
            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_monitor())
            
            self.state = InferenceState.READY
            
            logger.info(f"Inference engine started successfully with {num_workers} workers")
            
        except Exception as e:
            self.state = InferenceState.ERROR
            logger.error(f"Failed to start inference engine: {e}")
            raise ConfigurationError(
                config_field="engine_startup",
                details=f"Engine startup failed: {e}",
                cause=e
            )
    
    async def stop(self) -> None:
        """Stop the inference engine and cleanup resources."""
        logger.info("Stopping inference engine...")
        self.state = InferenceState.SHUTDOWN
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for workers to finish
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        # Final cleanup
        self._cleanup_resources()
        
        logger.info("Inference engine stopped successfully")
    
    async def predict(
        self, 
        inputs: Any, 
        priority: int = 0, 
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Make a prediction asynchronously.
        
        Args:
            inputs: Input data for prediction
            priority: Request priority (higher = processed first)
            timeout: Request timeout in seconds
            metadata: Optional metadata for the request
            
        Returns:
            Prediction result
            
        Raises:
            PredictionError: If prediction fails
            TimeoutError: If request times out
            ResourceExhaustedError: If resources are exhausted
        """
        if self.state != InferenceState.READY:
            raise PredictionError(
                details=f"Engine not ready (state: {self.state})",
                context={"engine_state": self.state.value}
            )
        
        request_id = f"req_{int(time.time() * 1000)}_{id(inputs)}"
        
        request = PredictionRequest(
            request_id=request_id,
            inputs=inputs,
            priority=priority,
            timeout=timeout or 30.0,
            metadata=metadata or {}
        )
        
        try:
            # Add to queue
            await self._request_queue.put((-priority, time.time(), request))
            
            # Update queue stats
            with self._stats_lock:
                self._stats.current_queue_size = self._request_queue.qsize()
                self._stats.peak_queue_size = max(
                    self._stats.peak_queue_size, 
                    self._stats.current_queue_size
                )
            
            # Wait for result with timeout
            result = await asyncio.wait_for(
                self._wait_for_result(request_id),
                timeout=request.timeout
            )
            
            if not result.success:
                raise PredictionError(
                    details=result.error or "Unknown prediction error",
                    context={
                        "request_id": request_id,
                        "processing_time": result.processing_time,
                        "queue_time": result.queue_time
                    }
                )
            
            return result.result
            
        except asyncio.TimeoutError:
            logger.warning(f"Request {request_id} timed out after {request.timeout}s")
            raise TimeoutError(
                operation="prediction",
                timeout_seconds=request.timeout,
                context={"request_id": request_id}
            )
        except Exception as e:
            if isinstance(e, TorchInferenceError):
                raise
            
            logger.error(f"Unexpected error in prediction: {e}")
            raise PredictionError(
                details=f"Unexpected error: {e}",
                cause=e,
                context={"request_id": request_id}
            )
    
    async def predict_batch(
        self, 
        inputs_list: List[Any], 
        priority: int = 0, 
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Make batch predictions.
        
        Args:
            inputs_list: List of input data for batch prediction
            priority: Request priority
            timeout: Request timeout in seconds
            
        Returns:
            List of prediction results
        """
        if not inputs_list:
            return []
        
        # For small batches, process individually for better error isolation
        if len(inputs_list) <= 4:
            tasks = [
                self.predict(inputs, priority, timeout)
                for inputs in inputs_list
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # For larger batches, use batch processing if supported
        try:
            return await self.predict(
                inputs_list, 
                priority, 
                timeout,
                metadata={"batch_size": len(inputs_list)}
            )
        except Exception as e:
            # Fallback to individual processing
            logger.warning(f"Batch processing failed, falling back to individual: {e}")
            tasks = [
                self.predict(inputs, priority, timeout)
                for inputs in inputs_list
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Health status dictionary
        """
        checks = {}
        overall_healthy = True
        
        # Engine state check
        checks["engine_state"] = {
            "healthy": self.state == InferenceState.READY,
            "state": self.state.value
        }
        
        if self.state != InferenceState.READY:
            overall_healthy = False
        
        # Model check
        try:
            checks["model"] = {
                "healthy": self.model is not None and hasattr(self.model, 'predict'),
                "type": type(self.model).__name__,
                "device": str(self.device)
            }
        except Exception as e:
            checks["model"] = {"healthy": False, "error": str(e)}
            overall_healthy = False
        
        # Device check
        try:
            if self.device.type == "cuda":
                checks["device"] = {
                    "healthy": torch.cuda.is_available(),
                    "type": "cuda",
                    "memory_allocated_mb": torch.cuda.memory_allocated(self.device) / (1024**2),
                    "memory_reserved_mb": torch.cuda.memory_reserved(self.device) / (1024**2)
                }
            else:
                checks["device"] = {
                    "healthy": True,
                    "type": self.device.type
                }
        except Exception as e:
            checks["device"] = {"healthy": False, "error": str(e)}
            overall_healthy = False
        
        # Memory check
        try:
            memory_stats = self._memory_manager.get_memory_stats()
            checks["memory"] = {
                "healthy": memory_stats["cache"]["cached_models"] < self._memory_manager.max_cached_models,
                "stats": memory_stats
            }
        except Exception as e:
            checks["memory"] = {"healthy": False, "error": str(e)}
        
        # Queue check
        with self._stats_lock:
            checks["queue"] = {
                "healthy": self._stats.current_queue_size < 100,  # Arbitrary threshold
                "size": self._stats.current_queue_size,
                "peak_size": self._stats.peak_queue_size
            }
            
            if self._stats.current_queue_size > 100:
                overall_healthy = False
        
        # Error rate check
        with self._stats_lock:
            total_requests = self._stats.requests_processed + self._stats.requests_failed
            error_rate = (self._stats.requests_failed / total_requests) if total_requests > 0 else 0
            
            checks["error_rate"] = {
                "healthy": error_rate < 0.1,  # Less than 10% error rate
                "rate": error_rate,
                "consecutive_errors": self._consecutive_errors
            }
            
            if error_rate >= 0.1 or self._consecutive_errors > 5:
                overall_healthy = False
        
        return {
            "healthy": overall_healthy,
            "checks": checks,
            "timestamp": time.time()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self._stats_lock:
            return {
                "requests_processed": self._stats.requests_processed,
                "requests_failed": self._stats.requests_failed,
                "total_processing_time": self._stats.total_processing_time,
                "total_queue_time": self._stats.total_queue_time,
                "average_processing_time": self._stats.average_processing_time,
                "average_queue_time": self._stats.average_queue_time,
                "current_queue_size": self._stats.current_queue_size,
                "peak_queue_size": self._stats.peak_queue_size,
                "memory_usage_mb": self._stats.memory_usage_mb,
                "gpu_utilization_percent": self._stats.gpu_utilization_percent,
                "error_rate": (
                    self._stats.requests_failed / 
                    (self._stats.requests_processed + self._stats.requests_failed)
                ) if (self._stats.requests_processed + self._stats.requests_failed) > 0 else 0.0,
                "consecutive_errors": self._consecutive_errors,
                "last_prediction_time": self._last_prediction_time
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        stats = self.get_stats()
        
        # Calculate percentiles for prediction times
        prediction_times = self._prediction_times[-1000:]  # Last 1000 predictions
        percentiles = {}
        
        if prediction_times:
            sorted_times = sorted(prediction_times)
            percentiles = {
                "p50": sorted_times[int(len(sorted_times) * 0.5)],
                "p90": sorted_times[int(len(sorted_times) * 0.9)],
                "p95": sorted_times[int(len(sorted_times) * 0.95)],
                "p99": sorted_times[int(len(sorted_times) * 0.99)],
            }
        
        return {
            "stats": stats,
            "performance": {
                "prediction_time_percentiles": percentiles,
                "throughput_predictions_per_second": (
                    len(prediction_times) / sum(prediction_times)
                ) if prediction_times else 0.0,
                "memory_efficiency": self._calculate_memory_efficiency(),
                "device_utilization": self._calculate_device_utilization()
            },
            "recommendations": self._generate_performance_recommendations()
        }
    
    def _get_device(self) -> torch.device:
        """Get the device for inference."""
        if hasattr(self.config, 'device') and hasattr(self.config.device, 'get_torch_device'):
            return self.config.device.get_torch_device()
        
        # Fallback device selection
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _validate_configuration(self) -> None:
        """Validate engine configuration."""
        if not self.model:
            raise ConfigurationError(
                config_field="model",
                details="Model is required for inference engine"
            )
        
        if not hasattr(self.model, 'predict'):
            raise ConfigurationError(
                config_field="model",
                details="Model must have a 'predict' method"
            )
        
        # Validate device availability
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise DeviceError(
                device=str(self.device),
                operation="initialization",
                details="CUDA device specified but CUDA is not available"
            )
    
    async def _initialize_model(self) -> None:
        """Initialize the model for inference."""
        try:
            logger.debug("Initializing model for inference...")
            
            # Move model to device if needed
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'to'):
                self.model.model = self.model.model.to(self.device)
            
            # Perform warmup
            if hasattr(self.model, 'warmup'):
                warmup_iterations = getattr(self.config, 'performance', {}).get('warmup_iterations', 3)
                await asyncio.get_event_loop().run_in_executor(
                    None, self.model.warmup, warmup_iterations
                )
            
            # Optimize for inference if supported
            if hasattr(self.model, 'optimize_for_inference'):
                await asyncio.get_event_loop().run_in_executor(
                    None, self.model.optimize_for_inference
                )
            
            logger.debug("Model initialization completed")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise ConfigurationError(
                config_field="model_initialization",
                details=f"Failed to initialize model: {e}",
                cause=e
            )
    
    def _get_optimal_worker_count(self) -> int:
        """Determine optimal number of worker threads."""
        if self.device.type == "cuda":
            # For GPU inference, fewer workers to avoid context switching
            return 2
        else:
            # For CPU inference, more workers can help with parallelization
            import os
            return min(4, os.cpu_count() or 1)
    
    async def _worker_loop(self, worker_name: str) -> None:
        """Worker loop for processing prediction requests."""
        logger.debug(f"Worker {worker_name} started")
        
        while not self._shutdown_event.is_set():
            try:
                # Get next request with timeout
                try:
                    _, queue_time, request = await asyncio.wait_for(
                        self._request_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the request
                await self._process_request(request, queue_time, worker_name)
                
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                self._consecutive_errors += 1
        
        logger.debug(f"Worker {worker_name} stopped")
    
    async def _process_request(self, request: PredictionRequest, queue_time: float, worker_name: str) -> None:
        """Process a single prediction request."""
        start_time = time.time()
        processing_start = start_time
        queue_duration = start_time - queue_time
        
        result = PredictionResult(
            request_id=request.request_id,
            result=None,
            success=False,
            queue_time=queue_duration
        )
        
        try:
            logger.debug(f"Worker {worker_name} processing request {request.request_id}")
            
            # Use memory management context
            with self._memory_manager.inference_context(model_name=getattr(self.model, 'model_name', None)):
                # Perform prediction
                prediction_result = await asyncio.get_event_loop().run_in_executor(
                    None, self._safe_predict, request.inputs
                )
                
                result.result = prediction_result
                result.success = True
                self._consecutive_errors = 0  # Reset consecutive error count
        
        except Exception as e:
            logger.error(f"Prediction failed for request {request.request_id}: {e}")
            result.error = str(e)
            result.success = False
            self._error_count += 1
            self._consecutive_errors += 1
        
        # Calculate timing
        end_time = time.time()
        result.processing_time = end_time - processing_start
        result.total_time = end_time - request.created_at
        
        # Update statistics
        self._update_stats(result)
        
        # Store result (in a real implementation, this would use a result store)
        # For now, we'll use a simple approach
        if not hasattr(self, '_results'):
            self._results = {}
        self._results[request.request_id] = result
        
        logger.debug(f"Request {request.request_id} completed in {result.total_time:.3f}s")
    
    def _safe_predict(self, inputs: Any) -> Any:
        """Safely perform prediction with error handling."""
        try:
            if hasattr(self.model, 'predict'):
                return self.model.predict(inputs)
            else:
                # Fallback for models without predict method
                if hasattr(self.model, '__call__'):
                    return self.model(inputs)
                else:
                    raise PredictionError(
                        details="Model has no predict method or is not callable",
                        context={"model_type": type(self.model).__name__}
                    )
        except torch.cuda.OutOfMemoryError as e:
            raise ResourceExhaustedError(
                resource="GPU memory",
                details=f"CUDA out of memory: {e}",
                cause=e
            )
        except Exception as e:
            raise PredictionError(
                details=f"Model prediction failed: {e}",
                cause=e,
                context={"model_type": type(self.model).__name__}
            )
    
    async def _wait_for_result(self, request_id: str) -> PredictionResult:
        """Wait for prediction result."""
        if not hasattr(self, '_results'):
            self._results = {}
        
        # Poll for result (in production, use proper async notifications)
        while request_id not in self._results:
            await asyncio.sleep(0.01)  # 10ms polling
        
        return self._results.pop(request_id)
    
    def _update_stats(self, result: PredictionResult) -> None:
        """Update engine statistics."""
        with self._stats_lock:
            if result.success:
                self._stats.requests_processed += 1
                self._stats.total_processing_time += result.processing_time
                self._stats.total_queue_time += result.queue_time
                
                # Update averages
                self._stats.average_processing_time = (
                    self._stats.total_processing_time / self._stats.requests_processed
                )
                self._stats.average_queue_time = (
                    self._stats.total_queue_time / self._stats.requests_processed
                )
                
                # Track prediction times for percentile calculation
                self._prediction_times.append(result.processing_time)
                if len(self._prediction_times) > 10000:  # Keep last 10k
                    self._prediction_times = self._prediction_times[-5000:]
                
                self._last_prediction_time = time.time()
            else:
                self._stats.requests_failed += 1
            
            # Update queue size
            self._stats.current_queue_size = self._request_queue.qsize()
    
    async def _health_monitor(self) -> None:
        """Background health monitoring."""
        while not self._shutdown_event.is_set():
            try:
                # Update memory statistics
                memory_stats = self._memory_manager.get_memory_stats()
                with self._stats_lock:
                    self._stats.memory_usage_mb = memory_stats.get("system", {}).get("percent_used", 0)
                
                # Update GPU utilization if available
                if self.device.type == "cuda":
                    # Note: Real GPU utilization would require nvidia-ml-py
                    allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
                    total = torch.cuda.get_device_properties(self.device).total_memory / (1024**2)
                    with self._stats_lock:
                        self._stats.gpu_utilization_percent = (allocated / total) * 100
                
                # Check for consecutive errors
                if self._consecutive_errors > 10:
                    logger.warning(f"High consecutive error count: {self._consecutive_errors}")
                    self.state = InferenceState.ERROR
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)  # Longer wait on error
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score (0-1)."""
        try:
            memory_stats = self._memory_manager.get_memory_stats()
            cache_stats = memory_stats.get("cache", {})
            hit_rate = cache_stats.get("hit_rate", 0)
            
            # Simple efficiency metric based on cache hit rate
            return hit_rate
        except Exception:
            return 0.0
    
    def _calculate_device_utilization(self) -> float:
        """Calculate device utilization score (0-1)."""
        try:
            if self.device.type == "cuda":
                allocated = torch.cuda.memory_allocated(self.device)
                total = torch.cuda.get_device_properties(self.device).total_memory
                return allocated / total
            else:
                # For CPU, use processing time as utilization proxy
                if self._prediction_times:
                    avg_time = sum(self._prediction_times[-100:]) / len(self._prediction_times[-100:])
                    # Normalize to 0-1 range (assume 1s is full utilization)
                    return min(1.0, avg_time)
                return 0.0
        except Exception:
            return 0.0
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Memory recommendations
        try:
            memory_stats = self._memory_manager.get_memory_stats()
            cache_stats = memory_stats.get("cache", {})
            
            if cache_stats.get("hit_rate", 0) < 0.8:
                recommendations.append("Consider increasing model cache size for better hit rate")
            
            if cache_stats.get("cached_models", 0) >= self._memory_manager.max_cached_models:
                recommendations.append("Model cache is full, consider increasing max_cached_models")
        except Exception:
            pass
        
        # Performance recommendations
        with self._stats_lock:
            if self._stats.average_processing_time > 1.0:
                recommendations.append("High average processing time, consider model optimization")
            
            if self._stats.average_queue_time > 0.1:
                recommendations.append("High queue times, consider increasing worker count")
            
            error_rate = (
                self._stats.requests_failed / 
                (self._stats.requests_processed + self._stats.requests_failed)
            ) if (self._stats.requests_processed + self._stats.requests_failed) > 0 else 0
            
            if error_rate > 0.05:
                recommendations.append("High error rate detected, check model and input validation")
        
        # Device recommendations
        if self.device.type == "cuda":
            try:
                utilization = self._calculate_device_utilization()
                if utilization < 0.3:
                    recommendations.append("Low GPU utilization, consider increasing batch size")
                elif utilization > 0.9:
                    recommendations.append("High GPU utilization, consider reducing batch size")
            except Exception:
                pass
        
        return recommendations
    
    def _cleanup_resources(self) -> None:
        """Cleanup engine resources."""
        logger.debug("Cleaning up inference engine resources...")
        
        # Clear result cache
        if hasattr(self, '_results'):
            self._results.clear()
        
        # Clear prediction times
        self._prediction_times.clear()
        
        # GPU memory cleanup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.debug("Resource cleanup completed")


def create_inference_engine(model: Any, config: InferenceConfig) -> InferenceEngine:
    """Create a standard inference engine."""
    return InferenceEngine(model, config)


def create_ultra_fast_inference_engine(model: Any, config: InferenceConfig) -> InferenceEngine:
    """Create an ultra-fast optimized inference engine."""
    # Apply additional optimizations
    engine = InferenceEngine(model, config)
    
    # Enable optimizations
    if hasattr(config, 'performance'):
        config.performance.enable_cuda_graphs = True
        config.performance.enable_jit_compilation = True
        config.performance.enable_memory_pool = True
    
    return engine


def create_hybrid_inference_engine(model: Any, config: InferenceConfig) -> InferenceEngine:
    """Create a hybrid inference engine with balanced performance and stability."""
    engine = InferenceEngine(model, config)
    
    # Apply moderate optimizations
    if hasattr(config, 'performance'):
        config.performance.enable_cuda_graphs = False  # More stable
        config.performance.enable_jit_compilation = True
        config.performance.enable_memory_pool = True
    
    return engine

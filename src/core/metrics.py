"""
Metrics collection and monitoring for PyTorch Inference Framework.

This module provides comprehensive metrics collection, aggregation,
and monitoring capabilities for performance tracking and observability.
"""

import logging
import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Individual metric value with timestamp."""
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    metric_type: MetricType
    description: str
    unit: Optional[str] = None
    labels: List[str] = field(default_factory=list)


class MetricCollector:
    """Base class for metric collectors."""
    
    def __init__(self, name: str):
        self.name = name
        self.values: deque = deque(maxlen=10000)  # Keep last 10k values
        self._lock = threading.Lock()
    
    def add_value(self, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Add a value to the metric."""
        with self._lock:
            metric_value = MetricValue(
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels or {}
            )
            self.values.append(metric_value)
    
    def get_recent_values(self, minutes: int = 5) -> List[MetricValue]:
        """Get recent values within specified time window."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        with self._lock:
            return [v for v in self.values if v.timestamp > cutoff_time]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the metric."""
        with self._lock:
            if not self.values:
                return {"count": 0, "latest": None}
            
            values = [v.value for v in self.values]
            latest = self.values[-1]
            
            return {
                "count": len(values),
                "latest": latest.value,
                "latest_timestamp": latest.timestamp.isoformat(),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values) if len(values) > 0 else 0
            }


class Counter(MetricCollector):
    """Counter metric that only increases."""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name)
        self.description = description
        self._total = 0
        self._lock = threading.Lock()
    
    def increment(self, value: Union[int, float] = 1, labels: Optional[Dict[str, str]] = None):
        """Increment the counter."""
        with self._lock:
            self._total += value
            self.add_value(self._total, labels)
    
    def get_total(self) -> Union[int, float]:
        """Get total count."""
        with self._lock:
            return self._total
    
    def reset(self):
        """Reset the counter."""
        with self._lock:
            self._total = 0
            self.values.clear()


class Gauge(MetricCollector):
    """Gauge metric that can go up and down."""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name)
        self.description = description
        self._current_value = 0
        self._lock = threading.Lock()
    
    def set(self, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Set the gauge value."""
        with self._lock:
            self._current_value = value
            self.add_value(value, labels)
    
    def increment(self, value: Union[int, float] = 1, labels: Optional[Dict[str, str]] = None):
        """Increment the gauge value."""
        with self._lock:
            self._current_value += value
            self.add_value(self._current_value, labels)
    
    def decrement(self, value: Union[int, float] = 1, labels: Optional[Dict[str, str]] = None):
        """Decrement the gauge value."""
        with self._lock:
            self._current_value -= value
            self.add_value(self._current_value, labels)
    
    def get_current(self) -> Union[int, float]:
        """Get current gauge value."""
        with self._lock:
            return self._current_value


class Histogram(MetricCollector):
    """Histogram metric for tracking distributions."""
    
    def __init__(self, name: str, description: str = "", buckets: Optional[List[float]] = None):
        super().__init__(name)
        self.description = description
        self.buckets = buckets or [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
        self.bucket_counts = {bucket: 0 for bucket in self.buckets}
        self._lock = threading.Lock()
    
    def observe(self, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Observe a value."""
        with self._lock:
            # Add to bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    self.bucket_counts[bucket] += 1
            
            self.add_value(value, labels)
    
    def get_buckets(self) -> Dict[float, int]:
        """Get bucket counts."""
        with self._lock:
            return self.bucket_counts.copy()
    
    def get_percentiles(self, percentiles: List[float] = None) -> Dict[float, float]:
        """Calculate percentiles from observed values."""
        percentiles = percentiles or [50, 90, 95, 99]
        
        with self._lock:
            if not self.values:
                return {p: 0 for p in percentiles}
            
            values = sorted([v.value for v in self.values])
            result = {}
            
            for p in percentiles:
                if p <= 0:
                    result[p] = values[0]
                elif p >= 100:
                    result[p] = values[-1]
                else:
                    index = int((p / 100) * (len(values) - 1))
                    result[p] = values[index]
            
            return result


class Timer:
    """Timer context manager for measuring execution time."""
    
    def __init__(self, metric_collector: MetricCollector, labels: Optional[Dict[str, str]] = None):
        self.metric_collector = metric_collector
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metric_collector.add_value(duration, self.labels)


class PerformanceMonitor:
    """Monitor for tracking performance metrics."""
    
    def __init__(self):
        self.request_count = Counter("requests_total", "Total number of requests")
        self.request_duration = Histogram("request_duration_seconds", "Request duration")
        self.error_count = Counter("errors_total", "Total number of errors")
        self.memory_usage = Gauge("memory_usage_bytes", "Memory usage in bytes")
        self.cpu_usage = Gauge("cpu_usage_percent", "CPU usage percentage")
        self.model_load_time = Histogram("model_load_time_seconds", "Model load time")
        self.prediction_time = Histogram("prediction_time_seconds", "Prediction time")
        
        self._active_requests = Gauge("active_requests", "Number of active requests")
        self._model_cache_size = Gauge("model_cache_size", "Number of cached models")
        
        self.custom_metrics: Dict[str, MetricCollector] = {}
    
    def record_request(self, duration: float, success: bool = True, 
                      labels: Optional[Dict[str, str]] = None):
        """Record a request with duration and success status."""
        self.request_count.increment(labels=labels)
        self.request_duration.observe(duration, labels=labels)
        
        if not success:
            self.error_count.increment(labels=labels)
    
    def record_prediction(self, duration: float, model_name: str):
        """Record a prediction with timing."""
        labels = {"model": model_name}
        self.prediction_time.observe(duration, labels=labels)
    
    def record_model_load(self, duration: float, model_name: str):
        """Record model loading time."""
        labels = {"model": model_name}
        self.model_load_time.observe(duration, labels=labels)
    
    def set_memory_usage(self, usage_bytes: int):
        """Set current memory usage."""
        self.memory_usage.set(usage_bytes)
    
    def set_cpu_usage(self, usage_percent: float):
        """Set current CPU usage."""
        self.cpu_usage.set(usage_percent)
    
    def set_active_requests(self, count: int):
        """Set number of active requests."""
        self._active_requests.set(count)
    
    def set_model_cache_size(self, size: int):
        """Set model cache size."""
        self._model_cache_size.set(size)
    
    def add_custom_metric(self, name: str, metric: MetricCollector):
        """Add a custom metric."""
        self.custom_metrics[name] = metric
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics summaries."""
        metrics = {
            "requests_total": self.request_count.get_summary(),
            "request_duration_seconds": self.request_duration.get_summary(),
            "errors_total": self.error_count.get_summary(),
            "memory_usage_bytes": self.memory_usage.get_summary(),
            "cpu_usage_percent": self.cpu_usage.get_summary(),
            "model_load_time_seconds": self.model_load_time.get_summary(),
            "prediction_time_seconds": self.prediction_time.get_summary(),
            "active_requests": self._active_requests.get_summary(),
            "model_cache_size": self._model_cache_size.get_summary()
        }
        
        # Add custom metrics
        for name, metric in self.custom_metrics.items():
            metrics[name] = metric.get_summary()
        
        return metrics


class MetricsCollector:
    """
    Main metrics collector for the inference framework.
    """
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.performance_monitor = PerformanceMonitor()
        self.system_metrics = SystemMetricsCollector()
        self.business_metrics = BusinessMetricsCollector()
        
        self._collection_interval = 30  # seconds
        self._collection_task = None
        self._running = False
        
        logger.debug("MetricsCollector initialized")
    
    async def start(self):
        """Start metrics collection."""
        if self._running:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collection started")
    
    async def stop(self):
        """Stop metrics collection."""
        self._running = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Metrics collection stopped")
    
    async def _collection_loop(self):
        """Main collection loop."""
        try:
            while self._running:
                try:
                    # Collect system metrics
                    await self.system_metrics.collect()
                    
                    # Update performance monitor with system metrics
                    system_stats = self.system_metrics.get_latest_stats()
                    if system_stats:
                        self.performance_monitor.set_memory_usage(
                            system_stats.get("memory_usage_bytes", 0)
                        )
                        self.performance_monitor.set_cpu_usage(
                            system_stats.get("cpu_usage_percent", 0)
                        )
                    
                    await asyncio.sleep(self._collection_interval)
                    
                except Exception as e:
                    logger.error(f"Metrics collection error: {e}")
                    await asyncio.sleep(5)  # Brief pause on error
                    
        except asyncio.CancelledError:
            logger.debug("Metrics collection loop cancelled")
    
    def record_request_start(self, request_id: str):
        """Record start of request processing."""
        self.business_metrics.record_request_start(request_id)
    
    def record_request_end(self, request_id: str, success: bool = True, 
                          model_name: Optional[str] = None):
        """Record end of request processing."""
        duration = self.business_metrics.record_request_end(request_id, success)
        
        if duration is not None:
            labels = {"model": model_name} if model_name else None
            self.performance_monitor.record_request(duration, success, labels)
    
    def record_model_operation(self, operation: str, model_name: str, duration: float):
        """Record model operation (load, unload, etc.)."""
        if operation == "load":
            self.performance_monitor.record_model_load(duration, model_name)
        
        self.business_metrics.record_model_operation(operation, model_name, duration)
    
    def record_prediction(self, model_name: str, duration: float, success: bool = True):
        """Record prediction operation."""
        self.performance_monitor.record_prediction(duration, model_name)
        self.business_metrics.record_prediction(model_name, duration, success)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "performance": self.performance_monitor.get_all_metrics(),
            "system": self.system_metrics.get_stats(),
            "business": self.business_metrics.get_stats()
        }
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health-related metrics."""
        performance_metrics = self.performance_monitor.get_all_metrics()
        
        # Calculate health indicators
        error_rate = 0
        if performance_metrics["requests_total"]["count"] > 0:
            error_rate = (performance_metrics["errors_total"]["count"] / 
                         performance_metrics["requests_total"]["count"])
        
        avg_response_time = performance_metrics["request_duration_seconds"].get("mean", 0)
        memory_usage = performance_metrics["memory_usage_bytes"].get("latest", 0)
        cpu_usage = performance_metrics["cpu_usage_percent"].get("latest", 0)
        
        return {
            "error_rate": error_rate,
            "avg_response_time_seconds": avg_response_time,
            "memory_usage_bytes": memory_usage,
            "cpu_usage_percent": cpu_usage,
            "active_requests": performance_metrics["active_requests"].get("latest", 0),
            "total_requests": performance_metrics["requests_total"]["count"],
            "total_errors": performance_metrics["errors_total"]["count"]
        }


class SystemMetricsCollector:
    """Collector for system-level metrics."""
    
    def __init__(self):
        self.latest_stats = {}
    
    async def collect(self):
        """Collect system metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            self.latest_stats = {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_bytes": memory.used,
                "memory_total_bytes": memory.total,
                "memory_usage_percent": memory.percent,
                "disk_usage_bytes": disk.used,
                "disk_total_bytes": disk.total,
                "disk_usage_percent": (disk.used / disk.total) * 100,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except ImportError:
            # psutil not available, use basic metrics
            self.latest_stats = {
                "cpu_usage_percent": 0,
                "memory_usage_bytes": 0,
                "memory_total_bytes": 0,
                "memory_usage_percent": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def get_latest_stats(self) -> Dict[str, Any]:
        """Get latest system statistics."""
        return self.latest_stats.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return self.get_latest_stats()


class BusinessMetricsCollector:
    """Collector for business-level metrics."""
    
    def __init__(self):
        self.active_requests: Dict[str, float] = {}
        self.request_history: List[Dict[str, Any]] = []
        self.model_operations: List[Dict[str, Any]] = []
        self.predictions: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def record_request_start(self, request_id: str):
        """Record start of request."""
        with self._lock:
            self.active_requests[request_id] = time.time()
    
    def record_request_end(self, request_id: str, success: bool = True) -> Optional[float]:
        """Record end of request and return duration."""
        with self._lock:
            start_time = self.active_requests.pop(request_id, None)
            
            if start_time:
                duration = time.time() - start_time
                
                self.request_history.append({
                    "request_id": request_id,
                    "duration": duration,
                    "success": success,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Keep only recent history
                cutoff_time = time.time() - 3600  # 1 hour
                self.request_history = [
                    r for r in self.request_history 
                    if datetime.fromisoformat(r["timestamp"]).timestamp() > cutoff_time
                ]
                
                return duration
            
            return None
    
    def record_model_operation(self, operation: str, model_name: str, duration: float):
        """Record model operation."""
        with self._lock:
            self.model_operations.append({
                "operation": operation,
                "model_name": model_name,
                "duration": duration,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Keep only recent operations
            cutoff_time = time.time() - 3600  # 1 hour
            self.model_operations = [
                op for op in self.model_operations
                if datetime.fromisoformat(op["timestamp"]).timestamp() > cutoff_time
            ]
    
    def record_prediction(self, model_name: str, duration: float, success: bool = True):
        """Record prediction operation."""
        with self._lock:
            self.predictions.append({
                "model_name": model_name,
                "duration": duration,
                "success": success,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Keep only recent predictions
            cutoff_time = time.time() - 3600  # 1 hour
            self.predictions = [
                p for p in self.predictions
                if datetime.fromisoformat(p["timestamp"]).timestamp() > cutoff_time
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get business statistics."""
        with self._lock:
            return {
                "active_requests": len(self.active_requests),
                "completed_requests": len(self.request_history),
                "successful_requests": sum(1 for r in self.request_history if r["success"]),
                "model_operations": len(self.model_operations),
                "predictions": len(self.predictions),
                "successful_predictions": sum(1 for p in self.predictions if p["success"])
            }


# Factory function

def create_metrics_collector() -> MetricsCollector:
    """
    Create a metrics collector instance.
    
    Returns:
        MetricsCollector instance
    """
    try:
        metrics_collector = MetricsCollector()
        logger.info("Metrics collector created successfully")
        return metrics_collector
        
    except Exception as e:
        logger.error(f"Failed to create metrics collector: {e}")
        raise


# Decorator for timing functions

def timed_operation(metric_name: str, metrics_collector: Optional[MetricsCollector] = None):
    """Decorator to time function execution and record metrics."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                if metrics_collector:
                    # Record timing metric (would need to be implemented based on specific needs)
                    pass
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                if metrics_collector:
                    # Record timing metric
                    pass
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

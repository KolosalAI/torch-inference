"""
Performance monitoring and metrics collection utilities.

This module provides comprehensive monitoring capabilities for the inference framework,
including performance metrics, resource usage tracking, and alerting.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from enum import Enum
import statistics
import json


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """Performance statistics."""
    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    avg: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    
    def update(self, value: float):
        """Update statistics with new value."""
        self.count += 1
        self.sum += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.avg = self.sum / self.count


@dataclass
class PerformanceMetrics:
    """Performance metrics for inference operations."""
    inference_time: float = 0.0
    preprocessing_time: float = 0.0
    postprocessing_time: float = 0.0
    total_time: float = 0.0
    throughput: float = 0.0
    memory_usage: int = 0
    gpu_utilization: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'inference_time': self.inference_time,
            'preprocessing_time': self.preprocessing_time,
            'postprocessing_time': self.postprocessing_time,
            'total_time': self.total_time,
            'throughput': self.throughput,
            'memory_usage': self.memory_usage,
            'gpu_utilization': self.gpu_utilization
        }


class MetricsCollector:
    """Thread-safe metrics collector."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._lock = threading.RLock()
        self._callbacks: List[Callable[[Metric], None]] = []
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        with self._lock:
            self._metrics[name].append(metric)
            
            # Trim history if needed
            if len(self._metrics[name]) > self.max_history:
                self._metrics[name] = self._metrics[name][-self.max_history:]
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"Metric callback failed: {e}")
    
    def record_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, tags)
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        self.record_metric(name, value, MetricType.GAUGE, tags)
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timer metric (duration in seconds)."""
        self.record_metric(name, duration, MetricType.TIMER, tags)
    
    def record_batch_metrics(self, batch_size: int, processing_time: float, 
                           queue_size: int, memory_usage: Dict[str, float]) -> None:
        """Record batch processing metrics."""
        self.record_gauge("batch_size", batch_size)
        self.record_timer("batch_processing_time", processing_time)
        self.record_gauge("queue_size", queue_size)
        
        for key, value in memory_usage.items():
            self.record_gauge(f"memory_{key}", value)
    
    def get_stats(self, name: str) -> Optional[PerformanceStats]:
        """Get statistics for a metric."""
        with self._lock:
            if name not in self._metrics:
                return None
            
            metrics = self._metrics[name]
            if not metrics:
                return None
            
            values = [m.value for m in metrics]
            stats = PerformanceStats()
            
            stats.count = len(values)
            stats.sum = sum(values)
            stats.min = min(values)
            stats.max = max(values)
            stats.avg = statistics.mean(values)
            
            if len(values) > 1:
                try:
                    stats.p50 = statistics.median(values)
                    stats.p95 = statistics.quantiles(values, n=20)[18]  # 95th percentile
                    stats.p99 = statistics.quantiles(values, n=100)[98]  # 99th percentile
                except statistics.StatisticsError:
                    pass  # Not enough data for quantiles
            
            return stats
    
    def get_recent_metrics(self, name: str, seconds: float = 60.0) -> List[Metric]:
        """Get metrics from the last N seconds."""
        cutoff_time = time.time() - seconds
        
        with self._lock:
            if name not in self._metrics:
                return []
            
            return [m for m in self._metrics[name] if m.timestamp >= cutoff_time]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        
        with self._lock:
            for name, metrics in self._metrics.items():
                if not metrics:
                    continue
                
                stats = self.get_stats(name)
                if stats:
                    summary[name] = {
                        "count": stats.count,
                        "avg": stats.avg,
                        "min": stats.min,
                        "max": stats.max,
                        "p50": stats.p50,
                        "p95": stats.p95,
                        "p99": stats.p99,
                        "latest": metrics[-1].value,
                        "latest_timestamp": metrics[-1].timestamp
                    }
        
        return summary
    
    def add_callback(self, callback: Callable[[Metric], None]) -> None:
        """Add a callback for metric events."""
        self._callbacks.append(callback)
    
    def clear_history(self) -> None:
        """Clear all metric history."""
        with self._lock:
            self._metrics.clear()


class PerformanceMonitor:
    """High-level performance monitoring."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self._timers: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Additional attributes for request tracking
        self.request_times = deque(maxlen=1000)
        self.total_requests = 0
        self.active_requests: Dict[str, float] = {}
        self.start_time = time.time()
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        with self._lock:
            self._timers[name] = time.time()
    
    def end_timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """End a named timer and record the duration."""
        with self._lock:
            if name not in self._timers:
                logger.warning(f"Timer '{name}' not found")
                return 0.0
            
            duration = time.time() - self._timers[name]
            del self._timers[name]
        
        self.metrics_collector.record_timer(name, duration, tags)
        return duration
    
    def start_request(self, request_id: str) -> None:
        """Start timing a request."""
        with self._lock:
            self.active_requests[request_id] = time.time()
    
    def end_request(self, request_id: str) -> float:
        """End timing a request and record the duration."""
        with self._lock:
            if request_id not in self.active_requests:
                logger.warning(f"Request '{request_id}' not found")
                return 0.0
            
            start_time = self.active_requests.pop(request_id)
            duration = time.time() - start_time
            
            self.request_times.append(duration)
            self.total_requests += 1
        
        self.metrics_collector.record_timer("request_duration", duration)
        return duration
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self._lock:
            stats = {
                "total_requests": self.total_requests,
                "active_requests": len(self.active_requests),
                "uptime": time.time() - self.start_time
            }
            
            if self.request_times:
                times = list(self.request_times)
                stats.update({
                    "avg_request_time": statistics.mean(times),
                    "min_request_time": min(times),
                    "max_request_time": max(times),
                    "recent_requests": len(times)
                })
                
                if len(times) > 1:
                    try:
                        stats["median_request_time"] = statistics.median(times)
                    except statistics.StatisticsError:
                        pass
        
        return stats
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        stats = self.get_current_stats()
        metrics_summary = self.metrics_collector.get_summary()
        
        return {
            "timestamp": time.time(),
            "stats": stats,
            "metrics": metrics_summary,
            "system_info": {
                "uptime": stats.get("uptime", 0),
                "total_requests": stats.get("total_requests", 0),
                "active_requests": stats.get("active_requests", 0)
            }
        }
    
    def time_context(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        return TimerContext(self, name, tags)
    
    def record_throughput(self, name: str, count: int, duration: float, 
                         tags: Optional[Dict[str, str]] = None) -> None:
        """Record throughput metrics."""
        if duration > 0:
            throughput = count / duration
            self.metrics_collector.record_gauge(f"{name}_throughput", throughput, tags)
            self.metrics_collector.record_timer(f"{name}_duration", duration, tags)
            self.metrics_collector.record_counter(f"{name}_count", count, tags)
    
    def get_performance_summary(self, window_seconds: float = 300.0) -> Dict[str, Any]:
        """Get performance summary for the last window."""
        summary = {
            "window_seconds": window_seconds,
            "timestamp": time.time(),
            "total_requests": self.total_requests,
            "metrics": {}
        }
        
        # Get summary from metrics collector
        all_metrics = self.metrics_collector.get_summary()
        
        # Filter and organize metrics
        for metric_name, stats in all_metrics.items():
            if metric_name.endswith("_throughput"):
                base_name = metric_name.replace("_throughput", "")
                if base_name not in summary["metrics"]:
                    summary["metrics"][base_name] = {}
                summary["metrics"][base_name]["throughput"] = stats
            elif metric_name.endswith("_duration"):
                base_name = metric_name.replace("_duration", "")
                if base_name not in summary["metrics"]:
                    summary["metrics"][base_name] = {}
                summary["metrics"][base_name]["duration"] = stats
            elif metric_name.endswith("_count"):
                base_name = metric_name.replace("_count", "")
                if base_name not in summary["metrics"]:
                    summary["metrics"][base_name] = {}
                summary["metrics"][base_name]["count"] = stats
            else:
                summary["metrics"][metric_name] = stats
        
        return summary


class TimerContext:
    """Context manager for performance timing."""
    
    def __init__(self, monitor: PerformanceMonitor, name: str, tags: Optional[Dict[str, str]] = None):
        self.monitor = monitor
        self.name = name
        self.tags = tags
    
    def __enter__(self):
        self.monitor.start_timer(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.end_timer(self.name, self.tags)


class AlertManager:
    """Simple alerting system for performance issues."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self._alert_rules: List[Dict[str, Any]] = []
        self._alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Add default alert callback
        self.metrics_collector.add_callback(self._check_alerts)
    
    def add_alert_rule(self, metric_name: str, condition: str, threshold: float, 
                      message: str, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Add an alert rule.
        
        Args:
            metric_name: Name of the metric to monitor
            condition: Condition ('>', '<', '>=', '<=', '==', '!=')
            threshold: Threshold value
            message: Alert message
            tags: Optional tags to filter metrics
        """
        rule = {
            "metric_name": metric_name,
            "condition": condition,
            "threshold": threshold,
            "message": message,
            "tags": tags or {},
            "triggered": False,
            "last_trigger_time": 0
        }
        self._alert_rules.append(rule)
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for alert events."""
        self._alert_callbacks.append(callback)
    
    def _check_alerts(self, metric: Metric) -> None:
        """Check if any alert rules are triggered."""
        for rule in self._alert_rules:
            if rule["metric_name"] != metric.name:
                continue
            
            # Check tags if specified
            if rule["tags"]:
                if not all(metric.tags.get(k) == v for k, v in rule["tags"].items()):
                    continue
            
            # Check condition
            triggered = self._evaluate_condition(metric.value, rule["condition"], rule["threshold"])
            
            # Trigger alert if condition met and not recently triggered
            if triggered and not rule["triggered"]:
                rule["triggered"] = True
                rule["last_trigger_time"] = time.time()
                
                alert = {
                    "rule": rule,
                    "metric": metric,
                    "timestamp": time.time(),
                    "message": rule["message"]
                }
                
                self._send_alert(alert)
            elif not triggered:
                rule["triggered"] = False
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold
        else:
            logger.error(f"Unknown condition: {condition}")
            return False
    
    def _send_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert to all callbacks."""
        logger.warning(f"ALERT: {alert['message']} (value: {alert['metric'].value})")
        
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")


# Global instances
_global_metrics_collector: Optional[MetricsCollector] = None
_global_performance_monitor: Optional[PerformanceMonitor] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor(get_metrics_collector())
    return _global_performance_monitor

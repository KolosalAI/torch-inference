"""
Enterprise monitoring and observability system.

This module provides comprehensive monitoring including:
- Distributed tracing
- Advanced metrics collection
- Real-time alerting
- Performance analytics
- Business intelligence dashboards
- SLA/SLO monitoring
"""

import time
import asyncio
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import json
import secrets
from abc import ABC, abstractmethod
import statistics

try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    JAEGER_AVAILABLE = True
    OPENTELEMETRY_BASE_AVAILABLE = True
except ImportError:
    # Handle missing dependencies gracefully
    trace = None
    metrics = None
    JaegerExporter = None
    JAEGER_AVAILABLE = False
    OPENTELEMETRY_BASE_AVAILABLE = False
    
try:
    from deprecated import deprecated
    DEPRECATED_AVAILABLE = True
except ImportError:
    # Create a dummy decorator if deprecated is not available
    def deprecated(reason=""):
        def decorator(func):
            return func
        return decorator
    DEPRECATED_AVAILABLE = False

try:
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
    OPENTELEMETRY_AVAILABLE = OPENTELEMETRY_BASE_AVAILABLE
except ImportError:
    # Handle missing opentelemetry dependencies
    PrometheusMetricReader = None
    TracerProvider = None
    BatchSpanProcessor = None
    MeterProvider = None
    RequestsInstrumentor = None
    Counter = None
    Histogram = None
    Gauge = None
    Summary = None
    CollectorRegistry = None
    OPENTELEMETRY_AVAILABLE = False

from .config import SecurityConfig


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class TraceContext:
    """Distributed trace context."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricPoint:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """Monitoring alert."""
    id: str
    name: str
    message: str
    severity: AlertSeverity
    metric_name: str
    threshold_value: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "message": self.message,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "threshold_value": self.threshold_value,
            "current_value": self.current_value,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "acknowledged": self.acknowledged
        }


@dataclass
class SLOTarget:
    """Service Level Objective target."""
    name: str
    target_percentage: float
    time_window_hours: int
    error_budget_consumed: float = 0.0
    
    def is_breached(self) -> bool:
        """Check if SLO is breached."""
        return self.error_budget_consumed > (100 - self.target_percentage)


class DistributedTracing:
    """Distributed tracing implementation."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.service_name = config.monitoring.tracing_service_name
        self.sampling_rate = config.monitoring.tracing_sampling_rate
        
        self._setup_tracing()
        if OPENTELEMETRY_AVAILABLE:
            self.tracer = trace.get_tracer(self.service_name)
        else:
            self.tracer = None
    
    def _setup_tracing(self) -> None:
        """Setup OpenTelemetry tracing."""
        if not OPENTELEMETRY_AVAILABLE or TracerProvider is None:
            logger.warning("OpenTelemetry not available, skipping tracing setup")
            return
            
        # Configure tracer provider
        trace.set_tracer_provider(TracerProvider())
        
        # Setup Jaeger exporter if configured
        if self.config.monitoring.jaeger_endpoint and JAEGER_AVAILABLE and JaegerExporter is not None:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
                collector_endpoint=self.config.monitoring.jaeger_endpoint,
            )
            
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Instrument requests
        if RequestsInstrumentor is not None:
            RequestsInstrumentor().instrument()
    
    def create_span(self, name: str, parent_context: Optional[TraceContext] = None) -> Optional[Any]:
        """Create new trace span."""
        if self.tracer is None:
            logger.debug("Tracer not available, skipping span creation")
            return None
            
        if parent_context:
            # Set parent context
            context = trace.set_span_in_context(
                trace.SpanContext(
                    trace_id=int(parent_context.trace_id, 16),
                    span_id=int(parent_context.span_id, 16),
                    is_remote=True
                )
            )
            span = self.tracer.start_span(name, context=context)
        else:
            span = self.tracer.start_span(name)
        
        return span
    
    def get_current_trace_context(self) -> Optional[TraceContext]:
        """Get current trace context."""
        if trace is None:
            return None
            
        current_span = trace.get_current_span()
        if current_span and current_span.get_span_context().is_valid:
            context = current_span.get_span_context()
            return TraceContext(
                trace_id=format(context.trace_id, '032x'),
                span_id=format(context.span_id, '016x')
            )
        return None
    
    def add_span_attribute(self, key: str, value: Any) -> None:
        """Add attribute to current span."""
        if trace is None:
            return
            
        current_span = trace.get_current_span()
        if current_span:
            current_span.set_attribute(key, str(value))
    
    def add_span_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add event to current span."""
        if trace is None:
            return
            
        current_span = trace.get_current_span()
        if current_span:
            current_span.add_event(name, attributes or {})


class PrometheusMetrics:
    """Prometheus metrics collection."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
        # Check if prometheus dependencies are available
        if not OPENTELEMETRY_AVAILABLE or CollectorRegistry is None:
            logger.warning("Prometheus metrics not available, metrics collection disabled")
            self.registry = None
            return
            
        self.registry = CollectorRegistry()
        
        # Initialize common metrics
        self._init_system_metrics()
        self._init_application_metrics()
        
    def _init_system_metrics(self) -> None:
        """Initialize system-level metrics."""
        if self.registry is None:
            return
            
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'active_connections',
            'Active connections',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
    
    def _init_application_metrics(self) -> None:
        """Initialize application-specific metrics."""
        if self.registry is None:
            return
            
        self.inference_count = Counter(
            'inference_requests_total',
            'Total inference requests',
            ['model', 'status', 'tenant'],
            registry=self.registry
        )
        
        self.inference_duration = Histogram(
            'inference_duration_seconds',
            'Inference duration',
            ['model'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Model accuracy score',
            ['model', 'version'],
            registry=self.registry
        )
        
        self.queue_size = Gauge(
            'request_queue_size',
            'Request queue size',
            registry=self.registry
        )
        
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        self.model_load_time = Histogram(
            'model_load_time_seconds',
            'Model loading time',
            ['model'],
            registry=self.registry
        )
    
    def record_request(self, method: str, endpoint: str, status: str, duration: float) -> None:
        """Record HTTP request metrics."""
        if self.registry is None:
            return
        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_inference(self, model: str, duration: float, status: str = "success", 
                        tenant: str = "default") -> None:
        """Record inference metrics."""
        if self.registry is None:
            return
        self.inference_count.labels(model=model, status=status, tenant=tenant).inc()
        self.inference_duration.labels(model=model).observe(duration)
    
    def update_system_metrics(self, cpu_percent: float, memory_bytes: Dict[str, float], 
                             active_conns: int) -> None:
        """Update system metrics."""
        if self.registry is None:
            return
        self.cpu_usage.set(cpu_percent)
        self.active_connections.set(active_conns)
        
        for mem_type, value in memory_bytes.items():
            self.memory_usage.labels(type=mem_type).set(value)
    
    def update_gpu_metrics(self, gpu_utilization: Dict[str, float]) -> None:
        """Update GPU metrics."""
        if self.registry is None:
            return
        for gpu_id, utilization in gpu_utilization.items():
            self.gpu_utilization.labels(gpu_id=gpu_id).set(utilization)
    
    def get_metric_families(self):
        """Get all metric families for Prometheus scraping."""
        if self.registry is None:
            return []
        return self.registry.collect()


class AlertManager:
    """Advanced alerting system."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.alert_rules: List[Dict[str, Any]] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Load default alert rules
        self._load_default_alert_rules()
    
    def _load_default_alert_rules(self) -> None:
        """Load default alert rules from configuration."""
        thresholds = self.config.monitoring.alert_thresholds
        
        # Error rate alerts
        self.add_alert_rule(
            name="high_error_rate",
            metric_name="error_rate",
            threshold=thresholds.get("error_rate", 0.05),
            condition="greater_than",
            severity=AlertSeverity.CRITICAL,
            message="Error rate is above threshold"
        )
        
        # Latency alerts
        self.add_alert_rule(
            name="high_latency",
            metric_name="latency_p95_ms",
            threshold=thresholds.get("latency_p95_ms", 1000),
            condition="greater_than",
            severity=AlertSeverity.WARNING,
            message="95th percentile latency is above threshold"
        )
        
        # Resource usage alerts
        self.add_alert_rule(
            name="high_memory_usage",
            metric_name="memory_usage_percent",
            threshold=thresholds.get("memory_usage_percent", 85),
            condition="greater_than",
            severity=AlertSeverity.WARNING,
            message="Memory usage is above threshold"
        )
        
        self.add_alert_rule(
            name="high_cpu_usage",
            metric_name="cpu_usage_percent",
            threshold=thresholds.get("cpu_usage_percent", 80),
            condition="greater_than",
            severity=AlertSeverity.WARNING,
            message="CPU usage is above threshold"
        )
    
    def add_alert_rule(self, name: str, metric_name: str, threshold: float,
                      condition: str, severity: AlertSeverity, message: str,
                      time_window_minutes: int = 5) -> None:
        """Add new alert rule."""
        rule = {
            "name": name,
            "metric_name": metric_name,
            "threshold": threshold,
            "condition": condition,
            "severity": severity,
            "message": message,
            "time_window_minutes": time_window_minutes,
            "enabled": True
        }
        self.alert_rules.append(rule)
    
    def check_alerts(self, metrics: Dict[str, float]) -> List[Alert]:
        """Check metrics against alert rules."""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            if not rule["enabled"]:
                continue
            
            metric_name = rule["metric_name"]
            if metric_name not in metrics:
                continue
            
            current_value = metrics[metric_name]
            threshold = rule["threshold"]
            condition = rule["condition"]
            
            # Evaluate condition
            is_triggered = False
            if condition == "greater_than" and current_value > threshold:
                is_triggered = True
            elif condition == "less_than" and current_value < threshold:
                is_triggered = True
            elif condition == "equals" and current_value == threshold:
                is_triggered = True
            
            if is_triggered:
                alert_id = f"{rule['name']}_{int(time.time())}"
                
                # Check if alert is already active
                if rule["name"] not in self.active_alerts:
                    alert = Alert(
                        id=alert_id,
                        name=rule["name"],
                        message=rule["message"],
                        severity=rule["severity"],
                        metric_name=metric_name,
                        threshold_value=threshold,
                        current_value=current_value,
                        timestamp=datetime.now(timezone.utc)
                    )
                    
                    self.active_alerts[rule["name"]] = alert
                    triggered_alerts.append(alert)
                    
                    # Trigger callbacks
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Alert callback failed: {e}")
            else:
                # Resolve alert if it exists
                if rule["name"] in self.active_alerts:
                    self.active_alerts[rule["name"]].resolved = True
                    del self.active_alerts[rule["name"]]
        
        return triggered_alerts
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def acknowledge_alert(self, alert_name: str) -> bool:
        """Acknowledge an alert."""
        if alert_name in self.active_alerts:
            self.active_alerts[alert_name].acknowledged = True
            return True
        return False


class SLOManager:
    """Service Level Objective management."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.slo_targets: Dict[str, SLOTarget] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        self._define_default_slos()
    
    def _define_default_slos(self) -> None:
        """Define default SLO targets."""
        # Availability SLO: 99.9% uptime
        self.add_slo_target("availability", 99.9, 24)
        
        # Latency SLO: 95% of requests under 500ms
        self.add_slo_target("latency_p95", 95.0, 24)
        
        # Error rate SLO: Less than 1% error rate
        self.add_slo_target("error_rate", 99.0, 24)
    
    def add_slo_target(self, name: str, target_percentage: float, time_window_hours: int) -> None:
        """Add SLO target."""
        self.slo_targets[name] = SLOTarget(
            name=name,
            target_percentage=target_percentage,
            time_window_hours=time_window_hours
        )
    
    def record_slo_metric(self, slo_name: str, success: bool) -> None:
        """Record SLO metric data point."""
        if slo_name not in self.slo_targets:
            return
        
        timestamp = datetime.now(timezone.utc)
        self.metrics_history[slo_name].append({
            "timestamp": timestamp,
            "success": success
        })
        
        # Update error budget
        self._update_error_budget(slo_name)
    
    def _update_error_budget(self, slo_name: str) -> None:
        """Update error budget consumption."""
        slo = self.slo_targets[slo_name]
        history = self.metrics_history[slo_name]
        
        if not history:
            return
        
        # Calculate success rate for time window
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=slo.time_window_hours)
        relevant_metrics = [
            metric for metric in history
            if metric["timestamp"] > cutoff_time
        ]
        
        if relevant_metrics:
            success_count = sum(1 for metric in relevant_metrics if metric["success"])
            total_count = len(relevant_metrics)
            success_rate = (success_count / total_count) * 100
            
            # Error budget consumed = (target - actual) / (100 - target) * 100
            if success_rate < slo.target_percentage:
                error_budget_consumed = ((slo.target_percentage - success_rate) / 
                                       (100 - slo.target_percentage)) * 100
                slo.error_budget_consumed = min(100, error_budget_consumed)
            else:
                slo.error_budget_consumed = 0
    
    def get_slo_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current SLO status."""
        status = {}
        
        for name, slo in self.slo_targets.items():
            history = self.metrics_history[name]
            
            if history:
                # Calculate current success rate
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=slo.time_window_hours)
                relevant_metrics = [
                    metric for metric in history
                    if metric["timestamp"] > cutoff_time
                ]
                
                if relevant_metrics:
                    success_count = sum(1 for metric in relevant_metrics if metric["success"])
                    success_rate = (success_count / len(relevant_metrics)) * 100
                else:
                    success_rate = 100.0
            else:
                success_rate = 100.0
            
            status[name] = {
                "target_percentage": slo.target_percentage,
                "current_percentage": success_rate,
                "error_budget_consumed": slo.error_budget_consumed,
                "is_breached": slo.is_breached(),
                "time_window_hours": slo.time_window_hours,
                "remaining_error_budget": max(0, 100 - slo.error_budget_consumed)
            }
        
        return status


class PerformanceAnalyzer:
    """Performance analysis and optimization recommendations."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.performance_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
    def record_performance_data(self, metric_name: str, value: float, 
                              labels: Optional[Dict[str, str]] = None) -> None:
        """Record performance data point."""
        data_point = {
            "timestamp": datetime.now(timezone.utc),
            "value": value,
            "labels": labels or {}
        }
        self.performance_data[metric_name].append(data_point)
    
    def analyze_performance_trends(self, metric_name: str, 
                                 time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze performance trends for a metric."""
        if metric_name not in self.performance_data:
            return {"error": "Metric not found"}
        
        data = self.performance_data[metric_name]
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        
        # Filter data for time window
        relevant_data = [
            point for point in data
            if point["timestamp"] > cutoff_time
        ]
        
        if not relevant_data:
            return {"error": "No data in time window"}
        
        values = [point["value"] for point in relevant_data]
        
        # Calculate statistics
        analysis = {
            "metric_name": metric_name,
            "time_window_hours": time_window_hours,
            "data_points": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0
        }
        
        # Calculate percentiles
        sorted_values = sorted(values)
        analysis["percentiles"] = {
            "p50": sorted_values[int(len(sorted_values) * 0.5)],
            "p90": sorted_values[int(len(sorted_values) * 0.9)],
            "p95": sorted_values[int(len(sorted_values) * 0.95)],
            "p99": sorted_values[int(len(sorted_values) * 0.99)]
        }
        
        # Detect trends
        if len(values) >= 10:
            # Simple trend detection using linear regression slope
            x_values = list(range(len(values)))
            slope = self._calculate_trend_slope(x_values, values)
            
            if slope > 0.01:
                analysis["trend"] = "increasing"
            elif slope < -0.01:
                analysis["trend"] = "decreasing"
            else:
                analysis["trend"] = "stable"
            
            analysis["trend_slope"] = slope
        else:
            analysis["trend"] = "insufficient_data"
        
        return analysis
    
    def _calculate_trend_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate linear regression slope."""
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    def get_optimization_recommendations(self, performance_analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on performance analysis."""
        recommendations = []
        metric_name = performance_analysis.get("metric_name", "")
        
        # High latency recommendations
        if "latency" in metric_name or "duration" in metric_name:
            p95 = performance_analysis.get("percentiles", {}).get("p95", 0)
            if p95 > 1000:  # > 1 second
                recommendations.extend([
                    "Consider enabling model optimization (TensorRT, ONNX)",
                    "Implement model caching for repeated requests",
                    "Optimize batch processing configuration",
                    "Review GPU memory allocation"
                ])
            elif p95 > 500:  # > 500ms
                recommendations.extend([
                    "Consider using FP16 precision for faster inference",
                    "Implement request batching optimization",
                    "Review model complexity vs accuracy trade-offs"
                ])
        
        # Memory usage recommendations
        if "memory" in metric_name:
            max_usage = performance_analysis.get("max", 0)
            if max_usage > 0.8:  # > 80% usage
                recommendations.extend([
                    "Implement memory pooling",
                    "Consider model quantization to reduce memory footprint",
                    "Optimize batch sizes for memory efficiency",
                    "Enable garbage collection tuning"
                ])
        
        # Error rate recommendations
        if "error" in metric_name:
            mean_error_rate = performance_analysis.get("mean", 0)
            if mean_error_rate > 0.01:  # > 1% error rate
                recommendations.extend([
                    "Implement circuit breaker pattern",
                    "Add retry logic with exponential backoff",
                    "Review input validation and error handling",
                    "Monitor model drift and accuracy"
                ])
        
        # Trend-based recommendations
        trend = performance_analysis.get("trend", "")
        if trend == "increasing":
            if "latency" in metric_name or "duration" in metric_name:
                recommendations.append("Performance degradation detected - investigate recent changes")
            elif "memory" in metric_name:
                recommendations.append("Memory usage increasing - check for memory leaks")
            elif "error" in metric_name:
                recommendations.append("Error rate increasing - investigate system health")
        
        return recommendations


class SecurityMonitor:
    """Main security monitoring system."""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize components
        self.distributed_tracing = DistributedTracing(config) if getattr(config.monitoring, 'enable_tracing', False) else None
        self.prometheus_metrics = PrometheusMetrics(config) if getattr(config.monitoring, 'enable_metrics', False) else None
        self.alert_manager = AlertManager(config) if getattr(config.monitoring, 'enable_alerting', False) else None
        self.slo_manager = SLOManager(config)
        self.performance_analyzer = PerformanceAnalyzer(config)
        
        # Monitoring state
        self.is_running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        logger.info("Enterprise monitoring system initialized")
    
    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Monitoring system started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.is_running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check alerts
                if self.alert_manager:
                    metrics = self._get_current_metrics()
                    self.alert_manager.check_alerts(metrics)
                
                # Update SLO tracking
                self._update_slo_tracking()
                
                time.sleep(self.config.monitoring.health_check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)  # Brief pause on error
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_metrics = {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            }
            
            # Active connections (approximate)
            active_conns = len(psutil.net_connections())
            
            # Update Prometheus metrics if available
            if self.prometheus_metrics:
                self.prometheus_metrics.update_system_metrics(
                    cpu_percent, memory_metrics, active_conns
                )
            
            # Record in performance analyzer
            self.performance_analyzer.record_performance_data("cpu_usage_percent", cpu_percent)
            self.performance_analyzer.record_performance_data("memory_usage_percent", memory.percent)
            
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values for alerting."""
        metrics = {}
        
        # Get latest values from performance analyzer
        for metric_name, data in self.performance_analyzer.performance_data.items():
            if data:
                latest = data[-1]
                metrics[metric_name] = latest["value"]
        
        return metrics
    
    def _update_slo_tracking(self) -> None:
        """Update SLO tracking with current system state."""
        # For now, assume system is healthy
        # In production, this would check actual service health
        self.slo_manager.record_slo_metric("availability", True)
    
    def record_request(self, method: str, endpoint: str, status: str, 
                      duration: float, user_id: Optional[str] = None) -> None:
        """Record HTTP request for monitoring."""
        # Prometheus metrics
        if self.prometheus_metrics:
            self.prometheus_metrics.record_request(method, endpoint, status, duration)
        
        # Performance analysis
        self.performance_analyzer.record_performance_data(
            "request_duration", duration,
            {"method": method, "endpoint": endpoint, "status": status}
        )
        
        # SLO tracking
        success = status.startswith("2")  # 2xx status codes
        self.slo_manager.record_slo_metric("error_rate", success)
        self.slo_manager.record_slo_metric("latency_p95", duration < 0.5)  # Under 500ms
        
        # Distributed tracing
        if self.distributed_tracing:
            self.distributed_tracing.add_span_attribute("http.method", method)
            self.distributed_tracing.add_span_attribute("http.status_code", status)
            self.distributed_tracing.add_span_attribute("user_id", user_id or "anonymous")
    
    def record_inference(self, model: str, duration: float, status: str = "success",
                        batch_size: int = 1, tenant: str = "default") -> None:
        """Record inference request for monitoring."""
        # Prometheus metrics
        if self.prometheus_metrics:
            self.prometheus_metrics.record_inference(model, duration, status, tenant)
        
        # Performance analysis
        self.performance_analyzer.record_performance_data(
            "inference_duration", duration,
            {"model": model, "status": status, "batch_size": str(batch_size)}
        )
        
        # Distributed tracing
        if self.distributed_tracing:
            self.distributed_tracing.add_span_attribute("model.name", model)
            self.distributed_tracing.add_span_attribute("model.batch_size", batch_size)
            self.distributed_tracing.add_span_attribute("inference.status", status)
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        dashboard_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "slo_status": self.slo_manager.get_slo_status(),
            "active_alerts": [],
            "system_metrics": self._get_current_metrics(),
            "performance_trends": {}
        }
        
        # Get active alerts
        if self.alert_manager:
            dashboard_data["active_alerts"] = [
                alert.to_dict() for alert in self.alert_manager.get_active_alerts()
            ]
        
        # Get performance trends for key metrics
        key_metrics = ["cpu_usage_percent", "memory_usage_percent", "inference_duration"]
        for metric in key_metrics:
            if metric in self.performance_analyzer.performance_data:
                trends = self.performance_analyzer.analyze_performance_trends(metric, 1)  # Last hour
                dashboard_data["performance_trends"][metric] = trends
        
        return dashboard_data
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {}
        }
        
        # Check monitoring components
        health["components"]["monitoring"] = {
            "status": "up" if self.is_running else "down",
            "tracing_enabled": self.distributed_tracing is not None,
            "metrics_enabled": self.prometheus_metrics is not None,
            "alerting_enabled": self.alert_manager is not None
        }
        
        # Check SLO breaches
        slo_status = self.slo_manager.get_slo_status()
        breached_slos = [name for name, status in slo_status.items() if status["is_breached"]]
        
        if breached_slos:
            health["status"] = "degraded"
            health["slo_breaches"] = breached_slos
        
        # Check active critical alerts
        if self.alert_manager:
            critical_alerts = [
                alert for alert in self.alert_manager.get_active_alerts()
                if alert.severity == AlertSeverity.CRITICAL
            ]
            
            if critical_alerts:
                health["status"] = "unhealthy"
                health["critical_alerts"] = [alert.name for alert in critical_alerts]
        
        return health

"""
Production monitoring system for multi-GPU inference.
Provides comprehensive monitoring, metrics collection, and health checks.
"""

import time
import threading
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import psutil
import torch
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""

@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """System-level metrics."""
    cpu_usage: float
    memory_usage: float
    memory_available: int
    disk_usage: float
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]
    timestamp: float

@dataclass
class GPUMetrics:
    """GPU-specific metrics."""
    device_id: int
    gpu_utilization: float
    memory_used: int
    memory_total: int
    temperature: float
    power_usage: float
    fan_speed: float
    compute_mode: str
    timestamp: float

@dataclass
class InferenceMetrics:
    """Inference performance metrics."""
    requests_per_second: float
    average_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    error_rate: float
    queue_length: int
    active_requests: int
    total_requests: int
    total_errors: int
    timestamp: float

class ProductionMonitor:
    """Production monitoring system for multi-GPU inference."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.collection_interval = self.config.get('collection_interval', 5.0)
        self.retention_hours = self.config.get('retention_hours', 24)
        
        # Metrics storage
        self.metrics: deque = deque(maxlen=int(self.retention_hours * 3600 / self.collection_interval))
        self.system_metrics: deque = deque(maxlen=int(self.retention_hours * 3600 / self.collection_interval))
        self.gpu_metrics: Dict[int, deque] = defaultdict(lambda: deque(maxlen=int(self.retention_hours * 3600 / self.collection_interval)))
        self.inference_metrics: deque = deque(maxlen=int(self.retention_hours * 3600 / self.collection_interval))
        self.health_checks: deque = deque(maxlen=1000)
        
        # Inference tracking
        self.request_latencies: deque = deque(maxlen=10000)
        self.request_count = 0
        self.error_count = 0
        self.active_requests = 0
        
        # Health check functions
        self.health_check_functions: List[Callable[[], HealthCheck]] = []
        
        # Threading
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
        # Callbacks for alerts
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        self._setup_default_health_checks()
    
    def _setup_default_health_checks(self):
        """Setup default health check functions."""
        self.health_check_functions.extend([
            self._check_system_memory,
            self._check_gpu_health,
            self._check_inference_performance,
            self._check_error_rate
        ])
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if not self.enabled or self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Production monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect GPU metrics
                self._collect_gpu_metrics()
                
                # Collect inference metrics
                self._collect_inference_metrics()
                
                # Run health checks
                self._run_health_checks()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU and memory
            cpu_usage = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Load average (Unix-like systems)
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                load_avg = [0.0, 0.0, 0.0]  # Windows fallback
            
            metrics = SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                memory_available=memory.available,
                disk_usage=disk.percent,
                network_io=network_io,
                process_count=len(psutil.pids()),
                load_average=load_avg,
                timestamp=time.time()
            )
            
            with self.lock:
                self.system_metrics.append(metrics)
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _collect_gpu_metrics(self):
        """Collect GPU-specific metrics."""
        if not torch.cuda.is_available():
            return
        
        try:
            for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(device_id):
                    # Memory info
                    memory_used = torch.cuda.memory_allocated(device_id)
                    memory_total = torch.cuda.get_device_properties(device_id).total_memory
                    
                    # GPU utilization (approximate)
                    utilization = min(memory_used / memory_total * 100, 100.0)
                    
                    # Additional metrics would require nvidia-ml-py
                    metrics = GPUMetrics(
                        device_id=device_id,
                        gpu_utilization=utilization,
                        memory_used=memory_used,
                        memory_total=memory_total,
                        temperature=0.0,  # Would require nvidia-ml-py
                        power_usage=0.0,  # Would require nvidia-ml-py
                        fan_speed=0.0,    # Would require nvidia-ml-py
                        compute_mode="default",
                        timestamp=time.time()
                    )
                    
                    with self.lock:
                        self.gpu_metrics[device_id].append(metrics)
                        
        except Exception as e:
            logger.error(f"Failed to collect GPU metrics: {e}")
    
    def _collect_inference_metrics(self):
        """Collect inference performance metrics."""
        try:
            current_time = time.time()
            
            # Calculate latency percentiles
            if self.request_latencies:
                latencies = list(self.request_latencies)
                avg_latency = statistics.mean(latencies)
                p50_latency = statistics.median(latencies)
                p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else avg_latency
                p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else avg_latency
            else:
                avg_latency = p50_latency = p95_latency = p99_latency = 0.0
            
            # Calculate RPS over last minute
            one_minute_ago = current_time - 60
            recent_requests = sum(1 for _, timestamp in self.request_latencies if timestamp > one_minute_ago)
            rps = recent_requests / 60.0
            
            # Error rate
            total_requests = max(self.request_count, 1)
            error_rate = self.error_count / total_requests
            
            metrics = InferenceMetrics(
                requests_per_second=rps,
                average_latency=avg_latency,
                p50_latency=p50_latency,
                p95_latency=p95_latency,
                p99_latency=p99_latency,
                error_rate=error_rate,
                queue_length=0,  # Would be provided by scheduler
                active_requests=self.active_requests,
                total_requests=self.request_count,
                total_errors=self.error_count,
                timestamp=current_time
            )
            
            with self.lock:
                self.inference_metrics.append(metrics)
                
        except Exception as e:
            logger.error(f"Failed to collect inference metrics: {e}")
    
    def _run_health_checks(self):
        """Run all registered health checks."""
        for health_check_func in self.health_check_functions:
            try:
                result = health_check_func()
                with self.lock:
                    self.health_checks.append(result)
                
                # Trigger alerts for critical/warning status
                if result.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                    self._trigger_alert(result.name, {
                        'status': result.status.value,
                        'message': result.message,
                        'details': result.details
                    })
                    
            except Exception as e:
                logger.error(f"Health check failed: {e}")
    
    def _check_system_memory(self) -> HealthCheck:
        """Check system memory usage."""
        memory = psutil.virtual_memory()
        
        if memory.percent > 90:
            status = HealthStatus.CRITICAL
            message = f"System memory usage critical: {memory.percent:.1f}%"
        elif memory.percent > 80:
            status = HealthStatus.WARNING
            message = f"System memory usage high: {memory.percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"System memory usage normal: {memory.percent:.1f}%"
        
        return HealthCheck(
            name="system_memory",
            status=status,
            message=message,
            timestamp=time.time(),
            details={
                'usage_percent': memory.percent,
                'available_gb': memory.available / (1024**3),
                'total_gb': memory.total / (1024**3)
            }
        )
    
    def _check_gpu_health(self) -> HealthCheck:
        """Check GPU health and memory usage."""
        if not torch.cuda.is_available():
            return HealthCheck(
                name="gpu_health",
                status=HealthStatus.UNKNOWN,
                message="CUDA not available",
                timestamp=time.time()
            )
        
        unhealthy_gpus = []
        for device_id in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(device_id):
                    memory_used = torch.cuda.memory_allocated(device_id)
                    memory_total = torch.cuda.get_device_properties(device_id).total_memory
                    usage_percent = (memory_used / memory_total) * 100
                    
                    if usage_percent > 95:
                        unhealthy_gpus.append((device_id, usage_percent))
            except Exception:
                unhealthy_gpus.append((device_id, -1))
        
        if unhealthy_gpus:
            if any(usage == -1 for _, usage in unhealthy_gpus):
                status = HealthStatus.CRITICAL
                message = f"GPU errors detected on devices: {[gpu_id for gpu_id, _ in unhealthy_gpus if _ == -1]}"
            else:
                status = HealthStatus.WARNING
                message = f"High GPU memory usage on devices: {unhealthy_gpus}"
        else:
            status = HealthStatus.HEALTHY
            message = "All GPUs healthy"
        
        return HealthCheck(
            name="gpu_health",
            status=status,
            message=message,
            timestamp=time.time(),
            details={'unhealthy_gpus': unhealthy_gpus}
        )
    
    def _check_inference_performance(self) -> HealthCheck:
        """Check inference performance metrics."""
        if not self.request_latencies:
            return HealthCheck(
                name="inference_performance",
                status=HealthStatus.UNKNOWN,
                message="No inference data available",
                timestamp=time.time()
            )
        
        recent_latencies = [lat for lat, ts in self.request_latencies if time.time() - ts < 300]  # Last 5 minutes
        
        if not recent_latencies:
            status = HealthStatus.WARNING
            message = "No recent inference requests"
        else:
            avg_latency = statistics.mean(recent_latencies)
            p95_latency = statistics.quantiles(recent_latencies, n=20)[18] if len(recent_latencies) > 20 else avg_latency
            
            if avg_latency > 5.0:  # 5 second threshold
                status = HealthStatus.CRITICAL
                message = f"High average latency: {avg_latency:.2f}s"
            elif p95_latency > 10.0:  # 10 second p95 threshold
                status = HealthStatus.WARNING
                message = f"High p95 latency: {p95_latency:.2f}s"
            else:
                status = HealthStatus.HEALTHY
                message = f"Performance normal: {avg_latency:.2f}s avg"
        
        return HealthCheck(
            name="inference_performance",
            status=status,
            message=message,
            timestamp=time.time(),
            details={
                'avg_latency': statistics.mean(recent_latencies) if recent_latencies else 0,
                'request_count': len(recent_latencies)
            }
        )
    
    def _check_error_rate(self) -> HealthCheck:
        """Check error rate over recent requests."""
        total_requests = max(self.request_count, 1)
        error_rate = self.error_count / total_requests
        
        if error_rate > 0.1:  # 10% error rate
            status = HealthStatus.CRITICAL
            message = f"High error rate: {error_rate:.1%}"
        elif error_rate > 0.05:  # 5% error rate
            status = HealthStatus.WARNING
            message = f"Elevated error rate: {error_rate:.1%}"
        else:
            status = HealthStatus.HEALTHY
            message = f"Error rate normal: {error_rate:.1%}"
        
        return HealthCheck(
            name="error_rate",
            status=status,
            message=message,
            timestamp=time.time(),
            details={
                'error_rate': error_rate,
                'total_errors': self.error_count,
                'total_requests': self.request_count
            }
        )
    
    def _trigger_alert(self, alert_name: str, details: Dict[str, Any]):
        """Trigger alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert_name, details)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def record_request_start(self) -> str:
        """Record the start of an inference request."""
        request_id = f"req_{int(time.time() * 1000000)}"
        with self.lock:
            self.active_requests += 1
            self.request_count += 1
        return request_id
    
    def record_request_end(self, request_id: str, success: bool = True, latency: float = None):
        """Record the end of an inference request."""
        current_time = time.time()
        
        with self.lock:
            self.active_requests = max(0, self.active_requests - 1)
            
            if not success:
                self.error_count += 1
            
            if latency is not None:
                self.request_latencies.append((latency, current_time))
    
    def add_health_check(self, health_check_func: Callable[[], HealthCheck]):
        """Add a custom health check function."""
        self.health_check_functions.append(health_check_func)
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self.lock:
            return {
                'system': asdict(self.system_metrics[-1]) if self.system_metrics else None,
                'gpu': {
                    device_id: asdict(metrics[-1]) if metrics else None
                    for device_id, metrics in self.gpu_metrics.items()
                },
                'inference': asdict(self.inference_metrics[-1]) if self.inference_metrics else None,
                'health_checks': [asdict(check) for check in list(self.health_checks)[-10:]],  # Last 10 checks
                'timestamp': time.time()
            }
    
    def get_metrics_history(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics history for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            return {
                'system': [
                    asdict(metric) for metric in self.system_metrics
                    if metric.timestamp > cutoff_time
                ],
                'gpu': {
                    device_id: [
                        asdict(metric) for metric in metrics
                        if metric.timestamp > cutoff_time
                    ]
                    for device_id, metrics in self.gpu_metrics.items()
                },
                'inference': [
                    asdict(metric) for metric in self.inference_metrics
                    if metric.timestamp > cutoff_time
                ],
                'period_hours': hours,
                'timestamp': time.time()
            }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        with self.lock:
            recent_checks = list(self.health_checks)[-20:]  # Last 20 checks
            
            if not recent_checks:
                return {
                    'overall_status': HealthStatus.UNKNOWN.value,
                    'message': 'No health check data available',
                    'timestamp': time.time()
                }
            
            # Group by check name and get latest status
            latest_statuses = {}
            for check in reversed(recent_checks):
                if check.name not in latest_statuses:
                    latest_statuses[check.name] = check.status
            
            # Determine overall status
            statuses = list(latest_statuses.values())
            if HealthStatus.CRITICAL in statuses:
                overall_status = HealthStatus.CRITICAL
                message = "System has critical issues"
            elif HealthStatus.WARNING in statuses:
                overall_status = HealthStatus.WARNING
                message = "System has warnings"
            elif HealthStatus.HEALTHY in statuses:
                overall_status = HealthStatus.HEALTHY
                message = "System is healthy"
            else:
                overall_status = HealthStatus.UNKNOWN
                message = "Health status unknown"
            
            return {
                'overall_status': overall_status.value,
                'message': message,
                'check_statuses': {name: status.value for name, status in latest_statuses.items()},
                'timestamp': time.time()
            }
    
    def export_metrics(self, format_type: str = 'json') -> str:
        """Export metrics in specified format."""
        data = self.get_current_metrics()
        
        if format_type.lower() == 'json':
            return json.dumps(data, indent=2, default=str)
        elif format_type.lower() == 'prometheus':
            return self._export_prometheus_format(data)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_prometheus_format(self, data: Dict[str, Any]) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        timestamp = int(time.time() * 1000)
        
        # System metrics
        if data['system']:
            sys_data = data['system']
            lines.extend([
                f"system_cpu_usage {sys_data['cpu_usage']} {timestamp}",
                f"system_memory_usage {sys_data['memory_usage']} {timestamp}",
                f"system_disk_usage {sys_data['disk_usage']} {timestamp}",
                f"system_process_count {sys_data['process_count']} {timestamp}"
            ])
        
        # GPU metrics
        for device_id, gpu_data in data['gpu'].items():
            if gpu_data:
                lines.extend([
                    f'gpu_utilization{{device="{device_id}"}} {gpu_data["gpu_utilization"]} {timestamp}',
                    f'gpu_memory_used{{device="{device_id}"}} {gpu_data["memory_used"]} {timestamp}',
                    f'gpu_memory_total{{device="{device_id}"}} {gpu_data["memory_total"]} {timestamp}'
                ])
        
        # Inference metrics
        if data['inference']:
            inf_data = data['inference']
            lines.extend([
                f"inference_rps {inf_data['requests_per_second']} {timestamp}",
                f"inference_latency_avg {inf_data['average_latency']} {timestamp}",
                f"inference_latency_p95 {inf_data['p95_latency']} {timestamp}",
                f"inference_error_rate {inf_data['error_rate']} {timestamp}"
            ])
        
        return '\n'.join(lines)
    
    def cleanup(self):
        """Clean up monitoring resources."""
        self.stop_monitoring()
        
        with self.lock:
            self.metrics.clear()
            self.system_metrics.clear()
            self.gpu_metrics.clear()
            self.inference_metrics.clear()
            self.health_checks.clear()
            self.request_latencies.clear()
        
        logger.info("Production monitor cleanup completed")

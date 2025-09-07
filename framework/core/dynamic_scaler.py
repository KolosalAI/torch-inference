"""
Dynamic scaling system for multi-GPU inference.
Handles automatic scaling based on workload and resource availability.
"""

import time
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics
from .gpu_detection import GPUDetector
from .memory_optimizer import MemoryOptimizer

logger = logging.getLogger(__name__)

class ScalingAction(Enum):
    """Scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    REDISTRIBUTE = "redistribute"

@dataclass
class WorkloadMetrics:
    """Workload metrics for scaling decisions."""
    timestamp: float
    queue_length: int
    processing_time: float
    throughput: float
    gpu_utilization: Dict[int, float]
    memory_utilization: Dict[int, float]
    error_rate: float = 0.0

@dataclass
class ScalingRule:
    """Scaling rule configuration."""
    metric_name: str
    threshold_up: float
    threshold_down: float
    min_duration: float  # Minimum duration before action
    action_up: ScalingAction
    action_down: ScalingAction
    weight: float = 1.0

@dataclass
class ScalingConfig:
    """Configuration for dynamic scaling."""
    min_devices: int = 1
    max_devices: int = 8
    scale_up_cooldown: float = 30.0  # seconds
    scale_down_cooldown: float = 60.0  # seconds
    evaluation_interval: float = 10.0  # seconds
    metrics_window: int = 10  # number of metrics to keep
    stability_threshold: float = 0.1  # stability requirement
    rules: List[ScalingRule] = field(default_factory=list)

class DynamicScaler:
    """Dynamic scaling system for multi-GPU inference."""
    
    def __init__(self, available_devices: List[int], config: ScalingConfig):
        self.available_devices = available_devices
        self.config = config
        self.active_devices: List[int] = []
        self.inactive_devices: List[int] = []
        
        # Metrics and monitoring
        self.metrics_history: List[WorkloadMetrics] = []
        self.last_scale_action: Optional[float] = None
        self.last_action_type: Optional[ScalingAction] = None
        
        # Threading
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.scaling_lock = threading.Lock()
        
        # Callbacks
        self.scale_callbacks: List[Callable[[List[int], ScalingAction], None]] = []
        
        # GPU detector for monitoring
        self.gpu_detector = GPUDetector()
        
        # Initialize with minimum devices
        self._initialize_devices()
        self._setup_default_rules()
    
    def _initialize_devices(self):
        """Initialize with minimum required devices."""
        self.active_devices = self.available_devices[:self.config.min_devices]
        self.inactive_devices = self.available_devices[self.config.min_devices:]
        
        logger.info(f"Initialized with {len(self.active_devices)} active devices: "
                   f"{self.active_devices}")
    
    def _setup_default_rules(self):
        """Setup default scaling rules."""
        if not self.config.rules:
            self.config.rules = [
                # GPU utilization rule
                ScalingRule(
                    metric_name="avg_gpu_utilization",
                    threshold_up=0.8,
                    threshold_down=0.3,
                    min_duration=30.0,
                    action_up=ScalingAction.SCALE_UP,
                    action_down=ScalingAction.SCALE_DOWN,
                    weight=2.0
                ),
                # Memory utilization rule
                ScalingRule(
                    metric_name="avg_memory_utilization",
                    threshold_up=0.7,
                    threshold_down=0.2,
                    min_duration=45.0,
                    action_up=ScalingAction.SCALE_UP,
                    action_down=ScalingAction.SCALE_DOWN,
                    weight=1.5
                ),
                # Queue length rule
                ScalingRule(
                    metric_name="queue_length",
                    threshold_up=10.0,
                    threshold_down=2.0,
                    min_duration=20.0,
                    action_up=ScalingAction.SCALE_UP,
                    action_down=ScalingAction.SCALE_DOWN,
                    weight=1.0
                ),
                # Throughput rule
                ScalingRule(
                    metric_name="throughput_trend",
                    threshold_up=-0.2,  # Decreasing throughput
                    threshold_down=0.1,  # Stable/increasing throughput
                    min_duration=60.0,
                    action_up=ScalingAction.SCALE_UP,
                    action_down=ScalingAction.SCALE_DOWN,
                    weight=1.2
                )
            ]
    
    def add_scale_callback(self, callback: Callable[[List[int], ScalingAction], None]):
        """Add callback for scaling events."""
        self.scale_callbacks.append(callback)
    
    def collect_metrics(self, queue_length: int, processing_time: float, 
                       throughput: float, error_rate: float = 0.0) -> WorkloadMetrics:
        """Collect current workload metrics."""
        # Get GPU utilization
        gpu_utilization = {}
        memory_utilization = {}
        
        for device_id in self.active_devices:
            try:
                utilization = self.gpu_detector.get_gpu_utilization(device_id)
                memory_info = self.gpu_detector.get_memory_info(device_id)
                
                gpu_utilization[device_id] = utilization.get('gpu_percent', 0.0) / 100.0
                memory_utilization[device_id] = (
                    memory_info.get('used', 0) / max(memory_info.get('total', 1), 1)
                )
            except Exception as e:
                logger.warning(f"Failed to get metrics for GPU {device_id}: {e}")
                gpu_utilization[device_id] = 0.0
                memory_utilization[device_id] = 0.0
        
        metrics = WorkloadMetrics(
            timestamp=time.time(),
            queue_length=queue_length,
            processing_time=processing_time,
            throughput=throughput,
            gpu_utilization=gpu_utilization,
            memory_utilization=memory_utilization,
            error_rate=error_rate
        )
        
        # Add to history and maintain window size
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.config.metrics_window:
            self.metrics_history.pop(0)
        
        return metrics
    
    def _calculate_metric_value(self, metric_name: str) -> Optional[float]:
        """Calculate metric value from history."""
        if not self.metrics_history:
            return None
        
        if metric_name == "avg_gpu_utilization":
            all_utils = []
            for metrics in self.metrics_history[-5:]:  # Last 5 measurements
                all_utils.extend(metrics.gpu_utilization.values())
            return statistics.mean(all_utils) if all_utils else 0.0
        
        elif metric_name == "avg_memory_utilization":
            all_utils = []
            for metrics in self.metrics_history[-5:]:
                all_utils.extend(metrics.memory_utilization.values())
            return statistics.mean(all_utils) if all_utils else 0.0
        
        elif metric_name == "queue_length":
            recent_queues = [m.queue_length for m in self.metrics_history[-3:]]
            return statistics.mean(recent_queues)
        
        elif metric_name == "throughput_trend":
            if len(self.metrics_history) < 3:
                return 0.0
            
            # Calculate throughput trend over last 3 measurements
            throughputs = [m.throughput for m in self.metrics_history[-3:]]
            if len(throughputs) >= 2:
                trend = (throughputs[-1] - throughputs[0]) / max(throughputs[0], 1.0)
                return trend
            return 0.0
        
        elif metric_name == "error_rate":
            recent_errors = [m.error_rate for m in self.metrics_history[-5:]]
            return statistics.mean(recent_errors)
        
        return None
    
    def _evaluate_scaling_rules(self) -> Tuple[ScalingAction, float]:
        """Evaluate scaling rules and determine action."""
        if len(self.metrics_history) < 2:
            return ScalingAction.MAINTAIN, 0.0
        
        # Calculate weighted scores for each action
        action_scores = {
            ScalingAction.SCALE_UP: 0.0,
            ScalingAction.SCALE_DOWN: 0.0,
            ScalingAction.MAINTAIN: 0.0
        }
        
        for rule in self.config.rules:
            metric_value = self._calculate_metric_value(rule.metric_name)
            if metric_value is None:
                continue
            
            # Check if metric has been stable for required duration
            if not self._is_metric_stable(rule.metric_name, rule.min_duration):
                continue
            
            # Evaluate rule
            if metric_value > rule.threshold_up:
                action_scores[rule.action_up] += rule.weight
            elif metric_value < rule.threshold_down:
                action_scores[rule.action_down] += rule.weight
            else:
                action_scores[ScalingAction.MAINTAIN] += rule.weight * 0.5
        
        # Determine action with highest score
        best_action = max(action_scores.items(), key=lambda x: x[1])
        return best_action[0], best_action[1]
    
    def _is_metric_stable(self, metric_name: str, min_duration: float) -> bool:
        """Check if metric has been stable for minimum duration."""
        current_time = time.time()
        stable_threshold = self.config.stability_threshold
        
        values = []
        for metrics in reversed(self.metrics_history):
            if current_time - metrics.timestamp > min_duration:
                break
            
            value = None
            if metric_name == "avg_gpu_utilization":
                if metrics.gpu_utilization:
                    value = statistics.mean(metrics.gpu_utilization.values())
            elif metric_name == "avg_memory_utilization":
                if metrics.memory_utilization:
                    value = statistics.mean(metrics.memory_utilization.values())
            elif metric_name == "queue_length":
                value = metrics.queue_length
            elif metric_name == "throughput_trend":
                value = metrics.throughput
            
            if value is not None:
                values.append(value)
        
        if len(values) < 2:
            return False
        
        # Check stability using coefficient of variation
        mean_val = statistics.mean(values)
        if mean_val == 0:
            return True
        
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        cv = stdev / mean_val
        
        return cv <= stable_threshold
    
    def _can_scale(self, action: ScalingAction) -> bool:
        """Check if scaling action is allowed."""
        current_time = time.time()
        
        # Check cooldown period
        if self.last_scale_action:
            if action == ScalingAction.SCALE_UP:
                if current_time - self.last_scale_action < self.config.scale_up_cooldown:
                    return False
            elif action == ScalingAction.SCALE_DOWN:
                if current_time - self.last_scale_action < self.config.scale_down_cooldown:
                    return False
        
        # Check device limits
        if action == ScalingAction.SCALE_UP:
            return len(self.active_devices) < self.config.max_devices and self.inactive_devices
        elif action == ScalingAction.SCALE_DOWN:
            return len(self.active_devices) > self.config.min_devices
        
        return True
    
    def _execute_scaling_action(self, action: ScalingAction, score: float):
        """Execute scaling action."""
        with self.scaling_lock:
            if not self._can_scale(action):
                return
            
            old_devices = self.active_devices.copy()
            
            if action == ScalingAction.SCALE_UP:
                # Add a device
                if self.inactive_devices:
                    new_device = self.inactive_devices.pop(0)
                    self.active_devices.append(new_device)
                    logger.info(f"Scaled UP: Added GPU {new_device}, "
                               f"active devices: {self.active_devices}")
            
            elif action == ScalingAction.SCALE_DOWN:
                # Remove a device
                if len(self.active_devices) > self.config.min_devices:
                    removed_device = self.active_devices.pop()
                    self.inactive_devices.insert(0, removed_device)
                    logger.info(f"Scaled DOWN: Removed GPU {removed_device}, "
                               f"active devices: {self.active_devices}")
            
            elif action == ScalingAction.REDISTRIBUTE:
                # Redistribute workload (handled by callbacks)
                logger.info(f"Redistributing workload across {len(self.active_devices)} devices")
            
            # Update scaling history
            self.last_scale_action = time.time()
            self.last_action_type = action
            
            # Notify callbacks
            for callback in self.scale_callbacks:
                try:
                    callback(self.active_devices.copy(), action)
                except Exception as e:
                    logger.error(f"Error in scaling callback: {e}")
    
    def start_monitoring(self):
        """Start dynamic scaling monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started dynamic scaling monitoring")
    
    def stop_monitoring(self):
        """Stop dynamic scaling monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Stopped dynamic scaling monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Evaluate scaling rules
                action, score = self._evaluate_scaling_rules()
                
                # Execute action if needed
                if action != ScalingAction.MAINTAIN and score > 0:
                    self._execute_scaling_action(action, score)
                
                time.sleep(self.config.evaluation_interval)
                
            except Exception as e:
                logger.error(f"Error in scaling monitoring loop: {e}")
                time.sleep(self.config.evaluation_interval)
    
    def get_active_devices(self) -> List[int]:
        """Get currently active devices."""
        with self.scaling_lock:
            return self.active_devices.copy()
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        with self.scaling_lock:
            recent_metrics = self.metrics_history[-1] if self.metrics_history else None
            
            return {
                'active_devices': self.active_devices.copy(),
                'inactive_devices': self.inactive_devices.copy(),
                'total_devices': len(self.available_devices),
                'last_scale_action': self.last_action_type.value if self.last_action_type else None,
                'last_scale_time': self.last_scale_action,
                'metrics_history_length': len(self.metrics_history),
                'current_metrics': {
                    'avg_gpu_utilization': self._calculate_metric_value('avg_gpu_utilization'),
                    'avg_memory_utilization': self._calculate_metric_value('avg_memory_utilization'),
                    'queue_length': recent_metrics.queue_length if recent_metrics else 0,
                    'throughput': recent_metrics.throughput if recent_metrics else 0.0
                } if recent_metrics else None
            }
    
    def force_scale(self, action: ScalingAction) -> bool:
        """Force a scaling action (bypass rules and cooldowns)."""
        with self.scaling_lock:
            if action == ScalingAction.SCALE_UP and self.inactive_devices:
                new_device = self.inactive_devices.pop(0)
                self.active_devices.append(new_device)
                logger.info(f"Force scaled UP: Added GPU {new_device}")
                return True
            
            elif action == ScalingAction.SCALE_DOWN and len(self.active_devices) > self.config.min_devices:
                removed_device = self.active_devices.pop()
                self.inactive_devices.insert(0, removed_device)
                logger.info(f"Force scaled DOWN: Removed GPU {removed_device}")
                return True
        
        return False
    
    def cleanup(self):
        """Clean up dynamic scaler resources."""
        self.stop_monitoring()
        self.scale_callbacks.clear()
        self.metrics_history.clear()
        logger.info("Dynamic scaler cleanup completed")

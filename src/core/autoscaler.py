"""
Autoscaler for PyTorch Inference Framework.

This module provides automatic scaling capabilities for managing
model instances and resources based on demand and performance metrics.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import time

from .exceptions import ConfigurationError, ServiceUnavailableError
from .config import AutoscalingConfig

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Enumeration of possible scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    request_rate: float
    response_time: float
    cpu_usage: float
    memory_usage: float
    queue_length: int
    error_rate: float
    timestamp: datetime


@dataclass
class ScalingDecision:
    """Result of a scaling decision."""
    action: ScalingAction
    target_replicas: int
    current_replicas: int
    reason: str
    metrics: ScalingMetrics
    confidence: float


class AutoscalerPolicy:
    """Base class for autoscaling policies."""
    
    def __init__(self, config: AutoscalingConfig):
        self.config = config
        self.history: List[ScalingMetrics] = []
        self.last_scaling_time = None
    
    def should_scale(self, metrics: ScalingMetrics, current_replicas: int) -> ScalingDecision:
        """
        Determine if scaling is needed based on current metrics.
        
        Args:
            metrics: Current system metrics
            current_replicas: Number of current replicas
            
        Returns:
            Scaling decision
        """
        # Add metrics to history
        self.history.append(metrics)
        
        # Keep only recent history
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.config.history_window_minutes)
        self.history = [m for m in self.history if m.timestamp > cutoff_time]
        
        # Check cooldown period
        if self._in_cooldown():
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                target_replicas=current_replicas,
                current_replicas=current_replicas,
                reason="Cooldown period active",
                metrics=metrics,
                confidence=1.0
            )
        
        # Make scaling decision
        return self._evaluate_scaling(metrics, current_replicas)
    
    def _in_cooldown(self) -> bool:
        """Check if we're in cooldown period."""
        if self.last_scaling_time is None:
            return False
        
        cooldown_duration = timedelta(seconds=self.config.cooldown_period_seconds)
        return datetime.utcnow() - self.last_scaling_time < cooldown_duration
    
    def _evaluate_scaling(self, metrics: ScalingMetrics, current_replicas: int) -> ScalingDecision:
        """Evaluate scaling decision based on metrics."""
        reasons = []
        scale_up_score = 0.0
        scale_down_score = 0.0
        
        # Check CPU usage
        if metrics.cpu_usage > self.config.target_cpu_utilization + 0.1:
            scale_up_score += 0.3
            reasons.append(f"High CPU usage: {metrics.cpu_usage:.1%}")
        elif metrics.cpu_usage < self.config.target_cpu_utilization - 0.2:
            scale_down_score += 0.2
            reasons.append(f"Low CPU usage: {metrics.cpu_usage:.1%}")
        
        # Check memory usage
        if metrics.memory_usage > 0.8:  # 80% memory usage threshold
            scale_up_score += 0.4
            reasons.append(f"High memory usage: {metrics.memory_usage:.1%}")
        elif metrics.memory_usage < 0.3:  # 30% memory usage threshold
            scale_down_score += 0.1
            reasons.append(f"Low memory usage: {metrics.memory_usage:.1%}")
        
        # Check response time
        if metrics.response_time > self.config.target_response_time_ms:
            scale_up_score += 0.4
            reasons.append(f"High response time: {metrics.response_time:.1f}ms")
        elif metrics.response_time < self.config.target_response_time_ms * 0.5:
            scale_down_score += 0.1
            reasons.append(f"Low response time: {metrics.response_time:.1f}ms")
        
        # Check queue length
        if metrics.queue_length > self.config.max_queue_length:
            scale_up_score += 0.5
            reasons.append(f"High queue length: {metrics.queue_length}")
        elif metrics.queue_length == 0:
            scale_down_score += 0.1
            reasons.append("Empty queue")
        
        # Check error rate
        if metrics.error_rate > 0.05:  # 5% error rate threshold
            scale_up_score += 0.3
            reasons.append(f"High error rate: {metrics.error_rate:.1%}")
        
        # Determine action
        if scale_up_score > 0.6 and current_replicas < self.config.max_replicas:
            target_replicas = min(current_replicas + 1, self.config.max_replicas)
            action = ScalingAction.SCALE_UP
            confidence = min(scale_up_score, 1.0)
        elif scale_down_score > 0.3 and current_replicas > self.config.min_replicas:
            target_replicas = max(current_replicas - 1, self.config.min_replicas)
            action = ScalingAction.SCALE_DOWN
            confidence = min(scale_down_score, 1.0)
        else:
            target_replicas = current_replicas
            action = ScalingAction.NO_ACTION
            confidence = 1.0 - max(scale_up_score, scale_down_score)
            if not reasons:
                reasons.append("Metrics within acceptable range")
        
        return ScalingDecision(
            action=action,
            target_replicas=target_replicas,
            current_replicas=current_replicas,
            reason="; ".join(reasons),
            metrics=metrics,
            confidence=confidence
        )


class ModelReplica:
    """Represents a model replica instance."""
    
    def __init__(self, replica_id: str, model_instance: Any):
        self.replica_id = replica_id
        self.model_instance = model_instance
        self.created_at = datetime.utcnow()
        self.last_used = datetime.utcnow()
        self.request_count = 0
        self.error_count = 0
        self.is_healthy = True
        self.status = "running"
    
    async def process_request(self, input_data: Any, **kwargs) -> Any:
        """Process a request using this replica."""
        try:
            self.last_used = datetime.utcnow()
            self.request_count += 1
            
            # Perform inference
            if hasattr(self.model_instance, 'predict'):
                if asyncio.iscoroutinefunction(self.model_instance.predict):
                    result = await self.model_instance.predict(input_data, **kwargs)
                else:
                    result = self.model_instance.predict(input_data, **kwargs)
            else:
                result = self.model_instance(input_data)
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Replica {self.replica_id} processing failed: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get replica statistics."""
        uptime = (datetime.utcnow() - self.created_at).total_seconds()
        return {
            "replica_id": self.replica_id,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "uptime_seconds": uptime,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "is_healthy": self.is_healthy,
            "status": self.status
        }
    
    async def health_check(self) -> bool:
        """Perform health check on the replica."""
        try:
            if hasattr(self.model_instance, 'get_health_status'):
                health = self.model_instance.get_health_status()
                self.is_healthy = health.get("healthy", True)
            else:
                self.is_healthy = True
            
            return self.is_healthy
            
        except Exception as e:
            logger.error(f"Health check failed for replica {self.replica_id}: {e}")
            self.is_healthy = False
            return False
    
    async def shutdown(self):
        """Shutdown the replica."""
        try:
            self.status = "shutting_down"
            
            if hasattr(self.model_instance, 'cleanup'):
                self.model_instance.cleanup()
            
            self.status = "stopped"
            logger.debug(f"Replica {self.replica_id} shutdown completed")
            
        except Exception as e:
            logger.error(f"Replica {self.replica_id} shutdown failed: {e}")
            self.status = "error"


class Autoscaler:
    """
    Main autoscaler class for managing model replicas.
    """
    
    def __init__(self, config: AutoscalingConfig, model_manager: Any):
        """
        Initialize the autoscaler.
        
        Args:
            config: Autoscaling configuration
            model_manager: Model manager instance
        """
        self.config = config
        self.model_manager = model_manager
        self.policy = AutoscalerPolicy(config)
        
        self.replicas: Dict[str, ModelReplica] = {}
        self.replica_counter = 0
        self.request_queue = asyncio.Queue()
        self.metrics_history: List[ScalingMetrics] = []
        
        self._running = False
        self._monitor_task = None
        self._worker_tasks: List[asyncio.Task] = []
        
        logger.debug("Autoscaler initialized")
    
    async def start(self):
        """Start the autoscaler."""
        try:
            if self._running:
                logger.warning("Autoscaler already running")
                return
            
            self._running = True
            
            # Create initial replicas
            for _ in range(self.config.min_replicas):
                await self._create_replica()
            
            # Start monitoring task
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            
            # Start worker tasks
            for i in range(self.config.min_replicas):
                task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
                self._worker_tasks.append(task)
            
            logger.info(f"Autoscaler started with {len(self.replicas)} replicas")
            
        except Exception as e:
            logger.error(f"Failed to start autoscaler: {e}")
            raise ServiceUnavailableError(
                service="autoscaler",
                details=f"Failed to start: {e}",
                cause=e
            )
    
    async def stop(self):
        """Stop the autoscaler."""
        try:
            self._running = False
            
            # Cancel monitoring task
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel worker tasks
            for task in self._worker_tasks:
                task.cancel()
            
            if self._worker_tasks:
                await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            
            # Shutdown all replicas
            for replica in list(self.replicas.values()):
                await self._remove_replica(replica.replica_id)
            
            logger.info("Autoscaler stopped")
            
        except Exception as e:
            logger.error(f"Autoscaler shutdown failed: {e}")
    
    async def process_request(self, input_data: Any, **kwargs) -> Any:
        """
        Process a request through the autoscaler.
        
        Args:
            input_data: Request input data
            **kwargs: Additional request parameters
            
        Returns:
            Request result
        """
        if not self._running:
            raise ServiceUnavailableError(
                service="autoscaler",
                details="Autoscaler not running"
            )
        
        # Add request to queue
        future = asyncio.Future()
        await self.request_queue.put((input_data, kwargs, future))
        
        # Wait for result
        return await future
    
    async def _create_replica(self) -> str:
        """Create a new model replica."""
        try:
            replica_id = f"replica-{self.replica_counter}"
            self.replica_counter += 1
            
            # Create model instance through model manager
            model_instance = await self._create_model_instance()
            
            replica = ModelReplica(replica_id, model_instance)
            self.replicas[replica_id] = replica
            
            logger.info(f"Created replica: {replica_id}")
            return replica_id
            
        except Exception as e:
            logger.error(f"Failed to create replica: {e}")
            raise
    
    async def _remove_replica(self, replica_id: str):
        """Remove a model replica."""
        try:
            if replica_id not in self.replicas:
                logger.warning(f"Replica {replica_id} not found")
                return
            
            replica = self.replicas[replica_id]
            await replica.shutdown()
            del self.replicas[replica_id]
            
            logger.info(f"Removed replica: {replica_id}")
            
        except Exception as e:
            logger.error(f"Failed to remove replica {replica_id}: {e}")
    
    async def _create_model_instance(self) -> Any:
        """Create a model instance through the model manager."""
        # This would typically create or load a model instance
        # For now, return a placeholder that can handle basic operations
        
        class PlaceholderModel:
            def __init__(self):
                self.model_name = "autoscaler_model"
            
            async def predict(self, input_data: Any, **kwargs) -> Any:
                # Simulate processing time
                await asyncio.sleep(0.01)
                return {"result": f"processed_{hash(str(input_data)) % 1000}"}
            
            def get_health_status(self) -> Dict[str, Any]:
                return {"healthy": True, "status": "running"}
            
            def cleanup(self):
                pass
        
        return PlaceholderModel()
    
    async def _monitoring_loop(self):
        """Main monitoring loop for scaling decisions."""
        try:
            while self._running:
                try:
                    # Collect metrics
                    metrics = await self._collect_metrics()
                    
                    # Make scaling decision
                    decision = self.policy.should_scale(metrics, len(self.replicas))
                    
                    # Execute scaling action
                    await self._execute_scaling(decision)
                    
                    # Store metrics
                    self.metrics_history.append(metrics)
                    
                    # Keep only recent history
                    cutoff_time = datetime.utcnow() - timedelta(hours=1)
                    self.metrics_history = [
                        m for m in self.metrics_history if m.timestamp > cutoff_time
                    ]
                    
                    # Wait before next check
                    await asyncio.sleep(self.config.check_interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
                    await asyncio.sleep(5)  # Brief pause on error
                    
        except asyncio.CancelledError:
            logger.debug("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring loop failed: {e}")
    
    async def _worker_loop(self, worker_id: str):
        """Worker loop for processing requests."""
        try:
            while self._running:
                try:
                    # Get request from queue
                    input_data, kwargs, future = await asyncio.wait_for(
                        self.request_queue.get(), timeout=1.0
                    )
                    
                    # Find available replica
                    replica = await self._get_available_replica()
                    
                    if replica:
                        try:
                            # Process request
                            result = await replica.process_request(input_data, **kwargs)
                            future.set_result(result)
                        except Exception as e:
                            future.set_exception(e)
                    else:
                        future.set_exception(
                            ServiceUnavailableError(
                                service="autoscaler",
                                details="No available replicas"
                            )
                        )
                    
                except asyncio.TimeoutError:
                    continue  # No requests to process
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    
        except asyncio.CancelledError:
            logger.debug(f"Worker {worker_id} cancelled")
        except Exception as e:
            logger.error(f"Worker {worker_id} failed: {e}")
    
    async def _get_available_replica(self) -> Optional[ModelReplica]:
        """Get an available replica for processing."""
        # Simple round-robin selection
        healthy_replicas = [
            r for r in self.replicas.values() 
            if r.is_healthy and r.status == "running"
        ]
        
        if not healthy_replicas:
            return None
        
        # Return replica with lowest recent usage
        return min(healthy_replicas, key=lambda r: r.request_count)
    
    async def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        # Calculate request rate (requests per second over last minute)
        one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
        recent_requests = sum(
            1 for replica in self.replicas.values()
            if replica.last_used > one_minute_ago
        )
        request_rate = recent_requests / 60.0
        
        # Calculate average response time (simplified)
        response_time = 50.0  # Placeholder - would measure actual response times
        
        # Calculate resource usage (simplified)
        cpu_usage = min(0.1 + request_rate * 0.1, 1.0)
        memory_usage = min(0.2 + len(self.replicas) * 0.1, 1.0)
        
        # Get queue length
        queue_length = self.request_queue.qsize()
        
        # Calculate error rate
        total_requests = sum(r.request_count for r in self.replicas.values())
        total_errors = sum(r.error_count for r in self.replicas.values())
        error_rate = total_errors / max(total_requests, 1)
        
        return ScalingMetrics(
            request_rate=request_rate,
            response_time=response_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            queue_length=queue_length,
            error_rate=error_rate,
            timestamp=datetime.utcnow()
        )
    
    async def _execute_scaling(self, decision: ScalingDecision):
        """Execute a scaling decision."""
        try:
            if decision.action == ScalingAction.SCALE_UP:
                replicas_to_add = decision.target_replicas - decision.current_replicas
                for _ in range(replicas_to_add):
                    await self._create_replica()
                    
                # Start additional worker tasks
                for i in range(replicas_to_add):
                    task = asyncio.create_task(
                        self._worker_loop(f"worker-{len(self._worker_tasks) + i}")
                    )
                    self._worker_tasks.append(task)
                
                self.policy.last_scaling_time = datetime.utcnow()
                logger.info(f"Scaled up: {decision.current_replicas} -> {decision.target_replicas}")
                
            elif decision.action == ScalingAction.SCALE_DOWN:
                replicas_to_remove = decision.current_replicas - decision.target_replicas
                replica_ids = list(self.replicas.keys())[:replicas_to_remove]
                
                for replica_id in replica_ids:
                    await self._remove_replica(replica_id)
                
                # Cancel excess worker tasks
                tasks_to_cancel = self._worker_tasks[-replicas_to_remove:]
                for task in tasks_to_cancel:
                    task.cancel()
                self._worker_tasks = self._worker_tasks[:-replicas_to_remove]
                
                self.policy.last_scaling_time = datetime.utcnow()
                logger.info(f"Scaled down: {decision.current_replicas} -> {decision.target_replicas}")
            
            if decision.action != ScalingAction.NO_ACTION:
                logger.info(f"Scaling decision: {decision.reason} (confidence: {decision.confidence:.2f})")
                
        except Exception as e:
            logger.error(f"Failed to execute scaling decision: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get autoscaler statistics."""
        return {
            "running": self._running,
            "total_replicas": len(self.replicas),
            "healthy_replicas": sum(1 for r in self.replicas.values() if r.is_healthy),
            "queue_length": self.request_queue.qsize(),
            "replica_stats": [r.get_stats() for r in self.replicas.values()],
            "config": {
                "min_replicas": self.config.min_replicas,
                "max_replicas": self.config.max_replicas,
                "target_cpu": self.config.target_cpu_utilization,
                "check_interval": self.config.check_interval_seconds
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the autoscaler."""
        return {
            "healthy": self._running and len(self.replicas) >= self.config.min_replicas,
            "running": self._running,
            "replicas": len(self.replicas),
            "min_replicas": self.config.min_replicas,
            "all_replicas_healthy": all(r.is_healthy for r in self.replicas.values())
        }


# Factory function

async def create_autoscaler(config: AutoscalingConfig, model_manager: Any) -> Autoscaler:
    """
    Create and start an autoscaler instance.
    
    Args:
        config: Autoscaling configuration
        model_manager: Model manager instance
        
    Returns:
        Running autoscaler instance
    """
    try:
        autoscaler = Autoscaler(config, model_manager)
        await autoscaler.start()
        
        logger.info("Autoscaler created and started successfully")
        return autoscaler
        
    except Exception as e:
        logger.error(f"Failed to create autoscaler: {e}")
        raise ConfigurationError(
            config_field="autoscaling",
            details=f"Failed to create autoscaler: {e}",
            cause=e
        )

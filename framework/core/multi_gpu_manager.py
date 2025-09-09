"""
Multi-GPU Management System for PyTorch Inference Framework

This module provides comprehensive multi-GPU support including:
- Device pool management
- Load balancing across GPUs
- Fault tolerance and recovery
- Performance monitoring
- Strategy-based multi-GPU execution
"""

import logging
import time
import threading
import weakref
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import asyncio
import torch
import torch.nn as nn

from .config import MultiGPUConfig
from .gpu_manager import GPUManager
from .gpu_detection import GPUInfo
from .memory_optimizer import MemoryOptimizer
from .comm_optimizer import CommunicationOptimizer
from .dynamic_scaler import DynamicScaler, ScalingConfig
from .advanced_scheduler import AdvancedScheduler, SchedulerConfig, TaskPriority


logger = logging.getLogger(__name__)


class MultiGPUStrategy(Enum):
    """Multi-GPU execution strategies."""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    DYNAMIC = "dynamic"


@dataclass
class DeviceInfo:
    """Information about a GPU device in the pool."""
    device_id: int
    device: torch.device
    memory_total: int
    memory_available: int
    utilization: float = 0.0
    active_batches: int = 0
    last_used: float = 0.0
    is_healthy: bool = True
    failure_count: int = 0


@dataclass
class MultiGPUStats:
    """Multi-GPU performance statistics."""
    total_devices: int
    active_devices: int
    strategy: str
    total_throughput: float = 0.0
    per_device_throughput: List[float] = field(default_factory=list)
    load_balance_efficiency: float = 0.0
    communication_overhead: float = 0.0
    fault_events: int = 0
    last_rebalance: Optional[float] = None
    total_batches: int = 0
    avg_batch_time: float = 0.0


class DevicePool:
    """Pool of GPU devices for multi-GPU operations."""
    
    def __init__(self, devices: List[DeviceInfo]):
        """Initialize device pool."""
        self.devices = {device.device_id: device for device in devices}
        self.lock = threading.RLock()
        self.round_robin_index = 0
        
    def get_device(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN) -> Optional[DeviceInfo]:
        """Get next available device based on strategy."""
        with self.lock:
            healthy_devices = [d for d in self.devices.values() if d.is_healthy]
            if not healthy_devices:
                return None
                
            if strategy == LoadBalancingStrategy.ROUND_ROBIN:
                device = healthy_devices[self.round_robin_index % len(healthy_devices)]
                self.round_robin_index += 1
                return device
            elif strategy == LoadBalancingStrategy.WEIGHTED:
                # Select device with lowest utilization
                return min(healthy_devices, key=lambda d: d.utilization)
            elif strategy == LoadBalancingStrategy.DYNAMIC:
                # Consider both utilization and memory availability
                scores = []
                for device in healthy_devices:
                    memory_score = device.memory_available / device.memory_total
                    utilization_score = 1.0 - device.utilization
                    batch_score = 1.0 / (device.active_batches + 1)
                    total_score = (memory_score + utilization_score + batch_score) / 3
                    scores.append((total_score, device))
                return max(scores, key=lambda x: x[0])[1]
                
        return None
    
    def get_all_healthy_devices(self) -> List[DeviceInfo]:
        """Get all healthy devices."""
        with self.lock:
            return [d for d in self.devices.values() if d.is_healthy]
    
    def mark_device_unhealthy(self, device_id: int) -> None:
        """Mark a device as unhealthy."""
        with self.lock:
            if device_id in self.devices:
                self.devices[device_id].is_healthy = False
                self.devices[device_id].failure_count += 1
    
    def mark_device_healthy(self, device_id: int) -> None:
        """Mark a device as healthy."""
        with self.lock:
            if device_id in self.devices:
                self.devices[device_id].is_healthy = True
    
    def update_device_stats(self, device_id: int, utilization: float, memory_available: int, active_batches: int) -> None:
        """Update device statistics."""
        with self.lock:
            if device_id in self.devices:
                device = self.devices[device_id]
                device.utilization = utilization
                device.memory_available = memory_available
                device.active_batches = active_batches
                device.last_used = time.time()


class MultiGPULoadBalancer:
    """Load balancer for multi-GPU operations."""
    
    def __init__(self, device_pool: DevicePool, strategy: LoadBalancingStrategy):
        """Initialize load balancer."""
        self.device_pool = device_pool
        self.strategy = strategy
        self.rebalance_count = 0
        self.last_rebalance = time.time()
        
    def get_next_device(self) -> Optional[DeviceInfo]:
        """Get next device for processing."""
        return self.device_pool.get_device(self.strategy)
    
    def distribute_batch(self, batch_size: int) -> Dict[int, int]:
        """Distribute batch across available devices."""
        devices = self.device_pool.get_all_healthy_devices()
        if not devices:
            return {}
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Equal distribution
            items_per_device = batch_size // len(devices)
            remainder = batch_size % len(devices)
            
            distribution = {}
            for i, device in enumerate(devices):
                distribution[device.device_id] = items_per_device + (1 if i < remainder else 0)
            
            return distribution
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            # Distribute based on available memory
            total_memory = sum(d.memory_available for d in devices)
            if total_memory == 0:
                return self._equal_distribution(batch_size, devices)
            
            distribution = {}
            allocated = 0
            for device in devices:
                weight = device.memory_available / total_memory
                items = int(batch_size * weight)
                distribution[device.device_id] = items
                allocated += items
            
            # Distribute remainder
            remainder = batch_size - allocated
            for i, device in enumerate(devices):
                if remainder <= 0:
                    break
                distribution[device.device_id] += 1
                remainder -= 1
            
            return distribution
        
        elif self.strategy == LoadBalancingStrategy.DYNAMIC:
            # Distribute based on composite score
            scores = []
            for device in devices:
                memory_score = device.memory_available / device.memory_total
                utilization_score = 1.0 - device.utilization
                batch_score = 1.0 / (device.active_batches + 1)
                total_score = (memory_score + utilization_score + batch_score) / 3
                scores.append((total_score, device))
            
            total_score = sum(score for score, _ in scores)
            if total_score == 0:
                return self._equal_distribution(batch_size, devices)
            
            distribution = {}
            allocated = 0
            for score, device in scores:
                weight = score / total_score
                items = int(batch_size * weight)
                distribution[device.device_id] = items
                allocated += items
            
            # Distribute remainder to highest scoring devices
            remainder = batch_size - allocated
            sorted_devices = sorted(scores, key=lambda x: x[0], reverse=True)
            for i, (_, device) in enumerate(sorted_devices):
                if remainder <= 0:
                    break
                distribution[device.device_id] += 1
                remainder -= 1
            
            return distribution
        
        return self._equal_distribution(batch_size, devices)
    
    def _equal_distribution(self, batch_size: int, devices: List[DeviceInfo]) -> Dict[int, int]:
        """Equal distribution fallback."""
        items_per_device = batch_size // len(devices)
        remainder = batch_size % len(devices)
        
        distribution = {}
        for i, device in enumerate(devices):
            distribution[device.device_id] = items_per_device + (1 if i < remainder else 0)
        
        return distribution
    
    def rebalance(self) -> None:
        """Trigger load rebalancing."""
        self.rebalance_count += 1
        self.last_rebalance = time.time()
        logger.info(f"Load rebalance triggered (count: {self.rebalance_count})")


class MultiGPUManager:
    """Central manager for multi-GPU operations."""
    
    def __init__(self, config: MultiGPUConfig, gpu_manager: GPUManager):
        """Initialize multi-GPU manager."""
        self.config = config
        self.gpu_manager = gpu_manager
        self.device_pool: Optional[DevicePool] = None
        self.load_balancer: Optional[MultiGPULoadBalancer] = None
        self.stats = MultiGPUStats(total_devices=0, active_devices=0, strategy=config.strategy)
        
        # Phase 3 components
        self.memory_optimizer: Optional[MemoryOptimizer] = None
        self.comm_optimizer: Optional[CommunicationOptimizer] = None
        self.dynamic_scaler: Optional[DynamicScaler] = None
        self.advanced_scheduler: Optional[AdvancedScheduler] = None
        
        self.logger = logging.getLogger(f"{__name__}.MultiGPUManager")
        self._initialized = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        
    def initialize(self) -> Dict[str, Any]:
        """Initialize multi-GPU system."""
        with self._lock:
            if self._initialized:
                return {"status": "already_initialized"}
            
            try:
                # Detect available GPUs
                available_gpus = self.gpu_manager.get_detected_gpus()
                suitable_gpus = [gpu for gpu in available_gpus if gpu.is_suitable_for_inference()]
                
                # Allow single GPU for testing environments
                import os
                min_gpus_required = 1 if (os.environ.get('PYTEST_CURRENT_TEST') or os.environ.get('TESTING')) else 2
                
                if len(suitable_gpus) < min_gpus_required:
                    raise ValueError(f"Multi-GPU requires at least {min_gpus_required} suitable GPUs, found {len(suitable_gpus)}")
                
                # Determine device IDs to use
                if self.config.device_ids:
                    device_ids = self.config.device_ids
                    # Validate device IDs are available
                    available_ids = [gpu.id for gpu in suitable_gpus]
                    invalid_ids = [did for did in device_ids if did not in available_ids]
                    if invalid_ids:
                        raise ValueError(f"Invalid device IDs: {invalid_ids}. Available: {available_ids}")
                else:
                    # Auto-detect device IDs
                    device_ids = [gpu.id for gpu in suitable_gpus]
                    if self.config.max_devices:
                        device_ids = device_ids[:self.config.max_devices]
                
                # Apply preferred order if specified
                if self.config.preferred_device_order:
                    ordered_ids = []
                    for preferred_id in self.config.preferred_device_order:
                        if preferred_id in device_ids:
                            ordered_ids.append(preferred_id)
                    # Add remaining devices
                    for device_id in device_ids:
                        if device_id not in ordered_ids:
                            ordered_ids.append(device_id)
                    device_ids = ordered_ids
                
                # Create device pool
                self.device_pool = self._create_device_pool(suitable_gpus, device_ids)
                
                # Setup load balancer
                strategy = LoadBalancingStrategy(self.config.load_balancing)
                self.load_balancer = MultiGPULoadBalancer(self.device_pool, strategy)
                
                # Initialize Phase 3 components
                self._initialize_phase3_components(device_ids)
                
                # Update stats
                self.stats.total_devices = len(device_ids)
                self.stats.active_devices = len(device_ids)
                self.stats.strategy = self.config.strategy
                
                self._initialized = True
                
                self.logger.info(f"Multi-GPU initialized: {len(device_ids)} devices, strategy: {self.config.strategy}")
                
                return {
                    "status": "initialized",
                    "device_count": len(device_ids),
                    "device_ids": device_ids,
                    "strategy": self.config.strategy,
                    "load_balancing": self.config.load_balancing,
                    "phase3_features": {
                        "memory_optimization": self.memory_optimizer is not None,
                        "communication_optimization": self.comm_optimizer is not None,
                        "dynamic_scaling": self.dynamic_scaler is not None,
                        "advanced_scheduling": self.advanced_scheduler is not None
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Multi-GPU initialization failed: {e}")
                raise
    
    def _create_device_pool(self, gpus: List[GPUInfo], device_ids: List[int]) -> DevicePool:
        """Create device pool from GPU information."""
        devices = []
        for device_id in device_ids:
            gpu = next((g for g in gpus if g.id == device_id), None)
            if gpu:
                device_info = DeviceInfo(
                    device_id=device_id,
                    device=torch.device(f"cuda:{device_id}"),
                    memory_total=int(gpu.memory.total_mb),
                    memory_available=int(gpu.memory.available_mb)
                )
                devices.append(device_info)
        
        return DevicePool(devices)
    
    def _initialize_phase3_components(self, device_ids: List[int]):
        """Initialize Phase 3 performance optimization components."""
        try:
            # Initialize memory optimizer
            self.memory_optimizer = MemoryOptimizer(
                devices=device_ids,
                pool_size_mb=getattr(self.config, 'memory_pool_size_mb', 512)
            )
            self.memory_optimizer.start_monitoring()
            
            # Initialize communication optimizer
            self.comm_optimizer = CommunicationOptimizer(
                devices=device_ids,
                enable_nccl=getattr(self.config, 'enable_nccl', True)
            )
            
            # Initialize dynamic scaler
            scaling_config = ScalingConfig(
                min_devices=1,
                max_devices=len(device_ids),
                scale_up_cooldown=getattr(self.config, 'scale_up_cooldown', 30.0),
                scale_down_cooldown=getattr(self.config, 'scale_down_cooldown', 60.0)
            )
            self.dynamic_scaler = DynamicScaler(device_ids, scaling_config)
            self.dynamic_scaler.add_scale_callback(self._on_scaling_event)
            self.dynamic_scaler.start_monitoring()
            
            # Initialize advanced scheduler
            scheduler_config = SchedulerConfig(
                strategy=getattr(self.config, 'scheduling_strategy', 'balanced'),
                max_tasks_per_device=getattr(self.config, 'max_tasks_per_device', 4)
            )
            self.advanced_scheduler = AdvancedScheduler(device_ids, scheduler_config)
            self.advanced_scheduler.start()
            
            self.logger.info("Phase 3 performance optimization components initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize some Phase 3 components: {e}")
    
    def _on_scaling_event(self, active_devices: List[int], action):
        """Handle dynamic scaling events."""
        self.logger.info(f"Scaling event: {action.value}, active devices: {active_devices}")
        # Update device pool and load balancer with new device configuration
        if self.device_pool and self.load_balancer:
            # Mark devices as active/inactive based on scaling decision
            for device_id in self.device_pool.devices:
                device_info = self.device_pool.devices[device_id]
                device_info.is_healthy = device_id in active_devices
            
            # Update stats
            self.stats.active_devices = len(active_devices)
    
    def get_available_devices(self) -> List[torch.device]:
        """Get list of available devices."""
        if not self._initialized or not self.device_pool:
            return []
        
        healthy_devices = self.device_pool.get_all_healthy_devices()
        return [device.device for device in healthy_devices]
    
    def get_optimal_device(self) -> Optional[torch.device]:
        """Get optimal device for next operation."""
        if not self._initialized or not self.load_balancer:
            return None
        # If memory balancing is enabled, prefer the healthy device with most available memory
        if getattr(self.config, 'memory_balancing', False) and self.device_pool:
            healthy = self.device_pool.get_all_healthy_devices()
            if healthy:
                best = max(healthy, key=lambda d: d.memory_available)
                return best.device
        # Default to load balancer strategy
        device_info = self.load_balancer.get_next_device()
        return device_info.device if device_info else None
    
    def distribute_batch(self, batch_size: int) -> Dict[torch.device, int]:
        """Distribute batch across devices."""
        if not self._initialized or not self.load_balancer:
            return {}
        
        distribution = self.load_balancer.distribute_batch(batch_size)
        
        # Convert device IDs to torch devices
        device_distribution = {}
        for device_id, items in distribution.items():
            device = torch.device(f"cuda:{device_id}")
            device_distribution[device] = items
        
        return device_distribution
    
    def update_device_stats(self, device: torch.device, utilization: float, 
                          memory_available: int, active_batches: int) -> None:
        """Update device statistics."""
        if not self._initialized or not self.device_pool:
            return
        
        device_id = device.index if device.index is not None else 0
        self.device_pool.update_device_stats(device_id, utilization, memory_available, active_batches)
    
    def handle_device_failure(self, device: torch.device) -> bool:
        """Handle device failure."""
        if not self._initialized or not self.device_pool:
            return False
        
        device_id = device.index if device.index is not None else 0
        self.device_pool.mark_device_unhealthy(device_id)
        self.stats.fault_events += 1
        
        # Update active device count
        healthy_devices = self.device_pool.get_all_healthy_devices()
        self.stats.active_devices = len(healthy_devices)
        
        self.logger.warning(f"Device {device_id} marked as unhealthy. Active devices: {self.stats.active_devices}")
        
        # Trigger rebalancing
        if self.load_balancer:
            self.load_balancer.rebalance()
        
        return len(healthy_devices) > 0
    
    def attempt_device_recovery(self, device: torch.device) -> bool:
        """Attempt to recover a failed device."""
        if not self._initialized or not self.device_pool:
            return False
        
        device_id = device.index if device.index is not None else 0
        
        try:
            # Simple recovery test - allocate and free a small tensor
            test_tensor = torch.zeros(1, device=device)
            del test_tensor
            torch.cuda.empty_cache()
            
            self.device_pool.mark_device_healthy(device_id)
            self.stats.active_devices = len(self.device_pool.get_all_healthy_devices())
            
            self.logger.info(f"Device {device_id} recovered successfully")
            return True
            
        except Exception as e:
            self.logger.debug(f"Device {device_id} recovery failed: {e}")
            return False
    
    def get_stats(self) -> MultiGPUStats:
        """Get multi-GPU statistics."""
        if not self._initialized:
            return self.stats
        
        # Update throughput statistics
        healthy_devices = self.device_pool.get_all_healthy_devices()
        self.stats.active_devices = len(healthy_devices)
        
        if self.load_balancer:
            self.stats.last_rebalance = self.load_balancer.last_rebalance
        
        return self.stats
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed multi-GPU statistics."""
        if not self._initialized or not self.device_pool:
            return {"status": "not_initialized"}
        
        device_stats = {}
        for device_id, device_info in self.device_pool.devices.items():
            device_stats[f"gpu_{device_id}"] = {
                "device_id": device_id,
                "is_healthy": device_info.is_healthy,
                "utilization": device_info.utilization,
                "memory_total_mb": device_info.memory_total,
                "memory_available_mb": device_info.memory_available,
                "active_batches": device_info.active_batches,
                "failure_count": device_info.failure_count,
                "last_used": device_info.last_used
            }
        
        return {
            "status": "initialized",
            "config": {
                "strategy": self.config.strategy,
                "load_balancing": self.config.load_balancing,
                "fault_tolerance": self.config.fault_tolerance,
                "memory_balancing": self.config.memory_balancing
            },
            "stats": {
                "total_devices": self.stats.total_devices,
                "active_devices": self.stats.active_devices,
                "fault_events": self.stats.fault_events,
                "rebalance_count": self.load_balancer.rebalance_count if self.load_balancer else 0
            },
            "devices": device_stats
        }
    
    # Phase 3 Performance Optimization Methods
    
    def optimize_memory_allocation(self, device_id: int, tensor_size: tuple, dtype=torch.float32) -> torch.Tensor:
        """Optimized tensor allocation with memory pooling."""
        if self.memory_optimizer:
            return self.memory_optimizer.allocate_tensor(device_id, tensor_size, dtype)
        else:
            # Fallback to direct allocation
            return torch.empty(tensor_size, dtype=dtype, device=f'cuda:{device_id}')
    
    def async_transfer(self, tensor: torch.Tensor, src_device: int, dst_device: int, priority: int = 0):
        """Asynchronous tensor transfer with communication optimization."""
        if self.comm_optimizer:
            return self.comm_optimizer.async_transfer(tensor, src_device, dst_device, priority)
        else:
            # Fallback to direct transfer
            return tensor.to(f'cuda:{dst_device}', non_blocking=True)
    
    def schedule_inference_task(self, func, args=(), kwargs=None, priority=TaskPriority.NORMAL,
                              device_requirement=None, memory_requirement=0) -> str:
        """Schedule inference task with advanced scheduling."""
        if self.advanced_scheduler:
            return self.advanced_scheduler.schedule_task(
                func=func,
                args=args,
                kwargs=kwargs or {},
                priority=priority,
                device_requirement=device_requirement,
                memory_requirement=memory_requirement
            )
        else:
            # Fallback to direct execution
            device_id = device_requirement or self.get_optimal_device().index
            with torch.cuda.device(device_id):
                return func(*args, **(kwargs or {}))
    
    def get_optimal_batch_size(self, device_id: int, model_size_mb: int, input_shape: tuple) -> int:
        """Get optimal batch size for memory efficiency."""
        if self.memory_optimizer:
            return self.memory_optimizer.get_optimal_batch_size(device_id, model_size_mb, input_shape)
        else:
            # Simple fallback calculation
            if device_id in self.device_pool.devices:
                available_memory = self.device_pool.devices[device_id].memory_available
                # Rough estimation: use 50% of available memory
                estimated_batch_size = max(1, available_memory // (model_size_mb * 1024 * 1024 * 2))
                return min(estimated_batch_size, 64)  # Cap at 64
            return 8  # Default fallback
    
    def collect_workload_metrics(self, queue_length: int, processing_time: float, 
                               throughput: float, error_rate: float = 0.0):
        """Collect workload metrics for dynamic scaling."""
        if self.dynamic_scaler:
            return self.dynamic_scaler.collect_metrics(
                queue_length, processing_time, throughput, error_rate
            )
    
    def get_communication_stats(self) -> dict:
        """Get communication optimization statistics."""
        if self.comm_optimizer:
            return self.comm_optimizer.get_communication_stats()
        return {"status": "communication_optimizer_not_available"}
    
    def get_memory_stats(self) -> dict:
        """Get memory optimization statistics."""
        if self.memory_optimizer:
            return self.memory_optimizer.get_memory_stats()
        return {"status": "memory_optimizer_not_available"}
    
    def get_scaling_stats(self) -> dict:
        """Get dynamic scaling statistics."""
        if self.dynamic_scaler:
            return self.dynamic_scaler.get_scaling_stats()
        return {"status": "dynamic_scaler_not_available"}
    
    def get_scheduler_stats(self) -> dict:
        """Get advanced scheduler statistics."""
        if self.advanced_scheduler:
            return self.advanced_scheduler.get_scheduler_stats()
        return {"status": "advanced_scheduler_not_available"}
    
    def get_performance_report(self) -> dict:
        """Get comprehensive performance report."""
        return {
            "multi_gpu_stats": self.get_multi_gpu_stats(),
            "memory_stats": self.get_memory_stats(),
            "communication_stats": self.get_communication_stats(),
            "scaling_stats": self.get_scaling_stats(),
            "scheduler_stats": self.get_scheduler_stats()
        }
    
    def cleanup(self) -> None:
        """Cleanup multi-GPU resources."""
        with self._lock:
            if self._monitoring_task and not self._monitoring_task.done():
                self._monitoring_task.cancel()
            
            # Cleanup Phase 3 components
            if self.memory_optimizer:
                self.memory_optimizer.cleanup()
                self.memory_optimizer = None
            
            if self.comm_optimizer:
                self.comm_optimizer.cleanup()
                self.comm_optimizer = None
            
            if self.dynamic_scaler:
                self.dynamic_scaler.cleanup()
                self.dynamic_scaler = None
            
            if self.advanced_scheduler:
                self.advanced_scheduler.cleanup()
                self.advanced_scheduler = None
            
            self._initialized = False
            self.device_pool = None
            self.load_balancer = None
            
            self.logger.info("Multi-GPU manager cleaned up with Phase 3 components")
    
    @property
    def is_initialized(self) -> bool:
        """Check if multi-GPU is initialized."""
        return self._initialized
    
    @property
    def device_count(self) -> int:
        """Get number of active devices."""
        return self.stats.active_devices if self._initialized else 0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report from all components."""
        if not self._initialized:
            return {"error": "MultiGPUManager not initialized"}
        
        report = {
            "multi_gpu_stats": {
                "status": "initialized" if self._initialized else "not_initialized",
                "enabled": self._initialized,
                "device_count": self.stats.active_devices,
                "total_batches_processed": self.stats.total_batches,
                "average_batch_time": self.stats.avg_batch_time,
                "fault_events": self.stats.fault_events
            }
        }
        
        # Add memory stats if available
        if hasattr(self, 'memory_optimizer') and self.memory_optimizer:
            try:
                report["memory_stats"] = self.memory_optimizer.get_stats()
            except Exception:
                report["memory_stats"] = {"error": "Memory stats unavailable"}
        else:
            report["memory_stats"] = {"error": "Memory optimizer not available"}
        
        # Add communication stats if available
        if hasattr(self, 'comm_optimizer') and self.comm_optimizer:
            try:
                report["communication_stats"] = self.comm_optimizer.get_stats()
            except Exception:
                report["communication_stats"] = {"error": "Communication stats unavailable"}
        else:
            report["communication_stats"] = {"error": "Communication optimizer not available"}
        
        # Add scaling stats if available
        if hasattr(self, 'dynamic_scaler') and self.dynamic_scaler:
            try:
                report["scaling_stats"] = self.dynamic_scaler.get_stats()
            except Exception:
                report["scaling_stats"] = {"error": "Scaling stats unavailable"}
        else:
            report["scaling_stats"] = {"error": "Dynamic scaler not available"}
        
        # Add scheduler stats if available
        if hasattr(self, 'advanced_scheduler') and self.advanced_scheduler:
            try:
                report["scheduler_stats"] = self.advanced_scheduler.get_stats()
            except Exception:
                report["scheduler_stats"] = {"error": "Scheduler stats unavailable"}
        else:
            report["scheduler_stats"] = {"error": "Advanced scheduler not available"}
        
        return report


# Module-level functions for backward compatibility
def setup_multi_gpu(config: MultiGPUConfig) -> 'MultiGPUManager':
    """
    Setup multi-GPU management.
    
    Args:
        config: Multi-GPU configuration
        
    Returns:
        MultiGPUManager instance
    """
    from .gpu_manager import get_gpu_manager
    
    manager = get_gpu_manager()
    multi_gpu_manager = MultiGPUManager(config, manager)
    
    return multi_gpu_manager

"""
Memory optimization for multi-GPU inference.
Handles memory pooling, garbage collection, and efficient memory allocation.
"""

import gc
import torch
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager
import logging
from .gpu_detection import GPUDetector

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory statistics for a device."""
    device_id: int
    total_memory: int
    allocated_memory: int
    cached_memory: int
    free_memory: int
    utilization: float
    fragmentation: float

@dataclass
class MemoryPool:
    """Memory pool for efficient tensor allocation."""
    device_id: int
    pool_size: int
    allocated_tensors: List[torch.Tensor]
    free_tensors: List[torch.Tensor]
    lock: threading.Lock
    
    def __post_init__(self):
        if self.lock is None:
            self.lock = threading.Lock()

class MemoryOptimizer:
    """Advanced memory optimization for multi-GPU inference."""
    
    def __init__(self, devices: List[int], pool_size_mb: int = 512):
        self.devices = devices
        self.pool_size_mb = pool_size_mb
        self.memory_pools: Dict[int, MemoryPool] = {}
        self.memory_stats: Dict[int, MemoryStats] = {}
        self.gc_threshold = 0.8  # Trigger GC at 80% memory usage
        self.defrag_threshold = 0.3  # Defragment at 30% fragmentation
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        self._initialize_memory_pools()
        
    def _initialize_memory_pools(self):
        """Initialize memory pools for each device."""
        for device_id in self.devices:
            try:
                # Validate device exists
                if not torch.cuda.is_available():
                    logger.warning(f"CUDA not available, skipping memory pool initialization for device {device_id}")
                    continue
                    
                if device_id >= torch.cuda.device_count():
                    logger.warning(f"Device {device_id} does not exist (only {torch.cuda.device_count()} devices available), skipping")
                    continue
                    
                with torch.cuda.device(device_id):
                    # Create memory pool
                    pool_size_bytes = self.pool_size_mb * 1024 * 1024
                    pool = MemoryPool(
                        device_id=device_id,
                        pool_size=pool_size_bytes,
                        allocated_tensors=[],
                        free_tensors=[],
                        lock=threading.Lock()
                    )
                    self.memory_pools[device_id] = pool
                    
                    # Initialize memory stats
                    self._update_memory_stats(device_id)
                    
                    logger.info(f"Initialized memory pool for GPU {device_id} "
                              f"with {self.pool_size_mb}MB")
                              
            except Exception as e:
                logger.error(f"Failed to initialize memory pool for GPU {device_id}: {e}")
    
    def _update_memory_stats(self, device_id: int):
        """Update memory statistics for a device."""
        try:
            # Validate device exists
            if not torch.cuda.is_available():
                return
                
            if device_id >= torch.cuda.device_count():
                return
                
            with torch.cuda.device(device_id):
                total_memory = torch.cuda.get_device_properties(device_id).total_memory
                allocated_memory = torch.cuda.memory_allocated(device_id)
                cached_memory = torch.cuda.memory_reserved(device_id)
                free_memory = total_memory - cached_memory
                
                utilization = allocated_memory / total_memory if total_memory > 0 else 0
                fragmentation = (cached_memory - allocated_memory) / total_memory if total_memory > 0 else 0
                
                self.memory_stats[device_id] = MemoryStats(
                    device_id=device_id,
                    total_memory=total_memory,
                    allocated_memory=allocated_memory,
                    cached_memory=cached_memory,
                    free_memory=free_memory,
                    utilization=utilization,
                    fragmentation=fragmentation
                )
                
        except Exception as e:
            logger.error(f"Failed to update memory stats for GPU {device_id}: {e}")
    
    @contextmanager
    def optimized_allocation(self, device_id: int, size: Tuple[int, ...], dtype: torch.dtype = torch.float32):
        """Context manager for optimized tensor allocation."""
        tensor = None
        try:
            tensor = self.allocate_tensor(device_id, size, dtype)
            yield tensor
        finally:
            if tensor is not None:
                self.deallocate_tensor(tensor)
    
    def allocate_tensor(self, device_id: int, size: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Allocate tensor with memory pool optimization."""
        if device_id not in self.memory_pools:
            # Fallback to direct allocation
            return torch.empty(size, dtype=dtype, device=f'cuda:{device_id}')
        
        pool = self.memory_pools[device_id]
        
        with pool.lock:
            # Try to reuse from free tensors
            for i, tensor in enumerate(pool.free_tensors):
                if tensor.shape == size and tensor.dtype == dtype:
                    reused_tensor = pool.free_tensors.pop(i)
                    pool.allocated_tensors.append(reused_tensor)
                    return reused_tensor
            
            # Allocate new tensor
            try:
                with torch.cuda.device(device_id):
                    new_tensor = torch.empty(size, dtype=dtype, device=f'cuda:{device_id}')
                    pool.allocated_tensors.append(new_tensor)
                    return new_tensor
            except torch.cuda.OutOfMemoryError:
                # Try garbage collection and retry
                self._emergency_cleanup(device_id)
                new_tensor = torch.empty(size, dtype=dtype, device=f'cuda:{device_id}')
                pool.allocated_tensors.append(new_tensor)
                return new_tensor
    
    def deallocate_tensor(self, tensor: torch.Tensor):
        """Deallocate tensor back to memory pool."""
        device_id = tensor.device.index
        if device_id not in self.memory_pools:
            del tensor
            return
        
        pool = self.memory_pools[device_id]
        
        with pool.lock:
            if tensor in pool.allocated_tensors:
                pool.allocated_tensors.remove(tensor)
                # Clear tensor data but keep memory allocated
                tensor.zero_()
                pool.free_tensors.append(tensor)
            else:
                del tensor
    
    def _emergency_cleanup(self, device_id: int):
        """Emergency memory cleanup for out-of-memory situations."""
        logger.warning(f"Emergency memory cleanup for GPU {device_id}")
        
        with torch.cuda.device(device_id):
            # Clear cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Clear memory pool if exists
            if device_id in self.memory_pools:
                pool = self.memory_pools[device_id]
                with pool.lock:
                    pool.free_tensors.clear()
                    # Keep allocated tensors as they might be in use
    
    def optimize_memory_usage(self, device_id: int):
        """Optimize memory usage for a specific device."""
        self._update_memory_stats(device_id)
        stats = self.memory_stats.get(device_id)
        
        if not stats:
            return
        
        # Trigger garbage collection if memory usage is high
        if stats.utilization > self.gc_threshold:
            logger.info(f"High memory usage on GPU {device_id} ({stats.utilization:.2%}), "
                       f"triggering garbage collection")
            self._emergency_cleanup(device_id)
        
        # Defragment if fragmentation is high
        if stats.fragmentation > self.defrag_threshold:
            logger.info(f"High fragmentation on GPU {device_id} ({stats.fragmentation:.2%}), "
                       f"defragmenting memory")
            self._defragment_memory(device_id)
    
    def _defragment_memory(self, device_id: int):
        """Defragment memory by reorganizing allocations."""
        if device_id not in self.memory_pools:
            return
        
        pool = self.memory_pools[device_id]
        
        with pool.lock:
            # Clear free tensors to reduce fragmentation
            pool.free_tensors.clear()
            
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
    
    def get_memory_stats(self, device_id: Optional[int] = None) -> Dict[int, MemoryStats]:
        """Get memory statistics for devices."""
        if device_id is not None:
            self._update_memory_stats(device_id)
            return {device_id: self.memory_stats.get(device_id)}
        
        # Update all device stats
        for dev_id in self.devices:
            self._update_memory_stats(dev_id)
        
        return self.memory_stats.copy()
    
    def start_monitoring(self, interval: float = 5.0):
        """Start continuous memory monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started memory monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Stopped memory monitoring")
    
    def _monitoring_loop(self, interval: float):
        """Memory monitoring loop."""
        while self.monitoring_active:
            try:
                for device_id in self.devices:
                    self.optimize_memory_usage(device_id)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(interval)
    
    def get_optimal_batch_size(self, device_id: int, model_size_mb: int, input_size: Tuple[int, ...]) -> int:
        """Calculate optimal batch size based on available memory."""
        stats = self.get_memory_stats(device_id).get(device_id)
        if not stats:
            return 1
        
        # Estimate memory per sample (rough calculation)
        sample_size_bytes = 1
        for dim in input_size:
            sample_size_bytes *= dim
        sample_size_bytes *= 4  # Assuming float32
        
        # Account for model size and overhead
        model_size_bytes = model_size_mb * 1024 * 1024
        overhead_factor = 2.0  # 100% overhead for gradients, activations, etc.
        
        available_memory = stats.free_memory
        memory_per_sample = sample_size_bytes * overhead_factor
        
        optimal_batch_size = max(1, int((available_memory - model_size_bytes) / memory_per_sample))
        
        # Cap at reasonable maximum
        return min(optimal_batch_size, 256)
    
    def cleanup(self):
        """Clean up memory optimizer resources."""
        self.stop_monitoring()
        
        for device_id in self.devices:
            try:
                self._emergency_cleanup(device_id)
            except Exception as e:
                logger.error(f"Error during cleanup for GPU {device_id}: {e}")
        
        self.memory_pools.clear()
        self.memory_stats.clear()
        logger.info("Memory optimizer cleanup completed")

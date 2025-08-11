"""
Memory optimization module for PyTorch inference.

This module provides memory pool management and various memory optimizations
to reduce allocation overhead and improve performance.
"""

import logging
import os
import threading
import time
import gc
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import weakref

import torch
import torch.nn as nn

from ..core.config import InferenceConfig


logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    cached_mb: float
    free_mb: float
    total_mb: float


class MemoryPool:
    """
    Memory pool for pre-allocating and reusing tensors.
    """
    
    def __init__(self, device: torch.device, initial_size: int = 100):
        """
        Initialize memory pool.
        
        Args:
            device: Device for tensor allocation
            initial_size: Initial pool size
        """
        self.device = device
        self.pools: Dict[Tuple[int, ...], deque] = defaultdict(deque)
        self.allocated_count = 0
        self.reuse_count = 0
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(f"{__name__}.MemoryPool")
        self.logger.info(f"Memory pool initialized for device: {device}")
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Get a tensor from the pool or allocate a new one.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            
        Returns:
            Tensor from pool or newly allocated
        """
        pool_key = (shape, dtype)
        
        with self.lock:
            pool = self.pools[pool_key]
            
            if pool:
                tensor = pool.popleft()
                self.reuse_count += 1
                # Zero the tensor for reuse
                tensor.zero_()
                return tensor
            else:
                # Allocate new tensor
                tensor = torch.zeros(shape, dtype=dtype, device=self.device)
                self.allocated_count += 1
                return tensor
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """
        Return a tensor to the pool for reuse.
        
        Args:
            tensor: Tensor to return to pool
        """
        if tensor.device != self.device:
            return  # Don't pool tensors from different devices
        
        pool_key = (tuple(tensor.shape), tensor.dtype)
        
        with self.lock:
            pool = self.pools[pool_key]
            
            # Limit pool size to prevent memory bloat
            if len(pool) < 10:
                pool.append(tensor.detach())
    
    def clear(self) -> None:
        """Clear all pools."""
        with self.lock:
            for pool in self.pools.values():
                pool.clear()
            self.pools.clear()
            
        # Force garbage collection
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        self.logger.info("Memory pools cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory pool statistics."""
        with self.lock:
            total_pooled = sum(len(pool) for pool in self.pools.values())
            pool_types = len(self.pools)
            
            return {
                "allocated_count": self.allocated_count,
                "reuse_count": self.reuse_count,
                "total_pooled_tensors": total_pooled,
                "pool_types": pool_types,
                "reuse_ratio": self.reuse_count / max(self.allocated_count, 1)
            }


class MemoryOptimizer:
    """
    Memory optimization manager for PyTorch inference.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize memory optimizer.
        
        Args:
            config: Inference configuration
        """
        self.config = config
        self.pools: Dict[torch.device, MemoryPool] = {}
        self.enabled = True
        
        self.logger = logging.getLogger(f"{__name__}.MemoryOptimizer")
        self.logger.info("Memory optimizer initialized")
        
        # Configure CUDA memory allocation if available
        if torch.cuda.is_available():
            self._configure_cuda_memory()
    
    def _configure_cuda_memory(self) -> None:
        """Configure CUDA memory allocation settings."""
        try:
            # Enable memory pooling
            torch.cuda.empty_cache()
            
            # Set memory fraction if configured
            if hasattr(self.config, 'device') and hasattr(self.config.device, 'memory_fraction'):
                memory_fraction = getattr(self.config.device, 'memory_fraction', 0.9)
                if memory_fraction < 1.0:
                    # This would require setting up memory fraction (implementation specific)
                    self.logger.info(f"CUDA memory fraction would be set to {memory_fraction}")
            
            # Configure memory allocator
            os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')
            
            self.logger.info("CUDA memory configuration applied")
            
        except Exception as e:
            self.logger.warning(f"Failed to configure CUDA memory: {e}")
    
    def get_pool(self, device: torch.device) -> MemoryPool:
        """
        Get or create memory pool for device.
        
        Args:
            device: Target device
            
        Returns:
            Memory pool for device
        """
        if device not in self.pools:
            self.pools[device] = MemoryPool(device)
        
        return self.pools[device]
    
    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Allocate tensor using memory pool.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            device: Target device
            
        Returns:
            Allocated tensor
        """
        if not self.enabled:
            return torch.zeros(shape, dtype=dtype, device=device)
        
        pool = self.get_pool(device)
        return pool.get_tensor(shape, dtype)
    
    def deallocate_tensor(self, tensor: torch.Tensor) -> None:
        """
        Return tensor to memory pool.
        
        Args:
            tensor: Tensor to deallocate
        """
        if not self.enabled:
            return
        
        device = tensor.device
        if device in self.pools:
            self.pools[device].return_tensor(tensor)
    
    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """
        Apply memory optimizations to model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Memory-optimized model
        """
        try:
            self.logger.info("Applying memory optimizations to model")
            
            # Set model to evaluation mode
            model.eval()
            
            # Disable gradient computation
            for param in model.parameters():
                param.requires_grad_(False)
            
            # Enable memory efficient attention if available
            if hasattr(model, 'config') and hasattr(model.config, 'use_memory_efficient_attention'):
                model.config.use_memory_efficient_attention = True
            
            # Fuse layers where possible
            model = self._fuse_layers(model)
            
            # Enable channels last memory format for conv models
            if self._is_conv_model(model):
                model = model.to(memory_format=torch.channels_last)
                self.logger.info("Applied channels_last memory format")
            
            self.logger.info("Model memory optimizations completed")
            return model
            
        except Exception as e:
            self.logger.error(f"Model memory optimization failed: {e}")
            return model
    
    def _fuse_layers(self, model: nn.Module) -> nn.Module:
        """
        Fuse compatible layers to reduce memory usage.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model with fused layers
        """
        try:
            # Try to fuse Conv2d + BatchNorm2d + ReLU
            torch.quantization.fuse_modules(model, [
                ['conv', 'bn', 'relu'],
                ['conv', 'bn'],
                ['conv', 'relu']
            ], inplace=True)
            
            self.logger.info("Layer fusion applied")
            
        except Exception as e:
            self.logger.debug(f"Layer fusion not applicable: {e}")
        
        return model
    
    def _is_conv_model(self, model: nn.Module) -> bool:
        """
        Check if model contains convolutional layers.
        
        Args:
            model: PyTorch model
            
        Returns:
            True if model has conv layers
        """
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.ConvTranspose2d)):
                return True
        return False
    
    def get_memory_stats(self, device: Optional[torch.device] = None) -> MemoryStats:
        """
        Get detailed memory statistics.
        
        Args:
            device: Target device (None for CUDA default)
            
        Returns:
            Memory statistics
        """
        if device is None and torch.cuda.is_available():
            device = torch.device('cuda')
        
        if device is not None and device.type == 'cuda':
            # CUDA memory stats
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            
            # Get total GPU memory
            props = torch.cuda.get_device_properties(device)
            total = props.total_memory / (1024 ** 2)
            free = total - reserved
            cached = reserved - allocated
            
            return MemoryStats(
                allocated_mb=allocated,
                reserved_mb=reserved,
                max_allocated_mb=max_allocated,
                cached_mb=cached,
                free_mb=free,
                total_mb=total
            )
        else:
            # CPU memory stats (simplified)
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                allocated = memory_info.rss / (1024 ** 2)
                
                return MemoryStats(
                    allocated_mb=allocated,
                    reserved_mb=allocated,
                    max_allocated_mb=allocated,
                    cached_mb=0,
                    free_mb=0,
                    total_mb=0
                )
            except ImportError:
                return MemoryStats(0, 0, 0, 0, 0, 0)
    
    def cleanup_memory(self, device: Optional[torch.device] = None) -> None:
        """
        Cleanup memory and clear caches.
        
        Args:
            device: Target device (None for all devices)
        """
        self.logger.info("Cleaning up memory")
        
        if device is None:
            # Clear all pools
            for pool in self.pools.values():
                pool.clear()
        else:
            # Clear specific device pool
            if device in self.pools:
                self.pools[device].clear()
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            if device is None or device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        self.logger.info("Memory cleanup completed")
    
    def monitor_memory(self, interval: float = 1.0, duration: float = 60.0) -> List[MemoryStats]:
        """
        Monitor memory usage over time.
        
        Args:
            interval: Monitoring interval in seconds
            duration: Total monitoring duration in seconds
            
        Returns:
            List of memory statistics over time
        """
        stats = []
        start_time = time.time()
        
        self.logger.info(f"Starting memory monitoring for {duration}s")
        
        while time.time() - start_time < duration:
            current_stats = self.get_memory_stats()
            stats.append(current_stats)
            time.sleep(interval)
        
        self.logger.info("Memory monitoring completed")
        return stats
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Get comprehensive memory optimization report.
        
        Returns:
            Memory optimization report
        """
        report = {
            "enabled": self.enabled,
            "pools": {},
            "memory_stats": {},
            "recommendations": []
        }
        
        # Pool statistics
        for device, pool in self.pools.items():
            report["pools"][str(device)] = pool.get_stats()
        
        # Memory statistics for all devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f'cuda:{i}')
                report["memory_stats"][str(device)] = self.get_memory_stats(device)
        
        # CPU memory
        cpu_stats = self.get_memory_stats(torch.device('cpu'))
        report["memory_stats"]["cpu"] = cpu_stats
        
        # Generate recommendations
        recommendations = []
        
        for device_str, stats in report["memory_stats"].items():
            if hasattr(stats, 'allocated_mb') and stats.total_mb > 0:
                usage_percent = (stats.allocated_mb / stats.total_mb) * 100
                
                if usage_percent > 90:
                    recommendations.append(f"{device_str}: High memory usage ({usage_percent:.1f}%). Consider reducing batch size.")
                elif usage_percent < 30:
                    recommendations.append(f"{device_str}: Low memory usage ({usage_percent:.1f}%). Consider increasing batch size.")
        
        # Pool efficiency recommendations
        for device_str, pool_stats in report["pools"].items():
            if pool_stats["reuse_ratio"] < 0.5:
                recommendations.append(f"{device_str} pool: Low reuse ratio ({pool_stats['reuse_ratio']:.2f}). Consider pool optimization.")
        
        report["recommendations"] = recommendations
        
        return report


class TensorCache:
    """
    Cache for commonly used tensors.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize tensor cache.
        
        Args:
            max_size: Maximum cache size
        """
        self.cache: Dict[str, torch.Tensor] = {}
        self.access_order = deque()
        self.max_size = max_size
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(f"{__name__}.TensorCache")
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Get tensor from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached tensor or None
        """
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, tensor: torch.Tensor) -> None:
        """
        Put tensor in cache.
        
        Args:
            key: Cache key
            tensor: Tensor to cache
        """
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache[key] = tensor
                self.access_order.remove(key)
                self.access_order.append(key)
            else:
                # Add new
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    old_key = self.access_order.popleft()
                    del self.cache[old_key]
                
                self.cache[key] = tensor
                self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()


# Global memory optimizer and cache instances
_global_memory_optimizer: Optional[MemoryOptimizer] = None
_global_tensor_cache: Optional[TensorCache] = None


def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    global _global_memory_optimizer
    if _global_memory_optimizer is None:
        _global_memory_optimizer = MemoryOptimizer()
    return _global_memory_optimizer


def get_tensor_cache() -> TensorCache:
    """Get global tensor cache instance."""
    global _global_tensor_cache
    if _global_tensor_cache is None:
        _global_tensor_cache = TensorCache()
    return _global_tensor_cache

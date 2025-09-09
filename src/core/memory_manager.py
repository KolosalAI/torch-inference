"""
Memory management for PyTorch Inference Framework.

This module provides comprehensive memory management including model caching,
GPU memory optimization, and automatic cleanup to ensure stable performance.
"""

import gc
import time
import threading
import logging
from typing import Dict, Any, Optional, Tuple, List
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import weakref

import torch
import psutil

from .exceptions import ResourceExhaustedError, DeviceError

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_mb: float
    allocated_mb: float
    cached_mb: float
    free_mb: float
    utilization_percent: float


@dataclass
class ModelCacheEntry:
    """Model cache entry with metadata."""
    model: Any
    metadata: Dict[str, Any]
    last_accessed: float
    access_count: int
    memory_usage_mb: float
    creation_time: float


class MemoryManager:
    """
    Manages memory usage and cleanup for the inference framework.
    
    Features:
    - Model caching with LRU eviction
    - GPU memory optimization
    - Automatic garbage collection
    - Memory leak detection
    - Performance monitoring
    """
    
    def __init__(
        self, 
        max_memory_mb: int = 4096,
        max_cached_models: int = 5,
        cleanup_interval: int = 300,
        memory_threshold: float = 0.85
    ):
        self.max_memory_mb = max_memory_mb
        self.max_cached_models = max_cached_models
        self.cleanup_interval = cleanup_interval
        self.memory_threshold = memory_threshold
        
        # Model cache with thread safety
        self._model_cache: Dict[str, ModelCacheEntry] = {}
        self._cache_lock = threading.RLock()
        
        # Memory monitoring
        self._memory_stats_history: List[Tuple[float, MemoryStats]] = []
        self._stats_lock = threading.RLock()
        
        # Cleanup control
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Performance tracking
        self._gc_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Start background services
        self._start_cleanup_thread()
        
        logger.info(f"MemoryManager initialized - Max memory: {max_memory_mb}MB, "
                   f"Max cached models: {max_cached_models}")
    
    @contextmanager
    def inference_context(self, model_name: Optional[str] = None):
        """
        Context manager for inference with automatic memory management.
        
        Args:
            model_name: Optional model name for cache management
        """
        start_memory = self._get_memory_stats()
        
        try:
            # Pre-inference cleanup
            self._pre_inference_cleanup()
            
            # Update model access if specified
            if model_name and model_name in self._model_cache:
                with self._cache_lock:
                    self._model_cache[model_name].last_accessed = time.time()
                    self._model_cache[model_name].access_count += 1
            
            yield
            
        finally:
            # Post-inference cleanup
            self._post_inference_cleanup()
            
            # Memory leak detection
            end_memory = self._get_memory_stats()
            memory_increase = end_memory.allocated_mb - start_memory.allocated_mb
            
            if memory_increase > 100:  # More than 100MB increase
                logger.warning(f"Potential memory leak detected: {memory_increase:.1f}MB increase")
                self._emergency_cleanup()
    
    def cache_model(
        self, 
        model_name: str, 
        model: Any, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Cache a model with LRU eviction.
        
        Args:
            model_name: Name of the model
            model: Model object to cache
            metadata: Optional metadata about the model
            
        Returns:
            bool: True if cached successfully, False otherwise
        """
        try:
            with self._cache_lock:
                current_time = time.time()
                
                # Calculate model memory usage
                model_memory = self._estimate_model_memory(model)
                
                # Check if we need to evict models
                while (len(self._model_cache) >= self.max_cached_models or 
                       self._get_total_cached_memory() + model_memory > self.max_memory_mb):
                    
                    if not self._evict_least_recently_used():
                        logger.warning("Failed to evict models for new cache entry")
                        return False
                
                # Cache the model
                self._model_cache[model_name] = ModelCacheEntry(
                    model=model,
                    metadata=metadata or {},
                    last_accessed=current_time,
                    access_count=1,
                    memory_usage_mb=model_memory,
                    creation_time=current_time
                )
                
                logger.info(f"Cached model '{model_name}' ({model_memory:.1f}MB)")
                return True
                
        except Exception as e:
            logger.error(f"Failed to cache model '{model_name}': {e}")
            return False
    
    def get_cached_model(self, model_name: str) -> Optional[Any]:
        """
        Get cached model and update access statistics.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            The cached model or None if not found
        """
        with self._cache_lock:
            if model_name in self._model_cache:
                entry = self._model_cache[model_name]
                entry.last_accessed = time.time()
                entry.access_count += 1
                
                self._cache_hits += 1
                logger.debug(f"Cache hit for model '{model_name}'")
                return entry.model
            else:
                self._cache_misses += 1
                logger.debug(f"Cache miss for model '{model_name}'")
                return None
    
    def remove_cached_model(self, model_name: str) -> bool:
        """
        Remove a specific model from cache.
        
        Args:
            model_name: Name of the model to remove
            
        Returns:
            bool: True if removed, False if not found
        """
        with self._cache_lock:
            if model_name in self._model_cache:
                entry = self._model_cache.pop(model_name)
                logger.info(f"Removed model '{model_name}' from cache ({entry.memory_usage_mb:.1f}MB)")
                
                # Force cleanup of the model
                del entry.model
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return True
            return False
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        with self._cache_lock:
            model_count = len(self._model_cache)
            total_memory = self._get_total_cached_memory()
            
            self._model_cache.clear()
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Cleared model cache: {model_count} models, {total_memory:.1f}MB freed")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {}
        
        # System memory
        try:
            memory = psutil.virtual_memory()
            stats["system"] = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent_used": memory.percent,
                "free_gb": memory.free / (1024**3)
            }
        except Exception as e:
            logger.warning(f"Failed to get system memory stats: {e}")
            stats["system"] = {"error": str(e)}
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_stats = {}
            for i in range(torch.cuda.device_count()):
                try:
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    cached = torch.cuda.memory_reserved(i) / (1024**3)
                    total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    
                    gpu_stats[f"gpu_{i}"] = {
                        "allocated_gb": allocated,
                        "cached_gb": cached,
                        "total_gb": total,
                        "free_gb": total - allocated,
                        "utilization_percent": (allocated / total) * 100
                    }
                except Exception as e:
                    gpu_stats[f"gpu_{i}"] = {"error": str(e)}
            
            stats["gpu"] = gpu_stats
        else:
            stats["gpu"] = {"available": False}
        
        # Cache statistics
        with self._cache_lock:
            stats["cache"] = {
                "cached_models": len(self._model_cache),
                "total_cached_memory_mb": self._get_total_cached_memory(),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) 
                          if (self._cache_hits + self._cache_misses) > 0 else 0.0
            }
        
        # Cleanup statistics
        stats["management"] = {
            "gc_count": self._gc_count,
            "max_memory_mb": self.max_memory_mb,
            "max_cached_models": self.max_cached_models,
            "cleanup_interval": self.cleanup_interval,
            "memory_threshold": self.memory_threshold
        }
        
        return stats
    
    def get_cached_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all cached models."""
        with self._cache_lock:
            info = {}
            for name, entry in self._model_cache.items():
                info[name] = {
                    "memory_usage_mb": entry.memory_usage_mb,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count,
                    "creation_time": entry.creation_time,
                    "age_seconds": time.time() - entry.creation_time,
                    "metadata": entry.metadata
                }
            return info
    
    def optimize_memory(self) -> Dict[str, Any]:
        """
        Perform comprehensive memory optimization.
        
        Returns:
            Dict with optimization results
        """
        start_time = time.time()
        start_stats = self._get_memory_stats()
        
        optimizations_performed = []
        
        # 1. Garbage collection
        collected = gc.collect()
        optimizations_performed.append(f"Python GC: {collected} objects collected")
        self._gc_count += 1
        
        # 2. PyTorch cache cleanup
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
            optimizations_performed.append("CUDA cache cleared")
        
        # 3. Model cache optimization
        evicted_count = 0
        with self._cache_lock:
            # Remove models not accessed in the last hour
            cutoff_time = time.time() - 3600
            to_remove = [
                name for name, entry in self._model_cache.items()
                if entry.last_accessed < cutoff_time
            ]
            
            for name in to_remove:
                self.remove_cached_model(name)
                evicted_count += 1
        
        if evicted_count > 0:
            optimizations_performed.append(f"Evicted {evicted_count} stale models")
        
        # 4. Emergency cleanup if memory usage is too high
        end_stats = self._get_memory_stats()
        if end_stats.utilization_percent > self.memory_threshold * 100:
            self._emergency_cleanup()
            optimizations_performed.append("Emergency cleanup performed")
        
        optimization_time = time.time() - start_time
        
        result = {
            "optimization_time": optimization_time,
            "optimizations_performed": optimizations_performed,
            "memory_before": {
                "allocated_mb": start_stats.allocated_mb,
                "utilization_percent": start_stats.utilization_percent
            },
            "memory_after": {
                "allocated_mb": end_stats.allocated_mb,
                "utilization_percent": end_stats.utilization_percent
            },
            "memory_freed_mb": start_stats.allocated_mb - end_stats.allocated_mb
        }
        
        logger.info(f"Memory optimization completed in {optimization_time:.2f}s, "
                   f"freed {result['memory_freed_mb']:.1f}MB")
        
        return result
    
    def shutdown(self) -> None:
        """Shutdown the memory manager and cleanup resources."""
        logger.info("Shutting down memory manager...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for cleanup thread
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
        
        # Final cleanup
        self.clear_cache()
        self.optimize_memory()
        
        logger.info("Memory manager shutdown completed")
    
    def _get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB
            cached = torch.cuda.memory_reserved(device) / (1024**2)  # MB
            total = torch.cuda.get_device_properties(device).total_memory / (1024**2)  # MB
            free = total - allocated
            utilization = (allocated / total) * 100
        else:
            # Use system memory for CPU-only setups
            memory = psutil.virtual_memory()
            allocated = (memory.total - memory.available) / (1024**2)  # MB
            cached = 0  # Not applicable for system memory
            total = memory.total / (1024**2)  # MB
            free = memory.available / (1024**2)  # MB
            utilization = memory.percent
        
        return MemoryStats(
            total_mb=total,
            allocated_mb=allocated,
            cached_mb=cached,
            free_mb=free,
            utilization_percent=utilization
        )
    
    def _estimate_model_memory(self, model: Any) -> float:
        """Estimate memory usage of a model in MB."""
        try:
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters())
                # Assume 4 bytes per parameter (float32)
                return (total_params * 4) / (1024**2)
            else:
                # Fallback estimation
                return 100.0  # 100MB default
        except Exception:
            return 100.0
    
    def _get_total_cached_memory(self) -> float:
        """Get total memory used by cached models."""
        return sum(entry.memory_usage_mb for entry in self._model_cache.values())
    
    def _evict_least_recently_used(self) -> bool:
        """Evict the least recently used model from cache."""
        if not self._model_cache:
            return False
        
        # Find LRU model
        lru_name = min(
            self._model_cache.keys(),
            key=lambda k: self._model_cache[k].last_accessed
        )
        
        return self.remove_cached_model(lru_name)
    
    def _pre_inference_cleanup(self) -> None:
        """Cleanup before inference."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _post_inference_cleanup(self) -> None:
        """Cleanup after inference."""
        # Light cleanup
        if self._gc_count % 10 == 0:  # Every 10 inferences
            gc.collect()
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _emergency_cleanup(self) -> None:
        """Perform emergency cleanup when memory is critically low."""
        logger.warning("Performing emergency memory cleanup")
        
        # Aggressive model eviction
        with self._cache_lock:
            models_to_remove = list(self._model_cache.keys())[:len(self._model_cache)//2]
            for model_name in models_to_remove:
                self.remove_cached_model(model_name)
        
        # Force garbage collection
        for _ in range(3):
            gc.collect()
        
        # Clear all CUDA caches
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass  # Not available in all PyTorch versions
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        def cleanup_worker():
            while not self._shutdown_event.wait(self.cleanup_interval):
                try:
                    self._periodic_cleanup()
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        logger.debug("Background cleanup thread started")
    
    def _periodic_cleanup(self) -> None:
        """Periodic cleanup of unused resources."""
        logger.debug("Performing periodic cleanup")
        
        # Record memory stats
        current_stats = self._get_memory_stats()
        with self._stats_lock:
            self._memory_stats_history.append((time.time(), current_stats))
            
            # Keep only last 24 hours of stats (assuming 5-minute intervals)
            max_entries = (24 * 60) // (self.cleanup_interval // 60)
            if len(self._memory_stats_history) > max_entries:
                self._memory_stats_history = self._memory_stats_history[-max_entries:]
        
        # Check if memory optimization is needed
        if current_stats.utilization_percent > self.memory_threshold * 100:
            logger.info(f"Memory utilization high ({current_stats.utilization_percent:.1f}%), "
                       "performing optimization")
            self.optimize_memory()
        
        # Remove very old models (not accessed in 2 hours)
        cutoff_time = time.time() - 7200  # 2 hours
        with self._cache_lock:
            old_models = [
                name for name, entry in self._model_cache.items()
                if entry.last_accessed < cutoff_time
            ]
            
            for model_name in old_models:
                self.remove_cached_model(model_name)
                logger.debug(f"Removed old model '{model_name}' from cache")


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(
    max_memory_mb: int = 4096,
    max_cached_models: int = 5,
    **kwargs
) -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(
            max_memory_mb=max_memory_mb,
            max_cached_models=max_cached_models,
            **kwargs
        )
    return _memory_manager


def cleanup_memory_manager() -> None:
    """Cleanup the global memory manager."""
    global _memory_manager
    if _memory_manager is not None:
        _memory_manager.shutdown()
        _memory_manager = None

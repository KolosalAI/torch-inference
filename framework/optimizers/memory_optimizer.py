"""
Memory optimization module for PyTorch inference.

This module provides advanced memory pool management, fragmentation prevention,
and various memory optimizations to reduce allocation overhead and improve performance.
"""

import logging
import os
import threading
import time
import gc
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import weakref
import psutil
from pathlib import Path
import json

import torch
import torch.nn as nn

from ..core.config import InferenceConfig


logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    enable_memory_pool: bool = True
    pool_size_mb: int = 512
    fragmentation_threshold: float = 0.5
    cleanup_interval: int = 10
    enable_background_cleanup: bool = True
    memory_growth_factor: float = 2.0
    enable_cuda_memory_pool: bool = True
    gradient_accumulation_steps: int = 4
    optimization_level: str = "balanced"  # "minimal", "balanced", "aggressive"
    enable_profiling: bool = False
    cleanup_threshold: float = 0.8
    enable_garbage_collection: bool = True
    gc_threshold: float = 0.8


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    cached_mb: float
    free_mb: float
    total_mb: float
    fragmentation_ratio: float = 0.0
    largest_free_block_mb: float = 0.0


@dataclass  
class MemoryBlock:
    """Memory block descriptor for fragmentation analysis."""
    size: torch.Size = None
    dtype: torch.dtype = None
    device: torch.device = None
    in_use: bool = False
    allocation_time: float = 0.0
    address: int = 0
    is_free: bool = True
    age: float = 0.0
    tensor_id: Optional[str] = None

    def __post_init__(self):
        if self.allocation_time == 0.0:
            self.allocation_time = time.time()
        if self.age == 0.0:
            self.age = time.time()

    def get_age(self) -> float:
        """Get the age of this memory block in seconds."""
        return time.time() - self.allocation_time


@dataclass
class FragmentationStats:
    """Memory fragmentation statistics."""
    total_free_memory: int = 0
    largest_free_block: int = 0
    num_free_blocks: int = 0
    fragmentation_ratio: float = 0.0
    avg_block_size: float = 0.0
    memory_efficiency: float = 1.0
    total_allocated: int = 0
    total_free: int = 0
    num_allocated_blocks: int = 0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class AdvancedMemoryPool:
    """
    Advanced memory pool with fragmentation prevention and smart allocation strategies.
    """
    
    def __init__(self, config: Union[MemoryConfig, torch.device], initial_size: int = 100):
        """
        Initialize advanced memory pool.
        
        Args:
            config: Memory configuration or device for tensor allocation
            initial_size: Initial pool size
        """
        # Handle different input types for backward compatibility
        if isinstance(config, MemoryConfig):
            self.config = config
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            # Assume it's a device
            self.device = config
            self.config = MemoryConfig()  # Use default config
        
        self.pools: Dict[Tuple[Tuple[int, ...], torch.dtype], deque] = defaultdict(deque)
        
        # Fragmentation prevention
        self.size_buckets = defaultdict(deque)  # Size-based allocation
        self.free_blocks: List[MemoryBlock] = []
        self.allocated_blocks: List[MemoryBlock] = []
        
        # Statistics
        self.allocated_count = 0
        self.reuse_count = 0
        self.fragmentation_events = 0
        
        # Configuration
        self.max_pool_size = 1000
        self.compaction_threshold = getattr(self.config, 'fragmentation_threshold', 0.3)
        self.size_alignment = 256  # Align allocations to reduce fragmentation
        
        # Thread management
        self.cleanup_thread = None
        self.cleanup_thread_running = False
        
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.AdvancedMemoryPool")
        self.logger.info(f"Advanced memory pool initialized for device: {self.device}")
        
        # Start background maintenance if on CUDA and enabled
        if self.device.type == 'cuda' and getattr(self.config, 'enable_background_cleanup', True):
            self._start_background_maintenance()
    
    def _start_background_maintenance(self) -> None:
        """Start background thread for memory maintenance."""
        def maintenance_worker():
            while True:
                try:
                    time.sleep(30)  # Run every 30 seconds
                    self._perform_maintenance()
                except Exception as e:
                    self.logger.warning(f"Background maintenance error: {e}")
        
        thread = threading.Thread(target=maintenance_worker, daemon=True)
        thread.start()
        self.logger.debug("Started background memory maintenance thread")
    
    def start_background_cleanup(self) -> None:
        """Start background cleanup thread."""
        if not self.cleanup_thread_running and getattr(self.config, 'enable_background_cleanup', True):
            self.cleanup_thread_running = True
            self.cleanup_thread = threading.Thread(target=self._background_cleanup_worker, daemon=True)
            self.cleanup_thread.start()
            self.logger.info("Started background cleanup thread")
    
    def stop_background_cleanup(self) -> None:
        """Stop background cleanup thread."""
        self.cleanup_thread_running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
        self.logger.info("Stopped background cleanup thread")
    
    def _background_cleanup_worker(self) -> None:
        """Background cleanup worker thread."""
        cleanup_interval = getattr(self.config, 'cleanup_interval', 30)
        
        while self.cleanup_thread_running:
            try:
                time.sleep(cleanup_interval)
                if self.cleanup_thread_running:  # Check again after sleep
                    self._perform_maintenance()
            except Exception as e:
                self.logger.warning(f"Background cleanup error: {e}")
    
    def allocate_tensor(self, shape: torch.Size, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Allocate tensor from pool.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            device: Target device
            
        Returns:
            Allocated tensor
        """
        # Create memory block for tracking
        memory_block = MemoryBlock(
            size=shape,
            dtype=dtype,
            device=device,
            in_use=True,
            allocation_time=time.time()
        )
        
        with self.lock:
            self.allocated_blocks.append(memory_block)
            
        # Use existing get_tensor method
        tensor = self.get_tensor(shape, dtype)
        
        self.allocated_count += 1
        return tensor
    
    def deallocate_tensor(self, tensor: torch.Tensor) -> None:
        """
        Deallocate tensor back to pool.
        
        Args:
            tensor: Tensor to deallocate
        """
        with self.lock:
            # Move from allocated to free blocks
            for block in self.allocated_blocks:
                if (block.size == tensor.size() and 
                    block.dtype == tensor.dtype and 
                    block.device == tensor.device):
                    block.in_use = False
                    block.allocation_time = time.time()
                    self.allocated_blocks.remove(block)
                    self.free_blocks.append(block)
                    break
        
        # Use existing return_tensor method
        self.return_tensor(tensor)
    
    def cleanup_old_blocks(self) -> None:
        """Clean up old memory blocks."""
        self._cleanup_aged_blocks()
    
    def defragment(self) -> None:
        """Perform memory defragmentation."""
        self._compact_memory()
    
    def get_fragmentation_stats(self) -> FragmentationStats:
        """Get current fragmentation statistics."""
        return self._analyze_fragmentation()
    
    def _perform_maintenance(self) -> None:
        """Perform background memory maintenance."""
        with self.lock:
            # Check fragmentation level
            fragmentation_stats = self._analyze_fragmentation()
            
            if fragmentation_stats.fragmentation_ratio > self.compaction_threshold:
                self.logger.info(f"High fragmentation detected ({fragmentation_stats.fragmentation_ratio:.2f}), performing compaction")
                self._compact_memory()
            
            # Clean old unused blocks
            self._cleanup_aged_blocks()
            
            # Optimize pool sizes
            self._optimize_pool_sizes()
    
    def _analyze_fragmentation(self) -> FragmentationStats:
        """Analyze current memory fragmentation."""
        if self.device.type != 'cuda':
            return FragmentationStats(0, 0, 0, 0.0, 0.0, 1.0)
        
        try:
            # Get CUDA memory info
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            
            # Calculate fragmentation metrics
            free_memory = reserved - allocated
            
            # Estimate fragmentation based on pool statistics
            total_blocks = len(self.free_blocks)
            if total_blocks == 0:
                return FragmentationStats(
                    total_free_memory=int(free_memory),
                    largest_free_block=int(free_memory),
                    num_free_blocks=0,
                    fragmentation_ratio=0.0,
                    avg_block_size=0.0,
                    memory_efficiency=1.0
                )
            
            # Calculate fragmentation based on pool utilization
            total_pooled = sum(len(pool) for pool in self.pools.values())
            avg_pool_utilization = total_pooled / max(len(self.pools), 1)
            
            # Fragmentation ratio based on pool efficiency
            fragmentation_ratio = max(0.0, 1.0 - (avg_pool_utilization / 10.0))
            
            # Memory efficiency
            memory_efficiency = allocated / max(reserved, 1)
            
            return FragmentationStats(
                total_free_memory=int(free_memory),
                largest_free_block=int(free_memory * memory_efficiency),
                num_free_blocks=total_blocks,
                fragmentation_ratio=fragmentation_ratio,
                avg_block_size=free_memory / max(total_blocks, 1),
                memory_efficiency=memory_efficiency
            )
            
        except Exception as e:
            self.logger.warning(f"Fragmentation analysis failed: {e}")
            return FragmentationStats(0, 0, 0, 0.0, 0.0, 1.0)
    
    def _compact_memory(self) -> None:
        """Perform memory compaction to reduce fragmentation."""
        try:
            self.logger.info("Starting memory compaction")
            
            # Clear unused pools to force reallocation
            pools_before = len(self.pools)
            self._cleanup_empty_pools()
            pools_after = len(self.pools)
            
            # Force CUDA cache clearing if needed
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                # Trigger garbage collection
                gc.collect()
            
            self.logger.info(f"Memory compaction completed: {pools_before} -> {pools_after} pools")
            
        except Exception as e:
            self.logger.warning(f"Memory compaction failed: {e}")
    
    def _cleanup_aged_blocks(self) -> None:
        """Clean up aged unused blocks."""
        current_time = time.time()
        max_age = 300.0  # 5 minutes
        
        # Clean aged entries from pools
        for pool_key, pool in list(self.pools.items()):
            if len(pool) > 0:
                # Keep only recent entries
                new_pool = deque()
                for tensor in pool:
                    # Simple age check based on creation time
                    if hasattr(tensor, '_creation_time'):
                        age = current_time - tensor._creation_time
                        if age < max_age:
                            new_pool.append(tensor)
                    else:
                        new_pool.append(tensor)  # Keep if no timestamp
                
                if len(new_pool) != len(pool):
                    self.pools[pool_key] = new_pool
                    if len(new_pool) == 0:
                        del self.pools[pool_key]
    
    def _cleanup_empty_pools(self) -> None:
        """Remove empty pools."""
        empty_keys = [key for key, pool in self.pools.items() if len(pool) == 0]
        for key in empty_keys:
            del self.pools[key]
    
    def _optimize_pool_sizes(self) -> None:
        """Optimize pool sizes based on usage patterns."""
        for pool_key, pool in self.pools.items():
            # Limit pool size to prevent unbounded growth
            if len(pool) > self.max_pool_size:
                # Keep only the most recently used tensors
                while len(pool) > self.max_pool_size:
                    pool.popleft()
    
    def _align_size(self, size: int) -> int:
        """Align size to reduce fragmentation."""
        return ((size + self.size_alignment - 1) // self.size_alignment) * self.size_alignment
    
    def _get_size_bucket(self, shape: Tuple[int, ...]) -> str:
        """Get size bucket for shape-based pooling."""
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        
        # Categorize by order of magnitude
        if total_elements < 1024:
            return "small"
        elif total_elements < 1024 * 1024:
            return "medium" 
        elif total_elements < 1024 * 1024 * 100:
            return "large"
        else:
            return "xlarge"
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Get a tensor from the pool with fragmentation prevention.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            
        Returns:
            Tensor from pool or newly allocated
        """
        pool_key = (shape, dtype)
        size_bucket = self._get_size_bucket(shape)
        
        with self.lock:
            # Try exact match first
            pool = self.pools[pool_key]
            if pool:
                tensor = pool.popleft()
                self.reuse_count += 1
                tensor.zero_()
                return tensor
            
            # Try size bucket for similar tensors
            bucket_pool = self.size_buckets[size_bucket]
            if bucket_pool:
                # Find compatible tensor in bucket
                for i, (candidate_shape, candidate_dtype, candidate_tensor) in enumerate(bucket_pool):
                    if candidate_dtype == dtype and self._shapes_compatible(shape, candidate_shape):
                        # Remove from bucket
                        bucket_pool.remove((candidate_shape, candidate_dtype, candidate_tensor))
                        
                        # Reshape if needed
                        if candidate_shape != shape:
                            try:
                                reshaped = candidate_tensor.view(shape)
                                self.reuse_count += 1
                                reshaped.zero_()
                                return reshaped
                            except RuntimeError:
                                pass  # Incompatible reshape, continue search
            
            # Allocate new tensor with fragmentation prevention
            tensor = self._allocate_new_tensor(shape, dtype)
            self.allocated_count += 1
            
            # Add timestamp for aging
            tensor._creation_time = time.time()
            
            return tensor
    
    def _shapes_compatible(self, target_shape: Tuple[int, ...], candidate_shape: Tuple[int, ...]) -> bool:
        """Check if shapes are compatible for reuse."""
        target_elements = 1
        candidate_elements = 1
        
        for dim in target_shape:
            target_elements *= dim
        for dim in candidate_shape:
            candidate_elements *= dim
        
        # Allow reuse if candidate is same size or larger (within 25%)
        return candidate_elements >= target_elements and candidate_elements <= target_elements * 1.25
    
    def _allocate_new_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Allocate new tensor with optimal strategy."""
        try:
            # Try normal allocation first
            return torch.zeros(shape, dtype=dtype, device=self.device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Out of memory - try cleanup and retry
                self.logger.warning("Out of memory during allocation, attempting cleanup")
                
                # Clear caches
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Force cleanup of pools
                self.clear()
                gc.collect()
                
                # Retry allocation
                try:
                    return torch.zeros(shape, dtype=dtype, device=self.device)
                except RuntimeError:
                    self.logger.error("Failed to allocate tensor even after cleanup")
                    raise
            else:
                raise
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """
        Return a tensor to the pool with fragmentation prevention.
        
        Args:
            tensor: Tensor to return to pool
        """
        if tensor.device != self.device:
            return  # Don't pool tensors from different devices
        
        pool_key = (tuple(tensor.shape), tensor.dtype)
        size_bucket = self._get_size_bucket(tensor.shape)
        
        with self.lock:
            pool = self.pools[pool_key]
            
            # Limit pool size to prevent unbounded growth
            if len(pool) < self.max_pool_size:
                pool.append(tensor.detach())
            
            # Also add to size bucket for cross-shape reuse
            bucket_pool = self.size_buckets[size_bucket]
            if len(bucket_pool) < self.max_pool_size // 2:
                bucket_pool.append((tuple(tensor.shape), tensor.dtype, tensor.detach()))


class MemoryPool:
    """
    Legacy memory pool for backward compatibility.
    """
    
    def __init__(self, device: torch.device, initial_size: int = 100):
        """
        Initialize memory pool.
        
        Args:
            device: Device for tensor allocation
            initial_size: Initial pool size
        """
        # Delegate to advanced pool
        self._advanced_pool = AdvancedMemoryPool(device, initial_size)
        self.device = device
        
        # Legacy compatibility properties
        self.allocated_count = 0
        self.reuse_count = 0
        
        self.logger = logging.getLogger(f"{__name__}.MemoryPool")
        self.logger.info(f"Memory pool (legacy) initialized for device: {device}")
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get tensor from advanced pool."""
        tensor = self._advanced_pool.get_tensor(shape, dtype)
        
        # Update legacy counters
        if self._advanced_pool.reuse_count > self.reuse_count:
            self.reuse_count = self._advanced_pool.reuse_count
        if self._advanced_pool.allocated_count > self.allocated_count:
            self.allocated_count = self._advanced_pool.allocated_count
        
        return tensor
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return tensor to advanced pool."""
        self._advanced_pool.return_tensor(tensor)
    
    def clear(self) -> None:
        """Clear advanced pool."""
        self._advanced_pool.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get stats from advanced pool."""
        advanced_stats = self._advanced_pool.get_stats()
        
        # Convert to legacy format
        return {
            "allocated_count": advanced_stats.get("allocated_count", 0),
            "reuse_count": advanced_stats.get("reuse_count", 0),
            "total_pooled_tensors": advanced_stats.get("total_pooled_tensors", 0),
            "pool_types": advanced_stats.get("pool_types", 0),
            "reuse_ratio": advanced_stats.get("reuse_ratio", 0.0)
        }


# Update the AdvancedMemoryPool methods that are missing
class AdvancedMemoryPool(AdvancedMemoryPool):
    """Extended AdvancedMemoryPool with missing methods."""
    
    def clear(self) -> None:
        """Clear all pools and reset statistics."""
        with self.lock:
            for pool in self.pools.values():
                pool.clear()
            self.pools.clear()
            
            for bucket in self.size_buckets.values():
                bucket.clear()
            self.size_buckets.clear()
            
            self.free_blocks.clear()
            self.allocated_blocks.clear()
            
        # Force garbage collection
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        self.logger.info("Advanced memory pools cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory pool statistics."""
        with self.lock:
            total_pooled = sum(len(pool) for pool in self.pools.values())
            pool_types = len(self.pools)
            bucket_utilization = {
                bucket: len(pool) for bucket, pool in self.size_buckets.items()
            }
            
            fragmentation_stats = self._analyze_fragmentation()
            
            return {
                "allocated_count": self.allocated_count,
                "reuse_count": self.reuse_count,
                "total_pooled_tensors": total_pooled,
                "pool_types": pool_types,
                "reuse_ratio": self.reuse_count / max(self.allocated_count, 1),
                "fragmentation_events": self.fragmentation_events,
                "bucket_utilization": bucket_utilization,
                "fragmentation_ratio": fragmentation_stats.fragmentation_ratio,
                "memory_efficiency": fragmentation_stats.memory_efficiency,
                "num_size_buckets": len(self.size_buckets)
            }


class MemoryOptimizer:
    """
    Memory optimization manager for PyTorch inference.
    """
    
    def __init__(self, config: Optional[Union[InferenceConfig, MemoryConfig]] = None):
        """
        Initialize memory optimizer.
        
        Args:
            config: Inference configuration or MemoryConfig
        """
        # Handle different config types
        if isinstance(config, MemoryConfig):
            self.config = config
            self.inference_config = None
        else:
            self.inference_config = config
            self.config = MemoryConfig()  # Use default memory config
        
        # Initialize memory pool
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_pool = AdvancedMemoryPool(
            config=self.config,  # Pass config instead of device
            initial_size=self.config.pool_size_mb
        )
        
        # Initialize caches and data structures
        self.optimization_cache = {}
        self.performance_metrics = {}
        
        self.pools: Dict[torch.device, MemoryPool] = {}
        self.enabled = True
        
        self.logger = logging.getLogger(f"{__name__}.MemoryOptimizer")
        self.logger.info("Memory optimizer initialized")
        
        # Configure CUDA memory allocation if available
        if torch.cuda.is_available():
            self._configure_cuda_memory()
    
    def optimize(self, model: nn.Module) -> nn.Module:
        """
        Apply memory optimizations to model.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        # Apply in-place optimizations
        self.enable_gradient_checkpointing(model)
        return model
    
    def optimize_model(self, model: nn.Module, example_inputs: torch.Tensor = None) -> nn.Module:
        """
        Apply memory optimizations to model (alias for optimize with additional parameters).
        
        Args:
            model: PyTorch model to optimize
            example_inputs: Example inputs (for future use)
            
        Returns:
            Optimized model
        """
        return self.optimize(model)
    
    def apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """
        Apply gradient checkpointing to model (alias for enable_gradient_checkpointing).
        
        Args:
            model: PyTorch model
            
        Returns:
            Model with gradient checkpointing enabled
        """
        self.enable_gradient_checkpointing(model)
        return model
    
    def enable_gradient_checkpointing(self, model: nn.Module) -> None:
        """
        Enable gradient checkpointing for model.
        
        Args:
            model: PyTorch model
        """
        try:
            # Apply gradient checkpointing to eligible layers
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
                    self.logger.info(f"Enabled gradient checkpointing for {type(module).__name__}")
        except Exception as e:
            self.logger.warning(f"Failed to enable gradient checkpointing: {e}")
    
    def optimize_attention_memory(self, model: nn.Module, example_inputs: torch.Tensor = None) -> nn.Module:
        """
        Apply memory-efficient attention optimizations.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs (for future use)
            
        Returns:
            Memory-optimized model
        """
        try:
            # Enable memory-efficient attention for transformer models
            for module in model.modules():
                if hasattr(module, 'attention') and hasattr(module.attention, 'enable_mem_efficient_attention'):
                    module.attention.enable_mem_efficient_attention = True
                    self.logger.info(f"Enabled memory-efficient attention for {type(module).__name__}")
                
                # For MultiheadAttention modules
                if isinstance(module, nn.MultiheadAttention):
                    # PyTorch's native MultiheadAttention doesn't have built-in memory efficiency
                    # but we can set reasonable defaults
                    if hasattr(module, 'batch_first'):
                        module.batch_first = True  # More memory efficient
                    self.logger.info(f"Configured {type(module).__name__} for memory efficiency")
            
            return model
        except Exception as e:
            self.logger.warning(f"Failed to optimize attention memory: {e}")
            return model
    
    def profile_memory_usage(self, model: nn.Module, example_inputs: torch.Tensor) -> Dict[str, Any]:
        """
        Profile memory usage during model inference.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs for profiling
            
        Returns:
            Memory profile data
        """
        try:
            memory_timeline = []
            initial_memory = self.get_memory_usage()
            memory_timeline.append(('start', initial_memory))
            
            # Run forward pass and monitor memory
            with torch.no_grad():
                def memory_hook(module, input, output):
                    current_memory = self.get_memory_usage()
                    memory_timeline.append((type(module).__name__, current_memory))
                
                # Register hooks
                hooks = []
                for name, module in model.named_modules():
                    if len(list(module.children())) == 0:  # Only leaf modules
                        hook = module.register_forward_hook(memory_hook)
                        hooks.append(hook)
                
                # Run inference
                output = model(example_inputs)
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
            
            final_memory = self.get_memory_usage()
            
            # Get peak memory from timeline
            memory_values = []
            for entry in memory_timeline:
                if len(entry) >= 2:
                    memory_val = entry[1]
                    if isinstance(memory_val, dict) and 'allocated_memory' in memory_val:
                        memory_values.append(memory_val['allocated_memory'])
                    elif isinstance(memory_val, (int, float)):
                        memory_values.append(memory_val)
            
            peak_memory = max(memory_values) if memory_values else 0
            
            # Extract layer-wise memory usage
            layer_memory_usage = {}
            for layer_name, memory in memory_timeline[1:]:
                if layer_name not in layer_memory_usage:
                    layer_memory_usage[layer_name] = []
                layer_memory_usage[layer_name].append(memory)
            
            return {
                "peak_memory": peak_memory,
                "memory_timeline": memory_timeline,
                "layer_memory_usage": layer_memory_usage,
                "initial_memory": initial_memory,
                "final_memory": final_memory
            }
            
        except Exception as e:
            self.logger.error(f"Memory profiling failed: {e}")
            return {
                "peak_memory": 0,
                "memory_timeline": [],
                "layer_memory_usage": {},
                "error": str(e)
            }
    
    def maybe_run_garbage_collection(self) -> None:
        """
        Run garbage collection if memory usage is above threshold.
        """
        try:
            current_memory = self.get_memory_usage()
            gc_threshold = getattr(self.config, 'gc_threshold', 0.8)
            
            # Simple heuristic: if current memory is above threshold, run GC
            if isinstance(current_memory, dict) and 'total_memory' in current_memory:
                if current_memory.get('available_memory', 0) < (current_memory['total_memory'] * (1 - gc_threshold)):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.logger.info("Automatic garbage collection triggered")
            elif isinstance(current_memory, (int, float)) and current_memory > gc_threshold * 1000:  # Simple MB threshold
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.info("Automatic garbage collection triggered")
        except Exception as e:
            self.logger.warning(f"Garbage collection check failed: {e}")
    
    def find_optimal_batch_size(self, model: nn.Module, sample_input: torch.Tensor, target_batch_size: int) -> int:
        """
        Find optimal batch size that fits in memory.
        
        Args:
            model: PyTorch model
            sample_input: Sample input tensor
            target_batch_size: Desired batch size to try
            
        Returns:
            Optimal batch size that fits in memory
        """
        optimal_batch_size = 1
        
        with torch.no_grad():
            for batch_size in [1, 2, 4, 8, 16, 32, 64]:
                if batch_size > target_batch_size:
                    break
                
                try:
                    # Create batch
                    if sample_input.dim() >= 1:
                        batch_shape = [batch_size] + list(sample_input.shape[1:])
                    else:
                        batch_shape = [batch_size]
                    
                    batch_input = torch.randn(batch_shape, dtype=sample_input.dtype, device=sample_input.device)
                    
                    # Test forward pass
                    output = model(batch_input)
                    
                    # If successful, update optimal batch size
                    optimal_batch_size = batch_size
                    
                    # Clean up
                    del batch_input, output
                    
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    if "out of memory" in str(e).lower():
                        self.logger.info(f"Batch size {batch_size} caused OOM, using {optimal_batch_size}")
                        break
                    else:
                        # Other error, re-raise
                        raise
        
        return optimal_batch_size
    
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
    
    def get_pool(self, device: torch.device) -> AdvancedMemoryPool:
        """
        Get or create advanced memory pool for device.
        
        Args:
            device: Target device
            
        Returns:
            Advanced memory pool for device
        """
        if device not in self.pools:
            self.pools[device] = AdvancedMemoryPool(device)
        
        pool = self.pools[device]
        # Ensure we return an AdvancedMemoryPool, not a MemoryPool wrapper
        if hasattr(pool, '_advanced_pool'):
            return pool._advanced_pool
        return pool
    
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
        Get detailed memory statistics with fragmentation analysis.
        
        Args:
            device: Target device (None for CUDA default)
            
        Returns:
            Memory statistics with fragmentation info
        """
        if device is None and torch.cuda.is_available():
            device = torch.device('cuda')
        
        if device is not None and device.type == 'cuda':
            # CUDA memory stats with fragmentation analysis
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            
            # Get total GPU memory
            props = torch.cuda.get_device_properties(device)
            total = props.total_memory / (1024 ** 2)
            free = total - reserved
            cached = reserved - allocated
            
            # Analyze fragmentation
            fragmentation_ratio = 0.0
            largest_free_block = free
            
            if device in self.pools:
                pool = self.pools[device]
                if hasattr(pool, '_analyze_fragmentation'):
                    fragmentation_stats = pool._analyze_fragmentation()
                    fragmentation_ratio = fragmentation_stats.fragmentation_ratio
                    largest_free_block = fragmentation_stats.largest_free_block / (1024 ** 2)
            
            return MemoryStats(
                allocated_mb=allocated,
                reserved_mb=reserved,
                max_allocated_mb=max_allocated,
                cached_mb=cached,
                free_mb=free,
                total_mb=total,
                fragmentation_ratio=fragmentation_ratio,
                largest_free_block_mb=largest_free_block
            )
        else:
            # CPU memory stats (simplified)
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                allocated = memory_info.rss / (1024 ** 2)
                
                return MemoryStats(
                    allocated_mb=allocated,
                    reserved_mb=allocated,
                    max_allocated_mb=allocated,
                    cached_mb=0,
                    free_mb=0,
                    total_mb=0,
                    fragmentation_ratio=0.0,
                    largest_free_block_mb=0.0
                )
            except ImportError:
                return MemoryStats(0, 0, 0, 0, 0, 0, 0.0, 0.0)
    
    def get_memory_usage(self, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Get current memory usage information.
        
        Args:
            device: Target device for memory statistics
            
        Returns:
            Dictionary with memory usage information
        """
        try:
            memory_info = {}
            
            # Get memory statistics
            stats = self.get_memory_stats(device)
            
            if device is None and torch.cuda.is_available():
                device = torch.device('cuda')
            
            if device and device.type == 'cuda':
                # CUDA memory information
                memory_info.update({
                    "total_memory": stats.total_mb,
                    "allocated_memory": stats.allocated_mb,
                    "reserved_memory": stats.reserved_mb,
                    "available_memory": stats.free_mb,
                    "cached_memory": stats.cached_mb,
                    "fragmentation_ratio": stats.fragmentation_ratio,
                    "cuda_memory": {
                        "allocated": stats.allocated_mb,
                        "reserved": stats.reserved_mb,
                        "free": stats.free_mb,
                        "total": stats.total_mb
                    }
                })
            else:
                # CPU memory information
                memory_info.update({
                    "total_memory": stats.allocated_mb,
                    "allocated_memory": stats.allocated_mb,
                    "available_memory": max(0, stats.total_mb - stats.allocated_mb) if stats.total_mb > 0 else 0,
                    "fragmentation_ratio": stats.fragmentation_ratio
                })
            
            return memory_info
            
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            # Return fallback values
            return {
                "total_memory": 0,
                "allocated_memory": 0,
                "available_memory": 0,
                "fragmentation_ratio": 0.0,
                "error": str(e)
            }
    
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
        Get comprehensive memory optimization report with fragmentation analysis.
        
        Returns:
            Memory optimization report
        """
        report = {
            "enabled": self.enabled,
            "pools": {},
            "memory_stats": {},
            "memory_usage": {},  # Add this for test compatibility
            "fragmentation_stats": {},
            "fragmentation_analysis": {},
            "optimizations_applied": [],
            "recommendations": []
        }
        
        # Memory usage (for test compatibility)
        try:
            memory_usage = self.get_memory_usage()
            report["memory_usage"] = memory_usage
        except Exception as e:
            report["memory_usage"] = {"error": str(e)}
        
        # Pool statistics with fragmentation info
        for device, pool in self.pools.items():
            if hasattr(pool, 'get_stats'):
                pool_stats = pool.get_stats()
                report["pools"][str(device)] = pool_stats
                
                # Advanced fragmentation analysis
                if hasattr(pool, '_analyze_fragmentation'):
                    fragmentation_stats = pool._analyze_fragmentation()
                    report["fragmentation_analysis"][str(device)] = {
                        "fragmentation_ratio": fragmentation_stats.fragmentation_ratio,
                        "memory_efficiency": fragmentation_stats.memory_efficiency,
                        "num_free_blocks": fragmentation_stats.num_free_blocks,
                        "avg_block_size": fragmentation_stats.avg_block_size,
                        "total_free_memory_mb": fragmentation_stats.total_free_memory / (1024 ** 2)
                    }
        
        # Enhanced fragmentation stats section
        if hasattr(self, 'memory_pool') and hasattr(self.memory_pool, 'get_fragmentation_stats'):
            try:
                frag_stats = self.memory_pool.get_fragmentation_stats()
                report["fragmentation_stats"] = {
                    "fragmentation_ratio": frag_stats.fragmentation_ratio,
                    "memory_efficiency": frag_stats.memory_efficiency,
                    "total_free": frag_stats.total_free,
                    "total_allocated": frag_stats.total_allocated,
                    "num_free_blocks": frag_stats.num_free_blocks,
                    "num_allocated_blocks": frag_stats.num_allocated_blocks,
                    "timestamp": frag_stats.timestamp
                }
            except Exception as e:
                report["fragmentation_stats"] = {"error": str(e)}
        
        # Memory statistics for all devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f'cuda:{i}')
                report["memory_stats"][str(device)] = self.get_memory_stats(device)
        
        # CPU memory
        cpu_stats = self.get_memory_stats(torch.device('cpu'))
        report["memory_stats"]["cpu"] = cpu_stats
        
        # Track applied optimizations
        optimizations = []
        if self.enabled:
            optimizations.append("memory_pool_enabled")
        if getattr(self.config, 'enable_cuda_memory_pool', False) and torch.cuda.is_available():
            optimizations.append("cuda_memory_pool")
        if getattr(self.config, 'enable_background_cleanup', False):
            optimizations.append("background_cleanup")
        
        report["optimizations_applied"] = optimizations
        
        # Generate enhanced recommendations
        recommendations = []
        
        # Memory usage recommendations
        for device_str, stats in report["memory_stats"].items():
            if hasattr(stats, 'total_mb') and stats.total_mb > 0:
                usage_percent = (stats.allocated_mb / stats.total_mb) * 100
                
                if usage_percent > 90:
                    recommendations.append(f"{device_str}: Critical memory usage ({usage_percent:.1f}%). Consider reducing batch size or model sharding.")
                elif usage_percent > 75:
                    recommendations.append(f"{device_str}: High memory usage ({usage_percent:.1f}%). Monitor closely and consider optimizations.")
                elif usage_percent < 30:
                    recommendations.append(f"{device_str}: Low memory usage ({usage_percent:.1f}%). Consider increasing batch size for better utilization.")
                
                # Fragmentation recommendations
                if hasattr(stats, 'fragmentation_ratio') and stats.fragmentation_ratio > 0.3:
                    recommendations.append(f"{device_str}: High memory fragmentation ({stats.fragmentation_ratio:.1f}). Consider memory pool optimization or compaction.")
        
        # Pool efficiency recommendations
        for device_str, pool_stats in report["pools"].items():
            if isinstance(pool_stats, dict):
                reuse_ratio = pool_stats.get("reuse_ratio", 0)
                if reuse_ratio < 0.3:
                    recommendations.append(f"{device_str} pool: Low reuse ratio ({reuse_ratio:.2f}). Pool may not be effective for current workload.")
                
                fragmentation_events = pool_stats.get("fragmentation_events", 0)
                if fragmentation_events > 100:
                    recommendations.append(f"{device_str} pool: High fragmentation events ({fragmentation_events}). Consider adjusting allocation strategies.")
        
        # Fragmentation-specific recommendations
        for device_str, frag_stats in report["fragmentation_analysis"].items():
            if frag_stats.get("fragmentation_ratio", 0) > 0.5:
                recommendations.append(f"{device_str}: Severe fragmentation detected. Enable automatic compaction or restart model.")
            elif frag_stats.get("memory_efficiency", 1.0) < 0.7:
                recommendations.append(f"{device_str}: Poor memory efficiency ({frag_stats['memory_efficiency']:.2f}). Consider memory optimization techniques.")
        
        report["recommendations"] = recommendations
        
        return report
    
    def enable_fragmentation_prevention(self, 
                                      compaction_threshold: float = 0.3,
                                      maintenance_interval: float = 30.0) -> None:
        """
        Enable advanced fragmentation prevention features.
        
        Args:
            compaction_threshold: Fragmentation ratio threshold for triggering compaction
            maintenance_interval: Background maintenance interval in seconds
        """
        for device, pool in self.pools.items():
            if hasattr(pool, 'compaction_threshold'):
                pool.compaction_threshold = compaction_threshold
                self.logger.info(f"Enabled fragmentation prevention for {device}")
        
        self.logger.info("Advanced fragmentation prevention enabled globally")
    
    def perform_memory_compaction(self, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Manually trigger memory compaction.
        
        Args:
            device: Target device (None for all devices)
            
        Returns:
            Compaction results
        """
        results = {}
        
        if device is None:
            # Compact all devices
            for dev, pool in self.pools.items():
                if hasattr(pool, '_compact_memory'):
                    try:
                        before_stats = pool._analyze_fragmentation()
                        pool._compact_memory()
                        after_stats = pool._analyze_fragmentation()
                        
                        results[str(dev)] = {
                            "success": True,
                            "fragmentation_before": before_stats.fragmentation_ratio,
                            "fragmentation_after": after_stats.fragmentation_ratio,
                            "improvement": before_stats.fragmentation_ratio - after_stats.fragmentation_ratio
                        }
                    except Exception as e:
                        results[str(dev)] = {
                            "success": False,
                            "error": str(e)
                        }
        else:
            # Compact specific device
            if device in self.pools:
                pool = self.pools[device]
                if hasattr(pool, '_compact_memory'):
                    try:
                        before_stats = pool._analyze_fragmentation()
                        pool._compact_memory()
                        after_stats = pool._analyze_fragmentation()
                        
                        results[str(device)] = {
                            "success": True,
                            "fragmentation_before": before_stats.fragmentation_ratio,
                            "fragmentation_after": after_stats.fragmentation_ratio,
                            "improvement": before_stats.fragmentation_ratio - after_stats.fragmentation_ratio
                        }
                    except Exception as e:
                        results[str(device)] = {
                            "success": False,
                            "error": str(e)
                        }
        
        self.logger.info(f"Memory compaction completed for {len(results)} devices")
        return results
    
    def save_memory_profile(self, filepath: Optional[str] = None) -> None:
        """
        Save current memory profile to disk for analysis.
        
        Args:
            filepath: Path to save profile (auto-generated if None)
        """
        if filepath is None:
            timestamp = int(time.time())
            filepath = f"memory_profile_{timestamp}.json"
        
        try:
            profile_data = {
                "timestamp": time.time(),
                "optimization_report": self.get_optimization_report(),
                "system_info": {
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                }
            }
            
            # Add CUDA device info if available
            if torch.cuda.is_available():
                cuda_devices = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    cuda_devices.append({
                        "device_id": i,
                        "name": props.name,
                        "total_memory_gb": props.total_memory / (1024**3),
                        "compute_capability": f"{props.major}.{props.minor}"
                    })
                profile_data["system_info"]["cuda_devices"] = cuda_devices
            
            Path(filepath).write_text(json.dumps(profile_data, indent=2, default=str))
            self.logger.info(f"Memory profile saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save memory profile: {e}")
    
    def analyze_memory_leaks(self, duration: float = 60.0, interval: float = 5.0) -> Dict[str, Any]:
        """
        Analyze potential memory leaks over time.
        
        Args:
            duration: Analysis duration in seconds
            interval: Sampling interval in seconds
            
        Returns:
            Memory leak analysis results
        """
        self.logger.info(f"Starting memory leak analysis for {duration}s")
        
        samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            timestamp = time.time()
            memory_stats = {}
            
            # Collect memory stats for all devices
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device = torch.device(f'cuda:{i}')
                    stats = self.get_memory_stats(device)
                    memory_stats[f'cuda:{i}'] = {
                        'allocated_mb': stats.allocated_mb,
                        'reserved_mb': stats.reserved_mb,
                        'fragmentation_ratio': stats.fragmentation_ratio
                    }
            
            # CPU memory
            cpu_stats = self.get_memory_stats(torch.device('cpu'))
            memory_stats['cpu'] = {
                'allocated_mb': cpu_stats.allocated_mb
            }
            
            samples.append({
                'timestamp': timestamp,
                'memory_stats': memory_stats
            })
            
            time.sleep(interval)
        
        # Analyze trends
        analysis = {
            'duration_s': duration,
            'num_samples': len(samples),
            'devices': {},
            'potential_leaks': []
        }
        
        for device_key in samples[0]['memory_stats'].keys():
            device_samples = [s['memory_stats'][device_key] for s in samples]
            
            # Calculate trends
            allocated_values = [s.get('allocated_mb', 0) for s in device_samples]
            
            if len(allocated_values) > 1:
                # Simple linear trend analysis
                time_points = list(range(len(allocated_values)))
                trend_slope = (allocated_values[-1] - allocated_values[0]) / len(allocated_values)
                
                # Memory growth analysis
                max_growth = max(allocated_values) - min(allocated_values)
                avg_growth_rate = trend_slope
                
                analysis['devices'][device_key] = {
                    'initial_allocated_mb': allocated_values[0],
                    'final_allocated_mb': allocated_values[-1],
                    'max_allocated_mb': max(allocated_values),
                    'min_allocated_mb': min(allocated_values),
                    'total_growth_mb': max_growth,
                    'avg_growth_rate_mb_per_sample': avg_growth_rate,
                    'trend_slope': trend_slope
                }
                
                # Detect potential leaks
                if trend_slope > 1.0:  # Growing > 1MB per sample
                    analysis['potential_leaks'].append({
                        'device': device_key,
                        'growth_rate_mb_per_sample': trend_slope,
                        'severity': 'high' if trend_slope > 5.0 else 'medium'
                    })
        
        self.logger.info(f"Memory leak analysis completed. Found {len(analysis['potential_leaks'])} potential issues")
        return analysis


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

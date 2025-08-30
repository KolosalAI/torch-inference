"""
High-Performance Batch Processor for PyTorch Inference

This module provides optimized batch processing capabilities:
- Dynamic batch sizing with adaptive algorithms
- Multi-stage pipeline processing
- Memory-efficient tensor operations
- Concurrent batch execution
- Advanced scheduling algorithms
- GPU memory management
- Performance monitoring and optimization
"""

import asyncio
import time
import logging
import threading
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
import torch
import torch.nn.functional as F
from enum import Enum
import weakref
import psutil

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Batch processing strategies"""
    FIXED_SIZE = "fixed_size"
    DYNAMIC_SIZE = "dynamic_size"
    ADAPTIVE_SIZE = "adaptive_size"
    MEMORY_AWARE = "memory_aware"


class ProcessingStage(Enum):
    """Processing pipeline stages"""
    PREPROCESSING = "preprocessing"
    INFERENCE = "inference"
    POSTPROCESSING = "postprocessing"


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    
    # Batch Size Settings
    min_batch_size: int = 1
    max_batch_size: int = 8
    default_batch_size: int = 4
    
    # Timing Settings
    batch_timeout_ms: float = 50
    max_wait_time_ms: float = 100.0
    processing_timeout_s: float = 30.0
    performance_target_ms: float = 100.0
    
    # Batching Strategy
    enable_adaptive_batching: bool = True
    enable_dynamic_batching: bool = True
    adaptive_scaling_factor: float = 1.2
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE_SIZE
    
    # Memory Management
    enable_memory_management: bool = True
    memory_threshold_mb: float = 1000.0
    max_memory_usage_gb: float = 8.0
    memory_safety_margin: float = 0.2  # 20% safety margin
    enable_memory_monitoring: bool = True
    memory_aware: bool = True
    
    # Pipeline Configuration
    enable_pipeline_processing: bool = True
    enable_pipelining: bool = True
    pipeline_stages: int = 3
    stage_overlap: bool = True
    
    # Performance Optimization
    enable_tensor_caching: bool = True
    enable_kernel_fusion: bool = True
    enable_mixed_precision: bool = True
    enable_compilation: bool = True
    
    # Scheduling
    priority_scheduling: bool = True
    load_balancing: bool = True
    adaptive_scheduling: bool = True
    
    # Monitoring
    performance_monitoring: bool = True
    detailed_metrics: bool = True


@dataclass
class BatchItem:
    """Individual item in a batch"""
    id: str
    data: Any
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    future: Optional[asyncio.Future] = None
    
    def __post_init__(self):
        if self.future is None:
            try:
                self.future = asyncio.Future()
            except RuntimeError:
                # No event loop running, will create it later when needed
                self.future = None
    
    def is_expired(self, timeout: float) -> bool:
        return (time.time() - self.timestamp) > timeout


@dataclass
class BatchResult:
    """Result of batch processing"""
    batch_id: str
    items: List[BatchItem]
    results: List[Any]
    processing_time: float
    stage_times: Dict[ProcessingStage, float]
    memory_usage: Dict[str, float]
    batch_size: int
    success: bool
    error: Optional[str] = None
    
    @property
    def throughput(self) -> float:
        """Calculate throughput (items/second)"""
        return self.batch_size / max(self.processing_time, 0.001)


class AdaptiveBatchSizer:
    """Adaptive batch sizing algorithm"""
    
    def __init__(self, config: BatchConfig = None, initial_size: int = None, min_size: int = None, max_size: int = None):
        if config is not None:
            self.config = config
            self.min_size = config.min_batch_size
            self.max_size = config.max_batch_size
            self.current_size = config.default_batch_size
            self.max_memory_bytes = config.max_memory_usage_gb * 1024 * 1024 * 1024
            self.safety_margin = config.memory_safety_margin
        else:
            # Support old-style initialization with individual parameters
            self.min_size = min_size or 1
            self.max_size = max_size or 8
            self.current_size = initial_size or 4
            self.max_memory_bytes = 8 * 1024 * 1024 * 1024  # 8GB default
            self.safety_margin = 0.2
            # Create a minimal config
            self.config = BatchConfig(
                min_batch_size=self.min_size,
                max_batch_size=self.max_size,
                default_batch_size=self.current_size
            )
        
        # Performance history
        self.performance_history = deque(maxlen=50)
        self.memory_history = deque(maxlen=20)
        
        # Adaptive parameters
        self.target_latency_ms = 50.0
        self.target_throughput = 100.0  # items/second
        self.learning_rate = 0.1
    
    def update_performance(self, batch_result: BatchResult):
        """Update performance history and adjust batch size"""
        self.performance_history.append({
            'batch_size': batch_result.batch_size,
            'processing_time': batch_result.processing_time * 1000,  # Convert to ms
            'throughput': batch_result.throughput,
            'memory_usage': batch_result.memory_usage.get('peak_memory', 0),
            'timestamp': time.time()
        })
        
        # Update memory history
        memory_usage = batch_result.memory_usage.get('peak_memory', 0)
        self.memory_history.append(memory_usage)
        
        # Adaptive size adjustment
        if self.config.strategy == BatchStrategy.ADAPTIVE_SIZE:
            self._adjust_adaptive_size(batch_result)
        elif self.config.strategy == BatchStrategy.MEMORY_AWARE:
            self._adjust_memory_aware_size(batch_result)
    
    def _adjust_adaptive_size(self, batch_result: BatchResult):
        """Adjust batch size based on performance metrics"""
        if len(self.performance_history) < 3:
            return
        
        recent_performance = list(self.performance_history)[-5:]
        avg_latency = sum(p['processing_time'] for p in recent_performance) / len(recent_performance)
        avg_throughput = sum(p['throughput'] for p in recent_performance) / len(recent_performance)
        
        # Adjust based on latency
        if avg_latency > self.target_latency_ms * 1.2:  # Too slow
            # Decrease batch size
            adjustment = -max(1, int(self.current_size * self.learning_rate))
        elif avg_latency < self.target_latency_ms * 0.8:  # Too fast, can increase
            # Check if throughput can be improved
            if avg_throughput < self.target_throughput:
                adjustment = max(1, int(self.current_size * self.learning_rate))
            else:
                adjustment = 0
        else:
            adjustment = 0
        
        # Apply adjustment
        new_size = self.current_size + adjustment
        self.current_size = max(self.min_size, min(self.max_size, new_size))
    
    def _adjust_memory_aware_size(self, batch_result: BatchResult):
        """Adjust batch size based on memory usage"""
        if not self.memory_history:
            return
        
        avg_memory = sum(self.memory_history) / len(self.memory_history)
        memory_ratio = avg_memory / self.max_memory_bytes
        
        if memory_ratio > (1.0 - self.safety_margin):  # Near memory limit
            # Aggressively reduce batch size
            adjustment = -max(1, int(self.current_size * 0.2))
        elif memory_ratio > 0.7:  # High memory usage
            # Slightly reduce batch size
            adjustment = -max(1, int(self.current_size * 0.1))
        elif memory_ratio < 0.4:  # Low memory usage
            # Can increase batch size
            adjustment = max(1, int(self.current_size * 0.1))
        else:
            adjustment = 0
        
        new_size = self.current_size + adjustment
        self.current_size = max(self.min_size, min(self.max_size, new_size))
    
    def get_batch_size(self, queue_size: int, available_memory: float) -> int:
        """Get optimal batch size based on current conditions"""
        if self.config.strategy == BatchStrategy.FIXED_SIZE:
            return min(self.config.default_batch_size, queue_size)
        
        elif self.config.strategy == BatchStrategy.DYNAMIC_SIZE:
            # Simple dynamic sizing based on queue
            if queue_size > self.max_size * 2:
                return self.max_size
            elif queue_size > self.max_size:
                return min(self.max_size, queue_size // 2)
            else:
                return min(self.current_size, queue_size)
        
        elif self.config.strategy == BatchStrategy.MEMORY_AWARE:
            # Memory-constrained sizing
            memory_ratio = (self.max_memory_bytes - available_memory) / self.max_memory_bytes
            max_safe_size = int(self.max_size * (1.0 - memory_ratio))
            return min(max_safe_size, self.current_size, queue_size)
        
        else:  # ADAPTIVE_SIZE
            return min(self.current_size, queue_size)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch sizer statistics"""
        if not self.performance_history:
            return {'current_batch_size': self.current_size}
        
        recent = list(self.performance_history)[-10:]
        avg_latency = sum(p['processing_time'] for p in recent) / len(recent)
        avg_throughput = sum(p['throughput'] for p in recent) / len(recent)
        
        return {
            'current_batch_size': self.current_size,
            'strategy': self.config.strategy.value,
            'avg_latency_ms': avg_latency,
            'avg_throughput': avg_throughput,
            'performance_samples': len(self.performance_history),
            'memory_samples': len(self.memory_history)
        }


class BatchQueue:
    """High-performance queue for batch items"""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self._items = deque()
        self._priority_items = deque()  # High priority items
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        
        # Statistics
        self._stats = {
            'items_queued': 0,
            'items_processed': 0,
            'items_expired': 0,
            'avg_queue_time': 0.0
        }
    
    async def put(self, item: BatchItem) -> bool:
        """Add item to queue"""
        async with self._not_empty:
            if item.priority > 0:
                self._priority_items.append(item)
            else:
                self._items.append(item)
            
            self._stats['items_queued'] += 1
            self._not_empty.notify()
            return True
    
    async def get_batch(self, batch_size: int, timeout_ms: float) -> List[BatchItem]:
        """Get batch of items"""
        async with self._not_empty:
            # Wait for items if queue is empty
            if len(self._items) == 0 and len(self._priority_items) == 0:
                try:
                    await asyncio.wait_for(
                        self._not_empty.wait(), 
                        timeout=timeout_ms / 1000.0
                    )
                except asyncio.TimeoutError:
                    return []
            
            batch = []
            current_time = time.time()
            
            # First, get high priority items
            while self._priority_items and len(batch) < batch_size:
                item = self._priority_items.popleft()
                if not item.is_expired(self.config.processing_timeout_s):
                    batch.append(item)
                else:
                    self._stats['items_expired'] += 1
                    if item.future and not item.future.done():
                        item.future.set_exception(asyncio.TimeoutError("Item expired"))
            
            # Then get regular priority items
            while self._items and len(batch) < batch_size:
                item = self._items.popleft()
                if not item.is_expired(self.config.processing_timeout_s):
                    batch.append(item)
                else:
                    self._stats['items_expired'] += 1
                    if item.future and not item.future.done():
                        item.future.set_exception(asyncio.TimeoutError("Item expired"))
            
            # Update queue time statistics
            if batch:
                avg_queue_time = sum(current_time - item.timestamp for item in batch) / len(batch)
                self._update_avg_queue_time(avg_queue_time)
            
            return batch
    
    def _update_avg_queue_time(self, queue_time: float):
        """Update average queue time"""
        current_avg = self._stats['avg_queue_time']
        processed = self._stats['items_processed']
        
        if processed == 0:
            self._stats['avg_queue_time'] = queue_time
        else:
            self._stats['avg_queue_time'] = (current_avg * processed + queue_time) / (processed + 1)
    
    def size(self) -> int:
        """Get current queue size"""
        return len(self._items) + len(self._priority_items)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return dict(self._stats)


class MemoryManager:
    """GPU/CPU memory manager for batch processing"""
    
    def __init__(self, config: BatchConfig = None, threshold_mb: float = None, warning_threshold: float = None, critical_threshold: float = None):
        if config is not None:
            self.config = config
            self.max_memory = config.max_memory_usage_gb * 1024 * 1024 * 1024  # Convert to bytes
        else:
            # Support old-style initialization
            self.max_memory = (threshold_mb or 1000.0) * 1024 * 1024  # Convert MB to bytes
            # Create minimal config
            self.config = BatchConfig()
            
        # Store thresholds
        self.warning_threshold = warning_threshold or 0.8
        self.critical_threshold = critical_threshold or 0.95
        self.threshold_mb = threshold_mb or (self.max_memory / (1024 * 1024))
        
        # Memory tracking
        self._peak_memory = 0
        self._current_memory = 0
        self._memory_history = deque(maxlen=100)
        
        # Tensor caching
        self._tensor_cache: Dict[str, torch.Tensor] = {}
        self._cache_usage = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Memory pools for different tensor shapes
        self._memory_pools: Dict[Tuple, List[torch.Tensor]] = defaultdict(list)
        
    def get_available_memory(self) -> float:
        """Get available memory in bytes"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            return total_memory - allocated_memory
        else:
            # CPU memory
            memory = psutil.virtual_memory()
            return memory.available
    
    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                       device: torch.device = None) -> torch.Tensor:
        """Allocate tensor with memory management"""
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check memory pools first
        pool_key = (shape, dtype, device)
        if pool_key in self._memory_pools and self._memory_pools[pool_key]:
            tensor = self._memory_pools[pool_key].pop()
            tensor.zero_()  # Clear the tensor
            return tensor
        
        # Allocate new tensor
        try:
            tensor = torch.zeros(shape, dtype=dtype, device=device)
            
            # Update memory tracking
            tensor_size = tensor.numel() * tensor.element_size()
            self._current_memory += tensor_size
            self._peak_memory = max(self._peak_memory, self._current_memory)
            
            return tensor
            
        except torch.cuda.OutOfMemoryError:
            # Try to free some memory
            self._cleanup_memory()
            # Retry with smaller tensor or fallback to CPU
            if device.type == 'cuda':
                return self.allocate_tensor(shape, dtype, torch.device('cpu'))
            else:
                raise
    
    def deallocate_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool for reuse"""
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        device = tensor.device
        pool_key = (shape, dtype, device)
        
        # Add to pool if not too many
        if len(self._memory_pools[pool_key]) < 10:  # Limit pool size
            self._memory_pools[pool_key].append(tensor.detach())
        
        # Update memory tracking
        tensor_size = tensor.numel() * tensor.element_size()
        self._current_memory = max(0, self._current_memory - tensor_size)
    
    def _cleanup_memory(self):
        """Clean up memory pools and caches"""
        # Clear oldest entries from pools
        for pool_key, pool in list(self._memory_pools.items()):
            if len(pool) > 5:  # Keep only 5 most recent
                removed = pool[:-5]
                self._memory_pools[pool_key] = pool[-5:]
                
                # Update memory tracking
                for tensor in removed:
                    tensor_size = tensor.numel() * tensor.element_size()
                    self._current_memory = max(0, self._current_memory - tensor_size)
        
        # Clear tensor cache if needed
        if len(self._tensor_cache) > 100:
            # Remove 50% of cache
            items_to_remove = len(self._tensor_cache) // 2
            keys_to_remove = list(self._tensor_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self._tensor_cache[key]
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_cached_tensor(self, key: str) -> Optional[torch.Tensor]:
        """Get tensor from cache"""
        if key in self._tensor_cache:
            self._cache_hits += 1
            return self._tensor_cache[key]
        else:
            self._cache_misses += 1
            return None
    
    def cache_tensor(self, key: str, tensor: torch.Tensor):
        """Cache tensor"""
        self._tensor_cache[key] = tensor.detach()
        self._cache_usage += tensor.numel() * tensor.element_size()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        available_memory = self.get_available_memory()
        
        return {
            'peak_memory': self._peak_memory,
            'current_memory': self._current_memory,
            'available_memory': available_memory,
            'memory_pools': {str(k): len(v) for k, v in self._memory_pools.items()},
            'cache_size': len(self._tensor_cache),
            'cache_usage': self._cache_usage,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': self._cache_hits / max(self._cache_hits + self._cache_misses, 1)
        }
    
    def is_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        try:
            available_memory = self.get_available_memory()
            memory_ratio = (self.max_memory - available_memory) / self.max_memory
            return memory_ratio > self.warning_threshold
        except Exception:
            return False


class BatchProcessor:
    """Main high-performance batch processor"""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.logger = logging.getLogger(f"{__name__}.BatchProcessor")
        
        # Core components
        self.batch_queue = BatchQueue(self.config)
        self.batch_sizer = AdaptiveBatchSizer(self.config)
        self.memory_manager = MemoryManager(self.config)
        
        # Processing components
        self._executor = ThreadPoolExecutor(
            max_workers=4, 
            thread_name_prefix="batch-processor"
        )
        
        # Pipeline stages
        self._preprocessing_workers = 2
        self._inference_workers = 1
        self._postprocessing_workers = 2
        
        # Background tasks
        self._processing_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        self._started = False
        
        # Statistics
        self._stats = {
            'batches_processed': 0,
            'items_processed': 0,
            'total_processing_time': 0.0,
            'avg_batch_size': 0.0,
            'avg_processing_time': 0.0,
            'throughput': 0.0,
            'errors': 0
        }
        
        # Stage timers
        self._stage_times: Dict[ProcessingStage, List[float]] = {
            stage: deque(maxlen=100) for stage in ProcessingStage
        }
    
    async def start(self):
        """Start the batch processor"""
        if self._running:
            return
        
        self._running = True
        self._started = True
        
        # Start processing task
        self._processing_task = asyncio.create_task(self._processing_loop())
        
        # Start monitoring task
        if self.config.performance_monitoring:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("Batch processor started")
    
    async def stop(self):
        """Stop the batch processor"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel tasks
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown executor (non-blocking to avoid deadlocks)
        self._executor.shutdown(wait=False)
        
        self.logger.info("Batch processor stopped")
    
    async def process_item(self, data: Any, priority: int = 0, 
                          metadata: Dict[str, Any] = None, handler: Callable = None) -> Any:
        """Process single item through batch processing"""
        item_id = f"item_{int(time.time() * 1000000)}"
        item = BatchItem(
            id=item_id,
            data=data,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Ensure future is created with current event loop
        if item.future is None:
            item.future = asyncio.Future()
        
        # Store handler in metadata for later use
        if handler is not None:
            item.metadata['handler'] = handler
        
        # Add to queue
        await self.batch_queue.put(item)
        
        # Wait for result
        try:
            result = await item.future
            return result
        except Exception as e:
            self.logger.error(f"Item processing failed: {e}")
            raise
    
    async def _processing_loop(self):
        """Main processing loop"""
        while self._running:
            try:
                # Get optimal batch size
                queue_size = self.batch_queue.size()
                available_memory = self.memory_manager.get_available_memory()
                batch_size = self.batch_sizer.get_batch_size(queue_size, available_memory)
                
                if batch_size == 0:
                    await asyncio.sleep(0.001)  # Short sleep if no work
                    continue
                
                # Get batch from queue
                timeout_ms = self.config.batch_timeout_ms
                batch = await self.batch_queue.get_batch(batch_size, timeout_ms)
                
                if not batch:
                    await asyncio.sleep(0.005)  # Short sleep if no batch
                    continue
                
                # Process batch
                batch_result = await self._process_batch(batch)
                
                # Update performance metrics
                self.batch_sizer.update_performance(batch_result)
                self._update_stats(batch_result)
                
                # Set results for futures
                for item, result in zip(batch, batch_result.results):
                    if item.future and not item.future.done():
                        if batch_result.success:
                            item.future.set_result(result)
                        else:
                            item.future.set_exception(Exception(batch_result.error or "Processing failed"))
                
            except Exception as e:
                self.logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, batch: List[BatchItem]) -> BatchResult:
        """Process a batch of items"""
        batch_id = f"batch_{int(time.time() * 1000000)}"
        start_time = time.time()
        stage_times = {}
        memory_usage = {}

        try:
            # Check if this batch has function handlers (non-tensor processing)
            has_handlers = any(item.metadata.get('handler') for item in batch)
            
            if has_handlers:
                # Process using function handlers directly
                # Group items by their handlers
                handler_groups = {}
                for item in batch:
                    handler = item.metadata.get('handler')
                    if handler not in handler_groups:
                        handler_groups[handler] = []
                    handler_groups[handler].append(item)
                
                results = []
                for handler, handler_items in handler_groups.items():
                    if handler:
                        try:
                            # Extract data from items
                            batch_data = [item.data for item in handler_items]
                            
                            if asyncio.iscoroutinefunction(handler):
                                handler_results = await handler(batch_data)
                            else:
                                handler_results = handler(batch_data)
                                
                            # Ensure we have results for each item
                            if not isinstance(handler_results, (list, tuple)):
                                handler_results = [handler_results] * len(handler_items)
                            
                            results.extend(handler_results)
                        except Exception as e:
                            self.logger.error(f"Handler processing failed: {e}")
                            # Add error results for all items in this batch
                            results.extend([e] * len(handler_items))
                    else:
                        # Default fallback for items without handlers
                        for item in handler_items:
                            results.append(f"processed_{item.data}")
                        
                processing_time = time.time() - start_time
                return BatchResult(
                    batch_id=batch_id,
                    items=batch,
                    results=results,
                    processing_time=processing_time,
                    stage_times={ProcessingStage.INFERENCE: processing_time},
                    memory_usage={'current_mb': 0},
                    batch_size=len(batch),
                    success=True
                )
            
            # Original tensor-based processing for ML models
            # Stage 1: Preprocessing
            preprocessing_start = time.time()
            preprocessed_data = await self._preprocess_batch(batch)
            stage_times[ProcessingStage.PREPROCESSING] = time.time() - preprocessing_start

            # Stage 2: Inference
            inference_start = time.time()
            inference_results = await self._inference_batch(preprocessed_data)
            stage_times[ProcessingStage.INFERENCE] = time.time() - inference_start

            # Stage 3: Postprocessing
            postprocessing_start = time.time()
            results = await self._postprocess_batch(inference_results, batch)
            stage_times[ProcessingStage.POSTPROCESSING] = time.time() - postprocessing_start

            # Get memory usage
            memory_usage = self.memory_manager.get_memory_stats()

            processing_time = time.time() - start_time

            return BatchResult(
                batch_id=batch_id,
                items=batch,
                results=results,
                processing_time=processing_time,
                stage_times=stage_times,
                memory_usage=memory_usage,
                batch_size=len(batch),
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Batch processing failed: {e}")
            
            return BatchResult(
                batch_id=batch_id,
                items=batch,
                results=[None] * len(batch),
                processing_time=processing_time,
                stage_times=stage_times,
                memory_usage=memory_usage,
                batch_size=len(batch),
                success=False,
                error=str(e)
            )
    
    async def _preprocess_batch(self, batch: List[BatchItem]) -> List[torch.Tensor]:
        """Preprocess batch data"""
        loop = asyncio.get_event_loop()
        
        def preprocess_item(item: BatchItem) -> torch.Tensor:
            data = item.data
            
            # Convert to tensor
            if isinstance(data, torch.Tensor):
                tensor = data
            elif isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data)
            elif isinstance(data, (list, tuple)):
                tensor = torch.tensor(data, dtype=torch.float32)
            elif isinstance(data, str):
                # For string data, create a dummy tensor or skip tensor conversion
                # This allows the batch processor to work with non-tensor data
                tensor = torch.tensor([0], dtype=torch.float32)  # Dummy tensor
            else:
                try:
                    # Try to convert to tensor
                    tensor = torch.tensor(data, dtype=torch.float32)
                except (ValueError, TypeError):
                    # If conversion fails, create a dummy tensor
                    tensor = torch.tensor([0], dtype=torch.float32)
            
            # Ensure correct device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            tensor = tensor.to(device)
            
            # Ensure batch dimension
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            
            return tensor
        
        # Process items concurrently
        tasks = []
        with ThreadPoolExecutor(max_workers=self._preprocessing_workers) as executor:
            for item in batch:
                task = loop.run_in_executor(executor, preprocess_item, item)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        
        return results
    
    async def _inference_batch(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Perform inference on batch"""
        # For now, this is a placeholder - in real implementation,
        # this would call the actual model
        
        def dummy_inference(tensor_batch: torch.Tensor) -> torch.Tensor:
            # Placeholder: just return the input transformed
            with torch.no_grad():
                # Simple transformation as placeholder
                result = torch.nn.functional.relu(tensor_batch.sum(dim=-1, keepdim=True))
                return result
        
        # Stack tensors for batch processing
        try:
            # Try to stack tensors (assumes same shape)
            stacked_tensors = torch.stack(tensors)
            
            # Run inference
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor, dummy_inference, stacked_tensors
            )
            
            # Split results back
            results = [result[i] for i in range(len(tensors))]
            return results
            
        except Exception as e:
            # Fallback: process individually
            self.logger.debug(f"Batch inference failed, processing individually: {e}")
            
            loop = asyncio.get_event_loop()
            tasks = []
            for tensor in tensors:
                task = loop.run_in_executor(
                    self._executor, dummy_inference, tensor.unsqueeze(0)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return [result.squeeze(0) for result in results]
    
    async def _postprocess_batch(self, inference_results: List[torch.Tensor], 
                                batch: List[BatchItem]) -> List[Any]:
        """Postprocess batch results"""
        def postprocess_item(tensor: torch.Tensor, item: BatchItem) -> Dict[str, Any]:
            # Convert to CPU and numpy
            result = tensor.detach().cpu().numpy()
            
            # Create response
            response = {
                'prediction': result.tolist(),
                'item_id': item.id,
                'processing_time': time.time() - item.timestamp,
                'metadata': item.metadata
            }
            
            return response
        
        # Process results
        loop = asyncio.get_event_loop()
        tasks = []
        
        with ThreadPoolExecutor(max_workers=self._postprocessing_workers) as executor:
            for tensor, item in zip(inference_results, batch):
                task = loop.run_in_executor(executor, postprocess_item, tensor, item)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        
        return results
    
    def _update_stats(self, batch_result: BatchResult):
        """Update processing statistics"""
        self._stats['batches_processed'] += 1
        self._stats['items_processed'] += batch_result.batch_size
        self._stats['total_processing_time'] += batch_result.processing_time
        
        if not batch_result.success:
            self._stats['errors'] += 1
        
        # Update averages
        batches = self._stats['batches_processed']
        self._stats['avg_batch_size'] = (
            (self._stats['avg_batch_size'] * (batches - 1) + batch_result.batch_size) / batches
        )
        self._stats['avg_processing_time'] = (
            self._stats['total_processing_time'] / batches
        )
        self._stats['throughput'] = (
            self._stats['items_processed'] / max(self._stats['total_processing_time'], 0.001)
        )
        
        # Update stage times
        for stage, stage_time in batch_result.stage_times.items():
            self._stage_times[stage].append(stage_time)
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._running:
            try:
                if self.config.detailed_metrics:
                    stats = self.get_stats()
                    self.logger.info(f"Batch processor stats: {stats}")
                
                await asyncio.sleep(30)  # Log every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = dict(self._stats)
        
        # Add component stats
        stats['queue'] = self.batch_queue.get_stats()
        stats['batch_sizer'] = self.batch_sizer.get_stats()
        stats['memory'] = self.memory_manager.get_memory_stats()
        
        # Calculate average batch size if we have processed batches
        if stats.get('batches_processed', 0) > 0 and stats.get('items_processed', 0) > 0:
            stats['average_batch_size'] = stats['items_processed'] / stats['batches_processed']
        else:
            stats['average_batch_size'] = 0.0
        
        # Add stage timing stats
        stage_stats = {}
        for stage, times in self._stage_times.items():
            if times:
                stage_stats[stage.value] = {
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'samples': len(times)
                }
        stats['stage_timings'] = stage_stats
        
        return stats
    
    async def submit_batch_item(self, batch_item: Dict[str, Any]) -> Any:
        """Submit batch item for processing - compatibility method"""
        # Extract info from batch_item dictionary
        data = batch_item.get('args', [])
        if isinstance(data, (list, tuple)) and len(data) > 0:
            data = data[0]  # Take first arg if it's a list
        
        handler = batch_item.get('function')
        future = batch_item.get('future')
        
        try:
            # Process through normal pipeline
            result = await self.process_item(data=data, handler=handler)
            
            # Set result on the provided future
            if future and not future.done():
                future.set_result(result)
                
            return result
            
        except Exception as e:
            if future and not future.done():
                future.set_exception(e)
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the batch processor"""
        try:
            queue_size = self.batch_queue.size()
            memory_stats = self.memory_manager.get_memory_stats()
            
            health_status = {
                'status': 'healthy' if self._running and queue_size < 1000 else 'degraded',
                'components': {
                    'batch_sizer': 'healthy',
                    'memory_manager': 'healthy' if memory_stats['available_memory'] > 0 else 'low_memory',
                    'scheduler': 'healthy',
                    'processing_loop': 'running' if self._running else 'stopped'
                },
                'metrics': {
                    'queue_size': queue_size,
                    'running': self._running,
                    'items_processed': self._stats['items_processed'],
                    'batches_processed': self._stats['batches_processed']
                }
            }
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


# Context manager for easy usage
@asynccontextmanager
async def batch_processor(config: BatchConfig = None):
    """Context manager for batch processor"""
    config = config or BatchConfig()
    processor = BatchProcessor(config)
    
    try:
        await processor.start()
        yield processor
    finally:
        await processor.stop()


class ProcessingPipeline:
    """Processing pipeline for batch operations"""
    
    def __init__(self, stages: List[Callable] = None, num_stages: int = None):
        self.stages = stages or []
        # Support num_stages parameter
        if num_stages is not None and not stages:
            # Create dummy stages
            self.stages = [lambda x: x for _ in range(num_stages)]
        self._stats = {
            'stages_completed': 0,
            'total_processing_time': 0.0,
            'items_processed': 0
        }
        self.logger = logging.getLogger(__name__)
    
    async def process(self, batch: List[BatchItem]) -> BatchResult:
        """Process a batch through all pipeline stages"""
        start_time = time.time()
        current_data = [item.data for item in batch]
        stage_times = {}
        
        try:
            for i, stage in enumerate(self.stages):
                stage_start = time.time()
                
                if asyncio.iscoroutinefunction(stage):
                    current_data = await stage(current_data)
                else:
                    current_data = stage(current_data)
                    
                stage_times[f"stage_{i}"] = time.time() - stage_start
                self._stats['stages_completed'] += 1
            
            processing_time = time.time() - start_time
            self._stats['total_processing_time'] += processing_time
            self._stats['items_processed'] += len(batch)
            
            # Create results
            results = []
            for i, (item, result_data) in enumerate(zip(batch, current_data)):
                results.append({
                    'id': item.id,
                    'data': result_data,
                    'processing_time': processing_time / len(batch)
                })
            
            return BatchResult(
                batch_id=f"batch_{int(time.time() * 1000000)}",
                items=batch,
                results=results,
                processing_time=processing_time,
                stage_times=stage_times,
                memory_usage={},
                batch_size=len(batch),
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline processing error: {e}")
            return BatchResult(
                batch_id=f"batch_{int(time.time() * 1000000)}",
                items=batch,
                results=[],
                processing_time=time.time() - start_time,
                stage_times=stage_times,
                memory_usage={},
                batch_size=len(batch),
                success=False,
                error=str(e)
            )
    
    def add_stage(self, stage: Callable):
        """Add a processing stage to the pipeline"""
        self.stages.append(stage)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return dict(self._stats)


class BatchScheduler:
    """Advanced batch scheduler with priority and resource management"""
    
    def __init__(self, config: BatchConfig = None, max_batch_size: int = None, timeout_ms: float = None, min_batch_size: int = None):
        if config is not None:
            self.config = config
            self.max_batch_size = config.max_batch_size
            self.timeout_ms = config.batch_timeout_ms
            self.min_batch_size = config.min_batch_size
        else:
            # Support old-style initialization
            self.max_batch_size = max_batch_size or 8
            self.timeout_ms = timeout_ms or 50
            self.min_batch_size = min_batch_size or 1
            # Create minimal config
            self.config = BatchConfig(
                max_batch_size=self.max_batch_size,
                batch_timeout_ms=self.timeout_ms,
                min_batch_size=self.min_batch_size
            )
            
        self.pending_items = []  # Add pending_items for compatibility
        self._scheduled_batches = deque()
        self._priority_queue = []  # Will use heapq for priority scheduling
        self._resource_usage = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'gpu_memory_percent': 0.0
        }
        self._stats = {
            'batches_scheduled': 0,
            'batches_completed': 0,
            'total_schedule_time': 0.0
        }
        self.logger = logging.getLogger(__name__)
    
    async def schedule_batch(self, batch: List[BatchItem], priority: int = 1) -> str:
        """Schedule a batch for processing with given priority"""
        import heapq
        
        batch_id = f"batch_{int(time.time() * 1000000)}"
        schedule_time = time.time()
        
        # Create batch entry
        batch_entry = {
            'id': batch_id,
            'batch': batch,
            'priority': priority,
            'schedule_time': schedule_time,
            'estimated_processing_time': self._estimate_processing_time(batch)
        }
        
        # Add to priority queue (negative priority for max-heap behavior)
        heapq.heappush(self._priority_queue, (-priority, schedule_time, batch_entry))
        
        self._stats['batches_scheduled'] += 1
        self.logger.debug(f"Scheduled batch {batch_id} with priority {priority}")
        
        return batch_id
    
    async def get_next_batch(self) -> Optional[Dict[str, Any]]:
        """Get the next batch to process based on priority and resources"""
        if not self._priority_queue:
            return None
        
        # Check resource availability
        if not self._check_resource_availability():
            return None
        
        import heapq
        _, _, batch_entry = heapq.heappop(self._priority_queue)
        return batch_entry
    
    def _estimate_processing_time(self, batch: List[BatchItem]) -> float:
        """Estimate processing time for a batch"""
        # Simple estimation based on batch size and historical data
        base_time = 0.1  # seconds
        return base_time + (len(batch) * 0.01)
    
    def _check_resource_availability(self) -> bool:
        """Check if resources are available for processing"""
        # Update current resource usage
        self._update_resource_usage()
        
        # Check thresholds
        if self._resource_usage['cpu_percent'] > 90:
            return False
        if self._resource_usage['memory_percent'] > 85:
            return False
        
        return True
    
    def _update_resource_usage(self):
        """Update current resource usage statistics"""
        try:
            # CPU usage
            self._resource_usage['cpu_percent'] = psutil.cpu_percent()
            
            # Memory usage
            memory = psutil.virtual_memory()
            self._resource_usage['memory_percent'] = memory.percent
            
            # GPU memory (if available)
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                self._resource_usage['gpu_memory_percent'] = gpu_memory
                
        except Exception as e:
            self.logger.debug(f"Resource monitoring error: {e}")
    
    def mark_completed(self, batch_id: str):
        """Mark a batch as completed"""
        self._stats['batches_completed'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        stats = dict(self._stats)
        stats['pending_batches'] = len(self._priority_queue)
        stats['resource_usage'] = dict(self._resource_usage)
        return stats

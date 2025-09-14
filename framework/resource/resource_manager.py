"""
Advanced Resource Management System

Provides:
- Memory management with leak detection
- Resource quotas and limits
- Queue management for request handling
- Connection limits and pool management
- Resource monitoring and enforcement
"""

import asyncio
import logging
import threading
import time
import psutil
import gc
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque, defaultdict
import weakref
from contextlib import asynccontextmanager
import tracemalloc

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Resource types for quota management."""
    MEMORY = "memory"
    CPU = "cpu"
    GPU_MEMORY = "gpu_memory"
    CONNECTIONS = "connections"
    REQUESTS_PER_MINUTE = "requests_per_minute"
    CONCURRENT_REQUESTS = "concurrent_requests"
    DISK_SPACE = "disk_space"


class QueuePriority(Enum):
    """Request priority levels."""
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class ResourceQuota:
    """Resource quota configuration."""
    resource_type: ResourceType
    limit: float
    warning_threshold: float = 0.8
    critical_threshold: float = 0.95
    enforcement: bool = True
    soft_limit: bool = False


@dataclass
class ResourceLimits:
    """Resource limits configuration."""
    max_memory_mb: int = 1024
    max_concurrent_requests: int = 100
    max_queue_size: int = 1000
    max_request_duration_seconds: int = 300
    max_cpu_percent: float = 80.0
    max_gpu_memory_mb: int = 2048
    
    def __post_init__(self):
        """Validate resource limits."""
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")
        if self.max_queue_size < 0:
            raise ValueError("max_queue_size cannot be negative")
        if self.max_request_duration_seconds <= 0:
            raise ValueError("max_request_duration_seconds must be positive")
        if self.max_cpu_percent <= 0 or self.max_cpu_percent > 100:
            raise ValueError("max_cpu_percent must be between 0 and 100")
        if self.max_gpu_memory_mb < 0:
            raise ValueError("max_gpu_memory_mb cannot be negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_memory_mb": self.max_memory_mb,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_queue_size": self.max_queue_size,
            "max_request_duration_seconds": self.max_request_duration_seconds,
            "max_cpu_percent": self.max_cpu_percent,
            "max_gpu_memory_mb": self.max_gpu_memory_mb
        }


@dataclass
class QueuedRequest:
    """Queued request representation."""
    id: str
    priority: QueuePriority
    data: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    timeout: Optional[float] = None
    callback: Optional[Callable] = None


@dataclass
class ResourceUsage:
    """Current resource usage."""
    resource_type: ResourceType
    current: float
    limit: float
    percentage: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MemoryTracker:
    """Advanced memory tracking and leak detection."""
    
    def __init__(self, warning_threshold_mb: int = 512, critical_threshold_mb: int = 1024, check_interval: float = 60.0):
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.check_interval = check_interval
        self._memory_history: deque = deque(maxlen=100)  # Keep last 100 measurements
        self._peak_memory = 0
        self._object_counts: Dict[str, int] = {}
        self._tracemalloc_enabled = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        
        # Enable tracemalloc if available
        try:
            tracemalloc.start()
            self._tracemalloc_enabled = True
            logger.info("Memory tracking with tracemalloc enabled")
        except Exception as e:
            logger.warning(f"Could not enable tracemalloc: {e}")
    
    def get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            return memory_mb
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return 0.0
    
    def update_memory_stats(self):
        """Update memory statistics."""
        try:
            current_memory = self.get_current_memory_usage()
            
            with self._lock:
                self._peak_memory = max(self._peak_memory, current_memory)
                self._memory_history.append((datetime.utcnow(), current_memory))
        except Exception as e:
            logger.error(f"Error updating memory stats: {e}")
    
    def check_memory_status(self) -> Dict[str, Any]:
        """Check current memory status against thresholds."""
        current_memory = self.get_current_memory_usage()
        
        status = {
            "current_mb": current_memory,
            "warning_threshold_mb": self.warning_threshold_mb,
            "critical_threshold_mb": self.critical_threshold_mb,
            "warning": current_memory >= self.warning_threshold_mb,
            "critical": current_memory >= self.critical_threshold_mb
        }
        
        if status["critical"]:
            status["level"] = "critical"
        elif status["warning"]:
            status["level"] = "warning"
        else:
            status["level"] = "normal"
        
        return status
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        return self._peak_memory
    
    def get_memory_history(self) -> List[Dict[str, Any]]:
        """Get memory usage history."""
        with self._lock:
            return [
                {
                    "timestamp": timestamp.isoformat(),
                    "memory_mb": memory_mb
                }
                for timestamp, memory_mb in self._memory_history
            ]
    
    def reset_stats(self):
        """Reset memory statistics."""
        with self._lock:
            self._peak_memory = 0
            self._memory_history.clear()
    
    async def start_monitoring(self):
        """Start memory monitoring."""
        if self._monitoring_task:
            return
        
        self._monitoring_task = asyncio.create_task(self._monitor_memory())
        logger.info("Memory monitoring started")
    
    async def stop_monitoring(self):
        """Stop memory monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("Memory monitoring stopped")
    
    async def _monitor_memory(self):
        """Memory monitoring loop."""
        try:
            while True:
                await asyncio.sleep(self.check_interval)
                
                try:
                    # Collect memory statistics
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    
                    usage = {
                        'timestamp': datetime.utcnow(),
                        'rss': memory_info.rss,  # Resident Set Size
                        'vms': memory_info.vms,  # Virtual Memory Size
                        'percent': process.memory_percent(),
                        'available': psutil.virtual_memory().available,
                        'total': psutil.virtual_memory().total
                    }
                    
                    # Add tracemalloc data if available
                    if self._tracemalloc_enabled:
                        current, peak = tracemalloc.get_traced_memory()
                        usage['traced_current'] = current
                        usage['traced_peak'] = peak
                    
                    with self._lock:
                        self._memory_history.append(usage)
                    
                    # Check for potential memory leaks
                    await self._check_memory_leaks()
                    
                except Exception as e:
                    logger.error(f"Error in memory monitoring: {e}")
        
        except asyncio.CancelledError:
            pass
    
    async def _check_memory_leaks(self):
        """Check for potential memory leaks."""
        with self._lock:
            if len(self._memory_history) < 10:
                return
            
            # Check if memory usage is consistently increasing
            recent_usage = list(self._memory_history)[-10:]
            memory_trend = [usage['rss'] for usage in recent_usage]
            
            # Simple trend detection: if last 5 measurements are all higher than first 5
            if len(memory_trend) >= 10:
                first_half_avg = sum(memory_trend[:5]) / 5
                second_half_avg = sum(memory_trend[5:]) / 5
                
                increase_percentage = (second_half_avg - first_half_avg) / first_half_avg * 100
                
                if increase_percentage > 20:  # 20% increase
                    logger.warning(f"Potential memory leak detected: {increase_percentage:.1f}% increase")
                    await self._analyze_memory_usage()
    
    async def _analyze_memory_usage(self):
        """Analyze memory usage for leak detection."""
        if not self._tracemalloc_enabled:
            return
        
        try:
            # Get top memory consumers
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            logger.info("Top 10 memory consumers:")
            for index, stat in enumerate(top_stats[:10], 1):
                logger.info(f"{index}. {stat}")
        
        except Exception as e:
            logger.error(f"Error analyzing memory usage: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            stats = {
                'current_rss': memory_info.rss,
                'current_vms': memory_info.vms,
                'current_percent': process.memory_percent(),
                'system_available': psutil.virtual_memory().available,
                'system_total': psutil.virtual_memory().total,
                'system_percent': psutil.virtual_memory().percent
            }
            
            if self._tracemalloc_enabled:
                current, peak = tracemalloc.get_traced_memory()
                stats['traced_current'] = current
                stats['traced_peak'] = peak
            
            with self._lock:
                if self._memory_history:
                    recent = list(self._memory_history)[-10:]
                    stats['history_avg'] = sum(h['rss'] for h in recent) / len(recent)
                    stats['history_max'] = max(h['rss'] for h in recent)
                    stats['history_min'] = min(h['rss'] for h in recent)
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return collection stats."""
        try:
            collected = {}
            for generation in range(3):
                collected[f'generation_{generation}'] = gc.collect(generation)
            
            logger.info(f"Garbage collection completed: {collected}")
            return collected
        
        except Exception as e:
            logger.error(f"Error in garbage collection: {e}")
            return {}


class RequestQueue:
    """Priority-based request queue with timeout handling."""
    
    def __init__(self, max_size: int = 1000, timeout_seconds: int = 30):
        self.max_size = max_size
        self.timeout_seconds = timeout_seconds
        self._queues = {
            QueuePriority.HIGH: deque(),
            QueuePriority.NORMAL: deque(),
            QueuePriority.LOW: deque()
        }
        self._pending_count = 0
        self._processed_count = 0
        self._timeout_count = 0
        self._condition = asyncio.Condition()
        self._lock = threading.RLock()
        
        # Cleanup task for expired requests
        self._cleanup_task: Optional[asyncio.Task] = None
    
    @property
    def size(self) -> int:
        """Get total number of pending requests."""
        with self._lock:
            return sum(len(q) for q in self._queues.values())
    
    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.size == 0
    
    @property
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self.size >= self.max_size
    
    async def enqueue(self, request_id: str, data: Any, priority: bool = False) -> None:
        """Enqueue a request."""
        request = QueuedRequest(
            id=request_id,
            priority=QueuePriority.HIGH if priority else QueuePriority.NORMAL,
            data=data,
            timeout=self.timeout_seconds
        )
        
        with self._lock:
            # Check if queue is full
            if self.size >= self.max_size:
                raise asyncio.QueueFull("Request queue is full")
            
            # Add to appropriate priority queue
            self._queues[request.priority].append(request)
            self._pending_count += 1
        
        # Notify waiting consumers
        async with self._condition:
            self._condition.notify()
    
    async def dequeue(self, timeout: Optional[float] = None) -> Tuple[str, Any, datetime]:
        """Dequeue the highest priority request."""
        async with self._condition:
            # Wait for available request
            start_time = time.time()
            while True:
                with self._lock:
                    # Try to get request from highest priority queue first
                    for priority in [QueuePriority.HIGH, QueuePriority.NORMAL, QueuePriority.LOW]:
                        queue = self._queues[priority]
                        if queue:
                            request = queue.popleft()
                            self._pending_count -= 1
                            self._processed_count += 1
                            return request.id, request.data, request.created_at
                
                # Check timeout
                if timeout and (time.time() - start_time) >= timeout:
                    raise asyncio.TimeoutError("Dequeue timeout")
                
                # Wait for notification
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    if timeout and (time.time() - start_time) >= timeout:
                        raise asyncio.TimeoutError("Dequeue timeout")
    
    def clear(self):
        """Clear all requests from the queue."""
        with self._lock:
            for queue in self._queues.values():
                queue.clear()
            self._pending_count = 0
    
    async def start(self):
        """Start the queue management."""
        if self._cleanup_task:
            return
        
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_requests())
        logger.info("Request queue started")
    
    async def stop(self):
        """Stop the queue management."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        # Clear all queues
        with self._lock:
            for queue in self._queues.values():
                queue.clear()
        
        logger.info("Request queue stopped")
    
    async def enqueue_request(self, request: QueuedRequest) -> bool:
        """Enqueue a request object directly."""
        with self._lock:
            # Check if queue is full
            total_pending = sum(len(q) for q in self._queues.values())
            if total_pending >= self.max_size:
                logger.warning("Request queue is full")
                return False
            
            # Add to appropriate priority queue
            self._queues[request.priority].append(request)
            self._pending_count += 1
        
        # Notify waiting consumers
        async with self._condition:
            self._condition.notify()
        
        return True
    
    async def _cleanup_expired_requests(self):
        """Clean up expired requests."""
        try:
            while True:
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
                now = datetime.utcnow()
                expired_count = 0
                
                with self._lock:
                    for priority, queue in self._queues.items():
                        # Check requests from the front (oldest first)
                        while queue:
                            request = queue[0]
                            
                            # Check if request has expired
                            if request.timeout:
                                elapsed = (now - request.created_at).total_seconds()
                                if elapsed > request.timeout:
                                    expired_request = queue.popleft()
                                    self._pending_count -= 1
                                    self._timeout_count += 1
                                    expired_count += 1
                                    
                                    # Call timeout callback if provided
                                    if expired_request.callback:
                                        try:
                                            if asyncio.iscoroutinefunction(expired_request.callback):
                                                asyncio.create_task(expired_request.callback(expired_request, "timeout"))
                                            else:
                                                expired_request.callback(expired_request, "timeout")
                                        except Exception as e:
                                            logger.error(f"Error in timeout callback: {e}")
                                else:
                                    break  # No more expired requests in this queue
                            else:
                                break  # Request has no timeout
                
                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired requests")
        
        except asyncio.CancelledError:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            total_pending = sum(len(q) for q in self._queues.values())
            return {
                'size': total_pending,  # Added for test compatibility
                'pending_high': len(self._queues[QueuePriority.HIGH]),
                'pending_normal': len(self._queues[QueuePriority.NORMAL]),
                'pending_low': len(self._queues[QueuePriority.LOW]),
                'total_pending': total_pending,
                'total_processed': self._processed_count,
                'total_timeouts': self._timeout_count,
                'max_size': self.max_size,
                'is_empty': total_pending == 0,
                'is_full': total_pending >= self.max_size,
                'timeout_seconds': self.timeout_seconds
            }


class ConnectionLimiter:
    """Connection limiting and management."""
    
    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self._active_connections: Set[str] = set()
        self._connection_history: deque = deque(maxlen=1000)
        self._semaphore = asyncio.Semaphore(max_connections)
        self._lock = threading.RLock()
        
        logger.info(f"Connection limiter initialized with max {max_connections} connections")
    
    @asynccontextmanager
    async def acquire_connection(self, connection_id: str):
        """Acquire a connection slot."""
        async with self._semaphore:
            with self._lock:
                self._active_connections.add(connection_id)
                self._connection_history.append({
                    'connection_id': connection_id,
                    'action': 'acquire',
                    'timestamp': datetime.utcnow(),
                    'active_count': len(self._active_connections)
                })
            
            try:
                yield
            finally:
                with self._lock:
                    self._active_connections.discard(connection_id)
                    self._connection_history.append({
                        'connection_id': connection_id,
                        'action': 'release',
                        'timestamp': datetime.utcnow(),
                        'active_count': len(self._active_connections)
                    })
    
    def get_active_connections(self) -> int:
        """Get number of active connections."""
        with self._lock:
            return len(self._active_connections)
    
    def get_available_connections(self) -> int:
        """Get number of available connection slots."""
        with self._lock:
            return self.max_connections - len(self._active_connections)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        with self._lock:
            recent_history = list(self._connection_history)[-100:]  # Last 100 events
            
            return {
                'active_connections': len(self._active_connections),
                'max_connections': self.max_connections,
                'available_connections': self.max_connections - len(self._active_connections),
                'utilization_percent': (len(self._active_connections) / self.max_connections) * 100,
                'recent_acquisitions': sum(1 for h in recent_history if h['action'] == 'acquire'),
                'recent_releases': sum(1 for h in recent_history if h['action'] == 'release')
            }


class ResourceQuotaManager:
    """Manages resource quotas and enforcement."""
    
    def __init__(self):
        self._quotas: Dict[str, Dict[str, float]] = {}
        self._usage: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._usage_history: Dict[ResourceType, deque] = defaultdict(lambda: deque(maxlen=100))
        self._monitoring_task: Optional[asyncio.Task] = None
        self._violation_callbacks: List[Callable] = []
        self._lock = threading.RLock()
    
    def set_quota(self, user_id: str, resource_type: str, limit: float):
        """Set a resource quota for a user."""
        with self._lock:
            if user_id not in self._quotas:
                self._quotas[user_id] = {}
            self._quotas[user_id][resource_type] = limit
        
        logger.info(f"Set quota for {user_id} {resource_type}: {limit}")
    
    def check_quota(self, user_id: str, resource_type: str) -> Tuple[bool, float]:
        """Check if user can consume resource quota."""
        with self._lock:
            if user_id not in self._quotas or resource_type not in self._quotas[user_id]:
                return True, float('inf')  # No quota set, allow unlimited
            
            limit = self._quotas[user_id][resource_type]
            
            # Check current usage
            current_usage = 0
            if user_id in self._usage and resource_type in self._usage[user_id]:
                usage_info = self._usage[user_id][resource_type]
                
                # Check if quota period has expired (e.g., per minute)
                current_time = time.time()
                if current_time >= usage_info.get('reset_time', 0):
                    # Reset quota
                    self._usage[user_id][resource_type] = {
                        'count': 0,
                        'reset_time': current_time + 60  # Reset after 1 minute
                    }
                    current_usage = 0
                else:
                    current_usage = usage_info['count']
            
            remaining = max(0, limit - current_usage)
            can_proceed = current_usage < limit
            
            return can_proceed, remaining
    
    def consume_quota(self, user_id: str, resource_type: str, amount: float = 1) -> bool:
        """Consume resource quota."""
        can_proceed, remaining = self.check_quota(user_id, resource_type)
        
        if not can_proceed or remaining < amount:
            return False
        
        with self._lock:
            if user_id not in self._usage:
                self._usage[user_id] = {}
            
            if resource_type not in self._usage[user_id]:
                self._usage[user_id][resource_type] = {
                    'count': 0,
                    'reset_time': time.time() + 60
                }
            
            self._usage[user_id][resource_type]['count'] += amount
        
        return True
    
    def get_quota_usage(self, user_id: str, resource_type: str) -> Dict[str, Any]:
        """Get quota usage information."""
        with self._lock:
            limit = self._quotas.get(user_id, {}).get(resource_type, float('inf'))
            usage_info = self._usage.get(user_id, {}).get(resource_type, {'count': 0, 'reset_time': time.time() + 60})
            
            used = usage_info['count']
            remaining = max(0, limit - used)
            
            return {
                'limit': limit,
                'used': used,
                'remaining': remaining,
                'reset_time': usage_info['reset_time']
            }
    
    def get_all_quotas(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """Get all quotas for a user."""
        with self._lock:
            result = {}
            user_quotas = self._quotas.get(user_id, {})
            
            for resource_type in user_quotas:
                result[resource_type] = self.get_quota_usage(user_id, resource_type)
            
            return result
    
    def remove_quota(self, user_id: str, resource_type: str):
        """Remove a quota."""
        with self._lock:
            if user_id in self._quotas and resource_type in self._quotas[user_id]:
                del self._quotas[user_id][resource_type]
                
                if not self._quotas[user_id]:  # Remove user if no quotas left
                    del self._quotas[user_id]
            
            # Also remove usage tracking
            if user_id in self._usage and resource_type in self._usage[user_id]:
                del self._usage[user_id][resource_type]
                
                if not self._usage[user_id]:
                    del self._usage[user_id]
    
    def set_resource_quota(self, quota: ResourceQuota):
        """Set a resource quota object."""
        with self._lock:
            self._quotas[quota.resource_type] = quota
        
        logger.info(f"Set quota for {quota.resource_type.value}: {quota.limit}")
    
    def add_violation_callback(self, callback: Callable[[ResourceType, ResourceUsage], None]):
        """Add a callback for quota violations."""
        self._violation_callbacks.append(callback)
    
    async def start_monitoring(self, check_interval: float = 30.0):
        """Start resource monitoring."""
        if self._monitoring_task:
            return
        
        self._monitoring_task = asyncio.create_task(
            self._monitor_resources(check_interval)
        )
        logger.info("Resource quota monitoring started")
    
    async def stop_monitoring(self):
        """Stop resource monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("Resource quota monitoring stopped")
    
    async def _monitor_resources(self, check_interval: float):
        """Resource monitoring loop."""
        try:
            while True:
                await asyncio.sleep(check_interval)
                
                try:
                    with self._lock:
                        quotas = dict(self._quotas)
                    
                    for resource_type, quota in quotas.items():
                        usage = await self._get_resource_usage(resource_type)
                        if usage:
                            await self._check_quota_compliance(quota, usage)
                
                except Exception as e:
                    logger.error(f"Error in resource monitoring: {e}")
        
        except asyncio.CancelledError:
            pass
    
    async def _get_resource_usage(self, resource_type: ResourceType) -> Optional[ResourceUsage]:
        """Get current resource usage."""
        try:
            if resource_type == ResourceType.MEMORY:
                process = psutil.Process()
                memory_info = process.memory_info()
                system_memory = psutil.virtual_memory()
                
                quota = self._quotas.get(resource_type)
                if quota:
                    current = memory_info.rss
                    percentage = (current / quota.limit) * 100
                    
                    return ResourceUsage(
                        resource_type=resource_type,
                        current=current,
                        limit=quota.limit,
                        percentage=percentage
                    )
            
            elif resource_type == ResourceType.CPU:
                process = psutil.Process()
                cpu_percent = process.cpu_percent()
                
                quota = self._quotas.get(resource_type)
                if quota:
                    return ResourceUsage(
                        resource_type=resource_type,
                        current=cpu_percent,
                        limit=quota.limit,
                        percentage=(cpu_percent / quota.limit) * 100
                    )
            
            elif resource_type == ResourceType.GPU_MEMORY:
                # This would integrate with GPU monitoring (nvidia-ml-py, etc.)
                # Placeholder implementation
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                        
                        quota = self._quotas.get(resource_type)
                        if quota:
                            return ResourceUsage(
                                resource_type=resource_type,
                                current=gpu_memory,
                                limit=quota.limit,
                                percentage=(gpu_memory / quota.limit) * 100
                            )
                except ImportError:
                    pass
            
            # Add more resource types as needed
            
        except Exception as e:
            logger.error(f"Error getting resource usage for {resource_type}: {e}")
        
        return None
    
    async def _check_quota_compliance(self, quota: ResourceQuota, usage: ResourceUsage):
        """Check if resource usage complies with quota."""
        with self._lock:
            self._usage_history[quota.resource_type].append(usage)
        
        # Check thresholds
        if usage.percentage >= quota.critical_threshold * 100:
            logger.critical(f"{quota.resource_type.value} usage critical: {usage.percentage:.1f}%")
            await self._handle_quota_violation(quota, usage, "critical")
        
        elif usage.percentage >= quota.warning_threshold * 100:
            logger.warning(f"{quota.resource_type.value} usage warning: {usage.percentage:.1f}%")
            await self._handle_quota_violation(quota, usage, "warning")
        
        # Enforce hard limits
        if quota.enforcement and not quota.soft_limit and usage.current >= quota.limit:
            logger.error(f"{quota.resource_type.value} quota exceeded: {usage.current} >= {quota.limit}")
            await self._handle_quota_violation(quota, usage, "exceeded")
    
    async def _handle_quota_violation(self, quota: ResourceQuota, usage: ResourceUsage, severity: str):
        """Handle quota violation."""
        # Call registered callbacks
        for callback in self._violation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(quota.resource_type, usage, severity)
                else:
                    callback(quota.resource_type, usage, severity)
            except Exception as e:
                logger.error(f"Error in quota violation callback: {e}")
    
    def get_resource_usage(self, resource_type: ResourceType) -> List[Dict[str, Any]]:
        """Get resource usage history."""
        with self._lock:
            history = list(self._usage_history.get(resource_type, []))
        
        return [
            {
                'timestamp': usage.timestamp.isoformat(),
                'current': usage.current,
                'limit': usage.limit,
                'percentage': usage.percentage
            }
            for usage in history
        ]
    
    def get_quota_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all quotas."""
        with self._lock:
            status = {}
            
            for resource_type, quota in self._quotas.items():
                recent_usage = list(self._usage_history.get(resource_type, []))
                current_usage = recent_usage[-1] if recent_usage else None
                
                status[resource_type.value] = {
                    'limit': quota.limit,
                    'warning_threshold': quota.warning_threshold,
                    'critical_threshold': quota.critical_threshold,
                    'enforcement': quota.enforcement,
                    'soft_limit': quota.soft_limit,
                    'current_usage': current_usage.current if current_usage else None,
                    'current_percentage': current_usage.percentage if current_usage else None,
                    'last_updated': current_usage.timestamp.isoformat() if current_usage else None
                }
            
            return status


class ResourceManager:
    """
    Comprehensive resource management system.
    
    Features:
    - Memory management with leak detection
    - Resource quotas and limits
    - Request queue management
    - Connection limiting
    - Resource monitoring and enforcement
    """
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.memory_tracker = MemoryTracker(
            warning_threshold_mb=int(self.limits.max_memory_mb * 0.8),
            critical_threshold_mb=self.limits.max_memory_mb
        )
        self.request_queue = RequestQueue(
            max_size=self.limits.max_queue_size,
            timeout_seconds=self.limits.max_request_duration_seconds
        )
        self.connection_limiter = ConnectionLimiter(self.limits.max_concurrent_requests)
        self.quota_manager = ResourceQuotaManager()
        
        # Request tracking
        self._active_requests = 0
        self._active_request_times: Dict[str, datetime] = {}
        self._request_lock = threading.RLock()
        
        # Setup default quotas
        self._setup_default_quotas()
        
        # Setup quota violation handling
        # self.quota_manager.add_violation_callback(self._handle_quota_violation)
        
        logger.info("Resource manager initialized")
    
    @property
    def active_requests(self) -> int:
        """Get number of active requests."""
        with self._request_lock:
            return self._active_requests
    
    def _setup_default_quotas(self):
        """Setup default resource quotas."""
        # Default quotas are set per user as needed
        pass
    
    async def can_accept_request(self, user_id: str) -> Tuple[bool, Optional[str]]:
        """Check if a request can be accepted."""
        # Check memory status
        memory_status = self.memory_tracker.check_memory_status()
        if memory_status["critical"]:
            return False, "Memory usage is critical"
        
        # Check concurrent request limit
        with self._request_lock:
            if self._active_requests >= self.limits.max_concurrent_requests:
                return False, "Concurrent request limit reached"
        
        # Check user quota
        can_proceed, remaining = self.quota_manager.check_quota(user_id, "requests_per_minute")
        if not can_proceed:
            return False, "User quota exceeded"
        
        return True, None
    
    async def acquire_resources(self, user_id: str, request_data: Any) -> Tuple[bool, Optional[str]]:
        """Acquire resources for a request."""
        # Check memory status
        memory_status = self.memory_tracker.check_memory_status()
        if memory_status["critical"]:
            return False, "Memory usage is critical"
        
        # Check user quota
        can_proceed, remaining = self.quota_manager.check_quota(user_id, "requests_per_minute")
        if not can_proceed:
            return False, "User quota exceeded"
        
        # Generate unique request ID
        import uuid
        request_id = f"req_{uuid.uuid4().hex[:16]}"
        
        # Consume quota first
        if not self.quota_manager.consume_quota(user_id, "requests_per_minute"):
            return False, "Failed to consume quota"
        
        # Check if we can accept the request immediately
        with self._request_lock:
            if self._active_requests < self.limits.max_concurrent_requests:
                # Can accept immediately
                self._active_requests += 1
                self._active_request_times[request_id] = datetime.utcnow()
                return True, request_id
            
        # Need to queue the request (don't increment active_requests for queued requests)
        try:
            await self.request_queue.enqueue(request_id, request_data)
            return True, request_id
        except asyncio.QueueFull:
            return False, "Request queue is full"
    
    async def release_resources(self, request_id: str, user_id: str):
        """Release resources after request completion."""
        with self._request_lock:
            if request_id in self._active_request_times:
                del self._active_request_times[request_id]
                self._active_requests = max(0, self._active_requests - 1)
    
    @asynccontextmanager
    async def acquire_for_request(self, user_id: str, request_data: Any):
        """Context manager for resource acquisition."""
        success, request_id = await self.acquire_resources(user_id, request_data)
        if not success:
            raise Exception(f"Resource acquisition failed: {request_id}")
        
        try:
            yield request_id
        finally:
            await self.release_resources(request_id, user_id)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics."""
        memory_status = self.memory_tracker.check_memory_status()
        queue_stats = self.request_queue.get_stats()
        
        with self._request_lock:
            request_stats = {
                "active": self._active_requests,
                "max_concurrent": self.limits.max_concurrent_requests,
                "utilization": self._active_requests / self.limits.max_concurrent_requests
            }
        
        return {
            "memory": {
                "current_mb": memory_status["current_mb"],
                "peak_mb": self.memory_tracker.get_peak_memory(),
                "status": memory_status
            },
            "requests": request_stats,
            "queue": queue_stats
        }
    
    async def cleanup_expired_requests(self) -> int:
        """Clean up expired requests."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.limits.max_request_duration_seconds)
        expired_requests = []
        
        with self._request_lock:
            for request_id, start_time in list(self._active_request_times.items()):
                if start_time < cutoff_time:
                    expired_requests.append(request_id)
            
            for request_id in expired_requests:
                del self._active_request_times[request_id]
                self._active_requests = max(0, self._active_requests - 1)
        
        return len(expired_requests)
    
    def update_limits(self, new_limits: ResourceLimits):
        """Update resource limits."""
        self.limits = new_limits
        # Update dependent components
        self.memory_tracker.warning_threshold_mb = int(new_limits.max_memory_mb * 0.8)
        self.memory_tracker.critical_threshold_mb = new_limits.max_memory_mb
    
    async def start(self):
        """Start all resource management components."""
        await self.memory_tracker.start_monitoring()
        await self.request_queue.start()
        await self.quota_manager.start_monitoring()
        
        logger.info("Resource manager started")
    
    async def stop(self):
        """Stop all resource management components."""
        await self.memory_tracker.stop_monitoring()
        await self.request_queue.stop()
        await self.quota_manager.stop_monitoring()
        
        logger.info("Resource manager stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system resource status."""
        return {
            'memory': self.memory_tracker.get_memory_stats(),
            'queue': self.request_queue.get_stats(),
            'connections': self.connection_limiter.get_connection_stats(),
            'quotas': self.quota_manager.get_quota_status(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @asynccontextmanager
    async def managed_request(self, request_id: str, priority: QueuePriority = QueuePriority.NORMAL):
        """Context manager for handling a managed request."""
        async with self.connection_limiter.acquire_connection(request_id):
            try:
                yield
            except Exception as e:
                logger.error(f"Error in managed request {request_id}: {e}")
                raise
    
    async def enqueue_request(self, request: QueuedRequest) -> bool:
        """Enqueue a request for processing."""
        return await self.request_queue.enqueue(request)
    
    async def process_next_request(self, timeout: Optional[float] = None) -> Optional[QueuedRequest]:
        """Process the next request from the queue."""
        return await self.request_queue.dequeue(timeout)


# Global resource manager
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager

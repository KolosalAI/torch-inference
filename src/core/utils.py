"""
Common utilities for the application.
"""

import os
import sys
import time
import hashlib
import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from datetime import datetime
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """Get hash of a file."""
    file_path = Path(file_path)
    hash_func = getattr(hashlib, algorithm)()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    import platform
    import psutil
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "disk_total_gb": psutil.disk_usage('/').total / (1024**3) if os.name != 'nt' else psutil.disk_usage('C:\\').total / (1024**3),
        "disk_free_gb": psutil.disk_usage('/').free / (1024**3) if os.name != 'nt' else psutil.disk_usage('C:\\').free / (1024**3),
    }


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation", logger_instance: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger_instance or logger
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        elapsed = self.elapsed_time
        self.logger.info(f"{self.name} completed in {elapsed:.3f}s")
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.perf_counter()
        return end_time - self.start_time


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retrying async functions."""
    def decorator(func: Callable[..., Awaitable[Any]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed: {e}")
            
            raise last_exception
        
        return wrapper
    return decorator


def retry_sync(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retrying sync functions."""
    def decorator(func: Callable[..., Any]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed: {e}")
            
            raise last_exception
        
        return wrapper
    return decorator


class ThreadSafeCounter:
    """Thread-safe counter."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, step: int = 1) -> int:
        """Increment counter and return new value."""
        with self._lock:
            self._value += step
            return self._value
    
    def decrement(self, step: int = 1) -> int:
        """Decrement counter and return new value."""
        with self._lock:
            self._value -= step
            return self._value
    
    @property
    def value(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value
    
    def reset(self, new_value: int = 0) -> int:
        """Reset counter to new value."""
        with self._lock:
            self._value = new_value
            return self._value


class AsyncBatch:
    """Utility for batching async operations."""
    
    def __init__(self, batch_size: int = 10, timeout: float = 1.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self._items = []
        self._results = []
        self._lock = asyncio.Lock()
    
    async def add(self, item: Any) -> Optional[List[Any]]:
        """Add item to batch. Returns batch if full, None otherwise."""
        async with self._lock:
            self._items.append(item)
            
            if len(self._items) >= self.batch_size:
                batch = self._items.copy()
                self._items.clear()
                return batch
            
            return None
    
    async def flush(self) -> List[Any]:
        """Get all remaining items in batch."""
        async with self._lock:
            batch = self._items.copy()
            self._items.clear()
            return batch


def safe_cast(value: Any, target_type: type, default: Any = None) -> Any:
    """Safely cast value to target type with default fallback."""
    try:
        return target_type(value)
    except (ValueError, TypeError, AttributeError):
        return default


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to max length with suffix."""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def get_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")


def validate_model_name(name: str) -> bool:
    """Validate model name format."""
    if not name or not isinstance(name, str):
        return False
    
    # Check length
    if len(name) < 1 or len(name) > 100:
        return False
    
    # Check allowed characters (alphanumeric, underscore, hyphen, dot)
    import re
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', name):
        return False
    
    return True


class PerformanceMonitor:
    """Monitor performance metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def record(self, metric_name: str, value: float):
        """Record a metric value."""
        with self._lock:
            if metric_name not in self._metrics:
                self._metrics[metric_name] = []
            
            self._metrics[metric_name].append(value)
            
            # Keep only recent values (last 1000)
            if len(self._metrics[metric_name]) > 1000:
                self._metrics[metric_name] = self._metrics[metric_name][-1000:]
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self._lock:
            values = self._metrics.get(metric_name, [])
            
            if not values:
                return {}
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1] if values else 0.0
            }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all metrics statistics."""
        return {name: self.get_stats(name) for name in self._metrics.keys()}

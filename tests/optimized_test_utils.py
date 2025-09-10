"""
Optimized test utilities for faster test execution and better resource management.
"""

import os
import time
import threading
import psutil
import torch
import gc
from functools import wraps
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from unittest.mock import Mock
import pytest


class TestResourceManager:
    """Centralized resource management for tests."""
    
    def __init__(self):
        self._cuda_initialized = False
        self._thread_pool = None
        self._memory_baseline = None
        
    def setup_test_session(self):
        """Setup resources once per test session."""
        # Pre-initialize CUDA if available to avoid per-test overhead
        if torch.cuda.is_available() and not self._cuda_initialized:
            torch.cuda.init()
            torch.cuda.empty_cache()
            self._cuda_initialized = True
            
        # Record memory baseline
        self._memory_baseline = psutil.Process().memory_info().rss
    
    def cleanup_light(self):
        """Light cleanup between tests - only essential cleanup."""
        # Only clean CUDA cache if memory usage is high
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            if current_memory > 100 * 1024 * 1024:  # > 100MB
                torch.cuda.empty_cache()
    
    def cleanup_heavy(self):
        """Heavy cleanup - run periodically or on failures."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Check for memory leaks
        if self._memory_baseline:
            current_memory = psutil.Process().memory_info().rss
            memory_growth = current_memory - self._memory_baseline
            if memory_growth > 500 * 1024 * 1024:  # > 500MB growth
                print(f"⚠️ Memory growth detected: {memory_growth / 1024 / 1024:.1f}MB")


# Global resource manager
resource_manager = TestResourceManager()


class FastMockFactory:
    """Factory for creating optimized mocks with cached configurations."""
    
    _mock_cache = {}
    
    @classmethod
    def get_mock_model(cls, model_type: str = "classification"):
        """Get cached mock model."""
        if model_type not in cls._mock_cache:
            mock = Mock()
            mock.eval.return_value = mock
            mock.to.return_value = mock
            mock.cuda.return_value = mock
            mock.cpu.return_value = mock
            mock.parameters.return_value = [torch.randn(10, 10)]
            
            def mock_forward(x):
                if isinstance(x, torch.Tensor):
                    if model_type == "classification":
                        return torch.randn(x.shape[0], 10)
                    elif model_type == "detection":
                        return torch.randn(x.shape[0], 5, 6)  # bbox format
                    else:
                        return torch.randn_like(x)
                return torch.randn(1, 10)
            
            mock.side_effect = mock_forward
            mock.__call__ = mock_forward
            cls._mock_cache[model_type] = mock
            
        return cls._mock_cache[model_type]
    
    @classmethod
    def get_mock_config(cls, config_type: str = "default"):
        """Get cached mock configuration."""
        cache_key = f"config_{config_type}"
        if cache_key not in cls._mock_cache:
            from framework.core.config import InferenceConfig, DeviceConfig, BatchConfig
            
            config = InferenceConfig(
                device=DeviceConfig(device_type="cpu", use_fp16=False),
                batch=BatchConfig(batch_size=1, max_batch_size=4)
            )
            cls._mock_cache[cache_key] = config
            
        return cls._mock_cache[cache_key]


def fast_test(category: str = None):
    """Decorator for fast tests that skip heavy setup."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Mark start time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Light cleanup only for fast tests
                resource_manager.cleanup_light()
                
                # Warn if test takes too long for "fast" category
                duration = time.time() - start_time
                if category == "fast" and duration > 1.0:
                    print(f"⚠️ Fast test {func.__name__} took {duration:.2f}s")
        
        # Add pytest marker
        wrapper = pytest.mark.unit(wrapper)
        return wrapper
    return decorator


def performance_test(timeout: int = 60):
    """Decorator for performance tests with extended timeouts."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Heavy cleanup before performance tests
            resource_manager.cleanup_heavy()
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if duration > timeout:
                    print(f"⚠️ Performance test {func.__name__} exceeded timeout: {duration:.2f}s > {timeout}s")
        
        # Add pytest markers
        wrapper = pytest.mark.performance(wrapper)
        wrapper = pytest.mark.timeout(timeout)(wrapper)
        return wrapper
    return decorator


class TestDataGenerator:
    """Optimized test data generation with caching."""
    
    _tensor_cache = {}
    
    @classmethod
    def get_tensor(cls, shape: tuple, dtype: torch.dtype = torch.float32, device: str = "cpu"):
        """Get cached tensor or create new one."""
        cache_key = (shape, dtype, device)
        
        if cache_key not in cls._tensor_cache:
            tensor = torch.randn(shape, dtype=dtype, device=device)
            # Cache small tensors only to avoid memory issues
            if tensor.numel() < 10000:  # < 10K elements
                cls._tensor_cache[cache_key] = tensor
            return tensor
        
        return cls._tensor_cache[cache_key].clone()
    
    @classmethod
    def clear_cache(cls):
        """Clear tensor cache to free memory."""
        cls._tensor_cache.clear()


@contextmanager
def isolated_test_environment():
    """Context manager for isolated test environment."""
    # Save current state
    original_threads = torch.get_num_threads()
    original_env = os.environ.copy()
    
    try:
        # Setup isolated environment
        os.environ.update({
            "ENVIRONMENT": "test",
            "LOG_LEVEL": "ERROR",  # Reduce logging in tests
            "DISABLE_PROFILING": "true"
        })
        
        yield
        
    finally:
        # Restore state
        torch.set_num_threads(original_threads)
        os.environ.clear()
        os.environ.update(original_env)


class BenchmarkTimer:
    """High-precision timer for benchmarking."""
    
    def __init__(self, warmup_iterations: int = 3):
        self.warmup_iterations = warmup_iterations
        self.times = []
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        pass
    
    def time_function(self, func, *args, **kwargs):
        """Time a function with warmup."""
        # Warmup
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        self.times.append(duration)
        return result, duration
    
    @property
    def average_time(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0
    
    @property
    def min_time(self) -> float:
        return min(self.times) if self.times else 0.0


def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )


def skip_if_slow_environment():
    """Skip test in slow CI environments."""
    return pytest.mark.skipif(
        os.getenv("CI") == "true" and os.getenv("SLOW_TESTS") != "true",
        reason="Skipping slow test in CI"
    )


class ParallelTestRunner:
    """Utilities for parallel test execution."""
    
    @staticmethod
    def can_run_parallel() -> bool:
        """Check if parallel tests can be run safely."""
        # Don't run parallel tests if GPU memory is limited
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            return total_memory > 4 * 1024**3  # > 4GB
        return True
    
    @staticmethod
    def get_optimal_worker_count() -> int:
        """Get optimal number of test workers."""
        cpu_count = os.cpu_count() or 4
        
        # Limit based on available memory
        available_memory = psutil.virtual_memory().available
        memory_limited_workers = available_memory // (2 * 1024**3)  # 2GB per worker
        
        return min(cpu_count, memory_limited_workers, 8)  # Max 8 workers


# Test fixtures for optimized testing

@pytest.fixture(scope="session")
def test_session_setup():
    """Session-level setup for optimal test execution."""
    resource_manager.setup_test_session()
    yield
    resource_manager.cleanup_heavy()


@pytest.fixture(scope="function")
def fast_cleanup():
    """Light cleanup between tests."""
    yield
    resource_manager.cleanup_light()


@pytest.fixture
def benchmark_timer():
    """Provide benchmark timer for performance tests."""
    return BenchmarkTimer()


@pytest.fixture
def mock_factory():
    """Provide fast mock factory."""
    return FastMockFactory


@pytest.fixture
def test_data_generator():
    """Provide optimized test data generator."""
    return TestDataGenerator


# Utility functions for test optimization

def assert_performance_improvement(baseline_time: float, optimized_time: float, 
                                 min_improvement: float = 0.1):
    """Assert that optimization provides minimum improvement."""
    improvement = (baseline_time - optimized_time) / baseline_time
    assert improvement >= min_improvement, (
        f"Insufficient performance improvement: {improvement:.2%} < {min_improvement:.2%}"
    )


def measure_memory_usage(func, *args, **kwargs):
    """Measure memory usage of a function."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        result = func(*args, **kwargs)
        
        peak_memory = torch.cuda.max_memory_allocated()
        return result, peak_memory
    else:
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        result = func(*args, **kwargs)
        
        final_memory = process.memory_info().rss
        return result, final_memory - initial_memory


def create_minimal_test_config():
    """Create minimal configuration for fast tests."""
    from framework.core.config import InferenceConfig, DeviceConfig, BatchConfig
    
    return InferenceConfig(
        device=DeviceConfig(
            device_type="cpu",
            use_fp16=False,
            enable_optimizations=False  # Disable for fast tests
        ),
        batch=BatchConfig(
            batch_size=1,
            max_batch_size=2  # Keep small for fast tests
        )
    )


# Test categorization helpers

def is_integration_test(test_path: str) -> bool:
    """Check if test is integration test."""
    return "integration" in test_path or "end_to_end" in test_path


def is_performance_test(test_path: str) -> bool:
    """Check if test is performance test."""
    return "performance" in test_path or "benchmark" in test_path


def get_test_category(test_path: str) -> str:
    """Get test category based on path."""
    if "unit" in test_path:
        return "unit"
    elif "integration" in test_path:
        return "integration"
    elif "performance" in test_path:
        return "performance"
    else:
        return "other"


# Configuration for different test environments

TEST_CONFIGS = {
    "fast": {
        "timeout": 10,
        "markers": ["unit", "fast"],
        "maxfail": 1,
        "disable_warnings": True
    },
    "standard": {
        "timeout": 30,
        "markers": ["unit", "integration"],
        "maxfail": 3,
        "disable_warnings": True
    },
    "comprehensive": {
        "timeout": 60,
        "markers": ["unit", "integration", "performance"],
        "maxfail": 5,
        "disable_warnings": False
    }
}


def get_pytest_args(config_name: str = "standard") -> List[str]:
    """Get pytest arguments for specific configuration."""
    config = TEST_CONFIGS.get(config_name, TEST_CONFIGS["standard"])
    
    args = [
        f"--timeout={config['timeout']}",
        "--timeout-method=thread",
        f"--maxfail={config['maxfail']}",
        "--tb=short",
        "-v"
    ]
    
    if config["disable_warnings"]:
        args.append("--disable-warnings")
    
    # Add marker filters
    if config["markers"]:
        marker_expr = " or ".join(config["markers"])
        args.extend(["-m", marker_expr])
    
    return args

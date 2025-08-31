"""
JIT Integration Module for PyTorch Inference Framework

This module provides seamless integration of Numba JIT compilation throughout the 
inference framework, optimizing computational performance without changing existing
class names or code structure.
"""

import logging
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Callable, Union
from functools import wraps
import time

# Import Numba optimizer
try:
    from ..optimizers.numba_optimizer import NumbaOptimizer
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)


class JITIntegrationManager:
    """
    Manager for JIT integration across the inference framework.
    
    This class provides a centralized way to enable/disable and manage
    Numba JIT optimizations throughout the framework without modifying
    existing class structures.
    """
    
    def __init__(self, enable_jit: bool = True):
        self.enable_jit = enable_jit and NUMBA_AVAILABLE
        self.numba_optimizer = None
        self.optimized_functions = {}
        self._performance_stats = {}
        
        if self.enable_jit:
            try:
                self.numba_optimizer = NumbaOptimizer()
                if self.numba_optimizer.is_available():
                    self._setup_optimized_functions()
                    logger.info("JIT Integration Manager initialized successfully")
                else:
                    self.enable_jit = False
                    logger.warning("Numba not functional, disabling JIT")
            except Exception as e:
                self.enable_jit = False
                logger.warning(f"Failed to initialize JIT manager: {e}")
    
    def _setup_optimized_functions(self):
        """Setup commonly used optimized functions."""
        try:
            self.optimized_functions = self.numba_optimizer.create_optimized_operations()
            logger.debug(f"Setup {len(self.optimized_functions)} optimized functions")
        except Exception as e:
            logger.debug(f"Failed to setup optimized functions: {e}")
    
    def apply_jit_optimization(self, func: Callable, target: str = "cpu", 
                              parallel: bool = True) -> Callable:
        """
        Apply JIT optimization to a function.
        
        Args:
            func: Function to optimize
            target: Target platform ("cpu", "cuda", "parallel")
            parallel: Enable parallel execution
            
        Returns:
            Optimized function or original if JIT is disabled
        """
        if not self.enable_jit:
            return func
        
        try:
            optimized = self.numba_optimizer.optimize_function(
                func, target=target, parallel=parallel
            )
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return optimized(*args, **kwargs)
                except Exception as e:
                    logger.debug(f"JIT function failed, falling back: {e}")
                    return func(*args, **kwargs)
            
            return wrapper
        except Exception as e:
            logger.debug(f"Failed to optimize function: {e}")
            return func
    
    def optimize_tensor_operation(self, tensor: torch.Tensor, 
                                 operation: str = "relu") -> torch.Tensor:
        """
        Apply JIT optimization to tensor operations.
        
        Args:
            tensor: Input tensor
            operation: Operation type
            
        Returns:
            Optimized tensor result
        """
        if not self.enable_jit or not self.numba_optimizer:
            return tensor
        
        try:
            return self.numba_optimizer.optimize_tensor_operation(tensor, operation)
        except Exception as e:
            logger.debug(f"Tensor optimization failed: {e}")
            return tensor
    
    def optimize_numpy_array(self, array: np.ndarray, 
                            operation: str = "relu") -> np.ndarray:
        """
        Apply JIT optimization to numpy array operations.
        
        Args:
            array: Input array
            operation: Operation type
            
        Returns:
            Optimized array result
        """
        if not self.enable_jit or operation not in self.optimized_functions:
            return array
        
        try:
            return self.optimized_functions[operation](array)
        except Exception as e:
            logger.debug(f"Array optimization failed: {e}")
            return array
    
    def benchmark_optimization(self, array_size: tuple = (1000, 1000), 
                              operation: str = "relu", iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark JIT optimization performance.
        
        Args:
            array_size: Size of test array
            operation: Operation to benchmark
            iterations: Number of iterations
            
        Returns:
            Performance statistics
        """
        if not self.enable_jit:
            return {"error": "JIT not available"}
        
        try:
            return self.numba_optimizer.benchmark_numba_performance(
                array_size, operation, iterations
            )
        except Exception as e:
            return {"error": str(e)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get JIT performance statistics."""
        if not self.enable_jit:
            return {"jit_enabled": False}
        
        stats = {
            "jit_enabled": True,
            "numba_available": NUMBA_AVAILABLE,
            "cuda_available": self.numba_optimizer.is_cuda_available() if self.numba_optimizer else False,
            "optimized_functions": len(self.optimized_functions),
        }
        
        if self.numba_optimizer:
            stats.update(self.numba_optimizer.get_optimization_stats())
        
        return stats
    
    def is_available(self) -> bool:
        """Check if JIT optimization is available."""
        return self.enable_jit and self.numba_optimizer is not None
    
    def disable_jit(self):
        """Disable JIT optimization."""
        self.enable_jit = False
        logger.info("JIT optimization disabled")
    
    def enable_jit_if_available(self):
        """Enable JIT optimization if available."""
        if NUMBA_AVAILABLE and self.numba_optimizer:
            self.enable_jit = True
            logger.info("JIT optimization enabled")
        else:
            logger.warning("Cannot enable JIT - Numba not available")


# Global JIT integration manager instance
_jit_manager = None

def get_jit_manager(enable_jit: bool = True) -> JITIntegrationManager:
    """Get the global JIT integration manager."""
    global _jit_manager
    if _jit_manager is None:
        _jit_manager = JITIntegrationManager(enable_jit)
    return _jit_manager


def jit_optimize(target: str = "cpu", parallel: bool = True):
    """
    Decorator for applying JIT optimization to functions.
    
    Args:
        target: Target platform ("cpu", "cuda", "parallel")
        parallel: Enable parallel execution
    """
    def decorator(func):
        manager = get_jit_manager()
        return manager.apply_jit_optimization(func, target, parallel)
    return decorator


def apply_tensor_jit(tensor: torch.Tensor, operation: str = "relu") -> torch.Tensor:
    """Apply JIT optimization to tensor operations."""
    manager = get_jit_manager()
    return manager.optimize_tensor_operation(tensor, operation)


def apply_array_jit(array: np.ndarray, operation: str = "relu") -> np.ndarray:
    """Apply JIT optimization to numpy array operations."""
    manager = get_jit_manager()
    return manager.optimize_numpy_array(array, operation)


# Optimized mathematical functions with JIT acceleration
@jit_optimize(target="parallel", parallel=True)
def fast_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast matrix multiplication with JIT acceleration."""
    return np.dot(a, b)


@jit_optimize(target="parallel", parallel=True)
def fast_element_wise_ops(a: np.ndarray, b: np.ndarray, operation: str = "add") -> np.ndarray:
    """Fast element-wise operations with JIT acceleration."""
    if operation == "add":
        return a + b
    elif operation == "mul":
        return a * b
    elif operation == "sub":
        return a - b
    elif operation == "div":
        return a / b
    else:
        return a + b  # Default to addition


@jit_optimize(target="parallel", parallel=True)
def fast_activation_functions(x: np.ndarray, activation: str = "relu") -> np.ndarray:
    """Fast activation functions with JIT acceleration."""
    if activation == "relu":
        return np.maximum(0, x)
    elif activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    elif activation == "tanh":
        return np.tanh(x)
    else:
        return np.maximum(0, x)  # Default to ReLU


def initialize_jit_integration(enable_jit: bool = True) -> JITIntegrationManager:
    """
    Initialize JIT integration for the entire framework.
    
    Args:
        enable_jit: Whether to enable JIT optimization
        
    Returns:
        JIT integration manager instance
    """
    manager = get_jit_manager(enable_jit)
    
    if manager.is_available():
        logger.info("JIT integration initialized successfully")
        logger.info(f"Performance stats: {manager.get_performance_stats()}")
    else:
        logger.warning("JIT integration not available")
    
    return manager


# Export public interface
__all__ = [
    'JITIntegrationManager',
    'get_jit_manager',
    'jit_optimize',
    'apply_tensor_jit',
    'apply_array_jit',
    'fast_matrix_multiply',
    'fast_element_wise_ops',
    'fast_activation_functions',
    'initialize_jit_integration',
]

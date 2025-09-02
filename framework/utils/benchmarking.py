"""
ROI Optimization 8: Measurement discipline utilities for accurate PyTorch inference benchmarking.

This module provides proper timing utilities that avoid self-deception in performance measurements.
"""

import time
import torch
import logging
import statistics
from typing import List, Dict, Callable, Any, Optional, Tuple
from contextlib import contextmanager
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    mean_time: float
    median_time: float
    p95_time: float
    p99_time: float
    min_time: float
    max_time: float
    std_dev: float
    num_iterations: int
    total_time: float
    throughput: float  # iterations per second


class PytorchBenchmarker:
    """
    ROI Optimization 8: Proper PyTorch benchmarking with measurement discipline.
    
    Implements best practices:
    - CUDA synchronization only around measurement boundaries
    - Multiple iterations with warmup
    - Statistical analysis (p50/p95/p99, not just best time)
    - Proper warmup handling
    """
    
    def __init__(self, device: Optional[torch.device] = None, warmup_iterations: int = 5):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.warmup_iterations = warmup_iterations
        
    def benchmark_function(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        num_iterations: int = 100,
        description: str = "Function"
    ) -> BenchmarkResult:
        """
        Benchmark a function with proper measurement discipline.
        
        Args:
            func: Function to benchmark
            args: Function arguments
            kwargs: Function keyword arguments
            num_iterations: Number of timing iterations (after warmup)
            description: Description for logging
            
        Returns:
            BenchmarkResult with statistical analysis
        """
        if kwargs is None:
            kwargs = {}
            
        logger.info(f"Benchmarking {description} with {num_iterations} iterations...")
        
        # Warmup phase
        logger.debug(f"Running {self.warmup_iterations} warmup iterations...")
        for _ in range(self.warmup_iterations):
            try:
                _ = func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Warmup iteration failed: {e}")
                continue
        
        # ROI Optimization 8: CUDA synchronization before measurement
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timing phase
        times = []
        total_start = time.perf_counter()
        
        for i in range(num_iterations):
            # ROI Optimization 8: Sync boundaries around measurement
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Iteration {i} failed: {e}")
                continue
            
            # ROI Optimization 8: Sync after computation completes
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        total_end = time.perf_counter()
        total_time = total_end - total_start
        
        if not times:
            raise RuntimeError("No successful iterations completed")
        
        # Statistical analysis
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        
        # Percentiles
        sorted_times = sorted(times)
        p95_idx = int(0.95 * len(sorted_times))
        p99_idx = int(0.99 * len(sorted_times))
        p95_time = sorted_times[min(p95_idx, len(sorted_times) - 1)]
        p99_time = sorted_times[min(p99_idx, len(sorted_times) - 1)]
        
        throughput = len(times) / total_time
        
        result = BenchmarkResult(
            mean_time=mean_time,
            median_time=median_time,
            p95_time=p95_time,
            p99_time=p99_time,
            min_time=min_time,
            max_time=max_time,
            std_dev=std_dev,
            num_iterations=len(times),
            total_time=total_time,
            throughput=throughput
        )
        
        logger.info(f"{description} benchmark results:")
        logger.info(f"  Mean: {mean_time*1000:.2f}ms, Median: {median_time*1000:.2f}ms")
        logger.info(f"  P95: {p95_time*1000:.2f}ms, P99: {p99_time*1000:.2f}ms")
        logger.info(f"  Min: {min_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")
        logger.info(f"  Throughput: {throughput:.1f} ops/sec")
        
        return result
    
    def benchmark_model_inference(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        num_iterations: int = 100,
        use_mixed_precision: bool = False,
        use_compile: bool = False
    ) -> BenchmarkResult:
        """
        Benchmark model inference with various optimization settings.
        
        Args:
            model: PyTorch model to benchmark
            input_tensor: Input tensor for inference
            num_iterations: Number of timing iterations
            use_mixed_precision: Whether to use mixed precision
            use_compile: Whether to compile the model
            
        Returns:
            BenchmarkResult with inference timing statistics
        """
        model.eval()
        model = model.to(self.device)
        input_tensor = input_tensor.to(self.device)
        
        # Apply optimizations
        if use_compile and hasattr(torch, 'compile'):
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("Model compiled for benchmarking")
        
        def inference_func():
            with torch.inference_mode():
                if use_mixed_precision and self.device.type == 'cuda':
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    with torch.amp.autocast('cuda', dtype=dtype):
                        return model(input_tensor)
                else:
                    return model(input_tensor)
        
        description = f"Model inference ({'mixed precision' if use_mixed_precision else 'fp32'}, {'compiled' if use_compile else 'standard'})"
        return self.benchmark_function(inference_func, description=description, num_iterations=num_iterations)
    
    @contextmanager
    def timer(self, description: str = "Operation"):
        """
        Context manager for timing individual operations with proper sync.
        
        Example:
            with benchmarker.timer("My operation"):
                result = my_function()
        """
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        yield
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        logger.info(f"{description}: {elapsed*1000:.2f}ms")


def compare_optimization_impact(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
    num_iterations: int = 50
) -> Dict[str, BenchmarkResult]:
    """
    Compare the impact of different optimizations on model performance.
    
    Args:
        model: Model to benchmark
        input_tensor: Input for inference
        device: Target device
        num_iterations: Iterations per benchmark
        
    Returns:
        Dictionary of benchmark results for each configuration
    """
    benchmarker = PytorchBenchmarker(device=device)
    results = {}
    
    # Baseline
    baseline_model = model.__class__()
    baseline_model.load_state_dict(model.state_dict())
    results['baseline'] = benchmarker.benchmark_model_inference(
        baseline_model, input_tensor, num_iterations, 
        use_mixed_precision=False, use_compile=False
    )
    
    # Mixed precision only
    mp_model = model.__class__()
    mp_model.load_state_dict(model.state_dict())
    results['mixed_precision'] = benchmarker.benchmark_model_inference(
        mp_model, input_tensor, num_iterations,
        use_mixed_precision=True, use_compile=False
    )
    
    # Compiled only
    compiled_model = model.__class__()
    compiled_model.load_state_dict(model.state_dict())
    results['compiled'] = benchmarker.benchmark_model_inference(
        compiled_model, input_tensor, num_iterations,
        use_mixed_precision=False, use_compile=True
    )
    
    # Both optimizations
    both_model = model.__class__()
    both_model.load_state_dict(model.state_dict())
    results['both_optimizations'] = benchmarker.benchmark_model_inference(
        both_model, input_tensor, num_iterations,
        use_mixed_precision=True, use_compile=True
    )
    
    # Log comparison
    baseline_time = results['baseline'].median_time
    logger.info("\nOptimization Impact Analysis:")
    for config, result in results.items():
        if config != 'baseline':
            speedup = baseline_time / result.median_time
            logger.info(f"  {config}: {speedup:.2f}x speedup ({result.median_time*1000:.2f}ms)")
    
    return results


def validate_optimization_correctness(
    model_func: Callable,
    input_data: torch.Tensor,
    baseline_output: Optional[torch.Tensor] = None,
    tolerance: float = 1e-4
) -> bool:
    """
    Validate that optimizations don't change model outputs significantly.
    
    Args:
        model_func: Function that returns model output
        input_data: Input tensor
        baseline_output: Expected output (computed if None)
        tolerance: Maximum allowed difference
        
    Returns:
        True if outputs match within tolerance
    """
    try:
        optimized_output = model_func()
        
        if baseline_output is None:
            logger.warning("No baseline output provided for correctness validation")
            return True
        
        if not torch.allclose(optimized_output, baseline_output, rtol=tolerance, atol=tolerance):
            max_diff = torch.max(torch.abs(optimized_output - baseline_output)).item()
            logger.warning(f"Optimization changed outputs! Max difference: {max_diff}")
            return False
        
        logger.debug("Optimization preserves output correctness")
        return True
        
    except Exception as e:
        logger.error(f"Correctness validation failed: {e}")
        return False

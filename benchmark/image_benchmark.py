"""
Image Model Benchmark Core Module.

Provides the main ImageBenchmarker class for running comprehensive image model benchmarks
with proper timing, concurrency control, and metric collection.
"""

import asyncio
import time
import logging
import threading
import queue
from typing import List, Dict, Optional, Callable, Any, Union, Coroutine
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

from .image_metrics import ImageRequestMetrics, ImageMetrics, aggregate_image_metrics

logger = logging.getLogger(__name__)


@dataclass
class ImageBenchmarkResult:
    """Results from a complete image model benchmark run."""
    metrics: ImageMetrics
    request_metrics: List[ImageRequestMetrics]
    config: Dict[str, Any] = field(default_factory=dict)
    
    def get_metric_summary(self) -> Dict[str, float]:
        """Get a summary of key metrics for easy comparison."""
        return {
            'ips': self.metrics.ips,
            'pps': self.metrics.pps,
            'sps': self.metrics.sps,
            'rps': self.metrics.rps,
            'ttfi_p50_ms': self.metrics.ttfi_p50 * 1000,
            'ttfi_p95_ms': self.metrics.ttfi_p95 * 1000,
            'success_rate': self.metrics.success_rate,
            'concurrency': self.metrics.concurrency_level,
            'avg_memory_mb': self.metrics.avg_memory_peak_mb
        }


class ImageBenchmarker:
    """
    Comprehensive image model benchmark runner implementing industry-standard metrics.
    
    Supports both synchronous and asynchronous image generation function benchmarking
    with proper concurrency control and detailed metric collection.
    """
    
    def __init__(
        self,
        default_width: int = 512,
        default_height: int = 512,
        default_steps: int = 20,
        warmup_requests: int = 3,
        timeout_seconds: float = 300.0,  # 5 minutes default for image generation
        monitor_memory: bool = True
    ):
        """
        Initialize image model benchmarker.
        
        Args:
            default_width: Default image width for consistency checking
            default_height: Default image height for consistency checking
            default_steps: Default number of inference steps
            warmup_requests: Number of warmup requests before timing
            timeout_seconds: Timeout for individual requests
            monitor_memory: Whether to monitor memory usage
        """
        self.default_width = default_width
        self.default_height = default_height
        self.default_steps = default_steps
        self.warmup_requests = warmup_requests
        self.timeout_seconds = timeout_seconds
        self.monitor_memory = monitor_memory
        
    def benchmark_sync_image_model(
        self,
        image_function: Callable[[str], Dict[str, Any]],
        test_prompts: List[str],
        concurrency_levels: List[int] = [1, 2, 4, 8, 16],
        iterations_per_level: int = 20,
        **image_kwargs
    ) -> Dict[int, ImageBenchmarkResult]:
        """
        Benchmark a synchronous image generation function across multiple concurrency levels.
        
        Args:
            image_function: Function that takes prompt and returns image info
                           Expected return: {'images': [...], 'width': int, 'height': int, ...}
            test_prompts: List of test prompts to generate images from
            concurrency_levels: List of concurrency levels to test
            iterations_per_level: Number of iterations per concurrency level
            **image_kwargs: Additional keyword arguments for image function
            
        Returns:
            Dictionary mapping concurrency level to benchmark results
        """
        results = {}
        
        for concurrency in concurrency_levels:
            logger.info(f"Running image model benchmark at concurrency level {concurrency}")
            
            # Run warmup
            self._run_warmup_sync(image_function, test_prompts[:self.warmup_requests], **image_kwargs)
            
            # Run actual benchmark
            request_metrics = self._run_sync_benchmark(
                image_function, test_prompts, concurrency, iterations_per_level, **image_kwargs
            )
            
            # Aggregate metrics
            metrics = aggregate_image_metrics(request_metrics, concurrency)
            
            results[concurrency] = ImageBenchmarkResult(
                metrics=metrics,
                request_metrics=request_metrics,
                config={
                    'concurrency': concurrency,
                    'iterations': iterations_per_level,
                    'test_prompts_count': len(test_prompts),
                    'image_kwargs': image_kwargs
                }
            )
            
            logger.info(f"Concurrency {concurrency}: IPS={metrics.ips:.3f}, "
                       f"TTFI p95={metrics.ttfi_p95*1000:.1f}ms, RPS={metrics.rps:.1f}")
        
        return results
    
    def benchmark_single_concurrency(
        self,
        image_function: Callable[[str], Dict[str, Any]],
        test_prompts: List[str],
        concurrency: int,
        iterations: int,
        **image_kwargs
    ) -> ImageBenchmarkResult:
        """
        Benchmark a synchronous image generation function at a single concurrency level.
        
        Args:
            image_function: Function that takes prompt and returns image info
            test_prompts: List of test prompts to generate images from
            concurrency: Concurrency level to test
            iterations: Number of iterations to run
            **image_kwargs: Additional keyword arguments for image function
            
        Returns:
            Single ImageBenchmarkResult for the specified concurrency level
        """
        logger.info(f"Running image model benchmark at concurrency level {concurrency}")
        
        # Run warmup
        self._run_warmup_sync(image_function, test_prompts[:self.warmup_requests], **image_kwargs)
        
        # Run actual benchmark
        request_metrics = self._run_sync_benchmark(
            image_function, test_prompts, concurrency, iterations, **image_kwargs
        )
        
        # Aggregate metrics
        metrics = aggregate_image_metrics(request_metrics, concurrency)
        
        result = ImageBenchmarkResult(
            metrics=metrics,
            request_metrics=request_metrics,
            config={
                'concurrency': concurrency,
                'iterations': iterations,
                'test_prompts_count': len(test_prompts),
                'image_kwargs': image_kwargs
            }
        )
        
        logger.info(f"Concurrency {concurrency}: IPS={metrics.ips:.3f}, "
                   f"TTFI p95={metrics.ttfi_p95*1000:.1f}ms, RPS={metrics.rps:.1f}")
        
        return result
    
    async def benchmark_async_image_model(
        self,
        image_function: Callable[[str], Coroutine[Any, Any, Dict[str, Any]]],
        test_prompts: List[str],
        concurrency_levels: List[int] = [1, 2, 4, 8, 16],
        iterations_per_level: int = 20,
        **image_kwargs
    ) -> Dict[int, ImageBenchmarkResult]:
        """
        Benchmark an asynchronous image generation function across multiple concurrency levels.
        
        Args:
            image_function: Async function that takes prompt and returns image info
            test_prompts: List of test prompts to generate images from
            concurrency_levels: List of concurrency levels to test
            iterations_per_level: Number of iterations per concurrency level
            **image_kwargs: Additional keyword arguments for image function
            
        Returns:
            Dictionary mapping concurrency level to benchmark results
        """
        results = {}
        
        for concurrency in concurrency_levels:
            logger.info(f"Running async image model benchmark at concurrency level {concurrency}")
            
            # Run warmup
            await self._run_warmup_async(image_function, test_prompts[:self.warmup_requests], **image_kwargs)
            
            # Run actual benchmark
            request_metrics = await self._run_async_benchmark(
                image_function, test_prompts, concurrency, iterations_per_level, **image_kwargs
            )
            
            # Aggregate metrics
            metrics = aggregate_image_metrics(request_metrics, concurrency)
            
            results[concurrency] = ImageBenchmarkResult(
                metrics=metrics,
                request_metrics=request_metrics,
                config={
                    'concurrency': concurrency,
                    'iterations': iterations_per_level,
                    'test_prompts_count': len(test_prompts),
                    'image_kwargs': image_kwargs
                }
            )
            
            logger.info(f"Concurrency {concurrency}: IPS={metrics.ips:.3f}, "
                       f"TTFI p95={metrics.ttfi_p95*1000:.1f}ms, RPS={metrics.rps:.1f}")
        
        return results
    
    def _run_warmup_sync(
        self,
        image_function: Callable,
        warmup_prompts: List[str],
        **image_kwargs
    ) -> None:
        """Run warmup requests to initialize the image generation system."""
        if not warmup_prompts:
            return
            
        logger.debug(f"Running {len(warmup_prompts)} warmup requests")
        
        for i, prompt in enumerate(warmup_prompts):
            try:
                result = image_function(prompt, **image_kwargs)
                logger.debug(f"Warmup {i+1}/{len(warmup_prompts)} completed")
            except Exception as e:
                logger.warning(f"Warmup request {i+1} failed: {e}")
    
    async def _run_warmup_async(
        self,
        image_function: Callable,
        warmup_prompts: List[str],
        **image_kwargs
    ) -> None:
        """Run async warmup requests to initialize the image generation system."""
        if not warmup_prompts:
            return
            
        logger.debug(f"Running {len(warmup_prompts)} async warmup requests")
        
        for i, prompt in enumerate(warmup_prompts):
            try:
                result = await image_function(prompt, **image_kwargs)
                logger.debug(f"Warmup {i+1}/{len(warmup_prompts)} completed")
            except Exception as e:
                logger.warning(f"Warmup request {i+1} failed: {e}")
    
    def _run_sync_benchmark(
        self,
        image_function: Callable,
        test_prompts: List[str],
        concurrency: int,
        iterations: int,
        **image_kwargs
    ) -> List[ImageRequestMetrics]:
        """Run synchronous benchmark with specified concurrency."""
        request_metrics = []
        
        # Prepare requests (cycle through test prompts)
        requests = []
        for i in range(iterations):
            prompt = test_prompts[i % len(test_prompts)]
            requests.append((f"req_{i}", prompt))
        
        if concurrency == 1:
            # Sequential execution
            for req_id, prompt in requests:
                metrics = self._execute_sync_request(req_id, image_function, prompt, **image_kwargs)
                request_metrics.append(metrics)
        else:
            # Concurrent execution
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_req = {
                    executor.submit(self._execute_sync_request, req_id, image_function, prompt, **image_kwargs): req_id
                    for req_id, prompt in requests
                }
                
                for future in as_completed(future_to_req, timeout=self.timeout_seconds * len(requests)):
                    try:
                        metrics = future.result()
                        request_metrics.append(metrics)
                    except Exception as e:
                        req_id = future_to_req[future]
                        error_metrics = ImageRequestMetrics(
                            request_id=req_id,
                            t_start=time.perf_counter(),
                            error=str(e)
                        )
                        request_metrics.append(error_metrics)
        
        return request_metrics
    
    async def _run_async_benchmark(
        self,
        image_function: Callable,
        test_prompts: List[str],
        concurrency: int,
        iterations: int,
        **image_kwargs
    ) -> List[ImageRequestMetrics]:
        """Run asynchronous benchmark with specified concurrency."""
        request_metrics = []
        
        # Prepare requests (cycle through test prompts)
        requests = []
        for i in range(iterations):
            prompt = test_prompts[i % len(test_prompts)]
            requests.append((f"req_{i}", prompt))
        
        if concurrency == 1:
            # Sequential execution
            for req_id, prompt in requests:
                metrics = await self._execute_async_request(req_id, image_function, prompt, **image_kwargs)
                request_metrics.append(metrics)
        else:
            # Concurrent execution with semaphore
            semaphore = asyncio.Semaphore(concurrency)
            
            async def bounded_request(req_id: str, prompt: str):
                async with semaphore:
                    return await self._execute_async_request(req_id, image_function, prompt, **image_kwargs)
            
            tasks = [bounded_request(req_id, prompt) for req_id, prompt in requests]
            request_metrics = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_metrics = []
            for i, result in enumerate(request_metrics):
                if isinstance(result, Exception):
                    error_metrics = ImageRequestMetrics(
                        request_id=requests[i][0],
                        t_start=time.perf_counter(),
                        error=str(result)
                    )
                    processed_metrics.append(error_metrics)
                else:
                    processed_metrics.append(result)
            
            request_metrics = processed_metrics
        
        return request_metrics
    
    def _execute_sync_request(
        self,
        request_id: str,
        image_function: Callable,
        prompt: str,
        **image_kwargs
    ) -> ImageRequestMetrics:
        """Execute a single synchronous image generation request and collect metrics."""
        metrics = ImageRequestMetrics(
            request_id=request_id,
            t_start=time.perf_counter(),
            prompt_len_chars=len(prompt)
        )
        
        # Extract parameters from kwargs
        negative_prompt = image_kwargs.get('negative_prompt', '')
        metrics.negative_prompt_len_chars = len(negative_prompt) if negative_prompt else 0
        metrics.width = image_kwargs.get('width', self.default_width)
        metrics.height = image_kwargs.get('height', self.default_height)
        metrics.num_inference_steps = image_kwargs.get('num_inference_steps', self.default_steps)
        metrics.guidance_scale = image_kwargs.get('guidance_scale', 7.5)
        metrics.seed = image_kwargs.get('seed', None)
        
        try:
            # Monitor memory before request if enabled
            if self.monitor_memory:
                import psutil
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # For streaming image generation, we would set t_first_image when first preview arrives
            # For now, assume non-streaming image generation
            result = image_function(prompt, **image_kwargs)
            
            metrics.t_first_image = time.perf_counter()  # Approximate for non-streaming
            metrics.t_end = time.perf_counter()
            
            # Extract image information from result
            if isinstance(result, dict):
                # Handle different result formats
                images = result.get('images', [])
                if images:
                    metrics.num_images = len(images)
                    
                    # Try to get image dimensions
                    if 'width' in result and 'height' in result:
                        metrics.width = result['width']
                        metrics.height = result['height']
                    elif hasattr(images[0], 'size'):  # PIL Image
                        metrics.width, metrics.height = images[0].size
                    
                    # Estimate file size (rough approximation)
                    metrics.file_size_bytes = metrics.total_pixels * 3  # RGB estimation
                
                # Extract other metadata
                if 'num_inference_steps' in result:
                    metrics.num_inference_steps = result['num_inference_steps']
                if 'guidance_scale' in result:
                    metrics.guidance_scale = result['guidance_scale']
                if 'seed' in result:
                    metrics.seed = result['seed']
            else:
                logger.warning(f"Image function returned unexpected type: {type(result)}")
            
            # Monitor memory after request if enabled
            if self.monitor_memory:
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                metrics.memory_peak_mb = max(initial_memory, final_memory)
                
                # Try to get GPU memory if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        metrics.gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                except ImportError:
                    pass
                    
        except Exception as e:
            metrics.error = str(e)
            metrics.t_end = time.perf_counter()
            logger.error(f"Request {request_id} failed: {e}")
        
        return metrics
    
    async def _execute_async_request(
        self,
        request_id: str,
        image_function: Callable,
        prompt: str,
        **image_kwargs
    ) -> ImageRequestMetrics:
        """Execute a single asynchronous image generation request and collect metrics."""
        metrics = ImageRequestMetrics(
            request_id=request_id,
            t_start=time.perf_counter(),
            prompt_len_chars=len(prompt)
        )
        
        # Extract parameters from kwargs
        negative_prompt = image_kwargs.get('negative_prompt', '')
        metrics.negative_prompt_len_chars = len(negative_prompt) if negative_prompt else 0
        metrics.width = image_kwargs.get('width', self.default_width)
        metrics.height = image_kwargs.get('height', self.default_height)
        metrics.num_inference_steps = image_kwargs.get('num_inference_steps', self.default_steps)
        metrics.guidance_scale = image_kwargs.get('guidance_scale', 7.5)
        metrics.seed = image_kwargs.get('seed', None)
        
        try:
            # Monitor memory before request if enabled
            if self.monitor_memory:
                import psutil
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # For streaming image generation, we would set t_first_image when first preview arrives
            result = await image_function(prompt, **image_kwargs)
            
            metrics.t_first_image = time.perf_counter()  # Approximate for non-streaming
            metrics.t_end = time.perf_counter()
            
            # Extract image information from result
            if isinstance(result, dict):
                # Handle different result formats
                images = result.get('images', [])
                if images:
                    metrics.num_images = len(images)
                    
                    # Try to get image dimensions
                    if 'width' in result and 'height' in result:
                        metrics.width = result['width']
                        metrics.height = result['height']
                    elif hasattr(images[0], 'size'):  # PIL Image
                        metrics.width, metrics.height = images[0].size
                    
                    # Estimate file size (rough approximation)
                    metrics.file_size_bytes = metrics.total_pixels * 3  # RGB estimation
                
                # Extract other metadata
                if 'num_inference_steps' in result:
                    metrics.num_inference_steps = result['num_inference_steps']
                if 'guidance_scale' in result:
                    metrics.guidance_scale = result['guidance_scale']
                if 'seed' in result:
                    metrics.seed = result['seed']
            else:
                logger.warning(f"Image function returned unexpected type: {type(result)}")
            
            # Monitor memory after request if enabled
            if self.monitor_memory:
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                metrics.memory_peak_mb = max(initial_memory, final_memory)
                
                # Try to get GPU memory if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        metrics.gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                except ImportError:
                    pass
                    
        except Exception as e:
            metrics.error = str(e)
            metrics.t_end = time.perf_counter()
            logger.error(f"Request {request_id} failed: {e}")
        
        return metrics


def create_streaming_image_wrapper(
    streaming_image_function: Callable,
    preview_callback: Optional[Callable[[str, float], None]] = None
):
    """
    Create a wrapper for streaming image generation functions to properly capture TTFI.
    
    Args:
        streaming_image_function: Function that yields image previews/progress
        preview_callback: Optional callback called with (request_id, timestamp) for first preview
        
    Returns:
        Wrapped function that captures streaming metrics
    """
    def wrapper(request_id: str, prompt: str, **kwargs):
        previews = []
        first_preview_time = None
        
        for preview in streaming_image_function(prompt, **kwargs):
            if first_preview_time is None:
                first_preview_time = time.perf_counter()
                if preview_callback:
                    preview_callback(request_id, first_preview_time)
            previews.append(preview)
        
        # Return final result
        final_image = previews[-1] if previews else None
        
        return {
            'images': [final_image] if final_image else [],
            'previews': previews,
            'first_preview_time': first_preview_time
        }
    
    return wrapper

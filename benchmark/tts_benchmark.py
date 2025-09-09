"""
TTS Benchmark Core Module.

Provides the main TTSBenchmarker class for running comprehensive TTS benchmarks
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

from .metrics import TTSRequestMetrics, TTSMetrics, aggregate_tts_metrics

logger = logging.getLogger(__name__)


@dataclass
class TTSBenchmarkResult:
    """Results from a complete TTS benchmark run."""
    metrics: TTSMetrics
    request_metrics: List[TTSRequestMetrics]
    config: Dict[str, Any] = field(default_factory=dict)
    
    def get_metric_summary(self) -> Dict[str, float]:
        """Get a summary of key metrics for easy comparison."""
        return {
            'asps': self.metrics.asps,
            'rtf_median': self.metrics.rtf_median,
            'rps': self.metrics.rps,
            'cps': self.metrics.cps,
            'ttfa_p50_ms': self.metrics.ttfa_p50 * 1000,
            'ttfa_p95_ms': self.metrics.ttfa_p95 * 1000,
            'success_rate': self.metrics.success_rate,
            'concurrency': self.metrics.concurrency_level
        }


class TTSBenchmarker:
    """
    Comprehensive TTS benchmark runner implementing industry-standard metrics.
    
    Supports both synchronous and asynchronous TTS function benchmarking
    with proper concurrency control and detailed metric collection.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        bit_depth: int = 16,
        warmup_requests: int = 3,
        timeout_seconds: float = 30.0
    ):
        """
        Initialize TTS benchmarker.
        
        Args:
            sample_rate: Audio sample rate for consistency checking
            bit_depth: Audio bit depth for consistency checking
            warmup_requests: Number of warmup requests before timing
            timeout_seconds: Timeout for individual requests
        """
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.warmup_requests = warmup_requests
        self.timeout_seconds = timeout_seconds
        
    def benchmark_sync_tts(
        self,
        tts_function: Callable[[str], Dict[str, Any]],
        test_texts: List[str],
        concurrency_levels: List[int] = [1, 2, 4, 8, 16, 32, 64],
        iterations_per_level: int = 100,
        **tts_kwargs
    ) -> Dict[int, TTSBenchmarkResult]:
        """
        Benchmark a synchronous TTS function across multiple concurrency levels.
        
        Args:
            tts_function: Function that takes text and returns audio info
                         Expected return: {'audio_duration': float, 'sample_rate': int, ...}
            test_texts: List of test texts to synthesize
            concurrency_levels: List of concurrency levels to test
            iterations_per_level: Number of iterations per concurrency level
            **tts_kwargs: Additional keyword arguments for TTS function
            
        Returns:
            Dictionary mapping concurrency level to benchmark results
        """
        results = {}
        
        for concurrency in concurrency_levels:
            logger.info(f"Running TTS benchmark at concurrency level {concurrency}")
            
            # Run warmup
            self._run_warmup_sync(tts_function, test_texts[:self.warmup_requests], **tts_kwargs)
            
            # Run actual benchmark
            request_metrics = self._run_sync_benchmark(
                tts_function, test_texts, concurrency, iterations_per_level, **tts_kwargs
            )
            
            # Aggregate metrics
            metrics = aggregate_tts_metrics(request_metrics, concurrency)
            
            results[concurrency] = TTSBenchmarkResult(
                metrics=metrics,
                request_metrics=request_metrics,
                config={
                    'concurrency': concurrency,
                    'iterations': iterations_per_level,
                    'test_texts_count': len(test_texts),
                    'tts_kwargs': tts_kwargs
                }
            )
            
            logger.info(f"Concurrency {concurrency}: ASPS={metrics.asps:.3f}, "
                       f"RTF={metrics.rtf_median:.3f}, RPS={metrics.rps:.1f}")
        
        return results
    
    async def benchmark_async_tts(
        self,
        tts_function: Callable[[str], Coroutine[Any, Any, Dict[str, Any]]],
        test_texts: List[str],
        concurrency_levels: List[int] = [1, 2, 4, 8, 16, 32, 64],
        iterations_per_level: int = 100,
        **tts_kwargs
    ) -> Dict[int, TTSBenchmarkResult]:
        """
        Benchmark an asynchronous TTS function across multiple concurrency levels.
        
        Args:
            tts_function: Async function that takes text and returns audio info
            test_texts: List of test texts to synthesize
            concurrency_levels: List of concurrency levels to test
            iterations_per_level: Number of iterations per concurrency level
            **tts_kwargs: Additional keyword arguments for TTS function
            
        Returns:
            Dictionary mapping concurrency level to benchmark results
        """
        results = {}
        
        for concurrency in concurrency_levels:
            logger.info(f"Running async TTS benchmark at concurrency level {concurrency}")
            
            # Run warmup
            await self._run_warmup_async(tts_function, test_texts[:self.warmup_requests], **tts_kwargs)
            
            # Run actual benchmark
            request_metrics = await self._run_async_benchmark(
                tts_function, test_texts, concurrency, iterations_per_level, **tts_kwargs
            )
            
            # Aggregate metrics
            metrics = aggregate_tts_metrics(request_metrics, concurrency)
            
            results[concurrency] = TTSBenchmarkResult(
                metrics=metrics,
                request_metrics=request_metrics,
                config={
                    'concurrency': concurrency,
                    'iterations': iterations_per_level,
                    'test_texts_count': len(test_texts),
                    'tts_kwargs': tts_kwargs
                }
            )
            
            logger.info(f"Concurrency {concurrency}: ASPS={metrics.asps:.3f}, "
                       f"RTF={metrics.rtf_median:.3f}, RPS={metrics.rps:.1f}")
        
        return results
    
    def _run_warmup_sync(
        self,
        tts_function: Callable,
        warmup_texts: List[str],
        **tts_kwargs
    ) -> None:
        """Run warmup requests to initialize the TTS system."""
        if not warmup_texts:
            return
            
        logger.debug(f"Running {len(warmup_texts)} warmup requests")
        
        for i, text in enumerate(warmup_texts):
            try:
                result = tts_function(text, **tts_kwargs)
                logger.debug(f"Warmup {i+1}/{len(warmup_texts)} completed")
            except Exception as e:
                logger.warning(f"Warmup request {i+1} failed: {e}")
    
    async def _run_warmup_async(
        self,
        tts_function: Callable,
        warmup_texts: List[str],
        **tts_kwargs
    ) -> None:
        """Run async warmup requests to initialize the TTS system."""
        if not warmup_texts:
            return
            
        logger.debug(f"Running {len(warmup_texts)} async warmup requests")
        
        for i, text in enumerate(warmup_texts):
            try:
                result = await tts_function(text, **tts_kwargs)
                logger.debug(f"Warmup {i+1}/{len(warmup_texts)} completed")
            except Exception as e:
                logger.warning(f"Warmup request {i+1} failed: {e}")
    
    def _run_sync_benchmark(
        self,
        tts_function: Callable,
        test_texts: List[str],
        concurrency: int,
        iterations: int,
        **tts_kwargs
    ) -> List[TTSRequestMetrics]:
        """Run synchronous benchmark with specified concurrency."""
        request_metrics = []
        
        # Prepare requests (cycle through test texts)
        requests = []
        for i in range(iterations):
            text = test_texts[i % len(test_texts)]
            requests.append((f"req_{i}", text))
        
        if concurrency == 1:
            # Sequential execution
            for req_id, text in requests:
                metrics = self._execute_sync_request(req_id, tts_function, text, **tts_kwargs)
                request_metrics.append(metrics)
        else:
            # Concurrent execution
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_req = {
                    executor.submit(self._execute_sync_request, req_id, tts_function, text, **tts_kwargs): req_id
                    for req_id, text in requests
                }
                
                for future in as_completed(future_to_req, timeout=self.timeout_seconds * len(requests)):
                    try:
                        metrics = future.result()
                        request_metrics.append(metrics)
                    except Exception as e:
                        req_id = future_to_req[future]
                        error_metrics = TTSRequestMetrics(
                            request_id=req_id,
                            t_start=time.perf_counter(),
                            error=str(e)
                        )
                        request_metrics.append(error_metrics)
        
        return request_metrics
    
    async def _run_async_benchmark(
        self,
        tts_function: Callable,
        test_texts: List[str],
        concurrency: int,
        iterations: int,
        **tts_kwargs
    ) -> List[TTSRequestMetrics]:
        """Run asynchronous benchmark with specified concurrency."""
        request_metrics = []
        
        # Prepare requests (cycle through test texts)
        requests = []
        for i in range(iterations):
            text = test_texts[i % len(test_texts)]
            requests.append((f"req_{i}", text))
        
        if concurrency == 1:
            # Sequential execution
            for req_id, text in requests:
                metrics = await self._execute_async_request(req_id, tts_function, text, **tts_kwargs)
                request_metrics.append(metrics)
        else:
            # Concurrent execution with semaphore
            semaphore = asyncio.Semaphore(concurrency)
            
            async def bounded_request(req_id: str, text: str):
                async with semaphore:
                    return await self._execute_async_request(req_id, tts_function, text, **tts_kwargs)
            
            tasks = [bounded_request(req_id, text) for req_id, text in requests]
            request_metrics = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_metrics = []
            for i, result in enumerate(request_metrics):
                if isinstance(result, Exception):
                    error_metrics = TTSRequestMetrics(
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
        tts_function: Callable,
        text: str,
        **tts_kwargs
    ) -> TTSRequestMetrics:
        """Execute a single synchronous TTS request and collect metrics."""
        metrics = TTSRequestMetrics(
            request_id=request_id,
            t_start=time.perf_counter(),
            text_len_chars=len(text),
            sample_rate=self.sample_rate,
            bit_depth=self.bit_depth
        )
        
        try:
            # For streaming TTS, we would set t_first_audio when first chunk arrives
            # For now, assume non-streaming TTS
            result = tts_function(text, **tts_kwargs)
            
            metrics.t_first_audio = time.perf_counter()  # Approximate for non-streaming
            metrics.t_end = time.perf_counter()
            
            # Extract audio information from result
            if isinstance(result, dict):
                metrics.audio_duration_sec = result.get('audio_duration', 0.0)
                if 'sample_rate' in result:
                    metrics.sample_rate = result['sample_rate']
                if 'text_tokens' in result:
                    metrics.text_len_tokens = result['text_tokens']
            else:
                logger.warning(f"TTS function returned unexpected type: {type(result)}")
                
        except Exception as e:
            metrics.error = str(e)
            metrics.t_end = time.perf_counter()
            logger.error(f"Request {request_id} failed: {e}")
        
        return metrics
    
    async def _execute_async_request(
        self,
        request_id: str,
        tts_function: Callable,
        text: str,
        **tts_kwargs
    ) -> TTSRequestMetrics:
        """Execute a single asynchronous TTS request and collect metrics."""
        metrics = TTSRequestMetrics(
            request_id=request_id,
            t_start=time.perf_counter(),
            text_len_chars=len(text),
            sample_rate=self.sample_rate,
            bit_depth=self.bit_depth
        )
        
        try:
            # For streaming TTS, we would set t_first_audio when first chunk arrives
            result = await tts_function(text, **tts_kwargs)
            
            metrics.t_first_audio = time.perf_counter()  # Approximate for non-streaming
            metrics.t_end = time.perf_counter()
            
            # Extract audio information from result
            if isinstance(result, dict):
                metrics.audio_duration_sec = result.get('audio_duration', 0.0)
                if 'sample_rate' in result:
                    metrics.sample_rate = result['sample_rate']
                if 'text_tokens' in result:
                    metrics.text_len_tokens = result['text_tokens']
            else:
                logger.warning(f"TTS function returned unexpected type: {type(result)}")
                
        except Exception as e:
            metrics.error = str(e)
            metrics.t_end = time.perf_counter()
            logger.error(f"Request {request_id} failed: {e}")
        
        return metrics


def create_streaming_tts_wrapper(
    streaming_tts_function: Callable,
    chunk_callback: Optional[Callable[[str, float], None]] = None
):
    """
    Create a wrapper for streaming TTS functions to properly capture TTFA.
    
    Args:
        streaming_tts_function: Function that yields audio chunks
        chunk_callback: Optional callback called with (request_id, timestamp) for first chunk
        
    Returns:
        Wrapped function that captures streaming metrics
    """
    def wrapper(request_id: str, text: str, **kwargs):
        chunks = []
        first_chunk_time = None
        
        for chunk in streaming_tts_function(text, **kwargs):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
                if chunk_callback:
                    chunk_callback(request_id, first_chunk_time)
            chunks.append(chunk)
        
        # Combine chunks and compute total duration
        # This is model-specific - adjust based on your audio format
        total_samples = sum(len(chunk) for chunk in chunks)
        sample_rate = kwargs.get('sample_rate', 22050)
        audio_duration = total_samples / sample_rate
        
        return {
            'audio_duration': audio_duration,
            'sample_rate': sample_rate,
            'chunks': chunks,
            'first_chunk_time': first_chunk_time
        }
    
    return wrapper

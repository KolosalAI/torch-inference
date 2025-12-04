#!/usr/bin/env python3
"""
ResNet Image Classification Benchmark Module.

This module provides benchmarking functionality specifically for ResNet and other
image classification models, adapting the image benchmark framework for classification
tasks instead of image generation.
"""

import os
import time
import random
import logging
import asyncio
import base64
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
from PIL import Image
import io
import numpy as np

from .image_benchmark import ImageBenchmarker, ImageBenchmarkResult
from .image_metrics import ImageRequestMetrics, ImageMetrics, aggregate_image_metrics

logger = logging.getLogger(__name__)


class ResNetImageBenchmarker(ImageBenchmarker):
    """
    ResNet Image Classification Benchmarker.
    
    Extends the ImageBenchmarker to work with image classification models
    instead of image generation models.
    """
    
    def __init__(
        self,
        default_width: int = 224,  # ResNet typical input size
        default_height: int = 224,
        warmup_requests: int = 5,
        timeout_seconds: float = 30.0,
        monitor_memory: bool = True,
        test_images_dir: Optional[str] = None
    ):
        """
        Initialize ResNet image classification benchmarker.
        
        Args:
            default_width: Default image width for ResNet (224x224 typical)
            default_height: Default image height for ResNet
            warmup_requests: Number of warmup requests before timing
            timeout_seconds: Timeout for individual requests
            monitor_memory: Whether to monitor memory usage
            test_images_dir: Directory containing test images
        """
        super().__init__(
            default_width=default_width,
            default_height=default_height,
            default_steps=1,  # Not applicable for classification
            warmup_requests=warmup_requests,
            timeout_seconds=timeout_seconds,
            monitor_memory=monitor_memory
        )
        
        self.test_images_dir = test_images_dir
        self._test_images_cache = {}
        
    def benchmark_resnet_model(
        self,
        classification_function: Callable[[bytes], Dict[str, Any]],
        test_images: Optional[List[bytes]] = None,
        concurrency_levels: List[int] = [1, 2, 4, 8, 16],
        iterations_per_level: int = 50,
        **kwargs
    ) -> Dict[int, ImageBenchmarkResult]:
        """
        Benchmark a ResNet classification function across multiple concurrency levels.
        
        Args:
            classification_function: Function that takes image bytes and returns classification results
            test_images: List of test images as bytes. If None, generates synthetic images
            concurrency_levels: List of concurrency levels to test
            iterations_per_level: Number of iterations per concurrency level
            **kwargs: Additional keyword arguments for classification function
            
        Returns:
            Dictionary mapping concurrency level to benchmark results
        """
        
        # Prepare test images
        if test_images is None:
            test_images = self._generate_test_images(min(iterations_per_level, 10))
        
        logger.info(f"Starting ResNet benchmark with {len(test_images)} test images")
        
        results = {}
        
        for concurrency in concurrency_levels:
            logger.info(f"Running ResNet benchmark at concurrency level {concurrency}")
            
            # Simple synchronous implementation for now
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import statistics
            
            request_metrics = []
            requests = []
            
            # Prepare requests (cycle through test images)
            for i in range(iterations_per_level):
                image_data = test_images[i % len(test_images)]
                requests.append((f"req_{i}", image_data))
            
            if concurrency == 1:
                # Sequential execution
                for req_id, image_data in requests:
                    start_time = time.perf_counter()
                    try:
                        result = classification_function(image_data, **kwargs)
                        end_time = time.perf_counter()
                        
                        # Create simplified metrics
                        metrics = ImageRequestMetrics(
                            request_id=req_id,
                            t_start=start_time,
                            t_first_image=end_time,
                            t_end=end_time,
                            prompt_len_chars=len(image_data),
                            width=224,
                            height=224,
                            num_images=1
                        )
                        request_metrics.append(metrics)
                    except Exception as e:
                        logger.error(f"Request {req_id} failed: {e}")
                        metrics = ImageRequestMetrics(
                            request_id=req_id,
                            t_start=start_time,
                            error=str(e)
                        )
                        request_metrics.append(metrics)
            else:
                # Concurrent execution
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    future_to_req = {}
                    for req_id, image_data in requests:
                        future = executor.submit(self._execute_classification_request, req_id, classification_function, image_data, **kwargs)
                        future_to_req[future] = req_id
                    
                    for future in as_completed(future_to_req, timeout=self.timeout_seconds * len(requests)):
                        try:
                            metrics = future.result()
                            request_metrics.append(metrics)
                        except Exception as e:
                            req_id = future_to_req[future]
                            metrics = ImageRequestMetrics(
                                request_id=req_id,
                                t_start=time.perf_counter(),
                                error=str(e)
                            )
                            request_metrics.append(metrics)
            
            # Aggregate metrics
            metrics = aggregate_image_metrics(request_metrics, concurrency)
            
            results[concurrency] = ImageBenchmarkResult(
                metrics=metrics,
                request_metrics=request_metrics,
                config={
                    'concurrency': concurrency,
                    'iterations': iterations_per_level,
                    'test_images_count': len(test_images),
                    'classification_kwargs': kwargs,
                    'model_type': 'image_classification'
                }
            )
            
            logger.info(f"Concurrency {concurrency}: "
                       f"Classifications/sec={metrics.ips:.3f}, "
                       f"RPS={metrics.rps:.1f}")
        
        return results
    
    def benchmark_single_concurrency_classification(
        self,
        classification_function: Callable[[bytes], Dict[str, Any]],
        concurrency: int,
        iterations: int,
        test_images: Optional[List[bytes]] = None,
        **kwargs
    ) -> ImageBenchmarkResult:
        """
        Benchmark ResNet classification at a single concurrency level.
        
        Args:
            classification_function: Function that takes image bytes and returns classification results
            concurrency: Number of concurrent requests
            iterations: Number of iterations to run
            test_images: List of test images as bytes. If None, generates synthetic images
            **kwargs: Additional keyword arguments for classification function
            
        Returns:
            ImageBenchmarkResult containing metrics for this concurrency level
        """
        # Prepare test images
        if test_images is None:
            test_images = self._generate_test_images(min(iterations, 10))
        
        logger.info(f"Running ResNet benchmark: concurrency={concurrency}, iterations={iterations}")
        
        request_metrics = []
        requests = []
        
        # Prepare requests (cycle through test images)
        for i in range(iterations):
            image_data = test_images[i % len(test_images)]
            requests.append((f"req_{i}", image_data))
        
        if concurrency == 1:
            # Sequential execution
            for req_id, image_data in requests:
                start_time = time.perf_counter()
                try:
                    result = classification_function(image_data, **kwargs)
                    end_time = time.perf_counter()
                    
                    # Create simplified metrics
                    metrics = ImageRequestMetrics(
                        request_id=req_id,
                        t_start=start_time,
                        t_first_image=end_time,
                        t_end=end_time,
                        prompt_len_chars=len(image_data),
                        width=224,
                        height=224,
                        num_images=1
                    )
                    request_metrics.append(metrics)
                except Exception as e:
                    logger.error(f"Request {req_id} failed: {e}")
                    metrics = ImageRequestMetrics(
                        request_id=req_id,
                        t_start=start_time,
                        error=str(e)
                    )
                    request_metrics.append(metrics)
        else:
            # Concurrent execution
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_req = {}
                for req_id, image_data in requests:
                    future = executor.submit(self._execute_classification_request, req_id, classification_function, image_data, **kwargs)
                    future_to_req[future] = req_id
                
                for future in as_completed(future_to_req, timeout=self.timeout_seconds * len(requests)):
                    try:
                        metrics = future.result()
                        request_metrics.append(metrics)
                    except Exception as e:
                        req_id = future_to_req[future]
                        metrics = ImageRequestMetrics(
                            request_id=req_id,
                            t_start=time.perf_counter(),
                            error=str(e)
                        )
                        request_metrics.append(metrics)
        
        # Aggregate metrics
        metrics = aggregate_image_metrics(request_metrics, concurrency)
        
        result = ImageBenchmarkResult(
            metrics=metrics,
            request_metrics=request_metrics,
            config={
                'concurrency': concurrency,
                'iterations': iterations,
                'test_images_count': len(test_images),
                'classification_kwargs': kwargs,
                'model_type': 'image_classification'
            }
        )
        
        return result
    
    def _execute_classification_request(
        self,
        request_id: str,
        classification_function: Callable,
        image_data: bytes,
        **kwargs
    ) -> ImageRequestMetrics:
        """Execute a single classification request and collect metrics."""
        start_time = time.perf_counter()
        
        try:
            result = classification_function(image_data, **kwargs)
            end_time = time.perf_counter()
            
            # Create metrics
            metrics = ImageRequestMetrics(
                request_id=request_id,
                t_start=start_time,
                t_first_image=end_time,
                t_end=end_time,
                prompt_len_chars=len(image_data),
                width=224,
                height=224,
                num_images=1
            )
            return metrics
        except Exception as e:
            logger.error(f"Classification request {request_id} failed: {e}")
            return ImageRequestMetrics(
                request_id=request_id,
                t_start=start_time,
                error=str(e)
            )
    
    def _generate_test_images(self, count: int = 10) -> List[bytes]:
        """Generate synthetic test images for benchmarking."""
        test_images = []
        
        for i in range(count):
            # Create a synthetic image with random content
            image_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            
            # Add some structured patterns
            if i % 3 == 0:
                image_array[::10, :] = 255  # Horizontal lines
            elif i % 3 == 1:
                image_array[:, ::10] = 255  # Vertical lines
            else:
                # Add a simple circle
                center_y, center_x = 112, 112
                radius = 50
                y, x = np.ogrid[:224, :224]
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
                image_array[mask] = [255, 255, 255]
            
            # Convert to PIL Image and then to bytes
            pil_image = Image.fromarray(image_array, 'RGB')
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=85)
            test_images.append(img_buffer.getvalue())
        
        logger.info(f"Generated {count} synthetic test images for benchmarking")
        return test_images
    
    def stress_test_resnet_model(
        self,
        classification_function: Callable[[bytes], Dict[str, Any]],
        duration_minutes: int = 5,
        max_concurrency: int = 64,
        ramp_up_seconds: int = 30,
        test_images: Optional[List[bytes]] = None,
        monitor_memory: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform stress testing on ResNet model with gradually increasing load.
        
        This function simulates real-world stress conditions by:
        1. Gradually ramping up concurrency over time
        2. Maintaining high load for extended periods
        3. Monitoring system resources and error rates
        4. Testing recovery and stability under stress
        
        Args:
            classification_function: Function that takes image bytes and returns classification results
            duration_minutes: Total duration of stress test in minutes
            max_concurrency: Maximum concurrent requests to reach
            ramp_up_seconds: Time to gradually ramp up to max concurrency
            test_images: List of test images as bytes. If None, generates synthetic images
            monitor_memory: Whether to monitor memory usage during test
            **kwargs: Additional keyword arguments for classification function
            
        Returns:
            Dictionary containing stress test results and metrics
        """
        import threading
        import psutil
        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        logger.info(f"Starting ResNet stress test: {duration_minutes}min duration, max concurrency {max_concurrency}")
        
        # Prepare test images
        if test_images is None:
            # Generate more diverse images for stress testing
            test_images = self._generate_test_images(20)
        
        start_time = time.perf_counter()
        end_time = start_time + (duration_minutes * 60)
        ramp_end_time = start_time + ramp_up_seconds
        
        # Metrics collection
        results_lock = threading.Lock()
        stress_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'error_types': defaultdict(int),
            'response_times': [],
            'memory_snapshots': [],
            'concurrency_levels': [],
            'requests_per_second': [],
            'timestamps': []
        }
        
        # Memory monitoring setup
        process = psutil.Process() if monitor_memory else None
        
        def collect_system_metrics(timestamp, current_concurrency):
            """Collect system metrics at regular intervals."""
            if not monitor_memory:
                return
            
            try:
                cpu_percent = psutil.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                with results_lock:
                    stress_metrics['memory_snapshots'].append({
                        'timestamp': timestamp,
                        'memory_mb': memory_mb,
                        'cpu_percent': cpu_percent,
                        'concurrency': current_concurrency
                    })
            except Exception as e:
                logger.warning(f"Failed to collect system metrics: {e}")
        
        def worker_function(worker_id: int):
            """Worker function for stress testing."""
            local_request_count = 0
            local_success_count = 0
            local_error_count = 0
            local_response_times = []
            local_errors = defaultdict(int)
            
            while time.perf_counter() < end_time:
                current_time = time.perf_counter()
                
                # Calculate current target concurrency based on ramp-up
                if current_time < ramp_end_time:
                    # Ramp up phase
                    progress = (current_time - start_time) / ramp_up_seconds
                    target_concurrency = int(progress * max_concurrency)
                else:
                    # Full load phase
                    target_concurrency = max_concurrency
                
                # Only proceed if this worker should be active
                if worker_id >= target_concurrency:
                    time.sleep(0.1)
                    continue
                
                # Select random test image
                image_data = test_images[local_request_count % len(test_images)]
                req_id = f"stress_req_{worker_id}_{local_request_count}"
                
                request_start = time.perf_counter()
                local_request_count += 1
                
                try:
                    # Execute classification request
                    result = classification_function(image_data, **kwargs)
                    request_end = time.perf_counter()
                    response_time = request_end - request_start
                    
                    local_response_times.append(response_time)
                    
                    if result.get('success', False):
                        local_success_count += 1
                    else:
                        local_error_count += 1
                        error_msg = result.get('error', 'Unknown error')
                        local_errors[error_msg] += 1
                        
                except Exception as e:
                    local_error_count += 1
                    local_errors[str(e)] += 1
                    local_response_times.append(time.perf_counter() - request_start)
                
                # Brief pause to prevent overwhelming the system
                time.sleep(0.001)
            
            # Aggregate worker results
            with results_lock:
                stress_metrics['total_requests'] += local_request_count
                stress_metrics['successful_requests'] += local_success_count
                stress_metrics['failed_requests'] += local_error_count
                stress_metrics['response_times'].extend(local_response_times)
                
                for error_type, count in local_errors.items():
                    stress_metrics['error_types'][error_type] += count
        
        # Start stress test workers
        logger.info("Starting stress test workers...")
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            # Submit worker tasks
            futures = []
            for worker_id in range(max_concurrency):
                future = executor.submit(worker_function, worker_id)
                futures.append(future)
            
            # Monitor progress and collect metrics
            metrics_interval = 10  # Collect metrics every 10 seconds
            last_metrics_time = start_time
            last_request_count = 0
            
            while time.perf_counter() < end_time:
                current_time = time.perf_counter()
                
                # Collect periodic metrics
                if current_time - last_metrics_time >= metrics_interval:
                    with results_lock:
                        current_requests = stress_metrics['total_requests']
                        requests_delta = current_requests - last_request_count
                        rps = requests_delta / metrics_interval
                        
                        # Calculate current concurrency
                        if current_time < ramp_end_time:
                            progress = (current_time - start_time) / ramp_up_seconds
                            current_concurrency = int(progress * max_concurrency)
                        else:
                            current_concurrency = max_concurrency
                        
                        stress_metrics['requests_per_second'].append(rps)
                        stress_metrics['concurrency_levels'].append(current_concurrency)
                        stress_metrics['timestamps'].append(current_time)
                        
                        last_request_count = current_requests
                        last_metrics_time = current_time
                    
                    # Collect system metrics
                    collect_system_metrics(current_time, current_concurrency)
                    
                    # Log progress
                    logger.info(f"Stress test progress: {(current_time - start_time) / 60:.1f}min, "
                               f"RPS: {rps:.1f}, Concurrency: {current_concurrency}, "
                               f"Success rate: {(stress_metrics['successful_requests'] / max(stress_metrics['total_requests'], 1)) * 100:.1f}%")
                
                time.sleep(1)
            
            # Wait for all workers to complete
            for future in as_completed(futures, timeout=30):
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Worker finished with exception: {e}")
        
        # Calculate final metrics
        total_duration = time.perf_counter() - start_time
        
        # Calculate statistics for response times
        response_times = stress_metrics['response_times']
        if response_times:
            response_times.sort()
            n = len(response_times)
            avg_response_time = sum(response_times) / n
            p50_response_time = response_times[int(0.5 * n)]
            p95_response_time = response_times[int(0.95 * n)]
            p99_response_time = response_times[int(0.99 * n)]
        else:
            avg_response_time = p50_response_time = p95_response_time = p99_response_time = 0.0
        
        # Calculate overall throughput
        overall_rps = stress_metrics['total_requests'] / total_duration if total_duration > 0 else 0
        success_rate = (stress_metrics['successful_requests'] / max(stress_metrics['total_requests'], 1)) * 100
        
        # Memory usage analysis
        memory_analysis = {}
        if stress_metrics['memory_snapshots']:
            memory_values = [s['memory_mb'] for s in stress_metrics['memory_snapshots']]
            cpu_values = [s['cpu_percent'] for s in stress_metrics['memory_snapshots']]
            
            memory_analysis = {
                'peak_memory_mb': max(memory_values),
                'avg_memory_mb': sum(memory_values) / len(memory_values),
                'memory_growth_mb': max(memory_values) - min(memory_values),
                'peak_cpu_percent': max(cpu_values),
                'avg_cpu_percent': sum(cpu_values) / len(cpu_values)
            }
        
        # Compile final results
        stress_results = {
            'test_configuration': {
                'duration_minutes': duration_minutes,
                'max_concurrency': max_concurrency,
                'ramp_up_seconds': ramp_up_seconds,
                'test_images_count': len(test_images),
                'actual_duration_seconds': total_duration
            },
            'performance_metrics': {
                'total_requests': stress_metrics['total_requests'],
                'successful_requests': stress_metrics['successful_requests'],
                'failed_requests': stress_metrics['failed_requests'],
                'success_rate_percent': success_rate,
                'overall_rps': overall_rps,
                'avg_response_time_ms': avg_response_time * 1000,
                'p50_response_time_ms': p50_response_time * 1000,
                'p95_response_time_ms': p95_response_time * 1000,
                'p99_response_time_ms': p99_response_time * 1000
            },
            'error_analysis': {
                'error_types': dict(stress_metrics['error_types']),
                'error_rate_percent': (stress_metrics['failed_requests'] / max(stress_metrics['total_requests'], 1)) * 100
            },
            'system_metrics': memory_analysis,
            'time_series_data': {
                'timestamps': stress_metrics['timestamps'],
                'requests_per_second': stress_metrics['requests_per_second'],
                'concurrency_levels': stress_metrics['concurrency_levels'],
                'memory_snapshots': stress_metrics['memory_snapshots']
            }
        }
        
        logger.info(f"Stress test completed: {stress_metrics['total_requests']} total requests, "
                   f"{success_rate:.1f}% success rate, {overall_rps:.1f} RPS average")
        
        return stress_results


def check_model_availability(
    model_name: str,
    base_url: str = "http://localhost:8000",
    auth_token: Optional[str] = None,
    auto_load: bool = True
) -> tuple[bool, str]:
    """
    Check if a model is available on the server, and optionally load it if not.
    
    Args:
        model_name: Name of the model to check
        base_url: Base URL of the torch-inference server
        auth_token: Optional authentication token
        auto_load: Whether to attempt loading the model if it's not available
        
    Returns:
        Tuple of (is_available, message)
    """
    import requests
    
    try:
        # Check server status first
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        
        # Get models list from /models endpoint
        try:
            models_response = requests.get(
                f"{base_url}/models",
                headers=headers,
                timeout=10
            )
            if models_response.status_code == 200:
                models_data = models_response.json()
                
                # Parse the response format we discovered
                available_models = models_data.get('available_models', [])
                if model_name in available_models:
                    return True, f"Model '{model_name}' is available"
                
                # Model not in available list, check if it exists in models_info
                models_info = models_data.get('models_info', {})
                if model_name in models_info:
                    model_info = models_info[model_name]
                    if not auto_load:
                        return False, f"Model '{model_name}' exists but is not loaded. Available models: {available_models}"
                    
                    # Try to load the model
                    logger.info(f"Model '{model_name}' exists but not loaded. Attempting to load...")
                    
                    # Try different loading approaches
                    loading_attempts = [
                        # Try /models/download with full parameters
                        lambda: requests.post(
                            f"{base_url}/models/download",
                            json={
                                "name": model_name,
                                "model_name": model_name,
                                "source": model_info.get("source", "torchvision"),
                                "model_id": model_info.get("model_id", model_name)
                            },
                            headers=headers,
                            timeout=30
                        ),
                        # Try manage endpoint with various actions
                        lambda: requests.post(
                            f"{base_url}/models/manage?action=enable&model_name={model_name}",
                            headers=headers,
                            timeout=15
                        ),
                        lambda: requests.post(
                            f"{base_url}/models/manage?action=activate&model_name={model_name}",
                            headers=headers,
                            timeout=15
                        ),
                    ]
                    
                    for attempt in loading_attempts:
                        try:
                            load_response = attempt()
                            if load_response.status_code == 200:
                                load_result = load_response.json()
                                if load_result.get("success", False):
                                    logger.info(f"Successfully initiated loading of model '{model_name}'")
                                    
                                    # Wait a moment and check again
                                    import time
                                    time.sleep(2)
                                    
                                    # Re-check availability
                                    recheck_response = requests.get(f"{base_url}/models", headers=headers, timeout=10)
                                    if recheck_response.status_code == 200:
                                        recheck_data = recheck_response.json()
                                        recheck_available = recheck_data.get('available_models', [])
                                        if model_name in recheck_available:
                                            return True, f"Model '{model_name}' successfully loaded and is now available"
                                        else:
                                            return False, f"Model '{model_name}' loading initiated but not yet available. Try again in a moment."
                                    
                                    return True, f"Model '{model_name}' loading initiated"
                        except Exception as e:
                            logger.debug(f"Loading attempt failed: {e}")
                            continue
                    
                    return False, f"Model '{model_name}' exists but could not be loaded automatically. Available models: {available_models}"
                else:
                    return False, f"Model '{model_name}' not found. Available models: {available_models}"
            else:
                logger.warning(f"Failed to get models list: HTTP {models_response.status_code}")
        except Exception as e:
            logger.debug(f"Models endpoint failed: {e}")
        
        # Fall back to making a test prediction request
        logger.info(f"Falling back to test prediction for model availability check")
        
        # Create a small test image
        test_image_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image_array, 'RGB')
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG', quality=85)
        test_image_data = img_buffer.getvalue()
        
        # Encode as base64
        image_b64 = base64.b64encode(test_image_data).decode('utf-8')
        
        # Prepare test payload
        payload = {
            "model_name": model_name,
            "inputs": image_b64,
            "metadata": {
                "input_type": "image",
                "format": "base64",
                "top_k": 1
            }
        }
        
        headers["Content-Type"] = "application/json"
        
        # Make test request
        response = requests.post(
            f"{base_url}/predict",
            json=payload,
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            # Check if there's an error about model availability
            if 'error' in result and 'not available' in result['error'].lower():
                return False, f"Model '{model_name}' is not available: {result['error']}"
            elif 'result' in result or 'predictions' in result:
                return True, f"Model '{model_name}' is available and responding"
            else:
                return False, f"Model '{model_name}' returned unexpected response: {result}"
        else:
            error_text = response.text
            if 'not available' in error_text.lower():
                return False, f"Model '{model_name}' is not available: {error_text}"
            else:
                return False, f"Server error (HTTP {response.status_code}): {error_text}"
                
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to server at {base_url}. Is the server running?"
    except requests.exceptions.Timeout:
        return False, f"Timeout connecting to server at {base_url}"
    except Exception as e:
        return False, f"Error checking model availability: {str(e)}"


def create_resnet_classification_function(
    model_name: str = "resnet18",
    base_url: str = "http://localhost:8000",
    auth_token: Optional[str] = None,
    top_k: int = 5,
    check_availability: bool = True,
    auto_load: bool = True
) -> Callable[[bytes], Dict[str, Any]]:
    """
    Create a ResNet classification function that calls the torch-inference API.
    
    Args:
        model_name: Name of the ResNet model to use
        base_url: Base URL of the torch-inference server
        auth_token: Optional authentication token
        top_k: Number of top predictions to return
        check_availability: Whether to check model availability before creating function
        auto_load: Whether to attempt loading the model if it's not available
        
    Returns:
        Function that takes image bytes and returns classification results
        
    Raises:
        RuntimeError: If check_availability is True and model is not available
    """
    import requests
    
    # Check model availability if requested
    if check_availability:
        is_available, message = check_model_availability(model_name, base_url, auth_token, auto_load)
        if not is_available:
            raise RuntimeError(f"Model availability check failed: {message}")
        else:
            logger.info(f"Model availability confirmed: {message}")
    
    def classify_image(image_data: bytes, **kwargs) -> Dict[str, Any]:
        """
        Classify an image using ResNet model via torch-inference API.
        
        Args:
            image_data: Image data as bytes
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with classification results
        """
        start_time = time.time()
        
        try:
            # Encode image as base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare request payload
            payload = {
                "model_name": model_name,
                "inputs": image_b64,
                "metadata": {
                    "input_type": "image",
                    "format": "base64",
                    "top_k": top_k
                }
            }
            
            # Add auth header if provided
            headers = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
            
            # Make request - Fixed URL
            response = requests.post(
                f"{base_url}/predict",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract predictions from response
                predictions = result.get('result', {})
                if isinstance(predictions, dict):
                    predictions = predictions.get('predictions', predictions)
                
                return {
                    'success': True,
                    'predictions': predictions,
                    'processing_time': processing_time,
                    'model_name': model_name,
                    'status_code': response.status_code,
                    'server_processing_time': result.get('processing_time', None)
                }
            else:
                error_text = response.text
                logger.warning(f"Server request failed: HTTP {response.status_code} - {error_text}")
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {error_text}",
                    'processing_time': processing_time,
                    'model_name': model_name,
                    'status_code': response.status_code
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'model_name': model_name
            }
    
    return classify_image


def create_demo_resnet_function() -> Callable[[bytes], Dict[str, Any]]:
    """
    Create a demo ResNet classification function for testing without a server.
    
    Returns:
        Function that simulates ResNet classification
    """
    # Imaginary ImageNet class names for demo
    demo_classes = [
        "Golden retriever", "Tabby cat", "Egyptian cat", "Tiger cat", "Persian cat",
        "Siamese cat", "Australian terrier", "English springer", "German shepherd",
        "Border collie", "Labrador retriever", "Cocker spaniel", "Poodle", "Dalmatian",
        "Husky", "Bulldog", "Beagle", "Rottweiler", "German short-haired pointer",
        "Yorkshire terrier", "Chihuahua", "Shih tzu", "Pug", "Boston terrier",
        "Elephant", "Lion", "Tiger", "Bear", "Zebra", "Giraffe", "Horse", "Cow",
        "Pig", "Sheep", "Chicken", "Duck", "Turkey", "Goose", "Swan", "Eagle",
        "Hawk", "Parrot", "Penguin", "Ostrich", "Flamingo", "Pelican", "Seagull"
    ]
    
    def demo_classify_image(image_data: bytes, **kwargs) -> Dict[str, Any]:
        """
        Demo classification function that simulates ResNet inference.
        
        Args:
            image_data: Image data as bytes
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with simulated classification results
        """
        start_time = time.time()
        
        # Simulate processing time (ResNet is typically fast)
        base_time = 0.05  # 50ms base processing time
        complexity_factor = len(image_data) / (224 * 224 * 3)  # Based on image size
        processing_time = base_time * complexity_factor * (0.8 + random.random() * 0.4)
        
        time.sleep(processing_time)
        
        # Generate realistic-looking predictions
        top_k = kwargs.get('top_k', 5)
        num_predictions = min(top_k, len(demo_classes))
        
        # Create predictions with decreasing confidence
        predictions = []
        selected_classes = random.sample(demo_classes, num_predictions)
        
        for i, class_name in enumerate(selected_classes):
            # First prediction has highest confidence
            confidence = random.uniform(0.15, 0.95) * (0.95 - i * 0.1)
            predictions.append({
                "class": class_name,
                "confidence": round(confidence, 4),
                "class_id": demo_classes.index(class_name)
            })
        
        # Sort by confidence (highest first)
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        actual_processing_time = time.time() - start_time
        
        return {
            'success': True,
            'predictions': predictions,
            'processing_time': actual_processing_time,
            'model_name': 'demo_resnet18',
            'image_size': len(image_data),
            'num_classes': len(demo_classes)
        }
    
    return demo_classify_image


def quick_resnet_test():
    """Quick test to verify ResNet benchmark functionality."""
    print("Quick ResNet Benchmark Test")
    print("-" * 30)
    
    benchmarker = ResNetImageBenchmarker(warmup_requests=1)
    demo_classifier = create_demo_resnet_function()
    
    # Quick single-concurrency test
    result = benchmarker.benchmark_single_concurrency_classification(
        classification_function=demo_classifier,
        concurrency=1,
        iterations=5
    )
    
    metrics = result.metrics
    print(f"✓ ResNet benchmark working!")
    print(f"  Classifications/sec: {metrics.ips:.2f}")
    print(f"  Average latency: {metrics.ttfi_p50*1000:.1f}ms")
    print(f"  Success rate: {metrics.success_rate:.1f}%")
    
    return result


def quick_stress_test():
    """Run a quick 30-second stress test for initial validation."""
    print("=" * 60)
    print("Quick ResNet Stress Test (30 seconds)")
    print("=" * 60)
    
    # Create benchmarker
    benchmarker = ResNetImageBenchmarker(
        default_width=224,
        default_height=224,
        warmup_requests=2,
        monitor_memory=True
    )
    
    # Use demo classifier
    demo_classifier = create_demo_resnet_function()
    
    print("Running quick stress test...")
    print("  Duration: 0.5 minutes (30 seconds)")
    print("  Max Concurrency: 16")
    print("  Ramp-up: 10 seconds")
    print()
    
    # Run stress test
    results = benchmarker.stress_test_resnet_model(
        classification_function=demo_classifier,
        duration_minutes=0.5,  # 30 seconds
        max_concurrency=16,
        ramp_up_seconds=10,
        monitor_memory=True
    )
    
    # Print summary
    perf = results['performance_metrics']
    print(f"\nQuick Stress Test Results:")
    print(f"  Total Requests: {perf['total_requests']:,}")
    print(f"  Success Rate: {perf['success_rate_percent']:.1f}%")
    print(f"  Average Throughput: {perf['overall_rps']:.1f} RPS")
    print(f"  Average Latency: {perf['avg_response_time_ms']:.1f}ms")
    print(f"  95th Percentile Latency: {perf['p95_response_time_ms']:.1f}ms")
    
    if perf['success_rate_percent'] >= 95:
        print("✅ Quick stress test PASSED - System stable under load")
    else:
        print("❌ Quick stress test FAILED - System experienced issues under load")
    
    return results

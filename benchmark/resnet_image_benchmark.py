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
                       f"RPS={metrics.rps:.1f}, "
                       f"Avg latency={metrics.ttfi_p50*1000:.1f}ms")
        
        return results
    
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

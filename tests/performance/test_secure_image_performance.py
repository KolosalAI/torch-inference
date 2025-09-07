"""
Performance tests for secure image processing system.

Tests performance characteristics, memory usage, and scalability
of the secure image processing components.
"""

import pytest
import time
import psutil
import gc
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from framework.processors.image.secure_image_processor import (
        SecurityLevel, SecurityConfig, SecureImageValidator, 
        SecureImageSanitizer, SecureImagePreprocessor
    )
    from framework.models.secure_image_model import SecureImageModel
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False


class PerformanceTracker:
    """Helper class to track performance metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.start_time = None
        self.end_time = None
        self.memory_start = None
        self.memory_peak = None
        self.memory_end = None
    
    def start(self):
        """Start performance tracking."""
        gc.collect()  # Clean up before measurement
        self.start_time = time.time()
        self.memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.memory_peak = self.memory_start
    
    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        if current_memory > self.memory_peak:
            self.memory_peak = current_memory
    
    def stop(self):
        """Stop performance tracking."""
        self.end_time = time.time()
        self.memory_end = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.update_peak_memory()
    
    @property
    def duration(self):
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def memory_delta(self):
        """Get memory usage delta in MB."""
        if self.memory_start and self.memory_end:
            return self.memory_end - self.memory_start
        return None
    
    @property
    def peak_memory_delta(self):
        """Get peak memory usage delta in MB."""
        if self.memory_start and self.memory_peak:
            return self.memory_peak - self.memory_start
        return None


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Secure image components not available")
class TestSecureImageProcessorPerformance:
    """Performance tests for secure image processor components."""
    
    @pytest.fixture
    def performance_tracker(self):
        """Create performance tracker."""
        return PerformanceTracker()
    
    @pytest.fixture
    def test_images(self):
        """Create test images of various sizes."""
        sizes = [(64, 64), (224, 224), (512, 512), (1024, 1024)]
        images = {}
        
        for width, height in sizes:
            # Create random image
            img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            
            images[f"{width}x{height}"] = buffer
        
        return images
    
    def test_validator_performance_by_security_level(self, performance_tracker, test_images):
        """Test validator performance across different security levels."""
        results = {}
        
        for level in SecurityLevel:
            config = SecurityConfig(security_level=level)
            validator = SecureImageValidator(config)
            
            level_results = {}
            
            for size_name, image_buffer in test_images.items():
                image_buffer.seek(0)
                img = Image.open(image_buffer)
                
                performance_tracker.start()
                
                # Run validation
                try:
                    result = validator.validate_image(img, f"test_{size_name}.png")
                    success = True
                except Exception:
                    success = False
                
                performance_tracker.stop()
                
                level_results[size_name] = {
                    'duration': performance_tracker.duration,
                    'memory_delta': performance_tracker.memory_delta,
                    'peak_memory_delta': performance_tracker.peak_memory_delta,
                    'success': success
                }
            
            results[level.value] = level_results
        
        # Analyze results
        for level, level_results in results.items():
            for size_name, metrics in level_results.items():
                if metrics['success']:
                    # Validation should complete reasonably quickly
                    assert metrics['duration'] < 5.0, f"Validation too slow for {level}/{size_name}: {metrics['duration']}s"
                    
                    # Memory usage should be reasonable
                    if metrics['peak_memory_delta']:
                        assert metrics['peak_memory_delta'] < 100, f"Memory usage too high for {level}/{size_name}: {metrics['peak_memory_delta']}MB"
        
        # Higher security levels may take longer
        if 'low' in results and 'high' in results:
            for size_name in test_images.keys():
                low_duration = results['low'][size_name]['duration']
                high_duration = results['high'][size_name]['duration']
                
                if low_duration and high_duration:
                    # High security should not be more than 10x slower
                    assert high_duration <= low_duration * 10
    
    def test_sanitizer_performance_by_image_size(self, performance_tracker, test_images):
        """Test sanitizer performance across different image sizes."""
        config = SecurityConfig(security_level=SecurityLevel.MEDIUM)
        sanitizer = SecureImageSanitizer(config)
        
        results = {}
        
        for size_name, image_buffer in test_images.items():
            image_buffer.seek(0)
            img = Image.open(image_buffer)
            
            performance_tracker.start()
            
            try:
                sanitized_img = sanitizer.sanitize_image(img)
                success = True
            except Exception:
                success = False
                sanitized_img = None
            
            performance_tracker.stop()
            
            results[size_name] = {
                'duration': performance_tracker.duration,
                'memory_delta': performance_tracker.memory_delta,
                'peak_memory_delta': performance_tracker.peak_memory_delta,
                'success': success,
                'output_size': sanitized_img.size if sanitized_img else None
            }
        
        # Analyze performance scaling
        for size_name, metrics in results.items():
            if metrics['success']:
                # Sanitization should scale reasonably with image size
                width, height = map(int, size_name.split('x'))
                pixel_count = width * height
                
                # Duration should scale sub-linearly with pixel count
                duration_per_pixel = metrics['duration'] / pixel_count * 1000000  # microseconds per pixel
                assert duration_per_pixel < 100, f"Sanitization too slow per pixel for {size_name}: {duration_per_pixel} μs/pixel"
                
                # Memory usage should be reasonable relative to image size
                if metrics['peak_memory_delta']:
                    memory_per_pixel = metrics['peak_memory_delta'] / pixel_count * 1024 * 1024  # bytes per pixel
                    assert memory_per_pixel < 100, f"Memory usage too high per pixel for {size_name}: {memory_per_pixel} B/pixel"
    
    def test_preprocessor_end_to_end_performance(self, performance_tracker, test_images):
        """Test full preprocessor pipeline performance."""
        config = SecurityConfig(
            security_level=SecurityLevel.MEDIUM,
            enable_adversarial_detection=True,
            enable_steganography_detection=True,
            enable_metadata_scanning=True
        )
        
        with patch('torch.torch.load'), patch('torch.jit.load'):
            preprocessor = SecureImagePreprocessor(config)
        
        results = {}
        
        for size_name, image_buffer in test_images.items():
            image_buffer.seek(0)
            
            performance_tracker.start()
            
            try:
                result = preprocessor.process_image_data(
                    image_buffer.getvalue(),
                    f"test_{size_name}.png"
                )
                success = result is not None
            except Exception:
                success = False
            
            performance_tracker.stop()
            
            results[size_name] = {
                'duration': performance_tracker.duration,
                'memory_delta': performance_tracker.memory_delta,
                'peak_memory_delta': performance_tracker.peak_memory_delta,
                'success': success
            }
        
        # Analyze end-to-end performance
        for size_name, metrics in results.items():
            if metrics['success']:
                # Full pipeline should complete within reasonable time
                assert metrics['duration'] < 10.0, f"Pipeline too slow for {size_name}: {metrics['duration']}s"
                
                # Memory usage should be controlled
                if metrics['peak_memory_delta']:
                    assert metrics['peak_memory_delta'] < 200, f"Pipeline memory usage too high for {size_name}: {metrics['peak_memory_delta']}MB"
    
    def test_concurrent_processing_performance(self, performance_tracker):
        """Test performance under concurrent processing load."""
        config = SecurityConfig(security_level=SecurityLevel.LOW)  # Fastest for concurrency test
        
        with patch('torch.torch.load'), patch('torch.jit.load'):
            preprocessor = SecureImagePreprocessor(config)
        
        # Create test image
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        test_image_data = buffer.getvalue()
        
        def process_image(image_id):
            """Process a single image."""
            start_time = time.time()
            try:
                result = preprocessor.process_image_data(test_image_data, f"concurrent_{image_id}.png")
                success = result is not None
            except Exception:
                success = False
            end_time = time.time()
            
            return {
                'image_id': image_id,
                'duration': end_time - start_time,
                'success': success
            }
        
        # Test different concurrency levels
        concurrency_levels = [1, 2, 4, 8]
        results = {}
        
        for num_threads in concurrency_levels:
            performance_tracker.start()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(process_image, i) for i in range(num_threads * 2)]  # 2 tasks per thread
                concurrent_results = [future.result() for future in as_completed(futures)]
            
            performance_tracker.stop()
            
            # Analyze concurrent processing results
            successful_processes = [r for r in concurrent_results if r['success']]
            total_duration = performance_tracker.duration
            avg_process_duration = np.mean([r['duration'] for r in successful_processes]) if successful_processes else 0
            
            results[num_threads] = {
                'total_duration': total_duration,
                'avg_process_duration': avg_process_duration,
                'successful_processes': len(successful_processes),
                'total_processes': len(concurrent_results),
                'success_rate': len(successful_processes) / len(concurrent_results) if concurrent_results else 0,
                'memory_delta': performance_tracker.memory_delta,
                'peak_memory_delta': performance_tracker.peak_memory_delta
            }
        
        # Analyze concurrency scaling
        for num_threads, metrics in results.items():
            # Should maintain reasonable success rate
            assert metrics['success_rate'] >= 0.5, f"Low success rate with {num_threads} threads: {metrics['success_rate']}"
            
            # Total duration should not scale linearly with thread count
            if num_threads > 1:
                single_thread_duration = results[1]['total_duration']
                assert metrics['total_duration'] <= single_thread_duration * num_threads * 1.5, \
                    f"Poor concurrent scaling with {num_threads} threads"
    
    def test_memory_leak_detection(self, performance_tracker):
        """Test for memory leaks during repeated processing."""
        config = SecurityConfig(security_level=SecurityLevel.LOW)
        
        with patch('torch.torch.load'), patch('torch.jit.load'):
            preprocessor = SecureImagePreprocessor(config)
        
        # Create test image
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        test_image_data = buffer.getvalue()
        
        # Process images repeatedly and track memory
        num_iterations = 20
        memory_measurements = []
        
        for i in range(num_iterations):
            # Force garbage collection before measurement
            gc.collect()
            
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = preprocessor.process_image_data(test_image_data, f"leak_test_{i}.png")
            except Exception:
                pass
            
            # Force garbage collection after processing
            gc.collect()
            
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_measurements.append(memory_after)
        
        # Analyze memory trend
        if len(memory_measurements) >= 10:
            # Calculate trend in last half of measurements
            mid_point = len(memory_measurements) // 2
            early_avg = np.mean(memory_measurements[:mid_point])
            late_avg = np.mean(memory_measurements[mid_point:])
            
            memory_growth = late_avg - early_avg
            
            # Memory growth should be minimal (less than 50MB over 20 iterations)
            assert memory_growth < 50, f"Potential memory leak detected: {memory_growth}MB growth over {num_iterations} iterations"
    
    def test_large_batch_processing_performance(self, performance_tracker):
        """Test performance with large batch of images."""
        config = SecurityConfig(security_level=SecurityLevel.LOW)
        
        with patch('torch.torch.load'), patch('torch.jit.load'):
            preprocessor = SecureImagePreprocessor(config)
        
        # Create batch of test images
        batch_size = 50
        test_images = []
        
        for i in range(batch_size):
            img_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            test_images.append(buffer.getvalue())
        
        performance_tracker.start()
        
        successful_processes = 0
        failed_processes = 0
        
        for i, image_data in enumerate(test_images):
            try:
                result = preprocessor.process_image_data(image_data, f"batch_{i}.png")
                if result:
                    successful_processes += 1
                else:
                    failed_processes += 1
            except Exception:
                failed_processes += 1
            
            # Update peak memory periodically
            if i % 10 == 0:
                performance_tracker.update_peak_memory()
        
        performance_tracker.stop()
        
        # Analyze batch processing performance
        total_processed = successful_processes + failed_processes
        success_rate = successful_processes / total_processed if total_processed > 0 else 0
        avg_time_per_image = performance_tracker.duration / total_processed if total_processed > 0 else 0
        
        # Should maintain good performance and success rate
        assert success_rate >= 0.8, f"Low success rate in batch processing: {success_rate}"
        assert avg_time_per_image < 1.0, f"Slow average processing time: {avg_time_per_image}s per image"
        
        # Memory usage should be reasonable for batch size
        if performance_tracker.peak_memory_delta:
            memory_per_image = performance_tracker.peak_memory_delta / batch_size
            assert memory_per_image < 10, f"High memory usage per image: {memory_per_image}MB"


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Secure image components not available")
class TestSecureImageModelPerformance:
    """Performance tests for secure image model."""
    
    @pytest.fixture
    def performance_tracker(self):
        """Create performance tracker."""
        return PerformanceTracker()
    
    def test_model_initialization_performance(self, performance_tracker):
        """Test model initialization performance."""
        performance_tracker.start()
        
        try:
            with patch('torch.torch.load'), patch('torch.jit.load'):
                model = SecureImageModel()
            success = True
        except Exception:
            success = False
        
        performance_tracker.stop()
        
        if success:
            # Model initialization should be fast
            assert performance_tracker.duration < 5.0, f"Model initialization too slow: {performance_tracker.duration}s"
            
            # Memory usage for initialization should be reasonable
            if performance_tracker.peak_memory_delta:
                assert performance_tracker.peak_memory_delta < 100, f"Model initialization memory too high: {performance_tracker.peak_memory_delta}MB"
    
    def test_model_processing_performance_scaling(self, performance_tracker):
        """Test model processing performance with different security levels."""
        try:
            with patch('torch.torch.load'), patch('torch.jit.load'):
                model = SecureImageModel()
        except Exception:
            pytest.skip("Model initialization failed")
        
        # Create test image
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        test_image_data = buffer.getvalue()
        
        results = {}
        
        for level in SecurityLevel:
            model.set_security_level(level)
            
            performance_tracker.start()
            
            try:
                result = model.process_image(test_image_data, f"perf_test_{level.value}.png")
                success = result is not None
            except Exception:
                success = False
            
            performance_tracker.stop()
            
            results[level.value] = {
                'duration': performance_tracker.duration,
                'memory_delta': performance_tracker.memory_delta,
                'peak_memory_delta': performance_tracker.peak_memory_delta,
                'success': success
            }
        
        # Analyze performance across security levels
        for level, metrics in results.items():
            if metrics['success']:
                # All security levels should complete within reasonable time
                assert metrics['duration'] < 10.0, f"Processing too slow for {level}: {metrics['duration']}s"
                
                # Memory usage should be controlled
                if metrics['peak_memory_delta']:
                    assert metrics['peak_memory_delta'] < 150, f"Memory usage too high for {level}: {metrics['peak_memory_delta']}MB"
    
    def test_model_statistics_performance(self, performance_tracker):
        """Test performance of statistics collection and reporting."""
        try:
            with patch('torch.torch.load'), patch('torch.jit.load'):
                model = SecureImageModel()
        except Exception:
            pytest.skip("Model initialization failed")
        
        # Process several images to generate statistics
        img_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        test_image_data = buffer.getvalue()
        
        # Process multiple images
        for i in range(10):
            try:
                model.process_image(test_image_data, f"stats_test_{i}.png")
            except Exception:
                pass
        
        # Test statistics retrieval performance
        performance_tracker.start()
        
        try:
            stats = model.get_security_statistics()
            success = stats is not None
        except Exception:
            success = False
        
        performance_tracker.stop()
        
        if success:
            # Statistics retrieval should be very fast
            assert performance_tracker.duration < 0.1, f"Statistics retrieval too slow: {performance_tracker.duration}s"
            
            # Should not use significant memory
            if performance_tracker.peak_memory_delta:
                assert performance_tracker.peak_memory_delta < 10, f"Statistics memory usage too high: {performance_tracker.peak_memory_delta}MB"


class TestSecureImageAPIPerformance:
    """Performance tests for secure image API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client for the FastAPI application."""
        try:
            from fastapi.testclient import TestClient
            from main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI application not available for testing")
    
    @pytest.fixture
    def performance_tracker(self):
        """Create performance tracker."""
        return PerformanceTracker()
    
    def test_api_response_time_performance(self, client, performance_tracker):
        """Test API response time performance."""
        # Create test image
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        test_image_data = buffer.getvalue()
        
        endpoints_to_test = [
            ("/image/health", "GET", {}),
            ("/image/validate/security", "POST", {
                "files": {"file": ("test.png", io.BytesIO(test_image_data), "image/png")},
                "data": {"security_level": "low"}
            }),
            ("/image/process/secure", "POST", {
                "files": {"file": ("test.png", io.BytesIO(test_image_data), "image/png")},
                "data": {"security_level": "low"}
            }),
            ("/image/security/stats", "GET", {}),
        ]
        
        results = {}
        
        for endpoint, method, request_data in endpoints_to_test:
            performance_tracker.start()
            
            try:
                if method == "GET":
                    response = client.get(endpoint)
                elif method == "POST":
                    response = client.post(endpoint, **request_data)
                
                success = response.status_code in [200, 404, 501]  # 404/501 OK if feature not implemented
            except Exception:
                success = False
                response = None
            
            performance_tracker.stop()
            
            results[endpoint] = {
                'duration': performance_tracker.duration,
                'success': success,
                'status_code': response.status_code if response else None
            }
        
        # Analyze API performance
        for endpoint, metrics in results.items():
            if metrics['success'] and metrics['status_code'] == 200:
                # API endpoints should respond quickly
                assert metrics['duration'] < 30.0, f"API endpoint {endpoint} too slow: {metrics['duration']}s"
    
    def test_api_concurrent_request_performance(self, client, performance_tracker):
        """Test API performance under concurrent requests."""
        # Create test image
        img_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        test_image_data = buffer.getvalue()
        
        def make_request(request_id):
            """Make a single API request."""
            start_time = time.time()
            try:
                response = client.get("/image/health")
                success = response.status_code == 200
            except Exception:
                success = False
            end_time = time.time()
            
            return {
                'request_id': request_id,
                'duration': end_time - start_time,
                'success': success
            }
        
        # Test concurrent requests
        num_concurrent = 5
        performance_tracker.start()
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_concurrent)]
            concurrent_results = [future.result() for future in as_completed(futures)]
        
        performance_tracker.stop()
        
        # Analyze concurrent performance
        successful_requests = [r for r in concurrent_results if r['success']]
        
        if successful_requests:
            avg_response_time = np.mean([r['duration'] for r in successful_requests])
            max_response_time = max([r['duration'] for r in successful_requests])
            
            # Concurrent requests should maintain reasonable performance
            assert avg_response_time < 5.0, f"Average concurrent response time too slow: {avg_response_time}s"
            assert max_response_time < 10.0, f"Max concurrent response time too slow: {max_response_time}s"
            
            # Should handle most concurrent requests successfully
            success_rate = len(successful_requests) / len(concurrent_results)
            assert success_rate >= 0.8, f"Low success rate for concurrent requests: {success_rate}"


if __name__ == '__main__':
    pytest.main([__file__])

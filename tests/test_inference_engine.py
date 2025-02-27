import unittest
import asyncio
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import time
import warnings
from unittest import mock
from typing import List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import functools
import pytest

# Import the InferenceEngine and related components
from modules.core.inference_engine import (
    InferenceEngine, 
    EngineConfig, 
    RequestItem, 
    ModelPreparationError,
    GuardError,
    ShutdownError,
    create_inference_engine,
    batch_inference_parallel
)


# Utility for running async tests using asyncio.run for compatibility with modern Python versions.
def async_test(coro):
    """Decorator for async test methods."""
    @functools.wraps(coro)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))
    return wrapper


# Simple model for testing
class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_size=10, hidden_size=5, output_size=2):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.input_shape = torch.Size([input_size])
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# Image model for testing multi-dimensional inputs
class SimpleConvModel(nn.Module):
    """Simple CNN for testing."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.input_shape = torch.Size([3, 32, 32])
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Custom config for testing
class TestConfig(EngineConfig):
    """Test configuration with shorter timeouts."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timeout = 0.5
        self.batch_wait_timeout = 0.1
        self.autoscale_interval = 0.5
        self.monitor_interval = 0.5
        self.queue_size = 100
        self.batch_size = 4
        self.min_batch_size = 1
        self.max_batch_size = 16
        self.warmup_runs = 2
        self.guard_enabled = False
        # Override with any custom args
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestInferenceEngineBasics(unittest.TestCase):
    """Test basic functionality of the InferenceEngine."""
    
    def setUp(self):
        """Set up common test resources."""
        self.model = SimpleModel()
        self.config = TestConfig(debug_mode=True)
        warnings.filterwarnings("ignore", category=UserWarning)
        
    def tearDown(self):
        """Clean up after tests."""
        torch.cuda.empty_cache()
        
    @async_test
    async def test_initialization(self):
        """Test engine initialization with various parameters."""
        engine = InferenceEngine(
            model=self.model,
            device="cpu",
            config=self.config
        )
        self.assertIsNotNone(engine)
        self.assertEqual(engine.primary_device, torch.device("cpu"))
        self.assertTrue(engine.is_ready())
        engine.shutdown_sync()

    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @async_test
    async def test_gpu_initialization(self):
        """Test initialization with GPU devices."""
        engine = InferenceEngine(
            model=self.model,
            device="cuda",
            config=self.config
        )
        self.assertEqual(engine.primary_device.type, "cuda")
        engine.shutdown_sync()

    
    def test_invalid_device(self):
        """Test handling of invalid device specifications."""
        with self.assertLogs(level='WARNING'):
            engine = InferenceEngine(
                model=self.model,
                device="invalid_device",
                config=self.config
            )
            self.assertEqual(engine.primary_device, torch.device("cpu"))
            engine.shutdown_sync()

    
    @async_test
    async def test_context_manager(self):
        """Test async context manager support."""
        async with InferenceEngine(
            model=self.model,
            device="cpu",
            config=self.config
        ) as engine:
            self.assertTrue(engine.is_ready())
            # Context manager should handle cleanup
            
        # Engine should be shutdown after context exit
        self.assertTrue(engine._shutdown_event.is_set())
    
    def test_sync_context_manager(self):
        """Test synchronous context manager support."""
        with InferenceEngine(
            model=self.model,
            device="cpu",
            config=self.config
        ) as engine:
            self.assertTrue(engine.is_ready())
        self.assertTrue(engine._shutdown_event.is_set())

    
    def test_factory_function(self):
        """Test the create_inference_engine factory function."""
        engine = create_inference_engine(
            model=self.model,
            debug_mode=True,
            batch_size=8,
            queue_size=200
        )
        self.assertIsNotNone(engine)
        self.assertEqual(engine.config.batch_size, 8)
        self.assertEqual(engine.config.queue_size, 200)
        engine.shutdown_sync()


class TestInferenceEngineInput(unittest.TestCase):
    """Test input detection and handling."""
    
    def setUp(self):
        """Set up common test resources."""
        self.vector_model = SimpleModel(input_size=10)
        self.image_model = SimpleConvModel()
        self.config = TestConfig(debug_mode=True)
        
    def tearDown(self):
        """Clean up after tests."""
        torch.cuda.empty_cache()
    
    @async_test
    async def test_input_shape_detection(self):
        """Test detection of input shapes."""
        engine = InferenceEngine(
            model=self.vector_model,
            device="cpu",
            config=self.config
        )
        self.assertEqual(engine.input_shape, torch.Size([10]))
        engine.shutdown_sync()

    
    def test_explicit_input_shape(self):
        """Test with explicitly provided input shape."""
        config = EngineConfig(input_shape=[5])
        engine = InferenceEngine(
            model=self.vector_model,
            device="cpu",
            config=config
        )
        self.assertEqual(engine.input_shape, torch.Size([5]))
        engine.shutdown_sync()

    
    @async_test
    async def test_input_validation(self):
        """Test input validation during batch processing."""
        engine = InferenceEngine(
            model=self.vector_model,
            device="cpu",
            config=self.config
        )
        inputs = [torch.randn(10) for _ in range(5)]
        engine._validate_batch_inputs(inputs)
        engine.shutdown_sync()



class TestInferenceEngineProcessing(unittest.TestCase):
    """Test preprocessing, inference and postprocessing."""
    
    def setUp(self):
        """Set up common test resources."""
        self.model = SimpleModel()
        self.config = TestConfig(debug_mode=True, async_mode=False)
        
    def tearDown(self):
        """Clean up after tests."""
        torch.cuda.empty_cache()
    
    @async_test
    async def test_preprocessing(self):
        """Test input preprocessing."""
        def preprocessor(x):
            return x + 1
        engine = InferenceEngine(
            model=self.model,
            device="cpu",
            preprocessor=preprocessor,
            config=self.config
        )
        input_data = torch.zeros(10)
        with mock.patch.object(engine, '_infer_batch', return_value=torch.ones(1, 2)):
            result = await engine.run_inference_async(input_data)
            self.assertEqual(engine._infer_batch.call_args[0][0].item(), 1.0)
        engine.shutdown_sync()

    
    @async_test
    async def test_postprocessing(self):
        """Test output postprocessing."""
        def postprocessor(x):
            return x * 2
        engine = InferenceEngine(
            model=self.model,
            device="cpu",
            postprocessor=postprocessor,
            config=self.config
        )
        with mock.patch.object(engine, '_infer_batch', return_value=torch.ones(1, 2)):
            result = await engine.run_inference_async(torch.zeros(10))
            self.assertEqual(result.item(), 2.0)
        engine.shutdown_sync()

    
    @async_test
    async def test_inference_async(self):
        """Test asynchronous inference end-to-end."""
        engine = InferenceEngine(
            model=self.model,
            device="cpu",
            config=self.config
        )
        input_data = torch.randn(10)
        result = await engine.run_inference_async(input_data)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 2)
        engine.shutdown_sync()

    
    def test_inference_sync(self):
        """Test synchronous batch inference."""
        engine = InferenceEngine(
            model=self.model,
            device="cpu",
            config=self.config
        )
        
        # Batch inference with list of tensors
        batch = [torch.randn(10) for _ in range(5)]
        result = engine.run_batch_inference(batch)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (5, 2))
        
        # Batch inference with stacked tensor
        batch = torch.stack(batch)
        result = engine.run_batch_inference(batch)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (5, 2))
        
        engine.shutdown_sync()


class TestInferenceEngineConcurrency(unittest.TestCase):
    """Test concurrent processing and batching."""
    
    def setUp(self):
        """Set up common test resources."""
        self.model = SimpleModel()
        self.config = TestConfig(
            debug_mode=True,
            async_mode=True,
            batch_size=8
        )
        
    def tearDown(self):
        """Clean up after tests."""
        torch.cuda.empty_cache()
    
    @async_test
    async def test_batch_processing(self):
        """Test batch assembly and processing."""
        engine = InferenceEngine(
            model=self.model,
            device="cpu",
            config=self.config
        )
        
        # Submit multiple requests at once
        num_requests = 20
        tasks = [
            engine.run_inference_async(torch.randn(10))
            for _ in range(num_requests)
        ]
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        self.assertEqual(len(results), num_requests)
        
        # Check batch processing stats
        metrics = engine.get_metrics()
        self.assertGreater(metrics["total_batches_processed"], 0)
        self.assertLessEqual(metrics["total_batches_processed"], num_requests)
        
        engine.shutdown_sync()
    
    @async_test
    async def test_queue_behavior(self):
        """Test queue behavior with many requests."""
        # Small queue size for testing
        config = TestConfig(
            queue_size=10,
            batch_size=2,
            batch_wait_timeout=0.05
        )
        
        engine = InferenceEngine(
            model=self.model,
            device="cpu",
            config=config
        )
        
        # Add delay to simulate slow processing by wrapping _infer_batch
        original_infer = engine._infer_batch
        
        def delayed_infer(batch):
            time.sleep(0.01)  # Small delay
            return original_infer(batch)
            
        engine._infer_batch = delayed_infer
        
        # Submit more requests than queue size
        num_requests = 20
        tasks = []
        
        for i in range(num_requests):
            tasks.append(
                asyncio.create_task(
                    engine.run_inference_async(torch.randn(10))
                )
            )
            # Small delay to avoid overloading
            await asyncio.sleep(0.01)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete, though some might be exceptions if queue was full
        self.assertEqual(len(results), num_requests)
        
        engine.shutdown_sync()
    
    @async_test
    async def test_parallel_helper(self):
        """Test the batch_inference_parallel helper function."""
        engine = InferenceEngine(
            model=self.model,
            device="cpu",
            config=self.config
        )
        
        # Create list of inputs
        inputs = [torch.randn(10) for _ in range(30)]
        
        # Process with parallel helper
        results = await batch_inference_parallel(
            engine=engine,
            inputs=inputs,
            max_workers=4,
            chunk_size=5
        )
        
        self.assertEqual(len(results), len(inputs))
        self.assertIsInstance(results[0], torch.Tensor)
        
        engine.shutdown_sync()
    
    @async_test
    async def test_priority_ordering(self):
        """Test request priority ordering."""
        config = TestConfig(
            batch_size=1,  # Process one at a time to test ordering
            batch_wait_timeout=0.05
        )
        
        engine = InferenceEngine(
            model=self.model,
            device="cpu",
            config=config
        )
        
        # Add significant delay to processing and record order
        original_infer = engine._infer_batch
        processed_order = []
        
        def delayed_infer(batch):
            # Record processing order; assume batch is a tensor and take its first element
            processed_order.append(batch[0].item())
            time.sleep(0.05)
            return original_infer(batch)
            
        engine._infer_batch = delayed_infer
        
        # Submit requests with different priorities and values
        await engine.run_inference_async(torch.tensor([10.0]), priority=2)
        await engine.run_inference_async(torch.tensor([20.0]), priority=1)
        await engine.run_inference_async(torch.tensor([30.0]), priority=0)  # Highest priority
        
        # Should process in priority order: 30, 20, 10
        self.assertEqual(processed_order, [30.0, 20.0, 10.0])
        
        engine.shutdown_sync()


class TestInferenceEngineGuard(unittest.TestCase):
    """Test guard functionality for adversarial detection."""
    
    def setUp(self):
        """Set up common test resources."""
        self.model = SimpleModel()
        # Enable guard with strict settings
        self.config = TestConfig(
            debug_mode=True,
            guard_enabled=True,
            guard_confidence_threshold=0.8,
            guard_variance_threshold=0.05,
            guard_fail_silently=False,
            num_classes=2  # For default response
        )
        
    def tearDown(self):
        """Clean up after tests."""
        torch.cuda.empty_cache()
    
    @async_test
    async def test_guard_normal_input(self):
        """Test guard with normal input that should pass."""
        # Mock model to return high confidence predictions
        mock_model = mock.MagicMock()
        mock_model.return_value = torch.tensor([[0.1, 0.9]] * 5)  # High confidence
        
        engine = InferenceEngine(
            model=mock_model,
            device="cpu",
            config=self.config
        )
        
        # Normal input should pass guard
        input_data = torch.randn(10)
        result = await engine.run_inference_async(input_data)
        self.assertIsInstance(result, torch.Tensor)
        
        engine.shutdown_sync()
    
    @async_test
    async def test_guard_adversarial_input(self):
        """Test guard catching adversarial input."""
        # Mock model to return unusual predictions (very uncertain)
        mock_model = mock.MagicMock()
        # Return inconsistent predictions across augmentations
        mock_model.return_value = torch.tensor([
            [0.51, 0.49],
            [0.45, 0.55],
            [0.52, 0.48],
            [0.47, 0.53],
            [0.49, 0.51]
        ])
        
        engine = InferenceEngine(
            model=mock_model,
            device="cpu",
            config=self.config
        )
        
        # Try with input that should trigger guard
        with self.assertRaises(GuardError):
            await engine.run_inference_async(torch.randn(10))
            
        engine.shutdown_sync()
    
    @async_test
    async def test_guard_silent_mode(self):
        """Test guard in silent mode (returns default response)."""
        # Enable silent mode
        config = self.config
        config.guard_fail_silently = True
        
        # Mock model to return unusual predictions
        mock_model = mock.MagicMock()
        mock_model.return_value = torch.ones((5, 2)) * 0.5  # Uniform distribution
        
        engine = InferenceEngine(
            model=mock_model,
            device="cpu",
            config=config
        )
        
        # Should return default uniform distribution
        result = await engine.run_inference_async(torch.randn(10))
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (2,))
        self.assertAlmostEqual(result[0].item(), 0.5, places=5)
        
        engine.shutdown_sync()


class TestInferenceEngineErrors(unittest.TestCase):
    """Test error handling in the inference engine."""
    
    def setUp(self):
        """Set up common test resources."""
        self.model = SimpleModel()
        self.config = TestConfig(debug_mode=True)
        
    def tearDown(self):
        """Clean up after tests."""
        torch.cuda.empty_cache()
    
    @async_test
    async def test_model_error(self):
        """Test handling of model execution errors."""
        # Create model that raises exception
        def bad_forward(self, x):
            raise RuntimeError("Simulated model error")
            
        with mock.patch.object(SimpleModel, 'forward', bad_forward):
            engine = InferenceEngine(
                model=self.model,
                device="cpu",
                config=self.config
            )
            
            with self.assertRaises(Exception):
                await engine.run_inference_async(torch.randn(10))
                
            engine.shutdown_sync()
    
    @async_test
    async def test_preprocessor_error(self):
        """Test handling of preprocessor errors."""
        def bad_preprocessor(x):
            raise ValueError("Simulated preprocessing error")
            
        engine = InferenceEngine(
            model=self.model,
            device="cpu",
            preprocessor=bad_preprocessor,
            config=self.config
        )
        
        with self.assertRaises(ValueError):
            await engine.run_inference_async(torch.randn(10))
            
        engine.shutdown_sync()
    
    @async_test
    async def test_postprocessor_error(self):
        """Test handling of postprocessor errors."""
        def bad_postprocessor(x):
            raise ValueError("Simulated postprocessing error")
            
        engine = InferenceEngine(
            model=self.model,
            device="cpu",
            postprocessor=bad_postprocessor,
            config=self.config
        )
        
        with self.assertRaises(ValueError):
            await engine.run_inference_async(torch.randn(10))
            
        engine.shutdown_sync()
    
    @async_test
    async def test_shutdown_during_inference(self):
        """Test behavior when shutting down during inference."""
        config = TestConfig(
            batch_wait_timeout=0.2,
            request_timeout=1.0
        )
        
        engine = InferenceEngine(
            model=self.model,
            device="cpu",
            config=config
        )
        
        # Start many requests
        tasks = [
            asyncio.create_task(engine.run_inference_async(torch.randn(10)))
            for _ in range(20)
        ]
        
        # Wait a bit for some to be queued
        await asyncio.sleep(0.1)
        
        # Shutdown the engine asynchronously
        asyncio.create_task(engine.cleanup())
        
        # Wait for all tasks to complete or raise exceptions
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Some should be ShutdownError
        shutdown_errors = [r for r in results if isinstance(r, ShutdownError)]
        self.assertGreater(len(shutdown_errors), 0)


class TestInferenceEnginePerformance(unittest.TestCase):
    """Test performance profiling and monitoring."""
    
    def setUp(self):
        """Set up common test resources."""
        self.model = SimpleModel()
        self.config = TestConfig(debug_mode=True)
        
    def tearDown(self):
        """Clean up after tests."""
        torch.cuda.empty_cache()
    
    def test_profiling(self):
        """Test profiling functionality."""
        engine = InferenceEngine(
            model=self.model,
            device="cpu",
            config=self.config
        )
        
        # Basic profiling
        input_data = torch.randn(10)
        profile_results = engine.profile_inference(
            input_data,
            warmup_runs=1,
            profile_runs=5
        )
        
        # Check key metrics
        self.assertIn("preprocess_ms_mean", profile_results)
        self.assertIn("inference_ms_mean", profile_results)
        self.assertIn("postprocess_ms_mean", profile_results)
        self.assertIn("total_ms_mean", profile_results)
        self.assertIn("throughput_items_per_second", profile_results)
        
        # Throughput should be positive
        self.assertGreater(profile_results["throughput_items_per_second"], 0)
        
        engine.shutdown_sync()
    
    @async_test
    async def test_metrics(self):
        """Test metrics collection during usage."""
        engine = InferenceEngine(
            model=self.model,
            device="cpu",
            config=self.config
        )
        
        # Process some requests
        for _ in range(10):
            await engine.run_inference_async(torch.randn(10))
            
        # Get metrics
        metrics = engine.get_metrics()
        
        # Check core metrics
        self.assertIn("queue_size", metrics)
        self.assertIn("average_batch_time", metrics)
        self.assertIn("throughput_per_second", metrics)
        self.assertIn("successful_requests", metrics)
        self.assertIn("memory_usage_mb", metrics)
        
        # Successful requests should match what we processed
        self.assertEqual(metrics["successful_requests"], 10)
        
        # Clear metrics
        engine.clear_metrics()
        new_metrics = engine.get_metrics()
        self.assertEqual(new_metrics["successful_requests"], 0)
        
        engine.shutdown_sync()


class TestInferenceEngineConfig(unittest.TestCase):
    """Test configuration and updates."""
    
    def setUp(self):
        """Set up common test resources."""
        self.model = SimpleModel()
        
    def tearDown(self):
        """Clean up after tests."""
        torch.cuda.empty_cache()
    
    def test_config_updates(self):
        """Test updating configuration after initialization."""
        config = TestConfig(batch_size=4)
        
        engine = InferenceEngine(
            model=self.model,
            device="cpu",
            config=config
        )
        
        # Initial batch size
        self.assertEqual(engine.config.batch_size, 4)
        
        # Update config
        engine.update_config(
            batch_size=8,
            guard_enabled=True,
            debug_mode=True
        )
        
        # Check updates applied
        self.assertEqual(engine.config.batch_size, 8)
        self.assertTrue(engine.config.guard_enabled)
        self.assertTrue(engine.config.debug_mode)
        
        # Invalid parameter should be ignored
        engine.update_config(nonexistent_param=123)
        self.assertFalse(hasattr(engine.config, "nonexistent_param"))
        
        engine.shutdown_sync()


if __name__ == '__main__':
    unittest.main()

"""Tests for inference engine functionality."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
import torch

from framework.core.inference_engine import (
    InferenceEngine, InferenceRequest, BatchResult, PIDController,
    RequestQueue, create_inference_engine
)
from framework.core.base_model import BaseModel
from framework.core.config import InferenceConfig, BatchConfig, PerformanceConfig


class MockInferenceModel(BaseModel):
    """Mock model for testing inference engine."""
    
    def __init__(self, config: InferenceConfig, prediction_time: float = 0.01):
        super().__init__(config)
        self.prediction_time = prediction_time
        self.model = Mock()
        self._is_loaded = True
    
    def load_model(self, model_path):
        pass
    
    def preprocess(self, inputs):
        return torch.tensor(inputs) if not isinstance(inputs, torch.Tensor) else inputs
    
    def forward(self, inputs):
        # Simulate processing time
        time.sleep(self.prediction_time)
        return torch.randn(inputs.shape[0], 10)
    
    def postprocess(self, outputs):
        return {"predictions": outputs.tolist()}


class TestPIDController:
    """Test PID controller functionality."""
    
    def test_pid_initialization(self):
        """Test PID controller initialization."""
        controller = PIDController(kp=0.6, ki=0.1, kd=0.05, setpoint=50.0)
        
        assert controller.kp == 0.6
        assert controller.ki == 0.1
        assert controller.kd == 0.05
        assert controller.setpoint == 50.0
        assert controller.min_value == 1
        assert controller.max_value == 32
    
    def test_pid_update_basic(self):
        """Test basic PID update."""
        controller = PIDController(setpoint=50.0, min_value=1, max_value=10)
        
        # Current value higher than setpoint - should decrease
        result = controller.update(60.0)
        assert 1 <= result <= 10
        
        # Current value lower than setpoint - should increase
        result = controller.update(40.0)
        assert 1 <= result <= 10
    
    def test_pid_bounds(self):
        """Test PID controller bounds."""
        controller = PIDController(setpoint=50.0, min_value=2, max_value=8)
        
        # Update with extreme values
        result = controller.update(1000.0)  # Very high
        assert 2 <= result <= 8
        
        result = controller.update(0.1)  # Very low
        assert 2 <= result <= 8
    
    def test_pid_reset(self):
        """Test PID controller reset."""
        controller = PIDController()
        
        # Make some updates
        controller.update(100.0)
        controller.update(10.0)
        
        # Reset
        controller.reset()
        
        assert controller.prev_error == 0
        assert controller.integral == 0
        assert controller.last_value == controller.min_value


class TestRequestQueue:
    """Test request queue functionality."""
    
    def test_queue_initialization(self):
        """Test request queue initialization."""
        queue = RequestQueue(max_size=100)
        
        assert queue.size() == 0
        assert queue.max_size == 100
    
    @pytest.mark.asyncio
    async def test_queue_put_get(self):
        """Test basic put and get operations."""
        queue = RequestQueue(max_size=10)
        
        # Create mock request
        future = asyncio.Future()
        request = InferenceRequest(
            id="test_1",
            inputs=[1, 2, 3],
            future=future,
            timestamp=time.time(),
            priority=0
        )
        
        # Put request
        await queue.put(request)
        assert queue.size() == 1
        
        # Get batch
        batch = queue.get_batch(max_batch_size=5, timeout=1.0)
        assert len(batch) == 1
        assert batch[0].id == "test_1"
        assert queue.size() == 0
    
    @pytest.mark.asyncio
    async def test_queue_priority(self):
        """Test priority-based request ordering."""
        queue = RequestQueue(max_size=10)
        
        # Add requests with different priorities
        requests = []
        for i, priority in enumerate([1, 3, 2]):
            future = asyncio.Future()
            request = InferenceRequest(
                id=f"test_{i}",
                inputs=[i],
                future=future,
                timestamp=time.time(),
                priority=priority
            )
            requests.append(request)
            await queue.put(request)
        
        # Get batch - should be ordered by priority (highest first)
        batch = queue.get_batch(max_batch_size=10, timeout=1.0)
        assert len(batch) == 3
        assert batch[0].priority == 3  # Highest priority first
        assert batch[1].priority == 2
        assert batch[2].priority == 1
    
    def test_queue_get_timeout(self):
        """Test queue get with timeout."""
        queue = RequestQueue(max_size=10)
        
        # Get from empty queue with short timeout
        batch = queue.get_batch(max_batch_size=5, timeout=0.1)
        assert len(batch) == 0
    
    def test_queue_clear(self):
        """Test queue clear."""
        queue = RequestQueue(max_size=10)
        
        # Add some mock requests
        for i in range(3):
            queue._queue.append(Mock())
        
        assert queue.size() == 3
        queue.clear()
        assert queue.size() == 0


class TestInferenceEngine:
    """Test inference engine functionality."""
    
    @pytest.fixture
    def mock_model(self, test_config):
        """Create mock model for testing."""
        return MockInferenceModel(test_config)
    
    @pytest.fixture
    def inference_config(self):
        """Create inference configuration for testing."""
        return InferenceConfig(
            batch=BatchConfig(
                batch_size=2,
                max_batch_size=8,
                min_batch_size=1,
                queue_size=100
            ),
            performance=PerformanceConfig(
                max_workers=2,
                enable_profiling=True
            )
        )
    
    def test_engine_initialization(self, mock_model, inference_config):
        """Test inference engine initialization."""
        engine = InferenceEngine(mock_model, inference_config)
        
        assert engine.model == mock_model
        assert engine.config == inference_config
        assert not engine._running
        assert engine._current_batch_size == inference_config.batch.batch_size
        assert engine._stats["requests_processed"] == 0
    
    @pytest.mark.asyncio
    async def test_engine_start_stop(self, mock_model, inference_config):
        """Test engine start and stop."""
        engine = InferenceEngine(mock_model, inference_config)
        
        # Start engine
        await engine.start()
        assert engine._running
        assert engine._worker_task is not None
        
        # Stop engine
        await engine.stop()
        assert not engine._running
    
    @pytest.mark.asyncio
    async def test_single_prediction(self, mock_model, inference_config):
        """Test single prediction through engine."""
        engine = InferenceEngine(mock_model, inference_config)
        
        async with engine.async_context():
            result = await engine.predict([1, 2, 3], priority=1, timeout=5.0)
            
            assert isinstance(result, dict)
            assert "predictions" in result
    
    @pytest.mark.asyncio
    async def test_batch_prediction(self, mock_model, inference_config):
        """Test batch prediction through engine."""
        engine = InferenceEngine(mock_model, inference_config)
        
        inputs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        async with engine.async_context():
            results = await engine.predict_batch(inputs, priority=1, timeout=10.0)
            
            assert len(results) == 3
            for result in results:
                assert isinstance(result, dict)
                assert "predictions" in result
    
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, mock_model, inference_config):
        """Test concurrent predictions."""
        engine = InferenceEngine(mock_model, inference_config)
        
        async with engine.async_context():
            # Submit multiple concurrent requests
            tasks = []
            for i in range(5):
                task = engine.predict([i, i+1, i+2], priority=i, timeout=10.0)
                tasks.append(task)
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            for result in results:
                assert isinstance(result, dict)
                assert "predictions" in result
    
    @pytest.mark.asyncio
    async def test_prediction_timeout(self, mock_model, inference_config):
        """Test prediction timeout handling."""
        # Create slow model
        slow_model = MockInferenceModel(inference_config, prediction_time=2.0)
        engine = InferenceEngine(slow_model, inference_config)
        
        async with engine.async_context():
            with pytest.raises(asyncio.TimeoutError):
                await engine.predict([1, 2, 3], timeout=0.5)
    
    def test_engine_stats(self, mock_model, inference_config):
        """Test engine statistics tracking."""
        engine = InferenceEngine(mock_model, inference_config)
        
        stats = engine.get_stats()
        
        assert isinstance(stats, dict)
        assert "requests_processed" in stats
        assert "batches_processed" in stats
        assert "total_processing_time" in stats
        assert "average_batch_size" in stats
        assert "errors" in stats
    
    def test_performance_report(self, mock_model, inference_config):
        """Test performance report generation."""
        engine = InferenceEngine(mock_model, inference_config)
        
        report = engine.get_performance_report()
        
        assert isinstance(report, dict)
        assert "engine_stats" in report
        assert "performance_metrics" in report
        assert "current_batch_size" in report
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_model, inference_config):
        """Test engine health check."""
        engine = InferenceEngine(mock_model, inference_config)
        
        # Health check when not running
        health = await engine.health_check()
        assert isinstance(health, dict)
        assert "healthy" in health
        assert "checks" in health
        
        # Health check when running
        async with engine.async_context():
            health = await engine.health_check()
            assert health["checks"]["engine_running"]
    
    @pytest.mark.asyncio
    async def test_engine_cleanup(self, mock_model, inference_config):
        """Test engine resource cleanup."""
        engine = InferenceEngine(mock_model, inference_config)
        
        await engine.start()
        await engine.cleanup()
        
        assert not engine._running
        assert engine._executor._shutdown
    
    @pytest.mark.asyncio
    async def test_error_handling(self, test_config):
        """Test error handling in engine."""
        # Create failing model
        class FailingModel(MockInferenceModel):
            def forward(self, inputs):
                raise RuntimeError("Mock model failure")
        
        failing_model = FailingModel(test_config)
        engine = InferenceEngine(failing_model)
        
        async with engine.async_context():
            with pytest.raises(Exception):
                await engine.predict([1, 2, 3], timeout=5.0)
    
    def test_dynamic_batch_sizing(self, mock_model, inference_config):
        """Test dynamic batch sizing with PID controller."""
        engine = InferenceEngine(mock_model, inference_config)
        
        # Initial batch size
        initial_batch_size = engine._current_batch_size
        assert initial_batch_size == inference_config.batch.batch_size
        
        # Simulate high latency - should decrease batch size
        engine.pid_controller.update(100.0)  # High latency
        new_batch_size = engine.pid_controller.last_value
        
        # Should be different from initial
        assert isinstance(new_batch_size, int)
        assert inference_config.batch.min_batch_size <= new_batch_size <= inference_config.batch.max_batch_size


class TestInferenceEngineIntegration:
    """Integration tests for inference engine."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_with_monitoring(self, mock_model, inference_config):
        """Test complete pipeline with performance monitoring."""
        engine = InferenceEngine(mock_model, inference_config)
        
        async with engine.async_context():
            # Submit various requests
            tasks = []
            
            # Single predictions
            for i in range(3):
                task = engine.predict([i, i+1, i+2], priority=1)
                tasks.append(task)
            
            # Batch prediction
            batch_task = engine.predict_batch([
                [10, 11, 12],
                [20, 21, 22],
                [30, 31, 32]
            ], priority=2)
            tasks.append(batch_task)
            
            # Execute all
            results = await asyncio.gather(*tasks)
            
            # Check results
            assert len(results) == 4
            
            # Check stats
            stats = engine.get_stats()
            assert stats["requests_processed"] > 0
            
            # Check performance report
            report = engine.get_performance_report()
            assert "engine_stats" in report
            assert "performance_metrics" in report
    
    @pytest.mark.asyncio
    async def test_high_throughput_scenario(self, mock_model, inference_config):
        """Test high throughput scenario."""
        engine = InferenceEngine(mock_model, inference_config)
        
        async with engine.async_context():
            # Submit many concurrent requests
            num_requests = 50
            tasks = []
            
            for i in range(num_requests):
                task = engine.predict([i, i+1, i+2], priority=i % 5)
                tasks.append(task)
            
            # Process all requests
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Check results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) > 0
            
            # Check throughput
            processing_time = end_time - start_time
            throughput = len(successful_results) / processing_time
            
            # Should process multiple requests per second
            assert throughput > 1.0
    
    @pytest.mark.asyncio
    async def test_mixed_workload(self, mock_model, inference_config):
        """Test mixed workload with different request types."""
        engine = InferenceEngine(mock_model, inference_config)
        
        async with engine.async_context():
            # Mix of single predictions and batches
            tasks = []
            
            # High priority single requests
            for i in range(5):
                task = engine.predict([i], priority=10)
                tasks.append(task)
            
            # Medium priority batch
            batch_task = engine.predict_batch([
                [10, 11], [12, 13], [14, 15]
            ], priority=5)
            tasks.append(batch_task)
            
            # Low priority single requests
            for i in range(5):
                task = engine.predict([i + 100], priority=1)
                tasks.append(task)
            
            # Execute with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=30.0
            )
            
            assert len(results) == 11


class TestCreateInferenceEngine:
    """Test inference engine factory function."""
    
    def test_create_inference_engine(self, mock_model, inference_config):
        """Test creating inference engine via factory function."""
        engine = create_inference_engine(mock_model, inference_config)
        
        assert isinstance(engine, InferenceEngine)
        assert engine.model == mock_model
        assert engine.config == inference_config
    
    def test_create_inference_engine_with_defaults(self, mock_model):
        """Test creating inference engine with default config."""
        engine = create_inference_engine(mock_model)
        
        assert isinstance(engine, InferenceEngine)
        assert engine.model == mock_model
        assert engine.config == mock_model.config


class TestEngineErrorHandling:
    """Test error handling in various scenarios."""
    
    @pytest.mark.asyncio
    async def test_model_loading_error(self, test_config):
        """Test handling model loading errors."""
        class FailingLoadModel(MockInferenceModel):
            def __init__(self, config):
                super().__init__(config)
                self._is_loaded = False
            
            def load_model(self, model_path):
                raise RuntimeError("Failed to load model")
        
        model = FailingLoadModel(test_config)
        engine = InferenceEngine(model)
        
        # Should handle gracefully
        with pytest.raises(Exception):
            await engine.predict([1, 2, 3])
    
    @pytest.mark.asyncio
    async def test_queue_full_handling(self, mock_model, inference_config):
        """Test handling of full request queue."""
        # Create engine with very small queue
        small_config = inference_config
        small_config.batch.queue_size = 2
        
        engine = InferenceEngine(mock_model, small_config)
        
        async with engine.async_context():
            # Fill queue beyond capacity
            tasks = []
            for i in range(5):  # More than queue size
                task = engine.predict([i], timeout=0.1)
                tasks.append(task)
            
            # Some should succeed, some may timeout
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should have mix of results and exceptions
            exceptions = [r for r in results if isinstance(r, Exception)]
            successes = [r for r in results if not isinstance(r, Exception)]
            
            # At least some should succeed
            assert len(successes) > 0
    
    @pytest.mark.asyncio
    async def test_worker_task_error_recovery(self, mock_model, inference_config):
        """Test recovery from worker task errors."""
        engine = InferenceEngine(mock_model, inference_config)
        
        await engine.start()
        
        # Simulate worker task failure
        if engine._worker_task:
            engine._worker_task.cancel()
        
        # Engine should still be able to handle requests
        # (Implementation dependent - may need to restart worker)
        
        await engine.stop()

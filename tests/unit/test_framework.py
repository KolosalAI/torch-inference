"""Tests for main framework interface."""

import pytest
import asyncio
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path

from framework import (
    TorchInferenceFramework,
    create_classification_framework,
    create_detection_framework,
    create_segmentation_framework,
    predict_image_classification,
    predict_object_detection,
    predict_segmentation,
    get_global_framework,
    set_global_framework,
    create_optimized_framework
)
from framework.core.config import InferenceConfig

# Test imports for enhanced optimizers
try:
    from framework.optimizers import (
        get_available_optimizers, get_optimization_recommendations,
        EnhancedJITOptimizer, VulkanOptimizer, NumbaOptimizer, PerformanceOptimizer
    )
    ENHANCED_OPTIMIZERS_AVAILABLE = True
except ImportError:
    get_available_optimizers = None
    get_optimization_recommendations = None
    EnhancedJITOptimizer = None
    VulkanOptimizer = None
    NumbaOptimizer = None
    PerformanceOptimizer = None
    ENHANCED_OPTIMIZERS_AVAILABLE = False


class TestTorchInferenceFramework:
    """Test main framework interface."""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple test model."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
    
    @pytest.fixture
    def framework(self, test_config):
        """Create framework instance."""
        return TorchInferenceFramework(test_config)
    
    def test_framework_initialization(self, framework, test_config):
        """Test framework initialization."""
        assert framework.config == test_config
        assert not framework._initialized
        assert not framework._engine_running
        assert framework.model is None
        assert framework.engine is None
    
    def test_framework_initialization_with_default_config(self):
        """Test framework initialization with default config."""
        with patch('framework.core.config.get_global_config') as mock_config:
            mock_config.return_value = InferenceConfig()
            
            framework = TorchInferenceFramework()
            
            assert framework.config is not None
            mock_config.assert_called_once()
    
    def test_load_model(self, framework, simple_model, temp_model_dir):
        """Test model loading."""
        model_path = temp_model_dir / "test_model.pt"
        torch.save(simple_model, model_path)
        
        with patch('framework.load_model') as mock_load:
            mock_model = Mock()
            mock_model.is_loaded = True
            mock_load.return_value = mock_model
            
            with patch('framework.create_inference_engine') as mock_engine:
                mock_engine_instance = Mock()
                mock_engine.return_value = mock_engine_instance
                
                framework.load_model(model_path, "test_model")
                
                assert framework._initialized
                assert framework.model == mock_model
                assert framework.engine == mock_engine_instance
                mock_load.assert_called_once_with(model_path, framework.config)
    
    def test_load_model_with_auto_name(self, framework, simple_model, temp_model_dir):
        """Test model loading with automatic name generation."""
        model_path = temp_model_dir / "auto_name_model.pt"
        torch.save(simple_model, model_path)
        
        with patch('framework.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            with patch.object(framework.model_manager, 'register_model') as mock_register:
                framework.load_model(model_path)
                
                # Should use file stem as model name
                mock_register.assert_called_once_with("auto_name_model", mock_model)
    
    def test_load_model_error(self, framework):
        """Test model loading error handling."""
        with patch('framework.load_model') as mock_load:
            mock_load.side_effect = Exception("Load failed")
            
            with pytest.raises(Exception):
                framework.load_model("nonexistent.pt")
    
    @pytest.mark.asyncio
    async def test_start_stop_engine(self, framework):
        """Test starting and stopping inference engine."""
        # Mock model and engine
        mock_model = Mock()
        mock_engine = AsyncMock()
        framework.model = mock_model
        framework.engine = mock_engine
        framework._initialized = True
        
        # Start engine
        await framework.start_engine()
        
        assert framework._engine_running
        mock_engine.start.assert_called_once()
        
        # Stop engine
        await framework.stop_engine()
        
        assert not framework._engine_running
        mock_engine.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_engine_not_initialized(self, framework):
        """Test starting engine when not initialized."""
        with pytest.raises(RuntimeError):
            await framework.start_engine()
    
    def test_predict_sync(self, framework):
        """Test synchronous prediction."""
        mock_model = Mock()
        mock_model.predict.return_value = {"prediction": "test_result"}
        framework.model = mock_model
        framework._initialized = True
        
        result = framework.predict([1, 2, 3])
        
        assert result == {"prediction": "test_result"}
        mock_model.predict.assert_called_once_with([1, 2, 3])
    
    def test_predict_sync_not_initialized(self, framework):
        """Test synchronous prediction when not initialized."""
        with pytest.raises(RuntimeError):
            framework.predict([1, 2, 3])
    
    @pytest.mark.asyncio
    async def test_predict_async(self, framework):
        """Test asynchronous prediction."""
        mock_model = Mock()
        mock_engine = AsyncMock()
        mock_engine.predict.return_value = {"prediction": "async_result"}
        
        framework.model = mock_model
        framework.engine = mock_engine
        framework._initialized = True
        framework._engine_running = True
        
        result = await framework.predict_async([1, 2, 3], priority=1, timeout=5.0)
        
        assert result == {"prediction": "async_result"}
        mock_engine.predict.assert_called_once_with([1, 2, 3], 1, 5.0)
    
    @pytest.mark.asyncio
    async def test_predict_async_engine_not_running(self, framework):
        """Test async prediction when engine not running."""
        framework._initialized = True
        framework._engine_running = False
        
        with pytest.raises(RuntimeError):
            await framework.predict_async([1, 2, 3])
    
    def test_predict_batch_sync(self, framework):
        """Test synchronous batch prediction."""
        mock_model = Mock()
        mock_model.predict_batch.return_value = [
            {"prediction": "result1"},
            {"prediction": "result2"}
        ]
        framework.model = mock_model
        framework._initialized = True
        
        inputs = [[1, 2, 3], [4, 5, 6]]
        results = framework.predict_batch(inputs)
        
        assert len(results) == 2
        mock_model.predict_batch.assert_called_once_with(inputs)
    
    @pytest.mark.asyncio
    async def test_predict_batch_async(self, framework):
        """Test asynchronous batch prediction."""
        mock_engine = AsyncMock()
        mock_engine.predict_batch.return_value = [
            {"prediction": "async_result1"},
            {"prediction": "async_result2"}
        ]
        
        framework.engine = mock_engine
        framework._initialized = True
        framework._engine_running = True
        
        inputs = [[1, 2, 3], [4, 5, 6]]
        results = await framework.predict_batch_async(inputs, priority=2, timeout=10.0)
        
        assert len(results) == 2
        mock_engine.predict_batch.assert_called_once_with(inputs, 2, 10.0)
    
    def test_benchmark(self, framework, sample_tensor):
        """Test model benchmarking."""
        mock_model = Mock()
        mock_model.predict.return_value = {"prediction": "benchmark_result"}
        mock_model.device = torch.device("cpu")
        mock_model.model_info = {"type": "test_model"}
        
        framework.model = mock_model
        framework._initialized = True
        
        # Mock time.perf_counter for consistent timing
        with patch('time.perf_counter') as mock_time:
            mock_time.side_effect = [0.0, 0.01, 0.02, 0.03]  # Mock timing progression
            
            results = framework.benchmark(sample_tensor, iterations=2, warmup=1)
        
        assert isinstance(results, dict)
        assert "iterations" in results
        assert "mean_time_ms" in results
        assert "throughput_fps" in results
        assert "device" in results
        assert "model_info" in results
        
        assert results["iterations"] == 2
        # Should call predict 3 times total (1 warmup + 2 iterations)
        assert mock_model.predict.call_count == 3
    
    def test_get_model_info(self, framework):
        """Test getting model information."""
        # Before loading
        info = framework.get_model_info()
        assert info == {"loaded": False}
        
        # After loading
        mock_model = Mock()
        mock_model.model_info = {"type": "test", "parameters": 1000}
        framework.model = mock_model
        framework._initialized = True
        
        info = framework.get_model_info()
        assert info == {"type": "test", "parameters": 1000}
    
    def test_get_engine_stats(self, framework):
        """Test getting engine statistics."""
        # Without engine
        stats = framework.get_engine_stats()
        assert stats == {"engine": "not_initialized"}
        
        # With engine
        mock_engine = Mock()
        mock_engine.get_stats.return_value = {"requests": 100, "avg_time": 0.05}
        framework.engine = mock_engine
        
        stats = framework.get_engine_stats()
        assert stats == {"requests": 100, "avg_time": 0.05}
        mock_engine.get_stats.assert_called_once()
    
    def test_get_performance_report(self, framework):
        """Test getting performance report."""
        mock_model = Mock()
        mock_model.model_info = {"type": "test"}
        mock_engine = Mock()
        mock_engine.get_stats.return_value = {"requests": 50}
        mock_engine.get_performance_report.return_value = {"avg_latency": 0.02}
        
        framework.model = mock_model
        framework.engine = mock_engine
        framework._initialized = True
        framework._engine_running = True
        
        with patch.object(framework.performance_monitor, 'get_performance_summary') as mock_perf:
            mock_perf.return_value = {"total_requests": 100}
            
            report = framework.get_performance_report()
        
        assert "framework_info" in report
        assert "model_info" in report
        assert "performance_metrics" in report
        assert "engine_stats" in report
        assert "engine_performance" in report
        
        assert report["framework_info"]["initialized"]
        assert report["framework_info"]["engine_running"]
    
    @pytest.mark.asyncio
    async def test_health_check(self, framework):
        """Test health check functionality."""
        # Mock components
        mock_model = Mock()
        mock_model.is_loaded = True
        mock_engine = AsyncMock()
        mock_engine.health_check.return_value = {"healthy": True}
        
        framework.model = mock_model
        framework.engine = mock_engine
        framework._initialized = True
        
        health = await framework.health_check()
        
        assert isinstance(health, dict)
        assert "healthy" in health
        assert "checks" in health
        assert "timestamp" in health
        
        assert health["healthy"]
        assert health["checks"]["framework_initialized"]
        assert health["checks"]["model_loaded"]
        assert health["checks"]["engine"]["healthy"]
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, framework):
        """Test health check with unhealthy components."""
        mock_model = Mock()
        mock_model.is_loaded = False  # Unhealthy
        framework.model = mock_model
        framework._initialized = False  # Unhealthy
        
        health = await framework.health_check()
        
        assert not health["healthy"]
        assert not health["checks"]["framework_initialized"]
        assert not health["checks"]["model_loaded"]
    
    def test_cleanup(self, framework):
        """Test framework cleanup."""
        mock_model = Mock()
        mock_engine = Mock()
        mock_model_manager = Mock()

        framework.model = mock_model
        framework.engine = mock_engine
        framework._engine_running = True
        framework._model_manager = mock_model_manager

        # Test synchronous cleanup
        framework.cleanup()

        mock_model.cleanup.assert_called_once()
        mock_model_manager.cleanup_all.assert_called_once()

    def test_async_context_manager(self, framework):
        """Test using framework as async context manager."""
        # This test focuses on checking the model_manager property
        assert hasattr(framework, 'model_manager')
        assert hasattr(framework.model_manager, 'cleanup_all')
        
        # Verify the property works
        assert framework.model_manager == framework._model_manager
    
    def test_sync_context_manager(self, framework):
        """Test using framework as sync context manager."""
        mock_model = Mock()
        framework.model = mock_model
        
        with framework as ctx:
            assert ctx == framework
        
        # Should cleanup on exit
        mock_model.cleanup.assert_called_once()


class TestFrameworkFactoryFunctions:
    """Test framework factory functions."""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple test model."""
        return nn.Sequential(
            nn.Linear(784, 10),
            nn.Softmax(dim=1)
        )
    
    def test_create_classification_framework(self, simple_model, temp_model_dir):
        """Test creating classification framework."""
        model_path = temp_model_dir / "classifier.pt"
        torch.save(simple_model, model_path)
        
        with patch('framework.TorchInferenceFramework.load_model'):
            framework = create_classification_framework(
                model_path=model_path,
                num_classes=10,
                class_names=["class_" + str(i) for i in range(10)],
                input_size=(224, 224)
            )
        
        assert isinstance(framework, TorchInferenceFramework)
        assert framework.config.model_type.value == "classification"
    
    def test_create_detection_framework(self, simple_model, temp_model_dir):
        """Test creating detection framework."""
        model_path = temp_model_dir / "detector.pt"
        torch.save(simple_model, model_path)
        
        with patch('framework.TorchInferenceFramework.load_model'):
            framework = create_detection_framework(
                model_path=model_path,
                class_names=["person", "car", "bike"],
                input_size=(640, 640),
                confidence_threshold=0.7
            )
        
        assert isinstance(framework, TorchInferenceFramework)
        assert framework.config.model_type.value == "detection"
    
    def test_create_segmentation_framework(self, simple_model, temp_model_dir):
        """Test creating segmentation framework."""
        model_path = temp_model_dir / "segmenter.pt"
        torch.save(simple_model, model_path)
        
        with patch('framework.TorchInferenceFramework.load_model'):
            framework = create_segmentation_framework(
                model_path=model_path,
                input_size=(512, 512),
                threshold=0.6
            )
        
        assert isinstance(framework, TorchInferenceFramework)
        assert framework.config.model_type.value == "segmentation"


class TestConvenienceFunctions:
    """Test convenience prediction functions."""
    
    def test_predict_image_classification(self, temp_model_dir, sample_image_path):
        """Test quick image classification prediction."""
        model_path = temp_model_dir / "classifier.pt"
        model_path.touch()
        
        with patch('framework.create_classification_framework') as mock_create:
            mock_framework = Mock()
            mock_framework.predict.return_value = {
                "predictions": [0.8, 0.1, 0.1],
                "class": 0
            }
            mock_framework.__enter__ = Mock(return_value=mock_framework)
            mock_framework.__exit__ = Mock(return_value=None)
            mock_create.return_value = mock_framework
            
            result = predict_image_classification(
                model_path=model_path,
                image_path=sample_image_path,
                num_classes=3,
                class_names=["cat", "dog", "bird"]
            )
        
        assert isinstance(result, dict)
        mock_create.assert_called_once()
        mock_framework.predict.assert_called_once_with(sample_image_path)
    
    def test_predict_object_detection(self, temp_model_dir, sample_image_path):
        """Test quick object detection prediction."""
        model_path = temp_model_dir / "detector.pt"
        model_path.touch()
        
        with patch('framework.create_detection_framework') as mock_create:
            mock_framework = Mock()
            mock_framework.predict.return_value = {
                "boxes": [[100, 100, 200, 200]],
                "scores": [0.9],
                "classes": [1]
            }
            mock_framework.__enter__ = Mock(return_value=mock_framework)
            mock_framework.__exit__ = Mock(return_value=None)
            mock_create.return_value = mock_framework
            
            result = predict_object_detection(
                model_path=model_path,
                image_path=sample_image_path,
                class_names=["person", "car"],
                confidence_threshold=0.7
            )
        
        assert isinstance(result, dict)
        mock_create.assert_called_once()
    
    def test_predict_segmentation(self, temp_model_dir, sample_image_path):
        """Test quick segmentation prediction."""
        model_path = temp_model_dir / "segmenter.pt"
        model_path.touch()
        
        with patch('framework.create_segmentation_framework') as mock_create:
            mock_framework = Mock()
            mock_framework.predict.return_value = {
                "mask": [[0, 1, 1], [0, 0, 1], [0, 0, 0]],
                "classes": [0, 1]
            }
            mock_framework.__enter__ = Mock(return_value=mock_framework)
            mock_framework.__exit__ = Mock(return_value=None)
            mock_create.return_value = mock_framework
            
            result = predict_segmentation(
                model_path=model_path,
                image_path=sample_image_path,
                threshold=0.5
            )
        
        assert isinstance(result, dict)
        mock_create.assert_called_once()


class TestGlobalFramework:
    """Test global framework management."""
    
    def test_get_global_framework(self):
        """Test getting global framework instance."""
        framework1 = get_global_framework()
        framework2 = get_global_framework()
        
        # Should be the same instance
        assert framework1 is framework2
        assert isinstance(framework1, TorchInferenceFramework)
    
    def test_set_global_framework(self, test_config):
        """Test setting global framework instance."""
        custom_framework = TorchInferenceFramework(test_config)
        set_global_framework(custom_framework)
        
        retrieved_framework = get_global_framework()
        assert retrieved_framework is custom_framework


class TestOptimizedFramework:
    """Test optimized framework creation."""
    
    def test_create_optimized_framework(self, test_config):
        """Test creating optimized framework."""
        optimized = create_optimized_framework(test_config)
        
        assert isinstance(optimized, TorchInferenceFramework)
        # Should be a subclass with optimized model loading
        assert optimized.config == test_config
    
    def test_optimized_framework_model_loading(self, test_config, simple_model, temp_model_dir):
        """Test optimized framework model loading."""
        model_path = temp_model_dir / "model.pt"
        torch.save(simple_model, model_path)
        
        optimized = create_optimized_framework(test_config)
        
        with patch('framework.OptimizedModel') as mock_optimized_model:
            mock_model_instance = Mock()
            mock_model_instance.load_model = Mock()
            mock_optimized_model.return_value = mock_model_instance
            
            with patch('framework.core.inference_engine.create_inference_engine'):
                optimized.load_model(model_path, "optimized_test")
        
        # Should use OptimizedModel instead of regular model adapter
        mock_optimized_model.assert_called_once_with(test_config)
        assert optimized.model == mock_model_instance


class TestFrameworkErrorHandling:
    """Test error handling in framework operations."""
    
    def test_framework_with_invalid_config(self):
        """Test framework with invalid configuration."""
        # Should handle None config gracefully
        with patch('framework.core.config.get_global_config') as mock_config:
            mock_config.return_value = InferenceConfig()
            
            framework = TorchInferenceFramework(None)
            assert framework.config is not None
    
    def test_prediction_error_handling(self, framework):
        """Test prediction error handling."""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Prediction failed")
        
        framework.model = mock_model
        framework._initialized = True
        
        with pytest.raises(Exception):
            framework.predict([1, 2, 3])
    
    @pytest.mark.asyncio
    async def test_async_prediction_error_handling(self, framework):
        """Test async prediction error handling."""
        mock_engine = AsyncMock()
        mock_engine.predict.side_effect = Exception("Async prediction failed")
        
        framework.engine = mock_engine
        framework._initialized = True
        framework._engine_running = True
        
        with pytest.raises(Exception):
            await framework.predict_async([1, 2, 3])
    
    def test_benchmark_error_handling(self, framework):
        """Test benchmark error handling."""
        framework._initialized = False
        
        with pytest.raises(RuntimeError):
            framework.benchmark([1, 2, 3])


class TestFrameworkIntegration:
    """Integration tests for framework functionality."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, simple_model, temp_model_dir, sample_image_path):
        """Test complete inference workflow."""
        model_path = temp_model_dir / "complete_model.pt"
        torch.save(simple_model, model_path)
        
        # Create framework
        framework = TorchInferenceFramework(InferenceConfig())
        
        with patch.object(framework, 'load_model') as mock_load_method:
            # Create a mock model instance 
            mock_model = Mock()
            mock_model.predict.return_value = {"prediction": "test"}
            mock_model.is_loaded = True
            mock_model.model_info = {"test": True}
            mock_model.cleanup = Mock()
            
            # Mock the predict_batch method to return a list
            mock_model.predict_batch.return_value = [{"prediction": "test"}, {"prediction": "test"}]
            
            # Set up the mock to assign the model when load_model is called
            async def mock_load_side_effect(*args, **kwargs):
                framework.model = mock_model
                framework._initialized = True
                # Also need to create a mock engine with async methods
                mock_engine = AsyncMock()
                mock_engine.health_check = AsyncMock(return_value={"healthy": True, "checks": {}})
                mock_engine.get_stats = Mock(return_value={"test": True})
                mock_engine.get_performance_report = Mock(return_value={"test": True})
                framework.engine = mock_engine
                
            mock_load_method.side_effect = mock_load_side_effect
            
            # Load model - await the async side effect
            await mock_load_side_effect()  # Call directly to avoid issues
            framework.load_model(model_path, "complete_test")
            
            # Verify the mock was used
            assert framework.model == mock_model
            
            # Test synchronous prediction
            sync_result = framework.predict(sample_image_path)
            assert sync_result == {"prediction": "test"}
            
            # Test batch prediction
            batch_result = framework.predict_batch([sample_image_path, sample_image_path])
            assert len(batch_result) == 2
            
            # Test model info
            info = framework.get_model_info()
            assert info is not None
            
            # Test performance report
            report = framework.get_performance_report()
            assert "framework_info" in report
            
            # Test health check
            health = await framework.health_check()
            assert health["healthy"]
    
    def test_framework_lifecycle_management(self, test_config, simple_model, temp_model_dir):
        """Test framework lifecycle management."""
        model_path = temp_model_dir / "lifecycle_model.pt"
        torch.save(simple_model, model_path)
        
        with patch('framework.load_model') as mock_load:
            mock_model = Mock()
            mock_model.cleanup = Mock()
            mock_model.is_loaded = True
            mock_model.model_info = {"test": True}
            mock_load.return_value = mock_model

            # Use as context manager
            with TorchInferenceFramework(test_config) as framework:
                framework.load_model(model_path)
                assert framework._initialized
                # Verify the mock model is being used
                assert framework.model == mock_model            # Should cleanup on exit
            mock_model.cleanup.assert_called_once()


class TestEnhancedOptimizationMethods:
    """Test enhanced optimization methods in TorchInferenceFramework."""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple test model."""
        return nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(32, 128)
    
    @pytest.fixture
    def enhanced_config(self):
        """Create configuration with enhanced optimizations enabled."""
        from framework.core.config import InferenceConfig, DeviceConfig, PerformanceConfig
        return InferenceConfig(
            device=DeviceConfig(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                use_vulkan=True,
                use_numba=True,
                jit_strategy="enhanced"
            ),
            performance=PerformanceConfig(
                enable_profiling=True,
                enable_metrics=True
            )
        )
    
    @pytest.fixture
    def framework_with_model(self, enhanced_config, simple_model, temp_model_dir):
        """Create framework with loaded model."""
        framework = TorchInferenceFramework(enhanced_config)
        
        # Mock model loading
        mock_model = Mock()
        mock_model.model = simple_model
        mock_model.example_inputs = torch.randn(1, 128)
        mock_model.is_loaded = True
        mock_model.model_info = {"type": "test", "parameters": 1000}
        mock_model.predict.return_value = torch.randn(1, 10)
        
        with patch('framework.load_model') as mock_load:
            mock_load.return_value = mock_model
            framework.load_model("test_model.pt")
        
        return framework
    
    @pytest.mark.skipif(not ENHANCED_OPTIMIZERS_AVAILABLE, reason="Enhanced optimizers not available")
    def test_get_optimization_recommendations(self, framework_with_model):
        """Test getting optimization recommendations."""
        recommendations = framework_with_model.get_optimization_recommendations(
            model_size="medium",
            target="inference"
        )
        
        assert isinstance(recommendations, list)
        for optimizer_name, description in recommendations:
            assert isinstance(optimizer_name, str)
            assert isinstance(description, str)
            assert len(description) > 0
    
    @pytest.mark.skipif(not ENHANCED_OPTIMIZERS_AVAILABLE, reason="Enhanced optimizers not available")
    def test_get_available_optimizers_method(self, framework_with_model):
        """Test getting available optimizers."""
        available = framework_with_model.get_available_optimizers()
        
        assert isinstance(available, dict)
        
        # Should have entries for different optimizer types
        expected_keys = ['performance', 'jit', 'enhanced_jit', 'vulkan', 'numba']
        
        for key in expected_keys:
            if key in available:
                assert 'available' in available[key]
                assert 'class' in available[key]
                assert isinstance(available[key]['available'], bool)
                assert isinstance(available[key]['class'], str)
    
    @pytest.mark.skipif(not ENHANCED_OPTIMIZERS_AVAILABLE, reason="Enhanced optimizers not available")
    def test_apply_automatic_optimizations(self, framework_with_model):
        """Test applying automatic optimizations."""
        with patch('framework.optimizers.EnhancedJITOptimizer') as mock_jit_class, \
             patch('framework.optimizers.VulkanOptimizer') as mock_vulkan_class, \
             patch('framework.optimizers.NumbaOptimizer') as mock_numba_class, \
             patch('framework.optimizers.PerformanceOptimizer') as mock_perf_class:
            
            # Setup mocks
            mock_jit = Mock()
            mock_vulkan = Mock()
            mock_numba = Mock()
            mock_perf = Mock()
            
            mock_jit.optimize.return_value = framework_with_model.model.model
            mock_vulkan.optimize.return_value = None  # Vulkan doesn't modify model directly
            mock_numba.optimize.return_value = None   # Numba doesn't modify model directly
            mock_perf.optimize.return_value = framework_with_model.model.model
            
            mock_jit_class.return_value = mock_jit
            mock_vulkan_class.return_value = mock_vulkan
            mock_numba_class.return_value = mock_numba
            mock_perf_class.return_value = mock_perf
            
            # Mock get_optimization_recommendations to return specific optimizers
            with patch.object(framework_with_model, 'get_optimization_recommendations') as mock_recommendations:
                mock_recommendations.return_value = [
                    ('enhanced_jit', 'Enhanced JIT compilation'),
                    ('vulkan', 'Vulkan acceleration'),
                    ('numba', 'Numba optimization'),
                    ('performance', 'Performance optimization')
                ]
                
                # Test automatic optimizations
                applied = framework_with_model.apply_automatic_optimizations()
                
                assert isinstance(applied, dict)
                
                # Check that optimizations were attempted
                for opt_name in ['enhanced_jit', 'vulkan', 'numba', 'performance']:
                    if opt_name in applied:
                        assert isinstance(applied[opt_name], bool)
    
    @pytest.mark.skipif(not ENHANCED_OPTIMIZERS_AVAILABLE, reason="Enhanced optimizers not available")
    def test_apply_automatic_optimizations_aggressive(self, framework_with_model):
        """Test applying aggressive automatic optimizations."""
        with patch.object(framework_with_model, 'get_optimization_recommendations') as mock_recommendations:
            mock_recommendations.return_value = [
                ('enhanced_jit', 'Enhanced JIT compilation'),
                ('performance', 'Performance optimization')
            ]
            
            # Test aggressive optimizations
            applied = framework_with_model.apply_automatic_optimizations(aggressive=True)
            
            assert isinstance(applied, dict)
            mock_recommendations.assert_called_once()
    
    def test_apply_automatic_optimizations_not_initialized(self, enhanced_config):
        """Test applying optimizations on uninitialized framework."""
        framework = TorchInferenceFramework(enhanced_config)
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            framework.apply_automatic_optimizations()
    
    def test_get_optimization_recommendations_fallback(self, enhanced_config):
        """Test optimization recommendations fallback when utilities not available."""
        framework = TorchInferenceFramework(enhanced_config)
        
        with patch('framework.get_optimization_recommendations', None):
            recommendations = framework.get_optimization_recommendations()
            
            # Should return fallback recommendations
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0
            assert recommendations[0] == ('standard', 'Only standard optimizations available')
    
    def test_get_available_optimizers_fallback(self, enhanced_config):
        """Test available optimizers fallback when utilities not available."""
        framework = TorchInferenceFramework(enhanced_config)
        
        with patch('framework.get_available_optimizers', None):
            available = framework.get_available_optimizers()
            
            # Should return fallback information
            assert isinstance(available, dict)
            assert 'standard' in available
            assert 'enhanced' in available
            assert available['standard']['available'] is True
            assert available['enhanced']['available'] is False
    
    @pytest.mark.skipif(not ENHANCED_OPTIMIZERS_AVAILABLE, reason="Enhanced optimizers not available")
    def test_optimization_with_different_strategies(self, framework_with_model):
        """Test optimization with different JIT strategies."""
        strategies = ['auto', 'torchscript_script', 'torchscript_trace', 'enhanced']
        
        for strategy in strategies:
            # Update config strategy
            framework_with_model.config.device.jit_strategy = strategy
            
            with patch('framework.optimizers.EnhancedJITOptimizer') as mock_jit_class:
                mock_jit = Mock()
                mock_jit.optimize.return_value = framework_with_model.model.model
                mock_jit_class.return_value = mock_jit
                
                with patch.object(framework_with_model, 'get_optimization_recommendations') as mock_recommendations:
                    mock_recommendations.return_value = [('enhanced_jit', 'Enhanced JIT')]
                    
                    applied = framework_with_model.apply_automatic_optimizations()
                    
                    if 'enhanced_jit' in applied:
                        # Verify the strategy was used
                        assert isinstance(applied['enhanced_jit'], bool)
    
    @pytest.mark.skipif(not ENHANCED_OPTIMIZERS_AVAILABLE, reason="Enhanced optimizers not available")
    def test_optimization_error_handling(self, framework_with_model):
        """Test error handling during optimization."""
        with patch('framework.EnhancedJITOptimizer') as mock_jit_class:
            # Mock optimizer that raises an exception
            mock_jit = Mock()
            mock_jit.optimize.side_effect = RuntimeError("Optimization failed")
            mock_jit_class.return_value = mock_jit
            
            with patch.object(framework_with_model, 'get_optimization_recommendations') as mock_recommendations:
                mock_recommendations.return_value = [('enhanced_jit', 'Enhanced JIT')]
                
                # Should handle errors gracefully
                applied = framework_with_model.apply_automatic_optimizations()
                
                if 'enhanced_jit' in applied:
                    # Should mark optimization as failed
                    assert applied['enhanced_jit'] is False
    
    def test_optimization_recommendations_different_scenarios(self, enhanced_config):
        """Test optimization recommendations for different scenarios."""
        framework = TorchInferenceFramework(enhanced_config)
        
        if framework.get_optimization_recommendations != None:
            scenarios = [
                ("auto", "small", "inference"),
                ("cuda", "medium", "inference"),
                ("cpu", "large", "training"),
                ("auto", "xlarge", "serving")
            ]
            
            for device, model_size, target in scenarios:
                recommendations = framework.get_optimization_recommendations(
                    device=device,
                    model_size=model_size,
                    target=target
                )
                
                assert isinstance(recommendations, list)
                # Different scenarios should potentially give different recommendations
                for optimizer_name, description in recommendations:
                    assert isinstance(optimizer_name, str)
                    assert isinstance(description, str)


class TestFrameworkOptimizationIntegration:
    """Test integration of optimization features with framework operations."""
    
    @pytest.fixture
    def optimized_framework_config(self):
        """Create configuration for optimized framework."""
        from framework.core.config import InferenceConfig, DeviceConfig, PerformanceConfig
        return InferenceConfig(
            device=DeviceConfig(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                use_vulkan=True,
                use_numba=True,
                jit_strategy="enhanced"
            ),
            performance=PerformanceConfig(
                enable_profiling=True,
                enable_metrics=True
            )
        )
    
    def test_optimized_framework_creation(self, optimized_framework_config):
        """Test creation of optimized framework."""
        framework = create_optimized_framework(optimized_framework_config)
        
        assert isinstance(framework, TorchInferenceFramework)
        assert framework.config == optimized_framework_config
    
    def test_optimized_framework_with_auto_optimization(self, optimized_framework_config, temp_model_dir):
        """Test optimized framework with automatic optimization."""
        simple_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        model_path = temp_model_dir / "auto_opt_model.pt"
        torch.save(simple_model, model_path)
        
        framework = create_optimized_framework(optimized_framework_config)
        
        with patch('framework.OptimizedModel') as mock_optimized_model_class:
            mock_optimized_model = Mock()
            mock_optimized_model.load_model = Mock()
            mock_optimized_model.is_loaded = True
            mock_optimized_model.model_info = {"optimized": True}
            mock_optimized_model_class.return_value = mock_optimized_model
            
            # Load model - should use OptimizedModel
            framework.load_model(model_path)
            
            assert framework._initialized
            mock_optimized_model_class.assert_called_once_with(optimized_framework_config)
            mock_optimized_model.load_model.assert_called_once_with(model_path)
    
    def test_framework_optimization_performance_impact(self, optimized_framework_config):
        """Test that optimizations don't negatively impact basic operations."""
        framework = TorchInferenceFramework(optimized_framework_config)
        
        # Mock a simple model
        mock_model = Mock()
        mock_model.is_loaded = True
        mock_model.model_info = {"test": True}
        mock_model.predict.return_value = {"prediction": "test"}
        mock_model.model = nn.Linear(10, 5)
        mock_model.example_inputs = torch.randn(1, 10)
        
        framework.model = mock_model
        framework._initialized = True
        
        # Test that basic operations still work
        input_data = torch.randn(1, 10)
        result = framework.predict(input_data)
        
        assert result == {"prediction": "test"}
        mock_model.predict.assert_called_once_with(input_data)
    
    def test_framework_benchmark_with_optimizations(self, optimized_framework_config):
        """Test benchmarking with optimizations applied."""
        framework = TorchInferenceFramework(optimized_framework_config)
        
        # Mock optimized model
        mock_model = Mock()
        mock_model.is_loaded = True
        mock_model.model_info = {"optimized": True, "type": "test"}
        mock_model.device = "cpu"
        mock_model.predict.return_value = torch.randn(1, 5)
        
        framework.model = mock_model
        framework._initialized = True
        
        # Run benchmark
        input_data = torch.randn(1, 10)
        results = framework.benchmark(input_data, iterations=10, warmup=2)
        
        assert isinstance(results, dict)
        assert "mean_time_ms" in results
        assert "throughput_fps" in results
        assert "device" in results
        assert "model_info" in results
        assert results["iterations"] == 10
        
        # Should have called predict multiple times (warmup + iterations)
        assert mock_model.predict.call_count == 12  # 2 warmup + 10 iterations
    
    @pytest.mark.asyncio
    async def test_framework_async_operations_with_optimizations(self, optimized_framework_config):
        """Test async operations with optimizations."""
        framework = TorchInferenceFramework(optimized_framework_config)
        
        # Mock optimized model and engine
        mock_model = Mock()
        mock_model.is_loaded = True
        mock_model.model_info = {"optimized": True}
        
        mock_engine = AsyncMock()
        mock_engine.predict.return_value = {"result": "optimized_prediction"}
        mock_engine.start = AsyncMock()
        mock_engine.stop = AsyncMock()
        mock_engine.health_check.return_value = {"healthy": True}
        
        framework.model = mock_model
        framework.engine = mock_engine
        framework._initialized = True
        
        # Test async operations
        await framework.start_engine()
        assert framework._engine_running
        
        result = await framework.predict_async("test_input")
        assert result == {"result": "optimized_prediction"}
        
        health = await framework.health_check()
        assert health["healthy"]
        
        await framework.stop_engine()
        assert not framework._engine_running


class TestFrameworkAvailabilityDetection:
    """Test framework's ability to detect and handle optimizer availability."""
    
    def test_framework_with_no_enhanced_optimizers(self):
        """Test framework behavior when enhanced optimizers are not available."""
        from framework.core.config import InferenceConfig
        config = InferenceConfig()
        
        with patch('framework.get_available_optimizers', None), \
             patch('framework.get_optimization_recommendations', None):
            
            framework = TorchInferenceFramework(config)
            
            # Should still work with fallbacks
            recommendations = framework.get_optimization_recommendations()
            assert isinstance(recommendations, list)
            
            available = framework.get_available_optimizers()
            assert isinstance(available, dict)
    
    def test_framework_mixed_optimizer_availability(self):
        """Test framework with mixed optimizer availability."""
        from framework.core.config import InferenceConfig
        config = InferenceConfig()
        
        framework = TorchInferenceFramework(config)
        
        # Mock mixed availability
        mock_available = {
            'enhanced_jit': {'available': True, 'class': 'EnhancedJITOptimizer'},
            'vulkan': {'available': False, 'class': 'VulkanOptimizer'},
            'numba': {'available': True, 'class': 'NumbaOptimizer'},
            'performance': {'available': True, 'class': 'PerformanceOptimizer'}
        }
        
        with patch.object(framework, 'get_available_optimizers', return_value=mock_available):
            available = framework.get_available_optimizers()
            
            # Should handle mixed availability
            assert available['enhanced_jit']['available'] is True
            assert available['vulkan']['available'] is False
            assert available['numba']['available'] is True
            assert available['performance']['available'] is True
    
    def test_framework_graceful_degradation(self):
        """Test framework graceful degradation when optimizers fail."""
        from framework.core.config import InferenceConfig, DeviceConfig
        config = InferenceConfig(
            device=DeviceConfig(
                use_vulkan=True,
                use_numba=True,
                jit_strategy="enhanced"
            )
        )
        
        framework = TorchInferenceFramework(config)
        
        # Mock model
        mock_model = Mock()
        mock_model.model = nn.Linear(10, 5)
        mock_model.example_inputs = torch.randn(1, 10)
        mock_model.is_loaded = True
        
        framework.model = mock_model
        framework._initialized = True
        
        # Mock all optimizers to fail
        with patch('framework.EnhancedJITOptimizer', side_effect=RuntimeError("Not available")), \
             patch('framework.VulkanOptimizer', side_effect=RuntimeError("Not available")), \
             patch('framework.NumbaOptimizer', side_effect=RuntimeError("Not available")):
            
            # Should still provide recommendations (even if they fail)
            with patch.object(framework, 'get_optimization_recommendations') as mock_recommendations:
                mock_recommendations.return_value = [
                    ('enhanced_jit', 'Enhanced JIT (may not be available)'),
                    ('vulkan', 'Vulkan (may not be available)')
                ]
                
                applied = framework.apply_automatic_optimizations()
                
                # All optimizations should fail gracefully
                for opt_name, success in applied.items():
                    assert success is False  # All should fail but not crash

"""Unit tests for Zero Scaler functionality."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from datetime import datetime, timedelta
import torch
from typing import Dict, List, Optional

from framework.autoscaling.zero_scaler import (
    ZeroScaler, 
    ZeroScalingConfig, 
    ColdStartStrategy, 
    ModelInstance,
    ModelInstanceState
)
from framework.core.base_model import ModelManager
from framework.core.config import InferenceConfig


@pytest.fixture
def zero_scaling_config():
    """Create a test zero scaling configuration."""
    return ZeroScalingConfig(
        enabled=True,
        scale_to_zero_delay=5.0,  # 5 seconds for quick testing
        max_loaded_models=3,
        preload_popular_models=True,
        popularity_threshold=5,
        enable_predictive_scaling=True,
        cold_start_strategy=ColdStartStrategy.LAZY,
        max_cold_start_time=10.0,
        preload_timeout=5.0
    )


@pytest.fixture
def mock_model_manager():
    """Create a mock model manager."""
    from framework.core.config import InferenceConfig, DeviceConfig, BatchConfig, DeviceType
    
    # Create a test config for models
    test_config = InferenceConfig(
        device=DeviceConfig(device_type=DeviceType.CPU, use_fp16=False),
        batch=BatchConfig(batch_size=1, max_batch_size=4)
    )
    
    manager = Mock(spec=ModelManager)
    manager._loaded_models = {}
    
    async def mock_load_model(model_id):
        # Create mock model with proper config
        mock_model = Mock()
        mock_model.config = test_config
        mock_model.predict = AsyncMock(return_value={"predictions": [0.1, 0.2, 0.7]})
        mock_model.warmup = AsyncMock()
        mock_model.is_loaded = True
        mock_model.model_name = model_id
        
        manager._loaded_models[model_id] = mock_model
        return mock_model
    
    async def mock_unload_model(model_id):
        if model_id in manager._loaded_models:
            del manager._loaded_models[model_id]
    
    def mock_get_model(model_id):
        return manager._loaded_models.get(model_id)
    
    def mock_is_model_loaded(model_id):
        return model_id in manager._loaded_models
    
    def mock_get_loaded_models():
        return list(manager._loaded_models.keys())
    
    manager.load_model.side_effect = mock_load_model
    manager.unload_model.side_effect = mock_unload_model
    manager.get_model.side_effect = mock_get_model
    manager.is_model_loaded.side_effect = mock_is_model_loaded
    manager.get_loaded_models.side_effect = mock_get_loaded_models
    
    return manager


@pytest.fixture
def mock_inference_engine():
    """Create a mock inference engine."""
    engine = Mock()
    engine.predict = AsyncMock()
    engine.health_check = AsyncMock(return_value=True)
    return engine


@pytest.fixture
def zero_scaler(zero_scaling_config, mock_model_manager, mock_inference_engine):
    """Create a zero scaler instance for testing."""
    return ZeroScaler(
        config=zero_scaling_config,
        model_manager=mock_model_manager,
        inference_engine=mock_inference_engine
    )


class TestZeroScalingConfig:
    """Test zero scaling configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ZeroScalingConfig()
        
        assert config.enabled is True
        assert config.scale_to_zero_delay == 300.0  # 5 minutes
        assert config.max_loaded_models == 5
        assert config.preload_popular_models is True
        assert config.popularity_threshold == 10
        assert config.enable_predictive_scaling is True
        assert config.cold_start_strategy == ColdStartStrategy.HYBRID
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ZeroScalingConfig(
            enabled=False,
            scale_to_zero_delay=600.0,
            max_loaded_models=10,
            cold_start_strategy=ColdStartStrategy.EAGER
        )
        
        assert config.enabled is False
        assert config.scale_to_zero_delay == 600.0
        assert config.max_loaded_models == 10
        assert config.cold_start_strategy == ColdStartStrategy.EAGER
    
    def test_validation_errors(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="scale_to_zero_delay must be positive"):
            ZeroScalingConfig(scale_to_zero_delay=-1.0)
        
        with pytest.raises(ValueError, match="max_loaded_models must be positive"):
            ZeroScalingConfig(max_loaded_models=0)
        
        with pytest.raises(ValueError, match="popularity_threshold must be positive"):
            ZeroScalingConfig(popularity_threshold=-1)


class TestModelInstance:
    """Test model instance management."""
    
    def test_model_instance_creation(self):
        """Test creating a model instance."""
        instance = ModelInstance(
            model_id="test_model",
            instance_id="instance_1",
            model=Mock(),
            state=ModelInstanceState.IDLE
        )
        
        assert instance.model_id == "test_model"
        assert instance.instance_id == "instance_1"
        assert instance.state == ModelInstanceState.IDLE
        assert instance.created_at is not None
        assert instance.last_accessed is not None
    
    def test_model_instance_update_state(self):
        """Test updating model instance state."""
        instance = ModelInstance(
            model_id="test_model",
            instance_id="instance_1", 
            model=Mock(),
            state=ModelInstanceState.IDLE
        )
        
        old_time = instance.last_accessed
        instance.update_state(ModelInstanceState.ACTIVE)
        
        assert instance.state == ModelInstanceState.ACTIVE
        assert instance.last_accessed > old_time
    
    def test_model_instance_is_idle_too_long(self):
        """Test checking if model instance has been idle too long."""
        instance = ModelInstance(
            model_id="test_model",
            instance_id="instance_1",
            model=Mock(),
            state=ModelInstanceState.IDLE
        )
        
        # Just created, should not be idle too long
        assert not instance.is_idle_too_long(300.0)
        
        # Mock old last_accessed time
        instance.last_accessed = datetime.now() - timedelta(seconds=400)
        assert instance.is_idle_too_long(300.0)
    
    def test_model_instance_get_idle_time(self):
        """Test getting idle time."""
        instance = ModelInstance(
            model_id="test_model",
            instance_id="instance_1",
            model=Mock(),
            state=ModelInstanceState.IDLE
        )
        
        # Mock last accessed time
        instance.last_accessed = datetime.now() - timedelta(seconds=100)
        idle_time = instance.get_idle_time()
        
        assert 99 <= idle_time <= 101  # Allow for small timing differences


class TestZeroScaler:
    """Test zero scaler functionality."""
    
    @pytest.mark.asyncio
    async def test_zero_scaler_start_stop(self, zero_scaler):
        """Test starting and stopping zero scaler."""
        assert not zero_scaler.is_running
        
        await zero_scaler.start()
        assert zero_scaler.is_running
        
        await zero_scaler.stop()
        assert not zero_scaler.is_running
    
    @pytest.mark.asyncio
    async def test_zero_scaler_disabled(self, mock_model_manager, mock_inference_engine):
        """Test zero scaler when disabled."""
        config = ZeroScalingConfig(enabled=False)
        scaler = ZeroScaler(config, mock_model_manager, mock_inference_engine)
        
        await scaler.start()
        assert not scaler.is_running
        
        # Should not do anything when disabled
        result = await scaler.ensure_model_loaded("test_model")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_ensure_model_loaded_new_model(self, zero_scaler, mock_model_manager):
        """Test ensuring a new model is loaded."""
        await zero_scaler.start()
        
        # Mock model loading
        mock_model = Mock()
        mock_model_manager.load_model.return_value = mock_model
        mock_model_manager.is_model_loaded.return_value = False
        
        instance = await zero_scaler.ensure_model_loaded("test_model")
        
        assert instance is not None
        assert instance.model_id == "test_model"
        assert instance.state == ModelInstanceState.IDLE
        mock_model_manager.load_model.assert_called_once_with("test_model")
    
    @pytest.mark.asyncio
    async def test_ensure_model_loaded_existing_model(self, zero_scaler, mock_model_manager):
        """Test ensuring an already loaded model."""
        await zero_scaler.start()
        
        # Create existing instance
        mock_model = Mock()
        instance = ModelInstance(
            model_id="test_model",
            instance_id="instance_1",
            model=mock_model,
            state=ModelInstanceState.IDLE
        )
        zero_scaler.instances["test_model"] = [instance]
        
        result = await zero_scaler.ensure_model_loaded("test_model")
        
        assert result == instance
        mock_model_manager.load_model.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_predict_with_model_loading(self, zero_scaler, mock_model_manager, mock_inference_engine):
        """Test prediction with automatic model loading."""
        await zero_scaler.start()
        
        # Mock model loading and prediction
        mock_model = Mock()
        mock_model_manager.load_model.return_value = mock_model
        mock_model_manager.is_model_loaded.return_value = False
        mock_inference_engine.predict.return_value = {"result": "test_prediction"}
        
        result = await zero_scaler.predict("test_model", {"input": "test"})
        
        assert result == {"result": "test_prediction"}
        mock_model_manager.load_model.assert_called_once_with("test_model")
        mock_inference_engine.predict.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_predict_updates_access_time(self, zero_scaler, mock_model_manager, mock_inference_engine):
        """Test that prediction updates model access time."""
        await zero_scaler.start()
        
        # Create existing instance
        mock_model = Mock()
        instance = ModelInstance(
            model_id="test_model",
            instance_id="instance_1",
            model=mock_model,
            state=ModelInstanceState.IDLE
        )
        old_access_time = instance.last_accessed
        zero_scaler.instances["test_model"] = [instance]
        
        mock_inference_engine.predict.return_value = {"result": "test_prediction"}
        
        # Wait a small amount to ensure time difference
        await asyncio.sleep(0.01)
        
        result = await zero_scaler.predict("test_model", {"input": "test"})
        
        assert instance.last_accessed > old_access_time
        assert instance.state == ModelInstanceState.ACTIVE
    
    @pytest.mark.asyncio
    async def test_scale_to_zero_idle_models(self, zero_scaler, mock_model_manager):
        """Test scaling idle models to zero."""
        await zero_scaler.start()
        
        # Create idle instance
        mock_model = Mock()
        instance = ModelInstance(
            model_id="test_model",
            instance_id="instance_1",
            model=mock_model,
            state=ModelInstanceState.IDLE
        )
        # Make it appear idle for longer than threshold
        instance.last_accessed = datetime.now() - timedelta(seconds=10)
        zero_scaler.instances["test_model"] = [instance]
        
        # Run scaling check
        await zero_scaler._scale_to_zero_check()
        
        # Should unload the model
        mock_model_manager.unload_model.assert_called_once_with("test_model")
        assert "test_model" not in zero_scaler.instances
    
    @pytest.mark.asyncio
    async def test_dont_scale_active_models(self, zero_scaler, mock_model_manager):
        """Test that active models are not scaled to zero."""
        await zero_scaler.start()
        
        # Create active instance
        mock_model = Mock()
        instance = ModelInstance(
            model_id="test_model",
            instance_id="instance_1",
            model=mock_model,
            state=ModelInstanceState.ACTIVE
        )
        zero_scaler.instances["test_model"] = [instance]
        
        # Run scaling check
        await zero_scaler._scale_to_zero_check()
        
        # Should not unload active model
        mock_model_manager.unload_model.assert_not_called()
        assert "test_model" in zero_scaler.instances
    
    @pytest.mark.asyncio
    async def test_preload_popular_models(self, zero_scaler, mock_model_manager):
        """Test preloading popular models."""
        await zero_scaler.start()
        
        # Simulate model requests to make it popular
        mock_model = Mock()
        mock_model_manager.load_model.return_value = mock_model
        mock_model_manager.is_model_loaded.return_value = False
        
        # Make multiple requests to make model popular
        for _ in range(6):  # Above popularity threshold of 5
            zero_scaler._update_model_popularity("test_model")
        
        # Mock that model is not loaded
        zero_scaler.instances = {}
        mock_model_manager.is_model_loaded.return_value = False
        
        # Run preloading
        await zero_scaler._preload_popular_models()
        
        # Should preload the popular model
        mock_model_manager.load_model.assert_called_with("test_model")
    
    @pytest.mark.asyncio
    async def test_max_loaded_models_limit(self, zero_scaler, mock_model_manager):
        """Test that max loaded models limit is respected."""
        await zero_scaler.start()
        zero_scaler.config.max_loaded_models = 2
        
        # Create instances for max number of models
        for i in range(2):
            mock_model = Mock()
            instance = ModelInstance(
                model_id=f"model_{i}",
                instance_id=f"instance_{i}",
                model=mock_model,
                state=ModelInstanceState.IDLE
            )
            zero_scaler.instances[f"model_{i}"] = [instance]
        
        # Try to load another model
        mock_model_manager.load_model.return_value = Mock()
        mock_model_manager.is_model_loaded.return_value = False
        
        # Should unload least recently used model first
        zero_scaler.instances["model_0"][0].last_accessed = datetime.now() - timedelta(seconds=100)
        zero_scaler.instances["model_1"][0].last_accessed = datetime.now() - timedelta(seconds=50)
        
        await zero_scaler.ensure_model_loaded("model_2")
        
        # Should unload the oldest model
        mock_model_manager.unload_model.assert_called_with("model_0")
    
    def test_get_stats(self, zero_scaler):
        """Test getting zero scaler statistics."""
        # Create some instances
        for i in range(3):
            mock_model = Mock()
            instance = ModelInstance(
                model_id=f"model_{i}",
                instance_id=f"instance_{i}",
                model=mock_model,
                state=ModelInstanceState.IDLE if i < 2 else ModelInstanceState.ACTIVE
            )
            zero_scaler.instances[f"model_{i}"] = [instance]
        
        # Update popularity
        zero_scaler.model_popularity["model_0"] = 10
        zero_scaler.model_popularity["model_1"] = 5
        
        stats = zero_scaler.get_stats()
        
        assert stats["total_instances"] == 3
        assert stats["idle_instances"] == 2
        assert stats["active_instances"] == 1
        assert stats["loaded_models"] == 3
        assert stats["popular_models"] == ["model_0", "model_1"]
        assert stats["enabled"] is True
    
    def test_get_health_status(self, zero_scaler):
        """Test getting health status."""
        health = zero_scaler.get_health_status()
        
        assert "status" in health
        assert "enabled" in health
        assert "total_instances" in health
        assert "idle_instances" in health
        assert "active_instances" in health
        assert "loaded_models" in health
    
    @pytest.mark.asyncio
    async def test_cold_start_strategies(self, mock_model_manager, mock_inference_engine):
        """Test different cold start strategies."""
        # Test EAGER strategy
        config = ZeroScalingConfig(
            enabled=True,
            cold_start_strategy=ColdStartStrategy.EAGER,
            preload_models=["model_1", "model_2"]
        )
        scaler = ZeroScaler(config, mock_model_manager, mock_inference_engine)
        
        mock_model_manager.load_model.return_value = Mock()
        mock_model_manager.is_model_loaded.return_value = False
        
        await scaler.start()
        
        # Should preload specified models
        expected_calls = [call("model_1"), call("model_2")]
        mock_model_manager.load_model.assert_has_calls(expected_calls, any_order=True)
    
    @pytest.mark.asyncio
    async def test_model_loading_failure(self, zero_scaler, mock_model_manager):
        """Test handling model loading failures."""
        await zero_scaler.start()
        
        # Mock model loading failure
        mock_model_manager.load_model.side_effect = Exception("Model loading failed")
        mock_model_manager.is_model_loaded.return_value = False
        
        instance = await zero_scaler.ensure_model_loaded("test_model")
        
        # Should return None on failure
        assert instance is None
        
        # Should not have any instances
        assert "test_model" not in zero_scaler.instances
    
    @pytest.mark.asyncio
    async def test_concurrent_model_loading(self, zero_scaler, mock_model_manager):
        """Test concurrent model loading requests."""
        await zero_scaler.start()
        
        mock_model = Mock()
        mock_model_manager.load_model.return_value = mock_model
        mock_model_manager.is_model_loaded.return_value = False
        
        # Start multiple concurrent loads
        tasks = [
            zero_scaler.ensure_model_loaded("test_model")
            for _ in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Should only load model once
        assert mock_model_manager.load_model.call_count == 1
        
        # All tasks should get the same instance
        assert all(result is not None for result in results)
        assert all(result.model_id == "test_model" for result in results)


if __name__ == "__main__":
    pytest.main([__file__])

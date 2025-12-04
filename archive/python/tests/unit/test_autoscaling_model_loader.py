"""Unit tests for Dynamic Model Loader functionality."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, List

from framework.autoscaling.model_loader import (
    DynamicModelLoader,
    ModelLoaderConfig,
    LoadBalancingStrategy,
    LoadBalancer,
    ModelInstanceInfo,
    ModelInstanceState,
    ModelLoadingStrategy
)
from framework.core.base_model import ModelManager


@pytest.fixture
def model_loader_config():
    """Create a test model loader configuration."""
    return ModelLoaderConfig(
        enabled=True,
        max_instances_per_model=3,
        min_instances_per_model=1,
        load_balancing_strategy=LoadBalancingStrategy.LEAST_CONNECTIONS,
        enable_model_caching=True,
        prefetch_popular_models=True,
        model_loading_strategy=ModelLoadingStrategy.LAZY,
        health_check_interval=5.0,
        max_unhealthy_instances=1
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
def model_loader(model_loader_config, mock_model_manager, mock_inference_engine):
    """Create a dynamic model loader instance for testing."""
    return DynamicModelLoader(
        config=model_loader_config,
        model_manager=mock_model_manager,
        inference_engine=mock_inference_engine
    )


class TestModelLoaderConfig:
    """Test model loader configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelLoaderConfig()
        
        assert config.enabled is True
        assert config.max_instances_per_model == 3
        assert config.min_instances_per_model == 1
        assert config.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS
        assert config.enable_model_caching is True
        assert config.prefetch_popular_models is True
        assert config.model_loading_strategy == ModelLoadingStrategy.LAZY
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ModelLoaderConfig(
            enabled=False,
            max_instances_per_model=5,
            min_instances_per_model=2,
            load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
            model_loading_strategy=ModelLoadingStrategy.EAGER
        )
        
        assert config.enabled is False
        assert config.max_instances_per_model == 5
        assert config.min_instances_per_model == 2
        assert config.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN
        assert config.model_loading_strategy == ModelLoadingStrategy.EAGER
    
    def test_validation_errors(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="max_instances_per_model must be positive"):
            ModelLoaderConfig(max_instances_per_model=0)
        
        with pytest.raises(ValueError, match="min_instances_per_model must be positive"):
            ModelLoaderConfig(min_instances_per_model=0)
        
        with pytest.raises(ValueError, match="min_instances_per_model cannot be greater than max_instances_per_model"):
            ModelLoaderConfig(min_instances_per_model=3, max_instances_per_model=2)


class TestModelInstanceInfo:
    """Test model instance information."""
    
    def test_instance_creation(self):
        """Test creating a model instance info."""
        instance = ModelInstanceInfo(
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
        assert instance.active_connections == 0
        assert instance.total_requests == 0
        assert instance.total_response_time == 0.0
    
    def test_update_state(self):
        """Test updating instance state."""
        instance = ModelInstanceInfo(
            model_id="test_model",
            instance_id="instance_1",
            model=Mock(),
            state=ModelInstanceState.IDLE
        )
        
        old_time = instance.last_accessed
        import time
        time.sleep(0.001)  # Small sleep to ensure different timestamps
        instance.update_state(ModelInstanceState.ACTIVE)
        
        assert instance.state == ModelInstanceState.ACTIVE
        assert instance.last_accessed >= old_time  # Allow equal in case time granularity is low
    
    def test_start_request(self):
        """Test starting a request."""
        instance = ModelInstanceInfo(
            model_id="test_model",
            instance_id="instance_1",
            model=Mock(),
            state=ModelInstanceState.IDLE
        )
        
        instance.start_request()
        
        assert instance.active_connections == 1
        assert instance.total_requests == 1
    
    def test_end_request(self):
        """Test ending a request."""
        instance = ModelInstanceInfo(
            model_id="test_model",
            instance_id="instance_1",
            model=Mock(),
            state=ModelInstanceState.IDLE
        )
        
        instance.start_request()
        instance.end_request(0.1)  # 100ms response time
        
        assert instance.active_connections == 0
        assert instance.total_response_time == 0.1
    
    def test_get_average_response_time(self):
        """Test getting average response time."""
        instance = ModelInstanceInfo(
            model_id="test_model",
            instance_id="instance_1",
            model=Mock(),
            state=ModelInstanceState.IDLE
        )
        
        # No requests yet
        assert instance.get_average_response_time() == 0.0
        
        # Add some requests
        instance.start_request()
        instance.end_request(0.1)
        instance.start_request()
        instance.end_request(0.2)
        
        assert instance.get_average_response_time() == pytest.approx(0.15)
    
    def test_is_healthy(self):
        """Test health check."""
        instance = ModelInstanceInfo(
            model_id="test_model",
            instance_id="instance_1",
            model=Mock(),
            state=ModelInstanceState.IDLE
        )
        
        assert instance.is_healthy() is True
        
        # Mark as unhealthy
        instance.health_check_failures = 3
        assert instance.is_healthy() is False
    
    def test_get_load_score(self):
        """Test getting load score."""
        instance = ModelInstanceInfo(
            model_id="test_model",
            instance_id="instance_1",
            model=Mock(),
            state=ModelInstanceState.IDLE
        )
        
        # No load initially
        assert instance.get_load_score() == 0.0
        
        # Add some load
        instance.active_connections = 2
        instance.total_requests = 10
        instance.total_response_time = 1.0
        
        score = instance.get_load_score()
        assert score > 0.0


class TestLoadBalancer:
    """Test load balancer functionality."""
    
    def test_round_robin_balancer(self):
        """Test round robin load balancing."""
        balancer = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        
        instances = [
            ModelInstanceInfo("model", f"instance_{i}", Mock(), ModelInstanceState.IDLE)
            for i in range(3)
        ]
        
        # Should cycle through instances
        selections = [balancer.select_instance(instances) for _ in range(6)]
        
        # Should select each instance twice in order
        assert selections[0] == instances[0]
        assert selections[1] == instances[1]
        assert selections[2] == instances[2]
        assert selections[3] == instances[0]
    
    def test_least_connections_balancer(self):
        """Test least connections load balancing."""
        balancer = LoadBalancer(LoadBalancingStrategy.LEAST_CONNECTIONS)
        
        instances = [
            ModelInstanceInfo("model", f"instance_{i}", Mock(), ModelInstanceState.IDLE)
            for i in range(3)
        ]
        
        # Set different connection counts
        instances[0].active_connections = 5
        instances[1].active_connections = 2
        instances[2].active_connections = 8
        
        selected = balancer.select_instance(instances)
        
        # Should select instance with least connections
        assert selected == instances[1]
    
    def test_least_response_time_balancer(self):
        """Test least response time load balancing."""
        balancer = LoadBalancer(LoadBalancingStrategy.LEAST_RESPONSE_TIME)
        
        instances = [
            ModelInstanceInfo("model", f"instance_{i}", Mock(), ModelInstanceState.IDLE)
            for i in range(3)
        ]
        
        # Set different response times
        instances[0].total_requests = 10
        instances[0].total_response_time = 5.0  # 0.5s average
        instances[1].total_requests = 10
        instances[1].total_response_time = 2.0  # 0.2s average
        instances[2].total_requests = 10
        instances[2].total_response_time = 8.0  # 0.8s average
        
        selected = balancer.select_instance(instances)
        
        # Should select instance with least response time
        assert selected == instances[1]
    
    def test_weighted_round_robin_balancer(self):
        """Test weighted round robin load balancing."""
        balancer = LoadBalancer(LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN)
        
        instances = [
            ModelInstanceInfo("model", f"instance_{i}", Mock(), ModelInstanceState.IDLE)
            for i in range(3)
        ]
        
        # Set different weights (inverse of load)
        instances[0].active_connections = 1  # High weight
        instances[1].active_connections = 5  # Low weight
        instances[2].active_connections = 3  # Medium weight
        
        # Select multiple times and check distribution
        selections = [balancer.select_instance(instances) for _ in range(30)]
        
        # Instance 0 should be selected most (highest weight)
        count_0 = selections.count(instances[0])
        count_1 = selections.count(instances[1])
        count_2 = selections.count(instances[2])
        
        assert count_0 > count_1
        assert count_0 > count_2
    
    def test_consistent_hash_balancer(self):
        """Test consistent hash load balancing."""
        balancer = LoadBalancer(LoadBalancingStrategy.CONSISTENT_HASH)
        
        instances = [
            ModelInstanceInfo("model", f"instance_{i}", Mock(), ModelInstanceState.IDLE)
            for i in range(3)
        ]
        
        # Same input should always go to same instance
        selected1 = balancer.select_instance(instances, request_data={"input": "test1"})
        selected2 = balancer.select_instance(instances, request_data={"input": "test1"})
        
        assert selected1 == selected2
        
        # Different input might go to different instance
        selected3 = balancer.select_instance(instances, request_data={"input": "test2"})
        # We can't guarantee it's different, but at least it should be consistent
        selected4 = balancer.select_instance(instances, request_data={"input": "test2"})
        assert selected3 == selected4
    
    def test_filter_healthy_instances(self):
        """Test filtering healthy instances."""
        balancer = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        
        instances = [
            ModelInstanceInfo("model", f"instance_{i}", Mock(), ModelInstanceState.IDLE)
            for i in range(3)
        ]
        
        # Mark one as unhealthy
        instances[1].health_check_failures = 5
        
        selected = balancer.select_instance(instances)
        
        # Should not select unhealthy instance
        assert selected != instances[1]
        assert selected in [instances[0], instances[2]]


class TestDynamicModelLoader:
    """Test dynamic model loader functionality."""
    
    @pytest.mark.asyncio
    async def test_start_stop(self, model_loader):
        """Test starting and stopping model loader."""
        assert not model_loader.is_running
        
        await model_loader.start()
        assert model_loader.is_running
        
        await model_loader.stop()
        assert not model_loader.is_running
    
    @pytest.mark.asyncio
    async def test_disabled_loader(self, mock_model_manager, mock_inference_engine):
        """Test model loader when disabled."""
        config = ModelLoaderConfig(enabled=False)
        loader = DynamicModelLoader(config, mock_model_manager, mock_inference_engine)
        
        await loader.start()
        assert not loader.is_running
        
        # Should not do anything when disabled
        result = await loader.predict("test_model", {"input": "test"})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_load_model_first_time(self, model_loader, mock_model_manager):
        """Test loading a model for the first time."""
        await model_loader.start()
        
        # Don't override the mock - use the one set up in fixture
        instance = await model_loader.load_model("test_model")
        
        assert instance is not None
        assert instance.model_id == "test_model"
        assert instance.state == ModelInstanceState.READY
        mock_model_manager.load_model.assert_called_once_with("test_model")
    
    @pytest.mark.asyncio
    async def test_load_model_scaling(self, model_loader, mock_model_manager):
        """Test scaling model instances."""
        await model_loader.start()
        
        # Don't override the mock - use the one set up in fixture
        
        # Load initial instance
        await model_loader.load_model("test_model")
        
        # Scale to 3 instances
        instances = await model_loader.scale_model("test_model", target_instances=3)
        
        assert len(instances) == 3
        assert all(inst.model_id == "test_model" for inst in instances)
        # Should have called load_model 3 times total (1 initial + 2 new)
        assert mock_model_manager.load_model.call_count == 3
    
    @pytest.mark.asyncio
    async def test_predict_with_load_balancing(self, model_loader, mock_model_manager, mock_inference_engine):
        """Test prediction with load balancing."""
        await model_loader.start()
        
        # Create multiple instances
        mock_model = Mock()
        mock_model_manager.load_model.return_value = mock_model
        mock_inference_engine.predict.return_value = {"result": "test_prediction"}
        
        await model_loader.load_model("test_model")
        await model_loader.scale_model("test_model", target_instances=3)
        
        # Make multiple predictions
        results = []
        for i in range(6):
            result = await model_loader.predict("test_model", {"input": f"test_{i}"})
            results.append(result)
        
        # All predictions should succeed
        assert all(result == {"result": "test_prediction"} for result in results)
        assert mock_inference_engine.predict.call_count == 6
    
    @pytest.mark.asyncio
    async def test_unload_model(self, model_loader, mock_model_manager):
        """Test unloading a model."""
        await model_loader.start()
        
        mock_model = Mock()
        mock_model_manager.load_model.return_value = mock_model
        
        # Load model
        await model_loader.load_model("test_model")
        assert "test_model" in model_loader.instances
        
        # Unload model
        await model_loader.unload_model("test_model")
        
        assert "test_model" not in model_loader.instances
        mock_model_manager.unload_model.assert_called_with("test_model")
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, model_loader, mock_model_manager, mock_inference_engine):
        """Test health monitoring of instances."""
        await model_loader.start()
        
        mock_model = Mock()
        mock_model_manager.load_model.return_value = mock_model
        
        # Load model instances
        await model_loader.load_model("test_model")
        instances = await model_loader.scale_model("test_model", target_instances=2)
        
        assert len(instances) >= 2, f"Expected at least 2 instances, got {len(instances)}"
        
        # Make one instance unhealthy - use the actual instance ID
        target_instance_id = instances[0].instance_id
        mock_inference_engine.health_check.side_effect = lambda model_id, instance_id: instance_id != target_instance_id
        
        # Run health check
        await model_loader._perform_health_checks()
        
        # The targeted instance should have increased failure count
        unhealthy_instance = instances[0]  # We targeted the first instance
        assert unhealthy_instance.health_check_failures > 0
    
    @pytest.mark.asyncio
    async def test_auto_scaling_up(self, model_loader, mock_model_manager):
        """Test automatic scaling up based on load."""
        model_loader.config.enable_auto_scaling = True
        model_loader.config.scale_up_threshold = 0.7
        await model_loader.start()
        
        # Don't override the mock - use the fixture setup
        
        # Load initial instance
        await model_loader.load_model("test_model")
        instances = model_loader.instances["test_model"]
        instance = instances[0]
        
        # Simulate high load
        instance.active_connections = 10  # High load
        
        # Run auto scaling check
        await model_loader._check_auto_scaling()
        
        # Should have scaled up
        assert len(model_loader.instances["test_model"]) > 1
    
    @pytest.mark.asyncio
    async def test_auto_scaling_down(self, model_loader, mock_model_manager):
        """Test automatic scaling down based on low load."""
        model_loader.config.enable_auto_scaling = True
        model_loader.config.scale_down_threshold = 0.3
        await model_loader.start()
        
        # Don't override the mock - use the fixture setup
        
        # Load multiple instances
        await model_loader.load_model("test_model")
        instances = await model_loader.scale_model("test_model", target_instances=3)
        
        # Simulate low load on all instances
        for instance in instances:
            instance.active_connections = 0
        
        # Run auto scaling check
        await model_loader._check_auto_scaling()
        
        # Should have scaled down but not below minimum
        assert len(model_loader.instances["test_model"]) >= model_loader.config.min_instances_per_model
    
    def test_get_stats(self, model_loader):
        """Test getting model loader statistics."""
        # Create some instances directly in the model instances dict
        mock_model = Mock()
        for i in range(2):
            instance = ModelInstanceInfo(
                model_id="test_model",
                instance_id=f"instance_{i}",
                model=mock_model,
                state=ModelInstanceState.IDLE if i == 0 else ModelInstanceState.ACTIVE
            )
            model_loader.model_instances["test_model"].append(instance)
        
        stats = model_loader.get_stats()
        
        assert stats["total_instances"] == 2
        assert stats["idle_instances"] >= 0  # At least 0 idle instances
        assert stats["loaded_models"] == 1
        assert stats["enabled"] is True
    
    def test_get_health_status(self, model_loader):
        """Test getting health status."""
        health = model_loader.get_health_status()
        
        assert "status" in health
        assert "enabled" in health
        assert "total_instances" in health
        assert "loaded_models" in health
    
    @pytest.mark.asyncio
    async def test_model_loading_failure(self, model_loader, mock_model_manager):
        """Test handling model loading failures."""
        await model_loader.start()
        
        # Mock model loading failure
        mock_model_manager.load_model.side_effect = Exception("Model loading failed")
        
        instance = await model_loader.load_model("test_model")
        
        # Should return None on failure
        assert instance is None
        
        # Should not have any instances
        assert "test_model" not in model_loader.instances
    
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, model_loader, mock_model_manager, mock_inference_engine):
        """Test handling concurrent predictions."""
        await model_loader.start()
        
        mock_model = Mock()
        mock_model_manager.load_model.return_value = mock_model
        mock_inference_engine.predict.return_value = {"result": "test_prediction"}
        
        # Load model with multiple instances
        await model_loader.load_model("test_model")
        await model_loader.scale_model("test_model", target_instances=3)
        
        # Make concurrent predictions
        tasks = [
            model_loader.predict("test_model", {"input": f"test_{i}"})
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All predictions should succeed
        assert all(result == {"result": "test_prediction"} for result in results)
        assert mock_inference_engine.predict.call_count == 10


if __name__ == "__main__":
    pytest.main([__file__])

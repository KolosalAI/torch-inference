"""Unit tests for main Autoscaler functionality."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, List

from framework.autoscaling.autoscaler import (
    Autoscaler,
    AutoscalerConfig,
    ScalingDecision,
    ScalingAction
)
from framework.autoscaling.zero_scaler import ZeroScaler, ZeroScalingConfig
from framework.autoscaling.model_loader import DynamicModelLoader, ModelLoaderConfig
from framework.autoscaling.metrics import MetricsCollector
from framework.core.base_model import ModelManager


@pytest.fixture
def autoscaler_config():
    """Create a test autoscaler configuration."""
    return AutoscalerConfig(
        enable_zero_scaling=True,
        enable_dynamic_loading=True,
        enable_monitoring=True,
        monitoring_interval=1.0,
        scaling_cooldown=5.0,
        max_concurrent_scalings=3,
        enable_predictive_scaling=True
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
def mock_zero_scaler():
    """Create a mock zero scaler."""
    scaler = Mock(spec=ZeroScaler)
    scaler.start = AsyncMock()
    scaler.stop = AsyncMock()
    scaler.predict = AsyncMock()
    scaler.ensure_model_loaded = AsyncMock()
    scaler.is_running = True
    scaler.get_stats = Mock(return_value={
        "total_instances": 2,
        "idle_instances": 1,
        "active_instances": 1,
        "loaded_models": 2
    })
    scaler.get_health_status = Mock(return_value={"status": "healthy"})
    return scaler


@pytest.fixture
def mock_model_loader():
    """Create a mock model loader."""
    loader = Mock(spec=DynamicModelLoader)
    loader.start = AsyncMock()
    loader.stop = AsyncMock()
    loader.predict = AsyncMock()
    loader.load_model = AsyncMock()
    loader.scale_model = AsyncMock()
    loader.unload_model = AsyncMock()
    loader.is_running = True
    loader.get_stats = Mock(return_value={
        "total_instances": 3,
        "idle_instances": 1,
        "active_instances": 2,
        "loaded_models": 2
    })
    loader.get_health_status = Mock(return_value={"status": "healthy"})
    return loader


@pytest.fixture
def mock_metrics_collector():
    """Create a mock metrics collector."""
    collector = Mock(spec=MetricsCollector)
    collector.start = AsyncMock()
    collector.stop = AsyncMock()
    collector.record_prediction = Mock()
    collector.record_scaling_event = Mock()
    collector.get_metrics = Mock(return_value={
        "request_rate": 10.5,
        "average_response_time": 0.05,
        "error_rate": 0.01
    })
    collector.get_prometheus_metrics = Mock(return_value="# Prometheus metrics")
    return collector


@pytest.fixture
def autoscaler(autoscaler_config, mock_model_manager, mock_inference_engine):
    """Create an autoscaler instance for testing."""
    return Autoscaler(
        config=autoscaler_config,
        model_manager=mock_model_manager,
        inference_engine=mock_inference_engine
    )


@pytest.fixture
def autoscaler_with_mocks(
    autoscaler_config, 
    mock_zero_scaler, 
    mock_model_loader, 
    mock_metrics_collector
):
    """Create an autoscaler with mocked components."""
    autoscaler = Autoscaler.__new__(Autoscaler)
    autoscaler.config = autoscaler_config
    autoscaler.zero_scaler = mock_zero_scaler
    autoscaler.model_loader = mock_model_loader
    autoscaler.metrics_collector = mock_metrics_collector
    
    # Initialize required attributes that would normally be set in __init__
    from framework.autoscaling.autoscaler import AutoscalerState
    import asyncio
    autoscaler.state = AutoscalerState.STOPPED
    autoscaler.state_lock = asyncio.Lock()
    autoscaler.model_manager = Mock()
    autoscaler.inference_engine = Mock()
    autoscaler.request_statistics = {}
    autoscaler.performance_history = []
    autoscaler.scaling_events = []
    autoscaler.active_alerts = set()
    autoscaler.alert_callbacks = []
    autoscaler.monitoring_task = None
    autoscaler.optimization_task = None 
    autoscaler.alerting_task = None
    autoscaler.logger = Mock()
    
    # Test-specific attributes
    autoscaler.scaling_locks = {}
    autoscaler.last_scaling_times = {}
    autoscaler.prediction_history = []
    autoscaler.scaling_history = []
    return autoscaler


class TestAutoscalerConfig:
    """Test autoscaler configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AutoscalerConfig()
        
        assert config.enable_zero_scaling is True
        assert config.enable_dynamic_loading is True
        assert config.enable_monitoring is True
        assert config.monitoring_interval == 30.0
        assert config.scaling_cooldown == 300.0
        assert config.max_concurrent_scalings == 3
        assert config.enable_predictive_scaling is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AutoscalerConfig(
            enable_zero_scaling=False,
            enable_dynamic_loading=False,
            monitoring_interval=10.0,
            scaling_cooldown=30.0,
            max_concurrent_scalings=3
        )
        
        assert config.enable_zero_scaling is False
        assert config.enable_dynamic_loading is False
        assert config.monitoring_interval == 10.0
        assert config.scaling_cooldown == 30.0
        assert config.max_concurrent_scalings == 3
    
    def test_validation_errors(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="monitoring_interval must be positive"):
            AutoscalerConfig(monitoring_interval=-1.0)
        
        with pytest.raises(ValueError, match="scaling_cooldown must be positive"):
            AutoscalerConfig(scaling_cooldown=-1.0)
        
        with pytest.raises(ValueError, match="max_concurrent_scalings must be positive"):
            AutoscalerConfig(max_concurrent_scalings=0)


class TestScalingDecision:
    """Test scaling decision logic."""
    
    def test_scaling_decision_creation(self):
        """Test creating a scaling decision."""
        decision = ScalingDecision(
            model_id="test_model",
            action=ScalingAction.SCALE_UP,
            current_instances=2,
            target_instances=3,
            reason="High load detected",
            confidence=0.8
        )
        
        assert decision.model_id == "test_model"
        assert decision.action == ScalingAction.SCALE_UP
        assert decision.current_instances == 2
        assert decision.target_instances == 3
        assert decision.reason == "High load detected"
        assert decision.confidence == 0.8
        assert decision.timestamp is not None
    
    def test_should_execute(self):
        """Test whether a scaling decision should be executed."""
        decision = ScalingDecision(
            model_id="test_model",
            action=ScalingAction.SCALE_UP,
            current_instances=2,
            target_instances=3,
            reason="High request rate",
            confidence=0.8
        )
        
        # High confidence should execute
        assert decision.should_execute(min_confidence=0.7) is True
        
        # Low confidence should not execute
        assert decision.should_execute(min_confidence=0.9) is False


class TestAutoscaler:
    """Test main autoscaler functionality."""
    
    @pytest.mark.asyncio
    async def test_autoscaler_start_stop(self, autoscaler_with_mocks):
        """Test starting and stopping autoscaler."""
        assert not autoscaler_with_mocks.is_running
        
        await autoscaler_with_mocks.start()
        assert autoscaler_with_mocks.is_running
        
        # Should start all components
        autoscaler_with_mocks.zero_scaler.start.assert_called_once()
        autoscaler_with_mocks.model_loader.start.assert_called_once()
        autoscaler_with_mocks.metrics_collector.start.assert_called_once()
        
        await autoscaler_with_mocks.stop()
        assert not autoscaler_with_mocks.is_running
        
        # Should stop all components
        autoscaler_with_mocks.zero_scaler.stop.assert_called_once()
        autoscaler_with_mocks.model_loader.stop.assert_called_once()
        autoscaler_with_mocks.metrics_collector.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_predict_routing(self, autoscaler_with_mocks):
        """Test prediction routing logic."""
        await autoscaler_with_mocks.start()
        
        # Configure mocks
        autoscaler_with_mocks.zero_scaler.predict.return_value = {"result": "zero_scaler"}
        autoscaler_with_mocks.model_loader.predict.return_value = {"result": "model_loader"}
        
        # Test with zero scaling enabled
        autoscaler_with_mocks.config.enable_zero_scaling = True
        autoscaler_with_mocks.config.enable_dynamic_loading = False
        
        result = await autoscaler_with_mocks.predict("test_model", {"input": "test"})
        
        assert result == {"result": "zero_scaler"}
        autoscaler_with_mocks.zero_scaler.predict.assert_called_once()
        autoscaler_with_mocks.model_loader.predict.assert_not_called()
        
        # Reset mocks
        autoscaler_with_mocks.zero_scaler.predict.reset_mock()
        autoscaler_with_mocks.model_loader.predict.reset_mock()
        
        # Test with dynamic loading enabled
        autoscaler_with_mocks.config.enable_zero_scaling = False
        autoscaler_with_mocks.config.enable_dynamic_loading = True
        
        result = await autoscaler_with_mocks.predict("test_model", {"input": "test"})
        
        assert result == {"result": "model_loader"}
        autoscaler_with_mocks.zero_scaler.predict.assert_not_called()
        autoscaler_with_mocks.model_loader.predict.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_predict_records_metrics(self, autoscaler_with_mocks):
        """Test that predictions are recorded in metrics."""
        await autoscaler_with_mocks.start()
        
        autoscaler_with_mocks.zero_scaler.predict.return_value = {"result": "test"}
        
        await autoscaler_with_mocks.predict("test_model", {"input": "test"})
        
        # Should record the prediction
        autoscaler_with_mocks.metrics_collector.record_prediction.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_scale_model(self, autoscaler_with_mocks):
        """Test scaling a model."""
        await autoscaler_with_mocks.start()
        
        # Mock successful scaling
        mock_instances = [Mock(), Mock(), Mock()]
        autoscaler_with_mocks.model_loader.scale_model.return_value = mock_instances
        
        result = await autoscaler_with_mocks.scale_model("test_model", target_instances=3)
        
        assert result == mock_instances
        autoscaler_with_mocks.model_loader.scale_model.assert_called_once_with(
            "test_model", target_instances=3
        )
        
        # Should record scaling event
        autoscaler_with_mocks.metrics_collector.record_scaling_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_model(self, autoscaler_with_mocks):
        """Test loading a model."""
        await autoscaler_with_mocks.start()
        
        mock_instance = Mock()
        autoscaler_with_mocks.model_loader.load_model.return_value = mock_instance
        
        result = await autoscaler_with_mocks.load_model("test_model")
        
        assert result == mock_instance
        autoscaler_with_mocks.model_loader.load_model.assert_called_once_with("test_model", "v1")
    
    @pytest.mark.asyncio
    async def test_unload_model(self, autoscaler_with_mocks):
        """Test unloading a model."""
        await autoscaler_with_mocks.start()
        
        await autoscaler_with_mocks.unload_model("test_model")
        
        autoscaler_with_mocks.model_loader.unload_model.assert_called_once_with("test_model", None)
    
    @pytest.mark.asyncio
    async def test_scaling_cooldown(self, autoscaler_with_mocks):
        """Test scaling cooldown functionality."""
        await autoscaler_with_mocks.start()
        
        # Record a recent scaling event
        autoscaler_with_mocks.last_scaling_times["test_model"] = time.time()
        
        # Try to scale again immediately
        mock_instances = [Mock()]
        autoscaler_with_mocks.model_loader.scale_model.return_value = mock_instances
        
        result = await autoscaler_with_mocks.scale_model("test_model", target_instances=2)
        
        # Should be blocked by cooldown
        assert result is None
        autoscaler_with_mocks.model_loader.scale_model.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_concurrent_scaling_limit(self, autoscaler_with_mocks):
        """Test concurrent scaling limit."""
        await autoscaler_with_mocks.start()
        autoscaler_with_mocks.config.max_concurrent_scalings = 1
        
        # Start first scaling operation
        mock_instances = [Mock()]
        autoscaler_with_mocks.model_loader.scale_model.return_value = mock_instances
        
        # Mock scaling to take time
        async def slow_scale(model_id, target_instances):
            await asyncio.sleep(0.1)
            return mock_instances
        
        autoscaler_with_mocks.model_loader.scale_model.side_effect = slow_scale
        
        # Start concurrent scaling operations
        tasks = [
            autoscaler_with_mocks.scale_model(f"model_{i}", target_instances=2)
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Only one should succeed due to limit
        successful_results = [r for r in results if r is not None]
        assert len(successful_results) == 1
    
    def test_get_stats(self, autoscaler_with_mocks):
        """Test getting autoscaler statistics."""
        stats = autoscaler_with_mocks.get_stats()
        
        assert "zero_scaler" in stats
        assert "model_loader" in stats
        assert "total_instances" in stats
        assert "loaded_models" in stats
        assert "scaling_history" in stats
        assert "prediction_history" in stats
    
    def test_get_health_status(self, autoscaler_with_mocks):
        """Test getting health status."""
        health = autoscaler_with_mocks.get_health_status()
        
        assert "status" in health
        assert "components" in health
        assert "zero_scaler" in health["components"]
        assert "model_loader" in health["components"]
        assert "metrics_collector" in health["components"]
    
    def test_get_metrics(self, autoscaler_with_mocks):
        """Test getting metrics."""
        metrics = autoscaler_with_mocks.get_metrics()
        
        assert "request_rate" in metrics
        assert "average_response_time" in metrics
        assert "error_rate" in metrics
        
        # Should also include stats
        assert "stats" in metrics
    
    @pytest.mark.asyncio
    async def test_make_scaling_decision(self, autoscaler_with_mocks):
        """Test making scaling decisions."""
        await autoscaler_with_mocks.start()
        
        # Mock metrics that suggest scaling up
        autoscaler_with_mocks.metrics_collector.get_metrics.return_value = {
            "models": {
                "test_model": {
                    "request_rate": 100.0,
                    "average_response_time": 0.5,
                    "active_instances": 1,
                    "average_load": 0.9
                }
            }
        }
        
        decision = await autoscaler_with_mocks._make_scaling_decision("test_model")
        
        assert decision is not None
        assert decision.model_id == "test_model"
        assert decision.action == ScalingAction.SCALE_UP
        assert decision.target_instances > decision.current_instances
    
    @pytest.mark.asyncio
    async def test_execute_scaling_decision(self, autoscaler_with_mocks):
        """Test executing scaling decisions."""
        await autoscaler_with_mocks.start()
        
        decision = ScalingDecision(
            model_id="test_model",
            action=ScalingAction.SCALE_UP,
            current_instances=1,
            target_instances=2,
            confidence=0.9
        )
        
        mock_instances = [Mock(), Mock()]
        autoscaler_with_mocks.model_loader.scale_model.return_value = mock_instances
        
        result = await autoscaler_with_mocks._execute_scaling_decision(decision)
        
        assert result is True
        autoscaler_with_mocks.model_loader.scale_model.assert_called_once_with(
            "test_model", target_instances=2
        )
    
    @pytest.mark.asyncio
    async def test_predictive_scaling(self, autoscaler_with_mocks):
        """Test predictive scaling functionality."""
        autoscaler_with_mocks.config.enable_predictive_scaling = True
        await autoscaler_with_mocks.start()
        
        # Add some prediction history to analyze
        for i in range(10):
            prediction_info = {
                "model_id": "test_model",
                "timestamp": datetime.now() - timedelta(minutes=i),
                "response_time": 0.1 + (i * 0.01)
            }
            autoscaler_with_mocks.prediction_history.append(prediction_info)
        
        # Mock metrics
        autoscaler_with_mocks.metrics_collector.get_metrics.return_value = {
            "models": {
                "test_model": {
                    "request_rate": 50.0,
                    "trend": "increasing"
                }
            }
        }
        
        prediction = await autoscaler_with_mocks._predict_future_load("test_model")
        
        assert prediction is not None
        assert "predicted_load" in prediction
        assert "confidence" in prediction
    
    @pytest.mark.asyncio
    async def test_monitoring_loop(self, autoscaler_with_mocks):
        """Test monitoring loop functionality."""
        autoscaler_with_mocks.config.monitoring_interval = 0.1  # Fast for testing
        await autoscaler_with_mocks.start()
        
        # Mock metrics that don't require scaling
        autoscaler_with_mocks.metrics_collector.get_metrics.return_value = {
            "models": {
                "test_model": {
                    "request_rate": 10.0,
                    "average_response_time": 0.1,
                    "active_instances": 2,
                    "average_load": 0.5
                }
            }
        }
        
        # Let monitoring loop run briefly
        await asyncio.sleep(0.2)
        
        # Metrics should have been checked
        assert autoscaler_with_mocks.metrics_collector.get_metrics.call_count > 0
    
    @pytest.mark.asyncio
    async def test_component_initialization(self, autoscaler, mock_model_manager, mock_inference_engine):
        """Test that components are properly initialized."""
        assert autoscaler.zero_scaler is not None
        assert autoscaler.model_loader is not None
        assert autoscaler.metrics_collector is not None
        
        # Components should have correct configuration
        assert autoscaler.zero_scaler.config.enabled == autoscaler.config.enable_zero_scaling
        assert autoscaler.model_loader.config.enabled == autoscaler.config.enable_dynamic_loading
    
    @pytest.mark.asyncio
    async def test_error_handling_in_prediction(self, autoscaler_with_mocks):
        """Test error handling during prediction."""
        await autoscaler_with_mocks.start()
        
        # Mock prediction failure
        autoscaler_with_mocks.zero_scaler.predict.side_effect = Exception("Prediction failed")
        
        result = await autoscaler_with_mocks.predict("test_model", {"input": "test"})
        
        # Should return None on error
        assert result is None
    
    @pytest.mark.asyncio
    async def test_error_handling_in_scaling(self, autoscaler_with_mocks):
        """Test error handling during scaling."""
        await autoscaler_with_mocks.start()
        
        # Mock scaling failure
        autoscaler_with_mocks.model_loader.scale_model.side_effect = Exception("Scaling failed")
        
        result = await autoscaler_with_mocks.scale_model("test_model", target_instances=3)
        
        # Should return None on error
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])

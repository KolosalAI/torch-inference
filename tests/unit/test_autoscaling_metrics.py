"""Unit tests for Metrics Collector functionality."""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, List

from framework.autoscaling.metrics import (
    MetricsCollector,
    MetricsConfig,
    ScalingMetrics,
    TimeSeriesMetric,
    AlertConfig,
    AlertThreshold,
    AlertChannel
)


@pytest.fixture
def metrics_config():
    """Create a test metrics configuration."""
    return MetricsConfig(
        enabled=True,
        collection_interval=1.0,
        retention_period=3600.0,  # 1 hour for testing
        enable_prometheus=True,
        enable_alerts=True,
        alert_check_interval=5.0
    )


@pytest.fixture
def alert_config():
    """Create a test alert configuration."""
    return AlertConfig(
        enabled=True,
        thresholds={
            "response_time": AlertThreshold(warning=0.1, critical=0.5),
            "error_rate": AlertThreshold(warning=0.05, critical=0.1),
            "memory_usage": AlertThreshold(warning=0.8, critical=0.9)
        },
        channels=[AlertChannel.SLACK],
        cooldown_period=300.0
    )


@pytest.fixture
def metrics_collector(metrics_config, alert_config):
    """Create a metrics collector instance for testing."""
    return MetricsCollector(
        config=metrics_config,
        alert_config=alert_config
    )


class TestMetricsConfig:
    """Test metrics configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MetricsConfig()
        
        assert config.enabled is True
        assert config.collection_interval == 30.0
        assert config.retention_period == 86400.0  # 24 hours
        assert config.enable_prometheus is True
        assert config.enable_alerts is True
        assert config.alert_check_interval == 60.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MetricsConfig(
            enabled=False,
            collection_interval=10.0,
            retention_period=3600.0,
            enable_prometheus=False,
            enable_alerts=False
        )
        
        assert config.enabled is False
        assert config.collection_interval == 10.0
        assert config.retention_period == 3600.0
        assert config.enable_prometheus is False
        assert config.enable_alerts is False
    
    def test_validation_errors(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="collection_interval must be positive"):
            MetricsConfig(collection_interval=-1.0)
        
        with pytest.raises(ValueError, match="retention_period must be positive"):
            MetricsConfig(retention_period=-1.0)
        
        with pytest.raises(ValueError, match="alert_check_interval must be positive"):
            MetricsConfig(alert_check_interval=-1.0)


class TestAlertConfig:
    """Test alert configuration."""
    
    def test_default_alert_config(self):
        """Test default alert configuration."""
        config = AlertConfig()
        
        assert config.enabled is True
        assert len(config.thresholds) > 0
        assert AlertChannel.SLACK in config.channels
        assert config.cooldown_period == 300.0
    
    def test_custom_alert_config(self):
        """Test custom alert configuration."""
        thresholds = {
            "response_time": AlertThreshold(warning=0.2, critical=1.0)
        }
        config = AlertConfig(
            enabled=False,
            thresholds=thresholds,
            channels=[AlertChannel.EMAIL],
            cooldown_period=600.0
        )
        
        assert config.enabled is False
        assert config.thresholds == thresholds
        assert config.channels == [AlertChannel.EMAIL]
        assert config.cooldown_period == 600.0


class TestScalingMetrics:
    """Test scaling metrics functionality."""
    
    def test_scaling_metrics_creation(self):
        """Test creating scaling metrics."""
        metrics = ScalingMetrics(
            model_id="test_model",
            timestamp=datetime.now(),
            active_instances=2,
            total_instances=3,
            request_rate=10.5,
            average_response_time=0.1,
            cpu_usage=0.7,
            memory_usage=0.6,
            error_rate=0.02
        )
        
        assert metrics.model_id == "test_model"
        assert metrics.active_instances == 2
        assert metrics.total_instances == 3
        assert metrics.request_rate == 10.5
        assert metrics.average_response_time == 0.1
        assert metrics.cpu_usage == 0.7
        assert metrics.memory_usage == 0.6
        assert metrics.error_rate == 0.02
    
    def test_scaling_metrics_to_dict(self):
        """Test converting scaling metrics to dictionary."""
        metrics = ScalingMetrics(
            model_id="test_model",
            timestamp=datetime.now(),
            active_instances=2,
            total_instances=3,
            request_rate=10.5
        )
        
        data = metrics.to_dict()
        
        assert data["model_id"] == "test_model"
        assert data["active_instances"] == 2
        assert data["total_instances"] == 3
        assert data["request_rate"] == 10.5
        assert "timestamp" in data


class TestTimeSeriesMetric:
    """Test time series metric functionality."""
    
    def test_time_series_creation(self):
        """Test creating a time series metric."""
        ts_metric = TimeSeriesMetric(
            name="response_time",
            max_points=100,
            retention_period=3600.0
        )
        
        assert ts_metric.name == "response_time"
        assert ts_metric.max_points == 100
        assert ts_metric.retention_period == 3600.0
        assert len(ts_metric.values) == 0
    
    def test_add_value(self):
        """Test adding values to time series."""
        ts_metric = TimeSeriesMetric("test_metric", max_points=5)
        
        # Add values
        for i in range(3):
            ts_metric.add_value(float(i), datetime.now())
        
        assert len(ts_metric.values) == 3
        assert ts_metric.values[0][0] == 0.0
        assert ts_metric.values[1][0] == 1.0
        assert ts_metric.values[2][0] == 2.0
    
    def test_max_points_limit(self):
        """Test max points limit."""
        ts_metric = TimeSeriesMetric("test_metric", max_points=3)
        
        # Add more values than limit
        for i in range(5):
            ts_metric.add_value(float(i), datetime.now())
        
        # Should only keep last 3 values
        assert len(ts_metric.values) == 3
        assert ts_metric.values[0][0] == 2.0
        assert ts_metric.values[1][0] == 3.0
        assert ts_metric.values[2][0] == 4.0
    
    def test_retention_period_cleanup(self):
        """Test retention period cleanup."""
        ts_metric = TimeSeriesMetric("test_metric", retention_period=1.0)  # 1 second
        
        # Add old values
        old_time = datetime.now() - timedelta(seconds=2)
        ts_metric.add_value(1.0, old_time)
        
        # Add recent value
        ts_metric.add_value(2.0, datetime.now())
        
        # Cleanup should remove old values
        ts_metric.cleanup_old_values()
        
        assert len(ts_metric.values) == 1
        assert ts_metric.values[0][0] == 2.0
    
    def test_get_recent_values(self):
        """Test getting recent values."""
        ts_metric = TimeSeriesMetric("test_metric")
        
        now = datetime.now()
        ts_metric.add_value(1.0, now - timedelta(seconds=10))
        ts_metric.add_value(2.0, now - timedelta(seconds=5))
        ts_metric.add_value(3.0, now)
        
        recent = ts_metric.get_recent_values(duration=timedelta(seconds=6))
        
        # Should get last 2 values
        assert len(recent) == 2
        assert recent[0][0] == 2.0
        assert recent[1][0] == 3.0
    
    def test_get_average(self):
        """Test getting average value."""
        ts_metric = TimeSeriesMetric("test_metric")
        
        now = datetime.now()
        ts_metric.add_value(1.0, now)
        ts_metric.add_value(2.0, now)
        ts_metric.add_value(3.0, now)
        
        avg = ts_metric.get_average()
        assert avg == 2.0
        
        # Test with duration
        avg_recent = ts_metric.get_average(duration=timedelta(seconds=1))
        assert avg_recent == 2.0
    
    def test_get_percentile(self):
        """Test getting percentile values."""
        ts_metric = TimeSeriesMetric("test_metric")
        
        now = datetime.now()
        for i in range(1, 11):  # Values 1-10
            ts_metric.add_value(float(i), now)
        
        p50 = ts_metric.get_percentile(50)
        p95 = ts_metric.get_percentile(95)
        
        assert p50 == 5.5  # Median of 1-10
        assert p95 == 9.5  # 95th percentile
    
    def test_to_dict(self):
        """Test converting time series to dictionary."""
        ts_metric = TimeSeriesMetric("test_metric")
        
        now = datetime.now()
        ts_metric.add_value(1.0, now)
        ts_metric.add_value(2.0, now)
        
        data = ts_metric.to_dict()
        
        assert data["name"] == "test_metric"
        assert data["count"] == 2
        assert data["average"] == 1.5
        assert "values" in data
        assert len(data["values"]) == 2


class TestMetricsCollector:
    """Test metrics collector functionality."""
    
    @pytest.mark.asyncio
    async def test_metrics_collector_start_stop(self, metrics_collector):
        """Test starting and stopping metrics collector."""
        assert not metrics_collector.is_running
        
        await metrics_collector.start()
        assert metrics_collector.is_running
        
        await metrics_collector.stop()
        assert not metrics_collector.is_running
    
    @pytest.mark.asyncio
    async def test_disabled_collector(self, alert_config):
        """Test metrics collector when disabled."""
        config = MetricsConfig(enabled=False)
        collector = MetricsCollector(config, alert_config)
        
        await collector.start()
        assert not collector.is_running
        
        # Should not do anything when disabled
        collector.record_prediction("test_model", 0.1, True)
        metrics = collector.get_metrics()
        assert metrics == {}
    
    def test_record_prediction(self, metrics_collector):
        """Test recording prediction metrics."""
        metrics_collector.record_prediction("test_model", 0.1, True)
        metrics_collector.record_prediction("test_model", 0.2, False)
        
        model_metrics = metrics_collector.model_metrics["test_model"]
        
        assert model_metrics["request_count"] == 2
        assert model_metrics["success_count"] == 1
        assert model_metrics["error_count"] == 1
        assert len(model_metrics["response_times"].values) == 2
    
    def test_record_scaling_event(self, metrics_collector):
        """Test recording scaling events."""
        metrics_collector.record_scaling_event(
            "test_model",
            "scale_up",
            old_instances=1,
            new_instances=2,
            reason="High load"
        )
        
        assert len(metrics_collector.scaling_events) == 1
        event = metrics_collector.scaling_events[0]
        
        assert event["model_id"] == "test_model"
        assert event["action"] == "scale_up"
        assert event["old_instances"] == 1
        assert event["new_instances"] == 2
        assert event["reason"] == "High load"
    
    def test_record_resource_usage(self, metrics_collector):
        """Test recording resource usage."""
        metrics_collector.record_resource_usage("test_model", cpu=0.7, memory=0.6, gpu=0.8)
        
        model_metrics = metrics_collector.model_metrics["test_model"]
        
        assert len(model_metrics["cpu_usage"].values) == 1
        assert len(model_metrics["memory_usage"].values) == 1
        assert len(model_metrics["gpu_usage"].values) == 1
        assert model_metrics["cpu_usage"].values[0][0] == 0.7
    
    def test_get_metrics_basic(self, metrics_collector):
        """Test getting basic metrics."""
        # Record some data
        metrics_collector.record_prediction("test_model", 0.1, True)
        metrics_collector.record_prediction("test_model", 0.2, True)
        metrics_collector.record_resource_usage("test_model", cpu=0.5, memory=0.4)
        
        metrics = metrics_collector.get_metrics()
        
        assert "models" in metrics
        assert "test_model" in metrics["models"]
        
        model_data = metrics["models"]["test_model"]
        assert model_data["request_count"] == 2
        assert model_data["success_count"] == 2
        assert model_data["error_count"] == 0
        assert model_data["average_response_time"] == 0.15
        assert model_data["error_rate"] == 0.0
    
    def test_get_prometheus_metrics(self, metrics_collector):
        """Test getting Prometheus format metrics."""
        # Record some data
        metrics_collector.record_prediction("test_model", 0.1, True)
        metrics_collector.record_resource_usage("test_model", cpu=0.5, memory=0.4)
        
        prometheus_text = metrics_collector.get_prometheus_metrics()
        
        assert isinstance(prometheus_text, str)
        assert "autoscaler_requests_total" in prometheus_text
        assert "autoscaler_response_time_seconds" in prometheus_text
        assert "autoscaler_cpu_usage" in prometheus_text
        assert "autoscaler_memory_usage" in prometheus_text
    
    def test_calculate_request_rate(self, metrics_collector):
        """Test calculating request rate."""
        now = datetime.now()
        
        # Record requests over time
        for i in range(5):
            metrics_collector.model_metrics["test_model"] = {
                "request_timestamps": []
            }
            for j in range(i + 1):
                timestamp = now - timedelta(seconds=60 - j * 10)
                metrics_collector.model_metrics["test_model"]["request_timestamps"].append(timestamp)
        
        # Mock the current implementation
        metrics_collector.model_metrics["test_model"]["request_timestamps"] = [
            now - timedelta(seconds=50),
            now - timedelta(seconds=30),
            now - timedelta(seconds=10),
        ]
        
        rate = metrics_collector._calculate_request_rate("test_model")
        
        # Should be 3 requests per minute = 3.0 requests/minute
        assert rate == 3.0
    
    def test_detect_trends(self, metrics_collector):
        """Test trend detection."""
        ts_metric = TimeSeriesMetric("test_metric")
        
        now = datetime.now()
        # Add increasing trend
        for i in range(10):
            ts_metric.add_value(float(i), now - timedelta(seconds=10 - i))
        
        trend = metrics_collector._detect_trend(ts_metric)
        
        assert trend == "increasing"
        
        # Test decreasing trend
        ts_metric = TimeSeriesMetric("test_metric")
        for i in range(10, 0, -1):
            ts_metric.add_value(float(i), now - timedelta(seconds=10 - i))
        
        trend = metrics_collector._detect_trend(ts_metric)
        assert trend == "decreasing"
    
    @pytest.mark.asyncio
    async def test_check_alerts(self, metrics_collector):
        """Test alert checking."""
        # Record metrics that exceed thresholds
        metrics_collector.record_prediction("test_model", 0.6, True)  # Exceeds critical response time
        
        with patch.object(metrics_collector, '_send_alert') as mock_send_alert:
            await metrics_collector._check_alerts()
            
            # Should trigger alert for high response time
            mock_send_alert.assert_called()
            call_args = mock_send_alert.call_args[0]
            assert "response_time" in call_args[0]  # metric_name
            assert call_args[1] == "critical"  # severity
    
    @pytest.mark.asyncio
    async def test_alert_cooldown(self, metrics_collector):
        """Test alert cooldown functionality."""
        metrics_collector.alert_history["test_model_response_time"] = datetime.now() - timedelta(seconds=100)
        
        # Should not send alert during cooldown
        with patch.object(metrics_collector, '_send_alert') as mock_send_alert:
            await metrics_collector._check_alert_threshold(
                "test_model",
                "response_time",
                0.6,
                AlertThreshold(warning=0.1, critical=0.5)
            )
            
            mock_send_alert.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_slack_alert_sending(self, metrics_collector):
        """Test sending Slack alerts."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 200
            
            await metrics_collector._send_slack_alert(
                "test_model",
                "response_time",
                "critical",
                0.6,
                0.5
            )
            
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "slack.com" in call_args[0][0]  # URL
            
            # Check message content
            json_data = call_args[1]["json"]
            assert "test_model" in json_data["text"]
            assert "response_time" in json_data["text"]
            assert "critical" in json_data["text"]
    
    def test_cleanup_old_metrics(self, metrics_collector):
        """Test cleaning up old metrics."""
        # Add old metrics
        old_time = datetime.now() - timedelta(hours=2)
        metrics_collector.model_metrics["test_model"] = {
            "response_times": TimeSeriesMetric("response_times", retention_period=3600.0),
            "cpu_usage": TimeSeriesMetric("cpu_usage", retention_period=3600.0)
        }
        
        # Add old values
        metrics_collector.model_metrics["test_model"]["response_times"].add_value(0.1, old_time)
        metrics_collector.model_metrics["test_model"]["cpu_usage"].add_value(0.5, old_time)
        
        # Add recent values
        metrics_collector.model_metrics["test_model"]["response_times"].add_value(0.2, datetime.now())
        metrics_collector.model_metrics["test_model"]["cpu_usage"].add_value(0.6, datetime.now())
        
        # Cleanup
        metrics_collector._cleanup_old_metrics()
        
        # Should only have recent values
        assert len(metrics_collector.model_metrics["test_model"]["response_times"].values) == 1
        assert len(metrics_collector.model_metrics["test_model"]["cpu_usage"].values) == 1
    
    @pytest.mark.asyncio
    async def test_metrics_collection_loop(self, metrics_collector):
        """Test metrics collection loop."""
        metrics_collector.config.collection_interval = 0.1  # Fast for testing
        
        with patch.object(metrics_collector, '_collect_system_metrics') as mock_collect:
            await metrics_collector.start()
            
            # Let collection loop run briefly
            await asyncio.sleep(0.2)
            
            # Should have collected metrics
            assert mock_collect.call_count > 0
            
            await metrics_collector.stop()
    
    def test_export_metrics_json(self, metrics_collector):
        """Test exporting metrics as JSON."""
        # Add some data
        metrics_collector.record_prediction("test_model", 0.1, True)
        metrics_collector.record_scaling_event("test_model", "scale_up", 1, 2)
        
        json_data = metrics_collector.export_metrics_json()
        
        # Should be valid JSON
        parsed = json.loads(json_data)
        
        assert "models" in parsed
        assert "scaling_events" in parsed
        assert "timestamp" in parsed
    
    def test_import_metrics_json(self, metrics_collector):
        """Test importing metrics from JSON."""
        # Create test data
        test_data = {
            "models": {
                "test_model": {
                    "request_count": 10,
                    "success_count": 9,
                    "error_count": 1
                }
            },
            "scaling_events": [
                {
                    "model_id": "test_model",
                    "action": "scale_up",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }
        
        json_data = json.dumps(test_data, default=str)
        
        metrics_collector.import_metrics_json(json_data)
        
        # Should have imported the data
        assert len(metrics_collector.scaling_events) == 1
        assert metrics_collector.scaling_events[0]["model_id"] == "test_model"


if __name__ == "__main__":
    pytest.main([__file__])

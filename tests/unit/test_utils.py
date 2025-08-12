"""Tests for utility modules."""

import pytest
import time
from unittest.mock import Mock, patch
from collections import deque

from framework.utils.monitoring import (
    PerformanceMonitor, MetricsCollector, PerformanceStats, PerformanceMetrics,
    get_performance_monitor, get_metrics_collector, Metric, MetricType
)


class TestPerformanceMetrics:
    """Test performance metrics data structure."""
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            inference_time=0.025,
            preprocessing_time=0.002,
            postprocessing_time=0.001,
            total_time=0.028,
            throughput=35.7,
            memory_usage=1024*1024*50,  # 50MB
            gpu_utilization=75.5
        )
        
        assert metrics.inference_time == 0.025
        assert metrics.preprocessing_time == 0.002
        assert metrics.postprocessing_time == 0.001
        assert metrics.total_time == 0.028
        assert metrics.throughput == 35.7
        assert metrics.memory_usage == 1024*1024*50
        assert metrics.gpu_utilization == 75.5
    
    def test_performance_metrics_defaults(self):
        """Test performance metrics with defaults."""
        metrics = PerformanceMetrics()
        
        assert metrics.inference_time == 0.0
        assert metrics.preprocessing_time == 0.0
        assert metrics.postprocessing_time == 0.0
        assert metrics.total_time == 0.0
        assert metrics.throughput == 0.0
        assert metrics.memory_usage == 0
        assert metrics.gpu_utilization is None


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor instance."""
        return PerformanceMonitor()
    
    def test_monitor_initialization(self, performance_monitor):
        """Test monitor initialization."""
        assert performance_monitor.request_times == deque(maxlen=1000)
        assert performance_monitor.total_requests == 0
        assert performance_monitor.start_time is not None
    
    def test_start_request_timing(self, performance_monitor):
        """Test starting request timing."""
        request_id = "test_request_1"
        performance_monitor.start_request(request_id)
        
        assert request_id in performance_monitor.active_requests
        assert performance_monitor.active_requests[request_id] > 0
    
    def test_end_request_timing(self, performance_monitor):
        """Test ending request timing."""
        request_id = "test_request_1"
        
        # Start request
        performance_monitor.start_request(request_id)
        time.sleep(0.01)  # Small delay
        
        # End request
        elapsed = performance_monitor.end_request(request_id)
        
        assert elapsed > 0
        assert request_id not in performance_monitor.active_requests
        assert len(performance_monitor.request_times) > 0
        assert performance_monitor.total_requests == 1
    
    def test_end_request_not_started(self, performance_monitor):
        """Test ending request that wasn't started."""
        elapsed = performance_monitor.end_request("nonexistent_request")
        assert elapsed == 0.0
    
    def test_get_current_stats(self, performance_monitor):
        """Test getting current statistics."""
        # Record some requests
        for i in range(5):
            request_id = f"request_{i}"
            performance_monitor.start_request(request_id)
            time.sleep(0.001)
            performance_monitor.end_request(request_id)
        
        stats = performance_monitor.get_current_stats()
        
        assert isinstance(stats, dict)
        assert "total_requests" in stats
        assert "avg_request_time" in stats
        assert "active_requests" in stats
        assert "uptime" in stats
        
        assert stats["total_requests"] == 5
        assert stats["avg_request_time"] > 0
        assert stats["uptime"] > 0
    
    def test_get_performance_summary(self, performance_monitor):
        """Test getting performance summary."""
        # Record requests with different timing patterns
        fast_times = [0.01, 0.015, 0.012, 0.018, 0.014]
        for i, duration in enumerate(fast_times):
            request_id = f"fast_request_{i}"
            performance_monitor.start_request(request_id)
            time.sleep(duration)
            performance_monitor.end_request(request_id)
        
        summary = performance_monitor.get_performance_summary()
        
        assert isinstance(summary, dict)
        assert "total_requests" in summary
        assert "window_seconds" in summary
        assert "timestamp" in summary
        assert "metrics" in summary
        
        assert summary["total_requests"] == 5
        assert summary["window_seconds"] > 0
        assert summary["timestamp"] > 0
    
    def test_record_batch_metrics(self, performance_monitor):
        """Test recording batch metrics."""
        metrics = PerformanceMetrics(
            inference_time=0.05,
            preprocessing_time=0.002,
            postprocessing_time=0.001,
            total_time=0.053,
            throughput=18.9,
            memory_usage=1024*1024*100,
            gpu_utilization=80.0
        )
        
        performance_monitor.record_batch_metrics(4, metrics)
        
        # Check that metrics were recorded
        assert len(performance_monitor.batch_metrics) > 0
        last_batch = performance_monitor.batch_metrics[-1]
        assert last_batch["batch_size"] == 4
        assert last_batch["metrics"] == metrics
    
    def test_get_batch_performance(self, performance_monitor):
        """Test getting batch performance statistics."""
        # Record multiple batch metrics
        for i in range(3):
            metrics = PerformanceMetrics(
                inference_time=0.02 + i * 0.01,
                total_time=0.025 + i * 0.01,
                throughput=40.0 - i * 5.0
            )
            performance_monitor.record_batch_metrics(2 + i, metrics)
        
        batch_perf = performance_monitor.get_batch_performance()
        
        assert isinstance(batch_perf, dict)
        assert "total_batches" in batch_perf
        assert "average_batch_size" in batch_perf
        assert "average_inference_time" in batch_perf
        assert "average_throughput" in batch_perf
        
        assert batch_perf["total_batches"] == 3
        assert batch_perf["average_batch_size"] > 2.0
    
    def test_reset_statistics(self, performance_monitor):
        """Test resetting statistics."""
        # Record some data
        performance_monitor.start_request("test")
        performance_monitor.end_request("test")
        
        assert performance_monitor.total_requests > 0
        
        # Reset
        performance_monitor.reset()
        
        assert performance_monitor.total_requests == 0
        assert len(performance_monitor.request_times) == 0
        assert len(performance_monitor.active_requests) == 0
        assert len(performance_monitor.batch_metrics) == 0
    
    def test_context_manager(self, performance_monitor):
        """Test using monitor as context manager."""
        with performance_monitor.time_request("context_test") as request_id:
            assert request_id == "context_test"
            assert "context_test" in performance_monitor.active_requests
            time.sleep(0.001)
        
        # Should automatically end timing
        assert "context_test" not in performance_monitor.active_requests
        assert performance_monitor.total_requests == 1


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector instance."""
        return MetricsCollector()
    
    def test_collector_initialization(self, metrics_collector):
        """Test collector initialization."""
        assert len(metrics_collector.get_summary()) == 0
        assert metrics_collector.max_history > 0
    
    def test_record_counter_metric(self, metrics_collector):
        """Test recording counter metrics."""
        metrics_collector.record_counter("requests_total", 1)
        metrics_collector.record_counter("requests_total", 1)
        metrics_collector.record_counter("requests_total", 3)
        
        # Check if the metric was recorded
        summary = metrics_collector.get_summary()
        assert "requests_total" in summary
        assert summary["requests_total"]["count"] == 3
    
    def test_record_gauge_metric(self, metrics_collector):
        """Test recording gauge metrics."""
        metrics_collector.record_gauge("memory_usage", 100)
        metrics_collector.record_gauge("memory_usage", 150)
        metrics_collector.record_gauge("memory_usage", 120)
        
        # Check if the metrics were recorded
        summary = metrics_collector.get_summary()
        assert "memory_usage" in summary
        assert summary["memory_usage"]["count"] == 3
        assert summary["memory_usage"]["latest"] == 120
    
    def test_record_histogram_metric(self, metrics_collector):
        """Test recording histogram metrics."""
        response_times = [0.01, 0.02, 0.015, 0.03, 0.012, 0.025]
        
        for rt in response_times:
            metrics_collector.record_timer("response_time", rt)
        
        # Check if the metrics were recorded
        summary = metrics_collector.get_summary()
        assert "response_time" in summary
        assert summary["response_time"]["count"] == len(response_times)
    
    def test_record_with_labels(self, metrics_collector):
        """Test recording metrics with labels."""
        # Record metrics with different labels
        metrics_collector.record(
            "model_requests", 5, 
            labels={"model": "bert", "version": "v1"}
        )
        metrics_collector.record(
            "model_requests", 3,
            labels={"model": "gpt", "version": "v2"}
        )
        
        # Should store separately by labels
        metrics = metrics_collector.get_all_metrics()
        model_requests = metrics["model_requests"]
        
        assert len(model_requests) == 2
        assert any(item["value"] == 5 for item in model_requests)
        assert any(item["value"] == 3 for item in model_requests)
    
    def test_get_metric_nonexistent(self, metrics_collector):
        """Test getting non-existent metric."""
        value = metrics_collector.get_metric("nonexistent_metric")
        assert value is None
    
    def test_get_all_metrics(self, metrics_collector):
        """Test getting all metrics."""
        # Record various metrics
        metrics_collector.record("counter_metric", 10, metric_type="counter")
        metrics_collector.record("gauge_metric", 50, metric_type="gauge")
        metrics_collector.record("histogram_metric", 0.1, metric_type="histogram")
        
        all_metrics = metrics_collector.get_all_metrics()
        
        assert isinstance(all_metrics, dict)
        assert "counter_metric" in all_metrics
        assert "gauge_metric" in all_metrics
        assert "histogram_metric" in all_metrics
    
    def test_calculate_percentiles(self, metrics_collector):
        """Test percentile calculations."""
        # Record histogram data
        values = list(range(1, 101))  # 1 to 100
        for value in values:
            metrics_collector.record("test_histogram", value, metric_type="histogram")
        
        percentiles = metrics_collector.calculate_percentiles("test_histogram", [50, 90, 95, 99])
        
        assert isinstance(percentiles, dict)
        assert 50 in percentiles
        assert 90 in percentiles
        assert 95 in percentiles
        assert 99 in percentiles
        
        # Check approximate correctness
        assert 45 <= percentiles[50] <= 55  # 50th percentile ~50
        assert 85 <= percentiles[90] <= 95  # 90th percentile ~90
    
    def test_calculate_percentiles_nonexistent(self, metrics_collector):
        """Test percentile calculation for non-existent metric."""
        percentiles = metrics_collector.calculate_percentiles("nonexistent", [50, 90])
        assert percentiles == {}
    
    def test_export_metrics(self, metrics_collector):
        """Test exporting metrics."""
        # Record various metrics
        metrics_collector.record("requests", 100, metric_type="counter")
        metrics_collector.record("cpu_usage", 75.5, metric_type="gauge")
        
        for i in range(10):
            metrics_collector.record("latency", 0.01 * (i + 1), metric_type="histogram")
        
        exported = metrics_collector.export_metrics()
        
        assert isinstance(exported, dict)
        assert "timestamp" in exported
        assert "metrics" in exported
        assert "collection_duration" in exported
        
        metrics = exported["metrics"]
        assert "requests" in metrics
        assert "cpu_usage" in metrics
        assert "latency" in metrics
    
    def test_reset_metrics(self, metrics_collector):
        """Test resetting all metrics."""
        # Record some metrics
        metrics_collector.record("test_counter", 5)
        metrics_collector.record("test_gauge", 10)
        
        assert len(metrics_collector.metrics) > 0
        
        # Reset
        metrics_collector.reset()
        
        assert len(metrics_collector.metrics) == 0


class TestMonitoringIntegration:
    """Integration tests for monitoring components."""
    
    def test_performance_monitor_metrics_collector_integration(self):
        """Test integration between performance monitor and metrics collector."""
        monitor = PerformanceMonitor()
        collector = MetricsCollector()
        
        # Record some performance data
        for i in range(10):
            request_id = f"request_{i}"
            monitor.start_request(request_id)
            time.sleep(0.001)
            elapsed = monitor.end_request(request_id)
            
            # Record in metrics collector
            collector.record("request_time", elapsed, metric_type="histogram")
            collector.record("requests_total", 1, metric_type="counter")
        
        # Get performance stats
        perf_stats = monitor.get_current_stats()
        
        # Get metrics
        total_requests = collector.get_metric("requests_total")
        request_times = collector.get_metric("request_time")
        
        assert perf_stats["total_requests"] == 10
        assert total_requests == 10
        assert len(request_times) == 10
    
    def test_monitoring_with_context_managers(self):
        """Test monitoring with context managers."""
        monitor = PerformanceMonitor()
        collector = MetricsCollector()
        
        # Use context managers for timing
        for i in range(5):
            with monitor.time_request(f"request_{i}"):
                time.sleep(0.001)
                # Simulate work
                collector.record("work_done", 1, metric_type="counter")
        
        stats = monitor.get_current_stats()
        work_done = collector.get_metric("work_done")
        
        assert stats["total_requests"] == 5
        assert work_done == 5


class TestGlobalMonitoringInstances:
    """Test global monitoring instances."""
    
    def test_get_performance_monitor_singleton(self):
        """Test global performance monitor singleton."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        # Should be the same instance
        assert monitor1 is monitor2
        assert isinstance(monitor1, PerformanceMonitor)
    
    def test_get_metrics_collector_singleton(self):
        """Test global metrics collector singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        # Should be the same instance
        assert collector1 is collector2
        assert isinstance(collector1, MetricsCollector)
    
    def test_global_instances_independence(self):
        """Test that global instances are independent."""
        monitor = get_performance_monitor()
        collector = get_metrics_collector()
        
        # They should be different objects
        assert monitor is not collector
        
        # Each should function independently
        monitor.start_request("test")
        collector.record("test_metric", 1)
        
        # Both should have recorded data
        assert len(monitor.active_requests) == 1
        assert len(collector.metrics) > 0


class TestMonitoringErrorHandling:
    """Test error handling in monitoring components."""
    
    def test_performance_monitor_error_handling(self):
        """Test performance monitor error handling."""
        monitor = PerformanceMonitor()
        
        # Test with None request ID
        monitor.start_request(None)  # Should handle gracefully
        elapsed = monitor.end_request(None)
        assert elapsed == 0.0
        
        # Test ending request multiple times
        monitor.start_request("test")
        monitor.end_request("test")
        elapsed = monitor.end_request("test")  # Already ended
        assert elapsed == 0.0
    
    def test_metrics_collector_error_handling(self):
        """Test metrics collector error handling."""
        collector = MetricsCollector()
        
        # Test with None values
        collector.record(None, 10)  # Should handle gracefully
        collector.record("test", None)  # Should handle gracefully
        
        # Test with invalid percentiles
        percentiles = collector.calculate_percentiles("nonexistent", [150, -10])
        assert percentiles == {}
    
    def test_monitoring_with_exceptions(self):
        """Test monitoring when exceptions occur."""
        monitor = PerformanceMonitor()
        collector = MetricsCollector()
        
        # Simulate work that raises exception
        try:
            with monitor.time_request("failing_request"):
                collector.record("attempts", 1, metric_type="counter")
                raise ValueError("Simulated error")
        except ValueError:
            collector.record("errors", 1, metric_type="counter")
        
        # Monitoring should still work
        stats = monitor.get_current_stats()
        attempts = collector.get_metric("attempts")
        errors = collector.get_metric("errors")
        
        assert stats["total_requests"] == 1
        assert attempts == 1
        assert errors == 1

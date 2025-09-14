"""
Tests for Metrics implementation.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from collections import defaultdict

from framework.observability.metrics import (
    Counter, Gauge, Histogram, MetricsCollector, 
    SLATracker, ResourceUtilizationTracker
)


class TestCounter:
    """Test counter metric functionality."""
    
    def test_counter_initialization(self):
        """Test counter initialization."""
        counter = Counter("test_counter", "Test counter description")
        
        assert counter.name == "test_counter"
        assert counter.description == "Test counter description"
        assert counter.value == 0
        assert counter.labels == {}
    
    def test_counter_with_labels(self):
        """Test counter with labels."""
        labels = {"endpoint": "/api/predict", "method": "POST"}
        counter = Counter("requests_total", "Total requests", labels=labels)
        
        assert counter.labels == labels
        assert counter.value == 0
    
    def test_counter_increment(self):
        """Test counter increment operation."""
        counter = Counter("test_counter")
        
        counter.increment()
        assert counter.value == 1
        
        counter.increment(5)
        assert counter.value == 6
    
    def test_counter_reset(self):
        """Test counter reset operation."""
        counter = Counter("test_counter")
        counter.increment(10)
        
        counter.reset()
        assert counter.value == 0
    
    def test_counter_thread_safety(self):
        """Test counter thread safety."""
        counter = Counter("thread_counter")
        
        def increment_worker():
            for _ in range(1000):
                counter.increment()
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=increment_worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert counter.value == 5000
    
    def test_counter_to_dict(self):
        """Test counter serialization to dict."""
        labels = {"service": "inference", "version": "1.0"}
        counter = Counter("test_counter", "Test description", labels=labels)
        counter.increment(42)
        
        counter_dict = counter.to_dict()
        
        assert counter_dict["name"] == "test_counter"
        assert counter_dict["description"] == "Test description"
        assert counter_dict["type"] == "counter"
        assert counter_dict["value"] == 42
        assert counter_dict["labels"] == labels
        assert "timestamp" in counter_dict


class TestGauge:
    """Test gauge metric functionality."""
    
    def test_gauge_initialization(self):
        """Test gauge initialization."""
        gauge = Gauge("test_gauge", "Test gauge description")
        
        assert gauge.name == "test_gauge"
        assert gauge.description == "Test gauge description"
        assert gauge.value == 0
    
    def test_gauge_set_value(self):
        """Test gauge set operation."""
        gauge = Gauge("cpu_usage")
        
        gauge.set(75.5)
        assert gauge.value == 75.5
        
        gauge.set(0)
        assert gauge.value == 0
    
    def test_gauge_increment_decrement(self):
        """Test gauge increment and decrement operations."""
        gauge = Gauge("active_connections")
        
        gauge.increment()
        assert gauge.value == 1
        
        gauge.increment(5)
        assert gauge.value == 6
        
        gauge.decrement(2)
        assert gauge.value == 4
        
        gauge.decrement()
        assert gauge.value == 3
    
    def test_gauge_negative_values(self):
        """Test gauge with negative values."""
        gauge = Gauge("temperature_delta")
        
        gauge.set(-10.5)
        assert gauge.value == -10.5
        
        gauge.increment(5)
        assert gauge.value == -5.5
    
    def test_gauge_thread_safety(self):
        """Test gauge thread safety."""
        gauge = Gauge("thread_gauge")
        
        def modify_gauge(increment_value):
            for _ in range(100):
                gauge.increment(increment_value)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=modify_gauge, args=(i + 1,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should be (1+2+3+4+5) * 100 = 1500
        assert gauge.value == 1500


class TestHistogram:
    """Test histogram metric functionality."""
    
    def test_histogram_initialization(self):
        """Test histogram initialization."""
        buckets = [0.1, 0.5, 1.0, 2.0, 5.0]
        histogram = Histogram("request_duration", "Request duration", buckets=buckets)
        
        assert histogram.name == "request_duration"
        assert histogram.buckets == buckets
        assert histogram.count == 0
        assert histogram.sum == 0
        assert len(histogram.bucket_counts) == len(buckets) + 1  # +1 for +Inf
    
    def test_histogram_observe(self):
        """Test histogram observe operation."""
        buckets = [1.0, 2.0, 5.0]
        histogram = Histogram("test_histogram", buckets=buckets)
        
        # Observe values
        histogram.observe(0.5)  # Falls in bucket <= 1.0
        histogram.observe(1.5)  # Falls in bucket <= 2.0
        histogram.observe(3.0)  # Falls in bucket <= 5.0
        histogram.observe(10.0) # Falls in bucket +Inf
        
        assert histogram.count == 4
        assert histogram.sum == 15.0  # 0.5 + 1.5 + 3.0 + 10.0
        
        # Check bucket counts
        assert histogram.bucket_counts[1.0] == 1  # 0.5
        assert histogram.bucket_counts[2.0] == 2  # 0.5, 1.5
        assert histogram.bucket_counts[5.0] == 3  # 0.5, 1.5, 3.0
        assert histogram.bucket_counts[float('inf')] == 4  # All values
    
    def test_histogram_percentiles(self):
        """Test histogram percentile calculations."""
        histogram = Histogram("test_percentiles", buckets=[1, 2, 5, 10])
        
        # Add values with known distribution
        values = [0.1, 0.5, 1.2, 1.8, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0]
        for value in values:
            histogram.observe(value)
        
        p50 = histogram.get_percentile(50)
        p95 = histogram.get_percentile(95)
        p99 = histogram.get_percentile(99)
        
        # Basic sanity checks (exact values depend on bucket interpolation)
        assert 0 < p50 < 10
        assert p95 > p50
        assert p99 >= p95
    
    def test_histogram_average(self):
        """Test histogram average calculation."""
        histogram = Histogram("test_average")
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            histogram.observe(value)
        
        avg = histogram.get_average()
        assert avg == 3.0  # (1+2+3+4+5) / 5
    
    def test_histogram_empty(self):
        """Test histogram with no observations."""
        histogram = Histogram("empty_histogram")
        
        assert histogram.get_average() == 0
        assert histogram.get_percentile(50) == 0
        assert histogram.get_percentile(95) == 0
    
    def test_histogram_reset(self):
        """Test histogram reset operation."""
        histogram = Histogram("test_reset", buckets=[1.0, 5.0])
        
        histogram.observe(2.0)
        histogram.observe(3.0)
        
        assert histogram.count == 2
        assert histogram.sum == 5.0
        
        histogram.reset()
        
        assert histogram.count == 0
        assert histogram.sum == 0
        assert all(count == 0 for count in histogram.bucket_counts.values())


class TestMetricsCollector:
    """Test metrics collector functionality."""
    
    @pytest.fixture
    def collector(self):
        """Create metrics collector."""
        return MetricsCollector()
    
    def test_register_counter(self, collector):
        """Test registering counter metric."""
        counter = collector.counter("requests_total", "Total requests")
        
        assert counter.name == "requests_total"
        assert "requests_total" in collector._metrics
        assert collector._metrics["requests_total"] is counter
    
    def test_register_gauge(self, collector):
        """Test registering gauge metric."""
        gauge = collector.gauge("cpu_usage", "CPU usage percentage")
        
        assert gauge.name == "cpu_usage"
        assert "cpu_usage" in collector._metrics
    
    def test_register_histogram(self, collector):
        """Test registering histogram metric."""
        buckets = [0.1, 0.5, 1.0, 2.0]
        histogram = collector.histogram("request_duration", "Request duration", buckets=buckets)
        
        assert histogram.name == "request_duration"
        assert histogram.buckets == buckets
        assert "request_duration" in collector._metrics
    
    def test_register_duplicate_metric(self, collector):
        """Test registering duplicate metric name."""
        collector.counter("duplicate_name", "First metric")
        
        with pytest.raises(ValueError, match="already registered"):
            collector.counter("duplicate_name", "Second metric")
    
    def test_get_metric(self, collector):
        """Test retrieving metric by name."""
        original_counter = collector.counter("test_counter")
        retrieved_counter = collector.get_metric("test_counter")
        
        assert retrieved_counter is original_counter
    
    def test_get_nonexistent_metric(self, collector):
        """Test retrieving non-existent metric."""
        metric = collector.get_metric("nonexistent")
        assert metric is None
    
    def test_get_all_metrics(self, collector):
        """Test retrieving all metrics."""
        counter = collector.counter("test_counter")
        gauge = collector.gauge("test_gauge")
        histogram = collector.histogram("test_histogram")
        
        all_metrics = collector.get_all_metrics()
        
        assert len(all_metrics) == 3
        assert "test_counter" in all_metrics
        assert "test_gauge" in all_metrics
        assert "test_histogram" in all_metrics
    
    def test_collect_metrics_data(self, collector):
        """Test collecting metrics data."""
        counter = collector.counter("requests_total")
        gauge = collector.gauge("cpu_usage")
        histogram = collector.histogram("latency", buckets=[0.1, 0.5, 1.0])
        
        # Update metrics
        counter.increment(10)
        gauge.set(75.5)
        histogram.observe(0.3)
        histogram.observe(0.8)
        
        metrics_data = collector.collect_metrics()
        
        assert len(metrics_data) == 3
        
        # Find each metric in the collected data
        counter_data = next(m for m in metrics_data if m["name"] == "requests_total")
        gauge_data = next(m for m in metrics_data if m["name"] == "cpu_usage")
        histogram_data = next(m for m in metrics_data if m["name"] == "latency")
        
        assert counter_data["value"] == 10
        assert gauge_data["value"] == 75.5
        assert histogram_data["count"] == 2


class TestSLATracker:
    """Test SLA tracker functionality."""
    
    def test_sla_tracker_initialization(self):
        """Test SLA tracker initialization."""
        tracker = SLATracker(
            sla_threshold=2.0,
            slo_targets={"p95": 1.0, "p99": 1.5}
        )
        
        assert tracker.sla_threshold == 2.0
        assert tracker.slo_targets == {"p95": 1.0, "p99": 1.5}
        assert tracker.total_requests == 0
        assert tracker.sla_violations == 0
    
    def test_sla_tracker_record_request(self):
        """Test recording requests in SLA tracker."""
        tracker = SLATracker(sla_threshold=1.0)
        
        # Record requests within SLA
        tracker.record_request(0.5, success=True)
        tracker.record_request(0.8, success=True)
        
        # Record requests violating SLA
        tracker.record_request(1.5, success=True)  # Slow but successful
        tracker.record_request(0.3, success=False)  # Fast but failed
        
        assert tracker.total_requests == 4
        assert tracker.successful_requests == 3
        assert tracker.sla_violations == 1  # Only the 1.5s request
    
    def test_sla_tracker_availability(self):
        """Test SLA availability calculation."""
        tracker = SLATracker()
        
        # Record mix of successful and failed requests
        for _ in range(95):
            tracker.record_request(0.5, success=True)
        
        for _ in range(5):
            tracker.record_request(0.5, success=False)
        
        availability = tracker.get_availability()
        assert availability == 0.95  # 95/100
    
    def test_sla_tracker_compliance_rate(self):
        """Test SLA compliance rate calculation."""
        tracker = SLATracker(sla_threshold=1.0)
        
        # 8 compliant requests, 2 non-compliant
        for _ in range(8):
            tracker.record_request(0.5, success=True)
        
        for _ in range(2):
            tracker.record_request(1.5, success=True)
        
        compliance_rate = tracker.get_sla_compliance_rate()
        assert compliance_rate == 0.8  # 8/10
    
    def test_sla_tracker_slo_status(self):
        """Test SLO status checking."""
        tracker = SLATracker(slo_targets={"p95": 1.0, "p99": 2.0})
        
        # Add requests with known distribution
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.5]
        for value in values:
            tracker.record_request(value, success=True)
        
        slo_status = tracker.get_slo_status()
        
        assert "p95" in slo_status
        assert "p99" in slo_status
        assert "met" in slo_status["p95"]
        assert "met" in slo_status["p99"]
    
    def test_sla_tracker_reset(self):
        """Test SLA tracker reset."""
        tracker = SLATracker()
        
        tracker.record_request(0.5, success=True)
        tracker.record_request(1.5, success=False)
        
        assert tracker.total_requests == 2
        
        tracker.reset()
        
        assert tracker.total_requests == 0
        assert tracker.successful_requests == 0
        assert tracker.sla_violations == 0


class TestResourceUtilizationTracker:
    """Test resource utilization tracker functionality."""
    
    @pytest.fixture
    def resource_tracker(self):
        """Create resource utilization tracker."""
        return ResourceUtilizationTracker()
    
    def test_resource_tracker_cpu_update(self, resource_tracker):
        """Test CPU utilization tracking."""
        with patch('psutil.cpu_percent', return_value=75.5):
            resource_tracker.update_cpu_usage()
            
            cpu_gauge = resource_tracker._collector.get_metric("cpu_usage_percent")
            assert cpu_gauge.value == 75.5
    
    def test_resource_tracker_memory_update(self, resource_tracker):
        """Test memory utilization tracking."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 60.0
            mock_memory.return_value.used = 8 * 1024 * 1024 * 1024  # 8GB
            mock_memory.return_value.total = 16 * 1024 * 1024 * 1024  # 16GB
            
            resource_tracker.update_memory_usage()
            
            memory_gauge = resource_tracker._collector.get_metric("memory_usage_percent")
            memory_used_gauge = resource_tracker._collector.get_metric("memory_used_bytes")
            
            assert memory_gauge.value == 60.0
            assert memory_used_gauge.value == 8 * 1024 * 1024 * 1024
    
    def test_resource_tracker_gpu_update(self, resource_tracker):
        """Test GPU utilization tracking."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2), \
             patch('torch.cuda.memory_stats') as mock_memory_stats, \
             patch('torch.cuda.utilization', return_value=85):
            
            # Mock memory stats for GPU 0
            mock_memory_stats.return_value = {
                'allocated_bytes.all.current': 2 * 1024 * 1024 * 1024,  # 2GB
                'reserved_bytes.all.current': 4 * 1024 * 1024 * 1024    # 4GB
            }
            
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
                
                resource_tracker.update_gpu_usage()
                
                gpu_util_gauge = resource_tracker._collector.get_metric("gpu_utilization_percent")
                gpu_memory_gauge = resource_tracker._collector.get_metric("gpu_memory_usage_percent")
                
                assert gpu_util_gauge.value == 85
                assert gpu_memory_gauge.value == 50.0  # 4GB / 8GB = 50%
    
    def test_resource_tracker_no_gpu(self, resource_tracker):
        """Test resource tracker when no GPU is available."""
        with patch('torch.cuda.is_available', return_value=False):
            resource_tracker.update_gpu_usage()
            
            # GPU metrics should not be created
            gpu_util_gauge = resource_tracker._collector.get_metric("gpu_utilization_percent")
            assert gpu_util_gauge is None
    
    def test_resource_tracker_update_all(self, resource_tracker):
        """Test updating all resource metrics at once."""
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('torch.cuda.is_available', return_value=False):
            
            mock_memory.return_value.percent = 40.0
            mock_memory.return_value.used = 4 * 1024 * 1024 * 1024
            
            resource_tracker.update_all_metrics()
            
            cpu_gauge = resource_tracker._collector.get_metric("cpu_usage_percent")
            memory_gauge = resource_tracker._collector.get_metric("memory_usage_percent")
            
            assert cpu_gauge.value == 50.0
            assert memory_gauge.value == 40.0
    
    def test_resource_tracker_get_current_stats(self, resource_tracker):
        """Test getting current resource statistics."""
        with patch('psutil.cpu_percent', return_value=30.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('torch.cuda.is_available', return_value=False):
            
            mock_memory.return_value.percent = 25.0
            mock_memory.return_value.used = 2 * 1024 * 1024 * 1024
            
            resource_tracker.update_all_metrics()
            stats = resource_tracker.get_current_stats()
            
            assert "cpu_usage_percent" in stats
            assert "memory_usage_percent" in stats
            assert stats["cpu_usage_percent"] == 30.0
            assert stats["memory_usage_percent"] == 25.0


class TestMetricsIntegration:
    """Test metrics integration scenarios."""
    
    def test_metrics_workflow_simulation(self):
        """Test complete metrics workflow simulation."""
        collector = MetricsCollector()
        sla_tracker = SLATracker(sla_threshold=1.0)
        
        # Create metrics
        request_counter = collector.counter("http_requests_total", "Total HTTP requests")
        response_time_histogram = collector.histogram(
            "http_request_duration_seconds", 
            "HTTP request duration",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        error_counter = collector.counter("http_errors_total", "Total HTTP errors")
        
        # Simulate request processing
        request_times = [0.05, 0.1, 0.15, 0.3, 0.8, 1.2, 0.2, 0.4, 0.6, 2.1]
        error_requests = [5, 9]  # Indices of error requests
        
        for i, response_time in enumerate(request_times):
            request_counter.increment()
            response_time_histogram.observe(response_time)
            
            success = i not in error_requests
            if not success:
                error_counter.increment()
            
            sla_tracker.record_request(response_time, success=success)
        
        # Verify metrics
        assert request_counter.value == 10
        assert error_counter.value == 2
        assert response_time_histogram.count == 10
        assert sla_tracker.total_requests == 10
        assert sla_tracker.successful_requests == 8
        assert sla_tracker.sla_violations == 2  # 1.2s and 2.1s requests
        
        # Check SLA compliance
        availability = sla_tracker.get_availability()
        compliance_rate = sla_tracker.get_sla_compliance_rate()
        
        assert availability == 0.8  # 8/10 successful
        assert compliance_rate == 0.8  # 8/10 within SLA
    
    def test_concurrent_metrics_collection(self):
        """Test concurrent metrics collection."""
        collector = MetricsCollector()
        counter = collector.counter("concurrent_counter")
        gauge = collector.gauge("concurrent_gauge")
        histogram = collector.histogram("concurrent_histogram")
        
        def worker_thread(thread_id):
            for i in range(100):
                counter.increment()
                gauge.set(thread_id * 100 + i)
                histogram.observe(thread_id + i * 0.01)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify final values
        assert counter.value == 500  # 5 threads * 100 increments
        assert histogram.count == 500
        
        # Gauge should have the last set value (non-deterministic which thread finishes last)
        assert gauge.value >= 0
    
    def test_metrics_export_format(self):
        """Test metrics export in standard format."""
        collector = MetricsCollector()
        
        counter = collector.counter(
            "test_counter", 
            "Test counter metric", 
            labels={"service": "inference"}
        )
        gauge = collector.gauge("test_gauge", "Test gauge metric")
        histogram = collector.histogram("test_histogram", "Test histogram metric")
        
        counter.increment(42)
        gauge.set(75.5)
        histogram.observe(0.5)
        histogram.observe(1.5)
        
        metrics_data = collector.collect_metrics()
        
        # Verify exportable format
        for metric_data in metrics_data:
            assert "name" in metric_data
            assert "type" in metric_data
            assert "value" in metric_data or "count" in metric_data
            assert "timestamp" in metric_data
            
            if metric_data["type"] == "counter":
                assert metric_data["value"] == 42
                assert "labels" in metric_data
                assert metric_data["labels"]["service"] == "inference"
            
            elif metric_data["type"] == "gauge":
                assert metric_data["value"] == 75.5
            
            elif metric_data["type"] == "histogram":
                assert metric_data["count"] == 2
                assert "buckets" in metric_data
                assert "percentiles" in metric_data

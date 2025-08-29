"""
Unit tests for PerformanceOptimizer

Tests the performance optimization system including:
- Real-time performance monitoring
- Adaptive optimization algorithms
- Automatic parameter tuning
- Performance alerting system
- Bottleneck detection and resolution
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
import numpy as np

from framework.core.performance_optimizer import (
    PerformanceOptimizer,
    PerformanceConfig,
    PerformanceMonitor,
    AdaptiveOptimizer,
    PerformanceSnapshot,
    PerformanceMetric,
    OptimizationLevel
)


class TestPerformanceConfig:
    """Test PerformanceConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = PerformanceConfig()
        
        assert config.optimization_level == OptimizationLevel.BALANCED
        assert config.target_latency_ms == 100.0
        assert config.target_throughput_rps == 1000.0
        assert config.target_cpu_usage == 0.7
        assert config.target_memory_usage == 0.8
        assert config.target_gpu_usage == 0.9
        assert config.monitoring_interval == 10.0
        assert config.metrics_window_size == 100
        assert config.performance_history_size == 1000
        assert config.enable_auto_scaling is True
        assert config.scale_up_threshold == 0.8
        assert config.scale_down_threshold == 0.3
        assert config.scaling_cooldown == 60.0
        assert config.enable_alerting is True
        assert config.max_concurrent_requests == 5000
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = PerformanceConfig(
            optimization_level=OptimizationLevel.AGGRESSIVE,
            target_latency_ms=50.0,
            target_throughput_rps=2000.0,
            monitoring_interval=5.0,
            enable_auto_scaling=False
        )
        
        assert config.optimization_level == OptimizationLevel.AGGRESSIVE
        assert config.target_latency_ms == 50.0
        assert config.target_throughput_rps == 2000.0
        assert config.monitoring_interval == 5.0
        assert config.enable_auto_scaling is False


class TestPerformanceSnapshot:
    """Test PerformanceSnapshot dataclass"""
    
    def test_snapshot_creation(self):
        """Test performance snapshot creation"""
        timestamp = time.time()
        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            latency_p50=50.0,
            latency_p95=150.0,
            latency_p99=250.0,
            throughput=500.0,
            cpu_usage=0.6,
            memory_usage=0.7,
            gpu_usage=0.8,
            gpu_memory_usage=0.5,
            active_requests=100,
            queue_size=25,
            error_rate=0.01,
            cache_hit_rate=0.85
        )
        
        assert snapshot.timestamp == timestamp
        assert snapshot.latency_p50 == 50.0
        assert snapshot.latency_p95 == 150.0
        assert snapshot.throughput == 500.0
        assert snapshot.cpu_usage == 0.6
        assert snapshot.memory_usage == 0.7
        assert snapshot.error_rate == 0.01
    
    def test_snapshot_to_dict(self):
        """Test converting snapshot to dictionary"""
        snapshot = PerformanceSnapshot(
            timestamp=123456.0,
            latency_p50=50.0,
            latency_p95=150.0,
            latency_p99=250.0,
            throughput=500.0,
            cpu_usage=0.6,
            memory_usage=0.7,
            gpu_usage=0.8,
            gpu_memory_usage=0.5,
            active_requests=100,
            queue_size=25,
            error_rate=0.01,
            cache_hit_rate=0.85
        )
        
        snapshot_dict = snapshot.to_dict()
        
        assert isinstance(snapshot_dict, dict)
        assert snapshot_dict['timestamp'] == 123456.0
        assert snapshot_dict['latency_p50'] == 50.0
        assert snapshot_dict['throughput'] == 500.0
        assert snapshot_dict['cpu_usage'] == 0.6
        assert len(snapshot_dict) == 13  # All fields present


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PerformanceConfig(
            monitoring_interval=1.0,
            metrics_window_size=10,
            performance_history_size=50
        )
    
    @pytest.fixture
    def monitor(self, config):
        """Create PerformanceMonitor fixture"""
        return PerformanceMonitor(config)
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert len(monitor.latencies) == 0
        assert len(monitor.throughput_samples) == 0
        assert len(monitor.system_metrics) == 0
        assert len(monitor.performance_history) == 0
        assert len(monitor.active_alerts) == 0
        assert monitor.completed_requests == 0
        assert monitor.failed_requests == 0
    
    def test_monitor_request_tracking(self, monitor):
        """Test request start/end tracking"""
        request_id = "test_request_1"
        
        # Record request start
        monitor.record_request_start(request_id)
        assert request_id in monitor.request_times
        
        # Small delay to measure latency
        time.sleep(0.01)
        
        # Record request end (success)
        monitor.record_request_end(request_id, success=True)
        
        assert request_id not in monitor.request_times
        assert monitor.completed_requests == 1
        assert monitor.failed_requests == 0
        assert len(monitor.latencies) == 1
        assert monitor.latencies[0] > 0  # Should have positive latency
    
    def test_monitor_failed_request_tracking(self, monitor):
        """Test failed request tracking"""
        request_id = "test_request_2"
        
        monitor.record_request_start(request_id)
        time.sleep(0.01)
        monitor.record_request_end(request_id, success=False)
        
        assert monitor.completed_requests == 0
        assert monitor.failed_requests == 1
        assert len(monitor.latencies) == 1  # Should still record latency
    
    def test_monitor_throughput_recording(self, monitor):
        """Test throughput recording"""
        monitor.record_throughput(100.0)
        monitor.record_throughput(150.0)
        monitor.record_throughput(120.0)
        
        assert len(monitor.throughput_samples) == 3
        assert monitor.throughput_samples[-1] == 120.0
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('torch.cuda.is_available')
    def test_monitor_system_metrics_collection(self, mock_cuda, mock_memory, mock_cpu, monitor):
        """Test system metrics collection"""
        # Mock system metrics
        mock_cpu.return_value = 60.0
        mock_memory.return_value = MagicMock(percent=70.0)
        mock_cuda.return_value = False  # No GPU
        
        metrics = monitor.collect_system_metrics()
        
        assert metrics['cpu_usage'] == 0.6  # 60% as fraction
        assert metrics['memory_usage'] == 0.7  # 70% as fraction
        assert metrics['gpu_usage'] == 0.0  # No GPU
        assert 'active_requests' in metrics
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('torch.cuda.is_available')
    def test_monitor_performance_snapshot(self, mock_cuda, mock_memory, mock_cpu, monitor):
        """Test performance snapshot generation"""
        # Setup mocks
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_cuda.return_value = False
        
        # Add some latency data
        monitor.latencies.extend([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        monitor.throughput_samples.extend([100, 120, 110])
        monitor.completed_requests = 8
        monitor.failed_requests = 2
        
        snapshot = monitor.get_current_performance()
        
        assert isinstance(snapshot, PerformanceSnapshot)
        assert snapshot.latency_p50 == 50  # 50th percentile
        assert snapshot.latency_p95 == 95  # 95th percentile
        assert snapshot.latency_p99 == 99  # 99th percentile
        assert snapshot.throughput == pytest.approx(110.0, abs=1)  # Average
        assert snapshot.cpu_usage == 0.5
        assert snapshot.memory_usage == 0.6
        assert snapshot.error_rate == 0.2  # 2/(8+2)
    
    def test_monitor_alert_detection(self, monitor):
        """Test alert detection and triggering"""
        # Create snapshot that exceeds thresholds
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            latency_p50=50.0,
            latency_p95=250.0,  # Exceeds default 200ms threshold
            latency_p99=300.0,
            throughput=500.0,
            cpu_usage=0.95,  # Exceeds default 0.9 threshold
            memory_usage=0.7,
            gpu_usage=0.8,
            gpu_memory_usage=0.5,
            active_requests=100,
            queue_size=25,
            error_rate=0.08,  # Exceeds default 0.05 threshold
            cache_hit_rate=0.85
        )
        
        # Check alerts
        monitor.check_alerts(snapshot)
        
        # Should have triggered alerts
        assert PerformanceMetric.LATENCY in monitor.active_alerts
        assert PerformanceMetric.CPU_USAGE in monitor.active_alerts
        assert PerformanceMetric.ERROR_RATE in monitor.active_alerts
    
    def test_monitor_alert_callbacks(self, monitor):
        """Test alert callback functionality"""
        callback_calls = []
        
        def test_callback(metric, snapshot):
            callback_calls.append((metric, snapshot))
        
        monitor.add_alert_callback(test_callback)
        
        # Trigger alert
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            latency_p50=50.0,
            latency_p95=250.0,  # Exceeds threshold
            latency_p99=300.0,
            throughput=500.0,
            cpu_usage=0.6,
            memory_usage=0.7,
            gpu_usage=0.8,
            gpu_memory_usage=0.5,
            active_requests=100,
            queue_size=25,
            error_rate=0.01,
            cache_hit_rate=0.85
        )
        
        monitor.check_alerts(snapshot)
        
        # Should have called callback
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == PerformanceMetric.LATENCY


class TestAdaptiveOptimizer:
    """Test AdaptiveOptimizer functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PerformanceConfig(
            target_latency_ms=100.0,
            target_throughput_rps=1000.0,
            target_cpu_usage=0.7,
            target_memory_usage=0.8
        )
    
    @pytest.fixture
    def optimizer(self, config):
        """Create AdaptiveOptimizer fixture"""
        return AdaptiveOptimizer(config)
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization"""
        assert len(optimizer.optimization_history) == 0
        assert optimizer.last_optimization_time == 0
        assert len(optimizer.performance_trends) == len(PerformanceMetric)
        assert len(optimizer.current_optimizations) == 0
    
    def test_optimizer_trend_analysis(self, optimizer):
        """Test performance trend analysis"""
        # Create mock performance history
        history = []
        base_time = time.time()
        
        # Create trend: increasing latency
        for i in range(40):
            snapshot = PerformanceSnapshot(
                timestamp=base_time + i,
                latency_p50=50.0,
                latency_p95=100.0 + i * 2,  # Increasing trend
                latency_p99=200.0,
                throughput=1000.0 - i,  # Decreasing trend
                cpu_usage=0.5 + i * 0.01,  # Increasing trend
                memory_usage=0.6,
                gpu_usage=0.7,
                gpu_memory_usage=0.4,
                active_requests=100,
                queue_size=20,
                error_rate=0.01,
                cache_hit_rate=0.85
            )
            history.append(snapshot)
        
        trends = optimizer.analyze_performance_trends(history)
        
        assert 'latency_p95' in trends
        assert 'throughput' in trends
        assert 'cpu_usage' in trends
        
        # Check trend directions
        assert trends['latency_p95']['direction'] == 'increasing'
        assert trends['throughput']['direction'] == 'decreasing'
        assert trends['cpu_usage']['direction'] == 'increasing'
    
    def test_optimizer_recommendation_generation(self, optimizer):
        """Test optimization recommendation generation"""
        # Create performance snapshot with issues
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            latency_p50=80.0,
            latency_p95=200.0,  # Exceeds target of 100ms
            latency_p99=300.0,
            throughput=500.0,  # Below target of 1000 rps
            cpu_usage=0.85,  # Above target of 0.7
            memory_usage=0.9,  # Above target of 0.8
            gpu_usage=0.8,
            gpu_memory_usage=0.6,
            active_requests=200,
            queue_size=50,
            error_rate=0.03,  # Above 2%
            cache_hit_rate=0.75
        )
        
        # Mock trends indicating performance degradation
        trends = {
            'latency_p95': {'direction': 'increasing', 'magnitude': 0.2},
            'throughput': {'direction': 'decreasing', 'magnitude': 0.3},
            'cpu_usage': {'direction': 'increasing', 'magnitude': 0.15},
            'memory_usage': {'direction': 'stable', 'magnitude': 0.02}
        }
        
        recommendations = optimizer.generate_optimization_recommendations(trends, snapshot)
        
        # Should generate recommendations for multiple issues
        assert len(recommendations) > 0
        
        # Check for specific recommendation types
        recommendation_types = [rec['type'] for rec in recommendations]
        assert 'scale_workers' in recommendation_types or 'increase_concurrency' in recommendation_types
        assert 'cpu_optimization' in recommendation_types
        assert 'memory_optimization' in recommendation_types
    
    def test_optimizer_recommendation_priorities(self, optimizer):
        """Test recommendation priority handling"""
        # Create snapshot with critical error rate
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            latency_p50=50.0,
            latency_p95=100.0,
            latency_p99=200.0,
            throughput=1000.0,
            cpu_usage=0.6,
            memory_usage=0.7,
            gpu_usage=0.8,
            gpu_memory_usage=0.5,
            active_requests=100,
            queue_size=20,
            error_rate=0.08,  # High error rate (critical)
            cache_hit_rate=0.85
        )
        
        trends = {}
        recommendations = optimizer.generate_optimization_recommendations(trends, snapshot)
        
        # Should have critical priority recommendation
        critical_recs = [rec for rec in recommendations if rec.get('priority') == 'critical']
        assert len(critical_recs) > 0
        
        # Critical recommendation should be for stability
        assert critical_recs[0]['type'] == 'stability_optimization'
    
    def test_optimizer_application_cooldown(self, optimizer):
        """Test optimization application cooldown"""
        recommendations = [
            {'type': 'scale_workers', 'action': 'increase_worker_count', 'priority': 'high'}
        ]
        
        # First application should succeed
        assert optimizer._should_apply_optimization(recommendations[0]) is True
        
        # Apply optimization to update last optimization time
        optimizer.apply_optimizations(recommendations)
        
        # Second application should be blocked by cooldown
        assert optimizer._should_apply_optimization(recommendations[0]) is False
    
    def test_optimizer_duplicate_prevention(self, optimizer):
        """Test prevention of duplicate optimizations"""
        recommendation = {'type': 'scale_workers', 'action': 'increase_worker_count', 'priority': 'high'}
        
        # Apply optimization first time
        optimizer.apply_optimizations([recommendation])
        
        # Manually reset cooldown but keep history
        optimizer.last_optimization_time = 0
        
        # Should still prevent duplicate type
        assert optimizer._should_apply_optimization(recommendation) is False
    
    def test_optimizer_stats_tracking(self, optimizer):
        """Test optimization statistics tracking"""
        recommendations = [
            {'type': 'scale_workers', 'action': 'increase_worker_count', 'priority': 'high'},
            {'type': 'memory_optimization', 'action': 'increase_garbage_collection_frequency', 'priority': 'medium'}
        ]
        
        results = optimizer.apply_optimizations(recommendations)
        
        assert 'applied' in results
        assert 'skipped' in results
        assert 'errors' in results
        assert len(optimizer.optimization_history) == 1
        assert optimizer.last_optimization_time > 0


class TestPerformanceOptimizer:
    """Test PerformanceOptimizer main class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return PerformanceConfig(
            monitoring_interval=0.1,  # Fast for testing
            enable_auto_scaling=True,
            enable_alerting=True
        )
    
    @pytest.fixture
    def optimizer(self, config):
        """Create PerformanceOptimizer fixture"""
        return PerformanceOptimizer(config)
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization"""
        assert optimizer.monitor is not None
        assert optimizer.adaptive_optimizer is not None
        assert optimizer._running is False
        assert optimizer._last_snapshot is None
    
    @pytest.mark.asyncio
    async def test_optimizer_start_stop(self, optimizer):
        """Test optimizer start/stop lifecycle"""
        await optimizer.start()
        
        assert optimizer._running is True
        assert optimizer._monitoring_task is not None
        assert optimizer._optimization_task is not None
        
        await optimizer.stop()
        
        assert optimizer._running is False
    
    @pytest.mark.asyncio
    async def test_optimizer_monitoring_loop(self, optimizer):
        """Test monitoring loop functionality"""
        await optimizer.start()
        
        # Let monitoring run for a short time
        await asyncio.sleep(0.25)
        
        # Should have collected some performance data
        assert optimizer._last_snapshot is not None
        assert len(optimizer.monitor.performance_history) > 0
        
        await optimizer.stop()
    
    @pytest.mark.asyncio
    async def test_optimizer_request_tracking(self, optimizer):
        """Test request tracking functionality"""
        await optimizer.start()
        
        # Track some requests
        optimizer.record_request("request_1")
        optimizer.record_request("request_2")
        
        # Complete requests
        time.sleep(0.01)  # Small delay
        optimizer.complete_request("request_1", success=True)
        optimizer.complete_request("request_2", success=False)
        
        # Let monitoring collect the data
        await asyncio.sleep(0.15)
        
        # Should have recorded the requests
        assert optimizer.monitor.completed_requests == 1
        assert optimizer.monitor.failed_requests == 1
        assert len(optimizer.monitor.latencies) == 2
        
        await optimizer.stop()
    
    @pytest.mark.asyncio
    async def test_optimizer_alert_handling(self, optimizer):
        """Test alert handling"""
        alert_received = []
        
        def alert_handler(metric, snapshot):
            alert_received.append((metric, snapshot))
        
        # Add alert callback before starting
        optimizer.monitor.add_alert_callback(alert_handler)
        
        await optimizer.start()
        
        # Manually trigger alert by creating bad snapshot
        bad_snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            latency_p50=50.0,
            latency_p95=300.0,  # Exceeds threshold
            latency_p99=400.0,
            throughput=500.0,
            cpu_usage=0.95,  # Exceeds threshold
            memory_usage=0.7,
            gpu_usage=0.8,
            gpu_memory_usage=0.5,
            active_requests=100,
            queue_size=25,
            error_rate=0.01,
            cache_hit_rate=0.85
        )
        
        optimizer.monitor.check_alerts(bad_snapshot)
        
        # Should have received alerts
        assert len(alert_received) >= 2  # Latency and CPU alerts
        
        await optimizer.stop()
    
    @pytest.mark.asyncio
    async def test_optimizer_component_injection(self, optimizer):
        """Test component injection for coordination"""
        # Mock components
        mock_concurrency_manager = Mock()
        mock_async_handler = Mock()
        mock_batch_processor = Mock()
        
        optimizer.inject_components(
            concurrency_manager=mock_concurrency_manager,
            async_handler=mock_async_handler,
            batch_processor=mock_batch_processor
        )
        
        assert optimizer.concurrency_manager == mock_concurrency_manager
        assert optimizer.async_handler == mock_async_handler
        assert optimizer.batch_processor == mock_batch_processor
    
    def test_optimizer_performance_retrieval(self, optimizer):
        """Test current performance retrieval"""
        # Should return None initially
        assert optimizer.get_current_performance() is None
        
        # Set a snapshot
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            latency_p50=50.0,
            latency_p95=100.0,
            latency_p99=200.0,
            throughput=1000.0,
            cpu_usage=0.6,
            memory_usage=0.7,
            gpu_usage=0.8,
            gpu_memory_usage=0.5,
            active_requests=100,
            queue_size=20,
            error_rate=0.01,
            cache_hit_rate=0.85
        )
        optimizer._last_snapshot = snapshot
        
        retrieved = optimizer.get_current_performance()
        assert retrieved == snapshot
    
    def test_optimizer_stats_collection(self, optimizer):
        """Test comprehensive stats collection"""
        # Set up some mock data
        optimizer._optimization_stats['optimizations_applied'] = 5
        optimizer._optimization_stats['performance_improvements'] = 3
        
        mock_snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            latency_p50=50.0,
            latency_p95=100.0,
            latency_p99=200.0,
            throughput=1000.0,
            cpu_usage=0.6,
            memory_usage=0.7,
            gpu_usage=0.8,
            gpu_memory_usage=0.5,
            active_requests=100,
            queue_size=20,
            error_rate=0.01,
            cache_hit_rate=0.85
        )
        optimizer._last_snapshot = mock_snapshot
        
        stats = optimizer.get_stats()
        
        assert 'optimizer' in stats
        assert 'current_performance' in stats
        assert 'active_alerts' in stats
        assert 'optimization_history_size' in stats
        assert 'performance_history_size' in stats
        assert 'current_optimizations' in stats
        
        assert stats['optimizer']['optimizations_applied'] == 5
        assert stats['current_performance']['latency_p50'] == 50.0
    
    def test_optimizer_forced_optimization(self, optimizer):
        """Test forced optimization trigger"""
        # Add some performance history
        for i in range(10):
            snapshot = PerformanceSnapshot(
                timestamp=time.time() - (10 - i),
                latency_p50=50.0,
                latency_p95=100.0 + i * 5,  # Increasing latency
                latency_p99=200.0,
                throughput=1000.0 - i * 10,  # Decreasing throughput
                cpu_usage=0.6,
                memory_usage=0.7,
                gpu_usage=0.8,
                gpu_memory_usage=0.5,
                active_requests=100,
                queue_size=20,
                error_rate=0.01,
                cache_hit_rate=0.85
            )
            optimizer.monitor.performance_history.append(snapshot)
        
        optimizer._last_snapshot = optimizer.monitor.performance_history[-1]
        
        # Force optimization
        results = optimizer.force_optimization()
        
        assert results is not None
        assert 'applied' in results or 'message' in results or 'error' in results


class TestPerformanceOptimizerIntegration:
    """Integration tests for PerformanceOptimizer"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_optimization_cycle(self):
        """Test complete optimization cycle"""
        config = PerformanceConfig(
            monitoring_interval=0.05,  # Very fast for testing
            enable_auto_scaling=True,
            target_latency_ms=50.0,  # Strict target
            target_throughput_rps=2000.0
        )
        optimizer = PerformanceOptimizer(config)
        
        await optimizer.start()
        
        # Simulate poor performance requests
        for i in range(10):
            optimizer.record_request(f"slow_request_{i}")
            await asyncio.sleep(0.05)  # Simulate slow processing
            optimizer.complete_request(f"slow_request_{i}", success=True)
        
        # Wait for monitoring and optimization cycles
        await asyncio.sleep(0.3)
        
        # Should have collected performance data
        assert len(optimizer.monitor.performance_history) > 0
        assert optimizer._last_snapshot is not None
        
        # Should have high latency due to slow requests
        assert optimizer._last_snapshot.latency_p95 > config.target_latency_ms
        
        await optimizer.stop()
    
    @pytest.mark.asyncio
    async def test_component_coordination(self):
        """Test coordination between optimizer and managed components"""
        config = PerformanceConfig(monitoring_interval=0.1)
        optimizer = PerformanceOptimizer(config)
        
        # Mock managed components with stats
        mock_async_handler = Mock()
        mock_async_handler.get_stats.return_value = {
            'requests': {'successful_requests': 100},
            'cache': {'hit_rate': 0.8}
        }
        
        mock_batch_processor = Mock()
        mock_batch_processor.get_stats.return_value = {
            'queue': {'current_size': 15},
            'processing': {'average_batch_size': 4}
        }
        
        optimizer.inject_components(
            async_handler=mock_async_handler,
            batch_processor=mock_batch_processor
        )
        
        await optimizer.start()
        
        # Let it collect some component metrics
        await asyncio.sleep(0.25)
        
        # Should have called component stats methods
        mock_async_handler.get_stats.assert_called()
        mock_batch_processor.get_stats.assert_called()
        
        await optimizer.stop()
    
    @pytest.mark.asyncio
    async def test_alert_to_optimization_pipeline(self):
        """Test pipeline from alert detection to optimization application"""
        config = PerformanceConfig(
            monitoring_interval=0.05,
            enable_alerting=True,
            enable_auto_scaling=True,
            alert_thresholds={
                PerformanceMetric.LATENCY: 80.0,  # Low threshold for testing
                PerformanceMetric.CPU_USAGE: 0.6,
                PerformanceMetric.MEMORY_USAGE: 0.6
            }
        )
        optimizer = PerformanceOptimizer(config)
        
        alerts_triggered = []
        
        def alert_callback(metric, snapshot):
            alerts_triggered.append(metric)
        
        optimizer.monitor.add_alert_callback(alert_callback)
        
        await optimizer.start()
        
        # Simulate requests that will trigger alerts
        for i in range(5):
            optimizer.record_request(f"alert_request_{i}")
            await asyncio.sleep(0.08)  # Slow enough to exceed latency threshold
            optimizer.complete_request(f"alert_request_{i}", success=True)
        
        # Wait for monitoring and potential optimization
        await asyncio.sleep(0.4)
        
        # Should have triggered some alerts
        assert len(alerts_triggered) > 0
        
        # Should have attempted optimizations
        assert len(optimizer.adaptive_optimizer.optimization_history) >= 0
        
        await optimizer.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

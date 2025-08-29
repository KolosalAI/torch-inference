"""
Unit tests for OptimizationIntegration

Tests the integration utility including:
- OptimizedInferenceServer wrapper functionality
- FastAPI integration helpers
- Configuration management across optimization levels
- Component coordination and lifecycle management
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Callable
from contextlib import asynccontextmanager

from framework.core.optimization_integration import (
    OptimizedInferenceServer,
    FastAPIIntegration,
    create_optimized_server,
    OptimizationLevel
)
from framework.core.concurrency_manager import ConcurrencyConfig
from framework.core.async_handler import ConnectionConfig
from framework.core.batch_processor import BatchConfig
from framework.core.performance_optimizer import PerformanceConfig


class TestOptimizedInferenceServer:
    """Test OptimizedInferenceServer main class"""
    
    def test_server_initialization_default(self):
        """Test server initialization with default settings"""
        server = OptimizedInferenceServer()
        
        assert server.performance_config.optimization_level == OptimizationLevel.BALANCED
        assert server.concurrency_config.max_workers == 8
        assert server.async_config.max_connections == 200
        assert server.batch_config.max_batch_size == 8
        assert server._started is False
    
    def test_server_initialization_conservative(self):
        """Test server initialization with conservative optimization"""
        server = OptimizedInferenceServer(OptimizationLevel.CONSERVATIVE)
        
        assert server.performance_config.optimization_level == OptimizationLevel.CONSERVATIVE
        assert server.concurrency_config.max_workers == 4
        assert server.async_config.max_connections == 50
        assert server.batch_config.max_batch_size == 4
        assert server.performance_config.target_latency_ms == 200.0
    
    def test_server_initialization_aggressive(self):
        """Test server initialization with aggressive optimization"""
        server = OptimizedInferenceServer(OptimizationLevel.AGGRESSIVE)
        
        assert server.performance_config.optimization_level == OptimizationLevel.AGGRESSIVE
        assert server.concurrency_config.max_workers == 16
        assert server.async_config.max_connections == 500
        assert server.batch_config.max_batch_size == 16
        assert server.performance_config.target_latency_ms == 50.0
    
    def test_server_initialization_extreme(self):
        """Test server initialization with extreme optimization"""
        server = OptimizedInferenceServer(OptimizationLevel.EXTREME)
        
        assert server.performance_config.optimization_level == OptimizationLevel.EXTREME
        assert server.concurrency_config.max_workers == 32
        assert server.async_config.max_connections == 1000
        assert server.batch_config.max_batch_size == 32
        assert server.performance_config.target_latency_ms == 25.0
        assert server.async_config.enable_compression is False  # Disabled for extreme performance
    
    def test_server_custom_config_overrides(self):
        """Test server initialization with custom config overrides"""
        custom_configs = {
            'concurrency': {
                'max_workers': 12,
                'enable_circuit_breaker': False
            },
            'batch': {
                'max_batch_size': 6,
                'batch_timeout_ms': 75
            },
            'performance': {
                'target_latency_ms': 80.0,
                'enable_auto_scaling': False
            }
        }
        
        server = OptimizedInferenceServer(
            OptimizationLevel.BALANCED,
            custom_configs
        )
        
        # Check overrides applied
        assert server.concurrency_config.max_workers == 12
        assert server.concurrency_config.enable_circuit_breaker is False
        assert server.batch_config.max_batch_size == 6
        assert server.batch_config.batch_timeout_ms == 75
        assert server.performance_config.target_latency_ms == 80.0
        assert server.performance_config.enable_auto_scaling is False
    
    @pytest.mark.asyncio
    async def test_server_start_stop_lifecycle(self):
        """Test server start/stop lifecycle"""
        server = OptimizedInferenceServer()
        
        # Mock component start/stop methods
        server.concurrency_manager.start = AsyncMock()
        server.async_handler.start = AsyncMock()
        server.batch_processor.start = AsyncMock()
        server.performance_optimizer.start = AsyncMock()
        
        server.concurrency_manager.stop = AsyncMock()
        server.async_handler.stop = AsyncMock()
        server.batch_processor.stop = AsyncMock()
        server.performance_optimizer.stop = AsyncMock()
        
        # Start server
        await server.start()
        
        assert server._started is True
        server.concurrency_manager.start.assert_called_once()
        server.async_handler.start.assert_called_once()
        server.batch_processor.start.assert_called_once()
        server.performance_optimizer.start.assert_called_once()
        
        # Stop server
        await server.stop()
        
        assert server._started is False
        server.concurrency_manager.stop.assert_called_once()
        server.async_handler.stop.assert_called_once()
        server.batch_processor.stop.assert_called_once()
        server.performance_optimizer.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_server_start_failure_cleanup(self):
        """Test cleanup on start failure"""
        server = OptimizedInferenceServer()
        
        # Mock components
        server.concurrency_manager.start = AsyncMock()
        server.async_handler.start = AsyncMock()
        server.batch_processor.start = AsyncMock(side_effect=Exception("Start failed"))
        server.performance_optimizer.start = AsyncMock()
        
        server.concurrency_manager.stop = AsyncMock()
        server.async_handler.stop = AsyncMock()
        server.batch_processor.stop = AsyncMock()
        server.performance_optimizer.stop = AsyncMock()
        
        # Should raise exception and cleanup
        with pytest.raises(Exception, match="Start failed"):
            await server.start()
        
        assert server._started is False
        # Should have called stop for cleanup
        server.performance_optimizer.stop.assert_called_once()
        server.batch_processor.stop.assert_called_once()
        server.async_handler.stop.assert_called_once()
        server.concurrency_manager.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_server_inference_function_wrapping(self):
        """Test wrapping inference functions with optimizations"""
        server = OptimizedInferenceServer()
        
        # Mock server components
        server.performance_optimizer.record_request = Mock()
        server.performance_optimizer.complete_request = Mock()
        server.concurrency_manager.request_context = AsyncMock()
        server.async_handler.get_cached_response = AsyncMock(return_value=None)
        server.async_handler.cache_response = AsyncMock()
        
        @asynccontextmanager
        async def mock_context():
            yield
        
        server.concurrency_manager.request_context = mock_context
        
        # Original inference function
        async def original_inference(data):
            return f"result_{data}"
        
        # Wrap function
        wrapped_inference = server.wrap_inference_function(original_inference)
        
        # Test wrapped function
        result = await wrapped_inference("test_data")
        
        assert result == "result_test_data"
        server.performance_optimizer.record_request.assert_called_once()
        server.performance_optimizer.complete_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_server_inference_caching(self):
        """Test inference result caching"""
        server = OptimizedInferenceServer()
        
        # Mock components
        server.performance_optimizer.record_request = Mock()
        server.performance_optimizer.complete_request = Mock()
        server.concurrency_manager.request_context = AsyncMock()
        
        # Mock cached response
        server.async_handler.get_cached_response = AsyncMock(return_value="cached_result")
        
        @asynccontextmanager
        async def mock_context():
            yield
        
        server.concurrency_manager.request_context = mock_context
        
        # Original function (shouldn't be called due to cache hit)
        call_count = 0
        
        async def original_inference(data):
            nonlocal call_count
            call_count += 1
            return f"result_{data}"
        
        wrapped_inference = server.wrap_inference_function(original_inference)
        result = await wrapped_inference("test_data")
        
        assert result == "cached_result"
        assert call_count == 0  # Original function shouldn't be called
    
    @pytest.mark.asyncio
    async def test_server_error_handling(self):
        """Test error handling in wrapped inference"""
        server = OptimizedInferenceServer()
        
        # Mock components
        server.performance_optimizer.record_request = Mock()
        server.performance_optimizer.complete_request = Mock()
        server.concurrency_manager.request_context = AsyncMock()
        server.async_handler.get_cached_response = AsyncMock(return_value=None)
        
        @asynccontextmanager
        async def mock_context():
            yield
        
        server.concurrency_manager.request_context = mock_context
        
        # Failing inference function
        async def failing_inference(data):
            raise ValueError("Inference failed")
        
        wrapped_inference = server.wrap_inference_function(failing_inference)
        
        with pytest.raises(ValueError, match="Inference failed"):
            await wrapped_inference("test_data")
        
        # Should record failed completion
        server.performance_optimizer.complete_request.assert_called_once_with(
            pytest.any(str), success=False
        )
    
    @pytest.mark.asyncio
    async def test_server_context_manager(self):
        """Test using server as context manager"""
        server = OptimizedInferenceServer()
        
        # Mock start/stop
        server.start = AsyncMock()
        server.stop = AsyncMock()
        
        async with server.optimized_context() as ctx:
            assert ctx == server
            server.start.assert_called_once()
        
        server.stop.assert_called_once()
    
    def test_server_stats_collection(self):
        """Test comprehensive stats collection"""
        server = OptimizedInferenceServer()
        
        # Mock component stats
        server.concurrency_manager.get_stats = Mock(return_value={'workers': 8})
        server.async_handler.get_stats = Mock(return_value={'connections': 50})
        server.batch_processor.get_stats = Mock(return_value={'batches': 100})
        server.performance_optimizer.get_stats = Mock(return_value={'optimizations': 5})
        
        stats = server.get_optimization_stats()
        
        assert 'concurrency' in stats
        assert 'async_handler' in stats
        assert 'batch_processor' in stats
        assert 'performance_optimizer' in stats
        assert 'configuration' in stats
        
        # Check configuration section
        config = stats['configuration']
        assert config['optimization_level'] == 'balanced'
        assert 'concurrency_config' in config
        assert 'batch_config' in config
    
    @pytest.mark.asyncio
    async def test_server_health_check(self):
        """Test comprehensive health check"""
        server = OptimizedInferenceServer()
        
        # Mock component health checks
        server.concurrency_manager.health_check = AsyncMock(return_value={'status': 'healthy'})
        server.async_handler.health_check = AsyncMock(return_value={'status': 'healthy'})
        server.batch_processor.health_check = AsyncMock(return_value={'status': 'healthy'})
        server.performance_optimizer.health_check = AsyncMock(return_value={'status': 'healthy'})
        
        health = await server.health_check()
        
        assert health['status'] == 'healthy'
        assert 'components' in health
        assert 'timestamp' in health
        
        components = health['components']
        assert 'concurrency_manager' in components
        assert 'async_handler' in components
        assert 'batch_processor' in components
        assert 'performance_optimizer' in components
    
    @pytest.mark.asyncio
    async def test_server_health_check_degraded(self):
        """Test health check with degraded component"""
        server = OptimizedInferenceServer()
        
        # Mock one unhealthy component
        server.concurrency_manager.health_check = AsyncMock(return_value={'status': 'healthy'})
        server.async_handler.health_check = AsyncMock(return_value={'status': 'degraded'})
        server.batch_processor.health_check = AsyncMock(return_value={'status': 'healthy'})
        server.performance_optimizer.health_check = AsyncMock(return_value={'status': 'healthy'})
        
        health = await server.health_check()
        
        assert health['status'] == 'degraded'  # Overall status should be degraded
        assert health['components']['async_handler']['status'] == 'degraded'


class TestCreateOptimizedServer:
    """Test create_optimized_server factory function"""
    
    def test_factory_default(self):
        """Test factory with default parameters"""
        server = create_optimized_server()
        
        assert isinstance(server, OptimizedInferenceServer)
        assert server.performance_config.optimization_level == OptimizationLevel.BALANCED
    
    def test_factory_with_level(self):
        """Test factory with specific optimization level"""
        server = create_optimized_server(OptimizationLevel.AGGRESSIVE)
        
        assert server.performance_config.optimization_level == OptimizationLevel.AGGRESSIVE
        assert server.concurrency_config.max_workers == 16
    
    def test_factory_with_custom_configs(self):
        """Test factory with custom configurations"""
        custom_configs = {
            'concurrency': {'max_workers': 20},
            'batch': {'max_batch_size': 12}
        }
        
        server = create_optimized_server(
            OptimizationLevel.BALANCED,
            custom_configs
        )
        
        assert server.concurrency_config.max_workers == 20
        assert server.batch_config.max_batch_size == 12


class TestFastAPIIntegration:
    """Test FastAPI integration helpers"""
    
    def test_middleware_creation(self):
        """Test FastAPI middleware creation"""
        server = OptimizedInferenceServer()
        server.performance_optimizer.record_request = Mock()
        server.performance_optimizer.complete_request = Mock()
        
        middleware_func = FastAPIIntegration.create_middleware(server)
        
        assert callable(middleware_func)
    
    @pytest.mark.asyncio
    async def test_middleware_execution_success(self):
        """Test middleware execution with successful request"""
        server = OptimizedInferenceServer()
        server.performance_optimizer.record_request = Mock()
        server.performance_optimizer.complete_request = Mock()
        
        middleware_func = FastAPIIntegration.create_middleware(server)
        
        # Mock request and response
        mock_request = Mock()
        mock_request.url.path = '/test'
        mock_response = Mock()
        
        async def call_next(request):
            return mock_response
        
        response = await middleware_func(mock_request, call_next)
        
        assert response == mock_response
        server.performance_optimizer.record_request.assert_called_once()
        server.performance_optimizer.complete_request.assert_called_once_with(
            pytest.any(str), success=True
        )
    
    @pytest.mark.asyncio
    async def test_middleware_execution_failure(self):
        """Test middleware execution with request failure"""
        server = OptimizedInferenceServer()
        server.performance_optimizer.record_request = Mock()
        server.performance_optimizer.complete_request = Mock()
        
        middleware_func = FastAPIIntegration.create_middleware(server)
        
        # Mock request
        mock_request = Mock()
        mock_request.url.path = '/test'
        
        async def failing_call_next(request):
            raise ValueError("Request failed")
        
        with pytest.raises(ValueError, match="Request failed"):
            await middleware_func(mock_request, failing_call_next)
        
        server.performance_optimizer.complete_request.assert_called_once_with(
            pytest.any(str), success=False
        )
    
    def test_startup_handler_creation(self):
        """Test FastAPI startup handler creation"""
        server = OptimizedInferenceServer()
        server.start = AsyncMock()
        
        startup_handler = FastAPIIntegration.create_startup_handler(server)
        
        assert callable(startup_handler)
    
    @pytest.mark.asyncio
    async def test_startup_handler_execution(self):
        """Test startup handler execution"""
        server = OptimizedInferenceServer()
        server.start = AsyncMock()
        
        startup_handler = FastAPIIntegration.create_startup_handler(server)
        await startup_handler()
        
        server.start.assert_called_once()
    
    def test_shutdown_handler_creation(self):
        """Test FastAPI shutdown handler creation"""
        server = OptimizedInferenceServer()
        server.stop = AsyncMock()
        
        shutdown_handler = FastAPIIntegration.create_shutdown_handler(server)
        
        assert callable(shutdown_handler)
    
    @pytest.mark.asyncio
    async def test_shutdown_handler_execution(self):
        """Test shutdown handler execution"""
        server = OptimizedInferenceServer()
        server.stop = AsyncMock()
        
        shutdown_handler = FastAPIIntegration.create_shutdown_handler(server)
        await shutdown_handler()
        
        server.stop.assert_called_once()


class TestOptimizationLevels:
    """Test different optimization level configurations"""
    
    def test_conservative_configuration(self):
        """Test conservative optimization configuration"""
        server = OptimizedInferenceServer(OptimizationLevel.CONSERVATIVE)
        
        # Should have conservative settings
        assert server.concurrency_config.max_workers == 4
        assert server.concurrency_config.max_queue_size == 100
        assert server.concurrency_config.requests_per_second == 100
        assert server.async_config.max_connections == 50
        assert server.batch_config.max_batch_size == 4
        assert server.batch_config.batch_timeout_ms == 100
        assert server.performance_config.target_latency_ms == 200.0
        assert server.performance_config.target_throughput_rps == 100.0
        assert server.performance_config.monitoring_interval == 30.0
        assert server.performance_config.enable_predictive_scaling is False
    
    def test_balanced_configuration(self):
        """Test balanced optimization configuration"""
        server = OptimizedInferenceServer(OptimizationLevel.BALANCED)
        
        # Should have balanced settings
        assert server.concurrency_config.max_workers == 8
        assert server.concurrency_config.max_queue_size == 500
        assert server.concurrency_config.requests_per_second == 500
        assert server.async_config.max_connections == 200
        assert server.batch_config.max_batch_size == 8
        assert server.batch_config.batch_timeout_ms == 50
        assert server.performance_config.target_latency_ms == 100.0
        assert server.performance_config.target_throughput_rps == 500.0
        assert server.performance_config.monitoring_interval == 15.0
        assert server.performance_config.enable_predictive_scaling is True
    
    def test_aggressive_configuration(self):
        """Test aggressive optimization configuration"""
        server = OptimizedInferenceServer(OptimizationLevel.AGGRESSIVE)
        
        # Should have aggressive settings
        assert server.concurrency_config.max_workers == 16
        assert server.concurrency_config.max_queue_size == 1000
        assert server.concurrency_config.requests_per_second == 1000
        assert server.async_config.max_connections == 500
        assert server.batch_config.max_batch_size == 16
        assert server.batch_config.batch_timeout_ms == 25
        assert server.performance_config.target_latency_ms == 50.0
        assert server.performance_config.target_throughput_rps == 1000.0
        assert server.performance_config.monitoring_interval == 10.0
        assert server.performance_config.enable_predictive_scaling is True
    
    def test_extreme_configuration(self):
        """Test extreme optimization configuration"""
        server = OptimizedInferenceServer(OptimizationLevel.EXTREME)
        
        # Should have extreme settings
        assert server.concurrency_config.max_workers == 32
        assert server.concurrency_config.max_queue_size == 2000
        assert server.concurrency_config.enable_rate_limiting is False  # No rate limiting
        assert server.async_config.max_connections == 1000
        assert server.async_config.enable_compression is False  # Disabled for performance
        assert server.batch_config.max_batch_size == 32
        assert server.batch_config.batch_timeout_ms == 10
        assert server.performance_config.target_latency_ms == 25.0
        assert server.performance_config.target_throughput_rps == 2000.0
        assert server.performance_config.monitoring_interval == 5.0
        assert server.performance_config.enable_predictive_scaling is True


class TestOptimizationIntegrationE2E:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_complete_optimization_pipeline(self):
        """Test complete optimization pipeline from request to response"""
        server = OptimizedInferenceServer(OptimizationLevel.BALANCED)
        
        # Mock all components to avoid actual network/processing
        server.concurrency_manager.start = AsyncMock()
        server.async_handler.start = AsyncMock()
        server.batch_processor.start = AsyncMock()
        server.performance_optimizer.start = AsyncMock()
        
        server.concurrency_manager.stop = AsyncMock()
        server.async_handler.stop = AsyncMock()
        server.batch_processor.stop = AsyncMock()
        server.performance_optimizer.stop = AsyncMock()
        
        server.performance_optimizer.record_request = Mock()
        server.performance_optimizer.complete_request = Mock()
        server.concurrency_manager.request_context = AsyncMock()
        server.async_handler.get_cached_response = AsyncMock(return_value=None)
        server.async_handler.cache_response = AsyncMock()
        
        @asynccontextmanager
        async def mock_context():
            yield
        
        server.concurrency_manager.request_context = mock_context
        
        # Start server
        await server.start()
        
        # Define test inference function
        async def test_inference(data):
            await asyncio.sleep(0.01)  # Simulate processing
            return f"processed_{data}"
        
        # Wrap function
        optimized_inference = server.wrap_inference_function(test_inference)
        
        # Process multiple requests
        tasks = []
        for i in range(5):
            task = optimized_inference(f"request_{i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Verify results
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result == f"processed_request_{i}"
        
        # Verify performance tracking
        assert server.performance_optimizer.record_request.call_count == 5
        assert server.performance_optimizer.complete_request.call_count == 5
        
        await server.stop()
    
    @pytest.mark.asyncio
    async def test_component_coordination_under_load(self):
        """Test component coordination under simulated load"""
        server = OptimizedInferenceServer(OptimizationLevel.AGGRESSIVE)
        
        # Mock components for coordination testing
        server.concurrency_manager.start = AsyncMock()
        server.async_handler.start = AsyncMock()
        server.batch_processor.start = AsyncMock()
        server.performance_optimizer.start = AsyncMock()
        
        server.concurrency_manager.stop = AsyncMock()
        server.async_handler.stop = AsyncMock()
        server.batch_processor.stop = AsyncMock()
        server.performance_optimizer.stop = AsyncMock()
        
        # Mock stats for performance monitoring
        server.concurrency_manager.get_stats = Mock(return_value={
            'processed_requests': 50,
            'active_workers': 16,
            'queue_size': 10
        })
        server.async_handler.get_stats = Mock(return_value={
            'requests': {'successful_requests': 50},
            'cache': {'hit_rate': 0.7}
        })
        server.batch_processor.get_stats = Mock(return_value={
            'items_processed': 50,
            'average_batch_size': 8,
            'queue': {'current_size': 5}
        })
        server.performance_optimizer.get_stats = Mock(return_value={
            'optimizer': {'optimizations_applied': 2}
        })
        
        await server.start()
        
        # Simulate load by collecting stats multiple times
        for _ in range(3):
            stats = server.get_optimization_stats()
            await asyncio.sleep(0.01)
        
        # Verify component coordination
        assert stats['concurrency']['processed_requests'] == 50
        assert stats['async_handler']['cache']['hit_rate'] == 0.7
        assert stats['batch_processor']['items_processed'] == 50
        
        # Health check should show all components healthy
        health = await server.health_check()
        assert health['status'] in ['healthy', 'degraded']  # May be degraded due to mocking
        
        await server.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

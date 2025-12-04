"""
Unit tests for AsyncRequestHandler

Tests the async request handling system including:
- Connection pooling and management
- Response caching with TTL and LRU eviction
- Rate limiting with token bucket algorithm
- WebSocket connection management
- Request retry logic with exponential backoff
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Optional
import json

from framework.core.async_handler import (
    AsyncRequestHandler,
    ConnectionConfig,
    RateLimiter,
    ResponseCache,
    WebSocketManager,
    RequestRetryHandler,
    ConnectionPoolManager
)


class TestConnectionConfig:
    """Test ConnectionConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ConnectionConfig()
        
        assert config.max_connections == 100
        assert config.connection_timeout == 30.0
        assert config.keep_alive_timeout == 120.0
        assert config.enable_http2 is True
        assert config.enable_compression is True
        assert config.max_request_size_mb == 10.0
        assert config.enable_websockets is True
        assert config.websocket_ping_interval == 30.0
        assert config.enable_request_caching is True
        assert config.cache_size_mb == 100.0
        assert config.cache_ttl_seconds == 300.0
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = ConnectionConfig(
            max_connections=200,
            connection_timeout=60.0,
            enable_http2=False,
            cache_size_mb=200.0
        )
        
        assert config.max_connections == 200
        assert config.connection_timeout == 60.0
        assert config.enable_http2 is False
        assert config.cache_size_mb == 200.0


class TestRateLimiter:
    """Test RateLimiter functionality"""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization"""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=20)
        
        assert limiter.requests_per_second == 10.0
        assert limiter.burst_size == 20
        assert limiter.tokens == 20  # Should start full
    
    def test_rate_limiter_acquire_tokens(self):
        """Test token acquisition"""
        limiter = RateLimiter(requests_per_second=100.0, burst_size=10)
        
        # Should be able to acquire tokens initially
        assert limiter.acquire() is True
        assert limiter.tokens == 9
        
        # Use up all tokens
        for _ in range(9):
            assert limiter.acquire() is True
        
        # Should be out of tokens now
        assert limiter.acquire() is False
    
    def test_rate_limiter_token_refill(self):
        """Test token refill over time"""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=5)
        
        # Use all tokens
        for _ in range(5):
            limiter.acquire()
        
        assert limiter.acquire() is False
        
        # Wait for refill
        time.sleep(0.6)  # Should refill ~6 tokens at 10/sec
        assert limiter.acquire() is True
    
    def test_rate_limiter_wait_for_token(self):
        """Test waiting for token availability"""
        async def test_wait():
            limiter = RateLimiter(requests_per_second=2.0, burst_size=1)
            
            # Use the one token
            assert limiter.acquire() is True
            
            # Wait for next token (should take ~0.5 seconds)
            start_time = time.time()
            await limiter.wait_for_token()
            wait_time = time.time() - start_time
            
            assert 0.4 < wait_time < 0.7  # Allow some variance
            assert limiter.acquire() is True
        
        asyncio.run(test_wait())


class TestResponseCache:
    """Test ResponseCache functionality"""
    
    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = ResponseCache(max_size_mb=50.0, ttl_seconds=300.0)
        
        assert cache.max_size_mb == 50.0
        assert cache.ttl_seconds == 300.0
        assert len(cache.cache) == 0
    
    def test_cache_put_get(self):
        """Test basic cache put/get operations"""
        cache = ResponseCache(max_size_mb=10.0, ttl_seconds=300.0)
        
        key = "test_key"
        value = {"result": "test_data", "status": "success"}
        
        cache.put(key, value)
        retrieved = cache.get(key)
        
        assert retrieved == value
    
    def test_cache_expiration(self):
        """Test cache TTL expiration"""
        cache = ResponseCache(max_size_mb=10.0, ttl_seconds=0.1)  # Very short TTL
        
        cache.put("test_key", "test_value")
        
        # Should be available immediately
        assert cache.get("test_key") == "test_value"
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired
        assert cache.get("test_key") is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = ResponseCache(max_size_mb=0.001, ttl_seconds=300.0)  # Very small cache
        
        # Fill cache
        cache.put("key1", "a" * 500)  # Fill most of cache
        cache.put("key2", "b" * 500)  # Should evict key1
        
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "b" * 500  # Still there
    
    def test_cache_stats(self):
        """Test cache statistics"""
        cache = ResponseCache(max_size_mb=10.0, ttl_seconds=300.0)
        
        # Generate some hits and misses
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        cache.get("key1")  # Hit
        
        stats = cache.get_stats()
        
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 2/3
        assert stats['size'] == 1
    
    def test_cache_clear(self):
        """Test cache clearing"""
        cache = ResponseCache(max_size_mb=10.0, ttl_seconds=300.0)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert len(cache.cache) == 2
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.get("key1") is None


class TestWebSocketManager:
    """Test WebSocketManager functionality"""
    
    def test_websocket_manager_initialization(self):
        """Test WebSocket manager initialization"""
        manager = WebSocketManager(
            max_connections=50,
            ping_interval=30.0,
            close_timeout=10.0
        )
        
        assert manager.max_connections == 50
        assert manager.ping_interval == 30.0
        assert manager.close_timeout == 10.0
        assert len(manager.connections) == 0
    
    @pytest.mark.asyncio
    async def test_websocket_connection_management(self):
        """Test WebSocket connection management"""
        manager = WebSocketManager(max_connections=5)
        
        # Mock WebSocket connections
        mock_ws1 = AsyncMock()
        mock_ws1.client_state = MagicMock()
        mock_ws1.client_state.name = 'OPEN'
        
        mock_ws2 = AsyncMock()
        mock_ws2.client_state = MagicMock()
        mock_ws2.client_state.name = 'OPEN'
        
        # Add connections
        connection_id_1 = await manager.add_connection(mock_ws1)
        connection_id_2 = await manager.add_connection(mock_ws2)
        
        assert len(manager.connections) == 2
        assert connection_id_1 in manager.connections
        assert connection_id_2 in manager.connections
        
        # Remove connection
        await manager.remove_connection(connection_id_1)
        assert len(manager.connections) == 1
        assert connection_id_1 not in manager.connections
    
    @pytest.mark.asyncio
    async def test_websocket_broadcast(self):
        """Test WebSocket message broadcasting"""
        manager = WebSocketManager(max_connections=5)
        
        # Mock WebSocket connections
        mock_ws1 = AsyncMock()
        mock_ws1.client_state = MagicMock()
        mock_ws1.client_state.name = 'OPEN'
        
        mock_ws2 = AsyncMock()
        mock_ws2.client_state = MagicMock()
        mock_ws2.client_state.name = 'OPEN'
        
        # Add connections
        await manager.add_connection(mock_ws1)
        await manager.add_connection(mock_ws2)
        
        # Broadcast message
        message = {"type": "notification", "data": "test"}
        await manager.broadcast(message)
        
        # Both connections should receive message
        mock_ws1.send_text.assert_called_once_with(json.dumps(message))
        mock_ws2.send_text.assert_called_once_with(json.dumps(message))
    
    @pytest.mark.asyncio
    async def test_websocket_send_to_connection(self):
        """Test sending message to specific connection"""
        manager = WebSocketManager(max_connections=5)
        
        mock_ws = AsyncMock()
        mock_ws.client_state = MagicMock()
        mock_ws.client_state.name = 'OPEN'
        
        connection_id = await manager.add_connection(mock_ws)
        
        message = {"type": "response", "data": "specific_message"}
        await manager.send_to_connection(connection_id, message)
        
        mock_ws.send_text.assert_called_once_with(json.dumps(message))
    
    @pytest.mark.asyncio
    async def test_websocket_connection_limit(self):
        """Test WebSocket connection limit enforcement"""
        manager = WebSocketManager(max_connections=2)
        
        # Add connections up to limit
        mock_ws1 = AsyncMock()
        mock_ws1.client_state = MagicMock()
        mock_ws1.client_state.name = 'OPEN'
        
        mock_ws2 = AsyncMock()
        mock_ws2.client_state = MagicMock()
        mock_ws2.client_state.name = 'OPEN'
        
        mock_ws3 = AsyncMock()
        mock_ws3.client_state = MagicMock()
        mock_ws3.client_state.name = 'OPEN'
        
        await manager.add_connection(mock_ws1)
        await manager.add_connection(mock_ws2)
        
        # Third connection should be rejected
        with pytest.raises(Exception):
            await manager.add_connection(mock_ws3)


class TestRequestRetryHandler:
    """Test RequestRetryHandler functionality"""
    
    def test_retry_handler_initialization(self):
        """Test retry handler initialization"""
        handler = RequestRetryHandler(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0
        )
        
        assert handler.max_retries == 3
        assert handler.base_delay == 1.0
        assert handler.max_delay == 60.0
        assert handler.exponential_base == 2.0
    
    def test_retry_delay_calculation(self):
        """Test exponential backoff delay calculation"""
        handler = RequestRetryHandler(
            max_retries=5,
            base_delay=1.0,
            exponential_base=2.0
        )
        
        # Test delay calculations
        assert handler.get_delay(0) == 1.0  # Base delay
        assert handler.get_delay(1) == 2.0  # 1.0 * 2^1
        assert handler.get_delay(2) == 4.0  # 1.0 * 2^2
        assert handler.get_delay(3) == 8.0  # 1.0 * 2^3
    
    def test_retry_delay_max_limit(self):
        """Test maximum delay limit"""
        handler = RequestRetryHandler(
            max_retries=10,
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0
        )
        
        # High attempt number should be capped at max_delay
        assert handler.get_delay(10) == 10.0  # Capped at max_delay
    
    @pytest.mark.asyncio
    async def test_retry_execution_success(self):
        """Test successful execution without retries"""
        handler = RequestRetryHandler(max_retries=3)
        
        call_count = 0
        
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await handler.execute_with_retry(successful_func)
        
        assert result == "success"
        assert call_count == 1  # Should succeed on first try
    
    @pytest.mark.asyncio
    async def test_retry_execution_with_failures(self):
        """Test execution with retries after failures"""
        handler = RequestRetryHandler(max_retries=3, base_delay=0.01)  # Fast retries for testing
        
        call_count = 0
        
        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = await handler.execute_with_retry(failing_then_success)
        
        assert result == "success"
        assert call_count == 3  # Should succeed on third try
    
    @pytest.mark.asyncio
    async def test_retry_execution_max_retries_exceeded(self):
        """Test execution failure after max retries"""
        handler = RequestRetryHandler(max_retries=2, base_delay=0.01)
        
        call_count = 0
        
        async def always_failing():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent failure")
        
        with pytest.raises(ConnectionError, match="Persistent failure"):
            await handler.execute_with_retry(always_failing)
        
        assert call_count == 3  # Initial attempt + 2 retries


class TestConnectionPoolManager:
    """Test ConnectionPoolManager functionality"""
    
    def test_pool_manager_initialization(self):
        """Test connection pool manager initialization"""
        manager = ConnectionPoolManager(
            max_connections=10,
            connection_timeout=30.0,
            keep_alive_timeout=120.0
        )
        
        assert manager.max_connections == 10
        assert manager.connection_timeout == 30.0
        assert manager.keep_alive_timeout == 120.0
    
    @pytest.mark.asyncio
    async def test_pool_connection_acquisition(self):
        """Test connection acquisition from pool"""
        manager = ConnectionPoolManager(max_connections=5)
        
        # Mock connection
        mock_connection = AsyncMock()
        
        with patch.object(manager, '_create_connection', return_value=mock_connection):
            connection = await manager.acquire_connection("http://example.com")
            
            assert connection == mock_connection
            assert len(manager.active_connections) == 1
    
    @pytest.mark.asyncio
    async def test_pool_connection_release(self):
        """Test connection release back to pool"""
        manager = ConnectionPoolManager(max_connections=5)
        
        mock_connection = AsyncMock()
        
        with patch.object(manager, '_create_connection', return_value=mock_connection):
            # Acquire connection
            connection = await manager.acquire_connection("http://example.com")
            assert len(manager.active_connections) == 1
            
            # Release connection
            await manager.release_connection(connection)
            assert len(manager.active_connections) == 0
    
    @pytest.mark.asyncio
    async def test_pool_connection_limit(self):
        """Test connection pool limit enforcement"""
        manager = ConnectionPoolManager(max_connections=2)
        
        with patch.object(manager, '_create_connection', side_effect=lambda x: AsyncMock()):
            # Acquire connections up to limit
            conn1 = await manager.acquire_connection("http://example1.com")
            conn2 = await manager.acquire_connection("http://example2.com")
            
            # Third connection should wait or be rejected
            with pytest.raises(Exception):  # Pool exhausted
                await asyncio.wait_for(
                    manager.acquire_connection("http://example3.com"),
                    timeout=0.1
                )


class TestAsyncRequestHandler:
    """Test AsyncRequestHandler main class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return ConnectionConfig(
            max_connections=10,
            connection_timeout=30.0,
            enable_request_caching=True,
            cache_size_mb=10.0,
            cache_ttl_seconds=60.0,
            enable_websockets=True
        )
    
    @pytest.fixture
    def handler(self, config):
        """Create AsyncRequestHandler fixture"""
        return AsyncRequestHandler(config)
    
    @pytest.mark.asyncio
    async def test_handler_initialization(self, handler):
        """Test handler initialization"""
        await handler.start()
        
        assert handler._started is True
        assert handler.connection_pool is not None
        assert handler.response_cache is not None
        assert handler.rate_limiter is not None
        
        await handler.stop()
    
    @pytest.mark.asyncio
    async def test_handler_request_processing(self, handler):
        """Test basic request processing"""
        await handler.start()
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        
        with patch('aiohttp.ClientSession.request', return_value=mock_response):
            response = await handler.make_request(
                method="GET",
                url="http://example.com/api",
                headers={"Content-Type": "application/json"}
            )
            
            assert response["result"] == "success"
        
        await handler.stop()
    
    @pytest.mark.asyncio
    async def test_handler_caching(self, handler):
        """Test response caching functionality"""
        await handler.start()
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "cached_data"}
        
        with patch('aiohttp.ClientSession.request', return_value=mock_response) as mock_request:
            # First request should hit the network
            response1 = await handler.make_request(
                method="GET",
                url="http://example.com/api/cache-test"
            )
            
            # Second identical request should use cache
            response2 = await handler.make_request(
                method="GET",
                url="http://example.com/api/cache-test"
            )
            
            assert response1 == response2
            # Should only make one actual request due to caching
            assert mock_request.call_count == 1
        
        await handler.stop()
    
    @pytest.mark.asyncio
    async def test_handler_rate_limiting(self, handler):
        """Test rate limiting functionality"""
        # Create handler with strict rate limiting
        config = ConnectionConfig(
            rate_limit_per_second=2.0,
            rate_limit_burst=2
        )
        handler = AsyncRequestHandler(config)
        await handler.start()
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "rate_limited"}
        
        with patch('aiohttp.ClientSession.request', return_value=mock_response):
            # First two requests should succeed
            response1 = await handler.make_request("GET", "http://example.com/1")
            response2 = await handler.make_request("GET", "http://example.com/2")
            
            assert response1["result"] == "rate_limited"
            assert response2["result"] == "rate_limited"
            
            # Third request should be rate limited
            start_time = time.time()
            response3 = await handler.make_request("GET", "http://example.com/3")
            elapsed = time.time() - start_time
            
            # Should have been delayed by rate limiter
            assert elapsed > 0.4  # Should wait ~0.5 seconds
        
        await handler.stop()
    
    @pytest.mark.asyncio
    async def test_handler_retry_logic(self, handler):
        """Test request retry logic"""
        await handler.start()
        
        # Mock failing then succeeding responses
        mock_responses = [
            AsyncMock(status_code=500),  # First attempt fails
            AsyncMock(status_code=500),  # Second attempt fails
            AsyncMock(status_code=200)   # Third attempt succeeds
        ]
        mock_responses[2].json.return_value = {"result": "retry_success"}
        
        call_count = 0
        def mock_request(*args, **kwargs):
            nonlocal call_count
            response = mock_responses[call_count]
            call_count += 1
            return response
        
        with patch('aiohttp.ClientSession.request', side_effect=mock_request):
            response = await handler.make_request_with_retry(
                method="GET",
                url="http://example.com/api/retry",
                max_retries=3
            )
            
            assert response["result"] == "retry_success"
            assert call_count == 3  # Should have made 3 attempts
        
        await handler.stop()
    
    @pytest.mark.asyncio
    async def test_handler_websocket_management(self, handler):
        """Test WebSocket management functionality"""
        await handler.start()
        
        mock_websocket = AsyncMock()
        mock_websocket.client_state = MagicMock()
        mock_websocket.client_state.name = 'OPEN'
        
        # Add WebSocket connection
        connection_id = await handler.add_websocket_connection(mock_websocket)
        
        assert connection_id is not None
        
        # Send message to WebSocket
        message = {"type": "test", "data": "websocket_message"}
        await handler.send_websocket_message(connection_id, message)
        
        mock_websocket.send_text.assert_called_once()
        
        # Remove WebSocket connection
        await handler.remove_websocket_connection(connection_id)
        
        await handler.stop()
    
    @pytest.mark.asyncio
    async def test_handler_stats_collection(self, handler):
        """Test statistics collection"""
        await handler.start()
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "stats_test"}
        
        with patch('aiohttp.ClientSession.request', return_value=mock_response):
            # Make some requests to generate stats
            await handler.make_request("GET", "http://example.com/1")
            await handler.make_request("GET", "http://example.com/2")
        
        stats = handler.get_stats()
        
        assert 'requests' in stats
        assert 'cache' in stats
        assert 'connections' in stats
        assert 'websockets' in stats
        assert stats['requests']['total_requests'] >= 2
        
        await handler.stop()
    
    @pytest.mark.asyncio
    async def test_handler_health_check(self, handler):
        """Test health check functionality"""
        await handler.start()
        
        health = await handler.health_check()
        
        assert health['status'] == 'healthy'
        assert 'connection_pool' in health['components']
        assert 'cache' in health['components']
        assert 'websocket_manager' in health['components']
        
        await handler.stop()
    
    @pytest.mark.asyncio
    async def test_handler_graceful_shutdown(self, handler):
        """Test graceful shutdown"""
        await handler.start()
        
        # Add some mock connections
        mock_websocket = AsyncMock()
        mock_websocket.client_state = MagicMock()
        mock_websocket.client_state.name = 'OPEN'
        
        connection_id = await handler.add_websocket_connection(mock_websocket)
        
        # Stop handler
        await handler.stop()
        
        # Should have closed WebSocket connections
        mock_websocket.close.assert_called_once()
        assert handler._started is False


class TestAsyncRequestHandlerIntegration:
    """Integration tests for AsyncRequestHandler"""
    
    @pytest.mark.asyncio
    async def test_high_concurrency_requests(self):
        """Test handling high concurrency requests"""
        config = ConnectionConfig(
            max_connections=50,
            rate_limit_per_second=100.0,
            enable_request_caching=True
        )
        handler = AsyncRequestHandler(config)
        await handler.start()
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "concurrent_success"}
        
        with patch('aiohttp.ClientSession.request', return_value=mock_response):
            # Submit many concurrent requests
            tasks = []
            for i in range(50):
                task = handler.make_request(
                    method="GET",
                    url=f"http://example.com/api/{i}"
                )
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            successful_results = [r for r in results if not isinstance(r, Exception)]
            
            # Should handle most requests successfully
            assert len(successful_results) >= 45
            
            # Should be reasonably fast due to connection pooling
            total_time = end_time - start_time
            assert total_time < 5.0
        
        await handler.stop()
    
    @pytest.mark.asyncio
    async def test_cache_efficiency_under_load(self):
        """Test cache efficiency under load"""
        config = ConnectionConfig(
            enable_request_caching=True,
            cache_size_mb=5.0,
            cache_ttl_seconds=60.0
        )
        handler = AsyncRequestHandler(config)
        await handler.start()
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "cached_response"}
        
        request_count = 0
        def count_requests(*args, **kwargs):
            nonlocal request_count
            request_count += 1
            return mock_response
        
        with patch('aiohttp.ClientSession.request', side_effect=count_requests):
            # Make repeated requests to same endpoints
            tasks = []
            for i in range(20):
                # Repeat same URLs to test caching
                url = f"http://example.com/api/{i % 5}"  # Only 5 unique URLs
                task = handler.make_request("GET", url)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Should have made significantly fewer actual requests due to caching
            assert request_count <= 10  # Should cache effectively
            assert len(results) == 20    # All requests should complete
            
            # Check cache stats
            stats = handler.get_stats()
            assert stats['cache']['hit_rate'] > 0.5  # Good cache hit rate
        
        await handler.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

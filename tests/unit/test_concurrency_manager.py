"""
Unit tests for ConcurrencyManager

Tests the concurrency management system including:
- Request queuing and priority handling
- Worker pool scaling
- Circuit breaker functionality
- Rate limiting
- Request coalescing
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from framework.core.concurrency_manager import (
    ConcurrencyManager,
    ConcurrencyConfig,
    RequestPriority,
    CircuitBreaker,
    RateLimiter,
    WorkerPool,
    RequestQueue
)


class TestConcurrencyConfig:
    """Test ConcurrencyConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ConcurrencyConfig()
        
        assert config.max_workers == 8
        assert config.max_queue_size == 1000
        assert config.worker_timeout == 300.0
        assert config.enable_circuit_breaker is True
        assert config.circuit_breaker_failure_threshold == 5
        assert config.circuit_breaker_timeout == 60.0
        assert config.enable_rate_limiting is True
        assert config.requests_per_second == 1000.0
        assert config.enable_request_coalescing is True
        assert config.enable_priority_queue is True
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = ConcurrencyConfig(
            max_workers=16,
            max_queue_size=2000,
            enable_circuit_breaker=False,
            requests_per_second=500.0
        )
        
        assert config.max_workers == 16
        assert config.max_queue_size == 2000
        assert config.enable_circuit_breaker is False
        assert config.requests_per_second == 500.0


class TestCircuitBreaker:
    """Test CircuitBreaker functionality"""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state"""
        breaker = CircuitBreaker(failure_threshold=3, timeout=60.0)
        
        assert breaker.state == "closed"
        assert breaker.can_execute() is True
    
    def test_circuit_breaker_failure_tracking(self):
        """Test failure tracking and state transitions"""
        breaker = CircuitBreaker(failure_threshold=2, timeout=1.0)
        
        # Record first failure
        breaker.record_failure()
        assert breaker.state == "closed"
        assert breaker.failure_count == 1
        
        # Record second failure - should open circuit
        breaker.record_failure()
        assert breaker.state == "open"
        assert breaker.can_execute() is False
    
    def test_circuit_breaker_success_reset(self):
        """Test success resets failure count"""
        breaker = CircuitBreaker(failure_threshold=3, timeout=60.0)
        
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.failure_count == 2
        
        breaker.record_success()
        assert breaker.failure_count == 0
    
    def test_circuit_breaker_half_open_state(self):
        """Test half-open state after timeout"""
        breaker = CircuitBreaker(failure_threshold=1, timeout=0.1)
        
        # Trigger circuit open
        breaker.record_failure()
        assert breaker.state == "open"
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Should transition to half-open
        assert breaker.can_execute() is True
        assert breaker.state == "half-open"
        
        # Success should close circuit
        breaker.record_success()
        assert breaker.state == "closed"


class TestRateLimiter:
    """Test RateLimiter functionality"""
    
    def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within limit"""
        limiter = RateLimiter(requests_per_second=10.0, bucket_size=10)
        
        # Should allow initial requests
        for _ in range(5):
            assert limiter.can_proceed() is True
    
    def test_rate_limiter_blocks_excess_requests(self):
        """Test rate limiter blocks excess requests"""
        limiter = RateLimiter(requests_per_second=1.0, bucket_size=2)
        
        # Use up tokens
        assert limiter.can_proceed() is True
        assert limiter.can_proceed() is True
        
        # Should be blocked now
        assert limiter.can_proceed() is False
    
    def test_rate_limiter_token_refill(self):
        """Test token bucket refill over time"""
        limiter = RateLimiter(requests_per_second=10.0, bucket_size=1)
        
        # Use token
        assert limiter.can_proceed() is True
        assert limiter.can_proceed() is False
        
        # Wait for refill
        time.sleep(0.15)  # Should refill at least 1 token
        assert limiter.can_proceed() is True


class TestRequestQueue:
    """Test RequestQueue functionality"""
    
    @pytest.fixture
    def queue(self):
        """Create RequestQueue fixture"""
        return RequestQueue(max_size=10, enable_priority=True)
    
    def test_queue_put_get(self, queue):
        """Test basic put/get operations"""
        request = {"id": "test", "data": "test_data"}
        
        queue.put(request, RequestPriority.NORMAL)
        assert queue.size() == 1
        
        retrieved = queue.get()
        assert retrieved == request
        assert queue.size() == 0
    
    def test_queue_priority_ordering(self, queue):
        """Test priority ordering in queue"""
        low_request = {"id": "low", "priority": "low"}
        normal_request = {"id": "normal", "priority": "normal"}
        high_request = {"id": "high", "priority": "high"}
        
        # Add in mixed order
        queue.put(normal_request, RequestPriority.NORMAL)
        queue.put(low_request, RequestPriority.LOW)
        queue.put(high_request, RequestPriority.HIGH)
        
        # Should get high priority first
        assert queue.get()["id"] == "high"
        assert queue.get()["id"] == "normal"
        assert queue.get()["id"] == "low"
    
    def test_queue_max_size_enforcement(self, queue):
        """Test queue size limit enforcement"""
        # Fill queue to capacity
        for i in range(10):
            queue.put({"id": i}, RequestPriority.NORMAL)
        
        # Should reject additional items
        with pytest.raises(Exception):  # Queue full exception
            queue.put({"id": "overflow"}, RequestPriority.NORMAL)
    
    def test_queue_empty_get(self, queue):
        """Test getting from empty queue"""
        with pytest.raises(Exception):  # Empty queue exception
            queue.get()


class TestWorkerPool:
    """Test WorkerPool functionality"""
    
    @pytest.fixture
    def worker_pool(self):
        """Create WorkerPool fixture"""
        return WorkerPool(max_workers=4, worker_timeout=10.0)
    
    @pytest.mark.asyncio
    async def test_worker_pool_initialization(self, worker_pool):
        """Test worker pool initialization"""
        await worker_pool.start()
        
        assert len(worker_pool.workers) == 4
        assert worker_pool.active_workers == 4
        
        await worker_pool.stop()
    
    @pytest.mark.asyncio
    async def test_worker_pool_task_execution(self, worker_pool):
        """Test task execution through worker pool"""
        await worker_pool.start()
        
        async def test_task():
            await asyncio.sleep(0.1)
            return "test_result"
        
        future = await worker_pool.submit_task(test_task)
        result = await future
        
        assert result == "test_result"
        
        await worker_pool.stop()
    
    @pytest.mark.asyncio
    async def test_worker_pool_scaling(self, worker_pool):
        """Test worker pool scaling"""
        await worker_pool.start()
        initial_workers = len(worker_pool.workers)
        
        # Scale up
        await worker_pool.scale_workers(8)
        assert len(worker_pool.workers) == 8
        
        # Scale down
        await worker_pool.scale_workers(2)
        assert len(worker_pool.workers) == 2
        
        await worker_pool.stop()
    
    @pytest.mark.asyncio
    async def test_worker_pool_error_handling(self, worker_pool):
        """Test worker pool error handling"""
        await worker_pool.start()
        
        async def failing_task():
            raise ValueError("Test error")
        
        future = await worker_pool.submit_task(failing_task)
        
        with pytest.raises(ValueError, match="Test error"):
            await future
        
        await worker_pool.stop()


class TestConcurrencyManager:
    """Test ConcurrencyManager main class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return ConcurrencyConfig(
            max_workers=4,
            max_queue_size=10,
            enable_circuit_breaker=True,
            circuit_breaker_failure_threshold=2,
            enable_rate_limiting=True,
            requests_per_second=10.0
        )
    
    @pytest.fixture
    def manager(self, config):
        """Create ConcurrencyManager fixture"""
        return ConcurrencyManager(config)
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, manager):
        """Test manager initialization"""
        await manager.start()
        
        assert manager._started is True
        assert manager.worker_pool is not None
        assert manager.request_queue is not None
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_request_processing(self, manager):
        """Test request processing through manager"""
        await manager.start()
        
        async def test_handler(data):
            return f"processed_{data}"
        
        result = await manager.process_request(
            handler=test_handler,
            data="test_data",
            priority=RequestPriority.NORMAL
        )
        
        assert result == "processed_test_data"
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_rate_limiting(self, manager):
        """Test rate limiting functionality"""
        await manager.start()
        
        async def fast_handler(data):
            return data
        
        # Submit multiple requests rapidly
        tasks = []
        for i in range(15):  # More than rate limit
            task = manager.process_request(
                handler=fast_handler,
                data=f"data_{i}",
                priority=RequestPriority.NORMAL
            )
            tasks.append(task)
        
        # Some should be rate limited
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should have some successful results and some rate limit exceptions
        successful = [r for r in results if not isinstance(r, Exception)]
        rate_limited = [r for r in results if isinstance(r, Exception)]
        
        assert len(successful) <= 10  # Rate limit
        assert len(rate_limited) > 0
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_circuit_breaker(self, manager):
        """Test circuit breaker functionality"""
        await manager.start()
        
        async def failing_handler(data):
            raise ValueError("Simulated failure")
        
        # Trigger circuit breaker
        for i in range(3):
            try:
                await manager.process_request(
                    handler=failing_handler,
                    data=f"data_{i}",
                    priority=RequestPriority.NORMAL
                )
            except ValueError:
                pass  # Expected
        
        # Circuit should now be open
        with pytest.raises(Exception):  # Circuit breaker exception
            await manager.process_request(
                handler=failing_handler,
                data="blocked_data",
                priority=RequestPriority.NORMAL
            )
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_request_coalescing(self, manager):
        """Test request coalescing functionality"""
        await manager.start()
        
        call_count = 0
        
        async def counted_handler(data):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate processing time
            return f"result_{data}"
        
        # Submit identical requests concurrently
        tasks = []
        for _ in range(5):
            task = manager.process_request(
                handler=counted_handler,
                data="same_data",
                priority=RequestPriority.NORMAL
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Should have coalesced requests (fewer handler calls)
        assert call_count < 5  # Coalescing should reduce calls
        assert all(r == "result_same_data" for r in results)
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_context_manager(self, manager):
        """Test using manager as context manager"""
        async with manager.request_context():
            # Should be able to process requests within context
            pass  # Context should work without errors
    
    @pytest.mark.asyncio
    async def test_manager_stats_collection(self, manager):
        """Test statistics collection"""
        await manager.start()
        
        async def test_handler(data):
            return data
        
        # Process some requests
        for i in range(3):
            await manager.process_request(
                handler=test_handler,
                data=f"data_{i}",
                priority=RequestPriority.NORMAL
            )
        
        stats = manager.get_stats()
        
        assert 'processed_requests' in stats
        assert 'failed_requests' in stats
        assert 'average_processing_time' in stats
        assert 'active_workers' in stats
        assert 'queue_size' in stats
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_health_check(self, manager):
        """Test health check functionality"""
        await manager.start()
        
        health = await manager.health_check()
        
        assert health['status'] == 'healthy'
        assert 'worker_pool' in health['components']
        assert 'circuit_breaker' in health['components']
        assert 'rate_limiter' in health['components']
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_graceful_shutdown(self, manager):
        """Test graceful shutdown with pending requests"""
        await manager.start()
        
        async def slow_handler(data):
            await asyncio.sleep(0.2)
            return data
        
        # Submit request and immediately stop
        task = asyncio.create_task(manager.process_request(
            handler=slow_handler,
            data="test_data",
            priority=RequestPriority.NORMAL
        ))
        
        await asyncio.sleep(0.05)  # Let request start
        await manager.stop()
        
        # Task should still complete
        result = await task
        assert result == "test_data"


class TestConcurrencyManagerIntegration:
    """Integration tests for ConcurrencyManager"""
    
    @pytest.mark.asyncio
    async def test_high_load_processing(self):
        """Test processing high load of concurrent requests"""
        config = ConcurrencyConfig(
            max_workers=8,
            max_queue_size=100,
            requests_per_second=50.0
        )
        manager = ConcurrencyManager(config)
        await manager.start()
        
        async def load_handler(data):
            await asyncio.sleep(0.01)  # Simulate work
            return f"processed_{data}"
        
        # Submit many concurrent requests
        tasks = []
        for i in range(50):
            task = manager.process_request(
                handler=load_handler,
                data=f"data_{i}",
                priority=RequestPriority.NORMAL
            )
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        # Should process most requests successfully
        assert len(successful_results) >= 40  # Allow for some rate limiting
        
        # Should be reasonably fast due to concurrency
        total_time = end_time - start_time
        assert total_time < 2.0  # Should be much faster than sequential
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_mixed_priority_processing(self):
        """Test processing requests with mixed priorities"""
        config = ConcurrencyConfig(
            max_workers=2,  # Limited workers to test priority
            max_queue_size=20
        )
        manager = ConcurrencyManager(config)
        await manager.start()
        
        processed_order = []
        
        async def tracking_handler(data):
            processed_order.append(data)
            await asyncio.sleep(0.05)
            return data
        
        # Submit mixed priority requests
        tasks = []
        
        # Add normal priority requests
        for i in range(3):
            task = manager.process_request(
                handler=tracking_handler,
                data=f"normal_{i}",
                priority=RequestPriority.NORMAL
            )
            tasks.append(task)
        
        # Add high priority request
        high_task = manager.process_request(
            handler=tracking_handler,
            data="high_priority",
            priority=RequestPriority.HIGH
        )
        tasks.append(high_task)
        
        # Add low priority request
        low_task = manager.process_request(
            handler=tracking_handler,
            data="low_priority",
            priority=RequestPriority.LOW
        )
        tasks.append(low_task)
        
        await asyncio.gather(*tasks)
        
        # High priority should be processed before low priority
        high_index = processed_order.index("high_priority")
        low_index = processed_order.index("low_priority")
        
        assert high_index < low_index
        
        await manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

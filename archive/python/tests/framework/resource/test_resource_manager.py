"""
Tests for Resource Management implementation.
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from framework.resource.resource_manager import (
    ResourceManager, MemoryTracker, RequestQueue, 
    ResourceQuotaManager, ResourceLimits
)


class TestResourceLimits:
    """Test resource limits data structure."""
    
    def test_resource_limits_creation(self):
        """Test creating resource limits."""
        limits = ResourceLimits(
            max_memory_mb=1024,
            max_concurrent_requests=100,
            max_queue_size=500,
            max_request_duration_seconds=300
        )
        
        assert limits.max_memory_mb == 1024
        assert limits.max_concurrent_requests == 100
        assert limits.max_queue_size == 500
        assert limits.max_request_duration_seconds == 300
    
    def test_resource_limits_validation(self):
        """Test resource limits validation."""
        # Valid limits should not raise exception
        ResourceLimits(max_memory_mb=512, max_concurrent_requests=50)
        
        # Invalid limits should raise ValueError
        with pytest.raises(ValueError):
            ResourceLimits(max_memory_mb=-1)  # Negative value
        
        with pytest.raises(ValueError):
            ResourceLimits(max_concurrent_requests=0)  # Zero value
    
    def test_resource_limits_to_dict(self):
        """Test converting resource limits to dict."""
        limits = ResourceLimits(
            max_memory_mb=2048,
            max_concurrent_requests=200
        )
        
        limits_dict = limits.to_dict()
        
        assert limits_dict["max_memory_mb"] == 2048
        assert limits_dict["max_concurrent_requests"] == 200
        assert "max_queue_size" in limits_dict
        assert "max_request_duration_seconds" in limits_dict


class TestMemoryTracker:
    """Test memory tracker functionality."""
    
    @pytest.fixture
    def memory_tracker(self):
        """Create memory tracker."""
        return MemoryTracker(warning_threshold_mb=512, critical_threshold_mb=1024)
    
    def test_memory_tracker_initialization(self, memory_tracker):
        """Test memory tracker initialization."""
        assert memory_tracker.warning_threshold_mb == 512
        assert memory_tracker.critical_threshold_mb == 1024
        assert memory_tracker._peak_memory == 0
        assert len(memory_tracker._memory_history) == 0
    
    @patch('psutil.Process')
    def test_get_current_memory_usage(self, mock_process, memory_tracker):
        """Test getting current memory usage."""
        # Mock process memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 128 * 1024 * 1024  # 128 MB in bytes
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        current_memory = memory_tracker.get_current_memory_usage()
        
        assert current_memory == 128  # Should return MB
    
    @patch('psutil.Process')
    def test_update_memory_stats(self, mock_process, memory_tracker):
        """Test updating memory statistics."""
        # Mock memory usage
        mock_memory_info = Mock()
        mock_memory_info.rss = 256 * 1024 * 1024  # 256 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        memory_tracker.update_memory_stats()
        
        assert memory_tracker._peak_memory == 256
        assert len(memory_tracker._memory_history) == 1
        assert memory_tracker._memory_history[0][1] == 256
    
    @patch('psutil.Process')
    def test_memory_threshold_detection(self, mock_process, memory_tracker):
        """Test memory threshold detection."""
        # Test normal memory usage
        mock_memory_info = Mock()
        mock_memory_info.rss = 256 * 1024 * 1024  # 256 MB (below warning)
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        status = memory_tracker.check_memory_status()
        assert status["level"] == "normal"
        assert not status["warning"]
        assert not status["critical"]
        
        # Test warning threshold
        mock_memory_info.rss = 600 * 1024 * 1024  # 600 MB (above warning, below critical)
        
        status = memory_tracker.check_memory_status()
        assert status["level"] == "warning"
        assert status["warning"]
        assert not status["critical"]
        
        # Test critical threshold
        mock_memory_info.rss = 1200 * 1024 * 1024  # 1200 MB (above critical)
        
        status = memory_tracker.check_memory_status()
        assert status["level"] == "critical"
        assert status["warning"]
        assert status["critical"]
    
    def test_get_peak_memory(self, memory_tracker):
        """Test getting peak memory usage."""
        # Simulate memory usage updates
        memory_tracker._peak_memory = 512
        
        peak = memory_tracker.get_peak_memory()
        assert peak == 512
    
    def test_get_memory_history(self, memory_tracker):
        """Test getting memory usage history."""
        # Add some history entries
        now = datetime.utcnow()
        memory_tracker._memory_history = [
            (now - timedelta(minutes=2), 256),
            (now - timedelta(minutes=1), 384),
            (now, 512)
        ]
        
        history = memory_tracker.get_memory_history()
        
        assert len(history) == 3
        assert history[0]["memory_mb"] == 256
        assert history[1]["memory_mb"] == 384
        assert history[2]["memory_mb"] == 512
    
    def test_reset_memory_stats(self, memory_tracker):
        """Test resetting memory statistics."""
        # Set some data
        memory_tracker._peak_memory = 1024
        memory_tracker._memory_history = [(datetime.utcnow(), 512)]
        
        memory_tracker.reset_stats()
        
        assert memory_tracker._peak_memory == 0
        assert len(memory_tracker._memory_history) == 0


class TestRequestQueue:
    """Test request queue functionality."""
    
    @pytest.fixture
    def request_queue(self):
        """Create request queue."""
        return RequestQueue(max_size=100, timeout_seconds=30)
    
    def test_request_queue_initialization(self, request_queue):
        """Test request queue initialization."""
        assert request_queue.max_size == 100
        assert request_queue.timeout_seconds == 30
        assert request_queue.size == 0
        assert request_queue.is_empty
        assert not request_queue.is_full
    
    @pytest.mark.asyncio
    async def test_enqueue_request(self, request_queue):
        """Test enqueueing requests."""
        request_data = {"operation": "predict", "data": {"input": "test"}}
        
        await request_queue.enqueue("req-1", request_data)
        
        assert request_queue.size == 1
        assert not request_queue.is_empty
    
    @pytest.mark.asyncio
    async def test_dequeue_request(self, request_queue):
        """Test dequeueing requests."""
        request_data = {"operation": "predict", "data": {"input": "test"}}
        
        await request_queue.enqueue("req-1", request_data)
        
        request_id, data, enqueue_time = await request_queue.dequeue()
        
        assert request_id == "req-1"
        assert data == request_data
        assert isinstance(enqueue_time, datetime)
        assert request_queue.size == 0
        assert request_queue.is_empty
    
    @pytest.mark.asyncio
    async def test_queue_fifo_order(self, request_queue):
        """Test queue FIFO ordering."""
        # Enqueue multiple requests
        for i in range(5):
            await request_queue.enqueue(f"req-{i}", {"index": i})
        
        # Dequeue and verify order
        for i in range(5):
            request_id, data, _ = await request_queue.dequeue()
            assert request_id == f"req-{i}"
            assert data["index"] == i
    
    @pytest.mark.asyncio
    async def test_queue_max_size(self, request_queue):
        """Test queue maximum size enforcement."""
        # Fill queue to capacity
        for i in range(100):
            await request_queue.enqueue(f"req-{i}", {"index": i})
        
        assert request_queue.size == 100
        assert request_queue.is_full
        
        # Adding one more should raise exception
        with pytest.raises(asyncio.QueueFull):
            await asyncio.wait_for(
                request_queue.enqueue("overflow", {}), 
                timeout=0.1
            )
    
    @pytest.mark.asyncio
    async def test_dequeue_timeout(self, request_queue):
        """Test dequeue timeout when queue is empty."""
        # Empty queue should timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(request_queue.dequeue(), timeout=0.1)
    
    @pytest.mark.asyncio
    async def test_priority_enqueue(self, request_queue):
        """Test priority enqueueing."""
        # Enqueue regular requests
        await request_queue.enqueue("regular-1", {"priority": "normal"})
        await request_queue.enqueue("regular-2", {"priority": "normal"})
        
        # Enqueue priority request
        await request_queue.enqueue("priority-1", {"priority": "high"}, priority=True)
        
        # Priority request should come first
        request_id, data, _ = await request_queue.dequeue()
        assert request_id == "priority-1"
        assert data["priority"] == "high"
    
    def test_get_queue_stats(self, request_queue):
        """Test getting queue statistics."""
        stats = request_queue.get_stats()
        
        assert "size" in stats
        assert "max_size" in stats
        assert "is_empty" in stats
        assert "is_full" in stats
        assert "timeout_seconds" in stats
        
        assert stats["size"] == 0
        assert stats["max_size"] == 100
        assert stats["is_empty"] is True
        assert stats["is_full"] is False
    
    @pytest.mark.asyncio
    async def test_clear_queue(self, request_queue):
        """Test clearing the queue."""
        # Add some requests
        for i in range(10):
            await request_queue.enqueue(f"req-{i}", {"index": i})
        
        assert request_queue.size == 10
        
        request_queue.clear()
        
        assert request_queue.size == 0
        assert request_queue.is_empty


class TestResourceQuotaManager:
    """Test resource quota manager functionality."""
    
    @pytest.fixture
    def quota_manager(self):
        """Create resource quota manager."""
        return ResourceQuotaManager()
    
    def test_quota_manager_initialization(self, quota_manager):
        """Test quota manager initialization."""
        assert len(quota_manager._quotas) == 0
        assert len(quota_manager._usage) == 0
    
    def test_set_quota(self, quota_manager):
        """Test setting resource quota."""
        quota_manager.set_quota("user123", "requests_per_minute", 100)
        quota_manager.set_quota("user123", "memory_mb", 512)
        
        assert quota_manager._quotas["user123"]["requests_per_minute"] == 100
        assert quota_manager._quotas["user123"]["memory_mb"] == 512
    
    def test_check_quota_under_limit(self, quota_manager):
        """Test quota check when under limit."""
        quota_manager.set_quota("user123", "requests_per_minute", 100)
        
        # Simulate usage below limit
        quota_manager._usage["user123"] = {
            "requests_per_minute": {"count": 50, "reset_time": time.time() + 30}
        }
        
        can_proceed, remaining = quota_manager.check_quota("user123", "requests_per_minute")
        
        assert can_proceed is True
        assert remaining == 50  # 100 - 50
    
    def test_check_quota_over_limit(self, quota_manager):
        """Test quota check when over limit."""
        quota_manager.set_quota("user123", "requests_per_minute", 100)
        
        # Simulate usage over limit
        quota_manager._usage["user123"] = {
            "requests_per_minute": {"count": 105, "reset_time": time.time() + 30}
        }
        
        can_proceed, remaining = quota_manager.check_quota("user123", "requests_per_minute")
        
        assert can_proceed is False
        assert remaining == 0
    
    def test_consume_quota(self, quota_manager):
        """Test consuming quota."""
        quota_manager.set_quota("user123", "requests_per_minute", 100)
        
        # First consumption
        success = quota_manager.consume_quota("user123", "requests_per_minute", 10)
        assert success is True
        
        # Check usage was recorded
        usage = quota_manager._usage["user123"]["requests_per_minute"]
        assert usage["count"] == 10
    
    def test_consume_quota_over_limit(self, quota_manager):
        """Test consuming quota over limit."""
        quota_manager.set_quota("user123", "requests_per_minute", 100)
        
        # Try to consume more than quota
        success = quota_manager.consume_quota("user123", "requests_per_minute", 150)
        assert success is False
        
        # Usage should not be recorded
        assert "user123" not in quota_manager._usage
    
    def test_quota_reset_after_period(self, quota_manager):
        """Test quota reset after time period."""
        quota_manager.set_quota("user123", "requests_per_minute", 100)
        
        # Consume some quota
        quota_manager.consume_quota("user123", "requests_per_minute", 80)
        
        # Simulate time passage by setting reset time in past
        quota_manager._usage["user123"]["requests_per_minute"]["reset_time"] = time.time() - 1
        
        # Check quota - should be reset
        can_proceed, remaining = quota_manager.check_quota("user123", "requests_per_minute")
        
        assert can_proceed is True
        assert remaining == 100  # Full quota available
    
    def test_get_quota_usage(self, quota_manager):
        """Test getting quota usage."""
        quota_manager.set_quota("user123", "requests_per_minute", 100)
        quota_manager.consume_quota("user123", "requests_per_minute", 25)
        
        usage_info = quota_manager.get_quota_usage("user123", "requests_per_minute")
        
        assert usage_info["limit"] == 100
        assert usage_info["used"] == 25
        assert usage_info["remaining"] == 75
        assert "reset_time" in usage_info
    
    def test_get_all_quotas(self, quota_manager):
        """Test getting all quotas for a user."""
        quota_manager.set_quota("user123", "requests_per_minute", 100)
        quota_manager.set_quota("user123", "memory_mb", 512)
        quota_manager.consume_quota("user123", "requests_per_minute", 20)
        
        all_quotas = quota_manager.get_all_quotas("user123")
        
        assert "requests_per_minute" in all_quotas
        assert "memory_mb" in all_quotas
        assert all_quotas["requests_per_minute"]["used"] == 20
        assert all_quotas["memory_mb"]["used"] == 0
    
    def test_remove_quota(self, quota_manager):
        """Test removing quota."""
        quota_manager.set_quota("user123", "requests_per_minute", 100)
        quota_manager.consume_quota("user123", "requests_per_minute", 50)
        
        quota_manager.remove_quota("user123", "requests_per_minute")
        
        assert "requests_per_minute" not in quota_manager._quotas.get("user123", {})
        
        # Should return True (no limit) when quota doesn't exist
        can_proceed, remaining = quota_manager.check_quota("user123", "requests_per_minute")
        assert can_proceed is True


class TestResourceManager:
    """Test resource manager functionality."""
    
    @pytest.fixture
    def resource_limits(self):
        """Create resource limits."""
        return ResourceLimits(
            max_memory_mb=1024,
            max_concurrent_requests=50,
            max_queue_size=200,
            max_request_duration_seconds=300
        )
    
    @pytest.fixture
    def resource_manager(self, resource_limits):
        """Create resource manager."""
        return ResourceManager(resource_limits)
    
    def test_resource_manager_initialization(self, resource_manager, resource_limits):
        """Test resource manager initialization."""
        assert resource_manager.limits is resource_limits
        assert isinstance(resource_manager.memory_tracker, MemoryTracker)
        assert isinstance(resource_manager.request_queue, RequestQueue)
        assert isinstance(resource_manager.quota_manager, ResourceQuotaManager)
        assert resource_manager.active_requests == 0
    
    @pytest.mark.asyncio
    async def test_can_accept_request_memory_ok(self, resource_manager):
        """Test request acceptance when memory is OK."""
        with patch.object(resource_manager.memory_tracker, 'check_memory_status') as mock_memory:
            mock_memory.return_value = {"level": "normal", "critical": False}
            
            can_accept, reason = await resource_manager.can_accept_request("user123")
            
            assert can_accept is True
            assert reason is None
    
    @pytest.mark.asyncio
    async def test_can_accept_request_memory_critical(self, resource_manager):
        """Test request rejection when memory is critical."""
        with patch.object(resource_manager.memory_tracker, 'check_memory_status') as mock_memory:
            mock_memory.return_value = {"level": "critical", "critical": True}
            
            can_accept, reason = await resource_manager.can_accept_request("user123")
            
            assert can_accept is False
            assert "memory" in reason.lower()
    
    @pytest.mark.asyncio
    async def test_can_accept_request_concurrent_limit(self, resource_manager):
        """Test request rejection when concurrent limit is reached."""
        # Mock memory as OK
        with patch.object(resource_manager.memory_tracker, 'check_memory_status') as mock_memory:
            mock_memory.return_value = {"level": "normal", "critical": False}
            
            # Set active requests to limit
            resource_manager._active_requests = 50  # At limit
            
            can_accept, reason = await resource_manager.can_accept_request("user123")
            
            assert can_accept is False
            assert "concurrent" in reason.lower()
    
    @pytest.mark.asyncio
    async def test_can_accept_request_quota_exceeded(self, resource_manager):
        """Test request rejection when user quota is exceeded."""
        with patch.object(resource_manager.memory_tracker, 'check_memory_status') as mock_memory:
            mock_memory.return_value = {"level": "normal", "critical": False}
            
            # Set user quota and consume it fully
            resource_manager.quota_manager.set_quota("user123", "requests_per_minute", 10)
            resource_manager.quota_manager.consume_quota("user123", "requests_per_minute", 10)
            
            can_accept, reason = await resource_manager.can_accept_request("user123")
            
            assert can_accept is False
            assert "quota" in reason.lower()
    
    @pytest.mark.asyncio
    async def test_acquire_resources(self, resource_manager):
        """Test acquiring resources for request."""
        with patch.object(resource_manager.memory_tracker, 'check_memory_status') as mock_memory:
            mock_memory.return_value = {"level": "normal", "critical": False}
            
            request_data = {"operation": "predict", "input": "test"}
            
            success, request_id = await resource_manager.acquire_resources("user123", request_data)
            
            assert success is True
            assert request_id is not None
            assert resource_manager.active_requests == 1
    
    @pytest.mark.asyncio
    async def test_release_resources(self, resource_manager):
        """Test releasing resources after request."""
        with patch.object(resource_manager.memory_tracker, 'check_memory_status') as mock_memory:
            mock_memory.return_value = {"level": "normal", "critical": False}
            
            # Acquire resources first
            success, request_id = await resource_manager.acquire_resources("user123", {})
            assert success is True
            
            # Release resources
            await resource_manager.release_resources(request_id, "user123")
            
            assert resource_manager.active_requests == 0
    
    @pytest.mark.asyncio
    async def test_resource_context_manager(self, resource_manager):
        """Test resource manager as context manager."""
        with patch.object(resource_manager.memory_tracker, 'check_memory_status') as mock_memory:
            mock_memory.return_value = {"level": "normal", "critical": False}
            
            async with resource_manager.acquire_for_request("user123", {}) as request_id:
                assert request_id is not None
                assert resource_manager.active_requests == 1
            
            # After context, resources should be released
            assert resource_manager.active_requests == 0
    
    @pytest.mark.asyncio
    async def test_context_manager_resource_denied(self, resource_manager):
        """Test context manager when resources are denied."""
        with patch.object(resource_manager.memory_tracker, 'check_memory_status') as mock_memory:
            mock_memory.return_value = {"level": "critical", "critical": True}
            
            with pytest.raises(Exception):  # Should raise resource unavailable exception
                async with resource_manager.acquire_for_request("user123", {}):
                    pass
    
    def test_get_resource_stats(self, resource_manager):
        """Test getting resource statistics."""
        stats = resource_manager.get_resource_stats()
        
        assert "memory" in stats
        assert "requests" in stats
        assert "queue" in stats
        
        assert "current_mb" in stats["memory"]
        assert "peak_mb" in stats["memory"]
        assert "status" in stats["memory"]
        
        assert "active" in stats["requests"]
        assert "max_concurrent" in stats["requests"]
        
        assert "size" in stats["queue"]
        assert "max_size" in stats["queue"]
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_requests(self, resource_manager):
        """Test cleanup of expired requests."""
        # Mock an expired request
        now = datetime.utcnow()
        expired_time = now - timedelta(seconds=400)  # Older than 300s limit
        
        resource_manager._active_request_times["expired-req"] = expired_time
        resource_manager._active_requests = 1
        
        cleaned_count = await resource_manager.cleanup_expired_requests()
        
        assert cleaned_count == 1
        assert resource_manager.active_requests == 0
        assert "expired-req" not in resource_manager._active_request_times
    
    @pytest.mark.asyncio
    async def test_concurrent_resource_management(self, resource_manager):
        """Test concurrent resource acquisition and release."""
        with patch.object(resource_manager.memory_tracker, 'check_memory_status') as mock_memory:
            mock_memory.return_value = {"level": "normal", "critical": False}
            
            async def acquire_and_release(user_id):
                async with resource_manager.acquire_for_request(user_id, {}):
                    await asyncio.sleep(0.1)  # Simulate work
                    return "completed"
            
            # Start multiple concurrent tasks
            tasks = []
            for i in range(10):
                task = asyncio.create_task(acquire_and_release(f"user{i}"))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # All should complete successfully
            assert all(r == "completed" for r in results)
            assert resource_manager.active_requests == 0
    
    def test_update_resource_limits(self, resource_manager):
        """Test updating resource limits."""
        new_limits = ResourceLimits(
            max_memory_mb=2048,
            max_concurrent_requests=100
        )
        
        resource_manager.update_limits(new_limits)
        
        assert resource_manager.limits.max_memory_mb == 2048
        assert resource_manager.limits.max_concurrent_requests == 100
    
    @pytest.mark.asyncio
    async def test_queue_integration(self, resource_manager):
        """Test integration with request queue."""
        with patch.object(resource_manager.memory_tracker, 'check_memory_status') as mock_memory:
            mock_memory.return_value = {"level": "normal", "critical": False}
            
            # Fill up concurrent request slots
            resource_manager._active_requests = resource_manager.limits.max_concurrent_requests
            
            # This request should go to queue
            request_data = {"operation": "predict"}
            success, request_id = await resource_manager.acquire_resources("user123", request_data)
            
            # Should succeed but be queued
            assert success is True
            assert resource_manager.request_queue.size > 0


class TestResourceManagerIntegration:
    """Test resource manager integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_resource_lifecycle(self):
        """Test complete resource management lifecycle."""
        limits = ResourceLimits(
            max_memory_mb=1024,
            max_concurrent_requests=5,
            max_queue_size=10
        )
        resource_manager = ResourceManager(limits)
        
        # Set user quotas
        resource_manager.quota_manager.set_quota("user1", "requests_per_minute", 100)
        resource_manager.quota_manager.set_quota("user2", "requests_per_minute", 50)
        
        with patch.object(resource_manager.memory_tracker, 'check_memory_status') as mock_memory:
            mock_memory.return_value = {"level": "normal", "critical": False}
            
            # Simulate multiple users making concurrent requests
            async def user_workflow(user_id, request_count):
                results = []
                for i in range(request_count):
                    try:
                        async with resource_manager.acquire_for_request(user_id, {"request": i}):
                            await asyncio.sleep(0.1)  # Simulate processing
                            results.append(f"{user_id}-request-{i}")
                    except Exception as e:
                        results.append(f"{user_id}-error-{i}")
                return results
            
            # Run concurrent workflows
            user1_task = asyncio.create_task(user_workflow("user1", 10))
            user2_task = asyncio.create_task(user_workflow("user2", 8))
            
            user1_results, user2_results = await asyncio.gather(user1_task, user2_task)
            
            # Verify results
            assert len(user1_results) == 10
            assert len(user2_results) == 8
            
            # Final state should be clean
            assert resource_manager.active_requests == 0
            assert resource_manager.request_queue.is_empty
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_and_recovery(self):
        """Test system behavior under resource exhaustion."""
        limits = ResourceLimits(
            max_memory_mb=512,
            max_concurrent_requests=2,
            max_queue_size=3
        )
        resource_manager = ResourceManager(limits)
        
        with patch.object(resource_manager.memory_tracker, 'check_memory_status') as mock_memory:
            # Start with normal memory
            mock_memory.return_value = {"level": "normal", "critical": False}
            
            # Acquire all concurrent slots
            contexts = []
            for i in range(2):
                ctx = resource_manager.acquire_for_request(f"user{i}", {})
                await ctx.__aenter__()
                contexts.append(ctx)
            
            # Fill the queue
            for i in range(3):
                success, _ = await resource_manager.acquire_resources(f"queued_user{i}", {})
                assert success is True
            
            # Next request should fail (queue full)
            success, _ = await resource_manager.acquire_resources("overflow_user", {})
            assert success is False
            
            # Simulate memory becoming critical
            mock_memory.return_value = {"level": "critical", "critical": True}
            
            can_accept, reason = await resource_manager.can_accept_request("new_user")
            assert can_accept is False
            assert "memory" in reason.lower()
            
            # Recovery: memory becomes normal and release resources
            mock_memory.return_value = {"level": "normal", "critical": False}
            
            for ctx in contexts:
                await ctx.__aexit__(None, None, None)
            
            # Should be able to accept requests again
            can_accept, reason = await resource_manager.can_accept_request("recovery_user")
            assert can_accept is True
    
    @pytest.mark.asyncio
    async def test_quota_enforcement_across_requests(self):
        """Test quota enforcement across multiple requests."""
        limits = ResourceLimits(max_concurrent_requests=10)
        resource_manager = ResourceManager(limits)
        
        # Set tight quota for testing
        resource_manager.quota_manager.set_quota("limited_user", "requests_per_minute", 3)
        
        with patch.object(resource_manager.memory_tracker, 'check_memory_status') as mock_memory:
            mock_memory.return_value = {"level": "normal", "critical": False}
            
            successful_requests = 0
            rejected_requests = 0
            
            # Try to make 5 requests (should only allow 3)
            for i in range(5):
                try:
                    async with resource_manager.acquire_for_request("limited_user", {}):
                        successful_requests += 1
                        await asyncio.sleep(0.01)
                except Exception:
                    rejected_requests += 1
            
            assert successful_requests == 3
            assert rejected_requests == 2
    
    def test_monitoring_and_alerting_integration(self):
        """Test resource manager monitoring capabilities."""
        limits = ResourceLimits(max_memory_mb=1024, max_concurrent_requests=50)
        resource_manager = ResourceManager(limits)
        
        # Simulate various resource states
        resource_manager._active_requests = 45  # Near limit
        
        with patch.object(resource_manager.memory_tracker, 'check_memory_status') as mock_memory:
            mock_memory.return_value = {
                "level": "warning", 
                "critical": False, 
                "current_mb": 800,
                "warning": True
            }
            
            stats = resource_manager.get_resource_stats()
            
            # Should indicate resource pressure
            assert stats["requests"]["active"] == 45
            assert stats["requests"]["utilization"] == 0.9  # 45/50
            assert stats["memory"]["status"]["warning"] is True
            
            # Could trigger alerts based on these stats
            alerts_needed = []
            if stats["requests"]["utilization"] > 0.8:
                alerts_needed.append("high_request_load")
            if stats["memory"]["status"]["warning"]:
                alerts_needed.append("memory_pressure")
            
            assert "high_request_load" in alerts_needed
            assert "memory_pressure" in alerts_needed

"""
Tests for Graceful Shutdown implementation.
"""

import pytest
import asyncio
import signal
import threading
import time
from unittest.mock import Mock, patch, AsyncMock, call
from datetime import datetime, timedelta

from framework.reliability.graceful_shutdown import (
    GracefulShutdown, ShutdownPhase, ActiveRequest, ShutdownState
)


class TestActiveRequest:
    """Test active request tracking."""
    
    def test_active_request_creation(self):
        """Test creating an active request."""
        request = ActiveRequest("req-123", "/api/predict")
        
        assert request.request_id == "req-123"
        assert request.endpoint == "/api/predict"
        assert isinstance(request.start_time, datetime)
        assert not request.is_finished
    
    def test_request_finish(self):
        """Test finishing a request."""
        request = ActiveRequest("req-123", "/api/predict")
        request.finish()
        
        assert request.is_finished
        assert isinstance(request.end_time, datetime)
        assert request.end_time >= request.start_time
    
    def test_request_duration(self):
        """Test request duration calculation."""
        request = ActiveRequest("req-123", "/api/predict")
        time.sleep(0.01)  # Small delay
        request.finish()
        
        duration = request.duration
        assert duration.total_seconds() > 0
        assert duration.total_seconds() < 1  # Should be very small
    
    def test_request_duration_unfinished(self):
        """Test duration calculation for unfinished request."""
        request = ActiveRequest("req-123", "/api/predict")
        time.sleep(0.01)
        
        duration = request.duration
        assert duration.total_seconds() > 0


class TestGracefulShutdown:
    """Test graceful shutdown functionality."""
    
    @pytest.fixture
    def shutdown_manager(self):
        """Create graceful shutdown manager."""
        from framework.reliability.graceful_shutdown import ShutdownConfig
        config = ShutdownConfig(
            graceful_timeout=2.0,
            cleanup_timeout=1.0
        )
        return GracefulShutdown(config=config)
    
    def test_initial_state(self, shutdown_manager):
        """Test initial shutdown manager state."""
        assert shutdown_manager._state == ShutdownState.RUNNING
        assert shutdown_manager.config.graceful_timeout == 2.0
        assert shutdown_manager.config.cleanup_timeout == 1.0
        assert len(shutdown_manager._active_requests) == 0
        assert len(shutdown_manager._shutdown_hooks) == 0
    
    def test_register_cleanup_task(self, shutdown_manager):
        """Test registering cleanup tasks."""
        async def cleanup_task():
            pass
        
        shutdown_manager.register_cleanup_task("test_cleanup", cleanup_task)
        
        assert "test_cleanup" in shutdown_manager.cleanup_tasks
        assert shutdown_manager.cleanup_tasks["test_cleanup"] is cleanup_task
    
    def test_register_duplicate_cleanup_task(self, shutdown_manager):
        """Test registering duplicate cleanup task."""
        async def cleanup_task():
            pass
        
        shutdown_manager.register_cleanup_task("test", cleanup_task)
        
        with pytest.raises(ValueError, match="already registered"):
            shutdown_manager.register_cleanup_task("test", cleanup_task)
    
    def test_add_active_request(self, shutdown_manager):
        """Test adding active request."""
        shutdown_manager.add_request("req-123", "/api/predict")
        
        assert "req-123" in shutdown_manager.active_requests
        request = shutdown_manager.active_requests["req-123"]
        assert request.endpoint == "/api/predict"
        assert not request.is_finished
    
    def test_remove_active_request(self, shutdown_manager):
        """Test removing active request."""
        shutdown_manager.add_request("req-123", "/api/predict")
        shutdown_manager.remove_request("req-123")
        
        assert "req-123" not in shutdown_manager.active_requests
    
    def test_remove_nonexistent_request(self, shutdown_manager):
        """Test removing non-existent request (should not raise)."""
        # Should not raise an exception
        shutdown_manager.remove_request("nonexistent")
    
    def test_get_active_request_count(self, shutdown_manager):
        """Test getting active request count."""
        assert shutdown_manager.get_active_request_count() == 0
        
        shutdown_manager.add_request("req-1", "/api/predict")
        shutdown_manager.add_request("req-2", "/api/health")
        
        assert shutdown_manager.get_active_request_count() == 2
    
    def test_get_active_requests_info(self, shutdown_manager):
        """Test getting active requests information."""
        shutdown_manager.add_request("req-1", "/api/predict")
        shutdown_manager.add_request("req-2", "/api/health")
        
        info = shutdown_manager.get_active_requests_info()
        
        assert len(info) == 2
        assert any(req["request_id"] == "req-1" for req in info)
        assert any(req["request_id"] == "req-2" for req in info)
        
        for req in info:
            assert "endpoint" in req
            assert "duration_seconds" in req
    
    @pytest.mark.asyncio
    async def test_signal_handler_setup(self, shutdown_manager):
        """Test signal handler setup."""
        with patch('signal.signal') as mock_signal:
            shutdown_manager.setup_signal_handlers()
            
            # Should register handlers for SIGTERM and SIGINT
            expected_calls = [
                call(signal.SIGTERM, shutdown_manager._signal_handler),
                call(signal.SIGINT, shutdown_manager._signal_handler)
            ]
            mock_signal.assert_has_calls(expected_calls, any_order=True)
    
    def test_signal_handler_triggers_shutdown(self, shutdown_manager):
        """Test signal handler triggers shutdown."""
        # Mock the shutdown method
        shutdown_manager.shutdown = Mock()
        
        # Simulate signal
        shutdown_manager._signal_handler(signal.SIGTERM, None)
        
        shutdown_manager.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_no_active_requests(self, shutdown_manager):
        """Test shutdown with no active requests."""
        cleanup_called = asyncio.Event()
        
        async def cleanup_task():
            cleanup_called.set()
        
        shutdown_manager.register_cleanup_task("test_cleanup", cleanup_task)
        
        # Start shutdown
        shutdown_task = asyncio.create_task(shutdown_manager.shutdown())
        
        # Wait a bit to ensure shutdown starts
        await asyncio.sleep(0.1)
        
        # Should transition to cleanup immediately
        assert shutdown_manager.phase == ShutdownPhase.CLEANUP
        
        # Wait for shutdown to complete
        await shutdown_task
        
        assert shutdown_manager.phase == ShutdownPhase.SHUTDOWN_COMPLETE
        assert cleanup_called.is_set()
    
    @pytest.mark.asyncio
    async def test_shutdown_with_active_requests(self, shutdown_manager):
        """Test shutdown waits for active requests."""
        # Add active request
        shutdown_manager.add_request("req-123", "/api/predict")
        
        # Start shutdown
        shutdown_task = asyncio.create_task(shutdown_manager.shutdown())
        
        # Wait a bit to ensure shutdown starts
        await asyncio.sleep(0.1)
        assert shutdown_manager.phase == ShutdownPhase.DRAINING
        
        # Remove request to allow shutdown to proceed
        shutdown_manager.remove_request("req-123")
        
        # Wait for shutdown to complete
        await shutdown_task
        
        assert shutdown_manager.phase == ShutdownPhase.SHUTDOWN_COMPLETE
    
    @pytest.mark.asyncio
    async def test_shutdown_timeout(self, shutdown_manager):
        """Test shutdown timeout with hanging requests."""
        # Add request that won't be removed
        shutdown_manager.add_request("hanging-req", "/api/predict")
        
        start_time = time.time()
        await shutdown_manager.shutdown()
        elapsed = time.time() - start_time
        
        # Should timeout after shutdown_timeout (2.0 seconds)
        assert 1.8 <= elapsed <= 2.5  # Allow some tolerance
        assert shutdown_manager.phase == ShutdownPhase.SHUTDOWN_COMPLETE
    
    @pytest.mark.asyncio
    async def test_cleanup_tasks_execution(self, shutdown_manager):
        """Test cleanup tasks are executed."""
        executed_tasks = []
        
        async def cleanup1():
            executed_tasks.append("cleanup1")
        
        async def cleanup2():
            executed_tasks.append("cleanup2")
        
        shutdown_manager.register_cleanup_task("cleanup1", cleanup1)
        shutdown_manager.register_cleanup_task("cleanup2", cleanup2)
        
        await shutdown_manager.shutdown()
        
        assert "cleanup1" in executed_tasks
        assert "cleanup2" in executed_tasks
    
    @pytest.mark.asyncio
    async def test_cleanup_task_exception_handling(self, shutdown_manager):
        """Test cleanup tasks exception handling."""
        executed_tasks = []
        
        async def failing_cleanup():
            raise Exception("Cleanup failed")
        
        async def normal_cleanup():
            executed_tasks.append("normal")
        
        shutdown_manager.register_cleanup_task("failing", failing_cleanup)
        shutdown_manager.register_cleanup_task("normal", normal_cleanup)
        
        # Should not raise exception, but complete shutdown
        await shutdown_manager.shutdown()
        
        # Normal cleanup should still execute
        assert "normal" in executed_tasks
        assert shutdown_manager.phase == ShutdownPhase.SHUTDOWN_COMPLETE
    
    @pytest.mark.asyncio
    async def test_cleanup_timeout(self, shutdown_manager):
        """Test cleanup timeout."""
        async def slow_cleanup():
            await asyncio.sleep(2.0)  # Longer than cleanup_timeout (1.0)
        
        shutdown_manager.register_cleanup_task("slow", slow_cleanup)
        
        start_time = time.time()
        await shutdown_manager.shutdown()
        elapsed = time.time() - start_time
        
        # Should timeout during cleanup phase
        assert 0.8 <= elapsed <= 1.5  # Allow some tolerance
        assert shutdown_manager.phase == ShutdownPhase.SHUTDOWN_COMPLETE
    
    def test_is_shutting_down(self, shutdown_manager):
        """Test is_shutting_down property."""
        assert not shutdown_manager.is_shutting_down
        
        shutdown_manager._phase = ShutdownPhase.DRAINING
        assert shutdown_manager.is_shutting_down
        
        shutdown_manager._phase = ShutdownPhase.CLEANUP
        assert shutdown_manager.is_shutting_down
        
        shutdown_manager._phase = ShutdownPhase.SHUTDOWN_COMPLETE
        assert not shutdown_manager.is_shutting_down
    
    def test_should_accept_requests(self, shutdown_manager):
        """Test should_accept_requests property."""
        assert shutdown_manager.should_accept_requests
        
        shutdown_manager._phase = ShutdownPhase.DRAINING
        assert not shutdown_manager.should_accept_requests
        
        shutdown_manager._phase = ShutdownPhase.CLEANUP
        assert not shutdown_manager.should_accept_requests
    
    @pytest.mark.asyncio
    async def test_wait_for_shutdown(self, shutdown_manager):
        """Test waiting for shutdown completion."""
        # Start shutdown in background
        shutdown_task = asyncio.create_task(shutdown_manager.shutdown())
        
        # Wait for shutdown to complete
        await shutdown_manager.wait_for_shutdown()
        
        assert shutdown_manager.phase == ShutdownPhase.SHUTDOWN_COMPLETE
        
        # Clean up
        await shutdown_task
    
    @pytest.mark.asyncio
    async def test_multiple_shutdown_calls(self, shutdown_manager):
        """Test multiple shutdown calls (should be idempotent)."""
        # Start first shutdown
        shutdown1 = asyncio.create_task(shutdown_manager.shutdown())
        
        # Start second shutdown (should not interfere)
        shutdown2 = asyncio.create_task(shutdown_manager.shutdown())
        
        # Both should complete successfully
        await shutdown1
        await shutdown2
        
        assert shutdown_manager.phase == ShutdownPhase.SHUTDOWN_COMPLETE


class TestGracefulShutdownIntegration:
    """Test graceful shutdown integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_fastapi_integration_simulation(self):
        """Test simulated FastAPI integration."""
        shutdown_manager = GracefulShutdown()
        
        # Simulate middleware that tracks requests
        async def simulate_request(request_id: str, duration: float):
            shutdown_manager.add_request(request_id, "/api/predict")
            try:
                await asyncio.sleep(duration)
                return f"Response for {request_id}"
            finally:
                shutdown_manager.remove_request(request_id)
        
        # Start several requests
        request_tasks = [
            asyncio.create_task(simulate_request("req-1", 0.5)),
            asyncio.create_task(simulate_request("req-2", 1.0)),
            asyncio.create_task(simulate_request("req-3", 0.3))
        ]
        
        # Let requests start
        await asyncio.sleep(0.1)
        
        # Start shutdown
        shutdown_task = asyncio.create_task(shutdown_manager.shutdown())
        
        # Wait for all requests to complete
        results = await asyncio.gather(*request_tasks)
        
        # Wait for shutdown to complete
        await shutdown_task
        
        assert len(results) == 3
        assert shutdown_manager.phase == ShutdownPhase.SHUTDOWN_COMPLETE
        assert shutdown_manager.get_active_request_count() == 0
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_simulation(self):
        """Test resource cleanup simulation."""
        shutdown_manager = GracefulShutdown()
        
        # Simulate database connections
        db_closed = asyncio.Event()
        cache_closed = asyncio.Event()
        
        async def close_database():
            await asyncio.sleep(0.1)  # Simulate cleanup time
            db_closed.set()
        
        async def close_cache():
            await asyncio.sleep(0.2)  # Simulate cleanup time
            cache_closed.set()
        
        shutdown_manager.register_cleanup_task("database", close_database)
        shutdown_manager.register_cleanup_task("cache", close_cache)
        
        await shutdown_manager.shutdown()
        
        assert db_closed.is_set()
        assert cache_closed.is_set()
        assert shutdown_manager.phase == ShutdownPhase.SHUTDOWN_COMPLETE
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_and_shutdown(self):
        """Test concurrent requests during shutdown."""
        shutdown_manager = GracefulShutdown(shutdown_timeout=1.0)
        
        async def long_request(request_id: str):
            shutdown_manager.add_request(request_id, "/api/predict")
            try:
                await asyncio.sleep(2.0)  # Longer than shutdown timeout
                return f"Completed {request_id}"
            except asyncio.CancelledError:
                return f"Cancelled {request_id}"
            finally:
                shutdown_manager.remove_request(request_id)
        
        # Start long-running request
        request_task = asyncio.create_task(long_request("long-req"))
        
        # Let request start
        await asyncio.sleep(0.1)
        
        # Start shutdown (should timeout)
        start_time = time.time()
        await shutdown_manager.shutdown()
        elapsed = time.time() - start_time
        
        # Should timeout after shutdown_timeout
        assert 0.8 <= elapsed <= 1.5
        assert shutdown_manager.phase == ShutdownPhase.SHUTDOWN_COMPLETE
        
        # Clean up request task
        request_task.cancel()
        try:
            await request_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_request_tracking_accuracy(self):
        """Test request tracking accuracy under load."""
        shutdown_manager = GracefulShutdown()
        
        async def tracked_request(request_id: str, duration: float):
            shutdown_manager.add_request(request_id, f"/api/{request_id}")
            try:
                await asyncio.sleep(duration)
                return request_id
            finally:
                shutdown_manager.remove_request(request_id)
        
        # Start many concurrent requests
        tasks = []
        for i in range(10):
            task = asyncio.create_task(
                tracked_request(f"req-{i}", 0.1 + (i * 0.05))
            )
            tasks.append(task)
        
        # Check request count during execution
        await asyncio.sleep(0.05)  # Let some requests start
        active_count = shutdown_manager.get_active_request_count()
        assert active_count > 0
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert shutdown_manager.get_active_request_count() == 0
    
    def test_thread_safety(self):
        """Test thread safety of request tracking."""
        shutdown_manager = GracefulShutdown()
        
        def add_remove_requests(thread_id: int):
            for i in range(100):
                request_id = f"thread-{thread_id}-req-{i}"
                shutdown_manager.add_request(request_id, "/api/test")
                time.sleep(0.001)  # Small delay
                shutdown_manager.remove_request(request_id)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_remove_requests, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have no active requests
        assert shutdown_manager.get_active_request_count() == 0


class TestShutdownPhases:
    """Test shutdown phase transitions."""
    
    def test_phase_enum_values(self):
        """Test shutdown phase enum values."""
        assert ShutdownPhase.RUNNING == "running"
        assert ShutdownPhase.DRAINING == "draining"
        assert ShutdownPhase.CLEANUP == "cleanup"
        assert ShutdownPhase.SHUTDOWN_COMPLETE == "shutdown_complete"
    
    def test_phase_ordering(self):
        """Test logical ordering of phases."""
        phases = [
            ShutdownPhase.RUNNING,
            ShutdownPhase.DRAINING,
            ShutdownPhase.CLEANUP,
            ShutdownPhase.SHUTDOWN_COMPLETE
        ]
        
        # This test ensures the phases follow logical progression
        assert len(phases) == 4
        assert phases[0] == ShutdownPhase.RUNNING
        assert phases[-1] == ShutdownPhase.SHUTDOWN_COMPLETE

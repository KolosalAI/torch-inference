"""
Tests for Circuit Breaker Pattern implementation.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta

from framework.reliability.circuit_breaker import (
    CircuitBreaker, CircuitBreakerState, CircuitBreakerConfig,
    get_circuit_breaker
)


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.success_threshold == 3
        assert config.timeout == 30.0


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,
            success_threshold=2,
            timeout=0.5
        )
        return CircuitBreaker("test_service", config)
    
    @pytest.fixture
    def mock_operation(self):
        """Create a mock operation."""
        return AsyncMock()
    
    def test_initial_state(self, circuit_breaker):
        """Test circuit breaker initial state."""
        assert circuit_breaker._state == CircuitBreakerState.CLOSED
        assert circuit_breaker._failure_count == 0
        assert circuit_breaker._success_count == 0
        assert circuit_breaker._last_failure_time == 0.0
    
    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker, mock_operation):
        """Test successful operation call."""
        mock_operation.return_value = "success"
        
        result = await circuit_breaker.call(mock_operation)
        
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker._failure_count == 0
        mock_operation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_failed_call(self, circuit_breaker, mock_operation):
        """Test failed operation call."""
        mock_operation.side_effect = Exception("Test error")
        
        with pytest.raises(Exception, match="Test error"):
            await circuit_breaker.call(mock_operation)
        
        assert circuit_breaker._failure_count == 1
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        mock_operation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, circuit_breaker, mock_operation):
        """Test circuit opens after failure threshold."""
        mock_operation.side_effect = Exception("Test error")
        
        # Fail enough times to open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(mock_operation)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker._failure_count == 3
    
    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self, circuit_breaker, mock_operation):
        """Test open circuit rejects calls immediately."""
        # Force circuit to open
        circuit_breaker._state = CircuitBreakerState.OPEN
        circuit_breaker._last_failure_time = datetime.utcnow()
        
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await circuit_breaker.call(mock_operation)
        
        # Operation should not have been called
        mock_operation.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self, circuit_breaker, mock_operation):
        """Test circuit transitions from open to half-open after timeout."""
        # Force circuit to open with old failure time
        circuit_breaker._state = CircuitBreakerState.OPEN
        circuit_breaker._last_failure_time = datetime.utcnow() - timedelta(seconds=2)
        
        mock_operation.return_value = "success"
        
        result = await circuit_breaker.call(mock_operation)
        
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN
        mock_operation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_half_open_closes_on_success(self, circuit_breaker, mock_operation):
        """Test half-open circuit closes after successful calls."""
        # Set to half-open state
        circuit_breaker._state = CircuitBreakerState.HALF_OPEN
        mock_operation.return_value = "success"
        
        # Make successful calls to close circuit
        for i in range(2):  # success_threshold = 2
            await circuit_breaker.call(mock_operation)
        
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker._failure_count == 0
        assert circuit_breaker._success_count == 0  # Reset after closing
    
    @pytest.mark.asyncio
    async def test_half_open_reopens_on_failure(self, circuit_breaker, mock_operation):
        """Test half-open circuit reopens on failure."""
        # Set to half-open state
        circuit_breaker._state = CircuitBreakerState.HALF_OPEN
        mock_operation.side_effect = Exception("Test error")
        
        with pytest.raises(Exception):
            await circuit_breaker.call(mock_operation)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker._failure_count == 1
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, circuit_breaker):
        """Test operation timeout handling."""
        async def slow_operation():
            await asyncio.sleep(1.0)  # Longer than timeout (0.5s)
            return "success"
        
        with pytest.raises(asyncio.TimeoutError):
            await circuit_breaker.call(slow_operation)
        
        assert circuit_breaker._failure_count == 1
    
    @pytest.mark.asyncio
    async def test_half_open_max_calls_limit(self, circuit_breaker, mock_operation):
        """Test half-open state respects max calls limit."""
        # Set to half-open state
        circuit_breaker._state = CircuitBreakerState.HALF_OPEN
        circuit_breaker._half_open_calls = 2  # At max limit
        
        with pytest.raises(Exception, match="Too many calls in HALF_OPEN state"):
            await circuit_breaker.call(mock_operation)
        
        mock_operation.assert_not_called()
    
    def test_get_stats(self, circuit_breaker):
        """Test circuit breaker statistics."""
        stats = circuit_breaker.get_stats()
        
        assert "state" in stats
        assert "failure_count" in stats
        assert "success_count" in stats
        assert "total_calls" in stats
        assert "last_failure_time" in stats
        assert "config" in stats
    
    @pytest.mark.asyncio
    async def test_reset_functionality(self, circuit_breaker, mock_operation):
        """Test circuit breaker reset functionality."""
        # Cause some failures
        mock_operation.side_effect = Exception("Test error")
        
        for i in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(mock_operation)
        
        # Reset circuit breaker
        circuit_breaker.reset()
        
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker._failure_count == 0
        assert circuit_breaker._success_count == 0
        assert circuit_breaker.last_failure_time is None


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration and edge cases."""
    
    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        """Test circuit breaker with concurrent calls."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
        circuit_breaker = CircuitBreaker("concurrent_test", config)
        
        async def operation(should_fail=False):
            if should_fail:
                raise Exception("Concurrent failure")
            return "success"
        
        # Test concurrent successful calls
        tasks = [circuit_breaker.call(operation) for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert all(r == "success" for r in results)
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_rapid_state_transitions(self):
        """Test rapid state transitions."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.01,  # Very short timeout
            success_threshold=1
        )
        circuit_breaker = CircuitBreaker("rapid_test", config)
        
        # Fail to open circuit
        with pytest.raises(Exception):
            await circuit_breaker.call(AsyncMock(side_effect=Exception("fail")))
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.02)
        
        # Success call should close circuit
        result = await circuit_breaker.call(AsyncMock(return_value="success"))
        
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
    
    def test_global_circuit_breaker(self):
        """Test global circuit breaker functionality."""
        cb1 = get_circuit_breaker("service1")
        cb2 = get_circuit_breaker("service1")  # Same service
        cb3 = get_circuit_breaker("service2")  # Different service
        
        assert cb1 is cb2  # Should be the same instance
        assert cb1 is not cb3  # Should be different instances
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, circuit_breaker, mock_operation):
        """Test metrics collection during operation."""
        # Successful calls
        mock_operation.return_value = "success"
        for i in range(3):
            await circuit_breaker.call(mock_operation)
        
        # Failed calls
        mock_operation.side_effect = Exception("error")
        for i in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(mock_operation)
        
        stats = circuit_breaker.get_stats()
        assert stats["total_calls"] == 5
        assert stats["failure_count"] == 2
    
    @pytest.mark.asyncio
    async def test_exception_types(self, circuit_breaker):
        """Test different exception types handling."""
        # Test with different exception types
        exceptions = [
            ValueError("value error"),
            RuntimeError("runtime error"),
            asyncio.TimeoutError("timeout"),
            ConnectionError("connection error")
        ]
        
        for exc in exceptions:
            with pytest.raises(type(exc)):
                await circuit_breaker.call(AsyncMock(side_effect=exc))
        
        # All should count as failures
        assert circuit_breaker._failure_count == len(exceptions)


@pytest.fixture
def cleanup_global_circuit_breakers():
    """Cleanup global circuit breakers after tests."""
    yield
    # Clear global circuit breakers
    from framework.reliability.circuit_breaker import _circuit_breakers
    _circuit_breakers.clear()


class TestCircuitBreakerConfiguration:
    """Test circuit breaker configuration scenarios."""
    
    def test_invalid_configuration(self):
        """Test invalid configuration values."""
        with pytest.raises(ValueError):
            CircuitBreakerConfig(failure_threshold=0)
        
        with pytest.raises(ValueError):
            CircuitBreakerConfig(recovery_timeout=-1)
        
        with pytest.raises(ValueError):
            CircuitBreakerConfig(success_threshold=0)
    
    def test_configuration_serialization(self):
        """Test configuration can be serialized."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=120.0
        )
        
        # Test that config can be converted to dict (for stats)
        config_dict = {
            "failure_threshold": config.failure_threshold,
            "recovery_timeout": config.recovery_timeout,
            "success_threshold": config.success_threshold,
            "timeout": config.timeout,
            "half_open_max_calls": config.half_open_max_calls
        }
        
        assert config_dict["failure_threshold"] == 10
        assert config_dict["recovery_timeout"] == 120.0

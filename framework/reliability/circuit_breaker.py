"""
Circuit Breaker Pattern Implementation

Provides circuit breaker functionality to prevent cascade failures in model inference.
Supports multiple failure detection modes and automatic recovery.
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"           # Normal operation
    OPEN = "open"              # Circuit is open, blocking requests
    HALF_OPEN = "half_open"    # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5                    # Number of failures before opening
    recovery_timeout: float = 60.0               # Time to wait before half-open (seconds)
    success_threshold: int = 3                   # Successes needed to close from half-open
    timeout: float = 30.0                       # Request timeout (seconds)
    expected_exception: tuple = (Exception,)     # Exceptions that count as failures
    
    # Advanced configuration
    failure_rate_threshold: float = 0.5         # Failure rate threshold (0.0-1.0)
    minimum_requests: int = 10                  # Minimum requests before calculating failure rate
    sliding_window_size: int = 100              # Size of sliding window for failure tracking
    exponential_backoff: bool = True            # Use exponential backoff for recovery
    max_recovery_timeout: float = 300.0         # Maximum recovery timeout
    
    # Monitoring configuration
    enable_metrics: bool = True                 # Enable metrics collection
    metrics_retention_seconds: float = 3600.0  # How long to retain metrics


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    circuit_opened_count: int = 0
    circuit_closed_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    current_failure_rate: float = 0.0
    average_response_time: float = 0.0
    
    # Sliding window for recent requests
    recent_requests: List[bool] = field(default_factory=list)
    request_times: List[float] = field(default_factory=list)


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    def __init__(self, message: str, state: CircuitBreakerState, metrics: CircuitBreakerMetrics):
        super().__init__(message)
        self.state = state
        self.metrics = metrics


class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascade failures.
    
    Features:
    - Configurable failure thresholds and recovery timeouts
    - Sliding window failure rate calculation
    - Exponential backoff recovery
    - Comprehensive metrics collection
    - Thread-safe operation
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self._state = CircuitBreakerState.CLOSED
        self._last_failure_time = 0.0
        self._failure_count = 0
        self._success_count = 0
        self._recovery_timeout = self.config.recovery_timeout
        
        # Metrics
        self.metrics = CircuitBreakerMetrics()
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Circuit breaker '{name}' initialized with config: {self.config}")
    
    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state
    
    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        with self._lock:
            if len(self.metrics.recent_requests) < self.config.minimum_requests:
                return 0.0
            
            failures = sum(1 for success in self.metrics.recent_requests if not success)
            return failures / len(self.metrics.recent_requests)
    
    def _update_metrics(self, success: bool, response_time: float):
        """Update circuit breaker metrics."""
        if not self.config.enable_metrics:
            return
        
        with self._lock:
            current_time = time.time()
            
            # Update basic counters
            self.metrics.total_requests += 1
            if success:
                self.metrics.successful_requests += 1
                self.metrics.last_success_time = current_time
            else:
                self.metrics.failed_requests += 1
                self.metrics.last_failure_time = current_time
            
            # Update sliding window
            self.metrics.recent_requests.append(success)
            self.metrics.request_times.append(response_time)
            
            # Maintain sliding window size
            if len(self.metrics.recent_requests) > self.config.sliding_window_size:
                self.metrics.recent_requests.pop(0)
                self.metrics.request_times.pop(0)
            
            # Update failure rate
            self.metrics.current_failure_rate = self.failure_rate
            
            # Update average response time
            if self.metrics.request_times:
                self.metrics.average_response_time = sum(self.metrics.request_times) / len(self.metrics.request_times)
            
            # Clean old metrics
            self._clean_old_metrics(current_time)
    
    def _clean_old_metrics(self, current_time: float):
        """Clean old metrics based on retention policy."""
        retention_cutoff = current_time - self.config.metrics_retention_seconds
        
        # This is a simplified implementation
        # In a production system, you might want to keep more detailed historical data
        if (self.metrics.last_failure_time and 
            self.metrics.last_failure_time < retention_cutoff):
            # Reset failure counts for old failures
            pass
    
    def _should_attempt_call(self) -> bool:
        """Check if call should be attempted based on circuit breaker state."""
        with self._lock:
            current_time = time.time()
            
            if self._state == CircuitBreakerState.CLOSED:
                return True
            
            elif self._state == CircuitBreakerState.OPEN:
                # Check if recovery timeout has elapsed
                if current_time - self._last_failure_time >= self._recovery_timeout:
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                    return True
                return False
            
            elif self._state == CircuitBreakerState.HALF_OPEN:
                return True
            
            return False
    
    def _on_success(self, response_time: float):
        """Handle successful request."""
        with self._lock:
            self._update_metrics(True, response_time)
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    self._recovery_timeout = self.config.recovery_timeout  # Reset backoff
                    self.metrics.circuit_closed_count += 1
                    logger.info(f"Circuit breaker '{self.name}' CLOSED after {self._success_count} successes")
            
            elif self._state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                if self._failure_count > 0:
                    self._failure_count = max(0, self._failure_count - 1)
    
    def _on_failure(self, response_time: float):
        """Handle failed request."""
        with self._lock:
            self._update_metrics(False, response_time)
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            # Check if we should open the circuit
            should_open = False
            
            # Threshold-based opening
            if self._failure_count >= self.config.failure_threshold:
                should_open = True
            
            # Rate-based opening (if we have enough requests)
            if (len(self.metrics.recent_requests) >= self.config.minimum_requests and
                self.failure_rate >= self.config.failure_rate_threshold):
                should_open = True
            
            if should_open and self._state != CircuitBreakerState.OPEN:
                self._state = CircuitBreakerState.OPEN
                self.metrics.circuit_opened_count += 1
                
                # Apply exponential backoff
                if self.config.exponential_backoff:
                    self._recovery_timeout = min(
                        self._recovery_timeout * 2,
                        self.config.max_recovery_timeout
                    )
                
                logger.warning(f"Circuit breaker '{self.name}' OPENED after {self._failure_count} failures "
                             f"(failure_rate: {self.failure_rate:.2f}, recovery_timeout: {self._recovery_timeout:.1f}s)")
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute an async function with circuit breaker protection."""
        if not self._should_attempt_call():
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is OPEN",
                self._state,
                self.metrics
            )
        
        start_time = time.time()
        try:
            # Apply timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            response_time = time.time() - start_time
            self._on_success(response_time)
            return result
            
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            self.metrics.timeouts += 1
            self._on_failure(response_time)
            raise
            
        except self.config.expected_exception as e:
            response_time = time.time() - start_time
            self._on_failure(response_time)
            raise
    
    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a sync function with circuit breaker protection."""
        if not self._should_attempt_call():
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is OPEN",
                self._state,
                self.metrics
            )
        
        start_time = time.time()
        try:
            # For sync functions, we can't easily implement timeout
            # without using threading, so we'll skip timeout for now
            result = func(*args, **kwargs)
            
            response_time = time.time() - start_time
            
            # Check if it took too long (manual timeout check)
            if response_time > self.config.timeout:
                self.metrics.timeouts += 1
                self._on_failure(response_time)
                raise CircuitBreakerError(
                    f"Function execution exceeded timeout ({response_time:.2f}s > {self.config.timeout}s)",
                    self._state,
                    self.metrics
                )
            
            self._on_success(response_time)
            return result
            
        except self.config.expected_exception as e:
            response_time = time.time() - start_time
            self._on_failure(response_time)
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker metrics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "recovery_timeout": self._recovery_timeout,
                "metrics": {
                    "total_requests": self.metrics.total_requests,
                    "successful_requests": self.metrics.successful_requests,
                    "failed_requests": self.metrics.failed_requests,
                    "timeouts": self.metrics.timeouts,
                    "circuit_opened_count": self.metrics.circuit_opened_count,
                    "circuit_closed_count": self.metrics.circuit_closed_count,
                    "current_failure_rate": self.metrics.current_failure_rate,
                    "average_response_time": self.metrics.average_response_time,
                    "last_failure_time": self.metrics.last_failure_time,
                    "last_success_time": self.metrics.last_success_time,
                },
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold,
                    "timeout": self.config.timeout,
                    "failure_rate_threshold": self.config.failure_rate_threshold,
                    "minimum_requests": self.config.minimum_requests,
                }
            }
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._recovery_timeout = self.config.recovery_timeout
            self.metrics = CircuitBreakerMetrics()
            logger.info(f"Circuit breaker '{self.name}' reset to initial state")


# Decorator for easy circuit breaker integration
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker protection to functions."""
    def decorator(func):
        cb = CircuitBreaker(name, config)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await cb.call_async(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return cb.call_sync(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper
        
        # Attach circuit breaker to wrapper for access to metrics
        wrapper.circuit_breaker = cb
        return wrapper
    
    return decorator


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers.
    
    Provides centralized management, monitoring, and configuration
    of multiple circuit breakers across the application.
    """
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        logger.info("Circuit breaker manager initialized")
    
    def create_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Create a new circuit breaker."""
        with self._lock:
            if name in self._breakers:
                logger.warning(f"Circuit breaker '{name}' already exists, returning existing instance")
                return self._breakers[name]
            
            breaker = CircuitBreaker(name, config)
            self._breakers[name] = breaker
            logger.info(f"Created circuit breaker '{name}'")
            return breaker
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get an existing circuit breaker."""
        with self._lock:
            return self._breakers.get(name)
    
    def remove_breaker(self, name: str) -> bool:
        """Remove a circuit breaker."""
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                logger.info(f"Removed circuit breaker '{name}'")
                return True
            return False
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all circuit breakers."""
        with self._lock:
            return {
                "circuit_breakers": {
                    name: breaker.get_metrics()
                    for name, breaker in self._breakers.items()
                },
                "total_breakers": len(self._breakers),
                "breaker_states": {
                    state.value: sum(1 for breaker in self._breakers.values() if breaker.state == state)
                    for state in CircuitBreakerState
                }
            }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("Reset all circuit breakers")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all circuit breakers."""
        with self._lock:
            healthy_count = sum(1 for breaker in self._breakers.values() 
                              if breaker.state == CircuitBreakerState.CLOSED)
            total_count = len(self._breakers)
            
            return {
                "healthy": healthy_count == total_count,
                "total_breakers": total_count,
                "healthy_breakers": healthy_count,
                "unhealthy_breakers": total_count - healthy_count,
                "health_percentage": (healthy_count / total_count * 100) if total_count > 0 else 100,
                "breaker_status": {
                    name: {
                        "state": breaker.state.value,
                        "healthy": breaker.state == CircuitBreakerState.CLOSED,
                        "failure_rate": breaker.failure_rate
                    }
                    for name, breaker in self._breakers.items()
                }
            }


# Global circuit breaker manager instance
_circuit_breaker_manager = CircuitBreakerManager()


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager instance."""
    return _circuit_breaker_manager


def create_model_circuit_breaker(model_name: str) -> CircuitBreaker:
    """Create a circuit breaker specifically configured for model inference."""
    config = CircuitBreakerConfig(
        failure_threshold=3,           # Open after 3 failures
        recovery_timeout=30.0,         # Wait 30s before retry
        success_threshold=2,           # Close after 2 successes
        timeout=30.0,                 # 30s timeout for inference
        failure_rate_threshold=0.6,   # Open if >60% failure rate
        minimum_requests=5,           # Need at least 5 requests
        exponential_backoff=True,     # Use exponential backoff
        max_recovery_timeout=300.0,   # Max 5min backoff
    )
    
    return _circuit_breaker_manager.create_breaker(f"model_{model_name}", config)


def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """Get an existing circuit breaker by name."""
    return _circuit_breaker_manager.get_breaker(name)

"""
Graceful Shutdown System

Provides proper SIGTERM handling to finish in-flight requests before shutdown.
Supports coordinated shutdown of all system components.
"""

import asyncio
import signal
import logging
import time
import threading
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ShutdownState(Enum):
    """Shutdown process states."""
    RUNNING = "running"
    SHUTDOWN_REQUESTED = "shutdown_requested"
    DRAINING = "draining"
    STOPPING_SERVICES = "stopping_services"
    CLEANUP = "cleanup"
    TERMINATED = "terminated"


# Alias for compatibility
ShutdownPhase = ShutdownState


@dataclass
class ActiveRequest:
    """Represents an active request during shutdown."""
    request_id: str
    endpoint: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    is_finished: bool = False
    
    def finish(self):
        """Mark the request as finished."""
        self.is_finished = True
    
    @property
    def duration(self) -> timedelta:
        """Get the duration since the request started."""
        return datetime.utcnow() - self.start_time


@dataclass
class ShutdownConfig:
    """Configuration for graceful shutdown."""
    # Timeout settings
    graceful_timeout: float = 30.0          # Total graceful shutdown timeout
    drain_timeout: float = 15.0             # Time to finish in-flight requests
    service_stop_timeout: float = 10.0      # Time for services to stop
    cleanup_timeout: float = 5.0            # Time for final cleanup
    
    # Behavior settings
    force_shutdown_after_timeout: bool = True  # Force shutdown if timeout exceeded
    log_progress: bool = True               # Log shutdown progress
    stop_accepting_requests: bool = True    # Stop accepting new requests during shutdown
    
    # Component priorities (lower number = higher priority for shutdown)
    component_shutdown_order: Dict[str, int] = field(default_factory=lambda: {
        "health_checks": 0,
        "request_handlers": 1,
        "inference_engines": 2,
        "model_managers": 3,
        "background_tasks": 4,
        "caches": 5,
        "connections": 6,
        "cleanup": 7
    })


@dataclass
class ShutdownHook:
    """A shutdown hook to be executed during graceful shutdown."""
    name: str
    callback: Callable
    timeout: float
    priority: int = 5  # Default priority
    is_async: bool = False


class GracefulShutdown:
    """
    Graceful shutdown manager that handles SIGTERM and coordinates
    shutdown of all system components.
    """
    
    def __init__(self, config: Optional[ShutdownConfig] = None):
        self.config = config or ShutdownConfig()
        
        # State management
        self._state = ShutdownState.RUNNING
        self._shutdown_event = asyncio.Event()
        self._shutdown_requested = False
        self._shutdown_start_time: Optional[float] = None
        
        # Component tracking
        self._active_requests: Set[str] = set()
        self._shutdown_hooks: List[ShutdownHook] = []
        self._components: Dict[str, Any] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Signal handlers
        self._original_sigterm_handler = None
        self._original_sigint_handler = None
        
        logger.info("Graceful shutdown manager initialized")
    
    @property
    def state(self) -> ShutdownState:
        """Get current shutdown state."""
        with self._lock:
            return self._state
    
    @property
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested
    
    @property
    def active_request_count(self) -> int:
        """Get number of active requests."""
        with self._lock:
            return len(self._active_requests)
    
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        try:
            # Store original handlers
            self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)
            self._original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
            
            logger.info("Signal handlers set up for graceful shutdown")
        except Exception as e:
            logger.error(f"Failed to set up signal handlers: {e}")
    
    def restore_signal_handlers(self):
        """Restore original signal handlers."""
        try:
            if self._original_sigterm_handler is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm_handler)
            if self._original_sigint_handler is not None:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
            
            logger.info("Original signal handlers restored")
        except Exception as e:
            logger.error(f"Failed to restore signal handlers: {e}")
    
    def _signal_handler(self, signum: int, frame):
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name} signal, initiating graceful shutdown...")
        
        # Trigger shutdown in a thread-safe way
        asyncio.create_task(self.shutdown())
    
    def register_component(self, name: str, component: Any):
        """Register a component for shutdown management."""
        with self._lock:
            self._components[name] = component
            logger.debug(f"Registered component for shutdown: {name}")
    
    def unregister_component(self, name: str):
        """Unregister a component."""
        with self._lock:
            if name in self._components:
                del self._components[name]
                logger.debug(f"Unregistered component: {name}")
    
    def add_shutdown_hook(self, 
                         name: str, 
                         callback: Callable, 
                         timeout: float = 10.0,
                         priority: int = 5,
                         is_async: bool = False):
        """Add a shutdown hook to be executed during graceful shutdown."""
        hook = ShutdownHook(
            name=name,
            callback=callback,
            timeout=timeout,
            priority=priority,
            is_async=is_async
        )
        
        with self._lock:
            self._shutdown_hooks.append(hook)
            # Sort hooks by priority (lower number = higher priority)
            self._shutdown_hooks.sort(key=lambda h: h.priority)
            
        logger.debug(f"Added shutdown hook: {name} (priority: {priority})")
    
    def remove_shutdown_hook(self, name: str) -> bool:
        """Remove a shutdown hook."""
        with self._lock:
            for i, hook in enumerate(self._shutdown_hooks):
                if hook.name == name:
                    del self._shutdown_hooks[i]
                    logger.debug(f"Removed shutdown hook: {name}")
                    return True
            return False
    
    def track_request(self, request_id: str):
        """Track an active request."""
        with self._lock:
            if not self._shutdown_requested or self._state == ShutdownState.RUNNING:
                self._active_requests.add(request_id)
                logger.debug(f"Tracking request: {request_id}")
                return True
            else:
                # Shutdown in progress, reject new requests
                logger.warning(f"Rejected request {request_id} - shutdown in progress")
                return False
    
    def untrack_request(self, request_id: str):
        """Stop tracking a request (request completed)."""
        with self._lock:
            self._active_requests.discard(request_id)
            logger.debug(f"Request completed: {request_id}")
    
    async def wait_for_shutdown(self) -> bool:
        """Wait for shutdown to be requested. Returns True if shutdown requested."""
        await self._shutdown_event.wait()
        return True
    
    async def shutdown(self) -> bool:
        """
        Perform graceful shutdown process.
        Returns True if shutdown completed successfully, False if forced.
        """
        with self._lock:
            if self._shutdown_requested:
                logger.warning("Shutdown already requested, ignoring duplicate request")
                return False
            
            self._shutdown_requested = True
            self._shutdown_start_time = time.time()
            self._state = ShutdownState.SHUTDOWN_REQUESTED
        
        logger.info("=== GRACEFUL SHUTDOWN INITIATED ===")
        if self.config.log_progress:
            logger.info(f"Shutdown timeout: {self.config.graceful_timeout}s")
        
        # Signal that shutdown has been requested
        self._shutdown_event.set()
        
        try:
            # Phase 1: Stop accepting new requests and drain existing ones
            await self._drain_requests()
            
            # Phase 2: Stop services in order
            await self._stop_services()
            
            # Phase 3: Execute shutdown hooks
            await self._execute_shutdown_hooks()
            
            # Phase 4: Final cleanup
            await self._final_cleanup()
            
            with self._lock:
                self._state = ShutdownState.TERMINATED
            
            total_time = time.time() - self._shutdown_start_time
            logger.info(f"=== GRACEFUL SHUTDOWN COMPLETED in {total_time:.2f}s ===")
            return True
            
        except asyncio.TimeoutError:
            total_time = time.time() - self._shutdown_start_time
            logger.error(f"Graceful shutdown timed out after {total_time:.2f}s")
            
            if self.config.force_shutdown_after_timeout:
                logger.warning("Forcing shutdown...")
                await self._force_shutdown()
                return False
            else:
                raise
        
        except Exception as e:
            total_time = time.time() - (self._shutdown_start_time or time.time())
            logger.error(f"Graceful shutdown failed after {total_time:.2f}s: {e}")
            
            if self.config.force_shutdown_after_timeout:
                logger.warning("Forcing shutdown due to error...")
                await self._force_shutdown()
                return False
            else:
                raise
    
    async def _drain_requests(self):
        """Phase 1: Drain in-flight requests."""
        with self._lock:
            self._state = ShutdownState.DRAINING
        
        if self.config.log_progress:
            logger.info("Phase 1: Draining in-flight requests...")
        
        start_time = time.time()
        
        # Wait for active requests to complete or timeout
        while True:
            active_count = self.active_request_count
            
            if active_count == 0:
                if self.config.log_progress:
                    logger.info("All requests drained successfully")
                break
            
            elapsed = time.time() - start_time
            if elapsed >= self.config.drain_timeout:
                logger.warning(f"Drain timeout exceeded with {active_count} requests still active")
                break
            
            if self.config.log_progress and int(elapsed) % 5 == 0:  # Log every 5 seconds
                logger.info(f"Waiting for {active_count} active requests... ({elapsed:.1f}s)")
            
            await asyncio.sleep(0.1)
    
    async def _stop_services(self):
        """Phase 2: Stop services in priority order."""
        with self._lock:
            self._state = ShutdownState.STOPPING_SERVICES
        
        if self.config.log_progress:
            logger.info("Phase 2: Stopping services...")
        
        # Group components by shutdown order
        component_groups = {}
        for name, component in self._components.items():
            priority = self.config.component_shutdown_order.get(
                name, 
                max(self.config.component_shutdown_order.values()) + 1
            )
            if priority not in component_groups:
                component_groups[priority] = []
            component_groups[priority].append((name, component))
        
        # Stop components in priority order
        for priority in sorted(component_groups.keys()):
            components = component_groups[priority]
            
            if self.config.log_progress:
                component_names = [name for name, _ in components]
                logger.info(f"Stopping components (priority {priority}): {component_names}")
            
            # Stop all components in this priority group in parallel
            tasks = []
            for name, component in components:
                task = self._stop_component(name, component)
                tasks.append(task)
            
            if tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=self.config.service_stop_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout stopping components at priority {priority}")
    
    async def _stop_component(self, name: str, component: Any):
        """Stop a single component."""
        try:
            # Try common shutdown methods
            if hasattr(component, 'shutdown') and callable(component.shutdown):
                if asyncio.iscoroutinefunction(component.shutdown):
                    await component.shutdown()
                else:
                    component.shutdown()
            elif hasattr(component, 'stop') and callable(component.stop):
                if asyncio.iscoroutinefunction(component.stop):
                    await component.stop()
                else:
                    component.stop()
            elif hasattr(component, 'close') and callable(component.close):
                if asyncio.iscoroutinefunction(component.close):
                    await component.close()
                else:
                    component.close()
            else:
                logger.debug(f"Component {name} has no shutdown method")
                
            logger.debug(f"Successfully stopped component: {name}")
            
        except Exception as e:
            logger.error(f"Error stopping component {name}: {e}")
    
    async def _execute_shutdown_hooks(self):
        """Phase 3: Execute shutdown hooks in priority order."""
        if not self._shutdown_hooks:
            return
        
        if self.config.log_progress:
            logger.info("Phase 3: Executing shutdown hooks...")
        
        for hook in self._shutdown_hooks:
            try:
                if self.config.log_progress:
                    logger.info(f"Executing shutdown hook: {hook.name}")
                
                if hook.is_async:
                    await asyncio.wait_for(hook.callback(), timeout=hook.timeout)
                else:
                    # Run sync callback in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    await asyncio.wait_for(
                        loop.run_in_executor(None, hook.callback),
                        timeout=hook.timeout
                    )
                
                logger.debug(f"Shutdown hook completed: {hook.name}")
                
            except asyncio.TimeoutError:
                logger.error(f"Shutdown hook timed out: {hook.name}")
            except Exception as e:
                logger.error(f"Error in shutdown hook {hook.name}: {e}")
    
    async def _final_cleanup(self):
        """Phase 4: Final cleanup."""
        with self._lock:
            self._state = ShutdownState.CLEANUP
        
        if self.config.log_progress:
            logger.info("Phase 4: Final cleanup...")
        
        try:
            # Clear tracking data
            with self._lock:
                self._active_requests.clear()
                self._components.clear()
            
            # Restore signal handlers
            self.restore_signal_handlers()
            
            logger.debug("Final cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during final cleanup: {e}")
    
    async def _force_shutdown(self):
        """Force immediate shutdown."""
        logger.warning("Executing force shutdown...")
        
        try:
            # Clear all tracking
            with self._lock:
                self._active_requests.clear()
                self._components.clear()
                self._state = ShutdownState.TERMINATED
            
            # Restore signal handlers
            self.restore_signal_handlers()
            
        except Exception as e:
            logger.error(f"Error during force shutdown: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current shutdown status."""
        with self._lock:
            elapsed_time = (time.time() - self._shutdown_start_time 
                          if self._shutdown_start_time else 0)
            
            return {
                "state": self._state.value,
                "shutdown_requested": self._shutdown_requested,
                "active_requests": len(self._active_requests),
                "registered_components": len(self._components),
                "shutdown_hooks": len(self._shutdown_hooks),
                "elapsed_time": elapsed_time,
                "config": {
                    "graceful_timeout": self.config.graceful_timeout,
                    "drain_timeout": self.config.drain_timeout,
                    "service_stop_timeout": self.config.service_stop_timeout,
                    "cleanup_timeout": self.config.cleanup_timeout
                }
            }


# Global graceful shutdown manager
_graceful_shutdown = None


def get_graceful_shutdown(config: Optional[ShutdownConfig] = None) -> GracefulShutdown:
    """Get the global graceful shutdown manager."""
    global _graceful_shutdown
    if _graceful_shutdown is None:
        _graceful_shutdown = GracefulShutdown(config)
    return _graceful_shutdown


def setup_graceful_shutdown(config: Optional[ShutdownConfig] = None) -> GracefulShutdown:
    """Set up graceful shutdown with signal handlers."""
    shutdown_manager = get_graceful_shutdown(config)
    shutdown_manager.setup_signal_handlers()
    return shutdown_manager


# Context manager for request tracking
class RequestTracker:
    """Context manager to track requests during shutdown."""
    
    def __init__(self, request_id: str, shutdown_manager: Optional[GracefulShutdown] = None):
        self.request_id = request_id
        self.shutdown_manager = shutdown_manager or get_graceful_shutdown()
        self.tracked = False
    
    def __enter__(self):
        self.tracked = self.shutdown_manager.track_request(self.request_id)
        return self.tracked
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tracked:
            self.shutdown_manager.untrack_request(self.request_id)


# Async context manager for request tracking
class AsyncRequestTracker:
    """Async context manager to track requests during shutdown."""
    
    def __init__(self, request_id: str, shutdown_manager: Optional[GracefulShutdown] = None):
        self.request_id = request_id
        self.shutdown_manager = shutdown_manager or get_graceful_shutdown()
        self.tracked = False
    
    async def __aenter__(self):
        self.tracked = self.shutdown_manager.track_request(self.request_id)
        return self.tracked
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.tracked:
            self.shutdown_manager.untrack_request(self.request_id)

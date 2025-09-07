"""
Simplified API Gateway for testing (no aiohttp dependency).
Provides core API gateway functionality for validation.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class RouteMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"

class LoadBalanceStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"

@dataclass
class Backend:
    """Backend service definition."""
    id: str
    host: str
    port: int
    weight: float = 1.0
    is_healthy: bool = True
    current_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_response_time: float = 0.0

@dataclass
class Route:
    """API route definition."""
    path: str
    methods: List[RouteMethod]
    backends: List[str]
    load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    rate_limit: Optional[int] = None
    auth_required: bool = True

@dataclass
class RequestMetrics:
    """Request metrics data."""
    request_id: str
    path: str
    method: str
    backend_id: Optional[str]
    start_time: datetime
    response_time: Optional[float] = None
    status_code: Optional[int] = None

class CircuitBreaker:
    """Circuit breaker for backend protection."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e

class ApiGateway:
    """Simplified API Gateway for testing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Backend management
        self.backends: Dict[str, Backend] = {}
        self.routes: Dict[str, Route] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Load balancing
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        
        # Rate limiting
        self.rate_limit_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Metrics
        self.request_metrics: deque = deque(maxlen=10000)
        self.active_requests: Dict[str, RequestMetrics] = {}
        
        logger.info("Simplified API Gateway initialized")
    
    def _find_route(self, path: str, method: str) -> Optional[Route]:
        """Find matching route for path and method."""
        for route in self.routes.values():
            if self._path_matches(path, route.path) and RouteMethod(method) in route.methods:
                return route
        return None
    
    def _path_matches(self, request_path: str, route_path: str) -> bool:
        """Check if request path matches route path."""
        if route_path.endswith('*'):
            return request_path.startswith(route_path[:-1])
        return request_path == route_path
    
    async def _select_backend(self, route: Route, exclude: List[str] = None) -> Optional[Backend]:
        """Select backend based on load balancing strategy."""
        exclude = exclude or []
        
        available_backends = [
            self.backends[backend_id]
            for backend_id in route.backends
            if backend_id in self.backends and 
               self.backends[backend_id].is_healthy and
               backend_id not in exclude
        ]
        
        if not available_backends:
            return None
        
        if route.load_balance_strategy == LoadBalanceStrategy.ROUND_ROBIN:
            counter = self.round_robin_counters[route.path]
            selected = available_backends[counter % len(available_backends)]
            self.round_robin_counters[route.path] = counter + 1
            return selected
        
        elif route.load_balance_strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return min(available_backends, key=lambda b: b.current_connections)
        
        return available_backends[0]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get gateway health status."""
        healthy_backends = sum(1 for b in self.backends.values() if b.is_healthy)
        total_backends = len(self.backends)
        
        return {
            'status': 'healthy' if healthy_backends > 0 else 'unhealthy',
            'backends': {
                'total': total_backends,
                'healthy': healthy_backends,
                'unhealthy': total_backends - healthy_backends
            },
            'routes': len(self.routes),
            'active_requests': len(self.active_requests)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get gateway metrics."""
        total_requests = len(self.request_metrics)
        
        backend_stats = {}
        for backend_id, backend in self.backends.items():
            backend_stats[backend_id] = {
                'healthy': backend.is_healthy,
                'total_requests': backend.total_requests,
                'total_errors': backend.total_errors,
                'current_connections': backend.current_connections,
                'avg_response_time': backend.avg_response_time
            }
        
        return {
            'total_requests': total_requests,
            'backends': backend_stats,
            'active_requests': len(self.active_requests)
        }
    
    def add_backend(self, backend: Backend):
        """Add backend to gateway."""
        self.backends[backend.id] = backend
        logger.info(f"Added backend: {backend.id}")
    
    def remove_backend(self, backend_id: str) -> bool:
        """Remove backend from gateway."""
        if backend_id in self.backends:
            del self.backends[backend_id]
            logger.info(f"Removed backend: {backend_id}")
            return True
        return False
    
    def add_route(self, route: Route):
        """Add route to gateway."""
        self.routes[route.path] = route
        logger.info(f"Added route: {route.path}")
    
    def remove_route(self, path: str) -> bool:
        """Remove route from gateway."""
        if path in self.routes:
            del self.routes[path]
            logger.info(f"Removed route: {path}")
            return True
        return False
    
    def validate_request(self, path: str, method: str) -> bool:
        """Validate if request can be routed."""
        route = self._find_route(path, method)
        return route is not None
    
    def cleanup(self):
        """Clean up gateway resources."""
        self.backends.clear()
        self.routes.clear()
        self.circuit_breakers.clear()
        logger.info("API Gateway cleanup completed")

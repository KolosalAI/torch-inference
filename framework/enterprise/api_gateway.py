"""
Enterprise API Gateway for multi-GPU inference.
Provides API management, routing, rate limiting, and request/response processing.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime, timezone, timedelta
import traceback
from collections import defaultdict, deque
import ssl
import weakref

# Optional imports
try:
    import aiohttp
    from aiohttp import web
    from aiohttp.web_exceptions import HTTPException
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None
    web = None
    HTTPException = None

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None

import json

logger = logging.getLogger(__name__)

class RouteMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

class LoadBalanceStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"

@dataclass
class Backend:
    """Backend service definition."""
    id: str
    host: str
    port: int
    weight: float = 1.0
    health_check_path: str = "/health"
    timeout: float = 30.0
    max_connections: int = 100
    is_healthy: bool = True
    current_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_response_time: float = 0.0
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Route:
    """API route definition."""
    path: str
    methods: List[RouteMethod]
    backends: List[str]  # Backend IDs
    load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    rate_limit: Optional[int] = None  # requests per minute
    auth_required: bool = True
    timeout: float = 30.0
    retries: int = 3
    circuit_breaker_threshold: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RequestMetrics:
    """Request metrics data."""
    request_id: str
    path: str
    method: str
    backend_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    response_time: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    user_id: Optional[str] = None
    ip_address: str = ""

class CircuitBreaker:
    """Circuit breaker for backend protection."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
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
    """Enterprise API Gateway."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available. API Gateway will run in mock mode.")
            self.app = None
        else:
            self.app = web.Application(middlewares=[
                self._create_auth_middleware(),
                self._create_rate_limit_middleware(),
                self._create_metrics_middleware(),
                self._create_error_handling_middleware()
            ])
        
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
        
        # Health checking
        self.health_check_interval = self.config.get('health_check_interval', 30)
        self.health_check_task = None
        
        # Session management
        self.session = None
        
        if self.app:
            self._setup_routes()
    
    def _create_auth_middleware(self):
        """Create authentication middleware."""
        async def middleware(request, handler):
            return await self.auth_middleware(request, handler)
        return middleware
    
    def _create_rate_limit_middleware(self):
        """Create rate limiting middleware."""
        async def middleware(request, handler):
            return await self.rate_limit_middleware(request, handler)
        return middleware
    
    def _create_metrics_middleware(self):
        """Create metrics middleware."""
        async def middleware(request, handler):
            return await self.metrics_middleware(request, handler)
        return middleware
    
    def _create_error_handling_middleware(self):
        """Create error handling middleware."""
        async def middleware(request, handler):
            return await self.error_handling_middleware(request, handler)
        return middleware
    
    async def _setup_routes(self):
        """Setup API gateway routes."""
        # Health check endpoint
        self.app.router.add_get('/gateway/health', self.gateway_health)
        
        # Metrics endpoint
        self.app.router.add_get('/gateway/metrics', self.gateway_metrics)
        
        # Admin endpoints
        self.app.router.add_get('/gateway/backends', self.list_backends)
        self.app.router.add_post('/gateway/backends', self.add_backend)
        self.app.router.add_delete('/gateway/backends/{backend_id}', self.remove_backend)
        
        # Route management
        self.app.router.add_get('/gateway/routes', self.list_routes)
        self.app.router.add_post('/gateway/routes', self.add_route)
        self.app.router.add_delete('/gateway/routes/{route_id}', self.remove_route)
        
        # Catch-all route for proxying
        self.app.router.add_route('*', '/{path:.*}', self.proxy_request)
    
    async def auth_middleware(self, request: Any, handler: Callable) -> Any:
        """Authentication middleware."""
        # Skip auth for gateway management endpoints
        if request.path.startswith('/gateway/'):
            return await handler(request)
        
        # Find matching route
        route = self._find_route(request.path, request.method)
        if not route or not route.auth_required:
            return await handler(request)
        
        # Check for authorization header
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return web.json_response(
                {'error': 'Missing or invalid authorization header'},
                status=401
            )
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        # Verify token (simplified - integrate with security manager)
        try:
            # This should integrate with the security manager
            payload = jwt.decode(
                token,
                self.config.get('jwt_secret', 'default_secret'),
                algorithms=['HS256']
            )
            request['user'] = payload
        except jwt.InvalidTokenError:
            return web.json_response(
                {'error': 'Invalid token'},
                status=401
            )
        
        return await handler(request)
    
    @middleware
    async def rate_limit_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """Rate limiting middleware."""
        # Skip rate limiting for gateway management endpoints
        if request.path.startswith('/gateway/'):
            return await handler(request)
        
        route = self._find_route(request.path, request.method)
        if not route or not route.rate_limit:
            return await handler(request)
        
        # Use user ID if available, otherwise IP address
        user_id = request.get('user', {}).get('username', '')
        client_ip = request.remote
        rate_limit_key = f"{route.path}:{user_id or client_ip}"
        
        current_time = time.time()
        window = self.rate_limit_windows[rate_limit_key]
        
        # Remove old requests (older than 1 minute)
        while window and current_time - window[0] > 60:
            window.popleft()
        
        # Check rate limit
        if len(window) >= route.rate_limit:
            return web.json_response(
                {'error': 'Rate limit exceeded'},
                status=429,
                headers={'Retry-After': '60'}
            )
        
        # Record this request
        window.append(current_time)
        
        return await handler(request)
    
    @middleware
    async def metrics_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """Metrics collection middleware."""
        request_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        # Create metrics record
        metrics = RequestMetrics(
            request_id=request_id,
            path=request.path,
            method=request.method,
            start_time=start_time,
            user_id=request.get('user', {}).get('username'),
            ip_address=request.remote or ''
        )
        
        self.active_requests[request_id] = metrics
        request['metrics'] = metrics
        
        try:
            response = await handler(request)
            metrics.status_code = response.status
            return response
        except Exception as e:
            metrics.error = str(e)
            raise
        finally:
            # Complete metrics
            end_time = datetime.now(timezone.utc)
            metrics.end_time = end_time
            metrics.response_time = (end_time - start_time).total_seconds()
            
            # Move to history
            self.request_metrics.append(metrics)
            self.active_requests.pop(request_id, None)
    
    @middleware
    async def error_handling_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """Error handling middleware."""
        try:
            return await handler(request)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unhandled error in API gateway: {e}\n{traceback.format_exc()}")
            return web.json_response(
                {'error': 'Internal server error'},
                status=500
            )
    
    def _find_route(self, path: str, method: str) -> Optional[Route]:
        """Find matching route for path and method."""
        for route in self.routes.values():
            if self._path_matches(path, route.path) and RouteMethod(method) in route.methods:
                return route
        return None
    
    def _path_matches(self, request_path: str, route_path: str) -> bool:
        """Check if request path matches route path (supports wildcards)."""
        if route_path.endswith('*'):
            return request_path.startswith(route_path[:-1])
        return request_path == route_path
    
    async def proxy_request(self, request: web.Request) -> web.Response:
        """Proxy request to backend service."""
        route = self._find_route(request.path, request.method)
        if not route:
            return web.json_response(
                {'error': 'Route not found'},
                status=404
            )
        
        # Select backend
        backend = await self._select_backend(route)
        if not backend:
            return web.json_response(
                {'error': 'No healthy backends available'},
                status=503
            )
        
        # Update metrics
        if 'metrics' in request:
            request['metrics'].backend_id = backend.id
        
        # Get circuit breaker
        circuit_breaker = self.circuit_breakers.get(backend.id)
        if not circuit_breaker:
            circuit_breaker = CircuitBreaker(route.circuit_breaker_threshold)
            self.circuit_breakers[backend.id] = circuit_breaker
        
        try:
            # Proxy the request with circuit breaker protection
            response = await circuit_breaker.call(
                self._make_backend_request,
                backend, request, route
            )
            
            # Update backend metrics
            backend.total_requests += 1
            if hasattr(request, 'metrics') and request['metrics'].response_time:
                # Update average response time
                backend.avg_response_time = (
                    (backend.avg_response_time * (backend.total_requests - 1) + 
                     request['metrics'].response_time) / backend.total_requests
                )
            
            return response
            
        except Exception as e:
            backend.total_errors += 1
            logger.error(f"Backend request failed: {e}")
            
            # Try retry if configured
            if route.retries > 0:
                for retry in range(route.retries):
                    try:
                        # Select different backend for retry
                        retry_backend = await self._select_backend(route, exclude=[backend.id])
                        if retry_backend:
                            return await self._make_backend_request(retry_backend, request, route)
                    except Exception:
                        continue
            
            return web.json_response(
                {'error': 'Backend service unavailable'},
                status=503
            )
    
    async def _select_backend(self, route: Route, exclude: List[str] = None) -> Optional[Backend]:
        """Select backend based on load balancing strategy."""
        exclude = exclude or []
        
        # Filter healthy backends
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
        
        elif route.load_balance_strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            # Simple weighted selection (could be improved)
            weights = [b.weight for b in available_backends]
            total_weight = sum(weights)
            counter = self.round_robin_counters[route.path] % total_weight
            
            current_weight = 0
            for backend in available_backends:
                current_weight += backend.weight
                if counter < current_weight:
                    self.round_robin_counters[route.path] += 1
                    return backend
        
        elif route.load_balance_strategy == LoadBalanceStrategy.RESPONSE_TIME:
            return min(available_backends, key=lambda b: b.avg_response_time)
        
        # Default fallback
        return available_backends[0]
    
    async def _make_backend_request(self, backend: Backend, request: web.Request, route: Route) -> web.Response:
        """Make request to backend service."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=route.timeout)
            connector = aiohttp.TCPConnector(limit_per_host=backend.max_connections)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        
        backend.current_connections += 1
        
        try:
            # Build backend URL
            backend_url = f"http://{backend.host}:{backend.port}{request.path_qs}"
            
            # Prepare headers (remove hop-by-hop headers)
            headers = dict(request.headers)
            hop_by_hop = ['connection', 'keep-alive', 'proxy-authenticate', 
                         'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade']
            for header in hop_by_hop:
                headers.pop(header, None)
            
            # Read request body
            body = await request.read()
            
            # Make request to backend
            async with self.session.request(
                request.method,
                backend_url,
                data=body,
                headers=headers
            ) as backend_response:
                
                # Read response
                response_body = await backend_response.read()
                
                # Copy response headers
                response_headers = dict(backend_response.headers)
                for header in hop_by_hop:
                    response_headers.pop(header, None)
                
                return web.Response(
                    body=response_body,
                    status=backend_response.status,
                    headers=response_headers
                )
        
        finally:
            backend.current_connections -= 1
    
    async def gateway_health(self, request: web.Request) -> web.Response:
        """Gateway health check endpoint."""
        healthy_backends = sum(1 for b in self.backends.values() if b.is_healthy)
        total_backends = len(self.backends)
        
        health_status = {
            'status': 'healthy' if healthy_backends > 0 else 'unhealthy',
            'backends': {
                'total': total_backends,
                'healthy': healthy_backends,
                'unhealthy': total_backends - healthy_backends
            },
            'routes': len(self.routes),
            'active_requests': len(self.active_requests),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return web.json_response(health_status, status=status_code)
    
    async def gateway_metrics(self, request: web.Request) -> web.Response:
        """Gateway metrics endpoint."""
        now = datetime.now(timezone.utc)
        
        # Recent metrics (last hour)
        recent_metrics = [
            m for m in self.request_metrics
            if now - m.start_time < timedelta(hours=1)
        ]
        
        # Calculate statistics
        total_requests = len(recent_metrics)
        successful_requests = sum(1 for m in recent_metrics if m.status_code and m.status_code < 400)
        failed_requests = total_requests - successful_requests
        
        avg_response_time = 0
        if recent_metrics:
            response_times = [m.response_time for m in recent_metrics if m.response_time]
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
        
        # Backend statistics
        backend_stats = {}
        for backend_id, backend in self.backends.items():
            backend_stats[backend_id] = {
                'healthy': backend.is_healthy,
                'total_requests': backend.total_requests,
                'total_errors': backend.total_errors,
                'current_connections': backend.current_connections,
                'avg_response_time': backend.avg_response_time
            }
        
        metrics = {
            'requests': {
                'total_last_hour': total_requests,
                'successful': successful_requests,
                'failed': failed_requests,
                'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
                'avg_response_time': avg_response_time
            },
            'backends': backend_stats,
            'active_requests': len(self.active_requests),
            'timestamp': now.isoformat()
        }
        
        return web.json_response(metrics)
    
    async def list_backends(self, request: web.Request) -> web.Response:
        """List all backends."""
        backend_list = [
            {
                'id': backend.id,
                'host': backend.host,
                'port': backend.port,
                'weight': backend.weight,
                'is_healthy': backend.is_healthy,
                'current_connections': backend.current_connections,
                'total_requests': backend.total_requests,
                'total_errors': backend.total_errors,
                'avg_response_time': backend.avg_response_time,
                'last_health_check': backend.last_health_check.isoformat() if backend.last_health_check else None
            }
            for backend in self.backends.values()
        ]
        
        return web.json_response({'backends': backend_list})
    
    async def add_backend(self, request: web.Request) -> web.Response:
        """Add new backend."""
        data = await request.json()
        
        backend = Backend(
            id=data['id'],
            host=data['host'],
            port=data['port'],
            weight=data.get('weight', 1.0),
            health_check_path=data.get('health_check_path', '/health'),
            timeout=data.get('timeout', 30.0),
            max_connections=data.get('max_connections', 100)
        )
        
        self.backends[backend.id] = backend
        
        return web.json_response({'message': f'Backend {backend.id} added successfully'})
    
    async def remove_backend(self, request: web.Request) -> web.Response:
        """Remove backend."""
        backend_id = request.match_info['backend_id']
        
        if backend_id in self.backends:
            del self.backends[backend_id]
            return web.json_response({'message': f'Backend {backend_id} removed successfully'})
        else:
            return web.json_response({'error': 'Backend not found'}, status=404)
    
    async def list_routes(self, request: web.Request) -> web.Response:
        """List all routes."""
        route_list = [
            {
                'path': route.path,
                'methods': [m.value for m in route.methods],
                'backends': route.backends,
                'load_balance_strategy': route.load_balance_strategy.value,
                'rate_limit': route.rate_limit,
                'auth_required': route.auth_required,
                'timeout': route.timeout,
                'retries': route.retries
            }
            for route in self.routes.values()
        ]
        
        return web.json_response({'routes': route_list})
    
    async def add_route(self, request: web.Request) -> web.Response:
        """Add new route."""
        data = await request.json()
        
        route = Route(
            path=data['path'],
            methods=[RouteMethod(m) for m in data['methods']],
            backends=data['backends'],
            load_balance_strategy=LoadBalanceStrategy(data.get('load_balance_strategy', 'round_robin')),
            rate_limit=data.get('rate_limit'),
            auth_required=data.get('auth_required', True),
            timeout=data.get('timeout', 30.0),
            retries=data.get('retries', 3)
        )
        
        self.routes[route.path] = route
        
        return web.json_response({'message': f'Route {route.path} added successfully'})
    
    async def remove_route(self, request: web.Request) -> web.Response:
        """Remove route."""
        route_id = request.match_info['route_id']
        
        if route_id in self.routes:
            del self.routes[route_id]
            return web.json_response({'message': f'Route {route_id} removed successfully'})
        else:
            return web.json_response({'error': 'Route not found'}, status=404)
    
    async def start_health_checks(self):
        """Start periodic health checks for backends."""
        async def health_check_worker():
            while True:
                try:
                    await self._check_backend_health()
                    await asyncio.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Health check worker error: {e}")
                    await asyncio.sleep(5)
        
        self.health_check_task = asyncio.create_task(health_check_worker())
    
    async def _check_backend_health(self):
        """Check health of all backends."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=5)
            self.session = aiohttp.ClientSession(timeout=timeout)
        
        for backend in self.backends.values():
            try:
                health_url = f"http://{backend.host}:{backend.port}{backend.health_check_path}"
                
                async with self.session.get(health_url) as response:
                    backend.is_healthy = response.status == 200
                    backend.last_health_check = datetime.now(timezone.utc)
                    
            except Exception as e:
                backend.is_healthy = False
                backend.last_health_check = datetime.now(timezone.utc)
                logger.debug(f"Health check failed for backend {backend.id}: {e}")
    
    async def start(self, host: str = '0.0.0.0', port: int = 8080, ssl_context=None):
        """Start the API gateway."""
        await self.start_health_checks()
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, host, port, ssl_context=ssl_context)
        await site.start()
        
        logger.info(f"API Gateway started on {host}:{port}")
        
        return runner
    
    async def stop(self):
        """Stop the API gateway."""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
        
        logger.info("API Gateway stopped")
    
    def cleanup(self):
        """Clean up gateway resources."""
        self.backends.clear()
        self.routes.clear()
        self.circuit_breakers.clear()
        self.rate_limit_windows.clear()
        self.request_metrics.clear()
        self.active_requests.clear()
        logger.info("API Gateway cleanup completed")

"""
Custom middleware for PyTorch Inference Framework.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.logging import get_api_logger

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all API requests and responses."""
    
    def __init__(self, app, enable_detailed_logging: bool = True):
        super().__init__(app)
        self.enable_detailed_logging = enable_detailed_logging
        self.api_logger = get_api_logger()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        start_time = time.time()
        
        # Extract request details
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        url = str(request.url)
        user_agent = request.headers.get("user-agent", "unknown")
        content_type = request.headers.get("content-type", "")
        
        # Log request
        if self.enable_detailed_logging:
            request_log = (
                f"[API REQUEST] {method} {url} - Client: {client_ip} - "
                f"User-Agent: {user_agent[:100]}"
            )
            logger.info(request_log)
            self.api_logger.info(
                f"REQUEST: {method} {url} | Client: {client_ip} | UA: {user_agent[:50]}"
            )
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log exceptions
            process_time = time.time() - start_time
            error_log = (
                f"[API ERROR] {method} {url} - Error: {str(e)} - "
                f"Time: {process_time:.3f}s - Client: {client_ip}"
            )
            logger.error(error_log)
            self.api_logger.error(
                f"ERROR: {method} {url} | Error: {str(e)} | Time: {process_time:.3f}s | Client: {client_ip}"
            )
            raise
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        if self.enable_detailed_logging:
            response_log = (
                f"[API RESPONSE] {method} {url} - Status: {response.status_code} - "
                f"Time: {process_time:.3f}s - Client: {client_ip}"
            )
            logger.info(response_log)
            self.api_logger.info(
                f"RESPONSE: {method} {url} | Status: {response.status_code} | "
                f"Time: {process_time:.3f}s | Client: {client_ip}"
            )
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for basic security headers and validation."""
    
    def __init__(self, app, max_request_size: int = 100 * 1024 * 1024):  # 100MB default
        super().__init__(app)
        self.max_request_size = max_request_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with security checks."""
        
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            from fastapi import HTTPException
            raise HTTPException(status_code=413, detail="Request entity too large")
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class CORSCustomMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware with additional features."""
    
    def __init__(self, app, allow_origins: list = None, allow_methods: list = None):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["*"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle CORS with custom logic."""
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Max-Age"] = "86400"  # 24 hours
            return response
        
        # Process request
        response = await call_next(request)
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response

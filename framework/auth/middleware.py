"""
Authentication middleware and decorators.

This module provides middleware for JWT authentication and authorization.
"""

import logging
from functools import wraps
from typing import Optional, List, Callable, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.requests import Request

from .jwt_handler import JWTHandler
from .user_store import UserStore
from .models import User

logger = logging.getLogger(__name__)

# Security scheme for FastAPI
security = HTTPBearer()


class AuthMiddleware:
    """Authentication middleware for FastAPI."""
    
    def __init__(self, jwt_handler: JWTHandler, user_store: UserStore):
        """
        Initialize auth middleware.
        
        Args:
            jwt_handler: JWT handler instance
            user_store: User store instance
        """
        self.jwt_handler = jwt_handler
        self.user_store = user_store
        self.logger = logging.getLogger(f"{__name__}.AuthMiddleware")
    
    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
        """
        Get current authenticated user from JWT token.
        
        Args:
            credentials: HTTP authorization credentials
            
        Returns:
            Current user
            
        Raises:
            HTTPException: If authentication fails
        """
        token = credentials.credentials
        
        try:
            # Verify JWT token
            payload = self.jwt_handler.verify_token(token)
            if not payload:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication token",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            # Check token type
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            # Get user
            username = payload.get("username")
            if not username:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            user = self.user_store.get_user(username)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User account is disabled",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            self.logger.debug(f"User authenticated via JWT: {username}")
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed",
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    def get_current_user_optional(self, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[User]:
        """
        Get current user (optional authentication).
        
        Args:
            credentials: HTTP authorization credentials
            
        Returns:
            Current user if authenticated, None otherwise
        """
        if not credentials:
            return None
        
        try:
            return self.get_current_user(credentials)
        except HTTPException:
            return None
    
    def authenticate_api_key(self, request: Request) -> Optional[User]:
        """
        Authenticate using API key from headers.
        
        Args:
            request: FastAPI request
            
        Returns:
            User if API key is valid, None otherwise
        """
        # Check for API key in headers
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")
        
        if not api_key or not api_key.startswith("sk_"):
            return None
        
        try:
            result = self.user_store.authenticate_api_key(api_key)
            if result:
                user, api_key_obj = result
                self.logger.debug(f"User authenticated via API key: {user.username}")
                return user
        except Exception as e:
            self.logger.error(f"API key authentication error: {e}")
        
        return None
    
    def require_auth(self, optional: bool = False):
        """
        Dependency for requiring authentication.
        
        Args:
            optional: If True, authentication is optional
            
        Returns:
            Dependency function
        """
        def dependency(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
            # Try API key first
            user = self.authenticate_api_key(request)
            if user:
                return user
            
            # Try JWT token
            if credentials:
                try:
                    return self.get_current_user(credentials)
                except HTTPException:
                    if not optional:
                        raise
            elif not optional:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            return None
        
        return dependency
    
    def require_roles(self, roles: List[str]):
        """
        Dependency for requiring specific roles.
        
        Args:
            roles: Required roles
            
        Returns:
            Dependency function
        """
        def dependency(user: User = Depends(self.require_auth())):
            if not any(role in user.roles for role in roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required roles: {roles}"
                )
            return user
        
        return dependency
    
    def require_admin(self):
        """Dependency for requiring admin role."""
        return self.require_roles(["admin"])


# Global middleware instance (will be initialized in main.py)
_auth_middleware: Optional[AuthMiddleware] = None


def init_auth_middleware(jwt_handler: JWTHandler, user_store: UserStore) -> AuthMiddleware:
    """
    Initialize global auth middleware.
    
    Args:
        jwt_handler: JWT handler instance
        user_store: User store instance
        
    Returns:
        Auth middleware instance
    """
    global _auth_middleware
    _auth_middleware = AuthMiddleware(jwt_handler, user_store)
    logger.info("Auth middleware initialized")
    return _auth_middleware


def get_auth_middleware() -> AuthMiddleware:
    """
    Get global auth middleware instance.
    
    Returns:
        Auth middleware instance
        
    Raises:
        RuntimeError: If middleware not initialized
    """
    if _auth_middleware is None:
        raise RuntimeError("Auth middleware not initialized. Call init_auth_middleware() first.")
    return _auth_middleware


# Convenience dependency functions
def get_current_user():
    """Get dependency for current user."""
    middleware = get_auth_middleware()
    return middleware.require_auth(optional=False)


def get_current_user_optional():
    """Get dependency for optional current user."""
    middleware = get_auth_middleware()
    return middleware.require_auth(optional=True)


def require_admin():
    """Get dependency for admin user."""
    middleware = get_auth_middleware()
    return middleware.require_admin()


def require_roles(roles: List[str]) -> Callable:
    """Get dependency for specific roles."""
    middleware = get_auth_middleware()
    return middleware.require_roles(roles)


# Decorator for route protection
def require_auth(optional: bool = False, roles: List[str] = None):
    """
    Decorator for requiring authentication on routes.
    
    Args:
        optional: If True, authentication is optional
        roles: Required roles
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # This is a simple decorator - in practice, you'd use FastAPI dependencies
            return await func(*args, **kwargs)
        
        # Add metadata for route inspection
        wrapper._auth_required = not optional
        wrapper._required_roles = roles or []
        
        return wrapper
    
    return decorator


# Rate limiting helpers
class RateLimiter:
    """Simple rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 1):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_minutes: Time window in minutes
        """
        self.max_requests = max_requests
        self.window_minutes = window_minutes
        self.requests: dict = {}  # user_id -> [timestamps]
    
    def is_allowed(self, user_id: str) -> bool:
        """
        Check if request is allowed for user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if request is allowed
        """
        import time
        current_time = time.time()
        window_start = current_time - (self.window_minutes * 60)
        
        # Clean old requests
        if user_id in self.requests:
            self.requests[user_id] = [
                req_time for req_time in self.requests[user_id]
                if req_time > window_start
            ]
        else:
            self.requests[user_id] = []
        
        # Check if under limit
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[user_id].append(current_time)
        return True

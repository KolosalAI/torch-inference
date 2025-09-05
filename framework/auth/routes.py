"""
Authentication API endpoints.

This module provides REST API endpoints for authentication and user management.
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse

from .models import (
    AuthRequest, RegisterRequest, ChangePasswordRequest, 
    GenerateAPIKeyRequest, RefreshTokenRequest,
    AuthResponse, UserResponse, APIKeyResponse, TokenRefreshResponse,
    Token, User
)
from .jwt_handler import JWTHandler
from .user_store import UserStore
from .middleware import AuthMiddleware
from .password import is_password_strong

logger = logging.getLogger(__name__)


class AuthRouter:
    """Authentication router for FastAPI."""
    
    def __init__(self, jwt_handler: JWTHandler, user_store: UserStore, auth_middleware: AuthMiddleware):
        """
        Initialize auth router.
        
        Args:
            jwt_handler: JWT handler instance
            user_store: User store instance
            auth_middleware: Auth middleware instance
        """
        self.jwt_handler = jwt_handler
        self.user_store = user_store
        self.auth_middleware = auth_middleware
        self.router = APIRouter(prefix="/auth", tags=["Authentication"])
        self.logger = logging.getLogger(f"{__name__}.AuthRouter")
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup authentication routes."""
        
        @self.router.post("/register", response_model=AuthResponse)
        async def register(request: RegisterRequest) -> AuthResponse:
            """Register new user."""
            try:
                # Check password strength
                is_strong, issues = is_password_strong(request.password)
                if not is_strong:
                    return AuthResponse(
                        success=False,
                        message="Password does not meet requirements",
                        error="; ".join(issues)
                    )
                
                # Create user
                user = self.user_store.create_user(
                    username=request.username,
                    email=request.email,
                    full_name=request.full_name,
                    password=request.password,
                    roles=["user"]
                )
                
                # Create tokens
                user_data = {
                    "sub": user.id,
                    "username": user.username,
                    "email": user.email,
                    "roles": user.roles
                }
                
                access_token = self.jwt_handler.create_access_token(user_data)
                refresh_token = self.jwt_handler.create_refresh_token(user_data)
                
                token = Token(
                    access_token=access_token,
                    refresh_token=refresh_token,
                    expires_in=self.jwt_handler.access_token_expire_minutes * 60
                )
                
                self.logger.info(f"User registered: {request.username}")
                
                return AuthResponse(
                    success=True,
                    message="User registered successfully",
                    token=token,
                    user=user.to_dict()
                )
                
            except ValueError as e:
                self.logger.warning(f"Registration failed: {e}")
                return AuthResponse(
                    success=False,
                    message="Registration failed",
                    error=str(e)
                )
            except Exception as e:
                self.logger.error(f"Registration error: {e}")
                return AuthResponse(
                    success=False,
                    message="Registration failed",
                    error="Internal server error"
                )
        
        @self.router.post("/login", response_model=AuthResponse)
        async def login(request: AuthRequest) -> AuthResponse:
            """Login user."""
            try:
                # Authenticate user
                user = self.user_store.authenticate_user(request.username, request.password)
                if not user:
                    return AuthResponse(
                        success=False,
                        message="Invalid credentials",
                        error="Username or password is incorrect"
                    )
                
                # Create tokens
                user_data = {
                    "sub": user.id,
                    "username": user.username,
                    "email": user.email,
                    "roles": user.roles
                }
                
                access_token = self.jwt_handler.create_access_token(user_data)
                refresh_token = self.jwt_handler.create_refresh_token(user_data)
                
                token = Token(
                    access_token=access_token,
                    refresh_token=refresh_token,
                    expires_in=self.jwt_handler.access_token_expire_minutes * 60
                )
                
                self.logger.info(f"User logged in: {request.username}")
                
                return AuthResponse(
                    success=True,
                    message="Login successful",
                    token=token,
                    user=user.to_dict()
                )
                
            except Exception as e:
                self.logger.error(f"Login error: {e}")
                return AuthResponse(
                    success=False,
                    message="Login failed",
                    error="Internal server error"
                )
        
        @self.router.post("/refresh", response_model=TokenRefreshResponse)
        async def refresh_token(request: RefreshTokenRequest) -> TokenRefreshResponse:
            """Refresh access token."""
            try:
                new_access_token = self.jwt_handler.refresh_access_token(request.refresh_token)
                if not new_access_token:
                    return TokenRefreshResponse(
                        success=False,
                        message="Token refresh failed",
                        error="Invalid or expired refresh token"
                    )
                
                return TokenRefreshResponse(
                    success=True,
                    access_token=new_access_token,
                    expires_in=self.jwt_handler.access_token_expire_minutes * 60,
                    message="Token refreshed successfully"
                )
                
            except Exception as e:
                self.logger.error(f"Token refresh error: {e}")
                return TokenRefreshResponse(
                    success=False,
                    message="Token refresh failed",
                    error="Internal server error"
                )
        
        @self.router.post("/logout")
        async def logout(current_user: User = Depends(self.auth_middleware.get_current_user)) -> Dict[str, Any]:
            """Logout user (revoke tokens)."""
            try:
                # Revoke all user tokens
                revoked_count = self.jwt_handler.revoke_user_tokens(current_user.id)
                
                self.logger.info(f"User logged out: {current_user.username}, tokens revoked: {revoked_count}")
                
                return {
                    "success": True,
                    "message": f"Logout successful. {revoked_count} tokens revoked."
                }
                
            except Exception as e:
                self.logger.error(f"Logout error: {e}")
                return {
                    "success": False,
                    "message": "Logout failed",
                    "error": "Internal server error"
                }
        
        @self.router.get("/profile", response_model=UserResponse)
        async def get_profile(current_user: User = Depends(self.auth_middleware.get_current_user)) -> UserResponse:
            """Get current user profile."""
            try:
                return UserResponse(
                    success=True,
                    user=current_user.to_dict(),
                    message="Profile retrieved successfully"
                )
                
            except Exception as e:
                self.logger.error(f"Profile retrieval error: {e}")
                return UserResponse(
                    success=False,
                    message="Failed to retrieve profile",
                    error="Internal server error"
                )
        
        @self.router.put("/password", response_model=Dict[str, Any])
        async def change_password(
            request: ChangePasswordRequest,
            current_user: User = Depends(self.auth_middleware.get_current_user)
        ) -> Dict[str, Any]:
            """Change user password."""
            try:
                # Check new password strength
                is_strong, issues = is_password_strong(request.new_password)
                if not is_strong:
                    return {
                        "success": False,
                        "message": "Password does not meet requirements",
                        "error": "; ".join(issues)
                    }
                
                # Change password
                success = self.user_store.change_password(
                    current_user.username,
                    request.current_password,
                    request.new_password
                )
                
                if not success:
                    return {
                        "success": False,
                        "message": "Password change failed",
                        "error": "Current password is incorrect"
                    }
                
                # Revoke all existing tokens to force re-login
                self.jwt_handler.revoke_user_tokens(current_user.id)
                
                self.logger.info(f"Password changed for user: {current_user.username}")
                
                return {
                    "success": True,
                    "message": "Password changed successfully. Please login again."
                }
                
            except Exception as e:
                self.logger.error(f"Password change error: {e}")
                return {
                    "success": False,
                    "message": "Password change failed",
                    "error": "Internal server error"
                }
        
        @self.router.post("/generate-key", response_model=APIKeyResponse)
        async def generate_api_key(
            request: GenerateAPIKeyRequest,
            current_user: User = Depends(self.auth_middleware.get_current_user)
        ) -> APIKeyResponse:
            """Generate new API key."""
            try:
                raw_key, api_key = self.user_store.create_api_key(
                    current_user.username,
                    request.name,
                    request.scopes,
                    request.expires_in_days
                )
                
                self.logger.info(f"API key generated for user {current_user.username}: {request.name}")
                
                return APIKeyResponse(
                    success=True,
                    api_key=raw_key,
                    key_info=api_key.to_dict(),
                    message="API key generated successfully"
                )
                
            except Exception as e:
                self.logger.error(f"API key generation error: {e}")
                return APIKeyResponse(
                    success=False,
                    message="API key generation failed",
                    error="Internal server error"
                )
        
        @self.router.get("/api-keys")
        async def list_api_keys(current_user: User = Depends(self.auth_middleware.get_current_user)) -> Dict[str, Any]:
            """List user's API keys."""
            try:
                api_keys = self.user_store.list_user_api_keys(current_user.username)
                
                return {
                    "success": True,
                    "api_keys": api_keys,
                    "count": len(api_keys)
                }
                
            except Exception as e:
                self.logger.error(f"API key listing error: {e}")
                return {
                    "success": False,
                    "message": "Failed to list API keys",
                    "error": "Internal server error"
                }
        
        @self.router.delete("/api-keys/{key_id}")
        async def revoke_api_key(
            key_id: str,
            current_user: User = Depends(self.auth_middleware.get_current_user)
        ) -> Dict[str, Any]:
            """Revoke API key."""
            try:
                success = self.user_store.revoke_api_key(current_user.username, key_id)
                
                if not success:
                    return {
                        "success": False,
                        "message": "API key not found",
                        "error": "Key not found or already revoked"
                    }
                
                self.logger.info(f"API key revoked for user {current_user.username}: {key_id}")
                
                return {
                    "success": True,
                    "message": "API key revoked successfully"
                }
                
            except Exception as e:
                self.logger.error(f"API key revocation error: {e}")
                return {
                    "success": False,
                    "message": "API key revocation failed",
                    "error": "Internal server error"
                }
        
        # Admin endpoints
        @self.router.get("/users", dependencies=[Depends(self.auth_middleware.require_admin())])
        async def list_users() -> Dict[str, Any]:
            """List all users (admin only)."""
            try:
                users = self.user_store.list_users()
                stats = self.user_store.get_stats()
                
                return {
                    "success": True,
                    "users": users,
                    "stats": stats
                }
                
            except Exception as e:
                self.logger.error(f"User listing error: {e}")
                return {
                    "success": False,
                    "message": "Failed to list users",
                    "error": "Internal server error"
                }
        
        @self.router.delete("/users/{username}", dependencies=[Depends(self.auth_middleware.require_admin())])
        async def delete_user(username: str) -> Dict[str, Any]:
            """Delete user (admin only)."""
            try:
                success = self.user_store.delete_user(username)
                
                if not success:
                    return {
                        "success": False,
                        "message": "User not found",
                        "error": "User does not exist"
                    }
                
                self.logger.info(f"User deleted by admin: {username}")
                
                return {
                    "success": True,
                    "message": f"User {username} deleted successfully"
                }
                
            except Exception as e:
                self.logger.error(f"User deletion error: {e}")
                return {
                    "success": False,
                    "message": "User deletion failed",
                    "error": "Internal server error"
                }
        
        @self.router.get("/stats", dependencies=[Depends(self.auth_middleware.require_admin())])
        async def get_auth_stats() -> Dict[str, Any]:
            """Get authentication statistics (admin only)."""
            try:
                user_stats = self.user_store.get_stats()
                jwt_stats = {
                    "active_tokens": self.jwt_handler.get_active_token_count(),
                    "blacklisted_tokens": len(self.jwt_handler.blacklisted_tokens)
                }
                
                return {
                    "success": True,
                    "user_store": user_stats,
                    "jwt_handler": jwt_stats,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Stats retrieval error: {e}")
                return {
                    "success": False,
                    "message": "Failed to retrieve stats",
                    "error": "Internal server error"
                }
        
        @self.router.post("/cleanup", dependencies=[Depends(self.auth_middleware.require_admin())])
        async def cleanup_expired() -> Dict[str, Any]:
            """Cleanup expired tokens and API keys (admin only)."""
            try:
                expired_tokens = self.jwt_handler.cleanup_expired_tokens()
                expired_keys = self.user_store.cleanup_expired_keys()
                
                self.logger.info(f"Cleanup completed - Tokens: {expired_tokens}, Keys: {expired_keys}")
                
                return {
                    "success": True,
                    "message": "Cleanup completed",
                    "expired_tokens": expired_tokens,
                    "expired_keys": expired_keys
                }
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                return {
                    "success": False,
                    "message": "Cleanup failed",
                    "error": "Internal server error"
                }
    
    def get_router(self) -> APIRouter:
        """Get the FastAPI router."""
        return self.router


def create_auth_router(jwt_handler: JWTHandler, user_store: UserStore, auth_middleware: AuthMiddleware) -> APIRouter:
    """
    Create authentication router.
    
    Args:
        jwt_handler: JWT handler instance
        user_store: User store instance
        auth_middleware: Auth middleware instance
        
    Returns:
        FastAPI router
    """
    auth_router = AuthRouter(jwt_handler, user_store, auth_middleware)
    return auth_router.get_router()

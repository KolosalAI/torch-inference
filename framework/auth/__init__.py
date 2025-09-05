"""
Authentication and authorization module.

This module provides JWT-based authentication with simple file-based user storage.
"""

from .jwt_handler import JWTHandler
from .password import hash_password, verify_password, generate_password, generate_api_key
from .models import User, Token, APIKey, AuthRequest, AuthResponse, RegisterRequest, ChangePasswordRequest, GenerateAPIKeyRequest
from .user_store import UserStore
from .middleware import AuthMiddleware, require_auth, init_auth_middleware, get_current_user, require_admin
from .routes import create_auth_router

__all__ = [
    "JWTHandler",
    "hash_password",
    "verify_password", 
    "generate_password",
    "generate_api_key",
    "User",
    "Token",
    "APIKey",
    "AuthRequest",
    "AuthResponse",
    "RegisterRequest",
    "ChangePasswordRequest", 
    "GenerateAPIKeyRequest",
    "UserStore",
    "AuthMiddleware",
    "require_auth",
    "init_auth_middleware",
    "get_current_user",
    "require_admin",
    "create_auth_router"
]

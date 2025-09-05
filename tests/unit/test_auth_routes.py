"""
Test suite for authentication API routes.

This module tests the FastAPI authentication endpoints.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from framework.auth.routes import AuthRouter, create_auth_router
from framework.auth.models import User
from framework.auth.jwt_handler import JWTHandler
from framework.auth.user_store import UserStore
from framework.auth.middleware import AuthMiddleware


class TestAuthRouter:
    """Test AuthRouter class."""
    
    @pytest.fixture
    def mock_jwt_handler(self):
        """Mock JWT handler."""
        handler = Mock(spec=JWTHandler)
        handler.access_token_expire_minutes = 30
        handler.revoke_user_tokens.return_value = 2
        handler.create_access_token.return_value = "mock_access_token_123"
        handler.create_refresh_token.return_value = "mock_refresh_token_456"
        return handler
    
    @pytest.fixture
    def mock_user_store(self):
        """Mock user store."""
        store = Mock(spec=UserStore)
        user = User(
            id="user_123",
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            hashed_password="hashed",
            roles=["user"]
        )
        store.authenticate_user.return_value = user
        return store
    
    @pytest.fixture
    def mock_auth_middleware(self):
        """Mock auth middleware."""
        middleware = Mock(spec=AuthMiddleware)
        user = User(
            id="test_user_123",
            username="testuser", 
            email="test@example.com",
            full_name="Test User",
            hashed_password="hashed",
            roles=["user"]
        )
        middleware.get_current_user.return_value = user
        return middleware
    
    @pytest.fixture
    def auth_router(self, mock_jwt_handler, mock_user_store, mock_auth_middleware):
        """Create auth router instance."""
        return AuthRouter(mock_jwt_handler, mock_user_store, mock_auth_middleware)
    
    @pytest.fixture
    def test_app(self, auth_router, mock_auth_middleware):
        """Create test FastAPI app."""
        app = FastAPI()
        app.include_router(auth_router.get_router())
        
        # Mock the middleware methods used in dependencies
        def mock_get_current_user():
            return User(
                id="test_user_123",
                username="testuser",
                email="test@example.com", 
                full_name="Test User",
                hashed_password="hashed",
                roles=["user"]
            )
        
        def mock_require_admin():
            return User(
                id="admin_user_123",
                username="admin",
                email="admin@example.com",
                full_name="Admin User", 
                hashed_password="hashed",
                roles=["admin"]
            )
        
        # Override the dependencies
        app.dependency_overrides[mock_auth_middleware.get_current_user] = mock_get_current_user
        app.dependency_overrides[mock_auth_middleware.require_admin] = mock_require_admin
        
        return app
    
    @pytest.fixture
    def client(self, test_app):
        """Create test client."""
        return TestClient(test_app)
    
    def test_login_success(self, client):
        """Test successful login."""
        response = client.post("/auth/login", json={
            "username": "testuser",
            "password": "testpass"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_logout_success(self, client):
        """Test successful logout."""
        response = client.post("/auth/logout")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

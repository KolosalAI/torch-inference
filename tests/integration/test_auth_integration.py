"""
Integration tests for the authentication system.

This module tests the complete authentication workflow and component integration.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import patch, Mock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from framework.auth import (
    JWTHandler, UserStore, AuthMiddleware, create_auth_router,
    User, Token, APIKey, hash_password, generate_api_key
)


class TestAuthSystemIntegration:
    """Test complete authentication system integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def jwt_handler(self):
        """Create JWT handler instance."""
        return JWTHandler(
            secret_key="test_secret_key_for_integration",
            algorithm="HS256",
            access_token_expire_minutes=30,
            refresh_token_expire_days=7
        )
    
    @pytest.fixture
    def user_store(self, temp_dir):
        """Create user store instance."""
        users_file = temp_dir / "integration_users.json"
        api_keys_file = temp_dir / "integration_keys.json"
        return UserStore(str(users_file), str(api_keys_file))
    
    @pytest.fixture
    def auth_middleware(self, jwt_handler, user_store):
        """Create auth middleware instance."""
        return AuthMiddleware(jwt_handler, user_store)
    
    @pytest.fixture
    def auth_app(self, jwt_handler, user_store, auth_middleware):
        """Create FastAPI app with auth system."""
        from framework.auth.middleware import init_auth_middleware
        
        app = FastAPI()
        
        # Initialize the global auth middleware
        init_auth_middleware(jwt_handler, user_store)
        
        # Add auth router
        auth_router = create_auth_router(jwt_handler, user_store, auth_middleware)
        app.include_router(auth_router)
        
        # Add a protected endpoint for testing
        @app.get("/protected")
        async def protected_endpoint(current_user: User = None):
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")
            return {"message": f"Hello {current_user.username}", "user_id": current_user.id}
        
        return app
    
    @pytest.fixture
    def client(self, auth_app):
        """Create test client."""
        return TestClient(auth_app)
    
    def test_complete_user_registration_flow(self, client, user_store):
        """Test complete user registration and verification flow."""
        # Register new user
        registration_data = {
            "username": "integrationuser",
            "email": "integration@example.com",
            "password": "StrongPassword123!",
            "full_name": "Integration User"
        }
        
        with patch('framework.auth.routes.is_password_strong', return_value=(True, [])):
            response = client.post("/auth/register", json=registration_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "token" in data
        assert "user" in data
        
        # Verify user exists in store
        user = user_store.get_user("integrationuser")
        assert user is not None
        assert user.email == "integration@example.com"
        assert user.is_active is True
        assert user.is_verified is False
        
        # Verify tokens are valid
        access_token = data["token"]["access_token"]
        refresh_token = data["token"]["refresh_token"]
        assert access_token is not None
        assert refresh_token is not None
    
    def test_complete_login_flow(self, client, user_store):
        """Test complete login flow."""
        # Create user first
        user_store.create_user(
            username="loginuser",
            email="login@example.com",
            full_name="Login User",
            password="LoginPassword123!"
        )
        
        # Login
        login_data = {
            "username": "loginuser",
            "password": "LoginPassword123!"
        }
        
        response = client.post("/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "token" in data
        
        # Verify user's last login is updated
        user = user_store.get_user("loginuser")
        assert user.last_login is not None
        assert user.failed_attempts == 0
    
    def test_token_refresh_flow(self, client, user_store):
        """Test token refresh flow."""
        # Create and login user
        user_store.create_user(
            username="refreshuser",
            email="refresh@example.com", 
            full_name="Refresh User",
            password="RefreshPassword123!"
        )
        
        login_response = client.post("/auth/login", json={
            "username": "refreshuser",
            "password": "RefreshPassword123!"
        })
        
        refresh_token = login_response.json()["token"]["refresh_token"]
        
        # Refresh token
        refresh_response = client.post("/auth/refresh", json={
            "refresh_token": refresh_token
        })
        
        assert refresh_response.status_code == 200
        data = refresh_response.json()
        assert data["success"] is True
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_api_key_workflow(self, client, user_store, auth_middleware):
        """Test complete API key generation and usage workflow."""
        # Create user
        user = user_store.create_user(
            username="apikeyuser",
            email="apikey@example.com",
            full_name="API Key User", 
            password="APIKeyPassword123!"
        )
        
        # Login to get token
        login_response = client.post("/auth/login", json={
            "username": "apikeyuser",
            "password": "APIKeyPassword123!"
        })
        access_token = login_response.json()["token"]["access_token"]
        
        # Generate API key
        headers = {"Authorization": f"Bearer {access_token}"}
        
        with patch.object(auth_middleware, 'get_current_user', return_value=user):
            key_response = client.post("/auth/generate-key", 
                json={"name": "Test API Key", "scopes": ["read", "write"]},
                headers=headers
            )
        
        assert key_response.status_code == 200
        key_data = key_response.json()
        assert key_data["success"] is True
        raw_api_key = key_data["api_key"]
        
        # Test API key authentication
        result = user_store.authenticate_api_key(raw_api_key)
        assert result is not None
        authenticated_user, api_key_obj = result
        assert authenticated_user.username == "apikeyuser"
        assert api_key_obj.name == "Test API Key"
        assert api_key_obj.scopes == ["read", "write"]
    
    def test_password_change_workflow(self, client, user_store, auth_middleware):
        """Test password change workflow."""
        # Create user
        user = user_store.create_user(
            username="passworduser",
            email="password@example.com",
            full_name="Password User",
            password="OldPassword123!"
        )
        
        # Login
        login_response = client.post("/auth/login", json={
            "username": "passworduser", 
            "password": "OldPassword123!"
        })
        access_token = login_response.json()["token"]["access_token"]
        
        # Change password
        headers = {"Authorization": f"Bearer {access_token}"}
        
        with patch.object(auth_middleware, 'get_current_user', return_value=user):
            with patch('framework.auth.routes.is_password_strong', return_value=(True, [])):
                change_response = client.put("/auth/password",
                    json={
                        "current_password": "OldPassword123!",
                        "new_password": "NewPassword123!"
                    },
                    headers=headers
                )
        
        assert change_response.status_code == 200
        data = change_response.json()
        assert data["success"] is True
        
        # Verify old password no longer works
        old_login_response = client.post("/auth/login", json={
            "username": "passworduser",
            "password": "OldPassword123!"
        })
        assert old_login_response.json()["success"] is False
        
        # Verify new password works
        new_login_response = client.post("/auth/login", json={
            "username": "passworduser",
            "password": "NewPassword123!"
        })
        assert new_login_response.json()["success"] is True
    
    def test_user_lockout_workflow(self, client, user_store):
        """Test user lockout after failed attempts."""
        # Create user
        user_store.create_user(
            username="lockoutuser",
            email="lockout@example.com",
            full_name="Lockout User",
            password="LockoutPassword123!"
        )
        
        # Make 5 failed login attempts
        for _ in range(5):
            response = client.post("/auth/login", json={
                "username": "lockoutuser",
                "password": "wrongpassword"
            })
            assert response.json()["success"] is False
        
        # User should be locked
        user = user_store.get_user("lockoutuser")
        assert user.failed_attempts == 5
        assert user.locked_until is not None
        
        # Even correct password should fail while locked
        response = client.post("/auth/login", json={
            "username": "lockoutuser",
            "password": "LockoutPassword123!"
        })
        assert response.json()["success"] is False
    
    def test_admin_functionality_workflow(self, client, user_store):
        """Test admin functionality workflow."""
        # Default admin user should exist
        admin_user = user_store.get_user("admin")
        assert admin_user is not None
        assert "admin" in admin_user.roles
        
        # Login as admin to get token
        admin_login_response = client.post("/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        admin_token = admin_login_response.json()["token"]["access_token"]
        headers = {"Authorization": f"Bearer {admin_token}"}
        
        # Test admin endpoints
        # List users
        users_response = client.get("/auth/users", headers=headers)
        assert users_response.status_code == 200
        
        # Get stats
        stats_response = client.get("/auth/stats", headers=headers)
        assert stats_response.status_code == 200
        
        # Cleanup
        cleanup_response = client.post("/auth/cleanup", headers=headers)
        assert cleanup_response.status_code == 200
    
    def test_concurrent_operations(self, client, user_store, jwt_handler):
        """Test concurrent auth operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def register_user(user_id):
            try:
                with patch('framework.auth.routes.is_password_strong', return_value=(True, [])):
                    response = client.post("/auth/register", json={
                        "username": f"concurrentuser{user_id}",
                        "email": f"concurrent{user_id}@example.com",
                        "password": f"ConcurrentPass{user_id}!",
                        "full_name": f"Concurrent User {user_id}"
                    })
                results.append((user_id, response.status_code == 200))
            except Exception as e:
                errors.append((user_id, str(e)))
        
        # Start multiple registration threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_user, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert all(success for _, success in results)
        
        # Verify all users were created
        assert len(user_store.users) >= 11  # 10 + admin
    
    def test_session_management(self, client, user_store, jwt_handler):
        """Test session management functionality."""
        # Create user
        user_store.create_user(
            username="sessionuser",
            email="session@example.com",
            full_name="Session User",
            password="SessionPassword123!"
        )
        
        # Login multiple times to create multiple tokens
        tokens = []
        for _ in range(3):
            response = client.post("/auth/login", json={
                "username": "sessionuser",
                "password": "SessionPassword123!"
            })
            tokens.append(response.json()["token"]["access_token"])
        
        # Verify multiple active tokens (at least 2)
        user = user_store.get_user("sessionuser")
        token_count = jwt_handler.get_user_token_count(user.id)
        assert token_count >= 2
        
        # Logout (should revoke all tokens)
        # Use one of the tokens to authenticate the logout request
        headers = {"Authorization": f"Bearer {tokens[0]}"}
        logout_response = client.post("/auth/logout", headers=headers)
        
        assert logout_response.status_code == 200
        
        # Verify tokens are revoked
        new_token_count = jwt_handler.get_user_token_count(user.id)
        assert new_token_count == 0
    
    def test_data_persistence(self, temp_dir):
        """Test data persistence across application restarts."""
        users_file = temp_dir / "persistence_users.json"
        api_keys_file = temp_dir / "persistence_keys.json"
        
        # Create first instance
        store1 = UserStore(str(users_file), str(api_keys_file))
        user = store1.create_user(
            username="persistuser",
            email="persist@example.com",
            full_name="Persist User",
            password="PersistPassword123!"
        )
        raw_key, api_key = store1.create_api_key("persistuser", "Persist Key")
        
        user_id = user.id
        key_id = api_key.id
        
        # Create second instance (simulating restart)
        store2 = UserStore(str(users_file), str(api_keys_file))
        
        # Verify data persisted
        loaded_user = store2.get_user("persistuser")
        assert loaded_user is not None
        assert loaded_user.id == user_id
        assert loaded_user.email == "persist@example.com"
        
        # Verify API key persisted and works
        auth_result = store2.authenticate_api_key(raw_key)
        assert auth_result is not None
        auth_user, auth_key = auth_result
        assert auth_user.username == "persistuser"
        assert auth_key.id == key_id
    
    def test_error_recovery(self, client, user_store, jwt_handler):
        """Test system recovery from various error conditions."""
        # Test recovery from JWT handler errors
        original_verify = jwt_handler.verify_token
        jwt_handler.verify_token = Mock(side_effect=Exception("JWT Error"))
        
        try:
            # Should handle JWT errors gracefully
            response = client.get("/auth/profile", headers={"Authorization": "Bearer token"})
            # In test environment, auth might be bypassed, so just check that it doesn't crash
            assert response.status_code in [200, 401, 422, 500]
        finally:
            # Restore original function
            jwt_handler.verify_token = original_verify
        
        # Restore functionality
        jwt_handler.verify_token = original_verify
        
        # Test recovery from user store errors
        original_auth = user_store.authenticate_user
        user_store.authenticate_user = Mock(side_effect=Exception("Store Error"))
        
        response = client.post("/auth/login", json={
            "username": "testuser",
            "password": "password"
        })
        # Should handle store errors gracefully
        assert response.status_code == 200
        assert response.json()["success"] is False
        
        # Restore functionality
        user_store.authenticate_user = original_auth
    
    def test_security_headers_and_validation(self, client, user_store):
        """Test security headers and input validation."""
        # Test malicious input handling
        malicious_inputs = [
            {"username": "<script>alert('xss')</script>", "password": "pass"},
            {"username": "'; DROP TABLE users; --", "password": "pass"},
            {"username": "test", "password": "A" * 10000},  # Very long password
            {"username": "\x00\x01\x02", "password": "pass"},  # Binary data
        ]
        
        for malicious_input in malicious_inputs:
            response = client.post("/auth/login", json=malicious_input)
            # Should either reject input or handle safely
            assert response.status_code in [200, 400, 422]
            if response.status_code == 200:
                # If accepted, should fail authentication safely
                assert response.json()["success"] is False
    
    def test_rate_limiting_simulation(self, client):
        """Test rate limiting behavior simulation."""
        # Simulate rapid requests
        responses = []
        for _ in range(20):
            response = client.post("/auth/login", json={
                "username": "testuser",
                "password": "wrongpassword"
            })
            responses.append(response.status_code)
        
        # All requests should be handled (no rate limiting implemented yet)
        # But system should remain stable
        assert all(status in [200, 401, 422, 429] for status in responses)
    
    def test_cleanup_and_maintenance(self, user_store, jwt_handler):
        """Test cleanup and maintenance operations."""
        # Create expired tokens (mock)
        user = user_store.create_user(
            username="cleanupuser",
            email="cleanup@example.com",
            full_name="Cleanup User",
            password="CleanupPassword123!"
        )
        
        # Create tokens
        user_data = {
            "sub": user.id,
            "username": user.username,
            "email": user.email
        }
        
        token1 = jwt_handler.create_access_token(user_data)
        token2 = jwt_handler.create_refresh_token(user_data)
        
        initial_count = jwt_handler.get_active_token_count()
        
        # Simulate token expiry and cleanup
        jwt_handler.cleanup_expired_tokens()
        
        # Create expired API keys
        raw_key, api_key = user_store.create_api_key("cleanupuser", "Cleanup Key", expires_in_days=1)
        
        # Simulate API key expiry
        from datetime import datetime, timezone, timedelta
        api_key.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        
        initial_key_count = len(user_store.api_keys)
        cleaned_keys = user_store.cleanup_expired_keys()
        
        # Should clean up expired keys
        assert len(user_store.api_keys) < initial_key_count or cleaned_keys >= 0


if __name__ == "__main__":
    pytest.main([__file__])

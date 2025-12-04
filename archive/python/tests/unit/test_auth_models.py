"""
Test suite for authentication models.

This module tests the authentication data models including User, APIKey, and Token.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from framework.auth.models import (
    User, APIKey, Token, AuthRequest, RegisterRequest, 
    ChangePasswordRequest, GenerateAPIKeyRequest, RefreshTokenRequest,
    AuthResponse, UserResponse, APIKeyResponse, TokenRefreshResponse,
    create_user_id, create_api_key_id
)


class TestUser:
    """Test User model."""
    
    def test_user_creation(self):
        """Test creating a user."""
        user = User(
            id="user_123",
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            hashed_password="hashed_password_123",
            roles=["user"],
            is_active=True,
            is_verified=False
        )
        
        assert user.id == "user_123"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.hashed_password == "hashed_password_123"
        assert user.roles == ["user"]
        assert user.is_active is True
        assert user.is_verified is False
        assert isinstance(user.created_at, datetime)
        assert user.last_login is None
        assert user.failed_attempts == 0
        assert user.locked_until is None
        assert user.metadata == {}
    
    def test_user_defaults(self):
        """Test user with default values."""
        user = User(
            id="user_456",
            username="defaultuser",
            email="default@example.com",
            full_name="Default User",
            hashed_password="hashed_pass"
        )
        
        assert user.roles == []
        assert user.api_keys == []
        assert user.is_active is True
        assert user.is_verified is False
        assert user.failed_attempts == 0
        assert user.locked_until is None
        assert user.metadata == {}
    
    def test_user_to_dict(self):
        """Test user serialization."""
        user = User(
            id="user_789",
            username="serialuser",
            email="serial@example.com",
            full_name="Serial User",
            hashed_password="secret_hash",
            roles=["admin", "user"],
            api_keys=["key1", "key2"],
            metadata={"department": "IT"}
        )
        
        user_dict = user.to_dict()
        
        assert user_dict["id"] == "user_789"
        assert user_dict["username"] == "serialuser"
        assert user_dict["email"] == "serial@example.com"
        assert user_dict["full_name"] == "Serial User"
        assert user_dict["roles"] == ["admin", "user"]
        assert user_dict["is_active"] is True
        assert user_dict["is_verified"] is False
        assert user_dict["api_key_count"] == 2
        assert user_dict["metadata"] == {"department": "IT"}
        
        # Sensitive data should not be included
        assert "hashed_password" not in user_dict
        assert "api_keys" not in user_dict
    
    def test_user_from_dict(self):
        """Test user deserialization."""
        data = {
            "id": "user_101",
            "username": "deserialuser",
            "email": "deserial@example.com",
            "full_name": "Deserial User",
            "hashed_password": "hash123",
            "roles": ["moderator"],
            "api_keys": ["key3"],
            "is_active": False,
            "is_verified": True,
            "failed_attempts": 2,
            "created_at": "2023-01-01T00:00:00+00:00",
            "metadata": {"role": "mod"}
        }
        
        user = User.from_dict(data)
        
        assert user.id == "user_101"
        assert user.username == "deserialuser"
        assert user.email == "deserial@example.com"
        assert user.full_name == "Deserial User"
        assert user.hashed_password == "hash123"
        assert user.roles == ["moderator"]
        assert user.api_keys == ["key3"]
        assert user.is_active is False
        assert user.is_verified is True
        assert user.failed_attempts == 2
        assert user.metadata == {"role": "mod"}
        assert isinstance(user.created_at, datetime)
    
    def test_user_role_management(self):
        """Test user role management methods."""
        user = User(
            id="user_role",
            username="roleuser",
            email="role@example.com",
            full_name="Role User",
            hashed_password="hash",
            roles=["user"]
        )
        
        # Test has_role
        assert user.has_role("user") is True
        assert user.has_role("admin") is False
        
        # Test add_role
        user.add_role("moderator")
        assert user.has_role("moderator") is True
        assert "moderator" in user.roles
        
        # Test adding duplicate role
        user.add_role("user")
        assert user.roles.count("user") == 1
        
        # Test remove_role
        user.remove_role("user")
        assert user.has_role("user") is False
        assert "user" not in user.roles
        
        # Test removing non-existent role
        user.remove_role("nonexistent")  # Should not raise error
        assert "moderator" in user.roles
    
    def test_user_lock_status(self):
        """Test user lock status."""
        user = User(
            id="user_lock",
            username="lockuser",
            email="lock@example.com",
            full_name="Lock User",
            hashed_password="hash"
        )
        
        # User should not be locked initially
        assert user.is_locked() is False
        
        # Lock user for 1 hour
        user.locked_until = datetime.now(timezone.utc) + timedelta(hours=1)
        assert user.is_locked() is True
        
        # Set lock time in the past
        user.locked_until = datetime.now(timezone.utc) - timedelta(hours=1)
        assert user.is_locked() is False
        
        # No lock time
        user.locked_until = None
        assert user.is_locked() is False
    
    def test_user_api_key_management(self):
        """Test user API key management."""
        user = User(
            id="user_api",
            username="apiuser",
            email="api@example.com",
            full_name="API User",
            hashed_password="hash"
        )
        
        # Initially no API keys
        assert len(user.api_keys) == 0
        
        # Add API key
        user.add_api_key("key_hash_1")
        assert "key_hash_1" in user.api_keys
        assert len(user.api_keys) == 1
        
        # Add duplicate key (should not add)
        user.add_api_key("key_hash_1")
        assert len(user.api_keys) == 1
        
        # Add another key
        user.add_api_key("key_hash_2")
        assert "key_hash_2" in user.api_keys
        assert len(user.api_keys) == 2
        
        # Remove key
        user.remove_api_key("key_hash_1")
        assert "key_hash_1" not in user.api_keys
        assert "key_hash_2" in user.api_keys
        assert len(user.api_keys) == 1
        
        # Remove non-existent key
        user.remove_api_key("nonexistent")  # Should not raise error
        assert len(user.api_keys) == 1


class TestAPIKey:
    """Test APIKey model."""
    
    def test_api_key_creation(self):
        """Test creating an API key."""
        api_key = APIKey(
            id="key_123",
            name="Test Key",
            key_hash="hash_123",
            user_id="user_456",
            scopes=["read", "write"],
            is_active=True,
            usage_count=5
        )
        
        assert api_key.id == "key_123"
        assert api_key.name == "Test Key"
        assert api_key.key_hash == "hash_123"
        assert api_key.user_id == "user_456"
        assert api_key.scopes == ["read", "write"]
        assert api_key.is_active is True
        assert api_key.usage_count == 5
        assert isinstance(api_key.created_at, datetime)
        assert api_key.last_used is None
        assert api_key.expires_at is None
    
    def test_api_key_defaults(self):
        """Test API key with default values."""
        api_key = APIKey(
            id="key_default",
            name="Default Key",
            key_hash="default_hash",
            user_id="user_default"
        )
        
        assert api_key.scopes == []
        assert api_key.is_active is True
        assert api_key.usage_count == 0
        assert isinstance(api_key.created_at, datetime)
        assert api_key.last_used is None
        assert api_key.expires_at is None
    
    def test_api_key_to_dict(self):
        """Test API key serialization."""
        expires_at = datetime.now(timezone.utc) + timedelta(days=30)
        api_key = APIKey(
            id="key_serial",
            name="Serial Key",
            key_hash="serial_hash",
            user_id="user_serial",
            scopes=["admin"],
            expires_at=expires_at,
            usage_count=10
        )
        
        key_dict = api_key.to_dict()
        
        assert key_dict["id"] == "key_serial"
        assert key_dict["name"] == "Serial Key"
        assert key_dict["key_hash"] == "serial_hash"
        assert key_dict["user_id"] == "user_serial"
        assert key_dict["scopes"] == ["admin"]
        assert key_dict["is_active"] is True
        assert key_dict["usage_count"] == 10
        assert "created_at" in key_dict
        assert "expires_at" in key_dict
        assert key_dict["last_used"] is None
    
    def test_api_key_from_dict(self):
        """Test API key deserialization."""
        data = {
            "id": "key_deserial",
            "name": "Deserial Key",
            "key_hash": "deserial_hash",
            "user_id": "user_deserial",
            "scopes": ["read"],
            "is_active": False,
            "created_at": "2023-01-01T00:00:00+00:00",
            "expires_at": "2023-12-31T23:59:59+00:00",
            "usage_count": 3
        }
        
        api_key = APIKey.from_dict(data)
        
        assert api_key.id == "key_deserial"
        assert api_key.name == "Deserial Key"
        assert api_key.key_hash == "deserial_hash"
        assert api_key.user_id == "user_deserial"
        assert api_key.scopes == ["read"]
        assert api_key.is_active is False
        assert api_key.usage_count == 3
        assert isinstance(api_key.created_at, datetime)
        assert isinstance(api_key.expires_at, datetime)
    
    def test_api_key_expiry(self):
        """Test API key expiry logic."""
        # Non-expiring key
        api_key = APIKey(
            id="key_no_expiry",
            name="No Expiry",
            key_hash="hash",
            user_id="user"
        )
        assert api_key.is_expired() is False
        
        # Future expiry
        future_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        api_key.expires_at = future_expiry
        assert api_key.is_expired() is False
        
        # Past expiry
        past_expiry = datetime.now(timezone.utc) - timedelta(hours=1)
        api_key.expires_at = past_expiry
        assert api_key.is_expired() is True
    
    def test_api_key_validity(self):
        """Test API key validity checks."""
        api_key = APIKey(
            id="key_valid",
            name="Valid Key",
            key_hash="hash",
            user_id="user",
            is_active=True
        )
        
        # Active and not expired
        assert api_key.is_valid() is True
        
        # Inactive
        api_key.is_active = False
        assert api_key.is_valid() is False
        
        # Active but expired
        api_key.is_active = True
        api_key.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        assert api_key.is_valid() is False
    
    def test_api_key_usage(self):
        """Test API key usage tracking."""
        api_key = APIKey(
            id="key_usage",
            name="Usage Key",
            key_hash="hash",
            user_id="user"
        )
        
        # Initial state
        assert api_key.usage_count == 0
        assert api_key.last_used is None
        
        # Use the key
        api_key.use()
        assert api_key.usage_count == 1
        assert isinstance(api_key.last_used, datetime)
        
        # Use again
        first_used = api_key.last_used
        api_key.use()
        assert api_key.usage_count == 2
        assert api_key.last_used >= first_used


class TestToken:
    """Test Token model."""
    
    def test_token_creation(self):
        """Test creating a token."""
        token = Token(
            access_token="access_123",
            refresh_token="refresh_123",
            token_type="bearer",
            expires_in=3600
        )
        
        assert token.access_token == "access_123"
        assert token.refresh_token == "refresh_123"
        assert token.token_type == "bearer"
        assert token.expires_in == 3600
    
    def test_token_defaults(self):
        """Test token with default values."""
        token = Token(access_token="token_123")
        
        assert token.access_token == "token_123"
        assert token.refresh_token is None
        assert token.token_type == "bearer"
        assert token.expires_in == 1800
    
    def test_token_to_dict(self):
        """Test token serialization."""
        token = Token(
            access_token="access_dict",
            refresh_token="refresh_dict",
            token_type="custom",
            expires_in=7200
        )
        
        token_dict = token.to_dict()
        
        assert token_dict["access_token"] == "access_dict"
        assert token_dict["refresh_token"] == "refresh_dict"
        assert token_dict["token_type"] == "custom"
        assert token_dict["expires_in"] == 7200


class TestRequestModels:
    """Test request models."""
    
    def test_auth_request(self):
        """Test AuthRequest model."""
        request = AuthRequest(username="testuser", password="testpass")
        assert request.username == "testuser"
        assert request.password == "testpass"
    
    def test_register_request(self):
        """Test RegisterRequest model."""
        request = RegisterRequest(
            username="newuser",
            email="new@example.com",
            password="newpass123",
            full_name="New User"
        )
        assert request.username == "newuser"
        assert request.email == "new@example.com"
        assert request.password == "newpass123"
        assert request.full_name == "New User"
    
    def test_change_password_request(self):
        """Test ChangePasswordRequest model."""
        request = ChangePasswordRequest(
            current_password="oldpassword",
            new_password="newpassword"
        )
        assert request.current_password == "oldpassword"
        assert request.new_password == "newpassword"
    
    def test_generate_api_key_request(self):
        """Test GenerateAPIKeyRequest model."""
        request = GenerateAPIKeyRequest(
            name="Test Key",
            scopes=["read", "write"],
            expires_in_days=30
        )
        assert request.name == "Test Key"
        assert request.scopes == ["read", "write"]
        assert request.expires_in_days == 30
    
    def test_generate_api_key_request_defaults(self):
        """Test GenerateAPIKeyRequest with defaults."""
        request = GenerateAPIKeyRequest(name="Default Key")
        assert request.name == "Default Key"
        assert request.scopes == []
        assert request.expires_in_days is None
    
    def test_refresh_token_request(self):
        """Test RefreshTokenRequest model."""
        request = RefreshTokenRequest(refresh_token="refresh_123")
        assert request.refresh_token == "refresh_123"


class TestResponseModels:
    """Test response models."""
    
    def test_auth_response(self):
        """Test AuthResponse model."""
        token = Token(access_token="test_token")
        user_data = {"id": "user_123", "username": "testuser"}
        
        response = AuthResponse(
            success=True,
            message="Login successful",
            token=token,
            user=user_data
        )
        
        assert response.success is True
        assert response.message == "Login successful"
        assert response.token == token
        assert response.user == user_data
        assert response.error is None
    
    def test_auth_response_error(self):
        """Test AuthResponse with error."""
        response = AuthResponse(
            success=False,
            message="Login failed",
            error="Invalid credentials"
        )
        
        assert response.success is False
        assert response.message == "Login failed"
        assert response.error == "Invalid credentials"
        assert response.token is None
        assert response.user is None
    
    def test_user_response(self):
        """Test UserResponse model."""
        user_data = {"id": "user_456", "username": "testuser"}
        
        response = UserResponse(
            success=True,
            user=user_data,
            message="User retrieved"
        )
        
        assert response.success is True
        assert response.user == user_data
        assert response.message == "User retrieved"
        assert response.error is None
    
    def test_api_key_response(self):
        """Test APIKeyResponse model."""
        key_info = {"id": "key_123", "name": "Test Key"}
        
        response = APIKeyResponse(
            success=True,
            api_key="raw_key_123",
            key_info=key_info,
            message="Key generated"
        )
        
        assert response.success is True
        assert response.api_key == "raw_key_123"
        assert response.key_info == key_info
        assert response.message == "Key generated"
        assert response.error is None
    
    def test_token_refresh_response(self):
        """Test TokenRefreshResponse model."""
        response = TokenRefreshResponse(
            success=True,
            access_token="new_access_token",
            token_type="bearer",
            expires_in=3600,
            message="Token refreshed"
        )
        
        assert response.success is True
        assert response.access_token == "new_access_token"
        assert response.token_type == "bearer"
        assert response.expires_in == 3600
        assert response.message == "Token refreshed"
        assert response.error is None


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_user_id(self):
        """Test user ID generation."""
        user_id = create_user_id()
        assert user_id.startswith("user_")
        assert len(user_id) > 5
        
        # Should generate unique IDs
        user_id2 = create_user_id()
        assert user_id != user_id2
    
    def test_create_api_key_id(self):
        """Test API key ID generation."""
        key_id = create_api_key_id()
        assert key_id.startswith("key_")
        assert len(key_id) > 4
        
        # Should generate unique IDs
        key_id2 = create_api_key_id()
        assert key_id != key_id2


if __name__ == "__main__":
    pytest.main([__file__])

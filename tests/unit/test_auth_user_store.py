"""
Test suite for user store functionality.

This module tests the file-based user storage system.
"""

import pytest
import tempfile
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, mock_open

from framework.auth.user_store import UserStore
from framework.auth.models import User, APIKey


class TestUserStore:
    """Test UserStore class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def user_store(self, temp_dir):
        """Create user store instance."""
        users_file = temp_dir / "users.json"
        api_keys_file = temp_dir / "api_keys.json"
        return UserStore(str(users_file), str(api_keys_file))
    
    @pytest.fixture
    def sample_user_data(self):
        """Sample user data for testing."""
        return {
            "username": "testuser",
            "email": "test@example.com",
            "full_name": "Test User",
            "password": "testpass123",
            "roles": ["user"]
        }
    
    def test_user_store_initialization(self, user_store):
        """Test user store initialization."""
        assert isinstance(user_store.users, dict)
        assert isinstance(user_store.api_keys, dict)
        assert user_store.users_file.exists()
        assert user_store.api_keys_file.exists()
        
        # Should create default admin user
        assert "admin" in user_store.users
        admin_user = user_store.users["admin"]
        assert admin_user.username == "admin"
        assert "admin" in admin_user.roles
        assert admin_user.is_active is True
        assert admin_user.is_verified is True
    
    def test_user_store_directory_creation(self, temp_dir):
        """Test that user store creates necessary directories."""
        nested_path = temp_dir / "nested" / "directory"
        users_file = nested_path / "users.json"
        api_keys_file = nested_path / "api_keys.json"
        
        user_store = UserStore(str(users_file), str(api_keys_file))
        
        assert nested_path.exists()
        assert users_file.exists()
        assert api_keys_file.exists()
    
    def test_create_user_success(self, user_store, sample_user_data):
        """Test successful user creation."""
        user = user_store.create_user(**sample_user_data)
        
        assert user.username == sample_user_data["username"]
        assert user.email == sample_user_data["email"]
        assert user.full_name == sample_user_data["full_name"]
        assert user.roles == sample_user_data["roles"]
        assert user.is_active is True
        assert user.is_verified is False
        assert user.username in user_store.users
        
        # Password should be hashed
        assert user.hashed_password != sample_user_data["password"]
    
    def test_create_user_duplicate_username(self, user_store, sample_user_data):
        """Test creating user with duplicate username."""
        user_store.create_user(**sample_user_data)
        
        with pytest.raises(ValueError) as exc_info:
            user_store.create_user(**sample_user_data)
        assert "already exists" in str(exc_info.value)
    
    def test_create_user_duplicate_email(self, user_store, sample_user_data):
        """Test creating user with duplicate email."""
        user_store.create_user(**sample_user_data)
        
        # Try to create another user with same email
        duplicate_email_data = sample_user_data.copy()
        duplicate_email_data["username"] = "differentuser"
        
        with pytest.raises(ValueError) as exc_info:
            user_store.create_user(**duplicate_email_data)
        assert "Email" in str(exc_info.value) and "already exists" in str(exc_info.value)
    
    def test_create_user_default_roles(self, user_store):
        """Test creating user with default roles."""
        user = user_store.create_user(
            username="defaultroleuser",
            email="default@example.com",
            full_name="Default Role User",
            password="password123"
        )
        
        assert user.roles == ["user"]
    
    def test_authenticate_user_success(self, user_store, sample_user_data):
        """Test successful user authentication."""
        user_store.create_user(**sample_user_data)
        
        authenticated_user = user_store.authenticate_user(
            sample_user_data["username"],
            sample_user_data["password"]
        )
        
        assert authenticated_user is not None
        assert authenticated_user.username == sample_user_data["username"]
        assert authenticated_user.failed_attempts == 0
        assert authenticated_user.locked_until is None
        assert isinstance(authenticated_user.last_login, datetime)
    
    def test_authenticate_user_invalid_username(self, user_store):
        """Test authentication with invalid username."""
        user = user_store.authenticate_user("nonexistent", "password")
        assert user is None
    
    def test_authenticate_user_invalid_password(self, user_store, sample_user_data):
        """Test authentication with invalid password."""
        user_store.create_user(**sample_user_data)
        
        user = user_store.authenticate_user(
            sample_user_data["username"],
            "wrongpassword"
        )
        
        assert user is None
        
        # Check failed attempts are incremented
        stored_user = user_store.users[sample_user_data["username"]]
        assert stored_user.failed_attempts == 1
    
    def test_authenticate_user_inactive(self, user_store, sample_user_data):
        """Test authentication with inactive user."""
        user_store.create_user(**sample_user_data)
        user_store.update_user(sample_user_data["username"], is_active=False)
        
        user = user_store.authenticate_user(
            sample_user_data["username"],
            sample_user_data["password"]
        )
        
        assert user is None
    
    def test_authenticate_user_locked(self, user_store, sample_user_data):
        """Test authentication with locked user."""
        user_store.create_user(**sample_user_data)
        
        # Lock the user
        stored_user = user_store.users[sample_user_data["username"]]
        stored_user.locked_until = datetime.now(timezone.utc) + timedelta(hours=1)
        
        user = user_store.authenticate_user(
            sample_user_data["username"],
            sample_user_data["password"]
        )
        
        assert user is None
    
    def test_authenticate_user_lockout_after_failed_attempts(self, user_store, sample_user_data):
        """Test user lockout after too many failed attempts."""
        user_store.create_user(**sample_user_data)
        
        # Make 5 failed attempts
        for _ in range(5):
            user_store.authenticate_user(
                sample_user_data["username"],
                "wrongpassword"
            )
        
        stored_user = user_store.users[sample_user_data["username"]]
        assert stored_user.failed_attempts == 5
        assert stored_user.locked_until is not None
        assert stored_user.locked_until > datetime.now(timezone.utc)
    
    def test_get_user_methods(self, user_store, sample_user_data):
        """Test various user retrieval methods."""
        created_user = user_store.create_user(**sample_user_data)
        
        # Test get_user
        user = user_store.get_user(sample_user_data["username"])
        assert user is not None
        assert user.username == sample_user_data["username"]
        
        # Test get_user_by_id
        user_by_id = user_store.get_user_by_id(created_user.id)
        assert user_by_id is not None
        assert user_by_id.id == created_user.id
        
        # Test get_user_by_email
        user_by_email = user_store.get_user_by_email(sample_user_data["email"])
        assert user_by_email is not None
        assert user_by_email.email == sample_user_data["email"]
        
        # Test with non-existent values
        assert user_store.get_user("nonexistent") is None
        assert user_store.get_user_by_id("nonexistent") is None
        assert user_store.get_user_by_email("nonexistent@example.com") is None
    
    def test_update_user(self, user_store, sample_user_data):
        """Test user update functionality."""
        user_store.create_user(**sample_user_data)
        
        # Update user
        success = user_store.update_user(
            sample_user_data["username"],
            email="newemail@example.com",
            full_name="New Full Name",
            roles=["admin", "user"],
            is_verified=True,
            metadata={"department": "IT"}
        )
        
        assert success is True
        
        updated_user = user_store.get_user(sample_user_data["username"])
        assert updated_user.email == "newemail@example.com"
        assert updated_user.full_name == "New Full Name"
        assert updated_user.roles == ["admin", "user"]
        assert updated_user.is_verified is True
        assert updated_user.metadata == {"department": "IT"}
    
    def test_update_user_nonexistent(self, user_store):
        """Test updating non-existent user."""
        success = user_store.update_user("nonexistent", email="new@example.com")
        assert success is False
    
    def test_change_password_success(self, user_store, sample_user_data):
        """Test successful password change."""
        user_store.create_user(**sample_user_data)
        
        success = user_store.change_password(
            sample_user_data["username"],
            sample_user_data["password"],
            "newpassword123"
        )
        
        assert success is True
        
        # Test authentication with new password
        user = user_store.authenticate_user(
            sample_user_data["username"],
            "newpassword123"
        )
        assert user is not None
        
        # Test old password no longer works
        user = user_store.authenticate_user(
            sample_user_data["username"],
            sample_user_data["password"]
        )
        assert user is None
    
    def test_change_password_invalid_old_password(self, user_store, sample_user_data):
        """Test password change with invalid old password."""
        user_store.create_user(**sample_user_data)
        
        success = user_store.change_password(
            sample_user_data["username"],
            "wrongoldpassword",
            "newpassword123"
        )
        
        assert success is False
    
    def test_delete_user(self, user_store, sample_user_data):
        """Test user deletion."""
        user_store.create_user(**sample_user_data)
        
        # Create API key for the user
        raw_key, api_key = user_store.create_api_key(
            sample_user_data["username"],
            "Test Key"
        )
        
        assert sample_user_data["username"] in user_store.users
        assert len(user_store.api_keys) > 0
        
        success = user_store.delete_user(sample_user_data["username"])
        assert success is True
        
        assert sample_user_data["username"] not in user_store.users
        
        # User's API keys should also be removed
        user_keys = [key for key in user_store.api_keys.values() 
                    if key.name == "Test Key"]
        assert len(user_keys) == 0
    
    def test_delete_user_nonexistent(self, user_store):
        """Test deleting non-existent user."""
        success = user_store.delete_user("nonexistent")
        assert success is False
    
    def test_list_users(self, user_store, sample_user_data):
        """Test listing all users."""
        user_store.create_user(**sample_user_data)
        
        users = user_store.list_users()
        assert len(users) >= 2  # admin + test user
        
        # Check that sensitive data is not included
        for user_dict in users:
            assert "hashed_password" not in user_dict
            assert "api_keys" not in user_dict
            assert "id" in user_dict
            assert "username" in user_dict
    
    def test_create_api_key_success(self, user_store, sample_user_data):
        """Test successful API key creation."""
        user_store.create_user(**sample_user_data)
        
        raw_key, api_key = user_store.create_api_key(
            sample_user_data["username"],
            "Test API Key",
            scopes=["read", "write"],
            expires_in_days=30
        )
        
        assert isinstance(raw_key, str)
        assert len(raw_key) > 0
        assert isinstance(api_key, APIKey)
        assert api_key.name == "Test API Key"
        assert api_key.scopes == ["read", "write"]
        assert api_key.expires_at is not None
        assert api_key.key_hash in user_store.api_keys
        
        # Check user has API key reference
        user = user_store.get_user(sample_user_data["username"])
        assert api_key.key_hash in user.api_keys
    
    def test_create_api_key_nonexistent_user(self, user_store):
        """Test API key creation for non-existent user."""
        with pytest.raises(ValueError) as exc_info:
            user_store.create_api_key("nonexistent", "Test Key")
        assert "User not found" in str(exc_info.value)
    
    def test_authenticate_api_key_success(self, user_store, sample_user_data):
        """Test successful API key authentication."""
        user_store.create_user(**sample_user_data)
        raw_key, api_key = user_store.create_api_key(
            sample_user_data["username"],
            "Auth Test Key"
        )
        
        result = user_store.authenticate_api_key(raw_key)
        assert result is not None
        
        user, authenticated_key = result
        assert user.username == sample_user_data["username"]
        assert authenticated_key.id == api_key.id
        assert authenticated_key.usage_count == 1
        assert authenticated_key.last_used is not None
    
    def test_authenticate_api_key_invalid(self, user_store):
        """Test API key authentication with invalid key."""
        result = user_store.authenticate_api_key("invalid_key")
        assert result is None
    
    def test_authenticate_api_key_expired(self, user_store, sample_user_data):
        """Test API key authentication with expired key."""
        user_store.create_user(**sample_user_data)
        raw_key, api_key = user_store.create_api_key(
            sample_user_data["username"],
            "Expired Key",
            expires_in_days=1
        )
        
        # Make the key expired
        api_key.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        
        result = user_store.authenticate_api_key(raw_key)
        assert result is None
    
    def test_authenticate_api_key_inactive_user(self, user_store, sample_user_data):
        """Test API key authentication with inactive user."""
        user_store.create_user(**sample_user_data)
        raw_key, api_key = user_store.create_api_key(
            sample_user_data["username"],
            "Inactive User Key"
        )
        
        # Deactivate user
        user_store.update_user(sample_user_data["username"], is_active=False)
        
        result = user_store.authenticate_api_key(raw_key)
        assert result is None
    
    def test_revoke_api_key(self, user_store, sample_user_data):
        """Test API key revocation."""
        user_store.create_user(**sample_user_data)
        raw_key, api_key = user_store.create_api_key(
            sample_user_data["username"],
            "Revoke Test Key"
        )
        
        success = user_store.revoke_api_key(
            sample_user_data["username"],
            api_key.id
        )
        
        assert success is True
        assert api_key.is_active is False
        
        # User should no longer have the key reference
        user = user_store.get_user(sample_user_data["username"])
        assert api_key.key_hash not in user.api_keys
        
        # Authentication should fail
        result = user_store.authenticate_api_key(raw_key)
        assert result is None
    
    def test_revoke_api_key_invalid(self, user_store, sample_user_data):
        """Test revoking non-existent API key."""
        user_store.create_user(**sample_user_data)
        
        success = user_store.revoke_api_key(
            sample_user_data["username"],
            "nonexistent_key_id"
        )
        
        assert success is False
    
    def test_list_user_api_keys(self, user_store, sample_user_data):
        """Test listing user API keys."""
        user_store.create_user(**sample_user_data)
        
        # Create multiple API keys
        user_store.create_api_key(sample_user_data["username"], "Key 1")
        user_store.create_api_key(sample_user_data["username"], "Key 2")
        
        api_keys = user_store.list_user_api_keys(sample_user_data["username"])
        
        assert len(api_keys) == 2
        key_names = [key["name"] for key in api_keys]
        assert "Key 1" in key_names
        assert "Key 2" in key_names
        
        # Check that sensitive data is removed
        for key_info in api_keys:
            assert "key_hash" not in key_info
            assert "id" in key_info
            assert "name" in key_info
    
    def test_list_user_api_keys_nonexistent_user(self, user_store):
        """Test listing API keys for non-existent user."""
        api_keys = user_store.list_user_api_keys("nonexistent")
        assert api_keys == []
    
    def test_cleanup_expired_keys(self, user_store, sample_user_data):
        """Test cleanup of expired API keys."""
        user_store.create_user(**sample_user_data)
        
        # Create expired key
        raw_key1, api_key1 = user_store.create_api_key(
            sample_user_data["username"],
            "Expired Key",
            expires_in_days=1
        )
        api_key1.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        
        # Create valid key
        raw_key2, api_key2 = user_store.create_api_key(
            sample_user_data["username"],
            "Valid Key"
        )
        
        initial_count = len(user_store.api_keys)
        cleaned_count = user_store.cleanup_expired_keys()
        
        assert cleaned_count == 1
        assert len(user_store.api_keys) == initial_count - 1
        assert api_key1.key_hash not in user_store.api_keys
        assert api_key2.key_hash in user_store.api_keys
        
        # Check user's key references are updated
        user = user_store.get_user(sample_user_data["username"])
        assert api_key1.key_hash not in user.api_keys
        assert api_key2.key_hash in user.api_keys
    
    def test_get_stats(self, user_store, sample_user_data):
        """Test getting storage statistics."""
        user_store.create_user(**sample_user_data)
        user_store.create_api_key(sample_user_data["username"], "Stats Key")
        
        stats = user_store.get_stats()
        
        assert "total_users" in stats
        assert "active_users" in stats
        assert "verified_users" in stats
        assert "total_api_keys" in stats
        assert "active_api_keys" in stats
        assert "users_file" in stats
        assert "api_keys_file" in stats
        
        assert stats["total_users"] >= 2  # admin + test user
        assert stats["active_users"] >= 2
        assert stats["total_api_keys"] >= 1
        assert stats["active_api_keys"] >= 1
    
    def test_file_persistence(self, temp_dir):
        """Test that data persists across user store instances."""
        users_file = temp_dir / "persistence_users.json"
        api_keys_file = temp_dir / "persistence_keys.json"
        
        # Create first instance and add user
        store1 = UserStore(str(users_file), str(api_keys_file))
        store1.create_user(
            username="persistuser",
            email="persist@example.com",
            full_name="Persist User",
            password="password123"
        )
        raw_key, api_key = store1.create_api_key("persistuser", "Persist Key")
        
        # Create second instance - should load existing data
        store2 = UserStore(str(users_file), str(api_keys_file))
        
        assert "persistuser" in store2.users
        assert len(store2.api_keys) > 0
        
        # Test authentication works
        user = store2.authenticate_user("persistuser", "password123")
        assert user is not None
        
        result = store2.authenticate_api_key(raw_key)
        assert result is not None
    
    def test_file_loading_error_handling(self, temp_dir):
        """Test error handling when loading corrupted files."""
        users_file = temp_dir / "corrupt_users.json"
        api_keys_file = temp_dir / "corrupt_keys.json"
        
        # Create corrupted JSON files
        with open(users_file, 'w') as f:
            f.write("invalid json content")
        
        with open(api_keys_file, 'w') as f:
            f.write("invalid json content")
        
        # Should handle errors gracefully
        store = UserStore(str(users_file), str(api_keys_file))
        
        # Should still create default admin user
        assert "admin" in store.users
        assert len(store.api_keys) == 0
    
    def test_atomic_file_writing(self, user_store, sample_user_data):
        """Test that file writing is atomic."""
        user_store.create_user(**sample_user_data)
        
        # Mock file operations to test atomic writing
        with patch('builtins.open', mock_open()) as mock_file:
            with patch.object(Path, 'replace') as mock_replace:
                user_store._save_users()
                
                # Should write to temporary file first
                mock_file.assert_called()
                mock_replace.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])

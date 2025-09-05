"""
Test suite for JWT handler functionality.

This module tests JWT token creation, validation, and management.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

from framework.auth.jwt_handler import JWTHandler, JWT_AVAILABLE


@pytest.mark.skipif(not JWT_AVAILABLE, reason="python-jose not available")
class TestJWTHandler:
    """Test JWTHandler class."""
    
    @pytest.fixture
    def jwt_handler(self):
        """Create JWT handler instance."""
        return JWTHandler(
            secret_key="test_secret_key_123",
            algorithm="HS256",
            access_token_expire_minutes=30,
            refresh_token_expire_days=7
        )
    
    @pytest.fixture
    def user_data(self):
        """Sample user data for tokens."""
        return {
            "sub": "user_123",
            "username": "testuser",
            "email": "test@example.com",
            "roles": ["user"]
        }
    
    def test_jwt_handler_initialization(self, jwt_handler):
        """Test JWT handler initialization."""
        assert jwt_handler.secret_key == "test_secret_key_123"
        assert jwt_handler.algorithm == "HS256"
        assert jwt_handler.access_token_expire_minutes == 30
        assert jwt_handler.refresh_token_expire_days == 7
        assert isinstance(jwt_handler.active_tokens, dict)
        assert isinstance(jwt_handler.blacklisted_tokens, set)
        assert len(jwt_handler.active_tokens) == 0
        assert len(jwt_handler.blacklisted_tokens) == 0
    
    def test_jwt_handler_initialization_no_secret(self):
        """Test JWT handler with auto-generated secret."""
        handler = JWTHandler()
        assert handler.secret_key is not None
        assert len(handler.secret_key) > 0
        assert handler.algorithm == "HS256"
        assert handler.access_token_expire_minutes == 30
        assert handler.refresh_token_expire_days == 7
    
    def test_jwt_handler_no_jose_library(self):
        """Test JWT handler when jose library is not available."""
        with patch('framework.auth.jwt_handler.JWT_AVAILABLE', False):
            with pytest.raises(RuntimeError) as exc_info:
                JWTHandler()
            assert "python-jose is required" in str(exc_info.value)
    
    def test_create_access_token(self, jwt_handler, user_data):
        """Test access token creation."""
        token = jwt_handler.create_access_token(user_data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        assert token in jwt_handler.active_tokens
        
        # Check token info
        token_info = jwt_handler.active_tokens[token]
        assert token_info["user_id"] == "user_123"
        assert token_info["username"] == "testuser"
        assert token_info["type"] == "access"
        assert isinstance(token_info["expires_at"], datetime)
    
    def test_create_access_token_custom_expiry(self, jwt_handler, user_data):
        """Test access token creation with custom expiry."""
        custom_expiry = timedelta(hours=2)
        token = jwt_handler.create_access_token(user_data, custom_expiry)
        
        assert isinstance(token, str)
        token_info = jwt_handler.active_tokens[token]
        
        # Check that custom expiry is used
        expected_expiry = datetime.now(timezone.utc) + custom_expiry
        actual_expiry = token_info["expires_at"]
        
        # Allow 1 minute tolerance for execution time
        assert abs((actual_expiry - expected_expiry).total_seconds()) < 60
    
    def test_create_refresh_token(self, jwt_handler, user_data):
        """Test refresh token creation."""
        token = jwt_handler.create_refresh_token(user_data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        assert token in jwt_handler.active_tokens
        
        # Check token info
        token_info = jwt_handler.active_tokens[token]
        assert token_info["user_id"] == "user_123"
        assert token_info["username"] == "testuser"
        assert token_info["type"] == "refresh"
        assert isinstance(token_info["expires_at"], datetime)
        
        # Refresh token should expire later than access token
        access_token = jwt_handler.create_access_token(user_data)
        access_info = jwt_handler.active_tokens[access_token]
        assert token_info["expires_at"] > access_info["expires_at"]
    
    def test_verify_token_valid(self, jwt_handler, user_data):
        """Test token verification with valid token."""
        token = jwt_handler.create_access_token(user_data)
        payload = jwt_handler.verify_token(token)
        
        assert payload is not None
        assert payload["sub"] == "user_123"
        assert payload["username"] == "testuser"
        assert payload["email"] == "test@example.com"
        assert payload["roles"] == ["user"]
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload
    
    def test_verify_token_invalid(self, jwt_handler):
        """Test token verification with invalid token."""
        payload = jwt_handler.verify_token("invalid_token")
        assert payload is None
    
    def test_verify_token_blacklisted(self, jwt_handler, user_data):
        """Test token verification with blacklisted token."""
        token = jwt_handler.create_access_token(user_data)
        jwt_handler.blacklisted_tokens.add(token)
        
        payload = jwt_handler.verify_token(token)
        assert payload is None
    
    def test_verify_token_not_in_active(self, jwt_handler, user_data):
        """Test token verification when token not in active tokens."""
        token = jwt_handler.create_access_token(user_data)
        
        # Remove from active tokens
        del jwt_handler.active_tokens[token]
        
        payload = jwt_handler.verify_token(token)
        assert payload is None
    
    def test_refresh_access_token_valid(self, jwt_handler, user_data):
        """Test access token refresh with valid refresh token."""
        refresh_token = jwt_handler.create_refresh_token(user_data)
        new_access_token = jwt_handler.refresh_access_token(refresh_token)
        
        assert new_access_token is not None
        assert isinstance(new_access_token, str)
        assert new_access_token != refresh_token
        assert new_access_token in jwt_handler.active_tokens
        
        # Verify new token
        payload = jwt_handler.verify_token(new_access_token)
        assert payload["sub"] == "user_123"
        assert payload["username"] == "testuser"
        assert payload["type"] == "access"
    
    def test_refresh_access_token_invalid(self, jwt_handler):
        """Test access token refresh with invalid refresh token."""
        new_token = jwt_handler.refresh_access_token("invalid_refresh_token")
        assert new_token is None
    
    def test_refresh_access_token_wrong_type(self, jwt_handler, user_data):
        """Test access token refresh with access token instead of refresh token."""
        access_token = jwt_handler.create_access_token(user_data)
        new_token = jwt_handler.refresh_access_token(access_token)
        assert new_token is None
    
    def test_revoke_token(self, jwt_handler, user_data):
        """Test token revocation."""
        token = jwt_handler.create_access_token(user_data)
        assert token in jwt_handler.active_tokens
        assert token not in jwt_handler.blacklisted_tokens
        
        success = jwt_handler.revoke_token(token)
        assert success is True
        assert token in jwt_handler.blacklisted_tokens
        assert token not in jwt_handler.active_tokens
        
        # Verify revoked token
        payload = jwt_handler.verify_token(token)
        assert payload is None
    
    def test_revoke_user_tokens(self, jwt_handler, user_data):
        """Test revoking all tokens for a user."""
        # Create multiple tokens for the user
        token1 = jwt_handler.create_access_token(user_data)
        token2 = jwt_handler.create_refresh_token(user_data)
        
        # Create token for different user
        other_user_data = {"sub": "user_456", "username": "otheruser"}
        other_token = jwt_handler.create_access_token(other_user_data)
        
        # Revoke tokens for user_123
        revoked_count = jwt_handler.revoke_user_tokens("user_123")
        
        assert revoked_count == 2
        assert token1 in jwt_handler.blacklisted_tokens
        assert token2 in jwt_handler.blacklisted_tokens
        assert other_token not in jwt_handler.blacklisted_tokens
        assert other_token in jwt_handler.active_tokens
    
    def test_cleanup_expired_tokens(self, jwt_handler, user_data):
        """Test cleanup of expired tokens."""
        # Create token with past expiry
        with patch('framework.auth.jwt_handler.datetime') as mock_datetime:
            past_time = datetime.now(timezone.utc) - timedelta(hours=2)
            mock_datetime.now.return_value = past_time
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            
            expired_token = jwt_handler.create_access_token(user_data)
        
        # Create valid token
        valid_token = jwt_handler.create_access_token(user_data)
        
        assert len(jwt_handler.active_tokens) == 2
        
        # Cleanup expired tokens
        cleaned_count = jwt_handler.cleanup_expired_tokens()
        
        assert cleaned_count == 1
        assert expired_token not in jwt_handler.active_tokens
        assert valid_token in jwt_handler.active_tokens
    
    def test_get_token_info(self, jwt_handler, user_data):
        """Test getting token information."""
        token = jwt_handler.create_access_token(user_data)
        token_info = jwt_handler.get_token_info(token)
        
        assert token_info is not None
        assert token_info["user_id"] == "user_123"
        assert token_info["username"] == "testuser"
        assert token_info["type"] == "access"
        
        # Non-existent token
        info = jwt_handler.get_token_info("nonexistent")
        assert info is None
    
    def test_is_token_active(self, jwt_handler, user_data):
        """Test token active status check."""
        token = jwt_handler.create_access_token(user_data)
        
        # Active token
        assert jwt_handler.is_token_active(token) is True
        
        # Blacklisted token
        jwt_handler.blacklisted_tokens.add(token)
        assert jwt_handler.is_token_active(token) is False
        
        # Remove from blacklist and active tokens
        jwt_handler.blacklisted_tokens.remove(token)
        del jwt_handler.active_tokens[token]
        assert jwt_handler.is_token_active(token) is False
    
    def test_is_token_active_expired(self, jwt_handler, user_data):
        """Test token active status with expired token."""
        # Create token with past expiry
        with patch('framework.auth.jwt_handler.datetime') as mock_datetime:
            past_time = datetime.now(timezone.utc) - timedelta(hours=2)
            mock_datetime.now.return_value = past_time
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            
            token = jwt_handler.create_access_token(user_data)
        
        # Token should be inactive due to expiry
        assert jwt_handler.is_token_active(token) is False
    
    def test_get_active_token_count(self, jwt_handler, user_data):
        """Test getting active token count."""
        assert jwt_handler.get_active_token_count() == 0
        
        token1 = jwt_handler.create_access_token(user_data)
        assert jwt_handler.get_active_token_count() == 1
        
        token2 = jwt_handler.create_refresh_token(user_data)
        assert jwt_handler.get_active_token_count() == 2
        
        jwt_handler.revoke_token(token1)
        assert jwt_handler.get_active_token_count() == 1
    
    def test_get_user_token_count(self, jwt_handler, user_data):
        """Test getting token count for specific user."""
        assert jwt_handler.get_user_token_count("user_123") == 0
        
        # Create tokens for user_123
        jwt_handler.create_access_token(user_data)
        jwt_handler.create_refresh_token(user_data)
        assert jwt_handler.get_user_token_count("user_123") == 2
        
        # Create token for different user
        other_user_data = {"sub": "user_456", "username": "otheruser"}
        jwt_handler.create_access_token(other_user_data)
        
        assert jwt_handler.get_user_token_count("user_123") == 2
        assert jwt_handler.get_user_token_count("user_456") == 1
        assert jwt_handler.get_user_token_count("nonexistent") == 0
    
    def test_create_token_error_handling(self, jwt_handler):
        """Test error handling in token creation."""
        with patch('framework.auth.jwt_handler.jwt.encode') as mock_encode:
            mock_encode.side_effect = Exception("Encoding error")
            
            with pytest.raises(Exception):
                jwt_handler.create_access_token({"sub": "user"})
    
    def test_refresh_token_error_handling(self, jwt_handler):
        """Test error handling in token refresh."""
        with patch.object(jwt_handler, 'create_access_token') as mock_create:
            mock_create.side_effect = Exception("Creation error")
            
            refresh_token = jwt_handler.create_refresh_token({"sub": "user", "username": "test"})
            result = jwt_handler.refresh_access_token(refresh_token)
            assert result is None


class TestJWTHandlerWithoutJose:
    """Test JWT handler behavior when jose library is not available."""
    
    @patch('framework.auth.jwt_handler.JWT_AVAILABLE', False)
    def test_initialization_without_jose(self):
        """Test that JWTHandler raises error when jose is not available."""
        with pytest.raises(RuntimeError) as exc_info:
            JWTHandler()
        assert "python-jose is required" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])

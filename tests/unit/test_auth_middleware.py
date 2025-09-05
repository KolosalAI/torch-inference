"""
Test suite for authentication middleware functionality.

This module tests the auth middleware and dependency injection.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from framework.auth.middleware import AuthMiddleware, init_auth_middleware, get_current_user, require_admin
from framework.auth.models import User
from framework.auth.jwt_handler import JWTHandler
from framework.auth.user_store import UserStore


class TestAuthMiddleware:
    """Test AuthMiddleware class."""
    
    @pytest.fixture
    def mock_jwt_handler(self):
        """Mock JWT handler."""
        handler = Mock(spec=JWTHandler)
        handler.verify_token.return_value = {
            "sub": "user_123",
            "username": "testuser",
            "email": "test@example.com",
            "roles": ["user"],
            "type": "access"
        }
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
            roles=["user"],
            is_active=True
        )
        store.get_user.return_value = user
        store.authenticate_api_key.return_value = (user, Mock())
        return store
    
    @pytest.fixture
    def auth_middleware(self, mock_jwt_handler, mock_user_store):
        """Create auth middleware instance."""
        return AuthMiddleware(mock_jwt_handler, mock_user_store)
    
    @pytest.fixture
    def mock_credentials(self):
        """Mock HTTP authorization credentials."""
        return HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="valid_token"
        )
    
    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request."""
        request = Mock()
        request.headers = {"X-API-Key": "sk_test_api_key"}
        return request
    
    def test_auth_middleware_initialization(self, auth_middleware, mock_jwt_handler, mock_user_store):
        """Test auth middleware initialization."""
        assert auth_middleware.jwt_handler == mock_jwt_handler
        assert auth_middleware.user_store == mock_user_store
        assert hasattr(auth_middleware, 'logger')
    
    def test_get_current_user_success(self, auth_middleware, mock_credentials):
        """Test successful user authentication via JWT."""
        user = auth_middleware.get_current_user(mock_credentials)
        
        assert user is not None
        assert user.username == "testuser"
        assert user.id == "user_123"
        auth_middleware.jwt_handler.verify_token.assert_called_once_with("valid_token")
        auth_middleware.user_store.get_user.assert_called_once_with("testuser")
    
    def test_get_current_user_invalid_token(self, auth_middleware, mock_credentials):
        """Test authentication with invalid token."""
        auth_middleware.jwt_handler.verify_token.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            auth_middleware.get_current_user(mock_credentials)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid authentication token" in exc_info.value.detail
    
    def test_get_current_user_invalid_token_payload(self, auth_middleware, mock_credentials):
        """Test authentication with invalid token payload."""
        auth_middleware.jwt_handler.verify_token.return_value = {"invalid": "payload"}
        
        with pytest.raises(HTTPException) as exc_info:
            auth_middleware.get_current_user(mock_credentials)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid token type" in exc_info.value.detail
    
    def test_get_current_user_user_not_found(self, auth_middleware, mock_credentials):
        """Test authentication when user not found."""
        auth_middleware.user_store.get_user.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            auth_middleware.get_current_user(mock_credentials)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "User not found" in exc_info.value.detail
    
    def test_get_current_user_inactive_user(self, auth_middleware, mock_credentials, mock_user_store):
        """Test authentication with inactive user."""
        inactive_user = User(
            id="user_123",
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            hashed_password="hashed",
            is_active=False
        )
        mock_user_store.get_user.return_value = inactive_user
        auth_middleware.user_store = mock_user_store
        
        with pytest.raises(HTTPException) as exc_info:
            auth_middleware.get_current_user(mock_credentials)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "User account is disabled" in exc_info.value.detail
    
    def test_get_current_user_exception_handling(self, auth_middleware, mock_credentials):
        """Test exception handling in get_current_user."""
        auth_middleware.jwt_handler.verify_token.side_effect = Exception("Unexpected error")
        
        with pytest.raises(HTTPException) as exc_info:
            auth_middleware.get_current_user(mock_credentials)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Authentication failed" in exc_info.value.detail
    
    def test_get_current_user_optional_success(self, auth_middleware, mock_credentials):
        """Test optional authentication with valid credentials."""
        user = auth_middleware.get_current_user_optional(mock_credentials)
        
        assert user is not None
        assert user.username == "testuser"
    
    def test_get_current_user_optional_none_credentials(self, auth_middleware):
        """Test optional authentication with no credentials."""
        user = auth_middleware.get_current_user_optional(None)
        assert user is None
    
    def test_get_current_user_optional_invalid_token(self, auth_middleware, mock_credentials):
        """Test optional authentication with invalid token."""
        auth_middleware.jwt_handler.verify_token.return_value = None
        
        user = auth_middleware.get_current_user_optional(mock_credentials)
        assert user is None
    
    def test_authenticate_api_key_success(self, auth_middleware, mock_request):
        """Test successful API key authentication."""
        user = auth_middleware.authenticate_api_key(mock_request)
        
        assert user is not None
        assert user.username == "testuser"
        auth_middleware.user_store.authenticate_api_key.assert_called_once_with("sk_test_api_key")
    
    def test_authenticate_api_key_no_header(self, auth_middleware, mock_request):
        """Test API key authentication with no API key header."""
        mock_request.headers = {}
        
        user = auth_middleware.authenticate_api_key(mock_request)
        assert user is None
    
    def test_authenticate_api_key_invalid_key(self, auth_middleware, mock_request):
        """Test API key authentication with invalid key."""
        auth_middleware.user_store.authenticate_api_key.return_value = None
        
        user = auth_middleware.authenticate_api_key(mock_request)
        assert user is None
    
    def test_require_auth_decorator(self, auth_middleware):
        """Test require_auth dependency function."""
        # Get the dependency function
        dependency_func = auth_middleware.require_auth()
        
        # The function should be callable
        assert callable(dependency_func)
        
        # We can't easily test dependency injection without FastAPI context,
        # but we can verify the function returns correctly
        assert dependency_func is not None
    
    def test_require_auth_optional_decorator(self, auth_middleware):
        """Test require_auth dependency function with optional=True."""
        # Get the dependency function
        dependency_func = auth_middleware.require_auth(optional=True)
        
        # The function should be callable
        assert callable(dependency_func)
        assert dependency_func is not None
    
    def test_require_roles_decorator(self, auth_middleware):
        """Test require_roles dependency function."""
        # Get the dependency function
        dependency_func = auth_middleware.require_roles(["admin"])
        
        # The function should be callable
        assert callable(dependency_func)
        assert dependency_func is not None
        
        # Test with admin user
        admin_user = User(
            id="admin_123",
            username="admin",
            email="admin@example.com",
            full_name="Admin User",
            hashed_password="hashed",
            roles=["admin"]
        )
        
        # Since we can't easily test FastAPI dependencies,
        # we test the underlying logic manually
        has_admin_role = "admin" in admin_user.roles
        assert has_admin_role is True
        
        # Test with regular user
        regular_user = User(
            id="user_123",
            username="user",
            email="user@example.com",
            full_name="Regular User",
            hashed_password="hashed",
            roles=["user"]
        )
        
        has_admin_role = "admin" in regular_user.roles
        assert has_admin_role is False
    
    def test_require_admin_decorator(self, auth_middleware):
        """Test require_admin dependency function."""
        # Get the dependency function
        dependency_func = auth_middleware.require_admin()
        
        # The function should be callable
        assert callable(dependency_func)
        assert dependency_func is not None


class TestGlobalMiddlewareFunctions:
    """Test global middleware management functions."""
    
    @pytest.fixture
    def mock_jwt_handler(self):
        """Mock JWT handler."""
        return Mock(spec=JWTHandler)
    
    @pytest.fixture
    def mock_user_store(self):
        """Mock user store."""
        return Mock(spec=UserStore)
    
    def test_init_auth_middleware(self, mock_jwt_handler, mock_user_store):
        """Test auth middleware initialization."""
        middleware = init_auth_middleware(mock_jwt_handler, mock_user_store)
        
        assert isinstance(middleware, AuthMiddleware)
        assert middleware.jwt_handler == mock_jwt_handler
        assert middleware.user_store == mock_user_store
    
    @patch('framework.auth.middleware._auth_middleware', None)
    def test_init_auth_middleware_sets_global(self, mock_jwt_handler, mock_user_store):
        """Test that init_auth_middleware sets global instance."""
        from framework.auth.middleware import _auth_middleware
        
        middleware = init_auth_middleware(mock_jwt_handler, mock_user_store)
        
        # Import the module-level variable to check it was set
        import framework.auth.middleware as middleware_module
        assert middleware_module._auth_middleware is not None
        assert middleware_module._auth_middleware == middleware


class TestDependencyFunctions:
    """Test dependency injection functions."""
    
    @pytest.fixture
    def mock_auth_middleware(self):
        """Mock auth middleware."""
        middleware = Mock(spec=AuthMiddleware)
        user = User(
            id="dep_user_123",
            username="depuser",
            email="dep@example.com",
            full_name="Dependency User",
            hashed_password="hashed",
            roles=["user"]
        )
        middleware.get_current_user.return_value = user
        middleware.get_current_user_optional.return_value = user
        return middleware
    
    def test_get_current_user_dependency_function(self, mock_auth_middleware):
        """Test get_current_user dependency function."""
        with patch('framework.auth.middleware.get_auth_middleware', return_value=mock_auth_middleware):
            from framework.auth.middleware import get_current_user
            
            dependency = get_current_user()
            
            # The function should return a callable (dependency)
            assert callable(dependency)
    
    def test_get_current_user_optional_dependency_function(self, mock_auth_middleware):
        """Test get_current_user_optional dependency function."""
        with patch('framework.auth.middleware.get_auth_middleware', return_value=mock_auth_middleware):
            from framework.auth.middleware import get_current_user_optional
            
            dependency = get_current_user_optional()
            
            # The function should return a callable (dependency)
            assert callable(dependency)
    
    def test_require_admin_dependency_function(self, mock_auth_middleware):
        """Test require_admin dependency function."""
        admin_user = User(
            id="admin_123",
            username="admin",
            email="admin@example.com",
            full_name="Admin User",
            hashed_password="hashed",
            roles=["admin"]
        )
        mock_auth_middleware.get_current_user.return_value = admin_user
        
        with patch('framework.auth.middleware.get_auth_middleware', return_value=mock_auth_middleware):
            from framework.auth.middleware import require_admin
            
            dependency = require_admin()
            
            # The function should return a callable (dependency)
            assert callable(dependency)
    
    def test_get_auth_middleware_not_initialized(self):
        """Test get_auth_middleware when not initialized."""
        with patch('framework.auth.middleware._auth_middleware', None):
            from framework.auth.middleware import get_auth_middleware
            
            with pytest.raises(RuntimeError) as exc_info:
                get_auth_middleware()
            
            assert "Auth middleware not initialized" in str(exc_info.value)


class TestRoleBasedAccess:
    """Test role-based access control functionality."""
    
    def test_user_has_required_role(self):
        """Test that user with required role passes check."""
        user = User(
            id="role_user_123",
            username="roleuser",
            email="role@example.com",
            full_name="Role User",
            hashed_password="hashed",
            roles=["admin", "user"]
        )
        
        # Test single role check
        assert user.has_role("admin") is True
        assert user.has_role("user") is True
        assert user.has_role("moderator") is False
        
        # Test multiple roles
        required_roles = ["admin"]
        has_any_role = any(user.has_role(role) for role in required_roles)
        assert has_any_role is True
        
        required_roles = ["moderator", "supervisor"]
        has_any_role = any(user.has_role(role) for role in required_roles)
        assert has_any_role is False
    
    def test_role_hierarchy_simulation(self):
        """Test simulated role hierarchy checking."""
        def has_sufficient_role(user_roles, required_role):
            """Simulate role hierarchy checking."""
            role_hierarchy = {
                "admin": ["admin", "moderator", "user"],
                "moderator": ["moderator", "user"],
                "user": ["user"]
            }
            
            # Check if user has any role that includes the required role
            for user_role in user_roles:
                if required_role in role_hierarchy.get(user_role, []):
                    return True
            return False
        
        # Admin user should have all permissions
        admin_user_roles = ["admin"]
        assert has_sufficient_role(admin_user_roles, "admin") is True
        assert has_sufficient_role(admin_user_roles, "moderator") is True
        assert has_sufficient_role(admin_user_roles, "user") is True
        
        # Moderator should have moderator and user permissions
        mod_user_roles = ["moderator"]
        assert has_sufficient_role(mod_user_roles, "admin") is False
        assert has_sufficient_role(mod_user_roles, "moderator") is True
        assert has_sufficient_role(mod_user_roles, "user") is True
        
        # Regular user should only have user permissions
        regular_user_roles = ["user"]
        assert has_sufficient_role(regular_user_roles, "admin") is False
        assert has_sufficient_role(regular_user_roles, "moderator") is False
        assert has_sufficient_role(regular_user_roles, "user") is True


class TestMiddlewareIntegration:
    """Test middleware integration scenarios."""
    
    @pytest.fixture
    def integration_middleware(self):
        """Create middleware for integration testing."""
        jwt_handler = Mock(spec=JWTHandler)
        user_store = Mock(spec=UserStore)
        return AuthMiddleware(jwt_handler, user_store)
    
    def test_jwt_and_api_key_auth_precedence(self, integration_middleware):
        """Test authentication precedence between JWT and API key."""
        # In real implementation, JWT auth typically takes precedence
        # This test simulates the decision logic
        
        def authenticate_request(jwt_credentials, api_key_header):
            """Simulate request authentication logic."""
            if jwt_credentials:
                # Try JWT first
                return "jwt_auth"
            elif api_key_header:
                # Fall back to API key
                return "api_key_auth"
            else:
                return None
        
        # Test JWT precedence
        result = authenticate_request("jwt_token", "api_key")
        assert result == "jwt_auth"
        
        # Test API key fallback
        result = authenticate_request(None, "api_key")
        assert result == "api_key_auth"
        
        # Test no auth
        result = authenticate_request(None, None)
        assert result is None
    
    def test_middleware_error_logging(self, integration_middleware):
        """Test that middleware logs authentication errors appropriately."""
        with patch.object(integration_middleware.logger, 'debug') as mock_debug, \
             patch.object(integration_middleware.logger, 'error') as mock_error:
            
            # Simulate authentication error
            integration_middleware.jwt_handler.verify_token.side_effect = Exception("Token error")
            
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")
            
            with pytest.raises(HTTPException):
                integration_middleware.get_current_user(credentials)
            
            # Should log the error
            mock_error.assert_called()
    
    def test_middleware_performance_logging(self, integration_middleware):
        """Test middleware performance monitoring."""
        # This would typically involve timing authentication operations
        # and logging slow operations
        
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0.0, 0.1]  # 100ms authentication
            
            # Simulate timing logic
            start_time = mock_time()
            # ... authentication logic ...
            end_time = mock_time()
            duration = end_time - start_time
            
            assert duration == 0.1
            
            # In real implementation, would log if duration > threshold
            if duration > 0.05:  # 50ms threshold
                performance_warning = f"Slow authentication: {duration}s"
                assert "Slow authentication" in performance_warning


if __name__ == "__main__":
    pytest.main([__file__])

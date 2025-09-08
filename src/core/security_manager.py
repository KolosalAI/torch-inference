"""
Security manager for PyTorch Inference Framework.

This module provides security features including authentication,
authorization, input validation, and security monitoring.
"""

import logging
import hashlib
import hmac
import secrets
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import jwt
import bcrypt
from functools import wraps

from .exceptions import SecurityError, ValidationError, ConfigurationError
from .config import SecurityConfig

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    ADMIN = "admin"
    SYSTEM = "system"


@dataclass
class SecurityEvent:
    """Security event for logging and monitoring."""
    event_type: str
    user_id: Optional[str]
    source_ip: Optional[str]
    timestamp: datetime
    details: Dict[str, Any]
    severity: str = "info"


@dataclass
class User:
    """User information for authentication."""
    user_id: str
    username: str
    email: Optional[str]
    roles: List[str]
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool = True


@dataclass
class APIKey:
    """API Key information."""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    permissions: List[str]
    expires_at: Optional[datetime]
    created_at: datetime
    last_used: Optional[datetime]
    is_active: bool = True


class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, identifier: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed under rate limit.
        
        Args:
            identifier: Unique identifier (IP, user ID, etc.)
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
        else:
            self.requests[identifier] = []
        
        # Check limit
        current_requests = len(self.requests[identifier])
        allowed = current_requests < self.max_requests
        
        if allowed:
            self.requests[identifier].append(now)
        
        return allowed, {
            "current_requests": current_requests,
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "reset_time": window_start + self.window_seconds,
            "allowed": allowed
        }


class InputValidator:
    """Input validation and sanitization."""
    
    @staticmethod
    def validate_text_input(text: str, max_length: int = 10000, 
                          allow_html: bool = False) -> str:
        """
        Validate and sanitize text input.
        
        Args:
            text: Input text to validate
            max_length: Maximum allowed length
            allow_html: Whether to allow HTML tags
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            raise ValidationError(
                field="text",
                details="Input must be a string"
            )
        
        if len(text) > max_length:
            raise ValidationError(
                field="text",
                details=f"Input too long (max {max_length} characters)"
            )
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Basic HTML sanitization if not allowed
        if not allow_html:
            text = text.replace('<', '&lt;').replace('>', '&gt;')
        
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text 
                      if ord(char) >= 32 or char in '\n\t\r')
        
        return text.strip()
    
    @staticmethod
    def validate_file_upload(file_data: bytes, max_size: int = 10 * 1024 * 1024,
                           allowed_types: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Validate file upload.
        
        Args:
            file_data: File data bytes
            max_size: Maximum file size in bytes
            allowed_types: Set of allowed MIME types
            
        Returns:
            Validation result
        """
        if len(file_data) > max_size:
            raise ValidationError(
                field="file",
                details=f"File too large (max {max_size} bytes)"
            )
        
        # Basic file type detection (simplified)
        file_type = "application/octet-stream"
        if file_data.startswith(b'\x89PNG'):
            file_type = "image/png"
        elif file_data.startswith(b'\xff\xd8\xff'):
            file_type = "image/jpeg"
        elif file_data.startswith(b'RIFF') and b'WAVE' in file_data[:20]:
            file_type = "audio/wav"
        elif file_data.startswith(b'ID3') or file_data.startswith(b'\xff\xfb'):
            file_type = "audio/mpeg"
        
        if allowed_types and file_type not in allowed_types:
            raise ValidationError(
                field="file_type",
                details=f"File type {file_type} not allowed"
            )
        
        return {
            "file_type": file_type,
            "file_size": len(file_data),
            "valid": True
        }


class SecurityManager:
    """
    Main security manager for the inference framework.
    """
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize the security manager.
        
        Args:
            config: Security configuration
        """
        self.config = config
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.security_events: List[SecurityEvent] = []
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.input_validator = InputValidator()
        
        # Initialize rate limiters
        if config.rate_limiting.enabled:
            self.rate_limiters["default"] = RateLimiter(
                config.rate_limiting.requests_per_minute,
                60
            )
            self.rate_limiters["api"] = RateLimiter(
                config.rate_limiting.requests_per_minute * 2,  # Higher limit for API keys
                60
            )
        
        # JWT secret key
        self.jwt_secret = config.jwt_secret or secrets.token_urlsafe(32)
        
        # Create default admin user if enabled
        if config.create_default_admin:
            self._create_default_admin()
        
        logger.debug("SecurityManager initialized")
    
    def _create_default_admin(self):
        """Create default admin user for initial setup."""
        admin_user = User(
            user_id="admin",
            username="admin",
            email="admin@localhost",
            roles=["admin"],
            permissions=["*"],
            created_at=datetime.utcnow(),
            last_login=None,
            is_active=True
        )
        self.users["admin"] = admin_user
        logger.info("Default admin user created")
    
    # Authentication methods
    
    def create_user(self, username: str, password: str, email: Optional[str] = None,
                   roles: Optional[List[str]] = None) -> User:
        """
        Create a new user.
        
        Args:
            username: Username
            password: Password
            email: Optional email address
            roles: Optional list of roles
            
        Returns:
            Created user
        """
        try:
            # Validate username
            if len(username) < 3 or len(username) > 50:
                raise ValidationError(
                    field="username",
                    details="Username must be 3-50 characters"
                )
            
            if username in [user.username for user in self.users.values()]:
                raise ValidationError(
                    field="username",
                    details="Username already exists"
                )
            
            # Validate password
            if len(password) < 8:
                raise ValidationError(
                    field="password",
                    details="Password must be at least 8 characters"
                )
            
            user_id = secrets.token_urlsafe(16)
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                roles=roles or ["user"],
                permissions=self._get_permissions_for_roles(roles or ["user"]),
                created_at=datetime.utcnow(),
                last_login=None,
                is_active=True
            )
            
            self.users[user_id] = user
            
            # Log security event
            self._log_security_event(
                "user_created",
                user_id=user_id,
                details={"username": username, "roles": roles or ["user"]}
            )
            
            logger.info(f"User created: {username}")
            return user
            
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise SecurityError(
                operation="create_user",
                details=f"Failed to create user: {e}",
                cause=e
            )
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Authenticated user or None
        """
        try:
            # Find user by username
            user = None
            for u in self.users.values():
                if u.username == username:
                    user = u
                    break
            
            if not user or not user.is_active:
                self._log_security_event(
                    "authentication_failed",
                    details={"username": username, "reason": "user_not_found"}
                )
                return None
            
            # For demo purposes, we'll accept any password for existing users
            # In production, you'd verify against stored password hash
            if password:  # Basic validation
                user.last_login = datetime.utcnow()
                
                self._log_security_event(
                    "authentication_success",
                    user_id=user.user_id,
                    details={"username": username}
                )
                
                return user
            
            self._log_security_event(
                "authentication_failed",
                details={"username": username, "reason": "invalid_password"}
            )
            return None
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            self._log_security_event(
                "authentication_error",
                details={"username": username, "error": str(e)}
            )
            return None
    
    def create_jwt_token(self, user: User, expires_in_hours: int = 24) -> str:
        """
        Create JWT token for user.
        
        Args:
            user: User to create token for
            expires_in_hours: Token expiration time
            
        Returns:
            JWT token string
        """
        try:
            payload = {
                "user_id": user.user_id,
                "username": user.username,
                "roles": user.roles,
                "permissions": user.permissions,
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(hours=expires_in_hours)
            }
            
            token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
            
            self._log_security_event(
                "token_created",
                user_id=user.user_id,
                details={"expires_in_hours": expires_in_hours}
            )
            
            return token
            
        except Exception as e:
            logger.error(f"Failed to create JWT token: {e}")
            raise SecurityError(
                operation="create_token",
                details=f"Failed to create token: {e}",
                cause=e
            )
    
    def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Token payload or None if invalid
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check if user still exists and is active
            user = self.users.get(payload["user_id"])
            if not user or not user.is_active:
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            self._log_security_event(
                "token_expired",
                details={"token": token[:20] + "..."}
            )
            return None
        except jwt.InvalidTokenError as e:
            self._log_security_event(
                "token_invalid",
                details={"error": str(e), "token": token[:20] + "..."}
            )
            return None
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return None
    
    # API Key management
    
    def create_api_key(self, user_id: str, name: str, 
                      permissions: Optional[List[str]] = None,
                      expires_in_days: Optional[int] = None) -> Tuple[str, APIKey]:
        """
        Create API key for user.
        
        Args:
            user_id: User ID
            name: API key name
            permissions: Optional permissions list
            expires_in_days: Optional expiration in days
            
        Returns:
            Tuple of (api_key_string, api_key_object)
        """
        try:
            if user_id not in self.users:
                raise ValidationError(
                    field="user_id",
                    details="User not found"
                )
            
            # Generate API key
            api_key_string = f"tif_{secrets.token_urlsafe(32)}"
            key_hash = hashlib.sha256(api_key_string.encode()).hexdigest()
            key_id = secrets.token_urlsafe(16)
            
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            api_key = APIKey(
                key_id=key_id,
                key_hash=key_hash,
                user_id=user_id,
                name=name,
                permissions=permissions or [],
                expires_at=expires_at,
                created_at=datetime.utcnow(),
                last_used=None,
                is_active=True
            )
            
            self.api_keys[key_id] = api_key
            
            self._log_security_event(
                "api_key_created",
                user_id=user_id,
                details={"key_id": key_id, "name": name}
            )
            
            return api_key_string, api_key
            
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise SecurityError(
                operation="create_api_key",
                details=f"Failed to create API key: {e}",
                cause=e
            )
    
    def validate_api_key(self, api_key_string: str) -> Optional[APIKey]:
        """
        Validate API key.
        
        Args:
            api_key_string: API key string
            
        Returns:
            API key object or None if invalid
        """
        try:
            if not api_key_string.startswith("tif_"):
                return None
            
            key_hash = hashlib.sha256(api_key_string.encode()).hexdigest()
            
            # Find matching API key
            for api_key in self.api_keys.values():
                if api_key.key_hash == key_hash and api_key.is_active:
                    # Check expiration
                    if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
                        return None
                    
                    # Update last used
                    api_key.last_used = datetime.utcnow()
                    return api_key
            
            return None
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return None
    
    # Authorization methods
    
    def check_permission(self, user_or_token: Any, required_permission: str) -> bool:
        """
        Check if user has required permission.
        
        Args:
            user_or_token: User object or token payload
            required_permission: Required permission string
            
        Returns:
            True if authorized
        """
        try:
            permissions = []
            
            if isinstance(user_or_token, User):
                permissions = user_or_token.permissions
            elif isinstance(user_or_token, dict):  # Token payload
                permissions = user_or_token.get("permissions", [])
            elif isinstance(user_or_token, APIKey):
                permissions = user_or_token.permissions
            
            # Check for wildcard permission
            if "*" in permissions:
                return True
            
            # Check exact permission
            if required_permission in permissions:
                return True
            
            # Check role-based permissions
            return False
            
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False
    
    def _get_permissions_for_roles(self, roles: List[str]) -> List[str]:
        """Get permissions for given roles."""
        role_permissions = {
            "admin": ["*"],
            "user": ["inference:predict", "models:list", "models:info"],
            "readonly": ["models:list", "models:info"]
        }
        
        permissions = set()
        for role in roles:
            permissions.update(role_permissions.get(role, []))
        
        return list(permissions)
    
    # Rate limiting
    
    def check_rate_limit(self, identifier: str, 
                        limiter_type: str = "default") -> Tuple[bool, Dict[str, Any]]:
        """
        Check rate limit for identifier.
        
        Args:
            identifier: Unique identifier
            limiter_type: Type of rate limiter to use
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        if not self.config.rate_limiting.enabled:
            return True, {"allowed": True, "rate_limiting_disabled": True}
        
        limiter = self.rate_limiters.get(limiter_type)
        if not limiter:
            return True, {"allowed": True, "no_limiter": True}
        
        allowed, info = limiter.is_allowed(identifier)
        
        if not allowed:
            self._log_security_event(
                "rate_limit_exceeded",
                details={
                    "identifier": identifier,
                    "limiter_type": limiter_type,
                    "rate_limit_info": info
                }
            )
        
        return allowed, info
    
    # Input validation
    
    def validate_request_input(self, input_data: Any, 
                             validation_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Validate request input data.
        
        Args:
            input_data: Input data to validate
            validation_config: Optional validation configuration
            
        Returns:
            Validated and sanitized input data
        """
        try:
            config = validation_config or {}
            
            if isinstance(input_data, str):
                return self.input_validator.validate_text_input(
                    input_data,
                    max_length=config.get("max_text_length", 10000),
                    allow_html=config.get("allow_html", False)
                )
            elif isinstance(input_data, bytes):
                return self.input_validator.validate_file_upload(
                    input_data,
                    max_size=config.get("max_file_size", 10 * 1024 * 1024),
                    allowed_types=config.get("allowed_file_types")
                )
            elif isinstance(input_data, dict):
                # Recursively validate dictionary values
                validated = {}
                for key, value in input_data.items():
                    validated[key] = self.validate_request_input(value, config)
                return validated
            elif isinstance(input_data, list):
                # Validate list items
                return [self.validate_request_input(item, config) for item in input_data]
            else:
                return input_data
                
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise ValidationError(
                field="input_data",
                details=f"Input validation failed: {e}",
                cause=e
            )
    
    # Security monitoring
    
    def _log_security_event(self, event_type: str, user_id: Optional[str] = None,
                          source_ip: Optional[str] = None, 
                          details: Optional[Dict[str, Any]] = None,
                          severity: str = "info"):
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            source_ip=source_ip,
            timestamp=datetime.utcnow(),
            details=details or {},
            severity=severity
        )
        
        self.security_events.append(event)
        
        # Keep only recent events
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        self.security_events = [
            e for e in self.security_events if e.timestamp > cutoff_time
        ]
        
        # Log high severity events
        if severity in ["warning", "error", "critical"]:
            logger.warning(f"Security event: {event_type} - {details}")
    
    def get_security_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent security events."""
        events = sorted(self.security_events, key=lambda x: x.timestamp, reverse=True)
        return [
            {
                "event_type": e.event_type,
                "user_id": e.user_id,
                "source_ip": e.source_ip,
                "timestamp": e.timestamp.isoformat(),
                "details": e.details,
                "severity": e.severity
            }
            for e in events[:limit]
        ]
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        return {
            "total_users": len(self.users),
            "active_users": sum(1 for u in self.users.values() if u.is_active),
            "total_api_keys": len(self.api_keys),
            "active_api_keys": sum(1 for k in self.api_keys.values() if k.is_active),
            "security_events_count": len(self.security_events),
            "rate_limiting_enabled": self.config.rate_limiting.enabled,
            "authentication_enabled": self.config.enable_auth
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get security manager health status."""
        return {
            "healthy": True,
            "authentication_enabled": self.config.enable_auth,
            "rate_limiting_enabled": self.config.rate_limiting.enabled,
            "total_users": len(self.users),
            "total_api_keys": len(self.api_keys),
            "recent_events": len([
                e for e in self.security_events 
                if e.timestamp > datetime.utcnow() - timedelta(hours=1)
            ])
        }


# Factory function

def create_security_manager(config: SecurityConfig) -> SecurityManager:
    """
    Create a security manager instance.
    
    Args:
        config: Security configuration
        
    Returns:
        Security manager instance
    """
    try:
        security_manager = SecurityManager(config)
        logger.info("Security manager created successfully")
        return security_manager
        
    except Exception as e:
        logger.error(f"Failed to create security manager: {e}")
        raise ConfigurationError(
            config_field="security",
            details=f"Failed to create security manager: {e}",
            cause=e
        )

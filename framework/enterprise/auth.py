"""
Enterprise authentication and authorization system.

This module provides comprehensive authentication and authorization features including:
- JWT-based authentication
- OAuth2/OIDC integration
- Role-based access control (RBAC)
- Multi-factor authentication
- API key management
- Session management
"""

import time
import secrets
import hashlib
import base64
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
import pyotp
import qrcode
from io import BytesIO

from .config import EnterpriseConfig, AuthProvider


logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions."""
    # Model permissions
    MODEL_CREATE = "model:create"
    MODEL_READ = "model:read"
    MODEL_UPDATE = "model:update"
    MODEL_DELETE = "model:delete"
    MODEL_DEPLOY = "model:deploy"
    
    # Inference permissions
    INFERENCE_PREDICT = "inference:predict"
    INFERENCE_BATCH = "inference:batch"
    INFERENCE_STREAM = "inference:stream"
    
    # Metrics permissions
    METRICS_READ = "metrics:read"
    METRICS_ADMIN = "metrics:admin"
    
    # Admin permissions
    USER_MANAGE = "user:manage"
    ROLE_MANAGE = "role:manage"
    SYSTEM_ADMIN = "system:admin"
    
    # Experiment permissions
    EXPERIMENT_CREATE = "experiment:create"
    EXPERIMENT_READ = "experiment:read"
    EXPERIMENT_UPDATE = "experiment:update"
    EXPERIMENT_DELETE = "experiment:delete"


@dataclass
class User:
    """User model."""
    id: str
    username: str
    email: str
    full_name: str
    roles: List[str]
    tenant_id: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    created_at: datetime = None
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
    
    def has_permission(self, permission: Union[Permission, str], rbac_manager: 'RBACManager') -> bool:
        """Check if user has specific permission."""
        if isinstance(permission, Permission):
            permission = permission.value
        
        return rbac_manager.user_has_permission(self.id, permission)
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role."""
        return role in self.roles
    
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary (excluding sensitive data)."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "roles": self.roles,
            "tenant_id": self.tenant_id,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "mfa_enabled": self.mfa_enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }


@dataclass
class APIKey:
    """API Key model."""
    id: str
    name: str
    key_hash: str
    user_id: str
    scopes: List[str]
    is_active: bool = True
    expires_at: Optional[datetime] = None
    created_at: datetime = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
    
    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if API key is valid."""
        return self.is_active and not self.is_expired()


@dataclass
class Session:
    """User session model."""
    id: str
    user_id: str
    access_token: str
    refresh_token: Optional[str]
    expires_at: datetime
    created_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if session is valid."""
        return self.is_active and not self.is_expired()


class AuthProvider(ABC):
    """Abstract authentication provider."""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[User]:
        """Authenticate user with credentials."""
        pass
    
    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        pass


class JWTManager:
    """JWT token management."""
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.secret_key = config.auth.secret_key
        self.algorithm = config.auth.algorithm
        self.access_token_expire = timedelta(minutes=config.auth.access_token_expire_minutes)
        self.refresh_token_expire = timedelta(days=config.auth.refresh_token_expire_days)
    
    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create access token for user."""
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + self.access_token_expire
        
        payload = {
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "roles": user.roles,
            "tenant_id": user.tenant_id,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user: User) -> str:
        """Create refresh token for user."""
        expire = datetime.now(timezone.utc) + self.refresh_token_expire
        
        payload = {
            "sub": user.id,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") not in ["access", "refresh"]:
                return None
            
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Create new access token from refresh token."""
        payload = self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None
        
        # In production, you would fetch the user from database
        # For now, we'll create a minimal token
        expire = datetime.now(timezone.utc) + self.access_token_expire
        new_payload = {
            "sub": payload["sub"],
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access"
        }
        
        return jwt.encode(new_payload, self.secret_key, algorithm=self.algorithm)


class PasswordManager:
    """Password hashing and validation."""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def generate_password(self, length: int = 16) -> str:
        """Generate secure random password."""
        return secrets.token_urlsafe(length)


class MFAManager:
    """Multi-factor authentication management."""
    
    def __init__(self, issuer: str = "TorchInference"):
        self.issuer = issuer
    
    def generate_secret(self) -> str:
        """Generate TOTP secret for user."""
        return pyotp.random_base32()
    
    def generate_qr_code(self, user: User, secret: str) -> bytes:
        """Generate QR code for TOTP setup."""
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=user.email,
            issuer_name=self.issuer
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = BytesIO()
        img.save(img_buffer, format="PNG")
        return img_buffer.getvalue()
    
    def verify_totp(self, secret: str, token: str, window: int = 1) -> bool:
        """Verify TOTP token."""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=window)
    
    def generate_backup_codes(self, count: int = 8) -> List[str]:
        """Generate backup codes for account recovery."""
        return [secrets.token_hex(4).upper() for _ in range(count)]


class APIKeyManager:
    """API key management."""
    
    def __init__(self):
        self.active_keys: Dict[str, APIKey] = {}
    
    def generate_key(self, user_id: str, name: str, scopes: List[str], 
                    expires_in_days: Optional[int] = None) -> Tuple[str, APIKey]:
        """Generate new API key."""
        # Generate random key
        raw_key = f"sk_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Create expiration date
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            id=secrets.token_urlsafe(16),
            name=name,
            key_hash=key_hash,
            user_id=user_id,
            scopes=scopes,
            expires_at=expires_at
        )
        
        self.active_keys[key_hash] = api_key
        return raw_key, api_key
    
    def verify_key(self, raw_key: str) -> Optional[APIKey]:
        """Verify API key and return key info."""
        if not raw_key.startswith("sk_"):
            return None
        
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        api_key = self.active_keys.get(key_hash)
        
        if not api_key or not api_key.is_valid():
            return None
        
        # Update usage statistics
        api_key.last_used = datetime.now(timezone.utc)
        api_key.usage_count += 1
        
        return api_key
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke API key."""
        for api_key in self.active_keys.values():
            if api_key.id == key_id:
                api_key.is_active = False
                return True
        return False
    
    def list_user_keys(self, user_id: str) -> List[APIKey]:
        """List all API keys for user."""
        return [key for key in self.active_keys.values() if key.user_id == user_id]


class RBACManager:
    """Role-based access control manager."""
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.roles = config.rbac.roles
        self.users: Dict[str, User] = {}
        self.user_permissions_cache: Dict[str, List[str]] = {}
    
    def add_user(self, user: User) -> None:
        """Add user to RBAC system."""
        self.users[user.id] = user
        self._invalidate_user_cache(user.id)
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def assign_role(self, user_id: str, role: str) -> bool:
        """Assign role to user."""
        user = self.users.get(user_id)
        if not user or role not in self.roles:
            return False
        
        if role not in user.roles:
            user.roles.append(role)
            self._invalidate_user_cache(user_id)
        
        return True
    
    def revoke_role(self, user_id: str, role: str) -> bool:
        """Revoke role from user."""
        user = self.users.get(user_id)
        if not user:
            return False
        
        if role in user.roles:
            user.roles.remove(role)
            self._invalidate_user_cache(user_id)
        
        return True
    
    def user_has_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has specific permission."""
        if permission == "*":
            return self.user_has_permission(user_id, "system:admin")
        
        user_permissions = self._get_user_permissions(user_id)
        return permission in user_permissions or "*" in user_permissions
    
    def user_has_role(self, user_id: str, role: str) -> bool:
        """Check if user has specific role."""
        user = self.users.get(user_id)
        return user and role in user.roles
    
    def get_role_permissions(self, role: str) -> List[str]:
        """Get permissions for role."""
        role_config = self.roles.get(role, {})
        return role_config.get("permissions", [])
    
    def _get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for user (with caching)."""
        if user_id in self.user_permissions_cache:
            return self.user_permissions_cache[user_id]
        
        user = self.users.get(user_id)
        if not user:
            return []
        
        permissions = set()
        for role in user.roles:
            role_permissions = self.get_role_permissions(role)
            permissions.update(role_permissions)
        
        permission_list = list(permissions)
        self.user_permissions_cache[user_id] = permission_list
        return permission_list
    
    def _invalidate_user_cache(self, user_id: str) -> None:
        """Invalidate user permissions cache."""
        if user_id in self.user_permissions_cache:
            del self.user_permissions_cache[user_id]


class SessionManager:
    """Session management."""
    
    def __init__(self, jwt_manager: JWTManager):
        self.jwt_manager = jwt_manager
        self.active_sessions: Dict[str, Session] = {}
    
    def create_session(self, user: User, ip_address: Optional[str] = None, 
                      user_agent: Optional[str] = None) -> Session:
        """Create new user session."""
        session_id = secrets.token_urlsafe(32)
        access_token = self.jwt_manager.create_access_token(user)
        refresh_token = self.jwt_manager.create_refresh_token(user)
        
        expires_at = datetime.now(timezone.utc) + self.jwt_manager.access_token_expire
        
        session = Session(
            id=session_id,
            user_id=user.id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            created_at=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.active_sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        session = self.active_sessions.get(session_id)
        if session and session.is_valid():
            session.last_activity = datetime.now(timezone.utc)
            return session
        return None
    
    def refresh_session(self, session_id: str) -> Optional[Session]:
        """Refresh session tokens."""
        session = self.active_sessions.get(session_id)
        if not session or not session.refresh_token:
            return None
        
        new_access_token = self.jwt_manager.refresh_access_token(session.refresh_token)
        if not new_access_token:
            return None
        
        session.access_token = new_access_token
        session.expires_at = datetime.now(timezone.utc) + self.jwt_manager.access_token_expire
        session.last_activity = datetime.now(timezone.utc)
        
        return session
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].is_active = False
            return True
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions."""
        expired_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if not session.is_valid()
        ]
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        return len(expired_sessions)


class EnterpriseAuth:
    """Main enterprise authentication system."""
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.jwt_manager = JWTManager(config)
        self.password_manager = PasswordManager()
        self.mfa_manager = MFAManager(config.auth.mfa_issuer)
        self.api_key_manager = APIKeyManager()
        self.rbac_manager = RBACManager(config)
        self.session_manager = SessionManager(self.jwt_manager)
        
        # Initialize admin user if configured
        self._initialize_admin_users()
    
    def _initialize_admin_users(self) -> None:
        """Initialize admin users from configuration."""
        for admin_username in self.config.rbac.admin_users:
            admin_user = User(
                id=f"admin_{secrets.token_urlsafe(8)}",
                username=admin_username,
                email=f"{admin_username}@example.com",
                full_name=f"Admin {admin_username}",
                roles=["admin"],
                is_active=True,
                is_verified=True
            )
            self.rbac_manager.add_user(admin_user)
    
    async def authenticate_password(self, username: str, password: str, 
                                  mfa_token: Optional[str] = None) -> Optional[User]:
        """Authenticate user with username/password."""
        # Find user by username (in production, this would query a database)
        user = None
        for u in self.rbac_manager.users.values():
            if u.username == username:
                user = u
                break
        
        if not user or not user.is_active or user.is_locked():
            return None
        
        # Verify password (in production, you'd have stored password hashes)
        # For demo purposes, we'll accept any password for existing users
        
        # Check MFA if enabled
        if user.mfa_enabled and user.mfa_secret:
            if not mfa_token or not self.mfa_manager.verify_totp(user.mfa_secret, mfa_token):
                return None
        
        # Update login info
        user.last_login = datetime.now(timezone.utc)
        user.failed_login_attempts = 0
        
        return user
    
    def authenticate_api_key(self, api_key: str) -> Optional[Tuple[User, APIKey]]:
        """Authenticate using API key."""
        key_info = self.api_key_manager.verify_key(api_key)
        if not key_info:
            return None
        
        user = self.rbac_manager.get_user(key_info.user_id)
        if not user or not user.is_active:
            return None
        
        return user, key_info
    
    def authenticate_token(self, token: str) -> Optional[User]:
        """Authenticate using JWT token."""
        payload = self.jwt_manager.verify_token(token)
        if not payload or payload.get("type") != "access":
            return None
        
        user_id = payload.get("sub")
        if not user_id:
            return None
        
        return self.rbac_manager.get_user(user_id)
    
    def create_user(self, username: str, email: str, full_name: str, 
                   password: str, roles: List[str] = None, 
                   tenant_id: Optional[str] = None) -> User:
        """Create new user."""
        if roles is None:
            roles = [self.config.rbac.default_role]
        
        # Validate roles
        for role in roles:
            if role not in self.config.rbac.roles:
                raise ValueError(f"Invalid role: {role}")
        
        user = User(
            id=f"user_{secrets.token_urlsafe(16)}",
            username=username,
            email=email,
            full_name=full_name,
            roles=roles,
            tenant_id=tenant_id,
            is_active=True,
            is_verified=False
        )
        
        self.rbac_manager.add_user(user)
        return user
    
    def enable_mfa(self, user_id: str) -> Tuple[str, bytes]:
        """Enable MFA for user."""
        user = self.rbac_manager.get_user(user_id)
        if not user:
            raise ValueError("User not found")
        
        secret = self.mfa_manager.generate_secret()
        qr_code = self.mfa_manager.generate_qr_code(user, secret)
        
        user.mfa_secret = secret
        user.mfa_enabled = True
        
        return secret, qr_code
    
    def create_api_key(self, user_id: str, name: str, scopes: List[str], 
                      expires_in_days: Optional[int] = None) -> Tuple[str, APIKey]:
        """Create API key for user."""
        user = self.rbac_manager.get_user(user_id)
        if not user:
            raise ValueError("User not found")
        
        return self.api_key_manager.generate_key(user_id, name, scopes, expires_in_days)
    
    def login(self, user: User, ip_address: Optional[str] = None, 
             user_agent: Optional[str] = None) -> Session:
        """Create login session for user."""
        return self.session_manager.create_session(user, ip_address, user_agent)
    
    def logout(self, session_id: str) -> bool:
        """Logout user session."""
        return self.session_manager.revoke_session(session_id)
    
    def check_permission(self, user_id: str, permission: Union[Permission, str]) -> bool:
        """Check if user has permission."""
        if isinstance(permission, Permission):
            permission = permission.value
        
        return self.rbac_manager.user_has_permission(user_id, permission)
    
    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information."""
        user = self.rbac_manager.get_user(user_id)
        return user.to_dict() if user else None

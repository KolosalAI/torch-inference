"""
Authentication data models.

This module defines the data structures for authentication and authorization.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import secrets


@dataclass
class User:
    """User data model."""
    
    id: str
    username: str
    email: str
    full_name: str
    hashed_password: str
    roles: List[str] = field(default_factory=list)
    api_keys: List[str] = field(default_factory=list)
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert user to dictionary.
        
        Args:
            include_sensitive: If True, includes sensitive data like hashed_password
        """
        data = {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "roles": self.roles,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "locked_until": self.locked_until.isoformat() if self.locked_until else None,
            "failed_attempts": self.failed_attempts,
            "metadata": self.metadata
        }
        
        if include_sensitive:
            data.update({
                "hashed_password": self.hashed_password,
                "api_keys": self.api_keys
            })
        else:
            data["api_key_count"] = len(self.api_keys)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create user from dictionary."""
        user = cls(
            id=data["id"],
            username=data["username"],
            email=data["email"],
            full_name=data["full_name"],
            hashed_password=data["hashed_password"],
            roles=data.get("roles", []),
            api_keys=data.get("api_keys", []),
            is_active=data.get("is_active", True),
            is_verified=data.get("is_verified", False),
            failed_attempts=data.get("failed_attempts", 0),
            metadata=data.get("metadata", {})
        )
        
        # Parse datetime fields
        if data.get("created_at"):
            if isinstance(data["created_at"], str):
                user.created_at = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
            else:
                user.created_at = data["created_at"]
        
        if data.get("last_login"):
            if isinstance(data["last_login"], str):
                user.last_login = datetime.fromisoformat(data["last_login"].replace('Z', '+00:00'))
            else:
                user.last_login = data["last_login"]
        
        if data.get("locked_until"):
            if isinstance(data["locked_until"], str):
                user.locked_until = datetime.fromisoformat(data["locked_until"].replace('Z', '+00:00'))
            else:
                user.locked_until = data["locked_until"]
        
        return user
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role."""
        return role in self.roles
    
    def add_role(self, role: str) -> None:
        """Add role to user."""
        if role not in self.roles:
            self.roles.append(role)
    
    def remove_role(self, role: str) -> None:
        """Remove role from user."""
        if role in self.roles:
            self.roles.remove(role)
    
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if not self.locked_until:
            return False
        return datetime.now(timezone.utc) < self.locked_until
    
    def add_api_key(self, api_key_hash: str) -> None:
        """Add API key hash to user."""
        if api_key_hash not in self.api_keys:
            self.api_keys.append(api_key_hash)
    
    def remove_api_key(self, api_key_hash: str) -> None:
        """Remove API key hash from user."""
        if api_key_hash in self.api_keys:
            self.api_keys.remove(api_key_hash)


@dataclass
class APIKey:
    """API Key data model."""
    
    id: str
    name: str
    key_hash: str
    user_id: str
    scopes: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert API key to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "key_hash": self.key_hash,
            "user_id": self.user_id,
            "scopes": self.scopes,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "usage_count": self.usage_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APIKey':
        """Create API key from dictionary."""
        api_key = cls(
            id=data["id"],
            name=data["name"],
            key_hash=data["key_hash"],
            user_id=data["user_id"],
            scopes=data.get("scopes", []),
            is_active=data.get("is_active", True),
            usage_count=data.get("usage_count", 0)
        )
        
        # Parse datetime fields
        if data.get("created_at"):
            if isinstance(data["created_at"], str):
                api_key.created_at = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
            else:
                api_key.created_at = data["created_at"]
        
        if data.get("last_used"):
            if isinstance(data["last_used"], str):
                api_key.last_used = datetime.fromisoformat(data["last_used"].replace('Z', '+00:00'))
            else:
                api_key.last_used = data["last_used"]
        
        if data.get("expires_at"):
            if isinstance(data["expires_at"], str):
                api_key.expires_at = datetime.fromisoformat(data["expires_at"].replace('Z', '+00:00'))
            else:
                api_key.expires_at = data["expires_at"]
        
        return api_key
    
    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if API key is valid."""
        return self.is_active and not self.is_expired()
    
    def use(self) -> None:
        """Mark API key as used."""
        self.last_used = datetime.now(timezone.utc)
        self.usage_count += 1


@dataclass
class Token:
    """Token data model."""
    
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert token to dictionary."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in
        }


# Pydantic models for API requests/responses

class AuthRequest(BaseModel):
    """Authentication request model."""
    
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class RegisterRequest(BaseModel):
    """User registration request model."""
    
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password")
    full_name: str = Field(..., min_length=1, max_length=100, description="Full name")


class ChangePasswordRequest(BaseModel):
    """Change password request model."""
    
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")


class GenerateAPIKeyRequest(BaseModel):
    """Generate API key request model."""
    
    name: str = Field(..., min_length=1, max_length=100, description="API key name")
    scopes: List[str] = Field(default_factory=list, description="API key scopes")
    expires_in_days: Optional[int] = Field(None, gt=0, le=365, description="Expiry in days")


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    
    refresh_token: str = Field(..., description="Refresh token")


class AuthResponse(BaseModel):
    """Authentication response model."""
    
    success: bool
    message: str
    token: Optional[Token] = None
    user: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class UserResponse(BaseModel):
    """User response model."""
    
    success: bool
    user: Optional[Dict[str, Any]] = None
    message: str
    error: Optional[str] = None


class APIKeyResponse(BaseModel):
    """API key response model."""
    
    success: bool
    api_key: Optional[str] = None
    key_info: Optional[Dict[str, Any]] = None
    message: str
    error: Optional[str] = None


class TokenRefreshResponse(BaseModel):
    """Token refresh response model."""
    
    success: bool
    access_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int = 1800
    message: str
    error: Optional[str] = None


def create_user_id() -> str:
    """Generate unique user ID."""
    return f"user_{secrets.token_urlsafe(16)}"


def create_api_key_id() -> str:
    """Generate unique API key ID."""
    return f"key_{secrets.token_urlsafe(12)}"

"""
Simple file-based user storage.

This module provides a simple JSON-based user storage system for authentication.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import secrets

from .models import User, APIKey, create_user_id, create_api_key_id
from .password import hash_password, verify_password, hash_api_key, verify_api_key

logger = logging.getLogger(__name__)


class UserStore:
    """File-based user storage system."""
    
    def __init__(self, users_file: str = "./data/users.json", 
                 api_keys_file: str = "./data/api_keys.json"):
        """
        Initialize user store.
        
        Args:
            users_file: Path to users JSON file
            api_keys_file: Path to API keys JSON file
        """
        self.users_file = Path(users_file)
        self.api_keys_file = Path(api_keys_file)
        
        # Ensure data directories exist
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        self.api_keys_file.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        
        # Load existing data
        self._load_users()
        self._load_api_keys()
        
        # Create default admin user if no users exist
        if not self.users:
            self._create_default_admin()
        
        # Ensure files exist after initialization
        self._save_users()
        self._save_api_keys()
        
        logger.info(f"User store initialized with {len(self.users)} users and {len(self.api_keys)} API keys")
    
    def _load_users(self) -> None:
        """Load users from file."""
        if not self.users_file.exists():
            logger.info("Users file does not exist, starting with empty store")
            return
        
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for user_data in data.get('users', []):
                user = User.from_dict(user_data)
                self.users[user.username] = user
            
            logger.info(f"Loaded {len(self.users)} users from file")
            
        except Exception as e:
            logger.error(f"Failed to load users from file: {e}")
    
    def _save_users(self) -> None:
        """Save users to file."""
        try:
            data = {
                'users': [user.to_dict(include_sensitive=True) for user in self.users.values()],
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Write to temporary file first
            temp_file = self.users_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomically replace the original file
            temp_file.replace(self.users_file)
            
            logger.debug(f"Saved {len(self.users)} users to file")
            
        except Exception as e:
            logger.error(f"Failed to save users to file: {e}")
    
    def _load_api_keys(self) -> None:
        """Load API keys from file."""
        if not self.api_keys_file.exists():
            logger.info("API keys file does not exist, starting with empty store")
            return
        
        try:
            with open(self.api_keys_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for key_data in data.get('api_keys', []):
                api_key = APIKey.from_dict(key_data)
                self.api_keys[api_key.key_hash] = api_key
            
            logger.info(f"Loaded {len(self.api_keys)} API keys from file")
            
        except Exception as e:
            logger.error(f"Failed to load API keys from file: {e}")
    
    def _save_api_keys(self) -> None:
        """Save API keys to file."""
        try:
            data = {
                'api_keys': [api_key.to_dict() for api_key in self.api_keys.values()],
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Write to temporary file first
            temp_file = self.api_keys_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomically replace the original file
            temp_file.replace(self.api_keys_file)
            
            logger.debug(f"Saved {len(self.api_keys)} API keys to file")
            
        except Exception as e:
            logger.error(f"Failed to save API keys to file: {e}")
    
    def _create_default_admin(self) -> None:
        """Create default admin user."""
        admin_password = "admin123"  # Change this in production!
        admin_user = User(
            id=create_user_id(),
            username="admin",
            email="admin@example.com",
            full_name="Administrator",
            hashed_password=hash_password(admin_password),
            roles=["admin"],
            is_active=True,
            is_verified=True
        )
        
        self.users["admin"] = admin_user
        self._save_users()
        
        logger.warning(f"Created default admin user - Username: admin, Password: {admin_password}")
        logger.warning("CHANGE THE DEFAULT ADMIN PASSWORD IMMEDIATELY!")
    
    def create_user(self, username: str, email: str, full_name: str, 
                   password: str, roles: List[str] = None) -> User:
        """
        Create new user.
        
        Args:
            username: Username
            email: Email address
            full_name: Full name
            password: Plain text password
            roles: User roles
            
        Returns:
            Created user
            
        Raises:
            ValueError: If user already exists
        """
        if username in self.users:
            raise ValueError(f"User {username} already exists")
        
        # Check email uniqueness
        for user in self.users.values():
            if user.email == email:
                raise ValueError(f"Email {email} already exists")
        
        user = User(
            id=create_user_id(),
            username=username,
            email=email,
            full_name=full_name,
            hashed_password=hash_password(password),
            roles=roles or ["user"],
            is_active=True,
            is_verified=False
        )
        
        self.users[username] = user
        self._save_users()
        
        logger.info(f"Created user: {username}")
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            User if authentication successful, None otherwise
        """
        user = self.users.get(username)
        if not user:
            logger.debug(f"User not found: {username}")
            return None
        
        if not user.is_active:
            logger.debug(f"User inactive: {username}")
            return None
        
        if user.is_locked():
            logger.debug(f"User locked: {username}")
            return None
        
        if not verify_password(password, user.hashed_password):
            # Increment failed attempts
            user.failed_attempts += 1
            state_changed = False
            if user.failed_attempts >= 5:
                # Lock user for 30 minutes
                from datetime import timedelta
                user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)
                logger.warning(f"User locked due to too many failed attempts: {username}")
                state_changed = True
            if user.failed_attempts == 1 or state_changed:
                self._save_users()
            logger.debug(f"Invalid password for user: {username}")
            return None
        # Reset failed attempts on successful login if needed
        if user.failed_attempts != 0 or user.locked_until is not None:
            user.failed_attempts = 0
            user.locked_until = None
            user.last_login = datetime.now(timezone.utc)
            self._save_users()
        else:
            user.last_login = datetime.now(timezone.utc)
        logger.info(f"User authenticated: {username}")
        return user
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.users.get(username)
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        for user in self.users.values():
            if user.id == user_id:
                return user
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        for user in self.users.values():
            if user.email == email:
                return user
        return None
    
    def update_user(self, username: str, **kwargs) -> bool:
        """
        Update user fields.
        
        Args:
            username: Username
            **kwargs: Fields to update
            
        Returns:
            True if user was updated
        """
        user = self.users.get(username)
        if not user:
            return False
        
        # Update allowed fields
        allowed_fields = ['email', 'full_name', 'roles', 'is_active', 'is_verified', 'metadata']
        for field, value in kwargs.items():
            if field in allowed_fields and hasattr(user, field):
                setattr(user, field, value)
        
        self._save_users()
        logger.info(f"Updated user: {username}")
        return True
    
    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """
        Change user password.
        
        Args:
            username: Username
            old_password: Current password
            new_password: New password
            
        Returns:
            True if password was changed
        """
        user = self.authenticate_user(username, old_password)
        if not user:
            return False
        
        user.hashed_password = hash_password(new_password)
        self._save_users()
        
        logger.info(f"Password changed for user: {username}")
        return True
    
    def delete_user(self, username: str) -> bool:
        """
        Delete user.
        
        Args:
            username: Username
            
        Returns:
            True if user was deleted
        """
        if username not in self.users:
            return False
        
        user = self.users.pop(username)
        
        # Remove user's API keys
        user_api_keys = [key for key in self.api_keys.values() if key.user_id == user.id]
        for api_key in user_api_keys:
            del self.api_keys[api_key.key_hash]
        
        self._save_users()
        self._save_api_keys()
        
        logger.info(f"Deleted user: {username}")
        return True
    
    def list_users(self) -> List[Dict[str, Any]]:
        """List all users (excluding sensitive data)."""
        return [user.to_dict() for user in self.users.values()]
    
    def create_api_key(self, username: str, name: str, scopes: List[str] = None,
                      expires_in_days: Optional[int] = None) -> tuple[str, APIKey]:
        """
        Create API key for user.
        
        Args:
            username: Username
            name: API key name
            scopes: API key scopes
            expires_in_days: Expiry in days
            
        Returns:
            Tuple of (raw_api_key, api_key_object)
            
        Raises:
            ValueError: If user not found
        """
        user = self.users.get(username)
        if not user:
            raise ValueError(f"User not found: {username}")
        
        # Generate API key
        from .password import generate_api_key
        raw_key = generate_api_key()
        key_hash = hash_api_key(raw_key)
        
        # Set expiry
        expires_at = None
        if expires_in_days:
            from datetime import timedelta
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            id=create_api_key_id(),
            name=name,
            key_hash=key_hash,
            user_id=user.id,
            scopes=scopes or [],
            expires_at=expires_at
        )
        
        # Store API key
        self.api_keys[key_hash] = api_key
        user.add_api_key(key_hash)
        
        self._save_api_keys()
        self._save_users()
        
        logger.info(f"Created API key for user {username}: {name}")
        return raw_key, api_key
    
    def authenticate_api_key(self, raw_key: str) -> Optional[tuple[User, APIKey]]:
        """
        Authenticate using API key.
        
        Args:
            raw_key: Raw API key
            
        Returns:
            Tuple of (user, api_key) if valid, None otherwise
        """
        key_hash = hash_api_key(raw_key)
        api_key = self.api_keys.get(key_hash)
        
        if not api_key or not api_key.is_valid():
            logger.debug("Invalid or expired API key used")
            return None
        
        user = self.get_user_by_id(api_key.user_id)
        if not user or not user.is_active:
            logger.debug(f"API key user not found or inactive: {api_key.user_id}")
            return None
        
        # Update usage statistics
        api_key.use()
        self._save_api_keys()
        
        logger.debug(f"API key authenticated for user: {user.username}")
        return user, api_key
    
    def revoke_api_key(self, username: str, key_id: str) -> bool:
        """
        Revoke API key.
        
        Args:
            username: Username
            key_id: API key ID
            
        Returns:
            True if key was revoked
        """
        user = self.users.get(username)
        if not user:
            return False
        
        # Find and revoke key
        for key_hash, api_key in self.api_keys.items():
            if api_key.id == key_id and api_key.user_id == user.id:
                api_key.is_active = False
                user.remove_api_key(key_hash)
                self._save_api_keys()
                self._save_users()
                logger.info(f"Revoked API key {key_id} for user {username}")
                return True
        
        return False
    
    def list_user_api_keys(self, username: str) -> List[Dict[str, Any]]:
        """
        List API keys for user.
        
        Args:
            username: Username
            
        Returns:
            List of API key info (excluding key hash)
        """
        user = self.users.get(username)
        if not user:
            return []
        
        user_keys = []
        for api_key in self.api_keys.values():
            if api_key.user_id == user.id:
                key_info = api_key.to_dict()
                key_info.pop('key_hash', None)  # Remove sensitive data
                user_keys.append(key_info)
        
        return user_keys
    
    def cleanup_expired_keys(self) -> int:
        """
        Remove expired API keys.
        
        Returns:
            Number of keys removed
        """
        expired_keys = []
        for key_hash, api_key in self.api_keys.items():
            if api_key.is_expired():
                expired_keys.append(key_hash)
        
        for key_hash in expired_keys:
            api_key = self.api_keys.pop(key_hash)
            # Remove from user's key list
            user = self.get_user_by_id(api_key.user_id)
            if user:
                user.remove_api_key(key_hash)
        
        if expired_keys:
            self._save_api_keys()
            self._save_users()
            logger.info(f"Cleaned up {len(expired_keys)} expired API keys")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        active_users = sum(1 for user in self.users.values() if user.is_active)
        verified_users = sum(1 for user in self.users.values() if user.is_verified)
        active_keys = sum(1 for key in self.api_keys.values() if key.is_active)
        
        return {
            "total_users": len(self.users),
            "active_users": active_users,
            "verified_users": verified_users,
            "total_api_keys": len(self.api_keys),
            "active_api_keys": active_keys,
            "users_file": str(self.users_file),
            "api_keys_file": str(self.api_keys_file)
        }

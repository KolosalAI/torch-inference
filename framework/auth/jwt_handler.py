"""
JWT Token Handler for authentication.

This module provides JWT token generation, validation, and management.
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
import logging

try:
    from jose import JWTError, jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None
    JWTError = Exception

logger = logging.getLogger(__name__)


class JWTHandler:
    """JWT token handler for authentication."""
    
    def __init__(self, secret_key: str = None, algorithm: str = "HS256", 
                 access_token_expire_minutes: int = 30,
                 refresh_token_expire_days: int = 7):
        """
        Initialize JWT handler.
        
        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT algorithm (default: HS256)
            access_token_expire_minutes: Access token expiry in minutes
            refresh_token_expire_days: Refresh token expiry in days
        """
        if not JWT_AVAILABLE:
            raise RuntimeError("python-jose is required for JWT functionality. Install with: pip install python-jose[cryptography]")
        
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        
        # Active tokens storage (in production, use Redis or database)
        self.active_tokens: Dict[str, Dict[str, Any]] = {}
        self.blacklisted_tokens: set = set()
        
        logger.info(f"JWT Handler initialized with algorithm: {algorithm}")
    
    def create_access_token(self, user_data: Dict[str, Any], 
                           expires_delta: Optional[timedelta] = None) -> str:
        """
        Create access token for user.
        
        Args:
            user_data: User data to include in token
            expires_delta: Custom expiration delta
            
        Returns:
            JWT access token
        """
        to_encode = user_data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access"
        })
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            
            # Store token info
            self.active_tokens[encoded_jwt] = {
                "user_id": user_data.get("sub"),
                "username": user_data.get("username"),
                "expires_at": expire,
                "type": "access"
            }
            
            logger.debug(f"Created access token for user: {user_data.get('username')}")
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise
    
    def create_refresh_token(self, user_data: Dict[str, Any]) -> str:
        """
        Create refresh token for user.
        
        Args:
            user_data: User data to include in token
            
        Returns:
            JWT refresh token
        """
        to_encode = {
            "sub": user_data.get("sub"),
            "username": user_data.get("username"),
            "type": "refresh"
        }
        
        expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc)
        })
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            
            # Store token info
            self.active_tokens[encoded_jwt] = {
                "user_id": user_data.get("sub"),
                "username": user_data.get("username"),
                "expires_at": expire,
                "type": "refresh"
            }
            
            logger.debug(f"Created refresh token for user: {user_data.get('username')}")
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Failed to create refresh token: {e}")
            raise
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Token payload if valid, None otherwise
        """
        if token in self.blacklisted_tokens:
            logger.warning("Attempted to use blacklisted token")
            return None
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token exists in active tokens
            if token not in self.active_tokens:
                logger.warning("Token not found in active tokens")
                return None
            
            # Verify token type exists
            if "type" not in payload:
                logger.warning("Token missing type field")
                return None
            
            logger.debug(f"Token verified successfully for user: {payload.get('username')}")
            return payload
            
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        Create new access token from refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New access token if successful, None otherwise
        """
        payload = self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            logger.warning("Invalid refresh token provided")
            return None
        
        # Create new access token
        user_data = {
            "sub": payload.get("sub"),
            "username": payload.get("username")
        }
        
        try:
            new_access_token = self.create_access_token(user_data)
            logger.info(f"Access token refreshed for user: {payload.get('username')}")
            return new_access_token
        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke (blacklist) a token.
        
        Args:
            token: Token to revoke
            
        Returns:
            True if revoked successfully
        """
        self.blacklisted_tokens.add(token)
        if token in self.active_tokens:
            del self.active_tokens[token]
        
        logger.info("Token revoked successfully")
        return True
    
    def revoke_user_tokens(self, user_id: str) -> int:
        """
        Revoke all tokens for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of tokens revoked
        """
        tokens_to_revoke = []
        for token, token_info in self.active_tokens.items():
            if token_info.get("user_id") == user_id:
                tokens_to_revoke.append(token)
        
        for token in tokens_to_revoke:
            self.revoke_token(token)
        
        logger.info(f"Revoked {len(tokens_to_revoke)} tokens for user: {user_id}")
        return len(tokens_to_revoke)
    
    def cleanup_expired_tokens(self) -> int:
        """
        Remove expired tokens from active tokens.
        
        Returns:
            Number of tokens cleaned up
        """
        current_time = datetime.now(timezone.utc)
        expired_tokens = []
        
        for token, token_info in self.active_tokens.items():
            if token_info.get("expires_at") and current_time > token_info["expires_at"]:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self.active_tokens[token]
        
        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")
        
        return len(expired_tokens)
    
    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a token.
        
        Args:
            token: Token to get info for
            
        Returns:
            Token information if exists
        """
        return self.active_tokens.get(token)
    
    def is_token_active(self, token: str) -> bool:
        """
        Check if token is active (not expired or blacklisted).
        
        Args:
            token: Token to check
            
        Returns:
            True if token is active
        """
        if token in self.blacklisted_tokens:
            return False
        
        token_info = self.active_tokens.get(token)
        if not token_info:
            return False
        
        expires_at = token_info.get("expires_at")
        if expires_at and datetime.now(timezone.utc) > expires_at:
            return False
        
        return True
    
    def get_active_token_count(self) -> int:
        """Get number of active tokens."""
        return len(self.active_tokens)
    
    def get_user_token_count(self, user_id: str) -> int:
        """Get number of active tokens for a user."""
        count = 0
        for token_info in self.active_tokens.values():
            if token_info.get("user_id") == user_id:
                count += 1
        return count

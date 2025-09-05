"""
Security utilities and configurations.
"""

import hashlib
import hmac
import secrets
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import jwt

logger = logging.getLogger(__name__)


class SecurityConfig:
    """Security configuration settings."""
    
    def __init__(self):
        self.secret_key = self._get_or_generate_secret_key()
        self.token_expiry_minutes = 60
        self.max_request_rate = 100  # requests per minute
        
        # Load CORS configuration from main config
        try:
            from .config import get_config
            config = get_config()
            self.enable_cors = config.security.enable_cors
            self.allowed_origins = config.security.allowed_origins
        except Exception:
            # Secure defaults - no wildcards
            self.enable_cors = True
            self.allowed_origins = ["http://localhost:3000", "http://localhost:8080"]
            
        self.enable_api_key_auth = False
        self.api_keys_file = "api_keys.txt"
    
    def _get_or_generate_secret_key(self) -> str:
        """Get existing secret key or generate a new one."""
        key_file = Path("secret_key.txt")
        
        if key_file.exists():
            try:
                return key_file.read_text().strip()
            except Exception as e:
                logger.warning(f"Failed to read secret key file: {e}")
        
        # Generate new secret key
        secret_key = secrets.token_hex(32)
        
        try:
            key_file.write_text(secret_key)
            logger.info("Generated new secret key")
        except Exception as e:
            logger.warning(f"Failed to save secret key: {e}")
        
        return secret_key


class SecurityManager:
    """Centralized security management."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self._request_counts: Dict[str, list] = {}
        self._blocked_ips: Dict[str, datetime] = {}
        self._api_keys: set = self._load_api_keys()
    
    def _load_api_keys(self) -> set:
        """Load API keys from file."""
        api_keys = set()
        api_keys_file = Path(self.config.api_keys_file)
        
        if api_keys_file.exists():
            try:
                with open(api_keys_file, 'r') as f:
                    for line in f:
                        key = line.strip()
                        if key and not key.startswith('#'):
                            api_keys.add(key)
                logger.info(f"Loaded {len(api_keys)} API keys")
            except Exception as e:
                logger.error(f"Failed to load API keys: {e}")
        
        return api_keys
    
    def generate_api_key(self) -> str:
        """Generate a new API key."""
        return secrets.token_urlsafe(32)
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key."""
        if not self.config.enable_api_key_auth:
            return True
        
        return api_key in self._api_keys
    
    def check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is within rate limits."""
        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        if client_ip in self._request_counts:
            self._request_counts[client_ip] = [
                req_time for req_time in self._request_counts[client_ip]
                if current_time - req_time < 60
            ]
        else:
            self._request_counts[client_ip] = []
        
        # Check rate limit
        if len(self._request_counts[client_ip]) >= self.config.max_request_rate:
            return False
        
        # Add current request
        self._request_counts[client_ip].append(current_time)
        return True
    
    def is_blocked(self, client_ip: str) -> bool:
        """Check if IP is temporarily blocked."""
        if client_ip in self._blocked_ips:
            if datetime.now() > self._blocked_ips[client_ip]:
                # Unblock expired IP
                del self._blocked_ips[client_ip]
                return False
            return True
        return False
    
    def block_ip(self, client_ip: str, duration_minutes: int = 10):
        """Temporarily block an IP address."""
        block_until = datetime.now() + timedelta(minutes=duration_minutes)
        self._blocked_ips[client_ip] = block_until
        logger.warning(f"Blocked IP {client_ip} until {block_until}")
    
    def create_token(self, payload: Dict[str, Any]) -> str:
        """Create a JWT token."""
        try:
            payload['exp'] = datetime.utcnow() + timedelta(minutes=self.config.token_expiry_minutes)
            payload['iat'] = datetime.utcnow()
            
            return jwt.encode(payload, self.config.secret_key, algorithm='HS256')
        except Exception as e:
            logger.error(f"Token creation failed: {e}")
            raise
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a JWT token."""
        try:
            payload = jwt.decode(token, self.config.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
        
        return None
    
    def hash_password(self, password: str) -> str:
        """Hash a password securely."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            salt, stored_hash = hashed_password.split(':')
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return hmac.compare_digest(stored_hash, password_hash.hex())
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def sanitize_input(self, user_input: str, max_length: int = 1000) -> str:
        """Sanitize user input to prevent injection attacks."""
        if not isinstance(user_input, str):
            user_input = str(user_input)
        
        # Truncate to max length
        user_input = user_input[:max_length]
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '\x00', '\r']
        for char in dangerous_chars:
            user_input = user_input.replace(char, '')
        
        return user_input.strip()
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get the global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def init_security(config: Optional[SecurityConfig] = None) -> SecurityManager:
    """Initialize the global security manager."""
    global _security_manager
    _security_manager = SecurityManager(config)
    return _security_manager

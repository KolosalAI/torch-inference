"""
Password hashing and validation utilities.

This module provides secure password hashing and verification using bcrypt.
"""

import secrets
import string
import logging
from typing import Optional

try:
    from passlib.context import CryptContext
    PASSLIB_AVAILABLE = True
    BCRYPT_AVAILABLE = True  # For test compatibility
except ImportError:
    PASSLIB_AVAILABLE = False
    BCRYPT_AVAILABLE = False  # For test compatibility
    CryptContext = None

logger = logging.getLogger(__name__)

# Password context for hashing

# Allow fast hash for tests
import os
pwd_context = None
if os.environ.get("TEST_FAST_HASH") == "1":
    pwd_context = None  # Use simple hash for tests
    logger.info("Password hashing: TEST_FAST_HASH enabled, using simple hash for tests")
elif PASSLIB_AVAILABLE:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    logger.info("Password hashing initialized with bcrypt")
else:
    logger.warning("passlib not available - password hashing will use simple fallback")


def hash_password(password: str) -> str:
    """
    Hash password using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    if not pwd_context:
        # Simple fallback for development (NOT SECURE for production)
        import hashlib
        # Add salt to prevent the same password from having the same hash
        salt = secrets.token_hex(16)
        hash_value = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"simple:{salt}:{hash_value}"
    
    try:
        hashed = pwd_context.hash(password)
        logger.debug("Password hashed successfully")
        return hashed
    except Exception as e:
        logger.error(f"Failed to hash password: {e}")
        raise


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches
    """
    if not pwd_context:
        # Simple fallback verification
        if hashed_password.startswith("simple:"):
            import hashlib
            # Handle both old format (simple:hash) and new format (simple:salt:hash)
            parts = hashed_password.split(":")
            if len(parts) == 3:  # New format with salt
                _, salt, stored_hash = parts
                expected_hash = hashlib.sha256((plain_password + salt).encode()).hexdigest()
                return expected_hash == stored_hash
            elif len(parts) == 2:  # Old format without salt (for backward compatibility)
                _, stored_hash = parts
                expected_hash = hashlib.sha256(plain_password.encode()).hexdigest()
                return expected_hash == stored_hash
        return False
    
    try:
        result = pwd_context.verify(plain_password, hashed_password)
        logger.debug(f"Password verification: {'success' if result else 'failed'}")
        return result
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def generate_password(length: int = 16, 
                     include_uppercase: bool = True,
                     include_lowercase: bool = True, 
                     include_digits: bool = True,
                     include_symbols: bool = True) -> str:
    """
    Generate secure random password.
    
    Args:
        length: Password length
        include_uppercase: Include uppercase letters
        include_lowercase: Include lowercase letters
        include_digits: Include digits
        include_symbols: Include symbols
        
    Returns:
        Generated password
    """
    if length < 4:
        raise ValueError("Password length must be at least 4 characters")
    
    character_sets = []
    
    if include_lowercase:
        character_sets.append(string.ascii_lowercase)
    if include_uppercase:
        character_sets.append(string.ascii_uppercase)
    if include_digits:
        character_sets.append(string.digits)
    if include_symbols:
        character_sets.append("!@#$%^&*()-_+=")
    
    if not character_sets:
        raise ValueError("At least one character set must be included")
    
    # Ensure at least one character from each selected set
    password_chars = []
    for char_set in character_sets:
        password_chars.append(secrets.choice(char_set))
    
    # Fill remaining length with random characters from all sets
    all_chars = ''.join(character_sets)
    for _ in range(length - len(password_chars)):
        password_chars.append(secrets.choice(all_chars))
    
    # Shuffle the password
    secrets.SystemRandom().shuffle(password_chars)
    
    password = ''.join(password_chars)
    logger.debug(f"Generated password of length {length}")
    return password


def generate_api_key(length: int = 32) -> str:
    """
    Generate secure API key.
    
    Args:
        length: Key length
        
    Returns:
        Generated API key
    """
    api_key = f"sk_{secrets.token_urlsafe(length)}"
    logger.debug("Generated API key")
    return api_key


def is_password_strong(password: str) -> tuple[bool, list[str]]:
    """
    Check if password meets strength requirements.
    
    Args:
        password: Password to check
        
    Returns:
        Tuple of (is_strong, list_of_issues)
    """
    issues = []
    
    if len(password) < 8:
        issues.append("Password must be at least 8 characters long")
    
    if not any(c.islower() for c in password):
        issues.append("Password must contain at least one lowercase letter")
    
    if not any(c.isupper() for c in password):
        issues.append("Password must contain at least one uppercase letter")
    
    if not any(c.isdigit() for c in password):
        issues.append("Password must contain at least one digit")
    
    if not any(c in "!@#$%^&*()-_+=" for c in password):
        issues.append("Password must contain at least one special character")
    
    # Check for common patterns
    common_patterns = ["123", "abc", "password", "admin", "user", "test"]
    if any(pattern in password.lower() for pattern in common_patterns):
        issues.append("Password contains common patterns")
    
    return len(issues) == 0, issues


def hash_api_key(api_key: str) -> str:
    """
    Hash API key for storage.
    
    Args:
        api_key: Plain API key
        
    Returns:
        Hashed API key
    """
    import hashlib
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """
    Verify API key against hash.
    
    Args:
        plain_key: Plain API key
        hashed_key: Hashed API key
        
    Returns:
        True if key matches
    """
    import hashlib
    return hashlib.sha256(plain_key.encode()).hexdigest() == hashed_key

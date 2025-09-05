"""
Authentication configuration loader.

This module loads authentication configuration from YAML files.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class AuthConfig:
    """Authentication configuration."""
    
    def __init__(self, config_data: Dict[str, Any]):
        """
        Initialize auth config.
        
        Args:
            config_data: Configuration dictionary
        """
        self.jwt_secret_key = config_data.get("jwt_secret_key", "your-secret-key-here")
        self.jwt_algorithm = config_data.get("jwt_algorithm", "HS256")
        self.access_token_expire_minutes = config_data.get("access_token_expire_minutes", 30)
        self.refresh_token_expire_days = config_data.get("refresh_token_expire_days", 7)
        self.api_key_length = config_data.get("api_key_length", 32)
        self.user_store_file = config_data.get("user_store_file", "./data/users.json")
        self.session_store_file = config_data.get("session_store_file", "./data/sessions.json")


class SecurityConfig:
    """Security configuration."""
    
    def __init__(self, config_data: Dict[str, Any]):
        """
        Initialize security config.
        
        Args:
            config_data: Configuration dictionary
        """
        self.enable_api_keys = config_data.get("enable_api_keys", True)
        self.enable_auth = config_data.get("enable_auth", True)
        self.rate_limit_per_minute = config_data.get("rate_limit_per_minute", 100)
        self.cors_origins = config_data.get("cors_origins", ["http://localhost:3000"])
        self.protected_endpoints = config_data.get("protected_endpoints", ["/inference", "/models", "/predict", "/stats"])
        self.public_endpoints = config_data.get("public_endpoints", ["/health", "/auth", "/"])


def load_auth_config(config_file: str = None, environment: str = "testing") -> tuple[AuthConfig, SecurityConfig]:
    """
    Load authentication configuration from YAML file.
    
    Args:
        config_file: Path to config file (optional)
        environment: Environment name (testing, development, production)
        
    Returns:
        Tuple of (AuthConfig, SecurityConfig)
    """
    if config_file is None:
        # Try to find config file based on environment
        config_dir = Path("config")
        if config_dir.exists():
            config_file = config_dir / f"{environment}.yaml"
            if not config_file.exists():
                config_file = config_dir / "default.yaml"
        else:
            # Fallback to project root
            config_file = f"{environment}.yaml"
    
    config_file = Path(config_file)
    
    if not config_file.exists():
        logger.warning(f"Config file {config_file} not found, using defaults")
        # Return default configuration
        auth_config = AuthConfig({})
        security_config = SecurityConfig({})
        return auth_config, security_config
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        auth_config = AuthConfig(config_data.get("auth", {}))
        security_config = SecurityConfig(config_data.get("security", {}))
        
        logger.info(f"Loaded auth configuration from {config_file}")
        return auth_config, security_config
        
    except Exception as e:
        logger.error(f"Failed to load config from {config_file}: {e}")
        # Return default configuration
        auth_config = AuthConfig({})
        security_config = SecurityConfig({})
        return auth_config, security_config


def get_environment() -> str:
    """
    Get current environment from environment variable.
    
    Returns:
        Environment name
    """
    return os.getenv("TORCH_ENV", "testing").lower()

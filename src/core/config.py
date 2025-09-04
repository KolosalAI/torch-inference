"""
Core configuration management for PyTorch Inference Framework.
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Server configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    reload: bool = False
    workers: int = 1
    
    
@dataclass
class SecurityConfig:
    """Security configuration settings."""
    max_file_size_mb: int = 100
    allowed_extensions: list = field(default_factory=lambda: [".wav", ".mp3", ".flac", ".m4a", ".ogg"])
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100


@dataclass 
class AudioConfig:
    """Audio processing configuration."""
    default_tts_model: str = "speecht5_tts"
    default_stt_model: str = "whisper-base"
    max_text_length: int = 5000
    default_sample_rate: int = 16000
    supported_formats: list = field(default_factory=lambda: ["wav", "mp3", "flac"])


@dataclass
class AppConfig:
    """Main application configuration."""
    environment: str = "development"
    debug: bool = True
    project_root: str = field(default_factory=lambda: str(Path(__file__).parent.parent.parent))
    
    # Sub-configurations
    server: ServerConfig = field(default_factory=ServerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)


class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent.parent / "config"
        self._config: Optional[AppConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML files."""
        try:
            # Load default config
            default_config = self._load_yaml_file("default.yaml")
            
            # Determine environment
            env = os.getenv("ENVIRONMENT", "development")
            
            # Load environment-specific config
            env_config = self._load_yaml_file(f"{env}.yaml") or {}
            
            # Merge configurations
            merged_config = self._merge_configs(default_config or {}, env_config)
            merged_config["environment"] = env
            
            # Create config object
            self._config = self._dict_to_config(merged_config)
            
            logger.info(f"Configuration loaded for environment: {env}")
            
        except Exception as e:
            logger.warning(f"Failed to load configuration files: {e}")
            # Fall back to default configuration
            self._config = AppConfig()
            
    def _load_yaml_file(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a YAML configuration file."""
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            return None
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig object."""
        # Extract sub-configurations
        server_dict = config_dict.get("server", {})
        security_dict = config_dict.get("security", {})
        audio_dict = config_dict.get("audio", {})
        
        return AppConfig(
            environment=config_dict.get("environment", "development"),
            debug=config_dict.get("debug", True),
            project_root=config_dict.get("project_root", str(Path(__file__).parent.parent.parent)),
            server=ServerConfig(**server_dict),
            security=SecurityConfig(**security_dict),
            audio=AudioConfig(**audio_dict)
        )
    
    @property
    def config(self) -> AppConfig:
        """Get the current configuration."""
        if self._config is None:
            self._load_config()
        return self._config
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration as dictionary for backward compatibility."""
        server = self.config.server
        return {
            "host": server.host,
            "port": server.port,
            "log_level": server.log_level,
            "reload": server.reload,
            "workers": server.workers
        }
    
    def get_inference_config(self):
        """Get inference configuration - delegates to framework config manager."""
        try:
            from framework.core.config_manager import get_config_manager
            framework_config_manager = get_config_manager()
            return framework_config_manager.get_inference_config()
        except ImportError:
            logger.warning("Framework config manager not available, using basic config")
            # Return a basic mock config for fallback
            from types import SimpleNamespace
            return SimpleNamespace(
                device=SimpleNamespace(
                    device_type=SimpleNamespace(value="cpu"),
                    device_id=0,
                    use_fp16=False,
                    use_tensorrt=False,
                    use_torch_compile=False,
                    get_torch_device=lambda: "cpu"
                ),
                batch=SimpleNamespace(
                    batch_size=1,
                    max_batch_size=8
                ),
                performance=SimpleNamespace(
                    warmup_iterations=3
                )
            )
    
    @property
    def environment(self) -> str:
        """Get current environment."""
        return self.config.environment
    
    def reload(self) -> None:
        """Reload configuration from files."""
        self._load_config()


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> AppConfig:
    """Get the current application configuration."""
    return get_config_manager().config

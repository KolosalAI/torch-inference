"""
Advanced configuration management for enterprise deployments.
Provides dynamic configuration, environment management, and validation.
"""

import os
import yaml
import json
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from datetime import datetime, timezone
import logging
from collections import defaultdict
import hashlib
import copy

logger = logging.getLogger(__name__)

class ConfigEnvironment(Enum):
    """Configuration environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class ConfigFormat(Enum):
    """Configuration file formats."""
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"

@dataclass
class ConfigSource:
    """Configuration source definition."""
    name: str
    path: str
    format: ConfigFormat
    priority: int = 100
    watch: bool = False
    required: bool = True
    reload_on_change: bool = True

@dataclass
class ConfigValidationRule:
    """Configuration validation rule."""
    path: str  # dot-separated path like "model.batch_size"
    validator: Callable[[Any], bool]
    error_message: str
    required: bool = True

class ConfigurationManager:
    """Advanced configuration management system."""
    
    def __init__(self, environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT):
        self.environment = environment
        self.config_data: Dict[str, Any] = {}
        self.sources: List[ConfigSource] = []
        self.validation_rules: List[ConfigValidationRule] = []
        self.watchers: Dict[str, threading.Thread] = {}
        self.callbacks: List[Callable[[str, Any, Any], None]] = []
        
        # Thread safety
        self.config_lock = threading.RLock()
        self.source_lock = threading.RLock()
        
        # Configuration history
        self.config_history: List[Dict[str, Any]] = []
        self.max_history = 50
        
        # Cache settings
        self.cache_enabled = True
        self.cached_configs: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, float] = {}
        
        self._setup_default_rules()
        self._load_default_sources()
    
    def _setup_default_rules(self):
        """Setup default validation rules."""
        default_rules = [
            ConfigValidationRule(
                path="gpu.device_count",
                validator=lambda x: isinstance(x, int) and x > 0,
                error_message="gpu.device_count must be a positive integer"
            ),
            ConfigValidationRule(
                path="model.batch_size",
                validator=lambda x: isinstance(x, int) and x > 0,
                error_message="model.batch_size must be a positive integer"
            ),
            ConfigValidationRule(
                path="api.port",
                validator=lambda x: isinstance(x, int) and 1024 <= x <= 65535,
                error_message="api.port must be between 1024 and 65535"
            ),
            ConfigValidationRule(
                path="security.jwt_secret",
                validator=lambda x: isinstance(x, str) and len(x) >= 32,
                error_message="security.jwt_secret must be at least 32 characters"
            ),
            ConfigValidationRule(
                path="monitoring.metrics_interval",
                validator=lambda x: isinstance(x, (int, float)) and x > 0,
                error_message="monitoring.metrics_interval must be positive"
            )
        ]
        
        self.validation_rules.extend(default_rules)
    
    def _load_default_sources(self):
        """Load default configuration sources based on environment."""
        base_path = Path("config")
        
        # Base configuration (lowest priority)
        self.add_source(ConfigSource(
            name="base",
            path=str(base_path / "default.yaml"),
            format=ConfigFormat.YAML,
            priority=1000,
            required=False
        ))
        
        # Environment-specific configuration
        env_file = base_path / f"{self.environment.value}.yaml"
        if env_file.exists():
            self.add_source(ConfigSource(
                name=f"env_{self.environment.value}",
                path=str(env_file),
                format=ConfigFormat.YAML,
                priority=500
            ))
        
        # Local overrides (highest priority)
        local_file = base_path / "local.yaml"
        if local_file.exists():
            self.add_source(ConfigSource(
                name="local",
                path=str(local_file),
                format=ConfigFormat.YAML,
                priority=100,
                watch=True
            ))
        
        # Environment variables override
        self.add_source(ConfigSource(
            name="env_vars",
            path="",  # Special case for env vars
            format=ConfigFormat.JSON,  # Irrelevant for env vars
            priority=50
        ))
    
    def add_source(self, source: ConfigSource):
        """Add configuration source."""
        with self.source_lock:
            self.sources.append(source)
            self.sources.sort(key=lambda s: s.priority, reverse=True)
            
            if source.watch and source.path and os.path.exists(source.path):
                self._start_file_watcher(source)
    
    def _start_file_watcher(self, source: ConfigSource):
        """Start file watcher for configuration source."""
        def watch_file():
            last_modified = 0
            while True:
                try:
                    if os.path.exists(source.path):
                        current_modified = os.path.getmtime(source.path)
                        if current_modified > last_modified:
                            last_modified = current_modified
                            if last_modified > 0:  # Skip initial load
                                logger.info(f"Configuration file {source.path} changed, reloading")
                                self._reload_config()
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error watching config file {source.path}: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=watch_file, daemon=True)
        thread.start()
        self.watchers[source.name] = thread
    
    def _load_file_config(self, source: ConfigSource) -> Dict[str, Any]:
        """Load configuration from file."""
        if not os.path.exists(source.path):
            if source.required:
                raise FileNotFoundError(f"Required config file not found: {source.path}")
            return {}
        
        try:
            with open(source.path, 'r', encoding='utf-8') as f:
                if source.format == ConfigFormat.YAML:
                    return yaml.safe_load(f) or {}
                elif source.format == ConfigFormat.JSON:
                    return json.load(f) or {}
                else:
                    raise ValueError(f"Unsupported format: {source.format}")
        except Exception as e:
            logger.error(f"Error loading config from {source.path}: {e}")
            if source.required:
                raise
            return {}
    
    def _load_env_vars(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        prefix = "TORCH_INFERENCE_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert TORCH_INFERENCE_MODEL__BATCH_SIZE to model.batch_size
                config_key = key[len(prefix):].lower().replace('__', '.')
                
                # Try to parse as JSON first, then as string
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
                
                # Set nested value
                self._set_nested_value(config, config_key, parsed_value)
        
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], key_path: str, value: Any):
        """Set nested dictionary value using dot notation."""
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _get_nested_value(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """Get nested dictionary value using dot notation."""
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _reload_config(self):
        """Reload configuration from all sources."""
        with self.config_lock:
            old_config = copy.deepcopy(self.config_data)
            
            # Start with empty config
            merged_config = {}
            
            # Load and merge configs in priority order (highest first)
            for source in sorted(self.sources, key=lambda s: s.priority):
                try:
                    if source.name == "env_vars":
                        source_config = self._load_env_vars()
                    else:
                        source_config = self._load_file_config(source)
                    
                    if source_config:
                        merged_config = self._merge_configs(source_config, merged_config)
                        logger.debug(f"Loaded config from source: {source.name}")
                
                except Exception as e:
                    logger.error(f"Error loading config source {source.name}: {e}")
                    if source.required:
                        raise
            
            # Validate configuration
            validation_errors = self._validate_config(merged_config)
            if validation_errors:
                logger.error(f"Configuration validation failed: {validation_errors}")
                raise ValueError(f"Configuration validation failed: {'; '.join(validation_errors)}")
            
            # Update configuration
            self.config_data = merged_config
            
            # Update history
            self.config_history.append({
                'timestamp': datetime.now(timezone.utc),
                'config': copy.deepcopy(merged_config)
            })
            
            if len(self.config_history) > self.max_history:
                self.config_history.pop(0)
            
            # Clear cache
            self.cached_configs.clear()
            self.cache_timestamps.clear()
            
            # Notify callbacks
            self._notify_config_change(old_config, merged_config)
            
            logger.info("Configuration reloaded successfully")
    
    def _validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration against rules."""
        errors = []
        
        for rule in self.validation_rules:
            value = self._get_nested_value(config, rule.path)
            
            if value is None:
                if rule.required:
                    errors.append(f"Required configuration '{rule.path}' is missing")
            else:
                try:
                    if not rule.validator(value):
                        errors.append(rule.error_message)
                except Exception as e:
                    errors.append(f"Validation error for '{rule.path}': {e}")
        
        return errors
    
    def _notify_config_change(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Notify callbacks of configuration changes."""
        # Find changed keys
        changed_keys = self._find_changed_keys(old_config, new_config)
        
        for callback in self.callbacks:
            try:
                for key in changed_keys:
                    old_value = self._get_nested_value(old_config, key)
                    new_value = self._get_nested_value(new_config, key)
                    callback(key, old_value, new_value)
            except Exception as e:
                logger.error(f"Configuration callback failed: {e}")
    
    def _find_changed_keys(self, old_config: Dict[str, Any], new_config: Dict[str, Any], prefix: str = "") -> List[str]:
        """Find keys that have changed between configurations."""
        changed = []
        
        all_keys = set(self._flatten_dict(old_config).keys()) | set(self._flatten_dict(new_config).keys())
        
        for key in all_keys:
            old_value = self._get_nested_value(old_config, key)
            new_value = self._get_nested_value(new_config, key)
            
            if old_value != new_value:
                changed.append(key)
        
        return changed
    
    def _flatten_dict(self, config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary to dot notation."""
        flattened = {}
        
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.update(self._flatten_dict(value, full_key))
            else:
                flattened[full_key] = value
        
        return flattened
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        # Check cache first
        if self.cache_enabled and key in self.cached_configs:
            cache_time = self.cache_timestamps.get(key, 0)
            if time.time() - cache_time < self.cache_ttl:
                return self.cached_configs[key]
        
        with self.config_lock:
            if not self.config_data:
                self._reload_config()
            
            value = self._get_nested_value(self.config_data, key, default)
            
            # Cache the value
            if self.cache_enabled:
                self.cached_configs[key] = value
                self.cache_timestamps[key] = time.time()
            
            return value
    
    def set(self, key: str, value: Any, persist: bool = False):
        """Set configuration value."""
        with self.config_lock:
            old_value = self._get_nested_value(self.config_data, key)
            self._set_nested_value(self.config_data, key, value)
            
            # Clear cache for this key
            self.cached_configs.pop(key, None)
            self.cache_timestamps.pop(key, None)
            
            # Validate single value
            temp_config = copy.deepcopy(self.config_data)
            validation_errors = self._validate_config(temp_config)
            if validation_errors:
                # Revert change
                if old_value is not None:
                    self._set_nested_value(self.config_data, key, old_value)
                else:
                    self._delete_nested_value(self.config_data, key)
                raise ValueError(f"Configuration validation failed: {'; '.join(validation_errors)}")
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(key, old_value, value)
                except Exception as e:
                    logger.error(f"Configuration callback failed: {e}")
            
            # Persist if requested
            if persist:
                self._persist_config(key, value)
    
    def _delete_nested_value(self, config: Dict[str, Any], key_path: str):
        """Delete nested dictionary value using dot notation."""
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                return
            current = current[key]
        
        current.pop(keys[-1], None)
    
    def _persist_config(self, key: str, value: Any):
        """Persist configuration change to local config file."""
        local_source = next((s for s in self.sources if s.name == "local"), None)
        if not local_source:
            logger.warning("No local config source found for persistence")
            return
        
        try:
            # Load current local config
            if os.path.exists(local_source.path):
                with open(local_source.path, 'r', encoding='utf-8') as f:
                    local_config = yaml.safe_load(f) or {}
            else:
                local_config = {}
            
            # Update value
            self._set_nested_value(local_config, key, value)
            
            # Save back
            os.makedirs(os.path.dirname(local_source.path), exist_ok=True)
            with open(local_source.path, 'w', encoding='utf-8') as f:
                yaml.dump(local_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Persisted configuration change: {key} = {value}")
        
        except Exception as e:
            logger.error(f"Failed to persist configuration: {e}")
    
    def reload(self):
        """Manually reload configuration."""
        self._reload_config()
    
    def add_validation_rule(self, rule: ConfigValidationRule):
        """Add configuration validation rule."""
        self.validation_rules.append(rule)
    
    def add_change_callback(self, callback: Callable[[str, Any, Any], None]):
        """Add configuration change callback."""
        self.callbacks.append(callback)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration data."""
        with self.config_lock:
            if not self.config_data:
                self._reload_config()
            return copy.deepcopy(self.config_data)
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        return self.config_history[-limit:]
    
    def export_config(self, format: ConfigFormat = ConfigFormat.YAML) -> str:
        """Export current configuration."""
        config = self.get_all()
        
        if format == ConfigFormat.YAML:
            return yaml.dump(config, default_flow_style=False, indent=2)
        elif format == ConfigFormat.JSON:
            return json.dumps(config, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            'environment': self.environment.value,
            'sources': [
                {
                    'name': source.name,
                    'path': source.path,
                    'format': source.format.value,
                    'priority': source.priority,
                    'exists': os.path.exists(source.path) if source.path else True,
                    'watch': source.watch
                }
                for source in self.sources
            ],
            'config_hash': hashlib.md5(
                json.dumps(self.config_data, sort_keys=True).encode()
            ).hexdigest(),
            'last_reload': self.config_history[-1]['timestamp'].isoformat() if self.config_history else None
        }
    
    def validate_all(self) -> List[str]:
        """Validate all configuration."""
        return self._validate_config(self.config_data)
    
    def cleanup(self):
        """Clean up configuration manager."""
        # Stop file watchers
        for watcher in self.watchers.values():
            if watcher.is_alive():
                # Note: We can't gracefully stop daemon threads
                pass
        
        self.watchers.clear()
        self.callbacks.clear()
        logger.info("Configuration manager cleanup completed")


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None

def get_config_manager(environment: ConfigEnvironment = None) -> ConfigurationManager:
    """Get global configuration manager instance."""
    global _config_manager
    
    if _config_manager is None:
        if environment is None:
            # Try to determine environment from env var
            env_name = os.environ.get('TORCH_INFERENCE_ENV', 'development').lower()
            try:
                environment = ConfigEnvironment(env_name)
            except ValueError:
                environment = ConfigEnvironment.DEVELOPMENT
        
        _config_manager = ConfigurationManager(environment)
        _config_manager.reload()
    
    return _config_manager

def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value using global manager."""
    return get_config_manager().get(key, default)

def set_config(key: str, value: Any, persist: bool = False):
    """Set configuration value using global manager."""
    get_config_manager().set(key, value, persist)

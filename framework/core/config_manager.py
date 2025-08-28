"""
Configuration Manager for PyTorch Inference Framework

This module provides unified configuration management that can read from:
- Environment variables (.env files)
- YAML configuration files
- Environment-specific overrides

Configuration precedence (highest to lowest):
1. Environment variables
2. config.yaml environment-specific overrides
3. config.yaml base configuration
4. Default values
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, TYPE_CHECKING
from dotenv import load_dotenv
import logging

from .config import (
    InferenceConfig, DeviceConfig, BatchConfig, PreprocessingConfig,
    PostprocessingConfig, PerformanceConfig, CacheConfig, 
    SecurityConfig as CoreSecurityConfig,
    DeviceType, ModelType
)

if TYPE_CHECKING:
    from ..security.config import SecurityConfig as SecurityFrameworkConfig

try:
    from ..security.config import SecurityConfig as SecurityFrameworkConfig
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    SecurityFrameworkConfig = None

logger = logging.getLogger(__name__)


class ConfigManager:
    """Unified configuration manager for the inference framework."""
    
    def __init__(self, 
                 env_file: Optional[Union[str, Path]] = None,
                 config_file: Optional[Union[str, Path]] = None,
                 config_dir: Optional[Union[str, Path]] = None,
                 environment: str = "development"):
        """
        Initialize the configuration manager.
        
        Args:
            env_file: Path to .env file (defaults to .env in project root or config_dir)
            config_file: Path to config.yaml file (defaults to config.yaml in project root or config_dir)
            config_dir: Base directory for config files
            environment: Current environment (development, staging, production)
        """
        self.environment = environment
        
        # Determine base directory
        if config_dir:
            base_dir = Path(config_dir)
        else:
            base_dir = Path(__file__).parent.parent.parent
        
        # Determine file paths
        self.env_file = Path(env_file) if env_file else base_dir / ".env"
        self.config_file = Path(config_file) if config_file else base_dir / "config.yaml"
        
        # Load configurations
        self._env_config = self._load_env_config()
        self._yaml_config = self._load_yaml_config()
        
        logger.info(f"Configuration loaded for environment: {self.environment}")
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Load environment variables from .env file."""
        if self.env_file.exists():
            load_dotenv(self.env_file, override=True)
            logger.debug(f"Loaded environment configuration from {self.env_file}")
        else:
            logger.warning(f"Environment file not found: {self.env_file}")
        
        # Return environment variables as dict
        return dict(os.environ)
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_file.exists():
            logger.warning(f"YAML config file not found: {self.config_file}")
            return {}
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # Apply environment-specific overrides  
            if 'environments' in config and self.environment in config['environments']:
                env_overrides = config['environments'][self.environment]
                config = self._deep_merge(config, env_overrides)
                # Remove the environments key to avoid confusion
                if 'environments' in config:
                    del config['environments']
            
            logger.debug(f"Loaded YAML configuration from {self.config_file}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load YAML config: {e}")
            return {}
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None, config_path: Optional[str] = None) -> Any:
        """
        Get configuration value with precedence: env > yaml > default.
        
        Args:
            key: Configuration key (for env vars)
            default: Default value if not found
            config_path: Dot-separated path for YAML config (e.g., 'device.type')
        
        Returns:
            Configuration value
        """
        # Check environment variables first
        env_value = self._env_config.get(key)
        if env_value is not None:
            return self._convert_type(env_value, default)
        
        # Check YAML config
        if config_path:
            yaml_value = self._get_nested_value(self._yaml_config, config_path)
            if yaml_value is not None:
                return yaml_value
        
        return default
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get nested value from config using dot-separated path."""
        keys = path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None
    
    def _convert_type(self, value: str, reference: Any) -> Any:
        """Convert string environment variable to appropriate type."""
        if reference is None:
            return value
        
        if isinstance(reference, bool):
            return value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(reference, int):
            try:
                return int(value)
            except ValueError:
                return reference
        elif isinstance(reference, float):
            try:
                return float(value)
            except ValueError:
                return reference
        elif isinstance(reference, list):
            # Handle comma-separated lists
            if isinstance(value, str):
                return [item.strip() for item in value.split(',') if item.strip()]
            return value
        
        return value
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration."""
        return {
            'host': self.get('HOST', '0.0.0.0', 'server.host'),
            'port': self.get('PORT', 8000, 'server.port'),
            'reload': self.get('RELOAD', False, 'server.reload'),
            'log_level': self.get('LOG_LEVEL', 'INFO', 'server.log_level'),
            'workers': self.get('WORKERS', 1, 'server.workers')
        }
    
    def get_inference_config(self) -> InferenceConfig:
        """Create InferenceConfig from loaded configuration."""
        # Reload environment to catch any changes made via patch.dict
        current_env = dict(os.environ)
        
        # Device configuration
        device_type = 'auto'  # Default
        if 'DEVICE' in current_env:
            device_type = current_env['DEVICE'].lower()
        elif 'device' in self._yaml_config:
            # Support both 'device_type' and 'type' keys for flexibility
            if 'device_type' in self._yaml_config['device']:
                device_type = str(self._yaml_config['device']['device_type']).lower()
            elif 'type' in self._yaml_config['device']:
                device_type = str(self._yaml_config['device']['type']).lower()
        
        # Auto-detect device if set to 'auto'
        device_config = None
        if device_type == 'auto':
            logger.info("Device type set to 'auto', using GPU manager for optimal device detection")
            try:
                from .gpu_manager import auto_configure_device
                device_config = auto_configure_device()
                logger.info(f"Auto-configured device: {device_config.device_type.value}")
            except Exception as e:
                logger.warning(f"Failed to auto-configure device: {e}, falling back to manual configuration")
                device_config = None
        
        # Manual device configuration or fallback
        if device_config is None:
            device_config = DeviceConfig(
                device_type=DeviceType.from_string(device_type),
                device_id=current_env.get('DEVICE_ID') and int(current_env.get('DEVICE_ID')),
                use_fp16=current_env.get('USE_FP16', 'false').lower() == 'true' if 'USE_FP16' in current_env else self._yaml_config.get('device', {}).get('use_fp16', False),
                use_torch_compile=current_env.get('USE_TORCH_COMPILE', 'false').lower() == 'true' if 'USE_TORCH_COMPILE' in current_env else self._yaml_config.get('device', {}).get('use_torch_compile', False)
            )
        else:
            # Override auto-detected config with manual settings if specified
            if 'DEVICE_ID' in current_env:
                device_config.device_id = int(current_env['DEVICE_ID'])
            if 'USE_FP16' in current_env:
                device_config.use_fp16 = current_env['USE_FP16'].lower() == 'true'
            elif 'use_fp16' in self._yaml_config.get('device', {}):
                device_config.use_fp16 = self._yaml_config['device']['use_fp16']
            if 'USE_TORCH_COMPILE' in current_env:
                device_config.use_torch_compile = current_env['USE_TORCH_COMPILE'].lower() == 'true'
            elif 'use_torch_compile' in self._yaml_config.get('device', {}):
                device_config.use_torch_compile = self._yaml_config['device']['use_torch_compile']
        
        # Batch configuration - handle environment variables properly
        batch_size = 2  # Default
        if 'BATCH_SIZE' in current_env:
            try:
                batch_size = int(current_env['BATCH_SIZE'])
            except (ValueError, TypeError):
                pass
        elif self._yaml_config.get('batch', {}).get('batch_size'):
            batch_size = self._yaml_config['batch']['batch_size']
                
        min_batch_size = 1
        if 'MIN_BATCH_SIZE' in current_env:
            try:
                min_batch_size = int(current_env['MIN_BATCH_SIZE'])
            except (ValueError, TypeError):
                pass
        elif self._yaml_config.get('batch', {}).get('min_batch_size'):
            min_batch_size = self._yaml_config['batch']['min_batch_size']
            
        max_batch_size = 16
        if 'MAX_BATCH_SIZE' in current_env:
            try:
                max_batch_size = int(current_env['MAX_BATCH_SIZE'])
            except (ValueError, TypeError):
                pass
        elif self._yaml_config.get('batch', {}).get('max_batch_size'):
            max_batch_size = self._yaml_config['batch']['max_batch_size']
        
        # Ensure batch_size doesn't exceed max_batch_size
        if batch_size > max_batch_size:
            max_batch_size = max(batch_size, 16)  # Expand max_batch_size if needed
        
        batch_config = BatchConfig(
            batch_size=batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            adaptive_batching=self.get('ADAPTIVE_BATCHING', True, 'batch.adaptive_batching'),
            timeout_seconds=self.get('BATCH_TIMEOUT', 5.0, 'batch.timeout_seconds'),
            queue_size=self.get('QUEUE_SIZE', 100, 'batch.queue_size')
        )
        
        # Preprocessing configuration
        input_size = (
            self.get('INPUT_SIZE_WIDTH', 224, 'preprocessing.input_size.width'),
            self.get('INPUT_SIZE_HEIGHT', 224, 'preprocessing.input_size.height')
        )
        mean = [
            self.get('MEAN_R', 0.485, 'preprocessing.normalization.mean.0'),
            self.get('MEAN_G', 0.456, 'preprocessing.normalization.mean.1'),
            self.get('MEAN_B', 0.406, 'preprocessing.normalization.mean.2')
        ]
        std = [
            self.get('STD_R', 0.229, 'preprocessing.normalization.std.0'),
            self.get('STD_G', 0.224, 'preprocessing.normalization.std.1'),
            self.get('STD_B', 0.225, 'preprocessing.normalization.std.2')
        ]
        
        # Handle YAML list format
        yaml_mean = self._get_nested_value(self._yaml_config, 'preprocessing.normalization.mean')
        if yaml_mean and isinstance(yaml_mean, list):
            mean = yaml_mean
        
        yaml_std = self._get_nested_value(self._yaml_config, 'preprocessing.normalization.std')
        if yaml_std and isinstance(yaml_std, list):
            std = yaml_std
        
        preprocessing_config = PreprocessingConfig(
            input_size=input_size,
            mean=mean,
            std=std,
            interpolation=self.get('INTERPOLATION', 'bilinear', 'preprocessing.interpolation'),
            center_crop=self.get('CENTER_CROP', True, 'preprocessing.center_crop'),
            normalize=self.get('NORMALIZE', True, 'preprocessing.normalize'),
            to_rgb=self.get('TO_RGB', True, 'preprocessing.to_rgb')
        )
        
        # Postprocessing configuration
        postprocessing_config = PostprocessingConfig(
            threshold=self.get('THRESHOLD', 0.5, 'postprocessing.threshold'),
            nms_threshold=self.get('NMS_THRESHOLD', 0.5, 'postprocessing.nms_threshold'),
            max_detections=self.get('MAX_DETECTIONS', 100, 'postprocessing.max_detections'),
            apply_sigmoid=self.get('APPLY_SIGMOID', False, 'postprocessing.apply_sigmoid'),
            apply_softmax=self.get('APPLY_SOFTMAX', False, 'postprocessing.apply_softmax')
        )
        
        # Performance configuration
        performance_config = PerformanceConfig(
            enable_profiling=self.get('ENABLE_PROFILING', False, 'performance.enable_profiling'),
            enable_metrics=self.get('ENABLE_METRICS', True, 'performance.enable_metrics'),
            warmup_iterations=self.get('WARMUP_ITERATIONS', 3, 'performance.warmup_iterations'),
            benchmark_iterations=self.get('BENCHMARK_ITERATIONS', 10, 'performance.benchmark_iterations'),
            log_level=self.get('LOG_LEVEL', 'INFO', 'monitoring.logging.level'),
            enable_async=self.get('ENABLE_ASYNC', True, 'performance.enable_async'),
            max_workers=self.get('MAX_WORKERS', 4, 'performance.max_workers')
        )
        
        # Cache configuration
        cache_config = CacheConfig(
            enable_caching=self.get('ENABLE_CACHING', True, 'cache.enable_caching'),
            cache_size=self.get('CACHE_SIZE', 100, 'cache.cache_size'),
            cache_ttl_seconds=self.get('CACHE_TTL_SECONDS', 3600, 'cache.cache_ttl_seconds'),
            disk_cache_path=self.get('DISK_CACHE_PATH', None, 'cache.disk_cache_path')
        )
        
        # Security configuration
        allowed_extensions = self.get('ALLOWED_EXTENSIONS', ['.jpg', '.jpeg', '.png', '.bmp'], 'security.allowed_extensions')
        if isinstance(allowed_extensions, str):
            allowed_extensions = [ext.strip() for ext in allowed_extensions.split(',')]
        
        security_config = CoreSecurityConfig(
            max_file_size_mb=self.get('MAX_FILE_SIZE_MB', 100, 'security.max_file_size_mb'),
            allowed_extensions=allowed_extensions,
            validate_inputs=self.get('VALIDATE_INPUTS', True, 'security.validate_inputs'),
            sanitize_outputs=self.get('SANITIZE_OUTPUTS', True, 'security.sanitize_outputs')
        )
        
        # Model type (default to CUSTOM)
        model_type = ModelType.CUSTOM
        
        return InferenceConfig(
            model_type=model_type,
            device=device_config,
            batch=batch_config,
            preprocessing=preprocessing_config,
            postprocessing=postprocessing_config,
            performance=performance_config,
            cache=cache_config,
            security=security_config
        )
    
    def get_security_config(self) -> Optional[Any]:
        """Get security configuration if security features are enabled."""
        security_enabled = self.get('SECURITY_ENABLED', True, 'security.enabled')
        
        if not security_enabled or not SECURITY_AVAILABLE:
            return None
        
        try:
            # Create base inference config
            inference_config = self.get_inference_config()
            
            # Create security config with inference config
            security_config = SecurityFrameworkConfig(inference=inference_config)
            
            # Set basic properties
            security_config.environment = self.get('ENVIRONMENT', 'development', 'security.environment')
            security_config.tenant_id = self.get('TENANT_ID', None, 'security.tenant_id')
            security_config.deployment_id = self.get('DEPLOYMENT_ID', '', 'security.deployment_id')
            security_config.version = self.get('VERSION', '1.0.0', 'app.version')
            
            # Auth configuration
            security_config.auth.secret_key = self.get('JWT_SECRET_KEY', '', 'security.auth.secret_key')
            security_config.auth.oauth2_client_id = self.get('OAUTH2_CLIENT_ID', '', 'security.auth.oauth2_client_id')
            security_config.auth.oauth2_client_secret = self.get('OAUTH2_CLIENT_SECRET', '', 'security.auth.oauth2_client_secret')
            
            # Security configuration
            security_config.security.enable_encryption_at_rest = self.get('ENABLE_ENCRYPTION_AT_REST', False, 'security.security.enable_encryption_at_rest')
            security_config.security.rate_limit_requests_per_minute = self.get('RATE_LIMIT_RPM', 100, 'security.security.rate_limit_requests_per_minute')
            security_config.security.enable_audit_logging = self.get('ENABLE_AUDIT_LOGGING', False, 'security.security.enable_audit_logging')
            
            # Monitoring configuration
            security_config.monitoring.jaeger_endpoint = self.get('JAEGER_ENDPOINT', '', 'monitoring.tracing.jaeger_endpoint')
            security_config.monitoring.metrics_port = self.get('METRICS_PORT', 9090, 'monitoring.metrics.port')
            security_config.monitoring.log_level = self.get('LOG_LEVEL', 'INFO', 'monitoring.logging.level')
            
            # Integration configuration
            security_config.integration.database_url = self.get('DATABASE_URL', '', 'security.integration.database_url')
            security_config.integration.cache_url = self.get('CACHE_URL', 'redis://localhost:6379/0', 'security.integration.cache_url')
            security_config.integration.message_broker_url = self.get('MESSAGE_BROKER_URL', '', 'security.integration.message_broker_url')
            
            return security_config
            
        except ImportError:
            logger.warning("Security features not available")
            return None
        except Exception as e:
            logger.error(f"Failed to create security config: {e}")
            return None
    
    def reload_config(self):
        """Reload configuration from files."""
        self._env_config = self._load_env_config()
        self._yaml_config = self._load_yaml_config()
        logger.info("Configuration reloaded")
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration for debugging/logging."""
        return {
            'environment': self.environment,
            'env_file': str(self.env_file),
            'config_file': str(self.config_file),
            'server': self.get_server_config(),
            'enterprise_enabled': self.get('ENTERPRISE_ENABLED', False, 'enterprise.enabled')
        }


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(environment: str = None) -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    
    if _config_manager is None or (environment and _config_manager.environment != environment):
        # Determine environment from various sources
        if not environment:
            environment = os.getenv('ENVIRONMENT', 'development')
        
        _config_manager = ConfigManager(environment=environment)
    
    return _config_manager


def set_config_manager(config_manager: ConfigManager):
    """Set the global configuration manager instance."""
    global _config_manager
    _config_manager = config_manager

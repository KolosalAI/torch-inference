"""
Core configuration management for PyTorch Inference Framework.

This module provides comprehensive configuration management using Pydantic for 
validation, with support for environment-specific configurations and proper 
error handling.
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator, validator, root_validator
import torch

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """Supported device types for inference."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class OptimizationType(str, Enum):
    """Supported optimization types."""
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    QUANTIZATION = "quantization"
    JIT = "jit"
    NONE = "none"


class LogLevel(str, Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TensorRTConfig(BaseModel):
    """TensorRT optimization configuration."""
    precision: str = Field(default="fp16", pattern="^(fp32|fp16|int8)$")
    max_batch_size: int = Field(default=32, gt=0, le=1024)
    workspace_size: int = Field(default=1<<30, gt=0)  # 1GB
    enable_dynamic_shapes: bool = True
    optimization_level: int = Field(default=3, ge=0, le=5)
    
    class Config:
        validate_assignment = True


class ONNXConfig(BaseModel):
    """ONNX optimization configuration."""
    providers: List[str] = Field(default=["CUDAExecutionProvider", "CPUExecutionProvider"])
    optimization_level: str = Field(default="all", pattern="^(basic|extended|all)$")
    enable_profiling: bool = False
    session_options: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True


class QuantizationConfig(BaseModel):
    """Quantization configuration."""
    enabled: bool = True
    dtype: str = Field(default="int8", pattern="^(int8|qint8|uint8)$")
    calibration_dataset_size: int = Field(default=100, gt=0, le=10000)
    enable_qat: bool = False  # Quantization Aware Training
    
    class Config:
        validate_assignment = True


class OptimizationConfig(BaseModel):
    """Model optimization configuration."""
    enabled: bool = True
    auto_select: bool = True
    fallback_enabled: bool = True
    benchmark_all: bool = False
    
    tensorrt: Optional[TensorRTConfig] = None
    onnx: Optional[ONNXConfig] = None
    quantization: Optional[QuantizationConfig] = None
    
    @model_validator(mode='after')
    def validate_optimization_config(self):
        """Validate optimization configuration."""
        if self.enabled and self.auto_select:
            # Auto-populate optimization configs if auto_select is enabled
            if not self.tensorrt and torch.cuda.is_available():
                self.tensorrt = TensorRTConfig()
            if not self.onnx:
                self.onnx = ONNXConfig()
            if not self.quantization:
                self.quantization = QuantizationConfig()
        
        return self
    
    class Config:
        validate_assignment = True


class DeviceConfig(BaseModel):
    """Device configuration for inference."""
    device_type: DeviceType = DeviceType.CUDA  # Default to CUDA for maximum performance
    device_id: Optional[int] = Field(default=None, ge=0)
    use_fp16: bool = Field(default=False)
    use_mixed_precision: bool = Field(default=False)
    memory_fraction: float = Field(default=0.9, gt=0.0, le=1.0)
    
    @field_validator('device_type')
    @classmethod
    def validate_device_type(cls, v):
        """Validate and auto-detect device type if needed."""
        if v == DeviceType.AUTO:
            if torch.cuda.is_available():
                logger.info("Auto-detected CUDA device for maximum performance")
                return DeviceType.CUDA
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Auto-detected MPS device")
                return DeviceType.MPS
            else:
                logger.warning("No GPU available - falling back to CPU (performance will be degraded)")
                return DeviceType.CPU
        elif v == DeviceType.CUDA and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available - falling back to CPU")
            return DeviceType.CPU
        return v
    
    @field_validator('use_fp16')
    @classmethod
    def validate_fp16(cls, v, info):
        """Validate FP16 usage."""
        device_type = info.data.get('device_type')
        if v and device_type == DeviceType.CPU:
            logger.warning("FP16 not supported on CPU, disabling")
            return False
        return v
    
    def get_torch_device(self) -> torch.device:
        """Get PyTorch device object."""
        if self.device_type == DeviceType.CUDA:
            device_id = self.device_id if self.device_id is not None else 0
            return torch.device(f"cuda:{device_id}")
        elif self.device_type == DeviceType.MPS:
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    class Config:
        validate_assignment = True
        use_enum_values = True


class BatchConfig(BaseModel):
    """Batch processing configuration."""
    batch_size: int = Field(default=1, gt=0, le=1024)
    max_batch_size: int = Field(default=32, gt=0, le=1024)
    enable_dynamic_batching: bool = True
    batch_timeout_ms: float = Field(default=100.0, gt=0)
    
    @validator('max_batch_size')
    def validate_max_batch_size(cls, v, values):
        """Ensure max_batch_size >= batch_size."""
        batch_size = values.get('batch_size', 1)
        if v < batch_size:
            raise ValueError(f"max_batch_size ({v}) must be >= batch_size ({batch_size})")
        return v
    
    class Config:
        validate_assignment = True


class AutoscalingConfig(BaseModel):
    """Autoscaling configuration."""
    enabled: bool = False
    scale_to_zero_delay: float = Field(default=300.0, gt=0)  # 5 minutes
    max_loaded_models: int = Field(default=5, gt=0, le=50)
    preload_popular_models: bool = True
    memory_threshold: float = Field(default=0.8, gt=0.0, le=1.0)
    
    class Config:
        validate_assignment = True


class SecurityConfig(BaseModel):
    """Security configuration."""
    enable_auth: bool = Field(default=False)
    enable_api_keys: bool = Field(default=False)
    enable_rate_limiting: bool = Field(default=True)
    rate_limit_per_minute: int = Field(default=100, gt=0)
    max_file_size_mb: int = Field(default=100, gt=0)
    allowed_extensions: List[str] = Field(default=[".wav", ".mp3", ".flac", ".m4a", ".ogg"])
    enable_cors: bool = Field(default=True)
    allowed_origins: List[str] = Field(default=["http://localhost:3000", "http://localhost:8080"])
    jwt_secret_key: Optional[str] = None
    
    @validator('jwt_secret_key')
    def validate_jwt_secret(cls, v, values):
        """Validate JWT secret key if auth is enabled."""
        if values.get('enable_auth') and not v:
            raise ValueError("jwt_secret_key is required when authentication is enabled")
        return v
    
    class Config:
        validate_assignment = True


class PerformanceConfig(BaseModel):
    """Performance configuration."""
    warmup_iterations: int = Field(default=3, ge=0, le=100)
    enable_cuda_graphs: bool = Field(default=True)
    enable_jit_compilation: bool = Field(default=True)
    thread_pool_size: Optional[int] = Field(default=None, gt=0)
    enable_memory_pool: bool = Field(default=True)
    gc_frequency: int = Field(default=100, gt=0)  # Garbage collection frequency
    
    @validator('thread_pool_size')
    def validate_thread_pool_size(cls, v):
        """Validate thread pool size."""
        if v is None:
            import os
            return min(32, (os.cpu_count() or 1) + 4)
        return v
    
    class Config:
        validate_assignment = True


class AudioConfig(BaseModel):
    """Audio processing configuration."""
    default_tts_model: str = "speecht5_tts"
    default_stt_model: str = "whisper-base"
    max_text_length: int = Field(default=5000, gt=0, le=50000)
    default_sample_rate: int = Field(default=16000, gt=0)
    supported_formats: List[str] = Field(default=["wav", "mp3", "flac"])
    enable_noise_reduction: bool = Field(default=True)
    enable_voice_activity_detection: bool = Field(default=True)
    
    class Config:
        validate_assignment = True


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, gt=0, le=65535)
    log_level: LogLevel = LogLevel.INFO
    reload: bool = Field(default=False)
    workers: int = Field(default=1, gt=0, le=32)
    access_log: bool = Field(default=True)
    max_request_size: int = Field(default=100 * 1024 * 1024)  # 100MB
    
    class Config:
        validate_assignment = True
        use_enum_values = True


class InferenceConfig(BaseModel):
    """Main inference configuration."""
    model_path: Optional[str] = None
    fallback_model: Optional[str] = None  # Fallback model for inference service
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    autoscaling: AutoscalingConfig = Field(default_factory=AutoscalingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    max_memory_usage: Optional[int] = None  # MB
    default_timeout: Optional[float] = 30.0  # Default timeout in seconds
    
    @validator('model_path')
    def validate_model_path(cls, v):
        """Validate model path if provided."""
        if v and not Path(v).exists():
            logger.warning(f"Model path does not exist: {v}")
        return v
    
    class Config:
        validate_assignment = True


class AppConfig(BaseModel):
    """Main application configuration."""
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    project_root: str = Field(default_factory=lambda: str(Path(__file__).parent.parent.parent))
    
    # Sub-configurations
    server: ServerConfig = Field(default_factory=ServerConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment value."""
        valid_envs = ['development', 'testing', 'staging', 'production']
        if v not in valid_envs:
            logger.warning(f"Unknown environment '{v}', using 'development'")
            return 'development'
        return v
    
    @model_validator(mode='after')
    def validate_config_consistency(self):
        """Validate configuration consistency."""
        if self.environment == 'production' and self.debug:
            logger.warning("Debug mode enabled in production environment")
            self.debug = False
        
        return self
    
    class Config:
        validate_assignment = True


class ConfigManager:
    """
    Centralized configuration manager with validation and environment support.
    
    Provides robust configuration loading with Pydantic validation, environment
    variable support, and graceful error handling.
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent.parent / "config"
        self._config: Optional[AppConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load and validate configuration from multiple sources."""
        try:
            # Load default config
            default_config = self._load_yaml_file("default.yaml") or {}
            
            # Determine environment
            env = os.getenv("ENVIRONMENT", "development")
            
            # Load environment-specific config
            env_config = self._load_yaml_file(f"{env}.yaml") or {}
            
            # Load environment variables
            env_vars = self._load_environment_variables()
            
            # Merge configurations (env vars have highest priority)
            merged_config = self._merge_configs(default_config, env_config, env_vars)
            merged_config["environment"] = env
            
            # Create and validate config object
            try:
                self._config = AppConfig(**merged_config)
                logger.info(f"Configuration loaded and validated for environment: {env}")
            except Exception as e:
                raise ConfigurationError(
                    config_field="root",
                    details=f"Configuration validation failed: {e}",
                    context={"environment": env, "merged_config_keys": list(merged_config.keys())}
                )
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            
            logger.error(f"Failed to load configuration: {e}")
            # Fall back to default configuration
            try:
                self._config = AppConfig()
                logger.warning("Using default configuration due to load failure")
            except Exception as fallback_error:
                raise ConfigurationError(
                    config_field="fallback",
                    details=f"Even default configuration failed: {fallback_error}",
                    cause=e
                )
    
    def _load_yaml_file(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a YAML configuration file with error handling."""
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            logger.debug(f"Configuration file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                logger.debug(f"Loaded configuration from {filename}")
                return content or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(
                config_field=filename,
                details=f"YAML parsing error: {e}",
                context={"filepath": str(filepath)}
            )
        except Exception as e:
            raise ConfigurationError(
                config_field=filename,
                details=f"File reading error: {e}",
                context={"filepath": str(filepath)}
            )
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Map environment variables to config structure
        env_mappings = {
            "TORCH_INFERENCE_HOST": ("server", "host"),
            "TORCH_INFERENCE_PORT": ("server", "port"),
            "TORCH_INFERENCE_LOG_LEVEL": ("server", "log_level"),
            "TORCH_INFERENCE_DEBUG": ("debug",),
            "TORCH_INFERENCE_DEVICE": ("inference", "device", "device_type"),
            "TORCH_INFERENCE_BATCH_SIZE": ("inference", "batch", "batch_size"),
            "TORCH_INFERENCE_MAX_BATCH_SIZE": ("inference", "batch", "max_batch_size"),
            "TORCH_INFERENCE_ENABLE_AUTH": ("security", "enable_auth"),
            "TORCH_INFERENCE_JWT_SECRET": ("security", "jwt_secret_key"),
            "TORCH_INFERENCE_RATE_LIMIT": ("security", "rate_limit_per_minute"),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(value, env_var)
                self._set_nested_config(env_config, config_path, converted_value)
        
        return env_config
    
    def _convert_env_value(self, value: str, env_var: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion for specific variables
        int_vars = ["PORT", "BATCH_SIZE", "MAX_BATCH_SIZE", "RATE_LIMIT"]
        if any(var in env_var for var in int_vars):
            try:
                return int(value)
            except ValueError:
                logger.warning(f"Invalid integer value for {env_var}: {value}")
                return value
        
        return value
    
    def _set_nested_config(self, config: Dict[str, Any], path: tuple, value: Any):
        """Set a nested configuration value."""
        for key in path[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[path[-1]] = value
    
    def _merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge multiple configuration dictionaries."""
        result = {}
        
        for config in configs:
            for key, value in config.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._merge_configs(result[key], value)
                else:
                    result[key] = value
        
        return result
    
    @property
    def config(self) -> AppConfig:
        """Get the current validated configuration."""
        if self._config is None:
            self._load_config()
        return self._config
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration as dictionary for backward compatibility."""
        server = self.config.server
        return {
            "host": server.host,
            "port": server.port,
            "log_level": server.log_level.value,
            "reload": server.reload,
            "workers": server.workers
        }
    
    def get_inference_config(self) -> InferenceConfig:
        """Get inference configuration."""
        return self.config.inference
    
    @property
    def environment(self) -> str:
        """Get current environment."""
        return self.config.environment
    
    def reload(self) -> None:
        """Reload configuration from files."""
        logger.info("Reloading configuration...")
        self._load_config()
        logger.info("Configuration reloaded successfully")
    
    def validate_config(self) -> List[str]:
        """Validate current configuration and return any issues."""
        issues = []
        
        try:
            # Re-validate the current config
            AppConfig(**self.config.dict())
        except Exception as e:
            issues.append(f"Configuration validation failed: {e}")
        
        # Additional custom validations
        config = self.config
        
        # Check device compatibility
        if config.inference.device.device_type == DeviceType.CUDA and not torch.cuda.is_available():
            issues.append("CUDA device specified but CUDA is not available")
        
        # Check memory settings
        if config.inference.max_memory_usage and config.inference.max_memory_usage < 1024:
            issues.append("max_memory_usage seems too low (< 1GB)")
        
        # Check batch size settings
        if config.inference.batch.max_batch_size > 128:
            issues.append("Very large max_batch_size may cause memory issues")
        
        return issues


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


def get_settings() -> AppConfig:
    """Get the current application settings (alias for get_config)."""
    return get_config()


def validate_configuration() -> List[str]:
    """Validate the current configuration and return any issues."""
    return get_config_manager().validate_config()

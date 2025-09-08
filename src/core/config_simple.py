"""
Simplified config for testing core functionality.
"""

import os
import yaml
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from pydantic import BaseModel, Field
from enum import Enum

class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"

class TensorRTConfig(BaseModel):
    enabled: bool = False
    precision: str = "fp16"

class ONNXConfig(BaseModel):
    enabled: bool = False

class QuantizationConfig(BaseModel):
    enabled: bool = False

class OptimizationConfig(BaseModel):
    enabled: bool = False
    auto_select: bool = True
    tensorrt: Optional[TensorRTConfig] = None
    onnx: Optional[ONNXConfig] = None
    quantization: Optional[QuantizationConfig] = None

class DeviceConfig(BaseModel):
    device_type: DeviceType = DeviceType.CUDA  # Default to CUDA for maximum performance
    device_index: int = 0
    use_fp16: bool = False
    memory_fraction: float = 0.9

class InferenceConfig(BaseModel):
    max_batch_size: int = 32
    timeout_seconds: int = 30
    enable_caching: bool = True
    device_config: DeviceConfig = DeviceConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    model_config: Dict[str, Any] = {}

class AutoscalingConfig(BaseModel):
    enabled: bool = False
    min_replicas: int = 1
    max_replicas: int = 5
    target_cpu_utilization: float = 0.7
    target_response_time_ms: float = 100.0
    max_queue_length: int = 10
    check_interval_seconds: int = 30
    cooldown_period_seconds: int = 60
    history_window_minutes: int = 5

class RateLimitingConfig(BaseModel):
    enabled: bool = False
    requests_per_minute: int = 60

class SecurityConfig(BaseModel):
    enable_auth: bool = False
    jwt_secret: Optional[str] = None
    create_default_admin: bool = True
    rate_limiting: RateLimitingConfig = RateLimitingConfig()

class ServerConfig(BaseModel):
    host: str = "localhost"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"

class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None

class AppConfig(BaseModel):
    environment: str = "development"
    debug: bool = False
    server: ServerConfig = ServerConfig()
    inference: InferenceConfig = InferenceConfig()
    logging: LoggingConfig = LoggingConfig()
    security: SecurityConfig = SecurityConfig()

# Global config instance
_config_instance: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config_instance
    
    if _config_instance is None:
        # Create default config
        _config_instance = AppConfig()
        
        # Try to load from config file if it exists
        config_paths = ["config.yaml", "config/default.yaml"]
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config_data = yaml.safe_load(f)
                    _config_instance = AppConfig(**config_data)
                    break
                except Exception as e:
                    print(f"Warning: Failed to load config from {config_path}: {e}")
    
    return _config_instance

def get_config_manager():
    """Get config manager (simplified for compatibility)."""
    return get_config()

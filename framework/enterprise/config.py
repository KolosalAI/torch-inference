"""
Enterprise configuration management.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class DeploymentMode(Enum):
    """Deployment modes."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class EnterpriseConfig:
    """Enterprise configuration settings."""
    
    deployment_mode: DeploymentMode = DeploymentMode.DEVELOPMENT
    enable_monitoring: bool = True
    enable_security: bool = True
    enable_multi_tenancy: bool = False
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    resource_limits: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.resource_limits is None:
            self.resource_limits = {
                'max_memory_mb': 4096,
                'max_cpu_cores': 4,
                'max_gpu_memory_mb': 8192
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'deployment_mode': self.deployment_mode.value,
            'enable_monitoring': self.enable_monitoring,
            'enable_security': self.enable_security,
            'enable_multi_tenancy': self.enable_multi_tenancy,
            'max_concurrent_requests': self.max_concurrent_requests,
            'request_timeout': self.request_timeout,
            'resource_limits': self.resource_limits
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnterpriseConfig':
        """Create config from dictionary."""
        config_dict = config_dict.copy()
        if 'deployment_mode' in config_dict:
            config_dict['deployment_mode'] = DeploymentMode(config_dict['deployment_mode'])
        return cls(**config_dict)


# Default enterprise configuration
DEFAULT_ENTERPRISE_CONFIG = EnterpriseConfig()

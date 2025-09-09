"""
Framework core modules.
"""

# Main classes and functions
from .config import MultiGPUConfig, DeviceConfig, InferenceConfig
from .gpu_manager import GPUManager, get_gpu_manager, setup_multi_gpu, validate_multi_gpu_setup
from .multi_gpu_manager import MultiGPUManager
# Don't import InferenceEngine here to avoid circular imports
from .gpu_detection import GPUDetector, GPUInfo

__all__ = [
    'MultiGPUConfig',
    'DeviceConfig', 
    'InferenceConfig',
    'GPUManager',
    'get_gpu_manager',
    'setup_multi_gpu',
    'validate_multi_gpu_setup',
    'MultiGPUManager',
    'GPUDetector',
    'GPUInfo'
]

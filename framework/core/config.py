"""
Production-level configuration system for PyTorch inference framework.

This module provides a centralized, type-safe configuration system with support for
environment variables, validation, and different model types.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from pathlib import Path
import os
import torch
from abc import ABC, abstractmethod


class ModelType(Enum):
    """Supported model types."""
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    REGRESSION = "regression"
    TTS = "text_to_speech"
    STT = "speech_to_text"
    CUSTOM = "custom"


class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    VULKAN = "vulkan"  # Vulkan compute
    AUTO = "auto"
    
    @classmethod
    def from_string(cls, value: str) -> "DeviceType":
        """Create DeviceType from string value."""
        if not value:
            return cls.AUTO
        
        value = value.lower()
        for device_type in cls:
            if device_type.value == value:
                return device_type
        
        # For explicitly invalid device types, raise an error
        valid_values = [dt.value for dt in cls]
        raise ValueError(f"Invalid device type: '{value}'. Must be one of: {valid_values}")


class OptimizationLevel(Enum):
    """Optimization levels for inference."""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"


@dataclass
class DeviceConfig:
    """Device and hardware configuration with conservative defaults."""
    device_type: DeviceType = DeviceType.AUTO
    device_id: Optional[int] = None
    use_fp16: bool = False  # Conservative default - let optimizer enable this
    use_int8: bool = False
    use_tensorrt: bool = False
    use_torch_compile: bool = False  # Conservative default - let optimizer enable this  
    compile_mode: str = "reduce-overhead"  # Balanced optimization
    memory_fraction: float = 0.9  # Fraction of CUDA memory to allocate
    
    # Enhanced JIT optimization options
    use_vulkan: bool = False  # Enable Vulkan compute acceleration
    use_numba: bool = False   # Enable Numba JIT compilation
    jit_strategy: str = "auto"  # JIT optimization strategy
    numba_target: str = "cpu"   # Numba target: cpu, cuda, parallel
    vulkan_device_id: Optional[int] = None  # Specific Vulkan device
    
    def __post_init__(self):
        """Validate device configuration after initialization."""
        if isinstance(self.device_type, str):
            # Convert string to DeviceType enum if possible
            try:
                self.device_type = DeviceType(self.device_type)
            except ValueError:
                # Check if it's a valid device string that torch would accept
                valid_devices = ['cpu', 'cuda', 'mps']
                if self.device_type not in valid_devices:
                    raise ValueError(f"Invalid device type: {self.device_type}. Must be one of {valid_devices} or a DeviceType enum value.")
    
    def get_torch_device(self) -> torch.device:
        """Get the actual torch device."""
        if self.device_type == DeviceType.AUTO:
            # Use GPU manager for auto-detection
            try:
                from .gpu_manager import auto_configure_device
                auto_config = auto_configure_device()
                return auto_config.get_torch_device()
            except Exception:
                # Fallback to manual detection
                if torch.cuda.is_available():
                    device_str = "cuda"
                    if self.device_id is not None:
                        device_str = f"cuda:{self.device_id}"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device_str = "mps"
                else:
                    device_str = "cpu"
        elif self.device_type == DeviceType.VULKAN:
            # Vulkan devices map to CPU for PyTorch, but with Vulkan acceleration
            device_str = "cpu"
        else:
            # Handle both DeviceType enum and string values
            if isinstance(self.device_type, str):
                device_str = self.device_type
            else:
                device_str = self.device_type.value
            if (isinstance(self.device_type, DeviceType) and self.device_type == DeviceType.CUDA) or \
               (isinstance(self.device_type, str) and self.device_type == "cuda"):
                if self.device_id is not None:
                    device_str = f"cuda:{self.device_id}"
        
        return torch.device(device_str)


@dataclass
class BatchConfig:
    """Batch processing configuration."""
    batch_size: int = 1
    min_batch_size: int = 1
    max_batch_size: int = 16
    adaptive_batching: bool = True
    timeout_seconds: float = 30.0
    queue_size: int = 100
    
    def __post_init__(self):
        """Validate batch configuration after initialization."""
        if self.batch_size > self.max_batch_size:
            raise ValueError(f"batch_size ({self.batch_size}) cannot be greater than max_batch_size ({self.max_batch_size})")
        if self.min_batch_size > self.batch_size:
            raise ValueError(f"min_batch_size ({self.min_batch_size}) cannot be greater than batch_size ({self.batch_size})")
        if self.min_batch_size < 1:
            raise ValueError("min_batch_size must be at least 1")


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    input_size: Tuple[int, int] = (224, 224)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    interpolation: str = "bilinear"
    center_crop: bool = True
    normalize: bool = True
    to_rgb: bool = True


@dataclass
class PostprocessingConfig:
    """Postprocessing configuration."""
    threshold: float = 0.5
    nms_threshold: float = 0.5
    max_detections: int = 100
    apply_sigmoid: bool = False
    apply_softmax: bool = False


@dataclass
class PerformanceConfig:
    """Performance and monitoring configuration with conservative defaults."""
    enable_profiling: bool = False
    enable_metrics: bool = True
    warmup_iterations: int = 5
    benchmark_iterations: int = 10
    log_level: str = "INFO"
    enable_async: bool = True
    max_workers: int = 8
    max_concurrent_requests: int = 10  # Conservative default


@dataclass
class CacheConfig:
    """Caching configuration."""
    enable_caching: bool = True
    cache_size: int = 100
    cache_ttl_seconds: int = 3600
    disk_cache_path: Optional[Path] = None


@dataclass
class PostDownloadOptimizationConfig:
    """Configuration for post-download model optimizations."""
    enable_optimization: bool = True
    enable_quantization: bool = True
    quantization_method: str = "dynamic"  # dynamic, static, qat, fx
    enable_low_rank_optimization: bool = True
    low_rank_method: str = "svd"  # svd, tucker, hlrtf
    target_compression_ratio: float = 0.7  # Target 30% reduction
    enable_tensor_factorization: bool = True
    preserve_accuracy_threshold: float = 0.02  # Max 2% accuracy loss
    enable_structured_pruning: bool = False  # More aggressive, disabled by default
    auto_select_best_method: bool = True  # Automatically choose best optimization
    benchmark_optimizations: bool = True  # Benchmark before/after optimization
    save_optimized_model: bool = True  # Save optimized version separately


@dataclass
class SecurityConfig:
    """Security and safety configuration."""
    max_file_size_mb: int = 100
    allowed_extensions: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp"])
    validate_inputs: bool = True
    sanitize_outputs: bool = True


@dataclass
class InferenceConfig:
    """Main inference configuration."""
    model_type: ModelType = ModelType.CUSTOM
    device: DeviceConfig = field(default_factory=lambda: DeviceConfig(device_type=DeviceType.AUTO))
    batch: BatchConfig = field(default_factory=BatchConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    post_download_optimization: PostDownloadOptimizationConfig = field(default_factory=PostDownloadOptimizationConfig)
    
    # Custom parameters for specific model types
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    # Property accessors for common configuration values
    @property
    def num_classes(self) -> Optional[int]:
        """Get number of classes from custom params."""
        return self.custom_params.get("num_classes")
    
    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        """Get input size from preprocessing config."""
        return self.preprocessing.input_size
    
    @property
    def threshold(self) -> float:
        """Get threshold from postprocessing config."""
        return self.postprocessing.threshold
    
    @property
    def optimizations(self) -> Dict[str, Any]:
        """Get optimization settings as a dictionary."""
        return {
            "tensorrt": self.device.use_tensorrt,
            "fp16": self.device.use_fp16,
            "torch_compile": self.device.use_torch_compile,
            "adaptive_batching": self.batch.adaptive_batching,
            "profiling": self.performance.enable_profiling
        }
    
    @classmethod
    def from_env(cls) -> "InferenceConfig":
        """Create config from environment variables."""
        config = cls()
        
        # Device configuration
        if os.getenv("DEVICE"):
            config.device.device_type = DeviceType(os.getenv("DEVICE", "auto"))
        if os.getenv("DEVICE_ID"):
            config.device.device_id = int(os.getenv("DEVICE_ID"))
        if os.getenv("USE_FP16"):
            config.device.use_fp16 = os.getenv("USE_FP16", "false").lower() == "true"
        
        # Batch configuration
        if os.getenv("BATCH_SIZE"):
            config.batch.batch_size = int(os.getenv("BATCH_SIZE"))
        if os.getenv("MAX_BATCH_SIZE"):
            config.batch.max_batch_size = int(os.getenv("MAX_BATCH_SIZE"))
        
        # Performance configuration
        if os.getenv("MAX_WORKERS"):
            config.performance.max_workers = int(os.getenv("MAX_WORKERS"))
        if os.getenv("LOG_LEVEL"):
            config.performance.log_level = os.getenv("LOG_LEVEL")
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "InferenceConfig":
        """Create config from dictionary."""
        # This would implement recursive dataclass creation from dict
        # For brevity, simplified implementation
        config = cls()
        
        if "device" in config_dict:
            device_dict = config_dict["device"]
            if "device_type" in device_dict:
                config.device.device_type = DeviceType(device_dict["device_type"])
            # ... other fields
        
        return config
    
    def validate(self) -> bool:
        """Validate configuration."""
        # Validate device configuration
        if self.device.device_type == DeviceType.CUDA and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        
        # Validate batch sizes
        if self.batch.min_batch_size > self.batch.max_batch_size:
            raise ValueError("min_batch_size cannot be greater than max_batch_size")
        
        if self.batch.batch_size < self.batch.min_batch_size:
            self.batch.batch_size = self.batch.min_batch_size
        elif self.batch.batch_size > self.batch.max_batch_size:
            self.batch.batch_size = self.batch.max_batch_size
        
        # Validate preprocessing
        if len(self.preprocessing.mean) != 3 or len(self.preprocessing.std) != 3:
            raise ValueError("Mean and std must have exactly 3 values (RGB)")
        
        # Validate thresholds
        if not 0 <= self.postprocessing.threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        return True


# Factory for creating model-specific configurations
class ConfigFactory:
    """Factory for creating model-specific configurations."""
    
    @staticmethod
    def create_classification_config(
        num_classes: int,
        input_size: Tuple[int, int] = (224, 224),
        use_softmax: bool = True
    ) -> InferenceConfig:
        """Create configuration for image classification."""
        config = InferenceConfig()
        config.model_type = ModelType.CLASSIFICATION
        config.preprocessing.input_size = input_size
        config.postprocessing.apply_softmax = use_softmax
        config.custom_params = {"num_classes": num_classes}
        return config
    
    @staticmethod
    def create_segmentation_config(
        input_size: Tuple[int, int] = (640, 640),
        threshold: float = 0.5,
        min_contour_area: int = 100
    ) -> InferenceConfig:
        """Create configuration for image segmentation."""
        config = InferenceConfig()
        config.model_type = ModelType.SEGMENTATION
        config.preprocessing.input_size = input_size
        config.postprocessing.threshold = threshold
        config.custom_params = {
            "min_contour_area": min_contour_area,
            "max_contours": 100
        }
        return config
    
    @staticmethod
    def create_detection_config(
        input_size: Tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        max_detections: int = 100
    ) -> InferenceConfig:
        """Create configuration for object detection."""
        config = InferenceConfig()
        config.model_type = ModelType.DETECTION
        config.preprocessing.input_size = input_size
        config.postprocessing.threshold = confidence_threshold
        config.postprocessing.nms_threshold = nms_threshold
        config.postprocessing.max_detections = max_detections
        return config
    
    @staticmethod
    def create_optimized_config(
        enable_tensorrt: bool = False,
        enable_fp16: bool = False,
        enable_torch_compile: bool = False,
        enable_cuda: bool = None
    ) -> InferenceConfig:
        """Create configuration optimized for performance."""
        config = InferenceConfig()
        
        # Enable performance optimizations
        config.device.use_fp16 = enable_fp16
        config.device.use_tensorrt = enable_tensorrt
        config.device.use_torch_compile = enable_torch_compile
        
        # Auto-detect CUDA if not specified
        if enable_cuda is None:
            try:
                import torch
                enable_cuda = torch.cuda.is_available()
            except ImportError:
                enable_cuda = False
        
        if enable_cuda:
            config.device.device_type = "cuda"
        
        # Optimize batch settings
        config.batch.adaptive_batching = True
        config.batch.max_batch_size = 32
        
        # Enable performance monitoring
        config.performance.enable_profiling = True
        config.performance.enable_metrics = True
        
        return config


# Global configuration instance
_global_config: Optional[InferenceConfig] = None


def get_global_config() -> InferenceConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = InferenceConfig.from_env()
        _global_config.validate()
    return _global_config


def set_global_config(config: InferenceConfig) -> None:
    """Set the global configuration instance."""
    global _global_config
    config.validate()
    _global_config = config

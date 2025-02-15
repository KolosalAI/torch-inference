# utils/config.py
import logging
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

# ------------------------------------------------------------------------------
# Base Configuration
# ------------------------------------------------------------------------------
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "inference.log",
            "formatter": "standard",
            "level": "DEBUG"
        }
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True
        }
    }
}

# ------------------------------------------------------------------------------
# Segmentation Configuration
# ------------------------------------------------------------------------------
SEGMENTATION_CONFIG: Dict[str, Any] = {
    # Model parameters
    "model_dir": "models/model_store",
    "model_name": "segmentation_unet.pt",
    "trt_engine_name": "segmentation_fp16.trt",
    
    # Input processing
    "input_size": (512, 512),
    "mean": (0.485, 0.456, 0.406),  # ImageNet normalization
    "std": (0.229, 0.224, 0.225),
    
    # Postprocessing
    "threshold": 0.65,
    "min_contour_area": 100,
    
    # Device settings
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "use_fp16": True,
    "use_tensorrt": False,
    
    # Performance
    "warmup_iterations": 10,
    "max_batch_size": 16,
    
    # Adversarial guard
    "guard_confidence_threshold": 0.7,
    "guard_variance_threshold": 0.03
}

# ------------------------------------------------------------------------------
# TensorRT Configuration
# ------------------------------------------------------------------------------
TENSORRT_SETTINGS: Dict[str, Any] = {
    "precision": "fp16",  # fp32/fp16/int8
    "workspace_size": 2048,  # MB
    "min_batch_size": 1,
    "opt_batch_size": 8,
    "max_batch_size": 16,
    "calibration_batches": 100,
    "calibration_cache": "models/calibration.cache",
    "dynamic_shapes": {
        "min": [1, 3, 256, 256],
        "opt": [8, 3, 512, 512],
        "max": [16, 3, 1024, 1024]
    }
}

# ------------------------------------------------------------------------------
# Engine Configuration (Dataclass)
# ------------------------------------------------------------------------------
@dataclass
class EngineConfig:
    """
    Configuration dataclass for the InferenceEngine with integrated guard system parameters.
    """
    # Core engine parameters
    num_workers: int = 1
    queue_size: int = 100
    batch_size: int = 32
    min_batch_size: int = 1
    max_batch_size: int = 128
    warmup_runs: int = 10
    timeout: float = 0.1
    autoscale_interval: float = 5.0
    queue_size_threshold_high: float = 80.0
    queue_size_threshold_low: float = 20.0
    enable_dynamic_batching: bool = True
    debug_mode: bool = False
    use_multigpu: bool = False
    device_ids: List[int] = field(default_factory=lambda: list(range(torch.cuda.device_count())))
    multigpu_strategy: str = 'dataparallel'
    log_file: str = "inference_engine.log"
    executor_type: str = "thread"
    
    # PID controller parameters
    pid_kp: float = 0.5
    pid_ki: float = 0.1
    pid_kd: float = 0.05
    
    # TensorRT parameters
    enable_trt: bool = False
    trt_mode: str = "static"
    trt_workspace_size: int = 1 << 30
    trt_min_block_size: int = 1
    trt_opt_shape: Optional[List[int]] = None
    trt_input_shape: Optional[List[int]] = None
    input_shape: Optional[torch.Size] = None
    use_tensorrt: bool = False
    num_classes : int = 10
    
    # Guard system parameters
    guard_enabled: bool = True
    guard_num_augmentations: int = 5
    guard_noise_level_range: Tuple[float, float] = (0.005, 0.02)
    guard_dropout_rate: float = 0.1
    guard_flip_prob: float = 0.5
    guard_confidence_threshold: float = 0.5
    guard_variance_threshold: float = 0.05
    guard_input_range: Tuple[float, float] = (0.0, 1.0)
    guard_augmentation_types: List[str] = field(
        default_factory=lambda: ["noise", "dropout", "flip"]
    )
    model_sources: Dict[str, Any] = field(default_factory=lambda: {
        "huggingface": {
            "enabled": True,
            "cache_dir": "models/hf_cache",
            "token": None  # Add your HF token here
        },
        "torchhub": {
            "enabled": True,
            "trust_repo": False
        }
    })
    max_concurrent_downloads: int = 3
    model_validation: Dict[str, Any] = field(default_factory=lambda: {
        "checksum_verification": True,
        "file_types": [".pt", ".pth", ".bin"],
        "max_size_mb": 5000
    })
    # Internal components (not configurable)
    pid_controller: object = field(init=False)
    device = "cuda:0"
    
    def __post_init__(self):
        """Initialize derived components after dataclass construction"""
        from core.pid import PIDController  # Import locally to avoid circular dependencies
        
        # Initialize PID controller
        self.pid_controller = PIDController(
            self.pid_kp, self.pid_ki, self.pid_kd, setpoint=50.0
        )
        
        # Validate device IDs
        if self.use_multigpu and not self.device_ids:
            self.device_ids = list(range(torch.cuda.device_count()))
            
        # Validate augmentation types
        valid_augmentations = {"noise", "dropout", "flip"}
        if invalid := set(self.guard_augmentation_types) - valid_augmentations:
            raise ValueError(f"Invalid augmentation types: {invalid}")

    def configure_logging(self):
        """Set up logging with guard system awareness"""
        level = logging.DEBUG if self.debug_mode else logging.INFO
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        
        logger = logging.getLogger()
        logger.handlers.clear()
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.setLevel(level)
        
        if self.debug_mode:
            logger.debug("Debug logging enabled.")
            if self.guard_enabled:
                logger.debug(f"Guard system configuration:\n{self._format_guard_config()}")

    def _format_guard_config(self) -> str:
        """Format guard configuration for debug logging"""
        return (
            f"Augmentations: {self.guard_num_augmentations} runs\n"
            f"Active augmentations: {', '.join(self.guard_augmentation_types)}\n"
            f"Noise range: {self.guard_noise_level_range}\n"
            f"Dropout rate: {self.guard_dropout_rate}\n"
            f"Flip probability: {self.guard_flip_prob}\n"
            f"Confidence threshold: {self.guard_confidence_threshold}\n"
            f"Variance threshold: {self.guard_variance_threshold}\n"
            f"Input range: {self.guard_input_range}"
        )

# ------------------------------------------------------------------------------
# Runtime Configuration
# ------------------------------------------------------------------------------
def get_runtime_config(use_tensorrt: bool = False) -> EngineConfig:
    """Factory function for creating runtime configurations"""
    base_config = EngineConfig()
    
    if use_tensorrt:
        return EngineConfig(
            use_tensorrt=True,
            use_fp16=True,
            batch_size=TENSORRT_SETTINGS["opt_batch_size"],
            max_batch_size=TENSORRT_SETTINGS["max_batch_size"],
            trt_precision=TENSORRT_SETTINGS["precision"]
        )
    
    return base_config
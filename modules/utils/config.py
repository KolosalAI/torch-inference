# utils/config.py
import logging
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from core.pid import PIDController

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
    "mean": [0.485, 0.456, 0.406],  # 3 values for RGB
    "std": [0.229, 0.224, 0.225],   # 3 values for RGB
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

################################################################################
# Minimal EngineConfig to match your tests
################################################################################
@dataclass
class EngineConfig:
    """
    Basic configuration class for the InferenceEngine. Contains the attributes
    your test code is referencing, with default values for demonstration.
    """
    def __init__(
        self,
        num_workers: int = 2,
        queue_size: int = 16,
        batch_size: int = 4,
        min_batch_size: int = 1,
        max_batch_size: int = 8,
        warmup_runs: int = 2,
        timeout: float = 2.0,
        batch_wait_timeout: float = 0.01,
        autoscale_interval: float = 0.5,
        queue_size_threshold_high: float = 80.0,
        queue_size_threshold_low: float = 20.0,
        enable_dynamic_batching: bool = False,
        debug_mode: bool = True,
        use_multigpu: bool = False,
        log_file: str = "engine.log",
        executor_type: str = "thread",  # or "process"
        enable_trt: bool = False,
        use_tensorrt: bool = False,
        num_classes: int = 10,
        guard_enabled: bool = True,
        guard_num_augmentations: int = 2,
        guard_noise_level_range: tuple = (0.001, 0.005),
        guard_dropout_rate: float = 0.0,
        guard_flip_prob: float = 0.0,
        guard_confidence_threshold: float = 0.6,
        guard_variance_threshold: float = 0.1,
        guard_input_range: tuple = (0.0, 1.0),
        guard_augmentation_types: List[str] = None,
        pid_kp: float = 0.1,
        pid_ki: float = 0.0,
        pid_kd: float = 0.0,
        trt_input_shape: Optional[List[tuple]] = None,
        async_mode: bool = True
    ):
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.warmup_runs = warmup_runs
        self.timeout = timeout
        self.batch_wait_timeout = batch_wait_timeout
        self.autoscale_interval = autoscale_interval
        self.queue_size_threshold_high = queue_size_threshold_high
        self.queue_size_threshold_low = queue_size_threshold_low
        self.enable_dynamic_batching = enable_dynamic_batching
        self.debug_mode = debug_mode
        self.use_multigpu = use_multigpu
        self.log_file = log_file
        self.executor_type = executor_type
        self.enable_trt = enable_trt
        self.use_tensorrt = use_tensorrt
        self.num_classes = num_classes
        self.guard_enabled = guard_enabled
        self.guard_num_augmentations = guard_num_augmentations
        self.guard_noise_level_range = guard_noise_level_range
        self.guard_dropout_rate = guard_dropout_rate
        self.guard_flip_prob = guard_flip_prob
        self.guard_confidence_threshold = guard_confidence_threshold
        self.guard_variance_threshold = guard_variance_threshold
        self.guard_input_range = guard_input_range
        self.guard_augmentation_types = guard_augmentation_types or ["noise", "dropout", "flip"]
        self.pid_kp = pid_kp
        self.pid_ki = pid_ki
        self.pid_kd = pid_kd
        self.trt_input_shape = trt_input_shape
        self.async_mode = async_mode

        # The user’s test code overrides __post_init__, but here we implement a simple default.
        self.__post_init__()

    def __post_init__(self):
        # Create a simple PID controller by default
        from core.pid import PIDController
        self.pid_controller = PIDController(self.pid_kp, self.pid_ki, self.pid_kd, setpoint=50.0)
        # Validate augmentation types
        valid_augmentations = {"noise", "dropout", "flip"}
        invalid = set(self.guard_augmentation_types) - valid_augmentations
        if invalid:
            raise ValueError(f"Invalid augmentation types: {invalid}")

    def configure_logging(self):
        level = logging.DEBUG if self.debug_mode else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            filename=self.log_file if self.log_file else None
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
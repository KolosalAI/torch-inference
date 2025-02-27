# utils/config.py
import logging
import torch
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import logging
import os
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
    Configuration class for the InferenceEngine.
    """
    debug_mode: bool = False
    async_mode: bool = True
    batch_size: int = 16
    min_batch_size: int = 1
    max_batch_size: int = 64
    queue_size: int = 1000
    timeout: float = 1.0
    batch_wait_timeout: float = 0.1
    warmup_runs: int = 5
    auto_tune_batch_size: bool = True
    target_memory_fraction: float = 0.7
    executor_type: str = "thread"  # "thread" or "process"
    num_workers: int = min(32, (os.cpu_count() or 4))
    guard_enabled: bool = False
    guard_num_augmentations: int = 5
    guard_confidence_threshold: float = 0.8
    guard_variance_threshold: float = 0.1
    guard_fail_silently: bool = False
    guard_augmentation_types: List[str] = field(default_factory=lambda: ["noise", "dropout", "flip"])
    guard_noise_level_range: Tuple[float, float] = (0.01, 0.05)
    guard_dropout_rate: float = 0.1
    guard_flip_prob: float = 0.3
    guard_input_range: Tuple[float, float] = (0.0, 1.0)
    num_classes: int = 0  # For default guard response
    trt_input_shape: Optional[List[Tuple[List[int], List[int], List[int]]]] = None
    trt_workspace_size: int = 1 << 30  # 1 GB
    use_jit: bool = False
    output_to_cpu: bool = False
    check_nan_inf: bool = False
    autoscale_interval: float = 5.0
    monitor_interval: float = 60.0
    request_timeout: float = 30.0
    pid_controller: Optional[Any] = None
    input_shape: Optional[List[int]] = None

    def configure_logging(self):
        """Configure logging based on debug mode."""
        logging.basicConfig(
            level=logging.DEBUG if self.debug_mode else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
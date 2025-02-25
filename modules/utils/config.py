# utils/config.py
import logging
import torch
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import logging
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
    
    Attributes:
        num_workers: Number of workers for thread/process executors.
        queue_size: Maximum size of the request queue.
        batch_size: Initial batch size for inference.
        min_batch_size: Minimum allowable batch size.
        max_batch_size: Maximum allowable batch size.
        warmup_runs: Number of warmup iterations.
        timeout: Timeout (in seconds) for waiting on new requests.
        batch_wait_timeout: Time (in seconds) to wait for additional requests in a batch.
        autoscale_interval: Interval (in seconds) to run the PID-based autoscaling.
        queue_size_threshold_high: Upper threshold for queue utilization (percentage).
        queue_size_threshold_low: Lower threshold for queue utilization (percentage).
        enable_dynamic_batching: Flag to enable dynamic batching.
        debug_mode: If True, enables debug logging.
        use_multigpu: If True, allows multiple GPUs to be used.
        log_file: Filename for logging output (None means console only).
        executor_type: Type of executor to use ("thread" or "process").
        use_tensorrt: Enable TensorRT optimizations.
        num_classes: Number of output classes (used for default responses in guard mode).
        guard_enabled: If True, activates guard logic for adversarial checks.
        guard_num_augmentations: Number of augmented samples for guard evaluation.
        guard_noise_level_range: Tuple specifying the range of noise levels to add.
        guard_dropout_rate: Dropout rate applied during guard augmentations.
        guard_flip_prob: Probability of flipping the image during guard augmentations.
        guard_confidence_threshold: Minimum confidence required for guard approval.
        guard_variance_threshold: Maximum variance allowed across augmentations.
        guard_input_range: Valid range for input tensor values.
        guard_augmentation_types: List of augmentation types to apply ("noise", "dropout", "flip").
        pid_kp: Proportional gain for PID controller.
        pid_ki: Integral gain for PID controller.
        pid_kd: Derivative gain for PID controller.
        trt_input_shape: Optional list of tuples specifying (min, opt, max) shapes for TRT.
        async_mode: If True, enables asynchronous processing.
        device: Device(s) to use for inference (e.g. "cuda:0" or ["cuda:0", "cuda:1"]).
    """
    num_workers: int = 2
    queue_size: int = 16
    batch_size: int = 4
    min_batch_size: int = 1
    max_batch_size: int = 8
    warmup_runs: int = 2
    timeout: float = 2.0
    batch_wait_timeout: float = 0.01
    autoscale_interval: float = 0.5
    queue_size_threshold_high: float = 80.0
    queue_size_threshold_low: float = 20.0
    enable_dynamic_batching: bool = False
    debug_mode: bool = True
    use_multigpu: bool = False
    log_file: Optional[str] = "engine.log"
    executor_type: str = "thread"  # Options: "thread", "process"
    use_tensorrt: bool = False
    num_classes: int = 10
    guard_enabled: bool = True
    guard_num_augmentations: int = 2
    guard_noise_level_range: Tuple[float, float] = (0.001, 0.005)
    guard_dropout_rate: float = 0.0
    guard_flip_prob: float = 0.0
    guard_confidence_threshold: float = 0.6
    guard_variance_threshold: float = 0.1
    guard_input_range: Tuple[float, float] = (0.0, 1.0)
    guard_augmentation_types: List[str] = field(default_factory=lambda: ["noise", "dropout", "flip"])
    pid_kp: float = 0.1
    pid_ki: float = 0.0
    pid_kd: float = 0.0
    trt_input_shape: Optional[List[Tuple[int, ...]]] = None
    async_mode: bool = True
    device: Union[str, List[str]] = "cuda:0"

    def __post_init__(self):
        # Create a simple PID controller by default.
        from core.pid import PIDController
        self.pid_controller = PIDController(self.pid_kp, self.pid_ki, self.pid_kd, setpoint=50.0)
        
        # Validate guard augmentation types.
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
import asyncio
import contextlib
import functools
import logging
import time
from contextlib import nullcontext, AsyncExitStack
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Tuple, cast
import sys
import os
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity


# If your directory structure needs it:
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# from utils.config import EngineConfig


################################################################################
# Enums and Constants
################################################################################
class ExecutorType(Enum):
    THREAD = auto()
    PROCESS = auto()


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Added support for Apple M-series GPUs


################################################################################
# Type Variables for Generic Functions
################################################################################
T = TypeVar('T')
Input = TypeVar('Input')
Output = TypeVar('Output')


################################################################################
# Daemon Thread Pool
################################################################################
class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that marks worker threads as daemon."""
    def _adjust_thread_count(self):
        super()._adjust_thread_count()
        for thread in self._threads:
            try:
                thread.daemon = True
            except RuntimeError:
                pass


################################################################################
# Custom Exceptions
################################################################################
class ModelError(Exception):
    """Base class for model-related errors."""
    pass


class ModelPreparationError(ModelError):
    """Exception for model preparation errors."""
    pass


class ModelInferenceError(ModelError):
    """Exception for inference execution errors."""
    pass


class GuardError(Exception):
    """Exception for guard-related errors."""
    pass


class ShutdownError(Exception):
    """Exception raised when engine is shutting down."""
    pass


################################################################################
# Request Item
################################################################################
@dataclass
class RequestItem:
    """
    Holds a single inference request with priority and future.
    """
    input: Any
    future: asyncio.Future
    priority: int = 0  # Lower values = higher priority
    timestamp: float = 0.0  # For tracking request age
    
    def __post_init__(self):
        self.timestamp = time.monotonic()
    
    def __lt__(self, other: "RequestItem"):
        # Priority queue uses __lt__ for ordering
        return self.priority < other.priority
    
    @property
    def age(self) -> float:
        """Return age of request in seconds."""
        return time.monotonic() - self.timestamp


################################################################################
# Memory Management Utilities
################################################################################
class MemoryManager:
    """Utilities for memory management and monitoring."""
    
    @staticmethod
    def get_gpu_memory_usage() -> Dict[int, Tuple[int, int]]:
        """
        Get GPU memory usage for each available device:
        Returns {device_idx: (allocated_bytes, cached_bytes)}.
        """
        if not torch.cuda.is_available():
            return {}
        
        result = {}
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                allocated = torch.cuda.memory_allocated()
                cached = torch.cuda.memory_reserved()
                result[i] = (allocated, cached)
        return result
    
    @staticmethod
    def estimate_batch_size(
        model: nn.Module, 
        sample_input: torch.Tensor,
        target_device: torch.device,
        target_memory_fraction: float = 0.7,
        max_batch_size: int = 512
    ) -> int:
        """
        Estimate an optimal batch size based on memory constraints for a
        given model and device. Returns the highest usable batch size up to
        'max_batch_size' that does not cause OOM.
        """
        if target_device.type != "cuda" or not torch.cuda.is_available():
            # No estimation needed for CPU or MPS
            return max_batch_size
            
        device_idx = target_device.index or 0
        device_props = torch.cuda.get_device_properties(device_idx)
        available_mem = device_props.total_memory * target_memory_fraction
        
        # Clear GPU stats before we start
        torch.cuda.empty_cache()
        
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            if batch_size > max_batch_size:
                return max_batch_size
            try:
                test_batch = sample_input.expand(batch_size, *sample_input.shape)
                test_batch = test_batch.to(target_device, non_blocking=True)
                
                with torch.inference_mode():
                    _ = model(test_batch)
                
                used_mem = torch.cuda.max_memory_allocated(device_idx)
                if used_mem >= available_mem:
                    return max(1, batch_size // 2)
                
                # Clean up
                del test_batch
                torch.cuda.reset_peak_memory_stats(device_idx)
                torch.cuda.empty_cache()
                
            except RuntimeError:
                return max(1, batch_size // 2)
                
        return max_batch_size


################################################################################
# Fallback EngineConfig (if not imported from elsewhere)
################################################################################
class EngineConfig:
    """
    Stand-in EngineConfig for demonstration. Replace with your own
    if you have a custom implementation.
    """
    def __init__(
        self,
        debug_mode: bool = False,
        batch_size: int = 16,
        queue_size: int = 1000,
        async_mode: bool = True,
        warmup_runs: int = 5,
        use_jit: bool = False,
        auto_tune_batch_size: bool = True,
        trt_workspace_size: int = 1 << 30,
        monitor_interval: float = 60.0,
        timeout: float = 2.0,
        batch_wait_timeout: float = 0.01,
        max_batch_size: int = 128,
        min_batch_size: int = 1,
        drop_requests_when_full: bool = False,
        queue_wait_timeout: float = 10.0,
        request_timeout: float = 30.0,
        check_nan_inf: bool = False,
        guard_enabled: bool = False,
        guard_num_augmentations: int = 3,
        guard_augmentation_types: Optional[List[str]] = None,
        guard_noise_level_range: Tuple[float, float] = (0.01, 0.05),
        guard_dropout_rate: float = 0.1,
        guard_flip_prob: float = 0.3,
        guard_scale_range: Tuple[float, float] = (0.95, 1.05),
        guard_input_range: Tuple[float, float] = (0.0, 1.0),
        guard_confidence_threshold: float = 0.7,
        guard_variance_threshold: float = 0.1,
        guard_fail_silently: bool = False,
        num_classes: int = 0,
        output_to_cpu: bool = False,
        pid_controller: Optional[Any] = None,
        autoscale_interval: float = 10.0,
        input_shape: Optional[Tuple[int, ...]] = None,
        target_memory_fraction: float = 0.7,
    ):
        self.debug_mode = debug_mode
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.async_mode = async_mode
        self.warmup_runs = warmup_runs
        self.use_jit = use_jit
        self.auto_tune_batch_size = auto_tune_batch_size
        self.trt_workspace_size = trt_workspace_size
        self.monitor_interval = monitor_interval
        self.timeout = timeout
        self.batch_wait_timeout = batch_wait_timeout
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.drop_requests_when_full = drop_requests_when_full
        self.queue_wait_timeout = queue_wait_timeout
        self.request_timeout = request_timeout
        self.check_nan_inf = check_nan_inf
        
        self.guard_enabled = guard_enabled
        self.guard_num_augmentations = guard_num_augmentations
        self.guard_augmentation_types = guard_augmentation_types or ["noise", "dropout"]
        self.guard_noise_level_range = guard_noise_level_range
        self.guard_dropout_rate = guard_dropout_rate
        self.guard_flip_prob = guard_flip_prob
        self.guard_scale_range = guard_scale_range
        self.guard_input_range = guard_input_range
        self.guard_confidence_threshold = guard_confidence_threshold
        self.guard_variance_threshold = guard_variance_threshold
        self.guard_fail_silently = guard_fail_silently
        self.num_classes = num_classes
        
        self.output_to_cpu = output_to_cpu
        self.pid_controller = pid_controller
        self.autoscale_interval = autoscale_interval
        
        self.input_shape = input_shape
        self.target_memory_fraction = target_memory_fraction
    
    def configure_logging(self):
        if self.debug_mode:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)


################################################################################
# Inference Engine
################################################################################
class InferenceEngine:
    def __init__(
        self,
        model: nn.Module,
        device: Optional[Union[str, torch.device, List[Union[str, torch.device]]]] = None,
        preprocessor: Optional[Callable[[Any], Any]] = None,
        postprocessor: Optional[Callable[[Any], Any]] = None,
        use_fp16: bool = False,
        use_tensorrt: bool = False,
        config: Optional[EngineConfig] = None,
    ):
        """
        An optimized asynchronous inference engine with parallel execution capabilities.

        Args:
            model:          A PyTorch model (nn.Module).
            device:         Device specification (e.g. "cpu", "cuda", torch.device)
                            or a list of devices for DataParallel.
                            If None, uses "cuda" if available, else "cpu" or "mps" if on Apple.
            preprocessor:   Optional callable to preprocess inputs.
            postprocessor:  Optional callable to postprocess outputs.
            use_fp16:       Enable FP16 inference if possible.
            use_tensorrt:   Use TensorRT optimization if possible.
            config:         Engine configuration object.
        """
        # ----------------------------------------------------------------------
        # Basic Setup
        # ----------------------------------------------------------------------
        self._shutdown_event = asyncio.Event()
        self._startup_complete = asyncio.Event()
        self._exit_stack = AsyncExitStack()

        # Configure logging
        self.config = config or EngineConfig(debug_mode=False)
        self.config.configure_logging()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.use_fp16 = use_fp16
        self.use_tensorrt = use_tensorrt
        self._ready = False

        # Handle device selection
        self.devices = self._normalize_devices(device)
        self.primary_device = self.devices[0]

        # Setup executors
        self._init_executors()

        # Pre/Post Processors
        self.preprocessor = preprocessor if preprocessor else (lambda x: x)
        self.postprocessor = postprocessor if postprocessor else (lambda x: x)

        # Metrics tracking
        self._batch_processing_times: List[float] = []
        self._last_metrics_reset = time.monotonic()
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._guard_triggered_count = 0
        
        # Priority queue for requests
        self.request_queue = asyncio.PriorityQueue(maxsize=self.config.queue_size)

        # Initialize and prepare model
        try:
            self.model = self._prepare_model(model)
            self.input_shape = self._detect_input_shape()

            # Auto-tune batch size if configured
            if self.config.auto_tune_batch_size and self.input_shape:
                self._tune_batch_size()
            
            # Warmup
            self._warmup()
            
            # Log device info
            self._log_device_info()
            self._ready = True

        except Exception as e:
            self.logger.error(f"Model preparation failed: {e}", exc_info=True)
            raise ModelPreparationError(f"Failed to prepare model: {str(e)}") from e

        # Start background tasks if async mode is enabled
        if self.config.async_mode:
            self.batch_processor_task = asyncio.create_task(self._process_batches())
            self.autoscale_task = asyncio.create_task(self._autoscale())
            self.monitor_task = asyncio.create_task(self._monitor_health())
        else:
            self.batch_processor_task = None
            self.autoscale_task = None
            self.monitor_task = None

        self._startup_complete.set()
        self.logger.info("InferenceEngine initialized successfully")

    ############################################################################
    # Internal Setup / Configuration
    ############################################################################
    def _normalize_devices(
        self,
        device: Optional[Union[str, torch.device, List[Union[str, torch.device]]]]
    ) -> List[torch.device]:
        """
        Converts device specification to a list of torch.device objects.
        Handles auto-selection and validation for CUDA or MPS.
        """
        if device is None:
            # Auto device selection with MPS (Apple Silicon) support
            if torch.cuda.is_available():
                return [torch.device("cuda")]
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                return [torch.device("mps")]
            else:
                return [torch.device("cpu")]
        
        if isinstance(device, (str, torch.device)):
            device_obj = torch.device(device)
            if device_obj.type == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available; falling back to CPU")
                return [torch.device("cpu")]
            if device_obj.type == "mps":
                if not (hasattr(torch, 'mps') and torch.backends.mps.is_available()):
                    self.logger.warning("MPS requested but not available; falling back to CPU")
                    return [torch.device("cpu")]
            return [device_obj]
        
        elif isinstance(device, list):
            if not device:  # Empty list fallback
                return self._normalize_devices(None)
            
            valid_devices = []
            for d in device:
                d_obj = torch.device(d)
                if d_obj.type == "cuda" and not torch.cuda.is_available():
                    self.logger.warning(f"Skipping unavailable device {d_obj}, falling back to CPU")
                    continue
                if d_obj.type == "mps":
                    if not (hasattr(torch, 'mps') and torch.backends.mps.is_available()):
                        self.logger.warning(f"Skipping unavailable device {d_obj}, fallback to CPU")
                        continue
                valid_devices.append(d_obj)
            if not valid_devices:
                self.logger.warning("No valid devices in list; falling back to CPU")
                return [torch.device("cpu")]
            
            return valid_devices
        else:
            raise ValueError(f"Invalid device type: {type(device)}")

    def _init_executors(self):
        """Create thread or process executors based on config."""
        executor_type = getattr(self.config, "executor_type", "thread")
        num_workers = getattr(self.config, "num_workers", min(32, os.cpu_count() or 4))
        
        if executor_type.lower() == "process":
            # Process pool executors
            self.executor = ProcessPoolExecutor(max_workers=num_workers)
            self.guard_executor = ProcessPoolExecutor(max_workers=max(1, num_workers // 2))
            self.inference_executor = ProcessPoolExecutor(max_workers=max(1, num_workers // 2))
            self.logger.info(f"Using ProcessPoolExecutor with {num_workers} workers")
        else:
            # Thread pool executors (default)
            self.executor = DaemonThreadPoolExecutor(max_workers=num_workers)
            self.guard_executor = DaemonThreadPoolExecutor(max_workers=max(1, num_workers // 2))
            self.inference_executor = DaemonThreadPoolExecutor(max_workers=max(1, num_workers // 2))
            self.logger.info(f"Using ThreadPoolExecutor with {num_workers} workers")

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Prepares model for inference: moves to device(s), sets eval mode,
        applies TorchScript or TensorRT if configured.
        """
        model.to(self.primary_device)
        
        # DataParallel if multiple GPU devices
        if len(self.devices) > 1 and all(d.type == "cuda" for d in self.devices):
            model = nn.DataParallel(model, device_ids=[d.index for d in self.devices if d.index is not None])
            self.logger.info(f"Using DataParallel across {len(self.devices)} CUDA devices")
        elif len(self.devices) > 1:
            self.logger.warning("Mixed or non-CUDA devices detected. Using only primary device.")
        
        model.eval()

        # Apply TorchScript if enabled
        if self.config.use_jit:
            try:
                self.logger.info("Applying TorchScript compilation...")
                dummy_input = self._create_dummy_input()
                if dummy_input is not None:
                    model = torch.jit.trace(model, dummy_input)
                    self.logger.info("Model successfully traced with TorchScript")
                else:
                    model = torch.jit.script(model)
                    self.logger.info("Model successfully scripted with TorchScript")
            except Exception as e:
                self.logger.warning(f"TorchScript compilation failed: {e}")

        # Apply TensorRT if requested
        if self.use_tensorrt and self.primary_device.type == "cuda":
            model = self._apply_tensorrt(model)

        return model

    def _create_dummy_input(self) -> Optional[torch.Tensor]:
        """
        Create a dummy input tensor for model tracing. If self.config.input_shape
        is known, that is used. Otherwise attempts a fallback.
        """
        try:
            # Start with fallback shape
            dummy_shape = (1, 10)
            if self.config.input_shape is not None:
                dummy_shape = (1,) + tuple(self.config.input_shape)
            return torch.randn(dummy_shape, device=self.primary_device)
        except Exception as e:
            self.logger.warning(f"Failed to create dummy input: {e}")
            return None

    def _apply_tensorrt(self, model: nn.Module) -> nn.Module:
        """Apply TensorRT optimizations using torch_tensorrt if available."""
        try:
            import torch_tensorrt
        except ImportError:
            self.logger.warning("torch_tensorrt is not installed; skipping TensorRT optimization")
            return model

        if len(self.devices) > 1:
            self.logger.warning("TensorRT is not applied to DataParallel models.")
            return model

        try:
            self.logger.info("Applying TensorRT optimization...")
            start_time = time.time()

            # Determine input specs
            input_specs = []
            sample_input = self._create_dummy_input()
            if sample_input is None:
                self.logger.error("TensorRT requires known input shape; skipping TRT compile.")
                return model

            input_specs.append(
                torch_tensorrt.Input(
                    min_shape=sample_input.shape,
                    opt_shape=sample_input.shape,
                    max_shape=sample_input.shape,
                    dtype=torch.half if self.use_fp16 else torch.float32
                )
            )

            enabled_precisions = {torch.half} if self.use_fp16 else {torch.float32}
            compiled_model = torch_tensorrt.compile(
                model,
                inputs=input_specs,
                enabled_precisions=enabled_precisions,
                workspace_size=self.config.trt_workspace_size,
                debug=self.config.debug_mode
            )
            compile_time = time.time() - start_time
            self.logger.info(f"TensorRT optimization completed in {compile_time:.2f}s")
            return compiled_model

        except Exception as e:
            self.logger.error(f"TensorRT optimization failed: {e}", exc_info=True)
            return model

    def _detect_input_shape(self) -> Optional[torch.Size]:
        """
        Detect input shape by either:
          1) Checking self.config.input_shape.
          2) Checking model.input_shape if it exists.
          3) Probing with a few typical shapes.
        """
        # 1. If config shape is provided
        if self.config.input_shape is not None:
            return torch.Size(self.config.input_shape)

        # 2. If model has an input_shape attribute
        if hasattr(self.model, "input_shape"):
            return torch.Size(self.model.input_shape)

        # 3. Probing with typical shapes
        test_shapes = [
            (1, 3, 224, 224),
            (1, 1, 28, 28),
            (1, 10),
        ]
        for shape in test_shapes:
            try:
                dummy_input = torch.randn(shape, device=self.primary_device)
                with torch.inference_mode():
                    _ = self.model(dummy_input)
                self.logger.debug(f"Inferred shape: {shape[1:]}")
                return torch.Size(shape[1:])
            except Exception:
                continue

        self.logger.warning("Unable to detect input shape.")
        return None

    def _tune_batch_size(self):
        """Auto-tune batch size using MemoryManager if input shape is known."""
        if not self.input_shape:
            self.logger.warning("Cannot auto-tune batch size: input shape unknown.")
            return

        sample_input = torch.zeros((1,) + self.input_shape, device=self.primary_device)
        max_batch = getattr(self.config, "max_batch_size", 128)

        tuned_bs = MemoryManager.estimate_batch_size(
            self.model,
            sample_input,
            self.primary_device,
            self.config.target_memory_fraction,
            max_batch_size=max_batch
        )
        self.config.batch_size = tuned_bs
        self.logger.info(f"Auto-tuned batch size to: {tuned_bs}")

    def _warmup(self):
        """
        Warm up the model on a dummy batch if input shape is known and warmup_runs>0.
        This helps the GPU load kernels and reduce first-inference latency.
        """
        if (not self._ready) or (not self.input_shape) or (self.config.warmup_runs <= 0):
            return
        
        self.logger.info(f"Warming up model with {self.config.warmup_runs} iterations...")
        dummy_input = torch.zeros((self.config.batch_size,) + self.input_shape, device=self.primary_device)

        with torch.inference_mode():
            for i in range(self.config.warmup_runs):
                try:
                    _ = self.model(dummy_input)
                    if i == 0 and self.primary_device.type == 'cuda':
                        # Clear GPU cache after first run overhead
                        torch.cuda.empty_cache()
                except Exception as e:
                    self.logger.error(f"Warmup iteration failed: {e}")
                    break

        if self.primary_device.type == 'cuda':
            torch.cuda.synchronize()
        self.logger.info("Warmup completed successfully.")

    def _log_device_info(self):
        """Log hardware info for the selected device(s)."""
        if len(self.devices) == 1:
            device = self.primary_device
            self.logger.info(f"Using device: {device}")
            if device.type == 'cuda':
                idx = device.index if device.index is not None else 0
                name = torch.cuda.get_device_name(idx)
                props = torch.cuda.get_device_properties(idx)
                total_mem_gb = props.total_memory / 1e9
                self.logger.info(
                    f"[Device {idx}] {name}, "
                    f"CUDA {props.major}.{props.minor}, "
                    f"Memory: {total_mem_gb:.2f} GB, "
                    f"SMs: {props.multi_processor_count}"
                )
            elif device.type == 'mps':
                self.logger.info("Using Apple Metal Performance Shaders (MPS).")
        else:
            self.logger.info(f"Using {len(self.devices)} devices in parallel: {self.devices}")
            for d in self.devices:
                if d.type == 'cuda':
                    idx = d.index if d.index is not None else 0
                    name = torch.cuda.get_device_name(idx)
                    props = torch.cuda.get_device_properties(idx)
                    total_mem_gb = props.total_memory / 1e9
                    self.logger.info(f"[Device {idx}] {name}, Mem: {total_mem_gb:.2f} GB")

    ############################################################################
    # Batch Processing Loop (async)
    ############################################################################
    async def _process_batches(self):
        """Main loop to process inference requests in batches."""
        self.logger.info("Starting batch processing loop.")
        while not self._shutdown_event.is_set():
            try:
                # Wait for the first item (or time out)
                try:
                    first_item = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=self.config.timeout
                    )
                except asyncio.TimeoutError:
                    if self.config.debug_mode:
                        self.logger.debug("No requests found before timeout in _process_batches.")
                    continue

                if first_item is None:
                    continue

                batch_start_time = time.monotonic()
                batch_items = [first_item]
                max_batch_size = self.config.batch_size

                # Adaptive small wait to allow queue to fill a bit, but only if queue is not near-full
                queue_size = self.request_queue.qsize()
                fill_factor = queue_size / self.config.queue_size
                wait_time = self.config.batch_wait_timeout * (1 - min(0.9, fill_factor))
                if queue_size < 0.9 * self.config.queue_size and wait_time > 0:
                    await asyncio.sleep(wait_time)

                # Collect additional items
                while len(batch_items) < max_batch_size:
                    try:
                        next_item = self.request_queue.get_nowait()
                        batch_items.append(next_item)
                    except asyncio.QueueEmpty:
                        break

                # Process batch
                await self._process_batch(batch_items, batch_start_time)

            except asyncio.CancelledError:
                self.logger.info("Batch processing loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in batch processing loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def _process_batch(self, batch_items: List[RequestItem], batch_start_time: float):
        """Process a single batch of requests."""
        if not batch_items:
            return
        
        batch_size = len(batch_items)
        inputs = [item.input for item in batch_items]
        futures = [item.future for item in batch_items]

        # Validate inputs
        try:
            self._validate_batch_inputs(inputs)
        except ValueError as e:
            self.logger.error(f"Batch input validation failed: {e}")
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)
            self._failed_requests += len(futures)
            return

        # Run inference
        try:
            stacked_tensor = torch.stack(inputs)
            if stacked_tensor.device != self.primary_device:
                stacked_tensor = stacked_tensor.to(self.primary_device, non_blocking=True)

            loop = asyncio.get_running_loop()
            outputs = await loop.run_in_executor(
                self.inference_executor,
                self._infer_batch,
                stacked_tensor
            )

            # Distribute results
            self._distribute_results(outputs, futures)
            self._successful_requests += len(futures)

            # Batch timing
            batch_time = time.monotonic() - batch_start_time
            self._batch_processing_times.append(batch_time)

            if self.config.debug_mode:
                self.logger.debug(
                    f"Processed {batch_size} items in {batch_time:.4f}s "
                    f"({batch_size/batch_time:.2f} items/s)."
                )

        except Exception as e:
            self.logger.error(f"Batch inference failed: {e}", exc_info=True)
            for fut in futures:
                if not fut.done():
                    fut.set_exception(ModelInferenceError(f"Inference failed: {str(e)}"))
            self._failed_requests += len(futures)

    def _validate_batch_inputs(self, inputs: List[torch.Tensor]) -> None:
        """
        Validate that all inputs in the batch are consistent (same shape, dtype).
        Raises ValueError if validation fails.
        """
        if not inputs:
            raise ValueError("Empty batch inputs.")
        
        base_shape = inputs[0].shape
        base_dtype = inputs[0].dtype
        for idx, tensor in enumerate(inputs):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Batch input at index {idx} is not a torch.Tensor.")
            if tensor.shape != base_shape:
                raise ValueError(
                    f"Input at index {idx} has shape {tensor.shape}, "
                    f"but expected {base_shape}."
                )
            if tensor.dtype != base_dtype:
                raise ValueError(
                    f"Input at index {idx} has dtype {tensor.dtype}, "
                    f"but expected {base_dtype}."
                )
            if self.config.check_nan_inf and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
                raise ValueError(f"Input at index {idx} contains NaN or Inf values.")

    def _infer_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference on the batch tensor. Uses AMP autocast if FP16 is enabled
        (on CUDA or MPS).
        """
        device_type = self.primary_device.type
        autocast_enabled = self.use_fp16 and (device_type in ["cuda", "mps"])
        
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device_type,
            enabled=autocast_enabled,
            dtype=torch.float16 if autocast_enabled else None
        ):
            outputs = self.model(batch_tensor)
            # Move outputs to CPU if configured
            if self.config.output_to_cpu and outputs.device.type != "cpu":
                outputs = outputs.cpu()
            return outputs

    def _distribute_results(self, outputs: torch.Tensor, futures: List[asyncio.Future]) -> None:
        """
        Postprocess and set individual future results from the batch output.
        """
        try:
            if isinstance(outputs, torch.Tensor):
                # If outputs has a batch dimension matching #futures, split it
                if outputs.dim() > 0 and outputs.size(0) == len(futures):
                    splitted = list(torch.split(outputs, 1, dim=0))
                    splitted = [s.squeeze(0) for s in splitted]
                else:
                    # Otherwise replicate the same output for each future
                    self.logger.warning("Output batch dimension mismatch. Duplicating for each future.")
                    splitted = [outputs] * len(futures)
            elif isinstance(outputs, (list, tuple)):
                if len(outputs) != len(futures):
                    raise ValueError(
                        f"Output length ({len(outputs)}) doesn't match number of futures ({len(futures)})."
                    )
                splitted = outputs
            else:
                splitted = [outputs] * len(futures)

            for fut, res in zip(futures, splitted):
                if not fut.done():
                    try:
                        processed_res = self.postprocessor(res)
                        fut.set_result(processed_res)
                    except Exception as e:
                        fut.set_exception(e)

        except Exception as e:
            self.logger.error(f"Error distributing results: {e}", exc_info=True)
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)

    ############################################################################
    # Autoscaling
    ############################################################################
    async def _autoscale(self):
        """
        PID-based autoscaling: adjusts batch size based on queue utilization.
        (Requires self.config.pid_controller to be set if you wish to use it.)
        """
        if not self.config.pid_controller:
            self.logger.info("Autoscaling disabled (no PID controller configured).")
            return
        
        self.logger.info("Starting autoscaling task.")
        last_time = time.monotonic()

        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(self.config.autoscale_interval)
                now = time.monotonic()
                dt = now - last_time
                last_time = now

                current_queue_size = self.request_queue.qsize()
                utilization = (current_queue_size / self.config.queue_size) * 100.0

                # Get average batch time
                recent_times = self._batch_processing_times[-20:]
                if not recent_times:
                    recent_times = [0.01]
                avg_batch_time = sum(recent_times) / len(recent_times)

                # PID adjustment
                adjustment = self.config.pid_controller.update(utilization, dt)

                # Optionally scale up more aggressively under heavy load
                if utilization > 75 and adjustment > 0:
                    adjustment *= 2.0
                # Scale down more gently when idle
                elif utilization < 25 and adjustment < 0 and avg_batch_time < 0.05:
                    adjustment *= 0.5

                new_bs = int(round(self.config.batch_size + adjustment))
                new_bs = max(self.config.min_batch_size, min(new_bs, self.config.max_batch_size))

                if new_bs != self.config.batch_size:
                    self.logger.info(
                        f"Autoscale: queue={current_queue_size}/{self.config.queue_size} "
                        f"({utilization:.1f}%), batch_size: {self.config.batch_size}→{new_bs}, "
                        f"avg_time={avg_batch_time*1000:.1f}ms"
                    )
                    self.config.batch_size = new_bs

        except asyncio.CancelledError:
            self.logger.info("Autoscale task cancelled.")
        except Exception as e:
            self.logger.error(f"Error in autoscale task: {e}", exc_info=True)

    ############################################################################
    # Health Monitoring
    ############################################################################
    async def _monitor_health(self):
        """
        Periodically logs health metrics such as queue size, memory usage,
        throughput, etc.
        """
        self.logger.info("Starting health monitoring task.")
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.monitor_interval)
                metrics = self.get_metrics()
                self.logger.info(
                    f"Health: queue={metrics['queue_size']}/{self.config.queue_size}, "
                    f"throughput={metrics['throughput_per_second']:.2f} req/s, "
                    f"batch_time={metrics['average_batch_time']*1000:.2f}ms, "
                    f"gpu_mem={metrics['gpu_memory_percent']:.1f}%, "
                    f"success={metrics['successful_requests']}, fails={metrics['failed_requests']}"
                )

                # Check stall condition
                if metrics['queue_size'] > 0 and metrics['throughput_per_second'] < 0.1:
                    self.logger.warning("Possible stall: queue has items but throughput is very low.")
                
                # Check for high GPU memory usage
                if self.primary_device.type == "cuda" and metrics['gpu_memory_percent'] > 95:
                    self.logger.warning("High GPU memory usage detected (>95%).")

            except asyncio.CancelledError:
                self.logger.info("Health monitor task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Monitor error: {e}", exc_info=True)

    ############################################################################
    # Guard Logic
    ############################################################################
    def _guard_sample(self, processed: torch.Tensor) -> bool:
        """
        Example advanced guard check to detect potential adversarial or invalid inputs.
        Returns True if sample passes guard checks.
        """
        if not self.config.guard_enabled:
            return True
        
        try:
            sample = processed
            if not isinstance(sample, torch.Tensor):
                return False
            
            # Ensure batch dimension
            if sample.dim() == len(self.input_shape):
                sample = sample.unsqueeze(0)
            sample = sample.to(self.primary_device)

            num_augs = self.config.guard_num_augmentations
            aug_batch = sample.repeat(num_augs, *([1] * (sample.dim() - 1)))

            # Augmentations (example: noise, dropout, flips, etc.)
            if "noise" in self.config.guard_augmentation_types:
                noise_levels = torch.empty(num_augs, device=self.primary_device).uniform_(
                    *self.config.guard_noise_level_range
                )
                noise_levels = noise_levels.view(num_augs, *([1] * (aug_batch.dim() - 1)))
                noise = torch.randn_like(aug_batch) * noise_levels
                aug_batch = aug_batch + noise

            if "dropout" in self.config.guard_augmentation_types:
                dropout_rate = self.config.guard_dropout_rate
                # Apply dropout to every nth sample for demonstration
                mask = (torch.rand_like(aug_batch) >= dropout_rate).float()
                aug_batch = aug_batch * mask

            if "flip" in self.config.guard_augmentation_types and aug_batch.dim() >= 3:
                flip_prob = self.config.guard_flip_prob
                flip_flags = torch.rand(num_augs, device=self.primary_device) < flip_prob
                flip_indices = flip_flags.nonzero(as_tuple=True)[0]
                if flip_indices.numel() > 0:
                    aug_batch[flip_indices] = torch.flip(aug_batch[flip_indices], dims=[-1])

            if "scale" in self.config.guard_augmentation_types:
                scale_range = self.config.guard_scale_range
                scale_factors = torch.empty(num_augs, device=self.primary_device).uniform_(*scale_range)
                scale_factors = scale_factors.view(num_augs, *([1] * (aug_batch.dim() - 1)))
                aug_batch = aug_batch * scale_factors

            # Clamp to valid range
            low, high = self.config.guard_input_range
            aug_batch = torch.clamp(aug_batch, min=low, max=high)

            # Inference on augmented batch
            with torch.inference_mode():
                autocast_en = self.use_fp16 and (self.primary_device.type == "cuda")
                with torch.amp.autocast(device_type=self.primary_device.type, enabled=autocast_en):
                    preds = self.model(aug_batch)

            # Convert to probabilities
            if preds.dim() > 1 and preds.size(1) > 1:
                preds_probs = torch.softmax(preds, dim=1)
            else:
                preds_probs = preds

            # Metrics: consistency, confidence, variance
            if preds_probs.dim() > 1:
                top_classes = preds_probs.argmax(dim=1)
                most_common_class = torch.mode(top_classes).values.item()
                class_consistency = (top_classes == most_common_class).float().mean().item()
                max_probs = preds_probs.max(dim=1)[0]
            else:
                # Single-dim output
                class_consistency = 1.0
                max_probs = preds_probs

            mean_confidence = max_probs.mean().item()
            confidence_variance = max_probs.var().item()

            # Basic checks
            passed = (
                class_consistency >= 0.8 and
                mean_confidence >= self.config.guard_confidence_threshold and
                confidence_variance <= self.config.guard_variance_threshold
            )

            if not passed:
                self._guard_triggered_count += 1

            return passed

        except Exception as e:
            self.logger.error(f"Error in guard check: {e}", exc_info=True)
            return False  # Fail closed

    ############################################################################
    # Public Methods
    ############################################################################
    async def run_inference_async(self, input_data: Any, priority: int = 0) -> torch.Tensor:
        """
        Asynchronously process an inference request via:
          1. Preprocessing
          2. Guard checks
          3. Batch-based inference
          4. Postprocessing

        Args:
            input_data: The raw input data (any format accepted by preprocessor).
            priority:   Request priority (lower = higher priority).

        Returns:
            Output from the postprocessor.

        Raises:
            ShutdownError, GuardError, ModelPreparationError, TimeoutError, etc.
        """
        if self._shutdown_event.is_set():
            raise ShutdownError("Engine is shutting down.")
        
        # Wait for startup
        if not self._startup_complete.is_set():
            await self._startup_complete.wait()
        if not self._ready:
            raise ModelPreparationError("Engine is not ready or failed initialization.")

        self._total_requests += 1
        loop = asyncio.get_running_loop()

        try:
            # Preprocess
            processed = await loop.run_in_executor(self.executor, self.preprocessor, input_data)
            if not isinstance(processed, torch.Tensor):
                processed = torch.as_tensor(processed, dtype=torch.float32)

            # Validate shape if known
            if self.input_shape is not None:
                if (processed.shape != self.input_shape) and (processed.shape[1:] != self.input_shape):
                    raise ValueError(
                        f"Preprocessed input shape {processed.shape} doesn't match "
                        f"expected {self.input_shape}."
                    )

            # Guard check
            if self.config.guard_enabled:
                is_safe = await loop.run_in_executor(
                    self.guard_executor,
                    self._guard_sample,
                    processed
                )
                if not is_safe:
                    self.logger.warning("Guard check failed - request rejected.")
                    if self.config.guard_fail_silently and self.config.num_classes > 0:
                        # Return uniform distribution
                        return torch.full((self.config.num_classes,), 1.0 / self.config.num_classes)
                    else:
                        raise GuardError("Input failed security checks.")

            # Enqueue request
            future = loop.create_future()
            request_item = RequestItem(processed, future, priority=priority)

            if self.request_queue.full():
                if self.config.drop_requests_when_full:
                    self.logger.warning("Request queue is full; dropping request.")
                    raise RuntimeError("Request queue is full.")
                else:
                    timeout = self.config.queue_wait_timeout
                    try:
                        await asyncio.wait_for(self.request_queue.put(request_item), timeout=timeout)
                    except asyncio.TimeoutError:
                        raise RuntimeError(f"Timed out waiting for queue space after {timeout}s.")
            else:
                await self.request_queue.put(request_item)

            # If async mode is off, we manually process the queue right now
            if not self.config.async_mode:
                asyncio.create_task(self._process_batches())

            # Wait for result
            timeout = self.config.request_timeout
            return await asyncio.wait_for(future, timeout)

        except Exception as e:
            self._failed_requests += 1
            if not isinstance(e, (GuardError, ValueError, ShutdownError, TimeoutError)):
                self.logger.error(f"run_inference_async error: {e}", exc_info=True)
            raise

    def run_batch_inference(self, batch: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Synchronously run inference on a batch of preprocessed inputs.
        """
        if self._shutdown_event.is_set():
            raise ShutdownError("Engine is shutting down.")
        
        if isinstance(batch, list):
            batch = torch.stack([
                x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
                for x in batch
            ])
        elif not isinstance(batch, torch.Tensor):
            raise TypeError("run_batch_inference expects a torch.Tensor or a list of Tensors.")
        
        if batch.device != self.primary_device:
            batch = batch.to(self.primary_device, non_blocking=True)

        try:
            return self._infer_batch(batch)
        except Exception as e:
            self.logger.error(f"Batch inference failed: {e}", exc_info=True)
            raise RuntimeError(f"Batch inference failed: {str(e)}")

    def profile_inference(
        self, 
        inputs: Any, 
        warmup_runs: int = 5,
        profile_runs: int = 50
    ) -> Dict[str, Any]:
        """
        Profile the inference pipeline with detailed performance metrics
        (preprocess, inference, postprocess).
        """
        if self._shutdown_event.is_set():
            raise ShutdownError("Engine is shutting down.")
        
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32)

        # Ensure batch dimension
        if self.input_shape and inputs.dim() == len(self.input_shape):
            inputs = inputs.unsqueeze(0)

        if inputs.device != self.primary_device:
            inputs = inputs.to(self.primary_device)

        # Warmup
        with torch.inference_mode():
            for _ in range(warmup_runs):
                pre = self.preprocessor(inputs)
                _ = self.model(pre)

        timings = {
            "preprocess_ms": [],
            "inference_ms": [],
            "postprocess_ms": [],
            "total_ms": []
        }

        device_type = self.primary_device.type
        # GPU profiling using cuda events
        if device_type == "cuda":
            for _ in range(profile_runs):
                start_ev = torch.cuda.Event(enable_timing=True)
                pre_ev = torch.cuda.Event(enable_timing=True)
                inf_ev = torch.cuda.Event(enable_timing=True)
                post_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)

                # Record start
                start_ev.record()
                with torch.inference_mode():
                    pre_data = self.preprocessor(inputs)
                pre_ev.record()

                with torch.inference_mode(), torch.amp.autocast(enabled=self.use_fp16, device_type="cuda"):
                    out = self.model(pre_data)
                inf_ev.record()

                with torch.inference_mode():
                    _ = self.postprocessor(out)
                post_ev.record()
                end_ev.record()

                torch.cuda.synchronize()
                t_pre = start_ev.elapsed_time(pre_ev)
                t_inf = pre_ev.elapsed_time(inf_ev)
                t_post = inf_ev.elapsed_time(post_ev)
                t_total = start_ev.elapsed_time(end_ev)
                timings["preprocess_ms"].append(t_pre)
                timings["inference_ms"].append(t_inf)
                timings["postprocess_ms"].append(t_post)
                timings["total_ms"].append(t_total)
        else:
            # CPU or MPS fallback using perf_counter
            for _ in range(profile_runs):
                start_total = time.perf_counter()
                start_pre = time.perf_counter()
                with torch.inference_mode():
                    pre_data = self.preprocessor(inputs)
                end_pre = time.perf_counter()

                start_inf = time.perf_counter()
                with torch.inference_mode(), torch.amp.autocast(enabled=(self.use_fp16 and device_type=="mps")):
                    out = self.model(pre_data)
                end_inf = time.perf_counter()

                start_post = time.perf_counter()
                with torch.inference_mode():
                    _ = self.postprocessor(out)
                end_post = time.perf_counter()
                end_total = time.perf_counter()

                timings["preprocess_ms"].append((end_pre - start_pre) * 1000)
                timings["inference_ms"].append((end_inf - start_inf) * 1000)
                timings["postprocess_ms"].append((end_post - start_post) * 1000)
                timings["total_ms"].append((end_total - start_total) * 1000)

        # Compute stats
        metrics = {}
        for k, vals in timings.items():
            arr = np.array(vals)
            metrics[f"{k}_mean"] = arr.mean()
            metrics[f"{k}_median"] = np.median(arr)
            metrics[f"{k}_min"] = arr.min()
            metrics[f"{k}_max"] = arr.max()
            metrics[f"{k}_std"] = arr.std()

        metrics["throughput_items_per_second"] = 1000.0 / metrics["total_ms_mean"]
        metrics["pipeline_efficiency"] = metrics["inference_ms_mean"] / metrics["total_ms_mean"]

        if device_type == "cuda":
            idx = self.primary_device.index or 0
            metrics["gpu_memory_usage_mb"] = torch.cuda.max_memory_allocated(idx) / (1024**2)
            torch.cuda.reset_peak_memory_stats(idx)

        self.logger.info(
            f"Profile: {profile_runs} runs, "
            f"Latency (mean): {metrics['total_ms_mean']:.2f}ms, "
            f"Throughput: {metrics['throughput_items_per_second']:.2f} items/s"
        )
        return metrics

    def get_metrics(self) -> Dict[str, Any]:
        """Return runtime metrics like throughput, memory, queue size, etc."""
        now = time.monotonic()
        elapsed = now - self._last_metrics_reset

        # Batch times
        if not self._batch_processing_times:
            avg_batch_time = 0.0
            p95_batch_time = 0.0
        else:
            recent_times = self._batch_processing_times[-100:]
            avg_batch_time = sum(recent_times) / len(recent_times)
            if len(recent_times) >= 20:
                p95_batch_time = float(np.percentile(recent_times, 95))
            else:
                p95_batch_time = avg_batch_time

        # Throughput
        throughput = self._successful_requests / max(1e-6, elapsed)

        # GPU memory
        memory_usage_mb = 0
        gpu_memory_percent = 0
        if self.primary_device.type == "cuda":
            idx = self.primary_device.index or 0
            props = torch.cuda.get_device_properties(idx)
            allocated = torch.cuda.memory_allocated(idx)
            total = props.total_memory
            memory_usage_mb = allocated / (1024**2)
            gpu_memory_percent = (allocated / total) * 100

        # Periodically reset metrics
        if elapsed > 300:  # reset every 5 minutes
            self._last_metrics_reset = now
            self._successful_requests = 0
            self._failed_requests = 0
            self._batch_processing_times = self._batch_processing_times[-100:]

        return {
            "queue_size": self.request_queue.qsize(),
            "average_batch_time": avg_batch_time,
            "p95_batch_time": p95_batch_time,
            "total_batches_processed": len(self._batch_processing_times),
            "throughput_per_second": throughput,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "guard_triggers": self._guard_triggered_count,
            "memory_usage_mb": memory_usage_mb,
            "gpu_memory_percent": gpu_memory_percent,
            "batch_size": self.config.batch_size,
            "uptime_seconds": elapsed,
        }

    async def cleanup(self):
        """Gracefully stop all background tasks, clear queues, and free resources."""
        if self._shutdown_event.is_set():
            return
        self.logger.info("Engine shutting down...")
        self._shutdown_event.set()

        # Cancel tasks
        tasks = [self.batch_processor_task, self.autoscale_task, self.monitor_task]
        tasks = [t for t in tasks if t is not None and not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Drain queue
        pending_requests = []
        while not self.request_queue.empty():
            try:
                item = self.request_queue.get_nowait()
                pending_requests.append(item)
            except asyncio.QueueEmpty:
                break

        for item in pending_requests:
            if not item.future.done():
                item.future.set_exception(ShutdownError("Engine shutting down."))

        # Shutdown executors
        self.logger.info("Shutting down executors...")
        self.executor.shutdown(wait=False)
        self.guard_executor.shutdown(wait=False)
        self.inference_executor.shutdown(wait=False)

        await self._exit_stack.aclose()
        
        if self.primary_device.type == 'cuda':
            torch.cuda.empty_cache()

        self.logger.info("Cleanup complete.")

    def shutdown_sync(self):
        """
        Synchronously shut down the engine for non-async contexts.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None or not loop.is_running():
            # No running loop, create one
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.cleanup())
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        else:
            fut = asyncio.run_coroutine_threadsafe(self.cleanup(), loop)
            fut.result(timeout=60)

    def register_managed_resource(self, resource):
        """Register a context manager resource for cleanup."""
        return self._exit_stack.enter_context(resource)

    async def register_async_resource(self, resource):
        """Register an async context manager resource for cleanup."""
        return await self._exit_stack.enter_async_context(resource)

    def is_ready(self) -> bool:
        """Check if engine is fully ready for inference."""
        return self._ready and not self._shutdown_event.is_set()

    def get_input_shape(self) -> Optional[torch.Size]:
        """Return the engine's detected input shape."""
        return self.input_shape

    def clear_metrics(self):
        """Reset all accumulated metrics."""
        self._batch_processing_times.clear()
        self._last_metrics_reset = time.monotonic()
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._guard_triggered_count = 0
        self.logger.info("Metrics reset.")

    def update_config(self, **kwargs):
        """Update specific config fields at runtime."""
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
                self.logger.info(f"Config updated: {k}={v}")
            else:
                self.logger.warning(f"Ignoring unknown config param: {k}")

    ############################################################################
    # Context Management
    ############################################################################
    async def __aenter__(self):
        await self._startup_complete.wait()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    def __enter__(self):
        while not self._startup_complete.is_set():
            time.sleep(0.01)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown_sync()


################################################################################
# Batch Inference Helper Function
################################################################################
async def batch_inference_parallel(
    engine: InferenceEngine,
    inputs: List[Any],
    max_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    batch_timeout: float = 30.0
) -> List[torch.Tensor]:
    """
    Process a large list of inputs in parallel, each via engine.run_inference_async.

    Args:
        engine:       The InferenceEngine instance.
        inputs:       List of inputs to process.
        max_workers:  Maximum number of parallel tasks (defaults to # CPU cores or 32).
        chunk_size:   How many items per chunk (defaults to 2 * engine.config.batch_size).
        batch_timeout:Timeout for each chunk's completion.

    Returns:
        List of outputs in the same order as 'inputs'.
    """
    if not inputs:
        return []

    n_inputs = len(inputs)
    if chunk_size is None:
        chunk_size = 2 * getattr(engine.config, "batch_size", 16)

    if max_workers is None:
        max_workers = min(32, os.cpu_count() or 4)

    n_chunks = (n_inputs + chunk_size - 1) // chunk_size
    max_workers = min(max_workers, n_chunks)

    async def process_chunk(sub_inputs):
        tasks = [
            asyncio.create_task(
                asyncio.wait_for(engine.run_inference_async(item), timeout=batch_timeout)
            )
            for item in sub_inputs
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    semaphore = asyncio.Semaphore(max_workers)

    async def process_with_semaphore(sub_inputs):
        async with semaphore:
            return await process_chunk(sub_inputs)

    # Split into chunks
    chunks = [inputs[i:i+chunk_size] for i in range(0, n_inputs, chunk_size)]
    chunk_tasks = [process_with_semaphore(chunk) for chunk in chunks]
    chunk_results = await asyncio.gather(*chunk_tasks)

    results = []
    for cr in chunk_results:
        results.extend(cr)

    # Raise exceptions in order
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            raise RuntimeError(f"Error at index {i}: {res}")
    return results


################################################################################
# Factory Function
################################################################################
def create_inference_engine(
    model: nn.Module,
    config: Optional[EngineConfig] = None,
    **kwargs
) -> InferenceEngine:
    """
    Factory function to create and initialize an InferenceEngine with defaults.

    Args:
        model:  A PyTorch model (nn.Module).
        config: Optional EngineConfig object.
        kwargs: Additional overrides for the InferenceEngine constructor.

    Returns:
        InferenceEngine instance.
    """
    if config is None:
        config = EngineConfig(
            debug_mode=kwargs.pop("debug_mode", False),
            batch_size=kwargs.pop("batch_size", 16),
            queue_size=kwargs.pop("queue_size", 1000),
            async_mode=kwargs.pop("async_mode", True),
            warmup_runs=kwargs.pop("warmup_runs", 5),
            auto_tune_batch_size=kwargs.pop("auto_tune_batch_size", True),
        )

    device = kwargs.pop("device", None)
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    return InferenceEngine(
        model=model,
        device=device,
        config=config,
        **kwargs
    )

import asyncio
import contextlib
import functools
import logging
import time
from contextlib import nullcontext, AsyncExitStack
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Tuple, cast, Set, Deque
import sys
import os
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
import weakref
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity


################################################################################
# Enums and Constants
################################################################################
class ExecutorType(Enum):
    THREAD = auto()
    PROCESS = auto()


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple M-series GPUs


class BatchState(Enum):
    """States for inflight batching state machine"""
    COLLECTING = auto()    # Accepting new items
    PROCESSING = auto()    # Processing but can still accept new items
    INFERENCE = auto()     # Running the model (no new items)
    DISTRIBUTING = auto()  # Distributing results
    COMPLETE = auto()      # Batch processing completed


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
    timestamp: float = field(default_factory=time.monotonic)  # For tracking request age
    processed_input: Optional[torch.Tensor] = None  # Cached preprocessed input
    
    def __lt__(self, other: "RequestItem"):
        # Priority queue uses __lt__ for ordering
        if self.priority == other.priority:
            return self.timestamp < other.timestamp
        return self.priority < other.priority
    
    @property
    def age(self) -> float:
        """Return age of request in seconds."""
        return time.monotonic() - self.timestamp


################################################################################
# Batch Container
################################################################################
@dataclass
class BatchContainer:
    """Container for inflight batching state management"""
    items: List[RequestItem] = field(default_factory=list)
    state: BatchState = BatchState.COLLECTING
    max_size: int = 32
    creation_time: float = field(default_factory=time.monotonic)
    inflight_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    # Tracking fields
    preprocessing_time: float = 0.0
    inference_time: float = 0.0
    postprocessing_time: float = 0.0
    
    def __len__(self):
        return len(self.items)
    
    @property
    def age(self) -> float:
        """Return age of batch in seconds."""
        return time.monotonic() - self.creation_time
    
    def can_accept_items(self) -> bool:
        """Check if this batch can accept new items."""
        return (len(self.items) < self.max_size and 
                self.state in (BatchState.COLLECTING, BatchState.PROCESSING))


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
        max_batch_size: int = 512,
        binary_search: bool = True
    ) -> int:
        """
        Estimate an optimal batch size based on memory constraints for a
        given model and device. Uses binary search for efficiency if enabled.
        """
        if target_device.type != "cuda" or not torch.cuda.is_available():
            # No estimation needed for CPU or MPS
            return max_batch_size
            
        device_idx = target_device.index or 0
        device_props = torch.cuda.get_device_properties(device_idx)
        available_mem = device_props.total_memory * target_memory_fraction
        
        # Clear GPU stats before we start
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_idx)
        
        if binary_search:
            # More efficient binary search approach
            low, high = 1, max_batch_size
            result = 1
            
            while low <= high:
                mid = (low + high) // 2
                try:
                    # Test with this batch size
                    test_batch = sample_input.expand(mid, *sample_input.shape)
                    test_batch = test_batch.to(target_device, non_blocking=True)
                    
                    with torch.inference_mode():
                        _ = model(test_batch)
                    
                    used_mem = torch.cuda.max_memory_allocated(device_idx)
                    if used_mem < available_mem:
                        result = mid  # This worked, but we might fit more
                        low = mid + 1
                    else:
                        high = mid - 1
                        
                    # Clean up
                    del test_batch
                    torch.cuda.reset_peak_memory_stats(device_idx)
                    torch.cuda.empty_cache()
                    
                except RuntimeError:  # OOM error
                    high = mid - 1
                    torch.cuda.empty_cache()
            
            return result
        
        # Original sequential approach as fallback
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
# Fallback EngineConfig with new settings
################################################################################
class EngineConfig:
    """
    Engine configuration with enhanced parameters for zero autoscaling and
    inflight batching.
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
        # New parameters
        enable_zero_autoscaling: bool = True,
        zero_scale_idle_threshold: float = 15.0,  # Seconds with no requests before scaling to 0
        zero_scale_wakeup_time: float = 0.05,     # Max seconds to wait when scaling up from 0
        enable_inflight_batching: bool = True,
        max_inflight_batches: int = 2,
        inflight_batch_timeout: float = 0.1,      # Max time to wait when adding to inflight batch
        inflight_preprocessing_limit: float = 0.5, # Max preprocessing time ratio for inflight batching
        use_async_preprocessing: bool = True,
        pin_memory: bool = True,
        max_preprocessing_workers: int = 4,
        prefetch_factor: int = 2,
        memory_efficient_inference: bool = True,
        adaptive_batch_timeout: bool = True,       # Adjust batch timeout based on load
        batch_timeout_scale_factor: float = 0.5,   # Scales batch timeout under load
        trace_inputs: bool = False,                # Track input distributions
        optimize_cuda_graphs: bool = False,        # Use CUDA graphs for optimization
        precompile_for_sizes: Optional[List[int]] = None,  # Batch sizes to precompile
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
        
        # New parameters for zero autoscaling
        self.enable_zero_autoscaling = enable_zero_autoscaling
        self.zero_scale_idle_threshold = zero_scale_idle_threshold
        self.zero_scale_wakeup_time = zero_scale_wakeup_time
        
        # New parameters for inflight batching
        self.enable_inflight_batching = enable_inflight_batching
        self.max_inflight_batches = max_inflight_batches
        self.inflight_batch_timeout = inflight_batch_timeout
        self.inflight_preprocessing_limit = inflight_preprocessing_limit
        
        # Optimizations
        self.use_async_preprocessing = use_async_preprocessing
        self.pin_memory = pin_memory
        self.max_preprocessing_workers = max_preprocessing_workers
        self.prefetch_factor = prefetch_factor
        self.memory_efficient_inference = memory_efficient_inference
        self.adaptive_batch_timeout = adaptive_batch_timeout
        self.batch_timeout_scale_factor = batch_timeout_scale_factor
        self.trace_inputs = trace_inputs
        self.optimize_cuda_graphs = optimize_cuda_graphs
        self.precompile_for_sizes = precompile_for_sizes
    
    def configure_logging(self):
        """Set up logging based on debug mode."""
        if self.debug_mode:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)


################################################################################
# Optimized Inference Engine
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
        Enhanced asynchronous inference engine with optimized execution and 
        advanced batching capabilities.

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
        self._sleep_task = None
        self._wake_event = asyncio.Event()

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
        self._batch_processing_times: Deque[float] = deque(maxlen=200)
        self._preprocessing_times: Deque[float] = deque(maxlen=100)
        self._last_metrics_reset = time.monotonic()
        self._last_request_time = time.monotonic()  # For zero autoscaling
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._guard_triggered_count = 0
        self._current_batch_size = self.config.batch_size
        self._paused = False  # For zero autoscaling
        
        # Priority queue for requests 
        self.request_queue = asyncio.PriorityQueue(maxsize=self.config.queue_size)

        # Inflight batching support
        self._active_batches: List[BatchContainer] = []
        self._inflight_batch_lock = asyncio.Lock()
        
        # CUDA Graphs support
        self._cuda_graph_cache = {}
        self._model_callable = None

        # Initialize and prepare model
        try:
            self.model = self._prepare_model(model)
            self.input_shape = self._detect_input_shape()

            # Auto-tune batch size if configured
            if self.config.auto_tune_batch_size and self.input_shape:
                self._tune_batch_size()
            
            # Precompile for common batch sizes if requested
            if self.config.optimize_cuda_graphs and self.config.precompile_for_sizes:
                self._precompile_cuda_graphs()
            
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
        max_preprocessing = self.config.max_preprocessing_workers
        
        if executor_type.lower() == "process":
            # Process pool executors
            self.executor = ProcessPoolExecutor(max_workers=num_workers)
            self.guard_executor = ProcessPoolExecutor(max_workers=max(1, num_workers // 2))
            self.inference_executor = ProcessPoolExecutor(max_workers=max(1, num_workers // 2))
            self.preprocessing_executor = ProcessPoolExecutor(max_workers=max_preprocessing)
            self.logger.info(f"Using ProcessPoolExecutor with {num_workers} workers")
        else:
            # Thread pool executors (default)
            self.executor = DaemonThreadPoolExecutor(max_workers=num_workers)
            self.guard_executor = DaemonThreadPoolExecutor(max_workers=max(1, num_workers // 2))
            self.inference_executor = DaemonThreadPoolExecutor(max_workers=max(1, num_workers // 2))
            self.preprocessing_executor = DaemonThreadPoolExecutor(max_workers=max_preprocessing)
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

        # Setup CUDA Graphs wrapper if enabled
        if (self.config.optimize_cuda_graphs and 
            self.primary_device.type == "cuda" and 
            torch.cuda.get_device_capability(self.primary_device)[0] >= 7):  # Volta+ GPUs
            
            self._model_callable = self._make_cuda_graph_wrapper(model)
            self.logger.info("CUDA Graphs optimization enabled")
        else:
            self._model_callable = model

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

    def _make_cuda_graph_wrapper(self, model: nn.Module) -> Callable:
        """Creates a wrapper for the model that uses CUDA graphs for optimization"""
        def run_with_cuda_graph(x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            
            # Check if we have a cached graph for this batch size
            if batch_size in self._cuda_graph_cache:
                graph_info = self._cuda_graph_cache[batch_size]
                # Copy input to the static input tensor
                graph_info['static_input'].copy_(x)
                # Replay the graph
                graph_info['graph'].replay()
                # Return a clone of the output to avoid modifying the static tensor
                return graph_info['static_output'].clone()
            
            # If no cached graph, run normally and try to capture (unless too many cached already)
            if len(self._cuda_graph_cache) < 10:  # Limit cache size
                try:
                    # Warmup 
                    for _ in range(3):
                        with torch.no_grad():
                            model(x)
                    
                    # Create static tensors for input and output
                    static_input = x.clone()
                    # Determine output shape with a dry run
                    with torch.no_grad():
                        static_output = model(static_input)
                    
                    # Capture the graph
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        static_output = model(static_input)
                    
                    # Cache the graph
                    self._cuda_graph_cache[batch_size] = {
                        'graph': g,
                        'static_input': static_input,
                        'static_output': static_output
                    }
                    
                    return static_output.clone()
                except Exception as e:
                    self.logger.warning(f"CUDA graph capture failed for batch_size={batch_size}: {e}")
            
            # Fallback to normal execution
            with torch.no_grad():
                return model(x)
        
        return run_with_cuda_graph

    def _precompile_cuda_graphs(self):
        """Precompile CUDA graphs for specified batch sizes"""
        if not self.config.optimize_cuda_graphs or not self.config.precompile_for_sizes:
            return
            
        if not (self.primary_device.type == "cuda" and torch.cuda.get_device_capability(self.primary_device)[0] >= 7):
            self.logger.warning("CUDA Graphs requires Volta+ GPUs, skipping precompilation")
            return
            
        self.logger.info(f"Precompiling CUDA graphs for batch sizes: {self.config.precompile_for_sizes}")
        
        for batch_size in self.config.precompile_for_sizes:
            try:
                dummy_input = torch.zeros((batch_size,) + self.input_shape, device=self.primary_device)
                # The wrapper will capture the graph
                _ = self._model_callable(dummy_input)
                self.logger.debug(f"Precompiled CUDA graph for batch_size={batch_size}")
            except Exception as e:
                self.logger.warning(f"Failed to precompile CUDA graph for batch_size={batch_size}: {e}")

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
                
            input_tensor = torch.randn(dummy_shape, device=self.primary_device)
            if self.config.pin_memory and self.primary_device.type != "cpu":
                input_tensor = input_tensor.pin_memory()
            return input_tensor
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
            sample_input = self._create_dummy_input()
            if sample_input is None:
                self.logger.error("TensorRT requires known input shape; skipping TRT compile.")
                return model
            
            # Create input specifications
            input_specs = [
                torch_tensorrt.Input(
                    min_shape=sample_input.shape,
                    opt_shape=sample_input.shape,
                    max_shape=sample_input.shape,
                    dtype=torch.half if self.use_fp16 else torch.float32
                )
            ]

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
            (1, 3, 224, 224),  # Standard ResNet/Vision
            (1, 3, 299, 299),  # Inception
            (1, 3, 384, 384),  # BERT-like
            (1, 1, 28, 28),    # MNIST
            (1, 784),          # Flattened MNIST
            (1, 512),          # Embedding
            (1, 10),           # Small vector
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
            max_batch_size=max_batch,
            binary_search=True  # Use optimized binary search
        )
        self.config.batch_size = tuned_bs
        self._current_batch_size = tuned_bs
        self.logger.info(f"Auto-tuned batch size to: {tuned_bs}")

    def _warmup(self):
        """
        Warm up the model to reduce first-inference latency and precompile kernels.
        """
        if (not self._ready) or (not self.input_shape) or (self.config.warmup_runs <= 0):
            return
        
        self.logger.info(f"Warming up model with {self.config.warmup_runs} iterations...")
        
        # Create dummy batches of different sizes to better prepare the model
        batch_sizes = [1, min(4, self.config.batch_size)]
        if self.config.batch_size > 4:
            batch_sizes.append(self.config.batch_size)
            
        for batch_size in batch_sizes:
            dummy_input = torch.zeros((batch_size,) + self.input_shape, device=self.primary_device)
            
            # For CUDA, ensure we use pinned memory if configured
            if self.config.pin_memory and self.primary_device.type == "cuda":
                cpu_tensor = torch.zeros((batch_size,) + self.input_shape).pin_memory()
                dummy_input.copy_(cpu_tensor.to(self.primary_device, non_blocking=True))
            
            with torch.inference_mode():
                for i in range(self.config.warmup_runs):
                    try:
                        if i == 0 and self._model_callable is not self.model:
                            # Ensure wrapper gets called for CUDA graphs
                            _ = self._model_callable(dummy_input)
                        else:
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
    # Zero Autoscaling Implementation
    ############################################################################
    async def _handle_zero_scaling(self):
        """
        Zero scaling logic: Pauses processing when idle, resumes on new requests.
        Creates a sleep task that can be cancelled when new requests arrive.
        """
        if not self.config.enable_zero_autoscaling:
            return
            
        # Check if we should enter sleep mode (idle for threshold duration)
        now = time.monotonic()
        idle_time = now - self._last_request_time
        queue_size = self.request_queue.qsize()
        
        if not self._paused and queue_size == 0 and idle_time > self.config.zero_scale_idle_threshold:
            self.logger.info(f"Zero autoscaling: Engine idle for {idle_time:.1f}s, entering sleep mode")
            self._paused = True
            
            # Free memory if using CUDA
            if self.primary_device.type == 'cuda' and self.config.memory_efficient_inference:
                # Keep model on CPU to reduce memory pressure but increase wakeup time
                if self.model is not self._model_callable:  # Not using CUDA graphs
                    self.model = self.model.cpu()
                torch.cuda.empty_cache()
                
            # Create a sleep task that can be cancelled
            self._wake_event.clear()
            self._sleep_task = asyncio.create_task(self._wake_event.wait())
            
        # Check if we should wake up (already paused but requests in queue)
        elif self._paused and (queue_size > 0 or self._wake_event.is_set()):
            self.logger.info("Zero autoscaling: Waking up engine")
            
            # Cancel sleep task if it exists
            if self._sleep_task and not self._sleep_task.done():
                self._sleep_task.cancel()
                self._sleep_task = None
            
            # Move model back to device if needed
            wake_start = time.monotonic()
            if self.primary_device.type == 'cuda' and self.model.device.type == 'cpu':
                self.model = self.model.to(self.primary_device)
                
            self._paused = False
            wake_time = time.monotonic() - wake_start
            if wake_time > self.config.zero_scale_wakeup_time:
                self.logger.warning(f"Slow wake-up time: {wake_time:.3f}s (target: {self.config.zero_scale_wakeup_time}s)")

    ############################################################################
    # Batch Processing Loop (async)
    ############################################################################
    async def _process_batches(self):
        """Main loop to process inference requests in batches."""
        self.logger.info("Starting batch processing loop.")
        while not self._shutdown_event.is_set():
            try:
                # Zero autoscaling - check if we should sleep or wake up
                await self._handle_zero_scaling()
                
                # If paused, briefly sleep (checking wake conditions frequently)
                if self._paused:
                    await asyncio.sleep(0.05)
                    continue
                
                # Either use inflight batching or standard batching
                if self.config.enable_inflight_batching:
                    await self._process_inflight_batches()
                else:
                    await self._process_standard_batches()

            except asyncio.CancelledError:
                self.logger.info("Batch processing loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in batch processing loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)
                
    async def _process_standard_batches(self):
        """Process batches in the standard (non-inflight) mode."""
        try:
            # Wait for the first item or timeout
            try:
                first_item = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                return

            if first_item is None:
                return

            batch_start_time = time.monotonic()
            batch_items = [first_item]
            max_batch_size = self._current_batch_size  # Use autoscaled batch size

            # Adaptive wait timeout based on load
            wait_timeout = self.config.batch_wait_timeout
            if self.config.adaptive_batch_timeout:
                queue_size = self.request_queue.qsize()
                fill_factor = queue_size / self.config.queue_size
                wait_timeout *= (1 - min(0.9, fill_factor * self.config.batch_timeout_scale_factor))
            
            # Wait a bit to collect items
            if wait_timeout > 0 and queue_size < 0.9 * self.config.queue_size:
                await asyncio.sleep(wait_timeout)

            # Collect additional items
            while len(batch_items) < max_batch_size:
                try:
                    next_item = self.request_queue.get_nowait()
                    batch_items.append(next_item)
                except asyncio.QueueEmpty:
                    break

            # Process collected batch
            await self._process_batch(batch_items, batch_start_time)

        except Exception as e:
            self.logger.error(f"Error in standard batch processing: {e}", exc_info=True)
    
    async def _process_inflight_batches(self):
        """Process batches with in-flight batching capabilities."""
        try:
            # Find existing batch that can accept items, or create new one
            async with self._inflight_batch_lock:
                active_batch = None
                # Try to find a batch accepting items
                for batch in self._active_batches:
                    if batch.can_accept_items():
                        active_batch = batch
                        break
                        
                # Create a new batch if needed
                if active_batch is None:
                    # Limit maximum number of active batches
                    if len(self._active_batches) >= self.config.max_inflight_batches:
                        # Wait for an existing batch to finish before creating a new one
                        # (small delay to avoid busy loop)
                        await asyncio.sleep(0.001)
                        return
                        
                    active_batch = BatchContainer(max_size=self._current_batch_size)
                    self._active_batches.append(active_batch)
            
            # Try to get an item from the queue with a short timeout
            try:
                item = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=self.config.inflight_batch_timeout
                )
                
                # Add to active batch
                async with active_batch.inflight_lock:
                    if active_batch.can_accept_items():
                        active_batch.items.append(item)
                    else:
                        # Add back to queue if batch is no longer accepting
                        await self.request_queue.put(item)
                        return
            except asyncio.TimeoutError:
                return
                
            # If this is the first item in the batch, start processing the batch
            if len(active_batch.items) == 1:
                asyncio.create_task(self._process_inflight_batch(active_batch))
            
            # If batch is full, mark it as not accepting new items
            if len(active_batch.items) >= active_batch.max_size:
                active_batch.state = BatchState.INFERENCE

        except Exception as e:
            self.logger.error(f"Error in inflight batch processing: {e}", exc_info=True)
            
    async def _process_inflight_batch(self, batch: BatchContainer):
        """
        Process a single inflight batch, which may receive new items while processing.
        This implements a state machine to handle concurrent preprocessing and dynamic batching.
        """
        try:
            batch_start_time = time.monotonic()
            preprocessing_start = time.monotonic()
            batch.state = BatchState.PROCESSING  # Accepting items during preprocessing
            
            # Process inputs in parallel
            input_tensors = []
            futures = []
            loop = asyncio.get_running_loop()
            
            # Gather preprocessed inputs (already processed or new)
            for item in batch.items:
                if item.processed_input is not None:
                    # Use already-preprocessed input
                    input_tensors.append(item.processed_input)
                    futures.append(item.future)
                else:
                    # Submit for preprocessing
                    futures.append(item.future)
                    if self.config.use_async_preprocessing:
                        # Async preprocessing
                        input_future = loop.run_in_executor(
                            self.preprocessing_executor,
                            self._preprocess_item,
                            item.input
                        )
                        input_tensors.append(input_future)
                    else:
                        # Synchronous preprocessing
                        processed = await loop.run_in_executor(
                            self.preprocessing_executor,
                            self._preprocess_item,
                            item.input
                        )
                        input_tensors.append(processed)
                        
            # Preprocess any async inputs
            if self.config.use_async_preprocessing:
                # If we have futures in input_tensors, await them
                for i, inp in enumerate(input_tensors):
                    if asyncio.isfuture(inp) or isinstance(inp, asyncio.Future):
                        try:
                            input_tensors[i] = await inp
                        except Exception as e:
                            # Mark the future as failed
                            if not futures[i].done():
                                futures[i].set_exception(e)
                            # Use a dummy tensor
                            input_tensors[i] = torch.zeros((1,) + self.input_shape, device=self.primary_device)
                            
            preprocessing_time = time.monotonic() - preprocessing_start
            batch.preprocessing_time = preprocessing_time
            
            # Calculate preprocessing ratio and update state
            time_ratio = preprocessing_time / (time.monotonic() - batch_start_time)
            if time_ratio < self.config.inflight_preprocessing_limit:
                # Fast preprocessing - keep accepting new items
                batch.state = BatchState.PROCESSING
            else:
                # Slow preprocessing - stop accepting new items
                batch.state = BatchState.INFERENCE
                
            # Now run inference - no more items can be added during this phase
            batch.state = BatchState.INFERENCE
            
            # Validate tensors and stack into batch
            valid_inputs = []
            valid_futures = []
            base_shape = None
            
            for tensor, future in zip(input_tensors, futures):
                try:
                    if not isinstance(tensor, torch.Tensor):
                        raise ValueError(f"Expected tensor, got {type(tensor)}")
                    
                    if base_shape is None:
                        base_shape = tensor.shape
                    elif tensor.shape != base_shape:
                        raise ValueError(f"Shape mismatch: {tensor.shape} vs {base_shape}")
                        
                    valid_inputs.append(tensor)
                    valid_futures.append(future)
                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
            
            if not valid_inputs:
                self.logger.warning("No valid inputs in batch")
                return
                
            # Run inference on validated batch
            try:
                inference_start = time.monotonic()
                
                # Stack tensors into batch
                stacked_tensor = torch.stack(valid_inputs)
                if stacked_tensor.device != self.primary_device:
                    stacked_tensor = stacked_tensor.to(self.primary_device, non_blocking=True)
                
                # Run inference
                outputs = await loop.run_in_executor(
                    self.inference_executor,
                    self._infer_batch,
                    stacked_tensor
                )
                
                batch.inference_time = time.monotonic() - inference_start
                self._distribute_results(outputs, valid_futures)
                self._successful_requests += len(valid_futures)

            except Exception as e:
                self.logger.error(f"Batch inference failed: {e}", exc_info=True)
                for fut in valid_futures:
                    if not fut.done():
                        fut.set_exception(ModelInferenceError(f"Inference failed: {str(e)}"))
                self._failed_requests += len(valid_futures)
                
            # Update batch statistics
            batch_time = time.monotonic() - batch_start_time
            self._batch_processing_times.append(batch_time)
            
            # Remove batch from active batches
            async with self._inflight_batch_lock:
                if batch in self._active_batches:
                    self._active_batches.remove(batch)
            
            batch.state = BatchState.COMPLETE
            
        except Exception as e:
            self.logger.error(f"Error processing inflight batch: {e}", exc_info=True)
            # Mark all unfulfilled futures as failed
            for item in batch.items:
                if not item.future.done():
                    item.future.set_exception(e)
            self._failed_requests += len(batch.items)
            
            # Remove batch from active batches
            async with self._inflight_batch_lock:
                if batch in self._active_batches:
                    self._active_batches.remove(batch)

    async def _process_batch(self, batch_items: List[RequestItem], batch_start_time: float):
        """Process a single batch of requests in standard mode."""
        if not batch_items:
            return
        
        batch_size = len(batch_items)
        preprocessed_inputs = []
        futures = [item.future for item in batch_items]
        
        # Preprocess inputs (with optional parallel preprocessing)
        try:
            loop = asyncio.get_running_loop()
            
            preprocess_start_time = time.monotonic()
            
            # Choose between parallel or sequential preprocessing based on configuration
            if self.config.use_async_preprocessing:
                # Parallel preprocessing tasks
                preprocess_tasks = [
                    loop.run_in_executor(self.preprocessing_executor, self._preprocess_item, item.input)
                    for item in batch_items
                ]
                preprocessed_inputs = await asyncio.gather(*preprocess_tasks, return_exceptions=True)
                
                # Check for preprocessing errors
                for i, result in enumerate(preprocessed_inputs):
                    if isinstance(result, Exception):
                        if not futures[i].done():
                            futures[i].set_exception(result)
                        preprocessed_inputs[i] = None  # Mark for filtering
                
                # Filter out None entries from failures
                valid_inputs = []
                valid_futures = []
                for inp, fut in zip(preprocessed_inputs, futures):
                    if inp is not None and not fut.done():
                        valid_inputs.append(inp)
                        valid_futures.append(fut)
                
                preprocessed_inputs = valid_inputs
                futures = valid_futures
            else:
                # Sequential preprocessing
                preprocessed_inputs = []
                for item in batch_items:
                    try:
                        processed = await loop.run_in_executor(
                            self.preprocessing_executor, 
                            self._preprocess_item, 
                            item.input
                        )
                        preprocessed_inputs.append(processed)
                    except Exception as e:
                        if not item.future.done():
                            item.future.set_exception(e)
            
            preprocess_time = time.monotonic() - preprocess_start_time
            self._preprocessing_times.append(preprocess_time)
            
            # Track empty batch case
            if not preprocessed_inputs:
                self.logger.warning("All items in batch failed preprocessing")
                return
            
            # Validate inputs
            self._validate_batch_inputs(preprocessed_inputs)
        except ValueError as e:
            self.logger.error(f"Batch input validation failed: {e}")
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)
            self._failed_requests += len(futures)
            return

        # Run inference
        try:
            stacked_tensor = torch.stack(preprocessed_inputs)
            if stacked_tensor.device != self.primary_device:
                stacked_tensor = stacked_tensor.to(self.primary_device, non_blocking=True)

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
                    f"({batch_size/batch_time:.2f} items/s). "
                    f"Preprocess: {preprocess_time:.4f}s ({preprocess_time/batch_time*100:.1f}%)"
                )

        except Exception as e:
            self.logger.error(f"Batch inference failed: {e}", exc_info=True)
            for fut in futures:
                if not fut.done():
                    fut.set_exception(ModelInferenceError(f"Inference failed: {str(e)}"))
            self._failed_requests += len(futures)

    def _preprocess_item(self, input_data: Any) -> torch.Tensor:
        """Preprocess a single input item and convert to tensor."""
        processed = self.preprocessor(input_data)
        
        # Convert to tensor if necessary
        if not isinstance(processed, torch.Tensor):
            processed = torch.as_tensor(processed, dtype=torch.float32)
            
        # Pin memory if using CUDA and configured
        if self.config.pin_memory and processed.device.type == "cpu" and self.primary_device.type == "cuda":
            processed = processed.pin_memory()
            
        # Check for NaN/Inf if enabled
        if self.config.check_nan_inf and (torch.isnan(processed).any() or torch.isinf(processed).any()):
            raise ValueError("Input contains NaN or Inf values.")
            
        # Validate shape
        if self.input_shape is not None:
            expected_shape = self.input_shape
            if processed.shape != expected_shape and processed.shape[1:] != expected_shape:
                raise ValueError(
                    f"Preprocessed input shape {processed.shape} doesn't match "
                    f"expected {expected_shape}."
                )
                
        return processed

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
            # Use the model callable (could be wrapped with CUDA graphs)
            outputs = self._model_callable(batch_tensor)
            
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
        Enhanced autoscaling with zero scaling capability: adjusts batch size 
        based on queue utilization and pauses processing when idle.
        """
        if not self.config.pid_controller and not self.config.enable_zero_autoscaling:
            self.logger.info("Autoscaling disabled.")
            return
        
        self.logger.info("Starting autoscaling task.")
        last_time = time.monotonic()

        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(self.config.autoscale_interval)
                now = time.monotonic()
                dt = now - last_time
                last_time = now
                
                # PID controller based batch size adjustment
                if self.config.pid_controller:
                    current_queue_size = self.request_queue.qsize()
                    utilization = (current_queue_size / self.config.queue_size) * 100.0

                    # Get average batch time
                    recent_times = list(self._batch_processing_times)[-20:]
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

                    new_bs = int(round(self._current_batch_size + adjustment))
                    new_bs = int(round(self._current_batch_size + adjustment))
                    new_bs = max(self.config.min_batch_size, min(new_bs, self.config.max_batch_size))
                    
                    if new_bs != self._current_batch_size:
                        self.logger.info(
                            f"Autoscaling batch size: {self._current_batch_size} → {new_bs} "
                            f"(utilization: {utilization:.1f}%, adjustment: {adjustment:.2f})"
                        )
                        self._current_batch_size = new_bs

        except asyncio.CancelledError:
            self.logger.info("Autoscaling task cancelled.")
        except Exception as e:
            self.logger.error(f"Error in autoscaling: {e}", exc_info=True)

    async def _monitor_health(self):
        """Periodically logs health metrics and performs cleanup."""
        self.logger.info("Starting health monitoring task.")
        last_time = time.monotonic()
        
        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(self.config.monitor_interval)
                now = time.monotonic()
                elapsed = now - last_time
                last_time = now
                
                # Log stats
                queue_size = self.request_queue.qsize()
                inflight_batches = len(self._active_batches) if hasattr(self, '_active_batches') else 0
                requests_per_second = self._successful_requests / elapsed if elapsed > 0 else 0
                success_rate = 100.0 * self._successful_requests / max(1, self._total_requests)
                
                # Get resource utilization
                gpu_info = ""
                if self.primary_device.type == "cuda":
                    mem_stats = MemoryManager.get_gpu_memory_usage()
                    for idx, (alloc, cached) in mem_stats.items():
                        alloc_gb = alloc / 1e9
                        cached_gb = cached / 1e9
                        gpu_info += f" GPU{idx}: {alloc_gb:.2f}GB/{cached_gb:.2f}GB"
                
                # Log batch timings
                avg_time = 0
                if self._batch_processing_times:
                    avg_time = sum(self._batch_processing_times) / len(self._batch_processing_times)
                
                self.logger.info(
                    f"Health Check: Queue={queue_size}, Batches={inflight_batches}, "
                    f"Requests={self._total_requests} ({requests_per_second:.1f}/s), "
                    f"Success rate={success_rate:.1f}%, AvgBatchTime={avg_time*1000:.1f}ms, "
                    f"BatchSize={self._current_batch_size}{gpu_info}"
                )
                
                # Reset counters
                self._total_requests = 0
                self._successful_requests = 0
                self._failed_requests = 0
                
                # Check for memory leaks
                if self.primary_device.type == "cuda" and self.config.memory_efficient_inference:
                    torch.cuda.empty_cache()
                
        except asyncio.CancelledError:
            self.logger.info("Health monitoring task cancelled.")
        except Exception as e:
            self.logger.error(f"Error in health monitoring: {e}", exc_info=True)
    
    ############################################################################
    # Public API
    ############################################################################
    async def infer(
        self, 
        input_data: Any, 
        priority: int = 0,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Asynchronously process a single inference request.
        
        Args:
            input_data: Input data to be processed
            priority: Processing priority (lower value = higher priority)
            timeout: Timeout in seconds for this request
            
        Returns:
            Processed output from the model
            
        Raises:
            asyncio.TimeoutError: If request times out
            ShutdownError: If engine is shutting down
            ModelError: For model-related errors
        """
        if self._shutdown_event.is_set():
            raise ShutdownError("Inference engine is shutting down")
            
        # Create a future for the result
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        # Update request tracking
        self._total_requests += 1
        self._last_request_time = time.monotonic()
        
        # If paused due to zero autoscaling, wake up
        if self._paused and self.config.enable_zero_autoscaling:
            self._wake_event.set()  # Wake up if sleeping
        
        # Create request item
        item = RequestItem(
            input=input_data,
            future=future,
            priority=priority,
            timestamp=time.monotonic()
        )
        
        # Fast path for empty queue with inflight batching
        if (self.config.enable_inflight_batching and 
            self.request_queue.empty() and 
            self._active_batches):
            
            # Try to add to an inflight batch
            for batch in self._active_batches:
                if batch.can_accept_items():
                    async with batch.inflight_lock:
                        if batch.can_accept_items():
                            batch.items.append(item)
                            # Don't need to put in queue
                            timeout_handle = None
                            break
            else:
                # No batch could accept, put in queue
                await self._enqueue_request(item)
        else:
            # Standard path - put in queue
            await self._enqueue_request(item)
        
        # Setup timeout if specified
        timeout_handle = None
        if timeout is not None:
            def on_timeout():
                if not future.done():
                    future.set_exception(asyncio.TimeoutError(
                        f"Request timed out after {timeout} seconds"
                    ))
            
            timeout_handle = loop.call_later(timeout, on_timeout)
        
        try:
            return await future
        finally:
            # Cancel timeout handler if it exists
            if timeout_handle is not None:
                timeout_handle.cancel()

    async def _enqueue_request(self, item: RequestItem) -> None:
        """Add a request to the queue with timeout handling."""
        try:
            # Use put_nowait with drop policy if configured
            if self.config.drop_requests_when_full and self.request_queue.full():
                if not item.future.done():
                    item.future.set_exception(asyncio.QueueFull(
                        "Request dropped due to queue full"
                    ))
                return
            
            # Otherwise try to put with timeout
            await asyncio.wait_for(
                self.request_queue.put(item),
                timeout=self.config.queue_wait_timeout
            )
        except asyncio.TimeoutError:
            if not item.future.done():
                item.future.set_exception(asyncio.TimeoutError(
                    f"Timed out after {self.config.queue_wait_timeout}s waiting for queue space"
                ))
            self._failed_requests += 1
    
    async def infer_batch(
        self, 
        inputs: List[Any], 
        priority: int = 0,
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Process a batch of inputs in one request. This creates a separate request
        for each input but ensures they are processed together.
        
        Args:
            inputs: List of input data
            priority: Processing priority (lower value = higher priority)
            timeout: Timeout in seconds for this batch request
            
        Returns:
            List of processed outputs
        """
        if not inputs:
            return []
            
        # Create a task for each input with same priority
        tasks = [
            self.infer(input_data, priority=priority, timeout=timeout)
            for input_data in inputs
        ]
        
        # Wait for all results
        return await asyncio.gather(*tasks)
    
    ############################################################################
    # Context Manager and Cleanup
    ############################################################################
    async def __aenter__(self):
        """Async context manager entry."""
        await self._startup_complete.wait()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown the inference engine."""
        if self._shutdown_event.is_set():
            return  # Already shutting down
            
        self.logger.info("Shutting down inference engine...")
        self._shutdown_event.set()
        
        # Cancel all tasks
        for task in [self.batch_processor_task, self.autoscale_task, self.monitor_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Cleanup executors
        self.executor.shutdown(wait=False)
        self.guard_executor.shutdown(wait=False)
        self.inference_executor.shutdown(wait=False)
        self.preprocessing_executor.shutdown(wait=False)
        
        # Clean up any remaining batch futures
        for batch in getattr(self, '_active_batches', []):
            for item in batch.items:
                if not item.future.done():
                    item.future.set_exception(ShutdownError("Inference engine shutting down"))
        
        # Clear the queue
        while not self.request_queue.empty():
            try:
                item = self.request_queue.get_nowait()
                if not item.future.done():
                    item.future.set_exception(ShutdownError("Inference engine shutting down"))
            except asyncio.QueueEmpty:
                break
                
        # Release CUDA memory if applicable
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Exit async stack
        await self._exit_stack.aclose()
        
        self.logger.info("Inference engine shutdown complete")

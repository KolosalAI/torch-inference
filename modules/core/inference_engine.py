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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.config import EngineConfig


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
        """Get GPU memory usage: {device_idx: (allocated_bytes, cached_bytes)}"""
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
        """Estimate optimal batch size based on memory constraints."""
        if target_device.type != "cuda" or not torch.cuda.is_available():
            return max_batch_size  # No estimation needed for CPU
            
        device_idx = target_device.index or 0
        device_props = torch.cuda.get_device_properties(device_idx)
        available_mem = device_props.total_memory * target_memory_fraction
        
        # Try incremental batch sizes
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            if batch_size > max_batch_size:
                return max_batch_size
                
            try:
                # Clear cache
                torch.cuda.empty_cache()
                
                # Create test batch
                test_batch = sample_input.expand(batch_size, *sample_input.shape)
                test_batch = test_batch.to(target_device)
                
                # Run model
                with torch.inference_mode():
                    _ = model(test_batch)
                    
                # Check memory usage
                used_mem = torch.cuda.max_memory_allocated(device_idx)
                if used_mem >= available_mem:
                    return max(1, batch_size // 2)
                
                # Try double size next
                del test_batch
                torch.cuda.empty_cache()
                
            except RuntimeError:
                # Memory error or other issue
                return max(1, batch_size // 2)
                
        return max_batch_size


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
        config: Optional["EngineConfig"] = None,
    ):
        """
        An optimized asynchronous inference engine with parallel execution capabilities.

        Args:
            model:          A PyTorch model (nn.Module).
            device:         Device specification (e.g. "cpu", "cuda", torch.device)
                            or a list of devices for DataParallel.
                            If None, uses "cuda" if available, else "cpu".
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

        # Feature flags
        self.use_fp16 = use_fp16
        self.use_tensorrt = use_tensorrt
        self._ready = False

        # Handle device selection
        self.devices = self._normalize_devices(device)
        self.primary_device = self.devices[0]

        # Setup executors
        self._init_executors()

        # Pre/Post Processors (with defaults)
        self.preprocessor = preprocessor if preprocessor else (lambda x: x)
        self.postprocessor = postprocessor if postprocessor else (lambda x: x)

        # Set up metrics tracking
        self._batch_processing_times = []
        self._last_metrics_reset = time.monotonic()
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._guard_triggered_count = 0
        
        # Initialize request queue with priority ordering
        self.request_queue = asyncio.PriorityQueue(maxsize=self.config.queue_size)

        # Initialize model and detect shapes for future validation
        try:
            self.model = self._prepare_model(model)
            self.input_shape = self._detect_input_shape()
            
            # Auto-tune batch size if configured
            if getattr(self.config, "auto_tune_batch_size", False) and self.input_shape:
                self._tune_batch_size()
                
            # Warmup the model
            self._warmup()
            
            # Show device info
            self._log_device_info()
            
            # Mark as ready for inference
            self._ready = True
            
        except Exception as e:
            self.logger.error(f"Model preparation failed: {e}", exc_info=True)
            raise ModelPreparationError(f"Failed to prepare model: {str(e)}") from e

        # Start background tasks if async mode is enabled
        if getattr(self.config, "async_mode", True):
            self.batch_processor_task = asyncio.create_task(self._process_batches())
            self.autoscale_task = asyncio.create_task(self._autoscale())
            self.monitor_task = asyncio.create_task(self._monitor_health())
        else:
            self.batch_processor_task = None
            self.autoscale_task = None
            self.monitor_task = None

        self._startup_complete.set()
        self.logger.info("InferenceEngine initialized successfully")

    # --------------------------------------------------------------------------
    # Internal Setup / Configuration
    # --------------------------------------------------------------------------
    def _normalize_devices(
        self,
        device: Optional[Union[str, torch.device, List[Union[str, torch.device]]]]
    ) -> List[torch.device]:
        """
        Converts device specification to a list of torch.device objects.
        Handles automatic device selection and validation.
        """
        if device is None:
            # Auto device selection with MPS (Apple Silicon) support
            if torch.cuda.is_available():
                return [torch.device("cuda")]
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                return [torch.device("mps")]
            else:
                return [torch.device("cpu")]
        elif isinstance(device, (str, torch.device)):
            # Convert single device
            device_obj = torch.device(device)
            
            # Validate device
            if device_obj.type == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available; falling back to CPU")
                return [torch.device("cpu")]
            elif device_obj.type == "mps" and not (hasattr(torch, 'mps') and torch.backends.mps.is_available()):
                self.logger.warning("MPS requested but not available; falling back to CPU")
                return [torch.device("cpu")]
                
            return [device_obj]
        elif isinstance(device, list):
            # Convert device list
            if not device:  # Empty list
                return self._normalize_devices(None)
                
            # Convert all devices
            device_objs = []
            for d in device:
                d_obj = torch.device(d)
                if (d_obj.type == "cuda" and not torch.cuda.is_available()) or \
                   (d_obj.type == "mps" and not (hasattr(torch, 'mps') and torch.backends.mps.is_available())):
                    self.logger.warning(f"Device {d_obj} requested but not available; skipping")
                    continue
                device_objs.append(d_obj)
                
            if not device_objs:  # No valid devices
                self.logger.warning("No valid devices in list; falling back to CPU")
                return [torch.device("cpu")]
                
            return device_objs
        else:
            raise ValueError(f"Invalid device argument type: {type(device)}")

    def _init_executors(self):
        """Creates thread/process executors according to config."""
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
        applies optimizations, and JIT compiles if enabled.
        """
        # Move model to primary device first
        model.to(self.primary_device)
        
        # Use DataParallel for multiple devices
        if len(self.devices) > 1:
            if all(d.type == "cuda" for d in self.devices):
                model = nn.DataParallel(model, device_ids=[d.index for d in self.devices if d.index is not None])
                self.logger.info(f"Using DataParallel across {len(self.devices)} CUDA devices")
            else:
                # Mixed device types - unsupported for now
                self.logger.warning("Mixed device types detected. Using only primary device.")

        # Set evaluation mode
        model.eval()
        
        # Apply torch.jit.script if configured
        if getattr(self.config, "use_jit", False):
            try:
                self.logger.info("Applying TorchScript compilation...")
                # Try exact tracing first
                dummy_input = self._create_dummy_input()
                if dummy_input is not None:
                    model = torch.jit.trace(model, dummy_input)
                    self.logger.info("Model successfully traced with TorchScript")
                else:
                    # Fallback to script mode (may not work for all models)
                    model = torch.jit.script(model)
                    self.logger.info("Model successfully compiled with TorchScript")
            except Exception as e:
                self.logger.warning(f"TorchScript compilation failed: {e}")
        
        # Apply TensorRT if configured
        if self.use_tensorrt and self.primary_device.type == "cuda":
            model = self._apply_tensorrt(model)
        
        return model

    def _create_dummy_input(self) -> Optional[torch.Tensor]:
        """Create a dummy input tensor for model tracing, based on inference shape."""
        try:
            # Start with basic guess
            dummy_shape = (1, 10)  # Fallback shape
            
            if self.config.input_shape is not None:
                # Use config-specified shape if available
                dummy_shape = (1,) + tuple(self.config.input_shape)
            elif hasattr(self.model, "input_shape"):
                # Use model's input_shape attribute if available
                dummy_shape = (1,) + tuple(self.model.input_shape)
            
            return torch.randn(dummy_shape, device=self.primary_device)
        except Exception as e:
            self.logger.warning(f"Failed to create dummy input: {e}")
            return None

    def _apply_tensorrt(self, model: nn.Module) -> nn.Module:
        """Apply TensorRT optimizations to the model."""
        try:
            import torch_tensorrt
        except ImportError:
            self.logger.warning("torch_tensorrt not installed; skipping TensorRT optimization")
            return model

        if len(self.devices) > 1:
            self.logger.warning("TensorRT not applied to DataParallel model")
            return model
            
        try:
            self.logger.info("Applying TensorRT optimization...")
            start_time = time.time()
            
            # Determine input specs based on available information
            input_specs = []
            if hasattr(self.config, "trt_input_shapes") and self.config.trt_input_shapes:
                # Use explicitly configured shapes if available
                for min_shape, opt_shape, max_shape in self.config.trt_input_shapes:
                    input_specs.append(
                        torch_tensorrt.Input(
                            min_shape=torch.Size(min_shape),
                            opt_shape=torch.Size(opt_shape),
                            max_shape=torch.Size(max_shape),
                            dtype=torch.half if self.use_fp16 else torch.float32
                        )
                    )
            else:
                # Create a default input spec using detected shape
                sample_input = self._create_dummy_input()
                if sample_input is not None:
                    input_specs.append(
                        torch_tensorrt.Input(
                            min_shape=sample_input.shape,
                            opt_shape=sample_input.shape,
                            max_shape=sample_input.shape,
                            dtype=torch.half if self.use_fp16 else torch.float32
                        )
                    )
                else:
                    raise ValueError("Could not determine input shape for TensorRT")
            
            # Apply TensorRT compilation
            enabled_precisions = {torch.half} if self.use_fp16 else {torch.float32}
            compiled_model = torch_tensorrt.compile(
                model, 
                inputs=input_specs,
                enabled_precisions=enabled_precisions,
                workspace_size=getattr(self.config, "trt_workspace_size", 1 << 30),
                debug=self.config.debug_mode
            )
            
            compile_time = time.time() - start_time
            self.logger.info(f"TensorRT optimization completed in {compile_time:.2f}s")
            return compiled_model
            
        except Exception as e:
            self.logger.error(f"TensorRT optimization failed: {e}", exc_info=True)
            return model  # Fall back to original model

    def _detect_input_shape(self) -> Optional[torch.Size]:
        """
        Detect input shape by examining model structure or running inference on dummy data.
        """
        try:
            # Try to extract from model attributes first
            if hasattr(self.model, "input_shape"):
                return self.model.input_shape
                
            # Try with config
            if hasattr(self.config, "input_shape") and self.config.input_shape:
                return torch.Size(self.config.input_shape)
                
            # Try with dummy run
            self.logger.debug("Probing model with dummy inputs to detect shape...")
            
            # Try common shapes
            shapes_to_try = [
                (1, 10),       # Basic vector
                (1, 3, 224, 224),  # Common image size (RGB)
                (1, 1, 28, 28),    # MNIST-style grayscale
                (1, 3, 299, 299),  # Inception-style
                (1, 3, 384, 384),  # BERT-style
                (1, 512),      # Embedding
            ]
            
            for shape in shapes_to_try:
                try:
                    dummy_input = torch.randn(shape, device=self.primary_device)
                    with torch.inference_mode():
                        _ = self.model(dummy_input)
                    self.logger.debug(f"Detected input shape: {shape[1:]}")
                    return torch.Size(shape[1:])  # Remove batch dimension
                except Exception:
                    continue
                    
            # Fallback: inspect model parameters for linear/conv layers
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    return torch.Size([module.in_features])
                elif isinstance(module, nn.Conv2d):
                    self.logger.debug(f"Found Conv2d layer {name} with in_channels={module.in_channels}")
                    return torch.Size([module.in_channels, 224, 224])  # Assumed image size
            
            self.logger.warning("Could not detect input shape")
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting input shape: {e}", exc_info=True)
            return None

    def _tune_batch_size(self):
        """
        Automatically determine optimal batch size based on memory constraints
        and model characteristics.
        """
        if self.input_shape is None:
            self.logger.warning("Cannot auto-tune batch size: input shape unknown")
            return
            
        self.logger.info("Auto-tuning batch size based on memory constraints...")
        
        # Create a sample input
        sample_input = torch.zeros((1,) + self.input_shape, device=self.primary_device)
        
        # Target memory fraction (avoid OOM)
        target_mem_fraction = getattr(self.config, "target_memory_fraction", 0.7)
        max_batch = getattr(self.config, "max_batch_size", 128)
        
        # Estimate optimal batch size
        optimal_batch_size = MemoryManager.estimate_batch_size(
            self.model,
            sample_input,
            self.primary_device,
            target_mem_fraction,
            max_batch
        )
        
        # Update config
        self.config.batch_size = optimal_batch_size
        self.config.max_batch_size = max(optimal_batch_size, self.config.max_batch_size)
        
        self.logger.info(f"Auto-tuned batch size: {optimal_batch_size}")

    def _warmup(self):
        """
        Warm up the model on dummy batches to optimize GPU kernels and caching.
        """
        if not self._ready or self.input_shape is None or self.config.warmup_runs <= 0:
            return
            
        self.logger.info(f"Warming up model with {self.config.warmup_runs} iterations...")
        
        # Create a dummy batch matching expected input shape
        dummy_input = torch.zeros(
            (self.config.batch_size,) + self.input_shape,
            device=self.primary_device
        )
        
        # Run warm-up iterations
        with torch.inference_mode():
            for i in range(self.config.warmup_runs):
                try:
                    self.model(dummy_input)
                    if i == 0 and self.primary_device.type == 'cuda':
                        # First run often has initialization overhead - clear cache
                        torch.cuda.empty_cache()
                except Exception as e:
                    self.logger.error(f"Warmup iteration failed: {e}")
                    break
                    
        # Ensure GPU completion
        if self.primary_device.type == 'cuda':
            torch.cuda.synchronize()
            
        self.logger.info("Warmup completed")

    def _log_device_info(self):
        """Log detailed information about the devices being used."""
        if len(self.devices) == 1:
            device = self.primary_device
            self.logger.info(f"Using device: {device}")
            
            if device.type == 'cuda':
                idx = device.index if device.index is not None else 0
                device_name = torch.cuda.get_device_name(idx)
                device_props = torch.cuda.get_device_properties(idx)
                total_mem_gb = device_props.total_memory / 1e9
                self.logger.info(
                    f"CUDA Device: {device_name}, Total Memory: {total_mem_gb:.2f} GB, "
                    f"CUDA: {device_props.major}.{device_props.minor}, "
                    f"SM count: {device_props.multi_processor_count}"
                )
            elif device.type == 'mps':
                self.logger.info("Using Apple Metal Performance Shaders (MPS) device")
        else:
            self.logger.info(f"Using {len(self.devices)} devices in parallel: {self.devices}")
            
            # Report memory for CUDA devices
            if any(d.type == 'cuda' for d in self.devices):
                for device in self.devices:
                    if device.type == 'cuda':
                        idx = device.index if device.index is not None else 0
                        device_name = torch.cuda.get_device_name(idx)
                        device_props = torch.cuda.get_device_properties(idx)
                        total_mem_gb = device_props.total_memory / 1e9
                        self.logger.info(
                            f"CUDA Device {idx}: {device_name}, Memory: {total_mem_gb:.2f} GB"
                        )

    # --------------------------------------------------------------------------
    # Batch Processing Loop (async)
    # --------------------------------------------------------------------------
    async def _process_batches(self):
        """
        Main batch processing loop: waits for requests, batches them,
        runs inference, and distributes results.
        """
        self.logger.info("Starting batch processing loop")
        
        while not self._shutdown_event.is_set():
            try:
                # Wait for the first request with timeout
                try:
                    first_item = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=self.config.timeout
                    )
                except asyncio.TimeoutError:
                    if self.config.debug_mode:
                        self.logger.debug("Batch processing loop: timeout waiting for requests")
                    continue
                    
                if first_item is None:
                    continue
                    
                # Start batch timing
                batch_start_time = time.monotonic()
                
                # Collect batch items
                batch_items = [first_item]
                max_batch_size = self.config.batch_size
                
                # Adaptive batch waiting time based on queue size
                queue_size = self.request_queue.qsize()
                queue_factor = queue_size / self.config.queue_size
                wait_time = self.config.batch_wait_timeout * (1 - min(0.9, queue_factor))
                
                # Only wait if queue isn't nearly full
                if queue_size < 0.9 * self.config.queue_size and wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                # Collect remaining items
                while len(batch_items) < max_batch_size:
                    try:
                        next_item = self.request_queue.get_nowait()
                        batch_items.append(next_item)
                    except asyncio.QueueEmpty:
                        break
                
                # Process the batch
                await self._process_batch(batch_items, batch_start_time)
                
            except asyncio.CancelledError:
                self.logger.info("Batch processing task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in batch processing loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)  # Brief pause to avoid tight error loop

    async def _process_batch(self, batch_items: List[RequestItem], batch_start_time: float):
        """Process a single batch of requests."""
        if not batch_items:
            return
            
        batch_size = len(batch_items)
        inputs = [item.input for item in batch_items]
        futures = [item.future for item in batch_items]
        
        # Safety check: validate inputs before running inference
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
            # Convert to torch tensor batch
            stacked_tensor = torch.stack(inputs)
            if stacked_tensor.device != self.primary_device:
                stacked_tensor = stacked_tensor.to(
                    self.primary_device, 
                    non_blocking=True
                )
            
            # Run inference in executor
            loop = asyncio.get_running_loop()
            outputs = await loop.run_in_executor(
                self.inference_executor,
                self._infer_batch,
                stacked_tensor
            )
            
            # Distribute results
            self._distribute_results(outputs, futures)
            
            # Update metrics
            batch_time = time.monotonic() - batch_start_time
            self._batch_processing_times.append(batch_time)
            self._successful_requests += len(futures)
            
            # Log performance
            if self.config.debug_mode:
                self.logger.debug(
                    f"Processed batch of {batch_size} items in {batch_time:.3f}s "
                    f"({batch_size/batch_time:.1f} items/s)"
                )
                
        except Exception as e:
            self.logger.error(f"Batch inference failed: {e}", exc_info=True)
            for fut in futures:
                if not fut.done():
                    fut.set_exception(ModelInferenceError(f"Inference failed: {str(e)}"))
            self._failed_requests += len(futures)

    def _validate_batch_inputs(self, inputs: List[torch.Tensor]) -> None:
        """
        Validate batch inputs for consistency and correctness.
        Raises ValueError if validation fails.
        """
        # Check if inputs are empty
        if not inputs:
            raise ValueError("Empty batch inputs")
            
        # Check tensor type
        for idx, tensor in enumerate(inputs):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Input at index {idx} is not a torch.Tensor")
                
        # Check shape consistency
        base_shape = inputs[0].shape
        for idx, tensor in enumerate(inputs[1:], 1):
            if tensor.shape != base_shape:
                raise ValueError(
                    f"Input at index {idx} has shape {tensor.shape}, "
                    f"expected {base_shape}"
                )
                
        # Check tensor dtype
        base_dtype = inputs[0].dtype
        for idx, tensor in enumerate(inputs[1:], 1):
            if tensor.dtype != base_dtype:
                raise ValueError(
                    f"Input at index {idx} has dtype {tensor.dtype}, "
                    f"expected {base_dtype}"
                )
                
        # Check for NaN/Inf values if configured
        if getattr(self.config, "check_nan_inf", False):
            for idx, tensor in enumerate(inputs):
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    raise ValueError(f"Input at index {idx} contains NaN or Inf values")

    def _infer_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a batch of inputs.
        Handles autocast and synchronization.
        """
        device_type = self.primary_device.type
        autocast_enabled = self.use_fp16 and device_type in ["cuda", "mps"]
        
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device_type, 
            enabled=autocast_enabled,
            dtype=torch.float16 if autocast_enabled else None,
            cache_enabled=True
        ):
            outputs = self.model(batch_tensor)
            
            # Ensure outputs are on CPU if needed for cheaper postprocessing
            if getattr(self.config, "output_to_cpu", False) and outputs.device.type != "cpu":
                outputs = outputs.cpu()
                
            return outputs

    def _distribute_results(self, outputs: torch.Tensor, futures: List[asyncio.Future]) -> None:
        """
        Distribute batch outputs to individual futures with proper postprocessing.
        """
        try:
            # Handle various output formats
            if isinstance(outputs, torch.Tensor):
                # If outputs is a single tensor with batch dimension
                if outputs.dim() > 0 and outputs.size(0) == len(futures):
                    # Split along batch dimension
                    splitted_outputs = list(torch.split(outputs, 1, dim=0))
                    splitted_outputs = [s.squeeze(0) for s in splitted_outputs]
                else:
                    # Assume same output for all requests (unusual)
                    self.logger.warning("Output tensor doesn't match batch size - duplicating result")
                    splitted_outputs = [outputs] * len(futures)
            elif isinstance(outputs, (list, tuple)):
                # Model returned a list/tuple of outputs
                if len(outputs) == len(futures):
                    splitted_outputs = outputs
                else:
                    raise ValueError(f"Output list length ({len(outputs)}) doesn't match futures count ({len(futures)})")
            else:
                # Fallback: duplicate same output for all requests
                splitted_outputs = [outputs] * len(futures)
                
            # Apply postprocessor and set results
            for fut, res in zip(futures, splitted_outputs):
                if not fut.done():
                    try:
                        processed_res = self.postprocessor(res)
                        fut.set_result(processed_res)
                    except Exception as e:
                        fut.set_exception(e)
                        
        except Exception as e:
            self.logger.error(f"Error distributing results: {e}", exc_info=True)
            # Set exception for all pending futures
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)

    # --------------------------------------------------------------------------
    # Autoscaling
    # --------------------------------------------------------------------------
    async def _autoscale(self):
        """
        PID-based autoscaling: adjusts batch size based on queue utilization.
        """
        if not hasattr(self.config, "pid_controller") or not self.config.pid_controller:
            self.logger.info("Autoscaling disabled (no PID controller configured)")
            return
            
        self.logger.info("Starting autoscaling task")
        last_time = time.monotonic()
        
        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(self.config.autoscale_interval)
                now = time.monotonic()
                dt = now - last_time
                last_time = now
                
                # Get metrics for PID control
                current_queue = self.request_queue.qsize()
                queue_capacity = self.config.queue_size
                utilization = (current_queue / queue_capacity) * 100.0
                
                # Batch time average
                recent_times = self._batch_processing_times[-20:] if self._batch_processing_times else [0.1]
                avg_batch_time = sum(recent_times) / len(recent_times)
                
                # Adjust batch size with PID controller
                adjustment = self.config.pid_controller.update(utilization, dt)
                
                # Apply adaptivity factors - increase more aggressively when queue filling up
                if utilization > 75 and adjustment > 0:
                    adjustment *= 2.0  # Scale up faster under load
                elif utilization < 25 and adjustment < 0 and avg_batch_time < 0.05:
                    adjustment *= 0.5  # Scale down more gently when idle
                
                # Apply adjustment with bounds
                new_batch_size = max(
                    self.config.min_batch_size,
                    min(int(round(self.config.batch_size + adjustment)), self.config.max_batch_size)
                )
                
                # Log significant changes
                if new_batch_size != self.config.batch_size:
                    self.logger.info(
                        f"Autoscale: queue={current_queue}/{queue_capacity} ({utilization:.1f}%), "
                        f"batch_size: {self.config.batch_size} → {new_batch_size}, "
                        f"avg_time: {avg_batch_time*1000:.1f}ms"
                    )
                    self.config.batch_size = new_batch_size
                    
        except asyncio.CancelledError:
            self.logger.info("Autoscale task cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in autoscale task: {e}", exc_info=True)

    # --------------------------------------------------------------------------
    # Health Monitoring
    # --------------------------------------------------------------------------
    async def _monitor_health(self):
        """
        Monitor engine health and log periodic status updates.
        Monitors memory usage, queue status, and request throughput.
        """
        self.logger.info("Starting health monitoring task")
        monitor_interval = getattr(self.config, "monitor_interval", 60.0)
        
        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(monitor_interval)
                
                # Collect metrics
                metrics = self.get_metrics()
                
                # Log status update
                self.logger.info(
                    f"Health status: queue={metrics['queue_size']}/{self.config.queue_size}, "
                    f"throughput={metrics['throughput_per_second']:.1f} req/s, "
                    f"batch_time={metrics['average_batch_time']*1000:.1f}ms, "
                    f"memory={metrics['memory_usage_mb']:.1f}MB"
                )
                
                # Check for stalled queue
                if (metrics['queue_size'] > 0 and 
                    metrics['throughput_per_second'] < 0.1 and 
                    len(self._batch_processing_times) > 10):
                    self.logger.warning(
                        "Possible stall detected: queue has items but throughput is low"
                    )
                    
                # Check for memory issues (GPU)
                if self.primary_device.type == "cuda" and metrics['gpu_memory_percent'] > 95:
                    self.logger.warning(
                        f"High GPU memory usage: {metrics['gpu_memory_percent']:.1f}%"
                    )
                    
                # Check for request timeouts
                old_request_count = 0
                threshold = getattr(self.config, "request_timeout", 30.0)
                
                # Find old requests (for debug purposes only - actual timeout handled by client)
                if self.request_queue.qsize() > 0:
                    for _ in range(min(10, self.request_queue.qsize())):
                        try:
                            item = self.request_queue._queue[_]  # Access private queue for inspection
                            if item.age > threshold:
                                old_request_count += 1
                        except (IndexError, AttributeError):
                            break
                            
                if old_request_count > 0:
                    self.logger.warning(f"Found {old_request_count} requests older than {threshold}s in queue")
                    
        except asyncio.CancelledError:
            self.logger.info("Health monitor task cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in health monitor: {e}", exc_info=True)

    # --------------------------------------------------------------------------
    # Guard Logic
    # --------------------------------------------------------------------------
    def _guard_sample(self, processed: torch.Tensor) -> bool:
        """
        Advanced guard with test-time augmentations for adversarial detection.
        Returns True if sample passes the guard checks.
        """
        if not getattr(self.config, "guard_enabled", False):
            return True  # Always pass if guard is disabled
            
        try:
            cfg = self.config
            sample = processed
            
            # Ensure sample is in correct format (add batch dim if needed)
            if not isinstance(sample, torch.Tensor):
                return False  # Non-tensor inputs fail guard
                
            if sample.dim() == len(self.input_shape):
                sample = sample.unsqueeze(0)
            sample = sample.to(self.primary_device)
            
            # Create augmented versions for robustness testing
            num_augs = cfg.guard_num_augmentations
            batch = sample.repeat(num_augs, *([1] * (sample.dim() - 1)))
            
            # Apply configured augmentations
            aug_types = getattr(cfg, "guard_augmentation_types", ["noise", "dropout"])
            
            # 1. Add noise augmentation
            if "noise" in aug_types:
                noise_level_range = getattr(cfg, "guard_noise_level_range", (0.01, 0.05))
                noise_levels = torch.empty(num_augs, device=self.primary_device).uniform_(*noise_level_range)
                noise_levels = noise_levels.view(num_augs, *([1] * (batch.dim() - 1)))
                noise = torch.randn_like(batch) * noise_levels
                batch = batch + noise
            
            # 2. Apply dropout
            if "dropout" in aug_types:
                dropout_rate = getattr(cfg, "guard_dropout_rate", 0.1)
                dropout_indices = torch.arange(num_augs, device=self.primary_device)
                dropout_indices = dropout_indices[dropout_indices % 3 == 0]  # Apply to every 3rd sample
                
                if len(dropout_indices) > 0:
                    mask = (torch.rand_like(batch[dropout_indices]) >= dropout_rate).float()
                    batch[dropout_indices] *= mask
            
            # 3. Flip (for 2D+ shapes)
            if "flip" in aug_types and len(self.input_shape) >= 2:
                flip_prob = getattr(cfg, "guard_flip_prob", 0.3)
                flip_flags = torch.rand(num_augs, device=self.primary_device) < flip_prob
                flip_indices = flip_flags.nonzero(as_tuple=True)[0]
                
                if flip_indices.numel() > 0:
                    batch[flip_indices] = torch.flip(batch[flip_indices], dims=[-1])
            
            # 4. Scaling
            if "scale" in aug_types:
                scale_range = getattr(cfg, "guard_scale_range", (0.95, 1.05))
                scale_factors = torch.empty(num_augs, device=self.primary_device).uniform_(*scale_range)
                scale_factors = scale_factors.view(num_augs, *([1] * (batch.dim() - 1)))
                batch = batch * scale_factors
            
            # Clamp to valid input range
            input_range = getattr(cfg, "guard_input_range", (0.0, 1.0))
            batch = torch.clamp(batch, min=input_range[0], max=input_range[1])
            
            # Run inference on augmented batch
            with torch.inference_mode():
                autocast_enabled = self.use_fp16 and (self.primary_device.type == "cuda")
                with torch.amp.autocast(device_type=self.primary_device.type, enabled=autocast_enabled):
                    preds = self.model(batch)
            
            # Convert to probabilities if needed
            if preds.dim() >= 2 and preds.size(1) > 1:  # Logits
                preds_probs = torch.softmax(preds, dim=1)
            else:  # Already probabilities or other output
                preds_probs = preds
            
            # Calculate robustness metrics
            confidence_threshold = getattr(cfg, "guard_confidence_threshold", 0.7)
            variance_threshold = getattr(cfg, "guard_variance_threshold", 0.1)
            
            # 1. Top class consistency
            if preds_probs.dim() >= 2:
                top_classes = preds_probs.argmax(dim=1)
                most_common_class = torch.mode(top_classes).values.item()
                class_consistency = (top_classes == most_common_class).float().mean().item()
            else:
                class_consistency = 1.0  # No classes to compare
                
            # 2. Confidence level
            if preds_probs.dim() >= 2:
                max_probs = preds_probs.max(dim=1)[0]
                mean_confidence = max_probs.mean().item()
                confidence_variance = max_probs.var().item()
            else:
                mean_confidence = preds_probs.mean().item()
                confidence_variance = preds_probs.var().item()
            
            # Advanced: check for adversarial signatures
            is_adversarial = False
            
            # Common adversarial signatures:
            # 1. Abnormally high confidence despite noise
            if mean_confidence > 0.99 and "noise" in aug_types:
                is_adversarial = True
                
            # 2. Uniform distribution across top classes
            if preds_probs.dim() >= 2 and preds_probs.size(1) > 3:
                sorted_probs, _ = torch.sort(preds_probs, dim=1, descending=True)
                top3_probs = sorted_probs[:, :3]
                top3_std = top3_probs.std(dim=1).mean().item()
                if top3_std < 0.05:  # Very uniform distribution among top classes
                    is_adversarial = True
            
            # Log guard metrics in debug mode
            if self.config.debug_mode:
                self.logger.debug(
                    f"Guard metrics: consistency={class_consistency:.3f}, "
                    f"confidence={mean_confidence:.3f}, variance={confidence_variance:.3f}, "
                    f"adversarial={is_adversarial}"
                )
            
            # Final decision: pass if meets all criteria
            passed = (
                class_consistency >= 0.8 and
                mean_confidence >= confidence_threshold and
                confidence_variance <= variance_threshold and
                not is_adversarial
            )
            
            if not passed:
                self._guard_triggered_count += 1
                
            return passed
            
        except Exception as e:
            self.logger.error(f"Error in guard check: {e}", exc_info=True)
            return False  # Fail closed on errors

    # --------------------------------------------------------------------------
    # Public Methods
    # --------------------------------------------------------------------------
    async def run_inference_async(self, input_data: Any, priority: int = 0) -> torch.Tensor:
        """
        Asynchronously process an inference request through the pipeline:
        preprocessing → guard → inference → postprocessing.
        
        Args:
            input_data: Raw input data to process
            priority: Request priority (lower = higher priority)
            
        Returns:
            Processed output tensor
            
        Raises:
            ShutdownError: If engine is shutting down
            GuardError: If input fails guard check
            ValueError: For invalid inputs
            ModelInferenceError: For inference failures
        """
        if self._shutdown_event.is_set():
            raise ShutdownError("Engine is shutting down")
            
        # Ensure engine is fully initialized
        if not self._startup_complete.is_set():
            await self._startup_complete.wait()
            
        if not self._ready:
            raise ModelPreparationError("Engine is not ready")
            
        self._total_requests += 1
        loop = asyncio.get_running_loop()
        
        try:
            # Preprocess input
            processed = await loop.run_in_executor(
                self.executor, self.preprocessor, input_data
            )
            
            # Convert to tensor if needed
            if not isinstance(processed, torch.Tensor):
                processed = torch.as_tensor(processed, dtype=torch.float32)
                
            # Validate shape if possible
            if self.input_shape is not None:
                if processed.shape != self.input_shape and processed.shape[1:] != self.input_shape:
                    raise ValueError(
                        f"Preprocessed input shape {processed.shape} doesn't match "
                        f"expected shape {self.input_shape}"
                    )
            
            # Check if guard is enabled and run guard check
            if getattr(self.config, "guard_enabled", False):
                is_safe = await loop.run_in_executor(
                    self.guard_executor, self._guard_sample, processed
                )
                
                if not is_safe:
                    self.logger.warning("Guard check failed - request rejected")
                    
                    # Return default response based on config
                    if getattr(self.config, "guard_fail_silently", False):
                        if self.config.num_classes > 0:
                            default_probs = torch.ones(self.config.num_classes, device="cpu") 
                            return default_probs / self.config.num_classes
                        else:
                            return torch.tensor([], device="cpu")
                    else:
                        raise GuardError("Input failed security checks")
            
            # Queue the request
            future = loop.create_future()
            request = RequestItem(processed, future, priority=priority)
            
            # Check if queue is full
            if self.request_queue.full():
                if getattr(self.config, "drop_requests_when_full", False):
                    self.logger.warning("Request queue full, dropping request")
                    raise RuntimeError("Request queue is full")
                else:
                    # Wait for space with timeout
                    timeout = getattr(self.config, "queue_wait_timeout", 10.0)
                    try:
                        # Try putting with timeout
                        await asyncio.wait_for(self.request_queue.put(request), timeout)
                    except asyncio.TimeoutError:
                        self.logger.error(f"Timed out after {timeout}s waiting for queue space")
                        raise RuntimeError(f"Timed out waiting for queue space after {timeout}s")
            else:
                # Queue has space
                await self.request_queue.put(request)
            
            if self.config.debug_mode:
                self.logger.debug(f"Request queued. Queue size: {self.request_queue.qsize()}")
            
            # If async mode is disabled, manually process right away
            if not getattr(self.config, "async_mode", True):
                # Create task to avoid blocking
                asyncio.create_task(self._process_batches())
            
            # Wait for result with timeout
            timeout = getattr(self.config, "request_timeout", 30.0)
            try:
                return await asyncio.wait_for(future, timeout)
            except asyncio.TimeoutError:
                self._failed_requests += 1
                self.logger.error(f"Inference request timed out after {timeout}s")
                raise TimeoutError(f"Inference request timed out after {timeout}s")
                
        except Exception as e:
            self._failed_requests += 1
            if not isinstance(e, (GuardError, ValueError, ShutdownError)):
                self.logger.error(f"Inference request failed: {e}", exc_info=True)
            raise

    def run_batch_inference(self, batch: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Synchronously run inference on a batch of preprocessed inputs.
        
        Args:
            batch: Either a batch tensor with shape (batch_size, ...) or a list of tensors
            
        Returns:
            Model output for the batch
            
        Raises:
            ValueError: For invalid inputs
            RuntimeError: For inference errors
        """
        if self._shutdown_event.is_set():
            raise ShutdownError("Engine is shutting down")
            
        # Convert list to tensor batch if needed
        if isinstance(batch, list):
            batch = torch.stack([
                x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
                for x in batch
            ], dim=0)
        elif not isinstance(batch, torch.Tensor):
            raise TypeError("run_batch_inference expects a torch.Tensor or list of Tensors")
            
        # Move to device if needed
        if batch.device != self.primary_device:
            batch = batch.to(self.primary_device, non_blocking=True)
            
        # Run inference
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
        Profile the inference pipeline with detailed performance metrics.
        
        Args:
            inputs: Input data to profile with
            warmup_runs: Number of warmup runs before profiling
            profile_runs: Number of runs to profile
            
        Returns:
            Dictionary of profiling metrics
        """
        if self._shutdown_event.is_set():
            raise ShutdownError("Engine is shutting down")
            
        # Convert to tensor if needed
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32)
            
        # Ensure batch dimension
        if self.input_shape is not None and inputs.dim() == len(self.input_shape):
            inputs = inputs.unsqueeze(0)
            
        # Move to target device
        if inputs.device != self.primary_device:
            inputs = inputs.to(self.primary_device)
            
        # Warmup
        with torch.inference_mode():
            for _ in range(warmup_runs):
                preprocessed = self.preprocessor(inputs)
                _ = self.model(preprocessed)
                
        # Detailed profiling
        timings = {
            "preprocess_ms": [],
            "inference_ms": [],
            "postprocess_ms": [],
            "total_ms": []
        }
        
        # Run profiling with CUDA events for precise GPU timing
        if self.primary_device.type == "cuda":
            for _ in range(profile_runs):
                start_event = torch.cuda.Event(enable_timing=True)
                preprocess_event = torch.cuda.Event(enable_timing=True)
                inference_event = torch.cuda.Event(enable_timing=True)
                postprocess_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                with torch.inference_mode():
                    preprocessed = self.preprocessor(inputs)
                    preprocessed = preprocessed.to(self.primary_device, non_blocking=True)
                    
                preprocess_event.record()
                with torch.inference_mode(), torch.amp.autocast(
                    device_type="cuda", enabled=self.use_fp16
                ):
                    output = self.model(preprocessed)
                    
                inference_event.record()
                with torch.inference_mode():
                    _ = self.postprocessor(output)
                    
                postprocess_event.record()
                end_event.record()
                
                # Synchronize and collect times
                torch.cuda.synchronize()
                timings["preprocess_ms"].append(start_event.elapsed_time(preprocess_event))
                timings["inference_ms"].append(preprocess_event.elapsed_time(inference_event))
                timings["postprocess_ms"].append(inference_event.elapsed_time(postprocess_event))
                timings["total_ms"].append(start_event.elapsed_time(end_event))
        else:
            # CPU profiling
            for _ in range(profile_runs):
                # Full pipeline
                start_total = time.perf_counter()
                
                # Preprocess
                start_pre = time.perf_counter()
                with torch.inference_mode():
                    preprocessed = self.preprocessor(inputs)
                end_pre = time.perf_counter()
                
                # Inference
                start_inf = time.perf_counter()
                with torch.inference_mode():
                    output = self.model(preprocessed)
                end_inf = time.perf_counter()
                
                # Postprocess
                start_post = time.perf_counter()
                with torch.inference_mode():
                    _ = self.postprocessor(output)
                end_post = time.perf_counter()
                
                end_total = time.perf_counter()
                
                timings["preprocess_ms"].append((end_pre - start_pre) * 1000)
                timings["inference_ms"].append((end_inf - start_inf) * 1000)
                timings["postprocess_ms"].append((end_post - start_post) * 1000)
                timings["total_ms"].append((end_total - start_total) * 1000)
                
        # Calculate statistics
        metrics = {}
        for key, values in timings.items():
            metrics[f"{key}_mean"] = np.mean(values)
            metrics[f"{key}_median"] = np.median(values)
            metrics[f"{key}_min"] = np.min(values)
            metrics[f"{key}_max"] = np.max(values)
            metrics[f"{key}_std"] = np.std(values)
            
        # Additional metrics
        metrics["throughput_items_per_second"] = 1000 / metrics["total_ms_mean"]
        metrics["pipeline_efficiency"] = (
            metrics["inference_ms_mean"] / metrics["total_ms_mean"]
        )
        
        # Detailed model info if available
        if self.primary_device.type == "cuda":
            with torch.cuda.device(self.primary_device):
                metrics["gpu_memory_usage_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
                torch.cuda.reset_peak_memory_stats()
                
        self.logger.info(
            f"Profiling results: {profile_runs} runs, "
            f"throughput: {metrics['throughput_items_per_second']:.1f} items/s, "
            f"latency: {metrics['total_ms_mean']:.2f}ms"
        )
            
        return metrics

    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns comprehensive health and performance metrics.
        """
        # Calculate time-window metrics
        time_now = time.monotonic()
        time_window = time_now - self._last_metrics_reset
        
        if not self._batch_processing_times:
            avg_batch_time = 0.0
            p95_batch_time = 0.0
        else:
            # Use recent batch times for better responsiveness
            recent_times = self._batch_processing_times[-100:]
            avg_batch_time = sum(recent_times) / len(recent_times)
            p95_batch_time = np.percentile(recent_times, 95) if len(recent_times) >= 20 else avg_batch_time
            
        # Calculate throughput (requests per second)
        throughput = self._successful_requests / max(0.001, time_window)
        
        # Reset counters periodically to keep metrics relevant
        if time_window > 300:  # Reset every 5 minutes
            self._last_metrics_reset = time_now
            self._successful_requests = 0
            self._failed_requests = 0
            self._batch_processing_times = self._batch_processing_times[-100:]
            
        # Memory metrics
        memory_usage_mb = 0
        gpu_memory_percent = 0
        
        if self.primary_device.type == "cuda":
            idx = self.primary_device.index if self.primary_device.index is not None else 0
            props = torch.cuda.get_device_properties(idx)
            allocated = torch.cuda.memory_allocated(idx)
            reserved = torch.cuda.memory_reserved(idx)
            
            memory_usage_mb = allocated / (1024 * 1024)
            total_memory = props.total_memory
            gpu_memory_percent = (allocated / total_memory) * 100
            
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
            "uptime_seconds": time.monotonic() - self._last_metrics_reset,
        }

    async def cleanup(self):
        """
        Gracefully clean up all resources and cancel background tasks.
        """
        if self._shutdown_event.is_set():
            return  # Already shutting down
            
        self.logger.info("Cleaning up engine resources...")
        self._shutdown_event.set()
        
        # Cancel all background tasks
        tasks = []
        for task in [self.batch_processor_task, self.autoscale_task, self.monitor_task]:
            if task is not None and not task.done():
                task.cancel()
                tasks.append(task)
                
        # Wait for tasks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        # Process any pending requests from the queue
        pending_requests = []
        while not self.request_queue.empty():
            try:
                item = self.request_queue.get_nowait()
                pending_requests.append(item)
            except asyncio.QueueEmpty:
                break
                
        # Cancel all pending requests
        for item in pending_requests:
            if not item.future.done():
                item.future.set_exception(ShutdownError("Engine shutting down"))
        
        # Shutdown executors
        self.logger.info("Shutting down executor pools...")
        self.executor.shutdown(wait=False)
        self.guard_executor.shutdown(wait=False)
        self.inference_executor.shutdown(wait=False)
        
        # Close managed resources
        await self._exit_stack.aclose()
        
        # Release CUDA cache if appropriate
        if self.primary_device.type == 'cuda':
            torch.cuda.empty_cache()
            
        self.logger.info("Engine cleanup completed")

    def shutdown_sync(self):
        """
        Synchronously shut down the engine, blocking until cleanup is complete.
        For non-async contexts.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No active event loop
            loop = None
            
        if loop is None or not loop.is_running():
            # No active loop, create one for cleanup
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.cleanup())
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        else:
            # Use running loop
            fut = asyncio.run_coroutine_threadsafe(self.cleanup(), loop)
            fut.result(timeout=60)  # Wait up to 60s for cleanup

    # --------------------------------------------------------------------------
    # Utility Methods
    # --------------------------------------------------------------------------
    def register_managed_resource(self, resource):
        """
        Register a resource to be automatically cleaned up.
        Resource should have a close() or aclose() method.
        """
        return self._exit_stack.enter_context(resource)
        
    async def register_async_resource(self, resource):
        """
        Register an async resource (with __aenter__/__aexit__) to be cleaned up.
        """
        return await self._exit_stack.enter_async_context(resource)
    
    def is_ready(self) -> bool:
        """Return True if the engine is ready for inference."""
        return self._ready and not self._shutdown_event.is_set()
    
    def get_input_shape(self) -> Optional[torch.Size]:
        """Return the expected input shape (without batch dimension)."""
        return self.input_shape
    
    def clear_metrics(self):
        """Reset all accumulated metrics."""
        self._batch_processing_times = []
        self._last_metrics_reset = time.monotonic()
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._guard_triggered_count = 0
        self.logger.info("Metrics have been reset")
        
    def update_config(self, **kwargs):
        """
        Update configuration parameters.
        Only updates parameters that exist in the current config.
        
        Args:
            **kwargs: Config parameter updates as keyword arguments
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config: {key} = {value}")
            else:
                self.logger.warning(f"Ignoring unknown config parameter: {key}")
    
    # --------------------------------------------------------------------------
    # Context Managers
    # --------------------------------------------------------------------------
    async def __aenter__(self):
        """Async context manager entry."""
        await self._startup_complete.wait()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.cleanup()
        
    def __enter__(self):
        """Synchronous context manager entry."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If there's a running event loop, wait for startup
            loop.run_until_complete(self._startup_complete.wait())
        else:
            # Otherwise, just block until startup is done
            while not self._startup_complete.is_set():
                time.sleep(0.01)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Synchronous context manager exit."""
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
    Process a large batch of inputs in parallel using the inference engine.
    
    Args:
        engine: The inference engine to use
        inputs: List of inputs to process
        max_workers: Maximum number of parallel tasks (default: min(32, #CPU cores))
        chunk_size: Size of each sub-batch (default: auto-determined)
        batch_timeout: Timeout in seconds for each batch
        
    Returns:
        List of output tensors corresponding to inputs
    """
    if not inputs:
        return []
        
    # Determine chunk size and number of workers
    n_inputs = len(inputs)
    if chunk_size is None:
        # Auto-determine chunk size based on engine batch size
        chunk_size = min(n_inputs, getattr(engine.config, "batch_size", 16) * 2)
        
    if max_workers is None:
        max_workers = min(32, os.cpu_count() or 4)
        
    # Calculate number of chunks
    n_chunks = (n_inputs + chunk_size - 1) // chunk_size
    max_workers = min(max_workers, n_chunks)  # Don't create more workers than chunks
    
    # Function to process a single chunk
    async def process_chunk(chunk_inputs):
        tasks = []
        for input_item in chunk_inputs:
            task = asyncio.create_task(
                asyncio.wait_for(
                    engine.run_inference_async(input_item),
                    timeout=batch_timeout
                )
            )
            tasks.append(task)
            
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process all chunks in parallel
    chunks = [inputs[i:i+chunk_size] for i in range(0, n_inputs, chunk_size)]
    
    # Use semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_workers)
    
    async def process_with_semaphore(chunk):
        async with semaphore:
            return await process_chunk(chunk)
            
    # Create tasks for all chunks
    chunk_tasks = [process_with_semaphore(chunk) for chunk in chunks]
    chunk_results = await asyncio.gather(*chunk_tasks)
    
    # Flatten results while preserving order
    results = []
    for chunk_result in chunk_results:
        results.extend(chunk_result)
        
    # Check for exceptions
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            raise RuntimeError(f"Error processing input at index {i}: {result}")
            
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
    Factory function to create and initialize an InferenceEngine with reasonable defaults.
    
    Args:
        model: The PyTorch model to use
        config: Optional engine configuration
        **kwargs: Additional arguments to pass to InferenceEngine constructor
        
    Returns:
        Initialized InferenceEngine instance
    """
    # Create default config if none provided
    if config is None:
        config = EngineConfig(
            debug_mode=kwargs.pop("debug_mode", False),
            batch_size=kwargs.pop("batch_size", 16),
            queue_size=kwargs.pop("queue_size", 1000),
            async_mode=kwargs.pop("async_mode", True),
            warmup_runs=kwargs.pop("warmup_runs", 5),
            auto_tune_batch_size=kwargs.pop("auto_tune_batch_size", True)
        )
        
    # Get device from kwargs or auto-select
    device = kwargs.pop("device", None)
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
    # Create and return engine
    return InferenceEngine(
        model=model,
        device=device,
        config=config,
        **kwargs
    )

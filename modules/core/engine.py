import asyncio
import logging
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union
from .preprocessor import BasePreprocessor
from .postprocessor import BasePostprocessor
import sys
import os
# Add the parent directory to sys.path so that imports like "core.engine" work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

# Assume a PIDController, BasePreprocessor, and BasePostprocessor are defined elsewhere.
# For example:



# ------------------------------------------------------------------------------
# Engine Configuration with Advanced Options (including TRT)
# ------------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import logging
import torch
from utils.config import EngineConfig

# ------------------------------------------------------------------------------
# Request Item Class
# ------------------------------------------------------------------------------
class RequestItem:
    """
    Holds a single inference request (processed input), an optional priority,
    and a Future to store the asynchronous result.
    """
    def __init__(self, input: Any, future: asyncio.Future, priority: int = 0):
        self.input = input
        self.future = future
        self.priority = priority  # Lower values indicate higher priority

    def __lt__(self, other: "RequestItem"):
        return self.priority < other.priority

# ------------------------------------------------------------------------------
# Inference Engine Class with Advanced Features, TensorRT, and Adversarial Guard
# ------------------------------------------------------------------------------
class InferenceEngine:
    """
    An asynchronous inference engine that supports:
      - Advanced dynamic batching (memory-, queue-, and PID-based autoscaling)
      - Multi-GPU support (DataParallel and a stub for DistributedDataParallel)
      - FP16 / mixed precision on CUDA
      - Option to use ThreadPool or ProcessPool for concurrency (if picklable)
      - Optional TensorRT compilation for improved GPU inference performance
      - Improved error handling, logging, profiling, and an adversarial guard.
    """
    def __init__(
        self,
        model: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        preprocessor: Optional[Callable[[Any], Any]] = None,
        postprocessor: Optional[Callable[[Any], Any]] = None,
        use_fp16: bool = False,
        use_tensorrt: bool = False,
        config: Optional["EngineConfig"] = None,
    ):
        # If no configuration is provided, create one with debug enabled.
        self.config = config or EngineConfig(debug_mode=True)
        self.config.configure_logging()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Automatic device detection and selection.
        self.device = self._auto_select_device(device)
        # Only enable FP16 if using a CUDA device.
        self.use_fp16 = use_fp16 and ('cuda' in str(self.device))
        # Store the use_tensorrt flag.
        self.use_tensorrt = use_tensorrt

        # Prepare the model (and wrap for multi-GPU if needed).
        self.model = self._prepare_model(model)

        # If enabled and using a CUDA device, compile the model with TensorRT.
        if self.use_tensorrt and self.device.type == "cuda":
            try:
                self.logger.info("Compiling model with TensorRT optimizations...")
                self.model, compile_time = self._compile_trt_model()
                self.logger.info(f"TensorRT model compiled in {compile_time:.2f}s")
            except Exception as e:
                self.logger.error(f"TensorRT compilation failed: {e}", exc_info=True)
                # Fall back to the original model if TRT compile fails.

        # Pre- and post-processing hooks.
        self.preprocessor = preprocessor if preprocessor is not None else BasePreprocessor()
        self.postprocessor = postprocessor if postprocessor is not None else BasePostprocessor()

        # Detect expected input shape (robust detection with fallback).
        self.input_shape = self._detect_input_shape()

        # Choose executor type for offloading inference.
        if self.config.executor_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=self.config.num_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        self.guard_executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        self.inference_executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        # Use an asyncio PriorityQueue to enable request prioritization.
        self.request_queue: asyncio.PriorityQueue["RequestItem"] = asyncio.PriorityQueue(maxsize=self.config.queue_size)

        # Create asynchronous tasks for batch processing and autoscaling.
        self.batch_processor_task = asyncio.create_task(self._process_batches())
        self.autoscale_task = asyncio.create_task(self._autoscale())

        # Warm up the model (especially important for CUDA).
        self._warmup()

        # Log device and configuration info.
        self._log_device_info()

    # --- Device and Model Setup ---
    def _auto_select_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """
        Automatically select an available device, falling back to CPU if no CUDA device is available.
        """
        device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Moves the model to the selected device and sets it to evaluation mode.
        If multi-GPU is enabled and available, wraps the model in DataParallel.
        A stub is provided for DistributedDataParallel.
        """
        # Move the model to the target device.
        model = model.to(self.device)

        # Set the model to evaluation mode.
        model.eval()

        # Retrieve configuration options.
        use_multigpu = self.config.use_multigpu
        device_ids = self.config.device_ids
        strategy = self.config.multigpu_strategy.lower() if hasattr(self.config, 'multigpu_strategy') else 'dataparallel'

        # Only proceed with multi-GPU setup if the device is CUDA and more than one GPU is available.
        if use_multigpu and self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            if strategy == 'dataparallel':
                self.logger.info(f"Using DataParallel on devices: {device_ids}")
                model = nn.DataParallel(model, device_ids=device_ids)
            elif strategy == 'distributed':
                raise NotImplementedError("DistributedDataParallel support is not yet implemented.")
            else:
                self.logger.warning(f"Unknown multi-GPU strategy: {self.config.multigpu_strategy}")
        elif use_multigpu and self.device.type != 'cuda':
            self.logger.warning("Multi-GPU is enabled but the selected device is not CUDA.")

        return model

    def _log_device_info(self):
        """
        Log useful details about the selected device.
        """
        self.logger.info(f"Running on device: {self.device}")
        
        if self.device.type == 'cuda':
            # Determine the GPU index; default to 0 if not specified.
            device_index = self.device.index if self.device.index is not None else 0
            
            # Retrieve device properties once.
            device_name = torch.cuda.get_device_name(device_index)
            device_props = torch.cuda.get_device_properties(device_index)
            cuda_version = torch.version.cuda
            
            total_mem_gb = device_props.total_memory / 1e9
            
            self.logger.info(f"GPU Name: {device_name}")
            self.logger.info(f"CUDA Version: {cuda_version}")
            self.logger.info(f"Total GPU Memory: {total_mem_gb:.2f} GB")

    def _detect_input_shape(self) -> Optional[torch.Size]:
        """
        Attempt to infer the model's expected (non-batch) input shape.
        First, try running a dummy input. If that fails, fall back to heuristics.
        """
        self.logger.debug("Detecting input shape via dummy run...")
        # Use a dummy input with an arbitrary shape. The shape can be adjusted if the expected input dimensionality is known.
        dummy_input = torch.randn(1, 10, device=self.device)
        try:
            with torch.no_grad():
                _ = self.model(dummy_input)
            # Exclude the batch dimension.
            shape = dummy_input.shape[1:]
            self.logger.debug(f"Inferred input shape from dummy run: {shape}")
            return shape
        except Exception as e:
            self.logger.debug(f"Dummy run failed: {e}; scanning modules for hints...")

            # Fall back to scanning the model's modules for hints.
            for module in self.model.modules():
                # If a Linear layer is found, infer the input shape from its in_features.
                if hasattr(module, 'in_features'):
                    shape = (module.in_features,)
                    self.logger.debug(f"Inferred shape from Linear layer: {shape}")
                    return torch.Size(shape)
                # If a Conv2d layer is found, assume a common image shape.
                elif isinstance(module, nn.Conv2d):
                    shape = (3, 224, 224)
                    self.logger.debug(f"Inferred default shape from Conv2d: {shape}")
                    return torch.Size(shape)

        self.logger.warning("Failed to detect input shape; please specify one via preprocessor or config.")
        return None

    def _warmup(self):
        """
        Run several forward passes to warm up the model (especially on GPU).
        """
        if self.input_shape is None:
            self.logger.warning("No input shape detected; skipping warmup.")
            return

        self.logger.info(f"Starting warmup with {self.config.warmup_runs} iterations...")
        # Build the dummy input batch using the configured batch size and detected input shape.
        batch_shape = (self.config.batch_size,) + tuple(self.input_shape)
        dummy_input = torch.randn(*batch_shape, device=self.device)

        with torch.no_grad():
            for i in range(self.config.warmup_runs):
                _ = self.model(dummy_input)
                self.logger.debug(f"Warmup iteration {i + 1}/{self.config.warmup_runs} completed.")

        # If running on GPU, synchronize to ensure all operations have finished.
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.logger.info("Warmup completed.")

    def _compile_trt_model(self) -> Union[torch.nn.Module, tuple]:
        """
        Compiles the PyTorch model with TensorRT.
        Supports multi-input models if `trt_input_shape` is provided in the config.
        
        Returns:
            A tuple (compiled_model, compile_time).
        """
        import torch_tensorrt  # Import here to avoid dependency if TRT is not used.

        # Build a list of TensorRT input specifications.
        if self.config.trt_input_shape is None:
            # Single-input model: use the detected or configured input shape.
            inputs = [
                torch_tensorrt.Input(
                    min_shape=self.input_shape,
                    opt_shape=self.input_shape,
                    max_shape=self.input_shape,
                    dtype=torch.float32
                )
            ]
        else:
            inputs = []
            for shape_tuple in self.config.trt_input_shape:
                try:
                    min_shape, opt_shape, max_shape = shape_tuple
                except Exception as e:
                    raise ValueError(
                        "Each TensorRT input shape must be a tuple: (min_shape, opt_shape, max_shape)."
                    ) from e

                inputs.append(
                    torch_tensorrt.Input(
                        min_shape=torch.Size(min_shape),
                        opt_shape=torch.Size(opt_shape),
                        max_shape=torch.Size(max_shape),
                        dtype=torch.float32
                    )
                )

        start_time = time.time()
        # Compile the model, enabling FP16 if desired.
        compiled_model = torch_tensorrt.compile(
            self.model,
            inputs=inputs,
            enabled_precisions={torch.half}  # Enable FP16.
        )
        compile_time = time.time() - start_time
        return compiled_model, compile_time

    def dynamic_batch_size(self, sample_tensor: torch.Tensor) -> int:
        """
        Optimized dynamic batch size computation that determines a batch size based on:
        - Available GPU memory (if on CUDA),
        - The current request queue length,
        - A PID controller adjustment.
        
        Improvements:
        - Cache the current queue size to avoid repeated calls.
        - Use list comprehension to compute memory-based batch sizes.
        - Compute delta time (dt) more cleanly with a default value.
        """
        # Cache current queue size once.
        current_queue = self.request_queue.qsize()
        
        # --- Memory-Based Computation ---
        if self.device.type == 'cuda':
            # Compute the memory footprint of one sample once.
            sample_bytes = sample_tensor.element_size() * sample_tensor.numel()
            # Compute the maximum number of samples that can fit on each GPU.
            possible_batches = [
                max(
                    (torch.cuda.get_device_properties(device_id).total_memory -
                    torch.cuda.memory_allocated(device_id)) // sample_bytes,
                    1
                )
                for device_id in self.config.device_ids
            ]
            memory_based = min(possible_batches)
        else:
            memory_based = self.config.max_batch_size

        # --- Queue-Based Computation ---
        # Add +1 to account for the current incoming request.
        queue_based = min(
            self.config.max_batch_size,
            max(self.config.min_batch_size, current_queue + 1)
        )

        # Base batch size is the lesser of memory- and queue-based sizes.
        base_batch_size = min(memory_based, queue_based)

        # --- PID Controller Adjustment ---
        # Use the cached queue length to compute utilization.
        utilization = (current_queue / self.config.queue_size) * 100.0

        # Compute dt (delta time) since the last autoscale check.
        now = time.monotonic()
        last_time = getattr(self, "_last_autoscale_time", None)
        dt = now - last_time if last_time is not None else 1.0
        self._last_autoscale_time = now

        # Get the PID adjustment.
        pid_adjustment = self.config.pid_controller.update(utilization, dt)

        # Adjust the base batch size with the PID controller output.
        new_batch = int(round(base_batch_size + pid_adjustment))
        new_batch = max(self.config.min_batch_size, min(new_batch, self.config.max_batch_size))

        if self.config.debug_mode:
            self.logger.debug(
                f"Optimized dynamic batch size computed: memory_based={memory_based}, "
                f"queue_based={queue_based}, utilization={utilization:.1f}%, "
                f"pid_adjustment={pid_adjustment:.2f} -> new_batch={new_batch}"
            )

        return new_batch

    # --- Batch Processing and Inference ---
    async def _process_batches(self):
        """
        Optimized asynchronous batching loop that:
        - Minimizes waiting by quickly draining the queue.
        - Yields control to allow more items to accumulate if near the desired batch size.
        - Moves the input tensor to the proper device in a non-blocking way.
        """
        self.logger.info("Starting optimized batch processing loop...")
        
        while True:
            try:
                # Wait for the first request with a short timeout.
                first_item = await asyncio.wait_for(
                    self.request_queue.get(), timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                continue  # No request arrived within the timeout period.

            batch_items = [first_item]
            
            # Compute the desired batch size.
            if self.config.enable_dynamic_batching:
                try:
                    desired_batch_size = self.dynamic_batch_size(first_item.input)
                except Exception as e:
                    self.logger.error("Error computing dynamic batch size", exc_info=True)
                    desired_batch_size = self.config.batch_size
            else:
                desired_batch_size = self.config.batch_size

            # Drain the queue immediately for available requests.
            while len(batch_items) < desired_batch_size:
                try:
                    next_item = self.request_queue.get_nowait()
                    batch_items.append(next_item)
                except asyncio.QueueEmpty:
                    break

            # Yield control to catch any straggling requests, but only once.
            if len(batch_items) < desired_batch_size:
                await asyncio.sleep(0)
                while len(batch_items) < desired_batch_size:
                    try:
                        next_item = self.request_queue.get_nowait()
                        batch_items.append(next_item)
                    except asyncio.QueueEmpty:
                        break

            # Extract inputs and futures from the collected batch.
            inputs = [item.input for item in batch_items]
            futures = [item.future for item in batch_items]

            try:
                # Stack the inputs into a single tensor.
                batch_tensor = torch.stack(inputs)
                # Transfer tensor to the target device if not already there.
                if batch_tensor.device != self.device:
                    batch_tensor = batch_tensor.to(self.device, non_blocking=True)
                
                # Offload the inference to the executor.
                loop = asyncio.get_running_loop()
                outputs = await loop.run_in_executor(self.executor, self._infer_batch, batch_tensor)
                
                # Postprocess each output.
                results = []
                for output in outputs:
                    result = self.postprocessor(output)
                    # Ensure a consistent batch dimension.
                    if result.dim() == 1:
                        result = result.unsqueeze(0)
                    results.append(result)

                # Set the result for each corresponding future.
                for fut, res in zip(futures, results):
                    if not fut.done():
                        fut.set_result(res)
                self.logger.debug(f"Processed batch of {len(batch_items)} items.")
            except Exception as e:
                self.logger.error("Batch processing error", exc_info=True)
                # Propagate the exception to all pending futures.
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)

 
    def _infer_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform optimized inference on a batch of inputs with proper precision context.
        
        Features:
        - Device-aware mixed precision using autocast.
        - Inference-mode optimization.
        - Tensor compatibility checks.
        
        Args:
            batch_tensor: Input tensor of shape [batch_size, ...]
            
        Returns:
            Model outputs tensor.
        
        Raises:
            RuntimeError: If tensor device mismatch occurs.
        """
        # Validate device compatibility.
        if batch_tensor.device != self.device:
            raise RuntimeError(
                f"Input tensor device {batch_tensor.device} doesn't match engine device {self.device}"
            )
        
        # Determine whether to enable autocasting for mixed precision.
        autocast_enabled = self.use_fp16 and self.device.type == "cuda"
        
        # Use inference mode and autocast for optimized inference.
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type, enabled=autocast_enabled):
            return self.model(batch_tensor)

    # --- Autoscaling Batch Size using PID ---
    async def _autoscale(self):
        """
        Optimized zero-autoscaling: immediately adjust batch size based on the current queue load.
        - When the queue is empty or below a low threshold, set the batch size to the minimum.
        - When the queue is above a high threshold, set it to the maximum.
        - For in-between utilization, linearly interpolate the batch size.
        """
        self.logger.info("Starting zero-autoscaling task...")
        cfg = self.config

        while True:
            try:
                # Sleep for the defined autoscale interval.
                await asyncio.sleep(cfg.autoscale_interval)
                
                # Get the current queue size and compute utilization.
                current_queue = self.request_queue.qsize()
                utilization = (current_queue / cfg.queue_size) * 100.0

                # --- Zero-autoscaling logic ---
                # Case 1: Low load (or no load) — scale down immediately.
                if current_queue == 0 or utilization < cfg.queue_size_threshold_low:
                    if cfg.batch_size != cfg.min_batch_size:
                        self.logger.info(
                            f"Queue utilization low ({utilization:.1f}%), setting batch size to minimum: {cfg.min_batch_size}"
                        )
                        cfg.batch_size = cfg.min_batch_size
                    continue

                # Case 2: High load — scale up immediately.
                elif utilization > cfg.queue_size_threshold_high:
                    if cfg.batch_size != cfg.max_batch_size:
                        self.logger.info(
                            f"Queue utilization high ({utilization:.1f}%), setting batch size to maximum: {cfg.max_batch_size}"
                        )
                        cfg.batch_size = cfg.max_batch_size
                    continue

                # Case 3: Moderate load — linearly interpolate between min and max batch sizes.
                else:
                    # Compute the scaling ratio between the low and high thresholds.
                    scale_ratio = ((utilization - cfg.queue_size_threshold_low) /
                                (cfg.queue_size_threshold_high - cfg.queue_size_threshold_low))
                    new_batch_size = int(cfg.min_batch_size + scale_ratio * (cfg.max_batch_size - cfg.min_batch_size))
                    new_batch_size = max(cfg.min_batch_size, min(new_batch_size, cfg.max_batch_size))

                    if new_batch_size != cfg.batch_size:
                        self.logger.info(
                            f"Queue utilization moderate ({utilization:.1f}%), adjusting batch size to {new_batch_size}"
                        )
                        cfg.batch_size = new_batch_size

            except asyncio.CancelledError:
                self.logger.info("Autoscale task cancelled.")
                break
            except Exception as e:
                self.logger.error("Error in autoscale task", exc_info=True)


    # --- Adversarial Guard: Test-Time Augmentation for Inference ---
    def _guard_sample(self, processed: torch.Tensor) -> bool:
        """
        Optimized guard sample: Applies test-time augmentations in a vectorized manner.
        
        Instead of performing a loop with multiple forward passes (one per augmentation),
        all augmentations are generated and processed in a single batched forward pass,
        thereby reducing inference overhead and improving latency.
        """
        # Retrieve configuration parameters.
        cfg = self.config
        num_augmentations = cfg.guard_num_augmentations
        noise_level_range = cfg.guard_noise_level_range
        dropout_rate = cfg.guard_dropout_rate
        flip_prob = cfg.guard_flip_prob
        confidence_threshold = cfg.guard_confidence_threshold
        variance_threshold = cfg.guard_variance_threshold
        input_range = cfg.guard_input_range

        # Ensure the sample has a batch dimension.
        # If processed tensor has shape (*input_shape), add a batch dimension.
        if processed.dim() == len(self.input_shape):
            sample = processed.unsqueeze(0)
        else:
            sample = processed
        sample = sample.to(self.device)

        # Create a batch by repeating the sample for each augmentation.
        # For a sample with shape [1, ...], this creates a tensor of shape [num_augmentations, ...].
        batch = sample.repeat(num_augmentations, *(1 for _ in range(sample.dim() - 1)))

        # ---- Vectorized Augmentations ----

        # 1. Random Gaussian Noise:
        #    Generate one noise level per augmentation and broadcast it over the sample dimensions.
        noise_levels = torch.empty(num_augmentations, device=self.device).uniform_(*noise_level_range)
        noise_levels = noise_levels.view(num_augmentations, *([1] * (batch.dim() - 1)))
        noise = torch.randn_like(batch) * noise_levels
        batch = batch + noise

        # 2. Random Dropout:
        #    With 30% chance per augmentation, apply dropout to randomly zero-out elements.
        dropout_flags = torch.rand(num_augmentations, device=self.device) < 0.3
        if dropout_flags.any():
            dropout_indices = dropout_flags.nonzero(as_tuple=True)[0]
            # For the selected augmentations, create a dropout mask.
            mask = (torch.rand_like(batch[dropout_indices]) >= dropout_rate).float()
            batch[dropout_indices] = batch[dropout_indices] * mask

        # 3. Random Horizontal Flip:
        #    For image-like data (assumed if input_shape has at least 2 dims), decide per augmentation.
        if len(self.input_shape) >= 2:
            flip_flags = torch.rand(num_augmentations, device=self.device) < flip_prob
            if flip_flags.any():
                flip_indices = flip_flags.nonzero(as_tuple=True)[0]
                batch[flip_indices] = torch.flip(batch[flip_indices], dims=[-1])

        # Clamp the augmented batch to the valid input range.
        batch = torch.clamp(batch, *input_range)

        # ---- Batched Inference ----
        with torch.no_grad():
            if self.use_fp16 and self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    preds = self.model(batch)
            else:
                preds = self.model(batch)

        # Compute probabilities via softmax.
        preds_probs = torch.softmax(preds, dim=1)

        # Aggregate predictions by averaging over the augmentations.
        aggregated = preds_probs.mean(dim=0)

        # Compute guard metrics: highest average probability and variance across augmentations.
        confidence = aggregated.max().item()
        max_probs = preds_probs.max(dim=1)[0]
        variance = max_probs.std().item()

        if cfg.debug_mode:
            self.logger.debug(f"Guard metrics - Confidence: {confidence:.3f}, Variance: {variance:.3f}")

        # Return True if the sample passes the guard.
        return confidence >= confidence_threshold and variance <= variance_threshold

    # --- Public Inference Interfaces ---
    async def run_inference_async(self, input_data: Any, priority: int = 0) -> Any:
        """
        Processes an inference request asynchronously.
        The input is preprocessed and passed through the guard in a thread,
        then queued for batch inference if the guard check passes.
        """
        loop = asyncio.get_running_loop()
        try:
            # Preprocess the input in a thread via the executor.
            processed = await loop.run_in_executor(self.guard_executor, self.preprocessor, input_data)
            if not isinstance(processed, torch.Tensor):
                processed = torch.as_tensor(processed, dtype=torch.float32)

            # Run the guard check concurrently in the executor.
            is_safe = await loop.run_in_executor(self.guard_executor, self._guard_sample, processed)
            if not is_safe:
                self.logger.warning("Guard: Potential adversarial input detected. Using default response.")
                num_classes = self.config.num_classes
                default_probs = torch.ones(1, num_classes, device=self.device) / num_classes
                # Remove the extra batch dimension before returning.
                return default_probs.squeeze(0)

            # Create a future for the response and queue the request.
            future = loop.create_future()
            await self.request_queue.put(RequestItem(input=processed, future=future, priority=priority))
            self.logger.debug(f"Queued request. Queue size: {self.request_queue.qsize()}")
            return await future

        except Exception as e:
            self.logger.error("Inference failed", exc_info=True)
            raise

    def run_batch_inference(self, batch: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Optimized synchronous inference on a batch of preprocessed inputs.
        
        Improvements:
        - Minimizes redundant tensor conversions.
        - Ensures the batch tensor is contiguous.
        - Transfers the batch to the correct device using non-blocking calls when appropriate.
        - Uses torch.inference_mode for maximum inference performance.
        
        Args:
            batch (Union[torch.Tensor, List[torch.Tensor]]): A batch of inputs.
            
        Returns:
            torch.Tensor: The model output.
        
        Raises:
            TypeError: If the batch is not a tensor or a list of tensors.
            ValueError: If stacking fails due to shape mismatches.
        """
        # If a list is provided, convert/stack elements into a single tensor.
        if isinstance(batch, list):
            try:
                # Convert non-tensor items on-the-fly and collect them.
                tensor_list = [
                    x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
                    for x in batch
                ]
                batch = torch.stack(tensor_list, dim=0)
            except Exception as e:
                raise ValueError(
                    "Failed to stack the input list into a tensor. "
                    "Ensure all elements are tensors or convertible to tensors with matching dimensions."
                ) from e

        # Confirm that batch is now a tensor.
        if not isinstance(batch, torch.Tensor):
            raise TypeError("The 'batch' must be a torch.Tensor or a list of tensors.")

        # Ensure the tensor is contiguous in memory.
        batch = batch.contiguous()

        # Transfer the batch to the correct device (non-blocking if on CUDA).
        if batch.device != self.device:
            batch = batch.to(self.device, non_blocking=True)

        # Perform inference using torch.inference_mode for optimal performance.
        with torch.inference_mode():
            output = self.model(batch)

        return output


    def profile_inference(self, inputs: Any) -> Dict[str, float]:
        # Convert inputs to a tensor without copying if already a tensor.
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32)

        # Ensure the input has a proper batch dimension if an input shape is defined.
        if self.input_shape is not None:
            expected_dim = len(self.input_shape) + 1  # +1 for batch dimension
            # If the input is missing a batch dimension, add one.
            if inputs.dim() == len(self.input_shape):
                inputs = inputs.unsqueeze(0)
            elif inputs.dim() != expected_dim:
                raise ValueError(
                    f"Input shape {inputs.shape} doesn't match expected shape with batch dimension."
                )

        # Run the inference while profiling key parts of the pipeline.
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True) as prof:
            with record_function("preprocess"):
                batch = self.preprocessor(inputs)
                if not isinstance(batch, torch.Tensor):
                    batch = torch.as_tensor(batch, dtype=torch.float32)
                batch = batch.to(self.device)

            with record_function("inference"):
                with torch.no_grad():
                    inference_output = self.model(batch)

            with record_function("postprocess"):
                _ = self.postprocessor(inference_output)

        # Gather profiling metrics for each recorded event.
        metrics = {}
        # Convert the key averages into a dict for easier lookup.
        stats = {stat.key: stat for stat in prof.key_averages()}
        for evt in ["preprocess", "inference", "postprocess"]:
            evt_stat = stats.get(evt)
            metrics[f"{evt}_ms"] = evt_stat.cpu_time_total / 1000.0 if evt_stat else 0.0

        metrics["total_ms"] = sum(metrics[step] for step in ["preprocess_ms", "inference_ms", "postprocess_ms"])

        self.logger.debug(
            "Inference profile results:\n" +
            "\n".join([f"{k}: {v:.2f} ms" for k, v in metrics.items()])
        )
        
        return metrics

    async def cleanup(self):
        self.logger.info("Cleaning up inference engine resources...")

        # Cancel the tasks if they exist.
        if self.batch_processor_task is not None:
            self.batch_processor_task.cancel()
        if self.autoscale_task is not None:
            self.autoscale_task.cancel()

        # Await the cancellation to ensure tasks are properly cleaned up.
        await asyncio.gather(
            *(t for t in [self.batch_processor_task, self.autoscale_task] if t is not None),
            return_exceptions=True
        )

        # Shut down the executor.
        self.executor.shutdown(wait=False)

        # Clear CUDA caches if needed.
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        self.logger.info("Cleanup completed.")

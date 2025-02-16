import asyncio
import logging
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union
import sys
import os
import numpy as np
# Add the parent directory to sys.path so that imports like "core.engine" work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

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
# Custom Exception
################################################################################
class ModelPreparationError(Exception):
    """Custom exception for model preparation errors."""
    pass


################################################################################
# Request Item
################################################################################
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
        # PriorityQueue uses __lt__ to determine ordering
        return self.priority < other.priority


################################################################################
# Inference Engine
################################################################################
class InferenceEngine:
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
        # ----------------------------------------------------------------------
        # Basic Setup
        # ----------------------------------------------------------------------
        self._shutdown_event = asyncio.Event()  # signals all tasks to stop

        self.config = config or EngineConfig(debug_mode=True)
        self.config.configure_logging()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.device = self._auto_select_device(device)
        self.use_fp16 = use_fp16 and (self.device.type == 'cuda')
        self.use_tensorrt = use_tensorrt
        self.model = self._prepare_model(model)

        # ----------------------------------------------------------------------
        # Optional TensorRT Compilation
        # ----------------------------------------------------------------------
        if self.use_tensorrt and self.device.type == "cuda":
            try:
                self.logger.info("Compiling model with TensorRT optimizations...")
                self.model, compile_time = self._compile_trt_model()
                self.logger.info(f"TensorRT model compiled in {compile_time:.2f}s")
            except Exception as e:
                self.logger.error(f"TensorRT compilation failed: {e}", exc_info=True)

        # ----------------------------------------------------------------------
        # Pre/Post Processors
        # ----------------------------------------------------------------------
        self.preprocessor = preprocessor if preprocessor is not None else (lambda x: x)
        self.postprocessor = postprocessor if postprocessor is not None else (lambda x: x)

        # Detect Input Shape
        self.input_shape = self._detect_input_shape()

        # ----------------------------------------------------------------------
        # Thread/Process Pool Executors
        # ----------------------------------------------------------------------
        if self.config.executor_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=self.config.num_workers)
            self.guard_executor = ProcessPoolExecutor(max_workers=self.config.num_workers)
            self.inference_executor = ProcessPoolExecutor(max_workers=self.config.num_workers)
        else:
            self.executor = DaemonThreadPoolExecutor(max_workers=self.config.num_workers)
            self.guard_executor = DaemonThreadPoolExecutor(max_workers=self.config.num_workers)
            self.inference_executor = DaemonThreadPoolExecutor(max_workers=self.config.num_workers)

        # Request Queue (PriorityQueue)
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=self.config.queue_size)

        # Start background tasks if async mode is on
        if getattr(self.config, "async_mode", True):
            self.batch_processor_task = asyncio.create_task(self._process_batches())
            self.autoscale_task = asyncio.create_task(self._autoscale())
        else:
            self.batch_processor_task = None
            self.autoscale_task = None

        # Track batch processing times for metrics
        self._batch_processing_times: List[float] = []

        # Warmup
        self._warmup()
        self._log_device_info()

    # --------------------------------------------------------------------------
    # Internal Setup Methods
    # --------------------------------------------------------------------------
    def _auto_select_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        model.eval()
        return model

    def _compile_trt_model(self):
        """Compile with TensorRT if available."""
        import torch_tensorrt

        # Build input specs
        if self.config.trt_input_shape is None:
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
                if not isinstance(shape_tuple, (tuple, list)) or len(shape_tuple) != 3:
                    raise ValueError(
                        "Each TensorRT input shape must be a tuple of (min_shape, opt_shape, max_shape)."
                    )
                min_shape, opt_shape, max_shape = shape_tuple
                inputs.append(
                    torch_tensorrt.Input(
                        min_shape=torch.Size(min_shape),
                        opt_shape=torch.Size(opt_shape),
                        max_shape=torch.Size(max_shape),
                        dtype=torch.float32
                    )
                )

        start_time = time.time()
        compiled_model = torch_tensorrt.compile(
            self.model,
            inputs=inputs,
            enabled_precisions={torch.half}  # FP16
        )
        compile_time = time.time() - start_time
        return compiled_model, compile_time

    def _detect_input_shape(self) -> Optional[torch.Size]:
        """
        Attempt to detect the model input shape with a dummy run, otherwise guess.
        """
        self.logger.debug("Detecting input shape via dummy run...")
        dummy_input = torch.randn(1, 10, device=self.device)
        try:
            with torch.no_grad():
                _ = self.model(dummy_input)
            return dummy_input.shape[1:]
        except Exception as e:
            self.logger.debug(f"Dummy run failed: {e}. Attempting shape scan.")
            # Fallback: scan the model for common modules
            for module in self.model.modules():
                if hasattr(module, 'in_features'):
                    return torch.Size([module.in_features])
                elif isinstance(module, nn.Conv2d):
                    return torch.Size([3, 224, 224])
            self.logger.warning("Failed to detect input shape.")
            return None

    def _warmup(self):
        """
        Warm up the model on a few dummy batches to optimize GPU execution.
        """
        if self.input_shape is None:
            self.logger.warning("No input shape detected; skipping warmup.")
            return

        self.logger.info(f"Warming up model for {self.config.warmup_runs} iterations...")
        dummy_input = torch.randn(
            (self.config.batch_size,) + self.input_shape,
            device=self.device
        )
        with torch.no_grad():
            for i in range(self.config.warmup_runs):
                _ = self.model(dummy_input)
                self.logger.debug(f"Warmup iteration {i+1}/{self.config.warmup_runs} complete.")
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.logger.info("Warmup completed.")

    def _log_device_info(self):
        """Log info about the selected device."""
        self.logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            idx = self.device.index if self.device.index is not None else 0
            device_name = torch.cuda.get_device_name(idx)
            device_props = torch.cuda.get_device_properties(idx)
            total_mem_gb = device_props.total_memory / 1e9
            self.logger.info(f"CUDA Device: {device_name}, Total Memory: {total_mem_gb:.2f} GB")

    # --------------------------------------------------------------------------
    # Batch Processing Loop
    # --------------------------------------------------------------------------
    async def _process_batches(self):
        """
        Continuously processes available requests in batches until shutdown.
        If no requests come in within `config.timeout`, it logs a timeout message
        and repeats until shutdown.
        """
        while not self._shutdown_event.is_set():
            try:
                # Wait for first item with a timeout
                first_item = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                self.logger.info("No requests received within timeout.")
                continue
            if first_item is None:
                continue

            batch_items = [first_item]
            batch_start_time = time.monotonic()

            # Small sleep to gather more requests
            await asyncio.sleep(self.config.batch_wait_timeout)

            # Drain the queue quickly
            while True:
                try:
                    next_item = self.request_queue.get_nowait()
                    batch_items.append(next_item)
                except asyncio.QueueEmpty:
                    break

            # Prepare: extract inputs and futures from batch items.
            inputs = [item.input for item in batch_items]
            futures = [item.future for item in batch_items]

            # Input checker: ensure all inputs are torch.Tensors and share the same shape.
            for idx, tensor in enumerate(inputs):
                if not isinstance(tensor, torch.Tensor):
                    error_msg = f"Input at index {idx} is not a torch.Tensor."
                    self.logger.error(error_msg)
                    for fut in futures:
                        if not fut.done():
                            fut.set_exception(ValueError(error_msg))
                    continue  # Skip processing this batch

            base_shape = inputs[0].shape
            for idx, tensor in enumerate(inputs):
                if tensor.shape != base_shape:
                    error_msg = f"Input at index {idx} does not match the expected shape {base_shape}."
                    self.logger.error(error_msg)
                    for fut in futures:
                        if not fut.done():
                            fut.set_exception(ValueError(error_msg))
                    continue  # Skip processing this batch

            try:
                # Stack and reshape inputs so that the batch tensor shape is (batch_size, *input_shape)
                stacked_tensor = torch.stack(inputs)
                batch_tensor = torch.reshape(stacked_tensor, (len(inputs), *base_shape))

                # Move tensor to device if needed.
                if batch_tensor.device != self.device:
                    batch_tensor = batch_tensor.to(self.device, non_blocking=True)

                loop = asyncio.get_running_loop()
                outputs = await loop.run_in_executor(
                    self.executor,
                    self._infer_batch,
                    batch_tensor
                )

                results = []
                for output in outputs:
                    # Post-process each item in the batch
                    res = self.postprocessor(output)
                    # Ensure it keeps a consistent shape (2D).
                    if res.dim() == 1:
                        res = res.unsqueeze(0)
                    results.append(res)

                for fut, res in zip(futures, results):
                    if not fut.done():
                        fut.set_result(res)

                batch_time = time.monotonic() - batch_start_time
                self._batch_processing_times.append(batch_time)
                self.logger.debug(
                    f"Processed batch of {len(batch_items)} items in {batch_time:.3f}s."
                )
            except Exception as e:
                self.logger.error("Error during batch processing", exc_info=True)
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)


    # --------------------------------------------------------------------------
    # Autoscaling via a PID Controller
    # --------------------------------------------------------------------------
    async def _autoscale(self):
        """
        Continuously adjusts self.config.batch_size based on queue utilization
        using a PID controller.
        """
        self.logger.info("Starting PID-based autoscaling task...")
        last_time = time.monotonic()
        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(self.config.autoscale_interval)
                now = time.monotonic()
                dt = now - last_time
                last_time = now

                current_queue = self.request_queue.qsize()
                utilization = (current_queue / self.config.queue_size) * 100.0

                # Check user-defined PID:
                adjustment = self.config.pid_controller.update(utilization, dt)
                new_batch_size = int(round(self.config.batch_size + adjustment))
                new_batch_size = max(
                    self.config.min_batch_size,
                    min(new_batch_size, self.config.max_batch_size)
                )

                if new_batch_size != self.config.batch_size:
                    self.logger.info(
                        f"PID autoscale: utilization={utilization:.1f}% -> "
                        f"changing batch size from {self.config.batch_size} to {new_batch_size}"
                    )
                    self.config.batch_size = new_batch_size
        except asyncio.CancelledError:
            self.logger.info("Autoscale task cancelled.")
            raise

    # --------------------------------------------------------------------------
    # Adversarial Guard
    # --------------------------------------------------------------------------
    def _guard_sample(self, processed: torch.Tensor) -> bool:
        """
        Applies test-time augmentations to check if the sample is adversarial.
        Returns True if the sample passes the guard, False otherwise.
        """
        if not self.config.guard_enabled:
            return True  # If guard not enabled, always pass

        cfg = self.config
        num_augmentations = cfg.guard_num_augmentations
        noise_level_range = cfg.guard_noise_level_range
        dropout_rate = cfg.guard_dropout_rate
        flip_prob = cfg.guard_flip_prob
        confidence_threshold = cfg.guard_confidence_threshold
        variance_threshold = cfg.guard_variance_threshold
        input_range = cfg.guard_input_range

        # Ensure batch dimension
        if processed.dim() == len(self.input_shape):
            sample = processed.unsqueeze(0)
        else:
            sample = processed
        sample = sample.to(self.device)

        # Create augmented batch
        batch = sample.repeat(num_augmentations, *(1 for _ in range(sample.dim() - 1)))

        # 1. Noise
        if "noise" in cfg.guard_augmentation_types:
            noise_levels = torch.empty(num_augmentations, device=self.device).uniform_(
                *noise_level_range
            )
            noise_levels = noise_levels.view(
                num_augmentations, *([1] * (batch.dim() - 1))
            )
            noise = torch.randn_like(batch) * noise_levels
            batch = batch + noise

        # 2. Dropout
        if "dropout" in cfg.guard_augmentation_types:
            dropout_flags = torch.rand(num_augmentations, device=self.device) < 0.3
            if dropout_flags.any():
                dropout_indices = dropout_flags.nonzero(as_tuple=True)[0]
                mask = (torch.rand_like(batch[dropout_indices]) >= dropout_rate).float()
                batch[dropout_indices] *= mask

        # 3. Flip
        # Typically for images, only if shape >= 2D
        if "flip" in cfg.guard_augmentation_types and len(self.input_shape) >= 2:
            flip_flags = torch.rand(num_augmentations, device=self.device) < flip_prob
            if flip_flags.any():
                flip_indices = flip_flags.nonzero(as_tuple=True)[0]
                batch[flip_indices] = torch.flip(batch[flip_indices], dims=[-1])

        # Clamp to valid range
        batch = torch.clamp(batch, min=input_range[0], max=input_range[1])

        with torch.no_grad():
            # Evaluate
            if self.use_fp16 and self.device.type == "cuda":
                with torch.amp.autocast(device_type=self.device.type, enabled=True):
                    preds = self.model(batch)
            else:
                preds = self.model(batch)

        # Softmax, aggregate
        preds_probs = torch.softmax(preds, dim=1)
        aggregated = preds_probs.mean(dim=0)
        confidence = aggregated.max().item()

        # Variance across augmentations
        max_probs = preds_probs.max(dim=1)[0]
        variance = max_probs.std().item()

        if self.config.debug_mode:
            self.logger.debug(
                f"Guard metrics -> confidence={confidence:.3f}, variance={variance:.3f}"
            )

        # Pass if confidence >= threshold and variance <= threshold
        return (confidence >= confidence_threshold) and (variance <= variance_threshold)

    # --------------------------------------------------------------------------
    # Public Methods
    # --------------------------------------------------------------------------
    async def run_inference_async(self, input_data: Any, priority: int = 0) -> torch.Tensor:
        """
        Asynchronously process an inference request. If guard fails,
        returns a default uniform distribution over classes.
        """
        loop = asyncio.get_running_loop()
        try:
            # Preprocessing
            processed = await loop.run_in_executor(
                self.guard_executor, self.preprocessor, input_data
            )
            if not isinstance(processed, torch.Tensor):
                processed = torch.as_tensor(processed, dtype=torch.float32)

            # Guard
            is_safe = await loop.run_in_executor(
                self.guard_executor, self._guard_sample, processed
            )
            if not is_safe:
                self.logger.warning("Guard triggered: returning default response.")
                num_classes = self.config.num_classes
                default_probs = torch.ones(1, num_classes, device=self.device) / num_classes
                return default_probs.squeeze(0)

            # Enqueue
            future = loop.create_future()
            await self.request_queue.put(RequestItem(processed, future, priority=priority))
            self.logger.debug(f"Queued request. Current queue size: {self.request_queue.qsize()}")

            # If async mode is disabled, process immediately (blocking)
            if not getattr(self.config, "async_mode", True):
                await self._process_batches()

            # Wait for inference result
            return await future
        except Exception as e:
            self.logger.error("Inference failed", exc_info=True)
            raise

    def run_batch_inference(self, batch: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Synchronous inference on a batch of preprocessed inputs.
        """
        if isinstance(batch, list):
            # Stack if list
            try:
                batch = torch.stack(
                    [x if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in batch],
                    dim=0
                )
            except Exception as e:
                raise ValueError(
                    "Cannot stack the provided batch list into a single tensor."
                ) from e

        if not isinstance(batch, torch.Tensor):
            raise TypeError("run_batch_inference expects a torch.Tensor or list of Tensors.")

        if batch.device != self.device:
            batch = batch.to(self.device, non_blocking=True)

        with torch.inference_mode():
            output = self.model(batch)

        return output

    def dynamic_batch_size(self, sample_tensor: torch.Tensor) -> int:
        """
        Decide a batch size based on GPU memory, queue length, and the PID controller.
        """
        current_queue = self.request_queue.qsize()

        if self.device.type == 'cuda':
            sample_bytes = sample_tensor.element_size() * sample_tensor.numel()
            possible_batches = [
                max(
                    (torch.cuda.get_device_properties(dev_id).total_memory -
                     torch.cuda.memory_allocated(dev_id)) // sample_bytes,
                    1
                )
                for dev_id in range(torch.cuda.device_count())
            ]
            memory_based = min(possible_batches) if possible_batches else self.config.max_batch_size
        else:
            memory_based = self.config.max_batch_size

        queue_based = min(
            self.config.max_batch_size,
            max(self.config.min_batch_size, current_queue + 1)
        )
        base_batch_size = min(memory_based, queue_based)

        utilization = (current_queue / self.config.queue_size) * 100.0
        now = time.monotonic()
        last_time = getattr(self, "_last_autoscale_time", None)
        dt = now - last_time if last_time is not None else 1.0
        self._last_autoscale_time = now

        pid_adjustment = self.config.pid_controller.update(utilization, dt)
        new_batch = int(round(base_batch_size + pid_adjustment))
        new_batch = max(self.config.min_batch_size, min(new_batch, self.config.max_batch_size))

        if self.config.debug_mode:
            self.logger.debug(
                f"Dynamic batch size -> memory_based={memory_based}, queue_based={queue_based}, "
                f"util={utilization:.1f}%, pid={pid_adjustment:.2f}, final={new_batch}"
            )
        return new_batch

    def profile_inference(self, inputs: Any) -> Dict[str, float]:
        """
        Profile the inference pipeline steps: preprocess, inference, postprocess.
        """
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32)

        if self.input_shape is not None:
            expected_dim = len(self.input_shape) + 1  # +1 for batch
            if inputs.dim() == len(self.input_shape):
                inputs = inputs.unsqueeze(0)
            elif inputs.dim() != expected_dim:
                raise ValueError(
                    f"Input shape {inputs.shape} doesn't match expected shape with batch dimension."
                )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True
        ) as prof:
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

        stats = {stat.key: stat for stat in prof.key_averages()}
        metrics = {}
        for evt in ["preprocess", "inference", "postprocess"]:
            evt_stat = stats.get(evt)
            metrics[f"{evt}_ms"] = evt_stat.cpu_time_total / 1_000.0 if evt_stat else 0.0

        metrics["total_ms"] = sum(
            metrics[f"{evt}_ms"] for evt in ["preprocess", "inference", "postprocess"]
        )

        self.logger.debug("Inference profile results:\n" + "\n".join([
            f"{k}: {v:.2f} ms" for k, v in metrics.items()
        ]))
        return metrics

    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns basic health/performance metrics.
        """
        avg_batch_time = (
            sum(self._batch_processing_times) / len(self._batch_processing_times)
            if self._batch_processing_times else 0.0
        )
        return {
            "queue_size": self.request_queue.qsize(),
            "average_batch_processing_time": avg_batch_time,
            "total_batches_processed": len(self._batch_processing_times),
        }

    def shutdown_sync(self):
        """
        Synchronously shut down the engine.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None or not loop.is_running():
            # No active loop: create one for cleanup
            new_loop = asyncio.new_event_loop()
            try:
                new_loop.run_until_complete(self.cleanup())
            finally:
                new_loop.close()
        else:
            fut = asyncio.run_coroutine_threadsafe(self.cleanup(), loop)
            fut.result()

    # --------------------------------------------------------------------------
    # Private / Helper
    # --------------------------------------------------------------------------
    def _infer_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Run the model in fp16 or fp32 mode, always reshaping the batch tensor to (-1, *model_input_size)."""
        if batch_tensor.device != self.device:
            raise RuntimeError(
                f"Mismatch: input device={batch_tensor.device}, engine device={self.device}"
            )
        
        # Ensure self.input_shape is defined.
        if not hasattr(self, "input_shape"):
            raise AttributeError("Model input size (self.input_shape) is not defined.")
        
        # Always reshape to (-1, *self.input_shape)
        batch_tensor = batch_tensor.reshape((-1, *self.input_shape))
        
        autocast_enabled = self.use_fp16 and (self.device.type == "cuda")
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type, enabled=autocast_enabled):
            return self.model(batch_tensor)

    # --------------------------------------------------------------------------
    # Cleanup & Async Context Manager
    # --------------------------------------------------------------------------
    async def cleanup(self):
        """
        Gracefully clean up all resources and cancel background tasks.
        """
        self.logger.info("Cleaning up engine resources...")
        self._shutdown_event.set()

        tasks = []
        for task in (self.batch_processor_task, self.autoscale_task):
            if task is not None:
                task.cancel()
                tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                    self.logger.error("Task raised an exception during cleanup", exc_info=res)

        # Shutdown executors
        self.executor.shutdown(wait=True)
        self.guard_executor.shutdown(wait=True)
        self.inference_executor.shutdown(wait=True)

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        self.logger.info("Cleanup completed.")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


################################################################################
# Minimal EngineConfig to match your tests
################################################################################
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
        debug_mode: bool = False,
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
        from .pid import PIDController
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


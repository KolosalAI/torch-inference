import asyncio
import logging
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union
import sys
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

# If your directory structure needs it:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.config import EngineConfig


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
        self.priority = priority  # Lower values = higher priority

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
        device: Optional[Union[str, torch.device, List[Union[str, torch.device]]]] = None,
        preprocessor: Optional[Callable[[Any], Any]] = None,
        postprocessor: Optional[Callable[[Any], Any]] = None,
        use_fp16: bool = False,
        use_tensorrt: bool = False,
        config: Optional["EngineConfig"] = None,
    ):
        """
        An improved, refactored InferenceEngine with optional parallelization.

        Args:
            model:          A PyTorch model (nn.Module).
            device:         Can be a single device (e.g. "cpu", "cuda", torch.device)
                            or a list of devices for DataParallel (e.g. ["cuda:0", "cuda:1"]).
                            If None, automatically picks "cuda" if available, else "cpu".
            preprocessor:   Optional callable to preprocess inputs.
            postprocessor:  Optional callable to postprocess outputs.
            use_fp16:       Enable FP16 autocast if device == "cuda" (or multi-GPU).
            use_tensorrt:   Use TensorRT (experimental) if True and device == "cuda".
            config:         Engine configuration object (EngineConfig). If None, a default is created.
        """
        # ----------------------------------------------------------------------
        # Basic Setup
        # ----------------------------------------------------------------------
        self._shutdown_event = asyncio.Event()  # signals tasks to stop

        self.config = config or EngineConfig(debug_mode=False)  # default
        self.config.configure_logging()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.use_fp16 = use_fp16
        self.use_tensorrt = use_tensorrt

        # Handle device (single or multiple)
        self.devices = self._normalize_devices(device)
        self.primary_device = self.devices[0]  # main device for guard checks, etc.

        # Setup model: including DataParallel if multiple devices
        self.model = self._prepare_model(model)

        # Potentially compile with TensorRT
        if self.use_tensorrt and self.primary_device.type == "cuda":
            self._maybe_compile_tensorrt()

        # Pre/Post Processors
        self.preprocessor = preprocessor if preprocessor else (lambda x: x)
        self.postprocessor = postprocessor if postprocessor else (lambda x: x)

        # Detect Input Shape
        self.input_shape = self._detect_input_shape()

        # Create executors
        self._init_executors()

        # PriorityQueue for requests
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self.config.queue_size
        )

        # Start background tasks if async mode is on
        self.batch_processor_task = None
        self.autoscale_task = None
        if getattr(self.config, "async_mode", True):
            self.batch_processor_task = asyncio.create_task(self._process_batches())
            self.autoscale_task = asyncio.create_task(self._autoscale())

        # Track batch processing times for metrics
        self._batch_processing_times: List[float] = []

        # Warmup
        self._warmup()
        self._log_device_info()

    # --------------------------------------------------------------------------
    # Internal Setup / Configuration
    # --------------------------------------------------------------------------
    def _normalize_devices(
        self,
        device: Optional[Union[str, torch.device, List[Union[str, torch.device]]]]
    ) -> List[torch.device]:
        """
        Converts the device argument into a list of torch.device objects.
        If device is None, picks CUDA if available, else CPU.
        """
        if device is None:
            # auto device selection
            return [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
        elif isinstance(device, (str, torch.device)):
            return [torch.device(device)]
        elif isinstance(device, list):
            # convert each item
            devices = [torch.device(d) for d in device]
            if not devices:
                devices = [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
            return devices
        else:
            raise ValueError("Invalid device argument")

    def _init_executors(self):
        """Creates thread/process executors according to config."""
        if self.config.executor_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=self.config.num_workers)
            self.guard_executor = ProcessPoolExecutor(max_workers=self.config.num_workers)
            self.inference_executor = ProcessPoolExecutor(max_workers=self.config.num_workers)
        else:
            self.executor = DaemonThreadPoolExecutor(max_workers=self.config.num_workers)
            self.guard_executor = DaemonThreadPoolExecutor(max_workers=self.config.num_workers)
            self.inference_executor = DaemonThreadPoolExecutor(max_workers=self.config.num_workers)

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Moves model to device(s). If multiple devices, wrap with DataParallel.
        """
        # If multiple devices, wrap in DataParallel
        if len(self.devices) > 1:
            # Use the first device as the default. DataParallel can gather outputs on the first device
            model.to(self.primary_device)
            model = nn.DataParallel(model, device_ids=self.devices)
        else:
            # single device
            model = model.to(self.primary_device)
        model.eval()
        return model

    def _maybe_compile_tensorrt(self) -> None:
        """
        Attempt to compile the (single-device) model with TensorRT if installed.
        If multiple devices are used (DataParallel), this is not currently supported.
        """
        # TensorRT + DataParallel is tricky. This snippet only supports single-GPU TRT.
        if len(self.devices) > 1:
            self.logger.warning(
                "TensorRT compilation is not supported with multiple devices in DataParallel. Skipping."
            )
            return

        try:
            import torch_tensorrt
        except ImportError:
            self.logger.error("torch_tensorrt is not installed; skipping TRT compilation.")
            return

        # Build input specs
        if self.config.trt_input_shape is None:
            # fallback
            min_opt_max = (
                self.input_shape
                if self.input_shape is not None
                else (1, 10)  # arbitrary
            )
            inputs = [
                torch_tensorrt.Input(
                    min_shape=min_opt_max,
                    opt_shape=min_opt_max,
                    max_shape=min_opt_max,
                    dtype=torch.float32
                )
            ]
        else:
            inputs = []
            for (min_shape, opt_shape, max_shape) in self.config.trt_input_shape:
                inputs.append(
                    torch_tensorrt.Input(
                        min_shape=torch.Size(min_shape),
                        opt_shape=torch.Size(opt_shape),
                        max_shape=torch.Size(max_shape),
                        dtype=torch.float32
                    )
                )

        self.logger.info("Compiling model with TensorRT optimizations...")
        start_time = time.time()
        try:
            compiled_model = torch_tensorrt.compile(
                self.model,
                inputs=inputs,
                enabled_precisions={torch.half} if self.use_fp16 else {torch.float32}
            )
            compile_time = time.time() - start_time
            self.logger.info(f"TensorRT model compiled in {compile_time:.2f}s")
            self.model = compiled_model
        except Exception as e:
            self.logger.error(f"TensorRT compilation failed: {e}", exc_info=True)

    def _detect_input_shape(self) -> Optional[torch.Size]:
        """
        Attempt to detect the model input shape with a dummy run.
        If that fails, fallback to scanning the model for linear/conv modules.
        """
        self.logger.debug("Detecting input shape via dummy run...")
        # Basic guess
        dummy_input = torch.randn(1, 10, device=self.primary_device)
        try:
            with torch.no_grad():
                _ = self.model(dummy_input)
            return dummy_input.shape[1:]
        except Exception as e:
            self.logger.debug(f"Dummy run failed: {e}. Attempting shape scan.")
            # fallback: check common modules
            for module in self.model.modules():
                if hasattr(module, 'in_features'):
                    return torch.Size([module.in_features])
                elif isinstance(module, nn.Conv2d):
                    return torch.Size([3, 224, 224])
            self.logger.warning("Failed to detect input shape.")
            return None

    def _warmup(self):
        """Warm up the model on a few dummy batches to optimize GPU execution."""
        if self.input_shape is None:
            self.logger.warning("No input shape detected; skipping warmup.")
            return
        if self.config.warmup_runs <= 0:
            return

        self.logger.info(f"Warming up model for {self.config.warmup_runs} iterations...")
        dummy_input = torch.randn(
            (self.config.batch_size,) + self.input_shape,
            device=self.primary_device
        )
        with torch.no_grad():
            for i in range(self.config.warmup_runs):
                _ = self.model(dummy_input)
                if self.config.debug_mode:
                    self.logger.debug(f"Warmup iteration {i+1}/{self.config.warmup_runs} complete.")
        if self.primary_device.type == 'cuda':
            torch.cuda.synchronize()
        self.logger.info("Warmup completed.")

    def _log_device_info(self):
        """Log info about the selected device(s)."""
        if len(self.devices) == 1:
            self.logger.info(f"Using single device: {self.primary_device}")
            if self.primary_device.type == 'cuda':
                idx = self.primary_device.index if self.primary_device.index is not None else 0
                device_name = torch.cuda.get_device_name(idx)
                device_props = torch.cuda.get_device_properties(idx)
                total_mem_gb = device_props.total_memory / 1e9
                self.logger.info(
                    f"CUDA Device: {device_name}, Total Memory: {total_mem_gb:.2f} GB"
                )
        else:
            self.logger.info(f"Using DataParallel across devices: {self.devices}")

    # --------------------------------------------------------------------------
    # Batch Processing Loop (async)
    # --------------------------------------------------------------------------
    async def _process_batches(self):
        """
        Continuously processes available requests in batches until shutdown.
        If no requests arrive within `config.timeout`, logs a timeout message
        and continues until shutdown.
        """
        while not self._shutdown_event.is_set():
            try:
                # Wait for the first item with a timeout
                first_item = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                self.logger.info("No requests received within timeout.")
                continue
            if first_item is None:
                continue

            # Start building the batch
            batch_items = [first_item]
            batch_start_time = time.monotonic()

            # Small delay to gather more requests
            await asyncio.sleep(self.config.batch_wait_timeout)

            # Drain the queue (non-blocking)
            while True:
                try:
                    next_item = self.request_queue.get_nowait()
                    batch_items.append(next_item)
                except asyncio.QueueEmpty:
                    break

            # Prepare batch
            inputs = [item.input for item in batch_items]
            futures = [item.future for item in batch_items]
            success = self._validate_batch_inputs(inputs, futures)
            if not success:
                continue  # skip if input shape mismatch or other error

            # Run inference in executor
            try:
                # stack
                stacked_tensor = torch.stack(inputs)
                if stacked_tensor.device != self.primary_device:
                    stacked_tensor = stacked_tensor.to(self.primary_device, non_blocking=True)

                loop = asyncio.get_running_loop()
                outputs = await loop.run_in_executor(
                    self.inference_executor,
                    self._infer_batch,
                    stacked_tensor
                )

                # Postprocess each output
                # If outputs is a single Tensor with shape (batch_size, num_classes, ...)
                # then we split along dim=0
                if outputs.dim() > 1 and outputs.size(0) == len(inputs):
                    splitted_outputs = list(torch.split(outputs, 1, dim=0))
                    splitted_outputs = [s.squeeze(0) for s in splitted_outputs]
                else:
                    # fallback
                    splitted_outputs = [outputs] * len(inputs)

                for fut, res in zip(futures, splitted_outputs):
                    if not fut.done():
                        processed_res = self.postprocessor(res)
                        fut.set_result(processed_res)

                batch_time = time.monotonic() - batch_start_time
                self._batch_processing_times.append(batch_time)

                if self.config.debug_mode:
                    self.logger.debug(
                        f"Processed batch of {len(batch_items)} items in {batch_time:.3f}s."
                    )
            except Exception as e:
                self.logger.error("Error during batch processing", exc_info=True)
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)

    def _validate_batch_inputs(
        self,
        inputs: List[torch.Tensor],
        futures: List[asyncio.Future]
    ) -> bool:
        """
        Basic checks for input validity (Tensor type, shape consistency).
        Returns True if valid, False otherwise.
        """
        for idx, tensor in enumerate(inputs):
            if not isinstance(tensor, torch.Tensor):
                error_msg = f"Input at index {idx} is not a torch.Tensor."
                self.logger.error(error_msg)
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(ValueError(error_msg))
                return False

        base_shape = inputs[0].shape
        for idx, tensor in enumerate(inputs):
            if tensor.shape != base_shape:
                error_msg = (
                    f"Input at index {idx} has shape {tensor.shape}, "
                    f"expected {base_shape}."
                )
                self.logger.error(error_msg)
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(ValueError(error_msg))
                return False
        return True

    def _infer_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Core batch inference in either FP16 or FP32.
        If using multiple devices in DataParallel, the model handles splitting automatically.
        """
        autocast_enabled = self.use_fp16 and (self.primary_device.type == "cuda")
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self.primary_device.type, enabled=autocast_enabled
        ):
            return self.model(batch_tensor)

    # --------------------------------------------------------------------------
    # Autoscaling (PID or custom logic)
    # --------------------------------------------------------------------------
    async def _autoscale(self):
        """
        Continuously adjusts batch_size based on queue utilization (PID controller).
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
    # Guard Logic
    # --------------------------------------------------------------------------
    def _guard_sample(self, processed: torch.Tensor) -> bool:
        """
        Applies test-time augmentations to check if the sample is adversarial.
        Returns True if the sample passes the guard, False otherwise.
        """
        if not self.config.guard_enabled:
            return True  # If guard not enabled, always pass

        cfg = self.config
        sample = processed
        if sample.dim() == len(self.input_shape):
            sample = sample.unsqueeze(0)
        sample = sample.to(self.primary_device)

        # Create N augmentations
        batch = sample.repeat(cfg.guard_num_augmentations, *([1] * (sample.dim() - 1)))

        # Add random noise
        if "noise" in cfg.guard_augmentation_types:
            noise_levels = torch.empty(cfg.guard_num_augmentations, device=self.primary_device).uniform_(
                *cfg.guard_noise_level_range
            )
            noise_levels = noise_levels.view(cfg.guard_num_augmentations, *([1] * (batch.dim() - 1)))
            noise = torch.randn_like(batch) * noise_levels
            batch = batch + noise

        # Dropout
        if "dropout" in cfg.guard_augmentation_types:
            dropout_flags = torch.rand(cfg.guard_num_augmentations, device=self.primary_device) < 0.3
            if dropout_flags.any():
                dropout_indices = dropout_flags.nonzero(as_tuple=True)[0]
                mask = (
                    torch.rand_like(batch[dropout_indices]) >= cfg.guard_dropout_rate
                ).float()
                batch[dropout_indices] *= mask

        # Flip (if 2D+ shape, e.g. images)
        if "flip" in cfg.guard_augmentation_types and len(self.input_shape) >= 2:
            flip_flags = torch.rand(cfg.guard_num_augmentations, device=self.primary_device) < cfg.guard_flip_prob
            if flip_flags.any():
                flip_indices = flip_flags.nonzero(as_tuple=True)[0]
                batch[flip_indices] = torch.flip(batch[flip_indices], dims=[-1])

        # Clamp to valid input range
        batch = torch.clamp(batch, min=cfg.guard_input_range[0], max=cfg.guard_input_range[1])

        with torch.no_grad():
            autocast_enabled = self.use_fp16 and (self.primary_device.type == "cuda")
            with torch.amp.autocast(device_type=self.primary_device.type, enabled=autocast_enabled):
                preds = self.model(batch)

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
        return (confidence >= cfg.guard_confidence_threshold) and (variance <= cfg.guard_variance_threshold)

    # --------------------------------------------------------------------------
    # Public Methods
    # --------------------------------------------------------------------------
    async def run_inference_async(self, input_data: Any, priority: int = 0) -> torch.Tensor:
        """
        Asynchronously process an inference request. If guard fails,
        returns a default uniform distribution over classes (if num_classes known).
        """
        loop = asyncio.get_running_loop()
        try:
            processed = await loop.run_in_executor(
                self.guard_executor, self.preprocessor, input_data
            )
            if not isinstance(processed, torch.Tensor):
                processed = torch.as_tensor(processed, dtype=torch.float32)

            # Guard check
            is_safe = await loop.run_in_executor(
                self.guard_executor, self._guard_sample, processed
            )
            if not is_safe:
                self.logger.warning("Guard triggered: returning default response.")
                if self.config.num_classes > 0:
                    probs = torch.ones(1, self.config.num_classes, device=self.primary_device)
                    return (probs / self.config.num_classes).squeeze(0)
                else:
                    return torch.tensor([], device=self.primary_device)

            # Enqueue
            future = loop.create_future()
            await self.request_queue.put(RequestItem(processed, future, priority=priority))

            if self.config.debug_mode:
                self.logger.debug(
                    f"Queued request. Current queue size: {self.request_queue.qsize()}"
                )

            # If async mode is disabled, we manually process right away (blocking)
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
        For parallel computing, DataParallel is used if multiple GPUs are specified.
        """
        if isinstance(batch, list):
            batch = torch.stack(
                [x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
                 for x in batch],
                dim=0
            )
        elif not isinstance(batch, torch.Tensor):
            raise TypeError("run_batch_inference expects a torch.Tensor or list of Tensors.")

        if batch.device != self.primary_device:
            batch = batch.to(self.primary_device, non_blocking=True)

        with torch.inference_mode():
            autocast_enabled = self.use_fp16 and (self.primary_device.type == "cuda")
            with torch.amp.autocast(device_type=self.primary_device.type, enabled=autocast_enabled):
                output = self.model(batch)
        return output

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
                    f"Input shape {inputs.shape} doesn't match expected shape + batch dimension."
                )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
            record_shapes=True
        ) as prof:
            with record_function("preprocess"):
                batch = self.preprocessor(inputs).to(self.primary_device)

            with record_function("inference"):
                with torch.inference_mode():
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
        """Synchronously shut down the engine."""
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
    # Cleanup
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
                    self.logger.error(
                        "Background task raised an exception during cleanup",
                        exc_info=res
                    )

        # Shutdown executors
        self.executor.shutdown(wait=True)
        self.guard_executor.shutdown(wait=True)
        self.inference_executor.shutdown(wait=True)

        if self.primary_device.type == 'cuda':
            torch.cuda.empty_cache()

        self.logger.info("Cleanup completed.")

    # --------------------------------------------------------------------------
    # Async Context Manager
    # --------------------------------------------------------------------------
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

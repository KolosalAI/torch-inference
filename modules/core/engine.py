#!/usr/bin/env python3
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Optional, List, Dict, Callable, Union, Any
import torch_tensorrt
from preprocessor import BasePreprocessor
from postprocessor import BasePostprocessor

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# ------------------------------------------------------------------------------
# Helper: PID Controller for Autoscaling Batch Size
# ------------------------------------------------------------------------------
class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, setpoint: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.last_error = 0.0
        self.integral = 0.0

    def update(self, current_value: float, dt: float) -> float:
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return output

# ------------------------------------------------------------------------------
# Engine Configuration with Advanced Options (including TRT)
# ------------------------------------------------------------------------------
class EngineConfig:
    """
    Configuration for the InferenceEngine.
    """
    def __init__(
        self,
        num_workers: int = 1,
        queue_size: int = 100,
        batch_size: int = 32,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        warmup_runs: int = 10,
        timeout: float = 0.1,
        autoscale_interval: float = 5.0,
        queue_size_threshold_high: float = 80.0,
        queue_size_threshold_low: float = 20.0,
        enable_dynamic_batching: bool = True,
        debug_mode: bool = False,
        use_multigpu: bool = False,
        device_ids: Optional[List[int]] = None,
        multigpu_strategy: str = 'dataparallel',  # or 'distributed'
        log_file: str = "inference_engine.log",
        executor_type: str = "thread",  # "thread" or "process"
        # PID controller parameters for autoscaling
        pid_kp: float = 0.5,
        pid_ki: float = 0.1,
        pid_kd: float = 0.05,
        # TensorRT configuration parameters:
        enable_trt: bool = False,
        trt_mode: str = "static",  # "static" or "dynamic"
        trt_workspace_size: int = 1 << 30,  # 1GB default workspace
        trt_min_block_size: int = 1,
        trt_opt_shape: Optional[List[int]] = None,  # Only used in dynamic mode
        trt_input_shape: Optional[List[int]] = None,  # Override input shape for TRT compile
        input_shape: Optional[torch.Size] = None,  # Expected input shape for the model,
        use_tensorrt: bool = False,
    ):
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.warmup_runs = warmup_runs
        self.timeout = timeout
        self.autoscale_interval = autoscale_interval
        self.queue_size_threshold_high = queue_size_threshold_high
        self.queue_size_threshold_low = queue_size_threshold_low
        self.enable_dynamic_batching = enable_dynamic_batching
        self.debug_mode = debug_mode
        self.use_multigpu = use_multigpu
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.multigpu_strategy = multigpu_strategy
        self.log_file = log_file
        self.executor_type = executor_type
        # PID controller for autoscaling (aim for 50% queue utilization)
        self.pid_controller = PIDController(pid_kp, pid_ki, pid_kd, setpoint=50.0)
        # TensorRT-specific settings
        self.enable_trt = enable_trt
        self.trt_mode = trt_mode
        self.trt_workspace_size = trt_workspace_size
        self.trt_min_block_size = trt_min_block_size
        self.trt_opt_shape = trt_opt_shape
        self.trt_input_shape = trt_input_shape
        self.input_shape = input_shape  # Add this field
        self.use_tensorrt = use_tensorrt


    def configure_logging(self):
        """Set up logging to both console and file using a simple formatter."""
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
# Inference Engine Class with Advanced Features and Optimized TensorRT
# ------------------------------------------------------------------------------

class InferenceEngine:
    """
    An asynchronous inference engine that supports:
      - Advanced dynamic batching (memory-, queue-, and PID-based autoscaling)
      - Multi-GPU support (DataParallel and a stub for DistributedDataParallel)
      - FP16 / mixed precision on CUDA
      - Option to use ThreadPool or ProcessPool for concurrency (if picklable)
      - Optional TensorRT compilation for improved GPU inference performance
      - Improved error handling, logging, and profiling
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
        self.config = config or EngineConfig()
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
        # If not provided, default to instances of the base classes.
        self.preprocessor = preprocessor if preprocessor is not None else BasePreprocessor()
        self.postprocessor = postprocessor if postprocessor is not None else BasePostprocessor()

        # Detect expected input shape (robust detection with fallback).
        self.input_shape = self._detect_input_shape()

        # Choose executor type for offloading inference.
        if self.config.executor_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=self.config.num_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)

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
        if device is None:
            if torch.cuda.is_available():
                return torch.device('cuda:0')
            return torch.device('cpu')
        return torch.device(device)
    
    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Moves the model to the selected device. Wraps the model in DataParallel if multi-GPU is enabled.
        A stub is provided for DistributedDataParallel.
        """
        model = model.to(self.device).eval()
        if self.config.use_multigpu and torch.cuda.device_count() > 1:
            if self.config.multigpu_strategy.lower() == 'dataparallel':
                self.logger.info(f"Using DataParallel on devices: {self.config.device_ids}")
                model = nn.DataParallel(model, device_ids=self.config.device_ids)
            elif self.config.multigpu_strategy.lower() == 'distributed':
                raise NotImplementedError("DistributedDataParallel support is not yet implemented.")
            else:
                self.logger.warning(f"Unknown multi-GPU strategy: {self.config.multigpu_strategy}")
        return model

    def _log_device_info(self):
        """
        Log useful details about the selected device.
        """
        self.logger.info(f"Running on device: {self.device}")
        if self.device.type == 'cuda':
            device_index = self.device.index if self.device.index is not None else 0
            self.logger.info(f"GPU Name: {torch.cuda.get_device_name(device_index)}")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            total_mem_gb = torch.cuda.get_device_properties(device_index).total_memory / 1e9
            self.logger.info(f"Total GPU Memory: {total_mem_gb:.2f} GB")

    def _detect_input_shape(self) -> Optional[torch.Size]:
        """
        Attempt to infer the model's expected (non-batch) input shape.
        First, try running a dummy input. If that fails, fall back to heuristics.
        """
        self.logger.debug("Detecting input shape via dummy run...")
        dummy_input = torch.randn(1, 10, device=self.device)
        try:
            with torch.no_grad():
                _ = self.model(dummy_input)
            shape = dummy_input.shape[1:]
            self.logger.debug(f"Inferred input shape from dummy run: {shape}")
            return shape
        except Exception as e:
            self.logger.debug(f"Dummy run failed: {e}; scanning modules for hints...")
            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    shape = (module.in_features,)
                    self.logger.debug(f"Inferred shape from Linear layer: {shape}")
                    return torch.Size(shape)
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

        if self.device.type == 'cuda':
            self.logger.info(f"Starting warmup with {self.config.warmup_runs} iterations...")
            batch_shape = (self.config.batch_size,) + tuple(self.input_shape)
            dummy_input = torch.randn(*batch_shape, device=self.device)
            with torch.no_grad():
                for _ in range(self.config.warmup_runs):
                    _ = self.model(dummy_input)
            torch.cuda.synchronize()
            self.logger.info("Warmup completed.")

    # --- Optimized TensorRT Compilation ---
    def _compile_trt_model(self) -> Union[torch.nn.Module, tuple]:
        """
        Compiles the PyTorch model with TensorRT.
        Supports multi-input models if `trt_input_shapes` is provided in the config.
        Returns a tuple (compiled_model, compile_time).
        """
        import torch_tensorrt  # Import here to avoid unnecessary dependency if TRT is not used.

        if self.config.trt_input_shapes is None:
            # Assume a single-input model using the detected or configured input shape.
            inputs = [torch_tensorrt.Input(
                min_shape=self.input_shape,
                opt_shape=self.input_shape,
                max_shape=self.input_shape,
                dtype=torch.float32
            )]
        else:
            inputs = []
            for shape_tuple in self.config.trt_input_shapes:
                try:
                    min_shape, opt_shape, max_shape = shape_tuple
                except Exception as e:
                    raise ValueError("Each TensorRT input shape must be a tuple: (min_shape, opt_shape, max_shape).") from e
                inputs.append(torch_tensorrt.Input(
                    min_shape=torch.Size(min_shape),
                    opt_shape=torch.Size(opt_shape),
                    max_shape=torch.Size(max_shape),
                    dtype=torch.float32
                ))
        start_time = time.time()
        # Compile the model to use FP16 if desired.
        compiled_model = torch_tensorrt.compile(
            self.model,
            inputs=inputs,
            enabled_precisions={torch.half}  # Enable FP16.
        )
        compile_time = time.time() - start_time
        return compiled_model, compile_time

    # --- Advanced Dynamic Batch Sizing ---
    def dynamic_batch_size(self, sample_tensor: torch.Tensor) -> int:
        """
        Determine a batch size based on free GPU memory, queue length,
        and a PID controller adjustment.
        """
        if self.device.type == 'cuda':
            sample_bytes = sample_tensor.element_size() * sample_tensor.nelement()
            batch_sizes = []
            for device_id in self.config.device_ids:
                props = torch.cuda.get_device_properties(device_id)
                total_mem = props.total_memory
                used_mem = torch.cuda.memory_allocated(device_id)
                free_mem = total_mem - used_mem
                possible_batch = max(int(free_mem // sample_bytes), 1)
                batch_sizes.append(possible_batch)
            memory_based = min(batch_sizes)
        else:
            memory_based = self.config.max_batch_size

        # Incorporate queued requests.
        queue_based = min(
            self.config.max_batch_size,
            max(self.config.min_batch_size, self.request_queue.qsize() + 1)
        )
        base_batch_size = min(memory_based, queue_based)

        # Use PID controller to adjust the batch size gradually.
        utilization = (self.request_queue.qsize() / self.config.queue_size) * 100.0
        now = time.time()
        dt = getattr(self, "_last_autoscale_time", now - 1.0)
        self._last_autoscale_time = now
        pid_adjustment = self.config.pid_controller.update(utilization, dt)
        new_batch = int(round(base_batch_size + pid_adjustment))
        new_batch = max(self.config.min_batch_size, min(new_batch, self.config.max_batch_size))
        if self.config.debug_mode:
            self.logger.debug(
                f"Dynamic batch size: memory_based={memory_based}, queue_based={queue_based}, "
                f"utilization={utilization:.1f}%, pid_adjustment={pid_adjustment:.2f} -> {new_batch}"
            )
        return new_batch

    # --- Batch Processing and Inference ---
    async def _process_batches(self):
        """
        Asynchronously aggregate requests into batches and perform inference.
        Uses advanced dynamic batching (with timeout and partial batches).
        """
        self.logger.info("Starting batch processing loop...")
        while True:
            batch_items: List["RequestItem"] = []
            try:
                first_item = await asyncio.wait_for(
                    self.request_queue.get(), timeout=self.config.timeout
                )
                batch_items.append(first_item)
            except asyncio.TimeoutError:
                continue

            if self.config.enable_dynamic_batching:
                try:
                    sample_tensor = first_item.input
                    desired_batch_size = self.dynamic_batch_size(sample_tensor)
                except Exception as e:
                    self.logger.error(f"Error computing dynamic batch size: {e}")
                    desired_batch_size = self.config.batch_size
            else:
                desired_batch_size = self.config.batch_size

            while len(batch_items) < desired_batch_size:
                try:
                    next_item = await asyncio.wait_for(self.request_queue.get(), timeout=0.01)
                    batch_items.append(next_item)
                except asyncio.TimeoutError:
                    break

            try:
                inputs = [item.input for item in batch_items]
                futures = [item.future for item in batch_items]
                batch_tensor = torch.stack(inputs).to(self.device)

                loop = asyncio.get_event_loop()
                inference_fn = lambda: self._infer_batch(batch_tensor)
                outputs = await loop.run_in_executor(self.executor, inference_fn)

                results = [self.postprocessor(output) for output in outputs]
                for fut, res in zip(futures, results):
                    if not fut.done():
                        fut.set_result(res)
                self.logger.debug(f"Processed batch of {len(batch_items)} items.")
            except Exception as e:
                self.logger.error("Batch processing error", exc_info=True)
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)

    def _infer_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Internal method to perform inference on a batch.
        Applies FP16 autocast if enabled.
        """
        with torch.no_grad():
            if self.use_fp16 and self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    return self.model(batch_tensor)
            else:
                return self.model(batch_tensor)

    # --- Autoscaling Batch Size using PID ---
    async def _autoscale(self):
        """
        Periodically check the queue and adjust the configured batch size using a PID controller.
        """
        self.logger.info("Starting autoscaling task...")
        while True:
            start_time = time.time()
            await asyncio.sleep(self.config.autoscale_interval)
            current_queue = self.request_queue.qsize()
            utilization = (current_queue / self.config.queue_size) * 100.0

            pid_adjustment = self.config.pid_controller.update(utilization, self.config.autoscale_interval)
            new_batch_size = int(round(self.config.batch_size + pid_adjustment))
            new_batch_size = max(self.config.min_batch_size, min(new_batch_size, self.config.max_batch_size))
            if new_batch_size != self.config.batch_size:
                old_size = self.config.batch_size
                self.config.batch_size = new_batch_size
                self.logger.info(f"Autoscale adjusted batch size from {old_size} to {new_batch_size} "
                                 f"(queue utilization: {utilization:.1f}%)")
            else:
                self.logger.debug(f"No autoscale change (batch size remains {self.config.batch_size}, "
                                  f"queue utilization: {utilization:.1f}%)")
            dt = time.time() - start_time

    # --- Public Inference Interfaces ---
    async def run_inference_async(self, input_data: Any, priority: int = 0) -> Any:
        try:
            # Process the input using the preprocessor.
            processed = self.preprocessor(input_data)
            if not isinstance(processed, torch.Tensor):
                processed = torch.tensor(processed, dtype=torch.float32)

            # Ensure that each request is a single sample with shape equal to self.input_shape.
            if self.input_shape is not None:
                sample_expected_dim = len(self.input_shape)  # e.g. (10,) → 1
                if processed.dim() == sample_expected_dim + 1:
                    # If there's a batch dimension and it is 1, remove it.
                    if processed.size(0) == 1:
                        processed = processed.squeeze(0)
                    else:
                        raise ValueError(
                            f"Input contains a batch dimension > 1: {processed.shape}. "
                            f"Expected single sample input matching shape {self.input_shape}."
                        )
                elif processed.dim() != sample_expected_dim:
                    raise ValueError(
                        f"Input shape {processed.shape} doesn't match expected shape {self.input_shape}"
                    )

            loop = asyncio.get_event_loop()
            future = loop.create_future()
            await self.request_queue.put(RequestItem(input=processed, future=future, priority=priority))
            self.logger.debug(f"Queued request. Queue size: {self.request_queue.qsize()}")
            return await future

        except Exception as e:
            self.logger.error("Inference failed", exc_info=True)
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            future.set_exception(e)
            return await future

    def run_batch_inference(self, batch: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Runs inference on a batch of preprocessed inputs.
        Ensures that the input is a Tensor (by stacking if a list is provided).
        This is a synchronous method. For non-blocking behavior, consider wrapping it
        with an executor or using an async version.
        """
        # If a list of tensors is provided, stack them into a single tensor.
        if isinstance(batch, list):
            try:
                batch = torch.stack(batch, dim=0)
            except Exception as e:
                raise ValueError("Failed to stack the input list into a tensor. "
                                "Ensure all elements are tensors with matching dimensions.") from e

        with torch.no_grad():
            output = self.model(batch)
        return output

    def profile_inference(self, inputs: Any) -> Dict[str, float]:
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)

        # Ensure the input has a batch dimension.
        if self.input_shape is not None:
            expected_dim = len(self.input_shape) + 1
            while inputs.dim() > expected_dim and inputs.size(0) == 1:
                inputs = inputs.squeeze(0)
            if inputs.dim() < expected_dim:
                inputs = inputs.unsqueeze(0)
            elif inputs.dim() > expected_dim:
                raise ValueError(f"Input shape {inputs.shape} doesn't match expected shape with batch dimension.")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True
        ) as prof:
            with record_function("preprocess"):
                batch = self.preprocessor(inputs)
                if not isinstance(batch, torch.Tensor):
                    batch = torch.tensor(batch, dtype=torch.float32)
                batch = batch.to(self.device)
            with record_function("inference"):
                with torch.no_grad():
                    inference_output = self.model(batch)
            with record_function("postprocess"):
                _ = self.postprocessor(inference_output)

        key_avgs = prof.key_averages()
        metrics = {}
        for evt in ["preprocess", "inference", "postprocess"]:
            match = [x for x in key_avgs if x.key == evt]
            metrics[f"{evt}_ms"] = match[0].cpu_time_total / 1000.0 if match else 0.0
        metrics["total_ms"] = sum(metrics.values())
        self.logger.debug("Inference profile results:\n" +
                        "\n".join([f"{k}: {v:.2f} ms" for k, v in metrics.items()]))
        return metrics

    def cleanup(self):
        """
        Cancel async tasks and release resources.
        """
        self.logger.info("Cleaning up inference engine resources...")
        self.batch_processor_task.cancel()
        self.autoscale_task.cancel()
        self.executor.shutdown(wait=False)
        if self.device.type == 'cuda':
            if isinstance(self.model, nn.DataParallel):
                for device_id in self.model.device_ids:
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
        self.logger.info("Cleanup completed.")


# ------------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------------

async def main():
    # Define a simple model: a linear layer mapping 10 features to 2 outputs.
    model = torch.nn.Linear(10, 2).to("cuda")

    # Create the engine configuration.
    # Here we only specify the minimal configuration needed for this example.
    config = EngineConfig(
        input_shape=[1, 10],  # The expected input shape (including batch dimension).
        # Optionally, specify TensorRT input shapes if using TRT:
        trt_input_shape=[([1, 10], [32, 10], [128, 10])],
        use_tensorrt=False,  # Change to True if TensorRT is desired and installed.
    )

    # Create the inference engine.
    # If your InferenceEngine supports a flag for FP16, pass it accordingly.
    engine = InferenceEngine(model=model, config=config)  # , use_fp16=True

    # Generate 100 valid inputs (each with 10 features).
    # Note: Each input is a 1D tensor, so we unsqueeze(0) to get a batch dimension.
    inputs = [torch.randn(10, device="cuda") for _ in range(100)]

    # Run asynchronous inference on all inputs concurrently.
    # Wrap each input with unsqueeze(0) so that it has shape [1, 10]
    tasks = [engine.run_inference_async(x.unsqueeze(0)) for x in inputs]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    engine.logger.info(f"Received {len(results)} asynchronous inference results.")

    # Optionally, profile a single inference call (if your engine supports profiling).
    if hasattr(engine, 'profile_inference'):
        # Ensure the input has a batch dimension.
        profile_metrics = engine.profile_inference(inputs[0].unsqueeze(0))
        engine.logger.info(f"Profile metrics: {profile_metrics}")

    # Synchronous inference for batch processing.
    # Here, engine.run_batch_inference will stack the list of [1, 10] tensors into a batch.
    sync_results = engine.run_batch_inference([x.unsqueeze(0) for x in inputs])
    engine.logger.info(f"Synchronous inference produced {sync_results.shape[0]} results.")

    # Clean up resources before exit.
    engine.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted!")
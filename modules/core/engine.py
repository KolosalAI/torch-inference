import logging
import asyncio
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Callable, Union, Any
from concurrent.futures import ThreadPoolExecutor

# For profiling
from torch.profiler import profile, record_function, ProfilerActivity

logger = logging.getLogger(__name__)

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
        log_file: str = "inference_engine.log"
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
        # Default to all GPUs if device_ids is None
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.multigpu_strategy = multigpu_strategy
        self.log_file = log_file

    def configure_logging(self):
        """Set up logging to both console and file."""
        level = logging.DEBUG if self.debug_mode else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_file)
            ]
        )
        if self.debug_mode:
            logger.debug("Debug logging enabled")


class RequestItem:
    """
    Holds a single inference request (processed input) and a Future
    to store the asynchronous result.
    """
    def __init__(self, input: Any, future: asyncio.Future):
        self.input = input
        self.future = future


class InferenceEngine:
    """
    Asynchronous inference engine that supports:
      - Dynamic batching
      - Multi-GPU (DataParallel)
      - Autoscaling batch size
      - FP16 execution on CUDA
    """
    def __init__(
        self,
        model: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        preprocessor: Optional[Callable] = None,
        postprocessor: Optional[Callable] = None,
        use_fp16: bool = False,
        config: Optional[EngineConfig] = None,
    ):
        self.config = config or EngineConfig()
        self.config.configure_logging()

        # Automatic device detection
        self.device = self._auto_select_device(device)
        # Only enable fp16 if a CUDA device is being used
        self.use_fp16 = use_fp16 and ('cuda' in str(self.device))

        # Prepare the model (possibly for multi-GPU)
        self.model = self._prepare_model(model)

        # Pre/post-processing
        self.preprocessor = preprocessor or (lambda x: x)
        self.postprocessor = postprocessor or (lambda x: x)

        # Detect expected input shape
        self.input_shape = self._detect_input_shape()

        # Thread-pool executor for offloading model inference
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)

        # Request queue for asynchronous inference
        self.request_queue = asyncio.Queue(maxsize=self.config.queue_size)

        # Create tasks for batch processing and autoscaling
        self.batch_processor_task = asyncio.create_task(self._process_batches())
        self.autoscale_task = asyncio.create_task(self._autoscale())

        # Warmup the model (especially important for CUDA)
        self._warmup()

        # Log device info (GPU memory, etc.)
        self._log_device_info()

    def _auto_select_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """
        Automatically select an available device, falling back to CPU if
        no CUDA device is detected or if none is specified.
        """
        if device is None:
            if torch.cuda.is_available():
                return torch.device('cuda:0')
            return torch.device('cpu')
        return torch.device(device)

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Move the model to the proper device and, if configured, wrap in
        DataParallel for multi-GPU usage.
        """
        model = model.to(self.device).eval()

        if self.config.use_multigpu and torch.cuda.device_count() > 1:
            if self.config.multigpu_strategy == 'dataparallel':
                logger.info(f"Using DataParallel on devices: {self.config.device_ids}")
                model = nn.DataParallel(model, device_ids=self.config.device_ids)
            elif self.config.multigpu_strategy == 'distributed':
                raise NotImplementedError("DistributedDataParallel support is not yet implemented.")
            else:
                logger.warning(f"Unknown multi-GPU strategy: {self.config.multigpu_strategy}")

        return model

    def _log_device_info(self):
        """
        Log useful details about the current device, such as name and memory.
        """
        logger.info(f"Running on device: {self.device}")
        if self.device.type == 'cuda':
            # Get the actual GPU index or default to 0
            device_index = self.device.index if self.device.index is not None else 0
            logger.info(f"GPU Name: {torch.cuda.get_device_name(device_index)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            total_mem_gb = torch.cuda.get_device_properties(device_index).total_memory / 1e9
            logger.info(f"Total GPU Memory: {total_mem_gb:.2f} GB")

    def _detect_input_shape(self) -> Optional[torch.Size]:
        """
        Attempt to infer the model's expected (non-batch) input shape by
        running a dummy input. If that fails, fallback heuristics for
        common layer types (Linear, Conv2d) are used.
        """
        if self.config.debug_mode:
            logger.debug("Starting input shape detection...")

        dummy_input = torch.randn(1, 10, device=self.device)
        try:
            with torch.no_grad():
                _ = self.model(dummy_input)
            shape = dummy_input.shape[1:]
            if self.config.debug_mode:
                logger.debug(f"Inferred input shape from dummy run: {shape}")
            return shape
        except Exception as e:
            if self.config.debug_mode:
                logger.debug(f"Dummy input failed: {e} -- searching modules...")

            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    shape = (module.in_features,)
                    logger.debug(f"Inferred shape (Linear): {shape}")
                    return shape
                elif isinstance(module, nn.Conv2d):
                    # A common default shape for image data
                    shape = (3, 224, 224)
                    logger.debug(f"Inferred shape (Conv2d default): {shape}")
                    return shape
        # If all else fails, return None
        return None

    def _warmup(self):
        """
        Run a few forward passes to 'warm up' the model (especially on GPU),
        which can help stabilize performance.
        """
        if self.input_shape is None:
            return  # No shape known, can't warm up

        if self.device.type == 'cuda':
            logger.info(f"Starting warmup with {self.config.warmup_runs} iterations...")
            batch_shape = (self.config.batch_size,) + self.input_shape
            dummy_input = torch.randn(*batch_shape, device=self.device)

            with torch.no_grad():
                for _ in range(self.config.warmup_runs):
                    _ = self.model(dummy_input)
            torch.cuda.synchronize()
            logger.info("Warmup completed.")

    def dynamic_batch_size(self, sample_tensor: torch.Tensor) -> int:
        """
        Calculate a batch size based on free GPU memory (naive approach)
        and queue length. This function can be customized.
        """
        if self.device.type == 'cuda':
            # Estimated input memory usage for one sample (bytes):
            sample_bytes = sample_tensor.element_size() * sample_tensor.nelement()

            # For multi-GPU, take a safe "minimum" approach across all device_ids
            batch_sizes = []
            for device_id in self.config.device_ids:
                prop = torch.cuda.get_device_properties(device_id)
                total_mem = prop.total_memory
                used_mem = torch.cuda.memory_allocated(device_id)
                free_mem = total_mem - used_mem

                # Extremely naive: how many samples could fit into free memory alone?
                # A real approach might use only a fraction of free_mem to avoid OOM.
                possible_batch = int(free_mem // sample_bytes)
                # Ensure at least 1 if there's a tiny bit of memory
                possible_batch = max(possible_batch, 1)
                batch_sizes.append(possible_batch)

            memory_based = min(batch_sizes)
        else:
            # On CPU, let's just use the max_batch_size by default (or a static approach)
            memory_based = self.config.max_batch_size

        # Also incorporate how many requests are queued up
        queue_based = min(
            self.config.max_batch_size,
            max(self.config.min_batch_size, self.request_queue.qsize() + 1)
        )

        # Final batch size is the min of memory-based and queue-based constraints
        return min(memory_based, queue_based)

    async def _process_batches(self):
        """
        Core loop that aggregates requests into batches and performs inference.
        Automatically handles dynamic batching (if enabled).
        """
        try:
            while True:
                batch_items = []
                try:
                    # Wait for at least one item in the queue
                    first_item = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=self.config.timeout
                    )
                    batch_items.append(first_item)
                except asyncio.TimeoutError:
                    # No items arrived within the timeout
                    continue

                # Determine batch size
                if self.config.enable_dynamic_batching:
                    try:
                        # We look at the first item to guess memory usage
                        sample_tensor = first_item.input
                        batch_size = self.dynamic_batch_size(sample_tensor)
                    except Exception as e:
                        logger.error(f"Batch size calculation error: {e}")
                        batch_size = self.config.batch_size
                else:
                    batch_size = self.config.batch_size

                # Collect as many items as we can up to batch_size
                while len(batch_items) < batch_size and not self.request_queue.empty():
                    batch_items.append(await self.request_queue.get())

                try:
                    # Prepare the inputs as a single batch tensor
                    inputs = [item.input for item in batch_items]
                    futures = [item.future for item in batch_items]

                    batch_tensor = torch.stack(inputs).to(self.device)

                    with torch.no_grad():
                        if self.use_fp16:
                            # Offload to thread pool for concurrency
                            with torch.cuda.amp.autocast():
                                outputs = await asyncio.get_event_loop().run_in_executor(
                                    self.executor,
                                    lambda: self.model(batch_tensor)
                                )
                        else:
                            outputs = await asyncio.get_event_loop().run_in_executor(
                                self.executor,
                                lambda: self.model(batch_tensor)
                            )

                    # Post-process each sample in the batch
                    results = [self.postprocessor(output) for output in outputs]

                    # Resolve futures
                    for future, result in zip(futures, results):
                        if not future.done():
                            future.set_result(result)

                except Exception as e:
                    logger.error(f"Batch processing error: {e}", exc_info=True)
                    # Propagate exceptions to all futures in this batch
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)
        except asyncio.CancelledError:
            logger.info("Batch processing task cancelled.")

    async def run_inference_async(self, input_data: Any) -> Any:
        """
        Accept a single input, pre-process it, enqueue the request, and
        return an async future that resolves once inference completes.
        """
        try:
            processed = self.preprocessor(input_data)

            if not isinstance(processed, torch.Tensor):
                processed = torch.tensor(processed, dtype=torch.float32)

            if self.config.debug_mode:
                logger.debug(f"Preprocessed input shape: {processed.shape}")

            # Basic shape alignment logic
            if self.input_shape is not None:
                # If we expect e.g. (10,) but the user passes (10,), that is fine
                # If we expect (10,) and the user passes (1,10), we might keep as is
                # If user passes shape that doesn't match, raise an error
                if processed.dim() == len(self.input_shape):
                    if processed.shape != self.input_shape:
                        raise ValueError(
                            f"Input shape {processed.shape} doesn't match expected {self.input_shape}"
                        )
                elif processed.dim() + 1 == len(self.input_shape):
                    # Possibly missing a batch dimension
                    if processed.shape == self.input_shape[1:]:
                        processed = processed.unsqueeze(0)
                        if self.config.debug_mode:
                            logger.debug("Added batch dimension to input.")
                    else:
                        raise ValueError(
                            f"Input shape {processed.shape} doesn't match expected {self.input_shape}"
                        )
                else:
                    # This is a rough check; you can refine as needed
                    raise ValueError(
                        f"Input shape {processed.shape} doesn't match expected {self.input_shape}"
                    )

            # Create a future for the result
            future = asyncio.get_event_loop().create_future()
            await self.request_queue.put(RequestItem(input=processed, future=future))

            if self.config.debug_mode:
                logger.debug(f"Queued request. Queue size: {self.request_queue.qsize()}")

            return await future

        except Exception as e:
            logger.error("Inference failed", exc_info=True)
            # If something fails early, create a future and set the exception
            future = asyncio.get_event_loop().create_future()
            future.set_exception(e)
            return await future

    async def _autoscale(self):
        """
        Periodically check the queue size and adjust the batch size
        within configured min/max bounds.
        """
        try:
            while True:
                await asyncio.sleep(self.config.autoscale_interval)
                current_size = self.request_queue.qsize()
                max_size = self.config.queue_size
                utilization = (current_size / max_size) * 100

                if self.config.debug_mode:
                    logger.debug(
                        f"Autoscale check: queue {current_size}/{max_size} "
                        f"({utilization:.1f}% utilization)"
                    )

                new_size = self.config.batch_size
                if utilization > self.config.queue_size_threshold_high:
                    new_size = min(self.config.batch_size * 2, self.config.max_batch_size)
                elif utilization < self.config.queue_size_threshold_low:
                    new_size = max(self.config.batch_size // 2, self.config.min_batch_size)

                if new_size != self.config.batch_size:
                    old_size = self.config.batch_size
                    self.config.batch_size = new_size
                    if self.config.debug_mode:
                        logger.debug(
                            f"Autoscale adjusted batch size from {old_size} to {new_size} "
                            f"(utilization {utilization:.1f}%)"
                        )
        except asyncio.CancelledError:
            if self.config.debug_mode:
                logger.debug("Autoscaling task cancelled.")

    def run_batch_inference(self, input_list: List[Any]) -> List[Any]:
        """
        Synchronous batch inference for a list of inputs. Useful for
        smaller offline jobs or testing.
        """
        if self.config.debug_mode:
            logger.debug(f"Starting synchronous batch inference on {len(input_list)} items.")

        # Preprocess all inputs
        processed_inputs = []
        for inp in input_list:
            tensor_inp = self.preprocessor(inp)
            if not isinstance(tensor_inp, torch.Tensor):
                tensor_inp = torch.tensor(tensor_inp, dtype=torch.float32)
            processed_inputs.append(tensor_inp)

        # Determine the batch size for this particular run
        if self.config.enable_dynamic_batching and len(processed_inputs) > 0:
            # Estimate from the first input
            batch_size = self.dynamic_batch_size(processed_inputs[0])
        else:
            batch_size = self.config.batch_size

        results = []
        for i in range(0, len(processed_inputs), batch_size):
            batch = torch.stack(processed_inputs[i:i + batch_size]).to(self.device)
            with torch.no_grad():
                if self.use_fp16 and self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)

            # Postprocess
            results.extend([self.postprocessor(o) for o in outputs])

        if self.config.debug_mode:
            logger.debug(f"Completed synchronous batch inference. Total results: {len(results)}")

        return results

    def profile_inference(self, inputs: Any) -> Dict[str, float]:
        """
        Profile the inference process (preprocess -> model -> postprocess).
        Returns a dict of timing metrics in milliseconds.
        """
        if self.config.debug_mode:
            logger.debug("Starting inference profiling...")

        # If user passes a single sample, turn it into a batch of size 1
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)

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
                    _ = self.model(batch)

            with record_function("postprocess"):
                _ = self.postprocessor(_)

        # Aggregate timing
        key_avgs = prof.key_averages()
        metrics = {}
        for evt in ["preprocess", "inference", "postprocess"]:
            match = [x for x in key_avgs if x.key == evt]
            if match:
                metrics[f"{evt}_ms"] = match[0].cpu_time_total / 1000.0
            else:
                metrics[f"{evt}_ms"] = 0.0

        metrics["total_ms"] = sum(metrics.values())

        if self.config.debug_mode:
            logger.debug("Inference profile results:\n" +
                         "\n".join([f"{k}: {v:.2f} ms" for k, v in metrics.items()]))

        return metrics

    def cleanup(self):
        """
        Cancel async tasks and release resources (GPU memory cache).
        """
        self.batch_processor_task.cancel()
        self.autoscale_task.cancel()
        self.executor.shutdown()

        if self.device.type == 'cuda':
            # If the model is DataParallel, free each device
            if isinstance(self.model, nn.DataParallel):
                for device_id in self.model.device_ids:
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()

        logger.info("Cleanup completed.")


# Example usage:
async def main():
    # Simple model that expects 10 features -> 2 outputs
    model = torch.nn.Linear(10, 2)

    # Configuration
    config = EngineConfig(
        debug_mode=True,
        use_multigpu=True,
        device_ids=[0, 1],  # Attempt to use first two GPUs
        multigpu_strategy='dataparallel',
        log_file="multi_gpu.log"
    )
    config.configure_logging()

    # Create engine
    engine = InferenceEngine(
        model=model,
        config=config,
        use_fp16=True
    )

    # Generate valid inputs (10 features each)
    inputs = [torch.randn(10) for _ in range(100)]

    # Run inference asynchronously
    tasks = [engine.run_inference_async(x) for x in inputs]
    results = await asyncio.gather(*tasks)

    logger.info(f"Received {len(results)} inference results.")

    # Profile a single inference call (optional)
    # profile_metrics = engine.profile_inference(inputs[0])
    # logger.info(f"Profile metrics: {profile_metrics}")

    # Cleanup
    engine.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

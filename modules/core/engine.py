import logging
import asyncio
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Callable, Union, Any
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class EngineConfig:
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
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.multigpu_strategy = multigpu_strategy
        self.log_file = log_file

    def configure_logging(self):
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
            logging.debug("Debug logging enabled")
            
class RequestItem:
    def __init__(self, input: Any, future: asyncio.Future):
        self.input = input
        self.future = future
        
class InferenceEngine:
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
        self.use_fp16 = use_fp16 and 'cuda' in str(self.device)
        
        # Multi-GPU setup
        self.model = self._prepare_model(model)
        self.preprocessor = preprocessor or (lambda x: x)
        self.postprocessor = postprocessor or (lambda x: x)
        self.input_shape = self._detect_input_shape()

        self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        self.request_queue = asyncio.Queue(maxsize=self.config.queue_size)
        
        self.batch_processor_task = asyncio.create_task(self._process_batches())
        self.autoscale_task = asyncio.create_task(self._autoscale())

        self._warmup()
        self._log_device_info()

    def _auto_select_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Automatically select available device with fallback"""
        if device is None:
            if torch.cuda.is_available():
                return torch.device('cuda:0')
            return torch.device('cpu')
        return torch.device(device)

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for single/multi-GPU execution"""
        model = model.to(self.device).eval()
        
        if self.config.use_multigpu and torch.cuda.device_count() > 1:
            if self.config.multigpu_strategy == 'dataparallel':
                logger.info(f"Using DataParallel on devices {self.config.device_ids}")
                model = nn.DataParallel(model, device_ids=self.config.device_ids)
            elif self.config.multigpu_strategy == 'distributed':
                raise NotImplementedError("DistributedDataParallel support coming soon")
        
        return model

    def _log_device_info(self):
        """Log detailed device information"""
        logger.info(f"Running on device: {self.device}")
        if 'cuda' in str(self.device):
            logger.info(f"GPU Name: {torch.cuda.get_device_name(self.device)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory/1e9:.2f} GB")

    def _detect_input_shape(self):
        """Improved input shape detection handling batch dimensions."""
        if self.config.debug_mode:
            logger.debug("Starting input shape detection")
            
        dummy_input = torch.randn(1, 10)
        try:
            with torch.no_grad():
                output = self.model(dummy_input)
            shape = dummy_input.shape[1:]
            if self.config.debug_mode:
                logger.debug(f"Shape detected through dummy input: {shape}")
            return shape
        except Exception as e:
            if self.config.debug_mode:
                logger.debug(f"Dummy input failed, searching modules: {str(e)}")
            
            for module in self.model.modules():
                if isinstance(module, torch.nn.Linear):
                    shape = (module.in_features,)
                    if self.config.debug_mode:
                        logger.debug(f"Detected Linear layer with input shape: {shape}")
                    return shape
                elif isinstance(module, torch.nn.Conv2d):
                    shape = (3, 224, 224)
                    if self.config.debug_mode:
                        logger.debug(f"Detected Conv2d layer with default shape: {shape}")
                    return shape
        return None

    def _warmup(self):
        """Enhanced warmup with multi-GPU support"""
        if self.input_shape and 'cuda' in str(self.device):
            batch_shape = (self.config.batch_size,) + self.input_shape
            dummy_input = torch.randn(*batch_shape, device=self.device)
            
            logger.info(f"Starting warmup with {self.config.warmup_runs} iterations")
            with torch.no_grad():
                for _ in range(self.config.warmup_runs):
                    _ = self.model(dummy_input)
            torch.cuda.synchronize()
            logger.info("Warmup completed")

    def dynamic_batch_size(self, input_size: int) -> int:
        """Enhanced batch size calculation for multi-GPU"""
        if 'cuda' in str(self.device):
            if self.config.use_multigpu:
                # Calculate based on all available GPUs
                batch_sizes = []
                for device_id in self.config.device_ids:
                    with torch.cuda.device(device_id):
                        free_mem = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                        batch_sizes.append(free_mem // (input_size * 4))
                memory_based = min(batch_sizes)
            else:
                free_memory = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                memory_based = max(1, int(free_memory / (input_size * 4)))
        else:
            memory_based = self.config.max_batch_size
        
        queue_based = min(
            self.config.max_batch_size,
            max(self.config.min_batch_size, self.request_queue.qsize() + 1)
        )
        return min(memory_based, queue_based)

    async def _process_batches(self):
        """Multi-GPU compatible batch processing"""
        try:
            while True:
                batch_items = []
                try:
                    first_item = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=self.config.timeout
                    )
                    batch_items.append(first_item)
                except asyncio.TimeoutError:
                    continue

                if self.config.enable_dynamic_batching:
                    try:
                        input_size = batch_items[0].input.numel() * 4
                        batch_size = self.dynamic_batch_size(input_size)
                    except Exception as e:
                        logger.error(f"Batch size calculation error: {str(e)}")
                        batch_size = self.config.batch_size
                else:
                    batch_size = self.config.batch_size

                while len(batch_items) < batch_size and not self.request_queue.empty():
                    batch_items.append(await self.request_queue.get())

                try:
                    inputs = [item.input for item in batch_items]
                    futures = [item.future for item in batch_items]

                    batch_tensor = torch.stack(inputs).to(self.device)
                    
                    with torch.no_grad():
                        if self.use_fp16:
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

                    results = [self.postprocessor(output) for output in outputs]
                    for future, result in zip(futures, results):
                        if not future.done():
                            future.set_result(result)
                except Exception as e:
                    logger.error(f"Batch processing error: {str(e)}")
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)
        except asyncio.CancelledError:
            logger.info("Batch processing task cancelled")

    async def run_inference_async(self, input: Any) -> Any:
        """Asynchronous inference with proper shape validation."""
        try:
            processed = self.preprocessor(input)
            
            if self.config.debug_mode:
                logger.debug(f"Preprocessed input shape: {processed.shape}")

            if self.input_shape and processed.shape != self.input_shape:
                if processed.dim() == len(self.input_shape) and processed.shape == self.input_shape:
                    if self.config.debug_mode:
                        logger.debug("Input shape matches without batch dimension")
                elif processed.dim() + 1 == len(self.input_shape) and processed.shape == self.input_shape[1:]:
                    processed = processed.unsqueeze(0)
                    if self.config.debug_mode:
                        logger.debug("Added batch dimension to input")
                else:
                    raise ValueError(
                        f"Input shape {processed.shape} doesn't match model expectation {self.input_shape}"
                    )

            future = asyncio.get_event_loop().create_future()
            await self.request_queue.put(RequestItem(input=processed, future=future))
            
            if self.config.debug_mode:
                logger.debug(f"Queued request. Queue size: {self.request_queue.qsize()}")
                
            return await future
        except Exception as e:
            logger.error("Inference failed", exc_info=True)
            future = asyncio.get_event_loop().create_future()
            future.set_exception(e)
            return await future
    async def _autoscale(self):
        """Autoscaling based on queue size."""
        try:
            while True:
                await asyncio.sleep(self.config.autoscale_interval)
                current_size = self.request_queue.qsize()
                max_size = self.config.queue_size
                utilization = (current_size / max_size) * 100

                if self.config.debug_mode:
                    logger.debug(
                        f"Autoscale check - Queue: {current_size}/{max_size} "
                        f"({utilization:.1f}% utilization)"
                    )

                new_size = self.config.batch_size
                if utilization > self.config.queue_size_threshold_high:
                    new_size = min(
                        self.config.batch_size * 2,
                        self.config.max_batch_size
                    )
                elif utilization < self.config.queue_size_threshold_low:
                    new_size = max(
                        self.config.batch_size // 2,
                        self.config.min_batch_size
                    )

                if new_size != self.config.batch_size:
                    old_size = self.config.batch_size
                    self.config.batch_size = new_size
                    if self.config.debug_mode:
                        logger.debug(
                            f"Autoscale adjusted batch size from {old_size} to {new_size} "
                            f"based on queue utilization {utilization:.1f}%"
                        )
        except asyncio.CancelledError:
            if self.config.debug_mode:
                logger.debug("Autoscaling task cancelled")
    def run_batch_inference(self, input_list: List[Any]) -> List[Any]:
        """Synchronous batch inference."""
        if self.config.debug_mode:
            logger.debug(f"Starting synchronous batch inference on {len(input_list)} items")

        batch_size = self.dynamic_batch_size(input_list[0].numel()) if self.config.enable_dynamic_batching else self.config.batch_size
        results = []
        
        for i in range(0, len(input_list), batch_size):
            if self.config.debug_mode:
                logger.debug(f"Processing batch {i//batch_size + 1}/{(len(input_list)-1)//batch_size + 1}")
                
            batch = torch.stack([self.preprocessor(input_) for input_ in input_list[i:i + batch_size]]).to(self.device)
            
            with torch.no_grad():
                if self.use_fp16 and "cuda" in str(self.device):
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)
                    
            results.extend([self.postprocessor(output) for output in outputs])

        if self.config.debug_mode:
            logger.debug(f"Completed synchronous batch inference. Total results: {len(results)}")
            
        return results
    def profile_inference(self, inputs: Any) -> Dict[str, float]:
        """Profile the inference process."""
        if self.config.debug_mode:
            logger.debug("Starting inference profiling")

        metrics = {}
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("preprocess"):
                batch = self.preprocessor(inputs)
            with record_function("inference"):
                with torch.no_grad():
                    outputs = self.model(batch)
            with record_function("postprocess"):
                results = self.postprocessor(outputs)

        metrics["preprocess_ms"] = prof.key_averages().get("preprocess").cpu_time_total / 1000
        metrics["inference_ms"] = prof.key_averages().get("inference").cpu_time_total / 1000
        metrics["postprocess_ms"] = prof.key_averages().get("postprocess").cpu_time_total / 1000
        metrics["total_ms"] = sum(metrics.values())

        if self.config.debug_mode:
            logger.debug("Inference profile results:\n" +
                         "\n".join([f"{k}: {v:.2f}ms" for k, v in metrics.items()]))

        return metrics
    def cleanup(self):
        """Enhanced cleanup for multi-GPU support"""
        self.batch_processor_task.cancel()
        self.autoscale_task.cancel()
        self.executor.shutdown()
        
        if 'cuda' in str(self.device):
            # Check if the model is using DataParallel
            if isinstance(self.model, nn.DataParallel):
                # Use the actual device IDs from the DataParallel module
                device_ids = self.model.device_ids
                for device_id in device_ids:
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
                    
        logger.info("Cleanup completed")
        
async def main():
    # Model expects 10 input features
    model = torch.nn.Linear(10, 2)
    
    # Configuration
    config = EngineConfig(
        debug_mode=True,
        use_multigpu=True,
        device_ids=[0, 1],  # Use first two GPUs
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
    
    # Run inference
    tasks = [engine.run_inference_async(x) for x in inputs]
    results = await asyncio.gather(*tasks)
    
    engine.cleanup()

asyncio.run(main())
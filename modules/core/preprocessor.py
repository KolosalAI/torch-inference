#!/usr/bin/env python3
"""
Enhanced Image Preprocessor with TensorRT compilation, asynchronous processing,
memory management, performance monitoring, and configuration management.

Usage Example:
    preprocessor = ImagePreprocessor(
        image_size=(224, 224),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        async_preproc=True,
        max_batch_size=16,
        max_cache_size=100,
        num_cuda_streams=2,
        compiled_module_cache="trt_module.pt"
    )
    # Synchronous call:
    tensor = preprocessor("sample.jpg")
    # Asynchronous call:
    def my_callback(result):
        print("Async result shape:", result.shape if result is not None else "Error")
    preprocessor.preprocess_async("sample.jpg", my_callback, timeout=2.0)
    # Benchmark:
    mean_time, std_time = preprocessor.benchmark("sample.jpg", iterations=10)
    print(f"Avg processing time: {mean_time:.4f}s ± {std_time:.4f}s")
    # Save configuration:
    preprocessor.save_config("preproc_config.pt")
    # Later you can load it:
    config = ImagePreprocessor.load_config("preproc_config.pt")
"""

import logging
import threading
import time
import os
import gc
from queue import Queue, Empty
from typing import Any, List, Optional, Union, Callable, Dict, Tuple
from contextlib import contextmanager, nullcontext

import numpy as np
import torchvision
import torch
import torch.nn.functional as F
import torch_tensorrt
from torchvision import transforms as T
from PIL import Image
from collections import OrderedDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BasePreprocessor:
    """
    A minimal preprocessor that returns the input data as a PyTorch tensor.
    Supports zero-copy processing for efficiency.
    """
    def __call__(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return inputs
        return torch.as_tensor(inputs, device="cuda", non_blocking=True)  # Zero-copy transfer


class TRTTransformsModule(torch.nn.Module):
    """
    A TorchScript‑compatible module that center‐crops images to match the target aspect ratio
    and then resizes and normalizes them. This module will be compiled using Torch‑TensorRT.
    """
    def __init__(self, image_size: Tuple[int, int], mean: List[float], std: List[float]):
        """
        Args:
            image_size (tuple): Target (H, W) dimensions.
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.
        """
        super().__init__()
        self.image_size = image_size  # Target (H, W)
        # Store mean and std as buffers with shape (1, C, 1, 1)
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def _center_crop_to_aspect(self, x: torch.Tensor) -> torch.Tensor:
        """
        Crop the image (or images) to match the target aspect ratio before resizing.
        Assumes input is a 4D tensor of shape (N, C, H, W).
        """
        target_h, target_w = self.image_size
        target_aspect = target_w / target_h
        N, C, H, W = x.shape
        current_aspect = W / H

        if current_aspect > target_aspect:
            # Image is too wide: crop width.
            new_w = int(H * target_aspect)
            start_x = (W - new_w) // 2
            x = x[:, :, :, start_x:start_x+new_w]
        elif current_aspect < target_aspect:
            # Image is too tall: crop height.
            new_h = int(W / target_aspect)
            start_y = (H - new_h) // 2
            x = x[:, :, start_y:start_y+new_h, :]
        # Else: aspect ratios match; no crop needed.
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        First, if necessary, center-crop the image(s) to match the target aspect ratio.
        Then, resize (or downscale/upscale) to the target image size and normalize.
        
        Args:
            x (torch.Tensor): Input tensor. Either 3D (C,H,W) or 4D (N,C,H,W).

        Returns:
            torch.Tensor: Processed tensor of shape (N, C, target_h, target_w) or (C, target_h, target_w).
        """
        is_single = x.dim() == 3
        if is_single:
            x = x.unsqueeze(0)  # Now (1, C, H, W)
        
        # First, center-crop to the target aspect ratio.
        x = self._center_crop_to_aspect(x)
        # Then, resize to the exact target dimensions.
        x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)
        # Normalize.
        x = (x - self.mean) / self.std
        
        if is_single:
            return x.squeeze(0)  # Return (C, H, W)
        return x


class ImagePreprocessor(BasePreprocessor):
    """
    Enhanced image preprocessor supporting multiple formats with optimized tensor conversion,
    asynchronous preprocessing, caching, performance monitoring, and configuration management.
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = (224, 224),
        mean: Union[List[float], float] = [0.485, 0.456, 0.406],
        std: Union[List[float], float] = [0.229, 0.224, 0.225],
        use_pinned_memory: bool = True,
        device: Union[str, torch.device] = "cuda",
        convert_grayscale: bool = True,
        trt_fp16: bool = False,  # Enable FP16 support if desired
        async_preproc: bool = False,  # Enable asynchronous preprocessing
        additional_transforms: Optional[List[Callable]] = None,  # Optional extra transforms
        compiled_module_cache: Optional[str] = None,  # Path to cache the compiled TRT module
        max_batch_size: int = 32,
        max_queue_size: int = 100,
        max_cache_size: int = 100,
        max_pinned_memory_mb: int = 512,  # Limit pinned memory usage
        num_cuda_streams: int = 1,
        fallback_to_cpu: bool = True  # Fallback to CPU if GPU/TRT fails
    ):
        """
        Args:
            image_size (int or tuple): Target image size. If int, square size is assumed.
            mean (list or float): Mean values for normalization. Can be a single numeric value or a list.
            std (list or float): Standard deviation values for normalization. Can be a single numeric value or a list.
            use_pinned_memory (bool): Whether to use pinned memory for CPU->GPU transfers.
            device (str or torch.device): Device to use for processing.
            convert_grayscale (bool): Convert grayscale images to RGB.
            trt_fp16 (bool): Enable FP16 precision in TensorRT.
            async_preproc (bool): Enable asynchronous preprocessing.
            additional_transforms (list): Additional transformations to apply.
            compiled_module_cache (str): Path to cache the compiled TensorRT module.
            max_batch_size (int): Maximum batch size for batch processing.
            max_queue_size (int): Maximum size of the async processing queue.
            max_cache_size (int): Maximum number of images to cache.
            max_pinned_memory_mb (int): Maximum size of pinned memory cache in MB.
            num_cuda_streams (int): Number of CUDA streams for parallel processing.
            fallback_to_cpu (bool): Whether to fall back to CPU processing if GPU/TRT fails.
        """
        # Process and validate mean/std parameters
        if isinstance(mean, (int, float)):
            mean = [float(mean)]
        elif not isinstance(mean, list):
            raise ValueError("mean must be a list or a numeric value for RGB")
            
        if isinstance(std, (int, float)):
            std = [float(std)]
        elif not isinstance(std, list):
            raise ValueError("std must be a list or a numeric value for RGB")

        # Expand single values to three channels
        if len(mean) == 1:
            mean = mean * 3
        elif len(mean) != 3:
            raise ValueError("mean must have either 1 or 3 elements for RGB")
            
        if len(std) == 1:
            std = std * 3
        elif len(std) != 3:
            raise ValueError("std must have either 1 or 3 elements for RGB")

        # Process image_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        elif not (isinstance(image_size, tuple) and len(image_size) == 2 and 
              all(isinstance(x, int) for x in image_size)):
            raise ValueError("image_size must be an int or a tuple of two ints")
            
        # Validate image_size bounds (within TRT supported range)
        min_bound, max_bound = 64, 1024
        if not (min_bound <= image_size[0] <= max_bound and min_bound <= image_size[1] <= max_bound):
            raise ValueError(f"image_size dimensions must be within [{min_bound}, {max_bound}]")

        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.use_pinned_memory = use_pinned_memory and torch.cuda.is_available()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.convert_grayscale = convert_grayscale
        self.async_preproc = async_preproc
        self.fallback_to_cpu = fallback_to_cpu
        
        # Validate additional transforms
        self.additional_transforms = []
        if additional_transforms:
            for t in additional_transforms:
                if not callable(t):
                    raise ValueError("All additional_transforms must be callable")
            self.additional_transforms = additional_transforms
            
        self.compiled_module_cache = compiled_module_cache
        self.max_batch_size = max(1, max_batch_size)
        self.max_pinned_memory_mb = max_pinned_memory_mb

        # Performance monitoring metrics
        self.metrics: Dict[str, List[float]] = {
            "processing_time": [], 
            "io_time": [],
            "gpu_time": []
        }

        # Image cache (simple LRU‑style using OrderedDict)
        self.max_cache_size = max_cache_size
        self.image_cache: "OrderedDict[Any, torch.Tensor]" = OrderedDict()

        # Pinned memory cache with size limit
        self.pinned_memory_cache: Dict[Tuple[Tuple[int, ...], torch.dtype], torch.Tensor] = {}
        self.pinned_memory_used = 0  # Track bytes used

        # CUDA streams for overlapping computation
        self.num_cuda_streams = max(1, num_cuda_streams) if self.device.type == 'cuda' else 0
        self.cuda_streams = [torch.cuda.Stream() for _ in range(self.num_cuda_streams)] if self.num_cuda_streams > 0 else []
        self._stream_index = 0

        # TensorRT compilation and fallback options
        self.use_trt = self.device.type == 'cuda'
        self.trt_enabled = False  # Will be set to True if compilation succeeds
        
        try:
            # Build a TorchScript‑compatible transforms module
            self.base_module = TRTTransformsModule(self.image_size, self.mean, self.std).eval()
            
            if self.use_trt:
                # Try to load or compile TRT module
                self.trt_transforms = self._load_or_compile_trt(trt_fp16)
                self.trt_enabled = True
            else:
                # CPU-only mode
                self.trt_transforms = torch.jit.script(self.base_module).to(self.device)
        except Exception as e:
            logger.warning(f"Failed to initialize TensorRT: {e}")
            if self.fallback_to_cpu:
                logger.info("Falling back to CPU processing")
                self.device = torch.device("cpu")
                self.trt_transforms = torch.jit.script(self.base_module).to(self.device)
            else:
                raise

        # Setup asynchronous preprocessing
        self._shutdown_event = threading.Event()
        self._worker = None
        if self.async_preproc:
            self.queue = Queue(maxsize=max_queue_size)
            self._start_worker()

        # Warmup
        self._warmup()

    def _load_or_compile_trt(self, trt_fp16: bool) -> torch.jit.ScriptModule:
        """
        Load cached TRT module or compile a new one.
        
        Args:
            trt_fp16 (bool): Whether to enable FP16 precision.
            
        Returns:
            torch.jit.ScriptModule: The loaded or compiled TRT module.
            
        Raises:
            RuntimeError: If TRT compilation fails and fallback_to_cpu is False.
        """
        if self.compiled_module_cache and os.path.exists(self.compiled_module_cache):
            try:
                module = torch.jit.load(self.compiled_module_cache)
                module = module.to(self.device)
                logger.info("Loaded cached TRT module from disk.")
                return module
            except Exception as e:
                logger.warning(f"Failed to load cached TRT module: {e}")
        
        # Need to compile a new module
        try:
            module = self._compile_trt_module(self.base_module.to(self.device), trt_fp16)
            
            # Try to save the compiled module if a cache path is provided
            if self.compiled_module_cache:
                try:
                    torch.jit.save(module, self.compiled_module_cache)
                    logger.info(f"Saved compiled TRT module to {self.compiled_module_cache}")
                except Exception as e:
                    logger.warning(f"Failed to save compiled TRT module: {e}")
            
            return module
            
        except Exception as e:
            logger.error(f"TensorRT compilation failed: {e}")
            if self.fallback_to_cpu:
                logger.info("Falling back to CPU JIT module")
                return torch.jit.script(self.base_module).to(torch.device("cpu"))
            else:
                raise RuntimeError(f"TensorRT compilation failed and fallback_to_cpu=False: {e}")

    def _compile_trt_module(self, module: torch.nn.Module, trt_fp16: bool) -> torch.jit.ScriptModule:
        """
        Compile a TensorRT module using torch_tensorrt with enhanced error handling.

        Args:
            module (torch.nn.Module): The module to compile.
            trt_fp16 (bool): Whether to enable FP16 precision.

        Returns:
            torch.jit.ScriptModule: The compiled TensorRT module.
            
        Raises:
            RuntimeError: If compilation fails.
        """
        logger.info(f"Compiling TensorRT module with FP16={trt_fp16}")
        try:
            # First ensure the module is on CUDA
            module = module.to(self.device)
            
            # Set up compile options
            input_sizes = {
                "min_shape": (3, 64, 64),
                "opt_shape": (3, self.image_size[0], self.image_size[1]),
                "max_shape": (3, 1024, 1024)
            }
            
            precision = {torch.float16} if trt_fp16 else {torch.float32}
            
            # Compile with TensorRT
            compiled = torch_tensorrt.compile(
                module,
                inputs=[torch_tensorrt.Input(**input_sizes)],
                enabled_precisions=precision,
                workspace_size=1 << 28,  # 256MB workspace
                min_block_size=1,        # Minimum number of ops in a TRT block
                embedding_to_mask=False  # Improves embedding lookup performance
            )
            
            logger.info("TensorRT compilation successful")
            return compiled
            
        except Exception as e:
            logger.error(f"TensorRT compilation error: {str(e)}")
            if self.fallback_to_cpu:
                logger.info("Using CPU JIT compilation instead")
                return torch.jit.script(self.base_module.cpu())
            else:
                raise RuntimeError(f"Failed to compile TRT module: {str(e)}")

    def _start_worker(self):
        """
        Start the async worker thread.
        """
        if self._worker is None or not self._worker.is_alive():
            self._shutdown_event.clear()
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()
            logger.debug("Started async worker thread")

    def _warmup(self, iterations: int = 3) -> None:
        """
        Warmup the processing pipeline by running dummy forward passes.

        Args:
            iterations (int): Number of warmup iterations.
        """
        try:
            # Create a dummy tensor matching our expected input shape
            dummy = torch.randn(3, self.image_size[0], self.image_size[1]).to(self.device)
            
            for i in range(iterations):
                with self._get_cuda_stream_context():
                    _ = self.trt_transforms(dummy)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                logger.debug(f"Warmup iteration {i+1}/{iterations} complete.")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}. This might affect initial performance.")

    @contextmanager
    def _get_cuda_stream_context(self):
        """
        Get a CUDA stream context (round‑robin) or nullcontext if not using CUDA.

        Returns:
            context manager: The current CUDA stream context or nullcontext.
        """
        if self.device.type == 'cuda' and self.cuda_streams:
            stream = self.cuda_streams[self._stream_index]
            self._stream_index = (self._stream_index + 1) % len(self.cuda_streams)
            with torch.cuda.stream(stream):
                yield
        else:
            with nullcontext():
                yield

    def _ensure_valid_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensures the tensor's spatial dimensions are within the allowed TRT range: [64, 1024].
        Upsamples if smaller than 64x64 and downsamples if larger than 1024x1024.

        Args:
            tensor (torch.Tensor): Input tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Tensor resized if necessary.
        """
        _, H, W = tensor.shape
        min_size, max_size = 64, 1024
        
        # Check if resizing is needed
        if H < min_size or W < min_size or H > max_size or W > max_size:
            new_H = max(min_size, min(H, max_size))
            new_W = max(min_size, min(W, max_size))
            
            # Use bilinear interpolation for resizing
            tensor = F.interpolate(
                tensor.unsqueeze(0), 
                size=(new_H, new_W),
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            logger.debug(f"Resized tensor from ({H}, {W}) to ({new_H}, {new_W})")
            
        return tensor

    def _get_from_cache(self, key: Any) -> Optional[torch.Tensor]:
        """
        Retrieve a processed image tensor from the cache.

        Args:
            key (Any): The key identifying the image (e.g., file path).

        Returns:
            Optional[torch.Tensor]: Cached tensor if available, None otherwise.
        """
        if key in self.image_cache:
            self.image_cache.move_to_end(key)
            logger.debug(f"Cache hit for key: {key}")
            return self.image_cache[key]
        logger.debug(f"Cache miss for key: {key}")
        return None

    def _add_to_cache(self, key: Any, tensor: torch.Tensor) -> None:
        """
        Add a processed image tensor to the cache with LRU eviction.

        Args:
            key (Any): The key identifying the image.
            tensor (torch.Tensor): Processed image tensor.
        """
        if self.max_cache_size <= 0:
            return
            
        # Add to cache
        self.image_cache[key] = tensor
        
        # Evict oldest items if cache is full
        while len(self.image_cache) > self.max_cache_size:
            removed_key, _ = self.image_cache.popitem(last=False)
            logger.debug(f"Cache full. Removed oldest item: {removed_key}")

    def _get_pinned_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Get a pinned-memory tensor from the pool if available, otherwise pin the tensor.
        Manages memory usage to stay within limits.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Pinned-memory tensor.
        """
        if not self.use_pinned_memory:
            return tensor
            
        key = (tuple(tensor.shape), tensor.dtype)
        cached = self.pinned_memory_cache.get(key)
        
        # If we have a suitable cached tensor, use it
        if cached is not None and cached.numel() >= tensor.numel():
            # Use the cached pinned tensor by copying data into it
            cached[:tensor.shape[0], :tensor.shape[1], :tensor.shape[2]] = tensor
            return cached
            
        # Need to create a new pinned tensor
        tensor_size_bytes = tensor.element_size() * tensor.numel()
        max_bytes = self.max_pinned_memory_mb * 1024 * 1024
        
        # If adding this tensor would exceed our limit, free some memory
        if self.pinned_memory_used + tensor_size_bytes > max_bytes:
            # Free memory until we have enough space
            while self.pinned_memory_cache and self.pinned_memory_used + tensor_size_bytes > max_bytes:
                # Remove a random key (first one)
                del_key = next(iter(self.pinned_memory_cache))
                del_tensor = self.pinned_memory_cache.pop(del_key)
                self.pinned_memory_used -= del_tensor.element_size() * del_tensor.numel()
                logger.debug(f"Freed pinned tensor of shape {del_key[0]}, dtype {del_key[1]}")
                del del_tensor  # Explicitly delete to free memory
            
            # Force garbage collection to release memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Create a new pinned tensor
        try:
            pinned = tensor.clone().pin_memory()
            self.pinned_memory_cache[key] = pinned
            self.pinned_memory_used += tensor_size_bytes
            logger.debug(f"Created new pinned tensor of shape {tensor.shape}, size {tensor_size_bytes/1024/1024:.2f}MB")
            return pinned
        except Exception as e:
            logger.warning(f"Failed to create pinned tensor: {e}. Using regular tensor.")
            return tensor

    def _worker_loop(self) -> None:
        """
        Worker thread for asynchronous preprocessing.
        Processes items in batches when possible for efficiency.
        Includes error handling and graceful shutdown.
        """
        logger.debug("Worker thread started")
        
        while not self._shutdown_event.is_set():
            try:
                # Get a batch of items to process
                batch = []
                callbacks = []
                
                # Try to get the first item with a timeout
                try:
                    item, callback = self.queue.get(timeout=0.5)
                    batch.append(item)
                    callbacks.append(callback)
                except Empty:
                    continue
                
                # Try to fill the batch if more items are available
                while len(batch) < self.max_batch_size:
                    try:
                        item, callback = self.queue.get_nowait()
                        batch.append(item)
                        callbacks.append(callback)
                    except Empty:
                        break
                
                logger.debug(f"Processing batch of {len(batch)} items")
                
                # Process each item and handle errors individually
                results = []
                for i, inp in enumerate(batch):
                    start_time = time.perf_counter()
                    try:
                        # Process the item
                        res = self._process_single(inp)
                        
                        # Apply additional transforms
                        for transform in self.additional_transforms:
                            try:
                                res = transform(res)
                            except Exception as e:
                                logger.error(f"Transform error: {e}")
                                res = None
                                break
                                
                        results.append(res)
                    except Exception as e:
                        logger.error(f"Processing error in worker: {e}")
                        results.append(None)
                        
                    elapsed = time.perf_counter() - start_time
                    self.metrics["processing_time"].append(elapsed)
                
                # Call each callback with its result
                for i, (result, callback) in enumerate(zip(results, callbacks)):
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                    finally:
                        self.queue.task_done()
                        
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}")
                # Brief sleep to avoid tight loop in case of persistent errors
                time.sleep(0.1)
                
        logger.debug("Worker thread stopped")

    def preprocess_async(self, inputs: Any, callback: Callable, timeout: Optional[float] = None) -> None:
        """
        Enqueue inputs for asynchronous preprocessing.
        The callback is invoked with the processed tensor once ready.

        Args:
            inputs (Any): Input image.
            callback (Callable): Function to call with the result.
            timeout (float, optional): Timeout for enqueuing the task.
            
        Raises:
            RuntimeError: If async processing is not enabled.
            ValueError: If the queue is full and timeout is reached.
        """
        if not self.async_preproc:
            raise RuntimeError("Async preprocessing not enabled. Initialize with async_preproc=True")
            
        # Restart worker if needed
        if self._worker is None or not self._worker.is_alive():
            self._start_worker()
            
        try:
            self.queue.put((inputs, callback), timeout=timeout)
            logger.debug("Task enqueued for async processing")
        except Exception as e:
            logger.error(f"Failed to enqueue task: {e}")
            try:
                callback(None)  # Notify callback about the failure
            except Exception as cb_error:
                logger.error(f"Callback error: {cb_error}")

    def __call__(self, inputs: Any) -> torch.Tensor:
        """
        Synchronous call that handles multiple input types and batch conversion.
        
        Args:
            inputs (Any): A single input or list of inputs.

        Returns:
            torch.Tensor: Processed tensor or batch of tensors.
            
        Raises:
            ValueError: If batch size exceeds maximum or processing fails.
        """
        # Handle single input and list inputs consistently
        is_batch_input = isinstance(inputs, list)
        input_list = inputs if is_batch_input else [inputs]
        
        # Validate batch size
        if len(input_list) > self.max_batch_size:
            raise ValueError(f"Batch size {len(input_list)} exceeds maximum allowed {self.max_batch_size}")
        
        # Process inputs synchronously
        processed = []
        for inp in input_list:
            start_time = time.perf_counter()
            try:
                # Process the input
                tensor = self._process_single(inp)
                
                # Apply additional transforms
                for transform in self.additional_transforms:
                    tensor = transform(tensor)
                    
                processed.append(tensor)
                
                # Record metrics
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                self.metrics["processing_time"].append(elapsed)
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                raise ValueError(f"Failed to process input: {e}")
                
        # Combine results
        if len(processed) == 1 and not is_batch_input:
            return processed[0]  # Return single tensor for single input
        else:
            # Stack tensors into a batch
            try:
                with self._get_cuda_stream_context():
                    batch_tensor = torch.stack(processed, dim=0)
                    
                    # Use pinned memory if needed
                    if self.use_pinned_memory and batch_tensor.device.type == 'cpu':
                        batch_tensor = self._get_pinned_tensor(batch_tensor)
                        
                    # Move tensor to target device with non-blocking transfer
                    batch_tensor = batch_tensor.to(self.device, non_blocking=True)
                    
                return batch_tensor
            except Exception as e:
                logger.error(f"Failed to combine processed tensors: {e}")
                raise ValueError(f"Failed to combine results: {e}")

    def _process_single(self, inp: Any) -> torch.Tensor:
        """
        Universal input handling with format detection.
        Supports file paths, PIL Images, torch.Tensors, numpy arrays, and numeric data.

        Args:
            inp (Any): Input data.

        Returns:
            torch.Tensor: Processed tensor.
            
        Raises:
            ValueError: If input format is not supported or processing fails.
        """
        cache_key = None
        
        # Handle file paths for caching
        if isinstance(inp, str):
            cache_key = inp
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached
        
        try:
            # Detect input type and process accordingly
            io_start = time.perf_counter()
            
            if isinstance(inp, str):
                # Process file path
                tensor = self._load_image_file(inp)
            elif isinstance(inp, Image.Image):
                # Process PIL image
                tensor = self._convert_pil_image(inp)
            elif isinstance(inp, np.ndarray):
                # Process NumPy array
                tensor = self._convert_numpy_array(inp)
            elif isinstance(inp, torch.Tensor):
                # Process torch tensor
                tensor = self._convert_tensor(inp)
            else:
                # Try to convert other types
                try:
                    tensor = torch.as_tensor(inp, device=self.device)
                except Exception as e:
                    raise ValueError(f"Unsupported input type: {type(inp)}")
            
            io_time = time.perf_counter() - io_start
            self.metrics["io_time"].append(io_time)
            
            # Cache the result if applicable
            if cache_key is not None:
                self._add_to_cache(cache_key, tensor)
                
            return tensor
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise ValueError(f"Failed to process input: {str(e)}")

    def _load_image_file(self, filepath: str) -> torch.Tensor:
        """
        Load image file using torchvision with error handling.

        Args:
            filepath (str): Path to the image file.

        Returns:
            torch.Tensor: Processed image tensor.
            
        Raises:
            ValueError: If file can't be loaded or processed.
        """
        if not os.path.exists(filepath):
            raise ValueError(f"Image file not found: {filepath}")
            
        try:
            # Try using torchvision's optimized reader first
            gpu_start = time.perf_counter()
            
            try:
                img_tensor = torchvision.io.read_image(filepath)
                
                # Convert to float and normalize to [0, 1]
                img_tensor = img_tensor.to(dtype=torch.float32) / 255.0
                
                # Handle grayscale conversion if needed
                if self.convert_grayscale and img_tensor.shape[0] == 1:
                    img_tensor = img_tensor.repeat(3, 1, 1)
                
                # Validate/adjust shape for TensorRT
                img_tensor = self._ensure_valid_shape(img_tensor)
                
                # Move to target device
                img_tensor = img_tensor.to(self.device, non_blocking=True)
                
                # Apply TRT transforms
                with self._get_cuda_stream_context():
                    result = self.trt_transforms(img_tensor)
                    
                gpu_time = time.perf_counter() - gpu_start
                self.metrics["gpu_time"].append(gpu_time)
                
                return result
                
            except Exception as e:
                # Fallback to PIL if torchvision fails
                logger.warning(f"torchvision.io.read_image failed: {e}, falling back to PIL")
                img = Image.open(filepath)
                return self._convert_pil_image(img)
                
        except Exception as e:
            logger.error(f"Image loading failed: {str(e)}")
            raise ValueError(f"Failed to load image file {filepath}: {str(e)}")

    def _convert_pil_image(self, image: Image.Image) -> torch.Tensor:
        """
        Convert PIL image to normalized tensor.

        Args:
            image (PIL.Image.Image): Input PIL image.

        Returns:
            torch.Tensor: Processed image tensor.
        """
        try:
            # Convert grayscale to RGB if needed
            if image.mode != 'RGB' and self.convert_grayscale:
                image = image.convert('RGB')
                
            # Convert to tensor (values in [0, 1])
            tensor = T.functional.to_tensor(image)
            
            # Validate and adjust shape
            tensor = self._ensure_valid_shape(tensor)
            
            # Move to target device
            tensor = tensor.to(self.device, non_blocking=True)
            
            # Apply TRT transforms
            gpu_start = time.perf_counter()
            with self._get_cuda_stream_context():
                result = self.trt_transforms(tensor)
            gpu_time = time.perf_counter() - gpu_start
            self.metrics["gpu_time"].append(gpu_time)
            
            return result
            
        except Exception as e:
            logger.error(f"PIL conversion failed: {str(e)}")
            raise ValueError(f"Failed to process PIL image: {str(e)}")

    def _convert_numpy_array(self, array: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy array to a normalized tensor.

        Args:
            array (np.ndarray): Input numpy array.

        Returns:
            torch.Tensor: Processed image tensor.
        """
        try:
            # Handle different input shapes
            if array.ndim == 2:
                # Single channel image (H,W)
                array = array[:, :, np.newaxis]
                
            if array.ndim == 3:
                # (H,W,C) to (C,H,W)
                if array.shape[2] in (1, 3, 4):
                    array = np.transpose(array, (2, 0, 1))
                    
                    # Handle RGBA by taking only RGB channels
                    if array.shape[0] == 4:
                        array = array[:3]
            
            # Convert to tensor
            tensor = torch.as_tensor(array, device=self.device, dtype=torch.float32)
            
            # Normalize to [0, 1] if needed
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
                
            # Handle single channel
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
                
            # Convert grayscale to RGB if needed
            if self.convert_grayscale and tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
                
            # Validate and adjust shape
            tensor = self._ensure_valid_shape(tensor)
            
            # Apply TRT transforms
            gpu_start = time.perf_counter()
            with self._get_cuda_stream_context():
                result = self.trt_transforms(tensor)
            gpu_time = time.perf_counter() - gpu_start
            self.metrics["gpu_time"].append(gpu_time)
            
            return result
            
        except Exception as e:
            logger.error(f"NumPy conversion failed: {str(e)}")
            raise ValueError(f"Failed to process NumPy array: {str(e)}")

    def _convert_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize existing tensors.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Processed image tensor.
        """
        try:
            # Handle various input tensor formats
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
                
            # Ensure correct device placement
            if tensor.device != self.device:
                tensor = tensor.to(self.device, non_blocking=True)
                
            # Convert to float32 and normalize to [0, 1] if needed
            if tensor.dtype != torch.float32:
                tensor = tensor.to(dtype=torch.float32)
                if tensor.max() > 1.0:
                    tensor /= 255.0
                    
            # Convert grayscale to RGB if needed
            if self.convert_grayscale and tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
                
            # Validate and adjust shape
            tensor = self._ensure_valid_shape(tensor)
            
            # Apply TRT transforms
            gpu_start = time.perf_counter()
            with self._get_cuda_stream_context():
                result = self.trt_transforms(tensor)
            gpu_time = time.perf_counter() - gpu_start
            self.metrics["gpu_time"].append(gpu_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Tensor conversion failed: {str(e)}")
            raise ValueError(f"Failed to process tensor: {str(e)}")

    def save_config(self, path: str) -> None:
        """
        Save the preprocessor configuration to disk.

        Args:
            path (str): Path to save the configuration.
        """
        try:
            config = {
                'image_size': self.image_size,
                'mean': self.mean,
                'std': self.std,
                'use_pinned_memory': self.use_pinned_memory,
                'device': str(self.device),
                'convert_grayscale': self.convert_grayscale,
                'trt_enabled': self.trt_enabled,
                'async_preproc': self.async_preproc,
                'max_batch_size': self.max_batch_size,
                'max_queue_size': getattr(self, 'queue', None) and self.queue.maxsize,
                'max_cache_size': self.max_cache_size,
                'max_pinned_memory_mb': self.max_pinned_memory_mb,
                'num_cuda_streams': self.num_cuda_streams,
                'fallback_to_cpu': self.fallback_to_cpu,
                # Don't save callbacks or other non-serializable objects
            }
            
            torch.save(config, path)
            logger.info(f"Configuration saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    @classmethod
    def load_config(cls, path: str) -> dict:
        """
        Load a configuration from disk.

        Args:
            path (str): Path to the configuration file.

        Returns:
            dict: The loaded configuration.
            
        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If the config file can't be loaded.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        try:
            config = torch.load(path)
            logger.info(f"Configuration loaded from {path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ValueError(f"Failed to load configuration from {path}: {e}")

    @classmethod
    def from_config(cls, config_path: str, **kwargs) -> 'ImagePreprocessor':
        """
        Create a new ImagePreprocessor instance from a saved configuration.
        
        Args:
            config_path (str): Path to the configuration file.
            **kwargs: Optional overrides for the loaded configuration.
            
        Returns:
            ImagePreprocessor: A new preprocessor instance.
        """
        config = cls.load_config(config_path)
        # Override loaded config with any provided kwargs
        config.update(kwargs)
        return cls(**config)

    def benchmark(self, sample_input: Any, iterations: int = 100, warmup: int = 5) -> Tuple[float, float, dict]:
        """
        Benchmark the preprocessing time with detailed metrics.

        Args:
            sample_input (Any): Sample input (file path, PIL image, etc.).
            iterations (int): Number of iterations.
            warmup (int): Number of warmup iterations.

        Returns:
            Tuple[float, float, dict]: Mean time, std deviation, and detailed metrics.
        """
        try:
            # Clear existing metrics
            for k in self.metrics:
                self.metrics[k] = []
                
            # Convert input to list if it's not already
            inputs = sample_input if isinstance(sample_input, list) else [sample_input]
            
            # Warmup
            for _ in range(warmup):
                _ = self(inputs)
                
            # Benchmark
            times = []
            for i in range(iterations):
                start = time.perf_counter()
                _ = self(inputs)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                logger.debug(f"Iteration {i+1}/{iterations}: {elapsed:.4f}s")
                
            # Calculate statistics
            mean_time = np.mean(times)
            std_time = np.std(times)
            
            # Gather detailed metrics
            metrics = {}
            for k, v in self.metrics.items():
                if v:  # Only include metrics that have data
                    metrics[k] = {
                        'mean': float(np.mean(v)),
                        'std': float(np.std(v)),
                        'min': float(np.min(v)),
                        'max': float(np.max(v)),
                    }
                    
            logger.info(f"Benchmark results: {iterations} iterations, mean={mean_time:.4f}s ± {std_time:.4f}s")
            return mean_time, std_time, metrics
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise

    def cancel_pending_tasks(self) -> None:
        """
        Cancel any pending asynchronous tasks.
        """
        if self.async_preproc and hasattr(self, 'queue'):
            try:
                with self.queue.mutex:
                    remaining = self.queue.qsize()
                    self.queue.queue.clear()
                logger.info(f"Cancelled {remaining} pending async tasks")
            except Exception as e:
                logger.error(f"Failed to cancel pending tasks: {e}")

    def cleanup(self) -> None:
        """
        Cleanup resources: shutdown worker thread, clear caches, and free CUDA memory.
        """
        # Signal worker thread to shut down
        self._shutdown_event.set()
        
        # Wait for worker thread to finish
        if self._worker is not None and self._worker.is_alive():
            logger.debug("Waiting for worker thread to finish...")
            self._worker.join(timeout=2)
            if self._worker.is_alive():
                logger.warning("Worker thread did not terminate cleanly")
                
        # Clear caches
        self.image_cache.clear()
        self.pinned_memory_cache.clear()
        self.pinned_memory_used = 0
        
        # Release metrics data
        for k in self.metrics:
            self.metrics[k] = []
            
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Cleanup complete. Resources released.")

    def __del__(self):
        """
        Ensure resources are released when the object is deleted.
        """
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    # Example unit tests and benchmarking.
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # Create a preprocessor with improved parameters
    try:
        preproc = ImagePreprocessor(
            image_size=(224, 224),
            mean=[0.485],  # Single value will be expanded to [0.485, 0.485, 0.485]
            std=[0.229],   # Single value will be expanded to [0.229, 0.229, 0.229]
            async_preproc=True,
            max_batch_size=4,
            max_queue_size=10,
            max_cache_size=50,
            num_cuda_streams=2,
            fallback_to_cpu=True,
            compiled_module_cache="trt_module_cache.pt"  # Enable caching
        )
        logger.info("Preprocessor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize preprocessor: {e}")
        sys.exit(1)

    # Test PIL image processing
    try:
        img = Image.new("RGB", (300, 300), color="red")
        tensor = preproc(img)
        logger.info(f"Synchronous PIL processing: output shape {tensor.shape}, device {tensor.device}")
    except Exception as e:
        logger.error(f"Error during synchronous PIL processing: {e}")

    # Test async processing with NumPy array
    def async_callback(result):
        if result is not None:
            logger.info(f"Asynchronous NumPy processing: output shape {result.shape}, device {result.device}")
        else:
            logger.error("Asynchronous processing failed.")

    # Create a test NumPy array
    np_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    preproc.preprocess_async(np_img, async_callback, timeout=2.0)

    # Test batch processing
    try:
        images = [Image.new("RGB", (300, 300), color=c) for c in ["red", "green", "blue"]]
        batch = preproc(images)
        logger.info(f"Batch processing: output shape {batch.shape}, device {batch.device}")
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")

    # Run benchmark
    try:
        mean_time, std_time, metrics = preproc.benchmark(np_img, iterations=5, warmup=2)
        logger.info(f"Benchmark result: {mean_time:.4f}s ± {std_time:.4f}s")
        logger.info(f"Detailed metrics: {metrics}")
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")

    # Test config saving and loading
    config_path = "preproc_config.pt"
    try:
        preproc.save_config(config_path)
        loaded_config = ImagePreprocessor.load_config(config_path)
        logger.info(f"Loaded config: {loaded_config}")
    except Exception as e:
        logger.error(f"Config saving/loading failed: {e}")

    # Create a new preprocessor from the loaded config
    try:
        new_preproc = ImagePreprocessor.from_config(config_path, device="cpu")
        logger.info(f"Created new preprocessor from config with device {new_preproc.device}")
    except Exception as e:
        logger.error(f"Failed to create preprocessor from config: {e}")

    # Clean up resources
    preproc.cleanup()
    logger.info("Tests completed")

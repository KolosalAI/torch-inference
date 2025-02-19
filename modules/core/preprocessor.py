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
from queue import Queue, Empty
from typing import Any, List, Optional, Union, Callable, Dict, Tuple

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
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Now (1, C, H, W)
        # First, center-crop to the target aspect ratio.
        x = self._center_crop_to_aspect(x)
        # Then, resize to the exact target dimensions.
        x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)
        # Normalize.
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        if x.shape[0] == 1:
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
        num_cuda_streams: int = 1
    ):
        """
        Args:
            image_size (int or tuple): Target image size. If int, square size is assumed.
            mean (list or float): Mean values for normalization. Can be a single numeric value or a list.
            std (list or float): Standard deviation values for normalization. Can be a single numeric value or a list.
            ...
        """
        # If mean or std are not lists, but single numbers, convert them into lists.
        if not isinstance(mean, list):
            if isinstance(mean, (int, float)):
                mean = [float(mean)]
            else:
                raise ValueError("mean must be a list or a numeric value for RGB")
        if not isinstance(std, list):
            if isinstance(std, (int, float)):
                std = [float(std)]
            else:
                raise ValueError("std must be a list or a numeric value for RGB")

        # Validate mean: allow either one value (to be repeated) or exactly three values.
        if not (len(mean) == 3 or len(mean) == 1):
            raise ValueError("mean must be a list with either 1 or 3 elements for RGB")
        if len(mean) == 1:
            mean = mean * 3  # Expand single value to three elements.

        # Validate std similarly.
        if not (len(std) == 3 or len(std) == 1):
            raise ValueError("std must be a list with either 1 or 3 elements for RGB")
        if len(std) == 1:
            std = std * 3  # Expand single value to three elements.

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        elif (isinstance(image_size, tuple) and len(image_size) == 2 and 
              all(isinstance(x, int) for x in image_size)):
            pass
        else:
            raise ValueError("image_size must be an int or a tuple of two ints")
        # Validate image_size bounds (within TRT supported range)
        min_bound, max_bound = 64, 1024
        if not (min_bound <= image_size[0] <= max_bound and min_bound <= image_size[1] <= max_bound):
            raise ValueError(f"image_size dimensions must be within [{min_bound}, {max_bound}]")

        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.use_pinned_memory = use_pinned_memory
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.convert_grayscale = convert_grayscale
        self.async_preproc = async_preproc
        self.additional_transforms = additional_transforms or []
        # Validate that additional_transforms are callables
        for t in self.additional_transforms:
            if not callable(t):
                raise ValueError("All additional_transforms must be callable")
        self.compiled_module_cache = compiled_module_cache
        self.max_batch_size = max_batch_size

        # Performance monitoring metrics
        self.metrics: Dict[str, List[float]] = {"processing_time": []}

        # Image cache (simple LRU‑style using OrderedDict)
        self.max_cache_size = max_cache_size
        self.image_cache: "OrderedDict[Any, torch.Tensor]" = OrderedDict()

        # Simple pinned memory pool cache (keyed by (shape, dtype))
        self.pinned_memory_cache: Dict[Tuple[Tuple[int, ...], torch.dtype], torch.Tensor] = {}

        # CUDA streams for overlapping computation
        self.num_cuda_streams = max(1, num_cuda_streams)
        self.cuda_streams = [torch.cuda.Stream() for _ in range(self.num_cuda_streams)] if self.device.type == 'cuda' else []
        self._stream_index = 0

        # Build a TorchScript‑compatible transforms module on the target device.
        trt_module = TRTTransformsModule(self.image_size, self.mean, self.std).eval().to(self.device)
        # Compile or load cached TRT module.
        if self.compiled_module_cache:
            try:
                self.trt_transforms = torch.jit.load(self.compiled_module_cache).to(self.device)
                logger.info("Loaded cached TRT module from disk.")
            except Exception as e:
                logger.info("No valid cached TRT module found; compiling a new one.")
                self.trt_transforms = self._compile_trt_module(trt_module, trt_fp16)
                try:
                    torch.jit.save(self.trt_transforms, self.compiled_module_cache)
                except Exception as save_e:
                    logger.error(f"Failed to save compiled module: {save_e}")
        else:
            self.trt_transforms = self._compile_trt_module(trt_module, trt_fp16)

        # Setup asynchronous preprocessing if enabled.
        self._shutdown_event = threading.Event()
        if self.async_preproc:
            self.queue = Queue(maxsize=max_queue_size)
            self.worker = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker.start()

        # Warmup the TRT engine for a few iterations to optimize batch sizes.
        self._warmup_trt_engine()

    def _compile_trt_module(self, module: torch.nn.Module, trt_fp16: bool) -> torch.jit.ScriptModule:
        """
        Compile a TRT module using torch_tensorrt.

        Args:
            module (torch.nn.Module): The module to compile.
            trt_fp16 (bool): Whether to enable FP16.

        Returns:
            torch.jit.ScriptModule: The compiled module.
        """
        try:
            compiled = torch_tensorrt.compile(
                module,
                inputs=[
                    torch_tensorrt.Input(
                        min_shape=(3, 64, 64),
                        opt_shape=(3, self.image_size[0], self.image_size[1]),
                        max_shape=(3, 1024, 1024),
                    )
                ],
                enabled_precisions={torch.float16} if trt_fp16 else {torch.float32},
            )
            return compiled
        except Exception as e:
            logger.error(f"TensorRT compilation failed: {e}")
            raise

    def _warmup_trt_engine(self, iterations: int = 3) -> None:
        """
        Warmup the TRT engine by running dummy forward passes.

        Args:
            iterations (int): Number of warmup iterations.
        """
        dummy = torch.randn(3, self.image_size[0], self.image_size[1]).to(self.device)
        for i in range(iterations):
            with self._get_cuda_stream_context():
                _ = self.trt_transforms(dummy)
            logger.debug(f"Warmup iteration {i+1}/{iterations} complete.")

    def _get_cuda_stream_context(self):
        """
        Get a CUDA stream context (round‑robin).

        Returns:
            context manager: The current CUDA stream context.
        """
        if self.device.type == 'cuda' and self.cuda_streams:
            stream = self.cuda_streams[self._stream_index]
            self._stream_index = (self._stream_index + 1) % len(self.cuda_streams)
            return torch.cuda.stream(stream)
        else:
            from contextlib import nullcontext
            return nullcontext()

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
        min_size = 64
        max_size = 1024
        new_H = min_size if H < min_size else (max_size if H > max_size else H)
        new_W = min_size if W < min_size else (max_size if W > max_size else W)
        if new_H != H or new_W != W:
            tensor = F.interpolate(tensor.unsqueeze(0), size=(new_H, new_W),
                                   mode='bilinear', align_corners=False).squeeze(0)
        return tensor

    def _get_from_cache(self, key: Any) -> Optional[torch.Tensor]:
        """
        Retrieve a processed image tensor from the cache.

        Args:
            key (Any): The key identifying the image (e.g., file path).

        Returns:
            Optional[torch.Tensor]: Cached tensor if available.
        """
        if key in self.image_cache:
            self.image_cache.move_to_end(key)
            logger.debug("Image retrieved from cache.")
            return self.image_cache[key]
        return None

    def _add_to_cache(self, key: Any, tensor: torch.Tensor) -> None:
        """
        Add a processed image tensor to the cache.

        Args:
            key (Any): The key identifying the image.
            tensor (torch.Tensor): Processed image tensor.
        """
        self.image_cache[key] = tensor
        if len(self.image_cache) > self.max_cache_size:
            removed = self.image_cache.popitem(last=False)
            logger.debug(f"Cache full. Removed oldest cached item: {removed[0]}")

    def _get_pinned_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Get a pinned-memory tensor from the pool if available, otherwise pin the tensor.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Pinned-memory tensor.
        """
        key = (tuple(tensor.shape), tensor.dtype)
        cached = self.pinned_memory_cache.get(key)
        if cached is not None and cached.numel() >= tensor.numel():
            cached.copy_(tensor)
            return cached
        else:
            pinned = tensor.clone().pin_memory()
            self.pinned_memory_cache[key] = pinned
            return pinned

    def _worker_loop(self) -> None:
        """
        Worker thread for asynchronous preprocessing.
        Processes items in batches if available.
        """
        while not self._shutdown_event.is_set():
            batch = []
            callbacks = []
            try:
                item, callback = self.queue.get(timeout=0.5)
                batch.append(item)
                callbacks.append(callback)
            except Empty:
                continue

            while len(batch) < self.max_batch_size:
                try:
                    item, callback = self.queue.get_nowait()
                    batch.append(item)
                    callbacks.append(callback)
                except Empty:
                    break

            results = []
            for inp in batch:
                start_time = time.perf_counter()
                try:
                    res = self._process_single(inp)
                    for t in self.additional_transforms:
                        res = t(res)
                    results.append(res)
                except Exception as e:
                    logger.error(f"Async preprocessing failed: {str(e)}")
                    results.append(None)
                elapsed = time.perf_counter() - start_time
                self.metrics["processing_time"].append(elapsed)

            for res, cb in zip(results, callbacks):
                cb(res)
                self.queue.task_done()

    def preprocess_async(self, inputs: Any, callback: Callable, timeout: Optional[float] = None) -> None:
        """
        Enqueue inputs for asynchronous preprocessing.
        The callback is invoked with the processed tensor once ready.

        Args:
            inputs (Any): Input image.
            callback (Callable): Function to call with the result.
            timeout (float, optional): Timeout for enqueuing the task.
        """
        try:
            self.queue.put((inputs, callback), timeout=timeout)
        except Exception as e:
            logger.error(f"Failed to enqueue async preprocessing task: {e}")
            callback(None)

    def __call__(self, inputs: Any) -> torch.Tensor:
        """
        Synchronous call that handles multiple input types and batch conversion.
        If asynchronous preprocessing is enabled, this will wait for the results.
        Applies CUDA stream contexts for overlapping computation.

        Args:
            inputs (Any): A single input or list of inputs.

        Returns:
            torch.Tensor: Batched processed tensor.
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) > self.max_batch_size:
            raise ValueError(f"Batch size {len(inputs)} exceeds maximum allowed {self.max_batch_size}")

        if self.async_preproc:
            results = [None] * len(inputs)
            events = [threading.Event() for _ in range(len(inputs))]

            def _make_callback(index):
                def _callback(result):
                    results[index] = result
                    events[index].set()
                return _callback

            for idx, inp in enumerate(inputs):
                self.preprocess_async(inp, _make_callback(idx), timeout=2.0)
            for ev in events:
                ev.wait()
            batch_tensor = torch.stack(results, dim=0)
        else:
            processed = []
            for inp in inputs:
                start_time = time.perf_counter()
                tensor = self._process_single(inp)
                for t in self.additional_transforms:
                    tensor = t(tensor)
                processed.append(tensor)
                self.metrics["processing_time"].append(time.perf_counter() - start_time)
            batch_tensor = torch.stack(processed, dim=0)

        with self._get_cuda_stream_context():
            batch_tensor = batch_tensor.to(self.device, non_blocking=True)

        if self.use_pinned_memory and batch_tensor.device.type == 'cpu':
            batch_tensor = self._get_pinned_tensor(batch_tensor)
        return batch_tensor

    def _process_single(self, inp: Any) -> torch.Tensor:
        """
        Universal input handling with format detection.
        Supports file paths, PIL Images, torch.Tensors, numpy arrays, and numeric data.

        Args:
            inp (Any): Input data.

        Returns:
            torch.Tensor: Processed tensor.
        """
        cache_key = None
        try:
            if isinstance(inp, str):
                cache_key = inp
                cached = self._get_from_cache(cache_key)
                if cached is not None:
                    return cached
                tensor = self._load_image_file(inp)
            elif isinstance(inp, Image.Image):
                tensor = self._convert_pil_image(inp)
            elif isinstance(inp, np.ndarray):
                tensor = self._convert_numpy_array(inp)
            elif isinstance(inp, torch.Tensor):
                tensor = self._convert_tensor(inp)
            else:
                tensor = torch.as_tensor(inp, device=self.device)
            if cache_key is not None:
                self._add_to_cache(cache_key, tensor)
            return tensor
        except Exception as e:
            logger.error(f"Image conversion failed: {str(e)}")
            raise

    def _load_image_file(self, filepath: str) -> torch.Tensor:
        """
        Load image file using torchvision with format detection.

        Args:
            filepath (str): Path to the image file.

        Returns:
            torch.Tensor: Normalized image tensor.
        """
        img_tensor = torchvision.io.read_image(filepath)
        if self.convert_grayscale and img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        img_tensor = img_tensor.to(dtype=torch.float32) / 255.0
        img_tensor = self._ensure_valid_shape(img_tensor)
        img_tensor = img_tensor.to(self.device, non_blocking=True)
        with self._get_cuda_stream_context():
            result = self.trt_transforms(img_tensor)
        return result

    def _convert_pil_image(self, image: Image.Image) -> torch.Tensor:
        """
        Convert PIL image to normalized tensor.

        Args:
            image (PIL.Image.Image): Input PIL image.

        Returns:
            torch.Tensor: Normalized image tensor.
        """
        if image.mode != 'RGB' and self.convert_grayscale:
            image = image.convert('RGB')
        tensor = T.functional.to_tensor(image)
        tensor = self._ensure_valid_shape(tensor)
        tensor = tensor.to(self.device, non_blocking=True)
        with self._get_cuda_stream_context():
            result = self.trt_transforms(tensor)
        return result

    def _convert_numpy_array(self, array: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy array to a normalized tensor.

        Args:
            array (np.ndarray): Input numpy array.

        Returns:
            torch.Tensor: Normalized image tensor.
        """
        if array.ndim == 3 and array.shape[2] in (1, 3):
            array = np.transpose(array, (2, 0, 1))
        tensor = torch.as_tensor(array, device=self.device, dtype=torch.float32)
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if self.convert_grayscale and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        tensor = self._ensure_valid_shape(tensor)
        with self._get_cuda_stream_context():
            result = self.trt_transforms(tensor)
        return result

    def _convert_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize existing tensors.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized image tensor.
        """
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)
            if tensor.max() > 1.0:
                tensor /= 255.0
        if self.convert_grayscale and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        tensor = self._ensure_valid_shape(tensor)
        with self._get_cuda_stream_context():
            result = self.trt_transforms(tensor)
        return result

    def save_config(self, path: str) -> None:
        """
        Save the preprocessor configuration to disk.

        Args:
            path (str): Path to save the configuration.
        """
        config = {
            'image_size': self.image_size,
            'mean': self.mean,
            'std': self.std,
            'use_pinned_memory': self.use_pinned_memory,
            'device': str(self.device),
            'convert_grayscale': self.convert_grayscale,
            'trt_fp16': self.trt_transforms is not None and any(p.dtype == torch.float16 for p in self.trt_transforms.parameters()),
            'async_preproc': self.async_preproc,
            'max_batch_size': self.max_batch_size,
            'max_queue_size': self.queue.maxsize if self.async_preproc else None,
            'max_cache_size': self.max_cache_size,
            'num_cuda_streams': self.num_cuda_streams,
        }
        torch.save(config, path)
        logger.info(f"Configuration saved to {path}")

    @staticmethod
    def load_config(path: str) -> dict:
        """
        Load a configuration from disk.

        Args:
            path (str): Path to the configuration file.

        Returns:
            dict: The loaded configuration.
        """
        config = torch.load(path)
        logger.info(f"Configuration loaded from {path}")
        return config

    def benchmark(self, sample_input: Any, iterations: int = 100) -> Tuple[float, float]:
        """
        Benchmark the preprocessing time.

        Args:
            sample_input (Any): Sample input (file path, PIL image, etc.).
            iterations (int): Number of iterations.

        Returns:
            Tuple[float, float]: Mean and standard deviation of processing time.
        """
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = self(sample_input)
            times.append(time.perf_counter() - start)
        mean_time = np.mean(times)
        std_time = np.std(times)
        logger.info(f"Benchmark: {iterations} iterations, mean={mean_time:.4f}s, std={std_time:.4f}s")
        return mean_time, std_time

    def cancel_pending_tasks(self) -> None:
        """
        Cancel any pending asynchronous tasks.
        """
        if self.async_preproc:
            with self.queue.mutex:
                self.queue.queue.clear()
            logger.info("Pending async tasks cancelled.")

    def cleanup(self) -> None:
        """
        Cleanup resources: shutdown worker thread, clear caches, and free CUDA memory.
        """
        self._shutdown_event.set()
        if self.async_preproc:
            self.worker.join(timeout=2)
        self.image_cache.clear()
        self.pinned_memory_cache.clear()
        torch.cuda.empty_cache()
        logger.info("Cleanup complete. Resources released.")


if __name__ == "__main__":
    # Example unit tests and benchmarking.
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    preproc = ImagePreprocessor(
        image_size=(224, 224),
        mean=[0.485],  # Single value will be expanded to [0.485, 0.485, 0.485]
        std=[0.229],   # Single value will be expanded to [0.229, 0.229, 0.229]
        async_preproc=True,
        max_batch_size=4,
        max_queue_size=10,
        max_cache_size=50,
        num_cuda_streams=2,
        compiled_module_cache=None  # Set to a file path to enable caching
    )

    try:
        img = Image.new("RGB", (300, 300), color="red")
        tensor = preproc(img)
        logger.info(f"Synchronous PIL processing: output shape {tensor.shape}")
    except Exception as e:
        logger.error(f"Error during synchronous PIL processing: {e}")

    def async_callback(result):
        if result is not None:
            logger.info(f"Asynchronous NumPy processing: output shape {result.shape}")
        else:
            logger.error("Asynchronous processing failed.")

    np_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    preproc.preprocess_async(np_img, async_callback, timeout=2.0)

    try:
        mean_time, std_time = preproc.benchmark(np_img, iterations=5)
        logger.info(f"Benchmark result: {mean_time:.4f}s ± {std_time:.4f}s")
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")

    config_path = "preproc_config.pt"
    preproc.save_config(config_path)
    loaded_config = ImagePreprocessor.load_config(config_path)
    logger.info(f"Loaded config: {loaded_config}")

    preproc.cleanup()

import logging
import threading
from queue import Queue, Empty
from typing import Any, List, Optional, Union, Callable

import torchvision
import torch
import torch.nn.functional as F
import torch_tensorrt
from torchvision import transforms as T
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    A TorchScript‑compatible module that resizes and normalizes images.
    This module will be compiled using Torch‑TensorRT.
    """
    def __init__(self, image_size: tuple, mean: List[float], std: List[float]):
        super().__init__()
        self.image_size = image_size  # Target (H, W)
        # Store mean and std as buffers with shape (1, C, 1, 1)
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resizes and normalizes the image(s).
        If input is 3D (unbatched), process as a single image.
        If input is 4D (batched), process all images directly.
        """
        if x.dim() == 3:
            # Input shape: (C, H, W)
            x = x.unsqueeze(0)  # Now (1, C, H, W)
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
            return x.squeeze(0)  # Return (C, H, W)
        elif x.dim() == 4:
            # Input shape: (N, C, H, W)
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
            return x
        else:
            raise RuntimeError("Input tensor must be 3D (C, H, W) or 4D (N, C, H, W)")



class ImagePreprocessor(BasePreprocessor):
    """
    Enhanced image preprocessor supporting multiple formats with optimized tensor conversion.
    Uses a Torch‑TensorRT–compiled module for resizing and normalization.
    Now includes asynchronous pre‑processing and caching support.
    """
    def __init__(
        self,
        image_size: Union[int, tuple] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        use_pinned_memory: bool = True,
        device: Union[str, torch.device] = "cuda",
        convert_grayscale: bool = True,
        trt_fp16: bool = False,  # Enable FP16 support if desired
        async_preproc: bool = False,  # Enable asynchronous preprocessing
        additional_transforms: Optional[List[Callable]] = None,  # Optional extra transforms
        compiled_module_cache: Optional[str] = None  # Path to cache the compiled module
    ):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.use_pinned_memory = use_pinned_memory
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.convert_grayscale = convert_grayscale
        self.async_preproc = async_preproc
        self.additional_transforms = additional_transforms or []
        self.compiled_module_cache = compiled_module_cache

        # Build a TorchScript-compatible transforms module on the target device.
        trt_module = TRTTransformsModule(self.image_size, self.mean, self.std).eval().to(self.device)

        # Check for cached compiled module if provided.
        if self.compiled_module_cache:
            try:
                self.trt_transforms = torch.jit.load(self.compiled_module_cache).to(self.device)
                logger.info("Loaded cached TRT module from disk.")
            except Exception as e:
                logger.info("No valid cached TRT module found; compiling a new one.")
                self.trt_transforms = self._compile_trt_module(trt_module, trt_fp16)
                torch.jit.save(self.trt_transforms, self.compiled_module_cache)
        else:
            self.trt_transforms = self._compile_trt_module(trt_module, trt_fp16)

        # Setup asynchronous preprocessing if enabled.
        if self.async_preproc:
            self.queue = Queue(maxsize=10)
            self.worker = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker.start()

    def _compile_trt_module(self, module: torch.nn.Module, trt_fp16: bool) -> torch.jit.ScriptModule:
        return torch_tensorrt.compile(
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

    def _ensure_valid_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensures the tensor's spatial dimensions are within the allowed TRT range: [64, 1024].
        Upsamples if smaller than 64x64 and downsamples if larger than 1024x1024.
        """
        _, H, W = tensor.shape
        min_size = 64
        max_size = 1024
        new_H = H
        new_W = W
        if H < min_size:
            new_H = min_size
        elif H > max_size:
            new_H = max_size
        if W < min_size:
            new_W = min_size
        elif W > max_size:
            new_W = max_size
        if new_H != H or new_W != W:
            tensor = F.interpolate(tensor.unsqueeze(0), size=(new_H, new_W),
                                   mode='bilinear', align_corners=False).squeeze(0)
        return tensor

    def _worker_loop(self):
        """Worker thread for asynchronous preprocessing."""
        while True:
            try:
                # Wait for an image input.
                input_item, callback = self.queue.get(timeout=1)
            except Empty:
                continue  # Check for shutdown signal if needed.
            try:
                result = self._process_single(input_item)
                # Apply additional transforms if any.
                for t in self.additional_transforms:
                    result = t(result)
                callback(result)
            except Exception as e:
                logger.error(f"Async preprocessing failed: {str(e)}")
                callback(None)
            finally:
                self.queue.task_done()

    def preprocess_async(self, inputs: Any, callback: Callable):
        """
        Enqueue inputs for asynchronous preprocessing.
        The callback is invoked with the processed tensor once ready.
        """
        self.queue.put((inputs, callback))

    def __call__(self, inputs: Any) -> torch.Tensor:
        """
        Synchronous call that handles multiple input types and batch conversion.
        If asynchronous preprocessing is enabled, this will wait for the results.
        """
        if self.async_preproc:
            results = []
            events = []

            def _make_callback(index):
                def _callback(result):
                    results.insert(index, result)
                    events[index].set()
                return _callback

            if not isinstance(inputs, list):
                inputs = [inputs]

            events = [threading.Event() for _ in range(len(inputs))]
            for idx, inp in enumerate(inputs):
                self.preprocess_async(inp, _make_callback(idx))
            # Wait for all events to complete.
            for ev in events:
                ev.wait()
            batch_tensor = torch.stack(results, dim=0)
        else:
            if not isinstance(inputs, list):
                inputs = [inputs]
            processed = [self._process_single(inp) for inp in inputs]
            # Apply additional transforms if provided.
            for i, tensor in enumerate(processed):
                for t in self.additional_transforms:
                    tensor = t(tensor)
                processed[i] = tensor
            batch_tensor = torch.stack(processed, dim=0)

        batch_tensor = batch_tensor.to(self.device, non_blocking=True)
        if self.use_pinned_memory and batch_tensor.device.type == 'cpu':
            batch_tensor = batch_tensor.pin_memory()
        return batch_tensor


    def _process_single(self, inp: Any) -> torch.Tensor:
        """Universal input handling with format detection."""
        try:
            if isinstance(inp, str):
                return self._load_image_file(inp)
            if isinstance(inp, Image.Image):
                return self._convert_pil_image(inp)
            if isinstance(inp, torch.Tensor):
                return self._convert_tensor(inp)
            # Assume numeric data convertible directly to tensor.
            return torch.as_tensor(inp, device=self.device)
        except Exception as e:
            logger.error(f"Image conversion failed: {str(e)}")
            raise

    def _load_image_file(self, filepath: str) -> torch.Tensor:
        """Load image file using torchvision with format detection."""
        img_tensor = torchvision.io.read_image(filepath)
        # Handle grayscale conversion.
        if self.convert_grayscale and img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        # Convert to float and scale to [0, 1].
        img_tensor = img_tensor.to(dtype=torch.float32) / 255.0
        # Ensure the tensor meets the valid shape requirements.
        img_tensor = self._ensure_valid_shape(img_tensor)
        # Move to the target device.
        img_tensor = img_tensor.to(self.device, non_blocking=True)
        return self.trt_transforms(img_tensor)

    def _convert_pil_image(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to normalized tensor."""
        if image.mode != 'RGB' and self.convert_grayscale:
            image = image.convert('RGB')
        tensor = T.functional.to_tensor(image)  # scales pixels to [0,1]
        tensor = self._ensure_valid_shape(tensor)
        tensor = tensor.to(self.device, non_blocking=True)
        return self.trt_transforms(tensor)

    def _convert_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize existing tensors."""
        if tensor.dim() == 2:  # Add channel dimension for grayscale.
            tensor = tensor.unsqueeze(0)
        if tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)
            tensor /= 255.0  # Assume byte tensor if not float.
        # Handle grayscale conversion.
        if self.convert_grayscale and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        tensor = self._ensure_valid_shape(tensor)
        tensor = tensor.to(self.device, non_blocking=True)
        return self.trt_transforms(tensor)

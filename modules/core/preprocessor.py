import logging
import threading
import time
from queue import Queue, Empty
from typing import Any, List, Optional, Union

import torch
from torchvision import transforms as T
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BasePreprocessor:
    """
    A minimal preprocessor that simply returns the input data as a PyTorch tensor
    or leaves it unchanged. Supports zero-copy processing for efficiency.
    """
    def __call__(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return inputs
        return torch.as_tensor(inputs, device="cuda", non_blocking=True)  # Zero-copy transfer


class ImagePreprocessor(BasePreprocessor):
    """
    Preprocessor for image data with batch optimizations, multi-task handling, and streaming support.
    """
    def __init__(
        self,
        image_size: Union[int, tuple] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        use_pinned_memory: bool = True,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.use_pinned_memory = use_pinned_memory
        self.device = torch.device(device)
        self.transforms = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])
    
    def __call__(self, inputs: Any) -> torch.Tensor:
        if not isinstance(inputs, list):
            inputs = [inputs]
        processed = [self.transforms(self._load_image(inp)) for inp in inputs]
        batch_tensor = torch.stack(processed, dim=0).to(self.device, non_blocking=True)
        if self.use_pinned_memory:
            batch_tensor = batch_tensor.pin_memory()
        return batch_tensor

    def _load_image(self, inp: Any) -> Image.Image:
        if isinstance(inp, Image.Image):
            return inp
        elif isinstance(inp, str):
            return Image.open(inp).convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(inp)}")


class MultiTaskPreprocessor(ImagePreprocessor):
    """
    Preprocessor for multi-task models including classification, detection, and segmentation.
    """
    def __init__(self, *args, multi_label: bool = False, threshold: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_label = multi_label
        self.threshold = threshold

    def __call__(self, inputs: Any) -> torch.Tensor:
        batch_tensor = super().__call__(inputs)
        if self.multi_label:
            batch_tensor = (batch_tensor > self.threshold).float()  # Convert to binary mask
        return batch_tensor


class TensorRTPreprocessor(ImagePreprocessor):
    """
    Preprocessor tailored for TensorRT inference.
    Inherits from ImagePreprocessor and converts the preprocessed batch to FP16
    if the TensorRT engine is configured for FP16 inference.
    """
    def __init__(
        self,
        image_size: Union[int, tuple] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        use_pinned_memory: bool = True,
        device: Union[str, torch.device] = "cuda",
        trt_fp16: bool = False,
    ):
        super().__init__(image_size=image_size, mean=mean, std=std, use_pinned_memory=use_pinned_memory, device=device)
        self.trt_fp16 = trt_fp16

    def __call__(self, inputs: Any) -> torch.Tensor:
        batch_tensor = super().__call__(inputs)
        if self.trt_fp16:
            batch_tensor = batch_tensor.half()
        return batch_tensor


class InflightBatcher:
    """
    A batcher that collects requests for preprocessing, supporting streaming and dynamic batch optimizations.
    The batcher collects incoming preprocessing requests and groups them into a batch,
    where the batch size is dynamically determined based on the current queue depth.
    """
    def __init__(self,
                 preprocessor: BasePreprocessor,
                 min_batch_size: int = 1,
                 max_batch_size: int = 8,
                 max_delay: float = 0.01):
        """
        Args:
            preprocessor: A callable that accepts a list of inputs and returns a batched tensor.
            min_batch_size: The minimum number of inputs to wait for before processing.
            max_batch_size: The maximum number of inputs in a batch.
            max_delay: Maximum delay (in seconds) to wait for additional requests.
        """
        self.preprocessor = preprocessor
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_delay = max_delay
        self.queue = Queue()
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.worker_thread.start()

    def submit(self, inp: Any) -> Queue:
        """
        Submit a preprocessing request.
        Returns a Queue from which the caller can later retrieve the processed result.
        """
        result_queue = Queue(maxsize=1)
        self.queue.put((inp, result_queue))
        return result_queue

    def shutdown(self):
        """Shutdown the batcher gracefully."""
        self._stop_event.set()
        self.worker_thread.join()

    def _batch_worker(self):
        buffer, result_queues = [], []
        last_time = time.time()
        while not self._stop_event.is_set():
            try:
                inp, rqueue = self.queue.get(timeout=self.max_delay)
                buffer.append(inp)
                result_queues.append(rqueue)
            except Empty:
                pass

            now = time.time()
            # Compute a dynamic desired batch size:
            # desired_batch_size = current buffered items plus what's still in the queue,
            # clamped between min_batch_size and max_batch_size.
            queued = self.queue.qsize()
            desired_batch_size = len(buffer) + queued
            if desired_batch_size < self.min_batch_size:
                desired_batch_size = self.min_batch_size
            if desired_batch_size > self.max_batch_size:
                desired_batch_size = self.max_batch_size

            # If we've collected enough items or waited long enough, process the batch.
            if len(buffer) >= desired_batch_size or ((now - last_time) >= self.max_delay and buffer):
                try:
                    # Preprocess the buffered inputs as a single batch.
                    batch = self.preprocessor(buffer)
                except Exception as e:
                    for rq in result_queues:
                        rq.put(e)
                else:
                    # Slice the batched tensor and return each sample to its caller.
                    for i, rq in enumerate(result_queues):
                        rq.put(batch[i])
                buffer, result_queues = [], []
                last_time = time.time()

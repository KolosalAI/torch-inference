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
    or leaves it unchanged. Useful as a placeholder or for unstructured data.
    """

    def __call__(self, inputs: Any) -> torch.Tensor:
        """
        Convert the input to a torch.Tensor (if applicable).
        Override for custom logic.

        Args:
            inputs (Any): Raw input (could be an existing torch.Tensor or other data structure).

        Returns:
            torch.Tensor: The processed/converted input.
        """
        if isinstance(inputs, torch.Tensor):
            # Already a tensor
            return inputs

        # If inputs is not a tensor, convert it to one
        return torch.as_tensor(inputs)


class ImagePreprocessor(BasePreprocessor):
    """
    A preprocessor for image data (PIL Images, file paths, or a list of them).
    Applies optional transformations (resize, normalization, etc.), then
    aggregates into a batch dimension.

    Can also leverage pinned memory for faster CPU->GPU transfers.
    """

    def __init__(
        self,
        image_size: Union[int, tuple] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        to_rgb: bool = True,
        use_pinned_memory: bool = False,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Args:
            image_size (Union[int, tuple]): Target image size for resizing. If int, becomes (int, int).
            mean (List[float]): Mean for normalization (commonly ImageNet mean).
            std (List[float]): Std for normalization (commonly ImageNet std).
            to_rgb (bool): Whether to ensure input channels are in RGB order.
            use_pinned_memory (bool): Use pinned (page-locked) memory for CPU->GPU transfer optimization.
            device (Union[str, torch.device]): Device to place the final tensor (e.g., 'cuda' for GPU, 'cpu' otherwise).
        """
        super().__init__()
        # Convert image_size to tuple if needed
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb
        self.use_pinned_memory = use_pinned_memory

        # Torch device (used if we want to create or move the final tensor to GPU/CPU)
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        # Build a set of TorchVision transforms
        self.transforms = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])

    def __call__(self, inputs: Any) -> torch.Tensor:
        """
        Preprocess image(s) into a batch of tensors.

        Args:
            inputs (Any): Could be:
                - A single PIL.Image.Image object
                - A single file path (string) to an image
                - A list of PIL.Image.Image objects
                - A list of file paths
                - etc.

        Returns:
            torch.Tensor: A batch tensor of shape (N, C, H, W), where N is the number of images.
        """
        # Convert all inputs into a list for uniform processing
        if not isinstance(inputs, list):
            inputs = [inputs]

        processed = []
        for inp in inputs:
            img = self._load_image(inp)
            if self.to_rgb:
                img = img.convert("RGB")
            tensor_img = self.transforms(img)
            processed.append(tensor_img)

        # Stack all images into a single batch dimension
        batch_tensor = torch.stack(processed, dim=0)

        if self.use_pinned_memory:
            # Only beneficial if your next step is to move this to GPU
            batch_tensor = batch_tensor.pin_memory()

        # Move to the desired device (e.g., 'cuda') if you want immediate GPU usage
        batch_tensor = batch_tensor.to(self.device, non_blocking=True)

        return batch_tensor

    def _load_image(self, inp: Any) -> Image.Image:
        """
        Utility method to handle different input types and load them as PIL images.

        Args:
            inp (Any): A single input (PIL image, file path, etc.)

        Returns:
            Image.Image: The loaded PIL Image in a standard format (RGB or original).
        """
        if isinstance(inp, Image.Image):
            return inp
        elif isinstance(inp, str):
            # Assume it's a file path
            img = Image.open(inp)
            return img
        else:
            raise ValueError(f"Unsupported image input type: {type(inp)}. "
                             "Expected PIL.Image.Image or file path.")


class TensorRTPreprocessor(ImagePreprocessor):
    """
    Extends ImagePreprocessor with additional considerations for TensorRT:

    - Ensures final dtype matches what TensorRT engine expects (e.g., float32 or float16).
    - Potentially handles more advanced batch dimension logic for TRT.
    """

    def __init__(
        self,
        image_size=(224, 224),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        to_rgb=True,
        use_pinned_memory=False,
        device="cuda",
        trt_fp16=False,
    ):
        """
        Args:
            trt_fp16 (bool): If True, casts final tensors to float16 for TRT.
            (Other arguments same as ImagePreprocessor).
        """
        super().__init__(image_size, mean, std, to_rgb, use_pinned_memory, device)
        self.trt_fp16 = trt_fp16

    def __call__(self, inputs: Any) -> torch.Tensor:
        batch_tensor = super().__call__(inputs)
        if self.trt_fp16:
            # Convert to half precision if that’s what your TensorRT engine expects
            batch_tensor = batch_tensor.half()
        else:
            # Ensure float32 just in case
            batch_tensor = batch_tensor.float()
        return batch_tensor


class InflightBatcher:
    """
    A minimalistic in-flight batcher that collects requests from multiple threads
    (or async coroutines) and assembles them into a single batch for preprocessing.

    Usage:
        1. Create an InflightBatcher instance with a preprocessor and desired batching parameters.
        2. Submit items with `submit(input_data)`.
        3. The batcher automatically collects items up to `max_batch_size` or `max_delay`,
           then calls `preprocessor` on the entire batch.
        4. Each call to `submit` returns a Future-like object that can be used to get the result.

    This is a simple design. In production, consider using a dedicated library or robust approach.
    """

    def __init__(
        self,
        preprocessor: BasePreprocessor,
        max_batch_size: int = 8,
        max_delay: float = 0.01
    ):
        """
        Args:
            preprocessor (BasePreprocessor): Preprocessing object.
            max_batch_size (int): Maximum items to collect before forming a batch.
            max_delay (float): Maximum delay in seconds before forcing a batch.
        """
        self.preprocessor = preprocessor
        self.max_batch_size = max_batch_size
        self.max_delay = max_delay

        self.queue = Queue()
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.worker_thread.start()

    def submit(self, inp: Any):
        """
        Submit an input for batching. Returns a tuple (thread-safe queue, unique_id)
        that can be used to retrieve the processed result.
        """
        result_queue = Queue(maxsize=1)
        self.queue.put((inp, result_queue))
        return result_queue

    def shutdown(self):
        """Stops the background worker."""
        self._stop_event.set()
        self.worker_thread.join()

    def _batch_worker(self):
        """
        Worker thread that assembles batches and processes them.
        """
        buffer = []
        result_queues = []
        last_time = time.time()

        while not self._stop_event.is_set():
            # Try to get a new item
            try:
                inp, rqueue = self.queue.get(timeout=self.max_delay)
                buffer.append(inp)
                result_queues.append(rqueue)
            except Empty:
                pass

            # Check if we have enough items or if we hit max delay
            now = time.time()
            if (len(buffer) >= self.max_batch_size) or (now - last_time >= self.max_delay and buffer):
                # Process the entire buffer as a batch
                try:
                    batch = self.preprocessor(buffer)
                except Exception as e:
                    # Send the error back if processing fails
                    for rq in result_queues:
                        rq.put(e)
                else:
                    # If success, distribute each slice of the batch to its corresponding queue
                    for i, rq in enumerate(result_queues):
                        # The i-th item in the batch is the i-th processed result
                        # For image-based batch, you might want to just return the entire batch 
                        # or slice out the single image. Below we return the i-th item:
                        rq.put(batch[i])

                # Clear buffers
                buffer = []
                result_queues = []
                last_time = time.time()


# ----------------
# Example usage:
# ----------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Suppose we want to do TensorRT-based FP16 preprocessing
    preprocessor = TensorRTPreprocessor(
        image_size=(224, 224),
        trt_fp16=True,  # assume our TRT engine expects FP16
        device="cuda",
    )

    # Create a batcher
    batcher = InflightBatcher(preprocessor, max_batch_size=4, max_delay=0.5)

    # Simulate concurrency: multiple threads or calls
    # For demonstration, we'll just submit some file paths or PIL.Image objects in a loop
    # (Replace these with real images or paths)
    test_images = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]

    futures = []
    for img in test_images:
        # Each submit returns a queue where we'll later get the result
        f = batcher.submit(img)
        futures.append(f)

    # Wait for each result
    for i, f in enumerate(futures):
        result = f.get()  # this will block until the preprocessor finishes
        if isinstance(result, Exception):
            logger.error(f"Error processing {test_images[i]}: {result}")
        else:
            logger.info(f"Processed {test_images[i]} with shape {result.shape}")

    # Clean up
    batcher.shutdown()

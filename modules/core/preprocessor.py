"""
core/preprocessor.py

Preprocessing module for PyTorch inference.

Classes:
    - BasePreprocessor: Minimal/base class that returns the input as-is or in a generic format.
    - ImagePreprocessor: A specialized class for image inputs that uses TorchVision transforms
      and includes best practices for performance (e.g., GPU-accelerated transforms, pinned memory).

Key Features & Optimizations:
    - Optionally uses GPU transforms if desired (via DALI or specialized libraries).
    - Batch-oriented design (if user provides multiple inputs).
    - Pinned memory usage for faster CPU-to-GPU transfer.
"""

import logging
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
    ):
        """
        Args:
            image_size (Union[int, tuple]): Target image size for resizing. If int, becomes (int, int).
            mean (List[float]): Mean for normalization (commonly ImageNet mean).
            std (List[float]): Std for normalization (commonly ImageNet std).
            to_rgb (bool): Whether to ensure input channels are in RGB order.
            use_pinned_memory (bool): Use pinned (page-locked) memory for CPU->GPU transfer optimization.
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

        # Build a set of TorchVision transforms
        # (feel free to add more advanced augmentations if needed)
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
                # Convert image to RGB if needed
                img = img.convert("RGB")

            # Apply the torchvision transforms
            tensor_img = self.transforms(img)
            processed.append(tensor_img)

        # Stack all images into a single batch dimension
        batch_tensor = torch.stack(processed, dim=0)

        # Optionally use pinned memory for faster CPU->GPU transfers
        # (only beneficial if your next step is to move this to GPU)
        if self.use_pinned_memory:
            batch_tensor = batch_tensor.pin_memory()

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
            # Attempt fallback or raise an error
            raise ValueError(f"Unsupported image input type: {type(inp)}. "
                             "Expected PIL.Image.Image or file path.")

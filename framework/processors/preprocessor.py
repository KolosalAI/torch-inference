"""
Generic preprocessing framework for various input types.

This module provides a flexible, extensible preprocessing system that can handle
different input types (images, text, audio, etc.) with pluggable transformations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from pathlib import Path
import hashlib
import time
from dataclasses import dataclass
from enum import Enum

from ..core.config import InferenceConfig


logger = logging.getLogger(__name__)


class InputType(Enum):
    """Supported input types."""
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    TENSOR = "tensor"
    NUMPY = "numpy"
    CUSTOM = "custom"


@dataclass
class PreprocessingResult:
    """Result of preprocessing operation."""
    data: torch.Tensor
    metadata: Dict[str, Any]
    original_shape: Optional[Tuple[int, ...]] = None
    processing_time: float = 0.0


class PreprocessingError(Exception):
    """Exception raised during preprocessing."""
    pass


class BasePreprocessor(ABC):
    """
    Abstract base class for all preprocessors.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = config.device.get_torch_device()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def supports_input_type(self, input_type: InputType) -> bool:
        """Check if this preprocessor supports the given input type."""
        pass
    
    @abstractmethod
    def preprocess(self, inputs: Any) -> PreprocessingResult:
        """Preprocess inputs synchronously."""
        pass
    
    async def preprocess_async(self, inputs: Any) -> PreprocessingResult:
        """Preprocess inputs asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.preprocess, inputs)
    
    def validate_inputs(self, inputs: Any) -> bool:
        """Validate input data."""
        return inputs is not None
    
    def get_cache_key(self, inputs: Any) -> Optional[str]:
        """Generate cache key for inputs."""
        try:
            if isinstance(inputs, str):
                return hashlib.md5(inputs.encode()).hexdigest()
            elif isinstance(inputs, (np.ndarray, torch.Tensor)):
                return hashlib.md5(str(inputs.shape).encode()).hexdigest()
            else:
                return None
        except Exception:
            return None


class CustomPreprocessor(BasePreprocessor):
    """
    Custom preprocessor for generic/unknown input types.
    """
    
    def supports_input_type(self, input_type: InputType) -> bool:
        """Supports custom input type."""
        return input_type == InputType.CUSTOM
    
    def preprocess(self, inputs: Any) -> PreprocessingResult:
        """Simple preprocessing for custom inputs."""
        start_time = time.time()
        
        # Handle different input types
        if isinstance(inputs, torch.Tensor):
            # Already a tensor, just move to device
            tensor = inputs.to(self.device)
        elif isinstance(inputs, (list, tuple)):
            # Convert list/tuple to tensor, preserving appropriate dtype
            try:
                # Try to infer dtype from the data
                first_elem = inputs[0] if inputs else 0
                if isinstance(first_elem, int):
                    tensor = torch.tensor(inputs, dtype=torch.long).to(self.device)
                else:
                    tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            except (ValueError, TypeError):
                # If can't convert to tensor, create a dummy tensor
                tensor = torch.zeros(1, 10, dtype=torch.float32).to(self.device)
        elif isinstance(inputs, (int, float)):
            # Single number to tensor
            dtype = torch.long if isinstance(inputs, int) else torch.float32
            tensor = torch.tensor([inputs], dtype=dtype).to(self.device)
        else:
            # For any other type, create a dummy tensor
            tensor = torch.zeros(1, 10, dtype=torch.float32).to(self.device)
        
        processing_time = time.time() - start_time

        return PreprocessingResult(
            data=tensor,
            original_shape=getattr(inputs, 'shape', None),
            metadata={
                "input_type": "custom",
                "preprocessing_time": processing_time,
                "device": str(self.device)
            },
            processing_time=processing_time
        )
class ImagePreprocessor(BasePreprocessor):
    """Preprocessor for image inputs."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.input_size = config.preprocessing.input_size
        self.mean = np.array(config.preprocessing.mean)
        self.std = np.array(config.preprocessing.std)
        self.interpolation = config.preprocessing.interpolation
        self.center_crop = config.preprocessing.center_crop
        self.normalize = config.preprocessing.normalize
        self.to_rgb = config.preprocessing.to_rgb
        
        # Setup transforms
        self._setup_transforms()
        
    def _setup_transforms(self):
        """Setup image transformation pipeline."""
        try:
            import torchvision.transforms as T
            from torchvision.transforms import InterpolationMode
            
            # Map interpolation string to torchvision enum
            interp_map = {
                "nearest": InterpolationMode.NEAREST,
                "bilinear": InterpolationMode.BILINEAR,
                "bicubic": InterpolationMode.BICUBIC,
            }
            
            transforms = []
            
            # Convert to tensor
            transforms.append(T.ToTensor())
            
            # Resize
            if self.input_size:
                transforms.append(T.Resize(
                    self.input_size, 
                    interpolation=interp_map.get(self.interpolation, InterpolationMode.BILINEAR),
                    antialias=True
                ))
            
            # Center crop
            if self.center_crop and self.input_size:
                transforms.append(T.CenterCrop(self.input_size))
            
            # Normalize
            if self.normalize:
                transforms.append(T.Normalize(mean=self.mean, std=self.std))
            
            self.transforms = T.Compose(transforms)
            self.use_torchvision = True
            
        except ImportError:
            self.use_torchvision = False
            self.logger.warning("torchvision not available, using OpenCV fallback")
    
    def supports_input_type(self, input_type: InputType) -> bool:
        """Check if this preprocessor supports the given input type."""
        return input_type == InputType.IMAGE
    
    def preprocess(self, inputs: Any) -> PreprocessingResult:
        """Preprocess image inputs."""
        start_time = time.time()
        
        try:
            # Load and convert image
            image = self._load_image(inputs)
            original_shape = image.shape
            
            # Validate image before processing
            if image.size == 0:
                raise ValueError("Empty image provided")
            
            # Log image characteristics for debugging
            self.logger.debug(f"Processing image with shape: {image.shape}, dtype: {image.dtype}")
            
            # Apply transforms
            if self.use_torchvision:
                tensor = self._apply_torchvision_transforms(image)
            else:
                tensor = self._apply_opencv_transforms(image)
            
            # Move to device
            tensor = tensor.to(self.device)
            
            # Add batch dimension if needed
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            
            processing_time = time.time() - start_time
            
            return PreprocessingResult(
                data=tensor,
                metadata={
                    "input_type": "image",
                    "original_shape": original_shape,
                    "final_shape": tuple(tensor.shape),
                    "preprocessor": self.__class__.__name__
                },
                original_shape=original_shape,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            self.logger.debug(f"Input type: {type(inputs)}, Input shape: {getattr(inputs, 'shape', 'N/A')}")
            
            # Try to create a fallback tensor for critical errors
            try:
                # Create a default image tensor (3x224x224 RGB image)
                fallback_tensor = torch.zeros(3, 224, 224, dtype=torch.float32, device=self.device)
                if self.normalize:
                    # Apply the same normalization as would be applied to real images
                    for i in range(3):
                        fallback_tensor[i] = (fallback_tensor[i] - self.mean[i]) / self.std[i]
                
                # Add batch dimension
                fallback_tensor = fallback_tensor.unsqueeze(0)
                
                processing_time = time.time() - start_time
                
                self.logger.warning("Using fallback tensor due to preprocessing error")
                return PreprocessingResult(
                    data=fallback_tensor,
                    metadata={
                        "input_type": "image",
                        "original_shape": getattr(inputs, 'shape', None),
                        "final_shape": tuple(fallback_tensor.shape),
                        "preprocessor": self.__class__.__name__,
                        "fallback": True,
                        "error": str(e)
                    },
                    processing_time=processing_time
                )
            except Exception as fallback_error:
                self.logger.error(f"Fallback tensor creation also failed: {fallback_error}")
                
            raise PreprocessingError(f"Image preprocessing failed: {e}") from e
    
    def _load_image(self, inputs: Any) -> np.ndarray:
        """Load image from various input formats."""
        try:
            if isinstance(inputs, str):
                return self._load_image_from_path(inputs)
            elif isinstance(inputs, list):
                # Handle nested lists that represent images (e.g., [C, H, W] format)
                image_array = np.array(inputs, dtype=np.float32)
                # Validate the resulting array
                if image_array.size == 0:
                    raise ValueError("Empty list provided")
                return image_array
            elif isinstance(inputs, np.ndarray):
                # Validate the numpy array
                if inputs.size == 0:
                    raise ValueError("Empty numpy array provided")
                return self._process_numpy_image(inputs)
            elif isinstance(inputs, torch.Tensor):
                # Validate the tensor
                if inputs.numel() == 0:
                    raise ValueError("Empty tensor provided")
                return self._tensor_to_numpy(inputs)
            elif hasattr(inputs, 'convert'):  # PIL Image
                return np.array(inputs.convert('RGB'))
            else:
                raise ValueError(f"Unsupported input type: {type(inputs)}")
        except Exception as e:
            self.logger.error(f"Failed to load image from input: {e}")
            # Re-raise with more context
            raise ValueError(f"Failed to load image from {type(inputs)}: {e}") from e
    
    def _load_image_from_path(self, path: str) -> np.ndarray:
        """Load image from file path or URL."""
        if path.startswith(('http://', 'https://')):
            return self._load_image_from_url(path)
        else:
            return self._load_image_from_file(path)
    
    def _load_image_from_file(self, path: str) -> np.ndarray:
        """Load image from local file."""
        try:
            from PIL import Image
            image = Image.open(path).convert('RGB')
            return np.array(image)
        except ImportError:
            # Fallback to OpenCV
            import cv2
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Failed to load image: {path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _load_image_from_url(self, url: str) -> np.ndarray:
        """Load image from URL."""
        import requests
        from io import BytesIO
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            from PIL import Image
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return np.array(image)
        except ImportError:
            # Fallback to OpenCV
            import cv2
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to decode image from URL: {url}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _process_numpy_image(self, image: np.ndarray) -> np.ndarray:
        """Process numpy array image."""
        # Handle different input formats
        if image.ndim == 1:
            # 1D array - convert to a square image if possible
            size = int(np.sqrt(image.size))
            if size * size == image.size:
                image = image.reshape(size, size)
            else:
                # Create a 1D "image" - pad or truncate to square
                target_size = 32  # minimum reasonable size
                if image.size < target_size * target_size:
                    # Pad with zeros
                    padded = np.zeros(target_size * target_size)
                    padded[:image.size] = image
                    image = padded.reshape(target_size, target_size)
                else:
                    # Truncate to square
                    image = image[:target_size * target_size].reshape(target_size, target_size)
                    
        if image.ndim == 2:  # Grayscale [H, W]
            h, w = image.shape
            # Handle degenerate cases
            if h <= 2 or w <= 2:
                self.logger.warning(f"Very small image dimensions {image.shape}, padding to minimum size")
                # Pad to minimum size
                min_size = 32
                padded = np.zeros((min_size, min_size), dtype=image.dtype)
                padded[:min(h, min_size), :min(w, min_size)] = image[:min(h, min_size), :min(w, min_size)]
                image = padded
            # Convert to RGB
            image = np.stack([image] * 3, axis=2)  # Convert to [H, W, 3]
            
        elif image.ndim == 3:
            h, w, c = image.shape
            
            # Handle degenerate spatial dimensions
            if h <= 2 or w <= 2:
                self.logger.warning(f"Very small spatial dimensions {(h, w)}, padding to minimum size")
                min_size = 32
                # Create new array with minimum size - ensure it's at least 3 channels for RGB
                target_channels = max(c, 3) if c <= 4 else 3
                new_image = np.zeros((min_size, min_size, target_channels), dtype=image.dtype)
                # Copy existing data, but limit channels to target
                copy_channels = min(c, target_channels)
                new_image[:min(h, min_size), :min(w, min_size), :copy_channels] = image[:min(h, min_size), :min(w, min_size), :copy_channels]
                # If we need more channels (e.g., grayscale to RGB), replicate
                if target_channels == 3 and copy_channels == 1:
                    new_image[:, :, 1] = new_image[:, :, 0]
                    new_image[:, :, 2] = new_image[:, :, 0]
                image = new_image
                h, w, c = image.shape
            
            # Check if it's in [C, H, W] format (channels first)
            if image.shape[0] == 3 and image.shape[1] > image.shape[0] and image.shape[2] > image.shape[0]:
                # Likely [C, H, W] format, convert to [H, W, C]
                image = np.transpose(image, (1, 2, 0))
                h, w, c = image.shape
            elif image.shape[0] <= 4 and image.shape[1] > 10 and image.shape[2] > 10:
                # Another case of [C, H, W] format
                image = np.transpose(image, (1, 2, 0))
                h, w, c = image.shape
                
            # Handle channel dimension
            if c == 1:  # Single channel [H, W, 1]
                image = np.concatenate([image] * 3, axis=2)
            elif c == 2:  # Two channels - duplicate one to make 3
                image = np.concatenate([image, image[:, :, :1]], axis=2)
            elif c == 4:  # RGBA [H, W, 4]
                image = image[:, :, :3]
            elif c > 4:  # Too many channels
                if c >= 3:
                    # Take first 3 channels as RGB
                    image = image[:, :, :3]
                else:
                    # Convert to grayscale and then RGB
                    image = np.mean(image, axis=2, keepdims=True)
                    image = np.concatenate([image] * 3, axis=2)
            elif c == 3:  # RGB [H, W, 3]
                # Check if BGR and convert to RGB
                if self.to_rgb:
                    # Simple heuristic: if blue channel has higher mean, likely BGR
                    if np.mean(image[:, :, 0]) > np.mean(image[:, :, 2]):
                        try:
                            import cv2
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        except ImportError:
                            # Manual BGR to RGB conversion
                            image = image[:, :, [2, 1, 0]]
        elif image.ndim == 4:
            # 4D tensor - take first image if it's a batch
            self.logger.warning(f"4D tensor provided with shape {image.shape}, taking first sample")
            image = image[0]
            # Recursively process the 3D image
            return self._process_numpy_image(image)
        elif image.ndim > 4:
            # Higher dimensional tensor - flatten to reasonable dimensions
            self.logger.warning(f"High dimensional tensor {image.shape}, flattening to 2D")
            # Flatten all but last 2 dimensions
            original_shape = image.shape
            image = image.reshape(-1, original_shape[-1]) if len(original_shape) > 1 else image.flatten()
            # Try to make it square-ish
            if image.ndim == 1:
                return self._process_numpy_image(image)  # Recursively handle 1D case
            else:
                # 2D case - convert to grayscale image
                h, w = image.shape
                min_dim = min(h, w)
                max_size = 224  # Reasonable max size
                if min_dim > max_size:
                    # Downsample
                    step_h = max(1, h // max_size)
                    step_w = max(1, w // max_size)
                    image = image[::step_h, ::step_w]
                # Convert to grayscale and then RGB
                if image.dtype == np.float32 or image.dtype == np.float64:
                    # Normalize to 0-255 range
                    image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
                image = np.stack([image] * 3, axis=2)
        
        return image
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        if tensor.ndim == 4:  # Batch dimension
            tensor = tensor[0]
        
        if tensor.ndim == 3:
            if tensor.shape[0] in [1, 3]:  # CHW format
                tensor = tensor.permute(1, 2, 0)  # Convert to HWC
        elif tensor.ndim == 2:
            # 2D tensor - assume it's a grayscale image (H, W)
            # Add channel dimension to make it (H, W, 1)
            tensor = tensor.unsqueeze(-1)
        
        return tensor.detach().cpu().numpy()
    
    def _apply_torchvision_transforms(self, image: np.ndarray) -> torch.Tensor:
        """Apply torchvision transforms."""
        from PIL import Image
        
        # Validate image shape and handle edge cases
        if image.size == 0:
            raise ValueError("Empty image array provided")
        
        # Log image shape for debugging
        self.logger.debug(f"Processing image with shape: {image.shape}, dtype: {image.dtype}")
        
        # Handle unusual tensor shapes that can't be processed as images
        if len(image.shape) == 3:
            h, w, c = image.shape
            # Check for degenerate cases (very small images or unusual channel counts)
            if h <= 2 or w <= 2 or c > 4:
                self.logger.warning(f"Unusual image shape {image.shape}, creating fallback image")
                # Create a default RGB image tensor for compatibility
                default_image = np.full((224, 224, 3), 128, dtype=np.uint8)  # Gray image
                return self._apply_torchvision_transforms(default_image)
        elif len(image.shape) == 2:
            h, w = image.shape
            if h <= 2 or w <= 2:
                self.logger.warning(f"Degenerate image shape {image.shape}, creating fallback image")
                # Create a default grayscale image tensor for compatibility
                default_image = np.full((224, 224), 128, dtype=np.uint8)  # Gray image
                return self._apply_torchvision_transforms(default_image)
        elif len(image.shape) == 1:
            self.logger.warning(f"1D array provided with shape {image.shape}, creating fallback image")
            default_image = np.full((224, 224), 128, dtype=np.uint8)
            return self._apply_torchvision_transforms(default_image)
        elif len(image.shape) > 3:
            self.logger.warning(f"High dimensional array provided with shape {image.shape}, creating fallback image")
            default_image = np.full((224, 224, 3), 128, dtype=np.uint8)
            return self._apply_torchvision_transforms(default_image)
        
        # Handle different input ranges and normalize to 0-255 for PIL
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Assume tensor is in normalized range (e.g., -1 to 1 or 0 to 1)
            # Normalize to 0-255 range
            image_min, image_max = image.min(), image.max()
            if image_min >= 0 and image_max <= 1:
                # Already in 0-1 range
                image = (image * 255).astype(np.uint8)
            elif image_min >= -1 and image_max <= 1:
                # In -1 to 1 range
                image = ((image + 1) * 127.5).astype(np.uint8)
            else:
                # Arbitrary range - normalize to 0-255
                image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Handle single channel by converting to RGB if normalization expects 3 channels
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image.squeeze(2)  # Remove single channel dimension
        elif len(image.shape) == 3 and image.shape[2] > 4:
            # Too many channels - take first 3 or convert to RGB
            self.logger.warning(f"Image has {image.shape[2]} channels, reducing to 3")
            if image.shape[2] >= 3:
                image = image[:, :, :3]  # Take first 3 channels
            else:
                # Convert to grayscale and then RGB
                image = np.mean(image, axis=2).astype(np.uint8)
        
        try:
            # Final validation before PIL conversion
            if len(image.shape) not in [2, 3]:
                raise ValueError(f"Invalid image shape after processing: {image.shape}")
            
            # More thorough dimension checking
            if len(image.shape) == 3:
                h, w, c = image.shape
                if h <= 0 or w <= 0 or c <= 0:
                    raise ValueError(f"Invalid dimensions: height={h}, width={w}, channels={c}")
                if h > 10000 or w > 10000:  # Prevent extremely large images
                    raise ValueError(f"Image dimensions too large: {h}x{w}")
                if c not in [1, 3, 4]:
                    raise ValueError(f"Invalid number of channels: {c}")
            elif len(image.shape) == 2:
                h, w = image.shape
                if h <= 0 or w <= 0:
                    raise ValueError(f"Invalid dimensions: height={h}, width={w}")
                if h > 10000 or w > 10000:  # Prevent extremely large images
                    raise ValueError(f"Image dimensions too large: {h}x{w}")
            
            # Additional safety check for PIL compatibility
            if len(image.shape) == 3:
                h, w, c = image.shape
                # PIL has issues with very small dimensions or unusual channel counts
                if h < 3 or w < 3 or c > 4:
                    self.logger.warning(f"Image dimensions {image.shape} may cause PIL issues, using fallback")
                    raise ValueError(f"Dimensions not suitable for PIL: {image.shape}")
            elif len(image.shape) == 2:
                h, w = image.shape
                if h < 3 or w < 3:
                    self.logger.warning(f"Image dimensions {image.shape} too small for PIL, using fallback")
                    raise ValueError(f"Dimensions too small for PIL: {image.shape}")
            
            if len(image.shape) == 2:
                # Convert grayscale to RGB by repeating the channel 3 times
                # This ensures compatibility with RGB normalization parameters
                pil_image = Image.fromarray(image, mode='L').convert('RGB')
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # Standard RGB image
                pil_image = Image.fromarray(image, mode='RGB')
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA image - convert to RGB
                pil_image = Image.fromarray(image, mode='RGBA').convert('RGB')
            elif len(image.shape) == 3 and image.shape[2] == 1:
                # Single channel - squeeze and convert to grayscale then RGB
                image = image.squeeze(2)
                pil_image = Image.fromarray(image, mode='L').convert('RGB')
            else:
                # Fallback: flatten to grayscale and convert to RGB
                if len(image.shape) == 3:
                    image = np.mean(image, axis=2).astype(np.uint8)
                pil_image = Image.fromarray(image, mode='L').convert('RGB')
                
            return self.transforms(pil_image)
            
        except (ValueError, TypeError) as e:
            self.logger.error(f"Failed to create PIL image from array with shape {image.shape} and dtype {image.dtype}: {e}")
            # Final fallback: create a default image
            default_image = np.full((224, 224, 3), 128, dtype=np.uint8)
            pil_image = Image.fromarray(default_image, mode='RGB')
            return self.transforms(pil_image)
    
    def _apply_opencv_transforms(self, image: np.ndarray) -> torch.Tensor:
        """Apply OpenCV-based transforms."""
        import cv2
        
        # Resize if needed
        if self.input_size:
            height, width = self.input_size
            if self.interpolation == "nearest":
                interp = cv2.INTER_NEAREST
            elif self.interpolation == "bicubic":
                interp = cv2.INTER_CUBIC
            else:
                interp = cv2.INTER_LINEAR
            
            image = cv2.resize(image, (width, height), interpolation=interp)
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply normalization
        if self.normalize:
            image = (image - self.mean) / self.std
        
        # Convert to CHW format
        image = np.transpose(image, (2, 0, 1))
        
        # Convert to tensor
        return torch.from_numpy(image).float()


class TextPreprocessor(BasePreprocessor):
    """Preprocessor for text inputs."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.max_length = config.custom_params.get("max_length", 512)
        self.tokenizer = None
        self._setup_tokenizer()
    
    def _setup_tokenizer(self):
        """Setup text tokenizer."""
        # This would be configured based on the specific model
        # For now, simple word-based tokenization
        pass
    
    def supports_input_type(self, input_type: InputType) -> bool:
        """Check if this preprocessor supports the given input type."""
        return input_type == InputType.TEXT
    
    def preprocess(self, inputs: Any) -> PreprocessingResult:
        """Preprocess text inputs."""
        start_time = time.time()
        
        try:
            if isinstance(inputs, str):
                text = inputs
            elif isinstance(inputs, list):
                text = " ".join(inputs)
            else:
                text = str(inputs)
            
            # Simple tokenization (would use proper tokenizer in practice)
            tokens = self._tokenize(text)
            
            # Convert to tensor
            tensor = torch.tensor(tokens, device=self.device).unsqueeze(0)
            
            processing_time = time.time() - start_time
            
            return PreprocessingResult(
                data=tensor,
                metadata={
                    "input_type": "text",
                    "text_length": len(text),
                    "num_tokens": len(tokens),
                    "preprocessor": self.__class__.__name__
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Text preprocessing failed: {e}")
            raise PreprocessingError(f"Text preprocessing failed: {e}") from e
    
    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization (placeholder)."""
        # This is a very basic implementation
        # In practice, would use proper tokenizers like transformers
        return [hash(word) % 10000 for word in text.split()]


class TensorPreprocessor(BasePreprocessor):
    """Preprocessor for tensor inputs."""
    
    def supports_input_type(self, input_type: InputType) -> bool:
        """Check if this preprocessor supports the given input type."""
        return input_type in [InputType.TENSOR, InputType.NUMPY]
    
    def preprocess(self, inputs: Any) -> PreprocessingResult:
        """Preprocess tensor inputs."""
        start_time = time.time()
        
        try:
            if isinstance(inputs, torch.Tensor):
                tensor = inputs.clone()
            elif isinstance(inputs, np.ndarray):
                # Preserve integer dtypes for token IDs
                if inputs.dtype in [np.int32, np.int64]:
                    tensor = torch.from_numpy(inputs).long()
                else:
                    tensor = torch.from_numpy(inputs)
            else:
                tensor = torch.tensor(inputs)
            
            # Move to device
            tensor = tensor.to(self.device)
            
            # Add batch dimension if needed
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 2:
                # Check if this looks like an image (both dimensions reasonably large)
                # vs a feature vector (one dimension small, likely batch or feature count)
                h, w = tensor.shape
                if h >= 32 and w >= 32:
                    # Likely a grayscale image [H, W] -> [1, 1, H, W]
                    tensor = tensor.unsqueeze(0).unsqueeze(0)
                # else: probably already properly shaped data (batch_size, features) or (features, batch_size)
            elif tensor.ndim == 3:
                # 3D tensor - check if it's an image [C, H, W] that needs batch dimension
                # But also consider [H, W, C] format which is common for numpy arrays
                dim0, dim1, dim2 = tensor.shape
                
                # Heuristic to detect format:
                # If first dim is 1-4 and others are larger, likely [C, H, W]
                # If last dim is 1-4 and others are larger, likely [H, W, C] 
                # If all dims are small, need special handling
                
                if dim0 in [1, 3, 4] and dim1 >= 32 and dim2 >= 32:
                    # Likely [C, H, W] format - standard image tensor
                    tensor = tensor.unsqueeze(0)  # Add batch: [1, C, H, W]
                elif dim2 in [1, 3, 4] and dim0 >= 32 and dim1 >= 32:
                    # Likely [H, W, C] format - need to transpose to [C, H, W] then add batch
                    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [C, H, W] -> [1, C, H, W]
                elif dim0 <= 32 and dim1 <= 32 and dim2 > 4:
                    # Small spatial dims with many "channels" - likely [H, W, C] with many channels
                    self.logger.warning(f"3D tensor with small spatial dims {dim0}x{dim1} and {dim2} channels, reshaping")
                    # Reshape to something more manageable - flatten spatial and treat as feature vector
                    tensor = tensor.view(-1, dim2)  # [H*W, C]
                    tensor = tensor.unsqueeze(0)    # Add batch: [1, H*W, C]
                elif dim0 > 4 and dim1 <= 32 and dim2 <= 32:
                    # Many "channels" with small spatial dims - likely [C, H, W] with many channels
                    self.logger.warning(f"3D tensor with {dim0} channels and small spatial dims {dim1}x{dim2}, reshaping")
                    # Flatten to feature vector
                    tensor = tensor.view(dim0, -1)  # [C, H*W]
                    tensor = tensor.unsqueeze(0)    # Add batch: [1, C, H*W]
                else:
                    # Other 3D tensor - add batch dimension conservatively
                    tensor = tensor.unsqueeze(0)  # [1, dim0, dim1, dim2]
            
            processing_time = time.time() - start_time
            
            return PreprocessingResult(
                data=tensor,
                metadata={
                    "input_type": "tensor",
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "preprocessor": self.__class__.__name__
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Tensor preprocessing failed: {e}")
            raise PreprocessingError(f"Tensor preprocessing failed: {e}") from e


class PreprocessorPipeline:
    """
    Pipeline for chaining multiple preprocessors.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.preprocessors: List[BasePreprocessor] = []
        self.cache_enabled = config.cache.enable_caching
        self.cache = {} if self.cache_enabled else None
        self.max_cache_size = config.cache.cache_size
        self.executor = ThreadPoolExecutor(max_workers=config.performance.max_workers)
        self.logger = logging.getLogger(f"{__name__}.PreprocessorPipeline")
        
        # Add default preprocessors
        self._add_default_preprocessors()
    
    def _add_default_preprocessors(self) -> None:
        """Add default preprocessors for each input type."""
        # Add image preprocessor
        image_processor = ImagePreprocessor(self.config)
        self.add_preprocessor(image_processor)
        
        # Add text preprocessor
        text_processor = TextPreprocessor(self.config)
        self.add_preprocessor(text_processor)
        
        # Add custom preprocessor for unknown types
        custom_processor = CustomPreprocessor(self.config)
        self.add_preprocessor(custom_processor)
    
    def add_preprocessor(self, preprocessor: BasePreprocessor) -> None:
        """Add a preprocessor to the pipeline."""
        self.preprocessors.append(preprocessor)
        self.logger.info(f"Added preprocessor: {preprocessor.__class__.__name__}")
    
    def detect_input_type(self, inputs: Any) -> InputType:
        """Detect the type of input data."""
        self.logger.debug(f"Detecting input type for: {type(inputs)}")
        
        if isinstance(inputs, str):
            # Check if it's an image path/URL
            if any(inputs.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
                self.logger.debug("Detected as IMAGE (file path)")
                return InputType.IMAGE
            elif inputs.startswith(('http://', 'https://')) and 'image' in inputs.lower():
                self.logger.debug("Detected as IMAGE (URL)")
                return InputType.IMAGE
            else:
                self.logger.debug("Detected as TEXT")
                return InputType.TEXT
        elif isinstance(inputs, list):
            # Handle nested lists that might represent images
            if len(inputs) == 3 and all(isinstance(channel, list) for channel in inputs):
                # Check if it looks like [C, H, W] format (3 channels)
                if all(len(channel) > 10 for channel in inputs):  # Reasonable height (lowered threshold)
                    first_channel = inputs[0]
                    if isinstance(first_channel, list) and len(first_channel) > 10:
                        # Check if all rows have same length (width)
                        if all(isinstance(row, list) and len(row) == len(first_channel[0]) for row in first_channel):
                            self.logger.debug(f"Detected as IMAGE (3D list [C,H,W] - shape: {len(inputs)}x{len(first_channel)}x{len(first_channel[0])})")
                            return InputType.IMAGE
            # Handle other list formats
            if isinstance(inputs, list) and len(inputs) > 0:
                # Check if it's a list of numbers (could be a feature vector)
                if all(isinstance(x, (int, float)) for x in inputs):
                    self.logger.debug("Detected as TENSOR (1D list of numbers)")
                    return InputType.TENSOR
                # Check if it's a nested list of numbers
                elif all(isinstance(x, list) and all(isinstance(y, (int, float)) for y in x) for x in inputs):
                    self.logger.debug("Detected as TENSOR (2D list of numbers)")
                    return InputType.TENSOR
            # Default fallback for lists
            self.logger.debug(f"Detected as CUSTOM (unrecognized list format - length: {len(inputs) if isinstance(inputs, list) else 'N/A'})")
            return InputType.CUSTOM
        elif isinstance(inputs, (np.ndarray, torch.Tensor)):
            shape = inputs.shape
            dtype = inputs.dtype if isinstance(inputs, torch.Tensor) else inputs.dtype
            self.logger.debug(f"Detected tensor/array with shape: {shape}, dtype: {dtype}")
            
            # If integer tensor, likely text tokens
            if isinstance(inputs, torch.Tensor):
                if inputs.dtype in [torch.int32, torch.int64, torch.long]:
                    self.logger.debug("Detected as TENSOR (integer tensor - likely tokens)")
                    return InputType.TENSOR
            elif isinstance(inputs, np.ndarray):
                if inputs.dtype in [np.int32, np.int64]:
                    self.logger.debug("Detected as TENSOR (integer array - likely tokens)")
                    return InputType.TENSOR
            
            # Handle batched tensors - if already batched, treat as tensor
            if len(shape) == 4:  # [batch, channels, height, width] 
                if shape[1] == 3 or shape[1] == 1:  # RGB or grayscale channels
                    # This is a batch of images - treat as tensor for batch processing
                    self.logger.debug(f"Detected as TENSOR (4D batch - shape: {shape})")
                    return InputType.TENSOR
                else:
                    # Some other 4D tensor
                    self.logger.debug(f"Detected as TENSOR (4D other - shape: {shape})")
                    return InputType.TENSOR
            elif len(shape) == 3:
                if shape[0] == 3 or shape[-1] == 3:  # Single image [C, H, W] or [H, W, C]
                    self.logger.debug(f"Detected as IMAGE (3D tensor - shape: {shape})")
                    return InputType.IMAGE
                else:
                    self.logger.debug(f"Detected as TENSOR (3D non-image - shape: {shape})")
                    return InputType.TENSOR
            elif len(shape) == 2:
                # Could be a grayscale image [H, W] or batch of 1D features
                # If dimensions suggest it could be an image (both dimensions > 10), treat as image
                if len(shape) == 2 and shape[0] >= 10 and shape[1] >= 10:
                    self.logger.debug(f"Detected as IMAGE (2D tensor - shape: {shape})")
                    return InputType.IMAGE
                else:
                    self.logger.debug(f"Detected as TENSOR (2D features - shape: {shape})")
                    return InputType.TENSOR
            else:
                self.logger.debug(f"Detected as TENSOR (other dimensions - shape: {shape})")
                return InputType.TENSOR
        elif hasattr(inputs, 'convert'):  # PIL Image
            self.logger.debug("Detected as IMAGE (PIL Image)")
            return InputType.IMAGE
        else:
            self.logger.debug(f"Detected as CUSTOM (unknown type: {type(inputs)})")
            return InputType.CUSTOM
    
    def preprocess(self, inputs: Any) -> PreprocessingResult:
        """Preprocess inputs using the appropriate preprocessor."""
        # Check cache first
        if self.cache_enabled:
            cache_key = self._get_cache_key(inputs)
            if cache_key and cache_key in self.cache:
                self.logger.debug(f"Cache hit for key: {cache_key}")
                return self.cache[cache_key]
        
        # Detect input type
        input_type = self.detect_input_type(inputs)
        
        # Find appropriate preprocessor
        preprocessor = self._find_preprocessor(input_type)
        if not preprocessor:
            raise PreprocessingError(f"No preprocessor found for input type: {input_type}")
        
        # Validate inputs
        if not preprocessor.validate_inputs(inputs):
            raise PreprocessingError("Input validation failed")
        
        # Preprocess
        result = preprocessor.preprocess(inputs)
        
        # Cache result
        if self.cache_enabled and cache_key:
            self._cache_result(cache_key, result)
        
        return result
    
    async def preprocess_async(self, inputs: Any) -> PreprocessingResult:
        """Preprocess inputs asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.preprocess, inputs)
    
    def preprocess_batch(self, inputs_list: List[Any]) -> List[PreprocessingResult]:
        """Preprocess a batch of inputs."""
        results = []
        for inputs in inputs_list:
            try:
                result = self.preprocess(inputs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to preprocess input: {e}")
                # Add error result
                results.append(PreprocessingResult(
                    data=torch.empty(0),
                    metadata={"error": str(e)},
                    processing_time=0.0
                ))
        return results
    
    async def preprocess_batch_async(self, inputs_list: List[Any]) -> List[PreprocessingResult]:
        """Preprocess a batch of inputs asynchronously."""
        tasks = [self.preprocess_async(inputs) for inputs in inputs_list]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def _find_preprocessor(self, input_type: InputType) -> Optional[BasePreprocessor]:
        """Find preprocessor for the given input type."""
        for preprocessor in self.preprocessors:
            if preprocessor.supports_input_type(input_type):
                return preprocessor
        return None
    
    def _get_cache_key(self, inputs: Any) -> Optional[str]:
        """Generate cache key for inputs."""
        try:
            if isinstance(inputs, str):
                return hashlib.md5(inputs.encode()).hexdigest()
            elif hasattr(inputs, 'shape'):
                return hashlib.md5(f"{inputs.shape}_{type(inputs)}".encode()).hexdigest()
            else:
                return hashlib.md5(str(inputs).encode()).hexdigest()
        except Exception:
            return None
    
    def _cache_result(self, cache_key: str, result: PreprocessingResult) -> None:
        """Cache preprocessing result."""
        if len(self.cache) >= self.max_cache_size:
            # Simple LRU: remove first item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        self.logger.debug(f"Cached result for key: {cache_key}")
    
    def clear_cache(self) -> None:
        """Clear preprocessing cache."""
        if self.cache:
            self.cache.clear()
            self.logger.info("Preprocessing cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        return {
            "num_preprocessors": len(self.preprocessors),
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.cache) if self.cache else 0,
            "max_cache_size": self.max_cache_size,
            "preprocessor_types": [p.__class__.__name__ for p in self.preprocessors]
        }


def create_default_preprocessing_pipeline(config: InferenceConfig) -> PreprocessorPipeline:
    """Create a default preprocessing pipeline with common preprocessors."""
    pipeline = PreprocessorPipeline(config)
    
    # Add common preprocessors
    pipeline.add_preprocessor(ImagePreprocessor(config))
    pipeline.add_preprocessor(TextPreprocessor(config))
    pipeline.add_preprocessor(TensorPreprocessor(config))
    
    return pipeline

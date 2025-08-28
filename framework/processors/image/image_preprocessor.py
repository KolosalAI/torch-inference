"""
Image preprocessing pipeline for the PyTorch inference framework.

This module provides comprehensive image preprocessing capabilities including:
- Image format conversion and loading
- Resizing, cropping, and normalization
- Computer vision transformations
- Image augmentation
- Batch processing utilities
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import torch
import logging
from pathlib import Path
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)


class ImagePreprocessorError(Exception):
    """Exception raised for image preprocessing errors."""
    pass


class BaseImagePreprocessor(ABC):
    """Base class for image preprocessors."""
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None, normalize: bool = True):
        self.target_size = target_size
        self.normalize = normalize
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def process(self, image: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Process image input."""
        pass


class ImageLoader:
    """Image file loader with support for multiple formats."""
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        self.target_size = target_size
        self.logger = logging.getLogger(f"{__name__}.ImageLoader")
        
        # Check available image libraries
        self._check_image_libraries()
    
    def _check_image_libraries(self):
        """Check which image libraries are available."""
        self.has_pil = False
        self.has_opencv = False
        self.has_skimage = False
        
        try:
            from PIL import Image
            self.has_pil = True
        except ImportError:
            pass
        
        try:
            import cv2
            self.has_opencv = True
        except ImportError:
            pass
        
        try:
            import skimage
            self.has_skimage = True
        except ImportError:
            pass
        
        if not any([self.has_pil, self.has_opencv, self.has_skimage]):
            self.logger.warning("No image libraries found. Install PIL, OpenCV, or scikit-image for image loading.")
    
    def load_image(self, file_path: Union[str, Path], 
                   target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Load image file with automatic format detection.
        
        Args:
            file_path: Path to image file or URL
            target_size: Target size (height, width) - None to use original size
            
        Returns:
            Image array in RGB format [H, W, C]
        """
        if isinstance(file_path, str):
            file_path = Path(file_path) if not file_path.startswith(('http://', 'https://')) else file_path
        
        # Handle URLs
        if isinstance(file_path, str) and file_path.startswith(('http://', 'https://')):
            return self._load_from_url(file_path, target_size)
        
        # Handle local files
        if isinstance(file_path, Path) and not file_path.exists():
            raise ImagePreprocessorError(f"Image file not found: {file_path}")
        
        size = target_size or self.target_size
        
        # Try loading with available libraries
        if self.has_pil:
            return self._load_with_pil(file_path, size)
        elif self.has_opencv:
            return self._load_with_opencv(file_path, size)
        elif self.has_skimage:
            return self._load_with_skimage(file_path, size)
        else:
            raise ImagePreprocessorError("No image loading library available")
    
    def _load_with_pil(self, file_path: Path, target_size: Optional[Tuple[int, int]]) -> np.ndarray:
        """Load image using PIL."""
        try:
            from PIL import Image
            
            image = Image.open(str(file_path)).convert('RGB')
            
            # Resize if target size specified
            if target_size:
                height, width = target_size
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            
            return np.array(image)
            
        except Exception as e:
            raise ImagePreprocessorError(f"Failed to load image with PIL: {e}")
    
    def _load_with_opencv(self, file_path: Path, target_size: Optional[Tuple[int, int]]) -> np.ndarray:
        """Load image using OpenCV."""
        try:
            import cv2
            
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError(f"Failed to load image: {file_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if target size specified
            if target_size:
                height, width = target_size
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
            
            return image
            
        except Exception as e:
            raise ImagePreprocessorError(f"Failed to load image with OpenCV: {e}")
    
    def _load_with_skimage(self, file_path: Path, target_size: Optional[Tuple[int, int]]) -> np.ndarray:
        """Load image using scikit-image."""
        try:
            from skimage import io, transform
            
            image = io.imread(str(file_path))
            
            # Convert to RGB if needed
            if image.ndim == 3 and image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]
            elif image.ndim == 2:  # Grayscale
                image = np.stack([image] * 3, axis=2)
            
            # Resize if target size specified
            if target_size:
                height, width = target_size
                image = transform.resize(image, (height, width), preserve_range=True, anti_aliasing=True)
                image = image.astype(np.uint8)
            
            return image
            
        except Exception as e:
            raise ImagePreprocessorError(f"Failed to load image with scikit-image: {e}")
    
    def _load_from_url(self, url: str, target_size: Optional[Tuple[int, int]]) -> np.ndarray:
        """Load image from URL."""
        try:
            import requests
            from io import BytesIO
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            if self.has_pil:
                from PIL import Image
                image = Image.open(BytesIO(response.content)).convert('RGB')
                
                # Resize if target size specified
                if target_size:
                    height, width = target_size
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                
                return np.array(image)
            else:
                # Fallback to OpenCV
                import cv2
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Failed to decode image from URL: {url}")
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize if target size specified
                if target_size:
                    height, width = target_size
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
                
                return image
                
        except Exception as e:
            raise ImagePreprocessorError(f"Failed to load image from URL: {e}")


class ImageTransforms:
    """Image transformation utilities."""
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None, 
                 mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
        """
        Initialize image transforms.
        
        Args:
            target_size: Target size (height, width)
            mean: Normalization mean values [R, G, B]
            std: Normalization std values [R, G, B]
        """
        self.target_size = target_size
        self.mean = np.array(mean or [0.485, 0.456, 0.406])
        self.std = np.array(std or [0.229, 0.224, 0.225])
        self.logger = logging.getLogger(f"{__name__}.ImageTransforms")
    
    def resize(self, image: np.ndarray, size: Tuple[int, int], 
               interpolation: str = "lanczos") -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image [H, W, C]
            size: Target size (height, width)
            interpolation: Interpolation method
            
        Returns:
            Resized image
        """
        try:
            # Try OpenCV first
            import cv2
            
            height, width = size
            interp_map = {
                "nearest": cv2.INTER_NEAREST,
                "linear": cv2.INTER_LINEAR,
                "cubic": cv2.INTER_CUBIC,
                "lanczos": cv2.INTER_LANCZOS4
            }
            
            interp = interp_map.get(interpolation, cv2.INTER_LANCZOS4)
            return cv2.resize(image, (width, height), interpolation=interp)
            
        except ImportError:
            # Fallback to PIL
            try:
                from PIL import Image
                
                height, width = size
                pil_image = Image.fromarray(image)
                
                interp_map = {
                    "nearest": Image.Resampling.NEAREST,
                    "linear": Image.Resampling.BILINEAR,
                    "cubic": Image.Resampling.BICUBIC,
                    "lanczos": Image.Resampling.LANCZOS
                }
                
                interp = interp_map.get(interpolation, Image.Resampling.LANCZOS)
                resized = pil_image.resize((width, height), interp)
                return np.array(resized)
                
            except ImportError:
                # Basic numpy resize (not ideal but functional)
                self.logger.warning("No image library available, using basic resize")
                return self._basic_resize(image, size)
    
    def _basic_resize(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Basic resize using numpy (low quality)."""
        height, width = size
        h, w = image.shape[:2]
        
        # Simple nearest neighbor resize
        y_indices = np.round(np.linspace(0, h - 1, height)).astype(int)
        x_indices = np.round(np.linspace(0, w - 1, width)).astype(int)
        
        resized = image[np.ix_(y_indices, x_indices)]
        return resized
    
    def center_crop(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Center crop image to target size.
        
        Args:
            image: Input image [H, W, C]
            size: Target size (height, width)
            
        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        target_h, target_w = size
        
        # Calculate crop coordinates
        start_h = max(0, (h - target_h) // 2)
        start_w = max(0, (w - target_w) // 2)
        end_h = min(h, start_h + target_h)
        end_w = min(w, start_w + target_w)
        
        cropped = image[start_h:end_h, start_w:end_w]
        
        # Pad if necessary
        if cropped.shape[0] < target_h or cropped.shape[1] < target_w:
            padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
            pad_h = (target_h - cropped.shape[0]) // 2
            pad_w = (target_w - cropped.shape[1]) // 2
            padded[pad_h:pad_h + cropped.shape[0], pad_w:pad_w + cropped.shape[1]] = cropped
            return padded
        
        return cropped
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image with mean and std.
        
        Args:
            image: Input image [H, W, C] in range [0, 255]
            
        Returns:
            Normalized image
        """
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply mean and std normalization
        image = (image - self.mean) / self.std
        
        return image
    
    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert numpy image to PyTorch tensor.
        
        Args:
            image: Input image [H, W, C]
            
        Returns:
            Image tensor [C, H, W]
        """
        # Convert HWC to CHW
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))
        elif image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        
        return torch.from_numpy(image.copy()).float()
    
    def apply_transforms(self, image: np.ndarray) -> torch.Tensor:
        """
        Apply complete transformation pipeline.
        
        Args:
            image: Input image [H, W, C]
            
        Returns:
            Transformed tensor [C, H, W]
        """
        # Resize if target size specified
        if self.target_size:
            image = self.resize(image, self.target_size)
        
        # Normalize
        image = self.normalize(image)
        
        # Convert to tensor
        tensor = self.to_tensor(image)
        
        return tensor


class ImageAugmentation:
    """Image augmentation utilities for data preprocessing."""
    
    def __init__(self, enable_augmentation: bool = False):
        self.enable_augmentation = enable_augmentation
        self.logger = logging.getLogger(f"{__name__}.ImageAugmentation")
    
    def random_flip(self, image: np.ndarray, prob: float = 0.5) -> np.ndarray:
        """Randomly flip image horizontally."""
        if not self.enable_augmentation or np.random.random() > prob:
            return image
        return np.fliplr(image)
    
    def random_rotation(self, image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
        """Randomly rotate image."""
        if not self.enable_augmentation:
            return image
        
        try:
            import cv2
            angle = np.random.uniform(-max_angle, max_angle)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, matrix, (w, h))
            
        except ImportError:
            self.logger.warning("OpenCV not available, skipping rotation")
            return image
    
    def adjust_brightness(self, image: np.ndarray, factor_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Randomly adjust brightness."""
        if not self.enable_augmentation:
            return image
        
        factor = np.random.uniform(*factor_range)
        adjusted = image.astype(np.float32) * factor
        return np.clip(adjusted, 0, 255).astype(image.dtype)
    
    def adjust_contrast(self, image: np.ndarray, factor_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Randomly adjust contrast."""
        if not self.enable_augmentation:
            return image
        
        factor = np.random.uniform(*factor_range)
        mean = image.mean()
        adjusted = (image.astype(np.float32) - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(image.dtype)
    
    def apply_augmentations(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations to image."""
        if not self.enable_augmentation:
            return image
        
        # Apply augmentations in sequence
        image = self.random_flip(image)
        image = self.random_rotation(image)
        image = self.adjust_brightness(image)
        image = self.adjust_contrast(image)
        
        return image


class ComprehensiveImagePreprocessor(BaseImagePreprocessor):
    """
    Comprehensive image preprocessor combining all functionality.
    """
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = (224, 224), 
                 normalize: bool = True, mean: Optional[List[float]] = None,
                 std: Optional[List[float]] = None, enable_augmentation: bool = False):
        super().__init__(target_size, normalize)
        
        self.enable_augmentation = enable_augmentation
        
        # Initialize components
        self.loader = ImageLoader(target_size)
        self.transforms = ImageTransforms(target_size, mean, std)
        self.augmentation = ImageAugmentation(enable_augmentation)
    
    def load_image(self, file_path: Union[str, Path], 
                   target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Load image file using the internal image loader.
        
        Args:
            file_path: Path to image file or URL
            target_size: Target size (height, width) - None to use preprocessor's target_size
            
        Returns:
            Image array in RGB format [H, W, C]
        """
        return self.loader.load_image(file_path, target_size)
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image array.
        
        Args:
            image: Input image [H, W, C]
            
        Returns:
            Preprocessed tensor [C, H, W]
        """
        # Apply augmentations if enabled
        if self.enable_augmentation:
            image = self.augmentation.apply_augmentations(image)
        
        # Apply transforms
        tensor = self.transforms.apply_transforms(image)
        
        return tensor
    
    def process(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Process image input with full pipeline.
        
        Args:
            image: Image input (file path, array, or tensor)
            
        Returns:
            Processed tensor [C, H, W]
        """
        # Load image if file path
        if isinstance(image, (str, Path)):
            image_array = self.loader.load_image(image, self.target_size)
        elif isinstance(image, torch.Tensor):
            # Convert tensor to numpy for processing
            image_array = self._tensor_to_numpy(image)
        elif isinstance(image, np.ndarray):
            image_array = image.copy()
        else:
            raise ImagePreprocessorError(f"Unsupported input type: {type(image)}")
        
        # Ensure RGB format
        image_array = self._ensure_rgb_format(image_array)
        
        # Preprocess
        tensor = self.preprocess_image(image_array)
        
        return tensor
    
    def process_batch(self, images: List[Union[str, np.ndarray, torch.Tensor]]) -> torch.Tensor:
        """
        Process a batch of images.
        
        Args:
            images: List of image inputs
            
        Returns:
            Batched tensor [B, C, H, W]
        """
        processed_images = []
        
        for image in images:
            try:
                tensor = self.process(image)
                processed_images.append(tensor)
            except Exception as e:
                self.logger.error(f"Failed to process image: {e}")
                # Add a zero tensor as fallback
                if self.target_size:
                    h, w = self.target_size
                    fallback = torch.zeros(3, h, w)
                else:
                    fallback = torch.zeros(3, 224, 224)
                processed_images.append(fallback)
        
        # Stack into batch
        return torch.stack(processed_images)
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        if tensor.dim() == 4:  # Batch dimension
            tensor = tensor[0]
        
        if tensor.dim() == 3:
            if tensor.shape[0] in [1, 3, 4]:  # CHW format
                tensor = tensor.permute(1, 2, 0)  # Convert to HWC
        elif tensor.dim() == 2:
            # 2D tensor - assume it's a grayscale image
            tensor = tensor.unsqueeze(-1)
        
        # Convert to numpy and ensure uint8 range
        image = tensor.detach().cpu().numpy()
        if image.dtype in [np.float32, np.float64]:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def _ensure_rgb_format(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is in RGB format [H, W, 3]."""
        if image.ndim == 2:
            # Grayscale to RGB
            image = np.stack([image] * 3, axis=2)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                # Single channel to RGB
                image = np.concatenate([image] * 3, axis=2)
            elif image.shape[2] == 4:
                # RGBA to RGB
                image = image[:, :, :3]
            elif image.shape[0] in [1, 3, 4] and image.shape[2] > image.shape[0]:
                # CHW to HWC format
                image = np.transpose(image, (1, 2, 0))
                if image.shape[2] == 1:
                    image = np.concatenate([image] * 3, axis=2)
                elif image.shape[2] == 4:
                    image = image[:, :, :3]
        
        return image

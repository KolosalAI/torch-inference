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
            raise PreprocessingError(f"Image preprocessing failed: {e}") from e
    
    def _load_image(self, inputs: Any) -> np.ndarray:
        """Load image from various input formats."""
        if isinstance(inputs, str):
            return self._load_image_from_path(inputs)
        elif isinstance(inputs, np.ndarray):
            return self._process_numpy_image(inputs)
        elif isinstance(inputs, torch.Tensor):
            return self._tensor_to_numpy(inputs)
        elif hasattr(inputs, 'convert'):  # PIL Image
            return np.array(inputs.convert('RGB'))
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
    
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
        # Ensure RGB format
        if image.ndim == 2:  # Grayscale
            image = np.stack([image] * 3, axis=2)
        elif image.ndim == 3:
            if image.shape[2] == 1:  # Single channel
                image = np.concatenate([image] * 3, axis=2)
            elif image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]
            elif image.shape[2] == 3:
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
        
        return image
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        if tensor.ndim == 4:  # Batch dimension
            tensor = tensor[0]
        
        if tensor.ndim == 3:
            if tensor.shape[0] in [1, 3]:  # CHW format
                tensor = tensor.permute(1, 2, 0)  # Convert to HWC
        
        return tensor.detach().cpu().numpy()
    
    def _apply_torchvision_transforms(self, image: np.ndarray) -> torch.Tensor:
        """Apply torchvision transforms."""
        from PIL import Image
        pil_image = Image.fromarray(image.astype(np.uint8))
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
                tensor = torch.from_numpy(inputs)
            else:
                tensor = torch.tensor(inputs)
            
            # Move to device
            tensor = tensor.to(self.device)
            
            # Add batch dimension if needed
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            
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
    
    def add_preprocessor(self, preprocessor: BasePreprocessor) -> None:
        """Add a preprocessor to the pipeline."""
        self.preprocessors.append(preprocessor)
        self.logger.info(f"Added preprocessor: {preprocessor.__class__.__name__}")
    
    def detect_input_type(self, inputs: Any) -> InputType:
        """Detect the type of input data."""
        if isinstance(inputs, str):
            # Check if it's an image path/URL
            if any(inputs.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
                return InputType.IMAGE
            elif inputs.startswith(('http://', 'https://')) and 'image' in inputs.lower():
                return InputType.IMAGE
            else:
                return InputType.TEXT
        elif isinstance(inputs, (np.ndarray, torch.Tensor)):
            # Heuristic: if 3D or 4D with appropriate dimensions, likely image
            shape = inputs.shape
            if len(shape) >= 3 and (shape[-1] == 3 or shape[-3] == 3):
                return InputType.IMAGE
            else:
                return InputType.TENSOR
        elif hasattr(inputs, 'convert'):  # PIL Image
            return InputType.IMAGE
        else:
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

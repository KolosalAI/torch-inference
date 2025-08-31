"""
Generic preprocessing framework for various input types.

This module provides a flexible, extensible preprocessing system that can handle
different input types (images, text, audio, etc.) with pluggable transformations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from pathlib import Path
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
import functools
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import weakref
import gc
import psutil
import os

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
class PreprocessorConfig:
    """Configuration for preprocessing operations."""
    
    input_type: str = "image"
    input_size: Tuple[int, int] = (224, 224)
    batch_size: int = 1
    normalize: bool = True
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    device: str = "auto"
    dtype: str = "float32"
    resize_mode: str = "bilinear"
    padding_mode: str = "constant"
    crop_mode: str = "center"
    interpolation: str = "bilinear"
    preserve_aspect_ratio: bool = True
    channel_order: str = "RGB"
    channel_first: bool = True
    apply_transforms: bool = True
    cache_enabled: bool = True
    max_cache_size: int = 1000
    preprocessing_threads: int = 4
    enable_timing: bool = False
    validation_enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.input_type not in ["image", "text", "audio", "video", "tensor", "numpy", "custom"]:
            raise ValueError(f"Unsupported input_type: {self.input_type}")
        
        if len(self.input_size) != 2 or any(s <= 0 for s in self.input_size):
            raise ValueError("input_size must be a tuple of two positive integers")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must have 3 values for RGB channels")
        
        if any(s <= 0 for s in self.std):
            raise ValueError("std values must be positive")
        
        if self.resize_mode not in ["nearest", "linear", "bilinear", "bicubic", "trilinear", "area"]:
            raise ValueError(f"Unsupported resize_mode: {self.resize_mode}")
        
        if self.crop_mode not in ["center", "random", "five_crop", "ten_crop"]:
            raise ValueError(f"Unsupported crop_mode: {self.crop_mode}")
        
        if self.channel_order not in ["RGB", "BGR", "RGBA", "BGRA"]:
            raise ValueError(f"Unsupported channel_order: {self.channel_order}")
        
        if self.max_cache_size <= 0:
            raise ValueError("max_cache_size must be positive")
        
        if self.preprocessing_threads <= 0:
            raise ValueError("preprocessing_threads must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'input_type': self.input_type,
            'input_size': self.input_size,
            'batch_size': self.batch_size,
            'normalize': self.normalize,
            'mean': self.mean,
            'std': self.std,
            'device': self.device,
            'dtype': self.dtype,
            'resize_mode': self.resize_mode,
            'padding_mode': self.padding_mode,
            'crop_mode': self.crop_mode,
            'interpolation': self.interpolation,
            'preserve_aspect_ratio': self.preserve_aspect_ratio,
            'channel_order': self.channel_order,
            'channel_first': self.channel_first,
            'apply_transforms': self.apply_transforms,
            'cache_enabled': self.cache_enabled,
            'max_cache_size': self.max_cache_size,
            'preprocessing_threads': self.preprocessing_threads,
            'enable_timing': self.enable_timing,
            'validation_enabled': self.validation_enabled
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PreprocessorConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


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
    Abstract base class for all preprocessors with built-in optimizations.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = config.device.get_torch_device()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance optimizations
        self._enable_optimizations = True
        self._tensor_cache = {}
        self._tensor_cache_lock = threading.RLock()
        self._memory_pool = []
        self._memory_pool_lock = threading.RLock()
        self._max_cache_size = getattr(config.cache, 'cache_size', 100)
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_preprocessing_time': 0.0,
            'num_processed': 0,
            'memory_pool_hits': 0
        }
        
        # Pre-allocate common tensor sizes for memory pool
        self._initialize_memory_pool()
        
        # Setup compilation optimizations
        self._setup_compilation_optimizations()
    
    def _initialize_memory_pool(self):
        """Pre-allocate common tensor sizes to reduce allocation overhead."""
        common_sizes = [
            (1, 3, 224, 224),  # Standard image batch
            (1, 3, 256, 256),  # Larger image batch
            (1, 512),          # Text tokens
            (1, 1024),         # Larger text tokens
            (1, 16000),        # Audio samples (1 second at 16kHz)
        ]
        
        for size in common_sizes:
            try:
                tensor = torch.zeros(size, device=self.device, dtype=torch.float32)
                with self._memory_pool_lock:
                    self._memory_pool.append({
                        'tensor': tensor,
                        'shape': size,
                        'in_use': False
                    })
            except Exception as e:
                self.logger.debug(f"Could not pre-allocate tensor of size {size}: {e}")
    
    def _setup_compilation_optimizations(self):
        """Setup compilation-based optimizations."""
        # Setup torch.compile if available and enabled
        if hasattr(torch, 'compile') and self.config.device.use_torch_compile:
            try:
                # Compile commonly used tensor operations
                self._compiled_ops = {}
                self._compiled_ops['normalize'] = torch.compile(self._normalize_tensor_compiled, mode=self.config.device.compile_mode)
                self._compiled_ops['resize'] = torch.compile(self._resize_tensor_compiled, mode=self.config.device.compile_mode)
                self._compiled_ops['to_device'] = torch.compile(self._to_device_compiled, mode=self.config.device.compile_mode)
                self.logger.info("Torch compilation enabled for preprocessing operations")
            except Exception as e:
                self.logger.warning(f"Could not setup torch compilation: {e}")
                self._compiled_ops = {}
        else:
            self._compiled_ops = {}
    
    def _get_tensor_from_pool(self, shape: tuple, dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
        """Get a pre-allocated tensor from memory pool if available."""
        with self._memory_pool_lock:
            for pool_item in self._memory_pool:
                if (pool_item['shape'] == shape and 
                    not pool_item['in_use'] and 
                    pool_item['tensor'].dtype == dtype):
                    pool_item['in_use'] = True
                    self._stats['memory_pool_hits'] += 1
                    return pool_item['tensor']
        return None
    
    def _return_tensor_to_pool(self, tensor: torch.Tensor):
        """Return a tensor to the memory pool."""
        with self._memory_pool_lock:
            for pool_item in self._memory_pool:
                if torch.equal(pool_item['tensor'], tensor):
                    pool_item['in_use'] = False
                    # Zero out the tensor for reuse
                    pool_item['tensor'].zero_()
                    break
    
    def _normalize_tensor_compiled(self, tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Compiled version of tensor normalization."""
        return (tensor - mean) / std
    
    def _resize_tensor_compiled(self, tensor: torch.Tensor, size: tuple) -> torch.Tensor:
        """Compiled version of tensor resizing."""
        return torch.nn.functional.interpolate(tensor, size=size, mode='bilinear', align_corners=False)
    
    def _to_device_compiled(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Compiled version of device transfer."""
        return tensor.to(device, non_blocking=True)
    
    @functools.lru_cache(maxsize=128)
    def _get_cached_transform_params(self, input_shape: tuple, target_shape: tuple) -> tuple:
        """Cache transformation parameters to avoid recalculation."""
        scale_x = target_shape[1] / input_shape[1] if input_shape[1] > 0 else 1.0
        scale_y = target_shape[0] / input_shape[0] if input_shape[0] > 0 else 1.0
        return scale_x, scale_y
    
    @abstractmethod
    def supports_input_type(self, input_type: InputType) -> bool:
        """Check if this preprocessor supports the given input type."""
        pass
    
    @abstractmethod
    def preprocess(self, inputs: Any) -> PreprocessingResult:
        """Preprocess inputs synchronously."""
        pass
    
    async def preprocess_async(self, inputs: Any) -> PreprocessingResult:
        """Preprocess inputs asynchronously with optimizations."""
        loop = asyncio.get_running_loop()
        
        # Use thread pool for CPU-bound operations
        if self.config.performance.enable_async:
            return await loop.run_in_executor(None, self._preprocess_optimized, inputs)
        else:
            return await loop.run_in_executor(None, self.preprocess, inputs)
    
    def _preprocess_optimized(self, inputs: Any) -> PreprocessingResult:
        """Optimized preprocessing with performance tracking."""
        start_time = time.perf_counter()
        
        try:
            # Check optimized cache first
            cache_key = self._get_optimized_cache_key(inputs)
            if cache_key:
                cached_result = self._get_from_optimized_cache(cache_key)
                if cached_result is not None:
                    self._stats['cache_hits'] += 1
                    return cached_result
                self._stats['cache_misses'] += 1
            
            # Perform optimized preprocessing
            result = self.preprocess(inputs)
            
            # Cache result with optimization
            if cache_key:
                self._store_in_optimized_cache(cache_key, result)
            
            # Update performance stats
            processing_time = time.perf_counter() - start_time
            self._stats['total_preprocessing_time'] += processing_time
            self._stats['num_processed'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimized preprocessing failed: {e}")
            # Fallback to basic preprocessing
            return self.preprocess(inputs)
    
    def _get_optimized_cache_key(self, inputs: Any) -> Optional[str]:
        """Generate optimized cache key with better hashing."""
        try:
            if isinstance(inputs, str):
                return hashlib.blake2b(inputs.encode(), digest_size=16).hexdigest()
            elif isinstance(inputs, (np.ndarray, torch.Tensor)):
                # Use shape and dtype for tensor-like inputs for faster hashing
                shape_str = str(inputs.shape)
                dtype_str = str(inputs.dtype)
                # Sample a few values for differentiation without full tensor hashing
                if hasattr(inputs, 'flatten'):
                    flat = inputs.flatten()
                    sample_values = flat[:min(10, len(flat))].tolist() if len(flat) > 0 else []
                    sample_str = str(sample_values)
                else:
                    sample_str = ""
                combined = f"{shape_str}_{dtype_str}_{sample_str}"
                return hashlib.blake2b(combined.encode(), digest_size=16).hexdigest()
            else:
                # For other types, use string representation
                return hashlib.blake2b(str(inputs).encode(), digest_size=16).hexdigest()
        except Exception:
            return None
    
    def _get_from_optimized_cache(self, cache_key: str) -> Optional[PreprocessingResult]:
        """Get result from optimized cache with thread safety."""
        with self._tensor_cache_lock:
            if cache_key in self._tensor_cache:
                cached_item = self._tensor_cache[cache_key]
                # Check if cached item is still valid
                if time.time() - cached_item['timestamp'] < self.config.cache.cache_ttl_seconds:
                    return cached_item['result']
                else:
                    # Remove expired item
                    del self._tensor_cache[cache_key]
        return None
    
    def _store_in_optimized_cache(self, cache_key: str, result: PreprocessingResult):
        """Store result in optimized cache with automatic cleanup."""
        with self._tensor_cache_lock:
            # Implement LRU eviction if cache is full
            if len(self._tensor_cache) >= self._max_cache_size:
                # Remove oldest item
                oldest_key = min(self._tensor_cache.keys(), 
                               key=lambda k: self._tensor_cache[k]['timestamp'])
                del self._tensor_cache[oldest_key]
            
            self._tensor_cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
    
    def optimize_tensor_operations(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply various tensor optimizations."""
        if not self._enable_optimizations:
            return tensor
        
        # Use compiled operations if available
        if 'to_device' in self._compiled_ops and tensor.device != self.device:
            tensor = self._compiled_ops['to_device'](tensor, self.device)
        elif tensor.device != self.device:
            tensor = tensor.to(self.device, non_blocking=True)
        
        # Ensure tensor is contiguous for better performance
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Use memory format optimization for 4D tensors (images)
        if tensor.dim() == 4 and tensor.size(1) == 3:  # RGB images
            tensor = tensor.to(memory_format=torch.channels_last)
        
        return tensor
    
    def validate_inputs(self, inputs: Any) -> bool:
        """Validate input data with optimized checks."""
        if inputs is None:
            return False
        
        # Fast type checks
        if isinstance(inputs, (str, list, tuple)):
            return len(inputs) > 0
        elif isinstance(inputs, np.ndarray):
            return inputs.size > 0
        elif isinstance(inputs, torch.Tensor):
            return inputs.numel() > 0
        
        return True
    
    def get_cache_key(self, inputs: Any) -> Optional[str]:
        """Generate cache key for inputs (fallback method)."""
        return self._get_optimized_cache_key(inputs)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this preprocessor."""
        stats = self._stats.copy()
        
        # Calculate derived metrics
        if stats['num_processed'] > 0:
            stats['avg_processing_time'] = stats['total_preprocessing_time'] / stats['num_processed']
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        else:
            stats['avg_processing_time'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        stats['cache_size'] = len(self._tensor_cache)
        stats['memory_pool_size'] = len(self._memory_pool)
        
        return stats
    
    def clear_cache(self):
        """Clear all caches and reset statistics."""
        with self._tensor_cache_lock:
            self._tensor_cache.clear()
        
        # Reset memory pool
        with self._memory_pool_lock:
            for pool_item in self._memory_pool:
                pool_item['in_use'] = False
                pool_item['tensor'].zero_()
        
        # Reset stats
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_preprocessing_time': 0.0,
            'num_processed': 0,
            'memory_pool_hits': 0
        }
        
        # Force garbage collection
        gc.collect()
    
    def enable_optimizations(self, enable: bool = True):
        """Enable or disable optimizations."""
        self._enable_optimizations = enable
        if enable:
            self.logger.info("Preprocessing optimizations enabled")
        else:
            self.logger.info("Preprocessing optimizations disabled")


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
    """Optimized preprocessor for image inputs with advanced performance features."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.input_size = config.preprocessing.input_size
        self.mean = np.array(config.preprocessing.mean, dtype=np.float32)
        self.std = np.array(config.preprocessing.std, dtype=np.float32)
        self.interpolation = config.preprocessing.interpolation
        self.center_crop = config.preprocessing.center_crop
        self.normalize = config.preprocessing.normalize
        self.to_rgb = config.preprocessing.to_rgb
        
        # Performance optimizations for image processing
        self._use_opencv_optimizations = True
        self._use_pillow_simd = False
        self._transform_cache = {}
        self._numpy_ops_compiled = False
        
        # Pre-compute normalization tensors for faster operations
        self._mean_tensor = torch.tensor(self.mean, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self._std_tensor = torch.tensor(self.std, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        
        # Setup transforms with optimizations
        self._setup_transforms()
        
        # Try to detect and use optimized libraries
        self._detect_optimized_libraries()
        
    def _detect_optimized_libraries(self):
        """Detect available optimized libraries."""
        try:
            import cv2
            self._opencv_available = True
            self.logger.debug("OpenCV available for optimized image processing")
        except ImportError:
            self._opencv_available = False
        
        try:
            import PIL.Image
            # Check if Pillow-SIMD is available
            if hasattr(PIL.Image, '_simd'):
                self._use_pillow_simd = True
                self.logger.debug("Pillow-SIMD detected for faster image processing")
        except ImportError:
            pass
        
        try:
            import numba
            # Try to compile numpy operations with numba
            self._setup_numba_optimizations()
        except ImportError:
            self.logger.debug("Numba not available for JIT compilation")
    
    def _setup_numba_optimizations(self):
        """Setup numba-compiled functions for faster numpy operations."""
        try:
            import numba as nb
            
            @nb.jit(nopython=True, cache=True)
            def normalize_array_fast(arr, mean, std):
                return (arr - mean) / std
            
            @nb.jit(nopython=True, cache=True)
            def rgb_to_bgr_fast(arr):
                return arr[:, :, [2, 1, 0]]
            
            @nb.jit(nopython=True, cache=True, parallel=True)
            def resize_bilinear_fast(src, dst_height, dst_width):
                """Fast bilinear resize using Numba."""
                src_height, src_width = src.shape[:2]
                
                # Calculate scale factors
                scale_x = src_width / dst_width
                scale_y = src_height / dst_height
                
                # Create output array
                if len(src.shape) == 3:
                    result = np.empty((dst_height, dst_width, src.shape[2]), dtype=src.dtype)
                else:
                    result = np.empty((dst_height, dst_width), dtype=src.dtype)
                
                # Perform bilinear interpolation
                    for x in range(dst_width):
                        # Calculate source coordinates
                        src_x = x * scale_x
                        src_y = y * scale_y
                        
                        # Get integer coordinates
                        x0 = int(src_x)
                        y0 = int(src_y)
                        x1 = min(x0 + 1, src_width - 1)
                        y1 = min(y0 + 1, src_height - 1)
                        
                        # Calculate interpolation weights
                        wx = src_x - x0
                        wy = src_y - y0
                        
                        # Bilinear interpolation
                        if len(src.shape) == 3:
                            for c in range(src.shape[2]):
                                top = src[y0, x0, c] * (1 - wx) + src[y0, x1, c] * wx
                                bottom = src[y1, x0, c] * (1 - wx) + src[y1, x1, c] * wx
                                result[y, x, c] = top * (1 - wy) + bottom * wy
                        else:
                            top = src[y0, x0] * (1 - wx) + src[y0, x1] * wx
                            bottom = src[y1, x0] * (1 - wx) + src[y1, x1] * wx
                            result[y, x] = top * (1 - wy) + bottom * wy
                
                return result
            
            @nb.jit(nopython=True, cache=True, parallel=True)
            def apply_brightness_contrast_fast(arr, brightness, contrast):
                """Fast brightness and contrast adjustment using Numba."""
                result = np.empty_like(arr)
                for i in nb.prange(arr.size):
                    result.flat[i] = np.clip(arr.flat[i] * contrast + brightness, 0, 255)
                return result
            
            self._normalize_fast = normalize_array_fast
            self._rgb_to_bgr_fast = rgb_to_bgr_fast
            self._resize_bilinear_fast = resize_bilinear_fast
            self._brightness_contrast_fast = apply_brightness_contrast_fast
            self._numpy_ops_compiled = True
            self.logger.debug("Enhanced Numba-compiled numpy operations enabled")
            
        except Exception as e:
            self.logger.debug(f"Could not setup numba optimizations: {e}")
            self._numpy_ops_compiled = False
        
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
        """Optimized image preprocessing with performance enhancements."""
        start_time = time.perf_counter()
        
        try:
            # Fast path for already processed tensors
            if isinstance(inputs, torch.Tensor) and inputs.device == self.device:
                if inputs.dim() == 4 and inputs.size(1) == 3 and inputs.size(2) == self.input_size[0]:
                    # Already in correct format, just apply normalization if needed
                    if self.normalize:
                        if 'normalize' in self._compiled_ops:
                            tensor = self._compiled_ops['normalize'](inputs, self._mean_tensor, self._std_tensor)
                        else:
                            tensor = (inputs - self._mean_tensor) / self._std_tensor
                    else:
                        tensor = inputs
                    
                    processing_time = time.perf_counter() - start_time
                    return PreprocessingResult(
                        data=tensor,
                        metadata={
                            "input_type": "image",
                            "fast_path": True,
                            "original_shape": tuple(inputs.shape),
                            "final_shape": tuple(tensor.shape),
                            "preprocessor": self.__class__.__name__
                        },
                        original_shape=tuple(inputs.shape),
                        processing_time=processing_time
                    )
            
            # Load and convert image with optimizations
            image = self._load_image_optimized(inputs)
            original_shape = image.shape
            
            # Validate image before processing
            if image.size == 0:
                raise ValueError("Empty image provided")
            
            # Apply optimized transforms
            if self.use_torchvision and hasattr(self, 'transforms'):
                tensor = self._apply_torchvision_transforms_optimized(image)
            else:
                tensor = self._apply_opencv_transforms_optimized(image)
            
            # Apply tensor optimizations
            tensor = self.optimize_tensor_operations(tensor)
            
            # Add batch dimension if needed
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            
            processing_time = time.perf_counter() - start_time
            
            return PreprocessingResult(
                data=tensor,
                metadata={
                    "input_type": "image",
                    "original_shape": original_shape,
                    "final_shape": tuple(tensor.shape),
                    "preprocessor": self.__class__.__name__,
                    "optimized": True
                },
                original_shape=original_shape,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Optimized image preprocessing failed: {e}")
            self.logger.debug(f"Input type: {type(inputs)}, Input shape: {getattr(inputs, 'shape', 'N/A')}")
            
            # Try fallback processing
            try:
                return self._fallback_preprocessing(inputs, start_time)
            except Exception as fallback_error:
                self.logger.error(f"Fallback preprocessing also failed: {fallback_error}")
                raise PreprocessingError(f"Image preprocessing failed: {e}") from e
    
    def _load_image_optimized(self, inputs: Any) -> np.ndarray:
        """Optimized image loading with caching and faster I/O."""
        if isinstance(inputs, str):
            return self._load_image_from_path_optimized(inputs)
        elif isinstance(inputs, list):
            # Handle nested lists with optimized numpy operations
            image_array = np.array(inputs, dtype=np.float32)
            if image_array.size == 0:
                raise ValueError("Empty list provided")
            return image_array
        elif isinstance(inputs, np.ndarray):
            if inputs.size == 0:
                raise ValueError("Empty numpy array provided")
            return self._process_numpy_image_optimized(inputs)
        elif isinstance(inputs, torch.Tensor):
            if inputs.numel() == 0:
                raise ValueError("Empty tensor provided")
            return self._tensor_to_numpy_optimized(inputs)
        elif hasattr(inputs, 'convert'):  # PIL Image
            # Optimized PIL to numpy conversion
            if self._use_opencv_optimizations:
                return np.array(inputs.convert('RGB'))
            else:
                return np.array(inputs.convert('RGB'))
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
    
    def _load_image_from_path_optimized(self, path: str) -> np.ndarray:
        """Optimized image loading from file path with caching."""
        # Check if this path was recently loaded (simple file-based caching)
        cache_key = f"file_{hashlib.blake2b(path.encode(), digest_size=8).hexdigest()}"
        
        if self._opencv_available:
            return self._load_with_opencv_optimized(path)
        else:
            return self._load_image_from_path(path)  # Fallback to original method
    
    def _load_with_opencv_optimized(self, path: str) -> np.ndarray:
        """Load image with OpenCV optimizations."""
        import cv2
        
        # Use OpenCV's optimized loading
        if path.startswith(('http://', 'https://')):
            # For URLs, use original method
            return self._load_image_from_url(path)
        else:
            # Fast local file loading with OpenCV
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to load image: {path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
    
    def _process_numpy_image_optimized(self, image: np.ndarray) -> np.ndarray:
        """Optimized numpy image processing."""
        # Use compiled functions if available
        if self._numpy_ops_compiled and self.to_rgb:
            try:
                # Fast BGR to RGB conversion with numba
                if image.ndim == 3 and image.shape[2] == 3:
                    return self._rgb_to_bgr_fast(image)
            except Exception:
                pass
        
        # Fallback to original processing
        return self._process_numpy_image(image)
    
    def _tensor_to_numpy_optimized(self, tensor: torch.Tensor) -> np.ndarray:
        """Optimized tensor to numpy conversion."""
        # Use optimized tensor operations
        if tensor.device != torch.device('cpu'):
            # Use non-blocking transfer for better performance
            tensor = tensor.cpu()
        
        if tensor.ndim == 4:  # Batch dimension
            tensor = tensor[0]
        
        if tensor.ndim == 3:
            if tensor.shape[0] in [1, 3]:  # CHW format
                tensor = tensor.permute(1, 2, 0)  # Convert to HWC
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        
        return tensor.detach().numpy()
    
    def _apply_torchvision_transforms_optimized(self, image: np.ndarray) -> torch.Tensor:
        """Apply torchvision transforms with optimizations."""
        from PIL import Image
        
        # Fast validation and preprocessing
        if len(image.shape) == 3:
            h, w, c = image.shape
            if h <= 2 or w <= 2 or c > 4:
                self.logger.warning(f"Unusual image shape {image.shape}, using fallback")
                return self._create_fallback_tensor()
        
        # Optimized dtype handling
        if image.dtype in [np.float32, np.float64]:
            # Normalize to 0-255 range more efficiently
            image_min, image_max = image.min(), image.max()
            if image_max > image_min:
                image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
            else:
                image = np.full_like(image, 128, dtype=np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Handle channel conversion efficiently
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image, mode='L').convert('RGB')
        elif len(image.shape) == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(image, mode='RGB')
        elif len(image.shape) == 3 and image.shape[2] == 4:
            pil_image = Image.fromarray(image, mode='RGBA').convert('RGB')
        else:
            # Fallback processing
            if len(image.shape) == 3:
                image = np.mean(image, axis=2).astype(np.uint8)
            pil_image = Image.fromarray(image, mode='L').convert('RGB')
        
        # Apply transforms
        try:
            return self.transforms(pil_image)
        except Exception as e:
            self.logger.error(f"Transform application failed: {e}")
            return self._create_fallback_tensor()
    
    def _apply_opencv_transforms_optimized(self, image: np.ndarray) -> torch.Tensor:
        """Apply OpenCV-based transforms with optimizations."""
        if not self._opencv_available:
            return self._apply_opencv_transforms(image)  # Fallback
        
        import cv2
        
        # Optimized resize with better interpolation mapping
        if self.input_size:
            height, width = self.input_size
            interp_map = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "lanczos": cv2.INTER_LANCZOS4
            }
            interp = interp_map.get(self.interpolation, cv2.INTER_LINEAR)
            
            # Use optimized resize
            image = cv2.resize(image, (width, height), interpolation=interp)
        
        # Convert to float and normalize efficiently
        image = image.astype(np.float32)
        image /= 255.0
        
        # Apply normalization with compiled functions if available
        if self.normalize:
            if self._numpy_ops_compiled:
                try:
                    image = self._normalize_fast(image, self.mean, self.std)
                except Exception:
                    image = (image - self.mean) / self.std
            else:
                image = (image - self.mean) / self.std
        
        # Convert to CHW format efficiently
        image = np.transpose(image, (2, 0, 1))
        
        # Convert to tensor with optimizations
        tensor = torch.from_numpy(image).float()
        return self.optimize_tensor_operations(tensor)
    
    def _create_fallback_tensor(self) -> torch.Tensor:
        """Create a fallback tensor with optimal memory usage."""
        # Try to get from memory pool first
        fallback_shape = (3, 224, 224)
        tensor = self._get_tensor_from_pool(fallback_shape)
        
        if tensor is None:
            tensor = torch.zeros(fallback_shape, dtype=torch.float32, device=self.device)
            
            if self.normalize:
                # Apply normalization to the fallback tensor
                for i in range(3):
                    tensor[i] = (0.5 - self.mean[i]) / self.std[i]  # Gray fallback
        
        return tensor
    
    def _fallback_preprocessing(self, inputs: Any, start_time: float) -> PreprocessingResult:
        """Fallback preprocessing when optimized version fails."""
        try:
            # Use original preprocessing logic
            image = self._load_image(inputs)
            original_shape = image.shape
            
            if self.use_torchvision:
                tensor = self._apply_torchvision_transforms(image)
            else:
                tensor = self._apply_opencv_transforms(image)
            
            tensor = tensor.to(self.device)
            
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            
            processing_time = time.perf_counter() - start_time
            
            return PreprocessingResult(
                data=tensor,
                metadata={
                    "input_type": "image",
                    "original_shape": original_shape,
                    "final_shape": tuple(tensor.shape),
                    "preprocessor": self.__class__.__name__,
                    "fallback": True
                },
                original_shape=original_shape,
                processing_time=processing_time
            )
        except Exception as e:
            # Final fallback - create a default tensor
            fallback_tensor = self._create_fallback_tensor()
            fallback_tensor = fallback_tensor.unsqueeze(0)
            
            processing_time = time.perf_counter() - start_time
            
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
            
            # For tensors that already have a batch dimension, don't modify them
            # This is especially important for feature vectors that are already properly shaped
            if tensor.ndim >= 2 and tensor.shape[0] == 1:
                # Already has batch dimension of 1, use as-is
                pass
            elif tensor.ndim == 1:
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


class AudioPreprocessor(BasePreprocessor):
    """Preprocessor for audio inputs."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.sample_rate = config.custom_params.get("sample_rate", 16000)
        self.max_length = config.custom_params.get("max_audio_length", 30.0)  # seconds
        self.normalize = config.custom_params.get("normalize_audio", True)
        
        # Try to import and initialize audio preprocessor
        self.audio_processor = None
        self._setup_audio_processor()
    
    def _setup_audio_processor(self):
        """Setup audio preprocessor."""
        try:
            from .audio.audio_preprocessor import ComprehensiveAudioPreprocessor
            self.audio_processor = ComprehensiveAudioPreprocessor(
                sample_rate=self.sample_rate,
                normalize=self.normalize
            )
        except ImportError:
            self.logger.warning("Audio preprocessor not available. Audio processing will be limited.")
    
    def supports_input_type(self, input_type: InputType) -> bool:
        """Check if this preprocessor supports the given input type."""
        return input_type == InputType.AUDIO
    
    def preprocess(self, inputs: Any) -> PreprocessingResult:
        """Preprocess audio inputs."""
        start_time = time.time()
        
        try:
            if self.audio_processor:
                # Use comprehensive audio processor
                audio_data = self.audio_processor.process(inputs)
                
                # Convert to tensor if needed
                if isinstance(audio_data, list):
                    # Multiple chunks - concatenate or take first chunk
                    if len(audio_data) > 0:
                        tensor = torch.from_numpy(audio_data[0]).float()
                    else:
                        tensor = torch.zeros(self.sample_rate * 10)  # 10 seconds of silence
                elif isinstance(audio_data, np.ndarray):
                    tensor = torch.from_numpy(audio_data).float()
                else:
                    tensor = torch.tensor(audio_data).float()
            else:
                # Basic audio processing fallback
                tensor = self._basic_audio_processing(inputs)
            
            # Move to device
            tensor = tensor.to(self.device)
            
            # Add batch dimension if needed
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            
            # Truncate or pad to max length
            max_samples = int(self.max_length * self.sample_rate)
            if tensor.size(-1) > max_samples:
                tensor = tensor[:, :max_samples]
            elif tensor.size(-1) < max_samples:
                padding = max_samples - tensor.size(-1)
                tensor = torch.nn.functional.pad(tensor, (0, padding))
            
            processing_time = time.time() - start_time
            
            return PreprocessingResult(
                data=tensor,
                metadata={
                    "input_type": "audio",
                    "sample_rate": self.sample_rate,
                    "duration": tensor.size(-1) / self.sample_rate,
                    "shape": tuple(tensor.shape),
                    "preprocessor": self.__class__.__name__
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            raise PreprocessingError(f"Audio preprocessing failed: {e}") from e
    
    def _basic_audio_processing(self, inputs: Any) -> torch.Tensor:
        """Basic audio processing when comprehensive processor is not available."""
        if isinstance(inputs, str):
            # File path - try to load with basic methods
            return self._load_audio_basic(inputs)
        elif isinstance(inputs, (np.ndarray, torch.Tensor)):
            # Audio data
            if isinstance(inputs, np.ndarray):
                tensor = torch.from_numpy(inputs).float()
            else:
                tensor = inputs.float()
            
            # Ensure 1D
            if tensor.ndim > 1:
                tensor = tensor.mean(dim=0)  # Convert to mono
            
            return tensor
        else:
            self.logger.warning(f"Unsupported audio input type: {type(inputs)}")
            # Return silence
            return torch.zeros(self.sample_rate * 10)
    
    def _load_audio_basic(self, file_path: str) -> torch.Tensor:
        """Basic audio loading."""
        try:
            # Try librosa first
            import librosa
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return torch.from_numpy(audio).float()
        except ImportError:
            try:
                # Try torchaudio
                import torchaudio
                waveform, sr = torchaudio.load(file_path)
                
                # Convert to mono and resample if needed
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                if sr != self.sample_rate:
                    import torchaudio.functional as F
                    waveform = F.resample(waveform, sr, self.sample_rate)
                
                return waveform.squeeze()
            except ImportError:
                self.logger.error("No audio library available for loading audio files")
                return torch.zeros(self.sample_rate * 10)


class PreprocessorPipeline:
    """
    High-performance pipeline for chaining multiple preprocessors with advanced optimizations.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.preprocessors: List[BasePreprocessor] = []
        self.cache_enabled = config.cache.enable_caching
        self.cache = {} if self.cache_enabled else None
        self.max_cache_size = config.cache.cache_size
        self.executor = ThreadPoolExecutor(
            max_workers=config.performance.max_workers,
            thread_name_prefix="PreprocessorPipeline"
        )
        self.logger = logging.getLogger(f"{__name__}.PreprocessorPipeline")
        
        # Performance optimizations
        self._enable_parallel_processing = True
        self._batch_processing_enabled = True
        self._max_batch_size = getattr(config.batch, 'max_batch_size', 16)
        self._adaptive_batching = getattr(config.batch, 'adaptive_batching', True)
        
        # Statistics and monitoring
        self._pipeline_stats = {
            'total_processed': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_executions': 0,
            'batch_executions': 0
        }
        
        # Memory management
        self._memory_monitor = MemoryMonitor() if self._should_enable_memory_monitoring() else None
        
        # Add default preprocessors
        self._add_default_preprocessors()
    
    def _should_enable_memory_monitoring(self) -> bool:
        """Check if memory monitoring should be enabled."""
        try:
            # Enable if system has sufficient resources
            memory = psutil.virtual_memory()
            return memory.total > 4 * 1024 * 1024 * 1024  # 4GB threshold
        except Exception:
            return False
    
    def _add_default_preprocessors(self) -> None:
        """Add default preprocessors for each input type with optimizations."""
        # Add optimized image preprocessor
        image_processor = ImagePreprocessor(self.config)
        self.add_preprocessor(image_processor)
        
        # Add optimized text preprocessor
        text_processor = TextPreprocessor(self.config)
        self.add_preprocessor(text_processor)
        
        # Add optimized audio preprocessor
        audio_processor = AudioPreprocessor(self.config)
        self.add_preprocessor(audio_processor)
        
        # Add optimized tensor preprocessor
        tensor_processor = TensorPreprocessor(self.config)
        self.add_preprocessor(tensor_processor)
        
        # Add custom preprocessor for unknown types
        custom_processor = CustomPreprocessor(self.config)
        self.add_preprocessor(custom_processor)
    
    def add_preprocessor(self, preprocessor: BasePreprocessor) -> None:
        """Add a preprocessor to the pipeline."""
        self.preprocessors.append(preprocessor)
        self.logger.info(f"Added preprocessor: {preprocessor.__class__.__name__}")
        
        # Enable optimizations on the preprocessor
        if hasattr(preprocessor, 'enable_optimizations'):
            preprocessor.enable_optimizations(True)
    
    def preprocess(self, inputs: Any) -> PreprocessingResult:
        """Optimized preprocessing with caching and performance monitoring."""
        start_time = time.perf_counter()
        
        # Memory management check
        if self._memory_monitor:
            self._memory_monitor.check_memory_usage()
        
        # Check cache first with optimized key
        if self.cache_enabled:
            cache_key = self._get_optimized_cache_key(inputs)
            if cache_key and cache_key in self.cache:
                self._pipeline_stats['cache_hits'] += 1
                self.logger.debug(f"Cache hit for key: {cache_key}")
                result = self.cache[cache_key]
                result.processing_time = time.perf_counter() - start_time  # Update timing
                return result
            if cache_key:
                self._pipeline_stats['cache_misses'] += 1
        
        # Detect input type with optimizations
        input_type = self.detect_input_type(inputs)
        
        # Find appropriate preprocessor
        preprocessor = self._find_preprocessor(input_type)
        if not preprocessor:
            raise PreprocessingError(f"No preprocessor found for input type: {input_type}")
        
        # Validate inputs with fast checks
        if not preprocessor.validate_inputs(inputs):
            # Try fallback processing for invalid inputs
            self.logger.warning("Input validation failed, attempting fallback processing")
            try:
                if hasattr(preprocessor, '_fallback_preprocessing'):
                    result = preprocessor._fallback_preprocessing(inputs, start_time)
                else:
                    # Create a basic fallback tensor
                    if hasattr(preprocessor, '_create_fallback_tensor'):
                        tensor = preprocessor._create_fallback_tensor()
                    else:
                        tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
                    
                    result = PreprocessingResult(
                        data=tensor,
                        metadata={"fallback": True, "original_input_type": str(type(inputs))},
                        processing_time=time.perf_counter() - start_time,
                        success=True
                    )
                
                # Cache fallback result if enabled
                if self.cache_enabled and cache_key:
                    self._cache_result_optimized(cache_key, result)
                
                return result
            except Exception as fallback_error:
                self.logger.error(f"Fallback processing failed: {fallback_error}")
                raise PreprocessingError("Input validation failed and fallback processing unsuccessful")
        
        # Preprocess with optimizations
        if hasattr(preprocessor, '_preprocess_optimized'):
            result = preprocessor._preprocess_optimized(inputs)
        else:
            result = preprocessor.preprocess(inputs)
        
        # Cache result with optimized storage
        if self.cache_enabled and cache_key:
            self._cache_result_optimized(cache_key, result)
        
        # Update statistics
        processing_time = time.perf_counter() - start_time
        self._pipeline_stats['total_processed'] += 1
        self._pipeline_stats['total_processing_time'] += processing_time
        
        return result
    
    async def preprocess_async(self, inputs: Any) -> PreprocessingResult:
        """Asynchronous preprocessing with optimizations."""
        if not self.config.performance.enable_async:
            return self.preprocess(inputs)
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.preprocess, inputs)
    
    def preprocess_batch(self, inputs_list: List[Any]) -> List[PreprocessingResult]:
        """Optimized batch preprocessing with parallel execution."""
        if not inputs_list:
            return []
        
        start_time = time.perf_counter()
        
        # Check if batch processing should be used
        if (self._batch_processing_enabled and 
            len(inputs_list) > 1 and 
            len(inputs_list) <= self._max_batch_size):
            
            return self._preprocess_batch_optimized(inputs_list, start_time)
        else:
            return self._preprocess_batch_sequential(inputs_list, start_time)
    
    def _preprocess_batch_optimized(self, inputs_list: List[Any], start_time: float) -> List[PreprocessingResult]:
        """Optimized batch preprocessing with grouping by input type."""
        # Group inputs by type for more efficient processing
        grouped_inputs = self._group_inputs_by_type(inputs_list)
        results = [None] * len(inputs_list)
        
        for input_type, (indices, inputs) in grouped_inputs.items():
            preprocessor = self._find_preprocessor(input_type)
            if not preprocessor:
                # Handle missing preprocessor
                for idx in indices:
                    results[idx] = PreprocessingResult(
                        data=torch.empty(0),
                        metadata={"error": f"No preprocessor for {input_type}"},
                        processing_time=0.0
                    )
                continue
            
            # Process batch of same type
            if self._enable_parallel_processing and len(inputs) > 1:
                batch_results = self._process_batch_parallel(preprocessor, inputs)
            else:
                batch_results = [preprocessor.preprocess(inp) for inp in inputs]
            
            # Map results back to original positions
            for i, result in enumerate(batch_results):
                results[indices[i]] = result
        
        # Update statistics
        processing_time = time.perf_counter() - start_time
        self._pipeline_stats['batch_executions'] += 1
        self._pipeline_stats['total_processing_time'] += processing_time
        
        return results
    
    def _preprocess_batch_sequential(self, inputs_list: List[Any], start_time: float) -> List[PreprocessingResult]:
        """Sequential batch preprocessing for large batches or when optimization is disabled."""
        results = []
        
        if self._enable_parallel_processing and len(inputs_list) > 2:
            # Use parallel processing for large batches
            with ThreadPoolExecutor(max_workers=min(len(inputs_list), self.config.performance.max_workers)) as executor:
                future_to_input = {executor.submit(self.preprocess, inputs): i for i, inputs in enumerate(inputs_list)}
                results = [None] * len(inputs_list)
                
                for future in as_completed(future_to_input):
                    try:
                        result = future.result()
                        idx = future_to_input[future]
                        results[idx] = result
                    except Exception as e:
                        self.logger.error(f"Failed to preprocess input: {e}")
                        idx = future_to_input[future]
                        results[idx] = PreprocessingResult(
                            data=torch.empty(0),
                            metadata={"error": str(e)},
                            processing_time=0.0
                        )
            
            self._pipeline_stats['parallel_executions'] += 1
        else:
            # Sequential processing for small batches
            for inputs in inputs_list:
                try:
                    result = self.preprocess(inputs)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to preprocess input: {e}")
                    results.append(PreprocessingResult(
                        data=torch.empty(0),
                        metadata={"error": str(e)},
                        processing_time=0.0
                    ))
        
        return results
    
    def _group_inputs_by_type(self, inputs_list: List[Any]) -> Dict[InputType, Tuple[List[int], List[Any]]]:
        """Group inputs by their detected type for more efficient batch processing."""
        grouped = {}
        
        for i, inputs in enumerate(inputs_list):
            input_type = self.detect_input_type(inputs)
            if input_type not in grouped:
                grouped[input_type] = ([], [])
            grouped[input_type][0].append(i)  # indices
            grouped[input_type][1].append(inputs)  # inputs
        
        return grouped
    
    def _process_batch_parallel(self, preprocessor: BasePreprocessor, inputs_list: List[Any]) -> List[PreprocessingResult]:
        """Process a batch of inputs in parallel using the same preprocessor."""
        with ThreadPoolExecutor(max_workers=min(len(inputs_list), 4)) as executor:
            futures = [executor.submit(preprocessor.preprocess, inputs) for inputs in inputs_list]
            results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Parallel preprocessing failed: {e}")
                    results.append(PreprocessingResult(
                        data=torch.empty(0),
                        metadata={"error": str(e)},
                        processing_time=0.0
                    ))
        
        return results
    
    async def preprocess_batch_async(self, inputs_list: List[Any]) -> List[PreprocessingResult]:
        """Asynchronous batch preprocessing with optimizations."""
        if not self.config.performance.enable_async:
            return self.preprocess_batch(inputs_list)
        
        # Process batches asynchronously
        tasks = []
        batch_size = min(self._max_batch_size, len(inputs_list))
        
        for i in range(0, len(inputs_list), batch_size):
            batch = inputs_list[i:i + batch_size]
            task = asyncio.create_task(self._preprocess_batch_async_chunk(batch))
            tasks.append(task)
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                self.logger.error(f"Async batch processing failed: {batch_result}")
                continue
            results.extend(batch_result)
        
        return results
    
    async def _preprocess_batch_async_chunk(self, batch: List[Any]) -> List[PreprocessingResult]:
        """Process a chunk of the batch asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.preprocess_batch, batch)
    
    def _get_optimized_cache_key(self, inputs: Any) -> Optional[str]:
        """Generate optimized cache key with better performance."""
        try:
            if isinstance(inputs, str):
                return hashlib.blake2b(inputs.encode(), digest_size=16).hexdigest()
            elif hasattr(inputs, 'shape'):
                # For tensors/arrays, use shape + sample values for faster hashing
                shape_str = str(inputs.shape)
                dtype_str = str(getattr(inputs, 'dtype', 'unknown'))
                combined = f"{shape_str}_{dtype_str}"
                return hashlib.blake2b(combined.encode(), digest_size=16).hexdigest()
            else:
                return hashlib.blake2b(str(type(inputs)).encode(), digest_size=16).hexdigest()
        except Exception:
            return None
    
    def _cache_result_optimized(self, cache_key: str, result: PreprocessingResult) -> None:
        """Store result in cache with optimized memory management."""
        if len(self.cache) >= self.max_cache_size:
            # Implement LRU eviction with better performance
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: getattr(self.cache[k], 'timestamp', 0))
            del self.cache[oldest_key]
        
        # Add timestamp for LRU tracking
        result.timestamp = time.time()
        self.cache[cache_key] = result
        self.logger.debug(f"Cached result for key: {cache_key}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self._pipeline_stats.copy()
        
        # Calculate derived metrics
        if stats['total_processed'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['total_processed']
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        else:
            stats['avg_processing_time'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        # Add preprocessor-specific stats
        preprocessor_stats = {}
        for preprocessor in self.preprocessors:
            if hasattr(preprocessor, 'get_performance_stats'):
                preprocessor_stats[preprocessor.__class__.__name__] = preprocessor.get_performance_stats()
        
        stats['preprocessor_stats'] = preprocessor_stats
        stats['cache_size'] = len(self.cache) if self.cache else 0
        stats['num_preprocessors'] = len(self.preprocessors)
        
        # Memory usage if monitoring is enabled
        if self._memory_monitor:
            stats['memory_usage'] = self._memory_monitor.get_memory_stats()
        
        return stats
    
    def optimize_for_throughput(self):
        """Optimize pipeline settings for maximum throughput."""
        self._enable_parallel_processing = True
        self._batch_processing_enabled = True
        self._max_batch_size = min(32, self.config.batch.max_batch_size * 2)
        
        # Enable optimizations on all preprocessors
        for preprocessor in self.preprocessors:
            if hasattr(preprocessor, 'enable_optimizations'):
                preprocessor.enable_optimizations(True)
        
        self.logger.info("Pipeline optimized for throughput")
    
    def optimize_for_latency(self):
        """Optimize pipeline settings for minimum latency."""
        self._enable_parallel_processing = False
        self._batch_processing_enabled = False
        self._max_batch_size = 1
        
        # Clear caches to reduce memory footprint
        self.clear_cache()
        
        self.logger.info("Pipeline optimized for latency")
    
    def clear_cache(self) -> None:
        """Clear all caches and reset statistics."""
        if self.cache:
            self.cache.clear()
        
        # Clear preprocessor caches
        for preprocessor in self.preprocessors:
            if hasattr(preprocessor, 'clear_cache'):
                preprocessor.clear_cache()
        
        # Reset statistics
        self._pipeline_stats = {
            'total_processed': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_executions': 0,
            'batch_executions': 0
        }
        
        # Force garbage collection
        gc.collect()
        
        self.logger.info("Pipeline cache cleared and statistics reset")
    
    def detect_input_type(self, inputs: Any) -> InputType:
        """Optimized input type detection with caching."""
        self.logger.debug(f"Detecting input type for: {type(inputs)}")
        
        if isinstance(inputs, str):
            # Check if it's an audio path/URL
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma']
            if any(inputs.lower().endswith(ext) for ext in audio_extensions):
                self.logger.debug("Detected as AUDIO (file path)")
                return InputType.AUDIO
            # Check if it's an image path/URL
            elif any(inputs.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
                self.logger.debug("Detected as IMAGE (file path)")
                return InputType.IMAGE
            elif inputs.startswith(('http://', 'https://')) and 'image' in inputs.lower():
                self.logger.debug("Detected as IMAGE (URL)")
                return InputType.IMAGE
            elif inputs.startswith(('http://', 'https://')) and any(ext in inputs.lower() for ext in audio_extensions):
                self.logger.debug("Detected as AUDIO (URL)")
                return InputType.AUDIO
            else:
                self.logger.debug("Detected as TEXT")
                return InputType.TEXT
        elif isinstance(inputs, list):
            # Optimized list detection
            if len(inputs) == 3 and all(isinstance(channel, list) for channel in inputs):
                if all(len(channel) > 10 for channel in inputs):
                    first_channel = inputs[0]
                    if isinstance(first_channel, list) and len(first_channel) > 10:
                        if all(isinstance(row, list) and len(row) == len(first_channel[0]) for row in first_channel):
                            self.logger.debug(f"Detected as IMAGE (3D list [C,H,W])")
                            return InputType.IMAGE
            
            if isinstance(inputs, list) and len(inputs) > 0:
                if all(isinstance(x, (int, float)) for x in inputs):
                    if len(inputs) > 1000:
                        self.logger.debug("Detected as AUDIO (1D list of samples)")
                        return InputType.AUDIO
                    else:
                        self.logger.debug("Detected as TENSOR (1D list of numbers)")
                        return InputType.TENSOR
                elif all(isinstance(x, list) and all(isinstance(y, (int, float)) for y in x) for x in inputs):
                    self.logger.debug("Detected as TENSOR (2D list of numbers)")
                    return InputType.TENSOR
            
            self.logger.debug("Detected as CUSTOM (unrecognized list format)")
            return InputType.CUSTOM
        elif isinstance(inputs, (np.ndarray, torch.Tensor)):
            shape = inputs.shape
            dtype = inputs.dtype if isinstance(inputs, torch.Tensor) else inputs.dtype
            self.logger.debug(f"Detected tensor/array with shape: {shape}, dtype: {dtype}")
            
            # Fast integer tensor detection
            if isinstance(inputs, torch.Tensor):
                if inputs.dtype in [torch.int32, torch.int64, torch.long]:
                    self.logger.debug("Detected as TENSOR (integer tensor)")
                    return InputType.TENSOR
            elif isinstance(inputs, np.ndarray):
                if inputs.dtype in [np.int32, np.int64]:
                    self.logger.debug("Detected as TENSOR (integer array)")
                    return InputType.TENSOR
            
            # Optimized shape-based detection
            ndim = len(shape)
            if ndim == 4:
                if shape[1] in [1, 3]:
                    self.logger.debug(f"Detected as TENSOR (4D batch - shape: {shape})")
                    return InputType.TENSOR
                else:
                    self.logger.debug(f"Detected as TENSOR (4D other - shape: {shape})")
                    return InputType.TENSOR
            elif ndim == 3:
                if shape[0] == 3 or shape[-1] == 3:
                    self.logger.debug(f"Detected as IMAGE (3D tensor - shape: {shape})")
                    return InputType.IMAGE
                else:
                    self.logger.debug(f"Detected as TENSOR (3D non-image - shape: {shape})")
                    return InputType.TENSOR
            elif ndim == 2:
                if shape[1] > shape[0] and shape[1] > 100:
                    self.logger.debug(f"Detected as AUDIO (2D spectrogram - shape: {shape})")
                    return InputType.AUDIO
                elif shape[0] >= 10 and shape[1] >= 10 and abs(shape[0] - shape[1]) < max(shape) * 0.5:
                    self.logger.debug(f"Detected as IMAGE (2D tensor - shape: {shape})")
                    return InputType.IMAGE
                else:
                    self.logger.debug(f"Detected as TENSOR (2D features - shape: {shape})")
                    return InputType.TENSOR
            elif ndim == 1:
                if shape[0] > 1000:
                    self.logger.debug(f"Detected as AUDIO (1D samples - shape: {shape})")
                    return InputType.AUDIO
                else:
                    self.logger.debug(f"Detected as TENSOR (1D features - shape: {shape})")
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
    
    def _find_preprocessor(self, input_type: InputType) -> Optional[BasePreprocessor]:
        """Find preprocessor for the given input type."""
        for preprocessor in self.preprocessors:
            if preprocessor.supports_input_type(input_type):
                return preprocessor
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        return {
            "num_preprocessors": len(self.preprocessors),
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.cache) if self.cache else 0,
            "max_cache_size": self.max_cache_size,
            "preprocessor_types": [p.__class__.__name__ for p in self.preprocessors],
            "performance_stats": self.get_performance_stats()
        }


class MemoryMonitor:
    """Memory monitoring utility for preprocessing pipeline."""
    
    def __init__(self):
        self.memory_threshold = 0.8  # 80% memory usage threshold
        self.last_check_time = 0
        self.check_interval = 10  # Check every 10 seconds
    
    def check_memory_usage(self):
        """Check current memory usage and trigger cleanup if needed."""
        current_time = time.time()
        if current_time - self.last_check_time < self.check_interval:
            return
        
        self.last_check_time = current_time
        
        try:
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold * 100:
                # Trigger garbage collection
                gc.collect()
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
        except Exception:
            pass  # Ignore memory monitoring errors
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent_used': memory.percent
            }
        except Exception:
            return {}
    
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
    pipeline.add_preprocessor(AudioPreprocessor(config))
    pipeline.add_preprocessor(TensorPreprocessor(config))
    
    return pipeline


def create_preprocessor(preprocessor_type: str, config: InferenceConfig) -> BasePreprocessor:
    """
    Create a preprocessor of the specified type.
    
    Args:
        preprocessor_type: Type of preprocessor ('image', 'text', 'audio', 'tensor')
        config: Inference configuration
        
    Returns:
        Preprocessor instance
        
    Raises:
        ValueError: If preprocessor type is not supported
    """
    preprocessor_type = preprocessor_type.lower()
    
    if preprocessor_type == "image":
        return ImagePreprocessor(config)
    elif preprocessor_type == "text":
        return TextPreprocessor(config)
    elif preprocessor_type == "audio":
        return AudioPreprocessor(config)
    elif preprocessor_type == "tensor":
        return TensorPreprocessor(config)
    else:
        raise ValueError(f"Unsupported preprocessor type: {preprocessor_type}")

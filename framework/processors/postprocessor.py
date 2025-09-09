"""
Generic postprocessor framework for various output types.

This module provides a flexible postprocessing system that can handle
different model outputs (classification, detection, segmentation, etc.)
and convert them to user-friendly formats.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import numpy as np
import torch
import time
from dataclasses import dataclass, field
from enum import Enum
import functools
import threading
import weakref
import gc
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.config import InferenceConfig, ModelType


logger = logging.getLogger(__name__)


class OutputType(Enum):
    """Types of model outputs."""
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    REGRESSION = "regression"
    EMBEDDING = "embedding"
    AUDIO = "audio"
    CUSTOM = "custom"


@dataclass
class PostprocessorConfig:
    """Configuration for postprocessing operations."""
    
    output_type: str = "classification"
    num_classes: int = 1000
    class_names: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5
    max_detections: int = 100
    top_k: int = 5
    apply_softmax: bool = True
    apply_sigmoid: bool = False
    normalize_probabilities: bool = True
    return_raw_outputs: bool = False
    batch_size: int = 1
    device: str = "auto"
    enable_caching: bool = True
    cache_size: int = 1000
    enable_timing: bool = False
    precision: str = "float32"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.output_type not in ["classification", "detection", "segmentation", "regression", "embedding", "audio", "custom"]:
            raise ValueError(f"Unsupported output_type: {self.output_type}")
        
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if not 0 <= self.nms_threshold <= 1:
            raise ValueError("nms_threshold must be between 0 and 1")
        
        if self.max_detections <= 0:
            raise ValueError("max_detections must be positive")
        
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        
        if self.precision not in ["float16", "float32", "float64"]:
            raise ValueError("precision must be one of: float16, float32, float64")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'output_type': self.output_type,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'max_detections': self.max_detections,
            'top_k': self.top_k,
            'apply_softmax': self.apply_softmax,
            'apply_sigmoid': self.apply_sigmoid,
            'normalize_probabilities': self.normalize_probabilities,
            'return_raw_outputs': self.return_raw_outputs,
            'batch_size': self.batch_size,
            'device': self.device,
            'enable_caching': self.enable_caching,
            'cache_size': self.cache_size,
            'enable_timing': self.enable_timing,
            'precision': self.precision
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PostprocessorConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass
class PostprocessingResult:
    """Result of postprocessing operation."""
    predictions: Any
    confidence_scores: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format for compatibility."""
        # Convert predictions to dict if they have a to_dict method
        if hasattr(self.predictions, 'to_dict'):
            predictions_data = self.predictions.to_dict()
        else:
            predictions_data = self.predictions
            
        result = {
            "predictions": predictions_data,
            "processing_time": self.processing_time,
            "metadata": self.metadata.copy() if self.metadata else {}
        }
        
        if self.confidence_scores is not None:
            result["confidence_scores"] = self.confidence_scores
            
        # Add compatibility fields based on prediction type
        if hasattr(self.predictions, 'predicted_class'):
            result["predicted_class"] = self.predictions.predicted_class
            result["confidence"] = self.predictions.confidence
        
        return result


@dataclass
@dataclass
class ClassificationResult:
    """Classification result."""
    predicted_class: int
    class_name: Optional[str]
    confidence: float
    top_k_classes: Optional[List[Tuple[int, str, float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        # For single predictions, return the predicted class as a regular Python type
        # instead of a tensor to ensure JSON serialization works correctly
        
        result = {
            "predictions": self.predicted_class,  # Return as int for compatibility
            "confidence": self.confidence,
            "predicted_class": self.predicted_class,
        }
        
        if self.class_name is not None:
            result["class_name"] = self.class_name
            
        if self.top_k_classes is not None:
            result["top_k_classes"] = self.top_k_classes
            
        return result


@dataclass
class DetectionResult:
    """Object detection result."""
    boxes: List[Tuple[float, float, float, float]]  # [x1, y1, x2, y2]
    classes: List[int]
    class_names: Optional[List[str]]
    confidences: List[float]
    masks: Optional[np.ndarray] = None  # For instance segmentation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "boxes": self.boxes,
            "classes": self.classes,
            "confidences": self.confidences,
            "predictions": torch.tensor(self.classes)  # For compatibility
        }
        
        if self.class_names is not None:
            result["class_names"] = self.class_names
            
        if self.masks is not None:
            result["masks"] = self.masks
            
        return result


@dataclass
class SegmentationResult:
    """Segmentation result."""
    mask: np.ndarray
    contours: List[np.ndarray]
    area_pixels: int
    coverage_percentage: float
    largest_contour_area: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "mask": self.mask,
            "contours": self.contours,
            "area_pixels": self.area_pixels,
            "coverage_percentage": self.coverage_percentage,
            "largest_contour_area": self.largest_contour_area,
            "predictions": self.mask  # For compatibility
        }


@dataclass
class AudioResult:
    """Audio generation/processing result."""
    audio_data: np.ndarray
    sample_rate: int
    duration: float
    channels: int = 1
    format_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "audio_data": self.audio_data,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "channels": self.channels,
            "predictions": self.audio_data  # For compatibility
        }
        
        if self.format_info is not None:
            result["format_info"] = self.format_info
            
        return result
    
    def save_audio(self, file_path: str, format: str = "wav") -> bool:
        """Save audio to file."""
        try:
            # Try to use soundfile first
            import soundfile as sf
            sf.write(file_path, self.audio_data, self.sample_rate)
            return True
        except ImportError:
            try:
                # Fallback to scipy
                from scipy.io import wavfile
                # Ensure audio is in the right format for scipy
                if self.audio_data.dtype == np.float32 or self.audio_data.dtype == np.float64:
                    # Convert to 16-bit PCM
                    audio_int16 = (self.audio_data * 32767).astype(np.int16)
                else:
                    audio_int16 = self.audio_data
                wavfile.write(file_path, self.sample_rate, audio_int16)
                return True
            except ImportError:
                return False


class PostprocessingError(Exception):
    """Exception raised during postprocessing."""
    pass


class BasePostprocessor(ABC):
    """
    Abstract base class for all postprocessors with built-in optimizations.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance optimizations
        self._enable_optimizations = True
        self._result_cache = {}
        self._result_cache_lock = threading.RLock()
        self._max_cache_size = getattr(config.cache, 'cache_size', 100)
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_postprocessing_time': 0.0,
            'num_processed': 0
        }
        
        # Pre-compiled operations for faster postprocessing
        self._compiled_ops = {}
        self._setup_compilation_optimizations()
    
    def _setup_compilation_optimizations(self):
        """Setup compilation-based optimizations for postprocessing."""
        if hasattr(torch, 'compile') and self.config.device.use_torch_compile:
            try:
                self._compiled_ops['softmax'] = torch.compile(torch.softmax, mode=self.config.device.compile_mode)
                self._compiled_ops['argmax'] = torch.compile(torch.argmax, mode=self.config.device.compile_mode)
                self._compiled_ops['topk'] = torch.compile(torch.topk, mode=self.config.device.compile_mode)
                self.logger.info("Torch compilation enabled for postprocessing operations")
            except Exception as e:
                self.logger.warning(f"Could not setup torch compilation: {e}")
                self._compiled_ops = {}
        else:
            self._compiled_ops = {}
    
    @functools.lru_cache(maxsize=256)
    def _get_cached_tensor_info(self, shape: tuple, dtype: str) -> dict:
        """Cache tensor metadata for faster processing."""
        return {
            'shape': shape,
            'dtype': dtype,
            'ndim': len(shape),
            'total_elements': np.prod(shape) if shape else 0
        }
    
    def _get_optimized_cache_key(self, outputs: torch.Tensor) -> Optional[str]:
        """Generate optimized cache key for postprocessing results."""
        try:
            import hashlib
            # Use tensor shape and first/last few values for quick differentiation
            shape_str = str(outputs.shape)
            dtype_str = str(outputs.dtype)
            
            # Sample a few values for differentiation
            flat = outputs.flatten()
            if len(flat) > 10:
                sample_values = torch.cat([flat[:5], flat[-5:]]).cpu().numpy().tolist()
            else:
                sample_values = flat.cpu().numpy().tolist()
            
            combined = f"{shape_str}_{dtype_str}_{str(sample_values)}"
            return hashlib.blake2b(combined.encode(), digest_size=16).hexdigest()
        except Exception:
            return None
    
    def _get_from_cache(self, cache_key: str) -> Optional[Union[Dict[str, Any], 'PostprocessingResult']]:
        """Get result from optimized cache."""
        with self._result_cache_lock:
            if cache_key in self._result_cache:
                cached_item = self._result_cache[cache_key]
                if time.time() - cached_item['timestamp'] < self.config.cache.cache_ttl_seconds:
                    self._stats['cache_hits'] += 1
                    return cached_item['result']
                else:
                    del self._result_cache[cache_key]
        return None
    
    def _store_in_cache(self, cache_key: str, result: Union[Dict[str, Any], 'PostprocessingResult']):
        """Store result in optimized cache."""
        with self._result_cache_lock:
            if len(self._result_cache) >= self._max_cache_size:
                oldest_key = min(self._result_cache.keys(), 
                               key=lambda k: self._result_cache[k]['timestamp'])
                del self._result_cache[oldest_key]
            
            self._result_cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
    
    def postprocess_optimized(self, outputs: torch.Tensor, **kwargs) -> Union[Dict[str, Any], 'PostprocessingResult']:
        """Optimized postprocessing with caching and performance monitoring."""
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            cache_key = self._get_optimized_cache_key(outputs)
            if cache_key:
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    return cached_result
                self._stats['cache_misses'] += 1
            
            # Optimize tensor before processing
            outputs = self._optimize_tensor_for_postprocessing(outputs)
            
            # Perform actual postprocessing
            result = self.postprocess(outputs, **kwargs)
            
            # Cache result
            if cache_key:
                self._store_in_cache(cache_key, result)
            
            # Update stats
            processing_time = time.perf_counter() - start_time
            self._stats['total_postprocessing_time'] += processing_time
            self._stats['num_processed'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimized postprocessing failed: {e}")
            return self.postprocess(outputs, **kwargs)  # Fallback
    
    def _optimize_tensor_for_postprocessing(self, outputs: torch.Tensor) -> torch.Tensor:
        """Apply tensor optimizations before postprocessing."""
        if not self._enable_optimizations:
            return outputs
        
        # Ensure tensor is contiguous for better performance
        if not outputs.is_contiguous():
            outputs = outputs.contiguous()
        
        # Move to CPU if needed for postprocessing (most postprocessing is CPU-bound)
        if outputs.device.type != 'cpu' and outputs.numel() < 10000:  # Small tensors
            outputs = outputs.cpu()
        
        return outputs
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this postprocessor."""
        stats = self._stats.copy()
        
        if stats['num_processed'] > 0:
            stats['avg_processing_time'] = stats['total_postprocessing_time'] / stats['num_processed']
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        else:
            stats['avg_processing_time'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        stats['cache_size'] = len(self._result_cache)
        return stats
    
    def clear_cache(self):
        """Clear cache and reset statistics."""
        with self._result_cache_lock:
            self._result_cache.clear()
        
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_postprocessing_time': 0.0,
            'num_processed': 0
        }
        
        gc.collect()
    
    def enable_optimizations(self, enable: bool = True):
        """Enable or disable optimizations."""
        self._enable_optimizations = enable
    
    @abstractmethod
    def supports_output_type(self, output_type: OutputType) -> bool:
        """Check if this postprocessor supports the given output type."""
        pass
    
    @abstractmethod
    def postprocess(self, outputs: torch.Tensor, **kwargs) -> Union[Dict[str, Any], PostprocessingResult]:
        """Postprocess model outputs."""
        pass
    
    def validate_outputs(self, outputs: torch.Tensor) -> bool:
        """Validate model outputs."""
        return outputs is not None and isinstance(outputs, torch.Tensor)


class CustomPostprocessor(BasePostprocessor):
    """
    Custom postprocessor for generic/unknown output types.
    """
    
    def supports_output_type(self, output_type: OutputType) -> bool:
        """Supports custom output type."""
        return output_type == OutputType.CUSTOM
    
    def postprocess(self, outputs: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Simple postprocessing for custom outputs."""
        # If outputs is a dict (e.g., {'logits': tensor}), extract the first tensor value
        if isinstance(outputs, dict):
            for v in outputs.values():
                if hasattr(v, 'detach'):
                    outputs = v
                    break
        # Convert tensor to numpy for easier handling
        outputs_np = outputs.detach().cpu().numpy()
        # For compatibility with old tests expecting dict, create a simple result
        return {
            "predictions": outputs_np.tolist(),
            "raw_output": outputs_np.tolist(),
            "shape": outputs.shape,
            "prediction": "custom_result",
            "metadata": {
                "output_type": "custom",
                "shape": list(outputs.shape),
                "dtype": str(outputs.dtype)
            }
        }

    def _optimize_tensor_for_postprocessing(self, outputs: torch.Tensor) -> torch.Tensor:
        """Apply tensor optimizations before postprocessing (handle dict outputs)."""
        # If outputs is a dict (e.g., {'logits': tensor}), extract the first tensor value
        if isinstance(outputs, dict):
            for v in outputs.values():
                if hasattr(v, 'is_contiguous'):
                    outputs = v
                    break
        if not getattr(self, '_enable_optimizations', True):
            return outputs
        # Ensure tensor is contiguous for better performance
        if hasattr(outputs, 'is_contiguous') and not outputs.is_contiguous():
            outputs = outputs.contiguous()
        # Move to CPU if needed for postprocessing (most postprocessing is CPU-bound)
        if hasattr(outputs, 'device') and outputs.device.type != 'cpu' and outputs.numel() < 10000:  # Small tensors
            outputs = outputs.cpu()
        return outputs
    
    def validate_outputs(self, outputs: torch.Tensor) -> bool:
        """Validate custom outputs (always accepts)."""
        return isinstance(outputs, torch.Tensor)


class ClassificationPostprocessor(BasePostprocessor):
    """Optimized postprocessor for classification outputs."""
    
    def __init__(self, config: InferenceConfig, class_names: Optional[List[str]] = None):
        super().__init__(config)
        self.class_names = class_names
        self.apply_softmax = config.postprocessing.apply_softmax
        self.top_k = config.custom_params.get("top_k", 5)
        
        # Pre-allocate common tensors for better performance
        self._device = config.device.get_torch_device()
        self._preallocated_tensors = {}
        
        # Cache for class name lookups
        self._class_name_cache = {}
        if class_names:
            self._class_name_cache = {i: name for i, name in enumerate(class_names)}
    
    def supports_output_type(self, output_type: OutputType) -> bool:
        """Check if this postprocessor supports the given output type."""
        return output_type == OutputType.CLASSIFICATION
    
    def postprocess(self, outputs: torch.Tensor, **kwargs) -> PostprocessingResult:
        """Optimized classification postprocessing."""
        start_time = time.perf_counter()
        
        try:
            # Fast path for common cases
            if outputs.dim() == 2 and outputs.size(0) == 1:
                return self._postprocess_single_sample_optimized(outputs[0], start_time)
            elif outputs.dim() == 1:
                return self._postprocess_single_sample_optimized(outputs, start_time)
            elif outputs.dim() == 2:
                return self._postprocess_batch_optimized(outputs, start_time)
            else:
                raise ValueError(f"Unexpected output shape: {outputs.shape}")
                
        except Exception as e:
            self.logger.error(f"Optimized classification postprocessing failed: {e}")
            # Fallback to original implementation
            return self._postprocess_fallback(outputs, start_time)
    
    def _postprocess_single_sample_optimized(self, logits: torch.Tensor, start_time: float) -> PostprocessingResult:
        """Optimized processing for single sample."""
        # Use compiled operations if available
        if self.apply_softmax:
            if 'softmax' in self._compiled_ops:
                probabilities = self._compiled_ops['softmax'](logits, dim=0)
            else:
                probabilities = torch.softmax(logits, dim=0)
        else:
            probabilities = logits
        
        # Fast argmax and confidence extraction
        if 'argmax' in self._compiled_ops:
            predicted_class_idx = self._compiled_ops['argmax'](probabilities).item()
        else:
            predicted_class_idx = torch.argmax(probabilities).item()
        
        confidence = probabilities[predicted_class_idx].item()
        
        # Fast class name lookup
        predicted_class_name = self._class_name_cache.get(predicted_class_idx)
        
        # Optimized top-k computation
        top_k_classes = None
        if self.top_k > 0:
            k = min(self.top_k, len(probabilities))
            if 'topk' in self._compiled_ops:
                top_k_indices = self._compiled_ops['topk'](probabilities, k).indices
            else:
                top_k_indices = torch.topk(probabilities, k).indices
            
            top_k_classes = []
            for idx in top_k_indices:
                idx_val = idx.item()
                class_name = self._class_name_cache.get(idx_val)
                top_k_classes.append((idx_val, class_name, probabilities[idx_val].item()))
        
        # Create result
        result = ClassificationResult(
            predicted_class=predicted_class_idx,
            class_name=predicted_class_name,
            confidence=confidence,
            top_k_classes=top_k_classes
        )
        
        processing_time = time.perf_counter() - start_time
        
        return PostprocessingResult(
            predictions=result,
            confidence_scores=[confidence],
            metadata={
                "output_type": "classification",
                "num_classes": len(probabilities),
                "top_k": self.top_k,
                "optimized": True
            },
            processing_time=processing_time
        )
    
    def _postprocess_batch_optimized(self, outputs: torch.Tensor, start_time: float) -> Dict[str, Any]:
        """Optimized batch processing for classification."""
        # Apply softmax to entire batch efficiently
        if self.apply_softmax:
            if 'softmax' in self._compiled_ops:
                probabilities = self._compiled_ops['softmax'](outputs, dim=1)
            else:
                probabilities = torch.softmax(outputs, dim=1)
        else:
            probabilities = outputs
        
        # Batch operations for better performance
        if 'argmax' in self._compiled_ops:
            predicted_classes = self._compiled_ops['argmax'](probabilities, dim=1)
        else:
            predicted_classes = torch.argmax(probabilities, dim=1)
        
        confidences = torch.max(probabilities, dim=1).values
        
        processing_time = time.perf_counter() - start_time
        
        # Return dict format for batch compatibility
        return {
            "predictions": predicted_classes.tolist(),
            "confidence": confidences.mean().item(),
            "confidences": confidences.tolist(),
            "probabilities": probabilities.tolist(),
            "metadata": {
                "output_type": "classification",
                "batch_size": outputs.size(0),
                "num_classes": outputs.size(1),
                "processing_time": processing_time,
                "optimized": True
            }
        }
    
    def _postprocess_fallback(self, outputs: torch.Tensor, start_time: float) -> PostprocessingResult:
        """Fallback to original postprocessing logic."""
        # Original implementation logic here (simplified for brevity)
        # This would contain the original postprocess method logic
        
        if outputs.dim() == 2 and outputs.size(0) == 1:
            logits = outputs[0]
        elif outputs.dim() == 1:
            logits = outputs
        else:
            # Take first sample for fallback
            logits = outputs.view(-1, outputs.size(-1))[0]
        
        if self.apply_softmax:
            probabilities = torch.softmax(logits, dim=0)
        else:
            probabilities = logits
        
        predicted_class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_idx].item()
        predicted_class_name = None
        if self.class_names and 0 <= predicted_class_idx < len(self.class_names):
            predicted_class_name = self.class_names[predicted_class_idx]
        
        result = ClassificationResult(
            predicted_class=predicted_class_idx,
            class_name=predicted_class_name,
            confidence=confidence
        )
        
        processing_time = time.perf_counter() - start_time
        
        return PostprocessingResult(
            predictions=result,
            confidence_scores=[confidence],
            metadata={
                "output_type": "classification",
                "fallback": True
            },
            processing_time=processing_time
        )
    
    def supports_output_type(self, output_type: OutputType) -> bool:
        """Check if this postprocessor supports the given output type."""
        return output_type == OutputType.CLASSIFICATION
    
    def postprocess(self, outputs: torch.Tensor, **kwargs) -> PostprocessingResult:
        """Postprocess classification outputs."""
        start_time = time.time()
        
        try:
            # Handle batch dimension
            if outputs.dim() == 2:
                # Batch of classifications (batch_size, num_classes)
                batch_size = outputs.size(0)
                
                if batch_size == 1:
                    # Single sample in batch
                    logits = outputs[0]
                    
                    # Apply softmax if requested
                    if self.apply_softmax:
                        probabilities = torch.softmax(logits, dim=0)
                    else:
                        probabilities = logits
                    
                    # Get top prediction
                    predicted_class_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_class_idx].item()
                    
                    # Get class name if available
                    predicted_class_name = None
                    if self.class_names and 0 <= predicted_class_idx < len(self.class_names):
                        predicted_class_name = self.class_names[predicted_class_idx]
                    
                    # Get top-k predictions
                    top_k_classes = None
                    if self.top_k > 0:
                        top_k_indices = torch.topk(probabilities, min(self.top_k, len(probabilities))).indices
                        top_k_classes = []
                        
                        for idx in top_k_indices:
                            idx = idx.item()
                            class_name = self.class_names[idx] if self.class_names else None
                            top_k_classes.append((idx, class_name, probabilities[idx].item()))
                    
                    # Create result
                    result = ClassificationResult(
                        predicted_class=predicted_class_idx,
                        class_name=predicted_class_name,
                        confidence=confidence,
                        top_k_classes=top_k_classes
                    )
                    
                    processing_time = time.time() - start_time
                    
                    return PostprocessingResult(
                        predictions=result,
                        confidence_scores=[confidence],
                        metadata={
                            "output_type": "classification",
                            "num_classes": len(probabilities),
                            "top_k": self.top_k
                        },
                        processing_time=processing_time
                    )
                else:
                    # Multiple samples in batch - return dict format for compatibility
                    # Apply softmax if requested
                    if self.apply_softmax:
                        probabilities = torch.softmax(outputs, dim=1)
                    else:
                        probabilities = outputs
                    
                    # Get predictions for entire batch
                    predicted_classes = torch.argmax(probabilities, dim=1)
                    confidences = torch.max(probabilities, dim=1).values
                    
                    processing_time = time.time() - start_time
                    
                    # Return dict format for batch compatibility
                    result = {
                        "predictions": predicted_classes.tolist(),  # Convert tensor to list
                        "confidence": confidences.mean().item(),  # Average confidence
                        "confidences": confidences.tolist(),  # Convert tensor to list
                        "probabilities": probabilities.tolist(),  # Convert tensor to list
                        "metadata": {
                            "output_type": "classification",
                            "batch_size": batch_size,
                            "num_classes": outputs.size(1),
                            "processing_time": processing_time
                        }
                    }
                    return result
                    
            elif outputs.dim() == 1:
                # Single prediction without batch dimension
                logits = outputs
                
                # Apply softmax if requested
                if self.apply_softmax:
                    probabilities = torch.softmax(logits, dim=0)
                else:
                    probabilities = logits
                
                # Get top prediction
                predicted_class_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class_idx].item()
                
                # Get class name if available
                predicted_class_name = None
                if self.class_names and 0 <= predicted_class_idx < len(self.class_names):
                    predicted_class_name = self.class_names[predicted_class_idx]
                
                # Get top-k predictions
                top_k_classes = None
                if self.top_k > 0:
                    top_k_indices = torch.topk(probabilities, min(self.top_k, len(probabilities))).indices
                    top_k_classes = []
                    
                    for idx in top_k_indices:
                        idx = idx.item()
                        class_name = self.class_names[idx] if self.class_names else None
                        top_k_classes.append((idx, class_name, probabilities[idx].item()))
                
                # Create result
                result = ClassificationResult(
                    predicted_class=predicted_class_idx,
                    class_name=predicted_class_name,
                    confidence=confidence,
                    top_k_classes=top_k_classes
                )
                
                processing_time = time.time() - start_time
                
                return PostprocessingResult(
                    predictions=result,
                    confidence_scores=[confidence],
                    metadata={
                        "output_type": "classification",
                        "num_classes": len(probabilities),
                        "top_k": self.top_k
                    },
                    processing_time=processing_time
                )
            else:
                raise ValueError(f"Unexpected output shape: {outputs.shape}")
            
        except Exception as e:
            self.logger.error(f"Classification postprocessing failed: {e}")
            raise PostprocessingError(f"Classification postprocessing failed: {e}") from e


class DetectionPostprocessor(BasePostprocessor):
    """Postprocessor for object detection outputs."""
    
    def __init__(self, config: InferenceConfig, class_names: Optional[List[str]] = None):
        super().__init__(config)
        self.class_names = class_names
        self.confidence_threshold = config.postprocessing.threshold
        self.nms_threshold = config.postprocessing.nms_threshold
        self.max_detections = config.postprocessing.max_detections
    
    def supports_output_type(self, output_type: OutputType) -> bool:
        """Check if this postprocessor supports the given output type."""
        return output_type == OutputType.DETECTION
    
    def postprocess(self, outputs: torch.Tensor, **kwargs) -> PostprocessingResult:
        """Postprocess detection outputs."""
        start_time = time.time()
        
        try:
            # Handle different output formats
            if self._is_yolo_format(outputs):
                result = self._postprocess_yolo(outputs, **kwargs)
            else:
                result = self._postprocess_generic(outputs, **kwargs)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Detection postprocessing failed: {e}")
            raise PostprocessingError(f"Detection postprocessing failed: {e}") from e
    
    def _is_yolo_format(self, outputs: torch.Tensor) -> bool:
        """Check if outputs are in YOLO format."""
        # YOLO typically outputs [batch, num_anchors, 5+num_classes] or similar
        return outputs.dim() == 3 and outputs.size(-1) > 5
    
    def _postprocess_yolo(self, outputs: torch.Tensor, **kwargs) -> PostprocessingResult:
        """Postprocess YOLO-style outputs."""
        # Handle batch dimension
        if outputs.size(0) == 1:
            detections = outputs[0]  # Remove batch dimension
        else:
            detections = outputs[0]  # Take first in batch
        
        # Filter by confidence
        obj_conf = detections[:, 4]  # Objectness confidence
        conf_mask = obj_conf > self.confidence_threshold
        detections = detections[conf_mask]
        
        if len(detections) == 0:
            return self._empty_detection_result()
        
        # Extract boxes, confidences, and classes
        boxes = detections[:, :4]  # x_center, y_center, width, height
        confidences = detections[:, 4]
        class_probs = detections[:, 5:]
        
        # Convert to corner format and get final confidences and classes
        final_boxes = []
        final_confidences = []
        final_classes = []
        
        for i, (box, obj_conf, class_prob) in enumerate(zip(boxes, confidences, class_probs)):
            # Get best class
            class_conf, class_idx = torch.max(class_prob, dim=0)
            final_conf = obj_conf * class_conf
            
            if final_conf > self.confidence_threshold:
                # Convert from center format to corner format
                x_center, y_center, width, height = box
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                final_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                final_confidences.append(final_conf.item())
                final_classes.append(class_idx.item())
        
        # Apply NMS
        if len(final_boxes) > 1:
            final_boxes, final_confidences, final_classes = self._apply_nms(
                final_boxes, final_confidences, final_classes
            )
        
        # Limit number of detections
        if len(final_boxes) > self.max_detections:
            # Sort by confidence and take top detections
            sorted_indices = sorted(range(len(final_confidences)), 
                                  key=lambda i: final_confidences[i], reverse=True)
            sorted_indices = sorted_indices[:self.max_detections]
            
            final_boxes = [final_boxes[i] for i in sorted_indices]
            final_confidences = [final_confidences[i] for i in sorted_indices]
            final_classes = [final_classes[i] for i in sorted_indices]
        
        # Get class names
        class_names = None
        if self.class_names:
            class_names = [self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}" 
                          for cls in final_classes]
        
        detection_result = DetectionResult(
            boxes=final_boxes,
            classes=final_classes,
            class_names=class_names,
            confidences=final_confidences
        )
        
        return PostprocessingResult(
            predictions=detection_result,
            confidence_scores=final_confidences,
            metadata={
                "output_type": "detection",
                "num_detections": len(final_boxes),
                "confidence_threshold": self.confidence_threshold,
                "nms_threshold": self.nms_threshold
            }
        )
    
    def _postprocess_generic(self, outputs: torch.Tensor, **kwargs) -> PostprocessingResult:
        """Postprocess generic detection outputs."""
        # Placeholder for generic detection postprocessing
        # This would be implemented based on specific model requirements
        return self._empty_detection_result()
    
    def _apply_nms(self, boxes: List[List[float]], confidences: List[float], 
                   classes: List[int]) -> Tuple[List[List[float]], List[float], List[int]]:
        """Apply Non-Maximum Suppression."""
        try:
            import torchvision.ops as ops
            
            # Convert to tensors
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            confidences_tensor = torch.tensor(confidences, dtype=torch.float32)
            
            # Apply NMS
            keep_indices = ops.nms(boxes_tensor, confidences_tensor, self.nms_threshold)
            
            # Filter results
            filtered_boxes = [boxes[i] for i in keep_indices]
            filtered_confidences = [confidences[i] for i in keep_indices]
            filtered_classes = [classes[i] for i in keep_indices]
            
            return filtered_boxes, filtered_confidences, filtered_classes
            
        except ImportError:
            # Fallback: simple NMS implementation
            self.logger.warning("torchvision not available, using simple NMS fallback")
            return self._simple_nms(boxes, confidences, classes)
    
    def _simple_nms(self, boxes: List[List[float]], confidences: List[float], 
                    classes: List[int]) -> Tuple[List[List[float]], List[float], List[int]]:
        """Simple NMS implementation."""
        # Sort by confidence
        indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
        
        keep = []
        while indices:
            current = indices.pop(0)
            keep.append(current)
            
            # Remove boxes with high IoU
            remaining = []
            for idx in indices:
                if self._calculate_iou(boxes[current], boxes[idx]) < self.nms_threshold:
                    remaining.append(idx)
            indices = remaining
        
        return ([boxes[i] for i in keep], 
                [confidences[i] for i in keep], 
                [classes[i] for i in keep])
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU)."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate union
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _empty_detection_result(self) -> PostprocessingResult:
        """Create empty detection result."""
        detection_result = DetectionResult(
            boxes=[],
            classes=[],
            class_names=[],
            confidences=[]
        )
        
        return PostprocessingResult(
            predictions=detection_result,
            confidence_scores=[],
            metadata={"output_type": "detection", "num_detections": 0}
        )


class SegmentationPostprocessor(BasePostprocessor):
    """Postprocessor for segmentation outputs."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.threshold = config.postprocessing.threshold
        self.apply_sigmoid = config.postprocessing.apply_sigmoid
        self.min_contour_area = config.custom_params.get("min_contour_area", 100)
        self.max_contours = config.custom_params.get("max_contours", 100)
    
    def supports_output_type(self, output_type: OutputType) -> bool:
        """Check if this postprocessor supports the given output type."""
        return output_type == OutputType.SEGMENTATION
    
    def postprocess(self, outputs: torch.Tensor, **kwargs) -> PostprocessingResult:
        """Postprocess segmentation outputs."""
        start_time = time.time()
        
        try:
            # Handle different output formats
            if hasattr(outputs, 'masks') and outputs.masks is not None:
                # YOLO-style segmentation
                result = self._postprocess_yolo_segmentation(outputs, **kwargs)
            else:
                # Generic segmentation tensor
                result = self._postprocess_generic_segmentation(outputs, **kwargs)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Segmentation postprocessing failed: {e}")
            raise PostprocessingError(f"Segmentation postprocessing failed: {e}") from e
    
    def _postprocess_yolo_segmentation(self, outputs: Any, **kwargs) -> PostprocessingResult:
        """Postprocess YOLO segmentation outputs."""
        try:
            # Extract masks from YOLO output
            if hasattr(outputs, 'masks') and outputs.masks is not None:
                masks = outputs.masks.data
            else:
                masks = outputs[0].masks.data if isinstance(outputs, list) else None
            
            if masks is None or len(masks) == 0:
                return self._empty_segmentation_result()
            
            # Combine all masks
            combined_mask = self._combine_masks(masks)
            
            # Find contours
            contours = self._find_contours(combined_mask)
            
            # Calculate metrics
            area_pixels = int(np.count_nonzero(combined_mask))
            total_pixels = combined_mask.size
            coverage_percentage = (area_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            largest_contour_area = 0
            if contours:
                try:
                    import cv2
                    largest_contour_area = max(cv2.contourArea(c) for c in contours)
                except ImportError:
                    largest_contour_area = len(contours[0]) if contours else 0
            
            segmentation_result = SegmentationResult(
                mask=combined_mask,
                contours=contours,
                area_pixels=area_pixels,
                coverage_percentage=coverage_percentage,
                largest_contour_area=largest_contour_area
            )
            
            return PostprocessingResult(
                predictions=segmentation_result,
                metadata={
                    "output_type": "segmentation",
                    "num_contours": len(contours),
                    "mask_shape": combined_mask.shape,
                    "coverage_percentage": coverage_percentage
                }
            )
            
        except Exception as e:
            self.logger.error(f"YOLO segmentation postprocessing failed: {e}")
            return self._empty_segmentation_result()
    
    def _postprocess_generic_segmentation(self, outputs: torch.Tensor, **kwargs) -> PostprocessingResult:
        """Postprocess generic segmentation outputs."""
        # Handle batch dimension
        if outputs.dim() == 4 and outputs.size(0) == 1:
            mask_logits = outputs[0]
        elif outputs.dim() == 3:
            mask_logits = outputs
        elif outputs.dim() == 2:
            mask_logits = outputs.unsqueeze(0)  # Add channel dimension
        else:
            raise ValueError(f"Unexpected output shape: {outputs.shape}")
        
        # Handle multi-class segmentation
        if mask_logits.size(0) > 1:
            # Take argmax across classes
            mask_logits = torch.argmax(mask_logits, dim=0).float()
        else:
            mask_logits = mask_logits[0]
        
        # Convert to numpy
        mask_np = mask_logits.detach().cpu().numpy()
        
        # Apply sigmoid if requested
        if self.apply_sigmoid:
            mask_np = 1 / (1 + np.exp(-mask_np))
        
        # Threshold to create binary mask
        binary_mask = (mask_np > self.threshold).astype(np.uint8) * 255
        
        # Find contours
        contours = self._find_contours(binary_mask)
        
        # Calculate metrics
        area_pixels = int(np.count_nonzero(binary_mask))
        total_pixels = binary_mask.size
        coverage_percentage = (area_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        largest_contour_area = 0
        if contours:
            try:
                import cv2
                largest_contour_area = max(cv2.contourArea(c) for c in contours)
            except ImportError:
                largest_contour_area = len(contours[0]) if contours else 0
        
        segmentation_result = SegmentationResult(
            mask=binary_mask,
            contours=contours,
            area_pixels=area_pixels,
            coverage_percentage=coverage_percentage,
            largest_contour_area=largest_contour_area
        )
        
        return PostprocessingResult(
            predictions=segmentation_result,
            metadata={
                "output_type": "segmentation",
                "num_contours": len(contours),
                "mask_shape": binary_mask.shape,
                "coverage_percentage": coverage_percentage
            }
        )
    
    def _combine_masks(self, masks: torch.Tensor) -> np.ndarray:
        """Combine multiple masks into one binary mask."""
        if len(masks) == 0:
            return np.zeros((100, 100), dtype=np.uint8)  # Default empty mask
        
        # Combine all masks with OR operation
        combined_mask = torch.zeros_like(masks[0], dtype=torch.uint8)
        
        for mask in masks:
            mask_binary = (mask > self.threshold).to(torch.uint8)
            combined_mask = torch.logical_or(combined_mask, mask_binary).to(torch.uint8)
        
        combined_mask_np = (combined_mask * 255).detach().cpu().numpy().astype(np.uint8)
        return combined_mask_np
    
    def _find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Find contours in the binary mask."""
        try:
            import cv2
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by area
            filtered_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
            
            # Limit number of contours
            if len(filtered_contours) > self.max_contours:
                filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
                filtered_contours = filtered_contours[:self.max_contours]
            
            return filtered_contours
            
        except ImportError:
            self.logger.warning("OpenCV not available, returning empty contours")
            return []
    
    def _empty_segmentation_result(self) -> PostprocessingResult:
        """Create empty segmentation result."""
        segmentation_result = SegmentationResult(
            mask=np.zeros((100, 100), dtype=np.uint8),
            contours=[],
            area_pixels=0,
            coverage_percentage=0.0,
            largest_contour_area=0.0
        )
        
        return PostprocessingResult(
            predictions=segmentation_result,
            metadata={"output_type": "segmentation", "num_contours": 0}
        )


class AudioPostprocessor(BasePostprocessor):
    """Postprocessor for audio outputs."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.sample_rate = config.custom_params.get("sample_rate", 16000)
        self.normalize_audio = config.custom_params.get("normalize_audio", True)
        self.output_format = config.custom_params.get("audio_output_format", "float32")
    
    def supports_output_type(self, output_type: OutputType) -> bool:
        """Check if this postprocessor supports the given output type."""
        return output_type == OutputType.AUDIO
    
    def postprocess(self, outputs: torch.Tensor, **kwargs) -> PostprocessingResult:
        """Postprocess audio outputs."""
        start_time = time.time()
        
        try:
            # Handle different output formats
            if outputs.dim() == 3:
                # Batch of audio samples [batch, channels, samples] or [batch, samples, features]
                if outputs.size(0) == 1:
                    audio_tensor = outputs[0]  # Remove batch dimension
                else:
                    # Multiple samples in batch - take first one for now
                    audio_tensor = outputs[0]
                    self.logger.warning(f"Multiple audio samples in batch ({outputs.size(0)}), taking first")
            elif outputs.dim() == 2:
                # Single audio sample [channels, samples] or [samples, features]
                audio_tensor = outputs
            elif outputs.dim() == 1:
                # Single channel audio [samples]
                audio_tensor = outputs.unsqueeze(0)  # Add channel dimension
            else:
                raise ValueError(f"Unexpected audio output shape: {outputs.shape}")
            
            # Convert to numpy
            audio_np = audio_tensor.detach().cpu().numpy()
            
            # Handle different formats
            if audio_np.ndim == 2:
                # Check if it's [channels, samples] or [samples, features]
                if audio_np.shape[0] <= 2:  # Likely [channels, samples] (mono or stereo)
                    channels, samples = audio_np.shape
                    if channels == 1:
                        audio_data = audio_np[0]  # Extract mono audio
                    else:
                        # Convert stereo to mono
                        audio_data = np.mean(audio_np, axis=0)
                        channels = 1
                else:
                    # Likely [samples, features] - extract main feature
                    if audio_np.shape[1] == 1:
                        audio_data = audio_np[:, 0]
                    else:
                        # Take first feature as audio data
                        audio_data = audio_np[:, 0]
                        self.logger.warning(f"Multiple features ({audio_np.shape[1]}), taking first")
                    channels = 1
            else:
                # 1D audio data
                audio_data = audio_np
                channels = 1
            
            # Normalize if requested
            if self.normalize_audio:
                audio_data = self._normalize_audio(audio_data)
            
            # Convert to desired output format
            if self.output_format == "int16":
                audio_data = (audio_data * 32767).astype(np.int16)
            elif self.output_format == "int32":
                audio_data = (audio_data * 2147483647).astype(np.int32)
            else:  # float32 default
                audio_data = audio_data.astype(np.float32)
            
            # Calculate duration
            duration = len(audio_data) / self.sample_rate
            
            # Create result
            audio_result = AudioResult(
                audio_data=audio_data,
                sample_rate=self.sample_rate,
                duration=duration,
                channels=channels,
                format_info={
                    "format": self.output_format,
                    "normalized": self.normalize_audio,
                    "original_shape": tuple(outputs.shape)
                }
            )
            
            processing_time = time.time() - start_time
            
            return PostprocessingResult(
                predictions=audio_result,
                metadata={
                    "output_type": "audio",
                    "sample_rate": self.sample_rate,
                    "duration": duration,
                    "channels": channels,
                    "format": self.output_format
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Audio postprocessing failed: {e}")
            raise PostprocessingError(f"Audio postprocessing failed: {e}") from e
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        if len(audio) == 0:
            return audio
        
        # Check if already normalized
        if np.max(np.abs(audio)) <= 1.0:
            return audio
        
        # Normalize to peak
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio / peak
        
        return audio


class PostprocessorPipeline:
    """
    High-performance pipeline for managing multiple postprocessors with advanced optimizations.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.postprocessors: Dict[OutputType, BasePostprocessor] = {}
        self.logger = logging.getLogger(f"{__name__}.PostprocessorPipeline")
        
        # Performance optimizations
        self._enable_parallel_processing = True
        self._batch_processing_enabled = True
        self._max_batch_size = getattr(config.batch, 'max_batch_size', 16)
        self.executor = ThreadPoolExecutor(
            max_workers=config.performance.max_workers,
            thread_name_prefix="PostprocessorPipeline"
        )
        
        # Statistics and monitoring
        self._pipeline_stats = {
            'total_processed': 0,
            'total_processing_time': 0.0,
            'parallel_executions': 0,
            'batch_executions': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Type detection cache for faster processing
        self._type_detection_cache = {}
        self._type_detection_cache_lock = threading.RLock()
        
        # Add default postprocessors
        self._add_default_postprocessors()
        
    def _add_default_postprocessors(self) -> None:
        """Add default postprocessors for each output type with optimizations."""
        # Add optimized classification postprocessor
        classification_processor = ClassificationPostprocessor(self.config)
        self.add_postprocessor(OutputType.CLASSIFICATION, classification_processor)
        
        # Add detection postprocessor
        detection_processor = DetectionPostprocessor(self.config)
        self.add_postprocessor(OutputType.DETECTION, detection_processor)
        
        # Add segmentation postprocessor
        segmentation_processor = SegmentationPostprocessor(self.config)
        self.add_postprocessor(OutputType.SEGMENTATION, segmentation_processor)
        
        # Add audio postprocessor
        audio_processor = AudioPostprocessor(self.config)
        self.add_postprocessor(OutputType.AUDIO, audio_processor)
        
        # Add custom postprocessor for unknown types
        custom_processor = CustomPostprocessor(self.config)
        self.add_postprocessor(OutputType.CUSTOM, custom_processor)
    
    def add_postprocessor(self, output_type: OutputType, postprocessor: BasePostprocessor) -> None:
        """Add a postprocessor for a specific output type."""
        self.postprocessors[output_type] = postprocessor
        self.logger.info(f"Added postprocessor for {output_type.value}: {postprocessor.__class__.__name__}")
        
        # Enable optimizations
        if hasattr(postprocessor, 'enable_optimizations'):
            postprocessor.enable_optimizations(True)
    
    def postprocess(self, outputs, output_type: OutputType, **kwargs) -> Union[Dict[str, Any], PostprocessingResult]:
        """Optimized postprocessing with performance monitoring."""
        start_time = time.perf_counter()
        
        if output_type not in self.postprocessors:
            raise PostprocessingError(f"No postprocessor found for output type: {output_type}")
        
        postprocessor = self.postprocessors[output_type]
        
        # Extract tensor from outputs if needed
        tensor_outputs = self._extract_tensor_optimized(outputs)
        
        # Validate outputs
        if isinstance(tensor_outputs, torch.Tensor) and not postprocessor.validate_outputs(tensor_outputs):
            raise PostprocessingError("Output validation failed")
        
        # Use optimized postprocessing if available
        if hasattr(postprocessor, 'postprocess_optimized'):
            result = postprocessor.postprocess_optimized(tensor_outputs, **kwargs)
        else:
            result = postprocessor.postprocess(tensor_outputs, **kwargs)
        
        # Update statistics
        processing_time = time.perf_counter() - start_time
        self._pipeline_stats['total_processed'] += 1
        self._pipeline_stats['total_processing_time'] += processing_time
        
        return result
    
    def _extract_tensor_optimized(self, outputs):
        """Optimized tensor extraction from various output formats."""
        # Fast path for common cases
        if isinstance(outputs, torch.Tensor):
            return outputs
        
        # Handle Hugging Face outputs
        if hasattr(outputs, 'logits'):
            return outputs.logits
        elif hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            tensor_outputs = outputs[0]
            if hasattr(tensor_outputs, 'logits'):
                return tensor_outputs.logits
            return tensor_outputs
        
        return outputs
    
    def detect_output_type(self, outputs) -> OutputType:
        """Optimized output type detection with caching."""
        # Generate cache key for type detection
        type_key = self._get_type_detection_cache_key(outputs)
        
        with self._type_detection_cache_lock:
            if type_key in self._type_detection_cache:
                return self._type_detection_cache[type_key]
        
        # Perform detection
        output_type = self._detect_output_type_impl(outputs)
        
        # Cache result
        with self._type_detection_cache_lock:
            if len(self._type_detection_cache) < 1000:  # Limit cache size
                self._type_detection_cache[type_key] = output_type
        
        return output_type
    
    def _get_type_detection_cache_key(self, outputs) -> str:
        """Generate cache key for output type detection."""
        try:
            if isinstance(outputs, torch.Tensor):
                return f"tensor_{outputs.shape}_{outputs.dtype}"
            elif hasattr(outputs, 'logits'):
                return f"hf_logits_{outputs.logits.shape}_{outputs.logits.dtype}"
            else:
                return f"other_{type(outputs).__name__}"
        except Exception:
            return f"unknown_{type(outputs).__name__}"
    
    def _detect_output_type_impl(self, outputs) -> OutputType:
        """Implementation of output type detection."""
        # Extract tensor for analysis
        tensor_outputs = self._extract_tensor_optimized(outputs)
        
        if not isinstance(tensor_outputs, torch.Tensor):
            return OutputType.CUSTOM
        
        # Fast dimension-based detection
        ndim = tensor_outputs.dim()
        shape = tensor_outputs.shape
        
        if ndim == 1:
            return OutputType.AUDIO if shape[0] > 1000 else OutputType.CUSTOM
        elif ndim == 2:
            if shape[1] > 1000 or shape[0] > 1000:
                return OutputType.AUDIO
            elif shape[1] > 1 and shape[1] < 1000:
                return OutputType.CLASSIFICATION
            else:
                return OutputType.CUSTOM
        elif ndim == 3:
            if shape[-1] > 1000:
                return OutputType.AUDIO
            elif shape[-1] > 5:
                return OutputType.DETECTION
            elif min(shape[-2:]) > 10:
                return OutputType.SEGMENTATION
            else:
                return OutputType.CUSTOM
        elif ndim >= 4:
            if min(shape[-2:]) > 10:
                return OutputType.SEGMENTATION
            else:
                return OutputType.CUSTOM
        else:
            return OutputType.CUSTOM
    
    def auto_postprocess(self, outputs, **kwargs) -> Union[Dict[str, Any], PostprocessingResult]:
        """Automatically detect output type and postprocess with optimizations."""
        output_type = self.detect_output_type(outputs)
        return self.postprocess(outputs, output_type, **kwargs)
    
    def postprocess_batch(self, outputs_list: List[Any], output_types: Optional[List[OutputType]] = None) -> List[Union[Dict[str, Any], PostprocessingResult]]:
        """Optimized batch postprocessing."""
        if not outputs_list:
            return []
        
        start_time = time.perf_counter()
        
        # Auto-detect output types if not provided
        if output_types is None:
            output_types = [self.detect_output_type(outputs) for outputs in outputs_list]
        
        # Group by output type for more efficient processing
        if self._batch_processing_enabled and len(outputs_list) > 1:
            results = self._postprocess_batch_grouped(outputs_list, output_types, start_time)
        else:
            results = self._postprocess_batch_sequential(outputs_list, output_types, start_time)
        
        return results
    
    def _postprocess_batch_grouped(self, outputs_list: List[Any], output_types: List[OutputType], start_time: float) -> List[Union[Dict[str, Any], PostprocessingResult]]:
        """Group outputs by type for more efficient batch processing."""
        # Group by output type
        grouped = {}
        for i, (outputs, output_type) in enumerate(zip(outputs_list, output_types)):
            if output_type not in grouped:
                grouped[output_type] = []
            grouped[output_type].append((i, outputs))
        
        # Process each group
        results = [None] * len(outputs_list)
        
        for output_type, items in grouped.items():
            if output_type not in self.postprocessors:
                # Handle missing postprocessor
                for idx, _ in items:
                    results[idx] = {"error": f"No postprocessor for {output_type}"}
                continue
            
            postprocessor = self.postprocessors[output_type]
            
            # Process group
            if self._enable_parallel_processing and len(items) > 1:
                group_results = self._process_group_parallel(postprocessor, items)
            else:
                group_results = [self.postprocess(outputs, output_type) for _, outputs in items]
            
            # Map results back to original positions
            for (idx, _), result in zip(items, group_results):
                results[idx] = result
        
        # Update statistics
        processing_time = time.perf_counter() - start_time
        self._pipeline_stats['batch_executions'] += 1
        self._pipeline_stats['total_processing_time'] += processing_time
        
        return results
    
    def _postprocess_batch_sequential(self, outputs_list: List[Any], output_types: List[OutputType], start_time: float) -> List[Union[Dict[str, Any], PostprocessingResult]]:
        """Sequential batch processing."""
        results = []
        
        if self._enable_parallel_processing and len(outputs_list) > 2:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=min(len(outputs_list), self.config.performance.max_workers)) as executor:
                future_to_idx = {
                    executor.submit(self.postprocess, outputs, output_type): i 
                    for i, (outputs, output_type) in enumerate(zip(outputs_list, output_types))
                }
                
                results = [None] * len(outputs_list)
                for future in as_completed(future_to_idx):
                    try:
                        result = future.result()
                        idx = future_to_idx[future]
                        results[idx] = result
                    except Exception as e:
                        self.logger.error(f"Parallel postprocessing failed: {e}")
                        idx = future_to_idx[future]
                        results[idx] = {"error": str(e)}
            
            self._pipeline_stats['parallel_executions'] += 1
        else:
            # Sequential processing
            for outputs, output_type in zip(outputs_list, output_types):
                try:
                    result = self.postprocess(outputs, output_type)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Sequential postprocessing failed: {e}")
                    results.append({"error": str(e)})
        
        return results
    
    def _process_group_parallel(self, postprocessor: BasePostprocessor, items: List[Tuple[int, Any]]) -> List[Union[Dict[str, Any], PostprocessingResult]]:
        """Process a group of outputs in parallel using the same postprocessor."""
        with ThreadPoolExecutor(max_workers=min(len(items), 4)) as executor:
            futures = [executor.submit(postprocessor.postprocess, outputs) for _, outputs in items]
            results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Group parallel postprocessing failed: {e}")
                    results.append({"error": str(e)})
        
        return results
    
    async def postprocess_async(self, outputs, output_type: OutputType, **kwargs) -> Union[Dict[str, Any], PostprocessingResult]:
        """Asynchronous postprocessing."""
        if not self.config.performance.enable_async:
            return self.postprocess(outputs, output_type, **kwargs)
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.postprocess, outputs, output_type, **kwargs)
    
    async def auto_postprocess_async(self, outputs, **kwargs) -> Union[Dict[str, Any], PostprocessingResult]:
        """Asynchronous auto postprocessing."""
        output_type = self.detect_output_type(outputs)
        return await self.postprocess_async(outputs, output_type, **kwargs)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self._pipeline_stats.copy()
        
        # Calculate derived metrics
        if stats['total_processed'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['total_processed']
        else:
            stats['avg_processing_time'] = 0.0
        
        # Add postprocessor-specific stats
        postprocessor_stats = {}
        for output_type, postprocessor in self.postprocessors.items():
            if hasattr(postprocessor, 'get_performance_stats'):
                postprocessor_stats[output_type.value] = postprocessor.get_performance_stats()
        
        stats['postprocessor_stats'] = postprocessor_stats
        stats['num_postprocessors'] = len(self.postprocessors)
        stats['type_detection_cache_size'] = len(self._type_detection_cache)
        
        return stats
    
    def optimize_for_throughput(self):
        """Optimize pipeline for maximum throughput."""
        self._enable_parallel_processing = True
        self._batch_processing_enabled = True
        self._max_batch_size = min(32, self.config.batch.max_batch_size * 2)
        
        # Enable optimizations on all postprocessors
        for postprocessor in self.postprocessors.values():
            if hasattr(postprocessor, 'enable_optimizations'):
                postprocessor.enable_optimizations(True)
        
        self.logger.info("Postprocessor pipeline optimized for throughput")
    
    def optimize_for_latency(self):
        """Optimize pipeline for minimum latency."""
        self._enable_parallel_processing = False
        self._batch_processing_enabled = False
        self._max_batch_size = 1
        
        # Clear caches
        self.clear_cache()
        
        self.logger.info("Postprocessor pipeline optimized for latency")
    
    def clear_cache(self):
        """Clear all caches and reset statistics."""
        with self._type_detection_cache_lock:
            self._type_detection_cache.clear()
        
        # Clear postprocessor caches
        for postprocessor in self.postprocessors.values():
            if hasattr(postprocessor, 'clear_cache'):
                postprocessor.clear_cache()
        
        # Reset statistics
        self._pipeline_stats = {
            'total_processed': 0,
            'total_processing_time': 0.0,
            'parallel_executions': 0,
            'batch_executions': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        gc.collect()
        self.logger.info("Postprocessor pipeline cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get postprocessing statistics."""
        return {
            "num_postprocessors": len(self.postprocessors),
            "supported_output_types": [ot.value for ot in self.postprocessors.keys()],
            "performance_stats": self.get_performance_stats()
        }


def create_default_postprocessing_pipeline(config: InferenceConfig, 
                                         class_names: Optional[List[str]] = None) -> PostprocessorPipeline:
    """Create a default postprocessing pipeline with common postprocessors."""
    pipeline = PostprocessorPipeline(config)
    
    # Add common postprocessors
    pipeline.add_postprocessor(OutputType.CLASSIFICATION, 
                              ClassificationPostprocessor(config, class_names))
    pipeline.add_postprocessor(OutputType.DETECTION, 
                              DetectionPostprocessor(config, class_names))
    pipeline.add_postprocessor(OutputType.SEGMENTATION, 
                              SegmentationPostprocessor(config))
    pipeline.add_postprocessor(OutputType.AUDIO, 
                              AudioPostprocessor(config))
    
    return pipeline


def create_postprocessor(output_type: Union[OutputType, str], 
                        config: Optional[InferenceConfig] = None,
                        class_names: Optional[List[str]] = None) -> BasePostprocessor:
    """
    Factory function to create a postprocessor for specific output type.
    
    Args:
        output_type: Type of output to process
        config: Inference configuration
        class_names: List of class names for classification/detection
        
    Returns:
        Postprocessor instance
    """
    if config is None:
        config = InferenceConfig()
    
    if isinstance(output_type, str):
        output_type = OutputType(output_type)
    
    if output_type == OutputType.CLASSIFICATION:
        return ClassificationPostprocessor(config, class_names)
    elif output_type == OutputType.DETECTION:
        return DetectionPostprocessor(config, class_names)
    elif output_type == OutputType.SEGMENTATION:
        return SegmentationPostprocessor(config)
    elif output_type == OutputType.REGRESSION:
        # Use custom postprocessor for regression
        return CustomPostprocessor(config, lambda x: x)  # Identity function for regression
    elif output_type == OutputType.EMBEDDING:
        # Use custom postprocessor for embeddings
        return CustomPostprocessor(config, lambda x: x.cpu().numpy() if torch.is_tensor(x) else x)
    elif output_type == OutputType.AUDIO:
        return AudioPostprocessor(config)
    else:
        # Return a custom postprocessor for unknown types
        return CustomPostprocessor(config, lambda x: x)

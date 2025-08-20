"""
Base model interface and abstract classes for the PyTorch inference framework.

This module defines the core interfaces that all model implementations must follow,
ensuring consistency and interoperability across different model types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

from ..core.config import InferenceConfig

logger = logging.getLogger(__name__)

# Import security mitigations
try:
    from .security import PyTorchSecurityMitigation, ECDSASecurityMitigation
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    logger.warning("Security mitigations not available")

# Initialize security if available
if SECURITY_AVAILABLE:
    _pytorch_security = PyTorchSecurityMitigation()
    logger.info("Security mitigations initialized for base models")
else:
    _pytorch_security = None


@dataclass
class ModelMetadata:
    """Metadata about a model."""
    name: str
    version: str
    model_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    description: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class ModelInferenceError(Exception):
    """Exception raised when model inference fails."""
    pass


class BaseModel(ABC):
    """
    Abstract base class for all model implementations.
    
    This class defines the interface that all model implementations must follow,
    ensuring consistency and proper resource management.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = config.device.get_torch_device()
        self.model: Optional[nn.Module] = None
        self.metadata: Optional[ModelMetadata] = None
        self._is_loaded = False
        self._compiled_model = None
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load the model from the given path.
        
        Args:
            model_path: Path to the model file
            
        Raises:
            ModelLoadError: If model loading fails
        """
        pass
    
    @abstractmethod
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """
        Preprocess inputs before inference.
        
        Args:
            inputs: Raw inputs (images, text, etc.)
            
        Returns:
            Preprocessed tensor ready for model inference
        """
        pass
    
    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass through the model.
        
        Args:
            inputs: Preprocessed input tensor
            
        Returns:
            Raw model outputs
        """
        pass
    
    @abstractmethod
    def postprocess(self, outputs: torch.Tensor) -> Any:
        """
        Postprocess model outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Processed outputs in the desired format
        """
        pass
    
    def predict(self, inputs: Any) -> Any:
        """
        Complete prediction pipeline: preprocess -> forward -> postprocess.
        
        Args:
            inputs: Raw inputs
            
        Returns:
            Final predictions
        """
        if not self._is_loaded:
            raise ModelInferenceError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess
            preprocessed_inputs = self.preprocess(inputs)
            
            # Forward pass
            with torch.no_grad():
                try:
                    raw_outputs = self.forward(preprocessed_inputs)
                except Exception as e:
                    # Handle compilation errors by falling back to non-compiled model
                    if "CppCompileError" in str(e) and self._compiled_model is not None:
                        self.logger.warning("Torch compilation failed, falling back to non-compiled model")
                        self.config.device.use_torch_compile = False
                        self._compiled_model = None
                        raw_outputs = self.forward(preprocessed_inputs)
                    else:
                        raise
            
            # Postprocess
            predictions = self.postprocess(raw_outputs)
            
            # Convert to dict for backward compatibility if needed
            if hasattr(predictions, 'to_dict'):
                return predictions.to_dict()
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise ModelInferenceError(f"Prediction failed: {e}") from e
    
    def predict_batch(self, inputs: List[Any]) -> List[Any]:
        """
        Batch prediction with automatic batching.
        
        Args:
            inputs: List of raw inputs
            
        Returns:
            List of predictions
        """
        if not inputs:
            return []
        
        batch_size = self.config.batch.batch_size
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            if len(batch) == 1:
                # Single item
                result = self.predict(batch[0])
                results.append(result)
            else:
                # True batch processing
                batch_result = self.predict_batch_internal(batch)
                results.extend(batch_result)
        
        return results
    
    def predict_batch_internal(self, inputs: List[Any]) -> List[Any]:
        """
        Internal batch processing method. Override for true batch processing.
        
        Args:
            inputs: List of raw inputs (batch)
            
        Returns:
            List of predictions
        """
        # Default implementation: process individually
        return [self.predict(inp) for inp in inputs]
    
    def warmup(self, num_iterations: int = None) -> None:
        """
        Warmup the model with dummy inputs.
        
        Args:
            num_iterations: Number of warmup iterations
        """
        if num_iterations is None:
            num_iterations = self.config.performance.warmup_iterations
        
        if not self._is_loaded:
            self.logger.warning("Model not loaded, skipping warmup")
            return
        
        self.logger.info(f"Warming up model with {num_iterations} iterations")
        
        try:
            # Create dummy input based on preprocessing config
            dummy_input = self._create_dummy_input()
            
            for i in range(num_iterations):
                try:
                    with torch.no_grad():
                        _ = self.forward(dummy_input)
                except Exception as e:
                    self.logger.warning(f"Warmup iteration {i+1} failed: {e}")
                    # If first iteration fails due to compilation, disable compilation and retry
                    if i == 0 and "CppCompileError" in str(e):
                        self.logger.warning("Disabling torch.compile due to compilation error")
                        self.config.device.use_torch_compile = False
                        self._compiled_model = None
                        try:
                            with torch.no_grad():
                                _ = self.forward(dummy_input)
                        except Exception as e2:
                            self.logger.error(f"Warmup failed even without compilation: {e2}")
                            break
                    else:
                        # For other errors, just continue
                        continue
            
            self.logger.info("Model warmup completed")
            
        except Exception as e:
            self.logger.warning(f"Warmup failed: {e}. Model may still work for inference.")
    
    def compile_model(self) -> None:
        """Compile the model using torch.compile for optimization."""
        if not self._is_loaded or not self.config.device.use_torch_compile:
            return
        
        if not hasattr(torch, 'compile'):
            self.logger.warning("torch.compile not available, skipping compilation")
            return
        
        try:
            self.logger.info("Compiling model with torch.compile")
            self._compiled_model = torch.compile(
                self.model,
                mode=self.config.device.compile_mode,
                fullgraph=False
            )
            self.logger.info("Model compilation completed")
        except Exception as e:
            self.logger.warning(f"Model compilation failed: {e}. Continuing without compilation.")
            # Don't raise the exception, just continue without compilation
    
    def get_model_for_inference(self) -> nn.Module:
        """Get the model instance to use for inference (compiled or original)."""
        return self._compiled_model if self._compiled_model is not None else self.model
    
    def optimize_for_inference(self) -> None:
        """Apply various optimizations for inference."""
        if not self._is_loaded:
            return
        
        # Set to evaluation mode
        self.model.eval()
        
        # Move to target device
        self.model.to(self.device)
        
        # Apply FP16 if requested
        if self.config.device.use_fp16:
            self.model.half()
        
        # Compile model if requested
        self.compile_model()
        
        # Configure CUDA optimizations
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def _create_dummy_input(self) -> torch.Tensor:
        """Create dummy input for warmup. Override in subclasses."""
        # Default implementation for image models
        if hasattr(self.config, 'preprocessing') and hasattr(self.config.preprocessing, 'input_size'):
            height, width = self.config.preprocessing.input_size
            return torch.randn(1, 3, height, width, device=self.device)
        else:
            # Generic dummy input
            return torch.randn(1, 10, device=self.device)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        memory_info = {}
        
        if self.device.type == 'cuda':
            memory_info['gpu_allocated_mb'] = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            memory_info['gpu_reserved_mb'] = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            memory_info['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        
        # CPU memory would require psutil
        try:
            import psutil
            process = psutil.Process()
            memory_info['cpu_memory_mb'] = process.memory_info().rss / (1024 ** 2)
        except ImportError:
            pass
        
        # Add model parameter info if available
        if self._is_loaded and self.model:
            try:
                # Handle both real models and Mock objects
                if hasattr(self.model, 'parameters') and callable(self.model.parameters):
                    total_params = sum(p.numel() for p in self.model.parameters())
                    trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    memory_info["total_params"] = total_params
                    memory_info["trainable_params"] = trainable_params
                    
                    # Estimate model size (very rough approximation)
                    # Assume 4 bytes per parameter (float32)
                    memory_info["model_size_mb"] = total_params * 4 / (1024 ** 2)
            except (TypeError, AttributeError):
                # Skip parameter counting for Mock objects or other types
                pass
        
        return memory_info
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    @is_loaded.setter
    def is_loaded(self, value: bool):
        """Set model loaded status."""
        self._is_loaded = value
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            "loaded": self._is_loaded,
            "device": str(self.device),
            "config": self.config,
        }
        
        if self.metadata:
            # Convert metadata to dict for compatibility
            if hasattr(self.metadata, '__dict__'):
                info["metadata"] = self.metadata.__dict__.copy()
            else:
                info["metadata"] = {
                    "model_type": getattr(self.metadata, 'model_type', 'pytorch'),
                    "input_shape": getattr(self.metadata, 'input_shape', None),
                    "output_shape": getattr(self.metadata, 'output_shape', None),
                    "num_parameters": getattr(self.metadata, 'num_parameters', None),
                    "framework_version": getattr(self.metadata, 'framework_version', None)
                }
        
        if self._is_loaded:
            info["memory_usage"] = self.get_memory_usage()
            
            # Model parameters count
            if self.model:
                try:
                    # Handle both real models and Mock objects
                    if hasattr(self.model, 'parameters') and callable(self.model.parameters):
                        total_params = sum(p.numel() for p in self.model.parameters())
                        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                        info["total_parameters"] = total_params
                        info["trainable_parameters"] = trainable_params
                except (TypeError, AttributeError):
                    # Skip parameter counting for Mock objects or other types
                    pass
        
        return info


class ModelManager:
    """
    Manager class for handling multiple model instances and lifecycle.
    """
    
    def __init__(self):
        self._models: Dict[str, BaseModel] = {}
        self.logger = logging.getLogger(f"{__name__}.ModelManager")
        
        # Import downloader here to avoid circular imports
        self._downloader = None
    
    def register_model(self, name: str, model: BaseModel) -> None:
        """Register a model instance."""
        if name in self._models:
            self.logger.warning(f"Model '{name}' already exists, replacing")
        
        self._models[name] = model
        self.logger.info(f"Registered model '{name}'")
    
    def get_model(self, name: str) -> BaseModel:
        """Get a registered model."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found")
        
        return self._models[name]
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self._models.keys())
    
    def is_model_loaded(self, name: str) -> bool:
        """Check if a model is loaded."""
        return name in self._models and self._models[name].is_loaded
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded models."""
        return [name for name, model in self._models.items() if model.is_loaded]
    
    def load_model(self, model_path: Union[str, Path], config: InferenceConfig) -> BaseModel:
        """Load a model with security mitigations."""
        # Create appropriate model adapter
        from ..adapters.model_adapters import ModelAdapterFactory
        
        adapter = ModelAdapterFactory.create_adapter(model_path, config)
        
        # Register the model in the manager
        self.register_model(str(model_path), adapter)
        
        # Use secure model loading if security is available
        if SECURITY_AVAILABLE and _pytorch_security:
            with _pytorch_security.secure_torch_context():
                adapter.load_model(model_path)
                adapter.optimize_for_inference()
                adapter.warmup()
            self.logger.info(f"Model '{model_path}' loaded with security mitigations")
        else:
            adapter.load_model(model_path)
            adapter.optimize_for_inference()
            adapter.warmup()
            self.logger.warning(f"Model '{model_path}' loaded without security mitigations")
        
        return adapter
    
    def load_registered_model(self, name: str, model_path: Union[str, Path]) -> None:
        """Load a registered model with security mitigations."""
        model = self.get_model(name)
        
        # Use secure model loading if security is available
        if SECURITY_AVAILABLE and _pytorch_security:
            with _pytorch_security.secure_torch_context():
                model.load_model(model_path)
                model.optimize_for_inference()
                model.warmup()
            self.logger.info(f"Model '{name}' loaded with security mitigations")
        else:
            model.load_model(model_path)
            model.optimize_for_inference()
            model.warmup()
            self.logger.warning(f"Model '{name}' loaded without security mitigations")
    
    def unload_model(self, name: str) -> None:
        """Unload and cleanup a model."""
        if name in self._models:
            self._models[name].cleanup()
            del self._models[name]
            self.logger.info(f"Unloaded model '{name}'")
    
    def cleanup_all(self) -> None:
        """Cleanup all models."""
        for name in list(self._models.keys()):
            self.unload_model(name)
    
    def get_downloader(self):
        """Get the model downloader instance."""
        if self._downloader is None:
            from .model_downloader import get_model_downloader
            self._downloader = get_model_downloader()
        return self._downloader
    
    def download_and_load_model(
        self, 
        source: str, 
        model_id: str, 
        name: str,
        config: Optional['InferenceConfig'] = None,
        **kwargs
    ) -> None:
        """
        Download a model from a source and load it into the framework.
        
        Args:
            source: Model source ('pytorch_hub', 'torchvision', 'huggingface', 'url')
            model_id: Model identifier
            name: Name to register the model with
            config: Inference configuration
            **kwargs: Additional arguments for download
        """
        try:
            # Download the model
            downloader = self.get_downloader()
            
            if source == "pytorch_hub":
                if "/" not in model_id:
                    raise ValueError("PyTorch Hub model_id should be in format 'repo/model'")
                # Parse PyTorch Hub model_id: repo/model or repo:version/model
                parts = model_id.rsplit("/", 1)  # Split from right to get last part as model
                if len(parts) != 2:
                    raise ValueError(f"Invalid PyTorch Hub model_id format: {model_id}. Expected: repo/model or repo:version/model")
                repo = parts[0]
                model = parts[1]
                model_path, model_info = downloader.download_pytorch_hub_model(
                    repo, model, name, **kwargs
                )
            elif source == "torchvision":
                model_path, model_info = downloader.download_torchvision_model(
                    model_id, custom_name=name, **kwargs
                )
            elif source == "huggingface":
                model_path, model_info = downloader.download_huggingface_model(
                    model_id, custom_name=name, **kwargs
                )
            elif source == "url":
                model_path, model_info = downloader.download_from_url(
                    model_id, name, **kwargs
                )
            else:
                raise ValueError(f"Unsupported source: {source}")
            
            # Create appropriate model adapter and load
            if config is None:
                from .config import get_global_config
                config = get_global_config()
            
            # Import here to avoid circular imports
            from ..adapters.model_adapters import ModelAdapterFactory
            
            adapter = ModelAdapterFactory.create_adapter(model_path, config)
            
            # Load the model
            adapter.load_model(model_path)
            adapter.optimize_for_inference()
            adapter.warmup()
            
            # Register the model
            self.register_model(name, adapter)
            
            self.logger.info(f"Successfully downloaded and loaded model '{name}' from {source}")
            
        except Exception as e:
            self.logger.error(f"Failed to download and load model '{name}': {e}")
            raise
    
    def list_available_downloads(self) -> Dict[str, Any]:
        """List available models that can be downloaded."""
        downloader = self.get_downloader()
        return downloader.list_available_models()
    
    def get_download_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a downloadable model."""
        downloader = self.get_downloader()
        info = downloader.get_model_info(model_name)
        return info.__dict__ if info else None


# Global model manager instance
_global_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager."""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager()
    return _global_model_manager

"""
Base model implementation for PyTorch Inference Framework.

This module provides base model classes and utilities for creating
and managing PyTorch models within the inference framework.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from datetime import datetime

from .exceptions import ModelLoadError, ValidationError, InferenceError
from .config import InferenceConfig

logger = logging.getLogger(__name__)


class BaseInferenceModel(ABC, nn.Module):
    """
    Abstract base class for inference models in the framework.
    
    All models used in the inference framework should inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(self, model_name: str, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base inference model.
        
        Args:
            model_name: Name of the model
            model_config: Optional model configuration
        """
        super().__init__()
        self.model_name = model_name
        self.model_config = model_config or {}
        # Default to GPU for maximum performance, fallback to CPU if unavailable
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            logger.info(f"BaseInferenceModel '{model_name}' defaulting to GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info(f"BaseInferenceModel '{model_name}' defaulting to Apple MPS")
        else:
            self.device = torch.device("cpu")
            logger.warning(f"BaseInferenceModel '{model_name}' defaulting to CPU (no GPU available)")
        self.is_loaded = False
        self.load_time = None
        self._model_info = {
            "name": model_name,
            "type": "base",
            "framework": "pytorch",
            "created_at": datetime.utcnow().isoformat()
        }
        
        logger.debug(f"BaseInferenceModel '{model_name}' initialized")
    
    @abstractmethod
    def load_model(self, device: Optional[torch.device] = None) -> None:
        """
        Load the model onto the specified device.
        
        Args:
            device: Target device for the model
        """
        pass
    
    @abstractmethod
    def forward(self, input_data: Any) -> Any:
        """
        Forward pass through the model.
        
        Args:
            input_data: Input data for inference
            
        Returns:
            Model output
        """
        pass
    
    @abstractmethod
    def preprocess(self, input_data: Any) -> torch.Tensor:
        """
        Preprocess input data for the model.
        
        Args:
            input_data: Raw input data
            
        Returns:
            Preprocessed tensor
        """
        pass
    
    @abstractmethod
    def postprocess(self, model_output: torch.Tensor) -> Any:
        """
        Postprocess model output.
        
        Args:
            model_output: Raw model output
            
        Returns:
            Processed output
        """
        pass
    
    def predict(self, input_data: Any) -> Any:
        """
        Perform end-to-end prediction.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Prediction result
        """
        try:
            if not self.is_loaded:
                raise ModelLoadError(
                    model_name=self.model_name,
                    details="Model not loaded - call load_model() first"
                )
            
            # Preprocess input
            processed_input = self.preprocess(input_data)
            
            # Ensure input is on correct device
            if isinstance(processed_input, torch.Tensor):
                processed_input = processed_input.to(self.device)
            
            # Run inference
            with torch.no_grad():
                self.eval()
                output = self.forward(processed_input)
            
            # Postprocess output
            result = self.postprocess(output)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for model {self.model_name}: {e}")
            raise InferenceError(
                details=f"Prediction failed: {e}",
                context={"model_name": self.model_name, "input_type": type(input_data).__name__},
                cause=e
            )
    
    def to_device(self, device: torch.device) -> 'BaseInferenceModel':
        """Move model to specified device."""
        try:
            self.device = device
            super().to(device)
            logger.debug(f"Model {self.model_name} moved to device: {device}")
            return self
        except Exception as e:
            logger.error(f"Failed to move model to device {device}: {e}")
            raise ModelLoadError(
                model_name=self.model_name,
                details=f"Failed to move to device {device}: {e}",
                cause=e
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = self._model_info.copy()
        info.update({
            "device": str(self.device),
            "is_loaded": self.is_loaded,
            "load_time": self.load_time,
            "parameters": sum(p.numel() for p in self.parameters()),
            "memory_usage": self.get_memory_usage()
        })
        return info
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get model memory usage."""
        try:
            param_size = sum(p.numel() * p.element_size() for p in self.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
            
            return {
                "parameters_bytes": param_size,
                "buffers_bytes": buffer_size,
                "total_bytes": param_size + buffer_size
            }
        except Exception as e:
            logger.error(f"Failed to calculate memory usage: {e}")
            return {"error": str(e)}
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data format."""
        return True  # Override in subclasses for specific validation
    
    def cleanup(self) -> None:
        """Cleanup model resources."""
        try:
            if hasattr(self, 'cpu'):
                self.cpu()
            self.is_loaded = False
            logger.debug(f"Model {self.model_name} cleaned up")
        except Exception as e:
            logger.error(f"Model cleanup failed: {e}")


class ExampleInferenceModel(BaseInferenceModel):
    """
    Example implementation of inference model for demonstration and testing.
    
    This is a simple linear model that can be used for basic testing
    of the inference framework.
    """
    
    def __init__(self, model_name: str = "example_model", 
                 input_size: int = 10, output_size: int = 1):
        """
        Initialize the example model.
        
        Args:
            model_name: Name of the model
            input_size: Input feature size
            output_size: Output size
        """
        super().__init__(model_name)
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Simple linear layer
        self.linear = nn.Linear(input_size, output_size)
        
        # Update model info
        self._model_info.update({
            "type": "example",
            "input_size": input_size,
            "output_size": output_size,
            "architecture": "linear"
        })
        
        logger.debug(f"ExampleInferenceModel created: {input_size} -> {output_size}")
    
    def load_model(self, device: Optional[torch.device] = None) -> None:
        """Load the example model."""
        try:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.to_device(device)
            
            # Initialize weights (example initialization)
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
            
            self.is_loaded = True
            self.load_time = datetime.utcnow().isoformat()
            
            logger.info(f"ExampleInferenceModel '{self.model_name}' loaded on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load example model: {e}")
            raise ModelLoadError(
                model_name=self.model_name,
                details=f"Failed to load example model: {e}",
                cause=e
            )
    
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear layer."""
        return self.linear(input_data)
    
    def preprocess(self, input_data: Any) -> torch.Tensor:
        """Preprocess input data to tensor format."""
        try:
            if isinstance(input_data, torch.Tensor):
                tensor = input_data.float()
            elif isinstance(input_data, (list, tuple)):
                tensor = torch.tensor(input_data, dtype=torch.float32)
            elif isinstance(input_data, str):
                # Convert string to numeric representation (example)
                tensor = torch.tensor([hash(input_data) % 1000 / 1000.0] * self.input_size, 
                                    dtype=torch.float32)
            else:
                # Try to convert to tensor
                tensor = torch.tensor(input_data, dtype=torch.float32)
            
            # Ensure correct shape
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)  # Add batch dimension
            
            # Ensure correct input size
            if tensor.size(-1) != self.input_size:
                if tensor.size(-1) < self.input_size:
                    # Pad with zeros
                    padding = torch.zeros(tensor.size(0), self.input_size - tensor.size(-1))
                    tensor = torch.cat([tensor, padding], dim=-1)
                else:
                    # Truncate
                    tensor = tensor[:, :self.input_size]
            
            return tensor
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise ValidationError(
                field="input_data",
                details=f"Failed to preprocess input: {e}",
                cause=e
            )
    
    def postprocess(self, model_output: torch.Tensor) -> Any:
        """Postprocess model output to appropriate format."""
        try:
            # Convert to CPU and numpy if needed
            output = model_output.cpu().numpy()
            
            # If single output, return scalar
            if output.shape == (1, 1):
                return float(output[0, 0])
            elif output.shape[0] == 1:
                return output[0].tolist()
            else:
                return output.tolist()
                
        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            return model_output.tolist()
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data for the example model."""
        try:
            processed = self.preprocess(input_data)
            return processed.size(-1) == self.input_size
        except Exception:
            return False


class TextToSpeechModel(BaseInferenceModel):
    """
    Base class for Text-to-Speech models.
    """
    
    def __init__(self, model_name: str, model_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, model_config)
        self._model_info["type"] = "text-to-speech"
        self.sample_rate = model_config.get("sample_rate", 22050) if model_config else 22050
    
    @abstractmethod
    def synthesize(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            **kwargs: Additional synthesis parameters
            
        Returns:
            Dict containing audio data and metadata
        """
        pass
    
    def preprocess(self, input_data: Any) -> Any:
        """Preprocess text input."""
        if isinstance(input_data, str):
            return input_data.strip()
        else:
            return str(input_data).strip()
    
    def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess TTS output."""
        return {
            "audio_data": model_output,
            "sample_rate": self.sample_rate,
            "model_name": self.model_name
        }


class SpeechToTextModel(BaseInferenceModel):
    """
    Base class for Speech-to-Text models.
    """
    
    def __init__(self, model_name: str, model_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, model_config)
        self._model_info["type"] = "speech-to-text"
        self.sample_rate = model_config.get("sample_rate", 16000) if model_config else 16000
    
    @abstractmethod
    def transcribe(self, audio_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Transcribe speech to text.
        
        Args:
            audio_data: Input audio data
            **kwargs: Additional transcription parameters
            
        Returns:
            Dict containing transcribed text and metadata
        """
        pass
    
    def preprocess(self, input_data: Any) -> torch.Tensor:
        """Preprocess audio input."""
        # This should be implemented based on specific audio preprocessing needs
        if isinstance(input_data, torch.Tensor):
            return input_data
        else:
            # Convert to tensor (placeholder implementation)
            return torch.tensor(input_data, dtype=torch.float32)
    
    def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess STT output."""
        return {
            "text": str(model_output),
            "model_name": self.model_name
        }


# Factory functions

def create_example_model(config: InferenceConfig) -> ExampleInferenceModel:
    """
    Create an example model for testing and demonstration.
    
    Args:
        config: Inference configuration
        
    Returns:
        Configured example model
    """
    try:
        model = ExampleInferenceModel(
            model_name="example_model",
            input_size=config.model_config.get("input_size", 10),
            output_size=config.model_config.get("output_size", 1)
        )
        
        # Load the model
        device = torch.device(config.device_config.device_type)
        model.load_model(device)
        
        logger.info("Example model created and loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to create example model: {e}")
        raise ModelLoadError(
            model_name="example_model",
            details=f"Failed to create example model: {e}",
            cause=e
        )


def create_model_from_config(model_name: str, model_config: Dict[str, Any], 
                           inference_config: InferenceConfig) -> BaseInferenceModel:
    """
    Create a model from configuration.
    
    Args:
        model_name: Name of the model
        model_config: Model-specific configuration
        inference_config: Global inference configuration
        
    Returns:
        Configured model instance
    """
    try:
        model_type = model_config.get("type", "example")
        
        if model_type == "example":
            model = ExampleInferenceModel(
                model_name=model_name,
                input_size=model_config.get("input_size", 10),
                output_size=model_config.get("output_size", 1)
            )
        elif model_type == "text-to-speech":
            # Placeholder for TTS model creation
            model = ExampleInferenceModel(model_name, 100, 1000)  # TTS typically has larger outputs
            model._model_info["type"] = "text-to-speech"
        elif model_type == "speech-to-text":
            # Placeholder for STT model creation
            model = ExampleInferenceModel(model_name, 1000, 100)  # STT typically has larger inputs
            model._model_info["type"] = "speech-to-text"
        else:
            logger.warning(f"Unknown model type '{model_type}', using example model")
            model = ExampleInferenceModel(model_name)
        
        # Load the model
        device = torch.device(inference_config.device_config.device_type)
        model.load_model(device)
        
        logger.info(f"Model '{model_name}' of type '{model_type}' created successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to create model '{model_name}': {e}")
        raise ModelLoadError(
            model_name=model_name,
            details=f"Failed to create model: {e}",
            cause=e
        )


# Model registry utilities

class ModelRegistry:
    """Registry for managing available model types and configurations."""
    
    def __init__(self):
        self._model_types = {
            "example": ExampleInferenceModel,
            "text-to-speech": TextToSpeechModel,
            "speech-to-text": SpeechToTextModel
        }
        self._model_configs = {}
    
    def register_model_type(self, type_name: str, model_class: type):
        """Register a new model type."""
        self._model_types[type_name] = model_class
        logger.debug(f"Registered model type: {type_name}")
    
    def get_model_class(self, type_name: str) -> Optional[type]:
        """Get model class by type name."""
        return self._model_types.get(type_name)
    
    def list_model_types(self) -> List[str]:
        """List available model types."""
        return list(self._model_types.keys())
    
    def register_model_config(self, model_name: str, config: Dict[str, Any]):
        """Register a model configuration."""
        self._model_configs[model_name] = config
        logger.debug(f"Registered model config: {model_name}")
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model configuration by name."""
        return self._model_configs.get(model_name)


# Global model registry instance
_model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Get the global model registry."""
    return _model_registry

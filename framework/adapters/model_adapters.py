"""
Model adapters for different deep learning frameworks.

This module provides adapters to load and use models from different frameworks
(PyTorch, ONNX, TensorRT, etc.) with a unified interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
import torch
import torch.nn as nn
import numpy as np

from ..core.base_model import BaseModel, ModelMetadata, ModelLoadError
from ..core.config import InferenceConfig
from ..processors.preprocessor import PreprocessingResult


logger = logging.getLogger(__name__)


class PyTorchModelAdapter(BaseModel):
    """Adapter for PyTorch models."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.model_path: Optional[Path] = None
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load PyTorch model."""
        try:
            model_path = Path(model_path)
            self.model_path = model_path
            
            self.logger.info(f"Loading PyTorch model from {model_path}")
            
            # Load model
            if model_path.suffix == '.pt' or model_path.suffix == '.pth':
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different save formats
                if isinstance(checkpoint, nn.Module):
                    self.model = checkpoint
                elif isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        self.model = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        # Need model architecture for state_dict
                        raise ModelLoadError("State dict found but no model architecture provided")
                    else:
                        # Assume the dict is the state dict
                        raise ModelLoadError("Model architecture required for state dict")
                else:
                    raise ModelLoadError(f"Unsupported checkpoint format: {type(checkpoint)}")
            
            elif model_path.suffix == '.torchscript':
                self.model = torch.jit.load(model_path, map_location=self.device)
            
            else:
                raise ModelLoadError(f"Unsupported file extension: {model_path.suffix}")
            
            # Set metadata
            self.metadata = ModelMetadata(
                name=model_path.stem,
                version="1.0",
                model_type="pytorch",
                input_shape=self._get_input_shape(),
                output_shape=self._get_output_shape(),
                description=f"PyTorch model loaded from {model_path}"
            )
            
            self._is_loaded = True
            self.logger.info(f"Successfully loaded PyTorch model: {self.metadata.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load PyTorch model: {e}")
            raise ModelLoadError(f"Failed to load PyTorch model: {e}") from e
    
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """Preprocess inputs for PyTorch model."""
        # Use the preprocessing pipeline
        from ..processors.preprocessor import create_default_preprocessing_pipeline
        
        if not hasattr(self, '_preprocessing_pipeline'):
            self._preprocessing_pipeline = create_default_preprocessing_pipeline(self.config)
        
        result = self._preprocessing_pipeline.preprocess(inputs)
        return result.data
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run forward pass through PyTorch model."""
        if not self._is_loaded:
            raise ModelLoadError("Model not loaded")
        
        model = self.get_model_for_inference()
        
        # Ensure inputs are on the correct device
        inputs = inputs.to(self.device)
        
        # Handle different model types
        if hasattr(model, 'predict') and callable(model.predict):
            # YOLO-style model
            outputs = model.predict(inputs)
        elif isinstance(model, torch.jit.ScriptModule):
            # TorchScript model
            outputs = model(inputs)
        else:
            # Standard PyTorch model
            outputs = model(inputs)
        
        return outputs
    
    def postprocess(self, outputs: torch.Tensor) -> Any:
        """Postprocess PyTorch model outputs."""
        # Use the postprocessing pipeline
        from ..processors.postprocessor import create_default_postprocessing_pipeline
        
        if not hasattr(self, '_postprocessing_pipeline'):
            self._postprocessing_pipeline = create_default_postprocessing_pipeline(self.config)
        
        result = self._postprocessing_pipeline.auto_postprocess(outputs)
        return result
    
    def _get_input_shape(self) -> Tuple[int, ...]:
        """Get model input shape."""
        try:
            if hasattr(self.model, 'input_shape'):
                return tuple(self.model.input_shape)
            elif hasattr(self.config, 'preprocessing') and hasattr(self.config.preprocessing, 'input_size'):
                h, w = self.config.preprocessing.input_size
                return (3, h, w)
            else:
                return (3, 224, 224)  # Default
        except Exception:
            return (3, 224, 224)  # Default
    
    def _get_output_shape(self) -> Tuple[int, ...]:
        """Get model output shape."""
        try:
            if hasattr(self.model, 'output_shape'):
                return tuple(self.model.output_shape)
            else:
                return (1000,)  # Default for classification
        except Exception:
            return (1000,)  # Default


class ONNXModelAdapter(BaseModel):
    """Adapter for ONNX models."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.model_path: Optional[Path] = None
        self.session = None
        self.input_names = []
        self.output_names = []
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ModelLoadError("onnxruntime not installed. Install with: pip install onnxruntime")
        
        try:
            model_path = Path(model_path)
            self.model_path = model_path
            
            self.logger.info(f"Loading ONNX model from {model_path}")
            
            # Configure ONNX Runtime providers
            providers = ['CPUExecutionProvider']
            if self.device.type == 'cuda':
                providers.insert(0, 'CUDAExecutionProvider')
            
            # Create inference session
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            
            # Get input and output names
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Get input and output shapes
            input_shapes = [input.shape for input in self.session.get_inputs()]
            output_shapes = [output.shape for output in self.session.get_outputs()]
            
            # Set metadata
            self.metadata = ModelMetadata(
                name=model_path.stem,
                version="1.0",
                model_type="onnx",
                input_shape=tuple(input_shapes[0]) if input_shapes else (1, 3, 224, 224),
                output_shape=tuple(output_shapes[0]) if output_shapes else (1, 1000),
                description=f"ONNX model loaded from {model_path}"
            )
            
            self._is_loaded = True
            self.logger.info(f"Successfully loaded ONNX model: {self.metadata.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            raise ModelLoadError(f"Failed to load ONNX model: {e}") from e
    
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """Preprocess inputs for ONNX model."""
        # Use the preprocessing pipeline
        from ..processors.preprocessor import create_default_preprocessing_pipeline
        
        if not hasattr(self, '_preprocessing_pipeline'):
            self._preprocessing_pipeline = create_default_preprocessing_pipeline(self.config)
        
        result = self._preprocessing_pipeline.preprocess(inputs)
        return result.data
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run forward pass through ONNX model."""
        if not self._is_loaded or not self.session:
            raise ModelLoadError("Model not loaded")
        
        # Convert to numpy for ONNX Runtime
        inputs_np = inputs.detach().cpu().numpy()
        
        # Prepare input dict
        input_dict = {self.input_names[0]: inputs_np}
        
        # Run inference
        outputs = self.session.run(self.output_names, input_dict)
        
        # Convert back to torch tensor
        output_tensor = torch.from_numpy(outputs[0]).to(self.device)
        
        return output_tensor
    
    def postprocess(self, outputs: torch.Tensor) -> Any:
        """Postprocess ONNX model outputs."""
        # Use the postprocessing pipeline
        from ..processors.postprocessor import create_default_postprocessing_pipeline
        
        if not hasattr(self, '_postprocessing_pipeline'):
            self._postprocessing_pipeline = create_default_postprocessing_pipeline(self.config)
        
        result = self._postprocessing_pipeline.auto_postprocess(outputs)
        return result


class TensorRTModelAdapter(BaseModel):
    """Adapter for TensorRT models."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.model_path: Optional[Path] = None
        self.engine = None
        self.context = None
        self.input_names = []
        self.output_names = []
        self.bindings = []
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load TensorRT model."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ModelLoadError("TensorRT or PyCUDA not installed")
        
        if self.device.type != 'cuda':
            raise ModelLoadError("TensorRT requires CUDA device")
        
        try:
            model_path = Path(model_path)
            self.model_path = model_path
            
            self.logger.info(f"Loading TensorRT model from {model_path}")
            
            # Load TensorRT engine
            with open(model_path, 'rb') as f:
                engine_data = f.read()
            
            # Create runtime and deserialize engine
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()
            
            # Get input and output information
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                
                if self.engine.binding_is_input(i):
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)
            
            # Set metadata
            self.metadata = ModelMetadata(
                name=model_path.stem,
                version="1.0",
                model_type="tensorrt",
                input_shape=tuple(self.engine.get_binding_shape(0)),
                output_shape=tuple(self.engine.get_binding_shape(1)) if self.engine.num_bindings > 1 else (1000,),
                description=f"TensorRT model loaded from {model_path}"
            )
            
            self._is_loaded = True
            self.logger.info(f"Successfully loaded TensorRT model: {self.metadata.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load TensorRT model: {e}")
            raise ModelLoadError(f"Failed to load TensorRT model: {e}") from e
    
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """Preprocess inputs for TensorRT model."""
        # Use the preprocessing pipeline
        from ..processors.preprocessor import create_default_preprocessing_pipeline
        
        if not hasattr(self, '_preprocessing_pipeline'):
            self._preprocessing_pipeline = create_default_preprocessing_pipeline(self.config)
        
        result = self._preprocessing_pipeline.preprocess(inputs)
        return result.data
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run forward pass through TensorRT model."""
        if not self._is_loaded or not self.engine or not self.context:
            raise ModelLoadError("Model not loaded")
        
        import pycuda.driver as cuda
        
        # Ensure inputs are contiguous and on GPU
        inputs = inputs.contiguous()
        
        # Allocate GPU memory for outputs
        output_shape = self.engine.get_binding_shape(1)
        outputs = torch.empty(output_shape, dtype=inputs.dtype, device=self.device)
        
        # Set up bindings
        bindings = [inputs.data_ptr(), outputs.data_ptr()]
        
        # Run inference
        self.context.execute_v2(bindings)
        
        return outputs
    
    def postprocess(self, outputs: torch.Tensor) -> Any:
        """Postprocess TensorRT model outputs."""
        # Use the postprocessing pipeline
        from ..processors.postprocessor import create_default_postprocessing_pipeline
        
        if not hasattr(self, '_postprocessing_pipeline'):
            self._postprocessing_pipeline = create_default_postprocessing_pipeline(self.config)
        
        result = self._postprocessing_pipeline.auto_postprocess(outputs)
        return result


class HuggingFaceModelAdapter(BaseModel):
    """Adapter for Hugging Face transformers models."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.model_name: Optional[str] = None
        self.tokenizer = None
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load Hugging Face model."""
        try:
            from transformers import AutoModel, AutoTokenizer, AutoConfig
        except ImportError:
            raise ModelLoadError("transformers not installed. Install with: pip install transformers")
        
        try:
            # Handle both local path and model name
            if isinstance(model_path, Path) and model_path.exists():
                model_name = str(model_path)
            else:
                model_name = str(model_path)
            
            self.model_name = model_name
            
            self.logger.info(f"Loading Hugging Face model: {model_name}")
            
            # Load config to get model info
            config = AutoConfig.from_pretrained(model_name)
            
            # Load model and tokenizer
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Move to device
            self.model.to(self.device)
            
            # Set metadata
            self.metadata = ModelMetadata(
                name=model_name.split('/')[-1] if '/' in model_name else model_name,
                version="1.0",
                model_type="huggingface",
                input_shape=(512,),  # Default sequence length
                output_shape=(config.hidden_size,) if hasattr(config, 'hidden_size') else (768,),
                description=f"Hugging Face model: {model_name}"
            )
            
            self._is_loaded = True
            self.logger.info(f"Successfully loaded Hugging Face model: {self.metadata.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Hugging Face model: {e}")
            raise ModelLoadError(f"Failed to load Hugging Face model: {e}") from e
    
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """Preprocess text inputs for Hugging Face model."""
        if not self.tokenizer:
            raise ModelLoadError("Tokenizer not loaded")
        
        if isinstance(inputs, str):
            text = inputs
        elif isinstance(inputs, list):
            text = inputs  # Assume list of strings
        else:
            text = str(inputs)
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        for key in encoded:
            encoded[key] = encoded[key].to(self.device)
        
        return encoded
    
    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Run forward pass through Hugging Face model."""
        if not self._is_loaded:
            raise ModelLoadError("Model not loaded")
        
        model = self.get_model_for_inference()
        
        if isinstance(inputs, dict):
            # Tokenized inputs
            outputs = model(**inputs)
        else:
            # Raw tensor
            outputs = model(inputs)
        
        # Extract appropriate outputs
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        elif hasattr(outputs, 'pooler_output'):
            return outputs.pooler_output
        else:
            return outputs[0] if isinstance(outputs, tuple) else outputs
    
    def postprocess(self, outputs: torch.Tensor) -> Any:
        """Postprocess Hugging Face model outputs."""
        # For transformers, typically return embeddings or logits directly
        return {
            "embeddings": outputs.detach().cpu().numpy(),
            "shape": list(outputs.shape)
        }


class ModelAdapterFactory:
    """Factory for creating model adapters."""
    
    @staticmethod
    def create_adapter(model_path: Union[str, Path], config: InferenceConfig) -> BaseModel:
        """Create appropriate model adapter based on file extension or model type."""
        model_path = Path(model_path) if isinstance(model_path, str) else model_path
        
        # Determine adapter type based on file extension or model name
        if model_path.suffix in ['.pt', '.pth', '.torchscript']:
            return PyTorchModelAdapter(config)
        elif model_path.suffix == '.onnx':
            return ONNXModelAdapter(config)
        elif model_path.suffix in ['.trt', '.engine']:
            return TensorRTModelAdapter(config)
        elif '/' in str(model_path) and not model_path.exists():
            # Likely a Hugging Face model name
            return HuggingFaceModelAdapter(config)
        else:
            # Default to PyTorch
            return PyTorchModelAdapter(config)
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """Get list of supported model formats."""
        return [
            '.pt', '.pth', '.torchscript',  # PyTorch
            '.onnx',  # ONNX
            '.trt', '.engine',  # TensorRT
            'huggingface'  # Hugging Face
        ]


def load_model(model_path: Union[str, Path], config: Optional[InferenceConfig] = None) -> BaseModel:
    """
    Convenient function to load any supported model.
    
    Args:
        model_path: Path to model file or model identifier
        config: Inference configuration
        
    Returns:
        Loaded model adapter
    """
    if config is None:
        from ..core.config import get_global_config
        config = get_global_config()
    
    # Create adapter
    adapter = ModelAdapterFactory.create_adapter(model_path, config)
    
    # Load model
    adapter.load_model(model_path)
    
    # Optimize for inference
    adapter.optimize_for_inference()
    
    # Warmup
    adapter.warmup()
    
    return adapter

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

# Import security mitigations
try:
    from ..core.security import PyTorchSecurityMitigation, ECDSASecurityMitigation
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Initialize security for model adapters (lazy initialization to prevent hanging)
_pytorch_security = None
_ecdsa_security = None

def _get_pytorch_security():
    """Lazy initialization of PyTorch security."""
    global _pytorch_security
    if SECURITY_AVAILABLE and _pytorch_security is None:
        try:
            _pytorch_security = PyTorchSecurityMitigation()
            logger.info("PyTorch security mitigation initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize PyTorch security: {e}")
            _pytorch_security = False  # Mark as failed to avoid retrying
    return _pytorch_security if _pytorch_security is not False else None

def _get_ecdsa_security():
    """Lazy initialization of ECDSA security."""
    global _ecdsa_security
    if SECURITY_AVAILABLE and _ecdsa_security is None:
        try:
            _ecdsa_security = ECDSASecurityMitigation()
            logger.info("ECDSA security mitigation initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize ECDSA security: {e}")
            _ecdsa_security = False  # Mark as failed to avoid retrying
    return _ecdsa_security if _ecdsa_security is not False else None


class PyTorchModelAdapter(BaseModel):
    """Adapter for PyTorch models."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.model_path: Optional[Path] = None
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load PyTorch model with security context."""
        try:
            model_path = Path(model_path)
            self.model_path = model_path
            
            self.logger.info(f"Loading PyTorch model from {model_path}")
            self.logger.info(f"Target device: {self.device}")
            
            # Use security context for model loading
            pytorch_security = _get_pytorch_security()
            if pytorch_security:
                with pytorch_security.secure_context():
                    # Load model
                    if model_path.suffix == '.pt' or model_path.suffix == '.pth':
                        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                        
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
            else:
                # Fallback without security context
                if model_path.suffix == '.pt' or model_path.suffix == '.pth':
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    
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
            
            # Ensure model is on the correct device
            self.model = self.model.to(self.device)
            self.logger.info(f"Model moved to device: {self.device}")
            
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
        # Handle tensors directly to avoid complex preprocessing pipeline issues
        if isinstance(inputs, torch.Tensor):
            # For tensors, use them directly if they already have proper shape
            tensor = inputs.clone()
            
            # Move to device
            tensor = tensor.to(self.device)
            
            # Add batch dimension only if needed
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            # For 2D tensors that already have batch dimension, use as-is
            
            return tensor
        
        # For non-tensor inputs, use the preprocessing pipeline
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

        # Check for quantized model and handle device compatibility
        is_quantized_model = self._is_quantized_model(model)
        if is_quantized_model and self.device.type == 'cuda':
            self.logger.warning("Quantized model detected on CUDA device. Moving model to CPU for compatibility. Preserving input dtype for embeddings.")
            # Move model to CPU for quantized operations
            model = model.cpu()
            # Convert all parameters and buffers to float32 to avoid dtype mismatch
            for param in model.parameters():
                if param.dtype == torch.float16 or param.dtype == torch.half:
                    param.data = param.data.float()
            for buffer_name, buffer in model.named_buffers():
                if buffer.dtype == torch.float16 or buffer.dtype == torch.half:
                    setattr(model, buffer_name, buffer.float())
            # Only cast to float if input is floating type; preserve integer types for embeddings
            if inputs.dtype in (torch.long, torch.int64, torch.int32, torch.int16, torch.int8):
                inputs = inputs.cpu()
            else:
                inputs = inputs.cpu().float()
            model_dtype = torch.float32
        else:
            # Handle dtype conversion carefully to avoid mismatches
            model_dtype = self._detect_model_dtype(model)
            inputs = self._match_input_dtype_to_model(inputs, model, model_dtype, is_quantized_model)

        # Handle different model types
        try:
            if hasattr(model, 'predict') and callable(model.predict):
                # YOLO-style model
                outputs = model.predict(inputs)
            elif isinstance(model, torch.jit.ScriptModule):
                # TorchScript model
                outputs = model(inputs)
            else:
                # Standard PyTorch model
                outputs = model(inputs)
            
            # Move outputs back to original device if we moved to CPU for quantization
            if is_quantized_model and self.device.type == 'cuda':
                outputs = self._move_outputs_to_device(outputs, self.device)
                
            return outputs
            
        except RuntimeError as e:
            if 'quantized' in str(e).lower() and 'cuda' in str(e).lower():
                self.logger.warning(f"Quantized operation failed on CUDA, retrying on CPU: {e}")
                # Force CPU execution for quantized operations
                model = model.cpu()
                inputs = inputs.cpu()
                
                if hasattr(model, 'predict') and callable(model.predict):
                    outputs = model.predict(inputs)
                elif isinstance(model, torch.jit.ScriptModule):
                    outputs = model(inputs)
                else:
                    outputs = model(inputs)
                
                # Move outputs back to original device
                outputs = self._move_outputs_to_device(outputs, self.device)
                return outputs
            else:
                raise
    
    def _detect_model_dtype(self, model):
        """Detect the primary dtype used by the model."""
        dtypes = set()
        param_count = 0
        
        # Sample parameters to determine the most common dtype
        for param in model.parameters():
            if param.dtype in (torch.float16, torch.float32, torch.float64):
                dtypes.add(param.dtype)
                param_count += 1
                # Sample first 10 parameters to avoid performance issues
                if param_count >= 10:
                    break
        
        # If any parameters are float16, assume the model uses mixed precision
        if torch.float16 in dtypes:
            return torch.float16
        elif torch.float32 in dtypes:
            return torch.float32
        else:
            # Default to float32 if we can't determine
            return torch.float32
            
    def _is_quantized_model(self, model) -> bool:
        """Check if the model contains quantized operations."""
        try:
            # Check for quantized modules or layers
            for module in model.modules():
                module_type = type(module).__name__
                if any(quant_indicator in module_type.lower() for quant_indicator in [
                    'quantized', 'quant', 'int8', 'qlinear', 'qdepthwise', 'qconv'
                ]):
                    return True
                
                # Check for quantized parameters
                for param_name, param in module.named_parameters():
                    if hasattr(param, 'qscheme') or 'quantized' in param_name.lower():
                        return True
                        
            # Check for quantized state in the model's state dict
            if hasattr(model, 'state_dict'):
                for key in model.state_dict().keys():
                    if any(quant_indicator in key.lower() for quant_indicator in [
                        'quantized', '_scale', '_zero_point', 'quant'
                    ]):
                        return True
                        
            return False
        except Exception as e:
            self.logger.debug(f"Could not determine if model is quantized: {e}")
            return False
    
    def _match_input_dtype_to_model(self, inputs: torch.Tensor, model, model_dtype: torch.dtype, is_quantized_model: bool) -> torch.Tensor:
        """Match input dtype to model dtype to prevent dtype mismatches."""
        try:
            # For embedding layers, preserve integer dtypes
            if inputs.dtype in (torch.long, torch.int64, torch.int32, torch.int16, torch.int8):
                # Don't convert integer tensors that are likely for embeddings
                return inputs
            
            # For floating point tensors, match to model dtype
            if inputs.dtype in (torch.float32, torch.float16, torch.float64, torch.bfloat16):
                # If model is quantized, use float32 to avoid issues
                if is_quantized_model:
                    return inputs.float()
                
                # Match model dtype
                if model_dtype == torch.float16:
                    return inputs.half()
                elif model_dtype == torch.float32:
                    return inputs.float()
                else:
                    return inputs.float()  # Default to float32
            
            # Return unchanged for other dtypes
            return inputs
            
        except Exception as e:
            self.logger.debug(f"Failed to match input dtype: {e}, using original")
            return inputs
    
    def _move_outputs_to_device(self, outputs, device: torch.device):
        """Move outputs to device, handling different output types."""
        try:
            if isinstance(outputs, torch.Tensor):
                return outputs.to(device)
            elif hasattr(outputs, 'to') and callable(outputs.to):
                # Some objects like ModelOutput have a 'to' method
                return outputs.to(device)
            elif hasattr(outputs, 'last_hidden_state') and isinstance(outputs.last_hidden_state, torch.Tensor):
                # Handle transformer outputs by moving individual tensor fields
                for attr_name in dir(outputs):
                    if not attr_name.startswith('_'):
                        attr_value = getattr(outputs, attr_name)
                        if isinstance(attr_value, torch.Tensor):
                            setattr(outputs, attr_name, attr_value.to(device))
                return outputs
            elif isinstance(outputs, (list, tuple)):
                # Handle list/tuple of tensors
                moved_outputs = []
                for output in outputs:
                    moved_outputs.append(self._move_outputs_to_device(output, device))
                return type(outputs)(moved_outputs)
            elif isinstance(outputs, dict):
                # Handle dictionary of tensors
                moved_dict = {}
                for key, value in outputs.items():
                    moved_dict[key] = self._move_outputs_to_device(value, device)
                return moved_dict
            else:
                # Can't move to device, return as is
                return outputs
        except Exception as e:
            self.logger.debug(f"Failed to move outputs to device: {e}, returning original")
            return outputs
    
    def postprocess(self, outputs: torch.Tensor) -> Any:
        """Postprocess PyTorch model outputs."""
        # Use the postprocessing pipeline
        from ..processors.postprocessor import create_default_postprocessing_pipeline
        
        if not hasattr(self, '_postprocessing_pipeline'):
            self._postprocessing_pipeline = create_default_postprocessing_pipeline(self.config)
        
        result = self._postprocessing_pipeline.auto_postprocess(outputs)
        
        # Convert to dict for backward compatibility
        if hasattr(result, 'to_dict'):
            return result.to_dict()
        
        return result
    
    def predict_batch(self, inputs_list: List[Any]) -> List[Any]:
        """
        Batch prediction optimized for PyTorch models.
        
        Args:
            inputs_list: List of input data
            
        Returns:
            List of predictions
        """
        if not inputs_list:
            return []
        
        # Try to batch process if possible
        try:
            # Preprocess all inputs
            preprocessed_inputs = [self.preprocess(inp) for inp in inputs_list]
            
            # Stack into batch tensor if possible
            if all(isinstance(inp, torch.Tensor) and inp.shape == preprocessed_inputs[0].shape for inp in preprocessed_inputs):
                # Check if inputs already have batch dimension of 1 - if so, remove it before stacking
                if len(preprocessed_inputs[0].shape) == 4 and preprocessed_inputs[0].shape[0] == 1:
                    # Remove the batch dimension from each input before stacking
                    squeezed_inputs = [inp.squeeze(0) for inp in preprocessed_inputs]
                    batch_tensor = torch.stack(squeezed_inputs, dim=0)
                else:
                    batch_tensor = torch.stack(preprocessed_inputs, dim=0)
                
                # Forward pass on batch
                with torch.no_grad():
                    batch_outputs = self.forward(batch_tensor)
                
                # Split batch results and postprocess
                if len(batch_outputs.shape) > 0:
                    outputs_list = torch.split(batch_outputs, 1, dim=0)
                    results = []
                    for output in outputs_list:
                        output = output.squeeze(0)  # Remove batch dimension
                        result = self.postprocess(output)
                        results.append(result)
                    return results
                    
            # Fallback to individual processing
            return [self.predict(inp) for inp in inputs_list]
            
        except Exception as e:
            self.logger.warning(f"Batch processing failed: {e}, falling back to individual processing")
            # Fallback to individual processing
            return [self.predict(inp) for inp in inputs_list]
    
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
            self.logger.info(f"Target device: {self.device}")
            
            # Configure ONNX Runtime providers based on detected device
            providers = ['CPUExecutionProvider']
            if self.device.type == 'cuda':
                providers.insert(0, 'CUDAExecutionProvider')
                self.logger.info("Using CUDA execution provider for ONNX")
            elif self.device.type == 'mps':
                # Note: ONNX Runtime doesn't support MPS directly, fallback to CPU
                self.logger.warning("MPS device detected but ONNX Runtime doesn't support MPS, using CPU")
            
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
            self.logger.info(f"Target device: {self.device}")
            
            # Load config to get model info
            config = AutoConfig.from_pretrained(model_name)
            
            # Load model and tokenizer
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Move to device
            self.model = self.model.to(self.device)
            self.logger.info(f"Hugging Face model moved to device: {self.device}")
            
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
        # Handle string identifiers vs file paths
        if isinstance(model_path, str):
            path_obj = Path(model_path)
            if path_obj.exists():
                # It's a file path
                model_path = path_obj
            else:
                # It's likely a model identifier
                if ('/' in model_path) or ('-' in model_path and '.' not in model_path):
                    # Likely a Hugging Face model name
                    return HuggingFaceModelAdapter(config)
                else:
                    # Default to PyTorch for unknown strings
                    return PyTorchModelAdapter(config)
        
        # Handle Path objects
        if isinstance(model_path, Path):
            # Determine adapter type based on file extension
            if model_path.suffix in ['.pt', '.pth', '.torchscript']:
                return PyTorchModelAdapter(config)
            elif model_path.suffix == '.onnx':
                return ONNXModelAdapter(config)
            elif model_path.suffix in ['.trt', '.engine']:
                return TensorRTModelAdapter(config)
            else:
                # Default to PyTorch
                return PyTorchModelAdapter(config)
        
        # Fallback to PyTorch
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
        
    Raises:
        ValueError: If model format is not supported
    """
    if config is None:
        from ..core.config import get_global_config
        config = get_global_config()
    
    # Handle string model identifiers (like HuggingFace model names)
    if isinstance(model_path, str):
        # Check if it's a file path
        path_obj = Path(model_path)
        if path_obj.exists():
            # It's an existing file
            model_path = path_obj
            # Validate model format
            if model_path.suffix not in ['.pt', '.pth', '.torchscript', '.onnx', '.trt', '.engine']:
                raise ValueError(f"Unsupported model format: {model_path.suffix}")
        else:
            # It's likely a model identifier (HuggingFace, etc.)
            # Don't convert to Path object for factory method
            pass
    else:
        # It's already a Path object
        model_path = model_path
        # Validate model format if file exists
        if model_path.exists() and model_path.suffix not in ['.pt', '.pth', '.torchscript', '.onnx', '.trt', '.engine']:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
    
    # Create adapter
    try:
        adapter = ModelAdapterFactory.create_adapter(model_path, config)
        
        # Load model
        adapter.load_model(model_path)
        
        # Optimize for inference
        adapter.optimize_for_inference()
        
        # Warmup
        adapter.warmup()
        
        return adapter
    except ModelLoadError as e:
        # Convert ModelLoadError to ValueError for unsupported formats
        if "Unsupported file extension" in str(e):
            raise ValueError(f"Unsupported model format: {model_path}") from e
        else:
            raise

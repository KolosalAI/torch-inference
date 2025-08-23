"""
Quantization optimization module for PyTorch models.

This module provides functionality to quantize PyTorch models for
improved performance and reduced memory usage.
"""

import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path

import torch
import torch.nn as nn

# Use modern torch.ao.quantization APIs
try:
    import torch.ao.quantization as ao_quant
    from torch.ao.quantization import QConfigMapping, get_default_qconfig_mapping
    # Also import legacy quantization for backward compatibility
    import torch.quantization as quant
    AO_QUANTIZATION_AVAILABLE = True
except ImportError:
    # Fallback to legacy APIs if ao.quantization is not available
    try:
        import torch.quantization as quant
        from torch.quantization import get_default_qconfig_mapping
        ao_quant = quant
        # Create a dummy QConfigMapping if not available
        try:
            from torch.quantization import QConfigMapping
        except ImportError:
            QConfigMapping = None
        AO_QUANTIZATION_AVAILABLE = True
    except ImportError:
        AO_QUANTIZATION_AVAILABLE = False

from ..core.config import InferenceConfig


logger = logging.getLogger(__name__)


class QuantizationOptimizer:
    """
    Quantization optimization manager for PyTorch models.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize quantization optimizer.
        
        Args:
            config: Inference configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.QuantizationOptimizer")
        
        # Check quantization backend availability
        self.backend = self._select_backend()
        self.logger.info(f"Quantization optimizer initialized with backend: {self.backend}")
    
    def _select_backend(self) -> str:
        """Select the best available quantization backend."""
        try:
            current_engine = torch.backends.quantized.engine
            # Check if current engine is already supported
            if current_engine in ['fbgemm', 'qnnpack']:
                return current_engine
        except:
            pass
        
        # Try to set a supported backend
        try:
            # On Windows/CPU, prefer fbgemm which is more widely supported
            torch.backends.quantized.engine = 'fbgemm'
            return 'fbgemm'
        except RuntimeError:
            try:
                # Fallback to qnnpack if available
                torch.backends.quantized.engine = 'qnnpack'
                return 'qnnpack'
            except RuntimeError:
                # If no backend is available, log warning and use current
                self.logger.warning("No quantization backend available, using default")
                return 'none'
    
    def optimize(self, 
                model: nn.Module, 
                quantization_type: str = "dynamic",
                **kwargs) -> nn.Module:
        """
        Main optimization method that dispatches to specific quantization methods.
        
        Args:
            model: Model to quantize
            quantization_type: Type of quantization ("dynamic", "static", "qat", "fx")
            **kwargs: Additional arguments for the specific quantization method
            
        Returns:
            Quantized model
        """
        # Skip quantization if no backend is available
        if self.backend == 'none':
            self.logger.warning("Quantization backend not available, returning original model")
            return model
            
        if quantization_type == "dynamic":
            return self.quantize_dynamic(model, **kwargs)
        elif quantization_type == "static":
            # For static quantization, we need calibration_loader
            calibration_loader = kwargs.pop('calibration_loader', None)
            if calibration_loader is None:
                raise ValueError("calibration_loader required for static quantization")
            return self.quantize_static(model, calibration_loader, **kwargs)
        elif quantization_type == "qat":
            # For QAT, we need training parameters
            train_loader = kwargs.pop('train_loader', None)
            optimizer = kwargs.pop('optimizer', None) 
            criterion = kwargs.pop('criterion', None)
            if not all([train_loader, optimizer, criterion]):
                raise ValueError("train_loader, optimizer, and criterion required for QAT")
            return self.quantize_qat(model, train_loader, optimizer, criterion, **kwargs)
        elif quantization_type == "fx":
            # For FX quantization, we need example inputs
            example_inputs = kwargs.pop('example_inputs', None)
            if example_inputs is None:
                raise ValueError("example_inputs required for FX quantization")
            return self.quantize_fx(model, example_inputs, **kwargs)
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
    
    def quantize_dynamic(self, 
                        model: nn.Module,
                        dtype: torch.dtype = torch.qint8,
                        qconfig_spec: Optional[Dict] = None,
                        **kwargs) -> nn.Module:
        """
        Apply dynamic quantization to model.
        
        Dynamic quantization quantizes weights statically and activations dynamically.
        Good for models where activation distributions vary significantly.
        
        Args:
            model: PyTorch model to quantize
            dtype: Quantization data type
            qconfig_spec: Quantization configuration specification
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            Dynamically quantized model
        """
        try:
            self.logger.info("Applying dynamic quantization")
            
            # Default layers to quantize
            if qconfig_spec is None:
                qconfig_spec = {
                    nn.Linear,
                    nn.Conv2d,
                    nn.Conv1d,
                    nn.ConvTranspose2d
                }
            
            # Set model to evaluation mode
            model.eval()
            
            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                qconfig_spec,
                dtype=dtype
            )
            
            self.logger.info("Dynamic quantization completed successfully")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Dynamic quantization failed: {e}")
            self.logger.warning("Returning original model")
            return model
    
    def quantize_static(self, 
                       model: nn.Module,
                       calibration_loader: torch.utils.data.DataLoader,
                       qconfig_mapping = None) -> nn.Module:
        """
        Apply static quantization to model.
        
        Static quantization quantizes both weights and activations statically
        based on calibration data.
        
        Args:
            model: PyTorch model to quantize
            calibration_loader: DataLoader for calibration
            qconfig_mapping: Quantization configuration mapping
            
        Returns:
            Statically quantized model
        """
        try:
            self.logger.info("Applying static quantization")
            
            # Set model to evaluation mode
            model.eval()
            
            # Default quantization configuration
            if qconfig_mapping is None:
                qconfig_mapping = get_default_qconfig_mapping(self.backend)
            
            # Prepare model for quantization
            prepared_model = torch.quantization.prepare(model, qconfig_mapping)
            
            # Calibrate with representative data
            self.logger.info("Calibrating model with representative data")
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(calibration_loader):
                    prepared_model(data)
                    if batch_idx >= 100:  # Limit calibration samples
                        break
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(prepared_model)
            
            self.logger.info("Static quantization completed successfully")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Static quantization failed: {e}")
            self.logger.warning("Returning original model")
            return model
    
    def quantize_qat(self, 
                    model: nn.Module,
                    train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    epochs: int = 3,
                    qconfig_mapping = None) -> nn.Module:
        """
        Apply Quantization Aware Training (QAT).
        
        QAT simulates quantization during training to maintain accuracy.
        
        Args:
            model: PyTorch model to quantize
            train_loader: Training data loader
            optimizer: Training optimizer
            criterion: Loss criterion
            epochs: Number of QAT epochs
            qconfig_mapping: Quantization configuration mapping
            
        Returns:
            QAT quantized model
        """
        try:
            self.logger.info(f"Starting Quantization Aware Training for {epochs} epochs")
            
            # Default quantization configuration for QAT
            if qconfig_mapping is None:
                qconfig_mapping = get_default_qconfig_mapping(self.backend)
            
            # Prepare model for QAT
            model.train()
            prepared_model = torch.quantization.prepare_qat(model, qconfig_mapping)
            
            # QAT training loop
            for epoch in range(epochs):
                self.logger.info(f"QAT Epoch {epoch + 1}/{epochs}")
                
                running_loss = 0.0
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    
                    output = prepared_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                    if batch_idx % 100 == 0:
                        self.logger.debug(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                
                avg_loss = running_loss / len(train_loader)
                self.logger.info(f"QAT Epoch {epoch + 1} completed, Average Loss: {avg_loss:.4f}")
            
            # Convert to quantized model
            prepared_model.eval()
            quantized_model = torch.quantization.convert(prepared_model)
            
            self.logger.info("Quantization Aware Training completed successfully")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Quantization Aware Training failed: {e}")
            self.logger.warning("Returning original model")
            return model
    
    def quantize_fx(self,
                   model: nn.Module,
                   example_inputs: torch.Tensor,
                   qconfig_mapping = None,
                   calibration_fn: Optional[Callable] = None) -> nn.Module:
        """
        Apply FX-based quantization (PyTorch 1.8+).
        
        FX quantization provides more flexibility and better debugging capabilities.
        
        Args:
            model: PyTorch model to quantize
            example_inputs: Example inputs for tracing
            qconfig_mapping: Quantization configuration mapping
            calibration_fn: Calibration function
            
        Returns:
            FX quantized model
        """
        if not AO_QUANTIZATION_AVAILABLE:
            self.logger.warning("AO quantization not available, falling back to legacy quantization")
            return self.quantize_dynamic(model)
        
        try:
            self.logger.info("Applying FX-based quantization")
            
            # Set model to evaluation mode
            model.eval()
            
            # Default quantization configuration
            if qconfig_mapping is None:
                qconfig_mapping = ao_quant.get_default_qconfig_mapping()
            
            # Prepare model using FX
            prepared_model = ao_quant.prepare_fx(model, qconfig_mapping, example_inputs)
            
            # Calibrate if calibration function is provided
            if calibration_fn is not None:
                self.logger.info("Running calibration")
                calibration_fn(prepared_model)
            else:
                # Simple calibration with example inputs
                with torch.no_grad():
                    prepared_model(example_inputs)
            
            # Convert to quantized model
            quantized_model = ao_quant.convert_fx(prepared_model)
            
            self.logger.info("FX quantization completed successfully")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"FX quantization failed: {e}")
            self.logger.warning("Returning original model")
            return model
    
    def benchmark_quantization(self,
                             original_model: nn.Module,
                             quantized_model: nn.Module,
                             example_inputs: torch.Tensor,
                             iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark original vs quantized model performance.
        
        Args:
            original_model: Original PyTorch model
            quantized_model: Quantized model
            example_inputs: Input tensor
            iterations: Number of benchmark iterations
            
        Returns:
            Benchmark results including performance and accuracy metrics
        """
        results = {}
        
        # Performance benchmarking
        original_model.eval()
        quantized_model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = original_model(example_inputs)
                _ = quantized_model(example_inputs)
        
        # Benchmark original model
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                original_output = original_model(example_inputs)
        original_time = time.time() - start_time
        
        # Benchmark quantized model
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                quantized_output = quantized_model(example_inputs)
        quantized_time = time.time() - start_time
        
        # Calculate performance metrics
        original_fps = iterations / original_time
        quantized_fps = iterations / quantized_time
        speedup = original_time / quantized_time
        
        # Calculate accuracy metrics (if possible)
        try:
            with torch.no_grad():
                original_out = original_model(example_inputs)
                quantized_out = quantized_model(example_inputs)
                
                # Mean squared error
                mse = torch.mean((original_out - quantized_out) ** 2).item()
                
                # Mean absolute error
                mae = torch.mean(torch.abs(original_out - quantized_out)).item()
                
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    original_out.flatten(), 
                    quantized_out.flatten(), 
                    dim=0
                ).item()
                
                accuracy_metrics = {
                    "mse": mse,
                    "mae": mae,
                    "cosine_similarity": cos_sim
                }
        except Exception as e:
            self.logger.warning(f"Could not compute accuracy metrics: {e}")
            accuracy_metrics = {}
        
        # Model size comparison
        original_size = self._get_model_size(original_model)
        quantized_size = self._get_model_size(quantized_model)
        size_reduction = (original_size - quantized_size) / original_size * 100
        
        results = {
            "iterations": iterations,
            "performance": {
                "original_time_s": original_time,
                "quantized_time_s": quantized_time,
                "original_fps": original_fps,
                "quantized_fps": quantized_fps,
                "speedup": speedup,
                "improvement_percent": (speedup - 1) * 100
            },
            "model_size": {
                "original_size_mb": original_size / (1024 * 1024),
                "quantized_size_mb": quantized_size / (1024 * 1024),
                "size_reduction_percent": size_reduction
            },
            "accuracy": accuracy_metrics
        }
        
        self.logger.info(f"Quantization speedup: {speedup:.2f}x, Size reduction: {size_reduction:.1f}%")
        
        return results
    
    def _get_model_size(self, model: nn.Module) -> int:
        """
        Calculate model size in bytes.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in bytes
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return param_size + buffer_size
    
    def save_quantized_model(self, model: nn.Module, filepath: str) -> bool:
        """
        Save quantized model to disk.
        
        Args:
            model: Quantized model
            filepath: Path to save model
            
        Returns:
            Success status
        """
        try:
            torch.save(model.state_dict(), filepath)
            self.logger.info(f"Quantized model saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save quantized model: {e}")
            return False
    
    def load_quantized_model(self, model: nn.Module, filepath: str, device: Optional[torch.device] = None) -> Optional[nn.Module]:
        """
        Load quantized model from disk.
        
        Args:
            model: Model architecture (for loading state dict)
            filepath: Path to model file
            device: Target device
            
        Returns:
            Loaded quantized model or None if failed
        """
        try:
            if device is None:
                device = torch.device("cpu")  # Quantized models typically run on CPU
            
            state_dict = torch.load(filepath, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            
            self.logger.info(f"Quantized model loaded from {filepath}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load quantized model: {e}")
            return None


class QuantizedModelWrapper:
    """
    Wrapper for quantized models with additional utilities.
    """
    
    def __init__(self, quantized_model: nn.Module, original_model: Optional[nn.Module] = None):
        """
        Initialize wrapper.
        
        Args:
            quantized_model: Quantized model
            original_model: Original model (for fallback)
        """
        self.quantized_model = quantized_model
        self.original_model = original_model
        self.logger = logging.getLogger(f"{__name__}.QuantizedModelWrapper")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fallback to original model on error.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        try:
            return self.quantized_model(x)
        except Exception as e:
            self.logger.warning(f"Quantized model inference failed: {e}")
            if self.original_model is not None:
                self.logger.info("Falling back to original model")
                return self.original_model(x)
            else:
                raise e
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Make the wrapper callable."""
        return self.forward(x)
    
    def eval(self):
        """Set to evaluation mode."""
        self.quantized_model.eval()
        if self.original_model is not None:
            self.original_model.eval()
        return self
    
    def to(self, device):
        """Move to device."""
        # Note: Quantized models typically stay on CPU
        if device.type != 'cpu':
            self.logger.warning("Quantized models typically run on CPU")
        
        self.quantized_model = self.quantized_model.to(device)
        if self.original_model is not None:
            self.original_model = self.original_model.to(device)
        return self


def quantize_model(model: nn.Module,
                  method: str = "dynamic",
                  config: Optional[InferenceConfig] = None,
                  **kwargs) -> QuantizedModelWrapper:
    """
    Convenience function to quantize PyTorch model.
    
    Args:
        model: PyTorch model
        method: Quantization method ("dynamic", "static", "qat", "fx")
        config: Inference configuration
        **kwargs: Additional quantization arguments
        
    Returns:
        Quantized model wrapper
    """
    optimizer = QuantizationOptimizer(config)
    
    # Remove 'quantization_type' from kwargs if present (for backwards compatibility)
    kwargs.pop('quantization_type', None)
    
    if method == "dynamic":
        quantized = optimizer.quantize_dynamic(model, **kwargs)
    elif method == "static":
        quantized = optimizer.quantize_static(model, **kwargs)
    elif method == "qat":
        quantized = optimizer.quantize_qat(model, **kwargs)
    elif method == "fx":
        quantized = optimizer.quantize_fx(model, **kwargs)
    else:
        logger.warning(f"Unknown quantization method: {method}, using dynamic")
        quantized = optimizer.quantize_dynamic(model, **kwargs)
    
    return QuantizedModelWrapper(quantized, model)


# Global quantization optimizer instance
_global_quant_optimizer: Optional[QuantizationOptimizer] = None


def get_quantization_optimizer() -> QuantizationOptimizer:
    """Get global quantization optimizer instance."""
    global _global_quant_optimizer
    if _global_quant_optimizer is None:
        _global_quant_optimizer = QuantizationOptimizer()
    return _global_quant_optimizer

"""
ONNX optimization module for PyTorch models.

This module provides functionality to convert PyTorch models to ONNX
and use ONNX Runtime for optimized inference.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import numpy as np

# ONNX imports with fallback
try:
    import onnx
    import onnxruntime as ort
    # Make onnxruntime available for tests
    onnxruntime = ort
    ONNX_AVAILABLE = True
except ImportError:
    onnx = None
    ort = None
    onnxruntime = None
    ONNX_AVAILABLE = False
    # Only warn when optimizer is actually used, not on import
    pass

from ..core.config import InferenceConfig


logger = logging.getLogger(__name__)


class ONNXOptimizer:
    """
    ONNX optimization manager for PyTorch models.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize ONNX optimizer.
        
        Args:
            config: Inference configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ONNXOptimizer")
        
        if not ONNX_AVAILABLE:
            self.logger.warning("ONNX/ONNXRuntime not available")
            self.enabled = False
            return
        
        self.enabled = True
        self.ort_session = None
        self.input_names = []
        self.output_names = []
        
        self.logger.info("ONNX optimizer initialized")
    
    def optimize(self, model: nn.Module, example_inputs: torch.Tensor, **kwargs) -> 'ONNXModelWrapper':
        """
        Optimize model by converting to ONNX.
        
        Args:
            model: PyTorch model to optimize
            example_inputs: Example inputs for export
            **kwargs: Additional arguments
            
        Returns:
            ONNX model wrapper
        """
        return self.optimize_model(model, example_inputs, **kwargs)
    
    def export_to_onnx(self, 
                      model: nn.Module,
                      example_inputs: torch.Tensor,
                      onnx_path: str,
                      input_names: Optional[List[str]] = None,
                      output_names: Optional[List[str]] = None,
                      dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                      opset_version: int = 11) -> bool:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            example_inputs: Example input tensor
            onnx_path: Path to save ONNX model
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes specification
            opset_version: ONNX opset version
            
        Returns:
            Success status
        """
        if not self.enabled:
            self.logger.warning("ONNX not enabled")
            return False
        
        try:
            self.logger.info(f"Exporting model to ONNX: {onnx_path}")
            
            # Set model to evaluation mode
            model.eval()
            
            # Default names if not provided
            if input_names is None:
                input_names = ["input"]
            if output_names is None:
                output_names = ["output"]
            
            # Default dynamic axes for batch dimension
            if dynamic_axes is None:
                dynamic_axes = {
                    input_names[0]: {0: "batch_size"},
                    output_names[0]: {0: "batch_size"}
                }
            
            # Export to ONNX
            torch.onnx.export(
                model,
                example_inputs,
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
                verbose=False
            )
            
            # Verify the exported model (optional check)
            try:
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                self.logger.info(f"ONNX export successful and verified: {onnx_path}")
            except Exception as verification_error:
                self.logger.warning(f"ONNX model verification failed: {verification_error}, but export succeeded")
            
            return True
            
        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            return False
    
    def create_ort_session(self,
                          onnx_path: str,
                          providers: Optional[List[str]] = None,
                          session_options: Optional[Any] = None) -> bool:
        """
        Create ONNX Runtime session for inference.
        
        Args:
            onnx_path: Path to ONNX model
            providers: List of execution providers
            session_options: ONNX Runtime session options
            
        Returns:
            Success status
        """
        if not self.enabled:
            return False
        
        try:
            # Default providers
            if providers is None:
                providers = []
                if torch.cuda.is_available():
                    providers.append('CUDAExecutionProvider')
                providers.append('CPUExecutionProvider')
            
            # Default session options
            if session_options is None:
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                # Set thread options
                if hasattr(self.config, 'performance') and self.config.performance.max_workers:
                    session_options.intra_op_num_threads = self.config.performance.max_workers
                    session_options.inter_op_num_threads = self.config.performance.max_workers
            
            # Create session
            self.ort_session = ort.InferenceSession(
                onnx_path,
                sess_options=session_options,
                providers=providers
            )
            
            # Get input/output names
            self.input_names = [inp.name for inp in self.ort_session.get_inputs()]
            self.output_names = [out.name for out in self.ort_session.get_outputs()]
            
            self.logger.info(f"ONNX Runtime session created with providers: {providers}")
            self.logger.info(f"Input names: {self.input_names}")
            self.logger.info(f"Output names: {self.output_names}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create ONNX Runtime session: {e}")
            return False
    
    def predict(self, inputs: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Run inference using ONNX Runtime.
        
        Args:
            inputs: Input tensor or array
            
        Returns:
            Prediction results as torch tensor
        """
        if not self.enabled or self.ort_session is None:
            raise RuntimeError("ONNX session not initialized")
        
        try:
            # Convert to numpy if needed
            if isinstance(inputs, torch.Tensor):
                if inputs.requires_grad:
                    inputs = inputs.detach()
                inputs_np = inputs.cpu().numpy()
            else:
                inputs_np = inputs
            
            # Prepare input dictionary
            input_dict = {self.input_names[0]: inputs_np}
            
            # Run inference
            outputs = self.ort_session.run(self.output_names, input_dict)
            
            # Convert back to torch tensor
            result = torch.from_numpy(outputs[0])
            
            return result
            
        except Exception as e:
            self.logger.error(f"ONNX inference failed: {e}")
            raise
    
    def optimize_model(self,
                      model: nn.Module,
                      example_inputs: torch.Tensor,
                      optimization_level: str = "all",
                      providers: Optional[List[str]] = None,
                      output_path: Optional[str] = None) -> 'ONNXModelWrapper':
        """
        Full optimization pipeline: PyTorch -> ONNX -> ONNX Runtime.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs
            optimization_level: ONNX optimization level
            providers: Execution providers
            output_path: Optional path to save ONNX model
            
        Returns:
            ONNX model wrapper
        """
        if not self.enabled:
            self.logger.warning("ONNX not enabled, returning original model")
            return ONNXModelWrapper(None, model)
        
        try:
            # Create temporary ONNX file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
                onnx_path = f.name
            
            # Export to ONNX
            success = self.export_to_onnx(model, example_inputs, onnx_path)
            if not success:
                if os.path.exists(onnx_path):
                    os.unlink(onnx_path)
                self.logger.warning("ONNX export failed, returning wrapper with original model")
                return ONNXModelWrapper(None, model)
            
            # Create ONNX Runtime session
            success = self.create_ort_session(onnx_path, providers)
            if not success:
                if os.path.exists(onnx_path):
                    os.unlink(onnx_path)
                self.logger.warning("ONNX Runtime session creation failed, returning wrapper with original model")
                return ONNXModelWrapper(None, model)
            
            # Create wrapper
            wrapper = ONNXModelWrapper(self, model)
            
            # Clean up temporary file
            os.unlink(onnx_path)
            
            self.logger.info("ONNX optimization completed successfully")
            return wrapper
            
        except Exception as e:
            self.logger.error(f"ONNX optimization failed: {e}")
            return ONNXModelWrapper(None, model)
    
    def benchmark_optimization(self,
                             original_model: nn.Module,
                             optimized_wrapper: 'ONNXModelWrapper',
                             example_inputs: torch.Tensor,
                             iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark original vs ONNX optimized model.
        
        Args:
            original_model: Original PyTorch model
            optimized_wrapper: ONNX model wrapper
            example_inputs: Input tensor
            iterations: Number of benchmark iterations
            
        Returns:
            Benchmark results
        """
        if not self.enabled:
            return {"error": "ONNX not enabled"}
        
        results = {}
        
        # Benchmark original model
        original_model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = original_model(example_inputs)
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = original_model(example_inputs)
        original_time = time.time() - start_time
        
        # Benchmark ONNX model
        # Warmup
        for _ in range(10):
            _ = optimized_wrapper(example_inputs)
        
        # Measure
        start_time = time.time()
        for _ in range(iterations):
            _ = optimized_wrapper(example_inputs)
        onnx_time = time.time() - start_time
        
        # Calculate results
        original_fps = iterations / original_time
        onnx_fps = iterations / onnx_time
        speedup = original_time / onnx_time
        
        results = {
            "iterations": iterations,
            "original_time_s": original_time,
            "onnx_time_s": onnx_time,
            "original_fps": original_fps,
            "onnx_fps": onnx_fps,
            "speedup": speedup,
            "improvement_percent": (speedup - 1) * 100
        }
        
        self.logger.info(f"ONNX optimization speedup: {speedup:.2f}x ({results['improvement_percent']:.1f}% faster)")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ONNX model information."""
        if not self.enabled or self.ort_session is None:
            return {"enabled": False}
        
        info = {
            "enabled": True,
            "providers": self.ort_session.get_providers(),
            "input_names": self.input_names,
            "output_names": self.output_names
        }
        
        # Get input/output shapes
        inputs_info = []
        for inp in self.ort_session.get_inputs():
            inputs_info.append({
                "name": inp.name,
                "shape": inp.shape,
                "type": inp.type
            })
        
        outputs_info = []
        for out in self.ort_session.get_outputs():
            outputs_info.append({
                "name": out.name,
                "shape": out.shape,
                "type": out.type
            })
        
        info["inputs"] = inputs_info
        info["outputs"] = outputs_info
        
        return info


class ONNXModelWrapper:
    """
    Wrapper for ONNX models with PyTorch-like interface.
    """
    
    def __init__(self, onnx_optimizer: Optional[ONNXOptimizer], fallback_model: Optional[nn.Module] = None):
        """
        Initialize wrapper.
        
        Args:
            onnx_optimizer: ONNX optimizer instance with session
            fallback_model: Fallback PyTorch model
        """
        self.onnx_optimizer = onnx_optimizer
        self.fallback_model = fallback_model
        self.logger = logging.getLogger(f"{__name__}.ONNXModelWrapper")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fallback to original model on error.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        if self.onnx_optimizer is not None and self.onnx_optimizer.enabled:
            try:
                return self.onnx_optimizer.predict(x)
            except Exception as e:
                self.logger.warning(f"ONNX inference failed: {e}")
        
        if self.fallback_model is not None:
            self.logger.info("Using fallback PyTorch model")
            return self.fallback_model(x)
        else:
            raise RuntimeError("No ONNX session and no fallback model available")
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Make the wrapper callable."""
        return self.forward(x)
    
    def eval(self):
        """Set to evaluation mode."""
        if self.fallback_model is not None:
            self.fallback_model.eval()
        return self
    
    def to(self, device):
        """Move to device (for fallback model only)."""
        if self.fallback_model is not None:
            self.fallback_model = self.fallback_model.to(device)
        return self
    
    def cuda(self):
        """Move to CUDA (for fallback model only)."""
        return self.to(torch.device("cuda"))
    
    def cpu(self):
        """Move to CPU (for fallback model only)."""
        return self.to(torch.device("cpu"))


def convert_to_onnx(model: nn.Module,
                   example_inputs: torch.Tensor,
                   config: Optional[InferenceConfig] = None,
                   **kwargs) -> ONNXModelWrapper:
    """
    Convenience function to convert PyTorch model to ONNX.
    
    Args:
        model: PyTorch model
        example_inputs: Example inputs
        config: Inference configuration
        **kwargs: Additional ONNX arguments
        
    Returns:
        ONNX model wrapper
    """
    optimizer = ONNXOptimizer(config)
    return optimizer.optimize(model, example_inputs, **kwargs)


# Global ONNX optimizer instance
_global_onnx_optimizer: Optional[ONNXOptimizer] = None


def get_onnx_optimizer() -> ONNXOptimizer:
    """Get global ONNX optimizer instance."""
    global _global_onnx_optimizer
    if _global_onnx_optimizer is None:
        _global_onnx_optimizer = ONNXOptimizer()
    return _global_onnx_optimizer

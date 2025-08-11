"""
TensorRT optimization module for PyTorch models.

This module provides functionality to convert PyTorch models to TensorRT
for significant performance improvements on NVIDIA GPUs.
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

# TensorRT modules - imported lazily
trt = None
torch_tensorrt = None
TRT_AVAILABLE = None

def _ensure_tensorrt_imported():
    """Ensure TensorRT modules are imported (lazy loading)."""
    global trt, torch_tensorrt, TRT_AVAILABLE
    
    if TRT_AVAILABLE is not None:
        return TRT_AVAILABLE
    
    try:
        import tensorrt as trt_module
        import torch_tensorrt as torch_tensorrt_module
        trt = trt_module
        torch_tensorrt = torch_tensorrt_module
        TRT_AVAILABLE = True
        return True
    except ImportError as e:
        warnings.warn(f"TensorRT not available: {e}. Install torch-tensorrt and tensorrt for optimization.")
        TRT_AVAILABLE = False
        return False

from ..core.config import InferenceConfig


logger = logging.getLogger(__name__)


class TensorRTOptimizer:
    """
    TensorRT optimization manager for PyTorch models.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize TensorRT optimizer.
        
        Args:
            config: Inference configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.TensorRTOptimizer")
        
        # Try to import TensorRT modules when initializing
        if not _ensure_tensorrt_imported():
            self.logger.warning("TensorRT not available")
            self.enabled = False
            return
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, TensorRT optimization disabled")
            self.enabled = False
            return
        
        self.enabled = True
        self.logger.info("TensorRT optimizer initialized")
    
    def optimize_model(self, 
                      model: nn.Module, 
                      example_inputs: torch.Tensor,
                      precision: str = "fp16",
                      workspace_size: int = 1 << 30,  # 1GB
                      max_batch_size: int = 32,
                      cache_dir: Optional[str] = None) -> nn.Module:
        """
        Optimize a PyTorch model using TensorRT.
        
        Args:
            model: PyTorch model to optimize
            example_inputs: Example input tensor for tracing
            precision: Precision mode ("fp32", "fp16", "int8")
            workspace_size: TensorRT workspace size in bytes
            max_batch_size: Maximum batch size
            cache_dir: Directory to cache compiled models
            
        Returns:
            TensorRT optimized model
        """
        if not self.enabled:
            self.logger.warning("TensorRT not enabled, returning original model")
            return model
        
        try:
            self.logger.info(f"Converting model to TensorRT with {precision} precision")
            
            # Set model to evaluation mode
            model.eval()
            
            # Move model and inputs to GPU
            device = torch.device("cuda")
            model = model.to(device)
            example_inputs = example_inputs.to(device)
            
            # Configure TensorRT compilation settings
            compile_settings = {
                "inputs": [
                    torch_tensorrt.Input(
                        shape=list(example_inputs.shape),
                        dtype=example_inputs.dtype,
                        name="input"
                    )
                ],
                "enabled_precisions": self._get_precision_set(precision),
                "workspace_size": workspace_size,
                "max_batch_size": max_batch_size,
                "truncate_long_and_double": True,
                "refit": False,
                "debug": False,
                "device": {
                    "device_type": torch_tensorrt.DeviceType.GPU,
                    "gpu_id": device.index or 0
                }
            }
            
            # Add cache directory if provided
            if cache_dir:
                cache_path = Path(cache_dir)
                cache_path.mkdir(parents=True, exist_ok=True)
                compile_settings["engine_cache_dir"] = str(cache_path)
                compile_settings["engine_cache_size"] = 1 << 30  # 1GB cache
            
            # Trace the model first
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_inputs)
            
            # Compile with TensorRT
            start_time = time.time()
            trt_model = torch_tensorrt.compile(traced_model, **compile_settings)
            compilation_time = time.time() - start_time
            
            self.logger.info(f"TensorRT compilation completed in {compilation_time:.2f}s")
            
            return trt_model
            
        except Exception as e:
            self.logger.error(f"TensorRT optimization failed: {e}")
            self.logger.warning("Falling back to original model")
            return model
    
    def _get_precision_set(self, precision: str) -> set:
        """Get TensorRT precision set based on string."""
        precision_map = {
            "fp32": {torch.float32},
            "fp16": {torch.float32, torch.half},
            "int8": {torch.float32, torch.half, torch.int8}
        }
        
        if precision not in precision_map:
            self.logger.warning(f"Unknown precision {precision}, using fp16")
            precision = "fp16"
        
        return precision_map[precision]
    
    def benchmark_optimization(self, 
                             original_model: nn.Module, 
                             optimized_model: nn.Module,
                             example_inputs: torch.Tensor,
                             iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark original vs optimized model performance.
        
        Args:
            original_model: Original PyTorch model
            optimized_model: TensorRT optimized model
            example_inputs: Input tensor for benchmarking
            iterations: Number of benchmark iterations
            
        Returns:
            Benchmark results
        """
        if not self.enabled:
            return {"error": "TensorRT not enabled"}
        
        device = torch.device("cuda")
        example_inputs = example_inputs.to(device)
        
        results = {}
        
        # Benchmark original model
        original_model.eval()
        torch.cuda.synchronize()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = original_model(example_inputs)
        torch.cuda.synchronize()
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = original_model(example_inputs)
        torch.cuda.synchronize()
        original_time = time.time() - start_time
        
        # Benchmark optimized model
        optimized_model.eval()
        torch.cuda.synchronize()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = optimized_model(example_inputs)
        torch.cuda.synchronize()
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = optimized_model(example_inputs)
        torch.cuda.synchronize()
        optimized_time = time.time() - start_time
        
        # Calculate results
        original_fps = iterations / original_time
        optimized_fps = iterations / optimized_time
        speedup = original_time / optimized_time
        
        results = {
            "iterations": iterations,
            "original_time_s": original_time,
            "optimized_time_s": optimized_time,
            "original_fps": original_fps,
            "optimized_fps": optimized_fps,
            "speedup": speedup,
            "improvement_percent": (speedup - 1) * 100
        }
        
        self.logger.info(f"TensorRT optimization speedup: {speedup:.2f}x ({results['improvement_percent']:.1f}% faster)")
        
        return results
    
    def save_engine(self, model: nn.Module, filepath: str) -> bool:
        """
        Save TensorRT engine to disk.
        
        Args:
            model: TensorRT model
            filepath: Path to save engine
            
        Returns:
            Success status
        """
        try:
            if hasattr(model, 'save'):
                model.save(filepath)
                self.logger.info(f"TensorRT engine saved to {filepath}")
                return True
            else:
                # For torch_tensorrt models, save as TorchScript
                torch.jit.save(model, filepath)
                self.logger.info(f"TensorRT model saved as TorchScript to {filepath}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to save TensorRT engine: {e}")
            return False
    
    def load_engine(self, filepath: str, device: Optional[torch.device] = None) -> Optional[nn.Module]:
        """
        Load TensorRT engine from disk.
        
        Args:
            filepath: Path to engine file
            device: Target device
            
        Returns:
            Loaded TensorRT model or None if failed
        """
        try:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            model = torch.jit.load(filepath, map_location=device)
            self.logger.info(f"TensorRT model loaded from {filepath}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load TensorRT engine: {e}")
            return None
    
    def check_compatibility(self, model: nn.Module) -> Dict[str, bool]:
        """
        Check if model is compatible with TensorRT optimization.
        
        Args:
            model: PyTorch model to check
            
        Returns:
            Compatibility report
        """
        compatibility = {
            "tensorrt_available": self.enabled,
            "cuda_available": torch.cuda.is_available(),
            "model_traceable": True,
            "unsupported_ops": []
        }
        
        if not self.enabled:
            return compatibility
        
        try:
            # Try to trace the model with dummy input
            model.eval()
            dummy_input = torch.randn(1, 3, 224, 224)  # Common image input
            if torch.cuda.is_available():
                model = model.cuda()
                dummy_input = dummy_input.cuda()
            
            with torch.no_grad():
                traced_model = torch.jit.trace(model, dummy_input)
                compatibility["model_traceable"] = True
                
        except Exception as e:
            compatibility["model_traceable"] = False
            compatibility["trace_error"] = str(e)
        
        return compatibility


def convert_to_tensorrt(model: nn.Module, 
                       example_inputs: torch.Tensor,
                       config: Optional[InferenceConfig] = None,
                       **kwargs) -> nn.Module:
    """
    Convenience function to convert PyTorch model to TensorRT.
    
    Args:
        model: PyTorch model
        example_inputs: Example inputs for tracing
        config: Inference configuration
        **kwargs: Additional TensorRT compilation arguments
        
    Returns:
        TensorRT optimized model
    """
    optimizer = TensorRTOptimizer(config)
    return optimizer.optimize_model(model, example_inputs, **kwargs)


class TensorRTModelWrapper:
    """
    Wrapper for TensorRT models with additional utilities.
    """
    
    def __init__(self, trt_model: nn.Module, original_model: Optional[nn.Module] = None):
        """
        Initialize wrapper.
        
        Args:
            trt_model: TensorRT optimized model
            original_model: Original PyTorch model (for fallback)
        """
        self.trt_model = trt_model
        self.original_model = original_model
        self.device = next(trt_model.parameters()).device
        self.logger = logging.getLogger(f"{__name__}.TensorRTModelWrapper")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fallback to original model on error.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        try:
            return self.trt_model(x)
        except Exception as e:
            self.logger.warning(f"TensorRT inference failed: {e}")
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
        self.trt_model.eval()
        if self.original_model is not None:
            self.original_model.eval()
        return self
    
    def to(self, device):
        """Move to device."""
        self.trt_model = self.trt_model.to(device)
        if self.original_model is not None:
            self.original_model = self.original_model.to(device)
        self.device = device
        return self
    
    def cuda(self):
        """Move to CUDA."""
        return self.to(torch.device("cuda"))
    
    def cpu(self):
        """Move to CPU."""
        return self.to(torch.device("cpu"))


def create_tensorrt_cache_key(model_config: Dict[str, Any], 
                            input_shape: Tuple[int, ...],
                            precision: str) -> str:
    """
    Create a cache key for TensorRT models.
    
    Args:
        model_config: Model configuration
        input_shape: Input tensor shape
        precision: Precision mode
        
    Returns:
        Cache key string
    """
    import hashlib
    
    key_data = {
        "input_shape": input_shape,
        "precision": precision,
        "model_config": model_config
    }
    
    key_str = str(sorted(key_data.items()))
    return hashlib.md5(key_str.encode()).hexdigest()


# Global TensorRT optimizer instance
_global_trt_optimizer: Optional[TensorRTOptimizer] = None


def get_tensorrt_optimizer() -> TensorRTOptimizer:
    """Get global TensorRT optimizer instance."""
    global _global_trt_optimizer
    if _global_trt_optimizer is None:
        _global_trt_optimizer = TensorRTOptimizer()
    return _global_trt_optimizer

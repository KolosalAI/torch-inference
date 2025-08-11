"""
JIT (Just-In-Time) compilation optimization module for PyTorch models.

This module provides TorchScript compilation and optimization for
improved inference performance.
"""

import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch import jit

from ..core.config import InferenceConfig


logger = logging.getLogger(__name__)


class JITOptimizer:
    """
    JIT optimization manager for PyTorch models.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize JIT optimizer.
        
        Args:
            config: Inference configuration
        """
        self.config = config
        self.compiled_models = {}
        
        self.logger = logging.getLogger(f"{__name__}.JITOptimizer")
        self.logger.info("JIT optimizer initialized")
    
    def trace_model(self, 
                   model: nn.Module, 
                   example_inputs: torch.Tensor,
                   strict: bool = True,
                   optimize: bool = True) -> torch.jit.ScriptModule:
        """
        Trace PyTorch model for JIT compilation.
        
        Tracing records operations during a sample execution and creates
        a static computation graph.
        
        Args:
            model: PyTorch model to trace
            example_inputs: Example inputs for tracing
            strict: Whether to be strict about type checking
            optimize: Whether to apply optimizations
            
        Returns:
            Traced TorchScript model
        """
        try:
            self.logger.info("Tracing model for JIT compilation")
            
            # Set model to evaluation mode
            model.eval()
            
            # Trace the model
            with torch.no_grad():
                traced_model = torch.jit.trace(
                    model, 
                    example_inputs, 
                    strict=strict,
                    check_trace=True
                )
            
            if optimize:
                traced_model = self._optimize_traced_model(traced_model)
            
            self.logger.info("Model tracing completed successfully")
            return traced_model
            
        except Exception as e:
            self.logger.error(f"Model tracing failed: {e}")
            raise
    
    def script_model(self, 
                    model: nn.Module,
                    optimize: bool = True) -> torch.jit.ScriptModule:
        """
        Script PyTorch model for JIT compilation.
        
        Scripting compiles model code directly, preserving control flow.
        
        Args:
            model: PyTorch model to script
            optimize: Whether to apply optimizations
            
        Returns:
            Scripted TorchScript model
        """
        try:
            self.logger.info("Scripting model for JIT compilation")
            
            # Set model to evaluation mode
            model.eval()
            
            # Script the model
            scripted_model = torch.jit.script(model)
            
            if optimize:
                scripted_model = self._optimize_scripted_model(scripted_model)
            
            self.logger.info("Model scripting completed successfully")
            return scripted_model
            
        except Exception as e:
            self.logger.error(f"Model scripting failed: {e}")
            raise
    
    def _optimize_traced_model(self, traced_model: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """
        Apply optimizations to traced model.
        
        Args:
            traced_model: Traced TorchScript model
            
        Returns:
            Optimized traced model
        """
        try:
            self.logger.info("Applying optimizations to traced model")
            
            # Freeze the model (eliminates certain overhead)
            traced_model = torch.jit.freeze(traced_model)
            
            # Apply optimization passes
            traced_model = torch.jit.optimize_for_inference(traced_model)
            
            self.logger.info("Traced model optimization completed")
            return traced_model
            
        except Exception as e:
            self.logger.warning(f"Traced model optimization failed: {e}")
            return traced_model
    
    def _optimize_scripted_model(self, scripted_model: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """
        Apply optimizations to scripted model.
        
        Args:
            scripted_model: Scripted TorchScript model
            
        Returns:
            Optimized scripted model
        """
        try:
            self.logger.info("Applying optimizations to scripted model")
            
            # Freeze the model
            scripted_model = torch.jit.freeze(scripted_model)
            
            # Apply optimization passes
            scripted_model = torch.jit.optimize_for_inference(scripted_model)
            
            self.logger.info("Scripted model optimization completed")
            return scripted_model
            
        except Exception as e:
            self.logger.warning(f"Scripted model optimization failed: {e}")
            return scripted_model
    
    def compile_model(self,
                     model: nn.Module,
                     example_inputs: torch.Tensor,
                     method: str = "trace",
                     model_name: str = "default") -> torch.jit.ScriptModule:
        """
        Compile PyTorch model using specified method.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs
            method: Compilation method ("trace" or "script")
            model_name: Name for the compiled model
            
        Returns:
            Compiled TorchScript model
        """
        try:
            if method == "trace":
                compiled_model = self.trace_model(model, example_inputs)
            elif method == "script":
                compiled_model = self.script_model(model)
            else:
                raise ValueError(f"Unknown compilation method: {method}")
            
            # Store compiled model
            self.compiled_models[model_name] = compiled_model
            
            self.logger.info(f"Model '{model_name}' compiled successfully using {method}")
            return compiled_model
            
        except Exception as e:
            self.logger.error(f"Model compilation failed: {e}")
            raise
    
    def save_compiled_model(self, 
                           compiled_model: torch.jit.ScriptModule,
                           filepath: str) -> bool:
        """
        Save compiled TorchScript model to disk.
        
        Args:
            compiled_model: Compiled TorchScript model
            filepath: Path to save model
            
        Returns:
            Success status
        """
        try:
            compiled_model.save(filepath)
            self.logger.info(f"Compiled model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save compiled model: {e}")
            return False
    
    def load_compiled_model(self, 
                           filepath: str,
                           device: Optional[torch.device] = None) -> Optional[torch.jit.ScriptModule]:
        """
        Load compiled TorchScript model from disk.
        
        Args:
            filepath: Path to model file
            device: Target device
            
        Returns:
            Loaded compiled model or None if failed
        """
        try:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            compiled_model = torch.jit.load(filepath, map_location=device)
            self.logger.info(f"Compiled model loaded from {filepath}")
            return compiled_model
            
        except Exception as e:
            self.logger.error(f"Failed to load compiled model: {e}")
            return None
    
    def benchmark_compilation(self,
                            original_model: nn.Module,
                            compiled_model: torch.jit.ScriptModule,
                            example_inputs: torch.Tensor,
                            iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark original vs JIT compiled model performance.
        
        Args:
            original_model: Original PyTorch model
            compiled_model: JIT compiled model
            example_inputs: Input tensor
            iterations: Number of benchmark iterations
            
        Returns:
            Benchmark results
        """
        results = {}
        
        device = next(original_model.parameters()).device
        example_inputs = example_inputs.to(device)
        
        # Benchmark original model
        original_model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = original_model(example_inputs)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = original_model(example_inputs)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        original_time = time.time() - start_time
        
        # Benchmark compiled model
        compiled_model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = compiled_model(example_inputs)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = compiled_model(example_inputs)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        compiled_time = time.time() - start_time
        
        # Calculate results
        original_fps = iterations / original_time
        compiled_fps = iterations / compiled_time
        speedup = original_time / compiled_time
        
        results = {
            "iterations": iterations,
            "original_time_s": original_time,
            "compiled_time_s": compiled_time,
            "original_fps": original_fps,
            "compiled_fps": compiled_fps,
            "speedup": speedup,
            "improvement_percent": (speedup - 1) * 100
        }
        
        self.logger.info(f"JIT compilation speedup: {speedup:.2f}x ({results['improvement_percent']:.1f}% faster)")
        
        return results
    
    def analyze_model_graph(self, model: torch.jit.ScriptModule) -> Dict[str, Any]:
        """
        Analyze the computation graph of a JIT compiled model.
        
        Args:
            model: JIT compiled model
            
        Returns:
            Graph analysis results
        """
        try:
            graph = model.graph
            
            # Count different types of operations
            op_counts = {}
            node_count = 0
            
            for node in graph.nodes():
                node_count += 1
                op_kind = node.kind()
                op_counts[op_kind] = op_counts.get(op_kind, 0) + 1
            
            # Get input/output information
            inputs = list(graph.inputs())
            outputs = list(graph.outputs())
            
            analysis = {
                "total_nodes": node_count,
                "operation_counts": op_counts,
                "input_count": len(inputs),
                "output_count": len(outputs),
                "graph_string": str(graph)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Graph analysis failed: {e}")
            return {"error": str(e)}
    
    def get_compilation_info(self) -> Dict[str, Any]:
        """
        Get information about compiled models.
        
        Returns:
            Compilation information
        """
        info = {
            "compiled_models": list(self.compiled_models.keys()),
            "jit_available": hasattr(torch, 'jit'),
            "optimization_passes": []
        }
        
        # Try to get available optimization passes
        try:
            # This is implementation-specific and may vary
            info["optimization_passes"] = ["freeze", "optimize_for_inference"]
        except:
            pass
        
        return info
    
    def clear_compiled_models(self) -> None:
        """Clear all compiled models from cache."""
        self.compiled_models.clear()
        self.logger.info("Compiled models cache cleared")


class JITModelWrapper:
    """
    Wrapper for JIT compiled models with fallback support.
    """
    
    def __init__(self, 
                 compiled_model: torch.jit.ScriptModule, 
                 original_model: Optional[nn.Module] = None):
        """
        Initialize JIT model wrapper.
        
        Args:
            compiled_model: JIT compiled model
            original_model: Original model for fallback
        """
        self.compiled_model = compiled_model
        self.original_model = original_model
        self.device = next(compiled_model.parameters()).device
        
        self.logger = logging.getLogger(f"{__name__}.JITModelWrapper")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fallback to original model on error.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        try:
            return self.compiled_model(x)
        except Exception as e:
            self.logger.warning(f"JIT model inference failed: {e}")
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
        self.compiled_model.eval()
        if self.original_model is not None:
            self.original_model.eval()
        return self
    
    def to(self, device):
        """Move to device."""
        self.compiled_model = self.compiled_model.to(device)
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
    
    def save(self, filepath: str) -> bool:
        """
        Save compiled model to disk.
        
        Args:
            filepath: Path to save model
            
        Returns:
            Success status
        """
        try:
            self.compiled_model.save(filepath)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False


def jit_compile_model(model: nn.Module,
                     example_inputs: torch.Tensor,
                     method: str = "trace",
                     config: Optional[InferenceConfig] = None,
                     **kwargs) -> JITModelWrapper:
    """
    Convenience function to JIT compile PyTorch model.
    
    Args:
        model: PyTorch model
        example_inputs: Example inputs
        method: Compilation method ("trace" or "script")
        config: Inference configuration
        **kwargs: Additional compilation arguments
        
    Returns:
        JIT model wrapper
    """
    optimizer = JITOptimizer(config)
    compiled_model = optimizer.compile_model(model, example_inputs, method, **kwargs)
    return JITModelWrapper(compiled_model, model)


# Global JIT optimizer instance
_global_jit_optimizer: Optional[JITOptimizer] = None


def get_jit_optimizer() -> JITOptimizer:
    """Get global JIT optimizer instance."""
    global _global_jit_optimizer
    if _global_jit_optimizer is None:
        _global_jit_optimizer = JITOptimizer()
    return _global_jit_optimizer

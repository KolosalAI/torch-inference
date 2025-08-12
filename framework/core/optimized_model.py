"""
Optimized model adapter for PyTorch inference.

This module provides an enhanced model adapter that automatically applies
various optimizations including TensorRT, ONNX, quantization, and more.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import warnings

import torch
import torch.nn as nn

from .base_model import BaseModel, ModelLoadError, ModelInferenceError
from .config import InferenceConfig
from ..optimizers import (
    TensorRTOptimizer, ONNXOptimizer, QuantizationOptimizer,
    MemoryOptimizer, CUDAOptimizer, JITOptimizer,
    convert_to_tensorrt, convert_to_onnx, quantize_model,
    enable_cuda_optimizations, jit_compile_model
)


logger = logging.getLogger(__name__)


class OptimizedModel(BaseModel):
    """
    Enhanced model class with automatic optimization capabilities.
    """
    
    def __init__(self, config: InferenceConfig):
        """
        Initialize optimized model.
        
        Args:
            config: Inference configuration with optimization settings
        """
        super().__init__(config)
        
        # Initialize optimizers
        self.optimizers = self._initialize_optimizers()
        self.optimized_models = {}
        self.active_optimization = "pytorch"  # Default to PyTorch
        
        # Performance tracking
        self.optimization_benchmarks = {}
        
        self.logger = logging.getLogger(f"{__name__}.OptimizedModel")
    
    def _initialize_optimizers(self) -> Dict[str, Any]:
        """Initialize all available optimizers."""
        optimizers = {
            'tensorrt': TensorRTOptimizer(self.config),
            'onnx': ONNXOptimizer(self.config),
            'quantization': QuantizationOptimizer(self.config),
            'memory': MemoryOptimizer(self.config),
            'cuda': CUDAOptimizer(self.config),
            'jit': JITOptimizer(self.config)
        }
        
        return optimizers
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load model and apply automatic optimizations.
        
        Args:
            model_path: Path to model file
        """
        super().load_model(model_path)
        
        if self._is_loaded:
            self.logger.info("Applying automatic optimizations")
            self._apply_optimizations()
    
    def _apply_optimizations(self) -> None:
        """Apply configured optimizations to the loaded model."""
        if not self._is_loaded or self.model is None:
            return
        
        # Create example input for optimization
        example_input = self._create_dummy_input()
        
        # Apply optimizations in priority order
        optimization_order = self._get_optimization_order()
        
        for optimization in optimization_order:
            try:
                self._apply_single_optimization(optimization, example_input)
            except Exception as e:
                self.logger.warning(f"Failed to apply {optimization} optimization: {e}")
        
        # Select best performing optimization
        self._select_best_optimization()
    
    def _get_optimization_order(self) -> List[str]:
        """Get the order in which optimizations should be applied."""
        order = []
        
        # CUDA optimizations first (if available)
        if torch.cuda.is_available() and getattr(self.config.device, 'use_cuda_optimizations', True):
            order.append('cuda')
        
        # JIT compilation
        if getattr(self.config.device, 'use_torch_compile', True):
            order.append('jit')
        
        # TensorRT (CUDA only, highest performance)
        if (torch.cuda.is_available() and 
            getattr(self.config.device, 'use_tensorrt', False)):
            order.append('tensorrt')
        
        # ONNX (good cross-platform performance)
        if getattr(self.config.device, 'use_onnx', False):
            order.append('onnx')
        
        # Quantization (memory and speed)
        if (getattr(self.config.device, 'use_int8', False) or 
            getattr(self.config.device, 'use_quantization', False)):
            order.append('quantization')
        
        # Memory optimizations (always beneficial)
        order.append('memory')
        
        return order
    
    def _apply_single_optimization(self, optimization: str, example_input: torch.Tensor) -> None:
        """Apply a single optimization technique."""
        self.logger.info(f"Applying {optimization} optimization")
        
        start_time = time.time()
        
        if optimization == 'tensorrt':
            self._apply_tensorrt_optimization(example_input)
        elif optimization == 'onnx':
            self._apply_onnx_optimization(example_input)
        elif optimization == 'quantization':
            self._apply_quantization_optimization()
        elif optimization == 'memory':
            self._apply_memory_optimization()
        elif optimization == 'cuda':
            self._apply_cuda_optimization()
        elif optimization == 'jit':
            self._apply_jit_optimization(example_input)
        
        optimization_time = time.time() - start_time
        self.logger.info(f"{optimization} optimization completed in {optimization_time:.2f}s")
    
    def _apply_tensorrt_optimization(self, example_input: torch.Tensor) -> None:
        """Apply TensorRT optimization."""
        optimizer = self.optimizers['tensorrt']
        
        if not optimizer.enabled:
            return
        
        try:
            precision = "fp16" if self.config.device.use_fp16 else "fp32"
            if self.config.device.use_int8:
                precision = "int8"
            
            optimized_model = optimizer.optimize_model(
                self.model,
                example_input,
                precision=precision,
                max_batch_size=self.config.batch.max_batch_size
            )
            
            self.optimized_models['tensorrt'] = optimized_model
            self.logger.info("TensorRT optimization successful")
            
        except Exception as e:
            self.logger.warning(f"TensorRT optimization failed: {e}")
    
    def _apply_onnx_optimization(self, example_input: torch.Tensor) -> None:
        """Apply ONNX optimization."""
        optimizer = self.optimizers['onnx']
        
        if not optimizer.enabled:
            return
        
        try:
            optimized_wrapper = optimizer.optimize_model(
                self.model,
                example_input
            )
            
            self.optimized_models['onnx'] = optimized_wrapper
            self.logger.info("ONNX optimization successful")
            
        except Exception as e:
            self.logger.warning(f"ONNX optimization failed: {e}")
    
    def _apply_quantization_optimization(self) -> None:
        """Apply quantization optimization."""
        optimizer = self.optimizers['quantization']
        
        try:
            method = "dynamic"  # Default to dynamic quantization
            
            if self.config.device.use_int8:
                method = "dynamic"  # Could be extended to static with calibration data
            
            optimized_wrapper = optimizer.quantize_model(
                self.model,
                method=method
            )
            
            self.optimized_models['quantization'] = optimized_wrapper
            self.logger.info("Quantization optimization successful")
            
        except Exception as e:
            self.logger.warning(f"Quantization optimization failed: {e}")
    
    def _apply_memory_optimization(self) -> None:
        """Apply memory optimization."""
        optimizer = self.optimizers['memory']
        
        try:
            optimized_model = optimizer.optimize_model_memory(self.model)
            self.optimized_models['memory'] = optimized_model
            self.logger.info("Memory optimization successful")
            
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
    
    def _apply_cuda_optimization(self) -> None:
        """Apply CUDA optimization."""
        optimizer = self.optimizers['cuda']
        
        if not optimizer.enabled:
            return
        
        try:
            optimized_model = optimizer.optimize_model_for_cuda(self.model)
            self.optimized_models['cuda'] = optimized_model
            self.logger.info("CUDA optimization successful")
            
        except Exception as e:
            self.logger.warning(f"CUDA optimization failed: {e}")
    
    def _apply_jit_optimization(self, example_input: torch.Tensor) -> None:
        """Apply JIT compilation optimization."""
        optimizer = self.optimizers['jit']
        
        try:
            method = "trace"  # Default to tracing
            
            optimized_wrapper = optimizer.compile_model(
                self.model,
                example_input,
                method=method
            )
            
            self.optimized_models['jit'] = optimized_wrapper
            self.logger.info("JIT optimization successful")
            
        except Exception as e:
            self.logger.warning(f"JIT optimization failed: {e}")
    
    def _select_best_optimization(self) -> None:
        """Benchmark optimizations and select the best one."""
        if not self.optimized_models:
            self.logger.info("No optimizations available, using original PyTorch model")
            return
        
        self.logger.info("Benchmarking optimizations to select the best one")
        
        example_input = self._create_dummy_input()
        benchmark_results = {}
        
        # Benchmark original model
        original_fps = self._benchmark_model(self.model, example_input)
        benchmark_results['pytorch'] = {'fps': original_fps, 'speedup': 1.0}
        
        # Benchmark optimized models
        for opt_name, opt_model in self.optimized_models.items():
            try:
                fps = self._benchmark_model(opt_model, example_input)
                speedup = fps / original_fps
                benchmark_results[opt_name] = {'fps': fps, 'speedup': speedup}
                
                self.logger.info(f"{opt_name}: {fps:.2f} FPS ({speedup:.2f}x speedup)")
                
            except Exception as e:
                self.logger.warning(f"Failed to benchmark {opt_name}: {e}")
        
        # Select best optimization
        best_optimization = max(
            benchmark_results.keys(),
            key=lambda k: benchmark_results[k]['fps']
        )
        
        self.active_optimization = best_optimization
        self.optimization_benchmarks = benchmark_results
        
        self.logger.info(f"Selected {best_optimization} as the best optimization "
                        f"({benchmark_results[best_optimization]['fps']:.2f} FPS)")
    
    def _benchmark_model(self, model: nn.Module, example_input: torch.Tensor, iterations: int = 50) -> float:
        """Benchmark model performance."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(example_input)
        
        # Synchronize if CUDA
        if example_input.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(example_input)
        
        if example_input.device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        fps = iterations / elapsed_time
        
        return fps
    
    def get_active_model(self) -> nn.Module:
        """Get the currently active (best performing) model."""
        if self.active_optimization == 'pytorch':
            return self.model
        elif self.active_optimization in self.optimized_models:
            return self.optimized_models[self.active_optimization]
        else:
            # Fallback to original model
            self.logger.warning(f"Active optimization {self.active_optimization} not found, using PyTorch model")
            return self.model
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the best performing model.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Model outputs
        """
        active_model = self.get_active_model()
        return active_model(inputs)
    
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """
        Preprocess inputs for inference.
        
        Args:
            inputs: Raw inputs
            
        Returns:
            Preprocessed tensor
        """
        # Basic preprocessing - convert to tensor and move to device
        if isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        elif isinstance(inputs, (list, tuple)):
            return torch.tensor(inputs, dtype=torch.float32, device=self.device)
        else:
            # For other types, try to convert to tensor
            return torch.tensor(inputs, dtype=torch.float32, device=self.device)
    
    def postprocess(self, outputs: torch.Tensor) -> Any:
        """
        Postprocess model outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Processed outputs
        """
        # Basic postprocessing - return as list or dict based on config
        outputs_cpu = outputs.detach().cpu()
        
        if self.config.model_type.value == "classification":
            # Apply softmax for classification
            if self.config.postprocessing.apply_softmax:
                outputs_cpu = torch.softmax(outputs_cpu, dim=-1)
            
            return {
                "predictions": outputs_cpu.tolist(),
                "raw_output": outputs.detach().cpu().tolist(),
                "shape": outputs.shape,
                "prediction": "optimized_result",
                "metadata": {
                    "output_type": "classification",
                    "shape": list(outputs.shape),
                    "dtype": str(outputs.dtype)
                }
            }
        else:
            # Generic output format
            return {
                "predictions": outputs_cpu.tolist(),
                "raw_output": outputs.detach().cpu().tolist(),
                "shape": outputs.shape,
                "prediction": "optimized_result",
                "metadata": {
                    "output_type": "optimized",
                    "shape": list(outputs.shape),
                    "dtype": str(outputs.dtype)
                }
            }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get detailed optimization report."""
        report = {
            "active_optimization": self.active_optimization,
            "available_optimizations": list(self.optimized_models.keys()),
            "benchmark_results": self.optimization_benchmarks,
            "optimizer_status": {}
        }
        
        # Get optimizer status
        for name, optimizer in self.optimizers.items():
            if hasattr(optimizer, 'enabled'):
                report["optimizer_status"][name] = {
                    "enabled": optimizer.enabled,
                    "available": True
                }
            else:
                report["optimizer_status"][name] = {
                    "enabled": True,
                    "available": True
                }
        
        # Memory usage comparison
        if 'memory' in self.optimizers:
            memory_stats = self.optimizers['memory'].get_memory_stats()
            report["memory_usage"] = memory_stats
        
        return report
    
    def switch_optimization(self, optimization_name: str) -> bool:
        """
        Switch to a different optimization.
        
        Args:
            optimization_name: Name of optimization to switch to
            
        Returns:
            Success status
        """
        if optimization_name == 'pytorch':
            self.active_optimization = optimization_name
            self.logger.info(f"Switched to {optimization_name}")
            return True
        elif optimization_name in self.optimized_models:
            self.active_optimization = optimization_name
            self.logger.info(f"Switched to {optimization_name}")
            return True
        else:
            self.logger.warning(f"Optimization {optimization_name} not available")
            return False
    
    def cleanup(self) -> None:
        """Cleanup all optimized models and resources."""
        super().cleanup()
        
        # Cleanup optimizers
        for optimizer in self.optimizers.values():
            if hasattr(optimizer, 'cleanup'):
                optimizer.cleanup()
        
        # Clear optimized models
        self.optimized_models.clear()
        
        self.logger.info("Optimization cleanup completed")


def create_optimized_model(config: InferenceConfig) -> OptimizedModel:
    """
    Factory function to create an optimized model.
    
    Args:
        config: Inference configuration
        
    Returns:
        OptimizedModel instance
    """
    return OptimizedModel(config)


class OptimizationConfig:
    """
    Configuration for model optimizations.
    """
    
    def __init__(self):
        # TensorRT settings
        self.use_tensorrt = False
        self.tensorrt_precision = "fp16"
        self.tensorrt_workspace_size = 1 << 30  # 1GB
        
        # ONNX settings
        self.use_onnx = False
        self.onnx_opset_version = 11
        
        # Quantization settings
        self.use_quantization = False
        self.quantization_method = "dynamic"
        
        # JIT settings
        self.use_jit = True
        self.jit_method = "trace"
        
        # CUDA settings
        self.use_cuda_optimizations = True
        self.use_cuda_graphs = False
        
        # Memory settings
        self.use_memory_optimizations = True
        self.memory_pool_size = 100
        
        # Benchmarking
        self.auto_select_best = True
        self.benchmark_iterations = 50
    
    def to_inference_config(self) -> Dict[str, Any]:
        """Convert to inference config dictionary."""
        return {
            'device': {
                'use_tensorrt': self.use_tensorrt,
                'use_fp16': self.tensorrt_precision == "fp16",
                'use_int8': self.tensorrt_precision == "int8",
                'use_onnx': self.use_onnx,
                'use_quantization': self.use_quantization,
                'use_torch_compile': self.use_jit,
                'use_cuda_optimizations': self.use_cuda_optimizations
            },
            'optimization': {
                'auto_select_best': self.auto_select_best,
                'benchmark_iterations': self.benchmark_iterations
            }
        }

"""
Performance Optimization Module for Hardware Acceleration

This module provides automatic hardware optimization with:
- GPU detection and optimization
- Memory management
- Hardware-specific acceleration
- Performance tuning
- Model optimization techniques
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
from ..core.config import InferenceConfig, DeviceConfig, DeviceType
from ..core.gpu_manager import GPUManager

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Automatic performance optimizer for inference workloads.
    
    Features:
    - Hardware detection and optimization
    - Memory optimization
    - Model compilation and fusion
    - Precision optimization
    - Batch size optimization
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.gpu_manager = GPUManager()
        self.optimizations_applied = []
        
    def optimize_device_config(self) -> DeviceConfig:
        """Optimize device configuration for best performance."""
        # Detect best GPU
        gpus, device_config = self.gpu_manager.detect_and_configure()
        
        # Apply performance-focused overrides
        if device_config.device_type == DeviceType.CUDA:
            device_config.use_fp16 = True  # Enable FP16 for speed
            device_config.use_torch_compile = True  # Enable compilation
            device_config.memory_fraction = 0.9  # Use more memory for performance
            self.optimizations_applied.append("CUDA FP16 enabled")
            self.optimizations_applied.append("torch.compile enabled")
        
        elif device_config.device_type == DeviceType.MPS:
            device_config.use_fp16 = True  # MPS supports FP16
            device_config.use_torch_compile = False  # May not be stable on MPS
            self.optimizations_applied.append("MPS FP16 enabled")
        
        # CPU optimizations
        elif device_config.device_type == DeviceType.CPU:
            device_config.use_torch_compile = True  # CPU can benefit from compilation
            self.optimizations_applied.append("CPU torch.compile enabled")
        
        logger.info(f"Device optimized: {device_config.device_type.value} - {', '.join(self.optimizations_applied)}")
        return device_config
    
    def optimize_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Optimize model for inference performance."""
        # Move to device first
        model = model.to(device)
        model.eval()
        
        # Apply optimizations
        optimized_model = self._apply_model_optimizations(model, device)
        
        return optimized_model
    
    def _apply_model_optimizations(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Apply various model optimizations."""
        # Set model to evaluation mode
        model.eval()
        
        # Disable gradients for inference
        for param in model.parameters():
            param.requires_grad = False
        
        # Apply torch.jit.script if possible
        try:
            if hasattr(torch.jit, 'script'):
                # Only script if model is scriptable
                model = torch.jit.script(model)
                self.optimizations_applied.append("TorchScript")
                logger.info("Model converted to TorchScript")
        except Exception as e:
            logger.debug(f"TorchScript conversion failed: {e}")
        
        # Apply torch.compile if available and enabled
        if (hasattr(torch, 'compile') and 
            self.config.device.use_torch_compile and 
            device.type in ['cuda', 'cpu']):
            try:
                model = torch.compile(
                    model,
                    mode='max-autotune',  # Aggressive optimization
                    fullgraph=False,      # Allow graph breaks
                    dynamic=True          # Handle dynamic shapes
                )
                self.optimizations_applied.append("torch.compile")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        # Enable CUDNN optimizations
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            self.optimizations_applied.append("CUDNN benchmark")
            logger.info("CUDNN benchmark mode enabled")
        
        # Apply layer fusion if available
        try:
            if hasattr(model, 'fuse_model'):
                model.fuse_model()
                self.optimizations_applied.append("Layer fusion")
                logger.info("Model layers fused")
        except Exception:
            pass
        
        return model
    
    def optimize_memory(self, device: torch.device) -> None:
        """Optimize memory usage for performance."""
        if device.type == 'cuda':
            # Clear cache
            torch.cuda.empty_cache()
            
            # Set memory fraction
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(self.config.device.memory_fraction)
            
            # Enable memory pool if available
            if hasattr(torch.cuda, 'memory_pool'):
                try:
                    # Pre-allocate memory pool
                    torch.cuda.memory_pool.resize_(device.index, 1024 * 1024 * 1024)  # 1GB
                    self.optimizations_applied.append("Memory pool optimization")
                except Exception:
                    pass
            
            self.optimizations_applied.append("CUDA memory optimization")
            logger.info("CUDA memory optimized")
    
    def get_optimal_batch_size(self, model: nn.Module, device: torch.device, 
                              input_shape: Tuple[int, ...] = (3, 224, 224)) -> int:
        """Determine optimal batch size for the model and device."""
        if device.type == 'cpu':
            return min(8, self.config.batch.max_batch_size)
        
        # For GPU, test different batch sizes
        optimal_batch_size = 1
        max_batch_size = self.config.batch.max_batch_size
        
        with torch.no_grad():
            for batch_size in [1, 2, 4, 8, 16, 32]:
                if batch_size > max_batch_size:
                    break
                
                try:
                    # Test memory usage
                    test_input = torch.randn(
                        batch_size, *input_shape,
                        device=device,
                        dtype=torch.float16 if self.config.device.use_fp16 else torch.float32
                    )
                    
                    # Try inference
                    _ = model(test_input)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                        # Check memory usage
                        memory_used = torch.cuda.memory_allocated(device) / 1024**3  # GB
                        memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
                        
                        if memory_used / memory_total < 0.8:  # Use less than 80% memory
                            optimal_batch_size = batch_size
                        else:
                            break
                    else:
                        optimal_batch_size = batch_size
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        break
                    else:
                        raise
                except Exception:
                    break
        
        logger.info(f"Optimal batch size determined: {optimal_batch_size}")
        return optimal_batch_size
    
    def warmup_model(self, model: nn.Module, device: torch.device, 
                    input_shape: Tuple[int, ...] = (3, 224, 224),
                    batch_sizes: Optional[list] = None) -> None:
        """Warmup model for stable performance."""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
        
        logger.info("Warming up model for optimal performance...")
        
        model.eval()
        with torch.no_grad():
            for batch_size in batch_sizes:
                if batch_size > self.config.batch.max_batch_size:
                    continue
                
                try:
                    # Create warmup input
                    warmup_input = torch.randn(
                        batch_size, *input_shape,
                        device=device,
                        dtype=torch.float16 if self.config.device.use_fp16 else torch.float32
                    )
                    
                    # Run multiple warmup iterations
                    for _ in range(5):
                        _ = model(warmup_input)
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                    
                    logger.debug(f"Warmup completed for batch size: {batch_size}")
                    
                except Exception as e:
                    logger.debug(f"Warmup failed for batch size {batch_size}: {e}")
        
        logger.info("Model warmup completed")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance optimization report."""
        return {
            "optimizations_applied": self.optimizations_applied,
            "device_config": {
                "device_type": self.config.device.device_type.value,
                "use_fp16": self.config.device.use_fp16,
                "use_torch_compile": self.config.device.use_torch_compile,
                "memory_fraction": self.config.device.memory_fraction
            },
            "recommendations": self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """Get performance recommendations."""
        recommendations = []
        
        if not self.config.device.use_fp16 and self.config.device.device_type in [DeviceType.CUDA, DeviceType.MPS]:
            recommendations.append("Enable FP16 for 2x speedup on modern GPUs")
        
        if not self.config.device.use_torch_compile:
            recommendations.append("Enable torch.compile for additional 20-50% speedup")
        
        if self.config.batch.batch_size < 4:
            recommendations.append("Consider larger batch sizes for better GPU utilization")
        
        if self.config.device.memory_fraction < 0.8:
            recommendations.append("Increase memory fraction for better performance")
        
        return recommendations


def optimize_for_inference(model: nn.Module, config: InferenceConfig) -> Tuple[nn.Module, DeviceConfig]:
    """
    Apply comprehensive optimizations for inference.
    
    Args:
        model: PyTorch model to optimize
        config: Inference configuration
    
    Returns:
        Tuple of (optimized_model, optimized_device_config)
    """
    optimizer = PerformanceOptimizer(config)
    
    # Optimize device configuration
    device_config = optimizer.optimize_device_config()
    device = device_config.get_torch_device()
    
    # Optimize model
    optimized_model = optimizer.optimize_model(model, device)
    
    # Optimize memory
    optimizer.optimize_memory(device)
    
    # Warmup model
    optimizer.warmup_model(optimized_model, device)
    
    # Log optimizations
    report = optimizer.get_performance_report()
    logger.info(f"Performance optimizations completed: {', '.join(report['optimizations_applied'])}")
    
    if report['recommendations']:
        logger.info(f"Performance recommendations: {', '.join(report['recommendations'])}")
    
    return optimized_model, device_config

"""
GPU Detection Integration for PyTorch Inference Framework

This module integrates the GPU detection system with the existing framework,
providing device configuration and optimization recommendations.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import torch

from .gpu_detection import (
    GPUDetector, GPUInfo, GPUVendor, AcceleratorType, 
    GPUArchitecture, detect_gpus, get_best_gpu
)
from .config import DeviceConfig, DeviceType, InferenceConfig


logger = logging.getLogger(__name__)


class GPUManager:
    """
    GPU management system that integrates detection with framework configuration.
    """
    
    def __init__(self):
        """Initialize GPU manager."""
        self.detector = GPUDetector(enable_benchmarks=True)
        self._gpus: Optional[List[GPUInfo]] = None
        self._best_gpu: Optional[GPUInfo] = None
        
    def detect_and_configure(self, force_refresh: bool = False) -> Tuple[List[GPUInfo], DeviceConfig]:
        """
        Detect GPUs and generate optimal device configuration.
        
        Args:
            force_refresh: Force refresh of GPU detection
            
        Returns:
            Tuple of (detected GPUs, recommended device config)
        """
        # Detect GPUs
        self._gpus = self.detector.detect_all_gpus(force_refresh=force_refresh)
        self._best_gpu = self.detector.get_best_gpu(self._gpus)
        
        # Generate device configuration
        device_config = self._generate_device_config()
        
        return self._gpus, device_config
    
    def _generate_device_config(self) -> DeviceConfig:
        """Generate optimal device configuration based on detected GPUs."""
        if not self._best_gpu or not self._best_gpu.is_suitable_for_inference():
            logger.info("No suitable GPU found, using CPU configuration")
            return DeviceConfig(
                device_type=DeviceType.CPU,
                use_fp16=False,
                use_int8=False,
                use_tensorrt=False,
                use_torch_compile=False
            )
        
        gpu = self._best_gpu
        logger.info(f"Configuring for GPU: {gpu.name} ({gpu.device_id})")
        
        # Determine device type
        if gpu.vendor == GPUVendor.NVIDIA and AcceleratorType.CUDA in gpu.supported_accelerators:
            device_type = DeviceType.CUDA
        elif gpu.vendor == GPUVendor.APPLE and AcceleratorType.MPS in gpu.supported_accelerators:
            device_type = DeviceType.MPS
        else:
            device_type = DeviceType.AUTO
        
        # Determine optimal precision settings
        recommended_precisions = gpu.get_recommended_precision()
        use_fp16 = "fp16" in recommended_precisions
        use_int8 = "int8" in recommended_precisions
        
        # TensorRT support
        use_tensorrt = gpu.tensorrt_support and gpu.vendor == GPUVendor.NVIDIA
        
        # Torch compile (be conservative)
        use_torch_compile = (
            gpu.vendor == GPUVendor.NVIDIA and 
            gpu.compute_capability and 
            gpu.compute_capability.major >= 7
        )
        
        device_config = DeviceConfig(
            device_type=device_type,
            device_id=gpu.id if gpu.vendor == GPUVendor.NVIDIA else None,
            use_fp16=use_fp16,
            use_int8=use_int8,
            use_tensorrt=use_tensorrt,
            use_torch_compile=use_torch_compile,
            compile_mode="reduce-overhead"
        )
        
        logger.info(f"Generated device config: {device_config}")
        return device_config
    
    def get_memory_recommendations(self) -> Dict[str, Any]:
        """Get memory optimization recommendations."""
        if not self._best_gpu:
            return {"recommendations": ["Use CPU-optimized batch sizes"]}
        
        gpu = self._best_gpu
        recommendations = []
        
        # Batch size recommendations
        if gpu.memory.total_mb < 4096:  # Less than 4GB
            recommendations.append("Use small batch sizes (1-4)")
            recommendations.append("Enable gradient checkpointing")
            recommendations.append("Consider model quantization")
        elif gpu.memory.total_mb < 8192:  # 4-8GB
            recommendations.append("Use moderate batch sizes (4-16)")
            recommendations.append("Enable mixed precision training")
        else:  # 8GB+
            recommendations.append("Can use larger batch sizes (16-64+)")
            recommendations.append("Enable tensor parallelism for very large models")
        
        # Memory optimization features
        if gpu.vendor == GPUVendor.NVIDIA:
            recommendations.append("Enable CUDA memory pool")
            recommendations.append("Use torch.cuda.empty_cache() periodically")
            
            if gpu.compute_capability and gpu.compute_capability.major >= 8:
                recommendations.append("Enable TensorFloat-32 (TF32) for Ampere+ GPUs")
        
        return {
            "total_memory_mb": gpu.memory.total_mb,
            "available_memory_mb": gpu.memory.available_mb,
            "estimated_max_batch_size": gpu.estimate_max_batch_size(),
            "recommendations": recommendations
        }
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations based on detected hardware."""
        if not self._best_gpu:
            return {"recommendations": ["CPU optimizations only"]}
        
        gpu = self._best_gpu
        recommendations = []
        
        # CUDA-specific optimizations
        if gpu.vendor == GPUVendor.NVIDIA:
            recommendations.append("Enable cuDNN benchmark mode")
            recommendations.append("Use CUDA graphs for repeated operations")
            
            if gpu.compute_capability:
                if gpu.compute_capability.supports_tensor_cores:
                    recommendations.append("Use Tensor Cores with FP16/BF16")
                
                if gpu.compute_capability.supports_tf32:
                    recommendations.append("Enable TensorFloat-32 (TF32)")
                
                if gpu.compute_capability.major >= 8:
                    recommendations.append("Consider using torch.compile")
        
        # Apple Silicon optimizations
        elif gpu.vendor == GPUVendor.APPLE:
            recommendations.append("Use Metal Performance Shaders (MPS)")
            recommendations.append("Optimize for unified memory architecture")
        
        # General optimizations
        if gpu.memory.total_mb >= 8192:
            recommendations.append("Enable large model optimizations")
            recommendations.append("Use model parallelism for very large models")
        
        return {
            "gpu_name": gpu.name,
            "vendor": gpu.vendor.value,
            "architecture": gpu.architecture.value,
            "recommendations": recommendations
        }
    
    def get_detected_gpus(self) -> List[GPUInfo]:
        """Get list of detected GPUs."""
        if self._gpus is None:
            self._gpus = self.detector.detect_all_gpus()
        return self._gpus
    
    def get_best_gpu_info(self) -> Optional[GPUInfo]:
        """Get information about the best detected GPU."""
        if self._best_gpu is None:
            self._best_gpu = self.detector.get_best_gpu()
        return self._best_gpu
    
    def generate_full_report(self) -> str:
        """Generate a comprehensive GPU report with recommendations."""
        if self._gpus is None:
            self._gpus = self.detector.detect_all_gpus()
        
        # Base detection report
        report = self.detector.generate_detection_report(self._gpus)
        
        # Add configuration recommendations
        _, device_config = self.detect_and_configure()
        memory_rec = self.get_memory_recommendations()
        optimization_rec = self.get_optimization_recommendations()
        
        report += "\n\nRECOMMENDATIONS:\n"
        report += "-" * 40 + "\n"
        
        report += f"\nDevice Configuration:\n"
        report += f"  Device Type: {device_config.device_type.value.upper()}\n"
        report += f"  FP16 Enabled: {device_config.use_fp16}\n"
        report += f"  INT8 Enabled: {device_config.use_int8}\n"
        report += f"  TensorRT Enabled: {device_config.use_tensorrt}\n"
        report += f"  Torch Compile Enabled: {device_config.use_torch_compile}\n"
        
        report += f"\nMemory Recommendations:\n"
        for rec in memory_rec.get("recommendations", []):
            report += f"  - {rec}\n"
        
        report += f"\nOptimization Recommendations:\n"
        for rec in optimization_rec.get("recommendations", []):
            report += f"  - {rec}\n"
        
        return report


def auto_configure_device() -> DeviceConfig:
    """
    Automatically configure device based on available hardware.
    
    Returns:
        Optimized device configuration
    """
    manager = GPUManager()
    _, device_config = manager.detect_and_configure()
    return device_config


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get GPU memory information and recommendations.
    
    Returns:
        Memory information and recommendations
    """
    manager = GPUManager()
    manager.detect_and_configure()
    return manager.get_memory_recommendations()


def print_gpu_configuration_report() -> None:
    """Print a comprehensive GPU configuration report."""
    manager = GPUManager()
    report = manager.generate_full_report()
    print(report)


# Global GPU manager instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get the global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager

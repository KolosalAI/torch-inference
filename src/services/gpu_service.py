"""
GPU detection and management service.
"""

import logging
import torch
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class GPUService:
    """Service for GPU detection and management."""
    
    def __init__(self):
        self.logger = logger
    
    def detect_gpus(self) -> Dict[str, Any]:
        """Detect available GPUs and their properties."""
        try:
            gpu_info = {
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpus": []
            }
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_info["gpus"].append({
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "total_memory_gb": gpu_props.total_memory / (1024**3),
                        "allocated_memory_gb": torch.cuda.memory_allocated(i) / (1024**3),
                        "reserved_memory_gb": torch.cuda.memory_reserved(i) / (1024**3),
                        "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                        "multiprocessor_count": gpu_props.multi_processor_count,
                    })
            
            # Check for Apple MPS
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                gpu_info["mps_available"] = True
                gpu_info["mps_device"] = {
                    "name": "Apple Metal Performance Shaders",
                    "type": "mps"
                }
            else:
                gpu_info["mps_available"] = False
            
            return gpu_info
            
        except Exception as e:
            self.logger.error(f"GPU detection failed: {e}")
            return {"error": str(e), "cuda_available": False, "gpu_count": 0}
    
    def get_best_gpu(self) -> Dict[str, Any]:
        """Get the best GPU for inference based on memory and compute capability."""
        try:
            gpu_info = self.detect_gpus()
            
            if not gpu_info.get("cuda_available", False):
                if gpu_info.get("mps_available", False):
                    return {
                        "best_gpu": gpu_info["mps_device"],
                        "reason": "MPS device available (Apple Silicon)",
                        "device_type": "mps"
                    }
                else:
                    return {
                        "best_gpu": {"name": "CPU", "type": "cpu"},
                        "reason": "No GPU acceleration available",
                        "device_type": "cpu"
                    }
            
            gpus = gpu_info.get("gpus", [])
            if not gpus:
                return {"error": "No GPUs found"}
            
            # Score GPUs based on memory and compute capability
            best_gpu = None
            best_score = 0
            
            for gpu in gpus:
                # Score based on free memory and compute capability
                free_memory = gpu["total_memory_gb"] - gpu["allocated_memory_gb"]
                compute_score = float(gpu["compute_capability"])
                
                # Weighted score: 70% memory, 30% compute capability
                score = (free_memory * 0.7) + (compute_score * 0.3)
                
                if score > best_score:
                    best_score = score
                    best_gpu = gpu
            
            return {
                "best_gpu": best_gpu,
                "score": best_score,
                "reason": "Selected based on available memory and compute capability",
                "device_type": "cuda"
            }
            
        except Exception as e:
            self.logger.error(f"Best GPU selection failed: {e}")
            return {"error": str(e)}
    
    def get_optimized_config(self) -> Dict[str, Any]:
        """Get GPU-optimized configuration recommendations."""
        try:
            best_gpu = self.get_best_gpu()
            
            if "error" in best_gpu:
                return best_gpu
            
            device_type = best_gpu.get("device_type", "cpu")
            config = {
                "device_type": device_type,
                "recommendations": {}
            }
            
            if device_type == "cuda":
                gpu = best_gpu["best_gpu"]
                memory_gb = gpu["total_memory_gb"]
                compute_capability = float(gpu["compute_capability"])
                
                # Memory-based recommendations
                if memory_gb >= 24:
                    config["recommendations"].update({
                        "batch_size": 16,
                        "use_fp16": True,
                        "enable_tensorrt": True,
                        "memory_optimization": "high_memory"
                    })
                elif memory_gb >= 12:
                    config["recommendations"].update({
                        "batch_size": 8,
                        "use_fp16": True,
                        "enable_tensorrt": True,
                        "memory_optimization": "medium_memory"
                    })
                elif memory_gb >= 6:
                    config["recommendations"].update({
                        "batch_size": 4,
                        "use_fp16": True,
                        "enable_tensorrt": False,
                        "memory_optimization": "low_memory"
                    })
                else:
                    config["recommendations"].update({
                        "batch_size": 2,
                        "use_fp16": True,
                        "enable_tensorrt": False,
                        "memory_optimization": "very_low_memory"
                    })
                
                # Compute capability based recommendations
                if compute_capability >= 8.0:
                    config["recommendations"]["tensor_cores"] = "ampere"
                elif compute_capability >= 7.0:
                    config["recommendations"]["tensor_cores"] = "turing"
                elif compute_capability >= 6.0:
                    config["recommendations"]["tensor_cores"] = "pascal"
                else:
                    config["recommendations"]["tensor_cores"] = "none"
                
                config["recommendations"]["device_id"] = gpu["id"]
                
            elif device_type == "mps":
                config["recommendations"].update({
                    "batch_size": 4,
                    "use_fp16": False,  # MPS has specific FP16 requirements
                    "enable_tensorrt": False,
                    "memory_optimization": "mps_optimized"
                })
            
            else:  # CPU
                config["recommendations"].update({
                    "batch_size": 1,
                    "use_fp16": False,
                    "enable_tensorrt": False,
                    "threads": torch.get_num_threads(),
                    "memory_optimization": "cpu_optimized"
                })
            
            return config
            
        except Exception as e:
            self.logger.error(f"GPU configuration generation failed: {e}")
            return {"error": str(e)}
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive GPU report."""
        try:
            gpu_info = self.detect_gpus()
            best_gpu = self.get_best_gpu()
            config = self.get_optimized_config()
            
            # System information
            system_info = {
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            }
            
            # Performance recommendations
            recommendations = []
            
            if gpu_info.get("cuda_available"):
                recommendations.append("CUDA acceleration available - use GPU for training and inference")
                if config.get("recommendations", {}).get("use_fp16"):
                    recommendations.append("FP16/Mixed precision recommended for this GPU")
                if config.get("recommendations", {}).get("enable_tensorrt"):
                    recommendations.append("TensorRT optimization available for improved inference speed")
            elif gpu_info.get("mps_available"):
                recommendations.append("Apple MPS acceleration available - use MPS for Apple Silicon")
            else:
                recommendations.append("No GPU acceleration available - consider upgrading hardware")
                recommendations.append("Optimize for CPU inference with appropriate threading")
            
            return {
                "system_info": system_info,
                "gpu_detection": gpu_info,
                "best_gpu": best_gpu,
                "optimized_config": config,
                "recommendations": recommendations,
                "summary": {
                    "acceleration_available": gpu_info.get("cuda_available") or gpu_info.get("mps_available"),
                    "recommended_device": best_gpu.get("device_type", "cpu"),
                    "performance_tier": self._get_performance_tier(best_gpu)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive GPU report failed: {e}")
            return {"error": str(e)}
    
    def _get_performance_tier(self, best_gpu: Dict[str, Any]) -> str:
        """Determine performance tier based on GPU capabilities."""
        try:
            device_type = best_gpu.get("device_type", "cpu")
            
            if device_type == "cpu":
                return "Basic"
            elif device_type == "mps":
                return "Good"
            elif device_type == "cuda":
                gpu = best_gpu.get("best_gpu", {})
                memory_gb = gpu.get("total_memory_gb", 0)
                compute_capability = float(gpu.get("compute_capability", "0.0"))
                
                if memory_gb >= 24 and compute_capability >= 8.0:
                    return "Excellent"
                elif memory_gb >= 12 and compute_capability >= 7.0:
                    return "Very Good"
                elif memory_gb >= 6 and compute_capability >= 6.0:
                    return "Good"
                else:
                    return "Fair"
            
            return "Unknown"
            
        except Exception:
            return "Unknown"

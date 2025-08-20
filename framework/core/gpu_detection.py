"""
GPU Detection System for PyTorch Inference Framework

This module provides comprehensive GPU detection and capability analysis,
including CUDA, ROCm, and other accelerator support with detailed hardware
information gathering and performance benchmarking.
"""

import os
import sys
import logging
import platform
import subprocess
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

import torch
import torch.nn as nn

try:
    import psutil
except ImportError:
    psutil = None

try:
    import GPUtil
except ImportError:
    GPUtil = None

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except (ImportError, Exception):
    pynvml = None
    NVML_AVAILABLE = False


logger = logging.getLogger(__name__)


class GPUVendor(Enum):
    """Supported GPU vendors."""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"


class AcceleratorType(Enum):
    """Types of accelerators."""
    CUDA = "cuda"
    ROCM = "rocm"
    OPENCL = "opencl"
    METAL = "metal"
    MPS = "mps"
    VULKAN = "vulkan"
    DIRECTML = "directml"
    CPU = "cpu"


class GPUArchitecture(Enum):
    """Known GPU architectures."""
    # NVIDIA
    KEPLER = "kepler"
    MAXWELL = "maxwell"
    PASCAL = "pascal"
    VOLTA = "volta"
    TURING = "turing"
    AMPERE = "ampere"
    ADA_LOVELACE = "ada_lovelace"
    HOPPER = "hopper"
    
    # AMD
    GCN = "gcn"
    RDNA = "rdna"
    RDNA2 = "rdna2"
    RDNA3 = "rdna3"
    CDNA = "cdna"
    CDNA2 = "cdna2"
    
    # Intel
    GEN9 = "gen9"
    GEN12 = "gen12"
    XE_HPG = "xe_hpg"
    XE_HPC = "xe_hpc"
    
    # Apple
    M1 = "m1"
    M2 = "m2"
    M3 = "m3"
    
    UNKNOWN = "unknown"


@dataclass
class MemoryInfo:
    """GPU memory information."""
    total_mb: float = 0.0
    available_mb: float = 0.0
    used_mb: float = 0.0
    free_mb: float = 0.0
    utilization_percent: float = 0.0
    bandwidth_gb_s: Optional[float] = None
    memory_clock_mhz: Optional[int] = None
    memory_type: Optional[str] = None
    bus_width: Optional[int] = None


@dataclass
class ComputeCapability:
    """CUDA compute capability information."""
    major: int = 0
    minor: int = 0
    
    @property
    def version(self) -> str:
        """Get version string."""
        return f"{self.major}.{self.minor}"
    
    @property
    def supports_fp16(self) -> bool:
        """Check if FP16 is supported."""
        return (self.major > 5) or (self.major == 5 and self.minor >= 3)
    
    @property
    def supports_int8(self) -> bool:
        """Check if INT8 is supported."""
        return (self.major > 6) or (self.major == 6 and self.minor >= 1)
    
    @property
    def supports_tensor_cores(self) -> bool:
        """Check if Tensor Cores are supported."""
        return self.major >= 7  # Volta and newer
    
    @property
    def supports_tf32(self) -> bool:
        """Check if TensorFloat-32 is supported."""
        return self.major >= 8  # Ampere and newer


@dataclass
class PerformanceMetrics:
    """GPU performance metrics."""
    gpu_utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    temperature_celsius: Optional[float] = None
    power_draw_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None
    fan_speed_percent: Optional[float] = None
    core_clock_mhz: Optional[int] = None
    memory_clock_mhz: Optional[int] = None
    
    # Computed metrics
    flops_fp32: Optional[float] = None
    flops_fp16: Optional[float] = None
    flops_int8: Optional[float] = None
    memory_bandwidth_gb_s: Optional[float] = None


@dataclass
class GPUInfo:
    """Comprehensive GPU information."""
    # Basic information
    id: int = 0
    name: str = "Unknown GPU"
    vendor: GPUVendor = GPUVendor.UNKNOWN
    architecture: GPUArchitecture = GPUArchitecture.UNKNOWN
    
    # Device identifiers
    device_id: Optional[str] = None
    pci_bus_id: Optional[str] = None
    uuid: Optional[str] = None
    
    # Capabilities
    compute_capability: Optional[ComputeCapability] = None
    supported_accelerators: List[AcceleratorType] = field(default_factory=list)
    
    # Hardware specs
    multiprocessor_count: Optional[int] = None
    cores_per_multiprocessor: Optional[int] = None
    total_cores: Optional[int] = None
    base_clock_mhz: Optional[int] = None
    boost_clock_mhz: Optional[int] = None
    
    # Memory
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    
    # Performance
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    # Software support
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    pytorch_support: bool = False
    tensorrt_support: bool = False
    
    # Additional capabilities
    supports_ecc: bool = False
    supports_nvlink: bool = False
    supports_mig: bool = False
    supports_ray_tracing: bool = False
    
    # Benchmark results
    benchmark_results: Dict[str, Any] = field(default_factory=dict)
    
    def is_suitable_for_inference(self) -> bool:
        """Check if GPU is suitable for inference."""
        if self.memory.total_mb < 1000:  # Less than 1GB
            return False
        
        if self.vendor == GPUVendor.NVIDIA:
            if self.compute_capability and self.compute_capability.major < 3:
                return False
                
        return self.pytorch_support
    
    def get_recommended_precision(self) -> List[str]:
        """Get recommended precision formats."""
        precisions = ["fp32"]
        
        if self.compute_capability:
            if self.compute_capability.supports_fp16:
                precisions.append("fp16")
            if self.compute_capability.supports_int8:
                precisions.append("int8")
            if self.compute_capability.supports_tf32:
                precisions.append("tf32")
        
        return precisions
    
    def estimate_max_batch_size(self, model_size_mb: float = 500) -> int:
        """Estimate maximum batch size for a given model size."""
        available_memory = self.memory.available_mb * 0.8  # Leave 20% headroom
        memory_per_sample = model_size_mb * 0.1  # Rough estimate
        
        if memory_per_sample <= 0:
            return 1
            
        return max(1, int(available_memory / memory_per_sample))


class GPUDetector:
    """Comprehensive GPU detection and analysis system."""
    
    def __init__(self, enable_benchmarks: bool = True, 
                 benchmark_duration: float = 5.0):
        """
        Initialize GPU detector.
        
        Args:
            enable_benchmarks: Whether to run performance benchmarks
            benchmark_duration: Duration for benchmarks in seconds
        """
        self.enable_benchmarks = enable_benchmarks
        self.benchmark_duration = benchmark_duration
        self.logger = logging.getLogger(f"{__name__}.GPUDetector")
        
        # Cache for detection results
        self._detection_cache: Optional[Dict[str, Any]] = None
        self._last_detection_time: float = 0
        self._cache_ttl: float = 300.0  # 5 minutes
    
    def detect_all_gpus(self, force_refresh: bool = False) -> List[GPUInfo]:
        """
        Detect all available GPUs with comprehensive information.
        
        Args:
            force_refresh: Force refresh of cached results
            
        Returns:
            List of GPU information objects
        """
        current_time = time.time()
        
        # Use cache if available and not expired
        if (not force_refresh and 
            self._detection_cache and 
            (current_time - self._last_detection_time) < self._cache_ttl):
            return self._detection_cache.get("gpus", [])
        
        self.logger.info("Starting comprehensive GPU detection...")
        
        gpus = []
        
        # CUDA GPUs
        cuda_gpus = self._detect_cuda_gpus()
        gpus.extend(cuda_gpus)
        
        # ROCm GPUs (AMD)
        rocm_gpus = self._detect_rocm_gpus()
        gpus.extend(rocm_gpus)
        
        # Apple Silicon (MPS)
        mps_gpus = self._detect_mps_gpus()
        gpus.extend(mps_gpus)
        
        # Intel GPUs
        intel_gpus = self._detect_intel_gpus()
        gpus.extend(intel_gpus)
        
        # Run benchmarks if enabled
        if self.enable_benchmarks:
            self.logger.info("Running GPU benchmarks...")
            for gpu in gpus:
                if gpu.pytorch_support:
                    gpu.benchmark_results = self._benchmark_gpu(gpu)
        
        # Update cache
        self._detection_cache = {
            "gpus": gpus,
            "detection_time": current_time,
            "system_info": self._get_system_info()
        }
        self._last_detection_time = current_time
        
        self.logger.info(f"GPU detection completed. Found {len(gpus)} GPU(s)")
        
        return gpus
    
    def _detect_cuda_gpus(self) -> List[GPUInfo]:
        """Detect NVIDIA CUDA GPUs."""
        gpus = []
        
        if not torch.cuda.is_available():
            self.logger.info("CUDA not available")
            return gpus
        
        device_count = torch.cuda.device_count()
        self.logger.info(f"Found {device_count} CUDA device(s)")
        
        for i in range(device_count):
            try:
                gpu_info = self._get_cuda_gpu_info(i)
                gpus.append(gpu_info)
            except Exception as e:
                self.logger.error(f"Error detecting CUDA GPU {i}: {e}")
        
        return gpus
    
    def _get_cuda_gpu_info(self, device_id: int) -> GPUInfo:
        """Get detailed information for a CUDA GPU."""
        device = torch.device(f"cuda:{device_id}")
        props = torch.cuda.get_device_properties(device)
        
        gpu_info = GPUInfo(
            id=device_id,
            name=props.name,
            vendor=GPUVendor.NVIDIA,
            device_id=f"cuda:{device_id}",
            multiprocessor_count=props.multi_processor_count,
            pytorch_support=True
        )
        
        # Compute capability
        gpu_info.compute_capability = ComputeCapability(
            major=props.major,
            minor=props.minor
        )
        
        # Memory information
        gpu_info.memory.total_mb = props.total_memory / (1024 ** 2)
        
        # Get current memory usage
        try:
            gpu_info.memory.available_mb = (
                props.total_memory - torch.cuda.memory_allocated(device)
            ) / (1024 ** 2)
            gpu_info.memory.used_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            gpu_info.memory.free_mb = gpu_info.memory.available_mb
        except Exception as e:
            self.logger.warning(f"Could not get memory usage for GPU {device_id}: {e}")
        
        # Architecture detection
        gpu_info.architecture = self._detect_nvidia_architecture(
            gpu_info.compute_capability, props.name
        )
        
        # Supported accelerators
        gpu_info.supported_accelerators = [AcceleratorType.CUDA]
        
        # Driver and CUDA version
        try:
            gpu_info.driver_version = torch.version.cuda
            gpu_info.cuda_version = torch.version.cuda
        except Exception:
            pass
        
        # NVML information if available
        if NVML_AVAILABLE:
            try:
                self._add_nvml_info(gpu_info, device_id)
            except Exception as e:
                self.logger.warning(f"Could not get NVML info for GPU {device_id}: {e}")
        
        # Additional capabilities
        gpu_info.supports_ecc = hasattr(props, 'ecc_enabled') and props.ecc_enabled
        
        # TensorRT support detection
        try:
            import tensorrt
            gpu_info.tensorrt_support = True
        except ImportError:
            gpu_info.tensorrt_support = False
        
        return gpu_info
    
    def _add_nvml_info(self, gpu_info: GPUInfo, device_id: int) -> None:
        """Add NVML information to GPU info."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            
            # UUID
            try:
                gpu_info.uuid = pynvml.nvmlDeviceGetUUID(handle).decode('utf-8')
            except Exception:
                pass
            
            # PCI info
            try:
                pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                gpu_info.pci_bus_id = pci_info.busId.decode('utf-8')
            except Exception:
                pass
            
            # Performance metrics
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_info.performance.gpu_utilization_percent = utilization.gpu
                gpu_info.performance.memory_utilization_percent = utilization.memory
            except Exception:
                pass
            
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_info.performance.temperature_celsius = temp
            except Exception:
                pass
            
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                gpu_info.performance.power_draw_watts = power
            except Exception:
                pass
            
            try:
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                gpu_info.performance.power_limit_watts = power_limit
            except Exception:
                pass
            
            # Clock speeds
            try:
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                gpu_info.performance.core_clock_mhz = graphics_clock
            except Exception:
                pass
            
            try:
                memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                gpu_info.performance.memory_clock_mhz = memory_clock
                gpu_info.memory.memory_clock_mhz = memory_clock
            except Exception:
                pass
            
            # Driver version
            try:
                driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                gpu_info.driver_version = driver_version
            except Exception:
                pass
            
        except Exception as e:
            self.logger.warning(f"NVML error for GPU {device_id}: {e}")
    
    def _detect_nvidia_architecture(self, compute_cap: ComputeCapability, 
                                   name: str) -> GPUArchitecture:
        """Detect NVIDIA GPU architecture."""
        major, minor = compute_cap.major, compute_cap.minor
        
        # Handle case where name might be a Mock object or None
        if name is None:
            name = ""
        try:
            name_lower = str(name).lower()
        except (AttributeError, TypeError):
            name_lower = ""
        
        # Architecture detection based on compute capability and name
        if major >= 9:
            return GPUArchitecture.HOPPER
        elif major == 8:
            if "a100" in name_lower or "a40" in name_lower:
                return GPUArchitecture.AMPERE
            elif "rtx 40" in name_lower or "ada" in name_lower:
                return GPUArchitecture.ADA_LOVELACE
            return GPUArchitecture.AMPERE
        elif major == 7:
            if "rtx" in name_lower and ("20" in name_lower or "16" in name_lower):
                return GPUArchitecture.TURING
            return GPUArchitecture.VOLTA
        elif major == 6:
            return GPUArchitecture.PASCAL
        elif major == 5:
            return GPUArchitecture.MAXWELL
        elif major == 3:
            return GPUArchitecture.KEPLER
        
        return GPUArchitecture.UNKNOWN
    
    def _detect_rocm_gpus(self) -> List[GPUInfo]:
        """Detect AMD ROCm GPUs."""
        gpus = []
        
        try:
            # Check if ROCm is available
            if hasattr(torch.version, 'hip') and torch.version.hip:
                # ROCm detected
                self.logger.info("ROCm support detected")
                # Add ROCm GPU detection logic here
        except Exception as e:
            self.logger.debug(f"ROCm detection failed: {e}")
        
        return gpus
    
    def _detect_mps_gpus(self) -> List[GPUInfo]:
        """Detect Apple Silicon GPUs (MPS)."""
        gpus = []
        
        try:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.logger.info("Apple Metal Performance Shaders (MPS) detected")
                
                gpu_info = GPUInfo(
                    id=0,
                    name="Apple Silicon GPU",
                    vendor=GPUVendor.APPLE,
                    device_id="mps:0",
                    pytorch_support=True
                )
                
                # Detect architecture
                if platform.machine() == "arm64":
                    system_name = platform.processor()
                    if "m3" in system_name.lower():
                        gpu_info.architecture = GPUArchitecture.M3
                    elif "m2" in system_name.lower():
                        gpu_info.architecture = GPUArchitecture.M2
                    elif "m1" in system_name.lower():
                        gpu_info.architecture = GPUArchitecture.M1
                
                gpu_info.supported_accelerators = [AcceleratorType.MPS, AcceleratorType.METAL]
                gpus.append(gpu_info)
                
        except Exception as e:
            self.logger.debug(f"MPS detection failed: {e}")
        
        return gpus
    
    def _detect_intel_gpus(self) -> List[GPUInfo]:
        """Detect Intel GPUs."""
        gpus = []
        
        try:
            # Check for Intel GPU support (XPU)
            if hasattr(torch, 'xpu') and hasattr(torch.xpu, 'is_available'):
                if torch.xpu.is_available():
                    device_count = torch.xpu.device_count()
                    self.logger.info(f"Found {device_count} Intel XPU device(s)")
                    
                    for i in range(device_count):
                        gpu_info = GPUInfo(
                            id=i,
                            name=f"Intel GPU {i}",
                            vendor=GPUVendor.INTEL,
                            device_id=f"xpu:{i}",
                            pytorch_support=True
                        )
                        gpu_info.supported_accelerators = [AcceleratorType.OPENCL]
                        gpus.append(gpu_info)
                        
        except Exception as e:
            self.logger.debug(f"Intel GPU detection failed: {e}")
        
        return gpus
    
    def _benchmark_gpu(self, gpu_info: GPUInfo) -> Dict[str, Any]:
        """Run performance benchmarks on a GPU."""
        if not gpu_info.pytorch_support:
            return {}
        
        results = {}
        
        try:
            device = torch.device(gpu_info.device_id)
            
            # Memory bandwidth benchmark
            results["memory_bandwidth"] = self._benchmark_memory_bandwidth(device)
            
            # Compute benchmarks
            results["fp32_performance"] = self._benchmark_compute(device, torch.float32)
            
            if gpu_info.compute_capability and gpu_info.compute_capability.supports_fp16:
                results["fp16_performance"] = self._benchmark_compute(device, torch.float16)
            
            # Matrix multiplication benchmark
            results["matmul_performance"] = self._benchmark_matmul(device)
            
        except Exception as e:
            self.logger.error(f"Benchmark failed for {gpu_info.device_id}: {e}")
            results["error"] = str(e)
        
        return results
    
    def _benchmark_memory_bandwidth(self, device: torch.device) -> Dict[str, float]:
        """Benchmark memory bandwidth."""
        size = 100 * 1024 * 1024  # 100MB
        
        # Allocate tensors
        tensor_a = torch.randn(size // 4, device=device, dtype=torch.float32)
        tensor_b = torch.empty_like(tensor_a)
        
        # Warmup
        for _ in range(10):
            tensor_b.copy_(tensor_a)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        
        # Benchmark
        start_time = time.time()
        iterations = 100
        
        for _ in range(iterations):
            tensor_b.copy_(tensor_a)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.time()
        
        elapsed = end_time - start_time
        bytes_transferred = size * iterations * 2  # Read + Write
        bandwidth_gb_s = (bytes_transferred / (1024 ** 3)) / elapsed
        
        return {
            "bandwidth_gb_s": bandwidth_gb_s,
            "elapsed_time_s": elapsed,
            "iterations": iterations
        }
    
    def _benchmark_compute(self, device: torch.device, dtype: torch.dtype) -> Dict[str, float]:
        """Benchmark compute performance."""
        size = (2048, 2048)
        
        tensor_a = torch.randn(size, device=device, dtype=dtype)
        tensor_b = torch.randn(size, device=device, dtype=dtype)
        
        # Warmup
        for _ in range(10):
            _ = tensor_a + tensor_b
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        
        # Benchmark
        start_time = time.time()
        iterations = 100
        
        for _ in range(iterations):
            result = tensor_a + tensor_b
            result = torch.sin(result)
            result = torch.exp(result)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.time()
        
        elapsed = end_time - start_time
        ops_per_second = (iterations * size[0] * size[1] * 3) / elapsed  # 3 operations per iteration
        
        return {
            "ops_per_second": ops_per_second,
            "elapsed_time_s": elapsed,
            "dtype": str(dtype),
            "tensor_size": size
        }
    
    def _benchmark_matmul(self, device: torch.device) -> Dict[str, float]:
        """Benchmark matrix multiplication performance."""
        sizes = [(512, 512), (1024, 1024), (2048, 2048)]
        results = {}
        
        for size in sizes:
            try:
                tensor_a = torch.randn(size, device=device)
                tensor_b = torch.randn(size, device=device)
                
                # Warmup
                for _ in range(10):
                    _ = torch.matmul(tensor_a, tensor_b)
                
                torch.cuda.synchronize() if device.type == "cuda" else None
                
                # Benchmark
                start_time = time.time()
                iterations = 50
                
                for _ in range(iterations):
                    _ = torch.matmul(tensor_a, tensor_b)
                
                torch.cuda.synchronize() if device.type == "cuda" else None
                end_time = time.time()
                
                elapsed = end_time - start_time
                flops = iterations * 2 * size[0] * size[1] * size[1]  # 2 * N^3 for matrix multiply
                gflops = (flops / (1e9)) / elapsed
                
                results[f"matmul_{size[0]}x{size[1]}"] = {
                    "gflops": gflops,
                    "elapsed_time_s": elapsed,
                    "iterations": iterations
                }
                
            except Exception as e:
                results[f"matmul_{size[0]}x{size[1]}"] = {"error": str(e)}
        
        return results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        }
        
        if psutil:
            system_info.update({
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024 ** 3),
                "cpu_brand": platform.processor()
            })
        
        return system_info
    
    def get_best_gpu(self, gpus: Optional[List[GPUInfo]] = None) -> Optional[GPUInfo]:
        """
        Get the best GPU for inference.
        
        Args:
            gpus: List of GPUs to choose from (default: detect all)
            
        Returns:
            Best GPU or None if no suitable GPU found
        """
        if gpus is None:
            gpus = self.detect_all_gpus()
        
        if not gpus:
            return None
        
        # Filter suitable GPUs
        suitable_gpus = [gpu for gpu in gpus if gpu.is_suitable_for_inference()]
        
        if not suitable_gpus:
            return None
        
        # Score GPUs based on multiple factors
        def score_gpu(gpu: GPUInfo) -> float:
            score = 0.0
            
            # Memory score (40% weight)
            score += (gpu.memory.total_mb / 16384) * 0.4  # Normalize to 16GB
            
            # Compute capability score (30% weight)
            if gpu.compute_capability:
                compute_score = gpu.compute_capability.major + (gpu.compute_capability.minor * 0.1)
                score += (compute_score / 10.0) * 0.3
            
            # Architecture bonus (20% weight)
            arch_scores = {
                GPUArchitecture.HOPPER: 1.0,
                GPUArchitecture.ADA_LOVELACE: 0.95,
                GPUArchitecture.AMPERE: 0.9,
                GPUArchitecture.TURING: 0.8,
                GPUArchitecture.VOLTA: 0.75,
                GPUArchitecture.PASCAL: 0.6,
                GPUArchitecture.M3: 0.7,
                GPUArchitecture.M2: 0.65,
                GPUArchitecture.M1: 0.6,
            }
            score += arch_scores.get(gpu.architecture, 0.3) * 0.2
            
            # Performance bonus (10% weight)
            if gpu.benchmark_results:
                if "fp32_performance" in gpu.benchmark_results:
                    perf = gpu.benchmark_results["fp32_performance"].get("ops_per_second", 0)
                    score += min(perf / 1e12, 1.0) * 0.1  # Normalize to 1T ops/sec
            
            return score
        
        # Find GPU with highest score
        best_gpu = max(suitable_gpus, key=score_gpu)
        
        return best_gpu
    
    def generate_detection_report(self, gpus: Optional[List[GPUInfo]] = None) -> str:
        """
        Generate a comprehensive detection report.
        
        Args:
            gpus: List of GPUs (default: detect all)
            
        Returns:
            Formatted report string
        """
        if gpus is None:
            gpus = self.detect_all_gpus()
        
        report = []
        report.append("=" * 80)
        report.append("  GPU DETECTION REPORT")
        report.append("=" * 80)
        
        if not gpus:
            report.append("No GPUs detected.")
            return "\n".join(report)
        
        # Summary
        report.append(f"\nSUMMARY:")
        report.append(f"  Total GPUs detected: {len(gpus)}")
        suitable_count = sum(1 for gpu in gpus if gpu.is_suitable_for_inference())
        report.append(f"  Suitable for inference: {suitable_count}")
        
        # Best GPU
        best_gpu = self.get_best_gpu(gpus)
        if best_gpu:
            report.append(f"  Recommended GPU: {best_gpu.name} (ID: {best_gpu.id})")
        
        # Detailed information
        report.append(f"\nDETAILED INFORMATION:")
        report.append("-" * 40)
        
        for i, gpu in enumerate(gpus):
            report.append(f"\nGPU {gpu.id}: {gpu.name}")
            report.append(f"  Vendor: {gpu.vendor.value.title()}")
            report.append(f"  Architecture: {gpu.architecture.value.title()}")
            report.append(f"  Device ID: {gpu.device_id}")
            
            if gpu.compute_capability:
                report.append(f"  Compute Capability: {gpu.compute_capability.version}")
                report.append(f"    - FP16 Support: {gpu.compute_capability.supports_fp16}")
                report.append(f"    - INT8 Support: {gpu.compute_capability.supports_int8}")
                report.append(f"    - Tensor Cores: {gpu.compute_capability.supports_tensor_cores}")
            
            report.append(f"  Memory: {gpu.memory.total_mb:.0f} MB total")
            if gpu.memory.available_mb > 0:
                report.append(f"    - Available: {gpu.memory.available_mb:.0f} MB")
                report.append(f"    - Used: {gpu.memory.used_mb:.0f} MB")
            
            if gpu.multiprocessor_count:
                report.append(f"  Multiprocessors: {gpu.multiprocessor_count}")
            
            report.append(f"  PyTorch Support: {gpu.pytorch_support}")
            report.append(f"  TensorRT Support: {gpu.tensorrt_support}")
            
            supported_acc = ", ".join([acc.value.upper() for acc in gpu.supported_accelerators])
            report.append(f"  Supported Accelerators: {supported_acc}")
            
            recommended_precisions = gpu.get_recommended_precision()
            report.append(f"  Recommended Precisions: {', '.join(recommended_precisions).upper()}")
            
            if gpu.driver_version:
                report.append(f"  Driver Version: {gpu.driver_version}")
            
            # Performance metrics
            if gpu.performance.temperature_celsius:
                report.append(f"  Temperature: {gpu.performance.temperature_celsius:.1f}°C")
            
            if gpu.performance.gpu_utilization_percent > 0:
                report.append(f"  GPU Utilization: {gpu.performance.gpu_utilization_percent:.1f}%")
            
            # Benchmark results
            if gpu.benchmark_results:
                report.append(f"  Benchmark Results:")
                if "memory_bandwidth" in gpu.benchmark_results:
                    bw = gpu.benchmark_results["memory_bandwidth"]["bandwidth_gb_s"]
                    report.append(f"    - Memory Bandwidth: {bw:.1f} GB/s")
                
                if "matmul_performance" in gpu.benchmark_results:
                    matmul_results = gpu.benchmark_results["matmul_performance"]
                    for key, result in matmul_results.items():
                        if isinstance(result, dict) and "gflops" in result:
                            report.append(f"    - {key}: {result['gflops']:.1f} GFLOPS")
            
            report.append(f"  Suitable for Inference: {gpu.is_suitable_for_inference()}")
            
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def detect_gpus(enable_benchmarks: bool = True) -> List[GPUInfo]:
    """
    Convenience function to detect all GPUs.
    
    Args:
        enable_benchmarks: Whether to run performance benchmarks
        
    Returns:
        List of detected GPUs
    """
    detector = GPUDetector(enable_benchmarks=enable_benchmarks)
    return detector.detect_all_gpus()


def get_best_gpu(enable_benchmarks: bool = True) -> Optional[GPUInfo]:
    """
    Convenience function to get the best available GPU.
    
    Args:
        enable_benchmarks: Whether to run performance benchmarks
        
    Returns:
        Best GPU or None if no suitable GPU found
    """
    detector = GPUDetector(enable_benchmarks=enable_benchmarks)
    return detector.get_best_gpu()


def print_gpu_report(enable_benchmarks: bool = True) -> None:
    """
    Print a comprehensive GPU detection report.
    
    Args:
        enable_benchmarks: Whether to run performance benchmarks
    """
    detector = GPUDetector(enable_benchmarks=enable_benchmarks)
    report = detector.generate_detection_report()
    print(report)


if __name__ == "__main__":
    # Example usage
    print_gpu_report(enable_benchmarks=True)

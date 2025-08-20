"""
Tests for GPU Detection System
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch

from framework.core.gpu_detection import (
    GPUDetector, GPUInfo, GPUVendor, AcceleratorType, 
    GPUArchitecture, ComputeCapability, MemoryInfo
)
from framework.core.gpu_manager import GPUManager, auto_configure_device
from framework.core.config import DeviceType


class TestGPUDetection:
    """Test GPU detection functionality."""
    
    @pytest.fixture
    def mock_cuda_available(self):
        """Mock CUDA availability."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            # Mock device properties
            mock_device = Mock()
            mock_device.name = "GeForce RTX 3080"
            mock_device.major = 8
            mock_device.minor = 6
            mock_device.total_memory = 10737418240  # 10GB
            mock_device.multi_processor_count = 68
            mock_device.ecc_enabled = False
            
            mock_props.return_value = mock_device
            yield mock_props
    
    @pytest.fixture
    def mock_no_cuda(self):
        """Mock no CUDA availability."""
        with patch('torch.cuda.is_available', return_value=False):
            yield
    
    @pytest.fixture
    def mock_mps_available(self):
        """Mock MPS (Apple Silicon) availability."""
        with patch('torch.backends.mps.is_available', return_value=True), \
             patch('platform.machine', return_value='arm64'), \
             patch('platform.processor', return_value='Apple M2'):
            yield
    
    def test_gpu_detector_initialization(self):
        """Test GPU detector initialization."""
        detector = GPUDetector(enable_benchmarks=False)
        assert detector is not None
        assert detector.enable_benchmarks is False
    
    def test_cuda_gpu_detection(self, mock_cuda_available):
        """Test CUDA GPU detection."""
        detector = GPUDetector(enable_benchmarks=False)
        
        with patch('torch.cuda.memory_allocated', return_value=1024*1024*1024), \
             patch('torch.cuda.memory_reserved', return_value=2*1024*1024*1024):
            
            gpus = detector._detect_cuda_gpus()
            
            assert len(gpus) == 1
            gpu = gpus[0]
            
            assert gpu.vendor == GPUVendor.NVIDIA
            assert gpu.name == "GeForce RTX 3080"
            assert gpu.compute_capability.major == 8
            assert gpu.compute_capability.minor == 6
            assert gpu.pytorch_support is True
            assert AcceleratorType.CUDA in gpu.supported_accelerators
    
    def test_no_cuda_detection(self, mock_no_cuda):
        """Test detection when CUDA is not available."""
        detector = GPUDetector(enable_benchmarks=False)
        gpus = detector._detect_cuda_gpus()
        
        assert len(gpus) == 0
    
    def test_mps_detection(self, mock_mps_available):
        """Test Apple Silicon MPS detection."""
        detector = GPUDetector(enable_benchmarks=False)
        gpus = detector._detect_mps_gpus()
        
        assert len(gpus) == 1
        gpu = gpus[0]
        
        assert gpu.vendor == GPUVendor.APPLE
        assert gpu.architecture == GPUArchitecture.M2
        assert gpu.pytorch_support is True
        assert AcceleratorType.MPS in gpu.supported_accelerators
    
    def test_compute_capability_properties(self):
        """Test compute capability properties."""
        # Test Volta (7.0)
        cap_volta = ComputeCapability(major=7, minor=0)
        assert cap_volta.supports_fp16 is True
        assert cap_volta.supports_int8 is True
        assert cap_volta.supports_tensor_cores is True
        assert cap_volta.supports_tf32 is False
        
        # Test Ampere (8.6)
        cap_ampere = ComputeCapability(major=8, minor=6)
        assert cap_ampere.supports_fp16 is True
        assert cap_ampere.supports_int8 is True
        assert cap_ampere.supports_tensor_cores is True
        assert cap_ampere.supports_tf32 is True
        
        # Test older architecture (5.2)
        cap_old = ComputeCapability(major=5, minor=2)
        assert cap_old.supports_fp16 is False
        assert cap_old.supports_int8 is False
        assert cap_old.supports_tensor_cores is False
        assert cap_old.supports_tf32 is False
    
    def test_gpu_suitability_check(self):
        """Test GPU suitability for inference."""
        # Suitable GPU
        gpu_good = GPUInfo(
            name="RTX 3080",
            vendor=GPUVendor.NVIDIA,
            pytorch_support=True
        )
        gpu_good.memory.total_mb = 10240  # 10GB
        assert gpu_good.is_suitable_for_inference() is True
        
        # GPU with insufficient memory
        gpu_low_mem = GPUInfo(
            name="GT 710",
            vendor=GPUVendor.NVIDIA,
            pytorch_support=True
        )
        gpu_low_mem.memory.total_mb = 512  # 512MB
        assert gpu_low_mem.is_suitable_for_inference() is False
        
        # GPU without PyTorch support
        gpu_no_pytorch = GPUInfo(
            name="Unknown GPU",
            vendor=GPUVendor.UNKNOWN,
            pytorch_support=False
        )
        gpu_no_pytorch.memory.total_mb = 8192  # 8GB
        assert gpu_no_pytorch.is_suitable_for_inference() is False
    
    def test_batch_size_estimation(self):
        """Test batch size estimation."""
        gpu = GPUInfo()
        gpu.memory.total_mb = 8192  # 8GB
        gpu.memory.available_mb = 6000  # 6GB available
        
        # Test with 500MB model
        batch_size = gpu.estimate_max_batch_size(model_size_mb=500)
        assert batch_size > 0
        assert isinstance(batch_size, int)
        
        # Test with very large model
        large_batch_size = gpu.estimate_max_batch_size(model_size_mb=5000)
        assert large_batch_size >= 1  # Should always be at least 1
    
    def test_nvidia_architecture_detection(self):
        """Test NVIDIA architecture detection."""
        detector = GPUDetector(enable_benchmarks=False)
        
        # Test Ampere detection
        cap_ampere = ComputeCapability(major=8, minor=6)
        arch = detector._detect_nvidia_architecture(cap_ampere, "GeForce RTX 3080")
        assert arch == GPUArchitecture.AMPERE
        
        # Test Turing detection
        cap_turing = ComputeCapability(major=7, minor=5)
        arch = detector._detect_nvidia_architecture(cap_turing, "GeForce RTX 2080")
        assert arch == GPUArchitecture.TURING
        
        # Test Volta detection
        cap_volta = ComputeCapability(major=7, minor=0)
        arch = detector._detect_nvidia_architecture(cap_volta, "Tesla V100")
        assert arch == GPUArchitecture.VOLTA


class TestGPUManager:
    """Test GPU manager functionality."""
    
    @pytest.fixture
    def mock_gpu_info(self):
        """Create mock GPU info."""
        gpu = GPUInfo(
            id=0,
            name="GeForce RTX 3080",
            vendor=GPUVendor.NVIDIA,
            device_id="cuda:0",
            pytorch_support=True
        )
        gpu.compute_capability = ComputeCapability(major=8, minor=6)
        gpu.memory.total_mb = 10240
        gpu.memory.available_mb = 8192
        gpu.supported_accelerators = [AcceleratorType.CUDA]
        gpu.tensorrt_support = True
        return gpu
    
    def test_gpu_manager_initialization(self):
        """Test GPU manager initialization."""
        manager = GPUManager()
        assert manager is not None
        assert manager.detector is not None
    
    def test_device_config_generation_nvidia(self, mock_gpu_info):
        """Test device configuration generation for NVIDIA GPU."""
        manager = GPUManager()
        manager._best_gpu = mock_gpu_info
        
        config = manager._generate_device_config()
        
        assert config.device_type == DeviceType.CUDA
        assert config.device_id == 0
        assert config.use_fp16 is True  # RTX 3080 supports FP16
        assert config.use_tensorrt is True
    
    def test_device_config_generation_cpu_fallback(self):
        """Test device configuration generation with CPU fallback."""
        manager = GPUManager()
        manager._best_gpu = None
        
        config = manager._generate_device_config()
        
        assert config.device_type == DeviceType.CPU
        assert config.use_fp16 is False
        assert config.use_tensorrt is False
        assert config.use_torch_compile is False
    
    def test_memory_recommendations(self, mock_gpu_info):
        """Test memory recommendations."""
        manager = GPUManager()
        manager._best_gpu = mock_gpu_info
        
        recommendations = manager.get_memory_recommendations()
        
        assert "total_memory_mb" in recommendations
        assert "available_memory_mb" in recommendations
        assert "estimated_max_batch_size" in recommendations
        assert "recommendations" in recommendations
        assert len(recommendations["recommendations"]) > 0
    
    def test_optimization_recommendations(self, mock_gpu_info):
        """Test optimization recommendations."""
        manager = GPUManager()
        manager._best_gpu = mock_gpu_info
        
        recommendations = manager.get_optimization_recommendations()
        
        assert "gpu_name" in recommendations
        assert "vendor" in recommendations
        assert "recommendations" in recommendations
        assert len(recommendations["recommendations"]) > 0
        
        # Should include CUDA-specific recommendations
        rec_text = " ".join(recommendations["recommendations"])
        assert "cuDNN" in rec_text or "CUDA" in rec_text
    
    def test_auto_configure_device(self):
        """Test automatic device configuration."""
        with patch('framework.core.gpu_manager.GPUManager.detect_and_configure') as mock_detect:
            mock_config = Mock()
            mock_detect.return_value = ([], mock_config)
            
            config = auto_configure_device()
            assert config == mock_config
            mock_detect.assert_called_once()


class TestBenchmarking:
    """Test GPU benchmarking functionality."""
    
    @pytest.fixture
    def mock_cuda_device(self):
        """Mock CUDA device for benchmarking."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.synchronize'), \
             patch('torch.device') as mock_device:
            
            mock_device.return_value.type = "cuda"
            yield mock_device.return_value
    
    def test_memory_bandwidth_benchmark(self, mock_cuda_device):
        """Test memory bandwidth benchmarking."""
        detector = GPUDetector(enable_benchmarks=True)
        
        with patch('torch.randn') as mock_randn, \
             patch('torch.empty_like') as mock_empty, \
             patch('time.time', side_effect=[0.0, 1.0]):  # 1 second elapsed
            
            mock_tensor = Mock()
            mock_tensor.copy_ = Mock()
            mock_randn.return_value = mock_tensor
            mock_empty.return_value = mock_tensor
            
            result = detector._benchmark_memory_bandwidth(mock_cuda_device)
            
            assert "bandwidth_gb_s" in result
            assert "elapsed_time_s" in result
            assert "iterations" in result
            assert result["elapsed_time_s"] == 1.0
    
    def test_compute_benchmark(self, mock_cuda_device):
        """Test compute performance benchmarking."""
        detector = GPUDetector(enable_benchmarks=True)
        
        with patch('torch.randn') as mock_randn, \
             patch('torch.sin') as mock_sin, \
             patch('torch.exp') as mock_exp, \
             patch('time.time', side_effect=[0.0, 1.0]):  # 1 second elapsed
            
            mock_tensor = Mock()
            # Make tensor addition work
            mock_tensor.__add__ = Mock(return_value=mock_tensor)
            mock_randn.return_value = mock_tensor
            mock_sin.return_value = mock_tensor
            mock_exp.return_value = mock_tensor
            
            result = detector._benchmark_compute(mock_cuda_device, torch.float32)
            
            assert "ops_per_second" in result
            assert "elapsed_time_s" in result
            assert "dtype" in result
            assert "tensor_size" in result
    
    def test_matmul_benchmark(self, mock_cuda_device):
        """Test matrix multiplication benchmarking."""
        detector = GPUDetector(enable_benchmarks=True)
        
        with patch('torch.randn') as mock_randn, \
             patch('torch.matmul') as mock_matmul, \
             patch('time.time', side_effect=[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]):  # Multiple benchmark runs
            
            mock_tensor = Mock()
            mock_randn.return_value = mock_tensor
            mock_matmul.return_value = mock_tensor
            
            result = detector._benchmark_matmul(mock_cuda_device)
            
            assert len(result) > 0
            # Should have results for different matrix sizes
            for key, value in result.items():
                if isinstance(value, dict) and "gflops" in value:
                    assert "gflops" in value
                    assert "elapsed_time_s" in value
                    assert "iterations" in value


class TestErrorHandling:
    """Test error handling in GPU detection."""
    
    def test_cuda_error_handling(self):
        """Test handling of CUDA-related errors."""
        detector = GPUDetector(enable_benchmarks=False)
        
        # Mock CUDA available but properties failing
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_properties', side_effect=RuntimeError("CUDA error")):
            
            gpus = detector._detect_cuda_gpus()
            # Should handle error gracefully and return empty list
            assert len(gpus) == 0
    
    def test_benchmark_error_handling(self):
        """Test handling of benchmark errors."""
        detector = GPUDetector(enable_benchmarks=True)
        
        gpu_info = GPUInfo(
            device_id="cuda:0",
            pytorch_support=True
        )
        
        with patch('torch.device', side_effect=RuntimeError("Device error")):
            result = detector._benchmark_gpu(gpu_info)
            
            # Should handle error and return error information
            assert "error" in result
    
    def test_invalid_device_handling(self):
        """Test handling of invalid device configurations."""
        manager = GPUManager()
        
        # Test with invalid GPU info
        invalid_gpu = GPUInfo(
            name="Invalid GPU",
            vendor=GPUVendor.UNKNOWN,
            pytorch_support=False
        )
        invalid_gpu.memory.total_mb = 100  # Very low memory
        
        manager._best_gpu = invalid_gpu
        config = manager._generate_device_config()
        
        # Should fallback to CPU configuration
        assert config.device_type == DeviceType.CPU


if __name__ == "__main__":
    pytest.main([__file__])

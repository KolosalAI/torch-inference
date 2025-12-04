"""
Unit tests for GPU service.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from src.services.gpu_service import GPUService


class TestGPUService:
    """Test suite for GPUService."""
    
    def setup_method(self):
        """Setup test method."""
        self.gpu_service = GPUService()
    
    def test_init(self):
        """Test GPUService initialization."""
        assert self.gpu_service is not None
        assert hasattr(self.gpu_service, 'logger')
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_detect_gpus_no_cuda(self, mock_device_count, mock_is_available):
        """Test GPU detection when CUDA is not available."""
        mock_is_available.return_value = False
        mock_device_count.return_value = 0
        
        result = self.gpu_service.detect_gpus()
        
        assert result['cuda_available'] is False
        assert result['gpu_count'] == 0
        assert len(result['gpus']) == 0
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_detect_gpus_with_cuda(
        self, 
        mock_memory_reserved,
        mock_memory_allocated, 
        mock_get_props,
        mock_get_name,
        mock_device_count,
        mock_is_available
    ):
        """Test GPU detection when CUDA is available."""
        mock_is_available.return_value = True
        mock_device_count.return_value = 1
        mock_get_name.return_value = "NVIDIA GeForce RTX 3080"
        
        # Mock device properties
        mock_props = MagicMock()
        mock_props.total_memory = 10 * (1024**3)  # 10GB
        mock_props.major = 8
        mock_props.minor = 6
        mock_props.multi_processor_count = 68
        mock_get_props.return_value = mock_props
        
        mock_memory_allocated.return_value = 1 * (1024**3)  # 1GB
        mock_memory_reserved.return_value = 2 * (1024**3)   # 2GB
        
        result = self.gpu_service.detect_gpus()
        
        assert result['cuda_available'] is True
        assert result['gpu_count'] == 1
        assert len(result['gpus']) == 1
        
        gpu = result['gpus'][0]
        assert gpu['id'] == 0
        assert gpu['name'] == "NVIDIA GeForce RTX 3080"
        assert gpu['total_memory_gb'] == 10.0
        assert gpu['allocated_memory_gb'] == 1.0
        assert gpu['reserved_memory_gb'] == 2.0
        assert gpu['compute_capability'] == "8.6"
        assert gpu['multiprocessor_count'] == 68
    
    def test_get_best_gpu_error_handling(self):
        """Test get_best_gpu with error handling."""
        with patch.object(self.gpu_service, 'detect_gpus') as mock_detect:
            mock_detect.side_effect = Exception("Test error")
            
            result = self.gpu_service.get_best_gpu()
            assert 'error' in result
            assert result['error'] == "Test error"
    
    def test_get_performance_tier(self):
        """Test performance tier calculation."""
        # Test CPU
        best_gpu_cpu = {"device_type": "cpu"}
        tier = self.gpu_service._get_performance_tier(best_gpu_cpu)
        assert tier == "Basic"
        
        # Test MPS
        best_gpu_mps = {"device_type": "mps"}
        tier = self.gpu_service._get_performance_tier(best_gpu_mps)
        assert tier == "Good"
        
        # Test CUDA excellent
        best_gpu_cuda_excellent = {
            "device_type": "cuda",
            "best_gpu": {
                "total_memory_gb": 24.0,
                "compute_capability": "8.0"
            }
        }
        tier = self.gpu_service._get_performance_tier(best_gpu_cuda_excellent)
        assert tier == "Excellent"
        
        # Test CUDA good
        best_gpu_cuda_good = {
            "device_type": "cuda",
            "best_gpu": {
                "total_memory_gb": 8.0,
                "compute_capability": "6.1"
            }
        }
        tier = self.gpu_service._get_performance_tier(best_gpu_cuda_good)
        assert tier == "Good"


if __name__ == "__main__":
    pytest.main([__file__])

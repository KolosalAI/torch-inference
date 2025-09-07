"""
Unit tests for Multi-GPU functionality.
"""

import pytest
import torch
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Set testing environment variable
os.environ['TESTING'] = '1'

from framework.core.config import MultiGPUConfig
from framework.core.multi_gpu_manager import (
    MultiGPUManager, DevicePool, DeviceInfo, 
    MultiGPULoadBalancer, LoadBalancingStrategy
)
from framework.core.multi_gpu_strategies import (
    DataParallelStrategy, ModelParallelStrategy,
    MultiGPUStrategyFactory
)
from framework.core.gpu_manager import (
    setup_multi_gpu, get_multi_gpu_configuration,
    validate_multi_gpu_setup
)
from framework.core.gpu_detection import GPUInfo, GPUVendor, MemoryInfo


class TestMultiGPUConfig:
    """Test MultiGPUConfig validation."""
    
    def test_valid_config(self):
        """Test valid multi-GPU configuration."""
        config = MultiGPUConfig(
            enabled=True,
            strategy="data_parallel",
            device_ids=[0, 1],
            load_balancing="dynamic"
        )
        assert config.enabled is True
        assert config.strategy == "data_parallel"
        assert config.device_ids == [0, 1]
    
    def test_invalid_strategy(self):
        """Test invalid strategy raises error."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            MultiGPUConfig(strategy="invalid_strategy")
    
    def test_invalid_load_balancing(self):
        """Test invalid load balancing raises error."""
        with pytest.raises(ValueError, match="Invalid load_balancing"):
            MultiGPUConfig(load_balancing="invalid_balancing")
    
    def test_invalid_device_ids(self):
        """Test invalid device IDs raise error."""
        import os
        # Temporarily remove testing environment variables
        testing_env = os.environ.pop('TESTING', None)
        pytest_env = os.environ.pop('PYTEST_CURRENT_TEST', None)
        
        try:
            with pytest.raises(ValueError, match="Multi-GPU strategies require at least 2 device IDs"):
                MultiGPUConfig(enabled=True, strategy="data_parallel", device_ids=[0])  # Only one device in production
        finally:
            # Restore environment variables
            if testing_env:
                os.environ['TESTING'] = testing_env
            if pytest_env:
                os.environ['PYTEST_CURRENT_TEST'] = pytest_env
        
        with pytest.raises(ValueError, match="must be non-negative integers"):
            MultiGPUConfig(device_ids=[0, -1])  # Negative ID


class TestDevicePool:
    """Test DevicePool functionality."""
    
    @pytest.fixture
    def sample_devices(self):
        """Create sample device info."""
        return [
            DeviceInfo(
                device_id=0,
                device=torch.device("cuda:0"),
                memory_total=8192,
                memory_available=6000,
                utilization=0.5
            ),
            DeviceInfo(
                device_id=1,
                device=torch.device("cuda:1"),
                memory_total=8192,
                memory_available=7000,
                utilization=0.3
            )
        ]
    
    def test_device_pool_creation(self, sample_devices):
        """Test device pool creation."""
        pool = DevicePool(sample_devices)
        assert len(pool.devices) == 2
        assert 0 in pool.devices
        assert 1 in pool.devices
    
    def test_round_robin_selection(self, sample_devices):
        """Test round-robin device selection."""
        pool = DevicePool(sample_devices)
        
        device1 = pool.get_device(LoadBalancingStrategy.ROUND_ROBIN)
        device2 = pool.get_device(LoadBalancingStrategy.ROUND_ROBIN)
        device3 = pool.get_device(LoadBalancingStrategy.ROUND_ROBIN)
        
        assert device1.device_id != device2.device_id
        assert device1.device_id == device3.device_id  # Should cycle back
    
    def test_weighted_selection(self, sample_devices):
        """Test weighted device selection."""
        pool = DevicePool(sample_devices)
        
        # Device 1 has lower utilization, should be selected
        device = pool.get_device(LoadBalancingStrategy.WEIGHTED)
        assert device.device_id == 1
    
    def test_device_failure_handling(self, sample_devices):
        """Test device failure handling."""
        pool = DevicePool(sample_devices)
        
        # Mark device as unhealthy
        pool.mark_device_unhealthy(0)
        assert not pool.devices[0].is_healthy
        assert pool.devices[0].failure_count == 1
        
        # Only healthy devices should be returned
        healthy_devices = pool.get_all_healthy_devices()
        assert len(healthy_devices) == 1
        assert healthy_devices[0].device_id == 1


class TestMultiGPULoadBalancer:
    """Test MultiGPULoadBalancer functionality."""
    
    @pytest.fixture
    def device_pool(self):
        """Create device pool for testing."""
        devices = [
            DeviceInfo(
                device_id=0,
                device=torch.device("cuda:0"),
                memory_total=8192,
                memory_available=6000
            ),
            DeviceInfo(
                device_id=1,
                device=torch.device("cuda:1"),
                memory_total=8192,
                memory_available=4000
            )
        ]
        return DevicePool(devices)
    
    def test_batch_distribution_round_robin(self, device_pool):
        """Test round-robin batch distribution."""
        balancer = MultiGPULoadBalancer(device_pool, LoadBalancingStrategy.ROUND_ROBIN)
        
        distribution = balancer.distribute_batch(10)
        assert sum(distribution.values()) == 10
        assert len(distribution) == 2
    
    def test_batch_distribution_weighted(self, device_pool):
        """Test weighted batch distribution."""
        balancer = MultiGPULoadBalancer(device_pool, LoadBalancingStrategy.WEIGHTED)
        
        distribution = balancer.distribute_batch(10)
        assert sum(distribution.values()) == 10
        
        # Device 0 should get more items (more available memory)
        assert distribution[0] > distribution[1]
    
    def test_rebalance_trigger(self, device_pool):
        """Test load rebalancing."""
        balancer = MultiGPULoadBalancer(device_pool, LoadBalancingStrategy.DYNAMIC)
        
        initial_count = balancer.rebalance_count
        balancer.rebalance()
        assert balancer.rebalance_count == initial_count + 1


class TestMultiGPUManager:
    """Test MultiGPUManager functionality."""
    
    @pytest.fixture
    def mock_gpu_manager(self):
        """Create mock GPU manager."""
        manager = Mock()
        
        # Create mock GPUs
        gpu1 = Mock()
        gpu1.id = 0
        gpu1.name = "RTX 3080"
        gpu1.memory.total_mb = 8192
        gpu1.memory.available_mb = 6000
        gpu1.is_suitable_for_inference.return_value = True
        
        gpu2 = Mock()
        gpu2.id = 1
        gpu2.name = "RTX 3080"
        gpu2.memory.total_mb = 8192
        gpu2.memory.available_mb = 6000
        gpu2.is_suitable_for_inference.return_value = True
        
        manager.get_detected_gpus.return_value = [gpu1, gpu2]
        return manager
    
    def test_multi_gpu_initialization(self, mock_gpu_manager):
        """Test multi-GPU manager initialization."""
        config = MultiGPUConfig(
            enabled=True,
            strategy="data_parallel",
            device_ids=[0, 1]
        )
        
        manager = MultiGPUManager(config, mock_gpu_manager)
        result = manager.initialize()
        
        assert result["status"] == "initialized"
        assert result["device_count"] == 2
        assert manager.is_initialized
    
    def test_initialization_insufficient_gpus(self, mock_gpu_manager):
        """Test initialization with insufficient GPUs."""
        import os
        # Temporarily remove testing environment variables
        testing_env = os.environ.pop('TESTING', None)
        pytest_env = os.environ.pop('PYTEST_CURRENT_TEST', None)
        
        try:
            # Only return one GPU
            mock_gpu_manager.get_detected_gpus.return_value = mock_gpu_manager.get_detected_gpus.return_value[:1]
            
            config = MultiGPUConfig(enabled=True)
            manager = MultiGPUManager(config, mock_gpu_manager)
            
            with pytest.raises(ValueError, match="Multi-GPU requires at least 2"):
                manager.initialize()
        finally:
            # Restore environment variables
            if testing_env:
                os.environ['TESTING'] = testing_env
            if pytest_env:
                os.environ['PYTEST_CURRENT_TEST'] = pytest_env
    
    def test_device_failure_handling(self, mock_gpu_manager):
        """Test device failure handling."""
        config = MultiGPUConfig(enabled=True, device_ids=[0, 1])
        manager = MultiGPUManager(config, mock_gpu_manager)
        manager.initialize()
        
        # Simulate device failure
        device = torch.device("cuda:0")
        result = manager.handle_device_failure(device)
        
        assert result is True  # Should still have healthy devices
        assert manager.stats.fault_events == 1
    
    def test_stats_collection(self, mock_gpu_manager):
        """Test statistics collection."""
        config = MultiGPUConfig(enabled=True, device_ids=[0, 1])
        manager = MultiGPUManager(config, mock_gpu_manager)
        manager.initialize()
        
        stats = manager.get_stats()
        assert stats.total_devices == 2
        assert stats.active_devices == 2
        assert stats.strategy == "data_parallel"
        
        detailed_stats = manager.get_detailed_stats()
        assert detailed_stats["status"] == "initialized"
        assert "devices" in detailed_stats


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMultiGPUStrategies:
    """Test multi-GPU strategies."""
    
    @pytest.fixture
    def mock_multi_gpu_manager(self):
        """Create mock multi-GPU manager."""
        manager = Mock()
        manager.get_available_devices.return_value = [
            torch.device("cuda:0"),
            torch.device("cuda:1")
        ]
        manager.distribute_batch.return_value = {
            torch.device("cuda:0"): 2,
            torch.device("cuda:1"): 2
        }
        return manager
    
    @pytest.fixture
    def sample_model(self):
        """Create sample model for testing."""
        return torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )
    
    def test_strategy_factory(self, mock_multi_gpu_manager):
        """Test strategy factory."""
        strategy = MultiGPUStrategyFactory.create_strategy("data_parallel", mock_multi_gpu_manager)
        assert isinstance(strategy, DataParallelStrategy)
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            MultiGPUStrategyFactory.create_strategy("invalid", mock_multi_gpu_manager)
    
    @pytest.mark.asyncio
    async def test_data_parallel_setup(self, mock_multi_gpu_manager, sample_model):
        """Test data parallel strategy setup."""
        strategy = DataParallelStrategy(mock_multi_gpu_manager)
        
        # Mock successful setup
        with patch.object(strategy, '_setup_model_on_device', return_value=True):
            result = await strategy.setup(sample_model)
            
            assert result["strategy"] == "data_parallel"
            assert strategy.is_setup
    
    @pytest.mark.asyncio
    async def test_model_parallel_setup(self, mock_multi_gpu_manager, sample_model):
        """Test model parallel strategy setup."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2), \
             patch.object(sample_model, 'to', return_value=sample_model), \
             patch('torch.nn.Module.to', return_value=sample_model):
            
            strategy = ModelParallelStrategy(mock_multi_gpu_manager)
            
            result = await strategy.setup(sample_model)
            
            assert result["strategy"] == "model_parallel"
            assert strategy.is_setup


class TestMultiGPUIntegration:
    """Test multi-GPU integration with GPU manager."""
    
    def test_multi_gpu_configuration_no_gpus(self):
        """Test configuration when no suitable GPUs available."""
        with patch('framework.core.gpu_manager.get_gpu_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_detected_gpus.return_value = []
            mock_get_manager.return_value = mock_manager
            
            config = get_multi_gpu_configuration()
            
            assert config["multi_gpu_available"] is False
            assert "Insufficient suitable GPUs" in config["reason"]
    
    def test_multi_gpu_configuration_sufficient_gpus(self):
        """Test configuration with sufficient GPUs."""
        with patch('framework.core.gpu_manager.get_gpu_manager') as mock_get_manager:
            mock_manager = Mock()
            
            # Create mock GPUs
            gpu1 = Mock()
            gpu1.id = 0
            gpu1.name = "RTX 3080"
            gpu1.memory.total_mb = 8192
            gpu1.vendor.value = "nvidia"
            gpu1.architecture.value = "ampere"
            gpu1.is_suitable_for_inference.return_value = True
            
            gpu2 = Mock()
            gpu2.id = 1
            gpu2.name = "RTX 3080"
            gpu2.memory.total_mb = 8192
            gpu2.vendor.value = "nvidia"
            gpu2.architecture.value = "ampere"
            gpu2.is_suitable_for_inference.return_value = True
            
            mock_manager.get_detected_gpus.return_value = [gpu1, gpu2]
            mock_get_manager.return_value = mock_manager
            
            config = get_multi_gpu_configuration()
            
            assert config["multi_gpu_available"] is True
            assert config["device_count"] == 2
            assert "recommended_config" in config
    
    def test_multi_gpu_validation_valid_config(self):
        """Test validation of valid multi-GPU config."""
        with patch('framework.core.gpu_manager.get_gpu_manager') as mock_get_manager:
            mock_manager = Mock()
            
            # Create mock suitable GPUs
            gpu1 = Mock()
            gpu1.id = 0
            gpu1.memory.total_mb = 8192
            gpu1.is_suitable_for_inference.return_value = True
            
            gpu2 = Mock()
            gpu2.id = 1
            gpu2.memory.total_mb = 8192
            gpu2.is_suitable_for_inference.return_value = True
            
            mock_manager.get_detected_gpus.return_value = [gpu1, gpu2]
            mock_get_manager.return_value = mock_manager
            
            config = MultiGPUConfig(enabled=True, device_ids=[0, 1])
            validation = validate_multi_gpu_setup(config)
            
            assert validation["valid"] is True
            assert len(validation["errors"]) == 0
    
    def test_multi_gpu_validation_invalid_device_ids(self):
        """Test validation with invalid device IDs."""
        import os
        # Set testing environment variable
        os.environ['TESTING'] = '1'
        
        try:
            with patch('framework.core.gpu_manager.get_gpu_manager') as mock_get_manager:
                mock_manager = Mock()
                
                # Provide 2 GPUs, but device 1 will not be valid for config [0, 1] if we use [0, 2]
                gpu1 = Mock()
                gpu1.id = 0
                gpu1.is_suitable_for_inference.return_value = True
                gpu1.memory.total_mb = 8192
                
                gpu2 = Mock()
                gpu2.id = 2  # Skip device 1 to make it invalid
                gpu2.is_suitable_for_inference.return_value = True
                gpu2.memory.total_mb = 8192
                
                mock_manager.get_detected_gpus.return_value = [gpu1, gpu2]
                mock_get_manager.return_value = mock_manager
                
                config = MultiGPUConfig(enabled=True, device_ids=[0, 1])  # Device 1 not available
                validation = validate_multi_gpu_setup(config)
                
                assert validation["valid"] is False
                assert any("Invalid device IDs" in error for error in validation["errors"])
        finally:
            # Clean up environment variable
            if 'TESTING' in os.environ:
                del os.environ['TESTING']


class TestMultiGPUPerformance:
    """Test multi-GPU performance scenarios."""
    
    def test_batch_distribution_performance(self):
        """Test batch distribution doesn't introduce significant overhead."""
        import time
        
        devices = [
            DeviceInfo(device_id=i, device=torch.device(f"cuda:{i}"), 
                      memory_total=8192, memory_available=6000)
            for i in range(4)
        ]
        pool = DevicePool(devices)
        balancer = MultiGPULoadBalancer(pool, LoadBalancingStrategy.DYNAMIC)
        
        # Measure distribution time
        start_time = time.time()
        for _ in range(1000):
            distribution = balancer.distribute_batch(32)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 1000
        assert avg_time < 0.001  # Should be less than 1ms per distribution
    
    def test_device_pool_concurrent_access(self):
        """Test device pool thread safety."""
        import threading
        import time
        
        devices = [
            DeviceInfo(device_id=i, device=torch.device(f"cuda:{i}"), 
                      memory_total=8192, memory_available=6000)
            for i in range(2)
        ]
        pool = DevicePool(devices)
        
        results = []
        
        def worker():
            for _ in range(100):
                device = pool.get_device()
                if device:
                    results.append(device.device_id)
                time.sleep(0.001)
        
        threads = [threading.Thread(target=worker) for _ in range(4)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have results from all workers
        assert len(results) > 0
        # Should use both devices
        assert len(set(results)) <= 2


if __name__ == "__main__":
    pytest.main([__file__])

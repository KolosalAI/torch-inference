"""
Phase 3 Multi-GPU Integration Tests
Tests advanced performance optimization features including memory optimization,
communication optimization, dynamic scaling, and advanced scheduling.
"""

import pytest
import torch
import time
import threading
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Set testing environment variable
os.environ['TESTING'] = '1'

# Add framework to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'framework'))

from core.config import MultiGPUConfig
from core.multi_gpu_manager import MultiGPUManager
from core.memory_optimizer import MemoryOptimizer, MemoryStats
from core.comm_optimizer import CommunicationOptimizer, CommPattern
from core.dynamic_scaler import DynamicScaler, ScalingConfig, ScalingAction
from core.advanced_scheduler import AdvancedScheduler, SchedulerConfig, TaskPriority
from core.gpu_manager import GPUManager

class TestPhase3Integration:
    """Test Phase 3 performance optimization integration."""
    
    @pytest.fixture
    def mock_gpu_manager(self):
        """Create a mock GPU manager."""
        gpu_manager = Mock(spec=GPUManager)
        
        # Mock GPU detection - provide 2 GPUs for testing
        mock_gpu1 = Mock()
        mock_gpu1.id = 0
        mock_gpu1.is_suitable_for_inference.return_value = True
        mock_gpu1.memory.total_mb = 8192
        mock_gpu1.memory.available_mb = 6144
        
        mock_gpu2 = Mock()
        mock_gpu2.id = 1
        mock_gpu2.is_suitable_for_inference.return_value = True
        mock_gpu2.memory.total_mb = 8192
        mock_gpu2.memory.available_mb = 6144
        
        gpu_manager.get_detected_gpus.return_value = [mock_gpu1, mock_gpu2]
        return gpu_manager
    
    @pytest.fixture
    def multi_gpu_config(self):
        """Create multi-GPU config with Phase 3 features."""
        return MultiGPUConfig(
            enabled=True,
            strategy="data_parallel",
            device_ids=[0, 1],
            load_balancing="dynamic",
            memory_pool_size_mb=256,
            enable_nccl=False,  # Disable for testing
            scale_up_cooldown=5.0,
            scale_down_cooldown=10.0,
            scheduling_strategy="balanced",
            max_tasks_per_device=2
        )
    
    @pytest.fixture
    def multi_gpu_manager(self, multi_gpu_config, mock_gpu_manager):
        """Create multi-GPU manager for testing."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2), \
             patch('torch.cuda.get_device_properties') as mock_props:

            mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
            
            manager = MultiGPUManager(multi_gpu_config, mock_gpu_manager)
            manager.initialize()
            yield manager
            manager.cleanup()
    
    def test_memory_optimizer_integration(self, multi_gpu_manager):
        """Test memory optimizer integration."""
        memory_optimizer = multi_gpu_manager.memory_optimizer
        assert memory_optimizer is not None
        
        # Test memory allocation
        with patch('torch.empty') as mock_empty:
            mock_tensor = Mock()
            mock_tensor.shape = (2, 3, 224, 224)
            mock_tensor.dtype = torch.float32
            mock_empty.return_value = mock_tensor
            
            tensor = multi_gpu_manager.optimize_memory_allocation(
                device_id=0,
                tensor_size=(2, 3, 224, 224),
                dtype=torch.float32
            )
            
            assert tensor is not None
    
    def test_communication_optimizer_integration(self, multi_gpu_manager):
        """Test communication optimizer integration."""
        comm_optimizer = multi_gpu_manager.comm_optimizer
        assert comm_optimizer is not None
        
        # Test async transfer
        with patch('torch.empty') as mock_empty:
            mock_tensor = Mock()
            mock_tensor.to.return_value = mock_tensor
            mock_empty.return_value = mock_tensor
            
            result = multi_gpu_manager.async_transfer(
                tensor=mock_tensor,
                src_device=0,
                dst_device=0,  # Same device for testing
                priority=5
            )
            
            assert result is not None
    
    def test_dynamic_scaler_integration(self, multi_gpu_manager):
        """Test dynamic scaler integration."""
        dynamic_scaler = multi_gpu_manager.dynamic_scaler
        assert dynamic_scaler is not None
        
        # Test workload metrics collection
        metrics = multi_gpu_manager.collect_workload_metrics(
            queue_length=5,
            processing_time=0.1,
            throughput=10.0,
            error_rate=0.0
        )
        
        assert metrics is not None
        assert metrics.queue_length == 5
        assert metrics.processing_time == 0.1
        assert metrics.throughput == 10.0
    
    def test_advanced_scheduler_integration(self, multi_gpu_manager):
        """Test advanced scheduler integration."""
        scheduler = multi_gpu_manager.advanced_scheduler
        assert scheduler is not None
        
        # Test task scheduling
        def dummy_task(device_id=None):
            return f"executed on device {device_id}"
        
        task_id = multi_gpu_manager.schedule_inference_task(
            func=dummy_task,
            priority=TaskPriority.HIGH,
            memory_requirement=1024 * 1024  # 1MB
        )
        
        assert isinstance(task_id, str)
        assert len(task_id) > 0
    
    def test_optimal_batch_size_calculation(self, multi_gpu_manager):
        """Test optimal batch size calculation."""
        batch_size = multi_gpu_manager.get_optimal_batch_size(
            device_id=0,
            model_size_mb=100,
            input_shape=(3, 224, 224)
        )
        
        assert isinstance(batch_size, int)
        assert batch_size >= 1
    
    def test_performance_report(self, multi_gpu_manager):
        """Test comprehensive performance report."""
        report = multi_gpu_manager.get_performance_report()
        
        assert 'multi_gpu_stats' in report
        assert 'memory_stats' in report
        assert 'communication_stats' in report
        assert 'scaling_stats' in report
        assert 'scheduler_stats' in report
        
        # Check that all components are reporting
        assert report['multi_gpu_stats']['status'] == 'initialized'
    
    def test_phase3_cleanup(self, multi_gpu_manager):
        """Test Phase 3 component cleanup."""
        # Verify components are initialized
        assert multi_gpu_manager.memory_optimizer is not None
        assert multi_gpu_manager.comm_optimizer is not None
        assert multi_gpu_manager.dynamic_scaler is not None
        assert multi_gpu_manager.advanced_scheduler is not None
        
        # Cleanup
        multi_gpu_manager.cleanup()
        
        # Verify components are cleaned up
        assert multi_gpu_manager.memory_optimizer is None
        assert multi_gpu_manager.comm_optimizer is None
        assert multi_gpu_manager.dynamic_scaler is None
        assert multi_gpu_manager.advanced_scheduler is None


class TestMemoryOptimizer:
    """Test memory optimizer standalone functionality."""
    
    @pytest.fixture
    def memory_optimizer(self):
        """Create memory optimizer for testing."""
        with patch('torch.cuda.device'), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
            
            optimizer = MemoryOptimizer(devices=[0], pool_size_mb=256)
            yield optimizer
            optimizer.cleanup()
    
    def test_memory_allocation(self, memory_optimizer):
        """Test memory allocation with pooling."""
        with patch('torch.empty') as mock_empty:
            mock_tensor = Mock()
            mock_empty.return_value = mock_tensor
            
            tensor = memory_optimizer.allocate_tensor(0, (2, 3, 224, 224))
            assert tensor is not None
    
    def test_memory_stats(self, memory_optimizer):
        """Test memory statistics collection."""
        with patch('torch.cuda.memory_allocated', return_value=1024*1024*1024), \
             patch('torch.cuda.memory_reserved', return_value=2048*1024*1024):
            
            stats = memory_optimizer.get_memory_stats(0)
            assert 0 in stats
            assert isinstance(stats[0].utilization, float)


class TestCommunicationOptimizer:
    """Test communication optimizer standalone functionality."""
    
    @pytest.fixture
    def comm_optimizer(self):
        """Create communication optimizer for testing."""
        optimizer = CommunicationOptimizer(devices=[0], enable_nccl=False)
        yield optimizer
        optimizer.cleanup()
    
    def test_point_to_point_transfer(self, comm_optimizer):
        """Test point-to-point transfer."""
        with patch('torch.empty') as mock_empty:
            mock_tensor = Mock()
            mock_tensor.to.return_value = mock_tensor
            mock_empty.return_value = mock_tensor
            
            future = comm_optimizer.async_transfer(mock_tensor, 0, 0)
            assert future is not None
    
    def test_communication_stats(self, comm_optimizer):
        """Test communication statistics."""
        stats = comm_optimizer.get_communication_stats()
        assert 'global_stats' in stats
        assert 'device_stats' in stats


class TestDynamicScaler:
    """Test dynamic scaler standalone functionality."""
    
    @pytest.fixture
    def dynamic_scaler(self):
        """Create dynamic scaler for testing."""
        config = ScalingConfig(
            min_devices=1,
            max_devices=2,
            evaluation_interval=1.0
        )
        
        # Create scaler and mock the gpu_detector after creation
        scaler = DynamicScaler(available_devices=[0, 1], config=config)
        scaler.gpu_detector = Mock()
        yield scaler
        scaler.cleanup()
    
    def test_metrics_collection(self, dynamic_scaler):
        """Test metrics collection."""
        metrics = dynamic_scaler.collect_metrics(
            queue_length=5,
            processing_time=0.1,
            throughput=10.0
        )
        
        assert metrics.queue_length == 5
        assert metrics.processing_time == 0.1
        assert metrics.throughput == 10.0
    
    def test_scaling_stats(self, dynamic_scaler):
        """Test scaling statistics."""
        stats = dynamic_scaler.get_scaling_stats()
        assert 'active_devices' in stats
        assert 'inactive_devices' in stats


class TestAdvancedScheduler:
    """Test advanced scheduler standalone functionality."""
    
    @pytest.fixture
    def scheduler(self):
        """Create advanced scheduler for testing."""
        config = SchedulerConfig(max_tasks_per_device=2)
        
        with patch('torch.cuda.device'), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024
            
            scheduler = AdvancedScheduler(devices=[0], config=config)
            scheduler.start()
            yield scheduler
            scheduler.stop()
    
    def test_task_scheduling(self, scheduler):
        """Test task scheduling."""
        def dummy_task(device_id=None):
            return f"executed on device {device_id}"
        
        task_id = scheduler.schedule_task(
            func=dummy_task,
            priority=TaskPriority.NORMAL
        )
        
        assert isinstance(task_id, str)
        assert len(task_id) > 0
    
    def test_scheduler_stats(self, scheduler):
        """Test scheduler statistics."""
        stats = scheduler.get_scheduler_stats()
        assert 'global_stats' in stats
        assert 'device_stats' in stats
        assert 'pending_tasks' in stats


class TestPhase3EndToEnd:
    """End-to-end Phase 3 integration tests."""
    
    @pytest.fixture
    def full_system(self):
        """Create full system with all Phase 3 components."""
        import os
        
        # Check if in testing environment
        is_testing = (
            os.environ.get("TESTING") == "1" or 
            "PYTEST_CURRENT_TEST" in os.environ
        )
        
        # Use single device for testing environments
        device_count = 1 if is_testing else 2
        device_ids = [0] if is_testing else [0, 1]
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=device_count), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024
            
            # Create config
            config = MultiGPUConfig(
                enabled=True,
                strategy="data_parallel",
                device_ids=device_ids,
                memory_pool_size_mb=256,
                enable_nccl=False
            )
            
            # Create mock GPU manager
            gpu_manager = Mock(spec=GPUManager)
            
            # Mock GPU detection - provide appropriate number of GPUs for testing
            mock_gpu1 = Mock()
            mock_gpu1.id = 0
            mock_gpu1.is_suitable_for_inference.return_value = True
            mock_gpu1.memory.total_mb = 8192
            mock_gpu1.memory.available_mb = 6144
            
            gpu_list = [mock_gpu1]
            if not is_testing:
                mock_gpu2 = Mock()
                mock_gpu2.id = 1
                mock_gpu2.is_suitable_for_inference.return_value = True
                mock_gpu2.memory.total_mb = 8192
                mock_gpu2.memory.available_mb = 6144
                gpu_list.append(mock_gpu2)
            
            gpu_manager.get_detected_gpus.return_value = gpu_list
            
            # Create manager
            manager = MultiGPUManager(config, gpu_manager)
            manager.initialize()
            
            yield manager
            manager.cleanup()
    
    def test_full_inference_pipeline(self, full_system):
        """Test full inference pipeline with Phase 3 optimizations."""
        manager = full_system
        
        # Simulate inference workload
        def inference_task(input_data, device_id=None):
            # Simulate memory allocation
            with patch('torch.empty') as mock_empty:
                mock_tensor = Mock()
                mock_empty.return_value = mock_tensor
                
                # Allocate input tensor
                tensor = manager.optimize_memory_allocation(
                    device_id or 0,
                    (1, 3, 224, 224)
                )
                
                # Simulate processing time
                time.sleep(0.01)
                
                return {"result": "inference_complete", "device": device_id}
        
        # Schedule multiple tasks
        task_ids = []
        for i in range(3):
            task_id = manager.schedule_inference_task(
                func=inference_task,
                args=(f"input_{i}",),
                priority=TaskPriority.NORMAL
            )
            task_ids.append(task_id)
        
        # Collect metrics
        manager.collect_workload_metrics(
            queue_length=len(task_ids),
            processing_time=0.05,
            throughput=20.0
        )
        
        # Get comprehensive report
        report = manager.get_performance_report()
        
        # Verify all components are active
        assert report['multi_gpu_stats']['status'] == 'initialized'
        assert 'memory_stats' in report
        assert 'communication_stats' in report
        assert 'scaling_stats' in report
        assert 'scheduler_stats' in report
    
    def test_performance_optimization_workflow(self, full_system):
        """Test complete performance optimization workflow."""
        manager = full_system
        
        # Step 1: Optimize memory allocation
        optimal_batch_size = manager.get_optimal_batch_size(
            device_id=0,
            model_size_mb=100,
            input_shape=(3, 224, 224)
        )
        assert optimal_batch_size >= 1
        
        # Step 2: Schedule tasks with different priorities
        high_priority_task = manager.schedule_inference_task(
            func=lambda x, device_id=None: f"high_priority_{device_id}",
            args=("data",),
            priority=TaskPriority.HIGH
        )
        
        normal_priority_task = manager.schedule_inference_task(
            func=lambda x, device_id=None: f"normal_priority_{device_id}",
            args=("data",),
            priority=TaskPriority.NORMAL
        )
        
        # Step 3: Simulate workload changes
        manager.collect_workload_metrics(
            queue_length=2,
            processing_time=0.08,
            throughput=12.5
        )
        
        # Step 4: Get performance insights
        memory_stats = manager.get_memory_stats()
        comm_stats = manager.get_communication_stats()
        scaling_stats = manager.get_scaling_stats()
        scheduler_stats = manager.get_scheduler_stats()
        
        # Verify all systems are working
        assert memory_stats is not None
        assert comm_stats is not None
        assert scaling_stats is not None
        assert scheduler_stats is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

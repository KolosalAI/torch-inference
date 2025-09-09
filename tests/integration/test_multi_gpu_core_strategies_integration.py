"""
Phase 2 Multi-GPU Integration Tests - Core Strategies Implementation
"""

import pytest
import torch
import asyncio
import time
import os
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Set testing environment variable
os.environ['TESTING'] = '1'

from framework.core.config import MultiGPUConfig, InferenceConfig, DeviceConfig
from framework.core.inference_engine import (
    create_multi_gpu_inference_engine, 
    create_hybrid_inference_engine,
    InferenceEngine
)
from framework.core.batch_processor import (
    create_multi_gpu_batch_processor,
    create_adaptive_batch_processor,
    BatchConfig,
    BatchItem
)
from framework.core.multi_gpu_manager import MultiGPUManager
from framework.core.multi_gpu_strategies import DataParallelStrategy


@pytest.mark.integration
@pytest.mark.multi_gpu
class TestPhase2MultiGPUIntegration:
    """Integration tests for Phase 2 multi-GPU core strategies."""
    
    @pytest.fixture
    def mock_multi_gpu_environment(self):
        """Mock environment with multiple GPUs."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2), \
             patch('torch.cuda.get_device_name') as mock_name, \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            device_names = ["RTX 3080", "RTX 3090"]
            mock_name.side_effect = lambda i: device_names[i]
            
            def mock_device_props(device_id):
                props = Mock()
                props.name = device_names[device_id]
                props.total_memory = 10 * 1024**3
                props.multi_processor_count = 68
                return props
            
            mock_props.side_effect = mock_device_props
            yield
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        model = Mock()
        model.device = torch.device("cuda:0")
        model.config = Mock()
        model.predict = Mock(return_value={"prediction": "test_result"})
        
        # Add model attribute for direct access
        torch_model = torch.nn.Linear(10, 1)
        model.model = torch_model
        model.model.eval()
        
        return model
    
    @pytest.fixture
    def multi_gpu_config(self):
        """Create multi-GPU configuration."""
        return MultiGPUConfig(
            enabled=True,
            strategy="data_parallel",
            device_ids=[0, 1],
            load_balancing="dynamic",
            fault_tolerance=True
        )
    
    @pytest.fixture
    def inference_config(self, multi_gpu_config):
        """Create inference configuration with multi-GPU."""
        config = InferenceConfig()
        config.device = DeviceConfig()
        config.device.multi_gpu = multi_gpu_config
        return config
    
    def test_multi_gpu_inference_engine_creation(self, mock_multi_gpu_environment, 
                                                sample_model, inference_config):
        """Test creation of multi-GPU inference engine."""
        with patch('framework.core.gpu_manager.get_gpu_manager') as mock_get_manager:
            mock_manager = self._create_mock_gpu_manager(2)
            mock_get_manager.return_value = mock_manager
            
            engine = create_multi_gpu_inference_engine(sample_model, inference_config)
            
            assert engine is not None
            assert isinstance(engine, InferenceEngine)
            assert engine._multi_gpu_enabled
            assert engine._multi_gpu_manager is not None
    
    def test_hybrid_inference_engine_auto_detection(self, mock_multi_gpu_environment,
                                                   sample_model, inference_config):
        """Test hybrid engine auto-detection of multi-GPU."""
        with patch('framework.core.gpu_manager.get_gpu_manager') as mock_get_manager:
            mock_manager = self._create_mock_gpu_manager(2)
            mock_get_manager.return_value = mock_manager
            
            engine = create_hybrid_inference_engine(sample_model, inference_config)
            
            assert engine is not None
            assert engine._multi_gpu_enabled
    
    @pytest.mark.asyncio
    async def test_multi_gpu_inference_engine_lifecycle(self, mock_multi_gpu_environment,
                                                       sample_model, inference_config):
        """Test multi-GPU inference engine start/stop lifecycle."""
        with patch('framework.core.gpu_manager.get_gpu_manager') as mock_get_manager:
            mock_manager = self._create_mock_gpu_manager(2)
            mock_get_manager.return_value = mock_manager
            
            engine = create_multi_gpu_inference_engine(sample_model, inference_config)
            
            # Test start
            await engine.start()
            assert engine._running
            
            # Test stop
            await engine.stop()
            assert not engine._running
    
    @pytest.mark.asyncio
    async def test_multi_gpu_inference_execution(self, mock_multi_gpu_environment,
                                                sample_model, inference_config):
        """Test actual inference execution with multi-GPU."""
        with patch('framework.core.gpu_manager.get_gpu_manager') as mock_get_manager:
            mock_manager = self._create_mock_gpu_manager(2)
            mock_get_manager.return_value = mock_manager
            
            engine = create_multi_gpu_inference_engine(sample_model, inference_config)
            
            await engine.start()
            
            try:
                # Test single prediction
                test_input = torch.randn(1, 10)
                result = await engine.predict(test_input)
                assert result is not None
                
                # Test batch prediction
                test_batch = [torch.randn(1, 10) for _ in range(4)]
                batch_results = await engine.predict_batch(test_batch)
                assert len(batch_results) == 4
                
            finally:
                await engine.stop()
    
    def test_multi_gpu_batch_processor_creation(self, mock_multi_gpu_environment):
        """Test creation of multi-GPU batch processor."""
        with patch('framework.core.gpu_manager.get_gpu_manager') as mock_get_manager:
            mock_manager = self._create_mock_gpu_manager(2)
            mock_get_manager.return_value = mock_manager
            
            # Create multi-GPU manager
            from framework.core.multi_gpu_manager import setup_multi_gpu
            multi_gpu_config = MultiGPUConfig(
                enabled=True,
                strategy="data_parallel",
                device_ids=[0, 1]
            )
            
            multi_gpu_manager = setup_multi_gpu(multi_gpu_config)
            
            # Create batch processor
            batch_processor = create_multi_gpu_batch_processor(
                multi_gpu_manager=multi_gpu_manager
            )
            
            assert batch_processor is not None
            assert batch_processor._multi_gpu_enabled
            assert batch_processor._multi_gpu_manager is multi_gpu_manager
    
    @pytest.mark.asyncio
    async def test_multi_gpu_batch_processor_lifecycle(self, mock_multi_gpu_environment):
        """Test multi-GPU batch processor start/stop lifecycle."""
        with patch('framework.core.gpu_manager.get_gpu_manager') as mock_get_manager:
            mock_manager = self._create_mock_gpu_manager(2)
            mock_get_manager.return_value = mock_manager
            
            from framework.core.multi_gpu_manager import setup_multi_gpu
            multi_gpu_config = MultiGPUConfig(
                enabled=True,
                strategy="data_parallel",
                device_ids=[0, 1]
            )
            
            multi_gpu_manager = setup_multi_gpu(multi_gpu_config)
            batch_processor = create_multi_gpu_batch_processor(
                multi_gpu_manager=multi_gpu_manager
            )
            
            # Test start
            await batch_processor.start()
            assert batch_processor._running
            
            # Test stop
            await batch_processor.stop()
            assert not batch_processor._running
    
    @pytest.mark.asyncio
    async def test_multi_gpu_batch_processing(self, mock_multi_gpu_environment):
        """Test actual batch processing with multi-GPU."""
        with patch('framework.core.gpu_manager.get_gpu_manager') as mock_get_manager:
            mock_manager = self._create_mock_gpu_manager(2)
            mock_get_manager.return_value = mock_manager
            
            from framework.core.multi_gpu_manager import setup_multi_gpu
            multi_gpu_config = MultiGPUConfig(
                enabled=True,
                strategy="data_parallel",
                device_ids=[0, 1]
            )
            
            multi_gpu_manager = setup_multi_gpu(multi_gpu_config)
            batch_processor = create_multi_gpu_batch_processor(
                config=BatchConfig(max_batch_size=4),
                multi_gpu_manager=multi_gpu_manager
            )
            
            await batch_processor.start()
            
            try:
                # Test batch processing
                test_data = [torch.randn(10) for _ in range(4)]
                results = []
                
                for i, data in enumerate(test_data):
                    result = await batch_processor.process_item(data, priority=0)
                    results.append(result)
                
                assert len(results) == 4
                
            finally:
                await batch_processor.stop()
    
    def test_adaptive_batch_processor_auto_detection(self, mock_multi_gpu_environment):
        """Test adaptive batch processor auto-detection."""
        with patch('framework.core.gpu_manager.get_gpu_manager') as mock_get_manager:
            mock_manager = self._create_mock_gpu_manager(2)
            mock_get_manager.return_value = mock_manager
            
            from framework.core.multi_gpu_manager import setup_multi_gpu
            multi_gpu_config = MultiGPUConfig(
                enabled=True,
                strategy="data_parallel",
                device_ids=[0, 1]
            )
            
            multi_gpu_manager = setup_multi_gpu(multi_gpu_config)
            
            # Should detect multi-GPU capability
            batch_processor = create_adaptive_batch_processor(
                multi_gpu_manager=multi_gpu_manager
            )
            
            assert batch_processor._multi_gpu_enabled
    
    def test_multi_gpu_statistics_collection(self, mock_multi_gpu_environment,
                                            sample_model, inference_config):
        """Test multi-GPU statistics collection."""
        with patch('framework.core.gpu_manager.get_gpu_manager') as mock_get_manager:
            mock_manager = self._create_mock_gpu_manager(2)
            mock_get_manager.return_value = mock_manager
            
            engine = create_multi_gpu_inference_engine(sample_model, inference_config)
            stats = engine.get_comprehensive_stats()
            
            assert "multi_gpu_stats" in stats
            assert stats["multi_gpu_stats"]["enabled"] is True
            assert stats["multi_gpu_stats"]["device_count"] == 2
            assert "engine_config" in stats
            assert stats["engine_config"]["multi_gpu_enabled"] is True
    
    @pytest.mark.asyncio
    async def test_multi_gpu_fault_tolerance(self, mock_multi_gpu_environment,
                                           sample_model, inference_config):
        """Test multi-GPU fault tolerance during inference."""
        with patch('framework.core.gpu_manager.get_gpu_manager') as mock_get_manager:
            mock_manager = self._create_mock_gpu_manager(2)
            mock_get_manager.return_value = mock_manager
            
            engine = create_multi_gpu_inference_engine(sample_model, inference_config)
            await engine.start()
            
            try:
                # Simulate device failure
                device = torch.device("cuda:1")
                engine._multi_gpu_manager.handle_device_failure(device)
                
                # Should still be able to process requests
                test_input = torch.randn(1, 10)
                result = await engine.predict(test_input)
                assert result is not None
                
                # Check fault tolerance stats
                stats = engine._multi_gpu_manager.get_stats()
                assert stats.fault_events == 1
                assert stats.active_devices == 1  # One device failed
                
            finally:
                await engine.stop()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_multi_gpu_performance_scaling(self, mock_multi_gpu_environment,
                                                sample_model, inference_config):
        """Test performance scaling with multi-GPU."""
        with patch('framework.core.gpu_manager.get_gpu_manager') as mock_get_manager:
            mock_manager = self._create_mock_gpu_manager(2)
            mock_get_manager.return_value = mock_manager
            
            engine = create_multi_gpu_inference_engine(sample_model, inference_config)
            await engine.start()
            
            try:
                # Test large batch processing
                large_batch = [torch.randn(1, 10) for _ in range(16)]
                
                start_time = time.time()
                results = await engine.predict_batch(large_batch)
                processing_time = time.time() - start_time
                
                assert len(results) == 16
                assert processing_time < 3.0  # Allow slower environments
                
                # Check throughput improvement
                stats = engine.get_comprehensive_stats()
                throughput = stats.get("requests_per_second", 0)
                assert throughput > 0  # Should have reasonable throughput
                
            finally:
                await engine.stop()
    
    def _create_mock_gpu_manager(self, gpu_count: int):
        """Helper to create mock GPU manager."""
        mock_manager = Mock()
        
        gpus = []
        for i in range(gpu_count):
            gpu = Mock()
            gpu.id = i
            gpu.name = f"RTX 308{i}"
            gpu.memory.total_mb = 10240
            gpu.memory.available_mb = 8000
            gpu.is_suitable_for_inference.return_value = True
            gpus.append(gpu)
        
        mock_manager.get_detected_gpus.return_value = gpus
        return mock_manager


@pytest.mark.integration
@pytest.mark.multi_gpu
@pytest.mark.strategy
class TestMultiGPUStrategyIntegration:
    """Integration tests for multi-GPU strategy integration."""
    
    @pytest.fixture
    def mock_strategy_environment(self):
        """Mock environment for strategy testing."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2):
            yield
    
    @pytest.mark.asyncio
    async def test_data_parallel_strategy_integration(self, mock_strategy_environment):
        """Test data parallel strategy integration with inference engine."""
        with patch('framework.core.gpu_manager.get_gpu_manager') as mock_get_manager:
            mock_manager = self._create_mock_gpu_manager(2)
            mock_get_manager.return_value = mock_manager
            
            from framework.core.multi_gpu_manager import setup_multi_gpu
            multi_gpu_config = MultiGPUConfig(
                enabled=True,
                strategy="data_parallel",
                device_ids=[0, 1]
            )
            
            multi_gpu_manager = setup_multi_gpu(multi_gpu_config)
            multi_gpu_manager.initialize()
            
            # Create strategy
            strategy = DataParallelStrategy(multi_gpu_manager)
            
            # Mock model for strategy setup
            mock_model = Mock()
            mock_model.model = torch.nn.Linear(10, 1)
            
            # Test strategy setup
            await strategy.setup(mock_model)
            assert strategy.is_setup
            
            # Test strategy forward pass
            test_input = torch.randn(4, 10)
            result = await strategy.forward(test_input)
            assert result is not None
            
            # Test strategy cleanup
            strategy.cleanup()
    
    def _create_mock_gpu_manager(self, gpu_count: int):
        """Helper to create mock GPU manager."""
        mock_manager = Mock()
        
        gpus = []
        for i in range(gpu_count):
            gpu = Mock()
            gpu.id = i
            gpu.name = f"RTX 308{i}"
            gpu.memory.total_mb = 10240
            gpu.memory.available_mb = 8000
            gpu.is_suitable_for_inference.return_value = True
            gpus.append(gpu)
        
        mock_manager.get_detected_gpus.return_value = gpus
        return mock_manager


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

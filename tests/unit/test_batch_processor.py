"""
Unit tests for BatchProcessor

Tests the batch processing system including:
- Adaptive batch sizing algorithms
- Memory-aware batch processing
- Pipeline processing with multiple stages
- Dynamic timeout adjustment
- Performance optimization and monitoring
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, Optional
import numpy as np

from framework.core.batch_processor import (
    BatchProcessor,
    BatchConfig,
    AdaptiveBatchSizer,
    MemoryManager,
    BatchItem,
    ProcessingPipeline,
    BatchScheduler
)


class TestBatchConfig:
    """Test BatchConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = BatchConfig()
        
        assert config.max_batch_size == 8
        assert config.min_batch_size == 1
        assert config.batch_timeout_ms == 50
        assert config.enable_adaptive_batching is True
        assert config.enable_memory_management is True
        assert config.memory_threshold_mb == 1000.0
        assert config.enable_dynamic_batching is True
        assert config.adaptive_scaling_factor == 1.2
        assert config.performance_target_ms == 100.0
        assert config.enable_pipeline_processing is True
        assert config.pipeline_stages == 3
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = BatchConfig(
            max_batch_size=16,
            batch_timeout_ms=100,
            enable_adaptive_batching=False,
            memory_threshold_mb=2000.0
        )
        
        assert config.max_batch_size == 16
        assert config.batch_timeout_ms == 100
        assert config.enable_adaptive_batching is False
        assert config.memory_threshold_mb == 2000.0


class TestBatchItem:
    """Test BatchItem dataclass"""
    
    def test_batch_item_creation(self):
        """Test batch item creation"""
        future = asyncio.Future()
        item = BatchItem(
            data={"input": "test_data"},
            future=future,
            priority=1,
            timestamp=time.time()
        )
        
        assert item.data == {"input": "test_data"}
        assert item.future == future
        assert item.priority == 1
        assert isinstance(item.timestamp, float)
    
    def test_batch_item_comparison(self):
        """Test batch item priority comparison"""
        future1 = asyncio.Future()
        future2 = asyncio.Future()
        
        item1 = BatchItem(data={}, future=future1, priority=1, timestamp=time.time())
        item2 = BatchItem(data={}, future=future2, priority=2, timestamp=time.time())
        
        # Higher priority should be considered "less than" for min-heap
        assert item2 < item1  # Priority 2 is higher than priority 1


class TestAdaptiveBatchSizer:
    """Test AdaptiveBatchSizer functionality"""
    
    def test_sizer_initialization(self):
        """Test adaptive batch sizer initialization"""
        sizer = AdaptiveBatchSizer(
            initial_size=4,
            min_size=1,
            max_size=16,
            scaling_factor=1.2,
            target_latency_ms=100.0
        )
        
        assert sizer.current_size == 4
        assert sizer.min_size == 1
        assert sizer.max_size == 16
        assert sizer.scaling_factor == 1.2
        assert sizer.target_latency_ms == 100.0
    
    def test_sizer_performance_feedback(self):
        """Test performance feedback adjustment"""
        sizer = AdaptiveBatchSizer(
            initial_size=8,
            min_size=2,
            max_size=16,
            scaling_factor=1.2,
            target_latency_ms=100.0
        )
        
        # Good performance should increase batch size
        sizer.update_performance(latency_ms=50.0, throughput=100.0)
        new_size = sizer.get_optimal_batch_size()
        assert new_size > 8  # Should increase
        
        # Reset for next test
        sizer.current_size = 8
        
        # Poor performance should decrease batch size
        sizer.update_performance(latency_ms=200.0, throughput=50.0)
        new_size = sizer.get_optimal_batch_size()
        assert new_size < 8  # Should decrease
    
    def test_sizer_bounds_enforcement(self):
        """Test batch size bounds enforcement"""
        sizer = AdaptiveBatchSizer(
            initial_size=8,
            min_size=2,
            max_size=12,
            scaling_factor=2.0  # Aggressive scaling
        )
        
        # Very good performance shouldn't exceed max
        sizer.update_performance(latency_ms=10.0, throughput=1000.0)
        optimal_size = sizer.get_optimal_batch_size()
        assert optimal_size <= 12
        
        # Very poor performance shouldn't go below min
        sizer.current_size = 4
        sizer.update_performance(latency_ms=1000.0, throughput=1.0)
        optimal_size = sizer.get_optimal_batch_size()
        assert optimal_size >= 2
    
    def test_sizer_statistics(self):
        """Test batch sizer statistics collection"""
        sizer = AdaptiveBatchSizer(initial_size=4)
        
        # Record some performance metrics
        sizer.update_performance(100.0, 50.0)
        sizer.update_performance(120.0, 45.0)
        sizer.update_performance(80.0, 60.0)
        
        stats = sizer.get_stats()
        
        assert 'current_size' in stats
        assert 'average_latency' in stats
        assert 'average_throughput' in stats
        assert 'adjustments_made' in stats
        assert stats['current_size'] == sizer.current_size


class TestMemoryManager:
    """Test MemoryManager functionality"""
    
    def test_memory_manager_initialization(self):
        """Test memory manager initialization"""
        manager = MemoryManager(
            threshold_mb=1000.0,
            warning_threshold=0.8,
            critical_threshold=0.95
        )
        
        assert manager.threshold_mb == 1000.0
        assert manager.warning_threshold == 0.8
        assert manager.critical_threshold == 0.95
    
    @patch('psutil.virtual_memory')
    def test_memory_usage_monitoring(self, mock_memory):
        """Test memory usage monitoring"""
        # Mock memory info
        mock_memory.return_value = MagicMock(
            total=8 * 1024 * 1024 * 1024,  # 8GB
            used=4 * 1024 * 1024 * 1024,   # 4GB used
            percent=50.0
        )
        
        manager = MemoryManager(threshold_mb=1000.0)
        
        usage = manager.get_memory_usage()
        assert usage['used_mb'] == 4 * 1024  # 4GB in MB
        assert usage['total_mb'] == 8 * 1024  # 8GB in MB
        assert usage['usage_percent'] == 50.0
    
    @patch('psutil.virtual_memory')
    def test_memory_pressure_detection(self, mock_memory):
        """Test memory pressure detection"""
        manager = MemoryManager(
            threshold_mb=1000.0,
            warning_threshold=0.7,
            critical_threshold=0.9
        )
        
        # Normal memory usage
        mock_memory.return_value = MagicMock(percent=60.0)
        assert manager.is_memory_pressure() is False
        
        # Warning level
        mock_memory.return_value = MagicMock(percent=75.0)
        assert manager.is_memory_pressure() is True
        
        # Critical level
        mock_memory.return_value = MagicMock(percent=95.0)
        assert manager.is_memory_pressure() is True
    
    def test_memory_optimization_suggestions(self):
        """Test memory optimization suggestions"""
        manager = MemoryManager(threshold_mb=1000.0)
        
        # Mock high memory usage
        with patch.object(manager, 'get_memory_usage', return_value={'usage_percent': 85.0}):
            suggestions = manager.get_optimization_suggestions()
            
            assert len(suggestions) > 0
            assert any('reduce_batch_size' in s['action'] for s in suggestions)
    
    def test_memory_cleanup_recommendations(self):
        """Test memory cleanup recommendations"""
        manager = MemoryManager(threshold_mb=1000.0)
        
        with patch.object(manager, 'is_memory_pressure', return_value=True):
            cleanup_actions = manager.suggest_cleanup_actions()
            
            assert len(cleanup_actions) > 0
            assert any('garbage_collect' in action for action in cleanup_actions)


class TestBatchScheduler:
    """Test BatchScheduler functionality"""
    
    def test_scheduler_initialization(self):
        """Test batch scheduler initialization"""
        scheduler = BatchScheduler(
            max_batch_size=8,
            timeout_ms=50,
            min_batch_size=2
        )
        
        assert scheduler.max_batch_size == 8
        assert scheduler.timeout_ms == 50
        assert scheduler.min_batch_size == 2
        assert len(scheduler.pending_items) == 0
    
    @pytest.mark.asyncio
    async def test_scheduler_item_batching(self):
        """Test item batching by scheduler"""
        scheduler = BatchScheduler(max_batch_size=3, timeout_ms=100)
        
        # Add items to scheduler
        items = []
        for i in range(3):
            future = asyncio.Future()
            item = BatchItem(
                data={"input": f"data_{i}"},
                future=future,
                priority=0,
                timestamp=time.time()
            )
            items.append(item)
            scheduler.add_item(item)
        
        # Should create batch when max size reached
        batch = scheduler.try_create_batch()
        assert len(batch) == 3
        assert len(scheduler.pending_items) == 0
    
    @pytest.mark.asyncio
    async def test_scheduler_timeout_batching(self):
        """Test timeout-based batching"""
        scheduler = BatchScheduler(max_batch_size=5, timeout_ms=50)
        
        # Add items but not enough to trigger size-based batching
        items = []
        for i in range(2):
            future = asyncio.Future()
            item = BatchItem(
                data={"input": f"data_{i}"},
                future=future,
                priority=0,
                timestamp=time.time()
            )
            items.append(item)
            scheduler.add_item(item)
        
        # Should not create batch immediately
        batch = scheduler.try_create_batch()
        assert batch is None
        
        # Wait for timeout
        await asyncio.sleep(0.06)
        
        # Should create batch after timeout
        batch = scheduler.try_create_batch()
        assert len(batch) == 2
    
    def test_scheduler_priority_ordering(self):
        """Test priority-based item ordering"""
        scheduler = BatchScheduler(max_batch_size=3, timeout_ms=100)
        
        # Add items with different priorities
        items = []
        for i, priority in enumerate([1, 3, 2]):  # Mixed priorities
            future = asyncio.Future()
            item = BatchItem(
                data={"input": f"data_{i}"},
                future=future,
                priority=priority,
                timestamp=time.time()
            )
            items.append(item)
            scheduler.add_item(item)
        
        batch = scheduler.try_create_batch()
        
        # Should be ordered by priority (highest first)
        assert batch[0].priority == 3
        assert batch[1].priority == 2
        assert batch[2].priority == 1
    
    def test_scheduler_statistics(self):
        """Test scheduler statistics"""
        scheduler = BatchScheduler(max_batch_size=4, timeout_ms=100)
        
        # Process some batches
        for i in range(8):  # Will create 2 batches of 4
            future = asyncio.Future()
            item = BatchItem(
                data={"input": f"data_{i}"},
                future=future,
                priority=0,
                timestamp=time.time()
            )
            scheduler.add_item(item)
            
            if (i + 1) % 4 == 0:  # Every 4 items
                batch = scheduler.try_create_batch()
                scheduler.record_batch_processed(len(batch))
        
        stats = scheduler.get_stats()
        
        assert stats['batches_created'] == 2
        assert stats['items_processed'] == 8
        assert stats['average_batch_size'] == 4.0


class TestProcessingPipeline:
    """Test ProcessingPipeline functionality"""
    
    def test_pipeline_initialization(self):
        """Test processing pipeline initialization"""
        pipeline = ProcessingPipeline(num_stages=3, stage_capacity=5)
        
        assert len(pipeline.stages) == 3
        assert all(stage.maxsize == 5 for stage in pipeline.stages)
    
    @pytest.mark.asyncio
    async def test_pipeline_stage_processing(self):
        """Test processing through pipeline stages"""
        pipeline = ProcessingPipeline(num_stages=2, stage_capacity=10)
        await pipeline.start()
        
        processed_data = []
        
        async def mock_stage_processor(stage_id, data):
            processed_data.append(f"stage_{stage_id}_{data}")
            return f"processed_{stage_id}_{data}"
        
        # Process data through pipeline
        result = await pipeline.process_through_stages(
            "test_data",
            mock_stage_processor
        )
        
        assert result == "processed_1_processed_0_test_data"
        assert len(processed_data) == 2
        
        await pipeline.stop()
    
    @pytest.mark.asyncio
    async def test_pipeline_parallel_processing(self):
        """Test parallel processing capability"""
        pipeline = ProcessingPipeline(num_stages=1, stage_capacity=5)
        await pipeline.start()
        
        async def slow_processor(stage_id, data):
            await asyncio.sleep(0.1)
            return f"processed_{data}"
        
        # Submit multiple items concurrently
        tasks = []
        for i in range(3):
            task = pipeline.process_through_stages(
                f"data_{i}",
                slow_processor
            )
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time
        
        # Should process in parallel (faster than sequential)
        assert elapsed_time < 0.25  # Much less than 0.3s sequential
        assert len(results) == 3
        
        await pipeline.stop()
    
    @pytest.mark.asyncio
    async def test_pipeline_backpressure(self):
        """Test pipeline backpressure handling"""
        pipeline = ProcessingPipeline(num_stages=1, stage_capacity=2)  # Small capacity
        await pipeline.start()
        
        async def slow_processor(stage_id, data):
            await asyncio.sleep(0.2)  # Slow processing
            return f"processed_{data}"
        
        # Submit more items than capacity
        tasks = []
        for i in range(4):  # More than capacity
            task = pipeline.process_through_stages(
                f"data_{i}",
                slow_processor
            )
            tasks.append(task)
        
        # Should handle backpressure gracefully
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        # Should complete all tasks eventually
        assert len(successful_results) == 4
        
        await pipeline.stop()


class TestBatchProcessor:
    """Test BatchProcessor main class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return BatchConfig(
            max_batch_size=4,
            min_batch_size=1,
            batch_timeout_ms=50,
            enable_adaptive_batching=True,
            enable_memory_management=True,
            memory_threshold_mb=100.0
        )
    
    @pytest.fixture
    def processor(self, config):
        """Create BatchProcessor fixture"""
        return BatchProcessor(config)
    
    @pytest.mark.asyncio
    async def test_processor_initialization(self, processor):
        """Test processor initialization"""
        await processor.start()
        
        assert processor._started is True
        assert processor.batch_sizer is not None
        assert processor.memory_manager is not None
        assert processor.scheduler is not None
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_processor_single_item(self, processor):
        """Test processing single item"""
        await processor.start()
        
        async def simple_handler(batch_data):
            return [f"processed_{item}" for item in batch_data]
        
        result = await processor.process_item(
            data="test_item",
            handler=simple_handler
        )
        
        assert result == "processed_test_item"
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_processor_batch_formation(self, processor):
        """Test automatic batch formation"""
        await processor.start()
        
        results = []
        
        async def batch_handler(batch_data):
            results.extend([f"processed_{item}" for item in batch_data])
            return [f"processed_{item}" for item in batch_data]
        
        # Submit multiple items quickly
        tasks = []
        for i in range(4):  # Matches max_batch_size
            task = processor.process_item(
                data=f"item_{i}",
                handler=batch_handler
            )
            tasks.append(task)
        
        # All should complete
        completed_results = await asyncio.gather(*tasks)
        
        assert len(completed_results) == 4
        assert all("processed_" in str(result) for result in completed_results)
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_processor_adaptive_sizing(self, processor):
        """Test adaptive batch sizing"""
        await processor.start()
        
        call_count = 0
        batch_sizes = []
        
        async def tracking_handler(batch_data):
            nonlocal call_count
            call_count += 1
            batch_sizes.append(len(batch_data))
            
            # Simulate different performance based on batch size
            if len(batch_data) > 3:
                await asyncio.sleep(0.1)  # Slower for large batches
            else:
                await asyncio.sleep(0.02)  # Faster for small batches
            
            return [f"processed_{item}" for item in batch_data]
        
        # Process multiple batches to trigger adaptation
        for batch_num in range(3):
            tasks = []
            for i in range(4):  # Submit full batches
                task = processor.process_item(
                    data=f"batch_{batch_num}_item_{i}",
                    handler=tracking_handler
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            await asyncio.sleep(0.1)  # Allow adaptation time
        
        # Should have made at least some batches
        assert call_count >= 3
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_processor_memory_management(self, processor):
        """Test memory management integration"""
        await processor.start()
        
        # Mock high memory usage
        with patch.object(processor.memory_manager, 'is_memory_pressure', return_value=True):
            async def memory_handler(batch_data):
                return [f"processed_{item}" for item in batch_data]
            
            # Should still process but may adjust batch sizes
            result = await processor.process_item(
                data="memory_test",
                handler=memory_handler
            )
            
            assert "processed_memory_test" == result
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_processor_priority_handling(self, processor):
        """Test priority-based processing"""
        await processor.start()
        
        processing_order = []
        
        async def order_tracking_handler(batch_data):
            processing_order.extend(batch_data)
            return [f"processed_{item}" for item in batch_data]
        
        # Submit items with different priorities
        tasks = []
        
        # Low priority items
        for i in range(2):
            task = processor.process_item(
                data=f"low_{i}",
                handler=order_tracking_handler,
                priority=0
            )
            tasks.append(task)
        
        # High priority item
        high_task = processor.process_item(
            data="high_priority",
            handler=order_tracking_handler,
            priority=2
        )
        tasks.append(high_task)
        
        await asyncio.gather(*tasks)
        
        # High priority should be processed first
        assert "high_priority" in processing_order
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_processor_error_handling(self, processor):
        """Test error handling in batch processing"""
        await processor.start()
        
        async def failing_handler(batch_data):
            if "error" in str(batch_data[0]):
                raise ValueError("Simulated processing error")
            return [f"processed_{item}" for item in batch_data]
        
        # Submit normal item
        normal_task = processor.process_item(
            data="normal_item",
            handler=failing_handler
        )
        
        # Submit failing item
        error_task = processor.process_item(
            data="error_item",
            handler=failing_handler
        )
        
        results = await asyncio.gather(normal_task, error_task, return_exceptions=True)
        
        # Normal item should succeed
        assert results[0] == "processed_normal_item"
        
        # Error item should raise exception
        assert isinstance(results[1], ValueError)
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_processor_stats_collection(self, processor):
        """Test statistics collection"""
        await processor.start()
        
        async def stats_handler(batch_data):
            await asyncio.sleep(0.01)  # Small delay
            return [f"processed_{item}" for item in batch_data]
        
        # Process several items
        tasks = []
        for i in range(6):
            task = processor.process_item(
                data=f"stats_item_{i}",
                handler=stats_handler
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        stats = processor.get_stats()
        
        assert 'items_processed' in stats
        assert 'batches_processed' in stats
        assert 'average_batch_size' in stats
        assert 'average_processing_time' in stats
        assert 'memory_usage' in stats
        
        assert stats['items_processed'] >= 6
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_processor_health_check(self, processor):
        """Test health check functionality"""
        await processor.start()
        
        health = await processor.health_check()
        
        assert health['status'] == 'healthy'
        assert 'batch_sizer' in health['components']
        assert 'memory_manager' in health['components']
        assert 'scheduler' in health['components']
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_processor_graceful_shutdown(self, processor):
        """Test graceful shutdown with pending items"""
        await processor.start()
        
        async def slow_handler(batch_data):
            await asyncio.sleep(0.1)
            return [f"processed_{item}" for item in batch_data]
        
        # Submit item and immediately stop
        task = asyncio.create_task(processor.process_item(
            data="shutdown_test",
            handler=slow_handler
        ))
        
        await asyncio.sleep(0.02)  # Let processing start
        await processor.stop()
        
        # Task should still complete
        result = await task
        assert result == "processed_shutdown_test"


class TestBatchProcessorIntegration:
    """Integration tests for BatchProcessor"""
    
    @pytest.mark.asyncio
    async def test_high_throughput_processing(self):
        """Test high throughput batch processing"""
        config = BatchConfig(
            max_batch_size=8,
            batch_timeout_ms=25,
            enable_adaptive_batching=True
        )
        processor = BatchProcessor(config)
        await processor.start()
        
        processed_count = 0
        
        async def throughput_handler(batch_data):
            nonlocal processed_count
            processed_count += len(batch_data)
            await asyncio.sleep(0.01)  # Small processing delay
            return [f"processed_{item}" for item in batch_data]
        
        # Submit many items for high throughput test
        tasks = []
        start_time = time.time()
        
        for i in range(100):
            task = processor.process_item(
                data=f"throughput_item_{i}",
                handler=throughput_handler
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time
        
        # Calculate throughput
        throughput = len(results) / elapsed_time
        
        # Should achieve good throughput through batching
        assert throughput > 50  # items per second
        assert len(results) == 100
        assert processed_count == 100
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_mixed_workload_processing(self):
        """Test processing mixed workload with different priorities and sizes"""
        config = BatchConfig(
            max_batch_size=6,
            enable_adaptive_batching=True,
            enable_pipeline_processing=True
        )
        processor = BatchProcessor(config)
        await processor.start()
        
        results_by_priority = {0: [], 1: [], 2: []}
        
        async def mixed_handler(batch_data):
            # Simulate different processing times
            await asyncio.sleep(0.02)
            return [f"processed_{item}" for item in batch_data]
        
        # Submit mixed priority items
        tasks = []
        
        # Low priority items
        for i in range(10):
            task = processor.process_item(
                data=f"low_{i}",
                handler=mixed_handler,
                priority=0
            )
            tasks.append(task)
        
        # Medium priority items
        for i in range(5):
            task = processor.process_item(
                data=f"med_{i}",
                handler=mixed_handler,
                priority=1
            )
            tasks.append(task)
        
        # High priority items
        for i in range(3):
            task = processor.process_item(
                data=f"high_{i}",
                handler=mixed_handler,
                priority=2
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Should process all items successfully
        assert len(results) == 18
        assert all("processed_" in str(result) for result in results)
        
        # Check stats for batching efficiency
        stats = processor.get_stats()
        assert stats['batches_processed'] < 18  # Should have batched items
        assert stats['items_processed'] == 18
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_adaptive_optimization_under_load(self):
        """Test adaptive optimization under varying load conditions"""
        config = BatchConfig(
            max_batch_size=10,
            enable_adaptive_batching=True,
            enable_memory_management=True,
            performance_target_ms=50.0
        )
        processor = BatchProcessor(config)
        await processor.start()
        
        # Track batch size changes
        batch_sizes = []
        
        async def adaptive_handler(batch_data):
            batch_sizes.append(len(batch_data))
            
            # Simulate variable processing times
            if len(batch_data) > 6:
                await asyncio.sleep(0.08)  # Slower for large batches
            else:
                await asyncio.sleep(0.03)  # Faster for small batches
            
            return [f"processed_{item}" for item in batch_data]
        
        # Process multiple rounds to trigger adaptation
        for round_num in range(5):
            tasks = []
            
            # Submit varying numbers of items
            num_items = 8 if round_num % 2 == 0 else 4
            
            for i in range(num_items):
                task = processor.process_item(
                    data=f"round_{round_num}_item_{i}",
                    handler=adaptive_handler
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            await asyncio.sleep(0.1)  # Allow adaptation time
        
        # Should have processed multiple batches with adaptation
        assert len(batch_sizes) >= 5
        
        # Batch sizes should show some variation due to adaptation
        unique_sizes = set(batch_sizes)
        assert len(unique_sizes) > 1  # Should have adapted batch sizes
        
        await processor.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

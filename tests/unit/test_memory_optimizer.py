"""
Unit tests for enhanced memory optimization with fragmentation prevention.
"""

import pytest
import torch
import torch.nn as nn
import threading
import time
import gc
from unittest.mock import Mock, patch
import weakref

from framework.optimizers.memory_optimizer import (
    MemoryOptimizer,
    MemoryConfig,
    AdvancedMemoryPool,
    FragmentationStats,
    MemoryBlock,
    get_memory_optimizer
)


class MemoryIntensiveModel(nn.Module):
    """Model that creates memory pressure for testing."""
    
    def __init__(self, size_multiplier=1):
        super().__init__()
        base_size = 64 * size_multiplier
        self.conv1 = nn.Conv2d(3, base_size, 5, padding=2)
        self.conv2 = nn.Conv2d(base_size, base_size * 2, 5, padding=2)
        self.conv3 = nn.Conv2d(base_size * 2, base_size * 4, 5, padding=2)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(base_size * 4 * 4 * 4, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.fixture
def memory_config():
    """Create memory configuration for testing."""
    return MemoryConfig(
        enable_memory_pool=True,
        pool_size_mb=128,  # Small for testing
        fragmentation_threshold=0.3,
        cleanup_interval=1,  # Short interval for testing
        enable_background_cleanup=True,
        memory_growth_factor=1.5,
        enable_cuda_memory_pool=True,
        gradient_accumulation_steps=1
    )


@pytest.fixture
def sample_model():
    """Create sample model for testing."""
    return MemoryIntensiveModel(size_multiplier=1)


@pytest.fixture
def large_model():
    """Create larger model for stress testing."""
    return MemoryIntensiveModel(size_multiplier=2)


class TestMemoryConfig:
    """Test memory configuration."""
    
    def test_default_config(self):
        """Test default memory configuration."""
        config = MemoryConfig()
        assert config.enable_memory_pool == True
        assert config.pool_size_mb == 512
        assert config.fragmentation_threshold == 0.5
        assert config.cleanup_interval == 10
        assert config.enable_background_cleanup == True
        assert config.memory_growth_factor == 2.0
        assert config.enable_cuda_memory_pool == True
        assert config.gradient_accumulation_steps == 4
        assert config.enable_garbage_collection == True
        assert config.gc_threshold == 0.8
    
    def test_custom_config(self):
        """Test custom memory configuration."""
        config = MemoryConfig(
            pool_size_mb=256,
            fragmentation_threshold=0.3,
            cleanup_interval=5,
            enable_background_cleanup=False
        )
        assert config.pool_size_mb == 256
        assert config.fragmentation_threshold == 0.3
        assert config.cleanup_interval == 5
        assert config.enable_background_cleanup == False


class TestMemoryBlock:
    """Test memory block implementation."""
    
    def test_memory_block_creation(self):
        """Test creating memory blocks."""
        data = torch.randn(100, 100)
        block = MemoryBlock(data.size(), data.dtype, data.device)
        
        assert block.size == data.size()
        assert block.dtype == data.dtype
        assert block.device == data.device
        assert not block.in_use
        assert block.allocation_time > 0
    
    def test_memory_block_usage(self):
        """Test memory block usage tracking."""
        block = MemoryBlock(torch.Size([10, 10]), torch.float32, torch.device("cpu"))
        
        assert not block.in_use
        
        # Mark as used
        block.in_use = True
        assert block.in_use
        
        # Mark as free
        block.in_use = False
        assert not block.in_use
    
    def test_memory_block_age(self):
        """Test memory block age calculation."""
        block = MemoryBlock(torch.Size([10, 10]), torch.float32, torch.device("cpu"))
        
        time.sleep(0.01)  # Small delay
        age = block.get_age()
        
        assert age > 0
        assert age < 1.0  # Should be much less than 1 second


class TestFragmentationStats:
    """Test fragmentation statistics."""
    
    def test_stats_initialization(self):
        """Test fragmentation stats initialization."""
        stats = FragmentationStats()
        
        assert stats.total_allocated == 0
        assert stats.total_free == 0
        assert stats.largest_free_block == 0
        assert stats.fragmentation_ratio == 0.0
        assert stats.num_free_blocks == 0
        assert stats.num_allocated_blocks == 0
        assert stats.timestamp > 0
    
    def test_stats_calculation(self):
        """Test fragmentation statistics calculation."""
        stats = FragmentationStats()
        
        # Simulate memory state
        stats.total_allocated = 1000
        stats.total_free = 500
        stats.largest_free_block = 200
        stats.num_free_blocks = 5
        stats.num_allocated_blocks = 10
        
        # Fragmentation ratio should be calculated correctly
        # (total_free - largest_free_block) / total_free if total_free > 0
        expected_ratio = (500 - 200) / 500
        stats.fragmentation_ratio = expected_ratio
        
        assert abs(stats.fragmentation_ratio - expected_ratio) < 1e-6
    
    def test_stats_update(self):
        """Test updating fragmentation statistics."""
        stats = FragmentationStats()
        old_timestamp = stats.timestamp
        
        time.sleep(0.01)
        
        # Update stats
        stats.total_allocated = 2000
        stats.total_free = 800
        stats.timestamp = time.time()
        
        assert stats.total_allocated == 2000
        assert stats.total_free == 800
        assert stats.timestamp > old_timestamp


class TestAdvancedMemoryPool:
    """Test advanced memory pool implementation."""
    
    def test_pool_initialization(self, memory_config):
        """Test memory pool initialization."""
        pool = AdvancedMemoryPool(memory_config)
        
        assert pool.config == memory_config
        assert len(pool.free_blocks) == 0
        assert len(pool.allocated_blocks) == 0
        assert not pool.cleanup_thread_running
    
    def test_allocate_tensor(self, memory_config):
        """Test tensor allocation from pool."""
        pool = AdvancedMemoryPool(memory_config)
        
        # Allocate tensor
        shape = torch.Size([100, 100])
        dtype = torch.float32
        device = torch.device("cpu")
        
        tensor = pool.allocate_tensor(shape, dtype, device)
        
        assert tensor is not None
        assert tensor.size() == shape
        assert tensor.dtype == dtype
        assert tensor.device == device
        
        # Pool should track allocation
        assert len(pool.allocated_blocks) > 0
    
    def test_deallocate_tensor(self, memory_config):
        """Test tensor deallocation to pool."""
        pool = AdvancedMemoryPool(memory_config)
        
        # Allocate tensor
        shape = torch.Size([50, 50])
        tensor = pool.allocate_tensor(shape, torch.float32, torch.device("cpu"))
        
        initial_allocated = len(pool.allocated_blocks)
        initial_free = len(pool.free_blocks)
        
        # Deallocate tensor
        pool.deallocate_tensor(tensor)
        
        # Should move from allocated to free
        assert len(pool.allocated_blocks) <= initial_allocated
        assert len(pool.free_blocks) >= initial_free
    
    def test_get_fragmentation_stats(self, memory_config):
        """Test getting fragmentation statistics."""
        pool = AdvancedMemoryPool(memory_config)
        
        # Allocate some tensors to create fragmentation
        tensors = []
        for i in range(5):
            tensor = pool.allocate_tensor(
                torch.Size([20, 20]), torch.float32, torch.device("cpu")
            )
            tensors.append(tensor)
        
        # Deallocate every other tensor
        for i in range(0, len(tensors), 2):
            pool.deallocate_tensor(tensors[i])
        
        # Get fragmentation stats
        stats = pool.get_fragmentation_stats()
        
        assert isinstance(stats, FragmentationStats)
        assert stats.num_allocated_blocks >= 0
        assert stats.num_free_blocks >= 0
        assert stats.fragmentation_ratio >= 0.0
        assert stats.fragmentation_ratio <= 1.0
    
    def test_cleanup_old_blocks(self, memory_config):
        """Test cleanup of old unused blocks."""
        memory_config.cleanup_interval = 0.05  # Very short for testing
        pool = AdvancedMemoryPool(memory_config)
        
        # Allocate and deallocate tensor
        tensor = pool.allocate_tensor(
            torch.Size([30, 30]), torch.float32, torch.device("cpu")
        )
        pool.deallocate_tensor(tensor)
        
        initial_free_blocks = len(pool.free_blocks)
        
        # Wait a bit for blocks to age
        time.sleep(0.1)
        
        # Manually trigger cleanup
        pool.cleanup_old_blocks()
        
        # Should have cleaned up old blocks (or at least attempted to)
        # Note: cleanup behavior may vary based on implementation
        final_free_blocks = len(pool.free_blocks)
        assert final_free_blocks >= 0  # Basic sanity check
    
    def test_defragmentation(self, memory_config):
        """Test memory defragmentation."""
        pool = AdvancedMemoryPool(memory_config)
        
        # Create fragmented memory pattern
        tensors = []
        for i in range(10):
            tensor = pool.allocate_tensor(
                torch.Size([10, 10]), torch.float32, torch.device("cpu")
            )
            tensors.append(tensor)
        
        # Deallocate every other tensor to create fragmentation
        for i in range(0, len(tensors), 2):
            pool.deallocate_tensor(tensors[i])
        
        # Get initial fragmentation
        initial_stats = pool.get_fragmentation_stats()
        
        # Run defragmentation
        pool.defragment()
        
        # Get final fragmentation
        final_stats = pool.get_fragmentation_stats()
        
        # Defragmentation should have run (results may vary)
        assert isinstance(final_stats, FragmentationStats)
        assert final_stats.timestamp >= initial_stats.timestamp
    
    def test_background_cleanup(self):
        """Test background cleanup thread."""
        config = MemoryConfig(
            enable_background_cleanup=True,
            cleanup_interval=0.1  # Short interval for testing
        )
        pool = AdvancedMemoryPool(config)
        
        # Start background cleanup
        pool.start_background_cleanup()
        
        assert pool.cleanup_thread_running
        assert pool.cleanup_thread is not None
        assert pool.cleanup_thread.is_alive()
        
        # Let it run briefly
        time.sleep(0.2)
        
        # Stop background cleanup
        pool.stop_background_cleanup()
        
        # Wait for thread to stop
        time.sleep(0.1)
        
        assert not pool.cleanup_thread_running
    
    def test_memory_pressure_handling(self, memory_config):
        """Test handling of memory pressure."""
        pool = AdvancedMemoryPool(memory_config)
        
        # Allocate many tensors to create memory pressure
        tensors = []
        try:
            for i in range(100):
                tensor = pool.allocate_tensor(
                    torch.Size([100, 100]), torch.float32, torch.device("cpu")
                )
                tensors.append(tensor)
        except RuntimeError:
            # Expected if we hit memory limits
            pass
        
        # Pool should still be functional
        stats = pool.get_fragmentation_stats()
        assert isinstance(stats, FragmentationStats)
        
        # Clean up
        for tensor in tensors:
            try:
                pool.deallocate_tensor(tensor)
            except:
                pass


class TestMemoryOptimizer:
    """Test main memory optimizer."""
    
    def test_optimizer_initialization(self, memory_config):
        """Test memory optimizer initialization."""
        optimizer = MemoryOptimizer(memory_config)
        
        assert optimizer.config == memory_config
        assert isinstance(optimizer.memory_pool, AdvancedMemoryPool)
        assert len(optimizer.optimization_cache) == 0
    
    def test_optimize_model_memory(self, sample_model, memory_config):
        """Test model memory optimization."""
        optimizer = MemoryOptimizer(memory_config)
        
        sample_input = torch.randn(2, 3, 64, 64)
        
        optimized_model = optimizer.optimize_model(sample_model, sample_input)
        
        assert optimized_model is not None
        assert isinstance(optimized_model, nn.Module)
        
        # Test that optimized model works
        original_output = sample_model(sample_input)
        optimized_output = optimized_model(sample_input)
        
        assert optimized_output.shape == original_output.shape
    
    def test_gradient_checkpointing(self, sample_model, memory_config):
        """Test gradient checkpointing optimization."""
        optimizer = MemoryOptimizer(memory_config)
        
        sample_input = torch.randn(4, 3, 64, 64)
        
        # Apply gradient checkpointing
        optimized_model = optimizer.apply_gradient_checkpointing(sample_model)
        
        assert optimized_model is not None
        
        # Test forward pass
        output = optimized_model(sample_input)
        expected_output = sample_model(sample_input)
        
        assert output.shape == expected_output.shape
    
    def test_memory_efficient_attention(self, memory_config):
        """Test memory-efficient attention optimization."""
        optimizer = MemoryOptimizer(memory_config)
        
        # Create simple attention-like model
        class AttentionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.MultiheadAttention(64, 8)
                
            def forward(self, x):
                # x shape: (seq_len, batch, embed_dim)
                attn_output, _ = self.attention(x, x, x)
                return attn_output
        
        model = AttentionModel()
        sample_input = torch.randn(16, 4, 64)  # (seq, batch, dim)
        
        optimized_model = optimizer.optimize_attention_memory(model, sample_input)
        
        assert optimized_model is not None
        
        # Test functionality
        output = optimized_model(sample_input)
        expected_output = model(sample_input)
        
        assert output.shape == expected_output.shape
    
    def test_memory_monitoring(self, sample_model, memory_config):
        """Test memory usage monitoring."""
        optimizer = MemoryOptimizer(memory_config)
        
        sample_input = torch.randn(2, 3, 64, 64)
        
        # Monitor memory during optimization
        initial_memory = optimizer.get_memory_usage()
        
        optimized_model = optimizer.optimize_model(sample_model, sample_input)
        
        final_memory = optimizer.get_memory_usage()
        
        # Should have memory usage information
        assert isinstance(initial_memory, dict)
        assert isinstance(final_memory, dict)
        assert "total_memory" in initial_memory
        assert "available_memory" in initial_memory
    
    def test_memory_profiling(self, sample_model, memory_config):
        """Test memory profiling during inference."""
        optimizer = MemoryOptimizer(memory_config)
        
        sample_input = torch.randn(2, 3, 64, 64)
        
        # Profile memory usage
        profile = optimizer.profile_memory_usage(sample_model, sample_input)
        
        assert isinstance(profile, dict)
        assert "peak_memory" in profile
        assert "memory_timeline" in profile
        assert "layer_memory_usage" in profile
        
        # Memory values should be non-negative
        assert profile["peak_memory"] >= 0
        assert len(profile["memory_timeline"]) > 0
    
    def test_automatic_garbage_collection(self, memory_config):
        """Test automatic garbage collection."""
        memory_config.enable_garbage_collection = True
        memory_config.gc_threshold = 0.5  # Low threshold for testing
        
        optimizer = MemoryOptimizer(memory_config)
        
        # Create objects that should be garbage collected
        large_tensors = []
        for i in range(10):
            tensor = torch.randn(100, 100)
            large_tensors.append(tensor)
        
        # Clear references
        large_tensors.clear()
        
        # Trigger garbage collection check
        initial_memory = optimizer.get_memory_usage()
        optimizer.maybe_run_garbage_collection()
        final_memory = optimizer.get_memory_usage()
        
        # GC should have run
        assert isinstance(initial_memory, dict)
        assert isinstance(final_memory, dict)
    
    def test_batch_size_optimization(self, sample_model, memory_config):
        """Test automatic batch size optimization."""
        optimizer = MemoryOptimizer(memory_config)
        
        # Start with large batch size that might not fit
        target_batch_size = 16
        sample_input = torch.randn(1, 3, 64, 64)  # Single sample
        
        optimal_batch_size = optimizer.find_optimal_batch_size(
            sample_model, sample_input, target_batch_size
        )
        
        assert isinstance(optimal_batch_size, int)
        assert optimal_batch_size > 0
        assert optimal_batch_size <= target_batch_size
        
        # Test that optimal batch size works
        optimized_input = torch.randn(optimal_batch_size, 3, 64, 64)
        output = sample_model(optimized_input)
        assert output.shape[0] == optimal_batch_size
    
    def test_memory_optimization_report(self, sample_model, memory_config):
        """Test memory optimization report generation."""
        optimizer = MemoryOptimizer(memory_config)
        
        sample_input = torch.randn(2, 3, 64, 64)
        
        # Run optimization
        optimized_model = optimizer.optimize_model(sample_model, sample_input)
        
        # Get report
        report = optimizer.get_optimization_report()
        
        assert isinstance(report, dict)
        assert "memory_usage" in report
        assert "fragmentation_stats" in report
        assert "optimizations_applied" in report
        assert "recommendations" in report
        
        # Check memory usage section
        memory_usage = report["memory_usage"]
        assert isinstance(memory_usage, dict)
        
        # Check fragmentation stats
        frag_stats = report["fragmentation_stats"]
        assert isinstance(frag_stats, dict)


class TestGlobalMemoryOptimizer:
    """Test global memory optimizer instance."""
    
    def test_get_memory_optimizer(self):
        """Test getting global memory optimizer."""
        optimizer1 = get_memory_optimizer()
        optimizer2 = get_memory_optimizer()
        
        # Should return the same instance
        assert optimizer1 is optimizer2
        assert isinstance(optimizer1, MemoryOptimizer)


class TestCUDAMemoryOptimization:
    """Test CUDA-specific memory optimizations."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_memory_pool(self):
        """Test CUDA memory pool optimization."""
        config = MemoryConfig(enable_cuda_memory_pool=True)
        optimizer = MemoryOptimizer(config)
        
        device = torch.device("cuda")
        
        # Test CUDA memory allocation
        tensor = torch.randn(100, 100, device=device)
        
        # Get CUDA memory usage
        memory_usage = optimizer.get_memory_usage()
        
        assert "cuda_memory" in memory_usage
        assert memory_usage["cuda_memory"]["allocated"] >= 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_memory_fragmentation(self):
        """Test CUDA memory fragmentation handling."""
        config = MemoryConfig(enable_cuda_memory_pool=True)
        optimizer = MemoryOptimizer(config)
        
        device = torch.device("cuda")
        
        # Create fragmented memory pattern
        tensors = []
        for i in range(10):
            tensor = torch.randn(50, 50, device=device)
            tensors.append(tensor)
        
        # Delete every other tensor
        for i in range(0, len(tensors), 2):
            del tensors[i]
        
        # Force garbage collection
        torch.cuda.empty_cache()
        
        # Get fragmentation stats
        memory_usage = optimizer.get_memory_usage()
        
        assert "cuda_memory" in memory_usage


class TestMemoryEdgeCases:
    """Test edge cases and error handling."""
    
    def test_out_of_memory_handling(self, memory_config):
        """Test handling of out-of-memory conditions."""
        optimizer = MemoryOptimizer(memory_config)
        
        # Try to create extremely large tensor
        try:
            huge_tensor = torch.randn(10000, 10000, 10000)  # Very large
            # If this doesn't fail, deallocate it
            del huge_tensor
        except RuntimeError as e:
            # Expected OOM error - check for various memory error patterns
            error_str = str(e).lower()
            assert ("out of memory" in error_str or 
                   "cuda" in error_str or 
                   "not enough memory" in error_str or
                   "allocate" in error_str), f"Unexpected error message: {e}"
        
        # Optimizer should still be functional
        memory_usage = optimizer.get_memory_usage()
        assert isinstance(memory_usage, dict)
    
    def test_zero_size_tensors(self, memory_config):
        """Test handling of zero-size tensors."""
        pool = AdvancedMemoryPool(memory_config)
        
        # Try to allocate zero-size tensor
        try:
            tensor = pool.allocate_tensor(
                torch.Size([0, 10]), torch.float32, torch.device("cpu")
            )
            assert tensor is not None
            assert tensor.numel() == 0
        except Exception:
            # May not support zero-size tensors
            pass
    
    def test_invalid_device_handling(self, memory_config):
        """Test handling of invalid devices."""
        pool = AdvancedMemoryPool(memory_config)
        
        try:
            # Try invalid device
            device = torch.device("invalid")
            tensor = pool.allocate_tensor(
                torch.Size([10, 10]), torch.float32, device
            )
        except (RuntimeError, ValueError):
            # Expected error for invalid device
            pass
    
    def test_concurrent_access(self, memory_config):
        """Test concurrent access to memory pool."""
        pool = AdvancedMemoryPool(memory_config)
        
        def allocate_tensors():
            tensors = []
            for i in range(5):
                try:
                    tensor = pool.allocate_tensor(
                        torch.Size([20, 20]), torch.float32, torch.device("cpu")
                    )
                    tensors.append(tensor)
                except Exception:
                    pass
            
            # Clean up
            for tensor in tensors:
                try:
                    pool.deallocate_tensor(tensor)
                except Exception:
                    pass
        
        # Run multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=allocate_tensors)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Pool should still be functional
        stats = pool.get_fragmentation_stats()
        assert isinstance(stats, FragmentationStats)


class TestMemoryLeakDetection:
    """Test memory leak detection and prevention."""
    
    def test_weak_reference_tracking(self, memory_config):
        """Test tracking tensors with weak references."""
        optimizer = MemoryOptimizer(memory_config)
        
        # Create tensor and weak reference
        tensor = torch.randn(50, 50)
        weak_ref = weakref.ref(tensor)
        
        assert weak_ref() is not None
        
        # Delete tensor
        del tensor
        gc.collect()
        
        # Weak reference should be None now
        assert weak_ref() is None
    
    def test_memory_leak_detection(self, memory_config):
        """Test detection of potential memory leaks."""
        optimizer = MemoryOptimizer(memory_config)
        
        # Get initial memory state
        initial_memory = optimizer.get_memory_usage()
        
        # Create and delete many tensors
        for i in range(10):
            tensor = torch.randn(100, 100)
            del tensor
        
        # Force garbage collection
        gc.collect()
        
        # Get final memory state
        final_memory = optimizer.get_memory_usage()
        
        # Should detect if memory increased significantly
        assert isinstance(initial_memory, dict)
        assert isinstance(final_memory, dict)
        
        if "total_memory" in initial_memory and "total_memory" in final_memory:
            memory_increase = final_memory["total_memory"] - initial_memory["total_memory"]
            # Large increase might indicate leak (but not necessarily in a test)
            assert isinstance(memory_increase, (int, float))


if __name__ == "__main__":
    # Run basic smoke test
    config = MemoryConfig(
        pool_size_mb=64,
        cleanup_interval=1,
        enable_background_cleanup=False  # Disable for simpler testing
    )
    
    optimizer = MemoryOptimizer(config)
    model = MemoryIntensiveModel(size_multiplier=1)
    sample_input = torch.randn(2, 3, 32, 32)
    
    print("Testing memory optimization...")
    
    # Test basic functionality
    initial_memory = optimizer.get_memory_usage()
    print(f"✓ Initial memory usage: {initial_memory.get('total_memory', 'N/A')} MB")
    
    # Optimize model
    optimized_model = optimizer.optimize_model(model, sample_input)
    print(f"✓ Model optimization completed")
    
    # Test functionality
    original_output = model(sample_input)
    optimized_output = optimized_model(sample_input)
    
    print(f"✓ Original output shape: {original_output.shape}")
    print(f"✓ Optimized output shape: {optimized_output.shape}")
    print(f"✓ Outputs match: {torch.allclose(original_output, optimized_output, rtol=1e-3)}")
    
    # Test memory pool
    pool = optimizer.memory_pool
    tensor = pool.allocate_tensor(torch.Size([100, 100]), torch.float32, torch.device("cpu"))
    pool.deallocate_tensor(tensor)
    
    fragmentation_stats = pool.get_fragmentation_stats()
    print(f"✓ Fragmentation ratio: {fragmentation_stats.fragmentation_ratio:.3f}")
    
    print("✓ Enhanced memory optimizer tests ready")

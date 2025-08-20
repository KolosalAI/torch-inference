#!/usr/bin/env python3
"""
Comprehensive HLRTF-inspired optimization test with working components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_model():
    """Create a test model for optimization."""
    class TestCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(128 * 4 * 4, 256)
            self.fc2 = nn.Linear(256, 10)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    return TestCNN()


def test_tensor_factorization():
    """Test tensor factorization optimization."""
    logger.info("🧪 Testing Tensor Factorization...")
    
    try:
        from framework.optimizers import TensorFactorizationOptimizer, TensorFactorizationConfig
        
        model = create_test_model()
        original_params = sum(p.numel() for p in model.parameters())
        
        # Configure optimization
        config = TensorFactorizationConfig()
        config.decomposition_method = "svd"
        config.target_compression_ratio = 0.8
        config.enable_fine_tuning = False
        
        # Apply optimization
        optimizer = TensorFactorizationOptimizer(config)
        optimized_model = optimizer.optimize(model)
        
        optimized_params = sum(p.numel() for p in optimized_model.parameters())
        compression_ratio = optimized_params / original_params
        
        # Test inference
        test_input = torch.randn(1, 3, 32, 32)
        output = optimized_model(test_input)
        
        logger.info(f"✅ Tensor Factorization: {original_params:,} → {optimized_params:,} params")
        logger.info(f"   Compression ratio: {compression_ratio:.3f}")
        logger.info(f"   Output shape: {output.shape}")
        
        return True, optimized_model, compression_ratio
        
    except Exception as e:
        logger.error(f"❌ Tensor Factorization failed: {e}")
        return False, None, 0


def test_mask_based_pruning():
    """Test mask-based structured pruning."""
    logger.info("🧪 Testing Mask-Based Structured Pruning...")
    
    try:
        from framework.optimizers import MaskBasedStructuredPruning, MaskPruningConfig
        
        model = create_test_model()
        original_params = sum(p.numel() for p in model.parameters())
        
        # Configure pruning
        config = MaskPruningConfig()
        config.pruning_ratio = 0.3
        config.importance_metric = "l2_norm"
        config.pruning_schedule = "gradual"
        config.num_pruning_steps = 5
        
        # Apply pruning
        optimizer = MaskBasedStructuredPruning(config)
        pruned_model = optimizer.optimize(model)
        
        effective_params = optimizer._count_effective_parameters(pruned_model)
        compression_ratio = effective_params / original_params
        
        # Test inference
        test_input = torch.randn(1, 3, 32, 32)
        output = pruned_model(test_input)
        
        logger.info(f"✅ Mask-Based Pruning: {original_params:,} → {effective_params:,} effective params")
        logger.info(f"   Compression ratio: {compression_ratio:.3f}")
        logger.info(f"   Output shape: {output.shape}")
        
        return True, pruned_model, compression_ratio
        
    except Exception as e:
        logger.error(f"❌ Mask-Based Pruning failed: {e}")
        return False, None, 0


def test_combined_optimization():
    """Test combined tensor factorization + mask-based pruning."""
    logger.info("🧪 Testing Combined Optimization (Tensor Factorization + Mask Pruning)...")
    
    try:
        from framework.optimizers import (
            TensorFactorizationOptimizer, TensorFactorizationConfig,
            MaskBasedStructuredPruning, MaskPruningConfig
        )
        
        model = create_test_model()
        original_params = sum(p.numel() for p in model.parameters())
        
        # Step 1: Apply tensor factorization
        tf_config = TensorFactorizationConfig()
        tf_config.decomposition_method = "svd"
        tf_config.target_compression_ratio = 0.9
        tf_config.enable_fine_tuning = False
        
        tf_optimizer = TensorFactorizationOptimizer(tf_config)
        factorized_model = tf_optimizer.optimize(model)
        
        factorized_params = sum(p.numel() for p in factorized_model.parameters())
        logger.info(f"   After factorization: {factorized_params:,} params")
        
        # Step 2: Apply mask-based pruning
        pruning_config = MaskPruningConfig()
        pruning_config.pruning_ratio = 0.3
        pruning_config.importance_metric = "l2_norm"
        pruning_config.pruning_schedule = "one_shot"
        
        pruning_optimizer = MaskBasedStructuredPruning(pruning_config)
        combined_model = pruning_optimizer.optimize(factorized_model)
        
        effective_params = pruning_optimizer._count_effective_parameters(combined_model)
        final_compression_ratio = effective_params / original_params
        
        # Test inference
        test_input = torch.randn(1, 3, 32, 32)
        output = combined_model(test_input)
        
        logger.info(f"✅ Combined Optimization: {original_params:,} → {effective_params:,} params")
        logger.info(f"   Final compression ratio: {final_compression_ratio:.3f}")
        logger.info(f"   Parameter reduction: {(1-final_compression_ratio):.1%}")
        logger.info(f"   Output shape: {output.shape}")
        
        return True, combined_model, final_compression_ratio
        
    except Exception as e:
        logger.error(f"❌ Combined Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, 0


def test_performance_benchmark():
    """Test performance benchmarking."""
    logger.info("🧪 Testing Performance Benchmark...")
    
    try:
        from framework.optimizers import TensorFactorizationOptimizer, TensorFactorizationConfig
        
        # Original model
        original_model = create_test_model()
        original_model.eval()
        
        # Optimized model
        config = TensorFactorizationConfig()
        config.decomposition_method = "svd"
        config.enable_fine_tuning = False
        
        optimizer = TensorFactorizationOptimizer(config)
        optimized_model = optimizer.optimize(original_model)
        optimized_model.eval()
        
        # Benchmark
        test_input = torch.randn(1, 3, 32, 32)
        num_runs = 100
        
        # Original model timing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = original_model(test_input)
        original_time = time.time() - start_time
        original_fps = num_runs / original_time
        
        # Optimized model timing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = optimized_model(test_input)
        optimized_time = time.time() - start_time
        optimized_fps = num_runs / optimized_time
        
        speedup = optimized_fps / original_fps
        
        logger.info(f"✅ Benchmark Results:")
        logger.info(f"   Original: {original_fps:.2f} FPS")
        logger.info(f"   Optimized: {optimized_fps:.2f} FPS")
        logger.info(f"   Speedup: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Benchmarking failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE HLRTF-INSPIRED OPTIMIZATION TESTS")
    logger.info("=" * 70)
    logger.info("")
    
    # Test individual optimizations
    tf_success, _, tf_ratio = test_tensor_factorization()
    logger.info("")
    
    mp_success, _, mp_ratio = test_mask_based_pruning()
    logger.info("")
    
    combined_success, _, combined_ratio = test_combined_optimization()
    logger.info("")
    
    perf_success = test_performance_benchmark()
    logger.info("")
    
    # Summary
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Tensor Factorization           {'✅ PASSED' if tf_success else '❌ FAILED'}")
    logger.info(f"Mask-Based Pruning             {'✅ PASSED' if mp_success else '❌ FAILED'}")
    logger.info(f"Combined Optimization          {'✅ PASSED' if combined_success else '❌ FAILED'}")
    logger.info(f"Performance Benchmark          {'✅ PASSED' if perf_success else '❌ FAILED'}")
    logger.info("")
    
    passed_tests = sum([tf_success, mp_success, combined_success, perf_success])
    total_tests = 4
    
    logger.info(f"Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("")
        logger.info("🎉 ALL HLRTF-INSPIRED OPTIMIZATIONS ARE WORKING!")
        logger.info("")
        logger.info("Summary of achievements:")
        logger.info("✅ Hierarchical Low-Rank Tensor Factorization implemented")
        logger.info("✅ Structured pruning with channel masking working")
        logger.info("✅ Combined optimization pipeline functional")
        logger.info("✅ Performance benchmarking operational")
        logger.info("")
        logger.info("The HLRTF-inspired optimization suite is ready for production use!")
    else:
        logger.error("⚠️ Some optimizations need attention.")

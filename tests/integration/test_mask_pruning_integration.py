#!/usr/bin/env python3
"""
Test mask-based structured pruning integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

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


def test_mask_based_pruning():
    """Test mask-based structured pruning integration."""
    logger.info("Testing Mask-Based Structured Pruning integration...")
    
    try:
        from framework.optimizers import MaskBasedStructuredPruning, MaskPruningConfig, prune_model_with_masks
        
        model = create_test_model()
        original_params = sum(p.numel() for p in model.parameters())
        
        # Test input
        test_input = torch.randn(1, 3, 32, 32)
        
        # Test original model
        original_output = model(test_input)
        logger.info(f"Original model: {original_params:,} params, output shape: {original_output.shape}")
        
        # Test mask-based pruning with optimizer class
        config = MaskPruningConfig()
        config.pruning_ratio = 0.3
        config.importance_metric = "l2_norm"
        config.pruning_schedule = "one_shot"
        
        optimizer = MaskBasedStructuredPruning(config)
        pruned_model = optimizer.optimize(model)
        
        # Test pruned model
        pruned_output = pruned_model(test_input)
        effective_params = optimizer._count_effective_parameters(pruned_model)
        
        logger.info(f"✅ Mask-based pruning: {original_params:,} → {effective_params:,} effective params")
        logger.info(f"   Compression ratio: {effective_params/original_params:.3f}")
        logger.info(f"   Output shape: {pruned_output.shape}")
        
        # Test convenience function
        logger.info("Testing convenience function...")
        convenience_model = prune_model_with_masks(create_test_model(), config)
        convenience_output = convenience_model(test_input)
        logger.info(f"✅ Convenience function output shape: {convenience_output.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Mask-based pruning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradual_pruning():
    """Test gradual pruning schedule."""
    logger.info("Testing Gradual Pruning...")
    
    try:
        from framework.optimizers import MaskBasedStructuredPruning, MaskPruningConfig
        
        model = create_test_model()
        original_params = sum(p.numel() for p in model.parameters())
        
        # Configure gradual pruning
        config = MaskPruningConfig()
        config.pruning_ratio = 0.5
        config.pruning_schedule = "gradual"
        config.num_pruning_steps = 5
        
        optimizer = MaskBasedStructuredPruning(config)
        pruned_model = optimizer.optimize(model)
        
        # Test forward pass
        test_input = torch.randn(1, 3, 32, 32)
        output = pruned_model(test_input)
        effective_params = optimizer._count_effective_parameters(pruned_model)
        
        logger.info(f"✅ Gradual pruning: {original_params:,} → {effective_params:,} effective params")
        logger.info(f"   Compression ratio: {effective_params/original_params:.3f}")
        logger.info(f"   Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Gradual pruning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("MASK-BASED STRUCTURED PRUNING INTEGRATION TESTS")
    logger.info("=" * 60)
    
    # Test basic mask-based pruning
    success1 = test_mask_based_pruning()
    
    logger.info("")
    
    # Test gradual pruning
    success2 = test_gradual_pruning()
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Mask-based Pruning      {'✅ PASSED' if success1 else '❌ FAILED'}")
    logger.info(f"Gradual Pruning         {'✅ PASSED' if success2 else '❌ FAILED'}")
    logger.info("")
    
    if success1 and success2:
        logger.info("🎉 All mask-based pruning tests passed!")
    else:
        logger.error("⚠️ Some tests failed!")

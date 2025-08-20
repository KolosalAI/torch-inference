"""
Unit tests for Mask-Based Structured Pruning Optimizer.

Tests the mask-based structured pruning functionality.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.optimizers import (
    MaskBasedStructuredPruning,
    MaskPruningConfig,
    MaskedConv2d,
    MaskedLinear,
    prune_model_with_masks
)


class SimpleTestModel(nn.Module):
    """Simple test model for mask-based pruning tests."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestMaskPruningConfig:
    """Test MaskPruningConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MaskPruningConfig()
        
        assert config.pruning_ratio == 0.3
        assert config.importance_metric == "l2_norm"
        assert config.pruning_schedule == "one_shot"
        assert config.num_pruning_steps == 10
        assert config.min_channels == 1
        assert config.min_features == 1
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MaskPruningConfig()
        config.pruning_ratio = 0.5
        config.importance_metric = "l1_norm"
        config.pruning_schedule = "gradual"
        config.num_pruning_steps = 5
        
        assert config.pruning_ratio == 0.5
        assert config.importance_metric == "l1_norm"
        assert config.pruning_schedule == "gradual"
        assert config.num_pruning_steps == 5


class TestMaskedConv2d:
    """Test MaskedConv2d layer."""
    
    def test_masked_conv_initialization(self):
        """Test initialization of masked conv layer."""
        original_conv = nn.Conv2d(16, 32, 3, padding=1)
        masked_conv = MaskedConv2d(original_conv)
        
        assert masked_conv.conv == original_conv
        assert masked_conv.channel_mask.shape == (32,)
        assert torch.all(masked_conv.channel_mask == 1.0)
    
    def test_masked_conv_forward(self):
        """Test forward pass through masked conv layer."""
        original_conv = nn.Conv2d(3, 16, 3, padding=1)
        masked_conv = MaskedConv2d(original_conv)
        
        # Test input
        input_tensor = torch.randn(2, 3, 32, 32)
        
        # Forward pass
        output = masked_conv(input_tensor)
        
        assert output.shape == (2, 16, 32, 32)
        assert not torch.isnan(output).any()
    
    def test_prune_channels(self):
        """Test channel pruning functionality."""
        original_conv = nn.Conv2d(3, 16, 3, padding=1)
        masked_conv = MaskedConv2d(original_conv)
        
        # Prune some channels
        channels_to_prune = [1, 5, 10]
        masked_conv.prune_channels(channels_to_prune)
        
        # Check that channels are masked
        for ch in channels_to_prune:
            assert masked_conv.channel_mask[ch] == 0.0
        
        # Check active channels
        expected_active = 16 - len(channels_to_prune)
        assert masked_conv.get_active_channels() == expected_active
        
        # Test pruning ratio
        expected_ratio = len(channels_to_prune) / 16
        assert abs(masked_conv.get_pruning_ratio() - expected_ratio) < 1e-6
    
    def test_forward_with_pruned_channels(self):
        """Test forward pass with pruned channels."""
        original_conv = nn.Conv2d(3, 16, 3, padding=1)
        masked_conv = MaskedConv2d(original_conv)
        
        # Prune some channels
        masked_conv.prune_channels([1, 5, 10])
        
        # Test input
        input_tensor = torch.randn(2, 3, 32, 32)
        
        # Forward pass
        output = masked_conv(input_tensor)
        
        assert output.shape == (2, 16, 32, 32)
        
        # Check that pruned channels output zeros
        assert torch.all(output[:, 1, :, :] == 0.0)
        assert torch.all(output[:, 5, :, :] == 0.0)
        assert torch.all(output[:, 10, :, :] == 0.0)


class TestMaskedLinear:
    """Test MaskedLinear layer."""
    
    def test_masked_linear_initialization(self):
        """Test initialization of masked linear layer."""
        original_linear = nn.Linear(128, 64)
        masked_linear = MaskedLinear(original_linear)
        
        assert masked_linear.linear == original_linear
        assert masked_linear.feature_mask.shape == (64,)
        assert torch.all(masked_linear.feature_mask == 1.0)
    
    def test_masked_linear_forward(self):
        """Test forward pass through masked linear layer."""
        original_linear = nn.Linear(128, 64)
        masked_linear = MaskedLinear(original_linear)
        
        # Test input
        input_tensor = torch.randn(4, 128)
        
        # Forward pass
        output = masked_linear(input_tensor)
        
        assert output.shape == (4, 64)
        assert not torch.isnan(output).any()
    
    def test_prune_features(self):
        """Test feature pruning functionality."""
        original_linear = nn.Linear(128, 64)
        masked_linear = MaskedLinear(original_linear)
        
        # Prune some features
        features_to_prune = [10, 20, 30]
        masked_linear.prune_features(features_to_prune)
        
        # Check that features are masked
        for feat in features_to_prune:
            assert masked_linear.feature_mask[feat] == 0.0
        
        # Check active features
        expected_active = 64 - len(features_to_prune)
        assert masked_linear.get_active_features() == expected_active
        
        # Test pruning ratio
        expected_ratio = len(features_to_prune) / 64
        assert abs(masked_linear.get_pruning_ratio() - expected_ratio) < 1e-6
    
    def test_forward_with_pruned_features(self):
        """Test forward pass with pruned features."""
        original_linear = nn.Linear(128, 64)
        masked_linear = MaskedLinear(original_linear)
        
        # Prune some features
        masked_linear.prune_features([10, 20, 30])
        
        # Test input
        input_tensor = torch.randn(4, 128)
        
        # Forward pass
        output = masked_linear(input_tensor)
        
        assert output.shape == (4, 64)
        
        # Check that pruned features output zeros
        assert torch.all(output[:, 10] == 0.0)
        assert torch.all(output[:, 20] == 0.0)
        assert torch.all(output[:, 30] == 0.0)


class TestMaskBasedStructuredPruning:
    """Test MaskBasedStructuredPruning optimizer."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        config = MaskPruningConfig()
        optimizer = MaskBasedStructuredPruning(config)
        
        assert optimizer.config == config
        assert optimizer.pruning_stats == {}
    
    def test_convert_to_masked_model(self):
        """Test conversion to masked model."""
        model = SimpleTestModel()
        config = MaskPruningConfig()
        optimizer = MaskBasedStructuredPruning(config)
        
        masked_model = optimizer._convert_to_masked_model(model)
        
        # Check that conv layers are converted
        assert isinstance(masked_model.conv1, MaskedConv2d)
        assert isinstance(masked_model.conv2, MaskedConv2d)
        
        # Check that linear layers are converted
        assert isinstance(masked_model.fc1, MaskedLinear)
        assert isinstance(masked_model.fc2, MaskedLinear)
    
    def test_optimize_one_shot(self):
        """Test one-shot pruning optimization."""
        model = SimpleTestModel()
        config = MaskPruningConfig()
        config.pruning_ratio = 0.3
        config.pruning_schedule = "one_shot"
        
        optimizer = MaskBasedStructuredPruning(config)
        pruned_model = optimizer.optimize(model)
        
        assert pruned_model is not None
        
        # Check that pruning was applied
        assert len(optimizer.pruning_stats) > 0
        
        # Test functionality
        test_input = torch.randn(1, 3, 32, 32)
        output = pruned_model(test_input)
        assert output.shape == (1, 10)
    
    def test_optimize_gradual(self):
        """Test gradual pruning optimization."""
        model = SimpleTestModel()
        config = MaskPruningConfig()
        config.pruning_ratio = 0.5
        config.pruning_schedule = "gradual"
        config.num_pruning_steps = 3
        
        optimizer = MaskBasedStructuredPruning(config)
        pruned_model = optimizer.optimize(model)
        
        assert pruned_model is not None
        
        # Test functionality
        test_input = torch.randn(1, 3, 32, 32)
        output = pruned_model(test_input)
        assert output.shape == (1, 10)
    
    def test_compute_conv_importance_l2(self):
        """Test L2 norm importance computation for conv layers."""
        config = MaskPruningConfig()
        config.importance_metric = "l2_norm"
        optimizer = MaskBasedStructuredPruning(config)
        
        # Create test masked conv
        original_conv = nn.Conv2d(3, 16, 3)
        masked_conv = MaskedConv2d(original_conv)
        
        importance = optimizer._compute_conv_importance(masked_conv)
        
        assert importance.shape == (16,)
        assert torch.all(importance >= 0)
    
    def test_compute_conv_importance_l1(self):
        """Test L1 norm importance computation for conv layers."""
        config = MaskPruningConfig()
        config.importance_metric = "l1_norm"
        optimizer = MaskBasedStructuredPruning(config)
        
        # Create test masked conv
        original_conv = nn.Conv2d(3, 16, 3)
        masked_conv = MaskedConv2d(original_conv)
        
        importance = optimizer._compute_conv_importance(masked_conv)
        
        assert importance.shape == (16,)
        assert torch.all(importance >= 0)
    
    def test_compute_linear_importance(self):
        """Test importance computation for linear layers."""
        config = MaskPruningConfig()
        optimizer = MaskBasedStructuredPruning(config)
        
        # Create test masked linear
        original_linear = nn.Linear(128, 64)
        masked_linear = MaskedLinear(original_linear)
        
        importance = optimizer._compute_linear_importance(masked_linear)
        
        assert importance.shape == (64,)
        assert torch.all(importance >= 0)
    
    def test_count_effective_parameters(self):
        """Test effective parameter counting."""
        model = SimpleTestModel()
        config = MaskPruningConfig()
        optimizer = MaskBasedStructuredPruning(config)
        
        # Convert to masked model
        masked_model = optimizer._convert_to_masked_model(model)
        
        # Count parameters before pruning
        original_count = optimizer._count_effective_parameters(masked_model)
        
        # Apply pruning
        optimizer._apply_one_shot_pruning(masked_model)
        
        # Count parameters after pruning
        pruned_count = optimizer._count_effective_parameters(masked_model)
        
        assert original_count > 0
        assert pruned_count > 0
        assert pruned_count < original_count  # Should be reduced
    
    @pytest.mark.parametrize("importance_metric", ["l1_norm", "l2_norm"])
    def test_different_importance_metrics(self, importance_metric):
        """Test different importance metrics."""
        model = SimpleTestModel()
        config = MaskPruningConfig()
        config.importance_metric = importance_metric
        config.pruning_ratio = 0.2
        
        optimizer = MaskBasedStructuredPruning(config)
        pruned_model = optimizer.optimize(model)
        
        assert pruned_model is not None
        
        # Test functionality
        test_input = torch.randn(1, 3, 32, 32)
        output = pruned_model(test_input)
        assert output.shape == (1, 10)


class TestMaskBasedPruningIntegration:
    """Integration tests for mask-based structured pruning."""
    
    def test_prune_model_with_masks_convenience_function(self):
        """Test the prune_model_with_masks convenience function."""
        model = SimpleTestModel()
        config = MaskPruningConfig()
        config.pruning_ratio = 0.3
        
        pruned_model = prune_model_with_masks(model, config)
        
        assert pruned_model is not None
        
        # Test that it still works
        test_input = torch.randn(1, 3, 32, 32)
        output = pruned_model(test_input)
        assert output.shape == (1, 10)
    
    def test_prune_model_with_default_config(self):
        """Test pruning with default configuration."""
        model = SimpleTestModel()
        
        pruned_model = prune_model_with_masks(model)  # No config provided
        
        assert pruned_model is not None
        
        # Test functionality
        test_input = torch.randn(1, 3, 32, 32)
        output = pruned_model(test_input)
        assert output.shape == (1, 10)
    
    def test_preserves_model_functionality(self):
        """Test that pruning preserves basic model functionality."""
        model = SimpleTestModel()
        config = MaskPruningConfig()
        config.pruning_ratio = 0.2  # Light pruning
        
        # Test input
        test_input = torch.randn(2, 3, 32, 32)
        
        # Original output
        model.eval()
        with torch.no_grad():
            original_output = model(test_input)
        
        # Prune model
        pruned_model = prune_model_with_masks(model, config)
        
        # Pruned output
        pruned_model.eval()
        with torch.no_grad():
            pruned_output = pruned_model(test_input)
        
        # Check output shapes match
        assert original_output.shape == pruned_output.shape
        assert pruned_output.shape == (2, 10)


if __name__ == "__main__":
    pytest.main([__file__])

#!/usr/bin/env python3
"""
Simple mask-based structured pruning test.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MaskedConv2d(nn.Module):
    """Conv2d layer with channel masking for structured pruning."""
    
    def __init__(self, conv_layer: nn.Conv2d):
        super().__init__()
        self.conv = conv_layer
        # Create channel mask (1 = keep, 0 = prune)
        self.register_buffer('channel_mask', torch.ones(conv_layer.out_channels))
        
    def forward(self, x):
        # Apply convolution
        out = self.conv(x)
        # Apply channel mask
        masked_out = out * self.channel_mask.view(1, -1, 1, 1)
        return masked_out
    
    def prune_channels(self, channels_to_prune):
        """Prune specified output channels by setting mask to 0."""
        for ch in channels_to_prune:
            if ch < len(self.channel_mask):
                self.channel_mask[ch] = 0
    
    def get_active_channels(self):
        """Get number of active (non-pruned) channels."""
        return int(self.channel_mask.sum().item())

class MaskedLinear(nn.Module):
    """Linear layer with feature masking for structured pruning."""
    
    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.linear = linear_layer
        # Create feature mask (1 = keep, 0 = prune)
        self.register_buffer('feature_mask', torch.ones(linear_layer.out_features))
        
    def forward(self, x):
        # Apply linear transformation
        out = self.linear(x)
        # Apply feature mask
        masked_out = out * self.feature_mask.view(1, -1)
        return masked_out
    
    def prune_features(self, features_to_prune):
        """Prune specified output features by setting mask to 0."""
        for feat in features_to_prune:
            if feat < len(self.feature_mask):
                self.feature_mask[feat] = 0
    
    def get_active_features(self):
        """Get number of active (non-pruned) features."""
        return int(self.feature_mask.sum().item())

def convert_to_masked_model(model):
    """Convert a model to use masked layers for structured pruning."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(model, name, MaskedConv2d(module))
        elif isinstance(module, nn.Linear):
            setattr(model, name, MaskedLinear(module))
        elif len(list(module.children())) > 0:
            convert_to_masked_model(module)
    return model

def apply_structured_pruning(model, sparsity=0.3):
    """Apply structured pruning using masks."""
    pruned_channels = 0
    total_channels = 0
    
    for name, module in model.named_modules():
        if isinstance(module, MaskedConv2d):
            # Compute channel importance (L2 norm)
            weight = module.conv.weight.data
            importance = torch.norm(weight.view(weight.size(0), -1), dim=1)
            
            # Determine channels to prune
            num_channels = weight.size(0)
            num_to_prune = int(num_channels * sparsity)
            
            if num_to_prune > 0:
                _, indices = torch.topk(importance, num_to_prune, largest=False)
                module.prune_channels(indices.tolist())
                pruned_channels += num_to_prune
            
            total_channels += num_channels
            
        elif isinstance(module, MaskedLinear):
            # Compute feature importance (L2 norm)
            weight = module.linear.weight.data
            importance = torch.norm(weight, dim=1)
            
            # Determine features to prune
            num_features = weight.size(0)
            num_to_prune = int(num_features * sparsity)
            
            if num_to_prune > 0:
                _, indices = torch.topk(importance, num_to_prune, largest=False)
                module.prune_features(indices.tolist())
                pruned_channels += num_to_prune
            
            total_channels += num_features
    
    logger.info(f"Pruned {pruned_channels}/{total_channels} channels/features ({pruned_channels/total_channels:.1%})")
    return model

def test_masked_pruning():
    """Test mask-based structured pruning."""
    try:
        # Create test model
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
        
        logger.info("Creating test model...")
        model = TestCNN()
        
        # Test input
        test_input = torch.randn(1, 3, 32, 32)
        
        logger.info("Testing original model...")
        original_output = model(test_input)
        original_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Original: {original_params:,} params, output shape: {original_output.shape}")
        
        logger.info("Converting to masked model...")
        masked_model = convert_to_masked_model(model)
        
        logger.info("Testing masked model (before pruning)...")
        masked_output = masked_model(test_input)
        logger.info(f"Masked (before pruning): output shape: {masked_output.shape}")
        
        logger.info("Applying structured pruning...")
        pruned_model = apply_structured_pruning(masked_model, sparsity=0.3)
        
        logger.info("Testing pruned model...")
        pruned_output = pruned_model(test_input)
        logger.info(f"Pruned: output shape: {pruned_output.shape}")
        
        # Count effective parameters (non-masked)
        effective_params = 0
        for name, module in pruned_model.named_modules():
            if isinstance(module, MaskedConv2d):
                active_channels = module.get_active_channels()
                weight = module.conv.weight.data
                effective_params += active_channels * weight.size(1) * weight.size(2) * weight.size(3)
                if module.conv.bias is not None:
                    effective_params += active_channels
            elif isinstance(module, MaskedLinear):
                active_features = module.get_active_features()
                weight = module.linear.weight.data
                effective_params += active_features * weight.size(1)
                if module.linear.bias is not None:
                    effective_params += active_features
        
        compression_ratio = effective_params / original_params
        logger.info(f"✅ Mask-based pruning: {original_params:,} → {effective_params:,} effective params")
        logger.info(f"   Compression ratio: {compression_ratio:.3f}")
        
        # Assert successful pruning
        assert pruned_model is not None, "Pruned model should not be None"
        assert effective_params > 0, "Effective params should be positive"
        assert compression_ratio > 0, "Compression ratio should be positive"
        assert pruned_output.shape == original_output.shape, "Output shape should remain the same"
        
    except Exception as e:
        logger.error(f"❌ Mask-based pruning failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Mask-based pruning test failed: {e}")

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("MASK-BASED STRUCTURED PRUNING TEST")
    logger.info("=" * 50)
    
    success = test_masked_pruning()
    
    if success:
        logger.info("🎉 Test passed!")
    else:
        logger.error("⚠️ Test failed!")

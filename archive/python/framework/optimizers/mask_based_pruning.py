"""
Mask-based Structured Pruning Optimizer.

This module implements structured pruning using masks to preserve model architecture
while achieving compression through channel/feature masking.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MaskPruningConfig:
    """Configuration for mask-based structured pruning."""
    
    # Pruning strategy
    pruning_ratio: float = 0.3  # Fraction of channels/features to prune
    importance_metric: str = "l2_norm"  # "l2_norm", "l1_norm"
    pruning_schedule: str = "one_shot"  # "one_shot", "gradual"
    
    # Gradual pruning settings
    num_pruning_steps: int = 10
    
    # Minimum channels/features to keep
    min_channels: int = 1
    min_features: int = 1


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
    
    def prune_channels(self, channels_to_prune: List[int]):
        """Prune specified output channels by setting mask to 0."""
        for ch in channels_to_prune:
            if ch < len(self.channel_mask):
                self.channel_mask[ch] = 0
    
    def get_active_channels(self) -> int:
        """Get number of active (non-pruned) channels."""
        return int(self.channel_mask.sum().item())
    
    def get_pruning_ratio(self) -> float:
        """Get current pruning ratio."""
        total = len(self.channel_mask)
        active = self.get_active_channels()
        return 1.0 - (active / total)


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
    
    def prune_features(self, features_to_prune: List[int]):
        """Prune specified output features by setting mask to 0."""
        for feat in features_to_prune:
            if feat < len(self.feature_mask):
                self.feature_mask[feat] = 0
    
    def get_active_features(self) -> int:
        """Get number of active (non-pruned) features."""
        return int(self.feature_mask.sum().item())
    
    def get_pruning_ratio(self) -> float:
        """Get current pruning ratio."""
        total = len(self.feature_mask)
        active = self.get_active_features()
        return 1.0 - (active / total)


class MaskBasedStructuredPruning:
    """
    Mask-based structured pruning optimizer that preserves model architecture.
    
    Uses channel/feature masks to achieve structured pruning without modifying
    the underlying model structure, ensuring forward pass compatibility.
    """
    
    def __init__(self, config: Optional[MaskPruningConfig] = None):
        self.config = config or MaskPruningConfig()
        self.logger = logging.getLogger(__name__)
        self.pruning_stats = {}
        
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        """Apply mask-based structured pruning to the model."""
        
        self.logger.info("Starting mask-based structured pruning optimization")
        
        # Convert model to use masked layers
        masked_model = self._convert_to_masked_model(model)
        
        # Count parameters before pruning
        original_params = self._count_effective_parameters(masked_model)
        
        # Apply pruning
        if self.config.pruning_schedule == "one_shot":
            self._apply_one_shot_pruning(masked_model)
        else:
            self._apply_gradual_pruning(masked_model)
        
        # Count parameters after pruning
        pruned_params = self._count_effective_parameters(masked_model)
        
        # Log results
        self._log_pruning_results(original_params, pruned_params)
        
        return masked_model
    
    def _convert_to_masked_model(self, model: nn.Module) -> nn.Module:
        """Convert a model to use masked layers for structured pruning."""
        import copy
        masked_model = copy.deepcopy(model)
        
        def replace_layers(module):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Conv2d):
                    setattr(module, name, MaskedConv2d(child))
                elif isinstance(child, nn.Linear):
                    setattr(module, name, MaskedLinear(child))
                else:
                    replace_layers(child)
        
        replace_layers(masked_model)
        return masked_model
    
    def _apply_one_shot_pruning(self, model: nn.Module):
        """Apply one-shot structured pruning."""
        self.logger.info("Applying one-shot structured pruning")
        
        pruned_count = 0
        total_count = 0
        
        for name, module in model.named_modules():
            if isinstance(module, MaskedConv2d):
                # Compute channel importance
                importance = self._compute_conv_importance(module)
                
                # Determine channels to prune
                num_channels = len(importance)
                num_to_prune = int(num_channels * self.config.pruning_ratio)
                num_to_prune = min(num_to_prune, num_channels - self.config.min_channels)
                
                if num_to_prune > 0:
                    _, indices = torch.topk(importance, num_to_prune, largest=False)
                    module.prune_channels(indices.tolist())
                    pruned_count += num_to_prune
                
                total_count += num_channels
                
                # Store statistics
                self.pruning_stats[name] = {
                    'original_channels': num_channels,
                    'pruned_channels': num_to_prune,
                    'remaining_channels': num_channels - num_to_prune,
                    'sparsity': num_to_prune / num_channels
                }
                
            elif isinstance(module, MaskedLinear):
                # Compute feature importance
                importance = self._compute_linear_importance(module)
                
                # Determine features to prune
                num_features = len(importance)
                num_to_prune = int(num_features * self.config.pruning_ratio)
                num_to_prune = min(num_to_prune, num_features - self.config.min_features)
                
                if num_to_prune > 0:
                    _, indices = torch.topk(importance, num_to_prune, largest=False)
                    module.prune_features(indices.tolist())
                    pruned_count += num_to_prune
                
                total_count += num_features
                
                # Store statistics
                self.pruning_stats[name] = {
                    'original_features': num_features,
                    'pruned_features': num_to_prune,
                    'remaining_features': num_features - num_to_prune,
                    'sparsity': num_to_prune / num_features
                }
        
        self.logger.info(f"Structured pruning completed:")
        self.logger.info(f"  Total channels/features pruned: {pruned_count}")
        self.logger.info(f"  Overall sparsity: {pruned_count/total_count:.3f}")
        self.logger.info(f"  Parameter reduction: {pruned_count/total_count:.1%}")
    
    def _apply_gradual_pruning(self, model: nn.Module):
        """Apply gradual structured pruning."""
        self.logger.info("Applying gradual structured pruning")
        
        for step in range(1, self.config.num_pruning_steps + 1):
            # Calculate target sparsity for this step
            target_sparsity = self.config.pruning_ratio * (step / self.config.num_pruning_steps) ** 3
            
            self.logger.info(f"Pruning step {step}/{self.config.num_pruning_steps}, target sparsity: {target_sparsity:.3f}")
            
            # Apply pruning for this step
            self._apply_step_pruning(model, target_sparsity)
    
    def _apply_step_pruning(self, model: nn.Module, target_sparsity: float):
        """Apply pruning for a single step."""
        for name, module in model.named_modules():
            if isinstance(module, MaskedConv2d):
                current_sparsity = module.get_pruning_ratio()
                if current_sparsity < target_sparsity:
                    # Calculate additional channels to prune
                    total_channels = len(module.channel_mask)
                    current_pruned = int(total_channels * current_sparsity)
                    target_pruned = int(total_channels * target_sparsity)
                    additional_to_prune = target_pruned - current_pruned
                    
                    if additional_to_prune > 0:
                        # Find least important active channels
                        active_mask = module.channel_mask > 0
                        if active_mask.sum() > additional_to_prune:
                            importance = self._compute_conv_importance(module)
                            # Only consider active channels
                            importance = importance * active_mask.float()
                            importance[~active_mask] = float('inf')  # Exclude already pruned
                            
                            _, indices = torch.topk(importance, additional_to_prune, largest=False)
                            module.prune_channels(indices.tolist())
                            
            elif isinstance(module, MaskedLinear):
                current_sparsity = module.get_pruning_ratio()
                if current_sparsity < target_sparsity:
                    # Calculate additional features to prune
                    total_features = len(module.feature_mask)
                    current_pruned = int(total_features * current_sparsity)
                    target_pruned = int(total_features * target_sparsity)
                    additional_to_prune = target_pruned - current_pruned
                    
                    if additional_to_prune > 0:
                        # Find least important active features
                        active_mask = module.feature_mask > 0
                        if active_mask.sum() > additional_to_prune:
                            importance = self._compute_linear_importance(module)
                            # Only consider active features
                            importance = importance * active_mask.float()
                            importance[~active_mask] = float('inf')  # Exclude already pruned
                            
                            _, indices = torch.topk(importance, additional_to_prune, largest=False)
                            module.prune_features(indices.tolist())
    
    def _compute_conv_importance(self, module: MaskedConv2d) -> torch.Tensor:
        """Compute importance scores for conv channels."""
        weight = module.conv.weight.data
        
        if self.config.importance_metric == "l1_norm":
            importance = torch.sum(torch.abs(weight.view(weight.size(0), -1)), dim=1)
        else:  # l2_norm
            importance = torch.norm(weight.view(weight.size(0), -1), dim=1)
        
        return importance
    
    def _compute_linear_importance(self, module: MaskedLinear) -> torch.Tensor:
        """Compute importance scores for linear features."""
        weight = module.linear.weight.data
        
        if self.config.importance_metric == "l1_norm":
            importance = torch.sum(torch.abs(weight), dim=1)
        else:  # l2_norm
            importance = torch.norm(weight, dim=1)
        
        return importance
    
    def _count_effective_parameters(self, model: nn.Module) -> int:
        """Count effective (non-masked) parameters."""
        effective_params = 0
        
        for module in model.modules():
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
                    
            elif isinstance(module, (nn.Conv2d, nn.Linear)):
                # Count regular layers that weren't converted
                for param in module.parameters():
                    effective_params += param.numel()
        
        return effective_params
    
    def _log_pruning_results(self, original_params: int, pruned_params: int):
        """Log pruning results."""
        compression_ratio = pruned_params / original_params
        reduction = 1.0 - compression_ratio
        
        self.logger.info("=" * 60)
        self.logger.info("MASK-BASED STRUCTURED PRUNING RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Original parameters: {original_params:,}")
        self.logger.info(f"Effective parameters: {pruned_params:,}")
        self.logger.info(f"Compression ratio: {compression_ratio:.3f}")
        self.logger.info(f"Parameter reduction: {reduction:.1%}")
        self.logger.info("=" * 60)


# Convenience function
def prune_model_with_masks(model: nn.Module, config: Optional[MaskPruningConfig] = None) -> nn.Module:
    """Convenience function to apply mask-based structured pruning."""
    optimizer = MaskBasedStructuredPruning(config)
    return optimizer.optimize(model)

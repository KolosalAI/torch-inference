"""
Structured Pruning Optimizer with Low-Rank Regularization.

This module implements structured pruning techniques with low-rank regularization
inspired by the HLRTF approach for neural network compression.

Key features:
- Channel-wise structured pruning
- Low-rank regularization during training
- Importance-based channel selection
- Gradual pruning with fine-tuning
- Hardware-efficient sparse patterns
"""

import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..core.config import InferenceConfig


class StructuredPruningConfig:
    """Configuration for structured pruning optimization."""
    
    def __init__(self):
        # General settings
        self.enabled = True
        self.target_sparsity = 0.5  # Target proportion of parameters to prune
        self.preserve_accuracy_threshold = 0.02  # Max acceptable accuracy loss
        
        # Pruning strategy
        self.pruning_method = "magnitude"  # magnitude, gradient, fisher, low_rank
        self.structured_type = "channel"  # channel, filter, block
        self.global_pruning = True  # Global vs layer-wise pruning
        self.gradual_pruning = True
        
        # Low-rank regularization (HLRTF-inspired)
        self.use_low_rank_regularization = True
        self.low_rank_weight = 0.001
        self.nuclear_norm_weight = 0.0001
        self.rank_constraint_weight = 0.01
        
        # Channel importance metrics
        self.importance_metric = "l2_norm"  # l2_norm, l1_norm, variance, gradient_based
        self.importance_accumulation = "mean"  # mean, max, sum
        self.use_batch_normalization_scaling = True
        
        # Gradual pruning schedule
        self.initial_sparsity = 0.0
        self.final_sparsity = 0.5
        self.pruning_steps = 10
        self.pruning_frequency = 100  # iterations between pruning steps
        
        # Fine-tuning settings
        self.enable_fine_tuning = True
        self.fine_tune_epochs = 5
        self.fine_tune_lr = 1e-4
        self.recovery_epochs = 3
        
        # Advanced options
        self.min_channels = 8  # Minimum channels to keep in any layer
        self.skip_depthwise_conv = True  # Skip depthwise convolutions
        self.preserve_first_last_layers = True
        self.use_knowledge_distillation = True
        self.distillation_temperature = 4.0
        self.distillation_alpha = 0.7


class ChannelImportanceCalculator:
    """Calculate channel importance for structured pruning."""
    
    def __init__(self, config: StructuredPruningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_importance(self, 
                           model: nn.Module, 
                           data_loader: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, torch.Tensor]:
        """
        Calculate channel importance scores for all layers.
        
        Args:
            model: PyTorch model
            data_loader: Optional data loader for gradient-based importance
            
        Returns:
            Dictionary mapping layer names to importance scores
        """
        importance_scores = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if self.config.importance_metric == "l2_norm":
                    scores = self._l2_norm_importance(module)
                elif self.config.importance_metric == "l1_norm":
                    scores = self._l1_norm_importance(module)
                elif self.config.importance_metric == "variance":
                    scores = self._variance_importance(module)
                elif self.config.importance_metric == "gradient_based" and data_loader:
                    scores = self._gradient_based_importance(model, name, data_loader)
                else:
                    scores = self._l2_norm_importance(module)  # fallback
                
                # Apply batch norm scaling if available
                if self.config.use_batch_normalization_scaling:
                    bn_scaling = self._get_bn_scaling(model, name)
                    if bn_scaling is not None:
                        # Ensure batch norm scaling matches the number of output channels
                        if len(bn_scaling) == len(scores):
                            scores = scores * bn_scaling
                        else:
                            self.logger.warning(f"Batch norm scaling size mismatch for layer {name}: "
                                              f"BN weights: {len(bn_scaling)}, scores: {len(scores)}. "
                                              f"Skipping BN scaling for this layer.")
                
                importance_scores[name] = scores
        
        return importance_scores
    
    def _l2_norm_importance(self, module: nn.Module) -> torch.Tensor:
        """Calculate L2 norm-based importance."""
        weight = module.weight.data
        
        if isinstance(module, nn.Conv2d):
            # For conv layers, compute norm across input channels, height, width
            importance = torch.norm(weight.reshape(weight.size(0), -1), dim=1)  # Flatten spatial dims
        else:  # Linear layer
            # For linear layers, compute norm across input features
            importance = torch.norm(weight, dim=1)  # Shape: [out_features]
        
        return importance
    
    def _l1_norm_importance(self, module: nn.Module) -> torch.Tensor:
        """Calculate L1 norm-based importance."""
        weight = module.weight.data
        
        if isinstance(module, nn.Conv2d):
            importance = torch.sum(torch.abs(weight), dim=(1, 2, 3))
        else:
            importance = torch.sum(torch.abs(weight), dim=1)
        
        return importance
    
    def _variance_importance(self, module: nn.Module) -> torch.Tensor:
        """Calculate variance-based importance."""
        weight = module.weight.data
        
        if isinstance(module, nn.Conv2d):
            importance = torch.var(weight, dim=(1, 2, 3))
        else:
            importance = torch.var(weight, dim=1)
        
        return importance
    
    def _gradient_based_importance(self, 
                                 model: nn.Module, 
                                 layer_name: str, 
                                 data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Calculate gradient-based importance using Fisher information."""
        model.eval()
        device = next(model.parameters()).device
        
        # Get the target layer
        target_layer = None
        for name, module in model.named_modules():
            if name == layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            return self._l2_norm_importance(target_layer)
        
        # Accumulate gradients
        gradient_accumulator = None
        num_samples = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= 10:  # Limit samples for efficiency
                break
            
            data, target = data.to(device), target.to(device)
            
            model.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            if target_layer.weight.grad is not None:
                grad = target_layer.weight.grad.data
                if isinstance(target_layer, nn.Conv2d):
                    grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1)
                else:
                    grad_norm = torch.norm(grad, dim=1)
                
                if gradient_accumulator is None:
                    gradient_accumulator = grad_norm.clone()
                else:
                    gradient_accumulator += grad_norm
                
                num_samples += data.size(0)
        
        if gradient_accumulator is not None:
            return gradient_accumulator / num_samples
        else:
            return self._l2_norm_importance(target_layer)
    
    def _get_bn_scaling(self, model: nn.Module, conv_name: str) -> Optional[torch.Tensor]:
        """Get batch normalization scaling factors for a convolutional layer."""
        # Look for the next batch norm layer
        modules = list(model.named_modules())
        conv_idx = None
        
        for i, (name, module) in enumerate(modules):
            if name == conv_name:
                conv_idx = i
                break
        
        if conv_idx is None:
            return None
        
        # Check the next few modules for BatchNorm
        for i in range(conv_idx + 1, min(conv_idx + 3, len(modules))):
            name, module = modules[i]
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    return torch.abs(module.weight.data)
        
        return None


class LowRankRegularizer:
    """Low-rank regularization inspired by HLRTF approach."""
    
    def __init__(self, config: StructuredPruningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def compute_low_rank_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Compute low-rank regularization loss for the model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Low-rank regularization loss
        """
        total_loss = 0.0
        num_layers = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                
                # Nuclear norm regularization (promotes low rank)
                nuclear_norm = torch.norm(weight.view(weight.size(0), -1), p='nuc')
                
                # Rank constraint regularization
                if isinstance(module, nn.Conv2d):
                    # For conv layers, reshape to 2D for SVD
                    weight_2d = weight.view(weight.size(0), -1)
                else:
                    weight_2d = weight
                
                # Compute approximate rank using SVD
                try:
                    _, S, _ = torch.svd(weight_2d)
                    rank_loss = torch.sum(S)  # Sum of singular values
                except:
                    rank_loss = torch.norm(weight_2d, p='fro')  # Fallback to Frobenius norm
                
                # Combine regularization terms
                layer_loss = (self.config.nuclear_norm_weight * nuclear_norm + 
                            self.config.rank_constraint_weight * rank_loss)
                
                total_loss += layer_loss
                num_layers += 1
        
        if num_layers > 0:
            return total_loss / num_layers
        else:
            return torch.tensor(0.0, requires_grad=True)


class StructuredPruningOptimizer:
    """
    Structured pruning optimizer with low-rank regularization.
    
    Implements channel-wise pruning with importance-based selection
    and low-rank regularization for improved compression.
    """
    
    def __init__(self, config: Optional[StructuredPruningConfig] = None):
        self.config = config or StructuredPruningConfig()
        self.logger = logging.getLogger(__name__)
        
        self.importance_calculator = ChannelImportanceCalculator(self.config)
        self.low_rank_regularizer = LowRankRegularizer(self.config)
        
        # Statistics
        self.pruning_stats = {}
        self.original_params = 0
        self.pruned_params = 0
    
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        """
        Main optimization method that applies structured pruning.
        
        Args:
            model: PyTorch model to optimize
            **kwargs: Additional optimization parameters
            
        Returns:
            Pruned model
        """
        self.logger.info("Starting structured pruning optimization")
        
        # Analyze model structure
        self._analyze_model(model)
        
        # Get data loader for importance calculation
        data_loader = kwargs.get('data_loader') or kwargs.get('train_loader')
        
        if self.config.gradual_pruning:
            pruned_model = self._gradual_pruning(model, data_loader)
        else:
            pruned_model = self._one_shot_pruning(model, data_loader)
        
        # Fine-tuning if enabled
        if self.config.enable_fine_tuning and data_loader:
            pruned_model = self._fine_tune_model(pruned_model, data_loader)
        
        # Log pruning results
        self._log_pruning_results()
        
        return pruned_model
    
    def _analyze_model(self, model: nn.Module):
        """Analyze model structure and determine pruning strategy."""
        self.original_params = sum(p.numel() for p in model.parameters())
        
        layer_info = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                params = sum(p.numel() for p in module.parameters())
                
                # Determine if layer should be pruned
                should_prune = True
                if self.config.preserve_first_last_layers:
                    # Simple heuristic: skip first and last layers
                    modules_list = list(model.named_modules())
                    if name == modules_list[1][0] or name == modules_list[-1][0]:
                        should_prune = False
                
                if isinstance(module, nn.Conv2d):
                    if module.groups == module.in_channels and self.config.skip_depthwise_conv:
                        should_prune = False  # Skip depthwise convolutions
                    if module.out_channels <= self.config.min_channels:
                        should_prune = False  # Skip if too few channels
                
                layer_info[name] = {
                    'type': type(module).__name__,
                    'params': params,
                    'shape': module.weight.shape,
                    'should_prune': should_prune,
                    'channels': module.out_channels if isinstance(module, nn.Conv2d) else module.out_features
                }
        
        self.layer_info = layer_info
        self.logger.info(f"Model analysis: {sum(1 for info in layer_info.values() if info['should_prune'])} "
                        f"prunable layers, {self.original_params:,} total parameters")
    
    def _gradual_pruning(self, model: nn.Module, data_loader) -> nn.Module:
        """Apply gradual structured pruning."""
        self.logger.info("Applying gradual structured pruning")
        
        import copy
        pruned_model = copy.deepcopy(model)
        
        # Calculate pruning schedule
        sparsity_schedule = self._get_sparsity_schedule()
        
        for step, target_sparsity in enumerate(sparsity_schedule):
            self.logger.info(f"Pruning step {step + 1}/{len(sparsity_schedule)}, "
                           f"target sparsity: {target_sparsity:.3f}")
            
            # Calculate importance scores
            importance_scores = self.importance_calculator.calculate_importance(
                pruned_model, data_loader
            )
            
            # Apply pruning
            self._apply_structured_pruning(pruned_model, importance_scores, target_sparsity)
            
            # Recovery training
            if data_loader:
                self._recovery_training(pruned_model, data_loader, self.config.recovery_epochs)
        
        return pruned_model
    
    def _one_shot_pruning(self, model: nn.Module, data_loader) -> nn.Module:
        """Apply one-shot structured pruning."""
        self.logger.info("Applying one-shot structured pruning")
        
        import copy
        pruned_model = copy.deepcopy(model)
        
        # Calculate importance scores
        importance_scores = self.importance_calculator.calculate_importance(
            pruned_model, data_loader
        )
        
        # Apply pruning
        self._apply_structured_pruning(pruned_model, importance_scores, self.config.target_sparsity)
        
        return pruned_model
    
    def _apply_structured_pruning(self, 
                                model: nn.Module, 
                                importance_scores: Dict[str, torch.Tensor], 
                                target_sparsity: float):
        """Apply structured pruning based on importance scores."""
        
        if self.config.global_pruning:
            # Global pruning: select channels across all layers
            self._global_channel_pruning(model, importance_scores, target_sparsity)
        else:
            # Layer-wise pruning: prune each layer independently
            self._layerwise_channel_pruning(model, importance_scores, target_sparsity)
    
    def _global_channel_pruning(self, 
                              model: nn.Module, 
                              importance_scores: Dict[str, torch.Tensor], 
                              target_sparsity: float):
        """Apply global channel pruning across all layers."""
        
        # Collect all importance scores
        all_scores = []
        layer_channel_info = []
        
        for name, scores in importance_scores.items():
            if name in self.layer_info and self.layer_info[name]['should_prune']:
                for i, score in enumerate(scores):
                    all_scores.append(score.item())
                    layer_channel_info.append((name, i))
        
        # Sort by importance
        sorted_indices = np.argsort(all_scores)
        
        # Determine channels to prune
        total_channels = len(all_scores)
        channels_to_prune = int(total_channels * target_sparsity)
        
        prune_mask = {}
        for i in range(channels_to_prune):
            layer_name, channel_idx = layer_channel_info[sorted_indices[i]]
            if layer_name not in prune_mask:
                prune_mask[layer_name] = []
            prune_mask[layer_name].append(channel_idx)
        
        # Apply pruning
        self._prune_channels(model, prune_mask)
    
    def _layerwise_channel_pruning(self, 
                                 model: nn.Module, 
                                 importance_scores: Dict[str, torch.Tensor], 
                                 target_sparsity: float):
        """Apply layer-wise channel pruning."""
        
        prune_mask = {}
        
        for name, scores in importance_scores.items():
            if name in self.layer_info and self.layer_info[name]['should_prune']:
                num_channels = len(scores)
                channels_to_prune = max(1, int(num_channels * target_sparsity))
                
                # Keep minimum number of channels
                channels_to_keep = num_channels - channels_to_prune
                if channels_to_keep < self.config.min_channels:
                    channels_to_prune = max(0, num_channels - self.config.min_channels)
                
                if channels_to_prune > 0:
                    _, sorted_indices = torch.sort(scores)
                    prune_mask[name] = sorted_indices[:channels_to_prune].tolist()
        
        # Apply pruning
        self._prune_channels(model, prune_mask)
        
        # After pruning, fix any linear layers that might have dimension mismatches
        self._fix_linear_layers_after_pruning(model)
    
    def _fix_linear_layers_after_pruning(self, model: nn.Module):
        """Fix all linear layers after pruning operations."""
        # Find all linear layers
        linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((name, module))
        
        # Update each linear layer
        for name, module in linear_layers:
            self._update_linear_layer_after_conv_pruning(model, module, name)
    
    def _prune_channels(self, model: nn.Module, prune_mask: Dict[str, List[int]]):
        """Prune channels from model layers (conservative approach)."""
        
        # Build dependencies first
        dependencies = self._build_layer_dependencies(model)
        
        for name, channel_indices in prune_mask.items():
            module = None
            for module_name, module_obj in model.named_modules():
                if module_name == name:
                    module = module_obj
                    break
            
            if module is None:
                continue
            
            # Create keep mask
            if isinstance(module, nn.Conv2d):
                total_channels = module.out_channels
                keep_indices = [i for i in range(total_channels) if i not in channel_indices]
                
                if len(keep_indices) == 0:
                    continue  # Skip if all channels would be pruned
                
                # Prune output channels
                module.weight.data = module.weight.data[keep_indices]
                if module.bias is not None:
                    module.bias.data = module.bias.data[keep_indices]
                
                # Update module parameters
                module.out_channels = len(keep_indices)
                
                # Update dependent layers
                self._update_dependent_layers(model, name, keep_indices, dependencies)
                
                # Store pruning statistics
                pruned_channels = len(channel_indices)
                self.pruning_stats[name] = {
                    'original_channels': total_channels,
                    'pruned_channels': pruned_channels,
                    'remaining_channels': len(keep_indices),
                    'sparsity': pruned_channels / total_channels
                }
                
            elif isinstance(module, nn.Linear):
                total_features = module.out_features
                keep_indices = [i for i in range(total_features) if i not in channel_indices]
                
                if len(keep_indices) == 0:
                    continue
                
                # Prune output features
                module.weight.data = module.weight.data[keep_indices]
                if module.bias is not None:
                    module.bias.data = module.bias.data[keep_indices]
                
                # Update module parameters
                module.out_features = len(keep_indices)
                
                # Update dependent layers (for linear layer outputs affecting next layers)
                self._update_dependent_layers(model, name, keep_indices, dependencies)
                
                # Store pruning statistics
                pruned_features = len(channel_indices)
                self.pruning_stats[name] = {
                    'original_features': total_features,
                    'pruned_features': pruned_features,
                    'remaining_features': len(keep_indices),
                    'sparsity': pruned_features / total_features
                }
    
    def _build_layer_dependencies(self, model: nn.Module) -> Dict[str, List[str]]:
        """Build a simple layer dependency graph for sequential models."""
        # This should map: layer_name -> [list of layers that DEPEND ON this layer]
        dependencies = {}
        layer_names = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layer_names.append(name)
        
        # For sequential models, when we prune a layer, we need to update the layers that come after it
        for i, name in enumerate(layer_names):
            dependencies[name] = []
            # Find layers that depend on this layer (layers that come after it)
            for j in range(i + 1, len(layer_names)):
                next_layer_name = layer_names[j]
                # Add the next layer as a dependent if there's a direct connection
                # For simplicity, we assume the immediate next layer depends on this one
                dependencies[name].append(next_layer_name)
                break  # Only add immediate next layer for now
        
        return dependencies
    
    def _update_dependent_layers(self, model: nn.Module, pruned_layer_name: str, 
                                keep_indices: List[int], dependencies: Dict[str, List[str]]):
        """Update input channels/features of dependent layers."""
        if pruned_layer_name not in dependencies:
            return
            
        for dependent_name in dependencies[pruned_layer_name]:
            dependent_module = None
            for module_name, module_obj in model.named_modules():
                if module_name == dependent_name:
                    dependent_module = module_obj
                    break
            
            if dependent_module is None:
                continue
            
            if isinstance(dependent_module, nn.Conv2d):
                # Update input channels
                old_weight = dependent_module.weight.data
                new_weight = old_weight[:, keep_indices, :, :]
                dependent_module.weight.data = new_weight
                dependent_module.in_channels = len(keep_indices)
                
            elif isinstance(dependent_module, nn.Linear):
                # For linear layers, we need to be more careful about input size
                # It might be the first linear layer after conv layers
                self._update_linear_layer_after_conv_pruning(model, dependent_module, dependent_name)
    
    def _update_linear_layer_after_conv_pruning(self, model: nn.Module, linear_layer: nn.Linear, layer_name: str):
        """Update linear layer dimensions after conv pruning by computing actual input size."""
        try:
            # Create a dummy input to trace through the model up to this linear layer
            dummy_input = torch.randn(1, 3, 32, 32)  # Standard test input
            
            # Trace through the model to find the actual input size to this linear layer
            actual_input_size = self._compute_linear_input_size_after_pruning(model, layer_name, dummy_input)
            
            if actual_input_size is None or actual_input_size == linear_layer.in_features:
                self.logger.info(f"No update needed for linear layer {layer_name}")
                return
                
            self.logger.info(f"Updating linear layer {layer_name}: {linear_layer.in_features} -> {actual_input_size} input features")
            
            # Create new linear layer with correct input size
            old_weight = linear_layer.weight.data
            old_bias = linear_layer.bias.data if linear_layer.bias is not None else None
            
            new_linear = nn.Linear(actual_input_size, linear_layer.out_features, 
                                 bias=linear_layer.bias is not None)
            
            # Initialize weights - truncate or pad as needed
            if actual_input_size <= old_weight.size(1):
                # Truncate weights
                new_linear.weight.data = old_weight[:, :actual_input_size]
            else:
                # Pad weights with zeros (shouldn't happen with pruning, but just in case)
                new_weight = torch.zeros(old_weight.size(0), actual_input_size)
                new_weight[:, :old_weight.size(1)] = old_weight
                new_linear.weight.data = new_weight
            
            if old_bias is not None:
                new_linear.bias.data = old_bias
            
            # Replace the layer in the model
            self._replace_module_in_model(model, layer_name, new_linear)
            self.logger.info(f"Successfully updated linear layer {layer_name}")
            
        except Exception as e:
            # If we can't compute the size, skip updating this layer
            self.logger.warning(f"Could not update linear layer {layer_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def _compute_linear_input_size_after_pruning(self, model: nn.Module, target_layer_name: str, dummy_input: torch.Tensor) -> int:
        """Compute the actual input size for a linear layer after pruning by forward tracing."""
        try:
            # Get all modules in order
            modules_list = list(model.named_modules())
            target_idx = None
            
            for i, (name, module) in enumerate(modules_list):
                if name == target_layer_name:
                    target_idx = i
                    break
            
            if target_idx is None:
                self.logger.warning(f"Target layer {target_layer_name} not found")
                return None
            
            # Forward trace through the model until we reach the target layer
            current_input = dummy_input
            model.eval()
            
            with torch.no_grad():
                for i, (name, module) in enumerate(modules_list):
                    if i >= target_idx:
                        break
                    
                    # Skip non-leaf modules (containers)
                    if len(list(module.children())) > 0:
                        continue
                    
                    # Apply the module
                    try:
                        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AdaptiveAvgPool2d, 
                                             nn.AvgPool2d, nn.ReLU, nn.Dropout, nn.BatchNorm2d, nn.BatchNorm1d)):
                            current_input = module(current_input)
                        elif hasattr(module, 'forward'):
                            current_input = module(current_input)
                    except Exception as e:
                        self.logger.warning(f"Error applying module {name}: {e}")
                        # Try to continue with a reasonable assumption
                        continue
            
            # At this point, current_input should be the input to the target linear layer
            # If it's still multi-dimensional, flatten it
            if len(current_input.shape) > 2:
                flattened_size = current_input.view(current_input.size(0), -1).size(1)
                self.logger.info(f"Computed flattened input size: {flattened_size}")
                return flattened_size
            else:
                self.logger.info(f"Computed input size: {current_input.size(1)}")
                return current_input.size(1)
                
        except Exception as e:
            self.logger.warning(f"Error computing input size for {target_layer_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _replace_module_in_model(self, model: nn.Module, module_name: str, new_module: nn.Module):
        """Replace a module in the model with a new module."""
        parts = module_name.split('.')
        current = model
        
        for part in parts[:-1]:
            current = getattr(current, part)
        
        setattr(current, parts[-1], new_module)
    
    def _get_sparsity_schedule(self) -> List[float]:
        """Get gradual pruning sparsity schedule."""
        if self.config.pruning_steps <= 1:
            return [self.config.final_sparsity]
        
        # Polynomial sparsity schedule
        schedule = []
        for step in range(self.config.pruning_steps):
            progress = step / (self.config.pruning_steps - 1)
            sparsity = (self.config.initial_sparsity + 
                       (self.config.final_sparsity - self.config.initial_sparsity) * 
                       (progress ** 3))  # Cubic schedule
            schedule.append(sparsity)
        
        return schedule
    
    def _recovery_training(self, model: nn.Module, data_loader, epochs: int):
        """Recovery training after pruning step."""
        device = next(model.parameters()).device
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.fine_tune_lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                
                # Main loss
                main_loss = criterion(output, target)
                
                # Add low-rank regularization
                if self.config.use_low_rank_regularization:
                    low_rank_loss = self.low_rank_regularizer.compute_low_rank_loss(model)
                    total_loss_val = main_loss + self.config.low_rank_weight * low_rank_loss
                else:
                    total_loss_val = main_loss
                
                total_loss_val.backward()
                optimizer.step()
                
                total_loss += total_loss_val.item()
                
                if batch_idx >= 50:  # Limit recovery training
                    break
            
            avg_loss = total_loss / min(51, len(data_loader))
            self.logger.debug(f"Recovery Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        
        model.eval()
    
    def _fine_tune_model(self, model: nn.Module, data_loader) -> nn.Module:
        """Fine-tune the pruned model to recover accuracy."""
        self.logger.info("Fine-tuning pruned model")
        
        device = next(model.parameters()).device
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.fine_tune_lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(self.config.fine_tune_epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                
                # Main loss
                main_loss = criterion(output, target)
                
                # Add low-rank regularization
                if self.config.use_low_rank_regularization:
                    low_rank_loss = self.low_rank_regularizer.compute_low_rank_loss(model)
                    total_loss_val = main_loss + self.config.low_rank_weight * low_rank_loss
                else:
                    total_loss_val = main_loss
                
                total_loss_val.backward()
                optimizer.step()
                
                total_loss += total_loss_val.item()
                
                if batch_idx % 100 == 0:
                    self.logger.debug(f"Fine-tune Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss_val.item():.4f}")
            
            avg_loss = total_loss / len(data_loader)
            self.logger.info(f"Fine-tune Epoch {epoch + 1}/{self.config.fine_tune_epochs}, Avg Loss: {avg_loss:.4f}")
        
        model.eval()
        return model
    
    def _log_pruning_results(self):
        """Log pruning statistics."""
        if not self.pruning_stats:
            return
        
        total_original_channels = sum(
            stats.get('original_channels', stats.get('original_features', 0)) 
            for stats in self.pruning_stats.values()
        )
        total_pruned_channels = sum(
            stats.get('pruned_channels', stats.get('pruned_features', 0)) 
            for stats in self.pruning_stats.values()
        )
        
        overall_sparsity = total_pruned_channels / total_original_channels if total_original_channels > 0 else 0.0
        
        self.logger.info(f"Structured pruning completed:")
        self.logger.info(f"  Layers pruned: {len(self.pruning_stats)}")
        self.logger.info(f"  Total channels/features pruned: {total_pruned_channels:,}")
        self.logger.info(f"  Overall sparsity: {overall_sparsity:.3f}")
        self.logger.info(f"  Parameter reduction: {overall_sparsity * 100:.1f}%")
    
    def benchmark_pruning(self,
                         original_model: nn.Module,
                         pruned_model: nn.Module,
                         example_inputs: torch.Tensor,
                         iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark the pruned model against the original.
        
        Args:
            original_model: Original model
            pruned_model: Pruned model
            example_inputs: Sample inputs for benchmarking
            iterations: Number of benchmark iterations
            
        Returns:
            Benchmark results
        """
        self.logger.info("Benchmarking structured pruning")
        
        device = example_inputs.device
        original_model = original_model.to(device).eval()
        pruned_model = pruned_model.to(device).eval()
        
        # Benchmark original model
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = original_model(example_inputs)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        original_time = time.time() - start_time
        
        # Benchmark pruned model
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = pruned_model(example_inputs)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        pruned_time = time.time() - start_time
        
        # Calculate metrics
        original_fps = iterations / original_time
        pruned_fps = iterations / pruned_time
        speedup = original_time / pruned_time
        
        # Model size comparison
        original_params = sum(p.numel() for p in original_model.parameters())
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        compression_ratio = pruned_params / original_params
        
        # Accuracy comparison
        try:
            with torch.no_grad():
                original_out = original_model(example_inputs)
                pruned_out = pruned_model(example_inputs)
                
                mse = torch.mean((original_out - pruned_out) ** 2).item()
                mae = torch.mean(torch.abs(original_out - pruned_out)).item()
                cos_sim = F.cosine_similarity(
                    original_out.flatten(), pruned_out.flatten(), dim=0
                ).item()
                
                accuracy_metrics = {
                    "mse": mse,
                    "mae": mae,
                    "cosine_similarity": cos_sim
                }
        except Exception as e:
            self.logger.warning(f"Could not compute accuracy metrics: {e}")
            accuracy_metrics = {}
        
        results = {
            "iterations": iterations,
            "performance": {
                "original_time_s": original_time,
                "pruned_time_s": pruned_time,
                "original_fps": original_fps,
                "pruned_fps": pruned_fps,
                "speedup": speedup,
                "improvement_percent": (speedup - 1) * 100
            },
            "model_size": {
                "original_params": original_params,
                "pruned_params": pruned_params,
                "compression_ratio": compression_ratio,
                "size_reduction_percent": (1 - compression_ratio) * 100
            },
            "accuracy": accuracy_metrics,
            "pruning_stats": self.pruning_stats
        }
        
        self.logger.info(f"Pruning speedup: {speedup:.2f}x, "
                        f"Size reduction: {(1 - compression_ratio) * 100:.1f}%")
        
        return results


def prune_model(model: nn.Module,
               method: str = "magnitude",
               config: Optional[StructuredPruningConfig] = None,
               **kwargs) -> nn.Module:
    """
    Convenience function to apply structured pruning to a PyTorch model.
    
    Args:
        model: PyTorch model
        method: Pruning method ("magnitude", "gradient", "fisher", "low_rank")
        config: Pruning configuration
        **kwargs: Additional pruning arguments
        
    Returns:
        Pruned model
    """
    if config is None:
        config = StructuredPruningConfig()
    
    config.pruning_method = method
    
    optimizer = StructuredPruningOptimizer(config)
    return optimizer.optimize(model, **kwargs)


# Export classes and functions
__all__ = [
    'StructuredPruningOptimizer',
    'StructuredPruningConfig',
    'ChannelImportanceCalculator',
    'LowRankRegularizer',
    'prune_model'
]

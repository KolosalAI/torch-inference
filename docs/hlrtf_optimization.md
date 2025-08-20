# HLRTF-Inspired Model Optimization Guide

This guide explains the Hierarchical Low-Rank Tensor Factorization (HLRTF) inspired optimization techniques implemented in the PyTorch Inference Framework. These methods are based on advanced tensor decomposition and structured pruning techniques for neural network compression.

## Overview

The HLRTF-inspired optimization suite provides comprehensive model compression through:

1. **Hierarchical Tensor Factorization** - Low-rank decomposition of weight tensors
2. **Structured Pruning with Low-Rank Regularization** - Channel-wise pruning with regularization
3. **Comprehensive Model Compression** - Multi-objective optimization combining multiple techniques
4. **Knowledge Distillation** - Accuracy preservation through teacher-student training

## Key Features

### 🎯 Hierarchical Tensor Factorization

Based on the HLRTF paper, this technique decomposes neural network weights into hierarchical low-rank factors:

- **Multi-level Decomposition**: Applies factorization at multiple hierarchical levels
- **Adaptive Rank Selection**: Automatically determines optimal ranks for compression
- **Hardware-Aware Optimization**: Considers target hardware constraints
- **Fine-tuning Support**: Recovers accuracy through post-compression training

### 🔧 Structured Pruning with Low-Rank Regularization

Advanced pruning that removes entire channels/filters while maintaining model structure:

- **Channel Importance Metrics**: Multiple methods to assess channel importance
- **Low-Rank Regularization**: Promotes low-rank structure during pruning
- **Gradual Pruning**: Progressive sparsity increase for better accuracy preservation
- **Global vs Layer-wise**: Flexible pruning strategies

### 🎛️ Comprehensive Model Compression

Multi-objective optimization combining multiple compression techniques:

- **Multi-Method Integration**: Combines factorization, pruning, and quantization
- **Pareto Optimization**: Finds optimal trade-offs between size, speed, and accuracy
- **Progressive Compression**: Staged optimization for gradual compression
- **Validation-Guided**: Uses validation metrics to guide optimization

## Quick Start

### 1. Basic Tensor Factorization

```python
from framework.optimizers import factorize_model, TensorFactorizationConfig

# Simple usage
compressed_model = factorize_model(model, method="hlrtf")

# Advanced configuration
config = TensorFactorizationConfig()
config.decomposition_method = "hlrtf"
config.target_compression_ratio = 0.4  # 60% parameter reduction
config.hierarchical_levels = 3
config.enable_fine_tuning = True

from framework.optimizers import TensorFactorizationOptimizer
optimizer = TensorFactorizationOptimizer(config)
compressed_model = optimizer.optimize(model, train_loader=dataloader)
```

### 2. Structured Pruning

```python
from framework.optimizers import prune_model, StructuredPruningConfig

# Simple usage
pruned_model = prune_model(model, method="magnitude")

# Advanced configuration
config = StructuredPruningConfig()
config.target_sparsity = 0.5  # 50% sparsity
config.use_low_rank_regularization = True
config.gradual_pruning = True
config.pruning_steps = 5

from framework.optimizers import StructuredPruningOptimizer
optimizer = StructuredPruningOptimizer(config)
pruned_model = optimizer.optimize(model, data_loader=dataloader)
```

### 3. Comprehensive Compression

```python
from framework.optimizers import compress_model_comprehensive, ModelCompressionConfig, CompressionMethod

# Simple usage
compressed_model = compress_model_comprehensive(model)

# Advanced configuration
config = ModelCompressionConfig()
config.enabled_methods = [
    CompressionMethod.TENSOR_FACTORIZATION,
    CompressionMethod.STRUCTURED_PRUNING,
    CompressionMethod.QUANTIZATION
]
config.targets.target_size_ratio = 0.3  # 70% reduction
config.progressive_compression = True

from framework.optimizers import ModelCompressionSuite
suite = ModelCompressionSuite(config)
compressed_model = suite.compress_model(model, validation_fn=my_validation_function)
```

## Detailed Configuration

### Tensor Factorization Configuration

```python
class TensorFactorizationConfig:
    # General settings
    enabled = True
    target_compression_ratio = 0.5  # Target model size reduction
    preserve_accuracy_threshold = 0.02  # Max acceptable accuracy loss
    
    # Decomposition settings
    decomposition_method = "hlrtf"  # hlrtf, tucker, svd, adaptive
    auto_rank_selection = True
    rank_selection_method = "energy"  # energy, nuclear_norm, adaptive
    energy_threshold = 0.95  # For energy-based rank selection
    
    # Layer-specific settings
    conv_rank_ratio = 0.5  # Rank as ratio of original dimensions
    linear_rank_ratio = 0.5
    min_rank = 4  # Minimum rank for any decomposition
    skip_small_layers = True
    min_params = 1000
    
    # Hierarchical settings (HLRTF-specific)
    hierarchical_levels = 3
    level_compression_ratios = [0.8, 0.6, 0.4]  # Per-level compression
    inter_level_regularization = 0.001
    
    # Fine-tuning settings
    enable_fine_tuning = True
    fine_tune_epochs = 5
    fine_tune_lr = 1e-4
    progressive_unfreezing = True
```

### Structured Pruning Configuration

```python
class StructuredPruningConfig:
    # General settings
    enabled = True
    target_sparsity = 0.5  # Target proportion of parameters to prune
    preserve_accuracy_threshold = 0.02
    
    # Pruning strategy
    pruning_method = "magnitude"  # magnitude, gradient, fisher, low_rank
    structured_type = "channel"  # channel, filter, block
    global_pruning = True  # Global vs layer-wise pruning
    gradual_pruning = True
    
    # Low-rank regularization (HLRTF-inspired)
    use_low_rank_regularization = True
    low_rank_weight = 0.001
    nuclear_norm_weight = 0.0001
    rank_constraint_weight = 0.01
    
    # Channel importance metrics
    importance_metric = "l2_norm"  # l2_norm, l1_norm, variance, gradient_based
    importance_accumulation = "mean"  # mean, max, sum
    use_batch_normalization_scaling = True
    
    # Gradual pruning schedule
    initial_sparsity = 0.0
    final_sparsity = 0.5
    pruning_steps = 10
    pruning_frequency = 100  # iterations between pruning steps
```

### Comprehensive Compression Configuration

```python
class ModelCompressionConfig:
    # Compression strategy
    enabled_methods = [
        CompressionMethod.TENSOR_FACTORIZATION,
        CompressionMethod.STRUCTURED_PRUNING,
        CompressionMethod.QUANTIZATION
    ]
    compression_order = [
        CompressionMethod.TENSOR_FACTORIZATION,
        CompressionMethod.STRUCTURED_PRUNING,
        CompressionMethod.QUANTIZATION
    ]
    
    # Optimization targets
    targets = CompressionTarget(
        target_size_ratio=0.5,      # Target model size reduction
        target_speedup=2.0,         # Target inference speedup
        max_accuracy_loss=0.02,     # Maximum acceptable accuracy loss
        memory_budget_mb=100,       # Memory budget in MB
        latency_budget_ms=50        # Latency budget in milliseconds
    )
    
    # Multi-objective optimization
    use_multi_objective = True
    pareto_optimization = True
    optimization_iterations = 10
    
    # Progressive compression
    progressive_compression = True
    compression_stages = 3
    intermediate_validation = True
    
    # Knowledge distillation
    enable_knowledge_distillation = True
    distillation_temperature = 4.0
    distillation_alpha = 0.7
    distillation_epochs = 5
```

## Compression Methods

### 1. Hierarchical Low-Rank Tensor Factorization (HLRTF)

The HLRTF method decomposes weight tensors into multiple hierarchical levels:

**Level 1: Channel Decomposition**
- Decomposes input/output channel relationships
- Uses parametrized factorization: `A_hat @ B_hat`

**Level 2: Spatial Decomposition**
- Separates spatial convolution patterns
- Applies sequential 1×1 and spatial convolutions

**Level 3: Refinement**
- Fine-grained factorization for accuracy recovery
- Residual connections for information preservation

**Mathematical Foundation:**
```
W_original ≈ Σ_i (A_i ⊗ B_i ⊗ C_i)
```
Where `⊗` represents tensor product and `A_i`, `B_i`, `C_i` are low-rank factors.

### 2. Structured Pruning with Low-Rank Regularization

Channel importance calculation:
```python
# L2 norm importance
importance = torch.norm(weight, dim=(1, 2, 3))  # For conv layers

# Gradient-based importance (Fisher information)
importance = gradient_norm_accumulation / num_samples

# Low-rank regularization loss
nuclear_norm = torch.norm(weight_2d, p='nuc')
rank_loss = torch.sum(singular_values)
total_loss = main_loss + λ1 * nuclear_norm + λ2 * rank_loss
```

### 3. Multi-Objective Optimization

Optimization score calculation:
```python
score = (accuracy * w_acc + 
         (1 - size_ratio) * w_size + 
         (speedup - 1) * w_speed)
```

With constraints:
- Accuracy loss ≤ max_accuracy_loss
- Size ratio ≤ target_size_ratio
- Memory usage ≤ memory_budget

## Performance Optimization Tips

### 1. Optimal Configuration Selection

**For Maximum Compression:**
```python
config.target_compression_ratio = 0.2  # 80% reduction
config.hierarchical_levels = 4
config.gradual_pruning = True
config.pruning_steps = 10
```

**For Maximum Speed:**
```python
config.decomposition_method = "svd"  # Faster than HLRTF
config.structured_type = "channel"  # Hardware-friendly
config.enable_fusion_optimization = True
```

**For Maximum Accuracy:**
```python
config.preserve_accuracy_threshold = 0.01  # Strict threshold
config.enable_knowledge_distillation = True
config.fine_tune_epochs = 10
config.progressive_compression = True
```

### 2. Hardware-Specific Optimization

**For CPU Inference:**
```python
config.conv_rank_ratio = 0.3  # Aggressive compression
config.structured_type = "channel"
config.enable_quantization = True
```

**For GPU Inference:**
```python
config.conv_rank_ratio = 0.5  # Moderate compression
config.use_mixed_precision = True
config.enable_fusion_optimization = True
```

**For Mobile/Edge:**
```python
config.target_size_ratio = 0.2  # Maximum compression
config.memory_budget_mb = 50
config.latency_budget_ms = 20
```

### 3. Layer-Specific Strategies

**Skip Critical Layers:**
```python
config.preserve_first_last_layers = True
config.skip_depthwise_conv = True
config.min_channels = 8
```

**Adaptive Compression:**
```python
config.auto_rank_selection = True
config.rank_selection_method = "energy"
config.energy_threshold = 0.95
```

## Benchmarking and Evaluation

### Performance Metrics

The optimization methods provide comprehensive benchmarking:

```python
benchmark_results = optimizer.benchmark_compression(
    original_model, compressed_model, example_inputs
)

# Results include:
# - Performance: speedup, FPS improvement
# - Model size: parameter reduction, memory usage
# - Accuracy: MSE, MAE, cosine similarity
# - Compression stats: method-specific metrics
```

### Validation Functions

For multi-objective optimization:

```python
def validation_function(model):
    accuracy = evaluate_accuracy(model, val_loader)
    latency = measure_latency(model, example_inputs)
    memory = estimate_memory_usage(model)
    
    return {
        'accuracy': accuracy,
        'speedup': baseline_latency / latency,
        'size_ratio': model_size / original_size,
        'memory_mb': memory
    }
```

## Advanced Usage

### 1. Custom Hierarchical Layers

```python
from framework.optimizers import HierarchicalTensorLayer

# Create custom hierarchical factorization
hierarchical_layer = HierarchicalTensorLayer(
    original_layer=conv_layer,
    ranks=[64, 32, 16],
    hierarchical_levels=3
)
```

### 2. Progressive Compression with Validation

```python
config = ModelCompressionConfig()
config.progressive_compression = True
config.compression_stages = 5
config.intermediate_validation = True

def stage_validation(model, stage):
    accuracy = evaluate_model(model)
    if accuracy < threshold:
        return False  # Stop compression
    return True

suite = ModelCompressionSuite(config)
compressed_model = suite.compress_model(
    model, 
    validation_fn=validation_function,
    stage_validation_fn=stage_validation
)
```

### 3. Knowledge Distillation Training

```python
from framework.optimizers import KnowledgeDistillationTrainer

# Set up knowledge distillation
teacher_model = original_model
kd_trainer = KnowledgeDistillationTrainer(
    teacher_model, 
    temperature=4.0, 
    alpha=0.7
)

# Train compressed model with distillation
trained_model = kd_trainer.distill_knowledge(
    compressed_model, 
    train_loader, 
    epochs=10
)
```

## Integration with Existing Framework

### 1. With Base Model

```python
from framework.core.base_model import BaseModel
from framework.optimizers import compress_model_comprehensive

class OptimizedModel(BaseModel):
    def optimize_for_inference(self):
        super().optimize_for_inference()
        
        # Apply HLRTF-inspired compression
        self.model = compress_model_comprehensive(
            self.model,
            config=self.compression_config
        )
```

### 2. With Inference Engine

```python
from framework.optimizers import ModelCompressionSuite

# In model loading pipeline
compression_config = ModelCompressionConfig()
compression_suite = ModelCompressionSuite(compression_config)

def load_and_optimize_model(model_path):
    model = load_model(model_path)
    optimized_model = compression_suite.compress_model(model)
    return optimized_model
```

## Troubleshooting

### Common Issues

**1. Memory Issues During Compression:**
```python
# Reduce batch size or use checkpointing
config.use_gradient_checkpointing = True
config.fine_tune_epochs = 3  # Reduce training
```

**2. Accuracy Loss Too High:**
```python
# Increase fine-tuning or use knowledge distillation
config.enable_knowledge_distillation = True
config.fine_tune_epochs = 10
config.preserve_accuracy_threshold = 0.01
```

**3. Slow Compression:**
```python
# Use simpler methods or reduce iterations
config.decomposition_method = "svd"  # Faster than HLRTF
config.optimization_iterations = 5
config.compression_stages = 3
```

### Performance Tuning

**Monitor Compression Progress:**
```python
import logging
logging.getLogger('framework.optimizers').setLevel(logging.DEBUG)

# Detailed logs will show:
# - Compression ratios per layer
# - Accuracy metrics during training
# - Performance benchmarks
```

**Optimize for Target Hardware:**
```python
# Profile on target device
benchmark_results = suite.benchmark_compression(
    original_model, compressed_model, example_inputs,
    iterations=100
)

# Adjust configuration based on results
if benchmark_results['performance']['speedup'] < target_speedup:
    config.targets.target_size_ratio *= 0.8  # More aggressive
```

## References

1. **HLRTF Paper**: "Hierarchical Low-Rank Tensor Factorization for Inverse Problems in Multi-Dimensional Imaging," CVPR 2022
2. **Tensor Decomposition**: "Tensor-Train Decomposition" and "Tucker Decomposition"
3. **Structured Pruning**: "Learning Efficient Convolutional Networks through Network Slimming"
4. **Knowledge Distillation**: "Distilling the Knowledge in a Neural Network"

## Examples

See `examples/hlrtf_optimization_example.py` for comprehensive demonstrations of all features.

---

This guide provides a complete reference for using HLRTF-inspired model optimization techniques in the PyTorch Inference Framework. The methods offer state-of-the-art compression capabilities while maintaining model accuracy and inference speed.

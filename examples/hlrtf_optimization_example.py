"""
Example script demonstrating HLRTF-inspired model optimization techniques.

This script shows how to use the new optimization methods inspired by the 
"Hierarchical Low-Rank Tensor Factorization" paper to compress and accelerate
PyTorch models while maintaining accuracy.
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Import our HLRTF-inspired optimizers
from framework.optimizers import (
    TensorFactorizationOptimizer, 
    TensorFactorizationConfig,
    StructuredPruningOptimizer,
    StructuredPruningConfig,
    ModelCompressionSuite,
    ModelCompressionConfig,
    CompressionMethod,
    factorize_model,
    prune_model,
    compress_model_comprehensive
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExampleCNN(nn.Module):
    """Example CNN model for demonstration."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Convolutional blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def create_sample_data():
    """Create sample data for demonstration."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Use a small subset of CIFAR-10 for quick demonstration
    try:
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        # Use only a subset for faster processing
        subset_indices = list(range(0, 1000))
        dataset = Subset(dataset, subset_indices)
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        return dataloader
    except Exception as e:
        logger.warning(f"Could not load CIFAR-10 dataset: {e}")
        # Create dummy data if CIFAR-10 is not available
        dummy_data = torch.randn(100, 3, 32, 32)
        dummy_targets = torch.randint(0, 10, (100,))
        dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_targets)
        return DataLoader(dummy_dataset, batch_size=32, shuffle=True)


def evaluate_model_accuracy(model, dataloader, device):
    """Evaluate model accuracy on dataset."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= 10:  # Limit evaluation for speed
                break
            
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def benchmark_model(model, example_inputs, device, iterations=50):
    """Benchmark model inference speed."""
    model = model.to(device).eval()
    example_inputs = example_inputs.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(example_inputs)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(example_inputs)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    total_time = time.time() - start_time
    
    return total_time, iterations / total_time


def demonstrate_tensor_factorization():
    """Demonstrate tensor factorization optimization."""
    logger.info("="*60)
    logger.info("DEMONSTRATING TENSOR FACTORIZATION OPTIMIZATION")
    logger.info("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model and data
    model = ExampleCNN(num_classes=10)
    model = model.to(device)
    dataloader = create_sample_data()
    example_inputs = torch.randn(1, 3, 32, 32).to(device)
    
    # Original model statistics
    original_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Original model parameters: {original_params:,}")
    
    # Original accuracy and speed
    original_accuracy = evaluate_model_accuracy(model, dataloader, device)
    original_time, original_fps = benchmark_model(model, example_inputs, device)
    logger.info(f"Original accuracy: {original_accuracy:.3f}")
    logger.info(f"Original inference speed: {original_fps:.2f} FPS")
    
    # Configure tensor factorization
    config = TensorFactorizationConfig()
    config.decomposition_method = "hlrtf"  # Use HLRTF method
    config.target_compression_ratio = 0.4  # 60% parameter reduction
    config.hierarchical_levels = 3
    config.enable_fine_tuning = True
    config.fine_tune_epochs = 3
    
    # Apply tensor factorization
    logger.info("Applying hierarchical tensor factorization...")
    optimizer = TensorFactorizationOptimizer(config)
    factorized_model = optimizer.optimize(model, train_loader=dataloader)
    
    # Factorized model statistics
    factorized_params = sum(p.numel() for p in factorized_model.parameters())
    compression_ratio = factorized_params / original_params
    logger.info(f"Factorized model parameters: {factorized_params:,}")
    logger.info(f"Compression ratio: {compression_ratio:.3f}")
    logger.info(f"Parameter reduction: {(1 - compression_ratio) * 100:.1f}%")
    
    # Factorized accuracy and speed
    factorized_accuracy = evaluate_model_accuracy(factorized_model, dataloader, device)
    factorized_time, factorized_fps = benchmark_model(factorized_model, example_inputs, device)
    speedup = original_time / factorized_time
    
    logger.info(f"Factorized accuracy: {factorized_accuracy:.3f}")
    logger.info(f"Factorized inference speed: {factorized_fps:.2f} FPS")
    logger.info(f"Speedup: {speedup:.2f}x")
    logger.info(f"Accuracy loss: {(original_accuracy - factorized_accuracy) * 100:.2f}%")
    
    # Benchmark comparison
    logger.info("\nBenchmarking tensor factorization...")
    benchmark_results = optimizer.benchmark_factorization(
        model, factorized_model, example_inputs, iterations=50
    )
    
    logger.info("Tensor Factorization Results:")
    logger.info(f"  Performance improvement: {benchmark_results['performance']['improvement_percent']:.1f}%")
    logger.info(f"  Size reduction: {benchmark_results['model_size']['size_reduction_percent']:.1f}%")
    if 'accuracy' in benchmark_results:
        logger.info(f"  Cosine similarity: {benchmark_results['accuracy']['cosine_similarity']:.4f}")
    
    return model, factorized_model


def demonstrate_structured_pruning():
    """Demonstrate structured pruning with low-rank regularization."""
    logger.info("="*60)
    logger.info("DEMONSTRATING STRUCTURED PRUNING WITH LOW-RANK REGULARIZATION")
    logger.info("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model and data
    model = ExampleCNN(num_classes=10)
    model = model.to(device)
    dataloader = create_sample_data()
    example_inputs = torch.randn(1, 3, 32, 32).to(device)
    
    # Original model statistics
    original_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Original model parameters: {original_params:,}")
    
    # Configure structured pruning
    config = StructuredPruningConfig()
    config.target_sparsity = 0.5  # 50% sparsity
    config.pruning_method = "magnitude"
    config.use_low_rank_regularization = True
    config.gradual_pruning = True
    config.pruning_steps = 5
    config.enable_fine_tuning = True
    config.fine_tune_epochs = 3
    
    # Apply structured pruning
    logger.info("Applying structured pruning with low-rank regularization...")
    optimizer = StructuredPruningOptimizer(config)
    pruned_model = optimizer.optimize(model, data_loader=dataloader)
    
    # Pruned model statistics
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    compression_ratio = pruned_params / original_params
    logger.info(f"Pruned model parameters: {pruned_params:,}")
    logger.info(f"Compression ratio: {compression_ratio:.3f}")
    logger.info(f"Parameter reduction: {(1 - compression_ratio) * 100:.1f}%")
    
    # Benchmark comparison
    logger.info("\nBenchmarking structured pruning...")
    benchmark_results = optimizer.benchmark_pruning(
        model, pruned_model, example_inputs, iterations=50
    )
    
    logger.info("Structured Pruning Results:")
    logger.info(f"  Performance improvement: {benchmark_results['performance']['improvement_percent']:.1f}%")
    logger.info(f"  Size reduction: {benchmark_results['model_size']['size_reduction_percent']:.1f}%")
    if 'accuracy' in benchmark_results:
        logger.info(f"  Cosine similarity: {benchmark_results['accuracy']['cosine_similarity']:.4f}")
    
    return model, pruned_model


def demonstrate_comprehensive_compression():
    """Demonstrate comprehensive model compression suite."""
    logger.info("="*60)
    logger.info("DEMONSTRATING COMPREHENSIVE MODEL COMPRESSION SUITE")
    logger.info("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model and data
    model = ExampleCNN(num_classes=10)
    model = model.to(device)
    dataloader = create_sample_data()
    example_inputs = torch.randn(1, 3, 32, 32).to(device)
    
    # Configure comprehensive compression
    config = ModelCompressionConfig()
    config.enabled_methods = [
        CompressionMethod.TENSOR_FACTORIZATION,
        CompressionMethod.STRUCTURED_PRUNING,
        CompressionMethod.QUANTIZATION
    ]
    config.targets.target_size_ratio = 0.3  # 70% parameter reduction
    config.targets.max_accuracy_loss = 0.05  # 5% max accuracy loss
    config.progressive_compression = True
    config.compression_stages = 3
    config.enable_knowledge_distillation = True
    
    # Set up method-specific configs
    config.tensor_factorization_config.decomposition_method = "hlrtf"
    config.tensor_factorization_config.target_compression_ratio = 0.6
    config.structured_pruning_config.target_sparsity = 0.4
    config.structured_pruning_config.use_low_rank_regularization = True
    
    # Define validation function
    def validation_fn(model_to_eval):
        accuracy = evaluate_model_accuracy(model_to_eval, dataloader, device)
        time_taken, fps = benchmark_model(model_to_eval, example_inputs, device, iterations=20)
        params = sum(p.numel() for p in model_to_eval.parameters())
        
        return {
            'accuracy': accuracy,
            'speedup': 1.0,  # Relative to original, would need original timing
            'size_ratio': params / sum(p.numel() for p in model.parameters())
        }
    
    # Apply comprehensive compression
    logger.info("Applying comprehensive model compression...")
    compression_suite = ModelCompressionSuite(config)
    compressed_model = compression_suite.compress_model(
        model, validation_fn=validation_fn, train_loader=dataloader
    )
    
    # Benchmark comprehensive results
    logger.info("\nBenchmarking comprehensive compression...")
    benchmark_results = compression_suite.benchmark_compression(
        model, compressed_model, example_inputs, iterations=50
    )
    
    logger.info("Comprehensive Compression Results:")
    logger.info(f"  Performance improvement: {benchmark_results['performance']['improvement_percent']:.1f}%")
    logger.info(f"  Size reduction: {benchmark_results['model_size']['size_reduction_percent']:.1f}%")
    if 'accuracy' in benchmark_results:
        logger.info(f"  Cosine similarity: {benchmark_results['accuracy']['cosine_similarity']:.4f}")
    if 'memory' in benchmark_results:
        logger.info(f"  Memory reduction: {benchmark_results['memory']['memory_reduction_percent']:.1f}%")
    
    return model, compressed_model


def demonstrate_convenience_functions():
    """Demonstrate convenience functions for quick optimization."""
    logger.info("="*60)
    logger.info("DEMONSTRATING CONVENIENCE FUNCTIONS")
    logger.info("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    ).to(device)
    
    example_inputs = torch.randn(1, 100).to(device)
    
    logger.info(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Quick tensor factorization
    logger.info("Applying quick tensor factorization...")
    factorized_model = factorize_model(model, method="svd")
    factorized_params = sum(p.numel() for p in factorized_model.parameters())
    logger.info(f"Factorized model parameters: {factorized_params:,}")
    logger.info(f"Factorization reduction: {(1 - factorized_params / sum(p.numel() for p in model.parameters())) * 100:.1f}%")
    
    # Quick structured pruning
    logger.info("Applying quick structured pruning...")
    pruned_model = prune_model(model, method="magnitude")
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    logger.info(f"Pruned model parameters: {pruned_params:,}")
    logger.info(f"Pruning reduction: {(1 - pruned_params / sum(p.numel() for p in model.parameters())) * 100:.1f}%")
    
    # Quick comprehensive compression
    logger.info("Applying quick comprehensive compression...")
    comprehensive_model = compress_model_comprehensive(model)
    comprehensive_params = sum(p.numel() for p in comprehensive_model.parameters())
    logger.info(f"Comprehensive model parameters: {comprehensive_params:,}")
    logger.info(f"Comprehensive reduction: {(1 - comprehensive_params / sum(p.numel() for p in model.parameters())) * 100:.1f}%")


def main():
    """Main demonstration function."""
    logger.info("Starting HLRTF-inspired Model Optimization Demonstration")
    logger.info("This example demonstrates advanced model compression techniques")
    logger.info("inspired by Hierarchical Low-Rank Tensor Factorization (HLRTF)")
    logger.info("")
    
    try:
        # Demonstrate different optimization techniques
        demonstrate_tensor_factorization()
        print("\n" + "="*80 + "\n")
        
        demonstrate_structured_pruning()
        print("\n" + "="*80 + "\n")
        
        demonstrate_comprehensive_compression()
        print("\n" + "="*80 + "\n")
        
        demonstrate_convenience_functions()
        print("\n" + "="*80 + "\n")
        
        logger.info("HLRTF-inspired optimization demonstration completed successfully!")
        logger.info("")
        logger.info("Key Benefits Demonstrated:")
        logger.info("  • Hierarchical tensor factorization for parameter reduction")
        logger.info("  • Structured pruning with low-rank regularization")
        logger.info("  • Multi-objective optimization for size/speed/accuracy trade-offs")
        logger.info("  • Knowledge distillation for accuracy preservation")
        logger.info("  • Progressive compression for gradual optimization")
        logger.info("  • Comprehensive benchmarking and evaluation")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

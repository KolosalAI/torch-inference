"""
Unit tests for Model Compression Suite.

Tests the multi-objective compression optimizer functionality.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.optimizers import (
    ModelCompressionSuite,
    CompressionConfig,
    CompressionMethod,
    TensorFactorizationConfig,
    MaskPruningConfig,
    compress_model
)


class SimpleTestModel(nn.Module):
    """Simple test model for compression tests."""
    
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


class TestCompressionConfig:
    """Test CompressionConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CompressionConfig()
        
        assert CompressionMethod.TENSOR_FACTORIZATION in config.enabled_methods
        assert CompressionMethod.STRUCTURED_PRUNING in config.enabled_methods
        assert config.enable_knowledge_distillation is True
        assert config.progressive_compression is True
        assert config.use_multi_objective is True
        assert config.distillation_temperature == 4.0
        assert config.distillation_alpha == 0.7
        assert config.distillation_epochs == 5
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CompressionConfig()
        config.target_compression_ratio = 0.3
        config.enable_knowledge_distillation = True
        config.compression_schedule = "parallel"
        config.distillation_temperature = 6.0
        
        assert config.target_compression_ratio == 0.3
        assert config.enable_knowledge_distillation is True
        assert config.compression_schedule == "parallel"
        assert config.distillation_temperature == 6.0
    
    def test_get_tensor_factorization_config(self):
        """Test tensor factorization config retrieval."""
        config = CompressionConfig()
        tf_config = config.get_tensor_factorization_config()
        
        assert isinstance(tf_config, TensorFactorizationConfig)
        assert tf_config.rank_ratio == 0.3  # Default value
    
    def test_get_mask_pruning_config(self):
        """Test mask pruning config retrieval."""
        config = CompressionConfig()
        mp_config = config.get_mask_pruning_config()
        
        assert isinstance(mp_config, MaskPruningConfig)
        assert mp_config.pruning_ratio == 0.3  # Default value


class TestModelCompressionSuite:
    """Test ModelCompressionSuite optimizer."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        config = CompressionConfig()
        optimizer = ModelCompressionSuite(config)
        
        assert optimizer.config == config
        assert optimizer.compression_stats == {}
    
    def test_sequential_compression(self):
        """Test sequential compression pipeline."""
        model = SimpleTestModel()
        config = CompressionConfig()
        config.compression_schedule = "sequential"
        config.target_compression_ratio = 0.6
        
        optimizer = ModelCompressionSuite(config)
        compressed_model = optimizer.optimize(model)
        
        assert compressed_model is not None
        
        # Test functionality
        test_input = torch.randn(1, 3, 32, 32)
        output = compressed_model(test_input)
        assert output.shape == (1, 10)
        
        # Check compression stats
        assert len(optimizer.compression_stats) > 0
        assert "final_compression_ratio" in optimizer.compression_stats
    
    def test_parallel_compression(self):
        """Test parallel compression pipeline."""
        model = SimpleTestModel()
        config = CompressionConfig()
        config.compression_schedule = "parallel"
        config.target_compression_ratio = 0.7
        
        optimizer = ModelCompressionSuite(config)
        compressed_model = optimizer.optimize(model)
        
        assert compressed_model is not None
        
        # Test functionality
        test_input = torch.randn(1, 3, 32, 32)
        output = compressed_model(test_input)
        assert output.shape == (1, 10)
    
    def test_compression_with_knowledge_distillation(self):
        """Test compression with knowledge distillation enabled."""
        model = SimpleTestModel()
        config = CompressionConfig()
        config.enable_knowledge_distillation = True
        config.max_distillation_epochs = 2  # Quick test
        config.target_compression_ratio = 0.5
        
        optimizer = ModelCompressionSuite(config)
        
        # Create some dummy training data
        train_data = [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))) for _ in range(5)]
        
        compressed_model = optimizer.optimize(model, train_data=train_data)
        
        assert compressed_model is not None
        
        # Test functionality
        test_input = torch.randn(1, 3, 32, 32)
        output = compressed_model(test_input)
        assert output.shape == (1, 10)
    
    def test_only_tensor_factorization(self):
        """Test compression with only tensor factorization."""
        model = SimpleTestModel()
        config = CompressionConfig()
        config.enable_tensor_factorization = True
        config.enable_structured_pruning = False
        config.enable_knowledge_distillation = False
        
        optimizer = ModelCompressionSuite(config)
        compressed_model = optimizer.optimize(model)
        
        assert compressed_model is not None
        
        # Test functionality
        test_input = torch.randn(1, 3, 32, 32)
        output = compressed_model(test_input)
        assert output.shape == (1, 10)
    
    def test_only_structured_pruning(self):
        """Test compression with only structured pruning."""
        model = SimpleTestModel()
        config = CompressionConfig()
        config.enable_tensor_factorization = False
        config.enable_structured_pruning = True
        config.enable_knowledge_distillation = False
        
        optimizer = ModelCompressionSuite(config)
        compressed_model = optimizer.optimize(model)
        
        assert compressed_model is not None
        
        # Test functionality
        test_input = torch.randn(1, 3, 32, 32)
        output = compressed_model(test_input)
        assert output.shape == (1, 10)
    
    def test_count_parameters(self):
        """Test parameter counting functionality."""
        model = SimpleTestModel()
        config = CompressionConfig()
        optimizer = ModelCompressionSuite(config)
        
        param_count = optimizer._count_parameters(model)
        
        assert param_count > 0
        assert isinstance(param_count, int)
    
    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        model = SimpleTestModel()
        config = CompressionConfig()
        optimizer = ModelCompressionSuite(config)
        
        original_params = optimizer._count_parameters(model)
        
        # Apply compression
        compressed_model = optimizer.optimize(model)
        compressed_params = optimizer._count_parameters(compressed_model)
        
        # Calculate compression ratio
        compression_ratio = compressed_params / original_params
        
        assert 0 < compression_ratio < 1  # Should be compressed
        
        # Check that stats are recorded
        assert "original_parameters" in optimizer.compression_stats
        assert "compressed_parameters" in optimizer.compression_stats
        assert "final_compression_ratio" in optimizer.compression_stats
    
    def test_performance_evaluation(self):
        """Test performance evaluation functionality."""
        model = SimpleTestModel()
        config = CompressionConfig()
        optimizer = ModelCompressionSuite(config)
        
        # Create test data
        test_data = [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))) for _ in range(3)]
        
        # Evaluate performance
        accuracy = optimizer._evaluate_performance(model, test_data)
        
        assert 0 <= accuracy <= 1
        assert isinstance(accuracy, float)
    
    def test_create_student_model_functionality(self):
        """Test student model creation."""
        teacher = SimpleTestModel()
        config = CompressionConfig()
        config.student_teacher_ratio = 0.5
        
        optimizer = ModelCompressionSuite(config)
        student = optimizer._create_student_model(teacher)
        
        assert student is not None
        
        # Test that student is smaller
        teacher_params = optimizer._count_parameters(teacher)
        student_params = optimizer._count_parameters(student)
        
        assert student_params < teacher_params
        
        # Test functionality
        test_input = torch.randn(1, 3, 32, 32)
        student_output = student(test_input)
        assert student_output.shape == (1, 10)
    
    @pytest.mark.parametrize("compression_schedule", ["sequential", "parallel"])
    def test_different_compression_schedules(self, compression_schedule):
        """Test different compression schedules."""
        model = SimpleTestModel()
        config = CompressionConfig()
        config.compression_schedule = compression_schedule
        config.target_compression_ratio = 0.6
        
        optimizer = ModelCompressionSuite(config)
        compressed_model = optimizer.optimize(model)
        
        assert compressed_model is not None
        
        # Test functionality
        test_input = torch.randn(1, 3, 32, 32)
        output = compressed_model(test_input)
        assert output.shape == (1, 10)
    
    @pytest.mark.parametrize("target_ratio", [0.3, 0.5, 0.8])
    def test_different_compression_ratios(self, target_ratio):
        """Test different target compression ratios."""
        model = SimpleTestModel()
        config = CompressionConfig()
        config.target_compression_ratio = target_ratio
        
        optimizer = ModelCompressionSuite(config)
        compressed_model = optimizer.optimize(model)
        
        assert compressed_model is not None
        
        # Test functionality
        test_input = torch.randn(1, 3, 32, 32)
        output = compressed_model(test_input)
        assert output.shape == (1, 10)


class TestKnowledgeDistillation:
    """Test knowledge distillation functionality."""
    
    def test_distillation_loss_calculation(self):
        """Test distillation loss calculation."""
        config = CompressionConfig()
        config.distillation_temperature = 4.0
        config.distillation_alpha = 0.7
        
        optimizer = ModelCompressionSuite(config)
        
        # Create dummy outputs
        teacher_output = torch.randn(4, 10)
        student_output = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        
        loss = optimizer._compute_distillation_loss(
            student_output, teacher_output, targets
        )
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_knowledge_distillation_training(self):
        """Test knowledge distillation training process."""
        teacher = SimpleTestModel()
        config = CompressionConfig()
        config.max_distillation_epochs = 1  # Quick test
        config.distillation_temperature = 3.0
        
        optimizer = ModelCompressionSuite(config)
        
        # Create student model
        student = optimizer._create_student_model(teacher)
        
        # Create training data
        train_data = [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))) for _ in range(3)]
        
        # Perform distillation
        distilled_student = optimizer._perform_knowledge_distillation(
            teacher, student, train_data
        )
        
        assert distilled_student is not None
        
        # Test functionality
        test_input = torch.randn(1, 3, 32, 32)
        output = distilled_student(test_input)
        assert output.shape == (1, 10)


class TestCompressionIntegration:
    """Integration tests for model compression suite."""
    
    def test_compress_model_convenience_function(self):
        """Test the compress_model convenience function."""
        model = SimpleTestModel()
        config = CompressionConfig()
        config.target_compression_ratio = 0.6
        
        compressed_model = compress_model(model, config)
        
        assert compressed_model is not None
        
        # Test functionality
        test_input = torch.randn(1, 3, 32, 32)
        output = compressed_model(test_input)
        assert output.shape == (1, 10)
    
    def test_compress_model_with_default_config(self):
        """Test compression with default configuration."""
        model = SimpleTestModel()
        
        compressed_model = compress_model(model)  # No config provided
        
        assert compressed_model is not None
        
        # Test functionality
        test_input = torch.randn(1, 3, 32, 32)
        output = compressed_model(test_input)
        assert output.shape == (1, 10)
    
    def test_create_student_model_convenience_function(self):
        """Test student model creation."""
        teacher = SimpleTestModel()
        config = CompressionConfig()
        config.student_teacher_ratio = 0.5
        
        optimizer = ModelCompressionSuite(config)
        student = optimizer._create_student_model(teacher)
        
        assert student is not None
        
        # Test functionality
        test_input = torch.randn(1, 3, 32, 32)
        output = student(test_input)
        assert output.shape == (1, 10)
    
    def test_end_to_end_compression_pipeline(self):
        """Test complete end-to-end compression pipeline."""
        model = SimpleTestModel()
        config = CompressionConfig()
        config.enable_tensor_factorization = True
        config.enable_structured_pruning = True
        config.enable_knowledge_distillation = True
        config.target_compression_ratio = 0.4
        config.max_distillation_epochs = 1  # Quick test
        
        # Create training data for distillation
        train_data = [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))) for _ in range(5)]
        
        compressed_model = compress_model(model, config, train_data=train_data)
        
        assert compressed_model is not None
        
        # Test functionality
        test_input = torch.randn(2, 3, 32, 32)
        output = compressed_model(test_input)
        assert output.shape == (2, 10)
        assert not torch.isnan(output).any()
    
    def test_compression_preserves_model_interface(self):
        """Test that compression preserves the model interface."""
        model = SimpleTestModel()
        config = CompressionConfig()
        config.target_compression_ratio = 0.7  # Light compression
        
        # Test input
        test_input = torch.randn(3, 3, 32, 32)
        
        # Original output
        model.eval()
        with torch.no_grad():
            original_output = model(test_input)
        
        # Compress model
        compressed_model = compress_model(model, config)
        
        # Compressed output
        compressed_model.eval()
        with torch.no_grad():
            compressed_output = compressed_model(test_input)
        
        # Check output shapes match
        assert original_output.shape == compressed_output.shape
        assert compressed_output.shape == (3, 10)


if __name__ == "__main__":
    pytest.main([__file__])

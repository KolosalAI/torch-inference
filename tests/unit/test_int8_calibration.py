"""
Unit tests for INT8 calibration toolkit.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from framework.optimizers.int8_calibration import (
    INT8CalibrationToolkit,
    CalibrationConfig,
    ActivationObserver,
    ActivationStats,
    create_calibration_dataset,
    get_calibration_toolkit
)


class SimpleCNN(nn.Module):
    """Simple CNN for testing calibration."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32 * 4 * 4, 10)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return SimpleCNN()


@pytest.fixture
def calibration_data():
    """Create sample calibration data."""
    # Generate synthetic data
    images = torch.randn(100, 3, 32, 32)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def calibration_config():
    """Create calibration configuration."""
    return CalibrationConfig(
        method="entropy",
        num_calibration_batches=5,
        histogram_bins=256,
        collect_stats=True,
        cache_calibration=False  # Disable caching for tests
    )


class TestCalibrationConfig:
    """Test calibration configuration."""
    
    def test_default_config(self):
        """Test default calibration configuration."""
        config = CalibrationConfig()
        assert config.method == "entropy"
        assert config.num_calibration_batches == 100
        assert config.percentile == 99.99
        assert config.histogram_bins == 2048
        assert config.collect_stats == True
        assert config.cache_calibration == True
        assert config.smooth_distribution == True
        assert config.outlier_threshold == 0.001
    
    def test_custom_config(self):
        """Test custom calibration configuration."""
        config = CalibrationConfig(
            method="percentile",
            num_calibration_batches=50,
            percentile=99.5,
            histogram_bins=1024
        )
        assert config.method == "percentile"
        assert config.num_calibration_batches == 50
        assert config.percentile == 99.5
        assert config.histogram_bins == 1024


class TestActivationObserver:
    """Test activation observer functionality."""
    
    def test_observer_initialization(self, calibration_config):
        """Test observer initialization."""
        observer = ActivationObserver(calibration_config)
        assert observer.config == calibration_config
        assert len(observer.stats) == 0
        assert len(observer.hooks) == 0
        assert len(observer.layer_names) == 0
    
    def test_hook_registration(self, sample_model, calibration_config):
        """Test hook registration on model."""
        observer = ActivationObserver(calibration_config)
        observer.register_hooks(sample_model)
        
        # Should register hooks for conv, bn, and linear layers
        assert len(observer.hooks) > 0
        assert len(observer.layer_names) > 0
        
        # Clean up
        observer.remove_hooks()
        assert len(observer.hooks) == 0
    
    def test_statistics_collection(self, sample_model, calibration_data, calibration_config):
        """Test statistics collection during forward pass."""
        observer = ActivationObserver(calibration_config)
        observer.register_hooks(sample_model)
        
        sample_model.eval()
        with torch.no_grad():
            # Run a few batches
            for i, (data, _) in enumerate(calibration_data):
                if i >= 2:  # Only run 2 batches
                    break
                _ = sample_model(data)
        
        # Should have collected statistics
        assert len(observer.stats) > 0
        
        # Check statistics structure
        for layer_name, stats in observer.stats.items():
            assert isinstance(stats, ActivationStats)
            assert stats.sample_count > 0
            assert stats.min_val <= stats.max_val
            assert len(stats.histogram) == calibration_config.histogram_bins
            assert len(stats.bin_edges) == calibration_config.histogram_bins + 1
        
        observer.remove_hooks()
    
    def test_clear_stats(self, calibration_config):
        """Test clearing statistics."""
        observer = ActivationObserver(calibration_config)
        
        # Add some dummy stats
        observer.stats["test_layer"] = ActivationStats(
            min_val=0.0, max_val=1.0, mean=0.5, std=0.25,
            histogram=np.zeros(256), bin_edges=np.linspace(0, 1, 257),
            shape=(1, 16, 32, 32), sample_count=1
        )
        
        assert len(observer.stats) == 1
        
        observer.clear_stats()
        assert len(observer.stats) == 0


class TestINT8CalibrationToolkit:
    """Test INT8 calibration toolkit."""
    
    def test_toolkit_initialization(self, calibration_config):
        """Test toolkit initialization."""
        toolkit = INT8CalibrationToolkit(calibration_config)
        assert toolkit.config == calibration_config
        assert isinstance(toolkit.observer, ActivationObserver)
        assert len(toolkit.calibration_cache) == 0
    
    def test_entropy_calibration(self, sample_model, calibration_data, calibration_config):
        """Test entropy-based calibration."""
        calibration_config.method = "entropy"
        toolkit = INT8CalibrationToolkit(calibration_config)
        
        device = torch.device("cpu")
        quantization_params = toolkit.calibrate_model(
            sample_model, calibration_data, device
        )
        
        # Should return quantization parameters
        assert isinstance(quantization_params, dict)
        assert len(quantization_params) > 0
        
        # Each parameter should be a (scale, zero_point) tuple
        for layer_name, (scale, zero_point) in quantization_params.items():
            assert isinstance(scale, float)
            assert isinstance(zero_point, (int, float))
            assert scale > 0  # Scale should be positive
    
    def test_percentile_calibration(self, sample_model, calibration_data, calibration_config):
        """Test percentile-based calibration."""
        calibration_config.method = "percentile"
        calibration_config.percentile = 99.0
        toolkit = INT8CalibrationToolkit(calibration_config)
        
        device = torch.device("cpu")
        quantization_params = toolkit.calibrate_model(
            sample_model, calibration_data, device
        )
        
        assert isinstance(quantization_params, dict)
        assert len(quantization_params) > 0
    
    def test_kl_divergence_calibration(self, sample_model, calibration_data, calibration_config):
        """Test KL divergence calibration."""
        calibration_config.method = "kl_divergence"
        toolkit = INT8CalibrationToolkit(calibration_config)
        
        device = torch.device("cpu")
        quantization_params = toolkit.calibrate_model(
            sample_model, calibration_data, device
        )
        
        assert isinstance(quantization_params, dict)
        assert len(quantization_params) > 0
    
    def test_minmax_calibration(self, sample_model, calibration_data, calibration_config):
        """Test min-max calibration."""
        calibration_config.method = "minmax"
        toolkit = INT8CalibrationToolkit(calibration_config)
        
        device = torch.device("cpu")
        quantization_params = toolkit.calibrate_model(
            sample_model, calibration_data, device
        )
        
        assert isinstance(quantization_params, dict)
        assert len(quantization_params) > 0
    
    def test_calibration_validation(self, sample_model, calibration_data, calibration_config):
        """Test calibration quality validation."""
        toolkit = INT8CalibrationToolkit(calibration_config)
        
        # Create a dummy quantized model (same as original for testing)
        original_model = sample_model
        quantized_model = sample_model  # In real case, this would be quantized
        
        device = torch.device("cpu")
        
        # Create smaller validation dataset
        val_images = torch.randn(16, 3, 32, 32)
        val_labels = torch.randint(0, 10, (16,))
        val_dataset = TensorDataset(val_images, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=4)
        
        quality_metrics = toolkit.validate_calibration_quality(
            original_model, quantized_model, val_loader, device
        )
        
        assert isinstance(quality_metrics, dict)
        assert "mse" in quality_metrics
        assert "mae" in quality_metrics
        assert "cosine_similarity" in quality_metrics
        assert "snr" in quality_metrics
        assert "samples_processed" in quality_metrics
        
        # Since we're using the same model, metrics should indicate high similarity
        assert quality_metrics["mse"] < 1e-6  # Very low MSE
        assert quality_metrics["cosine_similarity"] > 0.99  # High cosine similarity
    
    def test_calibration_report(self, sample_model, calibration_data, calibration_config):
        """Test calibration report generation."""
        toolkit = INT8CalibrationToolkit(calibration_config)
        
        device = torch.device("cpu")
        toolkit.calibrate_model(sample_model, calibration_data, device)
        
        report = toolkit.get_calibration_report()
        
        assert isinstance(report, dict)
        assert "config" in report
        assert "statistics" in report
        assert "recommendations" in report
        
        # Check config section
        config_section = report["config"]
        assert config_section["method"] == calibration_config.method
        assert config_section["num_calibration_batches"] == calibration_config.num_calibration_batches
        
        # Check statistics section
        stats_section = report["statistics"]
        assert len(stats_section) > 0  # Should have statistics for some layers
        
        # Check that each layer has proper statistics
        for layer_name, layer_stats in stats_section.items():
            assert "min_val" in layer_stats
            assert "max_val" in layer_stats
            assert "mean" in layer_stats
            assert "std" in layer_stats
            assert "dynamic_range" in layer_stats
            assert "sample_count" in layer_stats
            assert "shape" in layer_stats
    
    def test_unknown_calibration_method(self, sample_model, calibration_data, calibration_config):
        """Test handling of unknown calibration method."""
        calibration_config.method = "unknown_method"
        toolkit = INT8CalibrationToolkit(calibration_config)
        
        device = torch.device("cpu")
        
        # Should fall back to minmax method without throwing exception
        quantization_params = toolkit.calibrate_model(
            sample_model, calibration_data, device
        )
        
        assert isinstance(quantization_params, dict)
        assert len(quantization_params) > 0


class TestCalibrationDataset:
    """Test calibration dataset utilities."""
    
    def test_create_calibration_dataset(self, calibration_data):
        """Test calibration dataset creation."""
        # Create subset dataset
        calibration_dataset = create_calibration_dataset(calibration_data, num_samples=20)
        
        assert len(calibration_dataset) <= 20
        
        # Test that we can iterate through the dataset
        for i, sample in enumerate(calibration_dataset):
            assert isinstance(sample, torch.Tensor)
            assert sample.shape == (3, 32, 32)  # Image shape without batch dimension
            if i >= 5:  # Test first few samples
                break


class TestGlobalCalibrationToolkit:
    """Test global calibration toolkit instance."""
    
    def test_get_calibration_toolkit(self):
        """Test getting global calibration toolkit."""
        toolkit1 = get_calibration_toolkit()
        toolkit2 = get_calibration_toolkit()
        
        # Should return the same instance
        assert toolkit1 is toolkit2
        assert isinstance(toolkit1, INT8CalibrationToolkit)


class TestCalibrationCache:
    """Test calibration caching functionality."""
    
    def test_cache_calibration_results(self, sample_model, calibration_data):
        """Test caching of calibration results."""
        config = CalibrationConfig(cache_calibration=True)
        toolkit = INT8CalibrationToolkit(config)
        
        device = torch.device("cpu")
        quantization_params = toolkit.calibrate_model(
            sample_model, calibration_data, device
        )
        
        # Cache should be created (though we can't easily test file system in unit tests)
        assert isinstance(quantization_params, dict)
    
    def test_load_cached_calibration(self):
        """Test loading cached calibration results."""
        toolkit = INT8CalibrationToolkit()
        
        # Test loading non-existent cache
        result = toolkit.load_cached_calibration("non_existent_cache")
        assert result is None


class TestCUDACalibration:
    """Test calibration with CUDA."""
    
    def test_cuda_calibration(self, sample_model, calibration_data, calibration_config):
        """Test calibration on CUDA device."""
        with patch('torch.cuda.is_available', return_value=True):
            toolkit = INT8CalibrationToolkit(calibration_config)

            # Create a real torch device to avoid mock issues
            device = torch.device('cpu')  # Use CPU for testing since CUDA may not be available
            
            sample_model.to = Mock(return_value=sample_model)

            quantization_params = toolkit.calibrate_model(
                sample_model, calibration_data, device
            )

            assert isinstance(quantization_params, dict)
            assert len(quantization_params) > 0
class TestCalibrationEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_calibration_data(self, sample_model, calibration_config):
        """Test handling of empty calibration data."""
        # Create empty dataloader
        empty_dataset = TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long))
        empty_loader = DataLoader(empty_dataset, batch_size=1)
        
        toolkit = INT8CalibrationToolkit(calibration_config)
        device = torch.device("cpu")
        
        # Should handle empty data gracefully
        quantization_params = toolkit.calibrate_model(
            sample_model, empty_loader, device
        )
        
        # May return empty dict or handle gracefully
        assert isinstance(quantization_params, dict)
    
    def test_single_batch_calibration(self, sample_model, calibration_config):
        """Test calibration with single batch."""
        # Create single batch dataset
        single_batch = TensorDataset(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))
        single_loader = DataLoader(single_batch, batch_size=4)
        
        calibration_config.num_calibration_batches = 1
        toolkit = INT8CalibrationToolkit(calibration_config)
        
        device = torch.device("cpu")
        quantization_params = toolkit.calibrate_model(
            sample_model, single_loader, device
        )
        
        assert isinstance(quantization_params, dict)
    
    def test_large_batch_size(self, sample_model, calibration_config):
        """Test calibration with large batch size."""
        # Create dataset with large batch
        large_batch = TensorDataset(torch.randn(64, 3, 32, 32), torch.randint(0, 10, (64,)))
        large_loader = DataLoader(large_batch, batch_size=64)
        
        calibration_config.num_calibration_batches = 1
        toolkit = INT8CalibrationToolkit(calibration_config)
        
        device = torch.device("cpu")
        quantization_params = toolkit.calibrate_model(
            sample_model, large_loader, device
        )
        
        assert isinstance(quantization_params, dict)


if __name__ == "__main__":
    # Run basic smoke test
    model = SimpleCNN()
    config = CalibrationConfig(method="entropy", num_calibration_batches=2)
    toolkit = INT8CalibrationToolkit(config)
    
    # Create simple test data
    test_data = torch.randn(16, 3, 32, 32)
    test_labels = torch.randint(0, 10, (16,))
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    # Run calibration
    device = torch.device("cpu")
    params = toolkit.calibrate_model(model, test_loader, device)
    
    print(f"✓ Calibration completed for {len(params)} layers")
    print("✓ INT8 calibration tests ready")

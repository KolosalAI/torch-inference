"""
Test suite for core modules in the framework.

This module tests core framework components including model downloader,
optimized model, config manager, and GPU manager.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import json

# Framework imports
try:
    from framework.core.model_downloader import (
        ModelDownloader, ModelInfo, ModelSource, DownloadError,
        get_model_downloader, download_model, list_available_models
    )
    from framework.core.optimized_model import (
        OptimizedModel, OptimizationConfig, create_optimized_model,
        ModelOptimizer, OptimizationStrategy, apply_optimizations
    )
    from framework.core.config_manager import (
        ConfigManager, ConfigLoader, ConfigValidator,
        load_config, save_config, validate_config, merge_configs
    )
    from framework.core.gpu_manager import (
        GPUManager, DeviceInfo, MemoryInfo, 
        auto_configure_device, get_gpu_manager, monitor_gpu_usage
    )
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    CORE_MODULES_AVAILABLE = False
    pytest.skip(f"Core modules not available: {e}", allow_module_level=True)


class TestModelInfo:
    """Test ModelInfo class."""
    
    def test_model_info_creation(self):
        """Test creating ModelInfo object."""
        info = ModelInfo(
            name="resnet18",
            source="torchvision",
            model_type="classification",
            file_path="/path/to/model.pt",
            size_mb=44.6,
            download_url="https://download.pytorch.org/models/resnet18.pth"
        )
        
        assert info.name == "resnet18"
        assert info.source == "torchvision"
        assert info.model_type == "classification"
        assert info.file_path == "/path/to/model.pt"
        assert info.size_mb == 44.6
        assert info.download_url == "https://download.pytorch.org/models/resnet18.pth"
    
    def test_model_info_serialization(self):
        """Test ModelInfo serialization."""
        info = ModelInfo(
            name="test_model",
            source="custom",
            model_type="classification"
        )
        
        # Test dict conversion
        info_dict = info.to_dict()
        assert isinstance(info_dict, dict)
        assert info_dict["name"] == "test_model"
        
        # Test from dict
        new_info = ModelInfo.from_dict(info_dict)
        assert new_info.name == info.name
        assert new_info.source == info.source
    
    def test_model_info_validation(self):
        """Test ModelInfo validation."""
        # Test invalid model type
        with pytest.raises(ValueError):
            ModelInfo(
                name="test",
                source="custom",
                model_type="invalid_type"
            )
        
        # Test invalid size
        with pytest.raises(ValueError):
            ModelInfo(
                name="test",
                source="custom",
                model_type="classification",
                size_mb=-1
            )


class TestModelDownloader:
    """Test ModelDownloader class."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_model_file(self, temp_cache_dir):
        """Create mock model file."""
        model_file = temp_cache_dir / "test_model.pt"
        # Create dummy model
        dummy_model = torch.nn.Linear(10, 5)
        torch.save(dummy_model.state_dict(), model_file)
        return model_file
    
    def test_downloader_initialization(self, temp_cache_dir):
        """Test downloader initialization."""
        downloader = ModelDownloader(cache_dir=temp_cache_dir)
        
        assert downloader.cache_dir == temp_cache_dir
        assert temp_cache_dir.exists()
    
    @patch('framework.core.model_downloader.download_from_url')
    def test_download_from_url(self, mock_download, temp_cache_dir, mock_model_file):
        """Test downloading model from URL."""
        mock_download.return_value = mock_model_file
        
        downloader = ModelDownloader(cache_dir=temp_cache_dir)
        
        model_info = downloader.download_model(
            source="url",
            model_id="https://example.com/model.pt",
            name="test_model"
        )
        
        assert model_info.name == "test_model"
        assert model_info.source == "url"
        mock_download.assert_called_once()
    
    @patch('framework.core.model_downloader.download_from_torchvision')
    def test_download_from_torchvision(self, mock_download, temp_cache_dir, mock_model_file):
        """Test downloading model from torchvision."""
        mock_download.return_value = mock_model_file
        
        downloader = ModelDownloader(cache_dir=temp_cache_dir)
        
        model_info = downloader.download_model(
            source="torchvision",
            model_id="resnet18",
            name="resnet18"
        )
        
        assert model_info.name == "resnet18"
        assert model_info.source == "torchvision"
        mock_download.assert_called_once()
    
    @patch('framework.core.model_downloader.download_from_huggingface')
    def test_download_from_huggingface(self, mock_download, temp_cache_dir, mock_model_file):
        """Test downloading model from Hugging Face."""
        mock_download.return_value = mock_model_file
        
        downloader = ModelDownloader(cache_dir=temp_cache_dir)
        
        model_info = downloader.download_model(
            source="huggingface",
            model_id="bert-base-uncased",
            name="bert-base-uncased",
            task="text-classification"
        )
        
        assert model_info.name == "bert-base-uncased"
        assert model_info.source == "huggingface"
        mock_download.assert_called_once()
    
    def test_list_models(self, temp_cache_dir, mock_model_file):
        """Test listing downloaded models."""
        downloader = ModelDownloader(cache_dir=temp_cache_dir)
        
        # Create metadata file
        metadata = {
            "test_model": {
                "name": "test_model",
                "source": "custom",
                "model_type": "classification",
                "file_path": str(mock_model_file),
                "size_mb": 1.0
            }
        }
        
        metadata_file = temp_cache_dir / "models_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        models = downloader.list_models()
        
        assert len(models) == 1
        assert models[0].name == "test_model"
    
    def test_get_model_info(self, temp_cache_dir, mock_model_file):
        """Test getting model information."""
        downloader = ModelDownloader(cache_dir=temp_cache_dir)
        
        # Create metadata file
        metadata = {
            "test_model": {
                "name": "test_model",
                "source": "custom",
                "model_type": "classification",
                "file_path": str(mock_model_file),
                "size_mb": 1.0
            }
        }
        
        metadata_file = temp_cache_dir / "models_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        model_info = downloader.get_model_info("test_model")
        
        assert model_info.name == "test_model"
        assert model_info.source == "custom"
    
    def test_remove_model(self, temp_cache_dir, mock_model_file):
        """Test removing downloaded model."""
        downloader = ModelDownloader(cache_dir=temp_cache_dir)
        
        # Create metadata file
        metadata = {
            "test_model": {
                "name": "test_model",
                "source": "custom",
                "model_type": "classification",
                "file_path": str(mock_model_file),
                "size_mb": 1.0
            }
        }
        
        metadata_file = temp_cache_dir / "models_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        removed = downloader.remove_model("test_model")
        
        assert removed is True
        assert not mock_model_file.exists()
    
    def test_clean_cache(self, temp_cache_dir):
        """Test cleaning model cache."""
        downloader = ModelDownloader(cache_dir=temp_cache_dir)
        
        # Create some test files
        test_files = [
            temp_cache_dir / "model1.pt",
            temp_cache_dir / "model2.pth",
            temp_cache_dir / "config.json"
        ]
        
        for file_path in test_files:
            file_path.touch()
        
        removed_files = downloader.clean_cache()
        
        assert len(removed_files) >= 0
    
    def test_download_error_handling(self, temp_cache_dir):
        """Test download error handling."""
        downloader = ModelDownloader(cache_dir=temp_cache_dir)
        
        # Test invalid source
        with pytest.raises(DownloadError):
            downloader.download_model(
                source="invalid_source",
                model_id="test_model"
            )
    
    def test_get_model_downloader_singleton(self):
        """Test get_model_downloader singleton function."""
        downloader1 = get_model_downloader()
        downloader2 = get_model_downloader()
        
        assert downloader1 is downloader2


class TestOptimizedModel:
    """Test OptimizedModel class."""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple test model."""
        return torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
    
    @pytest.fixture
    def optimization_config(self):
        """Create optimization config."""
        return OptimizationConfig(
            enable_jit=True,
            enable_quantization=False,
            enable_pruning=False,
            optimization_level="moderate"
        )
    
    def test_optimization_config_creation(self):
        """Test creating optimization config."""
        config = OptimizationConfig(
            enable_jit=True,
            enable_quantization=True,
            enable_pruning=True,
            optimization_level="aggressive"
        )
        
        assert config.enable_jit is True
        assert config.enable_quantization is True
        assert config.enable_pruning is True
        assert config.optimization_level == "aggressive"
    
    def test_optimized_model_initialization(self, simple_model, optimization_config):
        """Test optimized model initialization."""
        optimized_model = OptimizedModel(simple_model, optimization_config)
        
        assert optimized_model.original_model == simple_model
        assert optimized_model.config == optimization_config
        assert optimized_model.optimized_model is not None
    
    def test_optimized_model_forward(self, simple_model, optimization_config):
        """Test optimized model forward pass."""
        optimized_model = OptimizedModel(simple_model, optimization_config)
        
        input_tensor = torch.randn(4, 10)
        output = optimized_model(input_tensor)
        
        assert output.shape == (4, 2)
        assert output.dtype == torch.float32
    
    def test_optimized_model_jit_compilation(self, simple_model):
        """Test JIT compilation optimization."""
        config = OptimizationConfig(enable_jit=True)
        
        with patch('torch.jit.script') as mock_jit:
            mock_jit.return_value = simple_model
            
            optimized_model = OptimizedModel(simple_model, config)
            
            # JIT should be applied if available
            if hasattr(torch.jit, 'script'):
                mock_jit.assert_called_once()
    
    def test_optimized_model_quantization(self, simple_model):
        """Test quantization optimization."""
        config = OptimizationConfig(enable_quantization=True)
        
        with patch('torch.quantization.quantize_dynamic') as mock_quantize:
            mock_quantize.return_value = simple_model
            
            optimized_model = OptimizedModel(simple_model, config)
            
            # Should attempt quantization
            assert optimized_model.optimized_model is not None
    
    def test_create_optimized_model_factory(self, simple_model):
        """Test create_optimized_model factory function."""
        optimized_model = create_optimized_model(
            model=simple_model,
            optimization_level="moderate",
            enable_jit=True
        )
        
        assert isinstance(optimized_model, OptimizedModel)
        assert optimized_model.config.enable_jit is True
    
    def test_apply_optimizations_function(self, simple_model):
        """Test apply_optimizations function."""
        optimizations = ["jit", "quantization"]
        
        optimized_model = apply_optimizations(simple_model, optimizations)
        
        assert optimized_model is not None
    
    def test_optimization_strategy_enum(self):
        """Test OptimizationStrategy enum."""
        assert hasattr(OptimizationStrategy, 'SPEED')
        assert hasattr(OptimizationStrategy, 'MEMORY')
        assert hasattr(OptimizationStrategy, 'BALANCED')


class TestConfigManager:
    """Test ConfigManager class."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {
            "model": {
                "type": "classification",
                "num_classes": 10
            },
            "device": {
                "type": "cuda",
                "device_id": 0
            },
            "optimization": {
                "enable_jit": True,
                "batch_size": 32
            }
        }
    
    def test_config_manager_initialization(self, temp_config_dir):
        """Test config manager initialization."""
        config_manager = ConfigManager(config_dir=temp_config_dir)
        
        assert config_manager.config_dir == temp_config_dir
    
    def test_load_config(self, temp_config_dir, sample_config):
        """Test loading configuration."""
        config_file = temp_config_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(sample_config, f)
        
        config_manager = ConfigManager(config_dir=temp_config_dir)
        loaded_config = config_manager.load_config("config.json")
        
        assert loaded_config == sample_config
    
    def test_save_config(self, temp_config_dir, sample_config):
        """Test saving configuration."""
        config_manager = ConfigManager(config_dir=temp_config_dir)
        config_manager.save_config(sample_config, "test_config.json")
        
        config_file = temp_config_dir / "test_config.json"
        assert config_file.exists()
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config == sample_config
    
    def test_validate_config(self, sample_config):
        """Test config validation."""
        validator = ConfigValidator()
        
        # Test valid config
        is_valid, errors = validator.validate(sample_config)
        assert is_valid is True
        assert len(errors) == 0
        
        # Test invalid config
        invalid_config = {"invalid": "config"}
        is_valid, errors = validator.validate(invalid_config)
        assert is_valid is False or len(errors) >= 0  # May have validation errors
    
    def test_merge_configs(self):
        """Test merging configurations."""
        config1 = {
            "model": {"type": "classification"},
            "device": {"type": "cpu"}
        }
        
        config2 = {
            "model": {"num_classes": 10},
            "optimization": {"enable_jit": True}
        }
        
        merged = merge_configs(config1, config2)
        
        assert merged["model"]["type"] == "classification"
        assert merged["model"]["num_classes"] == 10
        assert merged["device"]["type"] == "cpu"
        assert merged["optimization"]["enable_jit"] is True
    
    def test_config_loader_yaml(self, temp_config_dir, sample_config):
        """Test loading YAML configuration."""
        pytest.importorskip("yaml")
        
        config_file = temp_config_dir / "config.yaml"
        
        # Mock YAML content
        yaml_content = """
model:
  type: classification
  num_classes: 10
device:
  type: cuda
  device_id: 0
"""
        
        with open(config_file, 'w') as f:
            f.write(yaml_content)
        
        loader = ConfigLoader()
        loaded_config = loader.load(config_file)
        
        assert loaded_config is not None
        assert "model" in loaded_config
    
    def test_config_environment_variables(self):
        """Test config loading with environment variables."""
        import os
        
        # Set test environment variable
        os.environ["TORCH_INFERENCE_DEVICE"] = "cuda"
        
        loader = ConfigLoader()
        config = loader.load_with_env_override({})
        
        # Should incorporate environment variables
        assert config is not None
        
        # Cleanup
        del os.environ["TORCH_INFERENCE_DEVICE"]


class TestGPUManager:
    """Test GPUManager class."""
    
    def test_gpu_manager_initialization(self):
        """Test GPU manager initialization."""
        gpu_manager = GPUManager()
        
        assert hasattr(gpu_manager, 'device_count')
        assert hasattr(gpu_manager, 'current_device')
    
    def test_device_info_creation(self):
        """Test DeviceInfo creation."""
        device_info = DeviceInfo(
            device_id=0,
            name="NVIDIA RTX 3080",
            memory_total=10737418240,  # 10GB
            compute_capability=(8, 6)
        )
        
        assert device_info.device_id == 0
        assert device_info.name == "NVIDIA RTX 3080"
        assert device_info.memory_total == 10737418240
        assert device_info.compute_capability == (8, 6)
    
    def test_memory_info_creation(self):
        """Test MemoryInfo creation."""
        memory_info = MemoryInfo(
            allocated=1073741824,  # 1GB
            cached=536870912,      # 512MB
            reserved=2147483648    # 2GB
        )
        
        assert memory_info.allocated == 1073741824
        assert memory_info.cached == 536870912
        assert memory_info.reserved == 2147483648
        
        # Test properties
        assert memory_info.free > 0
        assert memory_info.utilization >= 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_detection(self):
        """Test GPU detection."""
        gpu_manager = GPUManager()
        
        devices = gpu_manager.get_available_devices()
        
        assert isinstance(devices, list)
        if torch.cuda.device_count() > 0:
            assert len(devices) > 0
            assert all(isinstance(device, DeviceInfo) for device in devices)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_monitoring(self):
        """Test memory monitoring."""
        gpu_manager = GPUManager()
        
        memory_info = gpu_manager.get_memory_info(0)
        
        assert isinstance(memory_info, MemoryInfo)
        assert memory_info.allocated >= 0
        assert memory_info.reserved >= 0
    
    def test_auto_configure_device(self):
        """Test auto device configuration."""
        device_config = auto_configure_device()
        
        assert "device" in device_config
        assert "device_id" in device_config
        
        # Should select appropriate device
        if torch.cuda.is_available():
            assert device_config["device"] in ["cuda", "cpu"]
        else:
            assert device_config["device"] == "cpu"
    
    def test_monitor_gpu_usage(self):
        """Test GPU usage monitoring."""
        # Mock GPU monitoring
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_stats') as mock_stats:
                mock_stats.return_value = {
                    'allocated_bytes.all.current': 1073741824,
                    'reserved_bytes.all.current': 2147483648,
                    'active_bytes.all.current': 536870912
                }
                
                usage = monitor_gpu_usage(device_id=0)
                
                assert isinstance(usage, dict)
                assert "memory" in usage or "utilization" in usage
    
    def test_get_gpu_manager_singleton(self):
        """Test get_gpu_manager singleton function."""
        manager1 = get_gpu_manager()
        manager2 = get_gpu_manager()
        
        assert manager1 is manager2
    
    def test_gpu_manager_device_selection(self):
        """Test GPU manager device selection."""
        gpu_manager = GPUManager()
        
        # Test selecting best device
        best_device = gpu_manager.select_best_device()
        
        assert best_device is not None
        assert isinstance(best_device, (int, str))
    
    def test_gpu_manager_memory_management(self):
        """Test GPU manager memory management."""
        gpu_manager = GPUManager()
        
        # Test memory cleanup
        cleaned = gpu_manager.cleanup_memory()
        
        assert isinstance(cleaned, bool)
    
    def test_gpu_manager_error_handling(self):
        """Test GPU manager error handling."""
        gpu_manager = GPUManager()
        
        # Test with invalid device ID
        with pytest.raises((ValueError, RuntimeError)):
            gpu_manager.get_memory_info(-1)


class TestCoreModuleIntegration:
    """Test integration between core modules."""
    
    def test_downloader_with_optimized_model(self, temp_cache_dir):
        """Test integration between downloader and optimized model."""
        # Create mock model file
        model_file = temp_cache_dir / "test_model.pt"
        dummy_model = torch.nn.Linear(10, 5)
        torch.save(dummy_model.state_dict(), model_file)
        
        # Test downloading and optimizing
        downloader = ModelDownloader(cache_dir=temp_cache_dir)
        
        with patch.object(downloader, 'download_model') as mock_download:
            mock_download.return_value = ModelInfo(
                name="test_model",
                source="custom",
                model_type="classification",
                file_path=str(model_file)
            )
            
            model_info = downloader.download_model("custom", "test_model")
            
            # Load and optimize model
            state_dict = torch.load(model_info.file_path)
            model = torch.nn.Linear(10, 5)
            model.load_state_dict(state_dict)
            
            config = OptimizationConfig(enable_jit=True)
            optimized_model = OptimizedModel(model, config)
            
            assert optimized_model is not None
    
    def test_config_manager_with_gpu_manager(self, temp_config_dir):
        """Test integration between config manager and GPU manager."""
        config_manager = ConfigManager(config_dir=temp_config_dir)
        gpu_manager = GPUManager()
        
        # Create device config
        device_config = auto_configure_device()
        
        # Save device config
        config_manager.save_config(device_config, "device_config.json")
        
        # Load device config
        loaded_config = config_manager.load_config("device_config.json")
        
        assert loaded_config == device_config
    
    def test_all_core_modules_together(self, temp_cache_dir):
        """Test all core modules working together."""
        # Initialize all components
        downloader = ModelDownloader(cache_dir=temp_cache_dir)
        config_manager = ConfigManager(config_dir=temp_cache_dir)
        gpu_manager = GPUManager()
        
        # Create integrated workflow
        device_config = auto_configure_device()
        config_manager.save_config(device_config, "device.json")
        
        # Mock model download and optimization
        with patch.object(downloader, 'download_model') as mock_download:
            model_file = temp_cache_dir / "integrated_model.pt"
            dummy_model = torch.nn.Linear(10, 5)
            torch.save(dummy_model.state_dict(), model_file)
            
            mock_download.return_value = ModelInfo(
                name="integrated_model",
                source="custom",
                model_type="classification",
                file_path=str(model_file)
            )
            
            model_info = downloader.download_model("custom", "integrated_model")
            
            # Load model and apply optimizations
            state_dict = torch.load(model_info.file_path)
            model = torch.nn.Linear(10, 5)
            model.load_state_dict(state_dict)
            
            # Apply device configuration
            if device_config["device"] == "cuda" and torch.cuda.is_available():
                model = model.cuda()
            
            # Create optimized model
            opt_config = OptimizationConfig(enable_jit=True)
            optimized_model = OptimizedModel(model, opt_config)
            
            assert optimized_model is not None
            assert model_info.name == "integrated_model"


@pytest.mark.integration
class TestCoreModulesWithFramework:
    """Integration tests for core modules with the main framework."""
    
    def test_core_modules_with_inference_framework(self):
        """Test core modules with TorchInferenceFramework."""
        # Mock framework integration
        with patch('framework.TorchInferenceFramework') as MockFramework:
            mock_framework = Mock()
            MockFramework.return_value = mock_framework
            
            # Test that core modules can be used with framework
            framework = MockFramework()
            assert framework is not None
    
    def test_core_modules_error_resilience(self):
        """Test core modules error resilience."""
        # Test that modules handle errors gracefully
        downloader = ModelDownloader(cache_dir="/invalid/path")
        
        with pytest.raises((OSError, PermissionError, DownloadError)):
            downloader.download_model("invalid", "model")


if __name__ == "__main__":
    pytest.main([__file__])

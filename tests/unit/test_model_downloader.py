"""
Unit tests for the model downloader functionality.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn

from framework.core.model_downloader import (
    ModelDownloader, ModelInfo, get_model_downloader,
    download_model, list_available_models
)


class TestModelInfo:
    """Test ModelInfo dataclass."""
    
    def test_model_info_creation(self):
        """Test ModelInfo can be created with required fields."""
        info = ModelInfo(
            name="test_model",
            source="torchvision",
            model_id="resnet18",
            task="classification"
        )
        
        assert info.name == "test_model"
        assert info.source == "torchvision"
        assert info.model_id == "resnet18" 
        assert info.task == "classification"
        assert info.tags == []
    
    def test_model_info_with_optional_fields(self):
        """Test ModelInfo with optional fields."""
        info = ModelInfo(
            name="test_model",
            source="torchvision",
            model_id="resnet18",
            task="classification",
            description="Test model",
            size_mb=44.7,
            license="MIT",
            tags=["test", "vision"]
        )
        
        assert info.description == "Test model"
        assert info.size_mb == 44.7
        assert info.license == "MIT"
        assert info.tags == ["test", "vision"]


class TestModelDownloader:
    """Test ModelDownloader class."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def downloader(self, temp_cache_dir):
        """Create ModelDownloader with temporary cache."""
        return ModelDownloader(temp_cache_dir)
    
    def test_downloader_initialization(self, downloader, temp_cache_dir):
        """Test downloader initializes correctly."""
        assert downloader.cache_dir == temp_cache_dir
        assert downloader.cache_dir.exists()
        assert downloader.registry == {}
    
    def test_cache_dir_creation(self, temp_cache_dir):
        """Test cache directory is created if it doesn't exist."""
        cache_dir = temp_cache_dir / "new_cache"
        assert not cache_dir.exists()
        
        downloader = ModelDownloader(cache_dir)
        assert cache_dir.exists()
        assert downloader.cache_dir == cache_dir
    
    def test_registry_persistence(self, downloader):
        """Test registry is saved and loaded correctly."""
        # Add model to registry
        test_info = {
            "path": "/test/path.pt",
            "info": {
                "name": "test_model",
                "source": "test",
                "model_id": "test_id",
                "task": "classification"
            }
        }
        
        downloader._register_model("test_model", test_info)
        
        # Create new downloader with same cache dir
        new_downloader = ModelDownloader(downloader.cache_dir)
        
        # Check registry was loaded
        assert "test_model" in new_downloader.registry
        assert new_downloader.registry["test_model"] == test_info
    
    def test_get_model_cache_dir(self, downloader):
        """Test model cache directory generation."""
        cache_dir = downloader._get_model_cache_dir("test_model")
        expected = downloader.cache_dir / "test_model"
        assert cache_dir == expected
        
        # Test with special characters
        cache_dir = downloader._get_model_cache_dir("test/model:v1")
        expected = downloader.cache_dir / "test_model_v1"
        assert cache_dir == expected
    
    def test_is_model_cached(self, downloader):
        """Test model cache checking."""
        assert not downloader.is_model_cached("test_model")
        
        # Add model to registry
        downloader._register_model("test_model", {"path": "/test/path.pt"})
        
        assert downloader.is_model_cached("test_model")
    
    def test_list_available_models(self, downloader):
        """Test listing available models."""
        # Initially empty
        models = downloader.list_available_models()
        assert models == {}
        
        # Add model
        test_info = ModelInfo(
            name="test_model",
            source="test", 
            model_id="test_id",
            task="classification"
        )
        
        downloader._register_model("test_model", {
            "path": "/test/path.pt",
            "info": test_info.__dict__
        })
        
        models = downloader.list_available_models()
        assert len(models) == 1
        assert "test_model" in models
        assert isinstance(models["test_model"], ModelInfo)
        assert models["test_model"].name == "test_model"
    
    def test_get_model_info(self, downloader):
        """Test getting model info."""
        # Non-existent model
        info = downloader.get_model_info("nonexistent")
        assert info is None
        
        # Add model
        test_info = ModelInfo(
            name="test_model",
            source="test",
            model_id="test_id", 
            task="classification"
        )
        
        downloader._register_model("test_model", {
            "path": "/test/path.pt",
            "info": test_info.__dict__
        })
        
        info = downloader.get_model_info("test_model")
        assert isinstance(info, ModelInfo)
        assert info.name == "test_model"
    
    def test_get_cache_size(self, downloader, temp_cache_dir):
        """Test cache size calculation."""
        # Initially zero
        size = downloader.get_cache_size()
        assert size == 0
        
        # Create dummy model file
        model_dir = temp_cache_dir / "test_model" 
        model_dir.mkdir()
        model_file = model_dir / "model.pt"
        
        # Create a 1MB file
        with open(model_file, 'wb') as f:
            f.write(b'0' * (1024 * 1024))
        
        # Register model
        downloader._register_model("test_model", {
            "path": str(model_file),
            "info": {}
        })
        
        size = downloader.get_cache_size()
        assert size == 1.0  # 1 MB
    
    @patch('torch.save')
    @patch('torchvision.models.resnet18')
    def test_download_torchvision_model(self, mock_resnet18, mock_torch_save, downloader):
        """Test downloading torchvision model."""
        # Mock model
        mock_model = Mock()
        mock_model.state_dict.return_value = {}
        mock_resnet18.return_value = mock_model
        
        # Mock file operations
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value = Mock(st_size=1024*1024)  # 1MB
            
            # Test download
            model_path, model_info = downloader.download_torchvision_model("resnet18")
        
        # Verify calls
        mock_resnet18.assert_called_once_with(weights='DEFAULT')
        assert mock_torch_save.call_count == 2  # Full model + state dict
        
        # Verify model info
        assert isinstance(model_info, ModelInfo)
        assert model_info.source == "torchvision"
        assert model_info.model_id == "resnet18"
        assert model_info.task == "image-classification"
        
        # Verify registration
        assert downloader.is_model_cached("torchvision_resnet18")
    
    @patch('torch.hub.load')
    def test_download_pytorch_hub_model(self, mock_hub_load, downloader):
        """Test downloading PyTorch Hub model."""
        # Mock model
        mock_model = Mock()
        mock_model.state_dict.return_value = {}
        mock_hub_load.return_value = mock_model
        
        # Test download
        with patch('torch.save') as mock_torch_save, \
             patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value = Mock(st_size=1024*1024)  # 1MB
            
            model_path, model_info = downloader.download_pytorch_hub_model(
                "pytorch/vision", "resnet18"
            )
        
        # Verify calls - PyTorch Hub still uses pretrained parameter
        mock_hub_load.assert_called_once_with(
            "pytorch/vision", "resnet18", pretrained=True
        )
        
        # Verify model info
        assert isinstance(model_info, ModelInfo)
        assert model_info.source == "pytorch_hub"
        assert "pytorch/vision" in model_info.model_id
        assert "resnet18" in model_info.model_id
    
    @patch('requests.get')
    def test_download_from_url(self, mock_get, downloader):
        """Test downloading model from URL."""
        # Mock response
        mock_response = Mock()
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content.return_value = [b'test' * 256]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test download
        model_path, model_info = downloader.download_from_url(
            "https://example.com/model.pt",
            "test_model"
        )
        
        # Verify model info
        assert isinstance(model_info, ModelInfo)
        assert model_info.source == "url"
        assert model_info.model_id == "https://example.com/model.pt"
        
        # Verify file was created
        assert model_path.exists()
    
    @patch('framework.core.model_downloader.TRANSFORMERS_AVAILABLE', True)
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoConfig.from_pretrained')
    def test_download_huggingface_model(self, mock_config, mock_tokenizer, 
                                       mock_model, downloader):
        """Test downloading Hugging Face model."""
        # Mock objects
        mock_model_instance = Mock()
        mock_model_instance.state_dict.return_value = {}
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_config_instance = Mock()
        mock_config_instance.save_pretrained = Mock()
        mock_config.return_value = mock_config_instance
        
        # Test download
        with patch('torch.save') as mock_torch_save, \
             patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value = Mock(st_size=1024*1024)  # 1MB
            
            model_path, model_info = downloader.download_huggingface_model(
                "bert-base-uncased"
            )
        
        # Verify calls
        mock_model.assert_called_once_with("bert-base-uncased")
        mock_tokenizer.assert_called_once_with("bert-base-uncased")
        mock_config.assert_called_once_with("bert-base-uncased")
        
        # Verify model info
        assert isinstance(model_info, ModelInfo)
        assert model_info.source == "huggingface"
        assert model_info.model_id == "bert-base-uncased"
    
    def test_remove_model(self, downloader, temp_cache_dir):
        """Test removing model from cache."""
        # Create dummy model
        model_dir = temp_cache_dir / "test_model"
        model_dir.mkdir()
        model_file = model_dir / "model.pt"
        model_file.write_text("dummy")
        
        # Register model
        downloader._register_model("test_model", {
            "path": str(model_file),
            "info": {}
        })
        
        assert downloader.is_model_cached("test_model")
        assert model_dir.exists()
        
        # Remove model
        success = downloader.remove_model("test_model")
        
        assert success
        assert not downloader.is_model_cached("test_model")
        assert not model_dir.exists()
    
    def test_clear_cache(self, downloader):
        """Test clearing all cached models."""
        # Add multiple models
        for i in range(3):
            downloader._register_model(f"model_{i}", {
                "path": f"/test/path_{i}.pt",
                "info": {}
            })
        
        assert len(downloader.registry) == 3
        
        # Clear cache
        count = downloader.clear_cache()
        
        assert count == 3
        assert len(downloader.registry) == 0


class TestGlobalFunctions:
    """Test global functions."""
    
    @patch('framework.core.model_downloader._global_downloader', None)
    def test_get_model_downloader(self):
        """Test global downloader singleton."""
        downloader1 = get_model_downloader()
        downloader2 = get_model_downloader()
        
        assert downloader1 is downloader2
        assert isinstance(downloader1, ModelDownloader)
    
    @patch('framework.core.model_downloader.get_model_downloader')
    def test_download_model_torchvision(self, mock_get_downloader):
        """Test download_model function with torchvision source."""
        mock_downloader = Mock()
        mock_downloader.download_torchvision_model.return_value = (
            Path("test.pt"), 
            ModelInfo("test", "torchvision", "resnet18", "classification")
        )
        mock_get_downloader.return_value = mock_downloader
        
        model_path, model_info = download_model(
            source="torchvision",
            model_id="resnet18"
        )
        
        mock_downloader.download_torchvision_model.assert_called_once_with(
            "resnet18", custom_name=None
        )
        assert isinstance(model_info, ModelInfo)
    
    @patch('framework.core.model_downloader.get_model_downloader')
    def test_download_model_pytorch_hub(self, mock_get_downloader):
        """Test download_model function with pytorch_hub source."""
        mock_downloader = Mock()
        mock_downloader.download_pytorch_hub_model.return_value = (
            Path("test.pt"),
            ModelInfo("test", "pytorch_hub", "pytorch/vision/resnet18", "classification")
        )
        mock_get_downloader.return_value = mock_downloader
        
        model_path, model_info = download_model(
            source="pytorch_hub",
            model_id="pytorch/vision/resnet18"
        )
        
        mock_downloader.download_pytorch_hub_model.assert_called_once_with(
            "pytorch", "vision/resnet18", None
        )
    
    def test_download_model_invalid_source(self):
        """Test download_model with invalid source."""
        with pytest.raises(ValueError, match="Unsupported source"):
            download_model("invalid_source", "model_id")
    
    def test_download_model_invalid_pytorch_hub_id(self):
        """Test download_model with invalid PyTorch Hub model_id."""
        with pytest.raises(ValueError, match="PyTorch Hub model_id should be in format"):
            download_model("pytorch_hub", "invalid_format")
    
    @patch('framework.core.model_downloader.get_model_downloader')
    def test_list_available_models(self, mock_get_downloader):
        """Test list_available_models function."""
        mock_downloader = Mock()
        mock_models = {
            "model1": ModelInfo("model1", "torchvision", "resnet18", "classification")
        }
        mock_downloader.list_available_models.return_value = mock_models
        mock_get_downloader.return_value = mock_downloader
        
        models = list_available_models()
        
        assert models == mock_models
        mock_downloader.list_available_models.assert_called_once()


class TestModelDownloaderErrorHandling:
    """Test error handling in model downloader."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def downloader(self, temp_cache_dir):
        """Create ModelDownloader with temporary cache."""
        return ModelDownloader(temp_cache_dir)
    
    @patch('framework.core.model_downloader.TORCHVISION_AVAILABLE', False)
    def test_torchvision_not_available(self, downloader):
        """Test error when torchvision is not available."""
        with pytest.raises(ImportError, match="torchvision not available"):
            downloader.download_torchvision_model("resnet18")
    
    @patch('framework.core.model_downloader.TRANSFORMERS_AVAILABLE', False)
    def test_transformers_not_available(self, downloader):
        """Test error when transformers is not available."""
        with pytest.raises(ImportError, match="transformers not available"):
            downloader.download_huggingface_model("bert-base-uncased")
    
    @patch('requests.get')
    def test_url_download_network_error(self, mock_get, downloader):
        """Test network error during URL download."""
        mock_get.side_effect = Exception("Network error")
        
        with pytest.raises(Exception, match="Network error"):
            downloader.download_from_url("https://example.com/model.pt", "test_model")
    
    @patch('requests.get')
    def test_url_download_hash_mismatch(self, mock_get, downloader):
        """Test hash mismatch during URL download."""
        # Mock response
        mock_response = Mock()
        mock_response.headers = {'content-length': '4'}
        mock_response.iter_content.return_value = [b'test']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test with wrong hash
        with pytest.raises(ValueError, match="Hash mismatch"):
            downloader.download_from_url(
                "https://example.com/model.pt", 
                "test_model",
                expected_hash="wrong_hash"
            )
    
    def test_get_model_path_missing_file(self, downloader, temp_cache_dir):
        """Test get_model_path when file is missing."""
        # Register model with non-existent file
        downloader._register_model("test_model", {
            "path": str(temp_cache_dir / "nonexistent.pt"),
            "info": {}
        })
        
        # Should return None and remove from registry
        path = downloader.get_model_path("test_model")
        assert path is None
        assert not downloader.is_model_cached("test_model")
    
    def test_remove_nonexistent_model(self, downloader):
        """Test removing non-existent model."""
        success = downloader.remove_model("nonexistent")
        assert not success


class TestModelDownloaderIntegration:
    """Integration tests for model downloader."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_full_download_cycle(self, temp_cache_dir):
        """Test full download, cache, and cleanup cycle."""
        downloader = ModelDownloader(temp_cache_dir)
        
        # Mock torchvision download
        with patch('torchvision.models.resnet18') as mock_resnet18, \
             patch('torch.save') as mock_torch_save, \
             patch('pathlib.Path.stat') as mock_stat:
            
            mock_model = Mock()
            mock_model.state_dict.return_value = {}
            mock_resnet18.return_value = mock_model
            mock_stat.return_value = Mock(st_size=1024*1024)  # 1MB
            
            # Download model
            model_path, model_info = downloader.download_torchvision_model("resnet18")
            
            # Verify model is cached
            assert downloader.is_model_cached("torchvision_resnet18")
            
            # Verify we can get model info
            cached_info = downloader.get_model_info("torchvision_resnet18")
            assert cached_info.name == model_info.name
            
            # Verify we can list models
            models = downloader.list_available_models()
            assert "torchvision_resnet18" in models
            
            # Verify we can remove model
            success = downloader.remove_model("torchvision_resnet18")
            assert success
            assert not downloader.is_model_cached("torchvision_resnet18")
    
    def test_registry_persistence_across_instances(self, temp_cache_dir):
        """Test registry persists across downloader instances."""
        # First downloader
        downloader1 = ModelDownloader(temp_cache_dir)
        downloader1._register_model("test_model", {
            "path": "/test/path.pt",
            "info": {"name": "test_model", "source": "test", "model_id": "test", "task": "test"}
        })
        
        # Second downloader with same cache
        downloader2 = ModelDownloader(temp_cache_dir)
        
        # Should have loaded the registry
        assert downloader2.is_model_cached("test_model")
        assert "test_model" in downloader2.registry
        
        # Modifications should persist
        downloader2.remove_model("test_model")
        
        # Third downloader
        downloader3 = ModelDownloader(temp_cache_dir)
        assert not downloader3.is_model_cached("test_model")

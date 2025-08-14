"""
Integration tests for model downloader with the framework.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from framework.core.model_downloader import ModelDownloader, ModelInfo
from framework.core.base_model import get_model_manager, ModelManager
from framework import TorchInferenceFramework
from framework.core.config import InferenceConfig


class TestModelDownloaderFrameworkIntegration:
    """Test integration between model downloader and the framework."""
    
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
    
    @pytest.fixture
    def model_manager(self):
        """Create a fresh ModelManager instance."""
        return ModelManager()
    
    def test_model_manager_downloader_integration(self, model_manager, temp_cache_dir):
        """Test ModelManager integration with downloader."""
        # Patch to use our temp cache
        with patch('framework.core.model_downloader.get_model_downloader') as mock_get:
            mock_downloader = ModelDownloader(temp_cache_dir)
            mock_get.return_value = mock_downloader
            
            # Get downloader from model manager
            downloader = model_manager.get_downloader()
            
            assert isinstance(downloader, ModelDownloader)
            assert downloader.cache_dir == temp_cache_dir
    
    @patch('torchvision.models.resnet18')
    @patch('torch.save')
    def test_model_manager_download_and_load(self, mock_torch_save, mock_resnet18, model_manager):
        """Test downloading and loading model through ModelManager."""
        # Mock model
        mock_model = Mock()
        mock_model.state_dict.return_value = {}
        mock_resnet18.return_value = mock_model
        
        # Mock model loading
        with patch('framework.adapters.model_adapters.ModelAdapterFactory.create_adapter') as mock_adapter_factory:
            mock_adapter = Mock()
            mock_adapter.load_model = Mock()
            mock_adapter.optimize_for_inference = Mock()
            mock_adapter.warmup = Mock()
            mock_adapter_factory.return_value = mock_adapter
            
            # Use temporary directory for cache to avoid path issues
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Patch the default cache directory in Path.home()
                with patch('pathlib.Path.home') as mock_home:
                    mock_home.return_value = Path(temp_dir)
                    
                    # Mock the size calculation specifically
                    def mock_save_with_file_creation(obj, path, *args, **kwargs):
                        # Ensure parent directory exists
                        Path(path).parent.mkdir(parents=True, exist_ok=True)
                        # Create a dummy file to simulate the save
                        Path(path).touch()
                        # Write some dummy data to give it a size
                        Path(path).write_bytes(b'x' * 1024 * 1024)  # 1MB dummy file
                        return None
                    
                    mock_torch_save.side_effect = mock_save_with_file_creation
                    
                    # Download and load model
                    model_manager.download_and_load_model(
                        source="torchvision",
                        model_id="resnet18",
                        name="test_resnet"
                    )
            
            # Verify model was registered
            assert "test_resnet" in model_manager.list_models()
            
            # Verify adapter methods were called
            mock_adapter.load_model.assert_called_once()
            mock_adapter.optimize_for_inference.assert_called_once()
            mock_adapter.warmup.assert_called_once()
    
    def test_model_manager_list_available_downloads(self, model_manager, temp_cache_dir):
        """Test listing available downloads through ModelManager."""
        # Create mock downloader with cached model
        with patch('framework.core.model_downloader.get_model_downloader') as mock_get:
            mock_downloader = ModelDownloader(temp_cache_dir)
            
            # Add a mock cached model
            test_info = ModelInfo(
                name="test_model",
                source="torchvision",
                model_id="resnet18",
                task="classification"
            )
            
            mock_downloader._register_model("test_model", {
                "path": "/test/path.pt",
                "info": test_info.__dict__
            })
            
            mock_get.return_value = mock_downloader
            
            # Get available downloads
            available = model_manager.list_available_downloads()
            
            assert "test_model" in available
            assert isinstance(available["test_model"], ModelInfo)
    
    def test_model_manager_get_download_info(self, model_manager, temp_cache_dir):
        """Test getting download info through ModelManager."""
        with patch('framework.core.model_downloader.get_model_downloader') as mock_get:
            mock_downloader = ModelDownloader(temp_cache_dir)
            
            # Add a mock cached model
            test_info = ModelInfo(
                name="test_model",
                source="torchvision",
                model_id="resnet18",
                task="classification"
            )
            
            mock_downloader._register_model("test_model", {
                "path": "/test/path.pt",
                "info": test_info.__dict__
            })
            
            mock_get.return_value = mock_downloader
            
            # Get download info
            info = model_manager.get_download_info("test_model")
            
            assert info is not None
            assert info["name"] == "test_model"
            assert info["source"] == "torchvision"
    
    @patch('torchvision.models.resnet18')
    @patch('torch.save')
    def test_framework_download_and_load_model(self, mock_torch_save, mock_resnet18):
        """Test downloading and loading model through TorchInferenceFramework."""
        # Mock model
        mock_model = Mock()
        mock_model.state_dict.return_value = {}
        mock_resnet18.return_value = mock_model
        
        framework = TorchInferenceFramework()
        
        # Mock model loading
        with patch('framework.load_model') as mock_load_model:
            mock_adapter = Mock()
            mock_load_model.return_value = mock_adapter
            
            # Mock inference engine creation
            with patch('framework.create_inference_engine') as mock_create_engine:
                mock_engine = Mock()
                mock_create_engine.return_value = mock_engine
                
                # Use temporary directory for cache to avoid path issues
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Patch the default cache directory in Path.home()
                    with patch('pathlib.Path.home') as mock_home:
                        mock_home.return_value = Path(temp_dir)
                        
                        # Mock the size calculation specifically
                        def mock_save_with_file_creation(obj, path, *args, **kwargs):
                            # Ensure parent directory exists
                            Path(path).parent.mkdir(parents=True, exist_ok=True)
                            # Create a dummy file to simulate the save
                            Path(path).touch()
                            # Write some dummy data to give it a size
                            Path(path).write_bytes(b'x' * 1024 * 1024)  # 1MB dummy file
                            return None
                        
                        mock_torch_save.side_effect = mock_save_with_file_creation
                        
                        # Download and load model
                        framework.download_and_load_model(
                            source="torchvision",
                            model_id="resnet18",
                            model_name="test_resnet"
                        )
                
                # Verify framework state
                assert framework._initialized
                assert framework.model is mock_adapter
                assert framework.engine is mock_engine
    
    def test_framework_list_available_downloads(self):
        """Test listing available downloads through TorchInferenceFramework."""
        framework = TorchInferenceFramework()
        
        # Mock the function at the framework level where it's actually imported
        with patch('framework.list_available_models') as mock_list:
            mock_models = {
                "test_model": ModelInfo("test", "torchvision", "resnet18", "classification")
            }
            mock_list.return_value = mock_models
            
            available = framework.list_available_downloads()
            
            assert available == mock_models
            mock_list.assert_called_once()
    
    def test_framework_get_model_downloader(self):
        """Test getting model downloader through TorchInferenceFramework."""
        framework = TorchInferenceFramework()
        
        # Mock the function at the framework level where it's actually imported
        with patch('framework.get_model_downloader') as mock_get:
            mock_downloader = Mock()
            mock_get.return_value = mock_downloader
            
            downloader = framework.get_model_downloader()
            
            assert downloader is mock_downloader
            mock_get.assert_called_once()


class TestConvenienceFunctions:
    """Test convenience functions for downloading models."""
    
    @patch('framework.TorchInferenceFramework.download_and_load_model')
    def test_download_resnet18(self, mock_download_and_load):
        """Test download_resnet18 convenience function."""
        from framework import download_resnet18
        
        mock_framework = Mock()
        
        with patch('framework.TorchInferenceFramework') as mock_framework_class:
            mock_framework_class.return_value = mock_framework
            
            result = download_resnet18(pretrained=True)
            
            assert result is mock_framework
            mock_framework.download_and_load_model.assert_called_once_with(
                source="torchvision",
                model_id="resnet18",
                pretrained=True
            )
    
    @patch('framework.TorchInferenceFramework.download_and_load_model')
    def test_download_pytorch_hub_model(self, mock_download_and_load):
        """Test download_pytorch_hub_model convenience function."""
        from framework import download_pytorch_hub_model
        
        mock_framework = Mock()
        
        with patch('framework.TorchInferenceFramework') as mock_framework_class:
            mock_framework_class.return_value = mock_framework
            
            result = download_pytorch_hub_model("pytorch/vision", "resnet50", pretrained=True)
            
            assert result is mock_framework
            mock_framework.download_and_load_model.assert_called_once_with(
                source="pytorch_hub",
                model_id="pytorch/vision/resnet50",
                pretrained=True
            )
    
    @patch('framework.TorchInferenceFramework.download_and_load_model')
    def test_download_huggingface_model(self, mock_download_and_load):
        """Test download_huggingface_model convenience function."""
        from framework import download_huggingface_model
        
        mock_framework = Mock()
        
        with patch('framework.TorchInferenceFramework') as mock_framework_class:
            mock_framework_class.return_value = mock_framework
            
            result = download_huggingface_model("bert-base-uncased", task="text-classification")
            
            assert result is mock_framework
            mock_framework.download_and_load_model.assert_called_once_with(
                source="huggingface",
                model_id="bert-base-uncased",
                task="text-classification"
            )


class TestModelDownloaderCLIIntegration:
    """Test CLI integration with model downloader."""
    
    def test_cli_download_command(self):
        """Test CLI download command integration."""
        from framework.scripts.download_models import download_command
        
        # Mock args
        args = Mock()
        args.source = "torchvision"
        args.model_id = "resnet18"
        args.name = None
        args.pretrained = True
        
        # Mock download_model function
        with patch('framework.scripts.download_models.download_model') as mock_download:
            mock_model_info = ModelInfo(
                name="torchvision_resnet18",
                source="torchvision",
                model_id="resnet18", 
                task="classification",
                size_mb=44.7,
                description="Test model"
            )
            mock_download.return_value = (Path("test.pt"), mock_model_info)
            
            result = download_command(args)
            
            assert result == 0  # Success
            mock_download.assert_called_once_with(
                source="torchvision",
                model_id="resnet18",
                model_name=None,
                pretrained=True
            )
    
    def test_cli_list_command(self):
        """Test CLI list command integration."""
        from framework.scripts.download_models import list_command
        
        args = Mock()
        
        # Mock list_available_models
        with patch('framework.scripts.download_models.list_available_models') as mock_list:
            mock_models = {
                "test_model": ModelInfo(
                    name="test_model",
                    source="torchvision",
                    model_id="resnet18",
                    task="classification",
                    size_mb=44.7,
                    description="Test model",
                    tags=["test", "vision"]
                )
            }
            mock_list.return_value = mock_models
            
            result = list_command(args)
            
            assert result == 0  # Success
            mock_list.assert_called_once()
    
    def test_cli_info_command(self):
        """Test CLI info command integration."""
        from framework.scripts.download_models import info_command
        
        args = Mock()
        args.name = "test_model"
        
        # Mock downloader
        with patch('framework.scripts.download_models.get_model_downloader') as mock_get:
            mock_downloader = Mock()
            mock_info = ModelInfo(
                name="test_model",
                source="torchvision",
                model_id="resnet18",
                task="classification",
                size_mb=44.7,
                description="Test model",
                license="MIT",
                tags=["test", "vision"]
            )
            mock_downloader.get_model_info.return_value = mock_info
            mock_downloader.get_model_path.return_value = Path("/test/path.pt")
            mock_get.return_value = mock_downloader
            
            result = info_command(args)
            
            assert result == 0  # Success
            mock_downloader.get_model_info.assert_called_once_with("test_model")
    
    def test_cli_remove_command(self):
        """Test CLI remove command integration."""
        from framework.scripts.download_models import remove_command
        
        args = Mock()
        args.name = "test_model"
        
        # Mock downloader
        with patch('framework.scripts.download_models.get_model_downloader') as mock_get:
            mock_downloader = Mock()
            mock_downloader.is_model_cached.return_value = True
            mock_downloader.remove_model.return_value = True
            mock_get.return_value = mock_downloader
            
            result = remove_command(args)
            
            assert result == 0  # Success
            mock_downloader.remove_model.assert_called_once_with("test_model")
    
    def test_cli_cache_command(self):
        """Test CLI cache command integration."""
        from framework.scripts.download_models import cache_command
        
        args = Mock()
        
        # Mock downloader
        with patch('framework.scripts.download_models.get_model_downloader') as mock_get:
            mock_downloader = Mock()
            mock_downloader.cache_dir = Path("/test/cache")
            mock_downloader.registry = {"model1": {}, "model2": {}}
            mock_downloader.get_cache_size.return_value = 100.5
            mock_get.return_value = mock_downloader
            
            result = cache_command(args)
            
            assert result == 0  # Success
            mock_downloader.get_cache_size.assert_called_once()


class TestModelDownloaderErrorScenarios:
    """Test error scenarios in model downloader integration."""
    
    def test_framework_download_with_invalid_source(self):
        """Test framework error handling with invalid source."""
        framework = TorchInferenceFramework()
        
        with pytest.raises(ValueError, match="Unsupported source"):
            framework.download_and_load_model(
                source="invalid_source",
                model_id="test_model"
            )
    
    def test_model_manager_download_nonexistent_model(self, model_manager):
        """Test error handling when downloading non-existent model."""
        # Initialize the downloader first
        downloader = model_manager.get_downloader()
        
        # Mock downloader to raise exception
        with patch.object(downloader, 'download_torchvision_model') as mock_download:
            mock_download.side_effect = Exception("Model not found")
            
            with pytest.raises(Exception, match="Model not found"):
                model_manager.download_and_load_model(
                    source="torchvision",
                    model_id="nonexistent_model",
                    name="test"
                )
    
    def test_cli_download_error_handling(self):
        """Test CLI error handling during download."""
        from framework.scripts.download_models import download_command
        
        args = Mock()
        args.source = "torchvision"
        args.model_id = "nonexistent"
        args.name = None
        args.pretrained = True
        
        # Mock download to fail
        with patch('framework.scripts.download_models.download_model') as mock_download:
            mock_download.side_effect = Exception("Download failed")
            
            result = download_command(args)
            
            assert result == 1  # Failure
    
    def test_cli_info_model_not_found(self):
        """Test CLI info command when model not found."""
        from framework.scripts.download_models import info_command
        
        args = Mock()
        args.name = "nonexistent_model"
        
        with patch('framework.scripts.download_models.get_model_downloader') as mock_get:
            mock_downloader = Mock()
            mock_downloader.get_model_info.return_value = None
            mock_get.return_value = mock_downloader
            
            result = info_command(args)
            
            assert result == 1  # Failure
    
    def test_cli_remove_model_not_found(self):
        """Test CLI remove command when model not found."""
        from framework.scripts.download_models import remove_command
        
        args = Mock()
        args.name = "nonexistent_model"
        
        with patch('framework.scripts.download_models.get_model_downloader') as mock_get:
            mock_downloader = Mock()
            mock_downloader.is_model_cached.return_value = False
            mock_get.return_value = mock_downloader
            
            result = remove_command(args)
            
            assert result == 1  # Failure

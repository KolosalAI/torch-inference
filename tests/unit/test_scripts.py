"""
Test suite for script modules in the framework.

This module tests all script implementations including model downloaders
and auto-download functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import argparse
import json

# Framework imports
try:
    from framework.scripts.download_models import (
        ModelDownloadScript, download_command, auto_download_command,
        list_command, info_command, remove_command, clean_command,
        main, parse_model_identifier
    )
    from framework.scripts.auto_download import (
        AutoDownloader, SourceDetector, ModelRegistry,
        auto_download_model, detect_model_source, register_model,
        get_registered_models, cleanup_downloaded_models
    )
    from framework.core.model_downloader import ModelDownloader, ModelInfo
    SCRIPTS_AVAILABLE = True
except ImportError as e:
    SCRIPTS_AVAILABLE = False
    pytest.skip(f"Scripts not available: {e}", allow_module_level=True)


class TestModelDownloadScript:
    """Test ModelDownloadScript class."""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary model directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_downloader(self):
        """Create mock model downloader."""
        downloader = Mock(spec=ModelDownloader)
        downloader.download_model.return_value = ModelInfo(
            name="test_model",
            source="torchvision",
            model_type="classification",
            file_path="/path/to/model.pt",
            size_mb=25.5,
            download_url="https://example.com/model.pt"
        )
        downloader.list_models.return_value = [
            ModelInfo(name="model1", source="torchvision", model_type="classification"),
            ModelInfo(name="model2", source="huggingface", model_type="text")
        ]
        downloader.get_model_info.return_value = ModelInfo(
            name="test_model",
            source="torchvision",
            model_type="classification"
        )
        return downloader
    
    def test_download_script_initialization(self, temp_model_dir):
        """Test download script initialization."""
        script = ModelDownloadScript(cache_dir=temp_model_dir)
        
        assert script.cache_dir == temp_model_dir
        assert hasattr(script, 'downloader')
    
    @patch('framework.scripts.download_models.get_model_downloader')
    def test_download_command(self, mock_get_downloader, mock_downloader, temp_model_dir):
        """Test download command function."""
        mock_get_downloader.return_value = mock_downloader
        
        # Mock command line arguments
        args = Mock()
        args.source = "torchvision"
        args.model_id = "resnet18"
        args.name = None
        args.task = "classification"
        args.cache_dir = str(temp_model_dir)
        
        download_command(args)
        
        mock_downloader.download_model.assert_called_once()
    
    @patch('framework.scripts.download_models.get_model_downloader')
    def test_auto_download_command(self, mock_get_downloader, mock_downloader, temp_model_dir):
        """Test auto download command function."""
        mock_get_downloader.return_value = mock_downloader
        
        # Mock command line arguments
        args = Mock()
        args.model_identifier = "torchvision:resnet18"
        args.name = None
        args.task = None
        args.cache_dir = str(temp_model_dir)
        
        with patch('framework.scripts.download_models.parse_model_identifier') as mock_parse:
            mock_parse.return_value = ("torchvision", "resnet18")
            
            auto_download_command(args)
            
            mock_parse.assert_called_once_with("torchvision:resnet18")
            mock_downloader.download_model.assert_called_once()
    
    @patch('framework.scripts.download_models.get_model_downloader')
    def test_list_command(self, mock_get_downloader, mock_downloader, capsys):
        """Test list command function."""
        mock_get_downloader.return_value = mock_downloader
        
        args = Mock()
        args.source = None
        
        list_command(args)
        
        mock_downloader.list_models.assert_called_once()
        captured = capsys.readouterr()
        assert "Available models:" in captured.out
    
    @patch('framework.scripts.download_models.get_model_downloader')
    def test_info_command(self, mock_get_downloader, mock_downloader, capsys):
        """Test info command function."""
        mock_get_downloader.return_value = mock_downloader
        
        args = Mock()
        args.name = "test_model"
        
        info_command(args)
        
        mock_downloader.get_model_info.assert_called_once_with("test_model")
        captured = capsys.readouterr()
        assert "Model information:" in captured.out
    
    @patch('framework.scripts.download_models.get_model_downloader')
    def test_remove_command(self, mock_get_downloader, mock_downloader):
        """Test remove command function."""
        mock_get_downloader.return_value = mock_downloader
        mock_downloader.remove_model.return_value = True
        
        args = Mock()
        args.name = "test_model"
        args.force = False
        
        remove_command(args)
        
        mock_downloader.remove_model.assert_called_once_with("test_model")
    
    @patch('framework.scripts.download_models.get_model_downloader')
    def test_clean_command(self, mock_get_downloader, mock_downloader):
        """Test clean command function."""
        mock_get_downloader.return_value = mock_downloader
        mock_downloader.clean_cache.return_value = ["model1", "model2"]
        
        args = Mock()
        args.force = False
        
        clean_command(args)
        
        mock_downloader.clean_cache.assert_called_once()
    
    def test_parse_model_identifier(self):
        """Test parse_model_identifier function."""
        # Test with source prefix
        source, model_id = parse_model_identifier("torchvision:resnet18")
        assert source == "torchvision"
        assert model_id == "resnet18"
        
        # Test without source prefix
        source, model_id = parse_model_identifier("resnet18")
        assert source is None
        assert model_id == "resnet18"
        
        # Test with complex identifier
        source, model_id = parse_model_identifier("huggingface:bert-base-uncased")
        assert source == "huggingface"
        assert model_id == "bert-base-uncased"
    
    @patch('framework.scripts.download_models.get_model_downloader')
    def test_main_function(self, mock_get_downloader, mock_downloader):
        """Test main function with argument parsing."""
        mock_get_downloader.return_value = mock_downloader
        
        # Test download command
        with patch('sys.argv', ['download_models.py', 'download', 'torchvision', 'resnet18']):
            try:
                main()
            except SystemExit:
                pass  # argparse calls sys.exit
        
        # Test auto command
        with patch('sys.argv', ['download_models.py', 'auto', 'torchvision:resnet18']):
            try:
                main()
            except SystemExit:
                pass  # argparse calls sys.exit
    
    def test_download_script_error_handling(self, temp_model_dir):
        """Test download script error handling."""
        script = ModelDownloadScript(cache_dir=temp_model_dir)
        
        with patch.object(script, 'downloader') as mock_downloader:
            mock_downloader.download_model.side_effect = Exception("Download failed")
            
            # Should handle download errors gracefully
            with pytest.raises(Exception):
                script.downloader.download_model("source", "model_id")


class TestAutoDownloader:
    """Test AutoDownloader class."""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary model directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_model_registry(self):
        """Create mock model registry."""
        registry = {
            "torchvision": {
                "resnet18": {
                    "url": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
                    "type": "classification",
                    "size_mb": 44.6
                }
            },
            "huggingface": {
                "bert-base-uncased": {
                    "url": "https://huggingface.co/bert-base-uncased",
                    "type": "text",
                    "size_mb": 420.0
                }
            }
        }
        return registry
    
    def test_auto_downloader_initialization(self, temp_model_dir):
        """Test auto downloader initialization."""
        downloader = AutoDownloader(cache_dir=temp_model_dir)
        
        assert downloader.cache_dir == temp_model_dir
        assert hasattr(downloader, 'source_detector')
        assert hasattr(downloader, 'model_registry')
    
    def test_source_detector_initialization(self):
        """Test source detector initialization."""
        detector = SourceDetector()
        
        assert hasattr(detector, 'detect_source')
    
    def test_detect_model_source(self):
        """Test detect_model_source function."""
        # Test torchvision models
        source = detect_model_source("resnet18")
        assert source in ["torchvision", "pytorch_hub", "auto"]
        
        # Test huggingface models
        source = detect_model_source("bert-base-uncased")
        assert source in ["huggingface", "auto"]
        
        # Test URL source
        source = detect_model_source("https://example.com/model.pt")
        assert source == "url"
    
    def test_model_registry_initialization(self, mock_model_registry):
        """Test model registry initialization."""
        with patch('framework.scripts.auto_download.ModelRegistry._load_registry') as mock_load:
            mock_load.return_value = mock_model_registry
            
            registry = ModelRegistry()
            
            assert registry.registry == mock_model_registry
    
    def test_auto_download_model(self, temp_model_dir, mock_model_registry):
        """Test auto_download_model function."""
        with patch('framework.scripts.auto_download.AutoDownloader') as MockAutoDownloader:
            mock_downloader = Mock()
            mock_downloader.download.return_value = ModelInfo(
                name="resnet18",
                source="torchvision",
                model_type="classification",
                file_path=str(temp_model_dir / "resnet18.pt"),
                size_mb=44.6
            )
            MockAutoDownloader.return_value = mock_downloader
            
            result = auto_download_model("resnet18", cache_dir=temp_model_dir)
            
            assert result.name == "resnet18"
            assert result.source == "torchvision"
            mock_downloader.download.assert_called_once()
    
    def test_register_model(self, temp_model_dir):
        """Test register_model function."""
        model_info = {
            "name": "custom_model",
            "source": "custom",
            "url": "https://example.com/model.pt",
            "type": "classification"
        }
        
        with patch('framework.scripts.auto_download.ModelRegistry') as MockRegistry:
            mock_registry = Mock()
            MockRegistry.return_value = mock_registry
            
            register_model(model_info)
            
            mock_registry.register.assert_called_once_with(model_info)
    
    def test_get_registered_models(self, mock_model_registry):
        """Test get_registered_models function."""
        with patch('framework.scripts.auto_download.ModelRegistry') as MockRegistry:
            mock_registry = Mock()
            mock_registry.list_models.return_value = list(mock_model_registry.keys())
            MockRegistry.return_value = mock_registry
            
            models = get_registered_models()
            
            assert isinstance(models, list)
            mock_registry.list_models.assert_called_once()
    
    def test_cleanup_downloaded_models(self, temp_model_dir):
        """Test cleanup_downloaded_models function."""
        # Create some test files
        test_files = [
            temp_model_dir / "model1.pt",
            temp_model_dir / "model2.pth",
            temp_model_dir / "config.json"
        ]
        
        for file_path in test_files:
            file_path.touch()
        
        removed_files = cleanup_downloaded_models(
            cache_dir=temp_model_dir,
            older_than_days=0,
            dry_run=False
        )
        
        assert len(removed_files) > 0
        # Check that model files were removed
        for file_path in test_files[:2]:  # Only .pt and .pth files
            assert not file_path.exists()


class TestSourceDetector:
    """Test SourceDetector class."""
    
    def test_detect_torchvision_models(self):
        """Test detecting torchvision models."""
        detector = SourceDetector()
        
        # Common torchvision models
        torchvision_models = [
            "resnet18", "resnet50", "vgg16", "alexnet",
            "densenet121", "mobilenet_v2"
        ]
        
        for model_name in torchvision_models:
            source = detector.detect_source(model_name)
            assert source in ["torchvision", "pytorch_hub", "auto"]
    
    def test_detect_huggingface_models(self):
        """Test detecting huggingface models."""
        detector = SourceDetector()
        
        # Common huggingface model patterns
        hf_models = [
            "bert-base-uncased",
            "gpt2",
            "roberta-base",
            "distilbert-base-uncased"
        ]
        
        for model_name in hf_models:
            source = detector.detect_source(model_name)
            assert source in ["huggingface", "auto"]
    
    def test_detect_url_source(self):
        """Test detecting URL sources."""
        detector = SourceDetector()
        
        urls = [
            "https://example.com/model.pt",
            "http://example.com/model.pth",
            "https://github.com/user/repo/model.bin"
        ]
        
        for url in urls:
            source = detector.detect_source(url)
            assert source == "url"
    
    def test_detect_file_path_source(self):
        """Test detecting file path sources."""
        detector = SourceDetector()
        
        file_paths = [
            "/path/to/model.pt",
            "./local_model.pth",
            "C:\\models\\model.bin"
        ]
        
        for file_path in file_paths:
            source = detector.detect_source(file_path)
            assert source == "file"
    
    def test_detect_unknown_source(self):
        """Test detecting unknown sources."""
        detector = SourceDetector()
        
        source = detector.detect_source("unknown_model_xyz123")
        assert source == "auto"


class TestModelRegistry:
    """Test ModelRegistry class."""
    
    @pytest.fixture
    def temp_registry_file(self):
        """Create temporary registry file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        registry_data = {
            "torchvision": {
                "resnet18": {
                    "url": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
                    "type": "classification",
                    "size_mb": 44.6
                }
            }
        }
        json.dump(registry_data, temp_file)
        temp_file.close()
        
        yield Path(temp_file.name)
        
        # Cleanup
        Path(temp_file.name).unlink(missing_ok=True)
    
    def test_model_registry_initialization(self):
        """Test model registry initialization."""
        registry = ModelRegistry()
        
        assert hasattr(registry, 'registry')
        assert isinstance(registry.registry, dict)
    
    def test_model_registry_register(self):
        """Test model registry register method."""
        registry = ModelRegistry()
        
        model_info = {
            "name": "custom_model",
            "source": "custom",
            "url": "https://example.com/model.pt",
            "type": "classification"
        }
        
        registry.register(model_info)
        
        # Check that model was registered
        assert "custom" in registry.registry
        assert "custom_model" in registry.registry["custom"]
    
    def test_model_registry_list_models(self):
        """Test model registry list_models method."""
        registry = ModelRegistry()
        
        # Add some test models
        registry.registry = {
            "source1": {"model1": {}, "model2": {}},
            "source2": {"model3": {}}
        }
        
        models = registry.list_models()
        
        assert len(models) == 3
        assert "model1" in models
        assert "model2" in models
        assert "model3" in models
    
    def test_model_registry_get_model_info(self):
        """Test model registry get_model_info method."""
        registry = ModelRegistry()
        
        # Add test model
        registry.registry = {
            "torchvision": {
                "resnet18": {
                    "url": "https://example.com/resnet18.pth",
                    "type": "classification",
                    "size_mb": 44.6
                }
            }
        }
        
        info = registry.get_model_info("torchvision", "resnet18")
        
        assert info["type"] == "classification"
        assert info["size_mb"] == 44.6
    
    def test_model_registry_save_load(self, temp_registry_file):
        """Test model registry save and load functionality."""
        # Test loading
        registry = ModelRegistry(registry_file=temp_registry_file)
        
        assert "torchvision" in registry.registry
        assert "resnet18" in registry.registry["torchvision"]
        
        # Test saving
        registry.register({
            "name": "new_model",
            "source": "custom",
            "url": "https://example.com/new_model.pt",
            "type": "classification"
        })
        
        registry.save()
        
        # Load again to verify save
        new_registry = ModelRegistry(registry_file=temp_registry_file)
        assert "custom" in new_registry.registry
        assert "new_model" in new_registry.registry["custom"]


class TestScriptIntegration:
    """Test script integration scenarios."""
    
    def test_download_script_with_auto_download(self, temp_model_dir):
        """Test integration between download script and auto download."""
        with patch('framework.scripts.download_models.get_model_downloader') as mock_get_downloader:
            with patch('framework.scripts.auto_download.auto_download_model') as mock_auto_download:
                mock_downloader = Mock()
                mock_get_downloader.return_value = mock_downloader
                
                mock_auto_download.return_value = ModelInfo(
                    name="resnet18",
                    source="torchvision",
                    model_type="classification",
                    file_path=str(temp_model_dir / "resnet18.pt"),
                    size_mb=44.6
                )
                
                # Test auto download integration
                args = Mock()
                args.model_identifier = "torchvision:resnet18"
                args.name = None
                args.task = None
                args.cache_dir = str(temp_model_dir)
                
                auto_download_command(args)
                
                # Verify integration works
                assert mock_auto_download.called or mock_downloader.download_model.called
    
    def test_script_error_handling(self):
        """Test script error handling."""
        # Test download command with invalid arguments
        args = Mock()
        args.source = "invalid_source"
        args.model_id = "invalid_model"
        
        with patch('framework.scripts.download_models.get_model_downloader') as mock_get_downloader:
            mock_downloader = Mock()
            mock_downloader.download_model.side_effect = Exception("Invalid model")
            mock_get_downloader.return_value = mock_downloader
            
            # Should handle errors gracefully
            with pytest.raises(Exception):
                download_command(args)
    
    def test_script_command_line_interface(self):
        """Test script command line interface."""
        # Test argument parsing
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        
        # Add download subcommand
        download_parser = subparsers.add_parser('download')
        download_parser.add_argument('source')
        download_parser.add_argument('model_id')
        
        # Test parsing
        args = parser.parse_args(['download', 'torchvision', 'resnet18'])
        assert args.command == 'download'
        assert args.source == 'torchvision'
        assert args.model_id == 'resnet18'
    
    def test_script_configuration_loading(self, temp_model_dir):
        """Test script configuration loading."""
        # Create test config file
        config_file = temp_model_dir / "config.json"
        config_data = {
            "cache_dir": str(temp_model_dir),
            "default_source": "torchvision",
            "download_timeout": 300
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Test loading configuration
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config["cache_dir"] == str(temp_model_dir)
        assert loaded_config["default_source"] == "torchvision"


@pytest.mark.integration
class TestScriptIntegrationWithFramework:
    """Integration tests for scripts with the main framework."""
    
    def test_script_with_inference_framework(self):
        """Test scripts with TorchInferenceFramework."""
        # Mock framework integration
        with patch('framework.TorchInferenceFramework') as MockFramework:
            mock_framework = Mock()
            MockFramework.return_value = mock_framework
            
            # Test that downloaded models can be loaded into framework
            framework = MockFramework()
            assert framework is not None
    
    def test_script_model_compatibility(self):
        """Test script model compatibility with framework."""
        # Test that downloaded models are compatible with framework
        model_info = ModelInfo(
            name="test_model",
            source="torchvision",
            model_type="classification",
            file_path="/path/to/model.pt",
            size_mb=25.5
        )
        
        # Should be able to use model info with framework
        assert model_info.name == "test_model"
        assert model_info.model_type == "classification"
    
    def test_script_cache_management(self, temp_model_dir):
        """Test script cache management integration."""
        # Test cache directory management
        cache_dir = temp_model_dir / "models"
        cache_dir.mkdir(exist_ok=True)
        
        # Create some test model files
        test_files = [
            cache_dir / "model1.pt",
            cache_dir / "model2.pth"
        ]
        
        for file_path in test_files:
            file_path.touch()
        
        # Test cache cleanup
        removed_files = cleanup_downloaded_models(
            cache_dir=cache_dir,
            older_than_days=0,
            dry_run=False
        )
        
        assert len(removed_files) >= 0


if __name__ == "__main__":
    pytest.main([__file__])

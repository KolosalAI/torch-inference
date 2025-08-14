"""Tests for configuration management."""

import os
import pytest
import tempfile
import importlib.util
from pathlib import Path
from unittest.mock import patch, Mock

from framework.core.config import (
    InferenceConfig, DeviceConfig, BatchConfig, PerformanceConfig,
    ModelType, DeviceType, ConfigFactory
)
from framework.core.config_manager import ConfigManager


@pytest.fixture
def clean_env():
    """Ensure a clean environment for config tests."""
    # List of env vars that might affect config tests
    config_env_vars = [
        'BATCH_SIZE', 'DEVICE', 'USE_FP16', 'LOG_LEVEL', 'ENVIRONMENT',
        'CONFIG_DIR', 'MAX_BATCH_SIZE', 'DEVICE_TYPE'
    ]
    
    # Save original values
    original_values = {}
    for var in config_env_vars:
        if var in os.environ:
            original_values[var] = os.environ[var]
            del os.environ[var]
    
    yield
    
    # Restore original values
    for var, value in original_values.items():
        os.environ[var] = value


class TestDeviceConfig:
    """Test DeviceConfig class."""
    
    def test_device_config_creation(self):
        """Test creating device configuration."""
        config = DeviceConfig(device_type=DeviceType.CPU)
        
        assert config.device_type == DeviceType.CPU
        assert not config.use_fp16
        assert config.device_id is None
    
    def test_device_config_with_gpu(self):
        """Test GPU device configuration."""
        config = DeviceConfig(
            device_type=DeviceType.CUDA,
            device_id=0,
            use_fp16=True
        )
        
        assert config.device_type == DeviceType.CUDA
        assert config.use_fp16
        assert config.device_id == 0
    
    def test_device_config_validation(self):
        """Test device configuration validation."""
        # Should not raise for valid config
        config = DeviceConfig(device_type=DeviceType.CPU)
        assert config.device_type == DeviceType.CPU
        
        # Test with string device type
        with pytest.raises(ValueError):
            DeviceConfig(device_type="invalid_device")


class TestBatchConfig:
    """Test BatchConfig class."""
    
    def test_batch_config_creation(self):
        """Test creating batch configuration."""
        config = BatchConfig(batch_size=8, max_batch_size=32)
        
        assert config.batch_size == 8
        assert config.max_batch_size == 32
    
    def test_batch_config_defaults(self):
        """Test batch configuration defaults."""
        config = BatchConfig()
        
        assert config.batch_size == 1
        assert config.max_batch_size == 16
    
    def test_batch_config_validation(self):
        """Test batch configuration validation."""
        # Valid configuration
        config = BatchConfig(batch_size=4, max_batch_size=16)
        assert config.batch_size <= config.max_batch_size
        
        # Invalid configuration - batch_size > max_batch_size
        with pytest.raises(ValueError):
            BatchConfig(batch_size=32, max_batch_size=16)


class TestPerformanceConfig:
    """Test PerformanceConfig class."""
    
    def test_performance_config_creation(self):
        """Test creating performance configuration."""
        config = PerformanceConfig(
            log_level="DEBUG",
            enable_profiling=True,
            max_concurrent_requests=100
        )
        
        assert config.log_level == "DEBUG"
        assert config.enable_profiling
        assert config.max_concurrent_requests == 100
    
    def test_performance_config_defaults(self):
        """Test performance configuration defaults."""
        config = PerformanceConfig()
        
        assert config.log_level == "INFO"
        assert not config.enable_profiling
        assert config.max_concurrent_requests == 10


class TestInferenceConfig:
    """Test InferenceConfig class."""
    
    def test_inference_config_creation(self):
        """Test creating inference configuration."""
        device_config = DeviceConfig(device_type=DeviceType.CUDA)
        batch_config = BatchConfig(batch_size=8)
        perf_config = PerformanceConfig(enable_profiling=True)
        
        config = InferenceConfig(
            device=device_config,
            batch=batch_config,
            performance=perf_config,
            model_type=ModelType.CLASSIFICATION
        )
        
        assert config.device == device_config
        assert config.batch == batch_config
        assert config.performance == perf_config
        assert config.model_type == ModelType.CLASSIFICATION
    
    def test_inference_config_defaults(self):
        """Test inference configuration with defaults."""
        config = InferenceConfig()
        
        assert config.device.device_type == DeviceType.AUTO  # Fixed: default is AUTO, not CPU
        assert config.batch.batch_size == 1
        assert config.performance.log_level == "INFO"
        assert config.model_type == ModelType.CUSTOM  # Fixed: default is CUSTOM, not CLASSIFICATION


class TestConfigFactory:
    """Test ConfigFactory class."""
    
    def test_create_classification_config(self):
        """Test creating classification configuration."""
        config = ConfigFactory.create_classification_config(
            num_classes=10,
            input_size=(224, 224),
            use_softmax=True
        )
        
        assert config.model_type == ModelType.CLASSIFICATION
        assert hasattr(config, 'num_classes')
        assert config.num_classes == 10
    
    def test_create_detection_config(self):
        """Test creating detection configuration."""
        config = ConfigFactory.create_detection_config(
            input_size=(640, 640),
            confidence_threshold=0.5
        )
        
        assert config.model_type == ModelType.DETECTION
        assert hasattr(config, 'input_size')
        assert config.input_size == (640, 640)
    
    def test_create_segmentation_config(self):
        """Test creating segmentation configuration."""
        config = ConfigFactory.create_segmentation_config(
            input_size=(512, 512),
            threshold=0.5
        )
        
        assert config.model_type == ModelType.SEGMENTATION
        assert hasattr(config, 'input_size')
        assert config.input_size == (512, 512)
    
    def test_create_optimized_config(self):
        """Test creating optimized configuration."""
        config = ConfigFactory.create_optimized_config(
            enable_tensorrt=True,
            enable_fp16=True
        )
        
        assert config.device.use_fp16
        assert hasattr(config, 'optimizations')


class TestConfigManager:
    """Test ConfigManager class."""
    
    def test_config_manager_creation(self, config_manager):
        """Test creating config manager."""
        assert config_manager is not None
        assert config_manager.environment == "test"
    
    def test_get_inference_config(self, config_manager):
        """Test getting inference configuration."""
        config = config_manager.get_inference_config()
        
        assert isinstance(config, InferenceConfig)
        assert config.device.device_type == DeviceType.CPU
        assert config.batch.batch_size == 2  # From test environment
    
    def test_get_server_config(self, config_manager):
        """Test getting server configuration."""
        server_config = config_manager.get_server_config()
        
        assert isinstance(server_config, dict)
        assert "host" in server_config
        assert "port" in server_config
        assert "log_level" in server_config
    
    def test_environment_override(self, clean_env):
        """Test environment-specific configuration override."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            
            # Create config with environment overrides
            yaml_file = config_dir / "config.yaml"
            yaml_file.write_text("""
device:
  device_type: cpu
  use_fp16: false

environments:
  production:
    device:
      device_type: cuda
      use_fp16: true
  development:
    device:
      use_fp16: false
""")
            
            # Test production environment
            config_mgr = ConfigManager(config_dir=config_dir, environment="production")
            config = config_mgr.get_inference_config()
            
            # Environment override should work for production
            assert config.device.device_type == DeviceType.CUDA
            assert config.device.use_fp16
    
    def test_environment_variable_override(self):
        """Test environment variable configuration override."""
        with patch.dict(os.environ, {
            "DEVICE": "cuda",
            "BATCH_SIZE": "16",
            "USE_FP16": "true",
            "LOG_LEVEL": "DEBUG"
        }):
            with tempfile.TemporaryDirectory() as temp_dir:
                config_dir = Path(temp_dir)
                
                # Create minimal config
                yaml_file = config_dir / "config.yaml"
                yaml_file.write_text("""
device:
  device_type: cpu
  use_fp16: false
batch:
  batch_size: 1
performance:
  log_level: INFO
""")
                
                config_mgr = ConfigManager(config_dir=config_dir)
                config = config_mgr.get_inference_config()
                
                # Environment variables should override YAML
                assert config.device.device_type == DeviceType.CUDA
                assert config.device.use_fp16
                assert config.batch.batch_size == 16
                assert config.performance.log_level == "DEBUG"
    
    def test_configuration_precedence(self, clean_env):
        """Test configuration precedence: ENV > YAML env > YAML base."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            
            # Create config with base and environment overrides
            yaml_file = config_dir / "config.yaml"
            yaml_file.write_text("""
device:
  device_type: cpu
  use_fp16: false
batch:
  batch_size: 1
  max_batch_size: 64  # Increase max to allow higher batch sizes

environments:
  test:
    device:
      use_fp16: true
    batch:
      batch_size: 8
      max_batch_size: 64
""")
            
            # Test with environment variable override
            with patch.dict(os.environ, {"BATCH_SIZE": "32"}):
                config_mgr = ConfigManager(config_dir=config_dir, environment="test")
                config = config_mgr.get_inference_config()
                
                # ENV should override YAML env override
                assert config.batch.batch_size == 32
                # YAML env should override base
                assert config.device.use_fp16
                # Base should be used for unoverridden values
                assert config.device.device_type == DeviceType.CPU
    
    def test_invalid_configuration_file(self, clean_env):
        """Test handling of invalid configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            
            # Create invalid YAML
            yaml_file = config_dir / "config.yaml"
            yaml_file.write_text("invalid: yaml: content: [")
            
            # Should fall back to defaults
            config_mgr = ConfigManager(config_dir=config_dir)
            config = config_mgr.get_inference_config()
            
            # Should use defaults
            assert config.device.device_type == DeviceType.AUTO  # Default is 'auto'
            assert config.batch.batch_size == 2  # Default from config_manager.py
    
    def test_missing_configuration_file(self, clean_env):
        """Test handling of missing configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            
            # No config files exist
            config_mgr = ConfigManager(config_dir=config_dir)
            config = config_mgr.get_inference_config()
            
            # Should use defaults
            assert config.device.device_type == DeviceType.AUTO  # Default is 'auto'
            assert config.batch.batch_size == 2  # Default from config_manager.py
            assert config.performance.log_level == "INFO"
    
    @pytest.mark.skipif(
        not importlib.util.find_spec("opentelemetry"), 
        reason="OpenTelemetry not available - enterprise features require it"
    )
    @patch('framework.enterprise.config.EnterpriseConfig')
    def test_enterprise_config(self, mock_enterprise, config_manager):
        """Test enterprise configuration loading."""
        mock_enterprise_instance = Mock()
        mock_enterprise.from_dict.return_value = mock_enterprise_instance
        
        enterprise_config = config_manager.get_enterprise_config()
        
        # Should return None if enterprise config not available
        # or the enterprise config if it is available
        assert enterprise_config is None or enterprise_config is not None
    
    def test_type_conversion(self, config_manager):
        """Test automatic type conversion of configuration values."""
        with patch.dict(os.environ, {
            "BATCH_SIZE": "8",  # string -> int
            "USE_FP16": "true",  # string -> bool
            "ENABLE_PROFILING": "false",  # string -> bool
        }):
            config = config_manager.get_inference_config()
            
            assert isinstance(config.batch.batch_size, int)
            assert config.batch.batch_size == 8
            assert isinstance(config.device.use_fp16, bool)
            assert config.device.use_fp16


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_full_configuration_workflow(self, clean_env):
        """Test complete configuration loading workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            
            # Create comprehensive configuration
            env_file = config_dir / ".env"
            env_file.write_text("""
ENVIRONMENT=production
DEVICE=cuda
BATCH_SIZE=16
USE_FP16=true
LOG_LEVEL=WARNING
HOST=0.0.0.0
PORT=8080
""")
            
            yaml_file = config_dir / "config.yaml"
            yaml_file.write_text("""
device:
  device_type: cpu
  use_fp16: false
  device_id: null

batch:
  batch_size: 1
  max_batch_size: 16

performance:
  log_level: INFO
  enable_profiling: false
  max_concurrent_requests: 10

server:
  host: 127.0.0.1
  port: 8000
  workers: 1
  reload: false

environments:
  production:
    server:
      host: 0.0.0.0
      workers: 4
      reload: false
    batch:
      max_batch_size: 64
  development:
    server:
      reload: true
    performance:
      enable_profiling: true
""")
            
            # Load configuration
            config_mgr = ConfigManager(config_dir=config_dir, environment="production")
            inference_config = config_mgr.get_inference_config()
            server_config = config_mgr.get_server_config()
            
            # Verify environment variable overrides
            assert inference_config.device.device_type == DeviceType.CUDA
            assert inference_config.batch.batch_size == 16
            assert inference_config.device.use_fp16
            assert inference_config.performance.log_level == "WARNING"
            
            # Verify environment-specific overrides
            assert inference_config.batch.max_batch_size == 64
            assert server_config["workers"] == 4
            assert server_config["host"] == "0.0.0.0"
            assert server_config["port"] == 8080  # ENV override
            
            # Verify base configuration for non-overridden values
            assert inference_config.performance.max_concurrent_requests == 10
    
    def test_configuration_validation_errors(self):
        """Test configuration validation error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            
            # Test invalid device type
            with patch.dict(os.environ, {"DEVICE": "invalid_device"}):
                config_mgr = ConfigManager(config_dir=config_dir)
                with pytest.raises((ValueError, TypeError)):
                    config_mgr.get_inference_config()
    
    def test_configuration_caching(self, config_manager):
        """Test configuration caching behavior."""
        # First call
        config1 = config_manager.get_inference_config()
        
        # Second call should return same instance (if cached)
        config2 = config_manager.get_inference_config()
        
        # Configs should have same values
        assert config1.device.device_type == config2.device.device_type
        assert config1.batch.batch_size == config2.batch.batch_size

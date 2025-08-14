"""Test configuration and fixtures."""

import os
import tempfile
import torch
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, MagicMock
import warnings

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from framework.core.config import InferenceConfig, DeviceConfig, BatchConfig, PerformanceConfig, DeviceType
from framework.core.config_manager import ConfigManager

# Test model loader
try:
    from tests.models.model_loader import TestModelLoader, get_test_model_loader
    REAL_MODELS_AVAILABLE = True
except ImportError:
    REAL_MODELS_AVAILABLE = False
    warnings.warn("Real models not available. Run 'python tests/models/create_test_models.py' to download them.")


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return InferenceConfig(
        device=DeviceConfig(device_type=DeviceType.CPU, use_fp16=False),
        batch=BatchConfig(batch_size=1, max_batch_size=4),
        performance=PerformanceConfig(log_level="INFO")
    )


@pytest.fixture
def test_model_loader():
    """Get the test model loader."""
    if REAL_MODELS_AVAILABLE:
        return get_test_model_loader()
    else:
        pytest.skip("Real models not available")


@pytest.fixture
def real_classification_model(test_model_loader):
    """Load a real classification model for testing."""
    try:
        model, info = test_model_loader.load_classification_model()
        return model, info
    except Exception as e:
        pytest.skip(f"Could not load classification model: {e}")


@pytest.fixture
def real_lightweight_model(test_model_loader):
    """Load the smallest available real model."""
    try:
        model, info = test_model_loader.load_lightweight_model()
        return model, info
    except Exception as e:
        pytest.skip(f"Could not load lightweight model: {e}")


@pytest.fixture
def real_image_model(test_model_loader):
    """Load a real image classification model."""
    try:
        model, info = test_model_loader.load_model_for_task("image-classification")
        return model, info
    except Exception as e:
        # Try to load any available model as fallback
        try:
            model, info = test_model_loader.load_lightweight_model()
            return model, info
        except Exception:
            pytest.skip(f"Could not load image model: {e}")


@pytest.fixture
def sample_input_for_model(test_model_loader):
    """Create sample input for a given model."""
    def _create_input(model_id: str, batch_size: int = 1):
        return test_model_loader.create_sample_input(model_id, batch_size)
    return _create_input


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for test models."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_torch_model():
    """Create a mock PyTorch model."""
    model = Mock()
    model.eval.return_value = model
    model.cuda.return_value = model
    model.cpu.return_value = model
    model.to.return_value = model
    model.parameters.return_value = [torch.randn(10, 10)]
    model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
    
    # Mock forward pass
    def mock_forward(x):
        if isinstance(x, torch.Tensor):
            return torch.randn(x.shape[0], 10)  # Mock classification output
        return torch.randn(1, 10)
    
    model.side_effect = mock_forward
    model.__call__ = mock_forward
    return model


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_image_path(temp_model_dir):
    """Create a sample image file for testing."""
    from PIL import Image
    
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='red')
    image_path = temp_model_dir / "test_image.jpg"
    img.save(image_path)
    return image_path


@pytest.fixture
def mock_model_file(temp_model_dir):
    """Create a mock model file."""
    model_path = temp_model_dir / "test_model.pt"
    # Create a simple mock model state dict
    torch.save({"layer.weight": torch.randn(10, 10)}, model_path)
    return model_path


@pytest.fixture
def config_manager():
    """Create a test config manager."""
    # Create temporary config files
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)
        
        # Create test .env file
        env_file = config_dir / ".env"
        env_file.write_text("""
ENVIRONMENT=test
DEVICE=cpu
BATCH_SIZE=2
LOG_LEVEL=DEBUG
""")
        
        # Create test config.yaml
        yaml_file = config_dir / "config.yaml"
        yaml_file.write_text("""
device:
  device_type: cpu
  use_fp16: false

batch:
  batch_size: 1
  max_batch_size: 4

performance:
  log_level: INFO
  enable_profiling: false

environments:
  test:
    device:
      device_type: cpu
    batch:
      batch_size: 2
    performance:
      log_level: DEBUG
""")
        
        # Save original environment variables (including those from .env file)
        original_env = {}
        env_vars_from_fixture = {"CONFIG_DIR": str(config_dir), "ENVIRONMENT": "test", "DEVICE": "cpu", "BATCH_SIZE": "2", "LOG_LEVEL": "DEBUG"}
        
        for key, value in env_vars_from_fixture.items():
            if key in os.environ:
                original_env[key] = os.environ[key]
            os.environ[key] = value
        
        try:
            yield ConfigManager(env_file=env_file, config_file=yaml_file, environment="test")
        finally:
            # Clean up ALL environment variables that could have been set by this fixture
            for key in env_vars_from_fixture.keys():
                if key in original_env:
                    os.environ[key] = original_env[key]
                elif key in os.environ:
                    del os.environ[key]


@pytest.fixture
def mock_cuda_available():
    """Mock CUDA availability."""
    original = torch.cuda.is_available
    torch.cuda.is_available = Mock(return_value=True)
    torch.cuda.device_count = Mock(return_value=1)
    torch.cuda.get_device_name = Mock(return_value="Mock GPU")
    yield
    torch.cuda.is_available = original


@pytest.fixture
def mock_no_cuda():
    """Mock CUDA not available."""
    original = torch.cuda.is_available
    torch.cuda.is_available = Mock(return_value=False)
    torch.cuda.device_count = Mock(return_value=0)
    yield
    torch.cuda.is_available = original


@pytest.fixture
def sample_batch_data():
    """Create sample batch data for testing."""
    return [
        torch.randn(1, 10),
        torch.randn(1, 10),
        torch.randn(1, 10),
    ]


@pytest.fixture
def mock_performance_metrics():
    """Mock performance metrics."""
    return {
        "inference_time": 0.015,
        "preprocessing_time": 0.002,
        "postprocessing_time": 0.001,
        "total_time": 0.018,
        "throughput": 55.56,
        "memory_usage": 1024 * 1024 * 100,  # 100MB
        "gpu_utilization": 85.5
    }


@pytest.fixture
def enterprise_config():
    """Create a test enterprise configuration."""
    from framework.enterprise.config import EnterpriseConfig, AuthConfig, SecurityConfig
    from framework.enterprise.config import AuthProvider, EncryptionAlgorithm
    
    return EnterpriseConfig(
        environment="test",
        auth=AuthConfig(
            provider=AuthProvider.OAUTH2,
            jwt_secret_key="test-secret-key"
        ),
        security=SecurityConfig(
            enable_encryption_at_rest=True,
            encryption_algorithm=EncryptionAlgorithm.AES256,
            enable_rate_limiting=True,
            max_requests_per_minute=100
        )
    )


class MockOptimizer:
    """Mock optimizer for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.optimized = False
    
    def optimize(self, model):
        """Mock optimization."""
        self.optimized = True
        return model
    
    def is_available(self) -> bool:
        """Mock availability check."""
        return True


@pytest.fixture
def mock_tensorrt_optimizer():
    """Mock TensorRT optimizer."""
    return MockOptimizer("TensorRT")


@pytest.fixture
def mock_onnx_optimizer():
    """Mock ONNX optimizer."""
    return MockOptimizer("ONNX")


# Test data generators

def create_classification_data(batch_size: int = 1, num_classes: int = 10):
    """Create classification test data."""
    inputs = torch.randn(batch_size, 3, 224, 224)
    outputs = torch.randint(0, num_classes, (batch_size,))
    return inputs, outputs


def create_detection_data(batch_size: int = 1, num_boxes: int = 5):
    """Create object detection test data."""
    inputs = torch.randn(batch_size, 3, 640, 640)
    # Mock detection output: [batch, num_boxes, 6] (x1, y1, x2, y2, conf, class)
    outputs = torch.rand(batch_size, num_boxes, 6)
    return inputs, outputs


def create_segmentation_data(batch_size: int = 1, num_classes: int = 21):
    """Create segmentation test data."""
    inputs = torch.randn(batch_size, 3, 512, 512)
    outputs = torch.randint(0, num_classes, (batch_size, 512, 512))
    return inputs, outputs


# Test utilities

def assert_tensor_equal(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol: float = 1e-5):
    """Assert two tensors are equal within tolerance."""
    assert torch.allclose(tensor1, tensor2, rtol=rtol), \
        f"Tensors not equal: {tensor1} vs {tensor2}"


def assert_model_output_valid(output: torch.Tensor, expected_shape: tuple):
    """Assert model output has valid shape and values."""
    assert output.shape == expected_shape, \
        f"Output shape {output.shape} != expected {expected_shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert torch.isfinite(output).all(), "Output contains infinite values"


def create_mock_model_with_output(output_shape: tuple):
    """Create a mock model that returns specific output shape."""
    def mock_forward(x):
        batch_size = x.shape[0] if len(x.shape) > 0 else 1
        return torch.randn(batch_size, *output_shape[1:])
    
    model = Mock()
    model.side_effect = mock_forward
    model.__call__ = mock_forward
    model.eval.return_value = model
    model.to.return_value = model
    model.cuda.return_value = model
    model.cpu.return_value = model
    
    return model


# Environment setup

def setup_test_environment():
    """Setup test environment variables."""
    test_env = {
        "ENVIRONMENT": "test",
        "DEVICE": "cpu",
        "BATCH_SIZE": "1",
        "LOG_LEVEL": "DEBUG",
        "ENABLE_PROFILING": "false",
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    return test_env


def cleanup_test_environment(env_vars: Dict[str, str]):
    """Clean up test environment variables."""
    for key in env_vars.keys():
        if key in os.environ:
            del os.environ[key]


# Additional missing fixtures

@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )
    model.eval()
    return model


@pytest.fixture
def framework(test_config):
    """Create a framework instance for testing."""
    from framework import TorchInferenceFramework
    return TorchInferenceFramework(test_config)


@pytest.fixture
def mock_model(test_config):
    """Create a mock model for testing."""
    from tests.unit.test_inference_engine import MockInferenceModel
    return MockInferenceModel(test_config)


@pytest.fixture
def inference_config():
    """Create inference config for testing."""
    from framework.core.config import InferenceConfig, BatchConfig, DeviceConfig, DeviceType
    return InferenceConfig(
        device=DeviceConfig(device_type=DeviceType.CPU),
        batch=BatchConfig(batch_size=2, max_batch_size=8)
    )

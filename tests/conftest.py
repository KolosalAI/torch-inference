"""Test configuration and fixtures."""

import os
import tempfile
import torch
import pytest
import numpy as np
import asyncio
import io
import json
import time
import gc
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, MagicMock, AsyncMock
import warnings
from PIL import Image, ImageDraw, ImageFont

from tests.utils.disk_management import managed_temp_dir, DiskSpaceManager

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

# Secure image components availability check
try:
    from framework.processors.image.secure_image_processor import (
        SecurityLevel, SecurityConfig, SecureImageValidator, 
        SecureImageSanitizer, SecureImagePreprocessor
    )
    from framework.models.secure_image_model import SecureImageModel
    SECURE_IMAGE_COMPONENTS_AVAILABLE = True
except ImportError:
    SECURE_IMAGE_COMPONENTS_AVAILABLE = False


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
    """Create a temporary directory for test models with disk space management."""
    disk_manager = DiskSpaceManager()
    
    # Check if there's enough space for model files (need at least 500MB)
    if not disk_manager.has_enough_space(required_mb=500):
        pytest.skip("Insufficient disk space for test models")
    
    try:
        with managed_temp_dir(required_space_mb=500) as temp_dir:
            yield temp_dir
    except OSError as e:
        if "No space left on device" in str(e):
            pytest.skip(f"Insufficient disk space for test models: {e}")
        else:
            raise


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
def mock_model_with_config(test_config):
    """Create a mock model with proper configuration for autoscaling tests."""
    model = Mock()
    
    # Set the config attribute that autoscaling expects
    model.config = test_config
    
    # Standard model methods
    model.eval.return_value = model
    model.cuda.return_value = model  
    model.cpu.return_value = model
    model.to.return_value = model
    model.parameters.return_value = [torch.randn(10, 10)]
    model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
    
    # Mock predict method
    async def mock_predict(inputs):
        await asyncio.sleep(0.001)  # Simulate processing
        return {
            "predictions": [0.1, 0.2, 0.7],
            "confidence": 0.8,
            "model_name": "mock_model"
        }
    
    model.predict = mock_predict
    
    # Mock other required methods
    model.warmup = AsyncMock()
    model.is_loaded = True
    model.model_name = "mock_model"
    
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
    from framework.security.config import EnterpriseConfig, AuthConfig, SecurityConfig
    from framework.security.config import AuthProvider, EncryptionAlgorithm
    
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


@pytest.fixture
def model_manager():
    """Create a model manager for testing."""
    from framework.core.base_model import ModelManager
    return ModelManager()


@pytest.fixture
def async_client():
    """Create async test client."""
    import httpx
    from main import app
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def mock_model_manager_with_config(test_config):
    """Create a mock model manager with proper configuration for autoscaling tests."""
    manager = Mock()
    manager._loaded_models = {}
    
    async def mock_load_model(model_id):
        # Create mock model with proper config
        mock_model = Mock()
        mock_model.config = test_config
        
        # Mock predict method
        async def mock_predict(inputs):
            await asyncio.sleep(0.001)  # Simulate processing
            return {
                "predictions": [0.1, 0.2, 0.7],
                "confidence": 0.8,
                "model_name": model_id
            }
        
        mock_model.predict = mock_predict
        mock_model.warmup = AsyncMock()
        mock_model.is_loaded = True
        mock_model.model_name = model_id
        
        manager._loaded_models[model_id] = mock_model
        return mock_model
    
    async def mock_unload_model(model_id):
        if model_id in manager._loaded_models:
            del manager._loaded_models[model_id]
    
    def mock_get_model(model_id):
        return manager._loaded_models.get(model_id)
    
    def mock_is_model_loaded(model_id):
        return model_id in manager._loaded_models
    
    def mock_get_loaded_models():
        return list(manager._loaded_models.keys())
    
    manager.load_model.side_effect = mock_load_model
    manager.unload_model.side_effect = mock_unload_model
    manager.get_model.side_effect = mock_get_model
    manager.is_model_loaded.side_effect = mock_is_model_loaded
    manager.get_loaded_models.side_effect = mock_get_loaded_models
    
    return manager


@pytest.fixture  
def autoscaler_config():
    """Create a test configuration for autoscaling."""
    from framework.autoscaling.autoscaler import AutoscalerConfig
    from framework.autoscaling.zero_scaler import ZeroScalingConfig
    from framework.autoscaling.model_loader import ModelLoaderConfig
    from framework.autoscaling.metrics import MetricsConfig
    
    return AutoscalerConfig(
        enable_zero_scaling=True,
        enable_dynamic_loading=True,
        enable_monitoring=True,
        monitoring_interval=0.1,
        scaling_cooldown=0.1,
        max_concurrent_scalings=5,
        enable_predictive_scaling=False,
        zero_scaling=ZeroScalingConfig(
            enabled=True,
            scale_to_zero_delay=1.0,
            max_loaded_models=5,
            preload_popular_models=False,
            popularity_threshold=3
        ),
        model_loader=ModelLoaderConfig(
            enabled=True,
            max_instances_per_model=3,
            min_instances_per_model=1,
            health_check_interval=0.5
        ),
        metrics=MetricsConfig(
            enabled=True,
            collection_interval=0.1,
            retention_period=300.0
        )
    )


@pytest.fixture
def client():
    """Create test client."""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


@pytest.fixture
def test_autoscaler(autoscaler_config, mock_model_manager_with_config):
    """Create a test autoscaler instance."""
    from framework.autoscaling.autoscaler import Autoscaler
    
    return Autoscaler(
        config=autoscaler_config,
        model_manager=mock_model_manager_with_config
    )


# Enhanced Optimizer Test Fixtures

@pytest.fixture
def enhanced_config():
    """Create configuration with enhanced optimizations enabled."""
    return InferenceConfig(
        device=DeviceConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_vulkan=True,
            use_numba=True,
            jit_strategy="enhanced",
            numba_target="auto"
        ),
        performance=PerformanceConfig(
            enable_optimizations=True,
            auto_optimize=True,
            optimization_level="balanced"
        )
    )


@pytest.fixture
def mock_vulkan_optimizer():
    """Mock Vulkan optimizer for testing."""
    optimizer = Mock()
    optimizer.is_available.return_value = True
    optimizer.optimize.return_value = None  # Vulkan doesn't modify model directly
    optimizer.device_manager = Mock()
    optimizer.compute_contexts = {}
    return optimizer


@pytest.fixture
def mock_numba_optimizer():
    """Mock Numba optimizer for testing."""
    optimizer = Mock()
    optimizer.is_available.return_value = True
    optimizer.optimize.return_value = None  # Numba wraps operations
    optimizer.target = "auto"
    optimizer.fastmath = True
    return optimizer


@pytest.fixture
def mock_enhanced_jit_optimizer():
    """Mock Enhanced JIT optimizer for testing."""
    optimizer = Mock()
    optimizer.is_available.return_value = True
    optimizer.strategy = "auto"
    optimizer.available_backends = {
        'torchscript': True,
        'vulkan': False,
        'numba': True
    }
    
    def mock_optimize(model, *args, **kwargs):
        # Return optimized model (could be same or different)
        return model
    
    optimizer.optimize.side_effect = mock_optimize
    return optimizer


@pytest.fixture
def mock_performance_optimizer():
    """Mock Performance optimizer for testing."""
    optimizer = Mock()
    optimizer.is_available.return_value = True
    
    def mock_optimize(model, *args, **kwargs):
        return model
    
    optimizer.optimize.side_effect = mock_optimize
    return optimizer


@pytest.fixture
def optimization_test_model():
    """Create a model suitable for optimization testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    )


@pytest.fixture
def optimization_test_input():
    """Create input tensor for optimization testing."""
    return torch.randn(32, 128)


@pytest.fixture
def mock_optimization_utilities():
    """Mock optimization utility functions."""
    utilities = Mock()
    
    utilities.get_available_optimizers.return_value = {
        'performance': {'available': True, 'class': 'PerformanceOptimizer'},
        'enhanced_jit': {'available': True, 'class': 'EnhancedJITOptimizer'},
        'vulkan': {'available': False, 'class': 'VulkanOptimizer'},
        'numba': {'available': True, 'class': 'NumbaOptimizer'},
        'jit': {'available': True, 'class': 'JITOptimizer'}
    }
    
    utilities.get_optimization_recommendations.return_value = [
        ('enhanced_jit', 'Enhanced JIT compilation with multi-backend support'),
        ('numba', 'JIT compilation for numerical operations'),
        ('performance', 'Comprehensive performance optimization')
    ]
    
    utilities.create_optimizer_pipeline.return_value = []
    
    return utilities


@pytest.fixture
def framework_with_enhanced_model(enhanced_config, optimization_test_model, temp_model_dir):
    """Create framework with loaded model for enhanced optimization testing."""
    from framework import TorchInferenceFramework
    
    # Save test model
    model_path = temp_model_dir / "enhanced_test_model.pt"
    torch.save(optimization_test_model, model_path)
    
    framework = TorchInferenceFramework(enhanced_config)
    
    # Mock model loading
    mock_model = Mock()
    mock_model.model = optimization_test_model
    mock_model.example_inputs = torch.randn(1, 128)
    mock_model.is_loaded = True
    mock_model.model_info = {"type": "test", "parameters": 100000}
    mock_model.predict.return_value = torch.randn(1, 10)
    mock_model.device = "cpu"
    
    # Mock the loading process
    from unittest.mock import patch
    with patch('framework.load_model') as mock_load:
        mock_load.return_value = mock_model
        framework.load_model(model_path)
    
    return framework


@pytest.fixture
def vulkan_availability():
    """Mock Vulkan availability detection."""
    return {
        'available': False,  # Default to False for testing
        'devices': [],
        'version': None
    }


@pytest.fixture
def numba_availability():
    """Mock Numba availability detection."""
    return {
        'available': True,   # Default to True for testing
        'cuda_available': False,
        'version': '0.60.0'
    }


@pytest.fixture
def mock_device_detection():
    """Mock device detection for optimization testing."""
    detection = Mock()
    
    detection.detect_vulkan_devices.return_value = []
    detection.detect_cuda_devices.return_value = []
    detection.get_cpu_info.return_value = {
        'name': 'Mock CPU',
        'cores': 8,
        'threads': 16
    }
    detection.get_memory_info.return_value = {
        'total': 16 * 1024 * 1024 * 1024,  # 16GB
        'available': 8 * 1024 * 1024 * 1024   # 8GB
    }
    
    return detection


@pytest.fixture
def optimization_benchmark_results():
    """Mock optimization benchmark results."""
    return {
        'torchscript_script': 0.015,
        'torchscript_trace': 0.012,
        'enhanced_jit': 0.010,
        'vulkan': 0.008,
        'numba': 0.009,
        'baseline': 0.020
    }


@pytest.fixture(params=["conservative", "balanced", "aggressive"])
def optimization_level(request):
    """Parametrized optimization levels for testing."""
    return request.param


@pytest.fixture(params=["cpu", "cuda"])
def test_device(request):
    """Parametrized devices for testing."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param


@pytest.fixture(params=["small", "medium", "large"])
def model_size_category(request):
    """Parametrized model sizes for testing."""
    return request.param


@pytest.fixture(params=["inference", "training", "serving"])
def optimization_target(request):
    """Parametrized optimization targets for testing."""
    return request.param


# Performance testing fixtures

@pytest.fixture
def performance_test_config():
    """Configuration for performance testing."""
    return {
        'warmup_iterations': 5,
        'benchmark_iterations': 20,
        'timeout_seconds': 30,
        'memory_limit_mb': 2048
    }


@pytest.fixture
def benchmark_model():
    """Create a model suitable for benchmarking."""
    return torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10)
    )


@pytest.fixture
def benchmark_input():
    """Create input for benchmarking."""
    return torch.randn(100, 1024)


# Marker for performance tests
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "performance: mark test as performance test (run with --performance flag)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security-focused test"
    )


# ============================================================================
# Secure Image Processing Test Fixtures and Utilities
# ============================================================================

class TestImageGenerator:
    """Utility class for generating test images with specific characteristics."""
    
    @staticmethod
    def create_normal_image(width=224, height=224, format='PNG'):
        """Create a normal, natural-looking image."""
        # Create gradient background
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                # Create a natural-looking pattern
                r = int(128 + 64 * np.sin(x / 20) * np.cos(y / 20))
                g = int(128 + 32 * np.sin(x / 15 + 1) * np.cos(y / 15 + 1))
                b = int(128 + 16 * np.sin(x / 25 + 2) * np.cos(y / 25 + 2))
                
                img_array[y, x] = [
                    max(0, min(255, r)),
                    max(0, min(255, g)),
                    max(0, min(255, b))
                ]
        
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)
        return buffer
    
    @staticmethod
    def create_high_entropy_image(width=224, height=224, format='PNG'):
        """Create an image with high entropy (random noise)."""
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)
        return buffer
    
    @staticmethod
    def create_low_entropy_image(width=224, height=224, format='PNG'):
        """Create an image with low entropy (uniform patterns)."""
        img_array = np.full((height, width, 3), [128, 128, 128], dtype=np.uint8)
        
        # Add some minimal variation
        for i in range(0, height, 50):
            for j in range(0, width, 50):
                img_array[i:i+25, j:j+25] = [100, 100, 100]
        
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)
        return buffer
    
    @staticmethod
    def create_adversarial_pattern_image(width=224, height=224, format='PNG'):
        """Create an image with adversarial-like patterns."""
        # Start with normal image
        base = np.random.randint(100, 156, (height, width, 3), dtype=np.uint8)
        
        # Add high-frequency adversarial-like noise
        noise_strength = 30
        for y in range(height):
            for x in range(width):
                # High-frequency pattern
                noise = noise_strength * np.sin(x * 0.8) * np.cos(y * 0.8)
                base[y, x] = np.clip(base[y, x] + noise, 0, 255)
        
        img = Image.fromarray(base.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)
        return buffer
    
    @staticmethod
    def create_steganography_pattern_image(width=224, height=224, format='PNG'):
        """Create an image with steganography-like LSB patterns."""
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Modify LSBs in a pattern
        for y in range(0, height, 2):
            for x in range(0, width, 2):
                # Set LSB to encode hidden data pattern
                img_array[y, x, :] = (img_array[y, x, :] & 0xFE) | ((x + y) % 2)
        
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)
        return buffer
    
    @staticmethod
    def create_text_image(text="TEST", width=224, height=224, format='PNG'):
        """Create an image with text content."""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        # Calculate text position to center it
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), text, fill='black', font=font)
        
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)
        return buffer


class TestDataManager:
    """Manager for test data and temporary files."""
    
    def __init__(self):
        self.temp_dir = None
        self.temp_files = []
    
    def setup(self):
        """Set up temporary directory for test data."""
        self.temp_dir = tempfile.mkdtemp(prefix="secure_image_test_")
        return self.temp_dir
    
    def cleanup(self):
        """Clean up temporary files and directory."""
        import shutil
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        self.temp_files.clear()
    
    def save_test_image(self, image_buffer, filename):
        """Save test image to temporary directory."""
        if not self.temp_dir:
            self.setup()
        
        filepath = Path(self.temp_dir) / filename
        with open(filepath, 'wb') as f:
            f.write(image_buffer.getvalue())
        
        self.temp_files.append(filepath)
        return filepath


class PerformanceTracker:
    """Helper class to track performance metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.start_time = None
        self.end_time = None
        self.memory_start = None
        self.memory_peak = None
        self.memory_end = None
    
    def start(self):
        """Start performance tracking."""
        gc.collect()  # Clean up before measurement
        self.start_time = time.time()
        try:
            import psutil
            self.memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            self.memory_peak = self.memory_start
        except ImportError:
            self.memory_start = None
            self.memory_peak = None
    
    def update_peak_memory(self):
        """Update peak memory usage."""
        try:
            import psutil
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            if current_memory > self.memory_peak:
                self.memory_peak = current_memory
        except ImportError:
            pass
    
    def stop(self):
        """Stop performance tracking."""
        self.end_time = time.time()
        try:
            import psutil
            self.memory_end = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            self.update_peak_memory()
        except ImportError:
            self.memory_end = None
    
    @property
    def duration(self):
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def memory_delta(self):
        """Get memory usage delta in MB."""
        if self.memory_start and self.memory_end:
            return self.memory_end - self.memory_start
        return None
    
    @property
    def peak_memory_delta(self):
        """Get peak memory usage delta in MB."""
        if self.memory_start and self.memory_peak:
            return self.memory_peak - self.memory_start
        return None


# Secure Image Testing Fixtures

@pytest.fixture(scope="session")
def test_image_generator():
    """Provide test image generator for the session."""
    return TestImageGenerator()


@pytest.fixture(scope="session")
def test_data_manager():
    """Provide test data manager for the session."""
    manager = TestDataManager()
    manager.setup()
    yield manager
    manager.cleanup()


@pytest.fixture
def performance_tracker():
    """Provide performance tracker for tests."""
    return PerformanceTracker()


@pytest.fixture
def secure_image_test_config():
    """Provide test configuration for secure image processing."""
    return {
        "security_test_timeout": 30.0,
        "performance_test_timeout": 60.0,
        "max_memory_usage_mb": 500,
        "max_processing_time_seconds": 10.0,
        "concurrent_test_workers": 4,
        "batch_test_size": 20
    }


@pytest.fixture
def mock_security_manager():
    """Provide mock security manager for testing."""
    mock_manager = Mock()
    mock_manager.log_security_event.return_value = None
    mock_manager.check_threat_level.return_value = "low"
    mock_manager.get_audit_trail.return_value = []
    mock_manager.validate_permissions.return_value = True
    return mock_manager


@pytest.fixture
def sample_images(test_image_generator):
    """Provide sample images for testing."""
    return {
        'normal': test_image_generator.create_normal_image(),
        'high_entropy': test_image_generator.create_high_entropy_image(),
        'low_entropy': test_image_generator.create_low_entropy_image(),
        'adversarial': test_image_generator.create_adversarial_pattern_image(),
        'steganography': test_image_generator.create_steganography_pattern_image(),
        'text': test_image_generator.create_text_image("SECURITY TEST")
    }


@pytest.fixture
def performance_test_images(test_image_generator):
    """Provide images specifically for performance testing."""
    sizes = [(128, 128), (256, 256), (512, 512)]
    images = {}
    
    for width, height in sizes:
        images[f"normal_{width}x{height}"] = test_image_generator.create_normal_image(width, height)
        images[f"random_{width}x{height}"] = test_image_generator.create_high_entropy_image(width, height)
    
    return images


@pytest.fixture
def api_test_client():
    """Provide FastAPI test client."""
    try:
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI application not available")


@pytest.fixture
def mock_secure_image_processor():
    """Provide mock secure image processor."""
    processor = Mock()
    processor.validate_image.return_value = {
        "is_safe": True,
        "threats_detected": [],
        "confidence_scores": {"entropy": 0.5}
    }
    processor.sanitize_image.return_value = Mock()  # Mock PIL Image
    processor.process_image_data.return_value = {
        "success": True,
        "processed_image": b"mock_processed_data",
        "threats_detected": [],
        "threats_mitigated": []
    }
    return processor


@pytest.fixture
def mock_secure_image_model():
    """Provide mock secure image model."""
    model = Mock()
    model.process_image.return_value = {
        "success": True,
        "processed_image": b"mock_processed_data",
        "security_analysis": {"threats_detected": []}
    }
    model.get_security_statistics.return_value = {
        "total_images_processed": 0,
        "threats_detected": 0,
        "threats_mitigated": 0
    }
    model.set_security_level.return_value = None
    return model


# Utility functions for secure image tests
def assert_valid_security_response(response_data):
    """Assert that a security response has valid structure."""
    assert isinstance(response_data, dict)
    assert "success" in response_data
    assert "is_safe" in response_data
    assert "threats_detected" in response_data
    assert isinstance(response_data["threats_detected"], list)
    
    if "confidence_scores" in response_data:
        assert isinstance(response_data["confidence_scores"], dict)
    
    if "recommendations" in response_data:
        assert isinstance(response_data["recommendations"], list)


def assert_valid_processing_response(response_data):
    """Assert that a processing response has valid structure."""
    assert isinstance(response_data, dict)
    assert "success" in response_data
    
    if response_data["success"]:
        assert "processed_image" in response_data
        assert "threats_detected" in response_data
        assert "threats_mitigated" in response_data
        assert isinstance(response_data["threats_detected"], list)
        assert isinstance(response_data["threats_mitigated"], list)
    else:
        assert "error" in response_data


def assert_performance_metrics(duration, memory_delta=None, max_duration=10.0, max_memory_mb=100):
    """Assert that performance metrics are within acceptable ranges."""
    assert duration is not None
    assert duration >= 0
    assert duration < max_duration, f"Operation too slow: {duration}s > {max_duration}s"
    
    if memory_delta is not None:
        assert memory_delta < max_memory_mb, f"Memory usage too high: {memory_delta}MB > {max_memory_mb}MB"


def create_mock_image_data(width=224, height=224, format='PNG'):
    """Create mock image data for testing."""
    img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()


def skip_if_secure_components_unavailable():
    """Skip test if secure image components are not available."""
    return not SECURE_IMAGE_COMPONENTS_AVAILABLE


# Test data constants for secure image processing
TEST_IMAGE_FORMATS = ['PNG', 'JPEG', 'BMP', 'TIFF']
TEST_SECURITY_LEVELS = ['low', 'medium', 'high', 'maximum']
TEST_IMAGE_SIZES = [(64, 64), (128, 128), (224, 224), (512, 512)]
TEST_THREAT_TYPES = [
    'adversarial_pattern',
    'steganography',
    'malware_signature',
    'suspicious_metadata',
    'format_exploit'
]


# ============================================================================
# End of Secure Image Processing Test Fixtures
# ============================================================================


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    if not config.getoption("--performance"):
        # Skip performance tests unless explicitly requested
        skip_performance = pytest.mark.skip(reason="need --performance option to run")
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_performance)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="run performance tests"
    )


# Removed async_cleanup fixture due to pytest compatibility issues


@pytest.fixture(autouse=True)
def resource_cleanup():
    """Enhanced automatic resource cleanup to prevent accumulation."""
    import asyncio
    import time
    from concurrent.futures import ThreadPoolExecutor
    
    # Store initial state
    initial_threads = threading.active_count()
    
    yield
    
    try:
        # 1. Clean up async resources
        _cleanup_async_resources()
        
        # 2. Clean up CUDA memory if available
        _cleanup_cuda_resources()
        
        # 3. Clean up thread pools and executors
        _cleanup_thread_resources()
        
        # 4. Force multiple garbage collection cycles
        _force_garbage_collection()
        
        # 5. Monitor and warn about thread leaks
        _monitor_thread_leaks(initial_threads)
        
        # 6. Short sleep to allow cleanup completion
        time.sleep(0.01)
        
    except Exception as e:
        warnings.warn(f"Resource cleanup failed: {e}")


def _cleanup_async_resources():
    """Clean up asyncio resources and event loops."""
    try:
        # Close any remaining event loops in the current thread
        try:
            loop = asyncio.get_event_loop()
            if loop and not loop.is_closed():
                # Cancel all pending tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    if not task.done():
                        task.cancel()
                
                # Run any final cleanup tasks
                if pending:
                    try:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    except:
                        pass
        except RuntimeError:
            # No event loop in current thread
            pass
            
        # Force cleanup of asyncio resources
        asyncio.set_event_loop(None)
        
    except Exception:
        pass


def _cleanup_cuda_resources():
    """Clean up CUDA memory and resources."""
    if torch.cuda.is_available():
        try:
            # Clear all device caches
            for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
        except RuntimeError:
            pass


def _cleanup_thread_resources():
    """Clean up thread pools and executors."""
    try:
        # Clean up any thread pool executors
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor
        
        # Force cleanup of default thread pool executor
        if hasattr(concurrent.futures.thread, '_threads_queues'):
            concurrent.futures.thread._threads_queues.clear()
            
        # Clean up any remaining ThreadPoolExecutor instances
        for obj in gc.get_objects():
            if isinstance(obj, ThreadPoolExecutor):
                try:
                    obj.shutdown(wait=False)
                except:
                    pass
                    
    except Exception:
        pass


def _force_garbage_collection():
    """Force multiple garbage collection cycles."""
    # Multiple GC cycles to ensure cleanup
    for _ in range(3):
        gc.collect()
        
    # Collect generation 0, 1, and 2 specifically
    if hasattr(gc, 'collect'):
        for gen in range(3):
            try:
                gc.collect(gen)
            except:
                pass


def _monitor_thread_leaks(initial_threads):
    """Monitor and warn about thread leaks."""
    current_threads = threading.active_count()
    thread_growth = current_threads - initial_threads
    
    # Lower threshold for thread warnings
    if current_threads > 8:  # Reduced from 10
        warnings.warn(f"High thread count detected: {current_threads} (growth: +{thread_growth})")
        
        # If thread count is extremely high, try to identify thread sources
        if current_threads > 20:
            thread_names = [t.name for t in threading.enumerate() if t.is_alive()]
            warnings.warn(f"Active threads: {thread_names[:10]}")  # Show first 10 thread names


@pytest.fixture(autouse=True)
def test_timeout():
    """Set a global timeout for individual tests."""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Test exceeded maximum time limit")
    
    # Set timeout only on Unix-like systems
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 60 second timeout per test
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # On Windows, just yield without signal-based timeout
        yield


# Hook to run after each test
def pytest_runtest_teardown(item, nextitem):
    """Clean up after each test."""
    # Clean up asyncio tasks if there's an event loop
    try:
        loop = asyncio.get_running_loop()
        tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
        
        if tasks:
            for task in tasks:
                if not task.cancelled():
                    task.cancel()
    except RuntimeError:
        pass  # No event loop running
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


# Hook to handle test failures and timeouts  
# Removed pytest_runtest_call as it was causing validation errors


@pytest.fixture
def mock_factory():
    """Provide fast mock factory."""
    from tests.optimized_test_utils import FastMockFactory
    return FastMockFactory


@pytest.fixture
def cleanup_global_circuit_breakers():
    """Cleanup global circuit breakers after test."""
    yield
    # Clear any global circuit breaker registry if it exists
    try:
        from framework.reliability.circuit_breaker import _circuit_breakers
        _circuit_breakers.clear()
    except (ImportError, NameError, AttributeError):
        pass


@pytest.fixture
def benchmark_timer():
    """Provide benchmark timer fixture for performance tests."""
    import time
    
    class BenchmarkTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def __enter__(self):
            self.start_time = time.perf_counter()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.perf_counter()
        
        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return 0
            return self.end_time - self.start_time
    
    return BenchmarkTimer


@pytest.fixture
def circuit_breaker():
    """Provide circuit breaker fixture for tests."""
    try:
        from framework.reliability.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout=5.0,
            success_threshold=2
        )
        breaker = CircuitBreaker(name="test_breaker", config=config)
        yield breaker
    except ImportError:
        # Create a mock if the real implementation isn't available
        from unittest.mock import Mock
        mock_breaker = Mock()
        mock_breaker.call = Mock()
        mock_breaker.get_stats = Mock(return_value={"total_calls": 0, "failure_count": 0})
        mock_breaker._failure_count = 0
        yield mock_breaker

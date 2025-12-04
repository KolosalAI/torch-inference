"""Tests for model adapters."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from framework.adapters.model_adapters import (
    PyTorchModelAdapter, ONNXModelAdapter, TensorRTModelAdapter,
    HuggingFaceModelAdapter, load_model
)
from framework.core.config import InferenceConfig, DeviceConfig, DeviceType
from framework.core.base_model import ModelLoadError


class TestPyTorchModelAdapter:
    """Test PyTorch model adapter."""
    
    @pytest.fixture
    def pytorch_adapter(self, test_config):
        """Create PyTorch model adapter."""
        return PyTorchModelAdapter(test_config)
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple PyTorch model."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
    
    def test_adapter_initialization(self, pytorch_adapter, test_config):
        """Test adapter initialization."""
        assert pytorch_adapter.config == test_config
        assert not pytorch_adapter.is_loaded
        assert pytorch_adapter.model_path is None
    
    def test_load_pytorch_model_direct(self, pytorch_adapter, simple_model, temp_model_dir):
        """Test loading PyTorch model (direct model save)."""
        model_path = temp_model_dir / "model.pt"
        torch.save(simple_model, model_path)
        
        pytorch_adapter.load_model(model_path)
        
        assert pytorch_adapter.is_loaded
        assert pytorch_adapter.model is not None
        assert pytorch_adapter.metadata is not None
        assert pytorch_adapter.metadata.model_type == "pytorch"
    
    def test_load_pytorch_model_state_dict(self, pytorch_adapter, simple_model, temp_model_dir):
        """Test loading PyTorch model from state dict."""
        model_path = temp_model_dir / "model.pth"
        torch.save({"state_dict": simple_model.state_dict()}, model_path)
        
        with pytest.raises(ModelLoadError):
            # Should fail because no model architecture provided
            pytorch_adapter.load_model(model_path)
    
    def test_load_pytorch_model_checkpoint(self, pytorch_adapter, simple_model, temp_model_dir):
        """Test loading PyTorch model from checkpoint."""
        model_path = temp_model_dir / "checkpoint.pt"
        torch.save({"model": simple_model, "epoch": 10}, model_path)
        
        pytorch_adapter.load_model(model_path)
        
        assert pytorch_adapter.is_loaded
        assert pytorch_adapter.model is not None
    
    def test_load_torchscript_model(self, pytorch_adapter, simple_model, temp_model_dir):
        """Test loading TorchScript model."""
        # Create TorchScript model
        scripted_model = torch.jit.script(simple_model)
        model_path = temp_model_dir / "model.torchscript"
        scripted_model.save(str(model_path))
        
        pytorch_adapter.load_model(model_path)
        
        assert pytorch_adapter.is_loaded
        assert pytorch_adapter.model is not None
    
    def test_load_unsupported_format(self, pytorch_adapter, temp_model_dir):
        """Test loading unsupported model format."""
        model_path = temp_model_dir / "model.txt"
        model_path.write_text("not a model")
        
        with pytest.raises(ModelLoadError):
            pytorch_adapter.load_model(model_path)
    
    def test_load_nonexistent_file(self, pytorch_adapter):
        """Test loading non-existent file."""
        with pytest.raises(ModelLoadError):
            pytorch_adapter.load_model("nonexistent.pt")
    
    def test_preprocess(self, pytorch_adapter, simple_model, temp_model_dir):
        """Test preprocessing."""
        model_path = temp_model_dir / "model.pt"
        torch.save(simple_model, model_path)
        pytorch_adapter.load_model(model_path)
        
        # Test with tensor input
        tensor_input = torch.randn(1, 10)
        processed = pytorch_adapter.preprocess(tensor_input)
        assert torch.equal(processed, tensor_input)
        
        # Test with list input
        list_input = [1.0, 2.0, 3.0]
        processed = pytorch_adapter.preprocess(list_input)
        assert isinstance(processed, torch.Tensor)
    
    def test_forward(self, pytorch_adapter, simple_model, temp_model_dir):
        """Test forward pass."""
        model_path = temp_model_dir / "model.pt"
        torch.save(simple_model, model_path)
        pytorch_adapter.load_model(model_path)
        
        inputs = torch.randn(1, 10)
        outputs = pytorch_adapter.forward(inputs)
        
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (1, 5)  # Based on simple_model output
    
    def test_postprocess(self, pytorch_adapter, simple_model, temp_model_dir):
        """Test postprocessing."""
        model_path = temp_model_dir / "model.pt"
        torch.save(simple_model, model_path)
        pytorch_adapter.load_model(model_path)
        
        outputs = torch.randn(1, 5)
        processed = pytorch_adapter.postprocess(outputs)
        
        assert isinstance(processed, dict)
        assert "predictions" in processed or "logits" in processed
    
    def test_prediction_pipeline(self, pytorch_adapter, simple_model, temp_model_dir):
        """Test complete prediction pipeline."""
        model_path = temp_model_dir / "model.pt"
        torch.save(simple_model, model_path)
        pytorch_adapter.load_model(model_path)
        
        inputs = torch.randn(1, 10)
        result = pytorch_adapter.predict(inputs)
        
        assert isinstance(result, dict)


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="onnxruntime not available")
class TestONNXModelAdapter:
    """Test ONNX model adapter."""
    
    @pytest.fixture
    def onnx_adapter(self, test_config):
        """Create ONNX model adapter."""
        return ONNXModelAdapter(test_config)
    
    def test_onnx_adapter_initialization(self, onnx_adapter):
        """Test ONNX adapter initialization."""
        assert not onnx_adapter.is_loaded
        assert onnx_adapter.session is None
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="onnxruntime not available")
    @patch('onnxruntime.InferenceSession')
    def test_load_onnx_model(self, mock_inference_session, onnx_adapter, temp_model_dir):
        """Test loading ONNX model."""
        # Mock ONNX Runtime session
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input", shape=[1, 10])]
        mock_session.get_outputs.return_value = [Mock(name="output", shape=[1, 5])]
        mock_inference_session.return_value = mock_session
        
        model_path = temp_model_dir / "model.onnx"
        model_path.touch()  # Create empty file
        
        onnx_adapter.load_model(model_path)
        
        assert onnx_adapter.is_loaded
        assert onnx_adapter.session == mock_session
        mock_inference_session.assert_called_once()
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="onnxruntime not available")
    @patch('onnxruntime.InferenceSession')
    def test_onnx_inference(self, mock_inference_session, onnx_adapter, temp_model_dir):
        """Test ONNX model inference."""
        # Mock ONNX Runtime session
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input", shape=[1, 10])]
        mock_session.get_outputs.return_value = [Mock(name="output", shape=[1, 5])]
        mock_session.run.return_value = [np.random.randn(1, 5)]
        mock_inference_session.return_value = mock_session
        
        model_path = temp_model_dir / "model.onnx"
        model_path.touch()
        
        onnx_adapter.load_model(model_path)
        
        inputs = torch.randn(1, 10)
        outputs = onnx_adapter.forward(inputs)
        
        assert isinstance(outputs, torch.Tensor)
        mock_session.run.assert_called_once()


@pytest.mark.skipif(TensorRTModelAdapter is None, reason="TensorRT adapter not available")
class TestTensorRTModelAdapter:
    """Test TensorRT model adapter."""
    
    @pytest.fixture
    def tensorrt_adapter(self, test_config):
        """Create TensorRT model adapter."""
        return TensorRTModelAdapter(test_config)
    
    def test_tensorrt_adapter_initialization(self, tensorrt_adapter):
        """Test TensorRT adapter initialization."""
        assert not tensorrt_adapter.is_loaded


@pytest.mark.skipif(HuggingFaceModelAdapter is None, reason="HuggingFace adapter not available")
class TestHuggingFaceModelAdapter:
    """Test HuggingFace model adapter."""
    
    @pytest.fixture
    def hf_adapter(self, test_config):
        """Create HuggingFace model adapter."""
        return HuggingFaceModelAdapter(test_config)
    
    def test_hf_adapter_initialization(self, hf_adapter):
        """Test HuggingFace adapter initialization."""
        assert not hf_adapter.is_loaded
    
    @patch('transformers.AutoConfig')
    @patch('transformers.AutoModel')
    @patch('transformers.AutoTokenizer')
    def test_load_hf_model(self, mock_tokenizer, mock_model, mock_config, hf_adapter):
        """Test loading HuggingFace model."""
        # Mock HuggingFace model, tokenizer, and config
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_config_instance = Mock()
        
        # Configure the mock model to return itself when .to() is called
        mock_model_instance.to.return_value = mock_model_instance
        
        # Configure the mock config to have a hidden_size attribute
        mock_config_instance.hidden_size = 768
        
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_config.from_pretrained.return_value = mock_config_instance
        
        hf_adapter.load_model("bert-base-uncased")
        
        assert hf_adapter.is_loaded
        assert hf_adapter.model == mock_model_instance
        mock_model.from_pretrained.assert_called_once_with("bert-base-uncased")
        mock_tokenizer.from_pretrained.assert_called_once_with("bert-base-uncased")
        mock_config.from_pretrained.assert_called_once_with("bert-base-uncased")


class TestModelAdapterFactory:
    """Test model adapter factory function."""
    
    def test_load_pytorch_model_factory(self, test_config, simple_model, temp_model_dir):
        """Test loading PyTorch model via factory function."""
        model_path = temp_model_dir / "model.pt"
        torch.save(simple_model, model_path)
        
        adapter = load_model(model_path, test_config)
        
        assert isinstance(adapter, PyTorchModelAdapter)
        assert adapter.is_loaded
    
    def test_load_model_auto_detection(self, test_config, simple_model, temp_model_dir):
        """Test automatic model type detection."""
        # PyTorch model
        pt_path = temp_model_dir / "model.pt"
        torch.save(simple_model, pt_path)
        
        pt_adapter = load_model(pt_path, test_config)
        assert isinstance(pt_adapter, PyTorchModelAdapter)
        
        # ONNX model (mock)
        onnx_path = temp_model_dir / "model.onnx"
        onnx_path.touch()
        
        with patch('framework.adapters.model_adapters.ONNXModelAdapter') as mock_onnx_adapter:
            mock_instance = Mock()
            mock_onnx_adapter.return_value = mock_instance
            
            onnx_adapter = load_model(onnx_path, test_config)
            mock_onnx_adapter.assert_called_once_with(test_config)
    
    def test_load_model_unsupported_format(self, test_config, temp_model_dir):
        """Test loading unsupported model format."""
        unsupported_path = temp_model_dir / "model.xyz"
        unsupported_path.touch()
        
        with pytest.raises(ValueError):
            load_model(unsupported_path, test_config)
    
    def test_load_model_from_string(self, test_config):
        """Test loading HuggingFace model from string identifier."""
        with patch('framework.adapters.model_adapters.HuggingFaceModelAdapter') as mock_hf_adapter:
            mock_instance = Mock()
            mock_hf_adapter.return_value = mock_instance
            
            adapter = load_model("bert-base-uncased", test_config)
            mock_hf_adapter.assert_called_once_with(test_config)


class TestModelAdapterIntegration:
    """Integration tests for model adapters."""
    
    def test_adapter_with_different_configs(self, simple_model, temp_model_dir):
        """Test adapter with different configurations."""
        model_path = temp_model_dir / "model.pt"
        torch.save(simple_model, model_path)
        
        # CPU config
        cpu_config = InferenceConfig(
            device=DeviceConfig(device_type=DeviceType.CPU)
        )
        cpu_adapter = PyTorchModelAdapter(cpu_config)
        cpu_adapter.load_model(model_path)
        
        assert cpu_adapter.is_loaded
        assert cpu_adapter.device.type == "cpu"
        
        # Test prediction
        result = cpu_adapter.predict(torch.randn(1, 10))
        assert isinstance(result, dict)
    
    def test_adapter_batch_processing(self, simple_model, temp_model_dir):
        """Test adapter batch processing."""
        model_path = temp_model_dir / "model.pt"
        torch.save(simple_model, model_path)
        
        adapter = PyTorchModelAdapter(InferenceConfig())
        adapter.load_model(model_path)
        
        # Test batch prediction
        batch_inputs = [
            torch.randn(1, 10),
            torch.randn(1, 10),
            torch.randn(1, 10)
        ]
        
        results = adapter.predict_batch(batch_inputs)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
    
    def test_adapter_error_recovery(self, test_config, temp_model_dir):
        """Test adapter error handling and recovery."""
        adapter = PyTorchModelAdapter(test_config)
        
        # Test prediction without loaded model
        with pytest.raises(Exception):
            adapter.predict(torch.randn(1, 10))
        
        # Test loading invalid model
        invalid_path = temp_model_dir / "invalid.pt"
        invalid_path.write_text("not a model")
        
        with pytest.raises(ModelLoadError):
            adapter.load_model(invalid_path)
        
        # Adapter should still be in valid state
        assert not adapter.is_loaded
    
    def test_adapter_model_info(self, simple_model, temp_model_dir):
        """Test adapter model information reporting."""
        model_path = temp_model_dir / "model.pt"
        torch.save(simple_model, model_path)
        
        adapter = PyTorchModelAdapter(InferenceConfig())
        
        # Before loading
        info_before = adapter.model_info
        assert not info_before["loaded"]
        
        # After loading
        adapter.load_model(model_path)
        info_after = adapter.model_info
        
        assert info_after["loaded"]
        assert "metadata" in info_after
        assert "total_parameters" in info_after
        assert info_after["metadata"]["model_type"] == "pytorch"
    
    def test_adapter_memory_management(self, simple_model, temp_model_dir):
        """Test adapter memory management."""
        model_path = temp_model_dir / "model.pt"
        torch.save(simple_model, model_path)
        
        adapter = PyTorchModelAdapter(InferenceConfig())
        adapter.load_model(model_path)
        
        # Get initial memory usage
        memory_before = adapter.get_memory_usage()
        
        # Run some predictions
        for _ in range(5):
            adapter.predict(torch.randn(1, 10))
        
        # Cleanup
        adapter.cleanup()
        
        # Memory usage should be tracked
        assert isinstance(memory_before, dict)


class TestModelAdapterErrorHandling:
    """Test error handling in model adapters."""
    
    def test_load_corrupted_model(self, test_config, temp_model_dir):
        """Test loading corrupted model file."""
        adapter = PyTorchModelAdapter(test_config)
        
        # Create corrupted model file
        corrupted_path = temp_model_dir / "corrupted.pt"
        corrupted_path.write_bytes(b"corrupted data")
        
        with pytest.raises(ModelLoadError):
            adapter.load_model(corrupted_path)
    
    def test_inference_error_handling(self, test_config, temp_model_dir):
        """Test inference error handling."""
        # Create a simple model and mock it to fail during inference
        simple_model = nn.Linear(10, 5)
        model_path = temp_model_dir / "failing_model.pt"
        torch.save(simple_model, model_path)
        
        adapter = PyTorchModelAdapter(test_config)
        adapter.load_model(model_path)
        
        # Mock the model's forward method to raise an error
        original_forward = adapter.model.forward
        def failing_forward(x):
            raise RuntimeError("Model inference failed")
        adapter.model.forward = failing_forward
        
        # Should raise ModelInferenceError
        with pytest.raises(Exception):
            adapter.predict(torch.randn(1, 10))
    
    def test_preprocessing_error_handling(self, test_config, simple_model, temp_model_dir):
        """Test preprocessing error handling."""
        model_path = temp_model_dir / "model.pt"
        torch.save(simple_model, model_path)
        
        adapter = PyTorchModelAdapter(test_config)
        adapter.load_model(model_path)
        
        # Mock preprocess to raise an error
        original_preprocess = adapter.preprocess
        def failing_preprocess(inputs):
            raise ValueError("Preprocessing failed")
        adapter.preprocess = failing_preprocess
        
        # Test with any input - should raise exception
        with pytest.raises(Exception):
            adapter.predict(torch.randn(1, 10))
    
    def test_device_mismatch_handling(self, simple_model, temp_model_dir):
        """Test handling device mismatches."""
        model_path = temp_model_dir / "model.pt"
        torch.save(simple_model, model_path)
        
        # Create config with specific device
        config = InferenceConfig(
            device=DeviceConfig(device_type=DeviceType.CPU)
        )
        adapter = PyTorchModelAdapter(config)
        adapter.load_model(model_path)
        
        # Model should be moved to correct device
        assert next(adapter.model.parameters()).device.type == adapter.device.type

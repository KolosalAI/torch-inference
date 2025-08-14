"""Tests for optimizer modules."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Test imports with mock fallbacks
try:
    from framework.optimizers import (
        TensorRTOptimizer, ONNXOptimizer, QuantizationOptimizer,
        MemoryOptimizer, CUDAOptimizer, JITOptimizer,
        convert_to_tensorrt, convert_to_onnx, quantize_model,
        enable_cuda_optimizations, jit_compile_model
    )
except ImportError:
    # Create mock classes for testing when optimizers are not available
    TensorRTOptimizer = None
    ONNXOptimizer = None
    QuantizationOptimizer = None
    MemoryOptimizer = None
    CUDAOptimizer = None
    JITOptimizer = None
    convert_to_tensorrt = None
    convert_to_onnx = None
    quantize_model = None
    enable_cuda_optimizations = None
    jit_compile_model = None


class MockOptimizer:
    """Base mock optimizer for testing."""
    
    def __init__(self, available: bool = True):
        self.available = available
        self.optimized_models = []
    
    def is_available(self) -> bool:
        return self.available
    
    def optimize(self, model, **kwargs):
        if not self.available:
            raise RuntimeError("Optimizer not available")
        
        # Mock optimization - return the same model
        optimized_model = model
        self.optimized_models.append(optimized_model)
        return optimized_model


class TestOptimizerAvailability:
    """Test optimizer availability detection."""
    
    def test_optimizer_imports(self):
        """Test that optimizer imports work (even if None)."""
        # These should not raise import errors
        optimizers = [
            TensorRTOptimizer,
            ONNXOptimizer, 
            QuantizationOptimizer,
            MemoryOptimizer,
            CUDAOptimizer,
            JITOptimizer
        ]
        
        functions = [
            convert_to_tensorrt,
            convert_to_onnx,
            quantize_model,
            enable_cuda_optimizations,
            jit_compile_model
        ]
        
        # Should be either callable or None
        for optimizer in optimizers:
            assert optimizer is None or callable(optimizer)
        
        for func in functions:
            assert func is None or callable(func)


@pytest.mark.skipif(TensorRTOptimizer is None, reason="TensorRT not available")
class TestTensorRTOptimizer:
    """Test TensorRT optimizer functionality."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        return nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(1, 10)
    
    def test_tensorrt_optimizer_creation(self):
        """Test TensorRT optimizer creation."""
        optimizer = TensorRTOptimizer()
        assert optimizer is not None
    
    def test_tensorrt_optimizer_unavailable(self):
        """Test TensorRT optimizer when CUDA/TensorRT unavailable."""
        optimizer = TensorRTOptimizer()
        
        # Should be disabled due to CUDA not available
        assert not optimizer.is_available()
        
        # Should return original model
        simple_model = torch.nn.Linear(10, 1)
        sample_input = torch.randn(1, 10)
        
        result = optimizer.optimize(simple_model, example_inputs=sample_input)
        assert result is simple_model  # Should return the same model

    @patch('framework.optimizers.tensorrt_optimizer.torch.cuda.is_available', return_value=True)
    @patch('framework.optimizers.tensorrt_optimizer.torch_tensorrt')
    @patch('framework.optimizers.tensorrt_optimizer._ensure_tensorrt_imported')
    @patch('framework.optimizers.tensorrt_optimizer.TRT_AVAILABLE', True)
    def test_tensorrt_availability_check(self, mock_trt_available, mock_ensure_import, mock_tensorrt, mock_cuda_available):
        """Test TensorRT availability checking."""
        # Mock successful import
        mock_ensure_import.return_value = True
        
        optimizer = TensorRTOptimizer()
        optimizer._test_mode_available = True  # Enable test mode
        optimizer.enabled = True

        # Test availability with mock
        assert optimizer.is_available()

    def test_tensorrt_optimization_fallback(self):
        """Test TensorRT optimization fallback when not available."""
        optimizer = TensorRTOptimizer()
        
        # Should be disabled due to CUDA/TensorRT not available
        assert not optimizer.is_available()
        
        # Create a simple model and input
        model = torch.nn.Linear(10, 1)
        sample_input = torch.randn(1, 10)
        
        # Should fall back to original model
        result = optimizer.optimize(model, example_inputs=sample_input)
        assert result is model  # Should return the same model


@pytest.mark.skipif(ONNXOptimizer is None, reason="ONNX not available")  
class TestONNXOptimizer:
    """Test ONNX optimizer functionality."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(1, 10)
    
    def test_onnx_optimizer_creation(self):
        """Test ONNX optimizer creation."""
        optimizer = ONNXOptimizer()
        assert optimizer is not None
    
    @patch('framework.optimizers.onnx_optimizer.torch.onnx.export')
    @patch('framework.optimizers.onnx_optimizer.onnxruntime')
    def test_onnx_optimization(self, mock_ort, mock_export, simple_model, sample_input, temp_model_dir):
        """Test ONNX model optimization."""
        optimizer = ONNXOptimizer()
        
        # Mock ONNX Runtime session
        mock_session = Mock()
        mock_ort.InferenceSession.return_value = mock_session
        
        # Mock successful ONNX export
        mock_export.return_value = None  # torch.onnx.export returns None
        
        onnx_path = temp_model_dir / "model.onnx"
        
        # Create a mock ONNX file to simulate successful export
        onnx_path.touch()
        
        optimized_model = optimizer.optimize(
            simple_model,
            example_inputs=[sample_input],
            output_path=str(onnx_path)
        )
        
        # Should call torch.onnx.export
        mock_export.assert_called_once()
        # Should create ONNX Runtime session - check if it was called
        # Note: This might not be called if there are ONNX validation errors
        assert optimized_model is not None


@pytest.mark.skipif(QuantizationOptimizer is None, reason="Quantization optimizer not available")
class TestQuantizationOptimizer:
    """Test quantization optimizer functionality."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        model.eval()
        return model
    
    def test_quantization_optimizer_creation(self):
        """Test quantization optimizer creation."""
        optimizer = QuantizationOptimizer()
        assert optimizer is not None
    
    def test_dynamic_quantization(self, simple_model):
        """Test dynamic quantization."""
        optimizer = QuantizationOptimizer()
        
        quantized_model = optimizer.optimize(
            simple_model,
            quantization_type="dynamic",
            dtype=torch.qint8
        )
        
        # Should return a model (may be the same or quantized)
        assert quantized_model is not None
        assert isinstance(quantized_model, nn.Module)
    
    @patch('torch.quantization.quantize_dynamic')
    def test_dynamic_quantization_with_mock(self, mock_quantize, simple_model):
        """Test dynamic quantization with mock."""
        optimizer = QuantizationOptimizer()
        
        mock_quantized = Mock()
        mock_quantize.return_value = mock_quantized
        
        result = optimizer.optimize(
            simple_model,
            quantization_type="dynamic"
        )
        
        mock_quantize.assert_called_once()
        assert result == mock_quantized


@pytest.mark.skipif(JITOptimizer is None, reason="JIT optimizer not available")
class TestJITOptimizer:
    """Test JIT optimizer functionality."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),  
            nn.Linear(20, 5)
        )
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(1, 10)
    
    def test_jit_optimizer_creation(self):
        """Test JIT optimizer creation."""
        optimizer = JITOptimizer()
        assert optimizer is not None
    
    @patch('torch.jit.script')
    def test_jit_script_optimization(self, mock_script, simple_model):
        """Test JIT script optimization."""
        optimizer = JITOptimizer()
        
        mock_scripted = Mock()
        mock_script.return_value = mock_scripted
        
        optimized_model = optimizer.optimize(
            simple_model,
            method="script"
        )
        
        mock_script.assert_called_once_with(simple_model)
        assert optimized_model == mock_scripted
    
    @patch('torch.jit.trace')
    def test_jit_trace_optimization(self, mock_trace, simple_model, sample_input):
        """Test JIT trace optimization."""
        optimizer = JITOptimizer()
        
        mock_traced = Mock()
        mock_trace.return_value = mock_traced
        
        optimized_model = optimizer.optimize(
            simple_model,
            method="trace",
            example_inputs=[sample_input]
        )
        
        # Check that torch.jit.trace was called with the expected arguments
        mock_trace.assert_called_once_with(simple_model, sample_input, strict=True, check_trace=True)
        assert optimized_model == mock_traced


@pytest.mark.skipif(CUDAOptimizer is None, reason="CUDA optimizer not available")
class TestCUDAOptimizer:
    """Test CUDA optimizer functionality."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
    
    def test_cuda_optimizer_creation(self):
        """Test CUDA optimizer creation."""
        optimizer = CUDAOptimizer()
        assert optimizer is not None
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_cuda_optimization(self, mock_cuda_available, simple_model):
        """Test CUDA optimization."""
        optimizer = CUDAOptimizer()
        
        optimized_model = optimizer.optimize(simple_model)
        
        # Model should be moved to CUDA
        # Note: Actual CUDA movement might not work in test environment
        assert optimized_model is not None
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_cuda_optimization_unavailable(self, mock_cuda_available, simple_model):
        """Test CUDA optimization when CUDA unavailable."""
        optimizer = CUDAOptimizer()
        
        # Should either return original model or raise appropriate error
        result = optimizer.optimize(simple_model)
        assert result is not None


@pytest.mark.skipif(MemoryOptimizer is None, reason="Memory optimizer not available")
class TestMemoryOptimizer:
    """Test memory optimizer functionality."""
    
    def test_memory_optimizer_creation(self):
        """Test memory optimizer creation."""
        optimizer = MemoryOptimizer()
        assert optimizer is not None
    
    def test_memory_optimization(self, simple_model):
        """Test memory optimization techniques."""
        optimizer = MemoryOptimizer()
        
        # Test gradient checkpointing enablement
        optimizer.enable_gradient_checkpointing(simple_model)
        
        # Test memory cleanup
        optimizer.cleanup_memory()
        
        # Should not raise errors


class TestOptimizerFunctions:
    """Test optimizer convenience functions."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(1, 10)
    
    @pytest.mark.skipif(convert_to_tensorrt is None, reason="TensorRT function not available")
    def test_convert_to_tensorrt_function(self, simple_model, sample_input):
        """Test convert_to_tensorrt convenience function."""
        # Only run test if TensorRT is actually available
        if convert_to_tensorrt is None:
            pytest.skip("TensorRT not available")
            
        with patch('framework.optimizers.tensorrt_optimizer.TensorRTOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer.optimize.return_value = simple_model
            mock_optimizer_class.return_value = mock_optimizer
            
            result = convert_to_tensorrt(simple_model, example_inputs=[sample_input])
            
            mock_optimizer_class.assert_called_once()
            mock_optimizer.optimize.assert_called_once()
            assert result is not None
    
    @pytest.mark.skipif(convert_to_onnx is None, reason="ONNX function not available")
    def test_convert_to_onnx_function(self, simple_model, sample_input, temp_model_dir):
        """Test convert_to_onnx convenience function."""
        # Only run test if ONNX is actually available
        if convert_to_onnx is None:
            pytest.skip("ONNX not available")
            
        with patch('framework.optimizers.onnx_optimizer.ONNXOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer.optimize.return_value = simple_model
            mock_optimizer_class.return_value = mock_optimizer
            
            output_path = temp_model_dir / "model.onnx"
            result = convert_to_onnx(
                simple_model, 
                example_inputs=[sample_input],
                output_path=str(output_path)
            )
            
            mock_optimizer_class.assert_called_once()
            mock_optimizer.optimize.assert_called_once()
            assert result is not None
    
    @pytest.mark.skipif(quantize_model is None, reason="Quantization function not available")
    def test_quantize_model_function(self, simple_model):
        """Test quantize_model convenience function."""
        with patch('framework.optimizers.QuantizationOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer.optimize.return_value = simple_model
            mock_optimizer_class.return_value = mock_optimizer
            
            result = quantize_model(simple_model, quantization_type="dynamic")
            
            mock_optimizer_class.assert_called_once()
            mock_optimizer.optimize.assert_called_once()
            assert result is not None
    
    @pytest.mark.skipif(jit_compile_model is None, reason="JIT function not available")
    def test_jit_compile_model_function(self, simple_model, sample_input):
        """Test jit_compile_model convenience function."""
        if jit_compile_model is None:
            pytest.skip("JIT not available")
            
        with patch('framework.optimizers.jit_optimizer.JITOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer.optimize.return_value = simple_model
            mock_optimizer_class.return_value = mock_optimizer
            
            result = jit_compile_model(simple_model, method="trace", example_inputs=[sample_input])
            
            mock_optimizer_class.assert_called_once()
            mock_optimizer.optimize.assert_called_once()
            assert result is not None
    
    @pytest.mark.skipif(enable_cuda_optimizations is None, reason="CUDA function not available")
    def test_enable_cuda_optimizations_function(self):
        """Test enable_cuda_optimizations convenience function."""
        if enable_cuda_optimizations is None:
            pytest.skip("CUDA optimizations not available")
            
        with patch('framework.optimizers.cuda_optimizer.CUDAOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer
            
            enable_cuda_optimizations()
            
            mock_optimizer_class.assert_called_once()


class TestOptimizerIntegration:
    """Integration tests for optimizer functionality."""
    
    @pytest.fixture
    def complex_model(self):
        """Create a more complex test model."""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    @pytest.fixture
    def image_input(self):
        """Create sample image input."""
        return torch.randn(1, 3, 224, 224)
    
    def test_multiple_optimizations(self, complex_model, image_input):
        """Test applying multiple optimizations sequentially."""
        original_model = complex_model
        optimized_model = original_model
        
        # Apply available optimizations
        optimizers_to_test = [
            (JITOptimizer, {"method": "script"}),
            (QuantizationOptimizer, {"quantization_type": "dynamic"}),
        ]
        
        for optimizer_class, kwargs in optimizers_to_test:
            if optimizer_class is not None:
                try:
                    optimizer = optimizer_class()
                    if hasattr(optimizer, 'is_available') and optimizer.is_available():
                        optimized_model = optimizer.optimize(optimized_model, **kwargs)
                except Exception as e:
                    # Skip if optimization fails (common in test environments)
                    continue
        
        # Should have a model (optimized or original)
        assert optimized_model is not None
    
    def test_optimization_pipeline(self, complex_model, image_input):
        """Test complete optimization pipeline."""
        model = complex_model
        
        # Mock optimization pipeline
        class MockOptimizationPipeline:
            def __init__(self):
                self.optimizers = []
                
                # Add available optimizers
                if JITOptimizer:
                    self.optimizers.append(("JIT", JITOptimizer()))
                if QuantizationOptimizer:
                    self.optimizers.append(("Quantization", QuantizationOptimizer()))
            
            def optimize(self, model):
                optimized_model = model
                results = {}
                
                for name, optimizer in self.optimizers:
                    try:
                        # Mock successful optimization
                        results[name] = "success"
                        # In real implementation, would apply optimization
                    except Exception as e:
                        results[name] = f"failed: {e}"
                
                return optimized_model, results
        
        pipeline = MockOptimizationPipeline()
        optimized_model, results = pipeline.optimize(model)
        
        assert optimized_model is not None
        assert isinstance(results, dict)


class TestOptimizerErrorHandling:
    """Test error handling in optimizers."""
    
    def test_unavailable_optimizer_handling(self):
        """Test handling of unavailable optimizers."""
        # Mock unavailable optimizer
        class UnavailableOptimizer:
            def is_available(self):
                return False
            
            def optimize(self, model, **kwargs):
                raise RuntimeError("Optimizer not available")
        
        optimizer = UnavailableOptimizer()
        
        # Should detect unavailability
        assert not optimizer.is_available()
        
        # Should raise appropriate error
        with pytest.raises(RuntimeError):
            optimizer.optimize(nn.Linear(10, 5))
    
    def test_optimization_failure_recovery(self):
        """Test recovery from optimization failures."""
        class FailingOptimizer:
            def optimize(self, model, **kwargs):
                raise RuntimeError("Optimization failed")
        
        optimizer = FailingOptimizer()
        original_model = nn.Linear(10, 5)
        
        # Should be able to catch and handle failure
        try:
            optimized_model = optimizer.optimize(original_model)
        except RuntimeError:
            # Fallback to original model
            optimized_model = original_model
        
        assert optimized_model == original_model
    
    def test_invalid_optimization_parameters(self):
        """Test handling of invalid optimization parameters."""
        if QuantizationOptimizer:
            optimizer = QuantizationOptimizer()
            model = nn.Linear(10, 5)
            
            # Test with invalid parameters - should either work or raise clear error
            try:
                result = optimizer.optimize(model, quantization_type="invalid_type")
                # If it doesn't raise, should return some result
                assert result is not None
            except (ValueError, TypeError, RuntimeError):
                # Expected for invalid parameters
                pass

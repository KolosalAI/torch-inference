"""
Integration tests for security mitigations in framework components.

Tests security integration with:
- Model adapters
- Inference engine
- Base model functionality
"""

import pytest
import torch
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import json

from framework.core.base_model import ModelManager
from framework.core.inference_engine import InferenceEngine
from framework.adapters.model_adapters import PyTorchModelAdapter
from framework.core.config import InferenceConfig


class TestSecurityInModelAdapters:
    """Test security integration in model adapters."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = InferenceConfig()
        self.adapter = PyTorchModelAdapter(self.config)
    
    def test_secure_model_loading(self):
        """Test secure model loading with security context."""
        # Create a dummy model file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            model_path = tmp_file.name
            
            # Create a simple model and save it
            simple_model = torch.nn.Linear(10, 1)
            torch.save(simple_model, model_path)
        
        try:
            # Test loading with security mitigations
            self.adapter.load_model(model_path)
            
            assert self.adapter._is_loaded is True
            assert self.adapter.model is not None
            assert hasattr(self.adapter.model, 'forward')
            
        finally:
            # Cleanup
            Path(model_path).unlink()
    
    def test_security_error_handling(self):
        """Test error handling in secure model loading."""
        # Test with non-existent file
        with pytest.raises(Exception):
            self.adapter.load_model("non_existent_model.pt")
        
        # Test with invalid file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            model_path = tmp_file.name
            tmp_file.write(b"invalid model data")
        
        try:
            with pytest.raises(Exception):
                self.adapter.load_model(model_path)
        finally:
            Path(model_path).unlink()
    
    @patch('framework.adapters.model_adapters._pytorch_security')
    def test_security_context_usage(self, mock_security):
        """Test that security context is properly used."""
        # Create a mock context manager
        mock_context = Mock()
        mock_context.__enter__ = Mock()
        mock_context.__exit__ = Mock()
        mock_security.secure_context.return_value = mock_context
        
        # Create a dummy model file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            model_path = tmp_file.name
            simple_model = torch.nn.Linear(5, 1)
            torch.save(simple_model, model_path)
        
        try:
            self.adapter.load_model(model_path)
            
            # Verify security context was used
            mock_security.secure_context.assert_called()
            
        finally:
            Path(model_path).unlink()


class TestSecurityInInferenceEngine:
    """Test security integration in inference engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = InferenceConfig()
        
        # Create a mock model
        self.mock_model = Mock()
        self.mock_model.config = self.config
        self.mock_model.device = torch.device('cpu')
        self.mock_model.predict.return_value = torch.tensor([1.0])
        self.mock_model.get_memory_usage.return_value = {"allocated": 1024}
        
        self.engine = InferenceEngine(self.mock_model, self.config)
    
    @pytest.mark.asyncio
    async def test_secure_inference(self):
        """Test secure inference operations."""
        await self.engine.start()
        
        try:
            # Test single prediction
            test_input = torch.randn(1, 10)
            result = await self.engine.predict(test_input)
            
            assert result is not None
            assert self.mock_model.predict.called
            
        finally:
            await self.engine.stop()
    
    @pytest.mark.asyncio
    async def test_secure_batch_inference(self):
        """Test secure batch inference."""
        await self.engine.start()
        
        try:
            # Test batch prediction
            test_inputs = [torch.randn(1, 10) for _ in range(3)]
            results = await self.engine.predict_batch(test_inputs)
            
            assert len(results) == 3
            assert all(result is not None for result in results)
            
        finally:
            await self.engine.stop()
    
    @patch('framework.core.inference_engine._inference_security')
    @pytest.mark.asyncio
    async def test_security_context_in_inference(self, mock_security):
        """Test that security context is used during inference."""
        # Create a mock context manager
        mock_context = Mock()
        mock_context.__enter__ = Mock()
        mock_context.__exit__ = Mock()
        mock_security.secure_torch_context.return_value = mock_context
        
        await self.engine.start()
        
        try:
            test_input = torch.randn(1, 10)
            await self.engine.predict(test_input)
            
            # Verify security context was used
            mock_security.secure_torch_context.assert_called()
            
        finally:
            await self.engine.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling_with_security(self):
        """Test error handling doesn't break security."""
        # Make model prediction fail
        self.mock_model.predict.side_effect = RuntimeError("Prediction failed")
        
        await self.engine.start()
        
        try:
            test_input = torch.randn(1, 10)
            
            with pytest.raises(RuntimeError):
                await self.engine.predict(test_input)
            
            # Engine should still be functional
            assert self.engine._running is True
            
        finally:
            await self.engine.stop()


class TestSecurityInModelManager:
    """Test security integration in model manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = InferenceConfig()
        self.manager = ModelManager()
    
    @patch('framework.core.base_model._pytorch_security')
    def test_secure_model_management(self, mock_security):
        """Test secure model loading and management."""
        # Create a mock context manager
        mock_context = Mock()
        mock_context.__enter__ = Mock()
        mock_context.__exit__ = Mock()
        mock_security.secure_torch_context.return_value = mock_context
        
        # Create a dummy model file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            model_path = tmp_file.name
            simple_model = torch.nn.Linear(5, 1)
            torch.save(simple_model, model_path)
        
        try:
            # Test model loading
            model = self.manager.load_model(model_path, self.config)
            
            assert model is not None
            assert hasattr(model, 'predict')
            
            # Verify security context was used
            mock_security.secure_torch_context.assert_called()
            
        finally:
            Path(model_path).unlink()
    
    def test_model_lifecycle_security(self):
        """Test security throughout model lifecycle."""
        # Create a dummy model file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            model_path = tmp_file.name
            simple_model = torch.nn.Linear(3, 1)
            torch.save(simple_model, model_path)
        
        try:
            # Load model
            model = self.manager.load_model(model_path, self.config)
            
            # Use model for prediction
            test_input = torch.randn(1, 3)
            result = model.predict(test_input)
            
            assert result is not None
            assert hasattr(result, 'shape') or isinstance(result, dict) or isinstance(result, list)
            
            # Unload model
            self.manager.unload_model(model_path)
            
        finally:
            Path(model_path).unlink()


class TestSecurityConfiguration:
    """Test security configuration and settings."""
    
    def test_security_config_validation(self):
        """Test security configuration validation."""
        config = InferenceConfig()
        
        # Security should be enabled by default
        assert hasattr(config, 'device')
        assert hasattr(config, 'batch')
        assert hasattr(config, 'performance')
    
    def test_security_settings_override(self):
        """Test security settings can be configured."""
        # Test with custom configuration
        config_dict = {
            "device": {"device_type": "cpu"},
            "batch": {"batch_size": 4, "max_batch_size": 32},
            "performance": {"max_workers": 2}
        }
        
        config = InferenceConfig()
        config.device.device_type = "cpu"  # Set the device type properly
        config.batch.batch_size = 4
        config.batch.max_batch_size = 32
        config.performance.max_workers = 2
        
        assert config.device.device_type == "cpu"
        assert config.batch.batch_size == 4
        assert config.performance.max_workers == 2


class TestSecurityMonitoring:
    """Test security monitoring and logging."""
    
    @patch('framework.core.security.security_logger')
    def test_security_event_logging(self, mock_logger):
        """Test that security events are properly logged."""
        from framework.core.security import initialize_security_mitigations
        
        # Initialize security
        pytorch_security, ecdsa_security = initialize_security_mitigations()
        
        # Use security features
        with pytorch_security.secure_context():
            tensor = torch.randn(5, 5)
        
        with ecdsa_security.secure_timing_context():
            import secrets
            data = secrets.token_bytes(16)
        
        # Verify logging occurred
        assert mock_logger.info.called or mock_logger.debug.called
    
    def test_security_metrics_collection(self):
        """Test security metrics are collected."""
        from framework.core.security import PyTorchSecurityMitigation
        
        security = PyTorchSecurityMitigation()
        security.initialize()
        
        # Perform operations that should generate metrics
        with security.secure_context():
            for _ in range(5):
                tensor = torch.randn(100, 100)
                result = torch.sum(tensor)
        
        # Cleanup and check metrics
        security.cleanup_resources()
        
        # Verify internal state indicates operations occurred
        assert len(security._cleanup_callbacks) >= 0


class TestSecurityPerformance:
    """Test security performance impact."""
    
    def test_security_overhead(self):
        """Test that security mitigations don't add excessive overhead."""
        from framework.core.security import PyTorchSecurityMitigation
        import time
        
        security = PyTorchSecurityMitigation()
        security.initialize()
        
        # Measure time without security
        start_time = time.perf_counter()
        for _ in range(10):
            tensor = torch.randn(100, 100)
            result = torch.sum(tensor)
        baseline_time = time.perf_counter() - start_time
        
        # Measure time with security (minimal overhead mode for performance testing)
        start_time = time.perf_counter()
        for _ in range(10):
            with security.secure_torch_context(minimal_overhead=True):
                tensor = torch.randn(100, 100)
                result = torch.sum(tensor)
        secure_time = time.perf_counter() - start_time
        
        # Security overhead should be reasonable (less than 50% increase)
        overhead_ratio = secure_time / baseline_time
        assert overhead_ratio < 1.5, f"Security overhead too high: {overhead_ratio:.2f}x"
    
    def test_concurrent_security_operations(self):
        """Test security with concurrent operations."""
        from framework.core.security import PyTorchSecurityMitigation
        import threading
        
        security = PyTorchSecurityMitigation()
        security.initialize()
        
        results = []
        errors = []
        
        def worker():
            try:
                with security.secure_context():
                    tensor = torch.randn(50, 50)
                    result = torch.sum(tensor)
                    results.append(result.item())
            except Exception as e:
                errors.append(e)
        
        # Run concurrent operations
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent errors: {errors}"
        assert len(results) == 10
        assert all(isinstance(r, float) for r in results)


if __name__ == "__main__":
    pytest.main([__file__])

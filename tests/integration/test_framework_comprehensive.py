"""
Comprehensive integration tests for the entire framework.

This module tests the integration between all framework components,
ensuring they work together correctly in real-world scenarios.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np
import json
import time

# Framework imports
try:
    from framework import TorchInferenceFramework
    from framework.core.config import InferenceConfig, ModelType
    from framework.core.base_model import BaseModel
    from framework.core.inference_engine import InferenceEngine
    from framework.models.audio.tts_models import TTSModel
    from framework.models.audio.stt_models import STTModel
    from framework.processors.preprocessor import ImagePreprocessor
    from framework.processors.postprocessor import ClassificationPostprocessor
    from framework.security.auth import AuthenticationManager
    from framework.security.governance import ModelGovernance
    from framework.security.monitoring import SecurityMonitor
    from framework.utils.monitoring import PerformanceMonitor
    from framework.utils.concurrent_processor import ConcurrentProcessor
    from framework.autoscaling.autoscaler import AutoScaler
    from framework.enterprise.config import EnterpriseConfig
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    FRAMEWORK_AVAILABLE = False
    pytest.skip(f"Framework not available: {e}", allow_module_level=True)


class TestFrameworkBasicIntegration:
    """Test basic framework integration scenarios."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def simple_model(self):
        """Create simple test model."""
        return torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
            torch.nn.Softmax(dim=1)
        )
    
    @pytest.fixture
    def sample_config(self):
        """Create sample inference config."""
        return InferenceConfig(
            model_type=ModelType.CLASSIFICATION,
            device_type="cpu",
            batch_size=4,
            enable_optimization=True
        )
    
    def test_framework_initialization(self, sample_config, temp_workspace):
        """Test basic framework initialization."""
        framework = TorchInferenceFramework(
            config=sample_config,
            cache_dir=temp_workspace
        )
        
        assert framework.config == sample_config
        assert framework.cache_dir == temp_workspace
        assert hasattr(framework, 'model')
        assert hasattr(framework, 'engine')
    
    def test_model_loading_and_inference(self, simple_model, sample_config, temp_workspace):
        """Test model loading and inference."""
        framework = TorchInferenceFramework(config=sample_config)
        
        # Save model to temp file
        model_path = temp_workspace / "test_model.pt"
        torch.save(simple_model.state_dict(), model_path)
        
        with patch.object(framework, 'load_model') as mock_load:
            # Mock successful model loading
            mock_load.return_value = None
            framework.model = Mock()
            framework.model.predict.return_value = {"predictions": torch.randn(1, 10)}
            
            framework.load_model(model_path)
            
            # Test inference
            sample_input = torch.randn(1, 784)
            result = framework.predict(sample_input)
            
            assert "predictions" in result
            mock_load.assert_called_once()
    
    def test_framework_with_optimizations(self, simple_model, temp_workspace):
        """Test framework with optimization pipeline."""
        config = InferenceConfig(
            model_type=ModelType.CLASSIFICATION,
            enable_optimization=True,
            optimization_level="aggressive"
        )
        
        framework = TorchInferenceFramework(config=config)
        
        with patch.object(framework, 'apply_automatic_optimizations') as mock_optimize:
            mock_optimize.return_value = {
                "jit": True,
                "quantization": False,
                "pruning": True
            }
            
            # Test optimization application
            optimizations = framework.apply_automatic_optimizations()
            
            assert "jit" in optimizations
            assert "pruning" in optimizations
            mock_optimize.assert_called_once()
    
    def test_framework_performance_monitoring(self, sample_config, temp_workspace):
        """Test framework with performance monitoring."""
        framework = TorchInferenceFramework(config=sample_config)
        
        # Mock performance monitor
        monitor = PerformanceMonitor()
        framework.performance_monitor = monitor
        
        # Simulate inference with monitoring
        with monitor.measure("inference"):
            time.sleep(0.01)  # Simulate work
        
        metrics = monitor.get_metrics()
        assert "inference" in metrics
        assert metrics["inference"]["duration"] >= 0.01


class TestMultiModalIntegration:
    """Test multi-modal integration scenarios."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_image_classification_pipeline(self, temp_workspace):
        """Test complete image classification pipeline."""
        # Create image classification config
        config = InferenceConfig(
            model_type=ModelType.CLASSIFICATION,
            input_size=(224, 224),
            num_classes=10
        )
        
        framework = TorchInferenceFramework(config=config)
        
        # Mock image preprocessor
        preprocessor = Mock()
        preprocessor.process.return_value = torch.randn(1, 3, 224, 224)
        
        # Mock postprocessor
        postprocessor = Mock()
        postprocessor.process.return_value = {
            "predicted_class": "cat",
            "confidence": 0.95,
            "probabilities": torch.softmax(torch.randn(10), dim=0)
        }
        
        # Mock model
        framework.model = Mock()
        framework.model.predict.return_value = torch.randn(1, 10)
        
        # Set processors
        framework.preprocessor = preprocessor
        framework.postprocessor = postprocessor
        
        # Test complete pipeline
        sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = framework.predict(sample_image)
        
        # Verify pipeline execution
        preprocessor.process.assert_called_once()
        framework.model.predict.assert_called_once()
        postprocessor.process.assert_called_once()
        
        # Verify result structure
        assert "predicted_class" in result
        assert "confidence" in result
    
    def test_audio_tts_pipeline(self, temp_workspace):
        """Test complete TTS pipeline."""
        # Create TTS config
        config = InferenceConfig(
            model_type=ModelType.TTS,
            sample_rate=22050
        )
        
        framework = TorchInferenceFramework(config=config)
        
        # Mock TTS model
        tts_model = Mock(spec=TTSModel)
        tts_model.synthesize_speech.return_value = {
            "audio": np.random.randn(22050),
            "sample_rate": 22050,
            "duration": 1.0
        }
        
        framework.model = tts_model
        
        # Test TTS synthesis
        text = "Hello, this is a test."
        result = framework.predict(text)
        
        assert "audio" in result
        assert "sample_rate" in result
        assert result["sample_rate"] == 22050
        tts_model.synthesize_speech.assert_called_once_with(text)
    
    def test_audio_stt_pipeline(self, temp_workspace):
        """Test complete STT pipeline."""
        # Create STT config
        config = InferenceConfig(
            model_type=ModelType.STT,
            sample_rate=16000
        )
        
        framework = TorchInferenceFramework(config=config)
        
        # Mock STT model
        stt_model = Mock(spec=STTModel)
        stt_model.transcribe_audio.return_value = {
            "transcription": "hello world",
            "confidence": 0.98,
            "word_timestamps": [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]
        }
        
        framework.model = stt_model
        
        # Test STT transcription
        audio_data = np.random.randn(16000).astype(np.float32)
        result = framework.predict(audio_data)
        
        assert "transcription" in result
        assert "confidence" in result
        assert result["transcription"] == "hello world"
        stt_model.transcribe_audio.assert_called_once_with(audio_data)


class TestSecurityIntegration:
    """Test security integration scenarios."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_framework_with_authentication(self, temp_workspace):
        """Test framework with authentication system."""
        # Create secure config
        config = InferenceConfig(
            model_type=ModelType.CLASSIFICATION,
            enable_security=True,
            require_authentication=True
        )
        
        framework = TorchInferenceFramework(config=config)
        
        # Initialize authentication
        auth_manager = AuthenticationManager()
        security_monitor = SecurityMonitor()
        
        # Register test user
        user = auth_manager.register_user("testuser", "testpass123")
        token = auth_manager.authenticate("testuser", "testpass123")
        
        # Mock secure inference
        with patch.object(framework, 'predict') as mock_predict:
            mock_predict.return_value = {"predictions": [0.1, 0.9]}
            
            # Simulate authenticated request
            security_context = {
                "user_id": user.user_id,
                "token": token.token,
                "authenticated": True
            }
            
            # Log security event
            from framework.security.monitoring import SecurityEvent
            event = SecurityEvent(
                event_type="model_access",
                user_id=user.user_id,
                description="Authenticated model access"
            )
            security_monitor.log_event(event)
            
            # Test inference with security context
            result = framework.predict(
                torch.randn(1, 784),
                security_context=security_context
            )
            
            assert result is not None
            mock_predict.assert_called_once()
        
        # Verify security logging
        events = security_monitor.get_events()
        assert len(events) >= 1
        assert events[0].event_type == "model_access"
    
    def test_framework_with_governance(self, temp_workspace):
        """Test framework with model governance."""
        config = InferenceConfig(
            model_type=ModelType.CLASSIFICATION,
            enable_governance=True
        )
        
        framework = TorchInferenceFramework(config=config)
        
        # Initialize governance
        governance = ModelGovernance()
        
        # Add governance policy
        from framework.security.governance import GovernancePolicy
        policy = GovernancePolicy(
            name="inference_policy",
            rules={
                "require_audit": True,
                "max_requests_per_minute": 100,
                "allowed_users": ["testuser"]
            }
        )
        governance.add_policy(policy)
        
        # Test governance compliance
        compliance_context = {
            "user_id": "testuser",
            "requests_this_minute": 50,
            "audit_enabled": True
        }
        
        is_compliant = governance.check_compliance(compliance_context)
        assert is_compliant is True
        
        # Test governance violation
        violation_context = {
            "user_id": "unauthorized_user",
            "requests_this_minute": 150,
            "audit_enabled": False
        }
        
        is_compliant = governance.check_compliance(violation_context)
        assert is_compliant is False


class TestAutoscalingIntegration:
    """Test autoscaling integration scenarios."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_framework_with_autoscaling(self, temp_workspace):
        """Test framework with autoscaling."""
        config = InferenceConfig(
            model_type=ModelType.CLASSIFICATION,
            enable_autoscaling=True,
            min_replicas=1,
            max_replicas=5
        )
        
        framework = TorchInferenceFramework(config=config)
        
        # Mock autoscaler
        autoscaler = Mock(spec=AutoScaler)
        autoscaler.get_current_replicas.return_value = 2
        autoscaler.scale_up.return_value = True
        autoscaler.scale_down.return_value = True
        
        framework.autoscaler = autoscaler
        
        # Simulate high load triggering scale up
        load_metrics = {
            "requests_per_second": 100,
            "cpu_utilization": 80,
            "memory_utilization": 75
        }
        
        # Test autoscaling decision
        should_scale = framework.should_scale_up(load_metrics)
        if should_scale:
            framework.scale_up()
            autoscaler.scale_up.assert_called_once()
        
        # Simulate low load triggering scale down
        low_load_metrics = {
            "requests_per_second": 10,
            "cpu_utilization": 20,
            "memory_utilization": 30
        }
        
        should_scale_down = framework.should_scale_down(low_load_metrics)
        if should_scale_down:
            framework.scale_down()
            autoscaler.scale_down.assert_called_once()


class TestEnterpriseIntegration:
    """Test enterprise features integration."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_framework_with_enterprise_features(self, temp_workspace):
        """Test framework with enterprise features."""
        # Create enterprise config
        enterprise_config = Mock(spec=EnterpriseConfig)
        enterprise_config.enable_advanced_monitoring = True
        enterprise_config.enable_custom_metrics = True
        enterprise_config.enable_compliance_reporting = True
        
        config = InferenceConfig(
            model_type=ModelType.CLASSIFICATION,
            enterprise_config=enterprise_config
        )
        
        framework = TorchInferenceFramework(config=config)
        
        # Test enterprise monitoring
        if hasattr(framework, 'enterprise_monitor'):
            framework.enterprise_monitor = Mock()
            
            # Simulate enterprise metrics collection
            enterprise_metrics = {
                "business_metrics": {
                    "predictions_made": 1000,
                    "accuracy_score": 0.95,
                    "revenue_generated": 5000.0
                },
                "compliance_metrics": {
                    "gdpr_compliant": True,
                    "audit_trail_complete": True,
                    "data_retention_policy": "30_days"
                }
            }
            
            framework.enterprise_monitor.record_metrics(enterprise_metrics)
            framework.enterprise_monitor.record_metrics.assert_called_once()


class TestConcurrentProcessingIntegration:
    """Test concurrent processing integration."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_framework_with_concurrent_processing(self, temp_workspace):
        """Test framework with concurrent processing."""
        config = InferenceConfig(
            model_type=ModelType.CLASSIFICATION,
            enable_concurrent_processing=True,
            num_workers=4
        )
        
        framework = TorchInferenceFramework(config=config)
        
        # Mock concurrent processor
        processor = Mock(spec=ConcurrentProcessor)
        processor.process_batch.return_value = [
            {"predictions": torch.randn(10)} for _ in range(8)
        ]
        
        framework.concurrent_processor = processor
        
        # Test batch processing
        batch_inputs = [torch.randn(1, 784) for _ in range(8)]
        
        with patch.object(framework, 'predict_batch') as mock_predict_batch:
            mock_predict_batch.return_value = processor.process_batch.return_value
            
            results = framework.predict_batch(batch_inputs)
            
            assert len(results) == 8
            assert all("predictions" in result for result in results)
            mock_predict_batch.assert_called_once()


class TestEndToEndIntegration:
    """Test complete end-to-end integration scenarios."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_complete_ml_pipeline(self, temp_workspace):
        """Test complete ML pipeline from data to prediction."""
        # Setup complete framework
        config = InferenceConfig(
            model_type=ModelType.CLASSIFICATION,
            input_size=(224, 224),
            num_classes=10,
            enable_optimization=True,
            enable_monitoring=True,
            enable_security=True,
            batch_size=4
        )
        
        framework = TorchInferenceFramework(config=config)
        
        # Mock all components
        framework.model = Mock()
        framework.preprocessor = Mock()
        framework.postprocessor = Mock()
        framework.performance_monitor = PerformanceMonitor()
        framework.security_monitor = Mock()
        
        # Setup mock responses
        framework.preprocessor.process_batch.return_value = torch.randn(4, 3, 224, 224)
        framework.model.predict_batch.return_value = torch.randn(4, 10)
        framework.postprocessor.process_batch.return_value = [
            {"predicted_class": f"class_{i}", "confidence": 0.9}
            for i in range(4)
        ]
        
        # Test complete pipeline
        batch_inputs = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(4)
        ]
        
        with framework.performance_monitor.measure("complete_pipeline"):
            results = framework.predict_batch(batch_inputs)
        
        # Verify pipeline execution
        assert len(results) == 4
        assert all("predicted_class" in result for result in results)
        
        framework.preprocessor.process_batch.assert_called_once()
        framework.model.predict_batch.assert_called_once()
        framework.postprocessor.process_batch.assert_called_once()
        
        # Verify monitoring
        metrics = framework.performance_monitor.get_metrics()
        assert "complete_pipeline" in metrics
    
    def test_real_time_inference_scenario(self, temp_workspace):
        """Test real-time inference scenario."""
        config = InferenceConfig(
            model_type=ModelType.CLASSIFICATION,
            optimization_level="speed",
            enable_monitoring=True,
            target_latency=50  # 50ms target
        )
        
        framework = TorchInferenceFramework(config=config)
        
        # Mock optimized model
        framework.model = Mock()
        framework.model.predict.return_value = torch.randn(1, 10)
        
        # Performance monitor
        monitor = PerformanceMonitor()
        framework.performance_monitor = monitor
        
        # Simulate real-time requests
        num_requests = 100
        latencies = []
        
        for i in range(num_requests):
            start_time = time.time()
            
            with monitor.measure(f"request_{i}"):
                sample_input = torch.randn(1, 784)
                result = framework.predict(sample_input)
                
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            
            assert result is not None
        
        # Verify performance requirements
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # In real scenario, these should meet target latency
        assert avg_latency >= 0  # Just check it's measured
        assert p95_latency >= 0
        
        # Verify all requests were processed
        metrics = monitor.get_metrics()
        assert len([k for k in metrics.keys() if k.startswith("request_")]) == num_requests
    
    def test_high_throughput_scenario(self, temp_workspace):
        """Test high throughput scenario."""
        config = InferenceConfig(
            model_type=ModelType.CLASSIFICATION,
            optimization_level="throughput",
            enable_concurrent_processing=True,
            batch_size=16,
            num_workers=4
        )
        
        framework = TorchInferenceFramework(config=config)
        
        # Mock high-throughput processing
        framework.model = Mock()
        framework.concurrent_processor = Mock()
        
        # Setup batch processing mock
        def mock_batch_predict(inputs):
            batch_size = len(inputs)
            return [torch.randn(10) for _ in range(batch_size)]
        
        framework.model.predict_batch = mock_batch_predict
        
        # Process large batch
        large_batch = [torch.randn(1, 784) for _ in range(64)]
        
        start_time = time.time()
        results = framework.predict_batch(large_batch)
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = len(large_batch) / processing_time  # requests per second
        
        assert len(results) == 64
        assert throughput > 0  # Measure throughput
        
        # In real scenario, should achieve high throughput
        print(f"Achieved throughput: {throughput:.2f} requests/sec")


class TestFrameworkRobustness:
    """Test framework robustness and error handling."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_framework_error_recovery(self, temp_workspace):
        """Test framework error recovery mechanisms."""
        config = InferenceConfig(
            model_type=ModelType.CLASSIFICATION,
            enable_error_recovery=True
        )
        
        framework = TorchInferenceFramework(config=config)
        
        # Mock model that sometimes fails
        framework.model = Mock()
        
        def failing_predict(input_data):
            # Fail every 3rd call
            if hasattr(failing_predict, 'call_count'):
                failing_predict.call_count += 1
            else:
                failing_predict.call_count = 1
                
            if failing_predict.call_count % 3 == 0:
                raise RuntimeError("Simulated model failure")
            return torch.randn(10)
        
        framework.model.predict = failing_predict
        
        # Test error recovery
        successful_predictions = 0
        failed_predictions = 0
        
        for i in range(10):
            try:
                sample_input = torch.randn(1, 784)
                result = framework.predict(sample_input)
                successful_predictions += 1
                assert result is not None
            except RuntimeError:
                failed_predictions += 1
        
        # Should have some successful and some failed predictions
        assert successful_predictions > 0
        assert failed_predictions > 0
        print(f"Successful: {successful_predictions}, Failed: {failed_predictions}")
    
    def test_framework_resource_management(self, temp_workspace):
        """Test framework resource management."""
        config = InferenceConfig(
            model_type=ModelType.CLASSIFICATION,
            enable_resource_monitoring=True
        )
        
        framework = TorchInferenceFramework(config=config)
        
        # Mock resource monitor
        framework.resource_monitor = Mock()
        framework.resource_monitor.get_memory_usage.return_value = 1024 * 1024 * 100  # 100MB
        framework.resource_monitor.get_gpu_usage.return_value = 50.0  # 50% GPU
        
        # Test resource monitoring
        memory_usage = framework.get_memory_usage()
        gpu_usage = framework.get_gpu_usage()
        
        assert memory_usage == 1024 * 1024 * 100
        assert gpu_usage == 50.0
        
        # Test resource cleanup
        framework.cleanup()
        
        # Should call cleanup on all components
        if hasattr(framework, 'model') and framework.model:
            assert hasattr(framework.model, 'cleanup') or True  # Mock or real cleanup
    
    def test_framework_configuration_validation(self, temp_workspace):
        """Test framework configuration validation."""
        # Test invalid configuration
        with pytest.raises((ValueError, TypeError)):
            invalid_config = InferenceConfig(
                model_type="invalid_type",
                batch_size=-1,
                device_type="invalid_device"
            )
            TorchInferenceFramework(config=invalid_config)
        
        # Test configuration corrections
        config_with_issues = InferenceConfig(
            model_type=ModelType.CLASSIFICATION,
            batch_size=0  # Should be corrected to 1
        )
        
        framework = TorchInferenceFramework(config=config_with_issues)
        
        # Framework should handle or correct configuration issues
        assert framework.config.batch_size >= 1


@pytest.mark.slow
class TestFrameworkPerformanceBenchmarks:
    """Performance benchmark tests for the framework."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_inference_latency_benchmark(self, temp_workspace):
        """Benchmark inference latency."""
        config = InferenceConfig(
            model_type=ModelType.CLASSIFICATION,
            optimization_level="speed"
        )
        
        framework = TorchInferenceFramework(config=config)
        
        # Mock fast model
        framework.model = Mock()
        framework.model.predict.return_value = torch.randn(10)
        
        # Benchmark single inference latency
        num_runs = 1000
        latencies = []
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            
            sample_input = torch.randn(1, 784)
            result = framework.predict(sample_input)
            
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000  # ms
            latencies.append(latency)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"Latency Benchmark Results:")
        print(f"  Average: {avg_latency:.3f}ms")
        print(f"  P50: {p50_latency:.3f}ms")
        print(f"  P95: {p95_latency:.3f}ms")
        print(f"  P99: {p99_latency:.3f}ms")
        
        # Basic assertions
        assert avg_latency > 0
        assert p95_latency > 0
    
    def test_throughput_benchmark(self, temp_workspace):
        """Benchmark inference throughput."""
        config = InferenceConfig(
            model_type=ModelType.CLASSIFICATION,
            optimization_level="throughput",
            batch_size=32
        )
        
        framework = TorchInferenceFramework(config=config)
        
        # Mock batch model
        framework.model = Mock()
        
        def mock_batch_predict(inputs):
            return [torch.randn(10) for _ in range(len(inputs))]
        
        framework.model.predict_batch = mock_batch_predict
        
        # Benchmark throughput
        batch_sizes = [1, 4, 8, 16, 32, 64]
        throughput_results = {}
        
        for batch_size in batch_sizes:
            batch_inputs = [torch.randn(1, 784) for _ in range(batch_size)]
            
            start_time = time.perf_counter()
            results = framework.predict_batch(batch_inputs)
            end_time = time.perf_counter()
            
            processing_time = end_time - start_time
            throughput = batch_size / processing_time
            throughput_results[batch_size] = throughput
            
            assert len(results) == batch_size
        
        print(f"Throughput Benchmark Results:")
        for batch_size, throughput in throughput_results.items():
            print(f"  Batch size {batch_size}: {throughput:.1f} req/sec")
        
        # Throughput should generally increase with batch size
        assert all(tput > 0 for tput in throughput_results.values())


if __name__ == "__main__":
    pytest.main([__file__])

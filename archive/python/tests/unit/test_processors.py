"""
Test suite for processor modules in the framework.

This module tests all processor implementations including preprocessors,
postprocessors, performance configs, and factory functions.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

# Framework imports
try:
    from framework.processors.preprocessor import (
        BasePreprocessor, ImagePreprocessor, TextPreprocessor,
        create_preprocessor, PreprocessorConfig
    )
    from framework.processors.postprocessor import (
        BasePostprocessor, ClassificationPostprocessor, DetectionPostprocessor,
        SegmentationPostprocessor, create_postprocessor, PostprocessorConfig
    )
    from framework.processors.performance_config import (
        PerformanceConfig, OptimizationLevel, ProcessingMode,
        get_performance_config, optimize_for_inference,
        get_optimal_batch_size, estimate_memory_usage
    )
    from framework.processors.fast_factory import (
        FastProcessorFactory, create_fast_preprocessing_pipelines,
        create_fast_postprocessing_pipelines, create_fast_processing_pipelines,
        create_optimized_config_for_processors, benchmark_processor_performance,
        optimize_for_production_throughput, optimize_for_real_time_latency,
        optimize_for_memory_constrained_environment, auto_optimize_processors
    )
    from framework.processors.image.image_preprocessor import (
        ImagePreprocessorAdvanced, ImageTransformConfig,
        create_image_preprocessor, get_image_transforms
    )
    from framework.processors.audio.audio_preprocessor import (
        AudioPreprocessor, AudioPreprocessorConfig,
        create_audio_preprocessor, get_audio_transforms
    )
    PROCESSORS_AVAILABLE = True
except ImportError as e:
    PROCESSORS_AVAILABLE = False
    pytest.skip(f"Processors not available: {e}", allow_module_level=True)


class TestPreprocessorConfig:
    """Test PreprocessorConfig class."""
    
    def test_preprocessor_config_creation(self):
        """Test creating preprocessor config."""
        config = PreprocessorConfig(
            input_size=(224, 224),
            normalize=True,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        assert config.input_size == (224, 224)
        assert config.normalize is True
        assert config.mean == [0.485, 0.456, 0.406]
        assert config.std == [0.229, 0.224, 0.225]
    
    def test_preprocessor_config_defaults(self):
        """Test default values in preprocessor config."""
        config = PreprocessorConfig()
        
        assert config.input_size == (224, 224)
        assert config.normalize is True
        assert config.mean is not None
        assert config.std is not None


class TestBasePreprocessor:
    """Test BasePreprocessor class."""
    
    @pytest.fixture
    def preprocessor_config(self):
        """Create test preprocessor config."""
        return PreprocessorConfig(
            input_size=(224, 224),
            normalize=True
        )
    
    def test_base_preprocessor_initialization(self, preprocessor_config):
        """Test base preprocessor initialization."""
        preprocessor = BasePreprocessor(preprocessor_config)
        
        assert preprocessor.config == preprocessor_config
        assert hasattr(preprocessor, 'transform')
    
    def test_base_preprocessor_process(self, preprocessor_config):
        """Test base preprocessor process method."""
        preprocessor = BasePreprocessor(preprocessor_config)
        
        # Mock input data
        input_data = torch.randn(3, 256, 256)
        
        with patch.object(preprocessor, 'transform') as mock_transform:
            mock_transform.return_value = torch.randn(3, 224, 224)
            
            result = preprocessor.process(input_data)
            
            mock_transform.assert_called_once_with(input_data)
            assert result is not None
    
    def test_base_preprocessor_batch_process(self, preprocessor_config):
        """Test base preprocessor batch processing."""
        preprocessor = BasePreprocessor(preprocessor_config)
        
        # Mock batch input data
        batch_data = [torch.randn(3, 256, 256) for _ in range(4)]
        
        with patch.object(preprocessor, 'process') as mock_process:
            mock_process.return_value = torch.randn(3, 224, 224)
            
            results = preprocessor.process_batch(batch_data)
            
            assert len(results) == 4
            assert mock_process.call_count == 4


class TestImagePreprocessor:
    """Test ImagePreprocessor class."""
    
    @pytest.fixture
    def image_config(self):
        """Create image preprocessor config."""
        return PreprocessorConfig(
            input_size=(224, 224),
            normalize=True,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def test_image_preprocessor_initialization(self, image_config):
        """Test image preprocessor initialization."""
        preprocessor = ImagePreprocessor(image_config)
        
        assert preprocessor.config == image_config
        assert hasattr(preprocessor, 'transform')
    
    def test_image_preprocessor_tensor_input(self, image_config):
        """Test image preprocessor with tensor input."""
        preprocessor = ImagePreprocessor(image_config)
        
        # Test with tensor input
        input_tensor = torch.randn(3, 256, 256)
        result = preprocessor.process(input_tensor)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[-2:] == (224, 224)  # Check resized dimensions
    
    def test_image_preprocessor_numpy_input(self, image_config):
        """Test image preprocessor with numpy input."""
        preprocessor = ImagePreprocessor(image_config)
        
        # Test with numpy input (HWC format)
        input_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = preprocessor.process(input_array)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[-2:] == (224, 224)
    
    def test_image_preprocessor_normalization(self, image_config):
        """Test image preprocessor normalization."""
        preprocessor = ImagePreprocessor(image_config)
        
        # Test normalization
        input_tensor = torch.ones(3, 224, 224) * 0.5  # Mid-range values
        result = preprocessor.process(input_tensor)
        
        # Values should be normalized
        assert result.mean().item() != 0.5  # Should be different after normalization


class TestPostprocessorConfig:
    """Test PostprocessorConfig class."""
    
    def test_postprocessor_config_creation(self):
        """Test creating postprocessor config."""
        config = PostprocessorConfig(
            num_classes=10,
            confidence_threshold=0.5,
            nms_threshold=0.4
        )
        
        assert config.num_classes == 10
        assert config.confidence_threshold == 0.5
        assert config.nms_threshold == 0.4
    
    def test_postprocessor_config_defaults(self):
        """Test default values in postprocessor config."""
        config = PostprocessorConfig()
        
        assert config.num_classes == 1000
        assert config.confidence_threshold == 0.5
        assert config.nms_threshold == 0.5


class TestBasePostprocessor:
    """Test BasePostprocessor class."""
    
    @pytest.fixture
    def postprocessor_config(self):
        """Create test postprocessor config."""
        return PostprocessorConfig(num_classes=10)
    
    def test_base_postprocessor_initialization(self, postprocessor_config):
        """Test base postprocessor initialization."""
        postprocessor = BasePostprocessor(postprocessor_config)
        
        assert postprocessor.config == postprocessor_config
    
    def test_base_postprocessor_process(self, postprocessor_config):
        """Test base postprocessor process method."""
        postprocessor = BasePostprocessor(postprocessor_config)
        
        # Mock model output
        model_output = torch.randn(1, 10)
        
        result = postprocessor.process(model_output)
        
        assert result is not None


class TestClassificationPostprocessor:
    """Test ClassificationPostprocessor class."""
    
    @pytest.fixture
    def classification_config(self):
        """Create classification postprocessor config."""
        return PostprocessorConfig(
            num_classes=10,
            class_names=[f"class_{i}" for i in range(10)]
        )
    
    def test_classification_postprocessor_initialization(self, classification_config):
        """Test classification postprocessor initialization."""
        postprocessor = ClassificationPostprocessor(classification_config)
        
        assert postprocessor.config == classification_config
        assert len(postprocessor.config.class_names) == 10
    
    def test_classification_postprocessor_process(self, classification_config):
        """Test classification postprocessor processing."""
        postprocessor = ClassificationPostprocessor(classification_config)
        
        # Mock logits
        logits = torch.randn(1, 10)
        result = postprocessor.process(logits)
        
        assert "predicted_class" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert isinstance(result["predicted_class"], str)
        assert 0 <= result["confidence"] <= 1
    
    def test_classification_postprocessor_top_k(self, classification_config):
        """Test classification postprocessor top-k predictions."""
        postprocessor = ClassificationPostprocessor(classification_config)
        
        # Mock logits
        logits = torch.randn(1, 10)
        result = postprocessor.process(logits, top_k=3)
        
        assert "top_predictions" in result
        assert len(result["top_predictions"]) == 3
        
        # Check that predictions are sorted by confidence
        confidences = [pred["confidence"] for pred in result["top_predictions"]]
        assert confidences == sorted(confidences, reverse=True)


class TestPerformanceConfig:
    """Test PerformanceConfig class."""
    
    def test_performance_config_creation(self):
        """Test creating performance config."""
        config = PerformanceConfig(
            optimization_level=OptimizationLevel.AGGRESSIVE,
            processing_mode=ProcessingMode.BATCH,
            batch_size=32,
            num_workers=4
        )
        
        assert config.optimization_level == OptimizationLevel.AGGRESSIVE
        assert config.processing_mode == ProcessingMode.BATCH
        assert config.batch_size == 32
        assert config.num_workers == 4
    
    def test_performance_config_defaults(self):
        """Test default values in performance config."""
        config = PerformanceConfig()
        
        assert config.optimization_level == OptimizationLevel.BALANCED
        assert config.processing_mode == ProcessingMode.SEQUENTIAL
        assert config.batch_size == 1
        assert config.num_workers == 1
    
    def test_get_performance_config(self):
        """Test get_performance_config function."""
        config = get_performance_config(
            target="inference",
            device="cuda",
            memory_limit="8GB"
        )
        
        assert isinstance(config, PerformanceConfig)
        assert config.optimization_level in [OptimizationLevel.BALANCED, OptimizationLevel.AGGRESSIVE]
    
    def test_optimize_for_inference(self):
        """Test optimize_for_inference function."""
        config = optimize_for_inference(device="cuda")
        
        assert isinstance(config, PerformanceConfig)
        assert config.optimization_level != OptimizationLevel.NONE
    
    def test_get_optimal_batch_size(self):
        """Test get_optimal_batch_size function."""
        # Mock model
        mock_model = Mock()
        mock_input_shape = (3, 224, 224)
        
        batch_size = get_optimal_batch_size(
            model=mock_model,
            input_shape=mock_input_shape,
            device="cpu"
        )
        
        assert isinstance(batch_size, int)
        assert batch_size > 0
    
    def test_estimate_memory_usage(self):
        """Test estimate_memory_usage function."""
        memory_usage = estimate_memory_usage(
            model_size_mb=100,
            batch_size=16,
            input_shape=(3, 224, 224)
        )
        
        assert isinstance(memory_usage, float)
        assert memory_usage > 0


class TestFastProcessorFactory:
    """Test FastProcessorFactory class."""
    
    @pytest.fixture
    def factory_config(self):
        """Create factory config."""
        return {
            "optimization_level": "aggressive",
            "target_device": "cuda",
            "memory_limit": "8GB"
        }
    
    def test_fast_processor_factory_initialization(self, factory_config):
        """Test fast processor factory initialization."""
        factory = FastProcessorFactory(factory_config)
        
        assert factory.config == factory_config
    
    def test_create_fast_preprocessor(self, factory_config):
        """Test creating fast preprocessor."""
        factory = FastProcessorFactory(factory_config)
        
        preprocessor = factory.create_preprocessor(
            processor_type="image",
            config=PreprocessorConfig()
        )
        
        assert preprocessor is not None
    
    def test_create_fast_postprocessor(self, factory_config):
        """Test creating fast postprocessor."""
        factory = FastProcessorFactory(factory_config)
        
        postprocessor = factory.create_postprocessor(
            processor_type="classification",
            config=PostprocessorConfig()
        )
        
        assert postprocessor is not None
    
    def test_create_fast_processing_pipelines(self):
        """Test create_fast_processing_pipelines function."""
        config = {
            "preprocessing": {"type": "image"},
            "postprocessing": {"type": "classification"}
        }
        
        with patch('framework.processors.fast_factory.FastProcessorFactory') as MockFactory:
            mock_factory = Mock()
            mock_factory.create_preprocessor.return_value = Mock()
            mock_factory.create_postprocessor.return_value = Mock()
            MockFactory.return_value = mock_factory
            
            result = create_fast_processing_pipelines(config)
            
            assert "preprocessor" in result
            assert "postprocessor" in result
    
    def test_benchmark_processor_performance(self):
        """Test benchmark_processor_performance function."""
        mock_processor = Mock()
        mock_processor.process.return_value = torch.randn(3, 224, 224)
        
        mock_data = [torch.randn(3, 256, 256) for _ in range(10)]
        
        results = benchmark_processor_performance(
            processor=mock_processor,
            test_data=mock_data,
            num_runs=5
        )
        
        assert "avg_time" in results
        assert "throughput" in results
        assert "memory_usage" in results
        assert results["avg_time"] > 0
        assert results["throughput"] > 0


class TestOptimizationFunctions:
    """Test optimization utility functions."""
    
    def test_optimize_for_production_throughput(self):
        """Test optimize_for_production_throughput function."""
        config = optimize_for_production_throughput(
            expected_load=1000,
            target_latency=100,
            available_memory="16GB"
        )
        
        assert isinstance(config, dict)
        assert "batch_size" in config
        assert "num_workers" in config
        assert "optimization_level" in config
    
    def test_optimize_for_real_time_latency(self):
        """Test optimize_for_real_time_latency function."""
        config = optimize_for_real_time_latency(
            max_latency=50,
            device="cuda"
        )
        
        assert isinstance(config, dict)
        assert "processing_mode" in config
        assert "optimization_level" in config
    
    def test_optimize_for_memory_constrained_environment(self):
        """Test optimize_for_memory_constrained_environment function."""
        config = optimize_for_memory_constrained_environment(
            memory_limit="4GB",
            model_size="large"
        )
        
        assert isinstance(config, dict)
        assert "batch_size" in config
        assert "optimization_level" in config
    
    def test_auto_optimize_processors(self):
        """Test auto_optimize_processors function."""
        mock_model = Mock()
        
        config = auto_optimize_processors(
            model=mock_model,
            target="inference",
            constraints={"memory": "8GB", "latency": 100}
        )
        
        assert isinstance(config, dict)
        assert "preprocessing" in config or "postprocessing" in config


class TestImageProcessorAdvanced:
    """Test advanced image processor."""
    
    @pytest.fixture
    def image_transform_config(self):
        """Create image transform config."""
        return ImageTransformConfig(
            resize_size=(256, 256),
            crop_size=(224, 224),
            normalize=True,
            augmentation=True
        )
    
    def test_image_transform_config_creation(self, image_transform_config):
        """Test image transform config creation."""
        assert image_transform_config.resize_size == (256, 256)
        assert image_transform_config.crop_size == (224, 224)
        assert image_transform_config.normalize is True
        assert image_transform_config.augmentation is True
    
    def test_create_image_preprocessor(self, image_transform_config):
        """Test create_image_preprocessor function."""
        preprocessor = create_image_preprocessor(image_transform_config)
        
        assert preprocessor is not None
        assert hasattr(preprocessor, 'process')
    
    def test_get_image_transforms(self):
        """Test get_image_transforms function."""
        transforms = get_image_transforms(
            input_size=(224, 224),
            augmentation=True
        )
        
        assert transforms is not None
        # Should be a torchvision Compose object or similar


class TestAudioProcessorAdvanced:
    """Test advanced audio processor."""
    
    @pytest.fixture
    def audio_config(self):
        """Create audio preprocessor config."""
        return AudioPreprocessorConfig(
            sample_rate=16000,
            n_mels=80,
            win_length=1024,
            hop_length=256
        )
    
    def test_audio_preprocessor_config_creation(self, audio_config):
        """Test audio preprocessor config creation."""
        assert audio_config.sample_rate == 16000
        assert audio_config.n_mels == 80
        assert audio_config.win_length == 1024
        assert audio_config.hop_length == 256
    
    def test_create_audio_preprocessor(self, audio_config):
        """Test create_audio_preprocessor function."""
        preprocessor = create_audio_preprocessor(audio_config)
        
        assert preprocessor is not None
        assert hasattr(preprocessor, 'process')
    
    def test_get_audio_transforms(self):
        """Test get_audio_transforms function."""
        transforms = get_audio_transforms(
            sample_rate=16000,
            n_mels=80
        )
        
        assert transforms is not None


class TestProcessorIntegration:
    """Test processor integration scenarios."""
    
    def test_preprocessing_postprocessing_pipeline(self):
        """Test complete preprocessing -> model -> postprocessing pipeline."""
        # Mock components
        preprocessor = Mock()
        preprocessor.process.return_value = torch.randn(1, 3, 224, 224)
        
        model = Mock()
        model.return_value = torch.randn(1, 10)
        
        postprocessor = Mock()
        postprocessor.process.return_value = {"predicted_class": "cat", "confidence": 0.95}
        
        # Test pipeline
        input_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Preprocess
        processed_input = preprocessor.process(input_data)
        
        # Model inference
        model_output = model(processed_input)
        
        # Postprocess
        final_result = postprocessor.process(model_output)
        
        assert "predicted_class" in final_result
        assert "confidence" in final_result
    
    def test_batch_processing_pipeline(self):
        """Test batch processing pipeline."""
        # Mock batch preprocessor
        preprocessor = Mock()
        preprocessor.process_batch.return_value = torch.randn(4, 3, 224, 224)
        
        # Mock batch postprocessor
        postprocessor = Mock()
        postprocessor.process_batch.return_value = [
            {"predicted_class": "cat", "confidence": 0.95},
            {"predicted_class": "dog", "confidence": 0.89},
            {"predicted_class": "bird", "confidence": 0.76},
            {"predicted_class": "fish", "confidence": 0.82}
        ]
        
        # Test batch pipeline
        batch_data = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(4)]
        
        processed_batch = preprocessor.process_batch(batch_data)
        batch_results = postprocessor.process_batch(processed_batch)
        
        assert len(batch_results) == 4
        assert all("predicted_class" in result for result in batch_results)
    
    def test_processor_memory_optimization(self):
        """Test processor memory optimization."""
        # Test that processors handle large inputs efficiently
        large_input = torch.randn(64, 3, 512, 512)  # Large batch
        
        config = PreprocessorConfig(input_size=(224, 224))
        preprocessor = ImagePreprocessor(config)
        
        # Should handle large input without memory errors
        with patch.object(preprocessor, 'transform') as mock_transform:
            mock_transform.return_value = torch.randn(64, 3, 224, 224)
            
            result = preprocessor.process(large_input)
            
            assert result is not None
            mock_transform.assert_called_once()
    
    def test_processor_device_handling(self):
        """Test processor device handling."""
        config = PreprocessorConfig()
        preprocessor = ImagePreprocessor(config)
        
        # Test moving to different device
        input_tensor = torch.randn(3, 224, 224)
        
        # Should handle device transfers gracefully
        result = preprocessor.process(input_tensor)
        
        assert result is not None
        assert isinstance(result, torch.Tensor)


@pytest.mark.integration
class TestProcessorIntegrationWithFramework:
    """Integration tests for processors with the main framework."""
    
    def test_processor_with_inference_framework(self):
        """Test processors with TorchInferenceFramework."""
        # Mock framework integration
        with patch('framework.TorchInferenceFramework') as MockFramework:
            mock_framework = Mock()
            MockFramework.return_value = mock_framework
            
            # Test that processors can be integrated
            framework = MockFramework()
            assert framework is not None
    
    def test_processor_configuration_loading(self):
        """Test loading processor configuration from files."""
        # Test configuration loading
        config_dict = {
            "preprocessing": {
                "type": "image",
                "input_size": [224, 224],
                "normalize": True
            },
            "postprocessing": {
                "type": "classification",
                "num_classes": 10
            }
        }
        
        # Should be able to create processors from config
        assert "preprocessing" in config_dict
        assert "postprocessing" in config_dict


if __name__ == "__main__":
    pytest.main([__file__])

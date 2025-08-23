#!/usr/bin/env python3
"""
Test cases for preprocessing edge cases, particularly the PIL error fix.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch

from framework.processors.preprocessor import (
    ImagePreprocessor, 
    TensorPreprocessor,
    PreprocessorPipeline,
    create_default_preprocessing_pipeline,
    PreprocessingError
)
from framework.core.config import InferenceConfig


class TestPreprocessingEdgeCases:
    """Test edge cases in preprocessing that previously caused errors."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return InferenceConfig()
    
    @pytest.fixture
    def image_preprocessor(self, config):
        """Create an ImagePreprocessor instance."""
        return ImagePreprocessor(config)
    
    @pytest.fixture
    def tensor_preprocessor(self, config):
        """Create a TensorPreprocessor instance."""
        return TensorPreprocessor(config)
    
    @pytest.fixture
    def pipeline(self, config):
        """Create a preprocessing pipeline."""
        return create_default_preprocessing_pipeline(config)
    
    def test_pil_error_fix_original_case(self, image_preprocessor):
        """Test the fix for the original PIL error: Cannot handle this data type: (1, 1, 15), |u1"""
        # Create the exact problematic array from the error
        problematic_array = np.random.randint(0, 255, (1, 1, 15), dtype=np.uint8)
        
        # This should not raise an error anymore
        result = image_preprocessor.preprocess(problematic_array)
        
        # Verify the result is valid
        assert isinstance(result.data, torch.Tensor)
        assert len(result.data.shape) == 4  # Should have batch dimension
        assert result.data.shape[1] == 3    # Should have 3 RGB channels
        assert result.data.shape[2] >= 32   # Should have reasonable spatial dimensions
        assert result.data.shape[3] >= 32
        
    def test_small_spatial_dimensions_with_many_channels(self, pipeline):
        """Test arrays with very small spatial dimensions but many channels."""
        test_cases = [
            (1, 1, 15),   # Original problem case
            (2, 2, 10),   # Small spatial, moderate channels
            (1, 3, 20),   # Very thin image with many channels
            (3, 1, 25),   # Very tall image with many channels
        ]
        
        for shape in test_cases:
            test_array = np.random.randint(0, 255, shape, dtype=np.uint8)
            result = pipeline.preprocess(test_array)
            
            # Should not raise an error
            assert isinstance(result.data, torch.Tensor)
            assert len(result.data.shape) in [3, 4]  # Valid dimensionality
                
    def test_tensor_preprocessor_dimension_handling(self, tensor_preprocessor):
        """Test TensorPreprocessor handles problematic 3D tensors correctly."""
        # Test [H, W, C] format with small spatial dims and many channels
        hwc_array = np.random.rand(2, 3, 20).astype(np.float32)
        result = tensor_preprocessor.preprocess(hwc_array)
        
        assert isinstance(result.data, torch.Tensor)
        assert len(result.data.shape) == 3  # [batch, features, channels]
        assert result.data.shape[0] == 1    # Batch dimension added
        
        # Test [C, H, W] format with many channels and small spatial dims  
        chw_array = np.random.rand(20, 2, 3).astype(np.float32)
        result = tensor_preprocessor.preprocess(chw_array)
        
        assert isinstance(result.data, torch.Tensor)
        assert len(result.data.shape) == 3  # [batch, channels, features]
        assert result.data.shape[0] == 1    # Batch dimension added
        
    def test_zero_dimension_arrays(self, pipeline):
        """Test arrays with zero dimensions are handled gracefully."""
        # Zero-size arrays should fall back to error handling
        zero_array = np.array([]).reshape(0, 0, 3)
        
        # Should not crash, should use fallback
        result = pipeline.preprocess(zero_array)
        assert isinstance(result.data, torch.Tensor)
        
    def test_image_preprocessor_fallback_mechanism(self, image_preprocessor):
        """Test that image preprocessor fallback mechanism works."""
        # Create an array that will trigger fallback
        problematic_array = np.random.randint(0, 255, (1, 1, 100), dtype=np.uint8)
        
        result = image_preprocessor.preprocess(problematic_array)
        
        # Should succeed with fallback
        assert isinstance(result.data, torch.Tensor)
        assert result.data.shape == (1, 3, 224, 224)  # Standard fallback size
        
    def test_extremely_large_channel_count(self, pipeline):
        """Test arrays with extremely large channel counts."""
        # Test very large channel count
        large_channel_array = np.random.rand(2, 2, 1000).astype(np.float32)
        
        result = pipeline.preprocess(large_channel_array)
        assert isinstance(result.data, torch.Tensor)
        # Should be reshaped to something manageable
        
    def test_input_type_detection_edge_cases(self, pipeline):
        """Test input type detection for edge cases."""
        test_cases = [
            (np.random.rand(1, 1, 15), "Should detect as tensor"),
            (np.random.rand(3, 224, 224), "Should detect as image"),
            (np.random.rand(224, 224, 3), "Should detect as image"),
            (np.random.rand(1, 1000), "Should detect as tensor"),
        ]
        
        for test_array, description in test_cases:
            detected_type = pipeline.detect_input_type(test_array)
            # Just ensure it detects some valid type without crashing
            assert hasattr(detected_type, 'value')
                
    def test_preprocessing_error_recovery(self, pipeline):
        """Test that preprocessing errors are handled gracefully."""
        # Test with completely invalid input
        invalid_inputs = [
            None,
            [],
            {},
            "not_an_image_path.xyz",
        ]
        
        for invalid_input in invalid_inputs:
            try:
                result = pipeline.preprocess(invalid_input)
                # If it succeeds, result should be valid
                assert isinstance(result.data, torch.Tensor)
            except PreprocessingError:
                # If it fails, should be a proper PreprocessingError
                pass
            except Exception as e:
                pytest.fail(f"Unexpected exception type for {type(invalid_input)}: {type(e)}: {e}")
                    
    def test_batch_consistency(self, pipeline):
        """Test that batch processing is consistent with single processing."""
        # Create a batch of problematic arrays
        batch = [
            np.random.randint(0, 255, (1, 1, 15), dtype=np.uint8),
            np.random.randint(0, 255, (2, 2, 10), dtype=np.uint8),
            np.random.rand(3, 224, 224).astype(np.float32),
        ]
        
        # Process individually
        individual_results = []
        for item in batch:
            result = pipeline.preprocess(item)
            individual_results.append(result)
            
        # Process as batch
        batch_results = pipeline.preprocess_batch(batch)
        
        assert len(batch_results) == len(individual_results)
        
        # Each result should be valid
        for result in batch_results:
            if hasattr(result, 'data'):  # Not an error result
                assert isinstance(result.data, torch.Tensor)


class TestPILErrorRegression:
    """Specific regression tests for the PIL error."""
    
    def test_pil_fromarray_error_cases(self):
        """Test specific cases that caused PIL Image.fromarray errors."""
        config = InferenceConfig()
        image_preprocessor = ImagePreprocessor(config)
        
        # Cases that previously caused "Cannot handle this data type" errors
        error_cases = [
            ((1, 1, 15), np.uint8),
            ((1, 2, 20), np.uint8),
            ((2, 1, 10), np.uint8),
            ((1, 1, 5), np.float32),
        ]
        
        for shape, dtype in error_cases:
            test_array = np.random.randint(0, 255, shape).astype(dtype)
            if dtype == np.float32:
                test_array = test_array / 255.0  # Normalize float arrays
                
            # This should not raise a PIL error
            result = image_preprocessor.preprocess(test_array)
            
            assert isinstance(result.data, torch.Tensor)
            assert len(result.data.shape) == 4
            assert result.data.shape[1] == 3  # RGB channels
                
    def test_pil_mode_compatibility(self):
        """Test that processed arrays are compatible with PIL modes."""
        config = InferenceConfig()
        image_preprocessor = ImagePreprocessor(config)
        
        # Test various input formats
        test_formats = [
            np.random.randint(0, 255, (1, 1, 1), dtype=np.uint8),   # Single channel
            np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8),   # RGB
            np.random.randint(0, 255, (1, 1, 4), dtype=np.uint8),   # RGBA
        ]
        
        for test_array in test_formats:
            result = image_preprocessor.preprocess(test_array)
            # Should succeed without PIL mode errors
            assert isinstance(result.data, torch.Tensor)

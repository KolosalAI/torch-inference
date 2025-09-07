"""
Unit tests for secure image model.

Tests the SecureImageModel functionality including secure processing,
security statistics, and integration with the secure processor.
"""

import pytest
import numpy as np
import torch
import tempfile
import io
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from framework.models.secure_image_model import (
    SecureImageModel,
    create_secure_image_model
)
from framework.processors.image.secure_image_processor import SecurityLevel


class TestSecureImageModel:
    """Test SecureImageModel functionality."""
    
    @pytest.fixture
    def secure_model(self):
        """Create a secure image model instance."""
        return SecureImageModel(
            base_model_name="test_model",
            security_level=SecurityLevel.MEDIUM
        )
    
    @pytest.fixture
    def test_image_data(self):
        """Create test image data."""
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    def test_model_initialization(self):
        """Test secure image model initialization."""
        model = SecureImageModel(
            base_model_name="test_model",
            security_level=SecurityLevel.HIGH
        )
        
        assert model.base_model_name == "test_model"
        assert model.security_level == SecurityLevel.HIGH
        assert model.model_name == "SecureImageModel_test_model"
        assert hasattr(model, 'secure_preprocessor')
        assert hasattr(model, '_security_stats')
        
        # Check initial security stats
        stats = model._security_stats
        assert stats['total_processed'] == 0
        assert stats['threats_detected'] == 0
        assert stats['sanitizations_applied'] == 0
    
    def test_process_image_secure_success(self, secure_model, test_image_data):
        """Test successful secure image processing."""
        result = secure_model.process_image_secure(
            test_image_data,
            enable_sanitization=True,
            enable_adversarial_detection=True,
            return_confidence_scores=True
        )
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'processed_image' in result
        assert 'threats_detected' in result
        assert 'threats_mitigated' in result
        assert 'confidence_scores' in result
        assert 'processing_time' in result
        assert 'model_name' in result
        assert 'security_level' in result
        
        assert result['model_name'] == secure_model.model_name
        assert result['security_level'] == secure_model.security_level.name
    
    def test_process_image_updates_stats(self, secure_model, test_image_data):
        """Test that processing updates security statistics."""
        initial_total = secure_model._security_stats['total_processed']
        
        result = secure_model.process_image_secure(test_image_data)
        
        # Stats should be updated
        assert secure_model._security_stats['total_processed'] == initial_total + 1
    
    def test_process_image_with_threats(self, secure_model):
        """Test processing image that triggers threat detection."""
        # Create suspicious image data
        suspicious_data = b'FAKE_SUSPICIOUS_IMAGE_DATA' * 1000
        
        result = secure_model.process_image_secure(suspicious_data)
        
        # Should handle the suspicious data (may succeed or fail depending on validation)
        assert isinstance(result, dict)
        assert 'success' in result
        
        if not result['success']:
            assert 'error' in result
            assert 'threats_detected' in result
    
    def test_validate_image_security(self, secure_model, test_image_data):
        """Test image security validation."""
        result = secure_model.validate_image_security(
            test_image_data, 
            filename='test.png'
        )
        
        assert isinstance(result, dict)
        assert 'is_safe' in result
        assert 'threats_detected' in result
        assert 'processing_time' in result
    
    def test_get_security_stats(self, secure_model):
        """Test security statistics retrieval."""
        stats = secure_model.get_security_stats()
        
        assert isinstance(stats, dict)
        assert 'total_processed' in stats
        assert 'threats_detected' in stats
        assert 'sanitizations_applied' in stats
        assert 'preprocessor_config' in stats
        assert 'configuration' in stats
        
        # Check configuration details
        config = stats['configuration']
        assert config['base_model'] == secure_model.base_model_name
        assert config['security_level'] == secure_model.security_level.name
        assert config['model_name'] == secure_model.model_name
    
    def test_set_security_level(self, secure_model):
        """Test security level modification."""
        original_level = secure_model.security_level
        new_level = SecurityLevel.HIGH
        
        secure_model.set_security_level(new_level)
        
        assert secure_model.security_level == new_level
        assert secure_model.secure_preprocessor.security_level == new_level
        assert secure_model.secure_preprocessor.config.security_level == new_level
    
    def test_model_info_property(self, secure_model):
        """Test model_info property."""
        info = secure_model.model_info
        
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'base_model' in info
        assert 'security_level' in info
        assert 'security_features' in info
        assert 'security_stats' in info
        
        # Check security features
        features = info['security_features']
        assert 'adversarial_detection' in features
        assert 'format_validation' in features
        assert 'sanitization' in features
        assert 'threat_monitoring' in features
    
    def test_error_handling(self, secure_model):
        """Test error handling with invalid input."""
        # Test with invalid image data
        result = secure_model.process_image_secure(b'invalid_data')
        
        assert result['success'] is False
        assert 'error' in result
        assert 'processing_error' in result['threats_detected']
    
    def test_multiple_processing_updates_stats(self, secure_model, test_image_data):
        """Test that multiple processing operations update stats correctly."""
        initial_stats = secure_model._security_stats.copy()
        
        # Process multiple images
        num_processes = 3
        for i in range(num_processes):
            secure_model.process_image_secure(test_image_data)
        
        final_stats = secure_model._security_stats
        assert final_stats['total_processed'] == initial_stats['total_processed'] + num_processes
    
    def test_different_security_levels_affect_processing(self, test_image_data):
        """Test that different security levels affect processing behavior."""
        models = {}
        results = {}
        
        # Create models with different security levels
        for level in [SecurityLevel.LOW, SecurityLevel.HIGH]:
            models[level] = SecureImageModel(
                base_model_name="test",
                security_level=level
            )
            results[level] = models[level].process_image_secure(test_image_data)
        
        # Both should process successfully but potentially with different mitigations
        for level in [SecurityLevel.LOW, SecurityLevel.HIGH]:
            assert isinstance(results[level], dict)
            assert 'security_level' in results[level]
            assert results[level]['security_level'] == level.name


class TestSecureImageModelCreation:
    """Test secure image model creation functions."""
    
    def test_create_secure_image_model(self):
        """Test create_secure_image_model function."""
        model = create_secure_image_model(
            base_model_name="test_model",
            security_level=SecurityLevel.HIGH
        )
        
        assert isinstance(model, SecureImageModel)
        assert model.base_model_name == "test_model"
        assert model.security_level == SecurityLevel.HIGH
    
    def test_create_secure_image_model_defaults(self):
        """Test create_secure_image_model with default parameters."""
        model = create_secure_image_model()
        
        assert isinstance(model, SecureImageModel)
        assert model.base_model_name == "default"
        assert model.security_level == SecurityLevel.MEDIUM


class TestSecureImageModelIntegration:
    """Integration tests for SecureImageModel with various scenarios."""
    
    def test_end_to_end_secure_processing(self):
        """Test complete end-to-end secure processing workflow."""
        # Create test image
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        # Create secure model
        model = SecureImageModel(
            base_model_name="integration_test",
            security_level=SecurityLevel.MEDIUM
        )
        
        # Process image
        result = model.process_image_secure(
            image_data,
            enable_sanitization=True,
            enable_adversarial_detection=True
        )
        
        # Validate result
        assert result['success'] is True
        assert 'processed_image' in result
        assert isinstance(result['processed_image'], bytes)
        
        # Validate security analysis
        assert 'security_analysis' in result
        security_analysis = result['security_analysis']
        assert 'is_safe' in security_analysis
        assert 'threats_detected' in security_analysis
        
        # Check that stats were updated
        stats = model.get_security_stats()
        assert stats['total_processed'] >= 1
    
    def test_security_level_escalation(self):
        """Test security level escalation scenario."""
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        # Start with low security
        model = SecureImageModel(security_level=SecurityLevel.LOW)
        
        # Process image
        low_result = model.process_image_secure(image_data)
        
        # Escalate security level
        model.set_security_level(SecurityLevel.MAXIMUM)
        
        # Process same image with higher security
        high_result = model.process_image_secure(image_data)
        
        # Both should process but potentially with different characteristics
        assert low_result['security_level'] == SecurityLevel.LOW.name
        assert high_result['security_level'] == SecurityLevel.MAXIMUM.name
    
    def test_batch_processing_stats(self):
        """Test statistics accumulation over batch processing."""
        model = SecureImageModel(security_level=SecurityLevel.MEDIUM)
        
        # Process multiple images
        batch_size = 5
        for i in range(batch_size):
            # Create different test images
            img_array = np.random.randint(0, 256, (50 + i * 10, 50 + i * 10, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            
            model.process_image_secure(buffer.getvalue())
        
        # Check accumulated stats
        stats = model.get_security_stats()
        assert stats['total_processed'] == batch_size
    
    def test_validation_only_workflow(self):
        """Test validation-only workflow."""
        model = SecureImageModel(security_level=SecurityLevel.HIGH)
        
        # Create test image
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        # Validate only
        validation_result = model.validate_image_security(image_data, 'test.png')
        
        # Should not affect processing stats
        initial_stats = model.get_security_stats()
        assert initial_stats['total_processed'] == 0
        
        # But should return validation results
        assert isinstance(validation_result, dict)
        assert 'is_safe' in validation_result
    
    def test_error_recovery(self):
        """Test error recovery and logging."""
        model = SecureImageModel(security_level=SecurityLevel.MEDIUM)
        
        # Process valid image first
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        valid_image = buffer.getvalue()
        
        valid_result = model.process_image_secure(valid_image)
        assert valid_result['success'] is True
        
        # Then process invalid data
        invalid_result = model.process_image_secure(b'invalid_data')
        assert invalid_result['success'] is False
        
        # Model should still be functional
        another_valid_result = model.process_image_secure(valid_image)
        assert another_valid_result['success'] is True
        
        # Stats should reflect all processing attempts
        stats = model.get_security_stats()
        assert stats['total_processed'] == 3  # 2 valid + 1 invalid


if __name__ == '__main__':
    pytest.main([__file__])

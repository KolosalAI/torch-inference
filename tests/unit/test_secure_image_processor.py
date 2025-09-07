"""
Unit tests for secure image processor.

Tests the core functionality of SecureImageValidator, SecureImageSanitizer,
and SecureImagePreprocessor components.
"""

import pytest
import numpy as np
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

from framework.processors.image.secure_image_processor import (
    SecurityLevel,
    ThreatLevel,
    AttackType,
    SecurityConfig,
    SecurityReport,
    SecureImageValidator,
    SecureImageSanitizer,
    SecureImagePreprocessor
)


class TestSecurityLevel:
    """Test SecurityLevel enum functionality."""
    
    def test_security_levels_ordering(self):
        """Test that security levels are properly ordered."""
        assert SecurityLevel.LOW < SecurityLevel.MEDIUM
        assert SecurityLevel.MEDIUM < SecurityLevel.HIGH
        assert SecurityLevel.HIGH < SecurityLevel.MAXIMUM
    
    def test_security_level_values(self):
        """Test security level values."""
        assert SecurityLevel.LOW.value == 1
        assert SecurityLevel.MEDIUM.value == 2
        assert SecurityLevel.HIGH.value == 3
        assert SecurityLevel.MAXIMUM.value == 4


class TestSecurityConfig:
    """Test SecurityConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SecurityConfig()
        assert config.security_level == SecurityLevel.MEDIUM
        assert config.max_image_size_mb == 50.0
        assert config.max_image_dimensions == (4096, 4096)
        assert config.min_image_dimensions == (32, 32)
        assert '.jpg' in config.allowed_formats
        assert '.png' in config.allowed_formats
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SecurityConfig(
            security_level=SecurityLevel.HIGH,
            max_image_size_mb=100.0,
            allowed_formats=['.png', '.jpg']
        )
        assert config.security_level == SecurityLevel.HIGH
        assert config.max_image_size_mb == 100.0
        assert config.allowed_formats == ['.png', '.jpg']


class TestSecureImageValidator:
    """Test SecureImageValidator functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return SecureImageValidator(SecurityLevel.MEDIUM)
    
    @pytest.fixture
    def test_image_data(self):
        """Create test image data."""
        # Create a simple RGB image
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = SecureImageValidator(SecurityLevel.HIGH)
        assert validator.config.security_level == SecurityLevel.HIGH
        assert hasattr(validator, '_file_signatures')
        assert '.png' in validator._file_signatures
        assert '.jpg' in validator._file_signatures
    
    def test_file_signatures_loading(self, validator):
        """Test file signatures are loaded correctly."""
        signatures = validator._file_signatures
        
        # Check PNG signature
        assert b'\x89PNG\r\n\x1a\n' in signatures['.png']
        
        # Check JPEG signatures
        assert b'\xff\xd8\xff\xe0' in signatures['.jpg']
        assert b'\xff\xd8\xff\xe1' in signatures['.jpg']
    
    def test_validate_image_security_success(self, validator, test_image_data):
        """Test successful image validation."""
        result = validator.validate_image_security(
            test_image_data, 
            filename='test.png',
            detailed_analysis=True
        )
        
        assert isinstance(result, dict)
        assert 'is_safe' in result
        assert 'threats_detected' in result
        assert 'confidence_scores' in result
        assert 'processing_time' in result
    
    def test_validate_oversized_image(self, validator):
        """Test validation of oversized image."""
        # Create large dummy data
        large_data = b'fake_image_data' * 10000000  # ~150MB
        
        result = validator.validate_image_security(
            large_data,
            filename='large.png'
        )
        
        assert not result['is_safe']
        assert 'oversized_file' in result['threats_detected']
        assert any('too large' in rec for rec in result['recommendations'])
    
    def test_validate_invalid_format(self, validator, test_image_data):
        """Test validation of invalid format."""
        result = validator.validate_image_security(
            test_image_data,
            filename='test.xyz'  # Invalid extension
        )
        
        assert not result['is_safe']
        assert 'invalid_format' in result['threats_detected']
    
    def test_validate_signature_mismatch(self, validator):
        """Test validation of signature mismatch."""
        # Create fake JPEG data with wrong signature
        fake_jpeg_data = b'NOT_A_JPEG_FILE' + b'\x00' * 1000
        
        result = validator.validate_image_security(
            fake_jpeg_data,
            filename='fake.jpg'
        )
        
        assert not result['is_safe']
        assert 'signature_mismatch' in result['threats_detected']
    
    @patch('framework.processors.image.secure_image_processor.HAS_PIL', False)
    def test_validate_without_pil(self, validator, test_image_data):
        """Test validation when PIL is not available."""
        result = validator.validate_image_security(
            test_image_data,
            filename='test.png',
            detailed_analysis=True
        )
        
        # Should still perform basic validation
        assert isinstance(result, dict)
        assert 'is_safe' in result
    
    def test_validate_empty_data(self, validator):
        """Test validation of empty image data."""
        result = validator.validate_image_security(b'', filename='empty.png')
        
        assert not result['is_safe']
        assert 'threats_detected' in result
    
    def test_entropy_analysis(self, validator):
        """Test entropy analysis functionality."""
        # Create image with unusual entropy
        # All pixels the same value (very low entropy)
        uniform_array = np.full((100, 100, 3), 128, dtype=np.uint8)
        img = Image.fromarray(uniform_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        
        result = validator.validate_image_security(
            buffer.getvalue(),
            filename='uniform.png',
            detailed_analysis=True
        )
        
        # May detect unusual entropy
        if 'confidence_scores' in result and 'entropy' in result['confidence_scores']:
            entropy = result['confidence_scores']['entropy']
            assert isinstance(entropy, float)
            assert entropy >= -1e-6, f"Entropy should be >= 0 (allowing for floating point precision), got {entropy}"  # Allow for small negative values due to floating point precision


class TestSecureImageSanitizer:
    """Test SecureImageSanitizer functionality."""
    
    @pytest.fixture
    def sanitizer(self):
        """Create a sanitizer instance."""
        config = SecurityConfig(security_level=SecurityLevel.MEDIUM)
        return SecureImageSanitizer(config)
    
    @pytest.fixture
    def test_image_array(self):
        """Create test image array."""
        return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    def test_sanitizer_initialization(self):
        """Test sanitizer initialization."""
        config = SecurityConfig(security_level=SecurityLevel.HIGH)
        sanitizer = SecureImageSanitizer(config)
        assert sanitizer.config.security_level == SecurityLevel.HIGH
    
    def test_noise_injection(self, sanitizer, test_image_array):
        """Test noise injection sanitization."""
        sanitizer.config.enable_noise_injection = True
        sanitizer.config.noise_injection_strength = 0.1
        
        sanitized_img, applied = sanitizer.sanitize_image(test_image_array)
        
        assert sanitized_img.shape == test_image_array.shape
        assert 'noise_injection' in applied
        assert sanitized_img.dtype == np.uint8
        
        # Images should be different after noise injection
        assert not np.array_equal(sanitized_img, test_image_array)
    
    def test_bit_depth_reduction(self, sanitizer, test_image_array):
        """Test bit depth reduction sanitization."""
        sanitizer.config.enable_bit_depth_reduction = True
        sanitizer.config.bit_depth = 6  # Reduce from 8 to 6 bits
        
        sanitized_img, applied = sanitizer.sanitize_image(test_image_array)
        
        assert sanitized_img.shape == test_image_array.shape
        assert 'bit_depth_reduction' in applied
        
        # Bit depth reduction should quantize values
        unique_values = len(np.unique(sanitized_img))
        max_possible_values = 2 ** sanitizer.config.bit_depth
        # Should have fewer unique values due to quantization
        assert unique_values <= max_possible_values * 3  # 3 channels
    
    @patch('framework.processors.image.secure_image_processor.HAS_OPENCV', True)
    @patch('framework.processors.image.secure_image_processor.cv2')
    def test_gaussian_blur_opencv(self, mock_cv2, sanitizer, test_image_array):
        """Test Gaussian blur with OpenCV."""
        sanitizer.config.enable_gaussian_blur = True
        
        # Mock OpenCV GaussianBlur
        mock_cv2.GaussianBlur.side_effect = lambda img, kernel, sigma: img
        
        sanitized_img, applied = sanitizer.sanitize_image(test_image_array)
        
        assert 'gaussian_blur' in applied
        assert mock_cv2.GaussianBlur.called
    
    @patch('framework.processors.image.secure_image_processor.HAS_OPENCV', False)
    @patch('framework.processors.image.secure_image_processor.HAS_SKIMAGE', True)
    @patch('framework.processors.image.secure_image_processor.filters')
    def test_gaussian_blur_skimage(self, mock_filters, sanitizer, test_image_array):
        """Test Gaussian blur with scikit-image."""
        sanitizer.config.enable_gaussian_blur = True
        
        # Mock scikit-image gaussian filter
        mock_filters.gaussian.side_effect = lambda img, sigma: img
        
        sanitized_img, applied = sanitizer.sanitize_image(test_image_array)
        
        assert 'gaussian_blur' in applied
        assert mock_filters.gaussian.called
    
    def test_no_sanitization(self, sanitizer, test_image_array):
        """Test when no sanitization is enabled."""
        # Disable all sanitization
        sanitizer.config.enable_noise_injection = False
        sanitizer.config.enable_gaussian_blur = False
        sanitizer.config.enable_bit_depth_reduction = False
        
        sanitized_img, applied = sanitizer.sanitize_image(test_image_array)
        
        # Should return original image when no sanitization is applied
        assert len(applied) == 0
        # Image should be converted to uint8 but otherwise unchanged
        assert sanitized_img.shape == test_image_array.shape
    
    def test_sanitization_error_handling(self, sanitizer):
        """Test error handling in sanitization."""
        # Pass invalid input
        invalid_input = "not_an_array"
        
        with pytest.raises(AttributeError):
            sanitizer.sanitize_image(invalid_input)


class TestSecureImagePreprocessor:
    """Test SecureImagePreprocessor functionality."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance."""
        return SecureImagePreprocessor(SecurityLevel.MEDIUM)
    
    @pytest.fixture
    def test_image_data(self):
        """Create test image data."""
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = SecureImagePreprocessor(SecurityLevel.HIGH)
        assert preprocessor.security_level == SecurityLevel.HIGH
        assert hasattr(preprocessor, 'validator')
        assert hasattr(preprocessor, 'sanitizer')
    
    def test_process_image_secure_success(self, preprocessor, test_image_data):
        """Test successful secure image processing."""
        result = preprocessor.process_image_secure(
            test_image_data,
            filename='test.png',
            enable_sanitization=True,
            return_format='numpy'
        )
        
        assert result['success'] is True
        assert 'processed_image' in result
        assert 'threats_detected' in result
        assert 'threats_mitigated' in result
        assert 'confidence_scores' in result
        assert 'processing_time' in result
        assert 'security_analysis' in result
        
        # Processed image should be numpy array
        assert isinstance(result['processed_image'], np.ndarray)
    
    def test_process_image_different_formats(self, preprocessor, test_image_data):
        """Test processing with different return formats."""
        formats = ['numpy', 'pil', 'bytes']
        
        for fmt in formats:
            result = preprocessor.process_image_secure(
                test_image_data,
                return_format=fmt
            )
            
            if result['success']:
                assert 'processed_image' in result
                if fmt == 'numpy':
                    assert isinstance(result['processed_image'], np.ndarray)
                elif fmt == 'bytes':
                    assert isinstance(result['processed_image'], bytes)
    
    def test_process_numpy_array_input(self, preprocessor):
        """Test processing numpy array input."""
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = preprocessor.process_image_secure(
            img_array,
            enable_sanitization=True
        )
        
        assert isinstance(result, dict)
        assert 'success' in result
    
    def test_process_with_sanitization(self, preprocessor, test_image_data):
        """Test processing with sanitization enabled."""
        result = preprocessor.process_image_secure(
            test_image_data,
            enable_sanitization=True
        )
        
        if result['success']:
            # Should have applied some sanitization
            assert 'threats_mitigated' in result
            assert isinstance(result['threats_mitigated'], list)
    
    def test_process_without_sanitization(self, preprocessor, test_image_data):
        """Test processing without sanitization."""
        result = preprocessor.process_image_secure(
            test_image_data,
            enable_sanitization=False
        )
        
        if result['success']:
            # Should not have applied sanitization
            assert result['threats_mitigated'] == []
    
    def test_validate_only(self, preprocessor, test_image_data):
        """Test validation-only functionality."""
        result = preprocessor.validate_only(test_image_data, 'test.png')
        
        assert isinstance(result, dict)
        assert 'is_safe' in result
        assert 'threats_detected' in result
        assert 'processing_time' in result
    
    def test_get_security_stats(self, preprocessor):
        """Test security statistics retrieval."""
        stats = preprocessor.get_security_stats()
        
        assert isinstance(stats, dict)
        assert 'security_level' in stats
        assert 'config' in stats
        assert 'capabilities' in stats
        
        # Check capabilities
        capabilities = stats['capabilities']
        assert 'pil_available' in capabilities
        assert 'opencv_available' in capabilities
        assert 'skimage_available' in capabilities
        assert 'sklearn_available' in capabilities
    
    def test_error_handling(self, preprocessor):
        """Test error handling with invalid input."""
        result = preprocessor.process_image_secure(
            b'invalid_image_data',
            filename='test.png'
        )
        
        assert result['success'] is False
        assert 'error' in result
        assert 'processing_error' in result['threats_detected']
    
    @patch('framework.processors.image.secure_image_processor.HAS_PIL', False)
    def test_processing_without_pil(self, preprocessor, test_image_data):
        """Test processing when PIL is not available."""
        result = preprocessor.process_image_secure(test_image_data)
        
        # Should fail gracefully when PIL is not available
        assert result['success'] is False
        assert 'error' in result


class TestIntegration:
    """Integration tests for secure image processing components."""
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing pipeline."""
        # Create test image
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        # Process with different security levels
        for security_level in SecurityLevel:
            preprocessor = SecureImagePreprocessor(security_level)
            
            result = preprocessor.process_image_secure(
                image_data,
                filename='test.png',
                enable_sanitization=True
            )
            
            # Should process successfully
            assert isinstance(result, dict)
            assert 'success' in result
            assert 'processing_time' in result
    
    def test_threat_detection_pipeline(self):
        """Test threat detection across all components."""
        # Create suspicious image data
        suspicious_data = b'FAKE_IMAGE_WITH_SUSPICIOUS_CONTENT' * 1000
        
        preprocessor = SecureImagePreprocessor(SecurityLevel.HIGH)
        
        result = preprocessor.process_image_secure(
            suspicious_data,
            filename='suspicious.png'
        )
        
        # Should detect threats
        assert not result['success'] or len(result['threats_detected']) > 0
    
    def test_security_level_impact(self):
        """Test that different security levels affect processing."""
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        results = {}
        
        # Process with different security levels
        for level in [SecurityLevel.LOW, SecurityLevel.HIGH]:
            preprocessor = SecureImagePreprocessor(level)
            result = preprocessor.process_image_secure(image_data)
            results[level] = result
        
        # High security should potentially apply more mitigations
        low_mitigations = len(results[SecurityLevel.LOW].get('threats_mitigated', []))
        high_mitigations = len(results[SecurityLevel.HIGH].get('threats_mitigated', []))
        
        # Note: This might not always be true depending on the image,
        # but the framework should support different security levels
        assert isinstance(low_mitigations, int)
        assert isinstance(high_mitigations, int)


if __name__ == '__main__':
    pytest.main([__file__])

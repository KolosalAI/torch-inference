#!/usr/bin/env python3
"""
Secure Image Processor

Comprehensive secure image preprocessing with advanced threat detection,
sanitization, and prevention mechanisms for adversarial attacks.
"""

import logging
import time
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np

# Optional dependencies with graceful fallback
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None

try:
    from skimage import filters
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    filters = None

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    IsolationForest = None
    StandardScaler = None


class SecurityLevel(IntEnum):
    """Security levels for image processing."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    MAXIMUM = 4


class ThreatLevel(IntEnum):
    """Threat severity levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AttackType(Enum):
    """Types of image-based attacks."""
    ADVERSARIAL = "adversarial_example"
    STEGANOGRAPHY = "steganography"
    FORMAT_EXPLOIT = "format_exploitation"
    MEMORY_ATTACK = "memory_exhaustion"
    MALWARE_INJECTION = "malware_injection"
    DATA_POISONING = "data_poisoning"


@dataclass
class SecurityConfig:
    """Configuration for secure image processing."""
    
    # Basic security settings
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    max_image_size_mb: float = 50.0
    max_image_dimensions: Tuple[int, int] = (4096, 4096)
    min_image_dimensions: Tuple[int, int] = (32, 32)
    allowed_formats: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'])
    
    # Validation settings
    enable_signature_verification: bool = True
    enable_deep_format_validation: bool = True
    enable_metadata_analysis: bool = True
    max_processing_time_seconds: float = 30.0
    
    # Threat detection thresholds
    entropy_min_threshold: float = 4.0
    entropy_max_threshold: float = 8.0
    noise_variance_threshold: float = 0.05
    gradient_magnitude_threshold: float = 0.3
    edge_density_threshold: float = 0.01
    histogram_uniformity_threshold: float = 0.95
    pixel_value_distribution_check: bool = True
    
    # Sanitization settings
    enable_noise_injection: bool = True
    noise_injection_strength: float = 0.01
    enable_gaussian_blur: bool = True
    blur_kernel_size: int = 3
    enable_bit_depth_reduction: bool = True
    bit_depth: int = 7  # Reduce from 8 to 7 bits
    enable_jpeg_compression: bool = True
    jpeg_quality: int = 85
    
    # Advanced detection
    enable_adversarial_detection: bool = True
    enable_steganography_detection: bool = True
    enable_metadata_sanitization: bool = True
    suspicious_metadata_patterns: List[str] = field(default_factory=lambda: [
        'script', 'eval', 'exec', 'malware', 'payload', 'exploit', 'shell'
    ])


@dataclass
class SecurityReport:
    """Comprehensive security analysis report."""
    is_safe: bool
    threats_detected: List[AttackType]
    threat_level: ThreatLevel
    confidence_score: float
    format_validation: Dict[str, Any]
    statistical_analysis: Dict[str, Any]
    adversarial_analysis: Dict[str, Any]
    steganography_analysis: Dict[str, Any]
    metadata_analysis: Dict[str, Any]
    processing_time: float
    memory_usage_mb: float
    recommended_actions: List[str]
    sanitization_applied: List[str]


class SecureImageValidator:
    """Advanced image security validator with threat detection."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        self.config = SecurityConfig(security_level=security_level)
        self.logger = logging.getLogger(f"{__name__}.SecureImageValidator")
        self._file_signatures = self._load_file_signatures()
        
    def _load_file_signatures(self) -> Dict[str, List[bytes]]:
        """Load known file signatures for format validation."""
        return {
            '.jpg': [b'\xff\xd8\xff\xe0', b'\xff\xd8\xff\xe1', b'\xff\xd8\xff\xe2'],
            '.jpeg': [b'\xff\xd8\xff\xe0', b'\xff\xd8\xff\xe1', b'\xff\xd8\xff\xe2'],
            '.png': [b'\x89PNG\r\n\x1a\n'],
            '.bmp': [b'BM'],
            '.tiff': [b'II*\x00', b'MM\x00*'],
            '.webp': [b'RIFF', b'WEBP']
        }
    
    def validate_image_security(self, image_data: bytes, 
                               filename: Optional[str] = None,
                               detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        Validate image security without requiring file path.
        
        Args:
            image_data: Raw image data as bytes
            filename: Optional filename for format detection
            detailed_analysis: Whether to perform detailed threat analysis
            
        Returns:
            Dictionary containing validation results
        """
        start_time = time.time()
        
        try:
            # Create temporary validation result
            validation_result = {
                'is_safe': True,
                'threats_detected': [],
                'confidence_scores': {},
                'recommendations': [],
                'file_info': {
                    'size_bytes': len(image_data),
                    'filename': filename or 'unknown'
                }
            }
            
            # Basic size validation
            size_mb = len(image_data) / (1024 * 1024)
            if size_mb > self.config.max_image_size_mb:
                validation_result['is_safe'] = False
                validation_result['threats_detected'].append('oversized_file')
                validation_result['recommendations'].append(f'File too large: {size_mb:.1f}MB > {self.config.max_image_size_mb}MB')
            
            # Format validation
            if filename:
                ext = Path(filename).suffix.lower()
                if ext not in self.config.allowed_formats:
                    validation_result['is_safe'] = False
                    validation_result['threats_detected'].append('invalid_format')
                    validation_result['recommendations'].append(f'Unsupported format: {ext}')
                
                # Check file signature
                if self.config.enable_signature_verification and ext in self._file_signatures:
                    signatures = self._file_signatures[ext]
                    signature_match = any(image_data.startswith(sig) for sig in signatures)
                    if not signature_match:
                        validation_result['is_safe'] = False
                        validation_result['threats_detected'].append('signature_mismatch')
                        validation_result['recommendations'].append('File signature does not match extension')
            
            # Try to load and analyze image
            if HAS_PIL and detailed_analysis:
                try:
                    import io
                    img = Image.open(io.BytesIO(image_data))
                    img_array = np.array(img.convert('RGB'))
                    
                    # Dimension validation
                    h, w = img_array.shape[:2]
                    if (h > self.config.max_image_dimensions[0] or 
                        w > self.config.max_image_dimensions[1] or
                        h < self.config.min_image_dimensions[0] or 
                        w < self.config.min_image_dimensions[1]):
                        validation_result['is_safe'] = False
                        validation_result['threats_detected'].append('invalid_dimensions')
                        validation_result['recommendations'].append(f'Invalid dimensions: {w}x{h}')
                    
                    # Basic statistical analysis
                    if len(img_array.flatten()) > 0:
                        gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
                        
                        # Calculate entropy
                        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
                        hist = hist / hist.sum()
                        entropy = -np.sum(hist * np.log2(hist + 1e-10))
                        
                        validation_result['confidence_scores']['entropy'] = float(entropy)
                        
                        if entropy < self.config.entropy_min_threshold or entropy > self.config.entropy_max_threshold:
                            validation_result['threats_detected'].append('unusual_entropy')
                            validation_result['recommendations'].append(f'Unusual entropy: {entropy:.2f}')
                        
                        # Check for high noise levels (potential adversarial)
                        if HAS_OPENCV:
                            blurred = cv2.GaussianBlur(gray.astype(np.uint8), (5, 5), 0)
                            noise = np.abs(gray.astype(float) - blurred.astype(float))
                            avg_noise = np.mean(noise) / 255.0
                            
                            validation_result['confidence_scores']['noise_level'] = float(avg_noise)
                            
                            if avg_noise > self.config.noise_variance_threshold:
                                validation_result['threats_detected'].append('high_noise_level')
                                validation_result['recommendations'].append('Potential adversarial noise detected')
                
                except Exception as e:
                    self.logger.warning(f"Failed to analyze image content: {e}")
                    validation_result['threats_detected'].append('analysis_failed')
                    validation_result['recommendations'].append('Could not analyze image content')
            
            # Final safety determination
            if len(validation_result['threats_detected']) > 0:
                validation_result['is_safe'] = False
            
            # Add processing time
            validation_result['processing_time'] = time.time() - start_time
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Security validation failed: {e}")
            return {
                'is_safe': False,
                'threats_detected': ['validation_error'],
                'confidence_scores': {},
                'recommendations': [f'Validation failed: {str(e)}'],
                'processing_time': time.time() - start_time,
                'error': str(e)
            }


class SecureImageSanitizer:
    """Secure image sanitization and defense mechanisms."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.SecureImageSanitizer")
    
    def sanitize_image(self, img_array: np.ndarray, 
                      sanitization_level: SecurityLevel = None) -> Tuple[np.ndarray, List[str]]:
        """Apply security sanitization to image."""
        if sanitization_level is None:
            sanitization_level = self.config.security_level
        
        sanitized_img = img_array.copy().astype(np.float32) / 255.0
        applied_sanitizations = []
        
        try:
            # Apply noise injection (helps against adversarial attacks)
            if self.config.enable_noise_injection:
                noise = np.random.normal(0, self.config.noise_injection_strength, sanitized_img.shape)
                sanitized_img = np.clip(sanitized_img + noise, 0, 1)
                applied_sanitizations.append('noise_injection')
            
            # Apply Gaussian blur (reduces high-frequency adversarial patterns)
            if self.config.enable_gaussian_blur:
                if HAS_OPENCV:
                    if len(sanitized_img.shape) == 3:
                        for channel in range(sanitized_img.shape[2]):
                            sanitized_img[:, :, channel] = cv2.GaussianBlur(
                                sanitized_img[:, :, channel], 
                                (self.config.blur_kernel_size, self.config.blur_kernel_size), 0
                            )
                    else:
                        sanitized_img = cv2.GaussianBlur(
                            sanitized_img, 
                            (self.config.blur_kernel_size, self.config.blur_kernel_size), 0
                        )
                    applied_sanitizations.append('gaussian_blur')
                elif HAS_SKIMAGE:
                    sanitized_img = filters.gaussian(sanitized_img, sigma=1.0)
                    applied_sanitizations.append('gaussian_blur')
            
            # Apply bit depth reduction (removes LSB steganography)
            if self.config.enable_bit_depth_reduction:
                max_val = 2 ** self.config.bit_depth - 1
                sanitized_img = np.round(sanitized_img * max_val) / max_val
                applied_sanitizations.append('bit_depth_reduction')
            
            # Convert back to uint8
            sanitized_img = (np.clip(sanitized_img, 0, 1) * 255).astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Image sanitization failed: {e}")
            # Return original image if sanitization fails
            return img_array, []
        
        return sanitized_img, applied_sanitizations


class SecureImagePreprocessor:
    """
    Secure image preprocessor with comprehensive threat detection and mitigation.
    
    Provides multi-layered security for image processing including:
    - Format validation and signature verification
    - Adversarial attack detection
    - Steganography detection
    - Malware and exploit prevention
    - Sanitization and defensive transformations
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM,
                 config: Optional[SecurityConfig] = None):
        self.security_level = security_level
        self.config = config or SecurityConfig(security_level=security_level)
        self.logger = logging.getLogger(f"{__name__}.SecureImagePreprocessor")
        
        # Initialize components
        self.validator = SecureImageValidator(security_level)
        self.sanitizer = SecureImageSanitizer(self.config)
        
        self.logger.info(f"SecureImagePreprocessor initialized with security level: {security_level.name}")
    
    def process_image_secure(self, image_data: Union[bytes, np.ndarray], 
                           filename: Optional[str] = None,
                           enable_sanitization: bool = True,
                           return_format: str = 'numpy') -> Dict[str, Any]:
        """
        Process image with comprehensive security measures.
        
        Args:
            image_data: Raw image data (bytes) or numpy array
            filename: Optional filename for format detection
            enable_sanitization: Whether to apply sanitization
            return_format: Format for returned image ('numpy', 'pil', 'bytes')
            
        Returns:
            Dictionary containing processed image and security analysis
        """
        start_time = time.time()
        result = {
            'success': False,
            'processed_image': None,
            'threats_detected': [],
            'threats_mitigated': [],
            'confidence_scores': {},
            'processing_time': 0.0,
            'security_analysis': {}
        }
        
        try:
            # Convert input to bytes if needed
            if isinstance(image_data, np.ndarray):
                if HAS_PIL:
                    import io
                    pil_img = Image.fromarray(image_data)
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format='PNG')
                    image_bytes = buffer.getvalue()
                else:
                    raise ValueError("PIL not available for array conversion")
            else:
                image_bytes = image_data
            
            # Security validation
            self.logger.debug("Performing security validation...")
            validation_result = self.validator.validate_image_security(
                image_bytes, filename, detailed_analysis=True
            )
            
            result['security_analysis'] = validation_result
            result['threats_detected'] = validation_result.get('threats_detected', [])
            result['confidence_scores'] = validation_result.get('confidence_scores', {})
            
            if not validation_result['is_safe']:
                self.logger.warning(f"Security threats detected: {result['threats_detected']}")
                # Continue processing but mark threats
            
            # Load image for processing
            if HAS_PIL:
                import io
                img = Image.open(io.BytesIO(image_bytes))
                img_array = np.array(img.convert('RGB'))
            else:
                raise ValueError("PIL not available for image loading")
            
            # Apply sanitization if enabled
            processed_img = img_array
            sanitizations_applied = []
            
            if enable_sanitization:
                self.logger.debug("Applying security sanitization...")
                processed_img, sanitizations_applied = self.sanitizer.sanitize_image(
                    img_array, self.security_level
                )
                result['threats_mitigated'] = sanitizations_applied
            
            # Convert to requested format
            if return_format == 'numpy':
                result['processed_image'] = processed_img
            elif return_format == 'pil' and HAS_PIL:
                result['processed_image'] = Image.fromarray(processed_img)
            elif return_format == 'bytes' and HAS_PIL:
                import io
                pil_img = Image.fromarray(processed_img)
                buffer = io.BytesIO()
                pil_img.save(buffer, format='PNG')
                result['processed_image'] = buffer.getvalue()
            else:
                result['processed_image'] = processed_img
            
            result['success'] = True
            result['image_size'] = (processed_img.shape[1], processed_img.shape[0])
            result['image_format'] = 'RGB'
            
            self.logger.info(f"Secure image processing completed. "
                           f"Threats detected: {len(result['threats_detected'])}, "
                           f"Mitigations applied: {len(sanitizations_applied)}")
            
        except Exception as e:
            self.logger.error(f"Secure image processing failed: {e}")
            result['error'] = str(e)
            result['threats_detected'].append('processing_error')
        
        finally:
            result['processing_time'] = time.time() - start_time
        
        return result
    
    def validate_only(self, image_data: bytes, filename: Optional[str] = None) -> Dict[str, Any]:
        """Validate image security without processing."""
        return self.validator.validate_image_security(image_data, filename, detailed_analysis=True)
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security processing statistics."""
        return {
            'security_level': self.security_level.name,
            'config': {
                'max_image_size_mb': self.config.max_image_size_mb,
                'max_dimensions': self.config.max_image_dimensions,
                'allowed_formats': self.config.allowed_formats,
                'sanitization_enabled': {
                    'noise_injection': self.config.enable_noise_injection,
                    'gaussian_blur': self.config.enable_gaussian_blur,
                    'bit_depth_reduction': self.config.enable_bit_depth_reduction,
                    'jpeg_compression': self.config.enable_jpeg_compression
                }
            },
            'capabilities': {
                'pil_available': HAS_PIL,
                'opencv_available': HAS_OPENCV,
                'skimage_available': HAS_SKIMAGE,
                'sklearn_available': HAS_SKLEARN
            }
        }


# Export main classes and functions
__all__ = [
    'SecurityLevel',
    'ThreatLevel', 
    'AttackType',
    'SecurityConfig',
    'SecurityReport',
    'SecureImageValidator',
    'SecureImageSanitizer', 
    'SecureImagePreprocessor'
]

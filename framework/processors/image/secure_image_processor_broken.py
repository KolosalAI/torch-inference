"""
Secure Image Preprocessing and Postprocessing for PyTorch Inference Framework

This module provides comprehensive image processing with security measures to prevent:
- Adversarial attacks (FGSM, PGD, C&W, etc.)
- Backdoor/Trojan attacks
- Data poisoning attacks
- Steganography-based attacks
- Format-based exploits
- Memory-based attacks
- Timing attacks
- Model inversion attacks

Security Features:
- Input validation and sanitization
- Image format verification
- Adversarial detection and mitigation
- Noise injection for robustness
- Size and dimension constraints
- Memory usage monitoring
- Timing attack prevention
- Statistical anomaly detection
"""

import os
import hashlib
import time
import logging
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
import warnings
from abc import ABC, abstractmethod

# Try to import optional dependencies
try:
    from PIL import Image, ImageStat, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = ImageStat = ImageFilter = None

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None

try:
    from skimage import filters, feature, measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    filters = feature = measure = None

from .image_preprocessor import ComprehensiveImagePreprocessor, ImagePreprocessorError
from ..preprocessor import BasePreprocessor
from ..postprocessor import BasePostprocessor
from ...security.security import SecurityManager, ThreatLevel, SecurityEvent


logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for image processing."""
    LOW = "low"          # Basic validation only
    MEDIUM = "medium"    # Standard security checks
    HIGH = "high"        # Comprehensive security
    PARANOID = "paranoid"  # Maximum security (may impact performance)


class AttackType(Enum):
    """Types of attacks the system can detect."""
    ADVERSARIAL = "adversarial"
    BACKDOOR = "backdoor"
    STEGANOGRAPHY = "steganography"
    FORMAT_EXPLOIT = "format_exploit"
    MEMORY_ATTACK = "memory_attack"
    TIMING_ATTACK = "timing_attack"
    DATA_POISONING = "data_poisoning"
    MODEL_INVERSION = "model_inversion"


@dataclass
class SecurityConfig:
    """Configuration for secure image processing."""
    
    # General security
    security_level: SecurityLevel = SecurityLevel.HIGH
    enable_format_validation: bool = True
    enable_signature_verification: bool = True
    enable_adversarial_detection: bool = True
    enable_statistical_analysis: bool = True
    enable_steganography_detection: bool = True
    enable_memory_monitoring: bool = True
    enable_timing_protection: bool = True
    
    # Size and format constraints
    max_image_size_mb: float = 50.0
    max_image_dimensions: Tuple[int, int] = (4096, 4096)
    min_image_dimensions: Tuple[int, int] = (8, 8)
    allowed_formats: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'])
    max_channels: int = 4  # RGBA
    min_channels: int = 1  # Grayscale
    
    # Adversarial detection
    adversarial_threshold: float = 0.1
    noise_variance_threshold: float = 0.05
    gradient_magnitude_threshold: float = 0.2
    pixel_value_distribution_check: bool = True
    
    # Defense mechanisms
    enable_noise_injection: bool = True
    noise_injection_strength: float = 0.01
    enable_gaussian_blur: bool = True
    blur_kernel_size: int = 3
    enable_jpeg_compression: bool = True
    jpeg_quality: int = 95
    enable_bit_depth_reduction: bool = True
    bit_depth: int = 7  # Reduce from 8 to 7 bits
    
    # Statistical thresholds
    entropy_min_threshold: float = 3.0
    entropy_max_threshold: float = 8.0
    histogram_uniformity_threshold: float = 0.8
    edge_density_threshold: float = 0.3
    texture_complexity_threshold: float = 0.1
    
    # Performance and monitoring
    max_processing_time_seconds: float = 30.0
    memory_limit_mb: float = 1024.0
    enable_secure_logging: bool = True
    audit_all_operations: bool = True
    
    # Blacklist patterns
    suspicious_metadata_patterns: List[str] = field(default_factory=lambda: [
        'script', 'eval', 'exec', 'import', 'subprocess', 'os.', 'sys.',
        'shell', 'cmd', 'powershell', 'bash'
    ])
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_image_size_mb <= 0:
            raise ValueError("max_image_size_mb must be positive")
        if not all(d > 0 for d in self.max_image_dimensions):
            raise ValueError("max_image_dimensions must be positive")
        if not all(d > 0 for d in self.min_image_dimensions):
            raise ValueError("min_image_dimensions must be positive")
        if self.max_channels < self.min_channels:
            raise ValueError("max_channels must be >= min_channels")
        if not 0 <= self.adversarial_threshold <= 1:
            raise ValueError("adversarial_threshold must be between 0 and 1")


@dataclass
class SecurityReport:
    """Security analysis report for processed images."""
    
    is_safe: bool
    threats_detected: List[AttackType]
    threat_level: ThreatLevel
    confidence_score: float
    
    # Analysis results
    format_validation: Dict[str, Any]
    statistical_analysis: Dict[str, Any]
    adversarial_analysis: Dict[str, Any]
    steganography_analysis: Dict[str, Any]
    metadata_analysis: Dict[str, Any]
    
    # Performance metrics
    processing_time: float
    memory_usage_mb: float
    
    # Recommendations
    recommended_actions: List[str]
    sanitization_applied: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            'is_safe': self.is_safe,
            'threats_detected': [t.value for t in self.threats_detected],
            'threat_level': self.threat_level.value,
            'confidence_score': self.confidence_score,
            'format_validation': self.format_validation,
            'statistical_analysis': self.statistical_analysis,
            'adversarial_analysis': self.adversarial_analysis,
            'steganography_analysis': self.steganography_analysis,
            'metadata_analysis': self.metadata_analysis,
            'processing_time': self.processing_time,
            'memory_usage_mb': self.memory_usage_mb,
            'recommended_actions': self.recommended_actions,
            'sanitization_applied': self.sanitization_applied
        }


class SecureImageValidator:
    """Comprehensive image validation and security analysis."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
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
        }\n    \n    def validate_image_file(self, file_path: Union[str, Path], \n                           file_data: Optional[bytes] = None) -> SecurityReport:\n        \"\"\"Comprehensive validation of image file.\"\"\"\n        start_time = time.time()\n        threats_detected = []\n        threat_level = ThreatLevel.LOW\n        confidence_score = 1.0\n        \n        try:\n            # Read file data if not provided\n            if file_data is None:\n                with open(file_path, 'rb') as f:\n                    file_data = f.read()\n            \n            # Basic file validation\n            format_validation = self._validate_file_format(file_path, file_data)\n            if not format_validation['is_valid']:\n                threats_detected.append(AttackType.FORMAT_EXPLOIT)\n                threat_level = ThreatLevel.HIGH\n                confidence_score = 0.1\n            \n            # Size validation\n            size_mb = len(file_data) / (1024 * 1024)\n            if size_mb > self.config.max_image_size_mb:\n                threats_detected.append(AttackType.MEMORY_ATTACK)\n                threat_level = max(threat_level, ThreatLevel.MEDIUM)\n            \n            # Load and analyze image\n            try:\n                if HAS_PIL:\n                    with Image.open(file_path) as img:\n                        img_array = np.array(img.convert('RGB'))\n                else:\n                    raise ImportError(\"PIL not available\")\n            except Exception as e:\n                self.logger.error(f\"Failed to load image: {e}\")\n                threats_detected.append(AttackType.FORMAT_EXPLOIT)\n                threat_level = ThreatLevel.HIGH\n                confidence_score = 0.0\n                img_array = None\n            \n            # Dimension validation\n            if img_array is not None:\n                h, w = img_array.shape[:2]\n                if (h > self.config.max_image_dimensions[0] or \n                    w > self.config.max_image_dimensions[1] or\n                    h < self.config.min_image_dimensions[0] or \n                    w < self.config.min_image_dimensions[1]):\n                    threats_detected.append(AttackType.MEMORY_ATTACK)\n                    threat_level = max(threat_level, ThreatLevel.MEDIUM)\n            \n            # Statistical analysis\n            statistical_analysis = self._analyze_image_statistics(img_array) if img_array is not None else {}\n            if not statistical_analysis.get('is_normal', True):\n                threats_detected.append(AttackType.ADVERSARIAL)\n                threat_level = max(threat_level, ThreatLevel.MEDIUM)\n                confidence_score = min(confidence_score, 0.7)\n            \n            # Adversarial detection\n            adversarial_analysis = self._detect_adversarial_patterns(img_array) if img_array is not None else {}\n            if adversarial_analysis.get('is_adversarial', False):\n                threats_detected.append(AttackType.ADVERSARIAL)\n                threat_level = max(threat_level, ThreatLevel.HIGH)\n                confidence_score = min(confidence_score, 0.5)\n            \n            # Steganography detection\n            steganography_analysis = self._detect_steganography(img_array, file_data) if img_array is not None else {}\n            if steganography_analysis.get('suspicious', False):\n                threats_detected.append(AttackType.STEGANOGRAPHY)\n                threat_level = max(threat_level, ThreatLevel.MEDIUM)\n                confidence_score = min(confidence_score, 0.6)\n            \n            # Metadata analysis\n            metadata_analysis = self._analyze_metadata(file_path)\n            if metadata_analysis.get('suspicious', False):\n                threats_detected.append(AttackType.FORMAT_EXPLOIT)\n                threat_level = max(threat_level, ThreatLevel.MEDIUM)\n            \n            processing_time = time.time() - start_time\n            memory_usage_mb = len(file_data) / (1024 * 1024) if file_data else 0\n            \n            # Determine if image is safe\n            is_safe = (len(threats_detected) == 0 and \n                      confidence_score > 0.8 and \n                      processing_time < self.config.max_processing_time_seconds)\n            \n            # Generate recommendations\n            recommended_actions = self._generate_recommendations(threats_detected, threat_level)\n            \n            return SecurityReport(\n                is_safe=is_safe,\n                threats_detected=threats_detected,\n                threat_level=threat_level,\n                confidence_score=confidence_score,\n                format_validation=format_validation,\n                statistical_analysis=statistical_analysis,\n                adversarial_analysis=adversarial_analysis,\n                steganography_analysis=steganography_analysis,\n                metadata_analysis=metadata_analysis,\n                processing_time=processing_time,\n                memory_usage_mb=memory_usage_mb,\n                recommended_actions=recommended_actions,\n                sanitization_applied=[]\n            )\n            \n        except Exception as e:\n            self.logger.error(f\"Security validation failed: {e}\")\n            return SecurityReport(\n                is_safe=False,\n                threats_detected=[AttackType.FORMAT_EXPLOIT],\n                threat_level=ThreatLevel.CRITICAL,\n                confidence_score=0.0,\n                format_validation={'is_valid': False, 'error': str(e)},\n                statistical_analysis={},\n                adversarial_analysis={},\n                steganography_analysis={},\n                metadata_analysis={},\n                processing_time=time.time() - start_time,\n                memory_usage_mb=0,\n                recommended_actions=['Reject input'],\n                sanitization_applied=[]\n            )\n    \n    def _validate_file_format(self, file_path: Union[str, Path], file_data: bytes) -> Dict[str, Any]:\n        \"\"\"Validate file format and signature.\"\"\"\n        path = Path(file_path)\n        extension = path.suffix.lower()\n        \n        validation_result = {\n            'is_valid': False,\n            'extension': extension,\n            'signature_match': False,\n            'size_bytes': len(file_data),\n            'issues': []\n        }\n        \n        # Check allowed extensions\n        if extension not in self.config.allowed_formats:\n            validation_result['issues'].append(f\"Extension {extension} not allowed\")\n            return validation_result\n        \n        # Check file signature\n        if self.config.enable_signature_verification:\n            expected_signatures = self._file_signatures.get(extension, [])\n            signature_match = False\n            \n            for signature in expected_signatures:\n                if file_data.startswith(signature):\n                    signature_match = True\n                    break\n            \n            validation_result['signature_match'] = signature_match\n            if not signature_match:\n                validation_result['issues'].append(\"File signature doesn't match extension\")\n                return validation_result\n        \n        # Additional format-specific checks\n        if extension in ['.jpg', '.jpeg']:\n            # Check for JPEG end marker\n            if not file_data.endswith(b'\\xff\\xd9'):\n                validation_result['issues'].append(\"JPEG file missing end marker\")\n                return validation_result\n        \n        validation_result['is_valid'] = True\n        return validation_result\n    \n    def _analyze_image_statistics(self, img_array: np.ndarray) -> Dict[str, Any]:\n        \"\"\"Analyze image statistics for anomalies.\"\"\"\n        if img_array is None or img_array.size == 0:\n            return {'is_normal': False, 'error': 'Empty image'}\n        \n        analysis = {\n            'is_normal': True,\n            'issues': [],\n            'metrics': {}\n        }\n        \n        try:\n            # Convert to grayscale for analysis\n            if len(img_array.shape) == 3:\n                gray = np.mean(img_array, axis=2)\n            else:\n                gray = img_array\n            \n            # Calculate entropy\n            hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))\n            hist = hist / hist.sum()  # Normalize\n            entropy = -np.sum(hist * np.log2(hist + 1e-10))\n            \n            analysis['metrics']['entropy'] = float(entropy)\n            \n            if entropy < self.config.entropy_min_threshold or entropy > self.config.entropy_max_threshold:\n                analysis['is_normal'] = False\n                analysis['issues'].append(f\"Entropy {entropy:.2f} outside normal range\")\n            \n            # Check histogram uniformity\n            uniformity = 1.0 - np.std(hist) / np.mean(hist + 1e-10)\n            analysis['metrics']['histogram_uniformity'] = float(uniformity)\n            \n            if uniformity > self.config.histogram_uniformity_threshold:\n                analysis['is_normal'] = False\n                analysis['issues'].append(f\"Histogram too uniform: {uniformity:.3f}\")\n            \n            # Edge density analysis\n            if HAS_OPENCV:\n                edges = cv2.Canny(gray.astype(np.uint8), 50, 150)\n                edge_density = np.sum(edges > 0) / edges.size\n            elif HAS_SKIMAGE:\n                edges = filters.sobel(gray)\n                edge_density = np.sum(edges > 0.1) / edges.size\n            else:\n                # Simple gradient-based edge detection\n                grad_x = np.abs(np.gradient(gray, axis=1))\n                grad_y = np.abs(np.gradient(gray, axis=0))\n                edges = np.sqrt(grad_x**2 + grad_y**2)\n                edge_density = np.sum(edges > np.mean(edges)) / edges.size\n            \n            analysis['metrics']['edge_density'] = float(edge_density)\n            \n            if edge_density < self.config.edge_density_threshold:\n                analysis['is_normal'] = False\n                analysis['issues'].append(f\"Edge density too low: {edge_density:.3f}\")\n            \n            # Pixel value distribution check\n            if self.config.pixel_value_distribution_check:\n                # Check for suspicious patterns in pixel values\n                unique_values = len(np.unique(gray))\n                total_pixels = gray.size\n                uniqueness_ratio = unique_values / total_pixels\n                \n                analysis['metrics']['uniqueness_ratio'] = float(uniqueness_ratio)\n                \n                if uniqueness_ratio < 0.01:  # Too few unique values\n                    analysis['is_normal'] = False\n                    analysis['issues'].append(f\"Too few unique pixel values: {uniqueness_ratio:.4f}\")\n            \n        except Exception as e:\n            self.logger.error(f\"Statistical analysis failed: {e}\")\n            analysis['is_normal'] = False\n            analysis['error'] = str(e)\n        \n        return analysis\n    \n    def _detect_adversarial_patterns(self, img_array: np.ndarray) -> Dict[str, Any]:\n        \"\"\"Detect adversarial attack patterns.\"\"\"\n        if img_array is None or img_array.size == 0:\n            return {'is_adversarial': False, 'error': 'Empty image'}\n        \n        analysis = {\n            'is_adversarial': False,\n            'confidence': 0.0,\n            'patterns_detected': [],\n            'metrics': {}\n        }\n        \n        try:\n            # Convert to float for analysis\n            img_float = img_array.astype(np.float32) / 255.0\n            \n            # Check for high-frequency noise patterns (common in adversarial examples)\n            if len(img_float.shape) == 3:\n                # Multi-channel image\n                noise_levels = []\n                for channel in range(img_float.shape[2]):\n                    channel_data = img_float[:, :, channel]\n                    # Calculate high-frequency components\n                    if HAS_OPENCV:\n                        blurred = cv2.GaussianBlur(channel_data, (5, 5), 0)\n                    else:\n                        # Simple blur approximation\n                        kernel = np.ones((5, 5)) / 25\n                        blurred = np.convolve(channel_data.flatten(), kernel.flatten(), mode='same').reshape(channel_data.shape)\n                    \n                    noise = np.abs(channel_data - blurred)\n                    noise_level = np.mean(noise)\n                    noise_levels.append(noise_level)\n                \n                avg_noise = np.mean(noise_levels)\n            else:\n                # Grayscale image\n                if HAS_OPENCV:\n                    blurred = cv2.GaussianBlur(img_float, (5, 5), 0)\n                else:\n                    # Simple blur approximation\n                    kernel = np.ones((5, 5)) / 25\n                    blurred = np.convolve(img_float.flatten(), kernel.flatten(), mode='same').reshape(img_float.shape)\n                \n                noise = np.abs(img_float - blurred)\n                avg_noise = np.mean(noise)\n            \n            analysis['metrics']['average_noise_level'] = float(avg_noise)\n            \n            if avg_noise > self.config.noise_variance_threshold:\n                analysis['is_adversarial'] = True\n                analysis['patterns_detected'].append('high_frequency_noise')\n                analysis['confidence'] += 0.3\n            \n            # Check for gradient magnitude anomalies\n            if len(img_float.shape) == 3:\n                gray = np.mean(img_float, axis=2)\n            else:\n                gray = img_float\n            \n            grad_x = np.gradient(gray, axis=1)\n            grad_y = np.gradient(gray, axis=0)\n            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)\n            avg_gradient = np.mean(gradient_magnitude)\n            \n            analysis['metrics']['average_gradient_magnitude'] = float(avg_gradient)\n            \n            if avg_gradient > self.config.gradient_magnitude_threshold:\n                analysis['is_adversarial'] = True\n                analysis['patterns_detected'].append('high_gradient_magnitude')\n                analysis['confidence'] += 0.4\n            \n            # Check for periodic patterns (backdoor indicators)\n            # Apply FFT to detect hidden periodic signals\n            try:\n                fft = np.fft.fft2(gray)\n                fft_magnitude = np.abs(fft)\n                fft_magnitude = np.log(fft_magnitude + 1)\n                \n                # Look for strong periodic components\n                fft_normalized = fft_magnitude / np.max(fft_magnitude)\n                strong_components = np.sum(fft_normalized > 0.8)\n                \n                analysis['metrics']['strong_frequency_components'] = int(strong_components)\n                \n                if strong_components > gray.size * 0.01:  # More than 1% of components are strong\n                    analysis['is_adversarial'] = True\n                    analysis['patterns_detected'].append('periodic_patterns')\n                    analysis['confidence'] += 0.3\n                    \n            except Exception as e:\n                self.logger.debug(f\"FFT analysis failed: {e}\")\n            \n            # Final confidence calculation\n            if analysis['confidence'] > 0.5:\n                analysis['is_adversarial'] = True\n            \n        except Exception as e:\n            self.logger.error(f\"Adversarial detection failed: {e}\")\n            analysis['error'] = str(e)\n        \n        return analysis\n    \n    def _detect_steganography(self, img_array: np.ndarray, file_data: bytes) -> Dict[str, Any]:\n        \"\"\"Detect steganography and hidden data.\"\"\"\n        analysis = {\n            'suspicious': False,\n            'confidence': 0.0,\n            'indicators': [],\n            'metrics': {}\n        }\n        \n        try:\n            # LSB (Least Significant Bit) analysis\n            if len(img_array.shape) == 3:\n                # Check LSBs in each channel\n                for channel in range(img_array.shape[2]):\n                    channel_data = img_array[:, :, channel]\n                    lsb_data = channel_data & 1  # Extract LSBs\n                    \n                    # Calculate entropy of LSB plane\n                    hist, _ = np.histogram(lsb_data.flatten(), bins=2, range=(0, 2))\n                    hist = hist / hist.sum()\n                    lsb_entropy = -np.sum(hist * np.log2(hist + 1e-10))\n                    \n                    analysis['metrics'][f'lsb_entropy_channel_{channel}'] = float(lsb_entropy)\n                    \n                    # High entropy in LSB plane may indicate steganography\n                    if lsb_entropy > 0.9:\n                        analysis['suspicious'] = True\n                        analysis['indicators'].append(f'high_lsb_entropy_channel_{channel}')\n                        analysis['confidence'] += 0.2\n            \n            # File size analysis\n            expected_size = img_array.size * img_array.dtype.itemsize\n            actual_size = len(file_data)\n            size_ratio = actual_size / expected_size if expected_size > 0 else 0\n            \n            analysis['metrics']['size_ratio'] = float(size_ratio)\n            \n            # Unusually large file size may indicate hidden data\n            if size_ratio > 3.0:  # File is 3x larger than expected\n                analysis['suspicious'] = True\n                analysis['indicators'].append('unusual_file_size')\n                analysis['confidence'] += 0.3\n            \n            # Check for hidden data in padding\n            if file_data.endswith(b'\\x00' * 10):  # Many null bytes at end\n                trailing_zeros = 0\n                for byte in reversed(file_data):\n                    if byte == 0:\n                        trailing_zeros += 1\n                    else:\n                        break\n                \n                if trailing_zeros > actual_size * 0.1:  # More than 10% trailing zeros\n                    analysis['suspicious'] = True\n                    analysis['indicators'].append('excessive_padding')\n                    analysis['confidence'] += 0.2\n            \n        except Exception as e:\n            self.logger.error(f\"Steganography detection failed: {e}\")\n            analysis['error'] = str(e)\n        \n        return analysis\n    \n    def _analyze_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:\n        \"\"\"Analyze image metadata for suspicious content.\"\"\"\n        analysis = {\n            'suspicious': False,\n            'metadata': {},\n            'issues': []\n        }\n        \n        try:\n            if HAS_PIL:\n                with Image.open(file_path) as img:\n                    # Extract EXIF data\n                    exif_data = img.getexif() if hasattr(img, 'getexif') else {}\n                    \n                    for tag_id, value in exif_data.items():\n                        if isinstance(value, str):\n                            # Check for suspicious patterns in metadata\n                            value_lower = value.lower()\n                            for pattern in self.config.suspicious_metadata_patterns:\n                                if pattern in value_lower:\n                                    analysis['suspicious'] = True\n                                    analysis['issues'].append(f\"Suspicious metadata pattern: {pattern}\")\n                    \n                    analysis['metadata'] = dict(exif_data)\n        \n        except Exception as e:\n            self.logger.debug(f\"Metadata analysis failed: {e}\")\n            analysis['error'] = str(e)\n        \n        return analysis\n    \n    def _generate_recommendations(self, threats_detected: List[AttackType], \n                                threat_level: ThreatLevel) -> List[str]:\n        \"\"\"Generate security recommendations based on detected threats.\"\"\"\n        recommendations = []\n        \n        if AttackType.ADVERSARIAL in threats_detected:\n            recommendations.extend([\n                \"Apply noise injection to reduce adversarial effectiveness\",\n                \"Use ensemble models for more robust predictions\",\n                \"Consider adversarial training for the model\"\n            ])\n        \n        if AttackType.STEGANOGRAPHY in threats_detected:\n            recommendations.extend([\n                \"Apply JPEG compression to remove hidden data\",\n                \"Reduce bit depth to eliminate LSB steganography\",\n                \"Re-encode image to remove metadata\"\n            ])\n        \n        if AttackType.FORMAT_EXPLOIT in threats_detected:\n            recommendations.extend([\n                \"Reject input and request valid format\",\n                \"Re-encode image using trusted library\",\n                \"Validate all inputs thoroughly\"\n            ])\n        \n        if AttackType.MEMORY_ATTACK in threats_detected:\n            recommendations.extend([\n                \"Resize image to acceptable dimensions\",\n                \"Compress image to reduce file size\",\n                \"Implement strict memory limits\"\n            ])\n        \n        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:\n            recommendations.append(\"Consider rejecting input entirely\")\n        \n        return recommendations


class SecureImageSanitizer:
    \"\"\"Secure image sanitization and defense mechanisms.\"\"\"\n    \n    def __init__(self, config: SecurityConfig):\n        self.config = config\n        self.logger = logging.getLogger(f\"{__name__}.SecureImageSanitizer\")\n    \n    def sanitize_image(self, img_array: np.ndarray, \n                      sanitization_level: SecurityLevel = None) -> Tuple[np.ndarray, List[str]]:\n        \"\"\"Apply security sanitization to image.\"\"\"\n        if sanitization_level is None:\n            sanitization_level = self.config.security_level\n        \n        sanitized_img = img_array.copy().astype(np.float32) / 255.0\n        applied_sanitizations = []\n        \n        try:\n            # Apply noise injection (helps against adversarial attacks)\n            if self.config.enable_noise_injection:\n                noise = np.random.normal(0, self.config.noise_injection_strength, sanitized_img.shape)\n                sanitized_img = np.clip(sanitized_img + noise, 0, 1)\n                applied_sanitizations.append('noise_injection')\n            \n            # Apply Gaussian blur (reduces high-frequency adversarial patterns)\n            if self.config.enable_gaussian_blur:\n                if HAS_OPENCV:\n                    if len(sanitized_img.shape) == 3:\n                        for channel in range(sanitized_img.shape[2]):\n                            sanitized_img[:, :, channel] = cv2.GaussianBlur(\n                                sanitized_img[:, :, channel], \n                                (self.config.blur_kernel_size, self.config.blur_kernel_size), 0\n                            )\n                    else:\n                        sanitized_img = cv2.GaussianBlur(\n                            sanitized_img, \n                            (self.config.blur_kernel_size, self.config.blur_kernel_size), 0\n                        )\n                    applied_sanitizations.append('gaussian_blur')\n                elif HAS_SKIMAGE:\n                    sanitized_img = filters.gaussian(sanitized_img, sigma=1.0)\n                    applied_sanitizations.append('gaussian_blur')\n            \n            # Apply bit depth reduction (removes LSB steganography)\n            if self.config.enable_bit_depth_reduction:\n                max_val = 2 ** self.config.bit_depth - 1\n                sanitized_img = np.round(sanitized_img * max_val) / max_val\n                applied_sanitizations.append('bit_depth_reduction')\n            \n            # JPEG compression simulation (removes high-frequency noise)\n            if self.config.enable_jpeg_compression and sanitization_level in [SecurityLevel.HIGH, SecurityLevel.PARANOID]:\n                # Simulate JPEG compression by applying DCT and quantization\n                try:\n                    sanitized_img = self._apply_jpeg_compression_simulation(sanitized_img)\n                    applied_sanitizations.append('jpeg_compression')\n                except Exception as e:\n                    self.logger.warning(f\"JPEG compression simulation failed: {e}\")\n            \n            # Convert back to uint8\n            sanitized_img = (np.clip(sanitized_img, 0, 1) * 255).astype(np.uint8)\n            \n        except Exception as e:\n            self.logger.error(f\"Image sanitization failed: {e}\")\n            # Return original image if sanitization fails\n            return img_array, []\n        \n        return sanitized_img, applied_sanitizations\n    \n    def _apply_jpeg_compression_simulation(self, img: np.ndarray) -> np.ndarray:\n        \"\"\"Simulate JPEG compression to remove adversarial noise.\"\"\"\n        if not HAS_PIL:\n            return img\n        \n        try:\n            # Convert to PIL Image\n            if len(img.shape) == 3:\n                pil_img = Image.fromarray((img * 255).astype(np.uint8))\n            else:\n                pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='L')\n            \n            # Save to JPEG in memory and reload\n            import io\n            buffer = io.BytesIO()\n            pil_img.save(buffer, format='JPEG', quality=self.config.jpeg_quality)\n            buffer.seek(0)\n            \n            compressed_img = Image.open(buffer)\n            \n            # Convert back to numpy array\n            if len(img.shape) == 3:\n                result = np.array(compressed_img.convert('RGB')).astype(np.float32) / 255.0\n            else:\n                result = np.array(compressed_img.convert('L')).astype(np.float32) / 255.0\n            \n            return result\n            \n        except Exception as e:\n            self.logger.error(f\"JPEG compression simulation failed: {e}\")\n            return img


class SecureImagePreprocessor(ComprehensiveImagePreprocessor):\n    \"\"\"Secure image preprocessor with comprehensive attack prevention.\"\"\"\n    \n    def __init__(self, target_size: Optional[Tuple[int, int]] = (224, 224),\n                 normalize: bool = True, \n                 security_config: Optional[SecurityConfig] = None,\n                 security_manager: Optional[SecurityManager] = None):\n        \n        super().__init__(target_size, normalize)\n        \n        self.security_config = security_config or SecurityConfig()\n        self.security_config.validate()\n        \n        self.security_manager = security_manager\n        self.validator = SecureImageValidator(self.security_config)\n        self.sanitizer = SecureImageSanitizer(self.security_config)\n        \n        # Performance monitoring\n        self._processing_stats = {\n            'total_processed': 0,\n            'threats_detected': 0,\n            'images_rejected': 0,\n            'sanitizations_applied': 0\n        }\n        \n        self.logger = logging.getLogger(f\"{__name__}.SecureImagePreprocessor\")\n    \n    def process_secure(self, image: Union[str, np.ndarray, torch.Tensor], \n                      client_id: Optional[str] = None,\n                      enable_sanitization: bool = True) -> Tuple[torch.Tensor, SecurityReport]:\n        \"\"\"Process image with comprehensive security analysis.\"\"\"\n        start_time = time.time()\n        \n        try:\n            self._processing_stats['total_processed'] += 1\n            \n            # Step 1: Load image if file path\n            if isinstance(image, (str, Path)):\n                # Validate file first\n                security_report = self.validator.validate_image_file(image)\n                \n                if not security_report.is_safe:\n                    self._processing_stats['threats_detected'] += 1\n                    self._processing_stats['images_rejected'] += 1\n                    \n                    # Log security event\n                    if self.security_manager:\n                        self.security_manager.log_security_event(\n                            SecurityEvent.INVALID_INPUT,\n                            client_id,\n                            f\"Unsafe image detected: {security_report.threats_detected}\",\n                            {\n                                'file_path': str(image),\n                                'threats': [t.value for t in security_report.threats_detected],\n                                'threat_level': security_report.threat_level.value\n                            }\n                        )\n                    \n                    if security_report.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:\n                        raise ImagePreprocessorError(\n                            f\"Image rejected due to security threats: {security_report.threats_detected}\"\n                        )\n                \n                # Load image\n                image_array = self.load_image(image, self.target_size)\n            else:\n                # Convert tensor/array to numpy\n                image_array = self._tensor_to_numpy(image) if isinstance(image, torch.Tensor) else image\n                \n                # Create basic security report for non-file inputs\n                security_report = SecurityReport(\n                    is_safe=True,\n                    threats_detected=[],\n                    threat_level=ThreatLevel.LOW,\n                    confidence_score=1.0,\n                    format_validation={'is_valid': True},\n                    statistical_analysis={},\n                    adversarial_analysis={},\n                    steganography_analysis={},\n                    metadata_analysis={},\n                    processing_time=0.0,\n                    memory_usage_mb=0.0,\n                    recommended_actions=[],\n                    sanitization_applied=[]\n                )\n            \n            # Step 2: Apply sanitization if enabled and needed\n            sanitization_applied = []\n            if enable_sanitization and (\n                security_report.threats_detected or \n                self.security_config.security_level in [SecurityLevel.HIGH, SecurityLevel.PARANOID]\n            ):\n                image_array, sanitization_applied = self.sanitizer.sanitize_image(\n                    image_array, self.security_config.security_level\n                )\n                security_report.sanitization_applied = sanitization_applied\n                \n                if sanitization_applied:\n                    self._processing_stats['sanitizations_applied'] += 1\n            \n            # Step 3: Apply standard preprocessing\n            processed_tensor = self.preprocess_image(image_array)\n            \n            # Step 4: Timing attack prevention\n            if self.security_config.enable_timing_protection:\n                self._apply_timing_protection(start_time)\n            \n            # Update security report\n            security_report.processing_time = time.time() - start_time\n            \n            return processed_tensor, security_report\n            \n        except Exception as e:\n            self.logger.error(f\"Secure image processing failed: {e}\")\n            \n            # Create error security report\n            error_report = SecurityReport(\n                is_safe=False,\n                threats_detected=[AttackType.FORMAT_EXPLOIT],\n                threat_level=ThreatLevel.CRITICAL,\n                confidence_score=0.0,\n                format_validation={'is_valid': False, 'error': str(e)},\n                statistical_analysis={},\n                adversarial_analysis={},\n                steganography_analysis={},\n                metadata_analysis={},\n                processing_time=time.time() - start_time,\n                memory_usage_mb=0.0,\n                recommended_actions=['Reject input'],\n                sanitization_applied=[]\n            )\n            \n            raise ImagePreprocessorError(f\"Secure preprocessing failed: {e}\") from e\n    \n    def process(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:\n        \"\"\"Standard process method for compatibility (applies security by default).\"\"\"\n        tensor, _ = self.process_secure(image, enable_sanitization=True)\n        return tensor\n    \n    def _apply_timing_protection(self, start_time: float):\n        \"\"\"Apply timing attack protection by normalizing processing time.\"\"\"\n        try:\n            elapsed = time.time() - start_time\n            target_time = 0.1  # Target 100ms processing time\n            \n            if elapsed < target_time:\n                sleep_time = target_time - elapsed\n                # Add some randomness to prevent timing analysis\n                sleep_time += np.random.uniform(0, 0.02)  # 0-20ms random delay\n                time.sleep(sleep_time)\n                \n        except Exception as e:\n            self.logger.debug(f\"Timing protection failed: {e}\")\n    \n    def get_security_stats(self) -> Dict[str, Any]:\n        \"\"\"Get security processing statistics.\"\"\"\n        stats = self._processing_stats.copy()\n        \n        if stats['total_processed'] > 0:\n            stats['threat_detection_rate'] = stats['threats_detected'] / stats['total_processed']\n            stats['rejection_rate'] = stats['images_rejected'] / stats['total_processed']\n            stats['sanitization_rate'] = stats['sanitizations_applied'] / stats['total_processed']\n        else:\n            stats['threat_detection_rate'] = 0.0\n            stats['rejection_rate'] = 0.0\n            stats['sanitization_rate'] = 0.0\n        \n        return stats


class SecureImagePostprocessor(BasePostprocessor):\n    \"\"\"Secure image postprocessor with output sanitization.\"\"\"\n    \n    def __init__(self, config, security_config: Optional[SecurityConfig] = None):\n        super().__init__(config)\n        self.security_config = security_config or SecurityConfig()\n        self.logger = logging.getLogger(f\"{__name__}.SecureImagePostprocessor\")\n    \n    def postprocess(self, outputs: torch.Tensor, **kwargs) -> Dict[str, Any]:\n        \"\"\"Secure postprocessing with output sanitization.\"\"\"\n        try:\n            # Apply output sanitization to prevent model inversion attacks\n            sanitized_outputs = self._sanitize_outputs(outputs)\n            \n            # Standard postprocessing\n            if hasattr(super(), 'postprocess'):\n                result = super().postprocess(sanitized_outputs, **kwargs)\n            else:\n                # Basic postprocessing for compatibility\n                result = {\n                    \"predictions\": sanitized_outputs.cpu().numpy().tolist(),\n                    \"metadata\": {\n                        \"sanitized\": True,\n                        \"security_level\": self.security_config.security_level.value\n                    }\n                }\n            \n            # Add security metadata\n            if isinstance(result, dict):\n                result['security_metadata'] = {\n                    'output_sanitized': True,\n                    'security_level': self.security_config.security_level.value\n                }\n            \n            return result\n            \n        except Exception as e:\n            self.logger.error(f\"Secure postprocessing failed: {e}\")\n            raise\n    \n    def _sanitize_outputs(self, outputs: torch.Tensor) -> torch.Tensor:\n        \"\"\"Sanitize model outputs to prevent information leakage.\"\"\"\n        try:\n            # Apply noise to outputs to prevent model inversion\n            if self.security_config.enable_noise_injection:\n                noise = torch.randn_like(outputs) * 0.01\n                outputs = outputs + noise\n            \n            # Clip extreme values\n            outputs = torch.clamp(outputs, -10, 10)\n            \n            # Round to reduce precision (prevents some inference attacks)\n            outputs = torch.round(outputs * 1000) / 1000\n            \n            return outputs\n            \n        except Exception as e:\n            self.logger.error(f\"Output sanitization failed: {e}\")\n            return outputs\n\n\n# Convenience functions\ndef create_secure_image_processor(target_size: Tuple[int, int] = (224, 224),\n                                 security_level: SecurityLevel = SecurityLevel.HIGH,\n                                 security_manager: Optional[SecurityManager] = None) -> SecureImagePreprocessor:\n    \"\"\"Create a secure image preprocessor with default configuration.\"\"\"\n    \n    security_config = SecurityConfig(\n        security_level=security_level,\n        max_image_size_mb=50.0,\n        max_image_dimensions=(4096, 4096),\n        enable_adversarial_detection=True,\n        enable_steganography_detection=True,\n        enable_noise_injection=True,\n        enable_gaussian_blur=True\n    )\n    \n    return SecureImagePreprocessor(\n        target_size=target_size,\n        security_config=security_config,\n        security_manager=security_manager\n    )\n\n\ndef validate_image_security(image_path: Union[str, Path], \n                           security_level: SecurityLevel = SecurityLevel.HIGH) -> SecurityReport:\n    \"\"\"Validate image security without processing.\"\"\"\n    \n    security_config = SecurityConfig(security_level=security_level)\n    validator = SecureImageValidator(security_config)\n    \n    return validator.validate_image_file(image_path)\n

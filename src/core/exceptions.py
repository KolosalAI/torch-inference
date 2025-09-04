"""
Custom exceptions for PyTorch Inference Framework.
"""

from typing import Any, Optional


class InferenceFrameworkError(Exception):
    """Base exception for PyTorch Inference Framework."""
    pass


class ConfigurationError(InferenceFrameworkError):
    """Configuration related errors."""
    pass


class ModelError(InferenceFrameworkError):
    """Model related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Model not found error."""
    pass


class ModelLoadError(ModelError):
    """Model loading error."""
    pass


class InferenceError(InferenceFrameworkError):
    """Inference related errors."""
    pass


class AudioProcessingError(InferenceFrameworkError):
    """Audio processing errors."""
    pass


class ValidationError(InferenceFrameworkError):
    """Data validation errors."""
    pass


class SecurityError(InferenceFrameworkError):
    """Security related errors."""
    pass


class ServiceUnavailableError(InferenceFrameworkError):
    """Service unavailable errors."""
    pass

"""
Custom exceptions for PyTorch Inference Framework.

This module provides a comprehensive error handling system with proper error 
context, logging, and graceful fallback support.
"""

import logging
from typing import Any, Optional, Dict, List
from datetime import datetime


logger = logging.getLogger(__name__)


class TorchInferenceError(Exception):
    """
    Base exception for torch-inference framework.
    
    Provides enhanced error context, automatic logging, and debugging information.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.cause = cause
        self.suggestions = suggestions or []
        self.timestamp = datetime.now()
        
        # Automatically log the error
        self._log_error()
    
    def _log_error(self):
        """Log the error with appropriate level and context."""
        log_data = {
            "error_code": self.error_code,
            "error_message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.cause:
            log_data["cause"] = str(self.cause)
        
        if self.suggestions:
            log_data["suggestions"] = self.suggestions
        
        logger.error(f"[{self.error_code}] {self.message}", extra={"error_details": log_data})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "context": self.context,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp.isoformat()
        }


class ModelLoadError(TorchInferenceError):
    """Raised when model loading fails."""
    
    def __init__(self, model_path: str, cause: Optional[Exception] = None, **kwargs):
        context = {"model_path": model_path}
        context.update(kwargs.get("context", {}))
        
        suggestions = [
            "Verify the model file exists and is accessible",
            "Check if the model format is compatible",
            "Ensure sufficient memory is available"
        ]
        
        if cause and "CUDA" in str(cause):
            suggestions.extend([
                "Try loading on CPU if GPU memory is insufficient",
                "Check CUDA compatibility and drivers"
            ])
        
        super().__init__(
            f"Failed to load model from {model_path}",
            error_code="MODEL_LOAD_ERROR",
            context=context,
            cause=cause,
            suggestions=suggestions
        )


class OptimizationError(TorchInferenceError):
    """Raised when model optimization fails."""
    
    def __init__(self, optimization_type: str, cause: Optional[Exception] = None, **kwargs):
        context = {"optimization_type": optimization_type}
        context.update(kwargs.get("context", {}))
        
        suggestions = [
            f"Check if {optimization_type} is properly installed and configured",
            "Verify model compatibility with the optimization technique",
            "Consider using fallback optimization or running without optimization"
        ]
        
        super().__init__(
            f"Model optimization failed: {optimization_type}",
            error_code="OPTIMIZATION_ERROR", 
            context=context,
            cause=cause,
            suggestions=suggestions
        )


class PredictionError(TorchInferenceError):
    """Raised when prediction fails."""
    
    def __init__(self, details: str, cause: Optional[Exception] = None, **kwargs):
        context = kwargs.get("context", {})
        
        suggestions = ["Check input data format and model compatibility"]
        
        if cause:
            if "out of memory" in str(cause).lower():
                suggestions.extend([
                    "Reduce batch size",
                    "Clear GPU cache",
                    "Use CPU inference if GPU memory is insufficient"
                ])
            elif "runtime error" in str(cause).lower():
                suggestions.extend([
                    "Verify input tensor shapes and types",
                    "Check model state and initialization"
                ])
        
        super().__init__(
            f"Prediction failed: {details}",
            error_code="PREDICTION_ERROR",
            context=context,
            cause=cause,
            suggestions=suggestions
        )


class ConfigurationError(TorchInferenceError):
    """Raised when configuration is invalid."""
    
    def __init__(self, config_field: str, details: str, **kwargs):
        context = {"config_field": config_field}
        context.update(kwargs.get("context", {}))
        
        suggestions = [
            f"Check the '{config_field}' configuration parameter",
            "Verify configuration file syntax and values",
            "Refer to documentation for valid configuration options"
        ]
        
        super().__init__(
            f"Configuration error in '{config_field}': {details}",
            error_code="CONFIGURATION_ERROR",
            context=context,
            suggestions=suggestions
        )


class DeviceError(TorchInferenceError):
    """Raised when device-related operations fail."""
    
    def __init__(self, device: str, operation: str, cause: Optional[Exception] = None, **kwargs):
        context = {"device": device, "operation": operation}
        context.update(kwargs.get("context", {}))
        
        suggestions = [
            f"Verify {device} is available and properly configured",
            "Check driver versions and compatibility"
        ]
        
        if "cuda" in device.lower():
            suggestions.extend([
                "Verify CUDA installation and drivers",
                "Check GPU memory availability",
                "Consider falling back to CPU"
            ])
        
        super().__init__(
            f"Device operation failed on {device}: {operation}",
            error_code="DEVICE_ERROR",
            context=context,
            cause=cause,
            suggestions=suggestions
        )


class AudioProcessingError(TorchInferenceError):
    """Raised when audio processing fails."""
    
    def __init__(self, operation: str, cause: Optional[Exception] = None, **kwargs):
        context = {"operation": operation}
        context.update(kwargs.get("context", {}))
        
        suggestions = [
            "Check audio file format and encoding",
            "Verify audio processing dependencies are installed",
            "Ensure audio data is in the correct format"
        ]
        
        super().__init__(
            f"Audio processing failed: {operation}",
            error_code="AUDIO_PROCESSING_ERROR",
            context=context,
            cause=cause,
            suggestions=suggestions
        )


class ValidationError(TorchInferenceError):
    """Raised when data validation fails."""
    
    def __init__(self, field: str, value: Any, expected: str, **kwargs):
        context = {"field": field, "value": str(value), "expected": expected}
        context.update(kwargs.get("context", {}))
        
        suggestions = [
            f"Ensure '{field}' matches the expected format: {expected}",
            "Check API documentation for correct parameter formats"
        ]
        
        super().__init__(
            f"Validation failed for '{field}': expected {expected}, got {type(value).__name__}",
            error_code="VALIDATION_ERROR",
            context=context,
            suggestions=suggestions
        )


class SecurityError(TorchInferenceError):
    """Raised when security-related operations fail."""
    
    def __init__(self, security_check: str, details: str, **kwargs):
        context = {"security_check": security_check}
        context.update(kwargs.get("context", {}))
        
        suggestions = [
            "Check authentication credentials",
            "Verify permissions and access rights",
            "Review security configuration"
        ]
        
        super().__init__(
            f"Security error in {security_check}: {details}",
            error_code="SECURITY_ERROR",
            context=context,
            suggestions=suggestions
        )


class ServiceUnavailableError(TorchInferenceError):
    """Raised when a service is unavailable."""
    
    def __init__(self, service: str, details: str, **kwargs):
        context = {"service": service}
        context.update(kwargs.get("context", {}))
        
        suggestions = [
            f"Check if {service} is running and accessible",
            "Verify network connectivity",
            "Try again later if this is a temporary issue"
        ]
        
        super().__init__(
            f"Service unavailable: {service} - {details}",
            error_code="SERVICE_UNAVAILABLE",
            context=context,
            suggestions=suggestions
        )


class ResourceExhaustedError(TorchInferenceError):
    """Raised when system resources are exhausted."""
    
    def __init__(self, resource: str, details: str, **kwargs):
        context = {"resource": resource}
        context.update(kwargs.get("context", {}))
        
        suggestions = [
            f"Free up {resource} resources",
            "Reduce batch size or model complexity",
            "Consider using a different device or configuration"
        ]
        
        super().__init__(
            f"Resource exhausted: {resource} - {details}",
            error_code="RESOURCE_EXHAUSTED",
            context=context,
            suggestions=suggestions
        )


class TimeoutError(TorchInferenceError):
    """Raised when operations timeout."""
    
    def __init__(self, operation: str, timeout_seconds: float, **kwargs):
        context = {"operation": operation, "timeout_seconds": timeout_seconds}
        context.update(kwargs.get("context", {}))
        
        suggestions = [
            f"Increase timeout for {operation}",
            "Optimize the operation for better performance",
            "Check for blocking operations or deadlocks"
        ]
        
        super().__init__(
            f"Operation timed out: {operation} (timeout: {timeout_seconds}s)",
            error_code="TIMEOUT_ERROR",
            context=context,
            suggestions=suggestions
        )


# Legacy exception classes for backward compatibility
class InferenceFrameworkError(TorchInferenceError):
    """Legacy base exception - use TorchInferenceError instead."""
    pass


class ModelError(TorchInferenceError):
    """Legacy model error - use ModelLoadError or specific error instead."""
    pass


class ModelNotFoundError(ModelLoadError):
    """Legacy model not found error."""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(
            model_path=model_name,
            message=f"Model '{model_name}' not found",
            **kwargs
        )


class InferenceError(PredictionError):
    """Legacy inference error - use PredictionError instead."""
    pass


class ProcessingError(TorchInferenceError):
    """Legacy processing error."""
    pass


class NotFoundError(TorchInferenceError):
    """Legacy not found error."""
    pass


class InternalServerError(TorchInferenceError):
    """Legacy internal server error."""
    pass

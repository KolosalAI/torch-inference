"""
Enhanced API schemas and models for PyTorch Inference Framework.

This module provides comprehensive Pydantic models for API requests and responses
with proper validation, error handling, and standardized formats.
"""

from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator
import uuid


class RequestStatus(str, Enum):
    """Request processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class ModelType(str, Enum):
    """Supported model types."""
    TEXT_TO_SPEECH = "text-to-speech"
    SPEECH_TO_TEXT = "speech-to-text"
    TEXT_CLASSIFICATION = "text-classification"
    IMAGE_CLASSIFICATION = "image-classification"
    FEATURE_EXTRACTION = "feature-extraction"
    OBJECT_DETECTION = "object-detection"
    CUSTOM = "custom"


class AudioFormat(str, Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


class BaseRequest(BaseModel):
    """Base request model with common fields."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token: Optional[str] = Field(default=None, description="Authentication token")
    priority: int = Field(default=0, ge=-10, le=10, description="Request priority (-10 to 10)")
    timeout: Optional[float] = Field(default=None, gt=0, le=300, description="Timeout in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        validate_assignment = True


class BaseResponse(BaseModel):
    """Base response model with standard fields."""
    request_id: str
    status: RequestStatus
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error details if any")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    
    class Config:
        validate_assignment = True
        use_enum_values = True


class PredictionRequest(BaseRequest):
    """Request model for general prediction endpoint."""
    model_name: str = Field(..., description="Name of the model to use")
    inputs: Union[Any, List[Any]] = Field(..., description="Input data for prediction")
    enable_batching: bool = Field(default=True, description="Enable batch processing optimization")
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model name format."""
        if not v or not isinstance(v, str):
            raise ValueError("Model name must be a non-empty string")
        return v.strip()
    
    @validator('inputs')
    def validate_inputs(cls, v):
        """Validate input data."""
        if v is None:
            raise ValueError("Inputs cannot be None")
        return v


class PredictionResponse(BaseResponse):
    """Response model for prediction endpoint."""
    result: Optional[Union[Any, List[Any]]] = Field(default=None, description="Prediction results")
    model_info: Optional[Dict[str, Any]] = Field(default=None, description="Model information")
    batch_info: Optional[Dict[str, Any]] = Field(default=None, description="Batch processing information")
    performance_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Performance metrics")


class TTSRequest(BaseRequest):
    """Request model for Text-to-Speech synthesis."""
    model_name: str = Field(..., description="TTS model to use")
    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    voice: Optional[str] = Field(default=None, description="Voice to use for synthesis")
    language: str = Field(default="en", description="Language code")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed multiplier")
    pitch: float = Field(default=1.0, ge=0.25, le=4.0, description="Pitch adjustment multiplier")
    volume: float = Field(default=1.0, ge=0.1, le=2.0, description="Volume level multiplier")
    emotion: Optional[str] = Field(default=None, description="Emotion for synthesis")
    output_format: AudioFormat = Field(default=AudioFormat.WAV, description="Output audio format")
    sample_rate: Optional[int] = Field(default=None, gt=0, le=96000, description="Output sample rate")
    
    # Advanced options
    enable_noise_reduction: bool = Field(default=True, description="Enable noise reduction")
    enable_enhancement: bool = Field(default=True, description="Enable audio enhancement")
    streaming: bool = Field(default=False, description="Enable streaming response")
    
    @validator('text')
    def validate_text(cls, v):
        """Validate input text."""
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()
    
    @validator('language')
    def validate_language(cls, v):
        """Validate language code."""
        valid_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
        if v not in valid_languages:
            raise ValueError(f"Language must be one of: {', '.join(valid_languages)}")
        return v


class TTSResponse(BaseResponse):
    """Response model for TTS synthesis."""
    audio_data: Optional[str] = Field(default=None, description="Base64 encoded audio data")
    audio_url: Optional[str] = Field(default=None, description="URL to download audio file")
    audio_format: Optional[AudioFormat] = None
    duration: Optional[float] = Field(default=None, description="Audio duration in seconds")
    sample_rate: Optional[int] = Field(default=None, description="Audio sample rate")
    file_size: Optional[int] = Field(default=None, description="Audio file size in bytes")
    model_info: Optional[Dict[str, Any]] = Field(default=None, description="Model information")
    synthesis_info: Optional[Dict[str, Any]] = Field(default=None, description="Synthesis details")


class STTRequest(BaseRequest):
    """Request model for Speech-to-Text transcription."""
    model_name: str = Field(default="whisper-base", description="STT model to use")
    language: str = Field(default="auto", description="Language code or 'auto' for detection")
    
    # Transcription options
    enable_timestamps: bool = Field(default=True, description="Include word-level timestamps")
    enable_word_confidence: bool = Field(default=True, description="Include word confidence scores")
    enable_speaker_diarization: bool = Field(default=False, description="Enable speaker identification")
    
    # Advanced options
    beam_size: int = Field(default=5, ge=1, le=20, description="Beam search size")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")
    suppress_blank: bool = Field(default=True, description="Suppress blank outputs")
    suppress_tokens: Optional[List[int]] = Field(default=None, description="Token IDs to suppress")
    initial_prompt: Optional[str] = Field(default=None, max_length=500, description="Initial context prompt")
    
    # Audio processing options
    enable_vad: bool = Field(default=True, description="Enable voice activity detection")
    enable_noise_reduction: bool = Field(default=True, description="Enable noise reduction")
    normalize_audio: bool = Field(default=True, description="Normalize audio levels")


class STTResponse(BaseResponse):
    """Response model for STT transcription."""
    text: Optional[str] = Field(default=None, description="Transcribed text")
    segments: Optional[List[Dict[str, Any]]] = Field(default=None, description="Detailed segments with timestamps")
    language: Optional[str] = Field(default=None, description="Detected or specified language")
    language_confidence: Optional[float] = Field(default=None, description="Language detection confidence")
    
    # Audio information
    duration: Optional[float] = Field(default=None, description="Audio duration in seconds")
    sample_rate: Optional[int] = Field(default=None, description="Audio sample rate")
    
    # Transcription quality metrics
    overall_confidence: Optional[float] = Field(default=None, description="Overall transcription confidence")
    word_count: Optional[int] = Field(default=None, description="Number of words transcribed")
    
    # Model and processing info
    model_info: Optional[Dict[str, Any]] = Field(default=None, description="Model information")
    processing_info: Optional[Dict[str, Any]] = Field(default=None, description="Processing details")


class HealthResponse(BaseResponse):
    """Response model for health check."""
    healthy: bool = Field(..., description="Overall health status")
    checks: Dict[str, Any] = Field(..., description="Individual health checks")
    uptime: Optional[float] = Field(default=None, description="System uptime in seconds")
    version: Optional[str] = Field(default=None, description="System version")
    environment: Optional[str] = Field(default=None, description="Environment name")


class ModelInfo(BaseModel):
    """Model information schema."""
    name: str = Field(..., description="Model name")
    display_name: str = Field(..., description="Human-readable model name")
    description: Optional[str] = Field(default=None, description="Model description")
    model_type: ModelType = Field(..., description="Type of model")
    version: Optional[str] = Field(default=None, description="Model version")
    
    # Capabilities
    supported_languages: List[str] = Field(default_factory=list, description="Supported languages")
    supported_formats: List[str] = Field(default_factory=list, description="Supported input/output formats")
    max_input_length: Optional[int] = Field(default=None, description="Maximum input length")
    
    # Status and performance
    status: str = Field(default="available", description="Model availability status")
    loaded: bool = Field(default=False, description="Whether model is currently loaded")
    load_time: Optional[float] = Field(default=None, description="Model load time in seconds")
    memory_usage_mb: Optional[float] = Field(default=None, description="Memory usage in MB")
    
    # Usage statistics
    total_requests: int = Field(default=0, description="Total number of requests processed")
    last_used: Optional[datetime] = Field(default=None, description="Last usage timestamp")
    average_processing_time: Optional[float] = Field(default=None, description="Average processing time")
    
    # Configuration
    optimization_enabled: bool = Field(default=False, description="Whether optimizations are enabled")
    optimization_info: Dict[str, Any] = Field(default_factory=dict, description="Optimization details")
    
    class Config:
        validate_assignment = True
        use_enum_values = True


class ModelListResponse(BaseResponse):
    """Response model for listing models."""
    models: List[ModelInfo] = Field(..., description="List of available models")
    total_models: int = Field(..., description="Total number of models")
    categories: Dict[str, List[str]] = Field(default_factory=dict, description="Models grouped by category")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Overall statistics")


class SystemInfoResponse(BaseResponse):
    """Response model for system information."""
    system: Dict[str, Any] = Field(..., description="System information")
    hardware: Dict[str, Any] = Field(..., description="Hardware information")
    software: Dict[str, Any] = Field(..., description="Software information")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")
    configuration: Dict[str, Any] = Field(..., description="System configuration")


class ErrorDetail(BaseModel):
    """Detailed error information."""
    error_code: str = Field(..., description="Specific error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[str] = Field(default=None, description="Additional error details")
    context: Dict[str, Any] = Field(default_factory=dict, description="Error context")
    suggestions: List[str] = Field(default_factory=list, description="Suggested fixes")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        validate_assignment = True


class ValidationErrorResponse(BaseResponse):
    """Response model for validation errors."""
    validation_errors: List[Dict[str, Any]] = Field(..., description="List of validation errors")
    invalid_fields: List[str] = Field(..., description="Names of invalid fields")


class FileUploadRequest(BaseModel):
    """Request model for file uploads."""
    file_type: str = Field(..., description="Type of file being uploaded")
    max_size_mb: int = Field(default=100, description="Maximum file size in MB")
    allowed_extensions: List[str] = Field(default_factory=list, description="Allowed file extensions")
    
    @validator('file_type')
    def validate_file_type(cls, v):
        """Validate file type."""
        valid_types = ['audio', 'image', 'text', 'model']
        if v not in valid_types:
            raise ValueError(f"File type must be one of: {', '.join(valid_types)}")
        return v


class FileUploadResponse(BaseResponse):
    """Response model for file uploads."""
    file_id: str = Field(..., description="Unique file identifier")
    file_name: str = Field(..., description="Original file name")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="Detected file type")
    content_type: str = Field(..., description="MIME content type")
    upload_url: Optional[str] = Field(default=None, description="URL where file was uploaded")
    expires_at: Optional[datetime] = Field(default=None, description="File expiration time")


class BatchRequest(BaseRequest):
    """Request model for batch operations."""
    operation: str = Field(..., description="Type of batch operation")
    items: List[Dict[str, Any]] = Field(..., min_items=1, max_items=1000, description="Batch items")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    fail_fast: bool = Field(default=False, description="Stop on first error")
    
    @validator('items')
    def validate_items(cls, v):
        """Validate batch items."""
        if not v:
            raise ValueError("Batch must contain at least one item")
        return v


class BatchResponse(BaseResponse):
    """Response model for batch operations."""
    total_items: int = Field(..., description="Total number of items processed")
    successful_items: int = Field(..., description="Number of successful items")
    failed_items: int = Field(..., description="Number of failed items")
    results: List[Dict[str, Any]] = Field(..., description="Individual item results")
    summary: Dict[str, Any] = Field(..., description="Batch processing summary")


class MetricsResponse(BaseResponse):
    """Response model for metrics and monitoring."""
    metrics: Dict[str, Any] = Field(..., description="System metrics")
    performance: Dict[str, Any] = Field(..., description="Performance statistics")
    resource_usage: Dict[str, Any] = Field(..., description="Resource utilization")
    alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Active alerts")


# Utility functions for response creation

def create_success_response(
    request_id: str,
    result: Any,
    processing_time: Optional[float] = None,
    **kwargs
) -> PredictionResponse:
    """Create a standardized success response."""
    return PredictionResponse(
        request_id=request_id,
        status=RequestStatus.SUCCESS,
        result=result,
        processing_time=processing_time,
        **kwargs
    )


def create_error_response(
    request_id: str,
    error: Exception,
    processing_time: Optional[float] = None,
    **kwargs
) -> PredictionResponse:
    """Create a standardized error response."""
    from .exceptions import TorchInferenceError
    
    if isinstance(error, TorchInferenceError):
        error_detail = error.to_dict()
    else:
        error_detail = {
            "error": "UNKNOWN_ERROR",
            "message": str(error),
            "type": type(error).__name__
        }
    
    return PredictionResponse(
        request_id=request_id,
        status=RequestStatus.ERROR,
        error=error_detail,
        processing_time=processing_time,
        **kwargs
    )


def create_health_response(
    checks: Dict[str, Any],
    healthy: bool = None,
    **kwargs
) -> HealthResponse:
    """Create a standardized health response."""
    if healthy is None:
        healthy = all(
            check.get("healthy", False) if isinstance(check, dict) else bool(check)
            for check in checks.values()
        )
    
    return HealthResponse(
        request_id=str(uuid.uuid4()),
        status=RequestStatus.SUCCESS,
        healthy=healthy,
        checks=checks,
        **kwargs
    )


def validate_request_model(request_data: Dict[str, Any], model_class: BaseModel) -> BaseModel:
    """
    Validate request data against a Pydantic model.
    
    Args:
        request_data: Request data dictionary
        model_class: Pydantic model class to validate against
        
    Returns:
        Validated model instance
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        return model_class(**request_data)
    except Exception as e:
        from .exceptions import ValidationError
        raise ValidationError(
            field="request_data",
            value=request_data,
            expected=model_class.__name__,
            context={"validation_error": str(e)}
        )

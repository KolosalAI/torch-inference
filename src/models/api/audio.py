"""
Audio processing API models.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    """Request model for Text-to-Speech synthesis."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to synthesize (1-5000 characters)"
    )
    model_name: str = Field(default="default", description="TTS model to use")
    voice: Optional[str] = Field(default=None, description="Voice to use for synthesis")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed (0.5-2.0)")
    pitch: float = Field(default=1.0, ge=0.5, le=2.0, description="Pitch adjustment (0.5-2.0)")
    volume: float = Field(default=1.0, ge=0.1, le=2.0, description="Volume level (0.1-2.0)")
    language: str = Field(default="en", description="Language code (e.g., 'en', 'es', 'fr')")
    emotion: Optional[str] = Field(default=None, description="Emotion for synthesis")
    output_format: str = Field(default="wav", pattern="^(wav|mp3|flac)$", description="Output audio format")


class TTSResponse(BaseModel):
    """Response model for TTS synthesis."""
    success: bool
    audio_data: Optional[str] = Field(default=None, description="Base64 encoded audio data")
    audio_format: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    processing_time: Optional[float] = None
    model_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class STTRequest(BaseModel):
    """Request model for Speech-to-Text transcription."""
    model_name: str = Field(default="whisper-base", description="STT model to use")
    language: str = Field(default="auto", description="Language code or 'auto' for detection")
    enable_timestamps: bool = Field(default=True, description="Include word-level timestamps")
    beam_size: int = Field(default=5, ge=1, le=10, description="Beam search size")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")
    suppress_blank: bool = Field(default=True, description="Suppress blank outputs")
    initial_prompt: Optional[str] = Field(default=None, description="Initial prompt for context")


class STTResponse(BaseModel):
    """Response model for STT transcription."""
    success: bool
    text: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = Field(default=None, description="Transcription segments with timestamps")
    language: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    model_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AudioValidationRequest(BaseModel):
    """Request model for audio file validation."""
    file_path: str = Field(..., description="Path to audio file")
    validate_format: bool = Field(default=True, description="Validate audio format")
    check_integrity: bool = Field(default=True, description="Check file integrity")


class AudioValidationResponse(BaseModel):
    """Response model for audio validation."""
    valid: bool = Field(..., description="Whether audio file is valid")
    file_path: str = Field(..., description="Path to validated file")
    format: Optional[str] = Field(None, description="Detected audio format")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")
    sample_rate: Optional[int] = Field(None, description="Audio sample rate")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")
    error: Optional[str] = Field(None, description="Validation error message")


class AudioProcessRequest(BaseModel):
    """Request model for audio processing operations."""
    operations: Dict[str, Any] = Field(..., description="Audio processing operations to apply")
    output_format: str = Field(default="wav", description="Output audio format")
    quality: Optional[str] = Field(None, description="Output quality setting")
    normalize: bool = Field(default=False, description="Normalize audio levels")

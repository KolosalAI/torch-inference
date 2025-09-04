"""
Domain model for audio processing.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


class TTSVoice(Enum):
    """Available TTS voices."""
    MALE_1 = "male_1"
    FEMALE_1 = "female_1"
    MALE_2 = "male_2"
    FEMALE_2 = "female_2"
    NEUTRAL = "neutral"


class AudioQuality(Enum):
    """Audio quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class AudioMetadata:
    """Audio file metadata."""
    sample_rate: int
    channels: int
    duration: float  # in seconds
    bit_depth: int
    format: AudioFormat
    size_bytes: int
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TTSRequest:
    """Text-to-Speech request."""
    id: str
    text: str
    model_name: str = "default"
    voice: Optional[TTSVoice] = None
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    language: str = "en"
    emotion: Optional[str] = None
    output_format: AudioFormat = AudioFormat.WAV
    quality: AudioQuality = AudioQuality.HIGH
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TTSResponse:
    """Text-to-Speech response."""
    request_id: str
    audio_data: bytes
    metadata: AudioMetadata
    processing_time: float
    model_info: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass
class STTRequest:
    """Speech-to-Text request."""
    id: str
    audio_data: bytes
    model_name: str = "whisper-base"
    language: str = "auto"
    enable_timestamps: bool = True
    beam_size: int = 5
    temperature: float = 0.0
    suppress_blank: bool = True
    initial_prompt: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class STTSegment:
    """Speech-to-Text segment with timestamp."""
    text: str
    start: float
    end: float
    confidence: float


@dataclass
class STTResponse:
    """Speech-to-Text response."""
    request_id: str
    text: str
    segments: List[STTSegment]
    language: str
    confidence: float
    processing_time: float
    model_info: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass
class AudioModelInfo:
    """Audio model information."""
    name: str
    type: str  # tts, stt
    supported_languages: List[str]
    supported_voices: List[TTSVoice] = field(default_factory=list)
    supported_formats: List[AudioFormat] = field(default_factory=list)
    model_size_mb: float = 0.0
    sample_rate: int = 22050
    is_loaded: bool = False
    loaded_at: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioProcessingMetrics:
    """Audio processing performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    total_audio_generated_seconds: float = 0.0
    total_audio_transcribed_seconds: float = 0.0
    real_time_factor: float = 0.0  # processing_time / audio_duration
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_tts_request(self, processing_time: float, audio_duration: float, success: bool):
        """Add a TTS request to metrics."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.total_processing_time += processing_time
            self.total_audio_generated_seconds += audio_duration
            self.average_processing_time = self.total_processing_time / self.successful_requests
            if audio_duration > 0:
                self.real_time_factor = processing_time / audio_duration
        else:
            self.failed_requests += 1
        self.last_updated = datetime.now()
    
    def add_stt_request(self, processing_time: float, audio_duration: float, success: bool):
        """Add a STT request to metrics."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.total_processing_time += processing_time
            self.total_audio_transcribed_seconds += audio_duration
            self.average_processing_time = self.total_processing_time / self.successful_requests
            if audio_duration > 0:
                self.real_time_factor = processing_time / audio_duration
        else:
            self.failed_requests += 1
        self.last_updated = datetime.now()

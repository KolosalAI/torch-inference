"""
Base classes for audio models in the PyTorch inference framework.

This module provides the foundation for audio processing models including
Text-to-Speech (TTS) and Speech-to-Text (STT) implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import logging
import numpy as np
from dataclasses import dataclass
from enum import Enum

from ...core.base_model import BaseModel, ModelMetadata
from ...core.config import InferenceConfig

logger = logging.getLogger(__name__)


class AudioModelType(Enum):
    """Types of audio models."""
    TTS = "text_to_speech"
    STT = "speech_to_text"
    AUDIO_CLASSIFICATION = "audio_classification"
    AUDIO_ENHANCEMENT = "audio_enhancement"
    VOICE_CONVERSION = "voice_conversion"


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    M4A = "m4a"
    OGG = "ogg"


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 16000
    chunk_duration: float = 30.0  # seconds
    overlap: float = 5.0  # seconds
    enable_vad: bool = True  # Voice Activity Detection
    supported_formats: List[str] = None
    max_audio_length: float = 300.0  # seconds
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["wav", "mp3", "flac", "m4a"]


@dataclass
class TTSConfig:
    """Configuration for Text-to-Speech models."""
    voice: str = "default"
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    language: str = "en"
    emotion: Optional[str] = None
    output_format: str = "wav"
    quality: str = "high"  # low, medium, high


@dataclass
class STTConfig:
    """Configuration for Speech-to-Text models."""
    language: str = "en"
    enable_timestamps: bool = True
    beam_size: int = 5
    temperature: float = 0.0
    suppress_blank: bool = True
    suppress_tokens: List[int] = None
    initial_prompt: Optional[str] = None
    condition_on_previous_text: bool = True
    
    def __post_init__(self):
        if self.suppress_tokens is None:
            self.suppress_tokens = [-1]


@dataclass
class AudioMetadata(ModelMetadata):
    """Extended metadata for audio models."""
    audio_model_type: AudioModelType = AudioModelType.STT
    sample_rate: int = 16000
    channels: int = 1
    max_duration: float = 300.0
    supported_languages: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.supported_languages is None:
            self.supported_languages = ["en"]


class AudioModelError(Exception):
    """Exception raised for audio model specific errors."""
    pass


class BaseAudioModel(BaseModel):
    """
    Abstract base class for all audio model implementations.
    
    This class extends BaseModel with audio-specific functionality for
    processing audio data, handling different audio formats, and managing
    audio-specific configurations.
    """
    
    def __init__(self, config: InferenceConfig, audio_config: Optional[AudioConfig] = None):
        super().__init__(config)
        self.audio_config = audio_config or AudioConfig()
        self.audio_metadata: Optional[AudioMetadata] = None
        
        # Audio processing components (will be initialized later)
        self._audio_processor = None
        self._feature_extractor = None
        
        # Setup audio-specific logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def get_audio_model_type(self) -> AudioModelType:
        """Get the type of audio model."""
        pass
    
    @abstractmethod
    def process_audio_input(self, audio_input: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Process audio input into the format expected by the model.
        
        Args:
            audio_input: Audio file path, numpy array, or tensor
            
        Returns:
            Processed audio tensor
        """
        pass
    
    @abstractmethod
    def process_audio_output(self, model_output: torch.Tensor) -> Any:
        """
        Process model output into the desired format.
        
        Args:
            model_output: Raw model output tensor
            
        Returns:
            Processed output (text, audio array, etc.)
        """
        pass
    
    def validate_audio_input(self, audio_input: Any) -> bool:
        """
        Validate that the audio input is in the correct format.
        
        Args:
            audio_input: Audio input to validate
            
        Returns:
            True if valid, False otherwise
        """
        if isinstance(audio_input, str):
            # Check if file exists and has valid extension
            path = Path(audio_input)
            if not path.exists():
                return False
            return path.suffix.lower().lstrip('.') in self.audio_config.supported_formats
        
        elif isinstance(audio_input, np.ndarray):
            # Check array properties
            if audio_input.ndim not in [1, 2]:
                return False
            if audio_input.ndim == 2 and audio_input.shape[0] > 2:  # Max 2 channels
                return False
            return True
        
        elif isinstance(audio_input, torch.Tensor):
            # Check tensor properties
            if audio_input.ndim not in [1, 2, 3]:
                return False
            return True
        
        return False
    
    def get_audio_duration(self, audio_input: Union[str, np.ndarray, torch.Tensor]) -> float:
        """
        Get the duration of audio input in seconds.
        
        Args:
            audio_input: Audio input
            
        Returns:
            Duration in seconds
        """
        if isinstance(audio_input, str):
            # Use librosa to get duration
            try:
                import librosa
                return librosa.get_duration(path=audio_input)
            except ImportError:
                self.logger.warning("librosa not available, cannot get audio duration from file")
                return 0.0
        
        elif isinstance(audio_input, np.ndarray):
            return len(audio_input) / self.audio_config.sample_rate
        
        elif isinstance(audio_input, torch.Tensor):
            # Assume last dimension is time
            time_dim = audio_input.shape[-1]
            return time_dim / self.audio_config.sample_rate
        
        return 0.0
    
    def chunk_audio(self, audio_input: Union[np.ndarray, torch.Tensor], 
                   chunk_duration: Optional[float] = None) -> List[Union[np.ndarray, torch.Tensor]]:
        """
        Split audio into chunks for processing.
        
        Args:
            audio_input: Audio array or tensor
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            List of audio chunks
        """
        if chunk_duration is None:
            chunk_duration = self.audio_config.chunk_duration
        
        sample_rate = self.audio_config.sample_rate
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(self.audio_config.overlap * sample_rate)
        
        chunks = []
        start = 0
        
        while start < len(audio_input):
            end = min(start + chunk_samples, len(audio_input))
            chunk = audio_input[start:end]
            chunks.append(chunk)
            
            if end >= len(audio_input):
                break
            
            start = end - overlap_samples
        
        return chunks
    
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """
        Preprocess inputs for audio models.
        
        Args:
            inputs: Audio input (file path, array, tensor, or text)
            
        Returns:
            Preprocessed tensor
        """
        try:
            # Validate input
            if not self.validate_audio_input(inputs):
                raise AudioModelError(f"Invalid audio input: {type(inputs)}")
            
            # Check audio duration
            duration = self.get_audio_duration(inputs)
            if duration > self.audio_config.max_audio_length:
                self.logger.warning(f"Audio duration ({duration}s) exceeds maximum ({self.audio_config.max_audio_length}s)")
            
            # Process audio input
            processed_audio = self.process_audio_input(inputs)
            
            # Move to device
            if isinstance(processed_audio, torch.Tensor):
                processed_audio = processed_audio.to(self.device)
                
                # Ensure proper shape for batching
                if processed_audio.ndim == 1:
                    processed_audio = processed_audio.unsqueeze(0)  # Add batch dimension
            
            return processed_audio
            
        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            raise AudioModelError(f"Audio preprocessing failed: {e}") from e
    
    def postprocess(self, outputs: torch.Tensor) -> Any:
        """
        Postprocess model outputs for audio models.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Processed outputs
        """
        try:
            return self.process_audio_output(outputs)
        except Exception as e:
            self.logger.error(f"Audio postprocessing failed: {e}")
            raise AudioModelError(f"Audio postprocessing failed: {e}") from e
    
    def _create_dummy_input(self) -> torch.Tensor:
        """Create dummy audio input for warmup."""
        # Create dummy audio tensor (1 second of silence)
        dummy_samples = int(self.audio_config.sample_rate * 1.0)
        dummy_audio = torch.zeros(1, dummy_samples, device=self.device)
        return dummy_audio
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get extended model information including audio-specific details."""
        info = super().model_info
        
        # Add audio-specific information
        info.update({
            "audio_model_type": self.get_audio_model_type().value,
            "audio_config": {
                "sample_rate": self.audio_config.sample_rate,
                "chunk_duration": self.audio_config.chunk_duration,
                "supported_formats": self.audio_config.supported_formats,
                "max_audio_length": self.audio_config.max_audio_length
            }
        })
        
        if self.audio_metadata:
            info["audio_metadata"] = {
                "sample_rate": self.audio_metadata.sample_rate,
                "channels": self.audio_metadata.channels,
                "max_duration": self.audio_metadata.max_duration,
                "supported_languages": self.audio_metadata.supported_languages
            }
        
        return info


class BaseTTSModel(BaseAudioModel):
    """
    Base class for Text-to-Speech models.
    """
    
    def __init__(self, config: InferenceConfig, audio_config: Optional[AudioConfig] = None,
                 tts_config: Optional[TTSConfig] = None):
        super().__init__(config, audio_config)
        self.tts_config = tts_config or TTSConfig()
    
    def get_audio_model_type(self) -> AudioModelType:
        return AudioModelType.TTS
    
    @abstractmethod
    def synthesize_speech(self, text: str, **kwargs) -> np.ndarray:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            **kwargs: Additional synthesis parameters
            
        Returns:
            Audio array
        """
        pass
    
    def process_audio_input(self, audio_input: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        For TTS models, audio_input is actually text.
        """
        if isinstance(audio_input, str):
            # Text input - this is normal for TTS
            # Convert text to tokens/embeddings (model-specific implementation)
            return self._text_to_tensor(audio_input)
        else:
            raise AudioModelError("TTS models expect text input, not audio")
    
    def process_audio_output(self, model_output: torch.Tensor) -> np.ndarray:
        """
        Convert model output to audio array.
        """
        # Convert tensor to numpy array
        if isinstance(model_output, torch.Tensor):
            audio_array = model_output.detach().cpu().numpy()
        else:
            audio_array = model_output
        
        # Ensure proper shape (samples,) or (channels, samples)
        if audio_array.ndim > 2:
            audio_array = audio_array.squeeze()
        
        return audio_array
    
    @abstractmethod
    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """Convert text to tensor representation."""
        pass
    
    def predict(self, inputs: str) -> Dict[str, Any]:
        """
        Predict audio from text input.
        
        Args:
            inputs: Text to synthesize
            
        Returns:
            Dictionary containing audio data and metadata
        """
        # Generate audio
        audio_array = self.synthesize_speech(inputs)
        
        return {
            "audio": audio_array,
            "sample_rate": self.audio_config.sample_rate,
            "duration": len(audio_array) / self.audio_config.sample_rate,
            "text": inputs,
            "voice": self.tts_config.voice,
            "format": self.tts_config.output_format
        }


class BaseSTTModel(BaseAudioModel):
    """
    Base class for Speech-to-Text models.
    """
    
    def __init__(self, config: InferenceConfig, audio_config: Optional[AudioConfig] = None,
                 stt_config: Optional[STTConfig] = None):
        super().__init__(config, audio_config)
        self.stt_config = stt_config or STTConfig()
    
    def get_audio_model_type(self) -> AudioModelType:
        return AudioModelType.STT
    
    @abstractmethod
    def transcribe_audio(self, audio: Union[str, np.ndarray, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio input
            **kwargs: Additional transcription parameters
            
        Returns:
            Transcription results with text and optional timestamps
        """
        pass
    
    def process_audio_input(self, audio_input: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Process audio input for STT models.
        """
        if isinstance(audio_input, str):
            # Load audio file
            audio_array = self._load_audio_file(audio_input)
            audio_tensor = torch.from_numpy(audio_array).float()
        elif isinstance(audio_input, np.ndarray):
            audio_tensor = torch.from_numpy(audio_input).float()
        elif isinstance(audio_input, torch.Tensor):
            audio_tensor = audio_input.float()
        else:
            raise AudioModelError(f"Unsupported audio input type: {type(audio_input)}")
        
        # Resample if necessary
        if hasattr(self, '_target_sample_rate'):
            audio_tensor = self._resample_audio(audio_tensor, self._target_sample_rate)
        
        return audio_tensor
    
    def process_audio_output(self, model_output: torch.Tensor) -> Dict[str, Any]:
        """
        Convert model output to transcription result.
        """
        # This is model-specific and should be implemented in subclasses
        return {"text": "", "confidence": 0.0}
    
    def _load_audio_file(self, file_path: str) -> np.ndarray:
        """Load audio file using available libraries."""
        try:
            import librosa
            audio, sr = librosa.load(file_path, sr=self.audio_config.sample_rate)
            return audio
        except ImportError:
            try:
                import soundfile as sf
                audio, sr = sf.read(file_path)
                if sr != self.audio_config.sample_rate:
                    # Basic resampling (not ideal, but works)
                    import scipy.signal
                    audio = scipy.signal.resample(
                        audio, 
                        int(len(audio) * self.audio_config.sample_rate / sr)
                    )
                return audio
            except ImportError:
                raise AudioModelError("No audio library available (librosa or soundfile required)")
    
    def _resample_audio(self, audio: torch.Tensor, target_sr: int) -> torch.Tensor:
        """Resample audio tensor to target sample rate."""
        try:
            import torchaudio.functional as F
            current_sr = self.audio_config.sample_rate
            if current_sr != target_sr:
                audio = F.resample(audio, current_sr, target_sr)
            return audio
        except ImportError:
            self.logger.warning("torchaudio not available, skipping resampling")
            return audio
    
    def predict(self, inputs: Union[str, np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Predict text from audio input.
        
        Args:
            inputs: Audio input (file path, array, or tensor)
            
        Returns:
            Dictionary containing transcription results
        """
        return self.transcribe_audio(inputs)

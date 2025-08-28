"""
Audio preprocessing pipeline for the PyTorch inference framework.

This module provides comprehensive audio preprocessing capabilities including:
- Audio format conversion and loading
- Resampling and sample rate conversion  
- Normalization and scaling
- Voice Activity Detection (VAD)
- Audio chunking and segmentation
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import numpy as np
import torch
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AudioPreprocessorConfig:
    """Configuration for audio preprocessing."""
    
    sample_rate: int = 16000
    n_mels: int = 80
    hop_length: int = 512
    win_length: Optional[int] = None
    n_fft: int = 1024
    normalize: bool = True
    normalization_method: str = "peak"  # "peak", "rms", "lufs"
    enable_vad: bool = True
    vad_aggressiveness: int = 1
    chunk_duration: Optional[float] = None
    overlap_duration: float = 5.0
    frame_duration: float = 0.03
    target_level: float = 0.95
    mono: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.win_length is None:
            self.win_length = self.n_fft
        if self.normalization_method not in ["peak", "rms", "lufs"]:
            raise ValueError("normalization_method must be one of: peak, rms, lufs")
        if not 0 <= self.vad_aggressiveness <= 3:
            raise ValueError("vad_aggressiveness must be between 0 and 3")
        if self.target_level <= 0 or self.target_level > 1:
            raise ValueError("target_level must be between 0 and 1")
        if self.frame_duration <= 0:
            raise ValueError("frame_duration must be positive")
        if self.overlap_duration < 0:
            raise ValueError("overlap_duration must be non-negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
            'n_fft': self.n_fft,
            'normalize': self.normalize,
            'normalization_method': self.normalization_method,
            'enable_vad': self.enable_vad,
            'vad_aggressiveness': self.vad_aggressiveness,
            'chunk_duration': self.chunk_duration,
            'overlap_duration': self.overlap_duration,
            'frame_duration': self.frame_duration,
            'target_level': self.target_level,
            'mono': self.mono
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AudioPreprocessorConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


class AudioPreprocessorError(Exception):
    """Exception raised for audio preprocessing errors."""
    pass


class BaseAudioPreprocessor(ABC):
    """Base class for audio preprocessors."""
    
    def __init__(self, sample_rate: int = 16000, normalize: bool = True):
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def process(self, audio: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Process audio input."""
        pass


class AudioLoader:
    """Audio file loader with support for multiple formats."""
    
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self.logger = logging.getLogger(f"{__name__}.AudioLoader")
        
        # Check available audio libraries
        self._check_audio_libraries()
    
    def _check_audio_libraries(self):
        """Check which audio libraries are available."""
        self.has_librosa = False
        self.has_soundfile = False
        self.has_torchaudio = False
        
        try:
            import librosa
            self.has_librosa = True
        except ImportError:
            pass
        
        try:
            import soundfile
            self.has_soundfile = True
        except ImportError:
            pass
        
        try:
            import torchaudio
            self.has_torchaudio = True
        except ImportError:
            pass
        
        if not any([self.has_librosa, self.has_soundfile, self.has_torchaudio]):
            self.logger.warning("No audio libraries found. Install librosa, soundfile, or torchaudio for audio loading.")
    
    def load_audio(self, file_path: Union[str, Path], 
                   sample_rate: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file with automatic format detection.
        
        Args:
            file_path: Path to audio file
            sample_rate: Target sample rate (None to use file's original rate)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise AudioPreprocessorError(f"Audio file not found: {file_path}")
        
        target_sr = sample_rate or self.target_sample_rate
        
        # Try loading with available libraries
        if self.has_librosa:
            return self._load_with_librosa(file_path, target_sr)
        elif self.has_soundfile:
            return self._load_with_soundfile(file_path, target_sr)
        elif self.has_torchaudio:
            return self._load_with_torchaudio(file_path, target_sr)
        else:
            raise AudioPreprocessorError("No audio loading library available")
    
    def _load_with_librosa(self, file_path: Path, sample_rate: int) -> Tuple[np.ndarray, int]:
        """Load audio using librosa."""
        try:
            import librosa
            audio, sr = librosa.load(str(file_path), sr=sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            raise AudioPreprocessorError(f"Failed to load audio with librosa: {e}")
    
    def _load_with_soundfile(self, file_path: Path, sample_rate: int) -> Tuple[np.ndarray, int]:
        """Load audio using soundfile."""
        try:
            import soundfile as sf
            audio, sr = sf.read(str(file_path))
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != sample_rate:
                audio = self._resample_scipy(audio, sr, sample_rate)
                sr = sample_rate
            
            return audio, sr
        except Exception as e:
            raise AudioPreprocessorError(f"Failed to load audio with soundfile: {e}")
    
    def _load_with_torchaudio(self, file_path: Path, sample_rate: int) -> Tuple[np.ndarray, int]:
        """Load audio using torchaudio."""
        try:
            import torchaudio
            import torchaudio.functional as F
            
            waveform, sr = torchaudio.load(str(file_path))
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != sample_rate:
                waveform = F.resample(waveform, sr, sample_rate)
                sr = sample_rate
            
            # Convert to numpy
            audio = waveform.squeeze().numpy()
            
            return audio, sr
        except Exception as e:
            raise AudioPreprocessorError(f"Failed to load audio with torchaudio: {e}")
    
    def _resample_scipy(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio using scipy."""
        try:
            from scipy import signal
            num_samples = int(len(audio) * target_sr / orig_sr)
            return signal.resample(audio, num_samples)
        except ImportError:
            self.logger.warning("scipy not available, cannot resample")
            return audio


class AudioNormalizer:
    """Audio normalization and scaling utilities."""
    
    def __init__(self, method: str = "peak", target_level: float = 0.95):
        """
        Initialize audio normalizer.
        
        Args:
            method: Normalization method ('peak', 'rms', 'lufs')
            target_level: Target level for normalization
        """
        self.method = method
        self.target_level = target_level
        self.logger = logging.getLogger(f"{__name__}.AudioNormalizer")
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio array.
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio array
        """
        if len(audio) == 0:
            return audio
        
        if self.method == "peak":
            return self._peak_normalize(audio)
        elif self.method == "rms":
            return self._rms_normalize(audio)
        elif self.method == "lufs":
            return self._lufs_normalize(audio)
        else:
            self.logger.warning(f"Unknown normalization method: {self.method}")
            return audio
    
    def _peak_normalize(self, audio: np.ndarray) -> np.ndarray:
        """Peak normalization."""
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio * (self.target_level / peak)
        return audio
    
    def _rms_normalize(self, audio: np.ndarray) -> np.ndarray:
        """RMS normalization."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            return audio * (self.target_level / rms)
        return audio
    
    def _lufs_normalize(self, audio: np.ndarray) -> np.ndarray:
        """LUFS normalization (simplified implementation)."""
        # This is a simplified version - proper LUFS requires more complex filtering
        return self._rms_normalize(audio)


class VoiceActivityDetector:
    """Voice Activity Detection (VAD) for audio preprocessing."""
    
    def __init__(self, frame_duration: float = 0.03, aggressiveness: int = 1):
        """
        Initialize VAD.
        
        Args:
            frame_duration: Duration of each frame in seconds
            aggressiveness: VAD aggressiveness (0-3, higher = more aggressive)
        """
        self.frame_duration = frame_duration
        self.aggressiveness = aggressiveness
        self.logger = logging.getLogger(f"{__name__}.VoiceActivityDetector")
        
        # Try to import webrtcvad
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(aggressiveness)
            self.has_webrtcvad = True
        except ImportError:
            self.has_webrtcvad = False
            self.logger.warning("webrtcvad not available, using energy-based VAD")
    
    def detect_voice_activity(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """
        Detect voice activity in audio.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate of audio
            
        Returns:
            List of (start_time, end_time) tuples for voice segments
        """
        if self.has_webrtcvad:
            return self._webrtc_vad(audio, sample_rate)
        else:
            return self._energy_based_vad(audio, sample_rate)
    
    def _webrtc_vad(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """Use WebRTC VAD for voice activity detection."""
        try:
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Frame size for WebRTC VAD (must be 10, 20, or 30ms)
            frame_duration_ms = int(self.frame_duration * 1000)
            if frame_duration_ms not in [10, 20, 30]:
                frame_duration_ms = 30
            
            frame_size = int(sample_rate * frame_duration_ms / 1000)
            
            voice_segments = []
            current_start = None
            
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i + frame_size].tobytes()
                
                try:
                    is_speech = self.vad.is_speech(frame, sample_rate)
                    time_offset = i / sample_rate
                    
                    if is_speech and current_start is None:
                        current_start = time_offset
                    elif not is_speech and current_start is not None:
                        voice_segments.append((current_start, time_offset))
                        current_start = None
                except Exception as e:
                    self.logger.debug(f"VAD frame processing error: {e}")
                    continue
            
            # Close final segment if needed
            if current_start is not None:
                voice_segments.append((current_start, len(audio) / sample_rate))
            
            return voice_segments
            
        except Exception as e:
            self.logger.warning(f"WebRTC VAD failed: {e}, falling back to energy-based VAD")
            return self._energy_based_vad(audio, sample_rate)
    
    def _energy_based_vad(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """Simple energy-based voice activity detection."""
        frame_size = int(sample_rate * self.frame_duration)
        
        # Calculate energy for each frame
        energies = []
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            energy = np.sum(frame ** 2) / len(frame)
            energies.append(energy)
        
        if not energies:
            return []
        
        # Determine threshold (adaptive)
        energy_array = np.array(energies)
        threshold = np.mean(energy_array) + self.aggressiveness * np.std(energy_array)
        
        # Find voice segments
        voice_segments = []
        current_start = None
        
        for i, energy in enumerate(energies):
            time_offset = i * self.frame_duration
            
            if energy > threshold and current_start is None:
                current_start = time_offset
            elif energy <= threshold and current_start is not None:
                voice_segments.append((current_start, time_offset))
                current_start = None
        
        # Close final segment if needed
        if current_start is not None:
            voice_segments.append((current_start, len(audio) / sample_rate))
        
        return voice_segments
    
    def remove_silence(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Remove silent parts from audio."""
        voice_segments = self.detect_voice_activity(audio, sample_rate)
        
        if not voice_segments:
            return audio
        
        # Concatenate voice segments
        voiced_audio_parts = []
        for start_time, end_time in voice_segments:
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            voiced_audio_parts.append(audio[start_idx:end_idx])
        
        return np.concatenate(voiced_audio_parts) if voiced_audio_parts else audio


class AudioChunker:
    """Audio chunking and segmentation utilities."""
    
    def __init__(self, chunk_duration: float = 30.0, overlap: float = 5.0):
        """
        Initialize audio chunker.
        
        Args:
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
        """
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.logger = logging.getLogger(f"{__name__}.AudioChunker")
    
    def chunk_audio(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[np.ndarray, float, float]]:
        """
        Split audio into overlapping chunks.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            List of (chunk_audio, start_time, end_time) tuples
        """
        chunk_samples = int(self.chunk_duration * sample_rate)
        overlap_samples = int(self.overlap * sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(audio):
            end_idx = min(start_idx + chunk_samples, len(audio))
            chunk = audio[start_idx:end_idx]
            
            start_time = start_idx / sample_rate
            end_time = end_idx / sample_rate
            
            chunks.append((chunk, start_time, end_time))
            
            if end_idx >= len(audio):
                break
            
            start_idx += step_samples
        
        return chunks
    
    def merge_chunks(self, chunks: List[Tuple[np.ndarray, float, float]], 
                    sample_rate: int) -> np.ndarray:
        """
        Merge overlapping chunks back into single audio.
        
        Args:
            chunks: List of (chunk_audio, start_time, end_time) tuples
            sample_rate: Sample rate
            
        Returns:
            Merged audio array
        """
        if not chunks:
            return np.array([])
        
        # Calculate total length
        last_chunk = chunks[-1]
        total_length = int(last_chunk[2] * sample_rate)  # end_time of last chunk
        
        # Initialize output array
        merged_audio = np.zeros(total_length)
        overlap_weights = np.zeros(total_length)
        
        # Add chunks with overlap handling
        for chunk_audio, start_time, end_time in chunks:
            start_idx = int(start_time * sample_rate)
            end_idx = start_idx + len(chunk_audio)
            
            # Add chunk to output
            if end_idx <= len(merged_audio):
                merged_audio[start_idx:end_idx] += chunk_audio
                overlap_weights[start_idx:end_idx] += 1
            else:
                # Handle final chunk that might exceed calculated length
                valid_length = len(merged_audio) - start_idx
                merged_audio[start_idx:] += chunk_audio[:valid_length]
                overlap_weights[start_idx:] += 1
        
        # Average overlapping regions
        overlap_weights[overlap_weights == 0] = 1  # Avoid division by zero
        merged_audio = merged_audio / overlap_weights
        
        return merged_audio


class ComprehensiveAudioPreprocessor(BaseAudioPreprocessor):
    """
    Comprehensive audio preprocessor combining all functionality.
    """
    
    def __init__(self, sample_rate: int = 16000, normalize: bool = True,
                 enable_vad: bool = True, chunk_duration: Optional[float] = None,
                 normalization_method: str = "peak"):
        super().__init__(sample_rate, normalize)
        
        self.enable_vad = enable_vad
        self.chunk_duration = chunk_duration
        
        # Initialize components
        self.loader = AudioLoader(sample_rate)
        self.normalizer = AudioNormalizer(method=normalization_method)
        self.vad = VoiceActivityDetector() if enable_vad else None
        self.chunker = AudioChunker(chunk_duration) if chunk_duration else None
    
    def load_audio(self, file_path: Union[str, Path], 
                   sample_rate: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file using the internal audio loader.
        
        Args:
            file_path: Path to audio file
            sample_rate: Target sample rate (None to use preprocessor's sample rate)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        return self.loader.load_audio(file_path, sample_rate)
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio using the internal normalizer.
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio array
        """
        return self.normalizer.normalize(audio)
    
    def process(self, audio: Union[str, np.ndarray, torch.Tensor]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Process audio input with full pipeline.
        
        Args:
            audio: Audio input (file path, array, or tensor)
            
        Returns:
            Processed audio (array or list of chunks)
        """
        # Load audio if file path
        if isinstance(audio, (str, Path)):
            audio_array, sr = self.loader.load_audio(audio, self.sample_rate)
        elif isinstance(audio, torch.Tensor):
            audio_array = audio.cpu().numpy()
            sr = self.sample_rate
        else:
            audio_array = audio
            sr = self.sample_rate
        
        # Ensure mono
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=0)
        
        # Normalize
        if self.normalize:
            audio_array = self.normalizer.normalize(audio_array)
        
        # Remove silence if VAD enabled
        if self.enable_vad and self.vad:
            audio_array = self.vad.remove_silence(audio_array, sr)
        
        # Chunk if enabled
        if self.chunker:
            chunks = self.chunker.chunk_audio(audio_array, sr)
            return [chunk[0] for chunk in chunks]  # Return just the audio arrays
        
        return audio_array


# Simple AudioPreprocessor alias for backward compatibility
class AudioPreprocessor(ComprehensiveAudioPreprocessor):
    """Simple audio preprocessor for backward compatibility."""
    pass


def create_audio_preprocessor(config: Optional[AudioPreprocessorConfig] = None) -> AudioPreprocessor:
    """
    Factory function to create an audio preprocessor.
    
    Args:
        config: Audio preprocessor configuration
        
    Returns:
        AudioPreprocessor instance
    """
    if config is None:
        config = AudioPreprocessorConfig()
    
    return AudioPreprocessor(
        sample_rate=config.sample_rate,
        normalize=config.normalize,
        enable_vad=config.enable_vad,
        chunk_duration=config.chunk_duration,
        normalization_method=config.normalization_method
    )


def get_audio_transforms(config: Optional[AudioPreprocessorConfig] = None) -> List[Callable]:
    """
    Get a list of audio transform functions based on configuration.
    
    Args:
        config: Audio preprocessor configuration
        
    Returns:
        List of transform functions
    """
    if config is None:
        config = AudioPreprocessorConfig()
    
    transforms = []
    
    # Add normalization transform
    if config.normalize:
        normalizer = AudioNormalizer(method=config.normalization_method, target_level=config.target_level)
        transforms.append(normalizer.normalize)
    
    # Add resampling transform if needed
    def resample_transform(audio: np.ndarray, original_sr: int = None) -> np.ndarray:
        if original_sr and original_sr != config.sample_rate:
            loader = AudioLoader(config.sample_rate)
            return loader._resample_scipy(audio, original_sr, config.sample_rate)
        return audio
    
    transforms.append(resample_transform)
    
    # Add VAD transform if enabled
    if config.enable_vad:
        vad = VoiceActivityDetector(
            frame_duration=config.frame_duration,
            aggressiveness=config.vad_aggressiveness
        )
        transforms.append(lambda audio: vad.remove_silence(audio, config.sample_rate))
    
    # Add chunking transform if enabled
    if config.chunk_duration:
        chunker = AudioChunker(
            chunk_duration=config.chunk_duration,
            overlap=config.overlap_duration
        )
        transforms.append(lambda audio: chunker.chunk_audio(audio, config.sample_rate))
    
    return transforms

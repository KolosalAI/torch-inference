"""
Spectrogram and frequency domain processing for audio models.

This module provides spectrogram generation, mel-scale processing, and
frequency domain transformations for audio preprocessing.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SpectrogramProcessorError(Exception):
    """Exception raised for spectrogram processing errors."""
    pass


class BaseSpectrogramProcessor(ABC):
    """Base class for spectrogram processors."""
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 2048):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute spectrogram from audio."""
        pass


class STFTProcessor:
    """Short-Time Fourier Transform processor."""
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 2048, 
                 hop_length: Optional[int] = None, win_length: Optional[int] = None,
                 window: str = "hann", center: bool = True):
        """
        Initialize STFT processor.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length between frames
            win_length: Window length
            window: Window function type
            center: Whether to center the window
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft
        self.window = window
        self.center = center
        self.logger = logging.getLogger(f"{__name__}.STFTProcessor")
    
    def stft(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            audio: Input audio array or tensor
            
        Returns:
            Complex STFT tensor of shape (n_freqs, n_frames)
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Generate window
        if self.window == "hann":
            window_tensor = torch.hann_window(self.win_length)
        elif self.window == "hamming":
            window_tensor = torch.hamming_window(self.win_length)
        elif self.window == "blackman":
            window_tensor = torch.blackman_window(self.win_length)
        else:
            window_tensor = torch.ones(self.win_length)
        
        # Compute STFT
        stft_result = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window_tensor,
            center=self.center,
            return_complex=True
        )
        
        return stft_result
    
    def istft(self, stft_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute Inverse Short-Time Fourier Transform.
        
        Args:
            stft_tensor: Complex STFT tensor
            
        Returns:
            Reconstructed audio tensor
        """
        # Generate window
        if self.window == "hann":
            window_tensor = torch.hann_window(self.win_length)
        elif self.window == "hamming":
            window_tensor = torch.hamming_window(self.win_length)
        elif self.window == "blackman":
            window_tensor = torch.blackman_window(self.win_length)
        else:
            window_tensor = torch.ones(self.win_length)
        
        # Compute ISTFT
        audio = torch.istft(
            stft_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window_tensor,
            center=self.center
        )
        
        return audio
    
    def magnitude_spectrogram(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute magnitude spectrogram.
        
        Args:
            audio: Input audio
            
        Returns:
            Magnitude spectrogram tensor
        """
        stft_result = self.stft(audio)
        magnitude = torch.abs(stft_result)
        return magnitude
    
    def power_spectrogram(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute power spectrogram.
        
        Args:
            audio: Input audio
            
        Returns:
            Power spectrogram tensor
        """
        magnitude = self.magnitude_spectrogram(audio)
        power = magnitude ** 2
        return power
    
    def phase_spectrogram(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute phase spectrogram.
        
        Args:
            audio: Input audio
            
        Returns:
            Phase spectrogram tensor
        """
        stft_result = self.stft(audio)
        phase = torch.angle(stft_result)
        return phase


class MelSpectrogramProcessor:
    """Mel-scale spectrogram processor."""
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 2048,
                 hop_length: Optional[int] = None, n_mels: int = 80,
                 fmin: float = 0.0, fmax: Optional[float] = None,
                 power: float = 2.0, normalized: bool = False):
        """
        Initialize Mel spectrogram processor.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length between frames
            n_mels: Number of mel filter banks
            fmin: Minimum frequency
            fmax: Maximum frequency (None for sample_rate/2)
            power: Exponent for power spectrogram
            normalized: Whether to normalize mel filter banks
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        self.power = power
        self.normalized = normalized
        
        # Initialize STFT processor
        self.stft_processor = STFTProcessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=self.hop_length
        )
        
        # Create mel filter bank
        self.mel_filter_bank = self._create_mel_filter_bank()
        self.logger = logging.getLogger(f"{__name__}.MelSpectrogramProcessor")
    
    def _create_mel_filter_bank(self) -> torch.Tensor:
        """Create mel filter bank matrix."""
        try:
            # Try using torchaudio if available
            import torchaudio.functional as F
            return F.melscale_fbanks(
                n_freqs=self.n_fft // 2 + 1,
                f_min=self.fmin,
                f_max=self.fmax,
                n_mels=self.n_mels,
                sample_rate=self.sample_rate,
                norm='slaney' if self.normalized else None
            )
        except ImportError:
            # Fallback implementation
            return self._create_mel_filter_bank_manual()
    
    def _create_mel_filter_bank_manual(self) -> torch.Tensor:
        """Manual implementation of mel filter bank creation."""
        # Convert Hz to mel scale
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Create mel-spaced frequencies
        mel_min = hz_to_mel(self.fmin)
        mel_max = hz_to_mel(self.fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert to FFT bin indices
        n_freqs = self.n_fft // 2 + 1
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # Create filter bank
        filter_bank = torch.zeros(n_freqs, self.n_mels)
        
        for m in range(1, self.n_mels + 1):
            left = bin_points[m - 1]
            center = bin_points[m]
            right = bin_points[m + 1]
            
            # Triangular filter
            for k in range(left, center):
                if center != left:
                    filter_bank[k, m - 1] = (k - left) / (center - left)
            
            for k in range(center, right):
                if right != center:
                    filter_bank[k, m - 1] = (right - k) / (right - center)
        
        return filter_bank.T  # Shape: (n_mels, n_freqs)
    
    def mel_spectrogram(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute mel-scale spectrogram.
        
        Args:
            audio: Input audio
            
        Returns:
            Mel spectrogram tensor of shape (n_mels, n_frames)
        """
        # Compute power spectrogram
        power_spec = self.stft_processor.power_spectrogram(audio)
        
        # Apply mel filter bank
        mel_spec = torch.matmul(self.mel_filter_bank, power_spec)
        
        # Apply power
        if self.power != 2.0:
            mel_spec = mel_spec ** (self.power / 2.0)
        
        return mel_spec
    
    def log_mel_spectrogram(self, audio: Union[np.ndarray, torch.Tensor], 
                           amin: float = 1e-10, dynamic_range_db: float = 80.0) -> torch.Tensor:
        """
        Compute log mel-scale spectrogram.
        
        Args:
            audio: Input audio
            amin: Minimum value for log computation
            dynamic_range_db: Dynamic range in dB
            
        Returns:
            Log mel spectrogram tensor
        """
        mel_spec = self.mel_spectrogram(audio)
        
        # Convert to log scale
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=amin))
        
        # Apply dynamic range compression
        log_mel_spec = torch.clamp(
            log_mel_spec,
            min=log_mel_spec.max() - dynamic_range_db / 20 * np.log(10)
        )
        
        return log_mel_spec


class MFCCProcessor:
    """Mel-Frequency Cepstral Coefficients processor."""
    
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 13,
                 n_fft: int = 2048, hop_length: Optional[int] = None,
                 n_mels: int = 40, fmin: float = 0.0, fmax: Optional[float] = None):
        """
        Initialize MFCC processor.
        
        Args:
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length between frames
            n_mels: Number of mel filter banks
            fmin: Minimum frequency
            fmax: Maximum frequency
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        
        # Initialize mel spectrogram processor
        self.mel_processor = MelSpectrogramProcessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=self.hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=self.fmax
        )
        
        # Create DCT matrix
        self.dct_matrix = self._create_dct_matrix()
        self.logger = logging.getLogger(f"{__name__}.MFCCProcessor")
    
    def _create_dct_matrix(self) -> torch.Tensor:
        """Create Discrete Cosine Transform matrix."""
        dct_matrix = torch.zeros(self.n_mfcc, self.n_mels)
        
        for k in range(self.n_mfcc):
            for n in range(self.n_mels):
                dct_matrix[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2 * self.n_mels))
        
        # Normalize
        dct_matrix[0, :] *= 1 / np.sqrt(2)
        dct_matrix *= np.sqrt(2 / self.n_mels)
        
        return dct_matrix
    
    def mfcc(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute MFCC features.
        
        Args:
            audio: Input audio
            
        Returns:
            MFCC tensor of shape (n_mfcc, n_frames)
        """
        # Compute log mel spectrogram
        log_mel_spec = self.mel_processor.log_mel_spectrogram(audio)
        
        # Apply DCT
        mfcc_features = torch.matmul(self.dct_matrix, log_mel_spec)
        
        return mfcc_features
    
    def delta_features(self, features: torch.Tensor, width: int = 9) -> torch.Tensor:
        """
        Compute delta (derivative) features.
        
        Args:
            features: Input feature tensor
            width: Width of the delta window
            
        Returns:
            Delta features tensor
        """
        # Pad features
        padded_features = F.pad(features, (width // 2, width // 2), mode='replicate')
        
        # Compute deltas
        deltas = torch.zeros_like(features)
        for t in range(features.shape[-1]):
            deltas[:, t] = (
                padded_features[:, t + width - 1] - 
                padded_features[:, t + 1]
            ) / 2
        
        return deltas
    
    def mfcc_with_deltas(self, audio: Union[np.ndarray, torch.Tensor], 
                        delta: bool = True, delta_delta: bool = True) -> torch.Tensor:
        """
        Compute MFCC with delta and delta-delta features.
        
        Args:
            audio: Input audio
            delta: Whether to include delta features
            delta_delta: Whether to include delta-delta features
            
        Returns:
            Combined MFCC features tensor
        """
        # Compute MFCC
        mfcc_features = self.mfcc(audio)
        feature_list = [mfcc_features]
        
        # Add delta features
        if delta:
            delta_features = self.delta_features(mfcc_features)
            feature_list.append(delta_features)
        
        # Add delta-delta features
        if delta_delta and delta:
            delta_delta_features = self.delta_features(delta_features)
            feature_list.append(delta_delta_features)
        
        # Concatenate features
        combined_features = torch.cat(feature_list, dim=0)
        
        return combined_features


class SpectralProcessor:
    """Additional spectral feature processor."""
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 2048,
                 hop_length: Optional[int] = None):
        """
        Initialize spectral processor.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length between frames
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        
        self.stft_processor = STFTProcessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=self.hop_length
        )
        self.logger = logging.getLogger(f"{__name__}.SpectralProcessor")
    
    def spectral_centroid(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute spectral centroid.
        
        Args:
            audio: Input audio
            
        Returns:
            Spectral centroid tensor
        """
        power_spec = self.stft_processor.power_spectrogram(audio)
        freqs = torch.linspace(0, self.sample_rate / 2, power_spec.shape[0])
        
        # Compute centroid
        centroid = torch.sum(freqs.unsqueeze(1) * power_spec, dim=0) / torch.sum(power_spec, dim=0)
        
        return centroid
    
    def spectral_rolloff(self, audio: Union[np.ndarray, torch.Tensor], 
                        roll_percent: float = 0.85) -> torch.Tensor:
        """
        Compute spectral rolloff.
        
        Args:
            audio: Input audio
            roll_percent: Rolloff percentage
            
        Returns:
            Spectral rolloff tensor
        """
        power_spec = self.stft_processor.power_spectrogram(audio)
        freqs = torch.linspace(0, self.sample_rate / 2, power_spec.shape[0])
        
        # Compute cumulative energy
        cumsum_spec = torch.cumsum(power_spec, dim=0)
        total_energy = cumsum_spec[-1, :]
        
        # Find rolloff frequency
        rolloff_threshold = roll_percent * total_energy
        rolloff_indices = torch.argmax((cumsum_spec >= rolloff_threshold).int(), dim=0)
        rolloff_freqs = freqs[rolloff_indices]
        
        return rolloff_freqs
    
    def zero_crossing_rate(self, audio: Union[np.ndarray, torch.Tensor],
                          frame_length: int = 2048, hop_length: Optional[int] = None) -> torch.Tensor:
        """
        Compute zero crossing rate.
        
        Args:
            audio: Input audio
            frame_length: Frame length for analysis
            hop_length: Hop length between frames
            
        Returns:
            Zero crossing rate tensor
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        if hop_length is None:
            hop_length = frame_length // 2
        
        # Compute zero crossings
        zcr = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            zero_crossings = torch.sum(torch.abs(torch.diff(torch.sign(frame)))) / 2
            zcr.append(zero_crossings / frame_length)
        
        return torch.tensor(zcr)


def create_spectrogram_processor(processor_type: str, **kwargs):
    """
    Factory function to create spectrogram processors.
    
    Args:
        processor_type: Type of processor ('stft', 'mel', 'mfcc', 'spectral')
        **kwargs: Additional arguments for processor initialization
        
    Returns:
        Spectrogram processor instance
    """
    processor_type = processor_type.lower()
    
    if processor_type == "stft":
        return STFTProcessor(**kwargs)
    elif processor_type == "mel":
        return MelSpectrogramProcessor(**kwargs)
    elif processor_type == "mfcc":
        return MFCCProcessor(**kwargs)
    elif processor_type == "spectral":
        return SpectralProcessor(**kwargs)
    else:
        raise SpectrogramProcessorError(f"Unsupported processor type: {processor_type}")

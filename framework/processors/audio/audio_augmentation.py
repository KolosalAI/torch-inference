"""
Audio data augmentation for improving model robustness.

This module provides various audio augmentation techniques including:
- Time-domain augmentations (time stretching, pitch shifting, noise)
- Frequency-domain augmentations (spectral masking, filtering)
- Advanced augmentations (reverb, compression, distortion)
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import numpy as np
import torch
import torch.nn.functional as F
import logging
import random
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AudioAugmentationError(Exception):
    """Exception raised for audio augmentation errors."""
    pass


class BaseAudioAugmentation(ABC):
    """Base class for audio augmentations."""
    
    def __init__(self, probability: float = 0.5):
        """
        Initialize augmentation.
        
        Args:
            probability: Probability of applying the augmentation
        """
        self.probability = probability
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def apply(self, audio: Union[np.ndarray, torch.Tensor], 
             sample_rate: int) -> Union[np.ndarray, torch.Tensor]:
        """Apply augmentation to audio."""
        pass
    
    def __call__(self, audio: Union[np.ndarray, torch.Tensor], 
                 sample_rate: int) -> Union[np.ndarray, torch.Tensor]:
        """Apply augmentation with probability."""
        if random.random() < self.probability:
            return self.apply(audio, sample_rate)
        return audio


class TimeStretchAugmentation(BaseAudioAugmentation):
    """Time stretching (speed change without pitch change)."""
    
    def __init__(self, stretch_range: Tuple[float, float] = (0.8, 1.2), 
                 probability: float = 0.5):
        """
        Initialize time stretch augmentation.
        
        Args:
            stretch_range: Range of stretch factors (min, max)
            probability: Probability of applying augmentation
        """
        super().__init__(probability)
        self.stretch_range = stretch_range
    
    def apply(self, audio: Union[np.ndarray, torch.Tensor], 
             sample_rate: int) -> Union[np.ndarray, torch.Tensor]:
        """Apply time stretching."""
        stretch_factor = random.uniform(*self.stretch_range)
        
        try:
            # Try using librosa for high-quality time stretching
            import librosa
            
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
                was_tensor = True
            else:
                audio_np = audio
                was_tensor = False
            
            stretched = librosa.effects.time_stretch(audio_np, rate=stretch_factor)
            
            if was_tensor:
                return torch.from_numpy(stretched).to(audio.device)
            return stretched
            
        except ImportError:
            # Fallback: simple resampling (changes pitch)
            self.logger.warning("librosa not available, using simple resampling")
            return self._simple_time_stretch(audio, stretch_factor)
    
    def _simple_time_stretch(self, audio: Union[np.ndarray, torch.Tensor], 
                           stretch_factor: float) -> Union[np.ndarray, torch.Tensor]:
        """Simple time stretching using interpolation."""
        if isinstance(audio, torch.Tensor):
            original_length = audio.shape[-1]
            new_length = int(original_length / stretch_factor)
            
            # Use interpolation for resampling
            audio_unsqueezed = audio.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            stretched = F.interpolate(
                audio_unsqueezed,
                size=new_length,
                mode='linear',
                align_corners=False
            )
            return stretched.squeeze(0).squeeze(0)
        else:
            # NumPy version
            from scipy.signal import resample
            original_length = len(audio)
            new_length = int(original_length / stretch_factor)
            return resample(audio, new_length)


class PitchShiftAugmentation(BaseAudioAugmentation):
    """Pitch shifting augmentation."""
    
    def __init__(self, pitch_range: Tuple[float, float] = (-2.0, 2.0),
                 probability: float = 0.5):
        """
        Initialize pitch shift augmentation.
        
        Args:
            pitch_range: Range of pitch shift in semitones
            probability: Probability of applying augmentation
        """
        super().__init__(probability)
        self.pitch_range = pitch_range
    
    def apply(self, audio: Union[np.ndarray, torch.Tensor], 
             sample_rate: int) -> Union[np.ndarray, torch.Tensor]:
        """Apply pitch shifting."""
        pitch_shift = random.uniform(*self.pitch_range)
        
        try:
            import librosa
            
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
                was_tensor = True
            else:
                audio_np = audio
                was_tensor = False
            
            shifted = librosa.effects.pitch_shift(
                audio_np, sr=sample_rate, n_steps=pitch_shift
            )
            
            if was_tensor:
                return torch.from_numpy(shifted).to(audio.device)
            return shifted
            
        except ImportError:
            self.logger.warning("librosa not available, skipping pitch shift")
            return audio


class NoiseAugmentation(BaseAudioAugmentation):
    """Add various types of noise to audio."""
    
    def __init__(self, noise_type: str = "white", 
                 snr_range: Tuple[float, float] = (10.0, 30.0),
                 probability: float = 0.5):
        """
        Initialize noise augmentation.
        
        Args:
            noise_type: Type of noise ('white', 'pink', 'brown')
            snr_range: Signal-to-noise ratio range in dB
            probability: Probability of applying augmentation
        """
        super().__init__(probability)
        self.noise_type = noise_type.lower()
        self.snr_range = snr_range
    
    def apply(self, audio: Union[np.ndarray, torch.Tensor], 
             sample_rate: int) -> Union[np.ndarray, torch.Tensor]:
        """Apply noise augmentation."""
        snr_db = random.uniform(*self.snr_range)
        
        if isinstance(audio, torch.Tensor):
            noise = self._generate_noise_tensor(audio.shape, self.noise_type, audio.device)
        else:
            noise = self._generate_noise_numpy(audio.shape, self.noise_type)
        
        # Calculate signal and noise power
        signal_power = torch.mean(audio ** 2) if isinstance(audio, torch.Tensor) else np.mean(audio ** 2)
        noise_power = torch.mean(noise ** 2) if isinstance(noise, torch.Tensor) else np.mean(noise ** 2)
        
        # Calculate noise scaling factor for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = (signal_power / (noise_power * snr_linear)) ** 0.5
        
        # Add scaled noise
        noisy_audio = audio + noise_scale * noise
        
        return noisy_audio
    
    def _generate_noise_tensor(self, shape: Tuple, noise_type: str, device: torch.device) -> torch.Tensor:
        """Generate noise tensor."""
        if noise_type == "white":
            return torch.randn(shape, device=device)
        elif noise_type == "pink":
            return self._generate_pink_noise_tensor(shape, device)
        elif noise_type == "brown":
            return self._generate_brown_noise_tensor(shape, device)
        else:
            return torch.randn(shape, device=device)
    
    def _generate_noise_numpy(self, shape: Tuple, noise_type: str) -> np.ndarray:
        """Generate noise numpy array."""
        if noise_type == "white":
            return np.random.randn(*shape)
        elif noise_type == "pink":
            return self._generate_pink_noise_numpy(shape)
        elif noise_type == "brown":
            return self._generate_brown_noise_numpy(shape)
        else:
            return np.random.randn(*shape)
    
    def _generate_pink_noise_tensor(self, shape: Tuple, device: torch.device) -> torch.Tensor:
        """Generate pink noise (1/f noise)."""
        # Simplified pink noise generation
        white_noise = torch.randn(shape, device=device)
        # Apply simple low-pass filtering for pink-ish characteristics
        if len(shape) == 1:
            # 1D signal
            alpha = 0.02
            pink_noise = torch.zeros_like(white_noise)
            pink_noise[0] = white_noise[0]
            for i in range(1, len(white_noise)):
                pink_noise[i] = alpha * white_noise[i] + (1 - alpha) * pink_noise[i-1]
            return pink_noise
        return white_noise  # Fallback
    
    def _generate_brown_noise_tensor(self, shape: Tuple, device: torch.device) -> torch.Tensor:
        """Generate brown noise (1/f^2 noise)."""
        # Simplified brown noise generation
        white_noise = torch.randn(shape, device=device)
        if len(shape) == 1:
            # 1D signal - cumulative sum approximates brown noise
            return torch.cumsum(white_noise, dim=0) / torch.sqrt(torch.arange(1, len(white_noise) + 1, device=device).float())
        return white_noise  # Fallback
    
    def _generate_pink_noise_numpy(self, shape: Tuple) -> np.ndarray:
        """Generate pink noise using numpy."""
        white_noise = np.random.randn(*shape)
        if len(shape) == 1:
            alpha = 0.02
            pink_noise = np.zeros_like(white_noise)
            pink_noise[0] = white_noise[0]
            for i in range(1, len(white_noise)):
                pink_noise[i] = alpha * white_noise[i] + (1 - alpha) * pink_noise[i-1]
            return pink_noise
        return white_noise
    
    def _generate_brown_noise_numpy(self, shape: Tuple) -> np.ndarray:
        """Generate brown noise using numpy."""
        white_noise = np.random.randn(*shape)
        if len(shape) == 1:
            return np.cumsum(white_noise) / np.sqrt(np.arange(1, len(white_noise) + 1))
        return white_noise


class VolumeAugmentation(BaseAudioAugmentation):
    """Volume scaling augmentation."""
    
    def __init__(self, volume_range: Tuple[float, float] = (0.5, 1.5),
                 probability: float = 0.5):
        """
        Initialize volume augmentation.
        
        Args:
            volume_range: Range of volume scaling factors
            probability: Probability of applying augmentation
        """
        super().__init__(probability)
        self.volume_range = volume_range
    
    def apply(self, audio: Union[np.ndarray, torch.Tensor], 
             sample_rate: int) -> Union[np.ndarray, torch.Tensor]:
        """Apply volume scaling."""
        volume_factor = random.uniform(*self.volume_range)
        return audio * volume_factor


class SpectralMaskingAugmentation(BaseAudioAugmentation):
    """Spectral masking augmentation for spectrograms."""
    
    def __init__(self, freq_mask_param: int = 15, time_mask_param: int = 35,
                 num_freq_masks: int = 1, num_time_masks: int = 1,
                 probability: float = 0.5):
        """
        Initialize spectral masking augmentation.
        
        Args:
            freq_mask_param: Maximum frequency masking length
            time_mask_param: Maximum time masking length
            num_freq_masks: Number of frequency masks
            num_time_masks: Number of time masks
            probability: Probability of applying augmentation
        """
        super().__init__(probability)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def apply(self, spectrogram: torch.Tensor, 
             sample_rate: int = None) -> torch.Tensor:
        """
        Apply spectral masking to spectrogram.
        
        Args:
            spectrogram: Input spectrogram tensor (freq, time)
            sample_rate: Not used for spectral masking
            
        Returns:
            Masked spectrogram
        """
        masked_spec = spectrogram.clone()
        
        # Apply frequency masks
        for _ in range(self.num_freq_masks):
            masked_spec = self._apply_freq_mask(masked_spec)
        
        # Apply time masks
        for _ in range(self.num_time_masks):
            masked_spec = self._apply_time_mask(masked_spec)
        
        return masked_spec
    
    def _apply_freq_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking."""
        freq_size = spectrogram.shape[0]
        mask_size = random.randint(0, min(self.freq_mask_param, freq_size))
        
        if mask_size > 0:
            start_freq = random.randint(0, freq_size - mask_size)
            spectrogram[start_freq:start_freq + mask_size, :] = 0
        
        return spectrogram
    
    def _apply_time_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply time masking."""
        time_size = spectrogram.shape[1]
        mask_size = random.randint(0, min(self.time_mask_param, time_size))
        
        if mask_size > 0:
            start_time = random.randint(0, time_size - mask_size)
            spectrogram[:, start_time:start_time + mask_size] = 0
        
        return spectrogram


class ReverbAugmentation(BaseAudioAugmentation):
    """Simple reverb augmentation."""
    
    def __init__(self, reverb_range: Tuple[float, float] = (0.1, 0.3),
                 delay_range: Tuple[float, float] = (0.05, 0.15),
                 probability: float = 0.5):
        """
        Initialize reverb augmentation.
        
        Args:
            reverb_range: Range of reverb intensity
            delay_range: Range of reverb delay in seconds
            probability: Probability of applying augmentation
        """
        super().__init__(probability)
        self.reverb_range = reverb_range
        self.delay_range = delay_range
    
    def apply(self, audio: Union[np.ndarray, torch.Tensor], 
             sample_rate: int) -> Union[np.ndarray, torch.Tensor]:
        """Apply simple reverb effect."""
        reverb_intensity = random.uniform(*self.reverb_range)
        delay_seconds = random.uniform(*self.delay_range)
        delay_samples = int(delay_seconds * sample_rate)
        
        if isinstance(audio, torch.Tensor):
            # Create delayed version
            delayed_audio = torch.zeros_like(audio)
            if delay_samples < len(audio):
                delayed_audio[delay_samples:] = audio[:-delay_samples]
            
            # Mix original with delayed version
            reverb_audio = audio + reverb_intensity * delayed_audio
        else:
            # NumPy version
            delayed_audio = np.zeros_like(audio)
            if delay_samples < len(audio):
                delayed_audio[delay_samples:] = audio[:-delay_samples]
            
            reverb_audio = audio + reverb_intensity * delayed_audio
        
        return reverb_audio


class CompressionAugmentation(BaseAudioAugmentation):
    """Dynamic range compression augmentation."""
    
    def __init__(self, threshold: float = 0.7, ratio: float = 4.0,
                 attack_time: float = 0.001, release_time: float = 0.1,
                 probability: float = 0.5):
        """
        Initialize compression augmentation.
        
        Args:
            threshold: Compression threshold (0-1)
            ratio: Compression ratio
            attack_time: Attack time in seconds
            release_time: Release time in seconds
            probability: Probability of applying augmentation
        """
        super().__init__(probability)
        self.threshold = threshold
        self.ratio = ratio
        self.attack_time = attack_time
        self.release_time = release_time
    
    def apply(self, audio: Union[np.ndarray, torch.Tensor], 
             sample_rate: int) -> Union[np.ndarray, torch.Tensor]:
        """Apply dynamic range compression."""
        # Simplified compression implementation
        if isinstance(audio, torch.Tensor):
            amplitude = torch.abs(audio)
            compressed = torch.where(
                amplitude > self.threshold,
                torch.sign(audio) * (
                    self.threshold + 
                    (amplitude - self.threshold) / self.ratio
                ),
                audio
            )
        else:
            amplitude = np.abs(audio)
            compressed = np.where(
                amplitude > self.threshold,
                np.sign(audio) * (
                    self.threshold + 
                    (amplitude - self.threshold) / self.ratio
                ),
                audio
            )
        
        return compressed


class AudioAugmentationPipeline:
    """Pipeline for applying multiple audio augmentations."""
    
    def __init__(self, augmentations: List[BaseAudioAugmentation],
                 shuffle: bool = True, max_augmentations: Optional[int] = None):
        """
        Initialize augmentation pipeline.
        
        Args:
            augmentations: List of augmentation instances
            shuffle: Whether to shuffle augmentation order
            max_augmentations: Maximum number of augmentations to apply (None for all)
        """
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.max_augmentations = max_augmentations
        self.logger = logging.getLogger(f"{__name__}.AudioAugmentationPipeline")
    
    def __call__(self, audio: Union[np.ndarray, torch.Tensor], 
                 sample_rate: int) -> Union[np.ndarray, torch.Tensor]:
        """Apply augmentation pipeline."""
        augmented_audio = audio
        
        # Select augmentations to apply
        if self.shuffle:
            selected_augs = random.sample(self.augmentations, len(self.augmentations))
        else:
            selected_augs = self.augmentations.copy()
        
        if self.max_augmentations is not None:
            selected_augs = selected_augs[:self.max_augmentations]
        
        # Apply augmentations
        for augmentation in selected_augs:
            try:
                augmented_audio = augmentation(augmented_audio, sample_rate)
            except Exception as e:
                self.logger.warning(f"Augmentation {type(augmentation).__name__} failed: {e}")
                continue
        
        return augmented_audio
    
    def add_augmentation(self, augmentation: BaseAudioAugmentation):
        """Add augmentation to pipeline."""
        self.augmentations.append(augmentation)
    
    def remove_augmentation(self, augmentation_type: type):
        """Remove augmentation of specific type from pipeline."""
        self.augmentations = [
            aug for aug in self.augmentations 
            if not isinstance(aug, augmentation_type)
        ]


def create_default_augmentation_pipeline(sample_rate: int = 16000,
                                       intensity: str = "medium") -> AudioAugmentationPipeline:
    """
    Create a default augmentation pipeline.
    
    Args:
        sample_rate: Audio sample rate
        intensity: Augmentation intensity ('light', 'medium', 'heavy')
        
    Returns:
        AudioAugmentationPipeline instance
    """
    intensity = intensity.lower()
    
    if intensity == "light":
        augmentations = [
            VolumeAugmentation(volume_range=(0.8, 1.2), probability=0.3),
            NoiseAugmentation(snr_range=(20.0, 40.0), probability=0.2),
        ]
    elif intensity == "medium":
        augmentations = [
            TimeStretchAugmentation(stretch_range=(0.9, 1.1), probability=0.4),
            PitchShiftAugmentation(pitch_range=(-1.0, 1.0), probability=0.3),
            VolumeAugmentation(volume_range=(0.7, 1.3), probability=0.5),
            NoiseAugmentation(snr_range=(15.0, 35.0), probability=0.4),
            ReverbAugmentation(probability=0.2),
        ]
    elif intensity == "heavy":
        augmentations = [
            TimeStretchAugmentation(stretch_range=(0.8, 1.2), probability=0.6),
            PitchShiftAugmentation(pitch_range=(-2.0, 2.0), probability=0.5),
            VolumeAugmentation(volume_range=(0.5, 1.5), probability=0.7),
            NoiseAugmentation(snr_range=(10.0, 30.0), probability=0.6),
            ReverbAugmentation(probability=0.4),
            CompressionAugmentation(probability=0.3),
        ]
    else:
        raise AudioAugmentationError(f"Unknown intensity level: {intensity}")
    
    return AudioAugmentationPipeline(augmentations, shuffle=True, max_augmentations=3)

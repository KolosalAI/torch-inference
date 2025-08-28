"""
Audio feature extraction for machine learning models.

This module provides comprehensive audio feature extraction including:
- Traditional features (MFCC, spectral features, etc.)
- Deep learning features (embeddings from pre-trained models)
- Time-domain features
- Statistical features
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import logging
from abc import ABC, abstractmethod

from .spectrogram_processor import (
    STFTProcessor, MelSpectrogramProcessor, 
    MFCCProcessor, SpectralProcessor
)

logger = logging.getLogger(__name__)


class FeatureExtractorError(Exception):
    """Exception raised for feature extraction errors."""
    pass


class BaseFeatureExtractor(ABC):
    """Base class for audio feature extractors."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def extract_features(self, audio: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract features from audio."""
        pass


class TraditionalFeatureExtractor(BaseFeatureExtractor):
    """Traditional audio feature extractor (MFCC, spectral features, etc.)."""
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 2048,
                 hop_length: Optional[int] = None, n_mels: int = 80, n_mfcc: int = 13):
        """
        Initialize traditional feature extractor.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length between frames
            n_mels: Number of mel filter banks
            n_mfcc: Number of MFCC coefficients
        """
        super().__init__(sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
        # Initialize processors
        self.stft_processor = STFTProcessor(sample_rate, n_fft, self.hop_length)
        self.mel_processor = MelSpectrogramProcessor(
            sample_rate, n_fft, self.hop_length, n_mels
        )
        self.mfcc_processor = MFCCProcessor(
            sample_rate, n_mfcc, n_fft, self.hop_length, n_mels
        )
        self.spectral_processor = SpectralProcessor(sample_rate, n_fft, self.hop_length)
    
    def extract_features(self, audio: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract comprehensive traditional features.
        
        Args:
            audio: Input audio
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        try:
            # Spectral features
            features['mel_spectrogram'] = self.mel_processor.mel_spectrogram(audio)
            features['log_mel_spectrogram'] = self.mel_processor.log_mel_spectrogram(audio)
            features['mfcc'] = self.mfcc_processor.mfcc(audio)
            features['mfcc_delta'] = self.mfcc_processor.delta_features(features['mfcc'])
            features['mfcc_delta_delta'] = self.mfcc_processor.delta_features(features['mfcc_delta'])
            
            # Statistical spectral features
            features['spectral_centroid'] = self.spectral_processor.spectral_centroid(audio)
            features['spectral_rolloff'] = self.spectral_processor.spectral_rolloff(audio)
            features['zero_crossing_rate'] = self.spectral_processor.zero_crossing_rate(audio)
            
            # Time-domain features
            features.update(self._extract_time_domain_features(audio))
            
            # Statistical features
            features.update(self._extract_statistical_features(audio))
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            raise FeatureExtractorError(f"Feature extraction failed: {e}")
        
        return features
    
    def _extract_time_domain_features(self, audio: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract time-domain features."""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        features = {}
        
        # Energy
        features['energy'] = torch.sum(audio ** 2)
        features['log_energy'] = torch.log(features['energy'] + 1e-8)
        
        # RMS
        features['rms'] = torch.sqrt(torch.mean(audio ** 2))
        
        # Peak amplitude
        features['peak_amplitude'] = torch.max(torch.abs(audio))
        
        # Crest factor
        features['crest_factor'] = features['peak_amplitude'] / (features['rms'] + 1e-8)
        
        return features
    
    def _extract_statistical_features(self, audio: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract statistical features."""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        features = {}
        
        # Basic statistics
        features['mean'] = torch.mean(audio)
        features['std'] = torch.std(audio)
        features['skewness'] = self._compute_skewness(audio)
        features['kurtosis'] = self._compute_kurtosis(audio)
        
        # Percentiles
        sorted_audio, _ = torch.sort(audio)
        n = len(sorted_audio)
        features['median'] = sorted_audio[n // 2]
        features['q25'] = sorted_audio[n // 4]
        features['q75'] = sorted_audio[3 * n // 4]
        features['iqr'] = features['q75'] - features['q25']
        
        return features
    
    def _compute_skewness(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute skewness of audio signal."""
        mean = torch.mean(audio)
        std = torch.std(audio)
        skewness = torch.mean(((audio - mean) / (std + 1e-8)) ** 3)
        return skewness
    
    def _compute_kurtosis(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute kurtosis of audio signal."""
        mean = torch.mean(audio)
        std = torch.std(audio)
        kurtosis = torch.mean(((audio - mean) / (std + 1e-8)) ** 4) - 3
        return kurtosis


class DeepLearningFeatureExtractor(BaseFeatureExtractor):
    """Deep learning-based feature extractor using pre-trained models."""
    
    def __init__(self, sample_rate: int = 16000, model_name: str = "wav2vec2",
                 layer_idx: Optional[int] = None):
        """
        Initialize deep learning feature extractor.
        
        Args:
            sample_rate: Audio sample rate
            model_name: Pre-trained model to use ('wav2vec2', 'hubert', 'whisper')
            layer_idx: Layer index to extract features from (None for last layer)
        """
        super().__init__(sample_rate)
        self.model_name = model_name.lower()
        self.layer_idx = layer_idx
        self.model = None
        self.processor = None
        
        self._load_pretrained_model()
    
    def _load_pretrained_model(self):
        """Load pre-trained model for feature extraction."""
        try:
            if self.model_name == "wav2vec2":
                self._load_wav2vec2()
            elif self.model_name == "hubert":
                self._load_hubert()
            elif self.model_name == "whisper":
                self._load_whisper_encoder()
            else:
                raise FeatureExtractorError(f"Unsupported model: {self.model_name}")
            
        except ImportError as e:
            raise FeatureExtractorError(f"Required library not available: {e}")
        except Exception as e:
            raise FeatureExtractorError(f"Failed to load model {self.model_name}: {e}")
    
    def _load_wav2vec2(self):
        """Load Wav2Vec2 model."""
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        
        model_name = "facebook/wav2vec2-base"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()
    
    def _load_hubert(self):
        """Load HuBERT model."""
        from transformers import HubertProcessor, HubertModel
        
        model_name = "facebook/hubert-base-ls960"
        self.processor = HubertProcessor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name)
        self.model.eval()
    
    def _load_whisper_encoder(self):
        """Load Whisper encoder for feature extraction."""
        from transformers import WhisperProcessor, WhisperModel
        
        model_name = "openai/whisper-base"
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name)
        self.model.eval()
    
    def extract_features(self, audio: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract deep learning features.
        
        Args:
            audio: Input audio
            
        Returns:
            Dictionary of extracted features
        """
        if self.model is None or self.processor is None:
            raise FeatureExtractorError("Model not loaded")
        
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        features = {}
        
        try:
            with torch.no_grad():
                if self.model_name in ["wav2vec2", "hubert"]:
                    # Process input
                    inputs = self.processor(
                        audio, 
                        sampling_rate=self.sample_rate, 
                        return_tensors="pt"
                    )
                    
                    # Extract features
                    outputs = self.model(**inputs, output_hidden_states=True)
                    
                    if self.layer_idx is not None:
                        hidden_states = outputs.hidden_states[self.layer_idx]
                    else:
                        hidden_states = outputs.last_hidden_state
                    
                    features['deep_features'] = hidden_states.squeeze(0)
                    features['pooled_features'] = torch.mean(hidden_states, dim=1).squeeze(0)
                
                elif self.model_name == "whisper":
                    # Process input
                    inputs = self.processor(
                        audio,
                        sampling_rate=self.sample_rate,
                        return_tensors="pt"
                    )
                    
                    # Extract encoder features
                    encoder_outputs = self.model.encoder(
                        inputs.input_features,
                        output_hidden_states=True
                    )
                    
                    if self.layer_idx is not None:
                        hidden_states = encoder_outputs.hidden_states[self.layer_idx]
                    else:
                        hidden_states = encoder_outputs.last_hidden_state
                    
                    features['deep_features'] = hidden_states.squeeze(0)
                    features['pooled_features'] = torch.mean(hidden_states, dim=1).squeeze(0)
                
        except Exception as e:
            self.logger.error(f"Deep learning feature extraction failed: {e}")
            raise FeatureExtractorError(f"Deep learning feature extraction failed: {e}")
        
        return features


class CombinedFeatureExtractor(BaseFeatureExtractor):
    """Combined feature extractor using both traditional and deep learning features."""
    
    def __init__(self, sample_rate: int = 16000, 
                 include_traditional: bool = True,
                 include_deep: bool = True,
                 deep_model: str = "wav2vec2",
                 feature_selection: Optional[List[str]] = None):
        """
        Initialize combined feature extractor.
        
        Args:
            sample_rate: Audio sample rate
            include_traditional: Whether to include traditional features
            include_deep: Whether to include deep learning features
            deep_model: Deep learning model to use
            feature_selection: List of specific features to extract (None for all)
        """
        super().__init__(sample_rate)
        self.include_traditional = include_traditional
        self.include_deep = include_deep
        self.feature_selection = feature_selection
        
        # Initialize extractors
        self.traditional_extractor = None
        self.deep_extractor = None
        
        if include_traditional:
            self.traditional_extractor = TraditionalFeatureExtractor(sample_rate)
        
        if include_deep:
            try:
                self.deep_extractor = DeepLearningFeatureExtractor(
                    sample_rate, deep_model
                )
            except Exception as e:
                self.logger.warning(f"Failed to load deep learning extractor: {e}")
                self.include_deep = False
    
    def extract_features(self, audio: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract combined features.
        
        Args:
            audio: Input audio
            
        Returns:
            Dictionary of extracted features
        """
        all_features = {}
        
        # Extract traditional features
        if self.include_traditional and self.traditional_extractor:
            try:
                traditional_features = self.traditional_extractor.extract_features(audio)
                all_features.update(traditional_features)
            except Exception as e:
                self.logger.warning(f"Traditional feature extraction failed: {e}")
        
        # Extract deep learning features
        if self.include_deep and self.deep_extractor:
            try:
                deep_features = self.deep_extractor.extract_features(audio)
                all_features.update(deep_features)
            except Exception as e:
                self.logger.warning(f"Deep learning feature extraction failed: {e}")
        
        # Apply feature selection if specified
        if self.feature_selection:
            selected_features = {}
            for feature_name in self.feature_selection:
                if feature_name in all_features:
                    selected_features[feature_name] = all_features[feature_name]
                else:
                    self.logger.warning(f"Feature {feature_name} not found")
            all_features = selected_features
        
        return all_features
    
    def get_feature_vector(self, audio: Union[np.ndarray, torch.Tensor],
                          flatten: bool = True) -> torch.Tensor:
        """
        Extract features and return as a single vector.
        
        Args:
            audio: Input audio
            flatten: Whether to flatten multi-dimensional features
            
        Returns:
            Feature vector tensor
        """
        features = self.extract_features(audio)
        
        feature_vectors = []
        for name, feature in features.items():
            if isinstance(feature, torch.Tensor):
                if flatten and feature.ndim > 1:
                    # Flatten multi-dimensional features
                    if feature.ndim == 2:
                        # For 2D features, take mean across time dimension
                        feature = torch.mean(feature, dim=-1)
                    else:
                        feature = feature.flatten()
                elif feature.ndim == 0:
                    # Convert scalar to 1D tensor
                    feature = feature.unsqueeze(0)
                
                feature_vectors.append(feature.flatten())
        
        if feature_vectors:
            return torch.cat(feature_vectors)
        else:
            return torch.tensor([])


class FrameBasedFeatureExtractor(BaseFeatureExtractor):
    """Frame-based feature extractor for real-time processing."""
    
    def __init__(self, sample_rate: int = 16000, frame_length: int = 2048,
                 hop_length: Optional[int] = None, feature_type: str = "mfcc"):
        """
        Initialize frame-based feature extractor.
        
        Args:
            sample_rate: Audio sample rate
            frame_length: Frame length in samples
            hop_length: Hop length between frames
            feature_type: Type of features to extract per frame
        """
        super().__init__(sample_rate)
        self.frame_length = frame_length
        self.hop_length = hop_length or frame_length // 2
        self.feature_type = feature_type.lower()
        
        # Initialize appropriate processor
        if self.feature_type == "mfcc":
            self.processor = MFCCProcessor(sample_rate)
        elif self.feature_type == "mel":
            self.processor = MelSpectrogramProcessor(sample_rate)
        elif self.feature_type == "spectral":
            self.processor = SpectralProcessor(sample_rate)
        else:
            raise FeatureExtractorError(f"Unsupported feature type: {feature_type}")
    
    def extract_features(self, audio: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract frame-based features.
        
        Args:
            audio: Input audio
            
        Returns:
            Dictionary of frame-based features
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        features = {}
        
        # Extract features based on type
        if self.feature_type == "mfcc":
            features['mfcc_frames'] = self.processor.mfcc(audio)
        elif self.feature_type == "mel":
            features['mel_frames'] = self.processor.mel_spectrogram(audio)
        elif self.feature_type == "spectral":
            features['spectral_centroid_frames'] = self.processor.spectral_centroid(audio)
            features['spectral_rolloff_frames'] = self.processor.spectral_rolloff(audio)
        
        return features
    
    def process_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Process a single frame and return features.
        
        Args:
            frame: Single audio frame
            
        Returns:
            Feature vector for the frame
        """
        if len(frame) < self.frame_length:
            # Pad frame if too short
            padding = self.frame_length - len(frame)
            frame = torch.cat([frame, torch.zeros(padding)])
        
        features = self.extract_features(frame)
        
        # Return first feature as frame representation
        feature_name = list(features.keys())[0]
        feature = features[feature_name]
        
        if feature.ndim > 1:
            # Take first column for frame-based processing
            return feature[:, 0]
        else:
            return feature


def create_feature_extractor(extractor_type: str, **kwargs) -> BaseFeatureExtractor:
    """
    Factory function to create feature extractors.
    
    Args:
        extractor_type: Type of extractor ('traditional', 'deep', 'combined', 'frame')
        **kwargs: Additional arguments for extractor initialization
        
    Returns:
        Feature extractor instance
    """
    extractor_type = extractor_type.lower()
    
    if extractor_type == "traditional":
        return TraditionalFeatureExtractor(**kwargs)
    elif extractor_type == "deep":
        return DeepLearningFeatureExtractor(**kwargs)
    elif extractor_type == "combined":
        return CombinedFeatureExtractor(**kwargs)
    elif extractor_type == "frame":
        return FrameBasedFeatureExtractor(**kwargs)
    else:
        raise FeatureExtractorError(f"Unsupported extractor type: {extractor_type}")


# Alias for backward compatibility
FeatureExtractor = TraditionalFeatureExtractor

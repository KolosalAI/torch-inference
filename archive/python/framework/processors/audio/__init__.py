"""
Audio processors package for the PyTorch inference framework.

This package provides comprehensive audio processing capabilities including:
- Audio preprocessing and loading
- Spectrogram and frequency domain processing
- Feature extraction for machine learning
- Audio data augmentation
"""

from .audio_preprocessor import (
    AudioPreprocessorError,
    BaseAudioPreprocessor,
    AudioLoader,
    AudioNormalizer,
    VoiceActivityDetector,
    AudioChunker,
    ComprehensiveAudioPreprocessor
)

from .spectrogram_processor import (
    SpectrogramProcessorError,
    BaseSpectrogramProcessor,
    STFTProcessor,
    MelSpectrogramProcessor,
    MFCCProcessor,
    SpectralProcessor,
    create_spectrogram_processor
)

from .feature_extractor import (
    FeatureExtractorError,
    BaseFeatureExtractor,
    TraditionalFeatureExtractor,
    DeepLearningFeatureExtractor,
    CombinedFeatureExtractor,
    FrameBasedFeatureExtractor,
    create_feature_extractor
)

from .audio_augmentation import (
    AudioAugmentationError,
    BaseAudioAugmentation,
    TimeStretchAugmentation,
    PitchShiftAugmentation,
    NoiseAugmentation,
    VolumeAugmentation,
    SpectralMaskingAugmentation,
    ReverbAugmentation,
    CompressionAugmentation,
    AudioAugmentationPipeline,
    create_default_augmentation_pipeline
)

__all__ = [
    # Preprocessing
    "AudioPreprocessorError",
    "BaseAudioPreprocessor",
    "AudioLoader",
    "AudioNormalizer", 
    "VoiceActivityDetector",
    "AudioChunker",
    "ComprehensiveAudioPreprocessor",
    
    # Spectrogram processing
    "SpectrogramProcessorError",
    "BaseSpectrogramProcessor",
    "STFTProcessor",
    "MelSpectrogramProcessor",
    "MFCCProcessor",
    "SpectralProcessor",
    "create_spectrogram_processor",
    
    # Feature extraction
    "FeatureExtractorError",
    "BaseFeatureExtractor",
    "TraditionalFeatureExtractor",
    "DeepLearningFeatureExtractor",
    "CombinedFeatureExtractor",
    "FrameBasedFeatureExtractor",
    "create_feature_extractor",
    
    # Augmentation
    "AudioAugmentationError",
    "BaseAudioAugmentation",
    "TimeStretchAugmentation",
    "PitchShiftAugmentation",
    "NoiseAugmentation",
    "VolumeAugmentation",
    "SpectralMaskingAugmentation",
    "ReverbAugmentation",
    "CompressionAugmentation",
    "AudioAugmentationPipeline",
    "create_default_augmentation_pipeline"
]

def create_audio_processor(processor_type: str, **kwargs):
    """
    Factory function to create audio processors.
    
    Args:
        processor_type: Type of processor ('preprocessor', 'spectrogram', 'features', 'augmentation')
        **kwargs: Additional arguments for processor initialization
        
    Returns:
        Audio processor instance
    """
    processor_type = processor_type.lower()
    
    if processor_type == "preprocessor":
        return ComprehensiveAudioPreprocessor(**kwargs)
    elif processor_type == "spectrogram":
        spectrogram_type = kwargs.pop('spectrogram_type', 'mel')
        return create_spectrogram_processor(spectrogram_type, **kwargs)
    elif processor_type == "features":
        extractor_type = kwargs.pop('extractor_type', 'combined')
        return create_feature_extractor(extractor_type, **kwargs)
    elif processor_type == "augmentation":
        intensity = kwargs.pop('intensity', 'medium')
        sample_rate = kwargs.pop('sample_rate', 16000)
        return create_default_augmentation_pipeline(sample_rate, intensity)
    else:
        raise ValueError(f"Unsupported processor type: {processor_type}")

def get_recommended_processors(task: str = "stt", sample_rate: int = 16000):
    """
    Get recommended processors for specific tasks.
    
    Args:
        task: Task type ('stt', 'tts', 'classification')
        sample_rate: Audio sample rate
        
    Returns:
        Dictionary of recommended processors
    """
    if task.lower() == "stt":
        return {
            "preprocessor": ComprehensiveAudioPreprocessor(
                sample_rate=sample_rate,
                normalize=True,
                enable_vad=True
            ),
            "features": create_feature_extractor(
                "combined",
                sample_rate=sample_rate,
                include_traditional=True,
                include_deep=True,
                deep_model="whisper"
            ),
            "augmentation": create_default_augmentation_pipeline(
                sample_rate=sample_rate,
                intensity="medium"
            )
        }
    elif task.lower() == "tts":
        return {
            "preprocessor": ComprehensiveAudioPreprocessor(
                sample_rate=sample_rate,
                normalize=True,
                enable_vad=False  # Keep all audio for TTS
            ),
            "spectrogram": create_spectrogram_processor(
                "mel",
                sample_rate=sample_rate,
                n_mels=80
            ),
            "augmentation": create_default_augmentation_pipeline(
                sample_rate=sample_rate,
                intensity="light"  # Lighter augmentation for TTS
            )
        }
    elif task.lower() == "classification":
        return {
            "preprocessor": ComprehensiveAudioPreprocessor(
                sample_rate=sample_rate,
                normalize=True,
                enable_vad=True
            ),
            "features": create_feature_extractor(
                "traditional",
                sample_rate=sample_rate
            ),
            "augmentation": create_default_augmentation_pipeline(
                sample_rate=sample_rate,
                intensity="heavy"
            )
        }
    else:
        raise ValueError(f"Unsupported task: {task}")

# Quick access functions
def load_and_preprocess_audio(file_path: str, sample_rate: int = 16000, **kwargs):
    """
    Quick function to load and preprocess audio.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        **kwargs: Additional preprocessing options
        
    Returns:
        Preprocessed audio array
    """
    preprocessor = ComprehensiveAudioPreprocessor(sample_rate=sample_rate, **kwargs)
    return preprocessor.process(file_path)

def extract_audio_features(audio, sample_rate: int = 16000, feature_type: str = "combined", **kwargs):
    """
    Quick function to extract audio features.
    
    Args:
        audio: Audio array or file path
        sample_rate: Audio sample rate
        feature_type: Type of features to extract
        **kwargs: Additional feature extraction options
        
    Returns:
        Extracted features dictionary
    """
    extractor = create_feature_extractor(feature_type, sample_rate=sample_rate, **kwargs)
    return extractor.extract_features(audio)

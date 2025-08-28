"""
Audio models package for the PyTorch inference framework.

This package provides comprehensive support for audio models including:
- Text-to-Speech (TTS) models
- Speech-to-Text (STT) models  
- Audio classification and enhancement models
- Audio preprocessing and feature extraction
"""

from .audio_base import (
    BaseAudioModel,
    BaseTTSModel, 
    BaseSTTModel,
    AudioModel,
    AudioModelType,
    AudioFormat,
    AudioTask,
    AudioConfig,
    AudioModelConfig,
    TTSConfig,
    STTConfig,
    AudioMetadata,
    AudioModelError,
    AudioInputError,
    AudioProcessingError
)

from .tts_models import (
    TTSModel,
    HuggingFaceTTSModel,
    TorchAudioTTSModel,
    CustomTTSModel,
    SpeechSynthesizer,
    create_tts_model,
    get_tts_model,
    available_tts_models,
    TTS_MODEL_REGISTRY
)

from .stt_models import (
    STTModel,
    WhisperSTTModel,
    Wav2Vec2STTModel,
    CustomSTTModel,
    SpeechRecognizer,
    create_stt_model,
    get_stt_model,
    available_stt_models,
    STT_MODEL_REGISTRY
)

__all__ = [
    # Base classes
    "BaseAudioModel",
    "BaseTTSModel",
    "BaseSTTModel",
    "AudioModel",
    
    # Enums and configs
    "AudioModelType",
    "AudioFormat",
    "AudioTask", 
    "AudioConfig",
    "AudioModelConfig",
    "TTSConfig",
    "STTConfig",
    "AudioMetadata",
    "AudioModelError",
    "AudioInputError",
    "AudioProcessingError",
    
    # TTS models
    "TTSModel",
    "HuggingFaceTTSModel",
    "TorchAudioTTSModel", 
    "CustomTTSModel",
    "SpeechSynthesizer",
    "create_tts_model",
    "get_tts_model",
    "available_tts_models",
    "TTS_MODEL_REGISTRY",
    
    # STT models
    "STTModel",
    "WhisperSTTModel",
    "Wav2Vec2STTModel",
    "CustomSTTModel",
    "SpeechRecognizer", 
    "create_stt_model",
    "get_stt_model",
    "available_stt_models",
    "STT_MODEL_REGISTRY"
]

# Model registry for easy access
AUDIO_MODEL_REGISTRY = {
    **TTS_MODEL_REGISTRY,
    **STT_MODEL_REGISTRY
}

def list_available_models() -> dict:
    """List all available audio models."""
    return AUDIO_MODEL_REGISTRY

def get_model_info(model_name: str) -> dict:
    """Get information about a specific model."""
    return AUDIO_MODEL_REGISTRY.get(model_name, {})

def create_audio_model(model_name: str, config, **kwargs):
    """
    Create an audio model by name.
    
    Args:
        model_name: Name of the model from registry
        config: Inference configuration
        **kwargs: Additional arguments
        
    Returns:
        Audio model instance
    """
    if model_name not in AUDIO_MODEL_REGISTRY:
        raise AudioModelError(f"Unknown model: {model_name}")
    
    model_info = AUDIO_MODEL_REGISTRY[model_name]
    model_type = model_info["type"]
    
    if model_type == "whisper":
        return WhisperSTTModel(config, model_size=model_info["model_size"], **kwargs)
    elif model_type == "wav2vec2":
        return Wav2Vec2STTModel(config, model_name=model_info["model_name"], **kwargs)
    elif model_type == "huggingface":
        return HuggingFaceTTSModel(config, model_name=model_info["model_name"], **kwargs)
    elif model_type == "torchaudio":
        return TorchAudioTTSModel(config, model_name=model_info["model_name"], **kwargs)
    else:
        raise AudioModelError(f"Unsupported model type: {model_type}")

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
    AudioModelType,
    AudioFormat,
    AudioConfig,
    TTSConfig,
    STTConfig,
    AudioMetadata,
    AudioModelError
)

from .tts_models import (
    HuggingFaceTTSModel,
    TorchAudioTTSModel,
    CustomTTSModel,
    create_tts_model,
    TTS_MODEL_REGISTRY
)

from .stt_models import (
    WhisperSTTModel,
    Wav2Vec2STTModel,
    CustomSTTModel,
    create_stt_model,
    STT_MODEL_REGISTRY
)

__all__ = [
    # Base classes
    "BaseAudioModel",
    "BaseTTSModel",
    "BaseSTTModel",
    
    # Enums and configs
    "AudioModelType",
    "AudioFormat", 
    "AudioConfig",
    "TTSConfig",
    "STTConfig",
    "AudioMetadata",
    "AudioModelError",
    
    # TTS models
    "HuggingFaceTTSModel",
    "TorchAudioTTSModel", 
    "CustomTTSModel",
    "create_tts_model",
    "TTS_MODEL_REGISTRY",
    
    # STT models
    "WhisperSTTModel",
    "Wav2Vec2STTModel",
    "CustomSTTModel", 
    "create_stt_model",
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

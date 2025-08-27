"""
Text-to-Speech (TTS) model implementations for the PyTorch inference framework.

This module provides implementations for various TTS models including:
- HuggingFace TTS models (SpeechT5, Bark, VALL-E)
- TorchAudio TTS models (Tacotron2, WaveGlow)
- Custom TTS model adapters
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

from .audio_base import BaseTTSModel, AudioConfig, TTSConfig, AudioModelError
from ...core.config import InferenceConfig

logger = logging.getLogger(__name__)


class HuggingFaceTTSModel(BaseTTSModel):
    """
    HuggingFace TTS model adapter supporting various architectures.
    
    Supports models like:
    - SpeechT5
    - Bark
    - VALL-E
    - FastSpeech2
    """
    
    def __init__(self, config: InferenceConfig, model_name: str = "microsoft/speecht5_tts",
                 audio_config: Optional[AudioConfig] = None, tts_config: Optional[TTSConfig] = None):
        super().__init__(config, audio_config, tts_config)
        self.model_name = model_name
        self.processor = None
        self.vocoder = None
        self.speaker_embeddings = None
        
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load HuggingFace TTS model and components."""
        try:
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            import datasets
            
            self.logger.info(f"Loading HuggingFace TTS model: {self.model_name}")
            
            # Load processor and model
            self.processor = SpeechT5Processor.from_pretrained(self.model_name)
            self.model = SpeechT5ForTextToSpeech.from_pretrained(self.model_name)
            
            # Load vocoder for audio generation
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            # Load speaker embeddings
            try:
                embeddings_dataset = datasets.load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
            except Exception as e:
                self.logger.warning(f"Could not load speaker embeddings: {e}")
                # Create default speaker embedding
                self.speaker_embeddings = torch.randn(1, 512)
            
            # Move to device
            self.model.to(self.device)
            self.vocoder.to(self.device)
            self.speaker_embeddings = self.speaker_embeddings.to(self.device)
            
            # Set evaluation mode
            self.model.eval()
            self.vocoder.eval()
            
            self._is_loaded = True
            self.logger.info("HuggingFace TTS model loaded successfully")
            
        except ImportError as e:
            raise AudioModelError(f"HuggingFace transformers not available: {e}")
        except Exception as e:
            raise AudioModelError(f"Failed to load HuggingFace TTS model: {e}")
    
    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """Convert text to model input tensor."""
        if not self.processor:
            raise AudioModelError("Model not loaded")
        
        # Process text input
        inputs = self.processor(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        return input_ids
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Generate speech from text tokens."""
        if not self._is_loaded:
            raise AudioModelError("Model not loaded")
        
        with torch.no_grad():
            # Generate speech features
            speech = self.model.generate_speech(
                inputs, 
                self.speaker_embeddings, 
                vocoder=self.vocoder
            )
        
        return speech
    
    def synthesize_speech(self, text: str, **kwargs) -> np.ndarray:
        """
        Synthesize speech from text using HuggingFace model.
        
        Args:
            text: Input text to synthesize
            **kwargs: Additional parameters (voice, speed, etc.)
            
        Returns:
            Audio array
        """
        # Process text
        input_tensor = self._text_to_tensor(text)
        
        # Generate speech
        speech_tensor = self.forward(input_tensor)
        
        # Convert to numpy
        audio_array = speech_tensor.cpu().numpy()
        
        # Apply post-processing based on TTS config
        if self.tts_config.speed != 1.0:
            audio_array = self._adjust_speed(audio_array, self.tts_config.speed)
        
        if self.tts_config.volume != 1.0:
            audio_array = audio_array * self.tts_config.volume
        
        return audio_array
    
    def _adjust_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Adjust audio playback speed."""
        try:
            import librosa
            return librosa.effects.time_stretch(audio, rate=speed)
        except ImportError:
            self.logger.warning("librosa not available, cannot adjust speed")
            return audio
    
    def process_audio_output(self, outputs: torch.Tensor) -> np.ndarray:
        """Process model output to audio array."""
        if isinstance(outputs, torch.Tensor):
            return outputs.detach().cpu().numpy()
        return outputs


class TorchAudioTTSModel(BaseTTSModel):
    """
    TorchAudio TTS model adapter supporting Tacotron2 and WaveGlow.
    """
    
    def __init__(self, config: InferenceConfig, model_name: str = "tacotron2",
                 audio_config: Optional[AudioConfig] = None, tts_config: Optional[TTSConfig] = None):
        super().__init__(config, audio_config, tts_config)
        self.model_name = model_name
        self.text_processor = None
        
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load TorchAudio TTS model."""
        try:
            import torchaudio
            
            self.logger.info(f"Loading TorchAudio TTS model: {self.model_name}")
            
            if self.model_name.lower() == "tacotron2":
                # Load Tacotron2 bundle
                bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
                self.model = bundle.get_tacotron2().to(self.device)
                self.vocoder = bundle.get_vocoder().to(self.device)
                self.text_processor = bundle.get_text_processor()
                
            else:
                raise AudioModelError(f"Unsupported TorchAudio model: {self.model_name}")
            
            # Set evaluation mode
            self.model.eval()
            if hasattr(self, 'vocoder'):
                self.vocoder.eval()
            
            self._is_loaded = True
            self.logger.info("TorchAudio TTS model loaded successfully")
            
        except ImportError as e:
            raise AudioModelError(f"TorchAudio not available: {e}")
        except Exception as e:
            raise AudioModelError(f"Failed to load TorchAudio TTS model: {e}")
    
    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """Convert text to phoneme tensor."""
        if not self.text_processor:
            raise AudioModelError("Model not loaded")
        
        # Convert text to phonemes
        processed_text = self.text_processor(text)
        return processed_text.to(self.device)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Generate speech from phoneme tokens."""
        if not self._is_loaded:
            raise AudioModelError("Model not loaded")
        
        with torch.no_grad():
            # Generate mel-spectrogram
            mel_spec, _ = self.model.infer(inputs)
            
            # Generate waveform using vocoder
            if hasattr(self, 'vocoder') and self.vocoder:
                waveform = self.vocoder(mel_spec)
            else:
                # Fallback: convert mel-spec to waveform (simplified)
                waveform = self._mel_to_waveform(mel_spec)
        
        return waveform.squeeze()
    
    def synthesize_speech(self, text: str, **kwargs) -> np.ndarray:
        """
        Synthesize speech from text using TorchAudio model.
        
        Args:
            text: Input text to synthesize
            **kwargs: Additional parameters
            
        Returns:
            Audio array
        """
        # Process text
        input_tensor = self._text_to_tensor(text)
        
        # Generate speech
        speech_tensor = self.forward(input_tensor)
        
        # Convert to numpy
        audio_array = speech_tensor.cpu().numpy()
        
        return audio_array
    
    def _mel_to_waveform(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Convert mel-spectrogram to waveform (simplified Griffin-Lim)."""
        try:
            import torchaudio.functional as F
            return F.griffinlim(mel_spec)
        except ImportError:
            self.logger.warning("Cannot convert mel-spec without torchaudio")
            # Return silence as fallback
            return torch.zeros(mel_spec.size(-1) * 256, device=self.device)
    
    def process_audio_output(self, outputs: torch.Tensor) -> np.ndarray:
        """Process model output to audio array."""
        if isinstance(outputs, torch.Tensor):
            return outputs.detach().cpu().numpy()
        return outputs


class CustomTTSModel(BaseTTSModel):
    """
    Custom TTS model adapter for user-defined models.
    """
    
    def __init__(self, config: InferenceConfig, 
                 audio_config: Optional[AudioConfig] = None, tts_config: Optional[TTSConfig] = None):
        super().__init__(config, audio_config, tts_config)
        self.tokenizer = None
        
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load custom TTS model."""
        try:
            self.logger.info(f"Loading custom TTS model from: {model_path}")
            
            # Load PyTorch model
            if isinstance(model_path, str):
                model_path = Path(model_path)
            
            if model_path.suffix == '.pt' or model_path.suffix == '.pth':
                self.model = torch.load(model_path, map_location=self.device)
            elif model_path.is_dir():
                # Load from directory (e.g., saved model)
                model_file = model_path / "model.pt"
                if model_file.exists():
                    self.model = torch.load(model_file, map_location=self.device)
                else:
                    raise FileNotFoundError(f"Model file not found in {model_path}")
            else:
                raise AudioModelError(f"Unsupported model format: {model_path}")
            
            # Set evaluation mode
            self.model.eval()
            self.model.to(self.device)
            
            self._is_loaded = True
            self.logger.info("Custom TTS model loaded successfully")
            
        except Exception as e:
            raise AudioModelError(f"Failed to load custom TTS model: {e}")
    
    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """Convert text to tensor representation."""
        # Simple character-level encoding as fallback
        if self.tokenizer:
            return self.tokenizer(text)
        else:
            # Basic character encoding
            chars = list(text.lower())
            char_to_idx = {chr(i): i - ord('a') for i in range(ord('a'), ord('z') + 1)}
            char_to_idx[' '] = 26
            char_to_idx[''] = 27  # Unknown character
            
            indices = [char_to_idx.get(c, 27) for c in chars]
            return torch.tensor(indices, device=self.device).unsqueeze(0)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through custom model."""
        if not self._is_loaded:
            raise AudioModelError("Model not loaded")
        
        with torch.no_grad():
            outputs = self.model(inputs)
        
        return outputs
    
    def synthesize_speech(self, text: str, **kwargs) -> np.ndarray:
        """
        Synthesize speech using custom model.
        
        Args:
            text: Input text to synthesize
            **kwargs: Additional parameters
            
        Returns:
            Audio array
        """
        # Process text
        input_tensor = self._text_to_tensor(text)
        
        # Generate speech
        speech_tensor = self.forward(input_tensor)
        
        # Convert to numpy
        audio_array = speech_tensor.cpu().numpy()
        
        # Ensure proper shape
        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze()
        
        return audio_array
    
    def process_audio_output(self, outputs: torch.Tensor) -> np.ndarray:
        """Process model output to audio array."""
        if isinstance(outputs, torch.Tensor):
            return outputs.detach().cpu().numpy()
        return outputs
    
    def set_tokenizer(self, tokenizer):
        """Set custom tokenizer for text processing."""
        self.tokenizer = tokenizer


def create_tts_model(model_type: str, config: InferenceConfig, **kwargs) -> BaseTTSModel:
    """
    Factory function to create TTS models.
    
    Args:
        model_type: Type of TTS model ('huggingface', 'torchaudio', 'custom')
        config: Inference configuration
        **kwargs: Additional arguments for model initialization
        
    Returns:
        TTS model instance
    """
    model_type = model_type.lower()
    
    if model_type == "huggingface" or model_type == "hf":
        return HuggingFaceTTSModel(config, **kwargs)
    elif model_type == "torchaudio" or model_type == "ta":
        return TorchAudioTTSModel(config, **kwargs)
    elif model_type == "custom":
        return CustomTTSModel(config, **kwargs)
    else:
        raise AudioModelError(f"Unsupported TTS model type: {model_type}")


# Available TTS models registry
TTS_MODEL_REGISTRY = {
    "speecht5": {
        "type": "huggingface",
        "model_name": "microsoft/speecht5_tts",
        "description": "Microsoft SpeechT5 TTS model"
    },
    "bark": {
        "type": "huggingface", 
        "model_name": "suno/bark",
        "description": "Suno Bark TTS model with voice cloning"
    },
    "tacotron2": {
        "type": "torchaudio",
        "model_name": "tacotron2",
        "description": "TorchAudio Tacotron2 with WaveRNN"
    }
}

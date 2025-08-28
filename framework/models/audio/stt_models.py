"""
Speech-to-Text (STT) model implementations for the PyTorch inference framework.

This module provides implementations for various STT models including:
- OpenAI Whisper models
- HuggingFace Wav2Vec2 models  
- Facebook Wav2Vec2
- Custom STT model adapters
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

from .audio_base import BaseSTTModel, AudioConfig, STTConfig, AudioModelError
from ...core.config import InferenceConfig

logger = logging.getLogger(__name__)


class WhisperSTTModel(BaseSTTModel):
    """
    OpenAI Whisper model adapter for speech-to-text.
    
    Supports all Whisper model sizes:
    - tiny, base, small, medium, large, large-v2, large-v3
    """
    
    def __init__(self, config: InferenceConfig, model_size: str = "base",
                 audio_config: Optional[AudioConfig] = None, stt_config: Optional[STTConfig] = None):
        super().__init__(config, audio_config, stt_config)
        self.model_size = model_size
        self.processor = None
        self._target_sample_rate = 16000  # Whisper requires 16kHz
        
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load Whisper model."""
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            self.logger.info(f"Loading Whisper model: {self.model_size}")
            
            # Determine model name
            if isinstance(model_path, (str, Path)) and Path(model_path).exists():
                # Load from local path
                model_name = str(model_path)
            else:
                # Load from HuggingFace hub
                model_name = f"openai/whisper-{self.model_size}"
            
            # Load processor and model
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            # Update audio config for Whisper
            self.audio_config.sample_rate = self._target_sample_rate
            
            self._is_loaded = True
            self.logger.info(f"Whisper model {self.model_size} loaded successfully")
            
        except ImportError as e:
            raise AudioModelError(f"Transformers library not available: {e}")
        except Exception as e:
            raise AudioModelError(f"Failed to load Whisper model: {e}")
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Generate text tokens from audio features."""
        if not self._is_loaded:
            raise AudioModelError("Model not loaded")
        
        with torch.no_grad():
            # Generate transcription
            if self.stt_config.language != "auto":
                # Force language
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=self.stt_config.language, task="transcribe"
                )
                predicted_ids = self.model.generate(
                    inputs, 
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=448,
                    num_beams=self.stt_config.beam_size,
                    temperature=self.stt_config.temperature if self.stt_config.temperature > 0 else None
                )
            else:
                # Auto-detect language
                predicted_ids = self.model.generate(
                    inputs,
                    max_length=448,
                    num_beams=self.stt_config.beam_size,
                    temperature=self.stt_config.temperature if self.stt_config.temperature > 0 else None
                )
        
        return predicted_ids
    
    def transcribe_audio(self, audio: Union[str, np.ndarray, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio to text using Whisper.
        
        Args:
            audio: Audio input (file path, array, or tensor)
            **kwargs: Additional transcription parameters
            
        Returns:
            Dictionary with transcription results
        """
        # Process audio input
        if isinstance(audio, str):
            audio_array = self._load_audio_file(audio)
        elif isinstance(audio, torch.Tensor):
            audio_array = audio.cpu().numpy()
        else:
            audio_array = audio
        
        # Ensure mono and correct sample rate
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=0)
        
        # Process with Whisper processor
        inputs = self.processor(
            audio_array, 
            sampling_rate=self._target_sample_rate, 
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)
        
        # Generate transcription
        predicted_ids = self.forward(input_features)
        
        # Decode transcription
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        # Prepare result
        result = {
            "text": transcription[0] if transcription else "",
            "language": self.stt_config.language,
            "model": f"whisper-{self.model_size}",
            "confidence": 1.0  # Whisper doesn't provide confidence scores
        }
        
        # Add timestamps if requested (requires additional processing)
        if self.stt_config.enable_timestamps:
            result["timestamps"] = self._extract_timestamps(input_features, predicted_ids)
        
        return result
    
    def _extract_timestamps(self, input_features: torch.Tensor, predicted_ids: torch.Tensor) -> List[Dict[str, float]]:
        """Extract word-level timestamps (simplified implementation)."""
        # This is a simplified implementation
        # For full timestamp support, you'd need whisper-timestamped or similar
        try:
            # Basic timestamp estimation based on input length
            audio_duration = input_features.shape[-1] * 0.02  # Assuming 20ms per frame
            num_tokens = predicted_ids.shape[-1]
            
            timestamps = []
            for i in range(num_tokens):
                start_time = (i / num_tokens) * audio_duration
                end_time = ((i + 1) / num_tokens) * audio_duration
                timestamps.append({
                    "start": start_time,
                    "end": end_time,
                    "token_id": predicted_ids[0, i].item()
                })
            
            return timestamps
        except Exception as e:
            self.logger.warning(f"Failed to extract timestamps: {e}")
            return []
    
    def process_audio_output(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Process model output to transcription result."""
        # Decode the output tokens
        if self.processor:
            text = self.processor.batch_decode(outputs, skip_special_tokens=True)
            return {"text": text[0] if text else "", "confidence": 1.0}
        return {"text": "", "confidence": 0.0}


class Wav2Vec2STTModel(BaseSTTModel):
    """
    Wav2Vec2 model adapter for speech-to-text.
    
    Supports both Facebook and HuggingFace Wav2Vec2 models.
    """
    
    def __init__(self, config: InferenceConfig, model_name: str = "facebook/wav2vec2-base-960h",
                 audio_config: Optional[AudioConfig] = None, stt_config: Optional[STTConfig] = None):
        super().__init__(config, audio_config, stt_config)
        self.model_name = model_name
        self.processor = None
        self._target_sample_rate = 16000
        
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load Wav2Vec2 model."""
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            
            self.logger.info(f"Loading Wav2Vec2 model: {self.model_name}")
            
            # Load processor and model
            if isinstance(model_path, (str, Path)) and Path(model_path).exists():
                model_name = str(model_path)
            else:
                model_name = self.model_name
            
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            # Update audio config
            self.audio_config.sample_rate = self._target_sample_rate
            
            self._is_loaded = True
            self.logger.info("Wav2Vec2 model loaded successfully")
            
        except ImportError as e:
            raise AudioModelError(f"Transformers library not available: {e}")
        except Exception as e:
            raise AudioModelError(f"Failed to load Wav2Vec2 model: {e}")
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through Wav2Vec2 model."""
        if not self._is_loaded:
            raise AudioModelError("Model not loaded")
        
        with torch.no_grad():
            logits = self.model(inputs).logits
        
        return logits
    
    def transcribe_audio(self, audio: Union[str, np.ndarray, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio using Wav2Vec2.
        
        Args:
            audio: Audio input
            **kwargs: Additional parameters
            
        Returns:
            Transcription results
        """
        # Process audio input
        if isinstance(audio, str):
            audio_array = self._load_audio_file(audio)
        elif isinstance(audio, torch.Tensor):
            audio_array = audio.cpu().numpy()
        else:
            audio_array = audio
        
        # Ensure mono
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=0)
        
        # Process with Wav2Vec2 processor
        inputs = self.processor(
            audio_array,
            sampling_rate=self._target_sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        
        # Get model predictions
        logits = self.forward(input_values)
        
        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        
        # Calculate confidence (simplified)
        confidence = torch.softmax(logits, dim=-1).max(dim=-1)[0].mean().item()
        
        result = {
            "text": transcription[0] if transcription else "",
            "confidence": float(confidence),
            "model": "wav2vec2",
            "language": self.stt_config.language
        }
        
        return result
    
    def process_audio_output(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Process model output to transcription result."""
        if self.processor:
            predicted_ids = torch.argmax(outputs, dim=-1)
            text = self.processor.batch_decode(predicted_ids)
            confidence = torch.softmax(outputs, dim=-1).max(dim=-1)[0].mean().item()
            return {"text": text[0] if text else "", "confidence": float(confidence)}
        return {"text": "", "confidence": 0.0}


class CustomSTTModel(BaseSTTModel):
    """
    Custom STT model adapter for user-defined models.
    """
    
    def __init__(self, config: InferenceConfig,
                 audio_config: Optional[AudioConfig] = None, stt_config: Optional[STTConfig] = None):
        super().__init__(config, audio_config, stt_config)
        self.vocab = None
        self.decoder = None
        
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load custom STT model."""
        try:
            self.logger.info(f"Loading custom STT model from: {model_path}")
            
            if isinstance(model_path, str):
                model_path = Path(model_path)
            
            if model_path.suffix in ['.pt', '.pth']:
                # Load PyTorch model
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict):
                    self.model = checkpoint['model']
                    self.vocab = checkpoint.get('vocab', None)
                else:
                    self.model = checkpoint
            elif model_path.is_dir():
                # Load from directory
                model_file = model_path / "model.pt"
                vocab_file = model_path / "vocab.txt"
                
                if model_file.exists():
                    self.model = torch.load(model_file, map_location=self.device)
                if vocab_file.exists():
                    with open(vocab_file, 'r') as f:
                        self.vocab = [line.strip() for line in f]
            
            # Set evaluation mode
            self.model.eval()
            self.model.to(self.device)
            
            self._is_loaded = True
            self.logger.info("Custom STT model loaded successfully")
            
        except Exception as e:
            raise AudioModelError(f"Failed to load custom STT model: {e}")
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through custom model."""
        if not self._is_loaded:
            raise AudioModelError("Model not loaded")
        
        with torch.no_grad():
            outputs = self.model(inputs)
        
        return outputs
    
    def transcribe_audio(self, audio: Union[str, np.ndarray, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio using custom model.
        
        Args:
            audio: Audio input
            **kwargs: Additional parameters
            
        Returns:
            Transcription results
        """
        # Process audio input
        audio_tensor = self.process_audio_input(audio)
        
        # Get model predictions
        outputs = self.forward(audio_tensor)
        
        # Decode outputs
        if self.vocab:
            # Use vocabulary for decoding
            predicted_ids = torch.argmax(outputs, dim=-1)
            text_tokens = [self.vocab[idx] for idx in predicted_ids.cpu().numpy().flatten() if idx < len(self.vocab)]
            transcription = " ".join(text_tokens)
        else:
            # Fallback decoding
            transcription = f"Model output shape: {outputs.shape}"
        
        # Calculate confidence
        confidence = torch.softmax(outputs, dim=-1).max(dim=-1)[0].mean().item()
        
        result = {
            "text": transcription,
            "confidence": float(confidence),
            "model": "custom",
            "language": self.stt_config.language
        }
        
        return result
    
    def process_audio_output(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Process model output to transcription result."""
        if self.vocab:
            predicted_ids = torch.argmax(outputs, dim=-1)
            text_tokens = [self.vocab[idx] for idx in predicted_ids.cpu().numpy().flatten() if idx < len(self.vocab)]
            text = " ".join(text_tokens)
            confidence = torch.softmax(outputs, dim=-1).max(dim=-1)[0].mean().item()
            return {"text": text, "confidence": float(confidence)}
        return {"text": "Custom model output", "confidence": 0.5}
    
    def set_vocabulary(self, vocab: List[str]):
        """Set vocabulary for decoding."""
        self.vocab = vocab
    
    def set_decoder(self, decoder):
        """Set custom decoder for output processing."""
        self.decoder = decoder


def create_stt_model(model_type: str, config: InferenceConfig, **kwargs) -> BaseSTTModel:
    """
    Factory function to create STT models.
    
    Args:
        model_type: Type of STT model ('whisper', 'wav2vec2', 'custom')
        config: Inference configuration
        **kwargs: Additional arguments for model initialization
        
    Returns:
        STT model instance
    """
    model_type = model_type.lower()
    
    if model_type == "whisper":
        return WhisperSTTModel(config, **kwargs)
    elif model_type == "wav2vec2":
        return Wav2Vec2STTModel(config, **kwargs)
    elif model_type == "custom":
        return CustomSTTModel(config, **kwargs)
    else:
        raise AudioModelError(f"Unsupported STT model type: {model_type}")


# Available STT models registry
STT_MODEL_REGISTRY = {
    "whisper-tiny": {
        "type": "whisper",
        "model_size": "tiny",
        "description": "OpenAI Whisper Tiny (39M params, ~1GB)"
    },
    "whisper-base": {
        "type": "whisper", 
        "model_size": "base",
        "description": "OpenAI Whisper Base (74M params, ~1GB)"
    },
    "whisper-small": {
        "type": "whisper",
        "model_size": "small", 
        "description": "OpenAI Whisper Small (244M params, ~2GB)"
    },
    "whisper-medium": {
        "type": "whisper",
        "model_size": "medium",
        "description": "OpenAI Whisper Medium (769M params, ~5GB)"
    },
    "whisper-large": {
        "type": "whisper",
        "model_size": "large-v3",
        "description": "OpenAI Whisper Large v3 (1550M params, ~10GB)"
    },
    "wav2vec2-base": {
        "type": "wav2vec2",
        "model_name": "facebook/wav2vec2-base-960h",
        "description": "Facebook Wav2Vec2 Base trained on LibriSpeech"
    },
    "wav2vec2-large": {
        "type": "wav2vec2",
        "model_name": "facebook/wav2vec2-large-960h",
        "description": "Facebook Wav2Vec2 Large trained on LibriSpeech"
    }
}

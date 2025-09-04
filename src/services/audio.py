"""
Audio processing service for TTS and STT operations.
"""

import logging
import time
import base64
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.exceptions import AudioProcessingError, ModelNotFoundError, ServiceUnavailableError
# Import models directly from specific files
from ..models.api.audio import TTSRequest, TTSResponse, STTRequest, STTResponse, AudioProcessRequest, AudioValidationRequest, AudioValidationResponse

logger = logging.getLogger(__name__)


class AudioService:
    """Service for audio processing operations."""
    
    def __init__(self, model_manager=None):
        self.model_manager = model_manager
        self.logger = logger
    
    async def synthesize_speech(self, request: TTSRequest) -> TTSResponse:
        """
        Convert text to speech using specified TTS model.
        
        Args:
            request: TTS request with text and synthesis options
            
        Returns:
            TTSResponse with audio data or error
        """
        start_time = time.perf_counter()
        
        try:
            # Validate text length
            if len(request.text) > 5000:
                raise AudioProcessingError("Text too long. Maximum 5000 characters allowed.")
            
            # Import audio modules dynamically
            try:
                from framework.models.audio import create_tts_model, AudioModelError
            except ImportError as e:
                self.logger.error(f"Audio modules not available: {e}")
                raise ServiceUnavailableError("Audio processing not available. Install audio dependencies.")
            
            # Get or create TTS model
            tts_model = await self._get_or_create_tts_model(request.model_name)
            
            # Prepare synthesis parameters
            synthesis_params = {
                "voice": request.voice,
                "speed": request.speed,
                "pitch": request.pitch,
                "volume": request.volume,
                "language": request.language,
                "emotion": request.emotion
            }
            
            # Perform TTS synthesis
            result = tts_model.predict(request.text)
            
            # Extract audio data and metadata
            audio_data = result["audio"]
            sample_rate = result["sample_rate"]
            duration = result.get("duration", None)
            
            # Apply additional parameters
            if request.speed != 1.0 and hasattr(tts_model, '_adjust_speed'):
                audio_data = tts_model._adjust_speed(audio_data, request.speed)
            
            if request.volume != 1.0:
                audio_data = audio_data * request.volume
            
            # Recalculate duration after processing
            if duration is None and sample_rate > 0:
                duration = len(audio_data) / sample_rate
            
            # Convert to requested format and encode
            audio_bytes = self._convert_audio_to_bytes(audio_data, sample_rate, request.output_format)
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            processing_time = time.perf_counter() - start_time
            
            self.logger.info(
                f"TTS synthesis completed - Duration: {duration:.2f}s, "
                f"Processing time: {processing_time:.3f}s"
            )
            
            return TTSResponse(
                success=True,
                audio_data=audio_base64,
                audio_format=request.output_format,
                duration=duration,
                sample_rate=sample_rate,
                processing_time=processing_time,
                model_info={
                    "model_name": request.model_name,
                    "voice": request.voice,
                    "language": request.language,
                    "actual_model": result.get("model_name", request.model_name)
                }
            )
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            self.logger.error(f"TTS synthesis failed: {e}")
            return TTSResponse(
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    async def transcribe_speech(
        self, 
        audio_file_path: str, 
        model_name: str = "whisper-base",
        language: str = "auto",
        enable_timestamps: bool = True,
        **kwargs
    ) -> STTResponse:
        """
        Transcribe audio file to text using specified STT model.
        
        Args:
            audio_file_path: Path to audio file
            model_name: STT model to use
            language: Language code or 'auto'
            enable_timestamps: Include timestamps
            **kwargs: Additional STT parameters
            
        Returns:
            STTResponse with transcription or error
        """
        start_time = time.perf_counter()
        
        try:
            # Import audio modules
            try:
                from framework.models.audio import create_stt_model
                from framework.processors.audio import ComprehensiveAudioPreprocessor as AudioPreprocessor
            except ImportError as e:
                self.logger.error(f"Audio modules not available: {e}")
                raise ServiceUnavailableError("Audio processing not available.")
            
            # Validate file
            if not Path(audio_file_path).exists():
                raise AudioProcessingError(f"Audio file not found: {audio_file_path}")
            
            # Get or create STT model
            stt_model = await self._get_or_create_stt_model(model_name)
            
            # Load and preprocess audio
            from ..core.config import get_config
            config = get_config()
            
            # Create a mock inference config for audio processor
            from types import SimpleNamespace
            inference_config = SimpleNamespace()
            
            audio_processor = AudioPreprocessor(inference_config)
            audio_data, sample_rate = audio_processor.load_audio(audio_file_path)
            
            # Prepare transcription parameters
            transcription_params = {
                "language": language if language != "auto" else None,
                "enable_timestamps": enable_timestamps,
                **{k: v for k, v in kwargs.items() if v is not None}
            }
            
            # Perform transcription
            result = await stt_model.transcribe(
                audio_data=audio_data,
                sample_rate=sample_rate,
                **transcription_params
            )
            
            processing_time = time.perf_counter() - start_time
            
            self.logger.info(
                f"STT transcription completed - Text length: {len(result.get('text', ''))}, "
                f"Processing time: {processing_time:.3f}s"
            )
            
            return STTResponse(
                success=True,
                text=result.get("text", ""),
                segments=result.get("segments", []) if enable_timestamps else None,
                language=result.get("language"),
                confidence=result.get("confidence"),
                processing_time=processing_time,
                model_info={
                    "model_name": model_name,
                    "language": result.get("language") or language
                }
            )
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            self.logger.error(f"STT transcription failed: {e}")
            return STTResponse(
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    def validate_audio_file(
        self, 
        file_path: str, 
        validate_format: bool = True,
        check_integrity: bool = True
    ) -> AudioValidationResponse:
        """
        Validate audio file format and integrity.
        
        Args:
            file_path: Path to audio file
            validate_format: Whether to validate format
            check_integrity: Whether to check integrity
            
        Returns:
            AudioValidationResponse with validation results
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return AudioValidationResponse(
                    valid=False,
                    file_path=file_path,
                    error=f"File not found: {file_path}"
                )
            
            # Get file info
            file_size = path.stat().st_size
            file_ext = path.suffix.lower()
            
            # Basic validation
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
            format_valid = file_ext in audio_extensions
            
            validation_result = AudioValidationResponse(
                valid=format_valid and file_size > 0,
                file_path=file_path,
                format=file_ext[1:] if file_ext else "unknown",
                size_bytes=file_size
            )
            
            # Try to get audio properties for WAV files
            if format_valid and file_size > 0 and file_ext == '.wav' and file_size > 44:
                # Simple heuristic for WAV files (44-byte header)
                estimated_duration = (file_size - 44) / (16000 * 2)  # Assume 16kHz, 16-bit
                validation_result.duration = estimated_duration
                validation_result.sample_rate = 16000  # Estimated
            
            self.logger.info(f"Audio validation completed - Valid: {validation_result.valid}")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Audio validation failed: {e}")
            return AudioValidationResponse(
                valid=False,
                file_path=file_path,
                error=str(e)
            )
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get available audio models (TTS and STT)."""
        try:
            # Try to import audio model registries
            try:
                from framework.models.audio import TTS_MODEL_REGISTRY, STT_MODEL_REGISTRY
                tts_models = TTS_MODEL_REGISTRY
                stt_models = STT_MODEL_REGISTRY
            except ImportError:
                # Fallback models
                tts_models = {
                    "speecht5_tts": {
                        "type": "huggingface",
                        "model_name": "microsoft/speecht5_tts",
                        "description": "Microsoft SpeechT5 TTS model"
                    },
                    "default": {
                        "type": "huggingface",
                        "model_name": "microsoft/speecht5_tts",
                        "description": "Default TTS model"
                    }
                }
                stt_models = {
                    "whisper-base": {
                        "type": "whisper",
                        "model_size": "base",
                        "description": "OpenAI Whisper Base model"
                    }
                }
            
            # Get currently loaded audio models
            loaded_models = []
            if self.model_manager:
                loaded_models = [
                    name for name in self.model_manager.list_models()
                    if any(audio_type in name.lower() for audio_type in [
                        'tts', 'stt', 'whisper', 'tacotron', 'wav2vec', 'speecht5', 'bark'
                    ])
                ]
            
            return {
                "tts_models": tts_models,
                "stt_models": stt_models,
                "loaded_models": loaded_models,
                "supported_tts_types": ["huggingface", "torchaudio", "custom"],
                "supported_stt_types": ["whisper", "wav2vec2", "custom"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return {
                "tts_models": {},
                "stt_models": {},
                "loaded_models": [],
                "error": str(e)
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get audio processing health status."""
        health_status = {
            "audio_available": False,
            "tts_available": False,
            "stt_available": False,
            "dependencies": {},
            "errors": []
        }
        
        # Check audio dependencies
        dependencies_to_check = [
            ("librosa", "Audio processing"),
            ("soundfile", "Audio I/O"),
            ("torchaudio", "PyTorch audio"),
            ("transformers", "HuggingFace models")
        ]
        
        for dep_name, dep_desc in dependencies_to_check:
            try:
                __import__(dep_name)
                health_status["dependencies"][dep_name] = {
                    "available": True, 
                    "description": dep_desc
                }
            except ImportError as e:
                health_status["dependencies"][dep_name] = {
                    "available": False,
                    "description": dep_desc,
                    "error": str(e)
                }
                health_status["errors"].append(f"{dep_name}: {str(e)}")
        
        # Check if audio modules can be imported
        try:
            from framework.models.audio import create_tts_model, create_stt_model
            health_status["tts_available"] = True
            health_status["stt_available"] = True
            health_status["audio_available"] = True
        except ImportError as e:
            health_status["errors"].append(f"Audio modules: {str(e)}")
        
        return health_status
    
    async def _get_or_create_tts_model(self, model_name: str):
        """Get or create TTS model."""
        from framework.models.audio import create_tts_model
        from ..core.config import get_config_manager
        
        # Model name mapping
        tts_model_mapping = {
            "speecht5_tts": {"type": "huggingface", "model_name": "microsoft/speecht5_tts"},
            "speecht5": {"type": "huggingface", "model_name": "microsoft/speecht5_tts"},
            "bark": {"type": "bark", "model_name": "suno/bark"},
            "tacotron2": {"type": "torchaudio", "model_name": "tacotron2"},
            "default": {"type": "huggingface", "model_name": "microsoft/speecht5_tts"}
        }
        
        if self.model_manager and model_name in self.model_manager.list_models():
            return self.model_manager.get_model(model_name)
        
        # Create new model
        config_manager = get_config_manager()
        config = config_manager.get_inference_config()
        
        model_config = tts_model_mapping.get(model_name, tts_model_mapping["default"])
        
        tts_model = create_tts_model(
            model_config["type"],
            config,
            model_name=model_config["model_name"]
        )
        
        # Load the model
        tts_model.load_model("dummy_path")
        
        # Register if model manager available
        if self.model_manager:
            self.model_manager.register_model(model_name, tts_model)
        
        return tts_model
    
    async def _get_or_create_stt_model(self, model_name: str):
        """Get or create STT model."""
        from framework.models.audio import create_stt_model
        from ..core.config import get_config_manager
        
        if self.model_manager and model_name in self.model_manager.list_models():
            return self.model_manager.get_model(model_name)
        
        # Create new model
        config_manager = get_config_manager()
        config = config_manager.get_inference_config()
        
        stt_model = create_stt_model(model_name, config)
        
        # Register if model manager available
        if self.model_manager:
            self.model_manager.register_model(model_name, stt_model)
        
        return stt_model
    
    def _convert_audio_to_bytes(self, audio_data, sample_rate: int, output_format: str) -> bytes:
        """Convert audio data to bytes in specified format."""
        import numpy as np
        import struct
        
        # Ensure audio data is in the right format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Clip to valid range
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        if output_format.lower() == "wav":
            # Convert to 16-bit PCM
            audio_pcm = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_pcm.tobytes()
            
            # Create WAV header
            chunk_size = 36 + len(audio_bytes)
            byte_rate = sample_rate * 1 * 2  # 1 channel, 2 bytes per sample
            block_align = 1 * 2
            
            wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
                b'RIFF', chunk_size, b'WAVE', b'fmt ', 16, 1, 1,
                sample_rate, byte_rate, block_align, 16, b'data', len(audio_bytes)
            )
            
            return wav_header + audio_bytes
        else:
            # For other formats, return raw PCM for now
            # TODO: Implement MP3, FLAC conversion
            audio_pcm = (audio_data * 32767).astype(np.int16)
            return audio_pcm.tobytes()

"""
Audio service wrapper for backward compatibility.

This module provides a wrapper around the model manager specifically
for audio-related operations (TTS and STT) while maintaining API
compatibility with existing code.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import asyncio
import io
import base64

from ..core.exceptions import ModelNotFoundError, ValidationError, InferenceError

logger = logging.getLogger(__name__)


class AudioServiceWrapper:
    """
    Wrapper around the model manager to provide audio-specific functionality
    for TTS and STT operations while maintaining backward compatibility.
    """
    
    def __init__(self, model_manager):
        """
        Initialize the audio service wrapper.
        
        Args:
            model_manager: Enhanced model manager instance
        """
        self.model_manager = model_manager
        self._initialized = False
        
        logger.debug("AudioServiceWrapper initialized")
    
    def initialize(self):
        """Initialize the service wrapper."""
        if not self._initialized:
            self._initialized = True
            logger.debug("AudioServiceWrapper initialization completed")
    
    # TTS (Text-to-Speech) methods
    
    def get_tts_models(self) -> List[str]:
        """Get available TTS models."""
        try:
            self.initialize()
            return self.model_manager.get_tts_models()
        except Exception as e:
            logger.error(f"Failed to get TTS models: {e}")
            return []
    
    def has_tts_model(self, model_name: str) -> bool:
        """Check if TTS model exists."""
        try:
            return self.model_manager.is_tts_model(model_name)
        except Exception as e:
            logger.error(f"Failed to check TTS model existence: {e}")
            return False
    
    def is_tts_model_loaded(self, model_name: str) -> bool:
        """Check if TTS model is currently loaded."""
        try:
            return (self.model_manager.is_tts_model(model_name) and 
                    self.model_manager.is_model_loaded(model_name))
        except Exception as e:
            logger.error(f"Failed to check TTS model load status: {e}")
            return False
    
    async def synthesize_speech(self, text: str, model_name: Optional[str] = None, 
                              voice: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Synthesize speech from text using TTS model.
        
        Args:
            text: Text to synthesize
            model_name: TTS model to use
            voice: Voice to use (if supported)
            **kwargs: Additional synthesis parameters
        
        Returns:
            Dict containing synthesized audio data and metadata
        """
        try:
            self.initialize()
            
            # Validate inputs
            if not text or not text.strip():
                raise ValidationError(
                    field="text",
                    details="Text cannot be empty"
                )
            
            # Use default TTS model if none specified
            if not model_name:
                tts_models = self.get_tts_models()
                if not tts_models:
                    raise ModelNotFoundError(
                        model_name="any_tts_model",
                        details="No TTS models available"
                    )
                model_name = tts_models[0]
                logger.info(f"Using default TTS model: {model_name}")
            
            # Validate model
            if not self.has_tts_model(model_name):
                raise ModelNotFoundError(
                    model_name=model_name,
                    details=f"Model '{model_name}' is not a TTS model"
                )
            
            # Prepare synthesis parameters
            synthesis_params = {
                "text": text,
                "model_name": model_name,
                **kwargs
            }
            
            if voice:
                synthesis_params["voice"] = voice
            
            # Perform synthesis (assuming model manager has TTS capability)
            if hasattr(self.model_manager, 'synthesize_speech'):
                result = await self.model_manager.synthesize_speech(**synthesis_params)
            else:
                # Fallback to generic inference
                result = await self._fallback_tts_synthesis(**synthesis_params)
            
            # Ensure result is in expected format
            if not isinstance(result, dict):
                result = {"audio_data": result}
            
            # Add metadata
            result.update({
                "text": text,
                "model_name": model_name,
                "voice": voice,
                "timestamp": datetime.utcnow().isoformat(),
                "service": "audio_service_wrapper",
                "type": "tts"
            })
            
            return result
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise InferenceError(
                details=f"TTS synthesis failed: {e}",
                context={"text": text[:100], "model_name": model_name, "voice": voice},
                cause=e
            )
    
    # STT (Speech-to-Text) methods
    
    def get_stt_models(self) -> List[str]:
        """Get available STT models."""
        try:
            self.initialize()
            return self.model_manager.get_stt_models()
        except Exception as e:
            logger.error(f"Failed to get STT models: {e}")
            return []
    
    def has_stt_model(self, model_name: str) -> bool:
        """Check if STT model exists."""
        try:
            return self.model_manager.is_stt_model(model_name)
        except Exception as e:
            logger.error(f"Failed to check STT model existence: {e}")
            return False
    
    def is_stt_model_loaded(self, model_name: str) -> bool:
        """Check if STT model is currently loaded."""
        try:
            return (self.model_manager.is_stt_model(model_name) and 
                    self.model_manager.is_model_loaded(model_name))
        except Exception as e:
            logger.error(f"Failed to check STT model load status: {e}")
            return False
    
    async def transcribe_audio(self, audio_data: Union[bytes, str, io.BytesIO], 
                             model_name: Optional[str] = None, 
                             language: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio to text using STT model.
        
        Args:
            audio_data: Audio data (bytes, base64 string, or BytesIO)
            model_name: STT model to use
            language: Language hint (if supported)
            **kwargs: Additional transcription parameters
        
        Returns:
            Dict containing transcribed text and metadata
        """
        try:
            self.initialize()
            
            # Validate inputs
            if not audio_data:
                raise ValidationError(
                    field="audio_data",
                    details="Audio data cannot be empty"
                )
            
            # Use default STT model if none specified
            if not model_name:
                stt_models = self.get_stt_models()
                if not stt_models:
                    raise ModelNotFoundError(
                        model_name="any_stt_model",
                        details="No STT models available"
                    )
                model_name = stt_models[0]
                logger.info(f"Using default STT model: {model_name}")
            
            # Validate model
            if not self.has_stt_model(model_name):
                raise ModelNotFoundError(
                    model_name=model_name,
                    details=f"Model '{model_name}' is not an STT model"
                )
            
            # Process audio data
            if isinstance(audio_data, str):
                # Assume base64 encoded
                try:
                    audio_bytes = base64.b64decode(audio_data)
                except Exception as e:
                    raise ValidationError(
                        field="audio_data",
                        details=f"Invalid base64 audio data: {e}"
                    )
            elif isinstance(audio_data, io.BytesIO):
                audio_bytes = audio_data.getvalue()
            else:
                audio_bytes = audio_data
            
            # Prepare transcription parameters
            transcription_params = {
                "audio_data": audio_bytes,
                "model_name": model_name,
                **kwargs
            }
            
            if language:
                transcription_params["language"] = language
            
            # Perform transcription (assuming model manager has STT capability)
            if hasattr(self.model_manager, 'transcribe_audio'):
                result = await self.model_manager.transcribe_audio(**transcription_params)
            else:
                # Fallback to generic inference
                result = await self._fallback_stt_transcription(**transcription_params)
            
            # Ensure result is in expected format
            if not isinstance(result, dict):
                result = {"text": result}
            
            # Add metadata
            result.update({
                "model_name": model_name,
                "language": language,
                "audio_size": len(audio_bytes),
                "timestamp": datetime.utcnow().isoformat(),
                "service": "audio_service_wrapper",
                "type": "stt"
            })
            
            return result
            
        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            raise InferenceError(
                details=f"STT transcription failed: {e}",
                context={"model_name": model_name, "language": language},
                cause=e
            )
    
    # Audio model management
    
    def get_audio_models(self) -> Dict[str, List[str]]:
        """Get all audio models grouped by type."""
        try:
            return {
                "tts": self.get_tts_models(),
                "stt": self.get_stt_models()
            }
        except Exception as e:
            logger.error(f"Failed to get audio models: {e}")
            return {"tts": [], "stt": []}
    
    def get_audio_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about an audio model."""
        try:
            if not (self.has_tts_model(model_name) or self.has_stt_model(model_name)):
                raise ModelNotFoundError(
                    model_name=model_name,
                    details=f"Model '{model_name}' is not an audio model"
                )
            
            info = self.model_manager.get_model_info(model_name)
            
            # Enhance with audio-specific information
            info.update({
                "is_tts": self.has_tts_model(model_name),
                "is_stt": self.has_stt_model(model_name),
                "audio_model": True,
                "service": "audio_service_wrapper"
            })
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get audio model info for {model_name}: {e}")
            return {
                "name": model_name,
                "available": False,
                "error": str(e),
                "service": "audio_service_wrapper"
            }
    
    def validate_audio_model(self, model_name: str, audio_type: str) -> Dict[str, Any]:
        """Validate an audio model for specific use case."""
        try:
            validation_result = {
                "valid": False,
                "model_name": model_name,
                "audio_type": audio_type,
                "issues": [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check if model exists and is audio model
            if audio_type == "tts":
                if not self.has_tts_model(model_name):
                    validation_result["issues"].append(f"Model '{model_name}' is not a TTS model")
                else:
                    validation_result["valid"] = True
            elif audio_type == "stt":
                if not self.has_stt_model(model_name):
                    validation_result["issues"].append(f"Model '{model_name}' is not an STT model")
                else:
                    validation_result["valid"] = True
            else:
                validation_result["issues"].append(f"Invalid audio type: {audio_type}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Audio model validation failed for {model_name}: {e}")
            return {
                "valid": False,
                "model_name": model_name,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get audio service statistics."""
        try:
            stats = {
                "tts_models": len(self.get_tts_models()),
                "stt_models": len(self.get_stt_models()),
                "total_audio_models": len(self.get_tts_models()) + len(self.get_stt_models()),
                "service": "audio_service_wrapper",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get audio service stats: {e}")
            return {"error": str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the audio service."""
        try:
            health = {
                "healthy": True,
                "service": "audio_service_wrapper",
                "initialized": self._initialized,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check if we have audio models available
            tts_models = self.get_tts_models()
            stt_models = self.get_stt_models()
            
            health.update({
                "tts_available": len(tts_models) > 0,
                "stt_available": len(stt_models) > 0,
                "tts_models_count": len(tts_models),
                "stt_models_count": len(stt_models)
            })
            
            if len(tts_models) == 0 and len(stt_models) == 0:
                health["healthy"] = False
                health["issue"] = "No audio models available"
            
            return health
            
        except Exception as e:
            logger.error(f"Audio service health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "service": "audio_service_wrapper",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # Fallback methods for when model manager doesn't have direct audio support
    
    async def _fallback_tts_synthesis(self, **params) -> Dict[str, Any]:
        """Fallback TTS synthesis using generic inference."""
        logger.warning("Using fallback TTS synthesis - limited functionality")
        
        # This would need to be implemented based on your specific TTS models
        # For now, return a placeholder
        return {
            "audio_data": b"",  # Empty audio data
            "fallback": True,
            "message": "TTS synthesis not fully supported in fallback mode"
        }
    
    async def _fallback_stt_transcription(self, **params) -> Dict[str, Any]:
        """Fallback STT transcription using generic inference."""
        logger.warning("Using fallback STT transcription - limited functionality")
        
        # This would need to be implemented based on your specific STT models
        # For now, return a placeholder
        return {
            "text": "",  # Empty transcription
            "fallback": True,
            "message": "STT transcription not fully supported in fallback mode"
        }
    
    # Legacy compatibility methods
    
    def synthesize_speech_sync(self, text: str, model_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Synchronous TTS synthesis for backward compatibility."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.synthesize_speech(text, model_name, **kwargs))
        except RuntimeError:
            return asyncio.run(self.synthesize_speech(text, model_name, **kwargs))
    
    def transcribe_audio_sync(self, audio_data: Union[bytes, str, io.BytesIO], 
                            model_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Synchronous STT transcription for backward compatibility."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.transcribe_audio(audio_data, model_name, **kwargs))
        except RuntimeError:
            return asyncio.run(self.transcribe_audio(audio_data, model_name, **kwargs))
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status (legacy method)."""
        return {
            "status": "running" if self._initialized else "not_initialized",
            "stats": self.get_service_stats(),
            "health": self.get_health_status()
        }


# Convenience function for creating the service
def create_audio_service(model_manager) -> AudioServiceWrapper:
    """Create an audio service wrapper instance."""
    return AudioServiceWrapper(model_manager)

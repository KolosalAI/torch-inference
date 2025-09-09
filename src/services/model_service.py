"""
Model service wrapper for backward compatibility.

This module provides a wrapper around the model manager to maintain
API compatibility with existing code while leveraging enhanced
model management capabilities.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import asyncio

from ..core.exceptions import ModelNotFoundError, ValidationError, ModelLoadError

logger = logging.getLogger(__name__)


class ModelServiceWrapper:
    """
    Wrapper around the model manager to provide backward compatibility
    with the existing service interface while leveraging enhanced capabilities.
    """
    
    def __init__(self, model_manager):
        """
        Initialize the model service wrapper.
        
        Args:
            model_manager: Enhanced model manager instance
        """
        self.model_manager = model_manager
        self._initialized = False
        
        logger.debug("ModelServiceWrapper initialized")
    
    def initialize(self):
        """Initialize the service wrapper."""
        if not self._initialized:
            self._initialized = True
            logger.debug("ModelServiceWrapper initialization completed")
    
    def list_models(self) -> List[str]:
        """List all available models."""
        try:
            self.initialize()
            return self.model_manager.list_models()
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def list_available_models(self) -> List[str]:
        """List available models (alias for backward compatibility)."""
        return self.list_models()
    
    def get_loaded_models(self) -> List[str]:
        """Get currently loaded models."""
        try:
            return self.model_manager.get_loaded_models()
        except Exception as e:
            logger.error(f"Failed to get loaded models: {e}")
            return []
    
    def has_model(self, model_name: str) -> bool:
        """Check if model exists."""
        try:
            return self.model_manager.has_model(model_name)
        except Exception as e:
            logger.error(f"Failed to check model existence: {e}")
            return False
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if model is currently loaded."""
        try:
            return self.model_manager.is_model_loaded(model_name)
        except Exception as e:
            logger.error(f"Failed to check model load status: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        try:
            if not self.has_model(model_name):
                raise ModelNotFoundError(
                    model_name=model_name,
                    details=f"Model '{model_name}' not found in available models"
                )
            
            info = self.model_manager.get_model_info(model_name)
            
            # Ensure consistent format
            info.update({
                "name": model_name,
                "available": True,
                "loaded": self.is_model_loaded(model_name),
                "timestamp": datetime.utcnow().isoformat(),
                "service": "model_service_wrapper"
            })
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return {
                "name": model_name,
                "available": False,
                "loaded": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "service": "model_service_wrapper"
            }
    
    def get_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available models."""
        try:
            models_info = {}
            for model_name in self.list_models():
                models_info[model_name] = self.get_model_info(model_name)
            return models_info
        except Exception as e:
            logger.error(f"Failed to get all models info: {e}")
            return {}
    
    async def load_model(self, model_name: str, **kwargs) -> bool:
        """Load a model."""
        try:
            if not self.has_model(model_name):
                raise ModelNotFoundError(
                    model_name=model_name,
                    details=f"Cannot load model '{model_name}' - not found in available models"
                )
            
            if self.is_model_loaded(model_name):
                logger.info(f"Model {model_name} already loaded")
                return True
            
            # Check if model manager supports async loading
            if hasattr(self.model_manager, 'load_model') and asyncio.iscoroutinefunction(self.model_manager.load_model):
                await self.model_manager.load_model(model_name, **kwargs)
            elif hasattr(self.model_manager, 'load_model'):
                self.model_manager.load_model(model_name, **kwargs)
            else:
                logger.warning(f"Model loading not supported by current model manager")
                return False
            
            return self.is_model_loaded(model_name)
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelLoadError(
                model_name=model_name,
                details=f"Failed to load model: {e}",
                cause=e
            )
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model."""
        try:
            if not self.is_model_loaded(model_name):
                logger.info(f"Model {model_name} not loaded")
                return True
            
            # Check if model manager supports async unloading
            if hasattr(self.model_manager, 'unload_model') and asyncio.iscoroutinefunction(self.model_manager.unload_model):
                await self.model_manager.unload_model(model_name)
            elif hasattr(self.model_manager, 'unload_model'):
                self.model_manager.unload_model(model_name)
            else:
                logger.warning(f"Model unloading not supported by current model manager")
                return False
            
            return not self.is_model_loaded(model_name)
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False
    
    def get_model_types(self) -> Dict[str, List[str]]:
        """Get models grouped by type."""
        try:
            model_types = {
                "text-to-speech": [],
                "speech-to-text": [], 
                "general": [],
                "unknown": []
            }
            
            for model_name in self.list_models():
                info = self.get_model_info(model_name)
                model_type = info.get("type", "unknown")
                
                if model_type in model_types:
                    model_types[model_type].append(model_name)
                else:
                    model_types["unknown"].append(model_name)
            
            return model_types
            
        except Exception as e:
            logger.error(f"Failed to get model types: {e}")
            return {
                "text-to-speech": [],
                "speech-to-text": [],
                "general": [],
                "unknown": []
            }
    
    def get_tts_models(self) -> List[str]:
        """Get TTS (Text-to-Speech) models."""
        try:
            if hasattr(self.model_manager, 'get_tts_models'):
                return self.model_manager.get_tts_models()
            else:
                # Fallback implementation
                return self.get_model_types()["text-to-speech"]
        except Exception as e:
            logger.error(f"Failed to get TTS models: {e}")
            return []
    
    def get_stt_models(self) -> List[str]:
        """Get STT (Speech-to-Text) models."""
        try:
            if hasattr(self.model_manager, 'get_stt_models'):
                return self.model_manager.get_stt_models()
            else:
                # Fallback implementation
                return self.get_model_types()["speech-to-text"]
        except Exception as e:
            logger.error(f"Failed to get STT models: {e}")
            return []
    
    def is_tts_model(self, model_name: str) -> bool:
        """Check if model is a TTS model."""
        try:
            if hasattr(self.model_manager, 'is_tts_model'):
                return self.model_manager.is_tts_model(model_name)
            else:
                return model_name in self.get_tts_models()
        except Exception as e:
            logger.error(f"Failed to check if {model_name} is TTS model: {e}")
            return False
    
    def is_stt_model(self, model_name: str) -> bool:
        """Check if model is an STT model."""
        try:
            if hasattr(self.model_manager, 'is_stt_model'):
                return self.model_manager.is_stt_model(model_name)
            else:
                return model_name in self.get_stt_models()
        except Exception as e:
            logger.error(f"Failed to check if {model_name} is STT model: {e}")
            return False
    
    def validate_model(self, model_name: str, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Validate a model for specific use case."""
        try:
            validation_result = {
                "valid": False,
                "model_name": model_name,
                "requested_type": model_type,
                "actual_type": None,
                "issues": [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check if model exists
            if not self.has_model(model_name):
                validation_result["issues"].append(f"Model '{model_name}' not found")
                return validation_result
            
            # Get model info
            info = self.get_model_info(model_name)
            actual_type = info.get("type", "unknown")
            validation_result["actual_type"] = actual_type
            
            # Check type compatibility if specified
            if model_type:
                if model_type == "tts" and not self.is_tts_model(model_name):
                    validation_result["issues"].append(f"Model is not a TTS model (actual: {actual_type})")
                elif model_type == "stt" and not self.is_stt_model(model_name):
                    validation_result["issues"].append(f"Model is not an STT model (actual: {actual_type})")
                elif model_type not in ["tts", "stt"] and actual_type != model_type:
                    validation_result["issues"].append(f"Model type mismatch (expected: {model_type}, actual: {actual_type})")
            
            # Check if model can be loaded
            if not self.is_model_loaded(model_name):
                validation_result["issues"].append("Model not currently loaded")
            
            validation_result["valid"] = len(validation_result["issues"]) == 0
            return validation_result
            
        except Exception as e:
            logger.error(f"Model validation failed for {model_name}: {e}")
            return {
                "valid": False,
                "model_name": model_name,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            stats = {
                "total_models": len(self.list_models()),
                "loaded_models": len(self.get_loaded_models()),
                "tts_models": len(self.get_tts_models()),
                "stt_models": len(self.get_stt_models()),
                "service": "model_service_wrapper",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add model manager stats if available
            if hasattr(self.model_manager, 'get_health_status'):
                manager_health = self.model_manager.get_health_status()
                stats.update(manager_health)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get service stats: {e}")
            return {"error": str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the service."""
        try:
            health = {
                "healthy": True,
                "service": "model_service_wrapper",
                "initialized": self._initialized,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check model manager health
            if hasattr(self.model_manager, 'get_health_status'):
                manager_health = self.model_manager.get_health_status()
                health.update({
                    "model_manager_healthy": manager_health.get("healthy", True),
                    "model_manager_status": manager_health
                })
                if not manager_health.get("healthy", True):
                    health["healthy"] = False
            
            # Add basic stats
            try:
                health.update({
                    "total_models": len(self.list_models()),
                    "loaded_models": len(self.get_loaded_models())
                })
            except Exception as e:
                health["healthy"] = False
                health["stats_error"] = str(e)
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "service": "model_service_wrapper",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def cleanup_all(self):
        """Cleanup all models."""
        try:
            if hasattr(self.model_manager, 'cleanup_all'):
                self.model_manager.cleanup_all()
            logger.info("ModelServiceWrapper cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    # Legacy compatibility methods
    
    def load_model_sync(self, model_name: str, **kwargs) -> bool:
        """Synchronous model loading for backward compatibility."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.load_model(model_name, **kwargs))
        except RuntimeError:
            # No event loop running
            return asyncio.run(self.load_model(model_name, **kwargs))
    
    def unload_model_sync(self, model_name: str) -> bool:
        """Synchronous model unloading for backward compatibility."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.unload_model(model_name))
        except RuntimeError:
            # No event loop running
            return asyncio.run(self.unload_model(model_name))
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status (legacy method)."""
        return {
            "status": "running" if self._initialized else "not_initialized",
            "stats": self.get_service_stats(),
            "health": self.get_health_status()
        }


# Convenience function for creating the service
def create_model_service(model_manager) -> ModelServiceWrapper:
    """Create a model service wrapper instance."""
    return ModelServiceWrapper(model_manager)

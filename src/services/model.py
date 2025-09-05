"""
Model management service.
"""

import logging
from typing import Any, Dict, List, Optional

from ..core.exceptions import ModelError, ModelNotFoundError
# Import models directly from specific files
from ..models.api.models import ModelInfo, ModelInfoResponse, ModelListResponse, ModelLoadRequest, ModelUnloadRequest, ModelCacheInfo

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing models."""
    
    def __init__(self, model_manager=None):
        self.model_manager = model_manager
        self.logger = logger
    
    def list_models(self) -> ModelListResponse:
        """List all available models."""
        try:
            if not self.model_manager:
                # Return basic model info if no manager available
                return ModelListResponse(
                    models=["example"],
                    model_info={
                        "example": ModelInfo(
                            name="example",
                            source="builtin",
                            model_id="example",
                            task="demonstration",
                            description="Built-in example model",
                            loaded=True
                        )
                    },
                    total_models=1
                )
            
            models = self.model_manager.list_models()
            model_info = {}
            
            for model_name in models:
                try:
                    model = self.model_manager.get_model(model_name)
                    info = model.model_info if hasattr(model, 'model_info') else {}
                    
                    model_info[model_name] = ModelInfo(
                        name=model_name,
                        source=info.get("source", "unknown"),
                        model_id=info.get("model_id", model_name),
                        task=info.get("task", "inference"),
                        description=info.get("description", ""),
                        size_mb=info.get("size_mb"),
                        tags=info.get("tags"),
                        loaded=True
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to get info for model {model_name}: {e}")
                    model_info[model_name] = ModelInfo(
                        name=model_name,
                        source="unknown",
                        model_id=model_name,
                        task="unknown",
                        loaded=True
                    )
            
            self.logger.info(f"Listed {len(models)} models successfully")
            
            return ModelListResponse(
                models=models,
                model_info=model_info,
                total_models=len(models)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            raise ModelError(f"Failed to list models: {e}")
    
    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get information about a specific model."""
        try:
            if not self.model_manager:
                if model_name == "example":
                    return ModelInfo(
                        name="example",
                        source="builtin",
                        model_id="example",
                        task="demonstration",
                        description="Built-in example model",
                        loaded=True
                    )
                else:
                    raise ModelNotFoundError(f"Model not found: {model_name}")
            
            if model_name not in self.model_manager.list_models():
                raise ModelNotFoundError(f"Model not found: {model_name}")
            
            model = self.model_manager.get_model(model_name)
            info = model.model_info if hasattr(model, 'model_info') else {}
            
            return ModelInfo(
                name=model_name,
                source=info.get("source", "unknown"),
                model_id=info.get("model_id", model_name),
                task=info.get("task", "inference"),
                description=info.get("description", ""),
                size_mb=info.get("size_mb"),
                tags=info.get("tags"),
                loaded=True
            )
            
        except ModelNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get model info for {model_name}: {e}")
            raise ModelError(f"Failed to get model info: {e}")
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        try:
            if not self.model_manager:
                return model_name == "example"
            return model_name in self.model_manager.list_models()
        except Exception:
            return False
    
    def get_cache_info(self) -> ModelCacheInfo:
        """Get model cache information."""
        try:
            if not self.model_manager or not hasattr(self.model_manager, 'get_downloader'):
                return ModelCacheInfo(
                    cache_directory="N/A",
                    total_models=0,
                    total_size_mb=0.0,
                    models=[]
                )
            
            downloader = self.model_manager.get_downloader()
            cache_size_mb = downloader.get_cache_size()
            cached_models = list(downloader.registry.keys()) if hasattr(downloader, 'registry') else []
            cache_dir = str(downloader.cache_dir) if hasattr(downloader, 'cache_dir') else "N/A"
            
            return ModelCacheInfo(
                cache_directory=cache_dir,
                total_models=len(cached_models),
                total_size_mb=cache_size_mb,
                models=cached_models
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get cache info: {e}")
            return ModelCacheInfo(
                cache_directory="Error",
                total_models=0,
                total_size_mb=0.0,
                models=[]
            )
    
    def list_available_downloads(self) -> Dict[str, Any]:
        """List models available for download."""
        try:
            if not self.model_manager or not hasattr(self.model_manager, 'list_available_downloads'):
                # Return popular TTS models as fallback
                return {
                    "speecht5_tts": {
                        "name": "speecht5_tts",
                        "source": "huggingface",
                        "model_id": "microsoft/speecht5_tts",
                        "task": "text-to-speech",
                        "description": "Microsoft SpeechT5 TTS model",
                        "size_mb": 2500
                    },
                    "bark_tts": {
                        "name": "bark_tts",
                        "source": "huggingface",
                        "model_id": "suno/bark",
                        "task": "text-to-speech",
                        "description": "Suno Bark TTS model",
                        "size_mb": 4000
                    }
                }
            
            return self.model_manager.list_available_downloads()
            
        except Exception as e:
            self.logger.error(f"Failed to list available downloads: {e}")
            return {}
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model from cache."""
        try:
            if not self.model_manager or not hasattr(self.model_manager, 'get_downloader'):
                return False
            
            downloader = self.model_manager.get_downloader()
            
            if not hasattr(downloader, 'is_model_cached') or not downloader.is_model_cached(model_name):
                raise ModelNotFoundError(f"Model not found in cache: {model_name}")
            
            success = downloader.remove_model(model_name)
            if success:
                self.logger.info(f"Successfully removed model from cache: {model_name}")
            return success
            
        except ModelNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to remove model {model_name}: {e}")
            raise ModelError(f"Failed to remove model: {e}")
    
    def optimize_model(self, model_name: str) -> bool:
        """Optimize a model for inference."""
        try:
            if not self.is_model_loaded(model_name):
                raise ModelNotFoundError(f"Model not loaded: {model_name}")
            
            model = self.model_manager.get_model(model_name)
            
            if hasattr(model, 'optimize_for_inference'):
                model.optimize_for_inference()
                self.logger.info(f"Successfully optimized model: {model_name}")
                return True
            else:
                self.logger.warning(f"Model {model_name} does not support optimization")
                return False
                
        except ModelNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to optimize model {model_name}: {e}")
            raise ModelError(f"Failed to optimize model: {e}")
    
    def cleanup_all(self) -> None:
        """Cleanup all models."""
        try:
            if self.model_manager and hasattr(self.model_manager, 'cleanup_all'):
                self.model_manager.cleanup_all()
                self.logger.info("Successfully cleaned up all models")
        except Exception as e:
            self.logger.error(f"Failed to cleanup models: {e}")

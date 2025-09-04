"""
FastAPI dependency injection for PyTorch Inference Framework.
"""

import logging
from typing import Optional
from fastapi import Depends, HTTPException

from ..core.config import get_config_manager, get_config
from ..core.logging import get_api_logger
from ..services import InferenceService, ModelService, AudioService, DownloadService

logger = logging.getLogger(__name__)

# Global service instances - these will be initialized at startup
_inference_service: Optional[InferenceService] = None
_model_service: Optional[ModelService] = None
_audio_service: Optional[AudioService] = None
_download_service: Optional[DownloadService] = None

# Global framework components - will be set by main application
_model_manager = None
_inference_engine = None
_autoscaler = None


def initialize_services(model_manager=None, inference_engine=None, autoscaler=None):
    """Initialize all services with framework components."""
    global _inference_service, _model_service, _audio_service, _download_service
    global _model_manager, _inference_engine, _autoscaler
    
    # Store framework components
    _model_manager = model_manager
    _inference_engine = inference_engine
    _autoscaler = autoscaler
    
    # Initialize services
    _inference_service = InferenceService(model_manager, inference_engine, autoscaler)
    _model_service = ModelService(model_manager)
    _audio_service = AudioService(model_manager)
    _download_service = DownloadService(model_manager)
    
    logger.info("All services initialized successfully")


def get_config_manager_dependency():
    """FastAPI dependency for configuration manager."""
    return get_config_manager()


def get_config_dependency():
    """FastAPI dependency for application configuration."""
    return get_config()


def get_api_logger_dependency():
    """FastAPI dependency for API logger."""
    return get_api_logger()


def get_inference_service() -> InferenceService:
    """FastAPI dependency for inference service."""
    if _inference_service is None:
        raise HTTPException(
            status_code=503,
            detail="Inference service not available. Server may be starting up."
        )
    return _inference_service


def get_model_service() -> ModelService:
    """FastAPI dependency for model service."""
    if _model_service is None:
        raise HTTPException(
            status_code=503,
            detail="Model service not available. Server may be starting up."
        )
    return _model_service


def get_audio_service() -> AudioService:
    """FastAPI dependency for audio service."""
    if _audio_service is None:
        raise HTTPException(
            status_code=503,
            detail="Audio service not available. Server may be starting up."
        )
    return _audio_service


def get_download_service() -> DownloadService:
    """FastAPI dependency for download service."""
    if _download_service is None:
        raise HTTPException(
            status_code=503,
            detail="Download service not available. Server may be starting up."
        )
    return _download_service


# Optional dependencies that may not be available
def get_model_manager():
    """Get model manager if available."""
    return _model_manager


def get_inference_engine():
    """Get inference engine if available."""
    return _inference_engine


def get_autoscaler():
    """Get autoscaler if available."""
    return _autoscaler


def cleanup_services():
    """Cleanup all services."""
    global _inference_service, _model_service, _audio_service, _download_service
    global _model_manager, _inference_engine, _autoscaler
    
    # Cleanup services
    if _model_service:
        _model_service.cleanup_all()
    
    # Clear references
    _inference_service = None
    _model_service = None
    _audio_service = None
    _download_service = None
    _model_manager = None
    _inference_engine = None
    _autoscaler = None
    
    logger.info("All services cleaned up")

"""
API package initialization for PyTorch Inference Framework.
"""

from .app import app, create_app
from .dependencies import (
    get_inference_service,
    get_model_service,
    get_audio_service,
    get_download_service,
    initialize_services,
    cleanup_services
)
from .lifespan import lifespan, get_framework_components
from .routers import (
    core_router,
    inference_router,
    models_router,
    audio_router,
    downloads_router
)

__all__ = [
    # Main app
    "app", 
    "create_app",
    
    # Dependencies
    "get_inference_service",
    "get_model_service", 
    "get_audio_service",
    "get_download_service",
    "initialize_services",
    "cleanup_services",
    
    # Lifespan
    "lifespan",
    "get_framework_components",
    
    # Routers
    "core_router",
    "inference_router",
    "models_router", 
    "audio_router",
    "downloads_router"
]

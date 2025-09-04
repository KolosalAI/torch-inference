"""
API Router initialization and registration.
"""

from .core import router as core_router
from .inference import router as inference_router  
from .models import router as models_router
from .audio import router as audio_router
from .downloads import router as downloads_router
from .gpu import router as gpu_router
from .autoscaler import router as autoscaler_router
from .server import router as server_router
from .logs import router as logs_router

# Export all routers
__all__ = [
    "core_router",
    "inference_router", 
    "models_router",
    "audio_router",
    "downloads_router",
    "gpu_router",
    "autoscaler_router", 
    "server_router",
    "logs_router"
]

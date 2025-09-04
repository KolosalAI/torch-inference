"""
Main FastAPI application for PyTorch Inference Framework.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ..core.config import get_config
from .lifespan import lifespan
from .middleware import RequestLoggingMiddleware, SecurityMiddleware, CORSCustomMiddleware
from .routers import (
    core_router,
    inference_router,
    models_router, 
    audio_router,
    downloads_router,
    gpu_router,
    autoscaler_router,
    server_router,
    logs_router
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    # Get configuration
    config = get_config()
    
    # Create FastAPI app with lifespan management
    app = FastAPI(
        title="PyTorch Inference API",
        description="High-performance PyTorch inference API with TTS, STT, and model management capabilities",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        # Additional configuration
        contact={
            "name": "PyTorch Inference Framework",
            "email": "support@pytorch-inference.com"
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        }
    )
    
    # Add middleware
    _setup_middleware(app, config)
    
    # Include routers
    _setup_routers(app)
    
    # Add event handlers
    _setup_event_handlers(app)
    
    logger.info("[APP] FastAPI application created and configured")
    
    return app


def _setup_middleware(app: FastAPI, config):
    """Setup application middleware."""
    logger.debug("[APP] Setting up middleware...")
    
    # Security middleware (should be first)
    app.add_middleware(SecurityMiddleware)
    
    # Request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # CORS middleware configuration
    cors_origins = ["*"]  # In production, specify exact origins
    if hasattr(config, 'security') and hasattr(config.security, 'cors_origins'):
        cors_origins = config.security.cors_origins
    
    # Add built-in CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Custom CORS middleware for additional control
    app.add_middleware(CORSCustomMiddleware)
    
    logger.debug("[APP] Middleware setup completed")


def _setup_routers(app: FastAPI):
    """Setup application routers."""
    logger.debug("[APP] Setting up routers...")
    
    # Include all routers with their prefixes
    app.include_router(core_router)
    app.include_router(inference_router)
    app.include_router(models_router)
    app.include_router(audio_router)
    app.include_router(downloads_router)
    app.include_router(gpu_router)
    app.include_router(autoscaler_router)
    app.include_router(server_router)
    app.include_router(logs_router)
    
    logger.debug("[APP] Routers setup completed")


def _setup_event_handlers(app: FastAPI):
    """Setup additional event handlers."""
    logger.debug("[APP] Setting up event handlers...")
    
    @app.get("/", 
             tags=["Root"],
             summary="API Root",
             description="API root endpoint with basic information")
    async def root():
        """Root endpoint."""
        return {
            "name": "PyTorch Inference API",
            "version": "1.0.0",
            "status": "running",
            "documentation": {
                "interactive_docs": "/docs",
                "redoc": "/redoc",
                "openapi_spec": "/openapi.json"
            },
            "endpoints": {
                "health_check": "/core/health",
                "system_info": "/core/info",
                "inference": "/inference/predict",
                "models": "/models/",
                "audio": "/audio/",
                "downloads": "/downloads/"
            }
        }
    
    logger.debug("[APP] Event handlers setup completed")


# Create the application instance
app = create_app()


# Optional: Add global exception handlers
@app.exception_handler(404)
async def custom_404_handler(request, exc):
    """Custom 404 handler."""
    return {
        "error": "Not Found",
        "message": f"The requested endpoint {request.url.path} was not found",
        "status_code": 404,
        "available_endpoints": [
            "/docs",
            "/core/health", 
            "/inference/predict",
            "/models/",
            "/audio/",
            "/downloads/"
        ]
    }


@app.exception_handler(500)
async def custom_500_handler(request, exc):
    """Custom 500 handler."""
    logger.error(f"[APP] Internal server error: {exc}")
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred. Please try again later.",
        "status_code": 500
    }


# Health check endpoint at root level for load balancers
@app.get("/health",
         tags=["Health"],
         summary="Simple Health Check",
         description="Simple health check endpoint for load balancers")
async def simple_health_check():
    """Simple health check for load balancers."""
    return {"status": "healthy"}


if __name__ == "__main__":
    # This allows running the app directly with python -m src.api.app
    import uvicorn
    
    config = get_config()
    host = getattr(config.server, 'host', '0.0.0.0')
    port = getattr(config.server, 'port', 8000)
    
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )

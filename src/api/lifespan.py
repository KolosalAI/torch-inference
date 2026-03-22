"""
Application lifespan management for PyTorch Inference Framework.
"""

import logging
from datetime import datetime
from contextlib import asynccontextmanager

from ..core.config import get_config
from ..core.exceptions import ServiceUnavailableError
from .dependencies import initialize_services, cleanup_services

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    """
    Application lifespan manager for FastAPI.
    Handles startup and shutdown procedures.
    """
    startup_time = datetime.now()
    
    logger.info("[SERVER] Starting PyTorch Inference API Server...")
    logger.info(f"[SERVER] Startup initiated at: {startup_time.isoformat()}")
    
    try:
        # Initialize framework components
        await initialize_framework_components()
        
        # Initialize services
        await initialize_application_services()
        
        ready_time = datetime.now()
        startup_duration = (ready_time - startup_time).total_seconds()
        
        logger.info(f"[SERVER] Server startup complete at: {ready_time.isoformat()}")
        logger.info(f"[SERVER] Startup duration: {startup_duration:.2f} seconds")
        logger.info("[SERVER] Ready to accept requests")
        logger.info("="*60)
        
        yield
        
    except Exception as e:
        logger.error(f"[SERVER] Failed to start server: {e}")
        raise ServiceUnavailableError(f"Server startup failed: {e}")
    
    finally:
        # Cleanup
        shutdown_time = datetime.now()
        logger.info("="*60)
        logger.info("[SERVER] Shutting down PyTorch Inference API Server...")
        logger.info(f"[SERVER] Shutdown initiated at: {shutdown_time.isoformat()}")
        
        try:
            await cleanup_application()
            logger.info("[SERVER] Application cleanup completed")
        except Exception as e:
            logger.error(f"[SERVER] Error during cleanup: {e}")
        
        shutdown_complete_time = datetime.now()
        shutdown_duration = (shutdown_complete_time - shutdown_time).total_seconds()
        
        logger.info(f"[SERVER] Server shutdown complete at: {shutdown_complete_time.isoformat()}")
        logger.info(f"[SERVER] Shutdown duration: {shutdown_duration:.2f} seconds")
        logger.info("="*60)


async def initialize_framework_components():
    """Initialize core framework components."""
    global _model_manager, _inference_engine, _autoscaler
    
    logger.info("[STARTUP] Initializing framework components...")
    
    try:
        # Import framework components with error handling
        framework_available = False
        try:
            # Test basic framework import
            import framework
            logger.info("✓ Framework module imported successfully")
            framework_available = True
        except ImportError as e:
            logger.warning(f"✗ Failed to import framework module: {e}")
            framework_available = False
        
        if framework_available:
            # Initialize framework components
            await _initialize_with_framework()
        else:
            # Initialize with fallback components
            await _initialize_fallback_components()
        
        logger.info("[STARTUP] Framework components initialized successfully")
        
    except Exception as e:
        logger.error(f"[STARTUP] Failed to initialize framework components: {e}")
        raise


async def initialize_application_services():
    """Initialize application services with framework components."""
    logger.info("[STARTUP] Initializing application services...")
    
    try:
        # Get framework components (may be None for fallback mode)
        model_manager = getattr(initialize_framework_components, '_model_manager', None)
        inference_engine = getattr(initialize_framework_components, '_inference_engine', None) 
        autoscaler = getattr(initialize_framework_components, '_autoscaler', None)
        
        # Initialize services
        initialize_services(model_manager, inference_engine, autoscaler)
        
        logger.info("[STARTUP] Application services initialized successfully")
        
    except Exception as e:
        logger.error(f"[STARTUP] Failed to initialize application services: {e}")
        raise


async def cleanup_application():
    """Cleanup application resources."""
    logger.info("[CLEANUP] Starting application cleanup...")
    
    try:
        # Cleanup services
        cleanup_services()
        
        # Cleanup framework components
        await _cleanup_framework_components()
        
        logger.info("[CLEANUP] Application cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"[CLEANUP] Error during cleanup: {e}")


async def _initialize_with_framework():
    """Initialize with full framework components."""
    from framework.core.config_manager import get_config_manager
    from framework.core.base_model import get_model_manager
    from framework.core.inference_engine import create_inference_engine
    from framework.autoscaling import Autoscaler, AutoscalerConfig, ZeroScalingConfig, ModelLoaderConfig
    
    logger.info("[STARTUP] Initializing with framework components...")
    
    # Initialize configuration
    framework_config_manager = get_config_manager()
    config = framework_config_manager.get_inference_config()
    
    # Initialize model manager
    model_manager = get_model_manager()
    
    # Create and load example model
    from main import ExampleModel  # Import from existing main.py
    example_model = ExampleModel(config)
    example_model.load_model("example")
    example_model.optimize_for_inference()
    
    # Register model
    model_manager.register_model("example", example_model)
    
    # Initialize autoscaler
    autoscaler_config = AutoscalerConfig(
        enable_zero_scaling=True,
        enable_dynamic_loading=True,
        zero_scaling=ZeroScalingConfig(
            enabled=True,
            scale_to_zero_delay=300.0,
            max_loaded_models=5,
            preload_popular_models=True
        ),
        model_loading=ModelLoaderConfig(
            max_instances_per_model=3,
            min_instances_per_model=1,
            enable_model_caching=True,
            prefetch_popular_models=True
        )
    )
    
    autoscaler = Autoscaler(autoscaler_config, model_manager)
    await autoscaler.start()
    
    # Create inference engine
    try:
        from framework.core.inference_engine import create_ultra_fast_inference_engine
        inference_engine = create_ultra_fast_inference_engine(example_model, config)
        logger.info("Using enhanced InferenceEngine with ultra-fast optimizations")
    except Exception as e:
        logger.warning(f"Failed to create ultra-fast inference engine, using standard: {e}")
        inference_engine = create_inference_engine(example_model, config)
    
    await inference_engine.start()
    
    # Warmup
    example_model.warmup(config.performance.warmup_iterations)
    
    # Store components for services
    _initialize_with_framework._model_manager = model_manager
    _initialize_with_framework._inference_engine = inference_engine
    _initialize_with_framework._autoscaler = autoscaler
    
    logger.info("[STARTUP] Framework components initialized successfully")


async def _initialize_fallback_components():
    """Initialize with fallback components when framework is not available."""
    logger.info("[STARTUP] Initializing with fallback components...")
    
    # Create minimal fallback components
    from types import SimpleNamespace
    
    # Mock model manager
    model_manager = SimpleNamespace()
    model_manager.list_models = lambda: ["example"]
    model_manager.get_model = lambda name: SimpleNamespace(
        predict=lambda inputs: {"result": "fallback_prediction"},
        model_info={"model_name": name, "device": "cpu", "loaded": True}
    )
    model_manager.register_model = lambda name, model: None
    model_manager.cleanup_all = lambda: None
    
    # Mock inference engine
    inference_engine = SimpleNamespace()
    inference_engine.predict = lambda inputs, **kwargs: {"result": "fallback_prediction"}
    inference_engine.predict_batch = lambda inputs_list, **kwargs: [{"result": "fallback_prediction"}] * len(inputs_list)
    inference_engine.health_check = lambda: {"healthy": True, "checks": {"fallback": True}, "timestamp": 0}
    inference_engine.get_stats = lambda: {"requests_processed": 0}
    inference_engine.get_performance_report = lambda: {"performance": "fallback_mode"}
    inference_engine.start = lambda: None
    inference_engine.stop = lambda: None
    inference_engine.device = "cpu"
    
    # No autoscaler in fallback mode
    autoscaler = None
    
    # Store fallback components
    _initialize_fallback_components._model_manager = model_manager
    _initialize_fallback_components._inference_engine = inference_engine
    _initialize_fallback_components._autoscaler = autoscaler
    
    logger.info("[STARTUP] Fallback components initialized successfully")


async def _cleanup_framework_components():
    """Cleanup framework components."""
    logger.info("[CLEANUP] Cleaning up framework components...")
    
    try:
        # Get components
        model_manager = getattr(_initialize_with_framework, '_model_manager', None) or \
                      getattr(_initialize_fallback_components, '_model_manager', None)
        inference_engine = getattr(_initialize_with_framework, '_inference_engine', None) or \
                          getattr(_initialize_fallback_components, '_inference_engine', None)
        autoscaler = getattr(_initialize_with_framework, '_autoscaler', None)
        
        # Cleanup autoscaler
        if autoscaler and hasattr(autoscaler, 'stop'):
            await autoscaler.stop()
            logger.info("[CLEANUP] Autoscaler stopped")
        
        # Cleanup inference engine
        if inference_engine and hasattr(inference_engine, 'stop'):
            await inference_engine.stop()
            logger.info("[CLEANUP] Inference engine stopped")
        
        # Cleanup model manager
        if model_manager and hasattr(model_manager, 'cleanup_all'):
            model_manager.cleanup_all()
            logger.info("[CLEANUP] Model manager cleaned up")
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("[CLEANUP] CUDA cache cleared")
        except ImportError:
            pass
        
        logger.info("[CLEANUP] Framework components cleaned up successfully")
        
    except Exception as e:
        logger.error(f"[CLEANUP] Error cleaning up framework components: {e}")


# Make components accessible for services
def get_framework_components():
    """Get initialized framework components."""
    model_manager = getattr(_initialize_with_framework, '_model_manager', None) or \
                   getattr(_initialize_fallback_components, '_model_manager', None)
    inference_engine = getattr(_initialize_with_framework, '_inference_engine', None) or \
                      getattr(_initialize_fallback_components, '_inference_engine', None)
    autoscaler = getattr(_initialize_with_framework, '_autoscaler', None)
    
    return model_manager, inference_engine, autoscaler

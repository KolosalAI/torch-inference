"""
Enhanced dependency injection system for PyTorch Inference Framework.

This module provides a centralized dependency injection system for managing
shared resources, services, and configurations across the application with
proper error handling and service lifecycle management.
"""

import logging
from typing import Optional, Any, Dict, List
from functools import lru_cache

from fastapi import HTTPException, status

from ..core.config import get_config_manager, get_config, InferenceConfig
from ..core.exceptions import ServiceUnavailableError, ConfigurationError

logger = logging.getLogger(__name__)


# Global service registry with enhanced management
_services: Dict[str, Any] = {}
_service_health: Dict[str, Dict[str, Any]] = {}


class ServiceRegistry:
    """Enhanced service registry with health monitoring and lifecycle management."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._service_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, service: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a service with optional metadata."""
        self._services[name] = service
        self._service_metadata[name] = metadata or {}
        logger.debug(f"Service '{name}' registered with metadata: {metadata}")
    
    def get(self, name: str) -> Optional[Any]:
        """Get a service by name."""
        return self._services.get(name)
    
    def unregister(self, name: str) -> bool:
        """Unregister a service."""
        if name in self._services:
            del self._services[name]
            self._service_metadata.pop(name, None)
            logger.debug(f"Service '{name}' unregistered")
            return True
        return False
    
    def list_services(self) -> List[str]:
        """List all registered services."""
        return list(self._services.keys())
    
    def get_service_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get service information including metadata."""
        if name in self._services:
            return {
                "name": name,
                "type": type(self._services[name]).__name__,
                "metadata": self._service_metadata.get(name, {}),
                "available": True
            }
        return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all services."""
        health = {}
        for name, service in self._services.items():
            try:
                if hasattr(service, 'get_health_status'):
                    health[name] = service.get_health_status()
                elif hasattr(service, 'health_check'):
                    health[name] = {"healthy": True, "status": "available"}
                else:
                    health[name] = {"healthy": True, "status": "available"}
            except Exception as e:
                health[name] = {"healthy": False, "error": str(e)}
        return health
    
    def clear(self) -> None:
        """Clear all services."""
        self._services.clear()
        self._service_metadata.clear()
        logger.debug("All services cleared from registry")


# Global service registry instance
_registry = ServiceRegistry()


def register_service(name: str, service: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Register a service in the global registry."""
    _registry.register(name, service, metadata)


def get_service(name: str) -> Optional[Any]:
    """Get a service from the global registry."""
    return _registry.get(name)


def unregister_service(name: str) -> bool:
    """Unregister a service from the global registry."""
    return _registry.unregister(name)


def list_services() -> List[str]:
    """List all registered services."""
    return _registry.list_services()


def get_service_info(name: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a service."""
    return _registry.get_service_info(name)


def clear_services() -> None:
    """Clear all registered services."""
    _registry.clear()


# Enhanced dependency functions for FastAPI

@lru_cache()
def get_app_config():
    """Get application configuration (cached)."""
    try:
        return get_config()
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Configuration service unavailable"
        )


def get_inference_config() -> InferenceConfig:
    """Get inference configuration."""
    try:
        config = get_app_config()
        return config.inference
    except Exception as e:
        logger.error(f"Failed to get inference configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference configuration unavailable"
        )


def get_memory_manager():
    """Get memory manager instance."""
    try:
        # Try to get from service registry first
        memory_manager = get_service("memory_manager")
        if memory_manager:
            return memory_manager
        
        # Create new instance if not in registry
        logger.warning("Memory manager not found in registry, creating new instance")
        
        from ..core.memory_manager import get_memory_manager as create_memory_manager
        
        config = get_app_config()
        max_memory_mb = getattr(config.inference, 'max_memory_usage', 4096) or 4096
        
        memory_manager = create_memory_manager(
            max_memory_mb=max_memory_mb,
            max_cached_models=5
        )
        
        # Register for future use
        register_service("memory_manager", memory_manager, {
            "type": "memory_manager",
            "max_memory_mb": max_memory_mb,
            "created_at": "runtime"
        })
        
        return memory_manager
        
    except Exception as e:
        logger.error(f"Failed to get memory manager: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory manager service unavailable"
        )


def get_inference_engine():
    """Get inference engine instance."""
    try:
        # Try to get from service registry
        inference_engine = get_service("inference_engine")
        if inference_engine:
            return inference_engine
        
        # If not available, this is an error since it should be initialized at startup
        logger.error("Inference engine not available in service registry")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized - server may be starting up"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get inference engine: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine service unavailable"
        )


def get_model_manager():
    """Get model manager instance."""
    try:
        # Try to get from service registry
        model_manager = get_service("model_manager")
        if model_manager:
            return model_manager
        
        # Create a basic model manager if not available
        logger.warning("Model manager not found in registry, creating basic instance")
        
        model_manager = BasicModelManager()
        register_service("model_manager", model_manager, {
            "type": "basic_model_manager",
            "created_at": "runtime"
        })
        
        return model_manager
        
    except Exception as e:
        logger.error(f"Failed to get model manager: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager service unavailable"
        )


def get_autoscaler():
    """Get autoscaler instance (optional)."""
    try:
        return get_service("autoscaler")
    except Exception as e:
        logger.debug(f"Autoscaler not available: {e}")
        return None


def get_security_manager():
    """Get security manager instance (optional)."""
    try:
        return get_service("security_manager")
    except Exception as e:
        logger.debug(f"Security manager not available: {e}")
        return None


def get_metrics_collector():
    """Get metrics collector instance (optional)."""
    try:
        return get_service("metrics_collector")
    except Exception as e:
        logger.debug(f"Metrics collector not available: {e}")
        return None


# Legacy service dependencies for backward compatibility
def get_inference_service():
    """Get inference service (legacy compatibility)."""
    try:
        service = get_service("inference_service")
        if service:
            return service
        
        # Create wrapper around inference engine
        inference_engine = get_inference_engine()
        model_manager = get_model_manager()
        
        from ..services.inference_service import InferenceServiceWrapper
        service = InferenceServiceWrapper(inference_engine, model_manager)
        
        register_service("inference_service", service, {
            "type": "inference_service_wrapper",
            "created_at": "runtime"
        })
        
        return service
        
    except Exception as e:
        logger.error(f"Failed to get inference service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service unavailable"
        )


def get_model_service():
    """Get model service (legacy compatibility)."""
    try:
        service = get_service("model_service")
        if service:
            return service
        
        # Create wrapper around model manager
        model_manager = get_model_manager()
        
        from ..services.model_service import ModelServiceWrapper
        service = ModelServiceWrapper(model_manager)
        
        register_service("model_service", service, {
            "type": "model_service_wrapper",
            "created_at": "runtime"
        })
        
        return service
        
    except Exception as e:
        logger.error(f"Failed to get model service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service unavailable"
        )


def get_audio_service():
    """Get audio service (legacy compatibility)."""
    try:
        service = get_service("audio_service")
        if service:
            return service
        
        # Create wrapper around model manager for audio models
        model_manager = get_model_manager()
        
        from ..services.audio_service import AudioServiceWrapper
        service = AudioServiceWrapper(model_manager)
        
        register_service("audio_service", service, {
            "type": "audio_service_wrapper", 
            "created_at": "runtime"
        })
        
        return service
        
    except Exception as e:
        logger.error(f"Failed to get audio service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audio service unavailable"
        )


def get_download_service():
    """Get download service (legacy compatibility)."""
    try:
        service = get_service("download_service")
        if service:
            return service
        
        # Create basic download service
        from ..services.download_service import BasicDownloadService
        service = BasicDownloadService()
        
        register_service("download_service", service, {
            "type": "basic_download_service",
            "created_at": "runtime"
        })
        
        return service
        
    except Exception as e:
        logger.error(f"Failed to get download service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Download service unavailable"
        )


# Service health check dependencies

def check_inference_service_health() -> Dict[str, Any]:
    """Check health of inference services."""
    return _registry.get_health_status()


def get_service_status() -> Dict[str, Any]:
    """Get comprehensive status of all services."""
    services = list_services()
    status = {
        "total_services": len(services),
        "services": {},
        "overall_healthy": True
    }
    
    for service_name in services:
        try:
            service = get_service(service_name)
            service_info = get_service_info(service_name)
            
            service_status = {
                "available": True,
                "type": service_info["type"] if service_info else "unknown",
                "metadata": service_info["metadata"] if service_info else {}
            }
            
            # Try to get health status if available
            if hasattr(service, 'get_health_status'):
                health = service.get_health_status()
                service_status.update(health)
                if not health.get("healthy", True):
                    status["overall_healthy"] = False
            elif hasattr(service, 'health_check'):
                service_status["healthy"] = True
            else:
                service_status["healthy"] = True
            
            status["services"][service_name] = service_status
            
        except Exception as e:
            status["services"][service_name] = {
                "available": False,
                "error": str(e),
                "healthy": False
            }
            status["overall_healthy"] = False
    
    return status


# Authentication dependencies (optional)

def get_current_user(token: Optional[str] = None):
    """Get current authenticated user (optional dependency)."""
    try:
        security_manager = get_security_manager()
        if security_manager and token:
            return security_manager.validate_token(token)
        return None
    except Exception as e:
        logger.debug(f"Authentication failed: {e}")
        return None


def require_authentication(token: Optional[str] = None):
    """Require authentication (raises exception if not authenticated)."""
    try:
        user = get_current_user(token)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable"
        )


# Service initialization and cleanup

async def initialize_services(app_config=None):
    """Initialize all services with enhanced error handling."""
    logger.info("Initializing enhanced services...")
    
    try:
        if not app_config:
            app_config = get_config()
        
        initialization_order = [
            "memory_manager",
            "model_manager", 
            "inference_engine",
            "autoscaler",
            "security_manager",
            "metrics_collector"
        ]
        
        # Initialize memory manager
        memory_manager = get_memory_manager()
        logger.info("✓ Memory manager initialized")
        
        # Initialize model manager
        model_manager = BasicModelManager()
        register_service("model_manager", model_manager, {
            "type": "basic_model_manager",
            "initialized_at": "startup"
        })
        logger.info("✓ Model manager initialized")
        
        # Initialize inference engine
        from ..core.engine import create_inference_engine
        from ..core.base_model import create_example_model
        
        example_model = create_example_model(app_config.inference)
        inference_engine = create_inference_engine(example_model, app_config.inference)
        await inference_engine.start()
        
        register_service("inference_engine", inference_engine, {
            "type": "inference_engine",
            "device": str(inference_engine.device),
            "initialized_at": "startup"
        })
        logger.info("✓ Inference engine initialized")
        
        # Initialize autoscaler (optional)
        if app_config.inference.autoscaling.enabled:
            try:
                from ..core.autoscaler import create_autoscaler
                autoscaler = await create_autoscaler(app_config.inference.autoscaling, model_manager)
                register_service("autoscaler", autoscaler, {
                    "type": "autoscaler",
                    "enabled": True,
                    "initialized_at": "startup"
                })
                logger.info("✓ Autoscaler initialized")
            except Exception as e:
                logger.warning(f"Autoscaler initialization failed: {e}")
        
        # Initialize security manager (optional)
        if app_config.security.enable_auth:
            try:
                from ..core.security_manager import create_security_manager
                security_manager = create_security_manager(app_config.security)
                register_service("security_manager", security_manager, {
                    "type": "security_manager",
                    "auth_enabled": True,
                    "initialized_at": "startup"
                })
                logger.info("✓ Security manager initialized")
            except Exception as e:
                logger.warning(f"Security manager initialization failed: {e}")
        
        # Initialize metrics collector (optional)
        try:
            from ..core.metrics import create_metrics_collector
            metrics_collector = create_metrics_collector()
            register_service("metrics_collector", metrics_collector, {
                "type": "metrics_collector",
                "initialized_at": "startup"
            })
            logger.info("✓ Metrics collector initialized")
        except Exception as e:
            logger.warning(f"Metrics collector initialization failed: {e}")
        
        # Validate critical services
        critical_services = ["memory_manager", "model_manager", "inference_engine"]
        missing_services = [name for name in critical_services if not get_service(name)]
        
        if missing_services:
            raise ConfigurationError(
                config_field="critical_services",
                details=f"Critical services missing: {missing_services}"
            )
        
        logger.info(f"Enhanced services initialization completed - {len(list_services())} services active")
        
        # Log service summary
        for service_name in list_services():
            info = get_service_info(service_name)
            logger.debug(f"  - {service_name}: {info['type']} ({info['metadata']})")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise ConfigurationError(
            config_field="services",
            details=f"Service initialization failed: {e}",
            cause=e
        )


async def cleanup_services():
    """Cleanup all services with proper error handling."""
    logger.info("Cleaning up services...")
    
    cleanup_order = [
        "autoscaler",
        "inference_engine", 
        "model_manager",
        "memory_manager",
        "security_manager",
        "metrics_collector"
    ]
    
    cleanup_errors = []
    
    for service_name in cleanup_order:
        try:
            service = get_service(service_name)
            if service:
                if hasattr(service, 'stop'):
                    await service.stop()
                    logger.debug(f"✓ {service_name} stopped")
                elif hasattr(service, 'shutdown'):
                    service.shutdown()
                    logger.debug(f"✓ {service_name} shutdown")
                elif hasattr(service, 'cleanup'):
                    service.cleanup()
                    logger.debug(f"✓ {service_name} cleaned up")
                
                unregister_service(service_name)
        except Exception as e:
            cleanup_errors.append(f"{service_name}: {e}")
            logger.error(f"Error cleaning up {service_name}: {e}")
    
    # Clear all remaining services
    clear_services()
    
    if cleanup_errors:
        logger.warning(f"Cleanup completed with {len(cleanup_errors)} errors: {cleanup_errors}")
    else:
        logger.info("Services cleanup completed successfully")


# Utility functions

def validate_service_dependencies() -> List[str]:
    """Validate that all required services are available and healthy."""
    issues = []
    
    required_services = ["memory_manager", "model_manager", "inference_engine"]
    
    for service_name in required_services:
        service = get_service(service_name)
        if not service:
            issues.append(f"Required service '{service_name}' not registered")
        else:
            try:
                # Check if service has health check capability
                if hasattr(service, 'get_health_status'):
                    health = service.get_health_status()
                    if not health.get("healthy", True):
                        issues.append(f"Required service '{service_name}' is unhealthy: {health}")
                elif hasattr(service, 'health_check'):
                    # Service has health check method, assume it's healthy if no exception
                    pass
            except Exception as e:
                issues.append(f"Required service '{service_name}' health check failed: {e}")
    
    return issues


# Context managers for service management

class ServiceContext:
    """Context manager for service lifecycle management."""
    
    def __init__(self, config=None):
        self.config = config
    
    async def __aenter__(self):
        await initialize_services(self.config)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await cleanup_services()


# Service decorators

def requires_service(service_name: str):
    """Decorator to require a specific service."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            service = get_service(service_name)
            if not service:
                raise ServiceUnavailableError(
                    service=service_name,
                    details=f"Required service '{service_name}' not available"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Basic model manager implementation for fallback

class BasicModelManager:
    """Basic model manager implementation for fallback scenarios."""
    
    def __init__(self):
        self._models = {}
        self._model_info = {}
    
    def list_models(self) -> List[str]:
        """List available models."""
        return list(self._models.keys())
    
    def get_loaded_models(self) -> List[str]:
        """Get currently loaded models."""
        return [name for name, model in self._models.items() if model is not None]
    
    def has_model(self, model_name: str) -> bool:
        """Check if model is available."""
        return model_name in self._models
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if model is loaded."""
        return model_name in self._models and self._models[model_name] is not None
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information."""
        return self._model_info.get(model_name, {
            "name": model_name,
            "status": "available",
            "type": "unknown"
        })
    
    def get_tts_models(self) -> List[str]:
        """Get TTS models."""
        return [name for name, info in self._model_info.items() 
                if info.get("type") == "text-to-speech"]
    
    def get_stt_models(self) -> List[str]:
        """Get STT models."""
        return [name for name, info in self._model_info.items() 
                if info.get("type") == "speech-to-text"]
    
    def is_tts_model(self, model_name: str) -> bool:
        """Check if model is a TTS model."""
        info = self.get_model_info(model_name)
        return info.get("type") == "text-to-speech"
    
    def is_stt_model(self, model_name: str) -> bool:
        """Check if model is an STT model.""" 
        info = self.get_model_info(model_name)
        return info.get("type") == "speech-to-text"
    
    def cleanup_all(self):
        """Cleanup all models."""
        self._models.clear()
        self._model_info.clear()
    
    def register_model(self, name: str, model: Any, info: Optional[Dict[str, Any]] = None):
        """Register a model."""
        self._models[name] = model
        self._model_info[name] = info or {}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        return {
            "healthy": True,
            "total_models": len(self._models),
            "loaded_models": len(self.get_loaded_models())
        }

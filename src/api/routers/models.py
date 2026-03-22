"""
Model management endpoints for loading, unloading, and managing ML models.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from pydantic import BaseModel

from ...services.model import ModelService
from ...models.api.models import ModelInfoResponse, ModelListResponse, ModelLoadRequest, ModelUnloadRequest
from ...models.api.base import SuccessResponse, ErrorResponse
from ...core.exceptions import ModelNotFoundError, ValidationError, InternalServerError
from ..dependencies import get_model_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["Model Management"])


class ModelStatus(BaseModel):
    """Model status information."""
    model_name: str
    status: str  # loaded, unloaded, loading, error
    device: Optional[str] = None
    memory_usage: Optional[int] = None
    load_time: Optional[float] = None
    last_used: Optional[datetime] = None
    error_message: Optional[str] = None


class ModelRegistryInfo(BaseModel):
    """Model registry information."""
    total_models: int
    loaded_models: int
    available_models: List[str]
    memory_usage: Optional[Dict[str, Any]] = None
    device_usage: Optional[Dict[str, Any]] = None


@router.get("/",
           response_model=ModelListResponse,
           summary="List Models",
           description="Get list of all available and loaded models")
async def list_models(
    include_details: bool = Query(False, description="Include detailed information for each model"),
    model_service: ModelService = Depends(get_model_service)
) -> ModelListResponse:
    """
    Get list of all available models.
    
    Args:
        include_details: Whether to include detailed model information
        model_service: Model service dependency
    
    Returns:
        ModelListResponse: List of models with optional details
    """
    try:
        logger.debug("[MODELS] Getting model list...")
        
        model_names = model_service.list_models()
        if hasattr(model_names, 'models'):
            model_names = model_names.models

        if include_details:
            model_info = {}
            for model_name in model_names:
                try:
                    info = model_service.get_model_info(model_name)
                    # If info is not a ModelInfo, convert if needed
                    if hasattr(info, 'dict'):
                        model_info[model_name] = info
                    else:
                        model_info[model_name] = info  # fallback
                except Exception as e:
                    logger.warning(f"Failed to get details for model {model_name}: {e}")
            models = list(model_info.keys())
        else:
            models = model_names
            model_info = {name: None for name in models}

        total_models = len(models)
        # If no models, ensure empty dict/list
        return ModelListResponse(models=models, model_info=model_info, total_models=total_models)
        
    except Exception as e:
        logger.error(f"[MODELS] Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {e}"
        )


@router.get("/registry",
           response_model=ModelRegistryInfo,
           summary="Model Registry Info",
           description="Get information about the model registry")
async def get_registry_info(
    model_service: ModelService = Depends(get_model_service)
) -> ModelRegistryInfo:
    """
    Get model registry information.
    
    Args:
        model_service: Model service dependency
    
    Returns:
        ModelRegistryInfo: Registry information with statistics
    """
    try:
        logger.debug("[MODELS] Getting registry information...")
        
        registry_info = await model_service.get_registry_info()
        
        return ModelRegistryInfo(
            total_models=registry_info.get("total_models", 0),
            loaded_models=registry_info.get("loaded_models", 0),
            available_models=registry_info.get("available_models", []),
            memory_usage=registry_info.get("memory_usage"),
            device_usage=registry_info.get("device_usage")
        )
        
    except Exception as e:
        logger.error(f"[MODELS] Failed to get registry info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get registry info: {e}"
        )


@router.get("/{model_name}",
           response_model=ModelInfoResponse,
           summary="Get Model Info",
           description="Get detailed information about a specific model")
async def get_model_info(
    model_name: str,
    model_service: ModelService = Depends(get_model_service)
) -> ModelInfoResponse:
    """
    Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model
        model_service: Model service dependency
    
    Returns:
        ModelInfoResponse: Detailed model information
    
    Raises:
        HTTPException: If model not found
    """
    try:
        logger.debug(f"[MODELS] Getting info for model: {model_name}")
        
        model_info = await model_service.get_model_info(model_name)
        
        return ModelInfoResponse(**model_info)
        
    except ModelNotFoundError:
        logger.warning(f"[MODELS] Model not found: {model_name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found"
        )
    
    except Exception as e:
        logger.error(f"[MODELS] Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {e}"
        )


@router.get("/{model_name}/status",
           response_model=ModelStatus,
           summary="Get Model Status",
           description="Get current status of a specific model")
async def get_model_status(
    model_name: str,
    model_service: ModelService = Depends(get_model_service)
) -> ModelStatus:
    """
    Get current status of a specific model.
    
    Args:
        model_name: Name of the model
        model_service: Model service dependency
    
    Returns:
        ModelStatus: Current model status
    """
    try:
        logger.debug(f"[MODELS] Getting status for model: {model_name}")
        
        status_info = await model_service.get_model_status(model_name)
        
        return ModelStatus(
            model_name=model_name,
            status=status_info.get("status", "unknown"),
            device=status_info.get("device"),
            memory_usage=status_info.get("memory_usage"),
            load_time=status_info.get("load_time"),
            last_used=status_info.get("last_used"),
            error_message=status_info.get("error_message")
        )
        
    except ModelNotFoundError:
        logger.warning(f"[MODELS] Model not found: {model_name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found"
        )
    
    except Exception as e:
        logger.error(f"[MODELS] Failed to get model status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model status: {e}"
        )


@router.post("/{model_name}/load",
            response_model=SuccessResponse,
            summary="Load Model",
            description="Load a model into memory for inference")
async def load_model(
    model_name: str,
    request: ModelLoadRequest,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(get_model_service)
) -> SuccessResponse:
    """
    Load a model into memory.
    
    Args:
        model_name: Name of the model to load
        request: Model load configuration
        background_tasks: Background tasks for async operations
        model_service: Model service dependency
    
    Returns:
        SuccessResponse: Load operation status
    """
    try:
        logger.info(f"[MODELS] Loading model: {model_name}")
        
        # Start loading in background
        background_tasks.add_task(
            _load_model_background,
            model_service=model_service,
            model_name=model_name,
            device=request.device,
            force_reload=request.force_reload
        )
        
        return SuccessResponse(
            success=True,
            message=f"Model '{model_name}' loading started",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"[MODELS] Failed to start model loading: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start model loading: {e}"
        )


@router.post("/{model_name}/unload",
            response_model=SuccessResponse,
            summary="Unload Model",
            description="Unload a model from memory")
async def unload_model(
    model_name: str,
    request: ModelUnloadRequest,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(get_model_service)
) -> SuccessResponse:
    """
    Unload a model from memory.
    
    Args:
        model_name: Name of the model to unload
        request: Unload configuration
        background_tasks: Background tasks for async operations
        model_service: Model service dependency
    
    Returns:
        SuccessResponse: Unload operation status
    """
    try:
        logger.info(f"[MODELS] Unloading model: {model_name}")
        
        if request.immediate:
            # Unload immediately
            await model_service.unload_model(model_name, force=request.force)
            message = f"Model '{model_name}' unloaded immediately"
        else:
            # Unload in background
            background_tasks.add_task(
                _unload_model_background,
                model_service=model_service,
                model_name=model_name,
                force=request.force
            )
            message = f"Model '{model_name}' unloading started"
        
        return SuccessResponse(
            success=True,
            message=message,
            timestamp=datetime.now()
        )
        
    except ModelNotFoundError:
        logger.warning(f"[MODELS] Model not found for unloading: {model_name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found"
        )
    
    except Exception as e:
        logger.error(f"[MODELS] Failed to unload model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unload model: {e}"
        )


@router.post("/{model_name}/reload",
            response_model=SuccessResponse,
            summary="Reload Model",
            description="Reload a model (unload and load again)")
async def reload_model(
    model_name: str,
    background_tasks: BackgroundTasks,
    device: Optional[str] = Query(None, description="Target device for reloading"),
    model_service: ModelService = Depends(get_model_service)
) -> SuccessResponse:
    """
    Reload a model (unload and load again).
    
    Args:
        model_name: Name of the model to reload
        background_tasks: Background tasks for async operations
        device: Target device for reloading
        model_service: Model service dependency
    
    Returns:
        SuccessResponse: Reload operation status
    """
    try:
        logger.info(f"[MODELS] Reloading model: {model_name}")
        
        # Start reloading in background
        background_tasks.add_task(
            _reload_model_background,
            model_service=model_service,
            model_name=model_name,
            device=device
        )
        
        return SuccessResponse(
            success=True,
            message=f"Model '{model_name}' reloading started",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"[MODELS] Failed to start model reloading: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start model reloading: {e}"
        )


# Background task functions

async def _load_model_background(model_service: ModelService, model_name: str, device: Optional[str] = None, force_reload: bool = False):
    """Load model in background."""
    try:
        logger.info(f"[MODELS] Starting background loading of model: {model_name}")
        
        await model_service.load_model(model_name, device=device, force_reload=force_reload)
        
        logger.info(f"[MODELS] Successfully loaded model in background: {model_name}")
        
    except Exception as e:
        logger.error(f"[MODELS] Background loading failed for model {model_name}: {e}")


async def _unload_model_background(model_service: ModelService, model_name: str, force: bool = False):
    """Unload model in background."""
    try:
        logger.info(f"[MODELS] Starting background unloading of model: {model_name}")
        
        await model_service.unload_model(model_name, force=force)
        
        logger.info(f"[MODELS] Successfully unloaded model in background: {model_name}")
        
    except Exception as e:
        logger.error(f"[MODELS] Background unloading failed for model {model_name}: {e}")


async def _reload_model_background(model_service: ModelService, model_name: str, device: Optional[str] = None):
    """Reload model in background."""
    try:
        logger.info(f"[MODELS] Starting background reloading of model: {model_name}")
        
        # First unload
        try:
            await model_service.unload_model(model_name, force=False)
        except ModelNotFoundError:
            pass  # Model wasn't loaded, continue with loading
        
        # Then load
        await model_service.load_model(model_name, device=device)
        
        logger.info(f"[MODELS] Successfully reloaded model in background: {model_name}")
        
    except Exception as e:
        logger.error(f"[MODELS] Background reloading failed for model {model_name}: {e}")

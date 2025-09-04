"""
Autoscaler management API endpoints.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks

from ...services.autoscaler_service import AutoscalerService
from ...models.api.common import APIResponse

router = APIRouter(prefix="/autoscaler", tags=["autoscaler"])
logger = logging.getLogger(__name__)

autoscaler_service = AutoscalerService()


@router.get("/stats")
async def get_autoscaler_stats() -> APIResponse:
    """Get autoscaler statistics."""
    try:
        stats = autoscaler_service.get_stats()
        return APIResponse(
            success=True,
            data=stats,
            message="Autoscaler statistics retrieved"
        )
    except Exception as e:
        logger.error(f"Autoscaler stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_autoscaler_health() -> APIResponse:
    """Get autoscaler health status."""
    try:
        health = autoscaler_service.get_health_status()
        return APIResponse(
            success=True,
            data=health,
            message="Autoscaler health check completed"
        )
    except Exception as e:
        logger.error(f"Autoscaler health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scale")
async def scale_model(
    model_name: str,
    target_instances: int,
    background_tasks: BackgroundTasks
) -> APIResponse:
    """Scale a model to target instances."""
    try:
        background_tasks.add_task(
            autoscaler_service.scale_model,
            model_name,
            target_instances
        )
        return APIResponse(
            success=True,
            data={"model_name": model_name, "target_instances": target_instances},
            message=f"Scaling request for {model_name} to {target_instances} instances"
        )
    except Exception as e:
        logger.error(f"Model scaling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_model(model_name: str, background_tasks: BackgroundTasks) -> APIResponse:
    """Load a model with autoscaling."""
    try:
        background_tasks.add_task(autoscaler_service.load_model, model_name)
        return APIResponse(
            success=True,
            data={"model_name": model_name},
            message=f"Model {model_name} loading started"
        )
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/unload")
async def unload_model(model_name: str, background_tasks: BackgroundTasks) -> APIResponse:
    """Unload a model."""
    try:
        background_tasks.add_task(autoscaler_service.unload_model, model_name)
        return APIResponse(
            success=True,
            data={"model_name": model_name},
            message=f"Model {model_name} unloading started"
        )
    except Exception as e:
        logger.error(f"Model unloading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_autoscaler_metrics() -> APIResponse:
    """Get detailed autoscaling metrics."""
    try:
        metrics = autoscaler_service.get_detailed_metrics()
        return APIResponse(
            success=True,
            data=metrics,
            message="Autoscaler metrics retrieved"
        )
    except Exception as e:
        logger.error(f"Autoscaler metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

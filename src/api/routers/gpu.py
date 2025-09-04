"""
GPU detection and management API endpoints.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

from ...services.gpu_service import GPUService
from ...models.api.common import APIResponse

router = APIRouter(prefix="/gpu", tags=["gpu"])
logger = logging.getLogger(__name__)

gpu_service = GPUService()


@router.get("/detect")
async def detect_gpus() -> APIResponse:
    """Detect available GPUs."""
    try:
        gpu_info = gpu_service.detect_gpus()
        return APIResponse(
            success=True,
            data=gpu_info,
            message="GPU detection completed successfully"
        )
    except Exception as e:
        logger.error(f"GPU detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/best")
async def get_best_gpu() -> APIResponse:
    """Get the best GPU for inference."""
    try:
        best_gpu = gpu_service.get_best_gpu()
        return APIResponse(
            success=True,
            data=best_gpu,
            message="Best GPU identified"
        )
    except Exception as e:
        logger.error(f"Best GPU selection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_gpu_config() -> APIResponse:
    """Get GPU-optimized configuration."""
    try:
        config = gpu_service.get_optimized_config()
        return APIResponse(
            success=True,
            data=config,
            message="GPU configuration retrieved"
        )
    except Exception as e:
        logger.error(f"GPU configuration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report")
async def get_gpu_report() -> APIResponse:
    """Get comprehensive GPU report."""
    try:
        report = gpu_service.get_comprehensive_report()
        return APIResponse(
            success=True,
            data=report,
            message="GPU report generated successfully"
        )
    except Exception as e:
        logger.error(f"GPU report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

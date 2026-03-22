"""
GPU detection and management API endpoints.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...services.gpu_service import GPUService
from ...models.api.common import APIResponse

router = APIRouter(prefix="/gpu", tags=["gpu"])
logger = logging.getLogger(__name__)

gpu_service = GPUService()


class MultiGPUConfigRequest(BaseModel):
    """Request model for multi-GPU configuration."""
    enabled: bool
    strategy: str = "data_parallel"
    device_ids: list[int] = None
    load_balancing: str = "dynamic"
    fault_tolerance: bool = True


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


@router.get("/multi-gpu/status")
async def get_multi_gpu_status() -> APIResponse:
    """Get multi-GPU status and configuration."""
    try:
        from ....framework.core.gpu_manager import get_multi_gpu_configuration
        
        config = get_multi_gpu_configuration()
        return APIResponse(
            success=True,
            data=config,
            message="Multi-GPU status retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Multi-GPU status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/multi-gpu/stats")
async def get_multi_gpu_stats() -> APIResponse:
    """Get detailed multi-GPU performance statistics."""
    try:
        from ....framework.core.gpu_manager import get_gpu_manager
        
        manager = get_gpu_manager()
        if manager.multi_gpu_manager and manager.multi_gpu_manager.is_initialized:
            stats = manager.multi_gpu_manager.get_detailed_stats()
            return APIResponse(
                success=True,
                data=stats,
                message="Multi-GPU statistics retrieved successfully"
            )
        else:
            return APIResponse(
                success=True,
                data={"status": "not_initialized", "message": "Multi-GPU not initialized"},
                message="Multi-GPU not active"
            )
    except Exception as e:
        logger.error(f"Multi-GPU stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi-gpu/configure")
async def configure_multi_gpu(config_request: MultiGPUConfigRequest) -> APIResponse:
    """Configure multi-GPU settings."""
    try:
        from ....framework.core.config import MultiGPUConfig
        from ....framework.core.gpu_manager import setup_multi_gpu, validate_multi_gpu_setup
        
        # Create config from request
        config = MultiGPUConfig(
            enabled=config_request.enabled,
            strategy=config_request.strategy,
            device_ids=config_request.device_ids,
            load_balancing=config_request.load_balancing,
            fault_tolerance=config_request.fault_tolerance
        )
        
        # Validate configuration
        validation = validate_multi_gpu_setup(config)
        if not validation["valid"]:
            return APIResponse(
                success=False,
                data=validation,
                message="Multi-GPU configuration invalid"
            )
        
        # Setup multi-GPU if enabled
        if config.enabled:
            multi_gpu_manager = setup_multi_gpu(config)
            result = multi_gpu_manager.initialize()
            
            return APIResponse(
                success=True,
                data={
                    "configuration": result,
                    "validation": validation
                },
                message="Multi-GPU configured successfully"
            )
        else:
            return APIResponse(
                success=True,
                data={"status": "disabled", "validation": validation},
                message="Multi-GPU disabled"
            )
            
    except Exception as e:
        logger.error(f"Multi-GPU configuration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/multi-gpu/devices")
async def list_multi_gpu_devices() -> APIResponse:
    """List all available GPU devices with utilization."""
    try:
        from ....framework.core.gpu_manager import get_gpu_manager
        
        manager = get_gpu_manager()
        gpus = manager.get_detected_gpus()
        
        device_list = []
        for gpu in gpus:
            device_info = {
                "id": gpu.id,
                "name": gpu.name,
                "vendor": gpu.vendor.value,
                "memory_total_mb": gpu.memory.total_mb,
                "memory_available_mb": gpu.memory.available_mb,
                "utilization": getattr(gpu, 'utilization', 0.0),
                "is_suitable": gpu.is_suitable_for_inference(),
                "pytorch_support": gpu.pytorch_support,
                "architecture": gpu.architecture.value if gpu.architecture else "unknown"
            }
            
            # Add multi-GPU specific info if available
            if manager.multi_gpu_manager and manager.multi_gpu_manager.is_initialized:
                if manager.multi_gpu_manager.device_pool and gpu.id in manager.multi_gpu_manager.device_pool.devices:
                    device_pool_info = manager.multi_gpu_manager.device_pool.devices[gpu.id]
                    device_info.update({
                        "is_healthy": device_pool_info.is_healthy,
                        "active_batches": device_pool_info.active_batches,
                        "failure_count": device_pool_info.failure_count,
                        "last_used": device_pool_info.last_used
                    })
            
            device_list.append(device_info)
        
        return APIResponse(
            success=True,
            data={
                "devices": device_list,
                "total_devices": len(device_list),
                "suitable_devices": len([d for d in device_list if d["is_suitable"]]),
                "multi_gpu_status": "initialized" if (manager.multi_gpu_manager and manager.multi_gpu_manager.is_initialized) else "not_initialized"
            },
            message="GPU devices listed successfully"
        )
        
    except Exception as e:
        logger.error(f"GPU device listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi-gpu/rebalance")
async def rebalance_multi_gpu_load() -> APIResponse:
    """Trigger manual load rebalancing."""
    try:
        from ....framework.core.gpu_manager import get_gpu_manager
        
        manager = get_gpu_manager()
        if not manager.multi_gpu_manager or not manager.multi_gpu_manager.is_initialized:
            raise HTTPException(status_code=400, detail="Multi-GPU not initialized")
        
        if manager.multi_gpu_manager.load_balancer:
            manager.multi_gpu_manager.load_balancer.rebalance()
            
            return APIResponse(
                success=True,
                data={
                    "rebalance_count": manager.multi_gpu_manager.load_balancer.rebalance_count,
                    "last_rebalance": manager.multi_gpu_manager.load_balancer.last_rebalance
                },
                message="Load rebalancing triggered successfully"
            )
        else:
            raise HTTPException(status_code=400, detail="Load balancer not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Load rebalancing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

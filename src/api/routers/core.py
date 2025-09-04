"""
Core system endpoints for health checks and system information.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ...services.inference import InferenceService
from ...models.api.base import HealthCheckResponse, SuccessResponse, ErrorResponse
from ...core.exceptions import ServiceUnavailableError, InternalServerError
from ..dependencies import get_inference_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/core", tags=["Core System"])


class SystemStatus(BaseModel):
    """System status information."""
    status: str
    timestamp: datetime
    uptime: Optional[float] = None
    version: str = "1.0.0"
    framework_version: Optional[str] = None


class SystemInfo(BaseModel):
    """Detailed system information."""
    server_time: datetime
    python_version: str
    pytorch_version: Optional[str] = None
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    device_count: int = 0
    memory_usage: Optional[Dict[str, Any]] = None
    environment: str = "development"


@router.get("/health", 
           response_model=HealthCheckResponse,
           summary="Health Check",
           description="Check if the API server is healthy and ready to serve requests")
async def health_check(
    inference_service: InferenceService = Depends(get_inference_service)
) -> HealthCheckResponse:
    """
    Perform a comprehensive health check of all system components.
    
    Returns:
        HealthCheckResponse: Health status with detailed component checks
    """
    try:
        logger.debug("[HEALTH] Performing health check...")
        
        # Get health status from inference service
        health_result = await inference_service.health_check()
        
        logger.debug(f"[HEALTH] Health check completed: {health_result['healthy']}")
        
        return HealthCheckResponse(
            healthy=health_result["healthy"],
            checks=health_result.get("checks", {}),
            timestamp=health_result.get("timestamp", datetime.now().timestamp()),
            message="Health check completed successfully" if health_result["healthy"] else "Some components are unhealthy"
        )
        
    except Exception as e:
        logger.error(f"[HEALTH] Health check failed: {e}")
        return HealthCheckResponse(
            healthy=False,
            checks={"error": str(e)},
            timestamp=datetime.now().timestamp(),
            message=f"Health check failed: {e}"
        )


@router.get("/status",
           response_model=SystemStatus,
           summary="System Status",
           description="Get basic system status information")
async def get_system_status() -> SystemStatus:
    """
    Get current system status.
    
    Returns:
        SystemStatus: Basic system status information
    """
    try:
        logger.debug("[STATUS] Getting system status...")
        
        # Calculate uptime if possible
        uptime = None
        try:
            import psutil
            import os
            uptime = datetime.now().timestamp() - psutil.Process(os.getpid()).create_time()
        except ImportError:
            pass
        
        # Get framework version if available
        framework_version = None
        try:
            import framework
            framework_version = getattr(framework, '__version__', 'unknown')
        except ImportError:
            pass
        
        return SystemStatus(
            status="healthy",
            timestamp=datetime.now(),
            uptime=uptime,
            framework_version=framework_version
        )
        
    except Exception as e:
        logger.error(f"[STATUS] Failed to get system status: {e}")
        raise InternalServerError(f"Failed to get system status: {e}")


@router.get("/info",
           response_model=SystemInfo,
           summary="System Information", 
           description="Get detailed system and environment information")
async def get_system_info() -> SystemInfo:
    """
    Get detailed system information including hardware and environment details.
    
    Returns:
        SystemInfo: Detailed system information
    """
    try:
        logger.debug("[INFO] Getting system information...")
        
        import sys
        
        # Get PyTorch info
        pytorch_version = None
        cuda_available = False
        cuda_version = None
        device_count = 0
        
        try:
            import torch
            pytorch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()
        except ImportError:
            logger.warning("[INFO] PyTorch not available")
        
        # Get memory usage
        memory_usage = None
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_usage = {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            }
            
            # Add GPU memory if available
            if cuda_available:
                try:
                    import torch
                    gpu_memory = {}
                    for i in range(device_count):
                        props = torch.cuda.get_device_properties(i)
                        memory_info = torch.cuda.mem_get_info(i)
                        gpu_memory[f"gpu_{i}"] = {
                            "name": props.name,
                            "total_memory": props.total_memory,
                            "free_memory": memory_info[0],
                            "used_memory": props.total_memory - memory_info[0]
                        }
                    memory_usage["gpu"] = gpu_memory
                except Exception as e:
                    logger.warning(f"[INFO] Failed to get GPU memory info: {e}")
                    
        except ImportError:
            logger.warning("[INFO] psutil not available for memory information")
        
        # Determine environment
        import os
        environment = os.getenv("ENVIRONMENT", "development").lower()
        
        return SystemInfo(
            server_time=datetime.now(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            pytorch_version=pytorch_version,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            device_count=device_count,
            memory_usage=memory_usage,
            environment=environment
        )
        
    except Exception as e:
        logger.error(f"[INFO] Failed to get system information: {e}")
        raise InternalServerError(f"Failed to get system information: {e}")


@router.get("/ping",
           response_model=Dict[str, Any],
           summary="Ping",
           description="Simple ping endpoint for basic connectivity testing")
async def ping() -> Dict[str, Any]:
    """
    Simple ping endpoint for basic connectivity testing.
    
    Returns:
        Dict[str, Any]: Ping response with timestamp
    """
    return {
        "ping": "pong",
        "timestamp": datetime.now().isoformat(),
        "status": "ok"
    }


@router.get("/version",
           response_model=Dict[str, str],
           summary="Version Information",
           description="Get API and framework version information")
async def get_version() -> Dict[str, str]:
    """
    Get version information for the API and framework.
    
    Returns:
        Dict[str, str]: Version information
    """
    try:
        # Get framework version if available
        framework_version = "not_available"
        try:
            import framework
            framework_version = getattr(framework, '__version__', 'unknown')
        except ImportError:
            pass
        
        # Get PyTorch version
        pytorch_version = "not_available"
        try:
            import torch
            pytorch_version = torch.__version__
        except ImportError:
            pass
        
        return {
            "api_version": "1.0.0",
            "framework_version": framework_version,
            "pytorch_version": pytorch_version
        }
        
    except Exception as e:
        logger.error(f"[VERSION] Failed to get version information: {e}")
        raise InternalServerError(f"Failed to get version information: {e}")

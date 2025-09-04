"""
Logging management API endpoints.
"""

import logging
from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...models.api.common import APIResponse

router = APIRouter(prefix="/logs", tags=["logs"])
logger = logging.getLogger(__name__)


@router.get("")
async def get_logging_info() -> APIResponse:
    """Get logging information and statistics."""
    try:
        log_dir = Path("logs")
        log_files = []
        
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                try:
                    file_size = log_file.stat().st_size
                    log_files.append({
                        "name": log_file.name,
                        "size_bytes": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2),
                        "path": str(log_file.absolute())
                    })
                except Exception as e:
                    log_files.append({
                        "name": log_file.name,
                        "error": str(e)
                    })
        
        return APIResponse(
            success=True,
            data={
                "log_directory": str(log_dir.absolute()),
                "log_files": log_files,
                "total_files": len(log_files)
            },
            message="Logging information retrieved"
        )
    except Exception as e:
        logger.error(f"Logging info retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{log_file}")
async def get_log_file(log_file: str):
    """Download or view specific log file."""
    try:
        log_path = Path("logs") / log_file
        
        if not log_path.exists():
            raise HTTPException(status_code=404, detail=f"Log file {log_file} not found")
        
        if not log_path.suffix == ".log":
            raise HTTPException(status_code=400, detail="Only .log files can be accessed")
        
        return FileResponse(
            path=str(log_path),
            media_type="text/plain",
            filename=log_file
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Log file retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{log_file}")
async def clear_log_file(log_file: str) -> APIResponse:
    """Clear specific log file."""
    try:
        log_path = Path("logs") / log_file
        
        if not log_path.exists():
            raise HTTPException(status_code=404, detail=f"Log file {log_file} not found")
        
        if not log_path.suffix == ".log":
            raise HTTPException(status_code=400, detail="Only .log files can be cleared")
        
        # Clear the file by opening in write mode
        with open(log_path, 'w') as f:
            f.write("")
        
        return APIResponse(
            success=True,
            data={"file": log_file},
            message=f"Log file {log_file} cleared successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Log file clearing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

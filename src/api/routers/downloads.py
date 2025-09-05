"""
Download endpoints for model downloads and file management.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Response
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, HttpUrl
import os
import mimetypes

from ...services.download import DownloadService
from ...models.api.downloads import ModelDownloadRequest, ModelDownloadResponse, DownloadStatus
from ...models.api.base import SuccessResponse, ErrorResponse
from ...core.exceptions import ValidationError, ProcessingError, InternalServerError, NotFoundError
from ..dependencies import get_download_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/downloads", tags=["Downloads"])


class DownloadInfo(BaseModel):
    """Download information."""
    download_id: str
    url: str
    filename: str
    status: str
    progress: Optional[float] = None
    file_size: Optional[int] = None
    downloaded_size: Optional[int] = None
    download_speed: Optional[float] = None
    eta: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class DownloadList(BaseModel):
    """List of downloads."""
    downloads: List[DownloadInfo]
    total_count: int
    active_downloads: int
    completed_downloads: int
    failed_downloads: int


class FileInfo(BaseModel):
    """File information."""
    filename: str
    file_path: str
    file_size: int
    mime_type: str
    created_at: datetime
    modified_at: datetime
    is_model_file: bool = False
    checksum: Optional[str] = None


@router.post("/models",
            response_model=ModelDownloadResponse,
            summary="Download Model",
            description="Download a model from external source")
async def download_model(
    request: ModelDownloadRequest,
    background_tasks: BackgroundTasks,
    download_service: DownloadService = Depends(get_download_service)
) -> ModelDownloadResponse:
    """
    Download a model from external source.
    
    Args:
        request: Model download request with URL and configuration
        background_tasks: Background tasks for async operations
        download_service: Download service dependency
    
    Returns:
        ModelDownloadResponse: Download status and metadata
    
    Raises:
        HTTPException: If download initiation fails
    """
    try:
        logger.info(f"[DOWNLOAD] Starting model download from: {request.url}")
        logger.debug(f"[DOWNLOAD] Model name: {request.model_name}, Force: {request.force_download}")
        
        # Validate URL
        if not str(request.url).startswith(('http://', 'https://')):
            raise ValidationError("Invalid URL format")
        
        # Start download
        result = await download_service.download_model(
            url=str(request.url),
            model_name=request.model_name,
            force_download=request.force_download,
            verify_checksum=request.verify_checksum,
            expected_checksum=request.expected_checksum
        )
        
        logger.info(f"[DOWNLOAD] Model download initiated: {result['download_id']}")
        
        # Add background monitoring
        background_tasks.add_task(
            _monitor_download,
            download_service=download_service,
            download_id=result['download_id']
        )
        
        return ModelDownloadResponse(**result)
        
    except ValidationError as e:
        logger.warning(f"[DOWNLOAD] Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except ProcessingError as e:
        logger.error(f"[DOWNLOAD] Processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Download failed: {e}"
        )
    
    except Exception as e:
        logger.error(f"[DOWNLOAD] Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start download: {e}"
        )


@router.get("/",
           response_model=DownloadList,
           summary="List Downloads",
           description="Get list of all downloads with status")
async def list_downloads(
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of downloads to return"),
    offset: int = Query(0, ge=0, description="Number of downloads to skip"),
    download_service: DownloadService = Depends(get_download_service)
) -> DownloadList:
    """
    Get list of downloads with filtering and pagination.
    
    Args:
        status_filter: Filter downloads by status
        limit: Maximum number of downloads to return
        offset: Number of downloads to skip
        download_service: Download service dependency
    
    Returns:
        DownloadList: List of downloads with metadata
    """
    try:
        logger.debug(f"[DOWNLOADS] Listing downloads - Filter: {status_filter}, Limit: {limit}, Offset: {offset}")
        
        downloads = await download_service.list_downloads(
            status_filter=status_filter,
            limit=limit,
            offset=offset
        )
        
        return DownloadList(**downloads)
        
    except Exception as e:
        logger.error(f"[DOWNLOADS] Failed to list downloads: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list downloads: {e}"
        )


@router.get("/{download_id}",
           response_model=DownloadInfo,
           summary="Get Download Info",
           description="Get detailed information about a specific download")
async def get_download_info(
    download_id: str,
    download_service: DownloadService = Depends(get_download_service)
) -> DownloadInfo:
    """
    Get detailed information about a specific download.
    
    Args:
        download_id: Download ID
        download_service: Download service dependency
    
    Returns:
        DownloadInfo: Detailed download information
    
    Raises:
        HTTPException: If download not found
    """
    try:
        logger.debug(f"[DOWNLOAD_INFO] Getting info for download: {download_id}")
        
        download_info = await download_service.get_download_info(download_id)
        
        if not download_info:
            raise NotFoundError(f"Download '{download_id}' not found")
        
        return DownloadInfo(**download_info)
        
    except NotFoundError:
        logger.warning(f"[DOWNLOAD_INFO] Download not found: {download_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Download '{download_id}' not found"
        )
    
    except Exception as e:
        logger.error(f"[DOWNLOAD_INFO] Failed to get download info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get download info: {e}"
        )


@router.get("/{download_id}/status",
           response_model=DownloadStatus,
           summary="Get Download Status",
           description="Get current status of a download")
async def get_download_status(
    download_id: str,
    download_service: DownloadService = Depends(get_download_service)
) -> DownloadStatus:
    """
    Get current download status.
    
    Args:
        download_id: Download ID
        download_service: Download service dependency
    
    Returns:
        DownloadStatus: Current download status
    """
    try:
        logger.debug(f"[DOWNLOAD_STATUS] Getting status for download: {download_id}")
        
        status_info = await download_service.get_download_status(download_id)
        
        if not status_info:
            raise NotFoundError(f"Download '{download_id}' not found")
        
        return DownloadStatus(**status_info)
        
    except NotFoundError:
        logger.warning(f"[DOWNLOAD_STATUS] Download not found: {download_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Download '{download_id}' not found"
        )
    
    except Exception as e:
        logger.error(f"[DOWNLOAD_STATUS] Failed to get download status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get download status: {e}"
        )


@router.delete("/{download_id}",
              response_model=SuccessResponse,
              summary="Cancel Download",
              description="Cancel an active download")
async def cancel_download(
    download_id: str,
    delete_partial: bool = Query(False, description="Delete partially downloaded files"),
    download_service: DownloadService = Depends(get_download_service)
) -> SuccessResponse:
    """
    Cancel an active download.
    
    Args:
        download_id: Download ID to cancel
        delete_partial: Whether to delete partially downloaded files
        download_service: Download service dependency
    
    Returns:
        SuccessResponse: Cancellation status
    """
    try:
        logger.info(f"[CANCEL_DOWNLOAD] Canceling download: {download_id}")
        
        result = await download_service.cancel_download(download_id, delete_partial=delete_partial)
        
        if not result:
            raise NotFoundError(f"Download '{download_id}' not found or cannot be canceled")
        
        return SuccessResponse(
            success=True,
            message=f"Download '{download_id}' canceled successfully",
            timestamp=datetime.now()
        )
        
    except NotFoundError:
        logger.warning(f"[CANCEL_DOWNLOAD] Download not found: {download_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Download '{download_id}' not found or cannot be canceled"
        )
    
    except Exception as e:
        logger.error(f"[CANCEL_DOWNLOAD] Failed to cancel download: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel download: {e}"
        )


@router.get("/files/",
           response_model=List[FileInfo],
           summary="List Downloaded Files",
           description="Get list of all downloaded files")
async def list_files(
    file_type: Optional[str] = Query(None, description="Filter by file type (model, audio, etc.)"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of files to return"),
    download_service: DownloadService = Depends(get_download_service)
) -> List[FileInfo]:
    """
    Get list of downloaded files.
    
    Args:
        file_type: Filter by file type
        limit: Maximum number of files to return
        download_service: Download service dependency
    
    Returns:
        List[FileInfo]: List of file information
    """
    try:
        logger.debug(f"[FILES] Listing files - Type: {file_type}, Limit: {limit}")
        
        files = await download_service.list_files(file_type_filter=file_type, limit=limit)
        
        return [FileInfo(**file_info) for file_info in files]
        
    except Exception as e:
        logger.error(f"[FILES] Failed to list files: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list files: {e}"
        )


@router.get("/files/{filename}",
           summary="Download File",
           description="Download a specific file")
async def download_file(
    filename: str,
    download_service: DownloadService = Depends(get_download_service)
) -> FileResponse:
    """
    Download a specific file.
    
    Args:
        filename: Name of the file to download
        download_service: Download service dependency
    
    Returns:
        FileResponse: File download response
    """
    try:
        logger.debug(f"[DOWNLOAD_FILE] Serving file: {filename}")
        
        file_path = await download_service.get_file_path(filename)
        
        if not file_path or not os.path.exists(file_path):
            raise NotFoundError(f"File '{filename}' not found")
        
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = 'application/octet-stream'
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=mime_type
        )
        
    except NotFoundError:
        logger.warning(f"[DOWNLOAD_FILE] File not found: {filename}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found"
        )
    
    except Exception as e:
        logger.error(f"[DOWNLOAD_FILE] Failed to serve file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to serve file: {e}"
        )


@router.delete("/files/{filename}",
              response_model=SuccessResponse,
              summary="Delete File",
              description="Delete a downloaded file")
async def delete_file(
    filename: str,
    download_service: DownloadService = Depends(get_download_service)
) -> SuccessResponse:
    """
    Delete a downloaded file.
    
    Args:
        filename: Name of the file to delete
        download_service: Download service dependency
    
    Returns:
        SuccessResponse: Deletion status
    """
    try:
        logger.info(f"[DELETE_FILE] Deleting file: {filename}")
        
        result = await download_service.delete_file(filename)
        
        if not result:
            raise NotFoundError(f"File '{filename}' not found")
        
        return SuccessResponse(
            success=True,
            message=f"File '{filename}' deleted successfully",
            timestamp=datetime.now()
        )
        
    except NotFoundError:
        logger.warning(f"[DELETE_FILE] File not found: {filename}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found"
        )
    
    except Exception as e:
        logger.error(f"[DELETE_FILE] Failed to delete file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {e}"
        )


# Background task functions

async def _monitor_download(download_service: DownloadService, download_id: str):
    """Monitor download progress in background."""
    try:
        logger.info(f"[MONITOR] Starting download monitoring for: {download_id}")
        
        # Periodic status checking would be implemented here
        # This is a placeholder for download monitoring logic
        
        import asyncio
        while True:
            status_info = await download_service.get_download_status(download_id)
            if not status_info:
                break
            
            status = status_info.get("status")
            if status in ["completed", "failed", "canceled"]:
                logger.info(f"[MONITOR] Download {download_id} finished with status: {status}")
                break
            
            await asyncio.sleep(5)  # Check every 5 seconds
        
    except Exception as e:
        logger.error(f"[MONITOR] Download monitoring failed for {download_id}: {e}")

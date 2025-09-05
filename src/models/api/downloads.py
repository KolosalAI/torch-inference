"""
Download API models for model downloads and file management.
"""

from typing import Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl


class ModelDownloadRequest(BaseModel):
    """Request model for downloading a model."""
    url: HttpUrl = Field(..., description="URL to download the model from")
    model_name: str = Field(..., description="Name to assign to the downloaded model")
    force_download: bool = Field(default=False, description="Force download even if file exists")
    verify_checksum: bool = Field(default=True, description="Verify file checksum after download")
    expected_checksum: Optional[str] = Field(None, description="Expected file checksum")


class ModelDownloadResponse(BaseModel):
    """Response model for model download initiation."""
    download_id: str = Field(..., description="Unique download identifier")
    model_name: str = Field(..., description="Name of the model being downloaded")
    url: str = Field(..., description="Download URL")
    status: str = Field(..., description="Initial download status")
    estimated_size: Optional[int] = Field(None, description="Estimated file size in bytes")
    created_at: datetime = Field(default_factory=datetime.now, description="Download creation time")


class DownloadStatus(BaseModel):
    """Download status information."""
    download_id: str = Field(..., description="Download identifier")
    status: str = Field(..., description="Current status (pending, downloading, completed, failed)")
    progress: Optional[float] = Field(None, description="Download progress (0.0 to 1.0)")
    downloaded_bytes: Optional[int] = Field(None, description="Number of bytes downloaded")
    total_bytes: Optional[int] = Field(None, description="Total file size in bytes")
    download_speed: Optional[float] = Field(None, description="Download speed in bytes per second")
    eta_seconds: Optional[float] = Field(None, description="Estimated time to completion in seconds")
    error_message: Optional[str] = Field(None, description="Error message if download failed")
    started_at: Optional[datetime] = Field(None, description="Download start time")
    completed_at: Optional[datetime] = Field(None, description="Download completion time")

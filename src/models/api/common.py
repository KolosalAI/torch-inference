"""
Common API response models and utilities.
"""

from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class APIResponse(BaseModel):
    """Standard API response model."""
    success: bool = Field(..., description="Whether the operation was successful")
    data: Optional[Any] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    error: Optional[str] = Field(None, description="Error message if operation failed")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Response timestamp")


class PaginatedResponse(APIResponse):
    """Paginated API response model."""
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")


class HealthResponse(BaseModel):
    """Health check response model."""
    healthy: bool = Field(..., description="Overall health status")
    checks: Dict[str, Any] = Field(..., description="Individual component health checks")
    timestamp: float = Field(..., description="Timestamp of health check")
    version: Optional[str] = Field(None, description="Application version")
    uptime: Optional[float] = Field(None, description="Application uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(False, description="Always false for error responses")
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Error timestamp")


class StatusResponse(BaseModel):
    """Status response model."""
    status: str = Field(..., description="Current status")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Status timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional status details")


# Common field types
class ModelInfo(BaseModel):
    """Model information schema."""
    name: str = Field(..., description="Model name")
    version: Optional[str] = Field(None, description="Model version")
    type: str = Field(..., description="Model type")
    device: str = Field(..., description="Device model is loaded on")
    memory_usage: Optional[float] = Field(None, description="Memory usage in MB")
    loaded_at: Optional[str] = Field(None, description="Timestamp when model was loaded")


class DeviceInfo(BaseModel):
    """Device information schema."""
    type: str = Field(..., description="Device type (cuda, mps, cpu)")
    id: Optional[int] = Field(None, description="Device ID for CUDA devices")
    name: str = Field(..., description="Device name")
    memory_total: Optional[float] = Field(None, description="Total memory in GB")
    memory_available: Optional[float] = Field(None, description="Available memory in GB")


class PerformanceMetrics(BaseModel):
    """Performance metrics schema."""
    requests_processed: int = Field(..., description="Total requests processed")
    average_latency: float = Field(..., description="Average latency in seconds")
    throughput: float = Field(..., description="Requests per second")
    error_rate: float = Field(..., description="Error rate as percentage")
    uptime: float = Field(..., description="Uptime in seconds")
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Last update timestamp")

"""
Common API models and base classes.
"""

from typing import Any, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """Base response model for all API responses."""
    success: bool = Field(..., description="Whether the request was successful")
    timestamp: Optional[str] = Field(None, description="Response timestamp")
    message: Optional[str] = Field(None, description="Optional message")


class SuccessResponse(BaseResponse):
    """Success response model."""
    success: bool = Field(default=True, description="Always true for success responses")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = Field(default=False, description="Always false for error responses")
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class HealthCheckResponse(BaseResponse):
    """Health check response model."""
    healthy: bool = Field(..., description="Whether the service is healthy")
    checks: Dict[str, Any] = Field(..., description="Individual health check results")
    timestamp: float = Field(..., description="Health check timestamp")
    message: str = Field(..., description="Health check message")


class ConfigResponse(BaseResponse):
    """Configuration response model."""
    configuration: Dict[str, Any] = Field(..., description="Configuration data")
    environment: str = Field(..., description="Current environment")
    

class StatsResponse(BaseResponse):
    """Statistics response model."""
    stats: Dict[str, Any] = Field(..., description="Service statistics")
    performance_report: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")

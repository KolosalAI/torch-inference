"""
Inference API models.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Request model for model-specific inference with inflight batching support."""
    inputs: Union[Any, List[Any]] = Field(..., description="Input data for inference (single item or list for batch)")
    priority: int = Field(default=0, description="Request priority (higher = processed first)")
    timeout: Optional[float] = Field(default=None, description="Request timeout in seconds")
    enable_batching: bool = Field(default=True, description="Enable inflight batching optimization")


class BatchInferenceRequest(BaseModel):
    """Request model for explicit batch inference."""
    inputs: List[Any] = Field(..., description="List of inputs for batch processing")
    priority: int = Field(default=0, description="Request priority (higher = processed first)")
    timeout: Optional[float] = Field(default=None, description="Request timeout in seconds")
    batch_size: Optional[int] = Field(default=None, description="Preferred batch size")


class InferenceResponse(BaseModel):
    """Response model for inference."""
    success: bool
    result: Union[Any, List[Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    model_info: Optional[Dict[str, Any]] = None
    batch_info: Optional[Dict[str, Any]] = None


class PredictionMetrics(BaseModel):
    """Model prediction performance metrics."""
    total_requests: int = Field(default=0, description="Total prediction requests processed")
    successful_requests: int = Field(default=0, description="Number of successful requests")
    failed_requests: int = Field(default=0, description="Number of failed requests")
    average_latency_ms: float = Field(default=0.0, description="Average prediction latency in milliseconds")
    throughput_rps: float = Field(default=0.0, description="Requests per second throughput")


class ModelPerformanceReport(BaseModel):
    """Comprehensive model performance report."""
    model_name: str = Field(..., description="Name of the model")
    metrics: PredictionMetrics = Field(..., description="Performance metrics")
    hardware_info: Dict[str, Any] = Field(..., description="Hardware utilization info")
    optimization_status: Dict[str, Any] = Field(..., description="Applied optimizations")
    timestamp: str = Field(..., description="Report generation timestamp")


class BatchInferenceResponse(BaseModel):
    """Response model for batch inference requests."""
    predictions: List[Any] = Field(..., description="List of predictions for each input")
    processing_time: float = Field(..., description="Total processing time in seconds") 
    total_processing_time: float = Field(..., description="Total processing time for all items")
    individual_times: Optional[List[float]] = Field(None, description="Processing time for each individual item")
    model_name: str = Field(..., description="Name of the model used for inference")
    batch_size: int = Field(..., description="Number of items in the batch")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional response metadata")

"""
Domain model for inference operations.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class InferenceStatus(Enum):
    """Inference operation status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(Enum):
    """Request priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class InferenceRequest:
    """Domain model for inference requests."""
    id: str
    inputs: Any
    model_name: str
    priority: Priority = Priority.NORMAL
    timeout: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class InferenceResult:
    """Domain model for inference results."""
    request_id: str
    outputs: Any
    status: InferenceStatus
    processing_time: float
    model_info: Dict[str, Any]
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.now()


@dataclass
class BatchInferenceRequest:
    """Domain model for batch inference requests."""
    id: str
    requests: List[InferenceRequest]
    batch_size: int
    parallel_processing: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class BatchInferenceResult:
    """Domain model for batch inference results."""
    batch_id: str
    results: List[InferenceResult]
    total_processing_time: float
    successful_count: int
    failed_count: int
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.now()


@dataclass
class InferenceMetrics:
    """Domain model for inference performance metrics."""
    requests_processed: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    max_processing_time: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0
    throughput: float = 0.0  # requests per second
    
    def add_request(self, processing_time: float, success: bool):
        """Add a request to the metrics."""
        self.requests_processed += 1
        self.total_processing_time += processing_time
        self.average_processing_time = self.total_processing_time / self.requests_processed
        self.min_processing_time = min(self.min_processing_time, processing_time)
        self.max_processing_time = max(self.max_processing_time, processing_time)
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.error_rate = (self.failed_requests / self.requests_processed) * 100
    
    def calculate_throughput(self, time_window_seconds: float):
        """Calculate throughput over a time window."""
        if time_window_seconds > 0:
            self.throughput = self.requests_processed / time_window_seconds
        else:
            self.throughput = 0.0

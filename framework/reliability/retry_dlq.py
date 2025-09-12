"""
Retry Mechanisms and Dead Letter Queue System

Provides:
- Exponential backoff for transient failures
- Dead letter queues for failed requests
- Retry policy configuration
- Failure analysis and recovery
"""

import asyncio
import logging
import time
import json
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import random
import threading
from collections import deque
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RetryPolicy:
    """Retry policy configuration."""
    strategy: str = "exponential_backoff"  # "exponential_backoff", "fixed_delay", "linear_backoff", "immediate"
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0  # Changed from backoff_multiplier for test compatibility
    jitter: bool = True
    
    # For compatibility with enum values
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE = "immediate"
    
    @property
    def backoff_multiplier(self) -> float:
        """Alias for exponential_base for backward compatibility."""
        return self.exponential_base
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if attempt == 0:
            return self.base_delay
        
        if self.strategy == "immediate":
            return 0.0
        elif self.strategy == "fixed_delay":
            delay = self.base_delay
        elif self.strategy == "linear_backoff":
            delay = self.base_delay * (attempt + 1)
        elif self.strategy == "exponential_backoff":
            delay = self.base_delay * (self.exponential_base ** attempt)
        else:
            delay = self.base_delay
        
        # Apply max delay limit
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            jitter_amount = delay * 0.5  # Up to 50% jitter
            delay += random.uniform(0, jitter_amount)
        
        return max(0, delay)
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if operation should be retried."""
        # Check if we've reached max attempts
        if attempt >= self.max_attempts:
            return False
        
        # Check if error is retryable
        retryable_errors = {
            ConnectionError,
            TimeoutError,
            OSError
        }
        
        non_retryable_errors = {
            ValueError,
            TypeError,
            KeyError
        }
        
        error_type = type(error)
        
        # Check if explicitly non-retryable
        if error_type in non_retryable_errors:
            return False
        
        # Check if explicitly retryable
        if error_type in retryable_errors:
            return True
        
        # Default to not retryable
        return False


class FailureReason(Enum):
    """Failure reason categories."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    UNKNOWN = "unknown"


class FailureClassification:
    """Classification system for failures to determine retry strategies."""
    
    def __init__(self):
        self._retryable_errors = {
            TimeoutError,
            ConnectionError,
            OSError
        }
        self._non_retryable_errors = {
            ValueError,
            TypeError,
            KeyError
        }
    
    def is_retryable(self, error: Exception) -> bool:
        """Determine if an error is retryable."""
        error_type = type(error)
        
        # Check if explicitly non-retryable
        if error_type in self._non_retryable_errors:
            return False
        
        # Check if explicitly retryable
        if error_type in self._retryable_errors:
            return True
        
        # HTTP status code based classification
        if hasattr(error, 'status_code'):
            status_code = error.status_code
            if 500 <= status_code < 600:  # Server errors
                return True
            if status_code == 429:  # Rate limit
                return True
            if 400 <= status_code < 500:  # Client errors (except rate limit)
                return False
        
        # Default to not retryable
        return False
    
    def classify_failure(self, error: Exception) -> FailureReason:
        """Classify the type of failure."""
        if isinstance(error, TimeoutError):
            return FailureReason.TIMEOUT
        elif isinstance(error, ConnectionError):
            return FailureReason.CONNECTION_ERROR
        elif hasattr(error, 'status_code'):
            if error.status_code == 429:
                return FailureReason.RATE_LIMIT
            elif 500 <= error.status_code < 600:
                return FailureReason.SERVER_ERROR
            elif 400 <= error.status_code < 500:
                return FailureReason.CLIENT_ERROR
        
        return FailureReason.UNKNOWN


@dataclass  
class ModelInferenceOperation:
    """Represents a model inference operation for retry/DLQ handling."""
    
    operation_id: str
    model_name: str
    input_data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    attempts: int = 0
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation_id': self.operation_id,
            'model_name': self.model_name,
            'input_data': self.input_data,
            'created_at': self.created_at.isoformat(),
            'attempts': self.attempts,
            'last_error': self.last_error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInferenceOperation':
        """Create from dictionary."""
        return cls(
            operation_id=data['operation_id'],
            model_name=data['model_name'],
            input_data=data['input_data'],
            created_at=datetime.fromisoformat(data['created_at']),
            attempts=data.get('attempts', 0),
            last_error=data.get('last_error')
        )


@dataclass
class RetryConfig:
    """Retry configuration."""
    policy: str = "exponential_backoff"  # Use string instead of enum
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_on_exceptions: List[type] = field(default_factory=lambda: [Exception])
    retry_on_status_codes: List[int] = field(default_factory=lambda: [500, 502, 503, 504])


@dataclass
class FailedRequest:
    """Failed request information."""
    id: str
    original_data: Any
    failure_reason: FailureReason
    error_message: str
    stack_trace: str
    attempts: int
    first_attempt_at: datetime
    last_attempt_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryAttempt:
    """Retry attempt information."""
    attempt_number: int
    timestamp: datetime
    delay_seconds: float
    error: Optional[str] = None
    success: bool = False


class RetryableOperation(ABC):
    """Abstract base class for retryable operations."""
    
    @abstractmethod
    async def execute(self, data: Any) -> Any:
        """Execute the operation."""
        pass
    
    @abstractmethod
    def should_retry(self, exception: Exception) -> bool:
        """Determine if operation should be retried based on exception."""
        pass
    
    @abstractmethod
    def classify_failure(self, exception: Exception) -> FailureReason:
        """Classify the failure reason."""
        pass


class ModelInferenceOperation(RetryableOperation):
    """Model inference retryable operation."""
    
    def __init__(self, model_callable: Callable, retry_config: RetryConfig):
        self.model_callable = model_callable
        self.retry_config = retry_config
    
    async def execute(self, data: Any) -> Any:
        """Execute model inference."""
        if asyncio.iscoroutinefunction(self.model_callable):
            return await self.model_callable(data)
        else:
            # Run in thread pool for sync functions
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.model_callable, data)
    
    def should_retry(self, exception: Exception) -> bool:
        """Determine if inference should be retried."""
        # Check if exception type is in retry list
        for retry_exception_type in self.retry_config.retry_on_exceptions:
            if isinstance(exception, retry_exception_type):
                return True
        
        # Check specific conditions
        if isinstance(exception, (TimeoutError, ConnectionError)):
            return True
        
        if isinstance(exception, OSError) and "CUDA" in str(exception):
            return True  # CUDA memory errors might be transient
        
        return False
    
    def classify_failure(self, exception: Exception) -> FailureReason:
        """Classify the failure reason."""
        if isinstance(exception, TimeoutError):
            return FailureReason.TIMEOUT
        elif isinstance(exception, ConnectionError):
            return FailureReason.CONNECTION_ERROR
        elif "CUDA out of memory" in str(exception):
            return FailureReason.RESOURCE_EXHAUSTED
        elif "rate limit" in str(exception).lower():
            return FailureReason.RATE_LIMIT
        else:
            return FailureReason.UNKNOWN


class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, policy_or_config):
        """Initialize with either RetryPolicy or RetryConfig."""
        if isinstance(policy_or_config, RetryPolicy):
            self.policy = policy_or_config
            # Convert to RetryConfig for internal use
            self.config = RetryConfig(
                policy=policy_or_config.strategy,
                max_attempts=policy_or_config.max_attempts,
                initial_delay=policy_or_config.base_delay,
                max_delay=policy_or_config.max_delay,
                backoff_multiplier=policy_or_config.exponential_base,
                jitter=policy_or_config.jitter
            )
        else:
            self.config = policy_or_config
            # Create RetryPolicy for compatibility
            self.policy = RetryPolicy(
                strategy=policy_or_config.policy,
                max_attempts=policy_or_config.max_attempts,
                base_delay=policy_or_config.initial_delay,
                max_delay=policy_or_config.max_delay,
                exponential_base=policy_or_config.backoff_multiplier,
                jitter=policy_or_config.jitter
            )
        
        self._attempt_history: Dict[str, List[RetryAttempt]] = {}
        self._lock = threading.RLock()
    
    async def execute_with_retry(self, operation, data: Any = None, 
                               request_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute operation with retry logic.
        
        Args:
            operation: Either a RetryableOperation instance or a callable function
            data: Data to pass to the operation (optional for direct callables)
            request_id: Optional request ID for tracking
            context: Optional context for the operation (not used but accepted for compatibility)
        """
        if request_id is None:
            import uuid
            request_id = f"req_{uuid.uuid4().hex[:12]}"
        
        attempts = []
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            delay = self._calculate_delay(attempt)
            
            if attempt > 1:
                logger.info(f"Retrying request {request_id}, attempt {attempt}/{self.config.max_attempts} after {delay}s delay")
                await asyncio.sleep(delay)
            
            attempt_info = RetryAttempt(
                attempt_number=attempt,
                timestamp=datetime.utcnow(),
                delay_seconds=delay
            )
            
            try:
                # Handle both RetryableOperation and direct callable
                if isinstance(operation, RetryableOperation):
                    result = await operation.execute(data)
                else:
                    # Direct callable
                    if asyncio.iscoroutinefunction(operation):
                        if data is not None:
                            result = await operation(data)
                        else:
                            result = await operation()
                    else:
                        # Sync function
                        if data is not None:
                            result = operation(data)
                        else:
                            result = operation()
                
                attempt_info.success = True
                
                with self._lock:
                    attempts.append(attempt_info)
                    self._attempt_history[request_id] = attempts
                
                if attempt > 1:
                    logger.info(f"Request {request_id} succeeded on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                attempt_info.error = str(e)
                attempts.append(attempt_info)
                
                logger.warning(f"Request {request_id} failed on attempt {attempt}: {e}")
                
                # Check if we should retry
                should_retry = False
                if isinstance(operation, RetryableOperation):
                    should_retry = operation.should_retry(e)
                else:
                    # Use policy's should_retry method
                    should_retry = self.policy.should_retry(e, attempt)
                
                if attempt == self.config.max_attempts or not should_retry:
                    break
        
        # All retries exhausted
        with self._lock:
            self._attempt_history[request_id] = attempts
        
        logger.error(f"Request {request_id} failed after {len(attempts)} attempts")
        raise last_exception
    
    def _calculate_delay(self, attempt_number: int) -> float:
        """Calculate delay for retry attempt."""
        if attempt_number == 1:
            return 0.0
        
        policy = self.config.policy if isinstance(self.config.policy, str) else self.config.policy.strategy
        
        if policy == "immediate":
            return 0.0
        
        elif policy == "fixed_delay":
            delay = self.config.initial_delay
        
        elif policy == "linear_backoff":
            delay = self.config.initial_delay * (attempt_number - 1)
        
        elif policy == "exponential_backoff":
            delay = self.config.initial_delay * (self.config.backoff_multiplier ** (attempt_number - 2))
        
        else:
            delay = self.config.initial_delay
        
        # Apply max delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def get_attempt_history(self, request_id: str) -> List[RetryAttempt]:
        """Get retry attempt history for a request."""
        with self._lock:
            return self._attempt_history.get(request_id, []).copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        with self._lock:
            total_requests = len(self._attempt_history)
            successful_requests = sum(
                1 for attempts in self._attempt_history.values()
                if any(attempt.success for attempt in attempts)
            )
            
            failed_requests = total_requests - successful_requests
            
            total_attempts = sum(
                len(attempts) for attempts in self._attempt_history.values()
            )
            
            # Calculate total retries (attempts beyond the first)
            total_retries = sum(
                max(0, len(attempts) - 1) for attempts in self._attempt_history.values()
            )
            
            avg_attempts = total_attempts / total_requests if total_requests > 0 else 0
            
            policy_str = self.config.policy if isinstance(self.config.policy, str) else self.config.policy.strategy
            
            return {
                'total_operations': total_requests,
                'total_requests': total_requests,  # Keep for compatibility
                'successful_operations': successful_requests,
                'successful_requests': successful_requests,  # Keep for compatibility
                'failed_operations': failed_requests,
                'failed_requests': failed_requests,  # Keep for compatibility
                'total_retries': total_retries,
                'success_rate': (successful_requests / total_requests * 100) if total_requests > 0 else 0,
                'total_attempts': total_attempts,
                'average_attempts_per_request': avg_attempts,
                'operations_by_status': {
                    'successful': successful_requests,
                    'failed': failed_requests
                },
                'config': {
                    'policy': policy_str,
                    'max_attempts': self.config.max_attempts,
                    'initial_delay': self.config.initial_delay,
                    'max_delay': self.config.max_delay
                }
            }
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """Alias for get_stats for test compatibility."""
        return self.get_stats()


class DeadLetterQueue:
    """Dead letter queue for failed requests."""
    
    def __init__(self, dlq_path_or_max_size=10000, persistence_path: Optional[str] = None):
        # Handle both old and new constructors
        if isinstance(dlq_path_or_max_size, str):
            # New constructor: DeadLetterQueue(dlq_path)
            self.dlq_path = Path(dlq_path_or_max_size)
            self.dlq_path.mkdir(parents=True, exist_ok=True)
            self.max_size = 10000
            self.persistence_path = None
        else:
            # Old constructor: DeadLetterQueue(max_size, persistence_path)
            self.max_size = dlq_path_or_max_size
            self.persistence_path = persistence_path
            self.dlq_path = None
        
        self._failed_requests: deque[FailedRequest] = deque(maxlen=self.max_size)
        self._failure_stats: Dict[FailureReason, int] = {reason: 0 for reason in FailureReason}
        self._lock = threading.RLock()
        
        # Load persisted failures if path provided
        if self.persistence_path:
            self._load_from_disk()
        
        logger.info(f"Dead letter queue initialized with max size: {self.max_size}")
    
    async def add_failed_operation(self, operation_id: str, operation_type: str,
                                  operation_data: Dict[str, Any], error: Exception,
                                  attempts: int):
        """Add a failed operation to the dead letter queue (test-compatible interface)."""
        # Classify the failure
        if isinstance(error, ConnectionError):
            failure_reason = FailureReason.CONNECTION_ERROR
        elif isinstance(error, TimeoutError):
            failure_reason = FailureReason.TIMEOUT
        elif isinstance(error, ValueError):
            failure_reason = FailureReason.CLIENT_ERROR
        else:
            failure_reason = FailureReason.UNKNOWN
        
        failed_request = FailedRequest(
            id=operation_id,
            original_data=operation_data,
            failure_reason=failure_reason,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            attempts=attempts,
            first_attempt_at=datetime.utcnow(),
            last_attempt_at=datetime.utcnow(),
            metadata={"operation_type": operation_type}
        )
        
        # Add to in-memory queue
        with self._lock:
            self._failed_requests.append(failed_request)
            self._failure_stats[failed_request.failure_reason] += 1
        
        # Save individual file if dlq_path is set
        if self.dlq_path:
            filename = f"{operation_id}_{int(time.time())}.json"
            file_path = self.dlq_path / filename
            
            data = {
                "operation_id": operation_id,
                "operation_type": operation_type,
                "operation_data": operation_data,
                "error_message": str(error),
                "error_type": type(error).__name__,
                "attempts": attempts,
                "timestamp": datetime.utcnow().isoformat(),
                "failure_reason": failure_reason.value
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        # Persist to single file if persistence_path is set
        if self.persistence_path:
            self._persist_to_disk()
        
        logger.warning(f"Added failed operation to DLQ: {operation_id} - {failure_reason.value}")
    
    def add_failed_request(self, failed_request: FailedRequest):
        """Add a failed request to the dead letter queue."""
        with self._lock:
            self._failed_requests.append(failed_request)
            self._failure_stats[failed_request.failure_reason] += 1
        
        # Persist to disk if configured
        if self.persistence_path:
            self._persist_to_disk()
        
        logger.warning(f"Added failed request to DLQ: {failed_request.id} - {failed_request.failure_reason.value}")
    
    def get_failed_operations(self, operation_type: Optional[str] = None, 
                            limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get failed operations from the queue (test-compatible interface)."""
        # Get all failed requests
        with self._lock:
            requests = list(self._failed_requests)
        
        # Filter by operation type if specified
        if operation_type:
            requests = [req for req in requests if req.metadata.get("operation_type") == operation_type]
        
        # Apply limit if specified
        if limit:
            requests = requests[-limit:]  # Get most recent
        
        # Convert to dictionary format for test compatibility
        result = []
        for req in requests:
            result.append({
                "operation_id": req.id,
                "operation_type": req.metadata.get("operation_type", "unknown"),
                "operation_data": req.original_data,
                "error_message": req.error_message,
                "error_type": req.error_message.split(":")[0] if ":" in req.error_message else "Exception",
                "attempts": req.attempts,
                "timestamp": req.last_attempt_at.isoformat(),
                "failure_reason": req.failure_reason.value
            })
        
        return result
    
    def get_failed_requests(self, limit: Optional[int] = None, 
                          failure_reason: Optional[FailureReason] = None) -> List[FailedRequest]:
        """Get failed requests from the queue."""
        with self._lock:
            requests = list(self._failed_requests)
        
        # Filter by failure reason if specified
        if failure_reason:
            requests = [req for req in requests if req.failure_reason == failure_reason]
        
        # Apply limit
        if limit:
            requests = requests[-limit:]  # Get most recent
        
        return requests
    
    def get_failed_operations_by_type(self, failure_type: str) -> List[FailedRequest]:
        """Get failed requests by failure type."""
        try:
            failure_reason = FailureReason(failure_type)
            return self.get_failed_requests(failure_reason=failure_reason)
        except ValueError:
            return []
    
    def retry_failed_request(self, request_id: str) -> Optional[FailedRequest]:
        """Remove and return a failed request for retry."""
        with self._lock:
            for i, request in enumerate(self._failed_requests):
                if request.id == request_id:
                    # Remove from deque by converting to list, removing, and recreating
                    requests_list = list(self._failed_requests)
                    removed_request = requests_list.pop(i)
                    self._failed_requests.clear()
                    self._failed_requests.extend(requests_list)
                    return removed_request
        
        return None
    
    async def reprocess_operation(self, request_id: str, processor_func) -> bool:
        """Reprocess a failed operation from the DLQ."""
        # Find and remove the request
        failed_request = self.retry_failed_request(request_id)
        if not failed_request:
            return False
        
        try:
            # Process the operation data
            if asyncio.iscoroutinefunction(processor_func):
                await processor_func(failed_request.original_data)
            else:
                processor_func(failed_request.original_data)
            return True
        except Exception as e:
            # If reprocessing fails, add it back to the queue
            failed_request.attempts += 1
            failed_request.last_attempt_at = datetime.utcnow()
            failed_request.error_message = str(e)
            self.add_failed_request(failed_request)
            return False
    
    async def reprocess_nonexistent_operation(self, request_id: str, processor_func) -> bool:
        """Try to reprocess a non-existent operation (for test compatibility)."""
        return await self.reprocess_operation(request_id, processor_func)
    
    def clear_old_requests(self, older_than_days: int = 7) -> int:
        """Clear requests older than specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        
        with self._lock:
            original_count = len(self._failed_requests)
            
            # Filter out old requests
            self._failed_requests = deque(
                [req for req in self._failed_requests if req.last_attempt_at > cutoff_date],
                maxlen=self.max_size
            )
            
            cleared_count = original_count - len(self._failed_requests)
        
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} old failed requests from DLQ")
        
        return cleared_count
    
    def cleanup_old_entries(self, older_than_days: int = 7) -> int:
        """Clean up old entries from both memory and disk."""
        cleaned_count = 0
        
        # Clean up in-memory entries
        memory_cleaned = self.clear_old_requests(older_than_days)
        cleaned_count += memory_cleaned
        
        # Clean up file-based entries if dlq_path is set
        if self.dlq_path:
            cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
            
            for json_file in self.dlq_path.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Check if the timestamp is older than cutoff
                    timestamp_str = data.get('timestamp', '')
                    if timestamp_str:
                        file_timestamp = datetime.fromisoformat(timestamp_str)
                        if file_timestamp < cutoff_date:
                            json_file.unlink()  # Delete the file
                            cleaned_count += 1
                            logger.info(f"Cleaned up old DLQ file: {json_file.name}")
                
                except Exception as e:
                    logger.error(f"Error cleaning up DLQ file {json_file}: {e}")
        
        return cleaned_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dead letter queue statistics."""
        with self._lock:
            total_failed = len(self._failed_requests)
            
            # Recent failures (last 24 hours)
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            recent_failures = sum(
                1 for req in self._failed_requests
                if req.last_attempt_at > recent_cutoff
            )
            
            # Group operations by type
            operations_by_type = {}
            for req in self._failed_requests:
                op_type = req.metadata.get("operation_type", "unknown")
                operations_by_type[op_type] = operations_by_type.get(op_type, 0) + 1
            
            return {
                'total_failed_operations': total_failed,  # Changed for test compatibility
                'total_failed_requests': total_failed,    # Keep for compatibility
                'recent_failures_24h': recent_failures,
                'failure_by_reason': dict(self._failure_stats),
                'operations_by_type': operations_by_type,
                'oldest_failure': min(
                    (req.first_attempt_at for req in self._failed_requests),
                    default=None
                ),
                'newest_failure': max(
                    (req.last_attempt_at for req in self._failed_requests),
                    default=None
                ),
                'oldest_entry': min(
                    (req.first_attempt_at for req in self._failed_requests),
                    default=None
                ),
                'newest_entry': max(
                    (req.last_attempt_at for req in self._failed_requests),
                    default=None
                ),
                'max_size': self.max_size,
                'utilization_percent': (total_failed / self.max_size * 100) if self.max_size > 0 else 0
            }
    
    def get_dlq_stats(self) -> Dict[str, Any]:
        """Alias for get_stats for test compatibility."""
        return self.get_stats()
    
    def _persist_to_disk(self):
        """Persist failed requests to disk."""
        if not self.persistence_path:
            return
        
        try:
            with self._lock:
                data = {
                    'failed_requests': [
                        {
                            'id': req.id,
                            'original_data': req.original_data,
                            'failure_reason': req.failure_reason.value,
                            'error_message': req.error_message,
                            'stack_trace': req.stack_trace,
                            'attempts': req.attempts,
                            'first_attempt_at': req.first_attempt_at.isoformat(),
                            'last_attempt_at': req.last_attempt_at.isoformat(),
                            'metadata': req.metadata
                        }
                        for req in self._failed_requests
                    ],
                    'failure_stats': {reason.value: count for reason, count in self._failure_stats.items()}
                }
            
            with open(self.persistence_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to persist DLQ to disk: {e}")
    
    def _load_from_disk(self):
        """Load failed requests from disk."""
        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)
            
            with self._lock:
                for req_data in data.get('failed_requests', []):
                    failed_request = FailedRequest(
                        id=req_data['id'],
                        original_data=req_data['original_data'],
                        failure_reason=FailureReason(req_data['failure_reason']),
                        error_message=req_data['error_message'],
                        stack_trace=req_data['stack_trace'],
                        attempts=req_data['attempts'],
                        first_attempt_at=datetime.fromisoformat(req_data['first_attempt_at']),
                        last_attempt_at=datetime.fromisoformat(req_data['last_attempt_at']),
                        metadata=req_data.get('metadata', {})
                    )
                    self._failed_requests.append(failed_request)
                
                # Load failure stats
                for reason_str, count in data.get('failure_stats', {}).items():
                    reason = FailureReason(reason_str)
                    self._failure_stats[reason] = count
            
            logger.info(f"Loaded {len(self._failed_requests)} failed requests from disk")
            
        except FileNotFoundError:
            logger.info("No existing DLQ persistence file found")
        except Exception as e:
            logger.error(f"Failed to load DLQ from disk: {e}")


class RetryAndDLQManager:
    """
    Combined retry and dead letter queue manager.
    
    Features:
    - Exponential backoff retry logic
    - Dead letter queue for ultimately failed requests
    - Failure analysis and classification
    - Retry statistics and monitoring
    """
    
    def __init__(self, retry_config: RetryConfig, dlq_max_size: int = 10000, 
                 dlq_persistence_path: Optional[str] = None):
        self.retry_manager = RetryManager(retry_config)
        self.dead_letter_queue = DeadLetterQueue(dlq_max_size, dlq_persistence_path)
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("Retry and DLQ manager initialized")
    
    async def start(self):
        """Start background tasks."""
        if self._cleanup_task:
            return
        
        self._cleanup_task = asyncio.create_task(self._cleanup_old_requests())
        logger.info("Retry and DLQ manager started")
    
    async def stop(self):
        """Stop background tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        logger.info("Retry and DLQ manager stopped")
    
    async def execute_with_retry_and_dlq(self, operation: RetryableOperation, data: Any, 
                                       request_id: Optional[str] = None) -> Any:
        """Execute operation with retry logic and DLQ fallback."""
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000000)}"
        
        first_attempt_time = datetime.utcnow()
        
        try:
            return await self.retry_manager.execute_with_retry(operation, data, request_id)
        
        except Exception as final_exception:
            # All retries exhausted, add to dead letter queue
            attempts = len(self.retry_manager.get_attempt_history(request_id))
            
            failed_request = FailedRequest(
                id=request_id,
                original_data=data,
                failure_reason=operation.classify_failure(final_exception),
                error_message=str(final_exception),
                stack_trace=traceback.format_exc(),
                attempts=attempts,
                first_attempt_at=first_attempt_time,
                last_attempt_at=datetime.utcnow(),
                metadata={
                    'operation_type': type(operation).__name__,
                    'retry_config': {
                        'policy': self.retry_manager.config.policy.value,
                        'max_attempts': self.retry_manager.config.max_attempts
                    }
                }
            )
            
            self.dead_letter_queue.add_failed_request(failed_request)
            
            # Re-raise the exception
            raise
    
    async def retry_from_dlq(self, request_id: str, operation: RetryableOperation) -> bool:
        """Retry a request from the dead letter queue."""
        failed_request = self.dead_letter_queue.retry_failed_request(request_id)
        if not failed_request:
            logger.warning(f"Request {request_id} not found in DLQ")
            return False
        
        try:
            await self.execute_with_retry_and_dlq(operation, failed_request.original_data, request_id)
            logger.info(f"Successfully retried request {request_id} from DLQ")
            return True
        
        except Exception as e:
            logger.error(f"Failed to retry request {request_id} from DLQ: {e}")
            return False
    
    async def _cleanup_old_requests(self):
        """Cleanup old requests periodically."""
        try:
            while True:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old DLQ entries (older than 7 days)
                self.dead_letter_queue.clear_old_requests(older_than_days=7)
        
        except asyncio.CancelledError:
            pass
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'retry_stats': self.retry_manager.get_stats(),
            'dlq_stats': self.dead_letter_queue.get_stats(),
            'timestamp': datetime.utcnow().isoformat()
        }


# Global retry and DLQ manager
_retry_dlq_manager: Optional[RetryAndDLQManager] = None


def get_retry_dlq_manager() -> RetryAndDLQManager:
    """Get the global retry and DLQ manager."""
    global _retry_dlq_manager
    if _retry_dlq_manager is None:
        config = RetryConfig(
            policy=RetryPolicy.EXPONENTIAL_BACKOFF,
            max_attempts=3,
            initial_delay=1.0,
            max_delay=60.0,
            backoff_multiplier=2.0,
            jitter=True
        )
        _retry_dlq_manager = RetryAndDLQManager(config, dlq_persistence_path="dlq_persistence.json")
    return _retry_dlq_manager


def create_model_inference_operation(model_callable: Callable, retry_config: Optional[RetryConfig] = None) -> ModelInferenceOperation:
    """Create a model inference operation with retry capabilities."""
    if retry_config is None:
        retry_config = RetryConfig()
    
    return ModelInferenceOperation(model_callable, retry_config)

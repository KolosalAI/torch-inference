"""
Structured Logging System

Provides structured JSON logging with:
- Request correlation IDs
- Distributed tracing support (Jaeger/Zipkin)
- Log aggregation compatibility (ELK/EFK stack)
"""

import json
import logging
import time
import uuid
import threading
import contextvars
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import traceback
import sys
import os
from pathlib import Path

# Context variables for correlation tracking
correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('correlation_id', default=None)
user_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('user_id', default=None)
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('request_id', default=None)
trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('trace_id', default=None)
span_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('span_id', default=None)


class LogLevel(Enum):
    """Log levels with numeric values."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class StructuredLogRecord:
    """Structured log record with all relevant fields."""
    timestamp: str
    level: str
    logger: str
    message: str
    
    # Context information
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Tracing information
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    # Application information
    service_name: Optional[str] = None
    service_version: Optional[str] = None
    environment: Optional[str] = None
    
    # Request information
    method: Optional[str] = None
    url: Optional[str] = None
    user_agent: Optional[str] = None
    remote_ip: Optional[str] = None
    
    # Performance information
    duration_ms: Optional[float] = None
    response_status: Optional[int] = None
    
    # Additional structured data
    extra: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Thread and process information
    thread_id: Optional[int] = None
    thread_name: Optional[str] = None
    process_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, removing None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured logging.
    
    Converts log records to structured JSON format with
    correlation IDs and tracing information.
    """
    
    def __init__(self,
                 service_name: str = "torch-inference",
                 service_version: str = "1.0.0",
                 environment: str = "development",
                 include_extra: bool = True):
        super().__init__()
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Create structured log record
        structured_record = StructuredLogRecord(
            timestamp=datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            
            # Context from context variables
            correlation_id=correlation_id_var.get(),
            request_id=request_id_var.get(),
            user_id=user_id_var.get(),
            trace_id=trace_id_var.get(),
            span_id=span_id_var.get(),
            
            # Service information
            service_name=self.service_name,
            service_version=self.service_version,
            environment=self.environment,
            
            # Thread and process information
            thread_id=record.thread,
            thread_name=record.threadName,
            process_id=record.process,
        )
        
        # Add exception information if present
        if record.exc_info:
            exc_type, exc_value, exc_traceback = record.exc_info
            structured_record.exception_type = exc_type.__name__ if exc_type else None
            structured_record.exception_message = str(exc_value) if exc_value else None
            
            if exc_traceback:
                structured_record.stack_trace = ''.join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback)
                )
        
        # Add extra fields from record
        if self.include_extra:
            extra_fields = {}
            
            # Standard extra fields we want to capture
            for attr in ['method', 'url', 'user_agent', 'remote_ip', 
                        'duration_ms', 'response_status']:
                if hasattr(record, attr):
                    value = getattr(record, attr)
                    if value is not None:
                        setattr(structured_record, attr, value)
            
            # Other extra fields go into the extra dict
            for key, value in record.__dict__.items():
                if (key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                              'pathname', 'filename', 'module', 'lineno', 
                              'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process',
                              'getMessage', 'exc_info', 'exc_text', 'stack_info',
                              'method', 'url', 'user_agent', 'remote_ip',
                              'duration_ms', 'response_status'] and
                    not key.startswith('_')):
                    extra_fields[key] = value
            
            if extra_fields:
                structured_record.extra = extra_fields
        
        return structured_record.to_json()


class CorrelationIDFilter(logging.Filter):
    """
    Filter that adds correlation ID to log records.
    
    Automatically injects correlation ID from context variables
    or generates a new one if not present.
    """
    
    def __init__(self, auto_generate: bool = True):
        super().__init__()
        self.auto_generate = auto_generate
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record."""
        # Get or generate correlation ID
        correlation_id = correlation_id_var.get()
        
        if not correlation_id and self.auto_generate:
            correlation_id = str(uuid.uuid4())
            correlation_id_var.set(correlation_id)
        
        # Add to record for backwards compatibility
        record.correlation_id = correlation_id
        
        return True


class RequestLoggingFilter(logging.Filter):
    """
    Filter for HTTP request logging with performance metrics.
    """
    
    def __init__(self, capture_body: bool = False, max_body_size: int = 1024):
        super().__init__()
        self.capture_body = capture_body
        self.max_body_size = max_body_size
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Enhance record with request information."""
        # This would be populated by middleware
        # For now, just ensure the filter passes through
        return True


class PerformanceLoggingFilter(logging.Filter):
    """
    Filter for performance logging with timing information.
    """
    
    def __init__(self, slow_request_threshold: float = 1.0):
        super().__init__()
        self.slow_request_threshold = slow_request_threshold
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance flags to log record."""
        if hasattr(record, 'duration_ms'):
            duration_seconds = record.duration_ms / 1000.0
            record.slow_request = duration_seconds > self.slow_request_threshold
        
        return True


class StructuredLogger:
    """
    High-level structured logger with context management.
    
    Provides convenient methods for structured logging with
    automatic correlation ID tracking and context management.
    """
    
    def __init__(self, logger: logging.Logger):
        self._logger = logger
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with structured data."""
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method."""
        # Extract exception info if present
        exc_info = kwargs.pop('exc_info', None)
        
        # Create extra dict with remaining kwargs
        extra = {}
        for key, value in kwargs.items():
            if key not in ['correlation_id', 'request_id', 'user_id', 'trace_id', 'span_id']:
                extra[key] = value
        
        # Log with extra information
        self._logger.log(level.value, message, extra=extra, exc_info=exc_info)
    
    def log_request_start(self, method: str, url: str, **kwargs):
        """Log request start."""
        self.info(f"Request started: {method} {url}", 
                 method=method, url=url, event_type="request_start", **kwargs)
    
    def log_request_end(self, method: str, url: str, status: int, duration_ms: float, **kwargs):
        """Log request end."""
        self.info(f"Request completed: {method} {url} -> {status} ({duration_ms:.1f}ms)",
                 method=method, url=url, response_status=status, 
                 duration_ms=duration_ms, event_type="request_end", **kwargs)
    
    def log_model_inference(self, model_name: str, duration_ms: float, **kwargs):
        """Log model inference."""
        self.info(f"Model inference: {model_name} ({duration_ms:.1f}ms)",
                 model_name=model_name, duration_ms=duration_ms, 
                 event_type="model_inference", **kwargs)
    
    def log_error_with_context(self, message: str, error: Exception, **kwargs):
        """Log error with full context."""
        self.error(message, 
                  exception_type=type(error).__name__,
                  exception_message=str(error),
                  exc_info=True, **kwargs)


class TraceContext:
    """
    Context manager for distributed tracing.
    
    Manages trace and span IDs for distributed tracing systems.
    """
    
    def __init__(self, 
                 operation_name: str,
                 trace_id: Optional[str] = None,
                 parent_span_id: Optional[str] = None):
        self.operation_name = operation_name
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_id = str(uuid.uuid4())
        self.parent_span_id = parent_span_id
        self.start_time = time.time()
        
        self._trace_token = None
        self._span_token = None
    
    def __enter__(self):
        """Enter trace context."""
        self._trace_token = trace_id_var.set(self.trace_id)
        self._span_token = span_id_var.set(self.span_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit trace context."""
        if self._trace_token:
            trace_id_var.reset(self._trace_token)
        if self._span_token:
            span_id_var.reset(self._span_token)
    
    def get_duration_ms(self) -> float:
        """Get operation duration in milliseconds."""
        return (time.time() - self.start_time) * 1000


class LoggingConfig:
    """
    Configuration for structured logging system.
    """
    
    def __init__(self,
                 level: str = "INFO",
                 service_name: str = "torch-inference",
                 service_version: str = "1.0.0",
                 environment: str = "development",
                 log_file: Optional[str] = None,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_correlation_id: bool = True,
                 auto_generate_correlation_id: bool = True,
                 capture_request_body: bool = False,
                 slow_request_threshold: float = 1.0):
        
        self.level = level
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.log_file = log_file
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_correlation_id = enable_correlation_id
        self.auto_generate_correlation_id = auto_generate_correlation_id
        self.capture_request_body = capture_request_body
        self.slow_request_threshold = slow_request_threshold


def setup_structured_logging(config: LoggingConfig) -> Dict[str, logging.Logger]:
    """
    Set up structured logging system.
    
    Returns dictionary of configured loggers.
    """
    # Create formatters
    structured_formatter = StructuredFormatter(
        service_name=config.service_name,
        service_version=config.service_version,
        environment=config.environment
    )
    
    # Create filters
    filters = []
    if config.enable_correlation_id:
        filters.append(CorrelationIDFilter(config.auto_generate_correlation_id))
    
    filters.append(RequestLoggingFilter(config.capture_request_body))
    filters.append(PerformanceLoggingFilter(config.slow_request_threshold))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    handlers = []
    
    # Console handler
    if config.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(structured_formatter)
        for filter_obj in filters:
            console_handler.addFilter(filter_obj)
        handlers.append(console_handler)
    
    # File handler
    if config.log_file:
        # Ensure log directory exists
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            config.log_file,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(structured_formatter)
        for filter_obj in filters:
            file_handler.addFilter(filter_obj)
        handlers.append(file_handler)
    
    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Create specific loggers
    loggers = {
        'root': root_logger,
        'api': logging.getLogger('api'),
        'inference': logging.getLogger('inference'),
        'security': logging.getLogger('security'),
        'performance': logging.getLogger('performance'),
        'audit': logging.getLogger('audit')
    }
    
    return loggers


def get_structured_logger(name: str = None) -> StructuredLogger:
    """Get a structured logger instance."""
    logger = logging.getLogger(name) if name else logging.getLogger()
    return StructuredLogger(logger)


# Context managers for correlation tracking
class CorrelationContext:
    """Context manager for correlation ID tracking."""
    
    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self._token = None
    
    def __enter__(self):
        self._token = correlation_id_var.set(self.correlation_id)
        return self.correlation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token:
            correlation_id_var.reset(self._token)


class RequestContext:
    """Context manager for request tracking."""
    
    def __init__(self, 
                 request_id: Optional[str] = None,
                 user_id: Optional[str] = None,
                 correlation_id: Optional[str] = None):
        self.request_id = request_id or str(uuid.uuid4())
        self.user_id = user_id
        self.correlation_id = correlation_id or correlation_id_var.get() or str(uuid.uuid4())
        
        self._request_token = None
        self._user_token = None
        self._correlation_token = None
    
    def __enter__(self):
        self._request_token = request_id_var.set(self.request_id)
        self._user_token = user_id_var.set(self.user_id)
        if self.correlation_id:
            self._correlation_token = correlation_id_var.set(self.correlation_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._request_token:
            request_id_var.reset(self._request_token)
        if self._user_token:
            user_id_var.reset(self._user_token)
        if self._correlation_token:
            correlation_id_var.reset(self._correlation_token)


# Helper functions for context management
def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from context."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str):
    """Set correlation ID in context."""
    correlation_id_var.set(correlation_id)


def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id_var.get()


def get_user_id() -> Optional[str]:
    """Get current user ID from context."""
    return user_id_var.get()


def get_trace_id() -> Optional[str]:
    """Get current trace ID from context."""
    return trace_id_var.get()


def get_span_id() -> Optional[str]:
    """Get current span ID from context."""
    return span_id_var.get()


# Global logger instances
_structured_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str = None) -> StructuredLogger:
    """Get or create a structured logger."""
    logger_name = name or 'default'
    
    if logger_name not in _structured_loggers:
        _structured_loggers[logger_name] = get_structured_logger(name)
    
    return _structured_loggers[logger_name]


# Default configuration
DEFAULT_LOGGING_CONFIG = LoggingConfig(
    level="INFO",
    service_name="torch-inference",
    environment=os.getenv("ENVIRONMENT", "development"),
    log_file="logs/structured.log",
    enable_console=True,
    enable_correlation_id=True
)

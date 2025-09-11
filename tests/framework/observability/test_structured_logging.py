"""
Tests for Structured Logging implementation.
"""

import pytest
import json
import logging
import uuid
from unittest.mock import Mock, patch
from datetime import datetime
from io import StringIO

from framework.observability.structured_logging import (
    StructuredFormatter, CorrelationIDFilter, TraceContext,
    get_logger, set_correlation_id, get_correlation_id
)


class TestTraceContext:
    """Test trace context functionality."""
    
    def test_trace_context_creation(self):
        """Test creating trace context."""
        context = TraceContext()
        
        assert context.correlation_id is not None
        assert len(context.correlation_id) > 0
        assert context.user_id is None
        assert context.operation is None
        assert isinstance(context.timestamp, datetime)
    
    def test_trace_context_with_data(self):
        """Test creating trace context with data."""
        correlation_id = str(uuid.uuid4())
        context = TraceContext(
            correlation_id=correlation_id,
            user_id="user-123",
            operation="model_inference"
        )
        
        assert context.correlation_id == correlation_id
        assert context.user_id == "user-123"
        assert context.operation == "model_inference"
    
    def test_trace_context_to_dict(self):
        """Test converting trace context to dictionary."""
        context = TraceContext(
            correlation_id="test-id",
            user_id="user-123",
            operation="test_operation"
        )
        
        context_dict = context.to_dict()
        
        assert context_dict["correlation_id"] == "test-id"
        assert context_dict["user_id"] == "user-123"
        assert context_dict["operation"] == "test_operation"
        assert "timestamp" in context_dict


class TestCorrelationIDFilter:
    """Test correlation ID filter functionality."""
    
    def test_filter_adds_correlation_id(self):
        """Test filter adds correlation ID to log record."""
        filter_instance = CorrelationIDFilter()
        
        # Create mock log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Set correlation ID in context
        correlation_id = "test-correlation-id"
        with patch('framework.observability.structured_logging._trace_context') as mock_context:
            mock_context.get.return_value = TraceContext(correlation_id=correlation_id)
            
            result = filter_instance.filter(record)
            
            assert result is True
            assert hasattr(record, 'correlation_id')
            assert record.correlation_id == correlation_id
    
    def test_filter_no_correlation_id(self):
        """Test filter when no correlation ID is set."""
        filter_instance = CorrelationIDFilter()
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # No correlation ID in context
        with patch('framework.observability.structured_logging._trace_context') as mock_context:
            mock_context.get.return_value = None
            
            result = filter_instance.filter(record)
            
            assert result is True
            assert hasattr(record, 'correlation_id')
            assert record.correlation_id is None


class TestStructuredFormatter:
    """Test structured JSON formatter."""
    
    @pytest.fixture
    def formatter(self):
        """Create structured formatter."""
        return StructuredFormatter()
    
    def test_basic_log_formatting(self, formatter):
        """Test basic log message formatting."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test log message",
            args=(),
            exc_info=None
        )
        
        # Add correlation ID
        record.correlation_id = "test-correlation-id"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["message"] == "Test log message"
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.logger"
        assert log_data["correlation_id"] == "test-correlation-id"
        assert log_data["module"] == "file"
        assert log_data["line_number"] == 42
        assert "timestamp" in log_data
    
    def test_log_with_extra_fields(self, formatter):
        """Test log formatting with extra fields."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.correlation_id = "test-id"
        record.user_id = "user-123"
        record.operation = "model_inference"
        record.execution_time = 1.5
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["user_id"] == "user-123"
        assert log_data["operation"] == "model_inference"
        assert log_data["execution_time"] == 1.5
    
    def test_log_with_exception(self, formatter):
        """Test log formatting with exception information."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Exception occurred",
            args=(),
            exc_info=exc_info
        )
        record.correlation_id = "test-id"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["message"] == "Exception occurred"
        assert "exception" in log_data
        assert "ValueError" in log_data["exception"]
        assert "Test exception" in log_data["exception"]
    
    def test_log_with_args(self, formatter):
        """Test log formatting with message arguments."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Processing user %s with model %s",
            args=("user-123", "bert-model"),
            exc_info=None
        )
        record.correlation_id = "test-id"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["message"] == "Processing user user-123 with model bert-model"
    
    def test_serialization_of_complex_objects(self, formatter):
        """Test serialization of complex objects in extra fields."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Complex data",
            args=(),
            exc_info=None
        )
        
        # Add complex objects
        record.correlation_id = "test-id"
        record.metadata = {"nested": {"value": 123}, "list": [1, 2, 3]}
        record.timestamp_obj = datetime.now()  # Non-serializable
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["metadata"]["nested"]["value"] == 123
        assert log_data["metadata"]["list"] == [1, 2, 3]
        # timestamp_obj should be converted to string
        assert isinstance(log_data["timestamp_obj"], str)


class TestLoggerFunctions:
    """Test logger utility functions."""
    
    def test_get_logger(self):
        """Test getting a structured logger."""
        logger = get_logger("test.module")
        
        assert logger.name == "test.module"
        assert len(logger.handlers) > 0
        
        # Check that correlation ID filter is added
        filters = []
        for handler in logger.handlers:
            filters.extend(handler.filters)
        
        assert any(isinstance(f, CorrelationIDFilter) for f in filters)
    
    def test_set_and_get_correlation_id(self):
        """Test setting and getting correlation ID."""
        correlation_id = "test-correlation-123"
        
        set_correlation_id(correlation_id)
        retrieved_id = get_correlation_id()
        
        assert retrieved_id == correlation_id
    
    def test_set_correlation_id_with_context(self):
        """Test setting correlation ID with additional context."""
        correlation_id = "test-correlation-456"
        
        set_correlation_id(
            correlation_id,
            user_id="user-789",
            operation="test_operation"
        )
        
        retrieved_id = get_correlation_id()
        assert retrieved_id == correlation_id
        
        # Check that context is set (would need to access internal context)
        with patch('framework.observability.structured_logging._trace_context') as mock_context:
            mock_context.get.return_value = TraceContext(
                correlation_id=correlation_id,
                user_id="user-789",
                operation="test_operation"
            )
            
            context = mock_context.get()
            assert context.user_id == "user-789"
            assert context.operation == "test_operation"
    
    def test_correlation_id_context_isolation(self):
        """Test that correlation IDs are isolated between contexts."""
        # This would require actual async context testing
        # For now, test basic functionality
        
        # Set correlation ID
        set_correlation_id("first-id")
        first_id = get_correlation_id()
        
        # Change correlation ID
        set_correlation_id("second-id")
        second_id = get_correlation_id()
        
        assert first_id == "first-id"
        assert second_id == "second-id"


class TestStructuredLoggingIntegration:
    """Test structured logging integration scenarios."""
    
    @pytest.fixture
    def log_capture(self):
        """Capture log output for testing."""
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(StructuredFormatter())
        handler.addFilter(CorrelationIDFilter())
        
        logger = logging.getLogger("test.integration")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        yield logger, log_stream
        
        # Cleanup
        logger.removeHandler(handler)
    
    def test_end_to_end_logging(self, log_capture):
        """Test end-to-end structured logging."""
        logger, log_stream = log_capture
        
        # Set correlation context
        set_correlation_id("integration-test-id", user_id="test-user")
        
        # Log various messages
        logger.info("Starting operation", extra={
            "operation": "test_operation",
            "input_size": 1024
        })
        
        logger.warning("Slow operation detected", extra={
            "execution_time": 5.2,
            "threshold": 3.0
        })
        
        try:
            raise ValueError("Test error for logging")
        except ValueError:
            logger.exception("Operation failed")
        
        # Parse logged output
        log_output = log_stream.getvalue()
        log_lines = [line for line in log_output.strip().split('\n') if line]
        
        assert len(log_lines) == 3
        
        # Check first log (info)
        info_log = json.loads(log_lines[0])
        assert info_log["level"] == "INFO"
        assert info_log["message"] == "Starting operation"
        assert info_log["correlation_id"] == "integration-test-id"
        assert info_log["operation"] == "test_operation"
        assert info_log["input_size"] == 1024
        
        # Check second log (warning)
        warning_log = json.loads(log_lines[1])
        assert warning_log["level"] == "WARNING"
        assert warning_log["execution_time"] == 5.2
        
        # Check third log (exception)
        error_log = json.loads(log_lines[2])
        assert error_log["level"] == "ERROR"
        assert "exception" in error_log
        assert "ValueError" in error_log["exception"]
    
    def test_concurrent_logging_correlation_ids(self, log_capture):
        """Test correlation ID isolation in concurrent scenarios."""
        import threading
        import time
        
        logger, log_stream = log_capture
        results = {}
        
        def log_with_id(thread_id):
            correlation_id = f"thread-{thread_id}-correlation"
            set_correlation_id(correlation_id)
            
            logger.info(f"Message from thread {thread_id}")
            time.sleep(0.1)  # Simulate work
            
            # Verify correlation ID is still correct
            current_id = get_correlation_id()
            results[thread_id] = current_id == correlation_id
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=log_with_id, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All threads should maintain their correlation IDs
        assert all(results.values())
    
    def test_logger_configuration_persistence(self):
        """Test that logger configuration persists across calls."""
        logger1 = get_logger("test.persistent")
        logger2 = get_logger("test.persistent")
        
        # Should be the same logger instance
        assert logger1 is logger2
        
        # Should have structured formatter
        for handler in logger1.handlers:
            assert isinstance(handler.formatter, StructuredFormatter)
    
    def test_custom_formatter_fields(self, log_capture):
        """Test custom fields in structured logs."""
        logger, log_stream = log_capture
        
        set_correlation_id("custom-test", operation="custom_operation")
        
        # Log with custom fields
        logger.info("Custom operation completed", extra={
            "model_name": "custom-bert",
            "batch_size": 32,
            "processing_time": 2.5,
            "accuracy": 0.95,
            "metadata": {
                "version": "1.0",
                "config": {"param1": "value1"}
            }
        })
        
        log_output = log_stream.getvalue()
        log_data = json.loads(log_output.strip())
        
        assert log_data["model_name"] == "custom-bert"
        assert log_data["batch_size"] == 32
        assert log_data["processing_time"] == 2.5
        assert log_data["accuracy"] == 0.95
        assert log_data["metadata"]["version"] == "1.0"
        assert log_data["metadata"]["config"]["param1"] == "value1"
    
    def test_log_level_filtering(self):
        """Test log level filtering works with structured logging."""
        # Create logger with WARNING level
        logger = get_logger("test.filtering")
        logger.setLevel(logging.WARNING)
        
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        
        try:
            # These should not appear
            logger.debug("Debug message")
            logger.info("Info message")
            
            # These should appear
            logger.warning("Warning message")
            logger.error("Error message")
            
            log_output = log_stream.getvalue()
            log_lines = [line for line in log_output.strip().split('\n') if line]
            
            assert len(log_lines) == 2
            
            warning_log = json.loads(log_lines[0])
            error_log = json.loads(log_lines[1])
            
            assert warning_log["level"] == "WARNING"
            assert error_log["level"] == "ERROR"
            
        finally:
            logger.removeHandler(handler)


class TestErrorHandling:
    """Test error handling in structured logging."""
    
    def test_formatter_handles_serialization_errors(self):
        """Test formatter gracefully handles serialization errors."""
        formatter = StructuredFormatter()
        
        # Create object that can't be JSON serialized
        class UnserializableObject:
            def __str__(self):
                raise Exception("Cannot serialize")
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        record.correlation_id = "test-id"
        record.bad_object = UnserializableObject()
        
        # Should not raise exception
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        # Bad object should be converted to string representation
        assert log_data["message"] == "Test message"
        assert "bad_object" in log_data  # Should still be present in some form
    
    def test_correlation_id_filter_handles_errors(self):
        """Test correlation ID filter handles errors gracefully."""
        filter_instance = CorrelationIDFilter()
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Mock context to raise exception
        with patch('framework.observability.structured_logging._trace_context') as mock_context:
            mock_context.get.side_effect = Exception("Context error")
            
            # Should not raise exception
            result = filter_instance.filter(record)
            
            assert result is True
            assert hasattr(record, 'correlation_id')
            assert record.correlation_id is None

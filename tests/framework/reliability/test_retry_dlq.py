"""
Tests for Retry and Dead Letter Queue implementation.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

from framework.reliability.retry_dlq import (
    RetryPolicy, RetryManager, DeadLetterQueue, 
    ModelInferenceOperation, FailureClassification
)


class TestRetryPolicy:
    """Test retry policy configuration."""
    
    def test_default_retry_policy(self):
        """Test default retry policy values."""
        policy = RetryPolicy()
        
        assert policy.max_attempts == 3
        assert policy.base_delay == 1.0
        assert policy.max_delay == 60.0
        assert policy.exponential_base == 2.0
        assert policy.jitter == True
    
    def test_custom_retry_policy(self):
        """Test custom retry policy configuration."""
        policy = RetryPolicy(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=1.5,
            jitter=False
        )
        
        assert policy.max_attempts == 5
        assert policy.base_delay == 2.0
        assert policy.max_delay == 120.0
        assert policy.exponential_base == 1.5
        assert policy.jitter == False
    
    def test_calculate_delay_no_jitter(self):
        """Test delay calculation without jitter."""
        policy = RetryPolicy(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=False
        )
        
        # Test exponential backoff
        assert policy.calculate_delay(0) == 1.0
        assert policy.calculate_delay(1) == 2.0
        assert policy.calculate_delay(2) == 4.0
        assert policy.calculate_delay(3) == 8.0
        
        # Test max delay cap
        assert policy.calculate_delay(10) == 10.0
    
    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        policy = RetryPolicy(
            base_delay=2.0,
            exponential_base=2.0,
            max_delay=20.0,
            jitter=True
        )
        
        # With jitter, delay should be within range
        delay = policy.calculate_delay(1)
        expected_base = 4.0  # 2.0 * 2^1
        
        # Jitter adds 0-50% of base delay
        assert expected_base <= delay <= expected_base * 1.5
    
    def test_should_retry_transient_errors(self):
        """Test retry decision for transient errors."""
        policy = RetryPolicy()
        
        # Transient errors should be retried
        transient_errors = [
            ConnectionError("Network error"),
            TimeoutError("Request timeout"),
            OSError("Resource unavailable")
        ]
        
        for error in transient_errors:
            assert policy.should_retry(error, attempt=1)
    
    def test_should_retry_permanent_errors(self):
        """Test retry decision for permanent errors."""
        policy = RetryPolicy()
        
        # Permanent errors should not be retried
        permanent_errors = [
            ValueError("Invalid input"),
            TypeError("Type mismatch"),
            KeyError("Key not found")
        ]
        
        for error in permanent_errors:
            assert not policy.should_retry(error, attempt=1)
    
    def test_should_retry_max_attempts(self):
        """Test retry decision at max attempts."""
        policy = RetryPolicy(max_attempts=3)
        
        error = ConnectionError("Network error")
        
        # Should retry up to max attempts
        assert policy.should_retry(error, attempt=1)
        assert policy.should_retry(error, attempt=2)
        assert not policy.should_retry(error, attempt=3)  # At max


class TestRetryManager:
    """Test retry manager functionality."""
    
    @pytest.fixture
    def retry_manager(self):
        """Create retry manager with test policy."""
        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.01,  # Small delay for testing
            max_delay=0.1,
            jitter=False
        )
        return RetryManager(policy)
    
    @pytest.fixture
    def mock_operation(self):
        """Create mock operation."""
        return AsyncMock()
    
    @pytest.mark.asyncio
    async def test_successful_operation_no_retry(self, retry_manager, mock_operation):
        """Test successful operation requires no retry."""
        mock_operation.return_value = "success"
        
        result = await retry_manager.execute_with_retry(mock_operation, "test-op")
        
        assert result == "success"
        assert mock_operation.call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self, retry_manager, mock_operation):
        """Test retry on transient failures."""
        # Fail twice, then succeed
        mock_operation.side_effect = [
            ConnectionError("Network error"),
            TimeoutError("Timeout"),
            "success"
        ]
        
        result = await retry_manager.execute_with_retry(mock_operation, "test-op")
        
        assert result == "success"
        assert mock_operation.call_count == 3
    
    @pytest.mark.asyncio
    async def test_no_retry_on_permanent_failure(self, retry_manager, mock_operation):
        """Test no retry on permanent failures."""
        mock_operation.side_effect = ValueError("Invalid input")
        
        with pytest.raises(ValueError):
            await retry_manager.execute_with_retry(mock_operation, "test-op")
        
        assert mock_operation.call_count == 1
    
    @pytest.mark.asyncio
    async def test_exhaust_retry_attempts(self, retry_manager, mock_operation):
        """Test exhausting retry attempts."""
        mock_operation.side_effect = ConnectionError("Persistent network error")
        
        with pytest.raises(ConnectionError):
            await retry_manager.execute_with_retry(mock_operation, "test-op")
        
        # Should try max_attempts times
        assert mock_operation.call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_with_delay(self, retry_manager, mock_operation):
        """Test retry includes appropriate delays."""
        mock_operation.side_effect = [
            ConnectionError("Network error"),
            ConnectionError("Network error"),
            "success"
        ]
        
        start_time = time.time()
        result = await retry_manager.execute_with_retry(mock_operation, "test-op")
        elapsed = time.time() - start_time
        
        assert result == "success"
        # Should have some delay (0.01 + 0.02 = 0.03s minimum)
        assert elapsed >= 0.025
    
    @pytest.mark.asyncio
    async def test_retry_with_context(self, retry_manager, mock_operation):
        """Test retry with operation context."""
        mock_operation.side_effect = [ConnectionError("Error"), "success"]
        
        context = {"user_id": "123", "model": "test-model"}
        result = await retry_manager.execute_with_retry(
            mock_operation, "test-op", context=context
        )
        
        assert result == "success"
    
    def test_get_retry_stats(self, retry_manager):
        """Test getting retry statistics."""
        stats = retry_manager.get_stats()
        
        assert "total_operations" in stats
        assert "successful_operations" in stats
        assert "failed_operations" in stats
        assert "total_retries" in stats
        assert "operations_by_status" in stats
    
    @pytest.mark.asyncio
    async def test_retry_stats_tracking(self, retry_manager, mock_operation):
        """Test retry statistics are tracked correctly."""
        # Successful operation
        mock_operation.return_value = "success"
        await retry_manager.execute_with_retry(mock_operation, "success-op")
        
        # Failed operation with retries
        mock_operation.side_effect = ConnectionError("Error")
        try:
            await retry_manager.execute_with_retry(mock_operation, "fail-op")
        except ConnectionError:
            pass
        
        stats = retry_manager.get_stats()
        assert stats["total_operations"] == 2
        assert stats["successful_operations"] == 1
        assert stats["failed_operations"] == 1
        assert stats["total_retries"] > 0


class TestDeadLetterQueue:
    """Test dead letter queue functionality."""
    
    @pytest.fixture
    def dlq_path(self, tmp_path):
        """Create temporary DLQ path."""
        return tmp_path / "dlq"
    
    @pytest.fixture
    def dead_letter_queue(self, dlq_path):
        """Create dead letter queue."""
        return DeadLetterQueue(str(dlq_path))
    
    @pytest.mark.asyncio
    async def test_add_failed_operation(self, dead_letter_queue):
        """Test adding failed operation to DLQ."""
        operation_data = {
            "operation_id": "op-123",
            "operation_type": "model_inference",
            "input_data": {"text": "test input"},
            "context": {"user_id": "user-123"}
        }
        
        error = ValueError("Processing failed")
        
        await dead_letter_queue.add_failed_operation(
            operation_id="op-123",
            operation_type="model_inference",
            operation_data=operation_data,
            error=error,
            attempts=3
        )
        
        # Check file was created
        dlq_files = list(Path(dead_letter_queue.dlq_path).glob("*.json"))
        assert len(dlq_files) == 1
        
        # Check file content
        with open(dlq_files[0]) as f:
            saved_data = json.load(f)
        
        assert saved_data["operation_id"] == "op-123"
        assert saved_data["operation_type"] == "model_inference"
        assert saved_data["attempts"] == 3
        assert "error_message" in saved_data
        assert "timestamp" in saved_data
    
    @pytest.mark.asyncio
    async def test_get_failed_operations(self, dead_letter_queue):
        """Test retrieving failed operations from DLQ."""
        # Add multiple failed operations
        for i in range(3):
            await dead_letter_queue.add_failed_operation(
                operation_id=f"op-{i}",
                operation_type="test",
                operation_data={"test": f"data-{i}"},
                error=Exception(f"Error {i}"),
                attempts=2
            )
        
        failed_ops = dead_letter_queue.get_failed_operations()
        
        assert len(failed_ops) == 3
        operation_ids = [op["operation_id"] for op in failed_ops]
        assert "op-0" in operation_ids
        assert "op-1" in operation_ids
        assert "op-2" in operation_ids
    
    @pytest.mark.asyncio
    async def test_get_failed_operations_by_type(self, dead_letter_queue):
        """Test retrieving failed operations by type."""
        # Add operations of different types
        await dead_letter_queue.add_failed_operation(
            "op-1", "model_inference", {}, Exception("Error"), 1
        )
        await dead_letter_queue.add_failed_operation(
            "op-2", "data_processing", {}, Exception("Error"), 1
        )
        await dead_letter_queue.add_failed_operation(
            "op-3", "model_inference", {}, Exception("Error"), 1
        )
        
        # Get only model_inference operations
        inference_ops = dead_letter_queue.get_failed_operations("model_inference")
        
        assert len(inference_ops) == 2
        assert all(op["operation_type"] == "model_inference" for op in inference_ops)
    
    @pytest.mark.asyncio
    async def test_reprocess_operation(self, dead_letter_queue):
        """Test reprocessing operation from DLQ."""
        # Add failed operation
        await dead_letter_queue.add_failed_operation(
            "op-reprocess", "test", {"data": "test"}, Exception("Error"), 2
        )
        
        # Mock successful reprocessing
        async def mock_processor(operation_data):
            return "reprocessed successfully"
        
        success = await dead_letter_queue.reprocess_operation(
            "op-reprocess", mock_processor
        )
        
        assert success
        
        # Operation should be removed from DLQ
        failed_ops = dead_letter_queue.get_failed_operations()
        assert not any(op["operation_id"] == "op-reprocess" for op in failed_ops)
    
    @pytest.mark.asyncio
    async def test_reprocess_nonexistent_operation(self, dead_letter_queue):
        """Test reprocessing non-existent operation."""
        async def mock_processor(operation_data):
            return "success"
        
        success = await dead_letter_queue.reprocess_operation(
            "nonexistent", mock_processor
        )
        
        assert not success
    
    @pytest.mark.asyncio
    async def test_cleanup_old_entries(self, dead_letter_queue):
        """Test cleanup of old DLQ entries."""
        # Add operation and manually set old timestamp
        await dead_letter_queue.add_failed_operation(
            "old-op", "test", {}, Exception("Error"), 1
        )
        
        # Manually modify file timestamp to be old
        dlq_files = list(Path(dead_letter_queue.dlq_path).glob("*.json"))
        old_file = dlq_files[0]
        
        with open(old_file) as f:
            data = json.load(f)
        
        # Set timestamp to 8 days ago (older than default 7 day retention)
        old_timestamp = (datetime.utcnow() - timedelta(days=8)).isoformat()
        data["timestamp"] = old_timestamp
        
        with open(old_file, 'w') as f:
            json.dump(data, f)
        
        # Run cleanup
        cleaned_count = dead_letter_queue.cleanup_old_entries()
        
        assert cleaned_count == 1
        assert len(list(Path(dead_letter_queue.dlq_path).glob("*.json"))) == 0
    
    def test_get_dlq_stats(self, dead_letter_queue):
        """Test getting DLQ statistics."""
        stats = dead_letter_queue.get_stats()
        
        assert "total_failed_operations" in stats
        assert "operations_by_type" in stats
        assert "oldest_entry" in stats
        assert "newest_entry" in stats


class TestModelInferenceOperation:
    """Test model inference operation wrapper."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = AsyncMock()
        model.predict = AsyncMock()
        return model
    
    def test_operation_creation(self, mock_model):
        """Test creating model inference operation."""
        operation = ModelInferenceOperation(
            model=mock_model,
            input_data={"text": "test input"},
            model_name="test-model"
        )
        
        assert operation.model is mock_model
        assert operation.input_data == {"text": "test input"}
        assert operation.model_name == "test-model"
    
    @pytest.mark.asyncio
    async def test_successful_inference(self, mock_model):
        """Test successful model inference."""
        mock_model.predict.return_value = {"prediction": "positive", "confidence": 0.95}
        
        operation = ModelInferenceOperation(
            model=mock_model,
            input_data={"text": "test input"},
            model_name="test-model"
        )
        
        result = await operation.execute()
        
        assert result == {"prediction": "positive", "confidence": 0.95}
        mock_model.predict.assert_called_once_with({"text": "test input"})
    
    @pytest.mark.asyncio
    async def test_inference_failure(self, mock_model):
        """Test model inference failure."""
        mock_model.predict.side_effect = RuntimeError("Model error")
        
        operation = ModelInferenceOperation(
            model=mock_model,
            input_data={"text": "test input"},
            model_name="test-model"
        )
        
        with pytest.raises(RuntimeError):
            await operation.execute()


class TestFailureClassification:
    """Test failure classification functionality."""
    
    def test_classify_transient_failures(self):
        """Test classification of transient failures."""
        transient_errors = [
            ConnectionError("Connection lost"),
            TimeoutError("Request timeout"),
            OSError("Resource temporarily unavailable"),
            asyncio.TimeoutError("Async timeout")
        ]
        
        for error in transient_errors:
            classification = FailureClassification.classify_failure(error)
            assert classification == "transient"
    
    def test_classify_permanent_failures(self):
        """Test classification of permanent failures."""
        permanent_errors = [
            ValueError("Invalid input format"),
            TypeError("Wrong type provided"),
            KeyError("Required key missing"),
            AttributeError("Method not found")
        ]
        
        for error in permanent_errors:
            classification = FailureClassification.classify_failure(error)
            assert classification == "permanent"
    
    def test_classify_model_failures(self):
        """Test classification of model-specific failures."""
        model_errors = [
            RuntimeError("CUDA out of memory"),
            RuntimeError("Model not loaded"),
            Exception("Inference failed")
        ]
        
        for error in model_errors:
            classification = FailureClassification.classify_failure(error)
            # Should be either "model" or "permanent" depending on message
            assert classification in ["model", "permanent", "transient"]
    
    def test_is_retryable_error(self):
        """Test retryable error detection."""
        # Transient errors should be retryable
        assert FailureClassification.is_retryable(ConnectionError("Error"))
        assert FailureClassification.is_retryable(TimeoutError("Error"))
        
        # Permanent errors should not be retryable
        assert not FailureClassification.is_retryable(ValueError("Error"))
        assert not FailureClassification.is_retryable(TypeError("Error"))
    
    def test_get_retry_delay(self):
        """Test getting appropriate retry delay for error types."""
        # Transient errors get standard delay
        transient_delay = FailureClassification.get_retry_delay(
            ConnectionError("Error"), attempt=1
        )
        assert transient_delay > 0
        
        # Model errors might get longer delay
        model_delay = FailureClassification.get_retry_delay(
            RuntimeError("CUDA error"), attempt=1
        )
        assert model_delay > 0


class TestIntegrationScenarios:
    """Test integration scenarios combining retry manager and DLQ."""
    
    @pytest.fixture
    def integrated_system(self, tmp_path):
        """Create integrated retry + DLQ system."""
        policy = RetryPolicy(max_attempts=2, base_delay=0.01)
        retry_manager = RetryManager(policy)
        dlq = DeadLetterQueue(str(tmp_path / "dlq"))
        return retry_manager, dlq
    
    @pytest.mark.asyncio
    async def test_retry_then_dlq_flow(self, integrated_system):
        """Test complete retry -> DLQ flow."""
        retry_manager, dlq = integrated_system
        
        async def failing_operation(data=None):
            raise ConnectionError("Persistent failure")
        
        # Try operation with retry
        try:
            await retry_manager.execute_with_retry(failing_operation, "test-op")
        except ConnectionError as e:
            # Add to DLQ when retries exhausted
            await dlq.add_failed_operation(
                operation_id="test-op",
                operation_type="test",
                operation_data={"test": "data"},
                error=e,
                attempts=retry_manager.policy.max_attempts
            )
        
        # Verify operation in DLQ
        failed_ops = dlq.get_failed_operations()
        assert len(failed_ops) == 1
        assert failed_ops[0]["operation_id"] == "test-op"
    
    @pytest.mark.asyncio
    async def test_successful_reprocessing_from_dlq(self, integrated_system):
        """Test successful reprocessing from DLQ."""
        retry_manager, dlq = integrated_system
        
        # Add failed operation to DLQ
        await dlq.add_failed_operation(
            "reprocess-op", "test", {"input": "test"}, Exception("Error"), 2
        )
        
        # Create successful reprocessor
        async def successful_reprocessor(operation_data):
            return "reprocessed successfully"
        
        # Reprocess with retry manager
        async def reprocess_with_retry(operation_data):
            return await successful_reprocessor(operation_data)
        
        success = await dlq.reprocess_operation("reprocess-op", reprocess_with_retry)
        
        assert success
    
    @pytest.mark.asyncio
    async def test_batch_reprocessing(self, integrated_system):
        """Test batch reprocessing of DLQ entries."""
        retry_manager, dlq = integrated_system
        
        # Add multiple failed operations
        for i in range(3):
            await dlq.add_failed_operation(
                f"batch-op-{i}", "test", {"data": i}, Exception("Error"), 1
            )
        
        # Reprocess all
        async def batch_processor(operation_data):
            return f"processed-{operation_data['data']}"
        
        failed_ops = dlq.get_failed_operations()
        success_count = 0
        
        for op in failed_ops:
            success = await dlq.reprocess_operation(
                op["operation_id"], batch_processor
            )
            if success:
                success_count += 1
        
        assert success_count == 3
        
        # DLQ should be empty after successful reprocessing
        remaining_ops = dlq.get_failed_operations()
        assert len(remaining_ops) == 0

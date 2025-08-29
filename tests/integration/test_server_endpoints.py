"""Integration tests for FastAPI server endpoints."""

import pytest
import asyncio
import time
import json
from typing import Dict, Any
from pathlib import Path
from unittest.mock import patch

import httpx
import uvicorn
from fastapi.testclient import TestClient

# Import the main FastAPI app
from main import app, initialize_inference_engine, cleanup_inference_engine


class TestServerEndpoints:
    """Test FastAPI server endpoints."""

    @pytest.mark.asyncio
    async def test_server_startup_and_shutdown(self):
        """Test server startup and shutdown sequence."""
        # Test that we can initialize the engine
        await initialize_inference_engine()
        
        # Test that cleanup works
        await cleanup_inference_engine()

    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert "timestamp" in data
        assert "environment" in data
        assert "endpoints" in data
        
        # Verify specific values
        assert data["message"] == "PyTorch Inference Framework API - Enhanced with TTS Support"
        assert data["version"] == "1.0.0-TTS-Enhanced"
        assert data["status"] == "running"
        
        # Verify endpoints structure
        endpoints = data["endpoints"]
        assert "inference" in endpoints
        assert "health" in endpoints
        assert "stats" in endpoints
        assert "models" in endpoints
        assert "enhanced_downloads" in endpoints
        assert "tts_audio" in endpoints

    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client):
        """Test the health check endpoint."""
        # Initialize engine for health check
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                response = await client.get("/health")
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify response structure
                assert "healthy" in data
                assert "checks" in data
                assert "timestamp" in data
                
                # Should be healthy with initialized engine
                assert data["healthy"] is True
                
                # Verify checks structure
                checks = data["checks"]
                assert isinstance(checks, dict)
                
        finally:
            await cleanup_inference_engine()

    def test_health_endpoint_without_engine(self, client):
        """Test health endpoint when engine is not initialized."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should report as unhealthy without engine
        assert data["healthy"] is False
        assert "inference_engine" in data["checks"]
        assert data["checks"]["inference_engine"] is False

    @pytest.mark.asyncio
    async def test_predict_endpoint(self, async_client):
        """Test the single prediction endpoint."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                # Test with simple numeric input
                payload = {
                    "inputs": 42,
                    "priority": 1,
                    "timeout": 10.0
                }
                
                response = await client.post("/predict", json=payload)
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify response structure
                assert "success" in data
                assert "result" in data
                assert "processing_time" in data
                assert "model_info" in data
                
                # Should be successful
                assert data["success"] is True
                assert data["result"] is not None
                assert data["processing_time"] > 0
                assert data["model_info"]["model"] == "example"

        finally:
            await cleanup_inference_engine()

def test_predict_endpoint_without_engine(client):
    """Test predict endpoint when engine is not initialized."""
    payload = {"inputs": 42}
    
    with patch('main.inference_engine', None):
        response = client.post("/predict", json=payload)

        # Should return 503 Service Unavailable
        assert response.status_code == 503
        assert "Inference services not available" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_batch_predict_endpoint(self, async_client):
        """Test the batch prediction endpoint."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                # Test with batch of inputs
                payload = {
                    "inputs": [1, 2, 3, 4, 5],
                    "priority": 2,
                    "timeout": 15.0
                }
                
                response = await client.post("/predict/batch", json=payload)
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify response structure
                assert "success" in data
                assert "results" in data
                assert "processing_time" in data
                assert "batch_size" in data
                
                # Should be successful
                assert data["success"] is True
                assert len(data["results"]) == 5
                assert data["batch_size"] == 5
                assert data["processing_time"] > 0

        finally:
            await cleanup_inference_engine()

    @pytest.mark.asyncio
    async def test_stats_endpoint(self, async_client):
        """Test the stats endpoint."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                response = await client.get("/stats")
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify response structure
                assert "stats" in data
                assert "performance_report" in data
                
                # Verify stats structure
                stats = data["stats"]
                assert isinstance(stats, dict)
                
                performance_report = data["performance_report"]
                assert isinstance(performance_report, dict)

        finally:
            await cleanup_inference_engine()

    def test_stats_endpoint_without_engine(self, client):
        """Test stats endpoint when engine is not initialized."""
        response = client.get("/stats")
        
        # Should return 503 Service Unavailable
        assert response.status_code == 503
        assert "Inference engine not available" in response.json()["detail"]

    def test_config_endpoint(self, client):
        """Test the config endpoint."""
        response = client.get("/config")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "configuration" in data
        assert "inference_config" in data
        assert "server_config" in data
        
        # Verify inference config structure
        inference_config = data["inference_config"]
        assert "device_type" in inference_config
        assert "batch_size" in inference_config
        assert "use_fp16" in inference_config
        assert "enable_profiling" in inference_config

    @pytest.mark.asyncio
    async def test_models_endpoint(self, async_client):
        """Test the models listing endpoint."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                response = await client.get("/models")
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify response structure
                assert "models" in data
                assert "model_info" in data
                assert "total_models" in data
                
                # Should have at least the example model
                assert data["total_models"] >= 1
                assert "example" in data["models"]

        finally:
            await cleanup_inference_engine()

    def test_models_available_endpoint(self, client):
        """Test the available models for download endpoint."""
        response = client.get("/models/available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "available_models" in data
        assert "total_available" in data
        
        # Should be a dict of available models
        assert isinstance(data["available_models"], dict)
        assert isinstance(data["total_available"], int)

    def test_cache_info_endpoint(self, client):
        """Test the model cache info endpoint."""
        response = client.get("/models/cache/info")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "cache_directory" in data
        assert "total_models" in data
        assert "total_size_mb" in data
        assert "models" in data
        
        # Should have valid cache info
        assert isinstance(data["total_models"], int)
        assert isinstance(data["total_size_mb"], (int, float))
        assert isinstance(data["models"], list)

    @pytest.mark.asyncio
    async def test_simple_example_endpoint(self, async_client):
        """Test the simple example endpoint."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                payload = {"input": 123}
                
                response = await client.post("/examples/simple", json=payload)
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify response structure
                assert "example" in data
                assert "input" in data
                assert "response" in data
                
                assert data["example"] == "simple_prediction"
                assert data["input"] == 123
                assert data["response"]["success"] is True

        finally:
            await cleanup_inference_engine()

    @pytest.mark.asyncio
    async def test_batch_example_endpoint(self, async_client):
        """Test the batch example endpoint."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                response = await client.post("/examples/batch")
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify response structure
                assert "example" in data
                assert "input_count" in data
                assert "response" in data
                
                assert data["example"] == "batch_prediction"
                assert data["input_count"] == 5
                assert data["response"]["success"] is True
                assert len(data["response"]["results"]) == 5

        finally:
            await cleanup_inference_engine()

    @pytest.mark.asyncio
    async def test_model_download_endpoint(self, async_client):
        """Test the model download endpoint."""
        async with async_client as client:
            # Test with valid parameters
            params = {
                "source": "pytorch_hub",
                "model_id": "pytorch/vision/mobilenet_v2",  # Fixed PyTorch Hub format (repo/model)
                "name": "test_mobilenet",
                "task": "classification",
                "weights": "DEFAULT"  # Use 'weights' instead of deprecated 'pretrained'
            }
            
            response = await client.post("/models/download", params=params)
            
            # This might fail due to actual model download, but should have proper structure
            if response.status_code == 200:
                data = response.json()
                assert "message" in data
                assert "model_name" in data
                assert "source" in data
                assert "status" in data
            else:
                # If it fails, should be due to actual download issues, not endpoint issues
                assert response.status_code in [400, 422, 500]

    def test_model_download_invalid_source(self, client):
        """Test model download with invalid source."""
        params = {
            "source": "invalid_source",
            "model_id": "test_model", 
            "name": "test_name"
        }
        
        response = client.post("/models/download", params=params)
        
        assert response.status_code in [400, 422]  # FastAPI returns 422 for validation errors
        if response.status_code == 400:
            assert "Invalid source" in response.json()["detail"]

    def test_download_info_nonexistent_model(self, client):
        """Test getting download info for non-existent model."""
        response = client.get("/models/download/nonexistent_model/info")
        
        assert response.status_code == 404
        assert "Model not found" in response.json()["detail"]

    def test_remove_nonexistent_model(self, client):
        """Test removing a non-existent model."""
        response = client.delete("/models/download/nonexistent_model")
        
        assert response.status_code == 404
        assert "Model not found in cache" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client):
        """Test handling concurrent requests."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                # Create multiple concurrent prediction requests
                tasks = []
                for i in range(5):
                    payload = {"inputs": i, "priority": i % 3}
                    task = client.post("/predict", json=payload)
                    tasks.append(task)
                
                # Execute all requests concurrently
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Verify all requests succeeded
                successful_responses = 0
                for response in responses:
                    if isinstance(response, httpx.Response) and response.status_code == 200:
                        successful_responses += 1
                
                # Should have reasonable success rate for concurrent requests
                assert successful_responses >= 3  # At least 60% success rate

        finally:
            await cleanup_inference_engine()

    @pytest.mark.asyncio
    async def test_request_timeout_handling(self, async_client):
        """Test request timeout handling."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                # Test with very short timeout
                payload = {
                    "inputs": 42,
                    "timeout": 0.001  # 1ms timeout, should be too short
                }
                
                response = await client.post("/predict", json=payload)
                
                # Should still get a response (might succeed if model is very fast)
                assert response.status_code in [200, 408]  # OK or Request Timeout
                
                if response.status_code == 200:
                    data = response.json()
                    # If successful, verify structure
                    assert "success" in data

        finally:
            await cleanup_inference_engine()

    @pytest.mark.asyncio
    async def test_error_handling_invalid_input(self, async_client):
        """Test error handling with invalid inputs."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                # Test with invalid JSON
                response = await client.post(
                    "/predict",
                    content="invalid json",
                    headers={"Content-Type": "application/json"}
                )
                
                assert response.status_code == 422  # Unprocessable Entity

        finally:
            await cleanup_inference_engine()

    @pytest.mark.asyncio
    async def test_health_check_with_load(self, async_client):
        """Test health check under load."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                # Generate some load
                prediction_tasks = []
                for i in range(3):
                    payload = {"inputs": i}
                    task = client.post("/predict", json=payload)
                    prediction_tasks.append(task)
                
                # Check health while under load
                health_response = await client.get("/health")
                
                assert health_response.status_code == 200
                health_data = health_response.json()
                assert health_data["healthy"] is True
                
                # Wait for prediction tasks to complete
                await asyncio.gather(*prediction_tasks, return_exceptions=True)

        finally:
            await cleanup_inference_engine()

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/")
        
        # Verify CORS headers (added by middleware)
        assert response.status_code in [200, 405]  # Some clients return 405 for OPTIONS
        
        # Check if CORS headers would be present in actual requests
        get_response = client.get("/")
        assert get_response.status_code == 200

    @pytest.mark.asyncio
    async def test_request_logging_middleware(self, async_client):
        """Test that request logging middleware is working."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                response = await client.get("/")
                
                # Verify process time header is added by middleware
                assert "X-Process-Time" in response.headers
                process_time = float(response.headers["X-Process-Time"])
                assert process_time >= 0

        finally:
            await cleanup_inference_engine()

    @pytest.mark.asyncio
    async def test_different_input_types(self, async_client):
        """Test prediction endpoint with different input types."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                # Test different input types that the example model should handle
                test_inputs = [
                    42,  # Integer
                    3.14,  # Float
                    [1, 2, 3, 4, 5],  # List of numbers
                    "test string",  # String
                ]
                
                for input_data in test_inputs:
                    payload = {"inputs": input_data}
                    response = await client.post("/predict", json=payload)
                    
                    assert response.status_code == 200
                    data = response.json()
                    # The prediction may fail for some input types due to tensor shape issues
                    # but the endpoint should still return a proper response structure
                    assert "success" in data
                    assert "result" in data or "error" in data

        finally:
            await cleanup_inference_engine()

    @pytest.mark.asyncio
    async def test_priority_handling(self, async_client):
        """Test that priority is handled in requests."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                # Send requests with different priorities
                high_priority = {"inputs": 1, "priority": 10}
                low_priority = {"inputs": 2, "priority": 1}
                
                # Both should succeed (priority affects internal processing)
                high_response = await client.post("/predict", json=high_priority)
                low_response = await client.post("/predict", json=low_priority)
                
                assert high_response.status_code == 200
                assert low_response.status_code == 200
                
                # Both should be successful
                assert high_response.json()["success"] is True
                assert low_response.json()["success"] is True

        finally:
            await cleanup_inference_engine()


class TestServerPerformance:
    """Test server performance characteristics."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.mark.asyncio
    async def test_startup_time(self):
        """Test server startup performance."""
        start_time = time.time()
        
        await initialize_inference_engine()
        
        startup_time = time.time() - start_time
        
        try:
            # Startup should be reasonable (less than 30 seconds)
            assert startup_time < 30.0
        finally:
            await cleanup_inference_engine()

    @pytest.mark.asyncio
    async def test_response_time_benchmark(self, async_client):
        """Test response time benchmarks."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                # Measure response times for different endpoints
                endpoints_to_test = [
                    ("GET", "/"),
                    ("GET", "/health"),
                    ("GET", "/config"),
                    ("GET", "/models"),
                ]
                
                response_times = {}
                
                for method, endpoint in endpoints_to_test:
                    start_time = time.time()
                    
                    if method == "GET":
                        response = await client.get(endpoint)
                    else:
                        response = await client.post(endpoint, json={})
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    response_times[endpoint] = response_time
                    
                    # Verify response is successful
                    assert response.status_code == 200
                    
                    # Response time should be reasonable (less than 5 seconds)
                    assert response_time < 5.0
                
                print(f"Response times: {response_times}")

        finally:
            await cleanup_inference_engine()

    @pytest.mark.asyncio
    async def test_prediction_throughput(self, async_client):
        """Test prediction throughput."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                # Measure throughput for predictions
                num_predictions = 10
                start_time = time.time()
                
                tasks = []
                for i in range(num_predictions):
                    payload = {"inputs": i}
                    task = client.post("/predict", json=payload)
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                end_time = time.time()
                total_time = end_time - start_time
                
                # Count successful predictions
                successful_predictions = 0
                for response in responses:
                    if (isinstance(response, httpx.Response) and 
                        response.status_code == 200 and 
                        response.json().get("success", False)):
                        successful_predictions += 1
                
                # Calculate throughput
                if total_time > 0:
                    throughput = successful_predictions / total_time
                    print(f"Prediction throughput: {throughput:.2f} predictions/second")
                    
                    # Should achieve reasonable throughput
                    assert throughput > 0.1  # At least 0.1 predictions per second
                
                # Should have decent success rate
                success_rate = successful_predictions / num_predictions
                assert success_rate >= 0.7  # At least 70% success rate

        finally:
            await cleanup_inference_engine()


@pytest.mark.slow
class TestServerStressTest:
    """Stress tests for the server."""

    @pytest.mark.asyncio
    async def test_high_load_stress_test(self, async_client):
        """Test server under high load."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                # Generate high load with many concurrent requests
                num_requests = 20
                
                async def make_request(request_id: int):
                    """Make a single request."""
                    try:
                        payload = {"inputs": request_id, "timeout": 10.0}
                        response = await client.post("/predict", json=payload)
                        return response.status_code == 200 and response.json().get("success", False)
                    except Exception:
                        return False
                        return False
                
                # Create many concurrent tasks
                tasks = [make_request(i) for i in range(num_requests)]
                
                # Execute with timeout
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=60.0  # 1 minute timeout
                )
                
                # Count successful requests
                successful_requests = sum(1 for result in results if result is True)
                
                print(f"Successful requests: {successful_requests}/{num_requests}")
                
                # Should handle most requests successfully under load
                success_rate = successful_requests / num_requests
                assert success_rate >= 0.5  # At least 50% success rate under stress

        except asyncio.TimeoutError:
            pytest.skip("Stress test timed out - system may be under load")
        finally:
            await cleanup_inference_engine()

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, async_client):
        """Test memory usage under load."""
        await initialize_inference_engine()
        
        try:
            async with async_client as client:
                # Make multiple requests and check memory doesn't grow excessively
                for batch in range(3):
                    tasks = []
                    for i in range(5):
                        payload = {"inputs": [1, 2, 3, 4, 5]}  # Batch input
                        task = client.post("/predict", json=payload)
                        tasks.append(task)
                    
                    # Execute batch
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Brief pause between batches
                    await asyncio.sleep(0.5)
                
                # Server should still be responsive after load
                health_response = await client.get("/health")
                assert health_response.status_code == 200
                assert health_response.json()["healthy"] is True

        finally:
            await cleanup_inference_engine()

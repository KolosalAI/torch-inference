"""Integration tests for autoscaling server endpoints."""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from fastapi.testclient import TestClient
import httpx

# Import the actual main app
from main import app


@pytest.fixture
def mock_autoscaler_for_server():
    """Create a mock autoscaler for server testing."""
    autoscaler = Mock()
    
    # Mock basic methods
    autoscaler.start = AsyncMock()
    autoscaler.stop = AsyncMock()
    autoscaler.is_running = True
    
    # Mock prediction
    async def mock_predict(model_id, inputs, **kwargs):
        return {
            "predictions": [0.1, 0.2, 0.7],
            "model_id": model_id,
            "confidence": 0.8
        }
    autoscaler.predict = AsyncMock(side_effect=mock_predict)
    
    # Mock scaling
    async def mock_scale_model(model_id, target_instances):
        return True  # Return success boolean
    autoscaler.scale_model = AsyncMock(side_effect=mock_scale_model)
    
    # Mock loading
    async def mock_load_model(model_id, version="v1"):
        return True  # Return success boolean
    autoscaler.load_model = AsyncMock(side_effect=mock_load_model)
    
    # Mock unloading
    async def mock_unload_model(model_id, version=None):
        return True  # Return success boolean
    autoscaler.unload_model = AsyncMock(side_effect=mock_unload_model)
    
    # Mock stats
    autoscaler.get_stats = Mock(return_value={
        "total_instances": 5,
        "loaded_models": 3,
        "zero_scaler": {
            "enabled": True,
            "idle_instances": 2,
            "active_instances": 1
        },
        "model_loader": {
            "enabled": True,
            "total_instances": 4,
            "loaded_models": 3
        }
    })
    
    # Mock metrics_collector attribute
    autoscaler.metrics_collector = Mock()
    autoscaler.metrics_collector.get_summary = Mock(return_value={
        "timestamp": 1629820000.0,
        "models": {
            "test_model": {
                "request_count": 100,
                "success_count": 95,
                "error_count": 5,
                "average_response_time": 0.05,
                "error_rate": 0.05
            }
        }
    })
    
    # Mock health
    autoscaler.get_health_status = Mock(return_value={
        "healthy": True,
        "timestamp": 1629820000.0,
        "components": {
            "zero_scaler": {"status": "healthy"},
            "model_loader": {"status": "healthy"},
            "metrics_collector": {"status": "healthy"}
        }
    })
    
    # Mock metrics with proper JSON-serializable structure
    autoscaler.get_metrics = Mock(return_value={
        "metrics": {
            "test_model": {
                "request_count": 100,
                "success_count": 95,
                "error_count": 5,
                "average_response_time": 0.05,
                "error_rate": 0.05
            }
        },
        "timestamp": 1629820000.0
    })
    
    # Mock Prometheus metrics
    autoscaler.get_prometheus_metrics = Mock(return_value="""
# HELP autoscaler_requests_total Total number of requests
# TYPE autoscaler_requests_total counter
autoscaler_requests_total{model="test_model"} 100

# HELP autoscaler_response_time_seconds Response time in seconds
# TYPE autoscaler_response_time_seconds histogram
autoscaler_response_time_seconds_sum{model="test_model"} 5.0
autoscaler_response_time_seconds_count{model="test_model"} 100
""".strip())
    
    return autoscaler


@pytest.fixture
def client_with_mock_autoscaler(mock_autoscaler_for_server):
    """Create test client with mocked autoscaler."""
    # Patch both the global autoscaler and the initialization function
    with patch('main.autoscaler', mock_autoscaler_for_server):
        with patch('main.initialize_inference_engine', AsyncMock()):
            # Set the autoscaler directly after patching
            import main
            main.autoscaler = mock_autoscaler_for_server
            
            with TestClient(app) as client:
                yield client


class TestAutoscalerServerEndpoints:
    """Test autoscaler endpoints in the FastAPI server."""
    
    def test_autoscaler_health_endpoint(self, client_with_mock_autoscaler):
        """Test autoscaler health endpoint."""
        response = client_with_mock_autoscaler.get("/autoscaler/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["healthy"] is True
        assert "timestamp" in data
    
    def test_autoscaler_stats_endpoint(self, client_with_mock_autoscaler):
        """Test autoscaler stats endpoint."""
        response = client_with_mock_autoscaler.get("/autoscaler/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_instances"] == 5
        assert data["loaded_models"] == 3
        assert "zero_scaler" in data
        assert "model_loader" in data
    
    def test_autoscaler_metrics_endpoint(self, client_with_mock_autoscaler):
        """Test autoscaler metrics endpoint."""
        response = client_with_mock_autoscaler.get("/autoscaler/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "metrics" in data
        assert "timestamp" in data
    
    def test_autoscaler_scale_endpoint(self, client_with_mock_autoscaler):
        """Test autoscaler scale endpoint."""
        response = client_with_mock_autoscaler.post(
            "/autoscaler/scale",
            params={"model_name": "test_model", "target_instances": 3}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["model_name"] == "test_model"
        assert data["target_instances"] == 3
    
    def test_autoscaler_scale_endpoint_invalid_params(self, client_with_mock_autoscaler):
        """Test autoscaler scale endpoint with invalid parameters."""
        # Missing parameters
        response = client_with_mock_autoscaler.post("/autoscaler/scale")
        assert response.status_code == 422
        
        # Invalid target_instances (out of range)
        response = client_with_mock_autoscaler.post(
            "/autoscaler/scale",
            params={"model_name": "test_model", "target_instances": 15}
        )
        assert response.status_code == 400
    
    def test_autoscaler_load_endpoint(self, client_with_mock_autoscaler):
        """Test autoscaler load model endpoint."""
        response = client_with_mock_autoscaler.post(
            "/autoscaler/load",
            params={"model_name": "new_model"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["model_name"] == "new_model"
    
    def test_autoscaler_unload_endpoint(self, client_with_mock_autoscaler):
        """Test autoscaler unload model endpoint."""
        response = client_with_mock_autoscaler.delete(
            "/autoscaler/unload",
            params={"model_name": "test_model"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["model_name"] == "test_model"
        assert "message" in data
    
    def test_predict_endpoint_with_autoscaler(self, client_with_mock_autoscaler):
        """Test that predict endpoint uses autoscaler when available."""
        response = client_with_mock_autoscaler.post(
            "/predict",
            json={
                "model_name": "example",  # Add required model_name field
                "inputs": {"text": "test input"}
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "result" in data


class TestAutoscalerServerErrorHandling:
    """Test error handling in autoscaler server endpoints."""
    
    @pytest.fixture
    def client_with_failing_autoscaler(self):
        """Create test client with failing autoscaler."""
        failing_autoscaler = Mock()
        failing_autoscaler.start = AsyncMock()
        failing_autoscaler.stop = AsyncMock()  # Add stop method for cleanup
        failing_autoscaler.get_health_status.side_effect = Exception("Autoscaler error")
        failing_autoscaler.get_stats.side_effect = Exception("Stats error")
        failing_autoscaler.get_metrics.side_effect = Exception("Metrics error")
        failing_autoscaler.scale_model = AsyncMock(side_effect=Exception("Scaling error"))
        failing_autoscaler.load_model = AsyncMock(side_effect=Exception("Loading error"))
        failing_autoscaler.unload_model = AsyncMock(side_effect=Exception("Unloading error"))
        failing_autoscaler.predict = AsyncMock(side_effect=Exception("Prediction error"))
        
        with patch('main.autoscaler', failing_autoscaler):
            with patch('main.initialize_inference_engine', AsyncMock()):
                # Set the autoscaler directly after patching
                import main
                main.autoscaler = failing_autoscaler
                
                with TestClient(app) as client:
                    yield client    
    def test_health_endpoint_error_handling(self, client_with_failing_autoscaler):
        """Test health endpoint error handling."""
        response = client_with_failing_autoscaler.get("/autoscaler/health")
        
        assert response.status_code == 200  # Health endpoint returns 200 with error details
        data = response.json()
        assert data["healthy"] is False
        assert "error" in data
    
    def test_stats_endpoint_error_handling(self, client_with_failing_autoscaler):
        """Test stats endpoint error handling."""
        response = client_with_failing_autoscaler.get("/autoscaler/stats")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data  # FastAPI uses "detail" for error messages
    
    def test_metrics_endpoint_error_handling(self, client_with_failing_autoscaler):
        """Test metrics endpoint error handling."""
        response = client_with_failing_autoscaler.get("/autoscaler/metrics")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data  # FastAPI uses "detail" for error messages
    
    def test_scale_endpoint_error_handling(self, client_with_failing_autoscaler):
        """Test scale endpoint error handling."""
        response = client_with_failing_autoscaler.post(
            "/autoscaler/scale",
            params={"model_name": "test_model", "target_instances": 2}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data  # FastAPI uses "detail" for error messages
    
    def test_load_endpoint_error_handling(self, client_with_failing_autoscaler):
        """Test load endpoint error handling."""
        response = client_with_failing_autoscaler.post(
            "/autoscaler/load",
            params={"model_name": "test_model"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data  # FastAPI uses "detail" for error messages
    
    def test_unload_endpoint_error_handling(self, client_with_failing_autoscaler):
        """Test unload endpoint error handling."""
        response = client_with_failing_autoscaler.delete(
            "/autoscaler/unload",
            params={"model_name": "test_model"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data  # FastAPI uses "detail" for error messages


class TestAutoscalerServerAsyncClient:
    """Test autoscaler endpoints using async client."""
    
    @pytest.mark.asyncio
    async def test_async_autoscaler_endpoints(self, mock_autoscaler_for_server):
        """Test autoscaler endpoints using async client."""
        with patch('main.autoscaler', mock_autoscaler_for_server):
            with patch('main.initialize_inference_engine', AsyncMock()):
                # Set the autoscaler directly after patching
                import main
                main.autoscaler = mock_autoscaler_for_server
                
                # Use httpx with ASGI transport for the FastAPI app
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    
                    # Test health endpoint
                    response = await client.get("/autoscaler/health")
                    assert response.status_code == 200
                    health_data = response.json()
                    assert health_data["healthy"] is True
                    
                    # Test stats endpoint
                    response = await client.get("/autoscaler/stats")
                    assert response.status_code == 200
                    stats_data = response.json()
                    assert stats_data["total_instances"] == 5
                    
                    # Test metrics endpoint
                    response = await client.get("/autoscaler/metrics")
                    assert response.status_code == 200
                    metrics_data = response.json()
                    assert "metrics" in metrics_data
                    
                    # Test scaling endpoint
                    response = await client.post(
                        "/autoscaler/scale",
                        params={"model_name": "test_model", "target_instances": 2}
                    )
                    assert response.status_code == 200
                    scale_data = response.json()
                    assert scale_data["success"] is True
                    
                    # Test load endpoint
                    response = await client.post(
                        "/autoscaler/load",
                        params={"model_name": "new_model"}
                    )
                    assert response.status_code == 200
                    load_data = response.json()
                    assert load_data["success"] is True
                    
                    # Test unload endpoint
                    response = await client.delete(
                        "/autoscaler/unload",
                        params={"model_name": "test_model"}
                    )
                    assert response.status_code == 200
                    unload_data = response.json()
                    assert unload_data["success"] is True
    
    @pytest.mark.asyncio
    async def test_concurrent_autoscaler_requests(self, mock_autoscaler_for_server):
        """Test concurrent requests to autoscaler endpoints."""
        with patch('main.autoscaler', mock_autoscaler_for_server):
            with patch('main.initialize_inference_engine', AsyncMock()):
                # Set the autoscaler directly after patching
                import main
                main.autoscaler = mock_autoscaler_for_server
                
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    
                    # Make concurrent requests
                    tasks = [
                        client.get("/autoscaler/health"),
                        client.get("/autoscaler/stats"),
                        client.get("/autoscaler/metrics"),
                        client.post("/autoscaler/scale", params={"model_name": "model1", "target_instances": 2}),
                        client.post("/autoscaler/scale", params={"model_name": "model2", "target_instances": 3}),
                    ]
                    
                    responses = await asyncio.gather(*tasks)
                    
                    # All requests should succeed
                    assert all(response.status_code == 200 for response in responses)
                    
                    # Check that scaling was called for both models
                    assert mock_autoscaler_for_server.scale_model.call_count == 2


class TestAutoscalerServerStartupShutdown:
    """Test autoscaler server startup and shutdown integration."""
    
    @pytest.mark.asyncio
    async def test_server_startup_with_autoscaler(self):
        """Test server startup with autoscaler initialization."""
        from main import app, initialize_inference_engine, cleanup_inference_engine
        
        # Create mock objects
        mock_autoscaler = Mock()
        mock_autoscaler.start = AsyncMock()
        mock_autoscaler.stop = AsyncMock()
        
        # Mock the initialize and cleanup functions to test lifecycle
        with patch('main.autoscaler', None) as mock_autoscaler_global:
            with patch('main.initialize_inference_engine') as mock_init:
                with patch('main.cleanup_inference_engine') as mock_cleanup:
                    # Simulate the lifespan function behavior
                    from main import lifespan
                    
                    # Test lifespan context manager
                    async with lifespan(app):
                        # Verify initialization was called
                        mock_init.assert_called_once()
                    
                    # Verify cleanup was called
                    mock_cleanup.assert_called_once()
    
    def test_server_endpoints_without_autoscaler(self):
        """Test server endpoints when autoscaler is not available."""
        import main
        from main import app
        
        # Patch both the autoscaler and the initialize function to prevent startup
        with patch.object(main, 'autoscaler', None):
            with patch.object(main, 'initialize_inference_engine', return_value=None):
                with TestClient(app) as client:
                    
                    # Health endpoint should return autoscaler not available
                    response = client.get("/autoscaler/health")
                    assert response.status_code == 200  # Health check itself should not fail
                    data = response.json()
                    assert data['healthy'] is False
                    assert "not available" in data["error"].lower()
                    
                    # Other endpoints should also handle missing autoscaler
                    response = client.get("/autoscaler/stats")
                    assert response.status_code == 503
                    
                    response = client.get("/autoscaler/metrics")
                    assert response.status_code == 503
                    
                    response = client.post(
                        "/autoscaler/scale",
                        params={"model_name": "test", "target_instances": 2}
                    )
                    assert response.status_code == 503


class TestAutoscalerServerDocumentation:
    """Test autoscaler API documentation endpoints."""
    
    def test_openapi_includes_autoscaler_endpoints(self, client_with_mock_autoscaler):
        """Test that OpenAPI spec includes autoscaler endpoints."""
        response = client_with_mock_autoscaler.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        paths = openapi_spec["paths"]
        
        # Check that autoscaler endpoints are documented
        assert "/autoscaler/health" in paths
        assert "/autoscaler/stats" in paths
        assert "/autoscaler/metrics" in paths
        assert "/autoscaler/scale" in paths
        assert "/autoscaler/load" in paths
        assert "/autoscaler/unload" in paths
        
        # Check endpoint details
        health_endpoint = paths["/autoscaler/health"]["get"]
        assert "summary" in health_endpoint
        assert "responses" in health_endpoint
        
        scale_endpoint = paths["/autoscaler/scale"]["post"]
        assert "parameters" in scale_endpoint
        assert len(scale_endpoint["parameters"]) >= 2  # model_name and target_instances


if __name__ == "__main__":
    pytest.main([__file__])

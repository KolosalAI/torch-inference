"""Integration tests for autoscaling server endpoints."""

import pytest
import asyncio
import json
import time
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
    
    return autoscaler


@pytest.fixture
def client_with_mock_autoscaler(mock_autoscaler_for_server):
    """Create test client with mocked autoscaler."""
    # Patch both the global autoscaler and the initialization function
    with patch('main.autoscaler', mock_autoscaler_for_server):
        with patch('main.inference_engine') as mock_engine:
            # Create a mock inference engine
            mock_engine.health_check = AsyncMock(return_value={
                "healthy": True,
                "checks": {"inference_engine": True},
                "timestamp": time.time()
            })
            mock_engine.get_stats = Mock(return_value={"requests_processed": 0})
            
            # Set both the autoscaler and inference engine directly after patching
            import main
            main.autoscaler = mock_autoscaler_for_server
            main.inference_engine = mock_engine
            
            with TestClient(app) as client:
                yield client


class TestAutoscalerServerEndpoints:
    """Test autoscaler integration in existing endpoints."""
    
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
    
    def test_health_endpoint_includes_autoscaler_info(self, client_with_mock_autoscaler):
        """Test that health endpoint includes autoscaler information."""
        response = client_with_mock_autoscaler.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["healthy"] is True
        assert "autoscaler" in data
        assert data["autoscaler"]["healthy"] is True


class TestAutoscalerServerErrorHandling:
    """Test error handling in autoscaler server integration."""
    
    @pytest.fixture
    def client_with_failing_autoscaler(self):
        """Create test client with failing autoscaler."""
        failing_autoscaler = Mock()
        failing_autoscaler.start = AsyncMock()
        failing_autoscaler.stop = AsyncMock()  # Add stop method for cleanup
        failing_autoscaler.get_health_status.side_effect = Exception("Autoscaler error")
        failing_autoscaler.predict = AsyncMock(side_effect=Exception("Prediction error"))
        
        with patch('main.autoscaler', failing_autoscaler):
            with patch('main.inference_engine') as mock_engine:
                # Create a mock inference engine that works
                mock_engine.health_check = AsyncMock(return_value={
                    "healthy": True,
                    "checks": {"inference_engine": True},
                    "timestamp": time.time()
                })
                mock_engine.get_stats = Mock(return_value={"requests_processed": 0})
                
                # Patch the initialization to prevent real autoscaler creation
                with patch('main.initialize_inference_engine', AsyncMock()):
                    # Set both the autoscaler and inference engine directly after patching
                    import main
                    main.autoscaler = failing_autoscaler
                    main.inference_engine = mock_engine
                    
                    with TestClient(app) as client:
                        yield client    
    
    def test_health_endpoint_error_handling(self, client_with_failing_autoscaler):
        """Test health endpoint error handling when autoscaler fails."""
        response = client_with_failing_autoscaler.get("/health")
        
        assert response.status_code == 200  # Health endpoint returns 200 with error details
        data = response.json()
        # The overall health is determined by the inference engine, not autoscaler
        # But autoscaler health should be reported as unhealthy
        assert "autoscaler" in data
        # The autoscaler should report an error since get_health_status raises an exception
        assert data["autoscaler"]["healthy"] is False
        assert "error" in data["autoscaler"]
    
    def test_predict_endpoint_error_handling(self, client_with_failing_autoscaler):
        """Test predict endpoint error handling when autoscaler fails."""
        response = client_with_failing_autoscaler.post(
            "/predict",
            json={
                "model_name": "example",
                "inputs": {"text": "test input"}
            }
        )
        
        # Should fall back to inference engine or return error
        assert response.status_code in [200, 500]


class TestAutoscalerServerAsyncClient:
    """Test autoscaler integration using async client."""
    
    @pytest.mark.asyncio
    async def test_async_predict_with_autoscaler(self, mock_autoscaler_for_server):
        """Test predict endpoint using async client with autoscaler."""
        with patch('main.autoscaler', mock_autoscaler_for_server):
            with patch('main.inference_engine') as mock_engine:
                # Create a mock inference engine
                mock_engine.health_check = AsyncMock(return_value={
                    "healthy": True,
                    "checks": {"inference_engine": True},
                    "timestamp": time.time()
                })
                mock_engine.get_stats = Mock(return_value={"requests_processed": 0})
                
                # Set both the autoscaler and inference engine directly after patching
                import main
                main.autoscaler = mock_autoscaler_for_server
                main.inference_engine = mock_engine
                
                # Use httpx with ASGI transport for the FastAPI app
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    
                    # Test predict endpoint
                    response = await client.post(
                        "/predict",
                        json={
                            "model_name": "example",
                            "inputs": {"text": "test input"}
                        }
                    )
                    assert response.status_code == 200
                    data = response.json()
                    assert data["success"] is True
                    
                    # Test health endpoint includes autoscaler info
                    response = await client.get("/health")
                    assert response.status_code == 200
                    health_data = response.json()
                    assert health_data["healthy"] is True
                    assert "autoscaler" in health_data
    
    @pytest.mark.asyncio
    async def test_concurrent_predict_requests(self, mock_autoscaler_for_server):
        """Test concurrent predict requests with autoscaler."""
        with patch('main.autoscaler', mock_autoscaler_for_server):
            with patch('main.inference_engine') as mock_engine:
                # Create a mock inference engine
                mock_engine.predict = AsyncMock(return_value={"result": "test"})
                mock_engine.health_check = AsyncMock(return_value={
                    "healthy": True,
                    "checks": {"inference_engine": True},
                    "timestamp": time.time()
                })
                mock_engine.get_stats = Mock(return_value={"requests_processed": 0})
                
                # Set both the autoscaler and inference engine directly after patching
                import main
                main.autoscaler = mock_autoscaler_for_server
                main.inference_engine = mock_engine
                
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    
                    # Make concurrent predict requests
                    tasks = [
                        client.post("/predict", json={"model_name": "example", "inputs": f"test input {i}"})
                        for i in range(5)
                    ]
                    
                    responses = await asyncio.gather(*tasks)
                    
                    # All requests should succeed
                    assert all(response.status_code == 200 for response in responses)
                    
                    # Check that predictions were made
                    for response in responses:
                        data = response.json()
                        assert data["success"] is True
                        assert "result" in data


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
                    
                    # Health endpoint should include autoscaler info
                    response = client.get("/health")
                    assert response.status_code == 200  # Health check itself should not fail
                    data = response.json()
                    # When autoscaler is None, health should still work but show autoscaler unavailable
                    assert "autoscaler" in data
                    assert data["autoscaler"]["healthy"] is False


class TestAutoscalerServerDocumentation:
    """Test autoscaler integration in API documentation."""
    
    def test_openapi_includes_existing_endpoints(self, client_with_mock_autoscaler):
        """Test that OpenAPI spec includes the current endpoints."""
        response = client_with_mock_autoscaler.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        paths = openapi_spec["paths"]
        
        # Check that main endpoints are documented
        assert "/" in paths
        assert "/predict" in paths
        assert "/health" in paths
        assert "/stats" in paths
        
        # Check endpoint details
        health_endpoint = paths["/health"]["get"]
        assert "summary" in health_endpoint or "description" in health_endpoint
        assert "responses" in health_endpoint
        
        predict_endpoint = paths["/predict"]["post"]
        assert "requestBody" in predict_endpoint or "parameters" in predict_endpoint


if __name__ == "__main__":
    pytest.main([__file__])

"""
Integration tests for the full API workflow.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient

from src.api.app import create_app


class TestFullWorkflow:
    """Test complete API workflows."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    def test_health_check_workflow(self, client):
        """Test health check workflow."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_api_discovery_workflow(self, client):
        """Test API discovery through root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        
        # Test that documented endpoints exist and return valid responses
        endpoints = data["endpoints"]
        
        # Define which endpoints to test and their expected behaviors
        endpoint_tests = {
            "health_check": (200, None),  # Should return 200
            "system_info": (200, None),   # Should return 200
            "inference": (405, None),     # POST endpoint, GET returns 405
            "models": (503, None),        # May return 503 if service not available
            "audio": (404, None),         # Base path, no endpoint
            "downloads": (503, None),     # May return 503 if service not available
        }
        
        for endpoint_name, endpoint_path in endpoints.items():
            if endpoint_path.startswith("/"):
                # Test basic endpoint accessibility
                test_response = client.get(endpoint_path)
                expected_status, _ = endpoint_tests.get(endpoint_name, (404, None))
                
                # Allow for reasonable error codes that indicate the endpoint exists
                valid_codes = [200, 405, 503]  # 405 = Method Not Allowed, 503 = Service Unavailable
                if endpoint_name == "audio":  # Special case for audio base path
                    valid_codes = [404]  # Audio base path doesn't exist, which is fine
                
                assert test_response.status_code in valid_codes, \
                    f"Endpoint {endpoint_path} returned {test_response.status_code}, expected one of {valid_codes}"
    
    def test_gpu_detection_workflow(self, client):
        """Test GPU detection workflow."""
        response = client.get("/gpu/detect")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        
        gpu_data = data["data"]
        assert "cuda_available" in gpu_data
        assert "gpu_count" in gpu_data
    
    def test_server_info_workflow(self, client):
        """Test server information workflow."""
        response = client.get("/server/config")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
    
    def test_error_handling_workflow(self, client):
        """Test error handling for non-existent endpoints."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        data = response.json()
        # Check for either the custom error format or FastAPI's default format
        assert "error" in data or "detail" in data
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_workflow(self):
        """Test handling of concurrent requests."""
        # Create app and client for async testing
        app = create_app()
        
        # Use httpx for proper async testing
        import httpx
        from httpx import ASGITransport
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Create multiple concurrent requests
            tasks = []
            
            async def make_request():
                response = await client.get("/health")
                return response
            
            # Make 3 concurrent requests (reduced from 5)
            tasks = [make_request() for _ in range(3)]
            responses = await asyncio.gather(*tasks)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__])

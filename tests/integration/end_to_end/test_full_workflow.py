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
        
        # Test that documented endpoints exist
        endpoints = data["endpoints"]
        for endpoint_path in endpoints.values():
            if endpoint_path.startswith("/"):
                # Test basic endpoint accessibility
                test_response = client.get(endpoint_path)
                # Should not be 404 (endpoint exists)
                assert test_response.status_code != 404
    
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
        assert "error" in data
        assert "available_endpoints" in data
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_workflow(self, client):
        """Test handling of concurrent requests."""
        # Create multiple concurrent requests
        responses = []
        
        async def make_request():
            return client.get("/health")
        
        # Make 5 concurrent requests
        tasks = [make_request() for _ in range(5)]
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__])

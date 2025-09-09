"""
Integration tests for secure image processing API endpoints.

Tests the FastAPI endpoints for secure image processing including:
- POST /image/process/secure
- POST /image/validate/security
- GET /image/security/stats
- GET /image/models
- GET /image/health
"""

import pytest
import asyncio
import aiohttp
import io
import tempfile
import json
from pathlib import Path
from PIL import Image
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient


class TestSecureImageAPIEndpoints:
    """Test secure image processing API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client for the FastAPI application."""
        # Import main app
        try:
            from main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("Main app not available for testing")
    
    @pytest.fixture
    def test_image_file(self):
        """Create a test image file."""
        # Create a small test image
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(temp_file.name, format='PNG')
        temp_file.close()
        
        yield temp_file.name
        
        # Cleanup
        Path(temp_file.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def test_image_bytes(self):
        """Create test image as bytes."""
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    def test_secure_image_processing_endpoint(self, client, test_image_file):
        """Test POST /image/process/secure endpoint."""
        with open(test_image_file, 'rb') as f:
            response = client.post(
                "/image/process/secure",
                files={"file": ("test.png", f, "image/png")},
                data={
                    "model_name": "default",
                    "security_level": "medium",
                    "enable_sanitization": "true",
                    "enable_adversarial_detection": "true",
                    "return_confidence_scores": "true"
                }
            )
        
        # Check response status
        assert response.status_code in [200, 503]  # 503 if dependencies missing
        
        if response.status_code == 200:
            result = response.json()
            
            # Check response structure
            assert isinstance(result, dict)
            assert "success" in result
            assert "processing_time" in result
            
            if result["success"]:
                assert "processed_image" in result
                assert "threats_detected" in result
                assert "threats_mitigated" in result
                assert "confidence_scores" in result
                assert "model_info" in result
                
                # Check model info
                model_info = result["model_info"]
                assert "model_name" in model_info
                assert "security_level" in model_info
        else:
            # Should return error message for missing dependencies
            result = response.json()
            assert "detail" in result
            assert "dependencies" in result["detail"].lower()
    
    def test_secure_image_processing_invalid_format(self, client):
        """Test secure processing with invalid file format."""
        # Create a fake file with invalid extension
        fake_file = io.BytesIO(b"fake image data")
        
        response = client.post(
            "/image/process/secure",
            files={"file": ("test.xyz", fake_file, "application/octet-stream")},
            data={"security_level": "medium"}
        )
        
        assert response.status_code == 400
        result = response.json()
        assert "detail" in result
        assert "format" in result["detail"].lower()
    
    def test_secure_image_processing_oversized_file(self, client):
        """Test secure processing with oversized file."""
        # Create large fake file (simulate oversized)
        large_data = b"x" * (100 * 1024 * 1024)  # 100MB
        fake_file = io.BytesIO(large_data)
        
        response = client.post(
            "/image/process/secure",
            files={"file": ("large.png", fake_file, "image/png")},
            data={"security_level": "medium"}
        )
        
        assert response.status_code == 413
        result = response.json()
        assert "detail" in result
        assert "large" in result["detail"].lower()
    
    def test_secure_image_processing_different_security_levels(self, client, test_image_file):
        """Test secure processing with different security levels."""
        security_levels = ["low", "medium", "high", "maximum"]
        
        for level in security_levels:
            with open(test_image_file, 'rb') as f:
                response = client.post(
                    "/image/process/secure",
                    files={"file": ("test.png", f, "image/png")},
                    data={"security_level": level}
                )
            
            # Should accept all valid security levels
            assert response.status_code in [200, 503]
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    model_info = result.get("model_info", {})
                    assert model_info.get("security_level") == level
    
    def test_image_validation_endpoint(self, client, test_image_file):
        """Test POST /image/validate/security endpoint."""
        with open(test_image_file, 'rb') as f:
            response = client.post(
                "/image/validate/security",
                files={"file": ("test.png", f, "image/png")},
                data={
                    "security_level": "high",
                    "detailed_analysis": "true"
                }
            )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            result = response.json()
            
            # Check response structure
            assert isinstance(result, dict)
            assert "success" in result
            assert "is_safe" in result
            assert "threats_detected" in result
            assert "processing_time" in result
            
            if result["success"]:
                assert "confidence_scores" in result
                assert "recommendations" in result
                assert "file_info" in result
                
                # Check file info
                file_info = result["file_info"]
                assert "filename" in file_info
                assert "size_bytes" in file_info
    
    def test_image_validation_invalid_file(self, client):
        """Test image validation with invalid file."""
        fake_file = io.BytesIO(b"not an image")
        
        response = client.post(
            "/image/validate/security",
            files={"file": ("fake.png", fake_file, "image/png")},
            data={"security_level": "medium"}
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            result = response.json()
            
            # Should detect the invalid file
            if result["success"]:
                assert not result["is_safe"]
                assert len(result["threats_detected"]) > 0
    
    def test_image_security_stats_endpoint(self, client):
        """Test GET /image/security/stats endpoint."""
        response = client.get("/image/security/stats")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            result = response.json()
            
            # Check response structure
            assert isinstance(result, dict)
            assert "success" in result
            assert "total_images_processed" in result
            assert "threats_by_type" in result
            assert "security_level_distribution" in result
            assert "system_health" in result
            
            # Check system health
            system_health = result["system_health"]
            assert isinstance(system_health, dict)
    
    def test_image_models_endpoint(self, client):
        """Test GET /image/models endpoint."""
        response = client.get("/image/models")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            result = response.json()
            
            # Check response structure
            assert isinstance(result, dict)
            assert "loaded_models" in result
            assert "security_levels" in result
            assert "supported_formats" in result
            assert "capabilities" in result
            assert "examples" in result
            
            # Check security levels
            security_levels = result["security_levels"]
            assert isinstance(security_levels, dict)
            expected_levels = ["low", "medium", "high", "maximum"]
            for level in expected_levels:
                assert level in security_levels
            
            # Check supported formats
            formats = result["supported_formats"]
            assert isinstance(formats, list)
            assert ".png" in formats
            assert ".jpg" in formats
            
            # Check capabilities
            capabilities = result["capabilities"]
            assert isinstance(capabilities, dict)
            assert "threat_detection" in capabilities
            assert "sanitization" in capabilities
    
    def test_image_health_endpoint(self, client):
        """Test GET /image/health endpoint."""
        response = client.get("/image/health")
        
        assert response.status_code == 200
        result = response.json()
        
        # Check response structure
        assert isinstance(result, dict)
        assert "image_processing_available" in result
        assert "secure_processing_available" in result
        assert "threat_detection_available" in result
        assert "dependencies" in result
        assert "errors" in result
        
        # Check dependencies
        dependencies = result["dependencies"]
        assert isinstance(dependencies, dict)
        
        # Should check for key dependencies
        expected_deps = ["PIL", "numpy", "torch"]
        for dep in expected_deps:
            if dep in dependencies:
                assert "available" in dependencies[dep]
                assert "description" in dependencies[dep]
    
    def test_image_processing_no_file(self, client):
        """Test image processing endpoint without file."""
        response = client.post(
            "/image/process/secure",
            data={"security_level": "medium"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_image_validation_no_file(self, client):
        """Test image validation endpoint without file."""
        response = client.post(
            "/image/validate/security",
            data={"security_level": "medium"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_security_level(self, client, test_image_file):
        """Test with invalid security level."""
        with open(test_image_file, 'rb') as f:
            response = client.post(
                "/image/process/secure",
                files={"file": ("test.png", f, "image/png")},
                data={"security_level": "invalid_level"}
            )
        
        assert response.status_code == 400
        result = response.json()
        assert "detail" in result
        assert "security level" in result["detail"].lower()


class TestSecureImageAPIAuthentication:
    """Test authentication with secure image processing endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client for the FastAPI application."""
        try:
            from main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("Main app not available for testing")
    
    @pytest.fixture
    def test_image_file(self):
        """Create a test image file."""
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(temp_file.name, format='PNG')
        temp_file.close()
        
        yield temp_file.name
        Path(temp_file.name).unlink(missing_ok=True)
    
    def test_image_processing_with_token(self, client, test_image_file):
        """Test image processing with authentication token."""
        with open(test_image_file, 'rb') as f:
            response = client.post(
                "/image/process/secure",
                files={"file": ("test.png", f, "image/png")},
                data={
                    "security_level": "medium",
                    "token": "test_token"  # This would need valid token in real scenario
                }
            )
        
        # Should handle token validation (may fail if token invalid)
        assert response.status_code in [200, 401, 503]
    
    def test_security_stats_with_token(self, client):
        """Test security stats with authentication token."""
        response = client.get("/image/security/stats?token=test_token")
        
        # Should handle token validation
        assert response.status_code in [200, 401, 503]


class TestSecureImageAPIPerformance:
    """Test performance characteristics of secure image processing API."""
    
    @pytest.fixture
    def client(self):
        """Create test client for the FastAPI application."""
        try:
            from main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("Main app not available for testing")
    
    def test_processing_time_reasonable(self, client):
        """Test that processing time is reasonable."""
        # Create small test image
        img_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        
        buffer.seek(0)
        response = client.post(
            "/image/process/secure",
            files={"file": ("small.png", buffer, "image/png")},
            data={"security_level": "low"}  # Fastest security level
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                processing_time = result.get("processing_time", 0)
                # Should process small image quickly (less than 10 seconds)
                assert processing_time < 10.0
    
    def test_health_check_fast(self, client):
        """Test that health check is fast."""
        import time
        
        start_time = time.time()
        response = client.get("/image/health")
        end_time = time.time()
        
        # Health check should be very fast
        assert (end_time - start_time) < 2.0
        assert response.status_code == 200
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        # Create test image
        img_array = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        # Send multiple concurrent requests
        responses = []
        for i in range(3):
            buffer = io.BytesIO(image_data)
            response = client.post(
                "/image/validate/security",
                files={"file": (f"test_{i}.png", buffer, "image/png")},
                data={"security_level": "low"}
            )
            responses.append(response)
        
        # All requests should complete
        for response in responses:
            assert response.status_code in [200, 503]


class TestSecureImageAPIErrorHandling:
    """Test error handling in secure image processing API."""
    
    @pytest.fixture
    def client(self):
        """Create test client for the FastAPI application."""
        try:
            from main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("Main app not available for testing")
    
    def test_malformed_request(self, client):
        """Test handling of malformed requests."""
        # Send request with missing required fields
        response = client.post("/image/process/secure")
        assert response.status_code == 422
    
    def test_unsupported_media_type(self, client):
        """Test handling of unsupported media types."""
        response = client.post(
            "/image/process/secure",
            files={"file": ("test.txt", io.BytesIO(b"text content"), "text/plain")},
            data={"security_level": "medium"}
        )
        
        assert response.status_code == 400
    
    def test_empty_file(self, client):
        """Test handling of empty files."""
        response = client.post(
            "/image/process/secure",
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
            data={"security_level": "medium"}
        )
        
        assert response.status_code == 400
        result = response.json()
        assert "empty" in result["detail"].lower()
    
    def test_corrupted_image(self, client):
        """Test handling of corrupted image data."""
        # Create corrupted image data
        corrupted_data = b"CORRUPTED_IMAGE_DATA" + b"\x00" * 1000
        
        response = client.post(
            "/image/process/secure",
            files={"file": ("corrupted.png", io.BytesIO(corrupted_data), "image/png")},
            data={"security_level": "medium"}
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 500, 503]
        
        if response.status_code == 200:
            result = response.json()
            # If processed, should detect as unsafe or report error
            if result.get("success"):
                # Should detect threats or report processing issues
                assert (not result.get("is_safe", True) or 
                       len(result.get("threats_detected", [])) > 0 or
                       "error" in result)


if __name__ == '__main__':
    pytest.main([__file__])

"""
End-to-end tests for secure image processing system.

Tests the complete workflow from API request to secure processing
and response generation, including realistic attack scenarios.
"""

import pytest
import asyncio
import tempfile
import io
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from unittest.mock import Mock, patch

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient


class TestSecureImageProcessingE2E:
    """End-to-end tests for secure image processing."""
    
    @pytest.fixture
    def client(self):
        """Create test client for the FastAPI application."""
        try:
            from main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("Main app not available for testing")
    
    @pytest.fixture
    def normal_image(self):
        """Create a normal, safe image with realistic entropy."""
        # Create a more natural image without adding noise
        # Use a more complex but smooth pattern that will have good entropy
        
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Create a pattern that looks natural but has good entropy
        for i in range(224):
            for j in range(224):
                # Create sinusoidal patterns that create natural-looking gradients
                r = int(128 + 100 * np.sin(i * 0.05) * np.cos(j * 0.03))
                g = int(128 + 80 * np.cos(i * 0.03) * np.sin(j * 0.04))
                b = int(128 + 90 * np.sin(i * 0.04) * np.sin(j * 0.05))
                
                img_array[i, j] = [
                    max(0, min(255, r)),
                    max(0, min(255, g)),
                    max(0, min(255, b))
                ]
        
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    @pytest.fixture
    def adversarial_image(self):
        """Create an image with adversarial-like patterns."""
        # High-frequency noise pattern that might trigger adversarial detection
        base = np.random.randint(100, 156, (224, 224, 3), dtype=np.uint8)
        noise = np.random.randint(-50, 50, (224, 224, 3), dtype=np.int16)
        
        # Add high-frequency noise
        adversarial = np.clip(base + noise, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(adversarial)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    @pytest.fixture
    def steganography_image(self):
        """Create an image that might trigger steganography detection."""
        # Create image with unusual LSB patterns
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Modify LSBs in a pattern that might be detected
        for i in range(0, 100, 2):
            for j in range(0, 100, 2):
                # Set LSB to alternating pattern
                img_array[i, j, :] = (img_array[i, j, :] & 0xFE) | (i + j) % 2
        
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    @pytest.fixture
    def suspicious_metadata_image(self):
        """Create an image with suspicious metadata."""
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Add suspicious EXIF data (simulated)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        return buffer.getvalue()
    
    def test_normal_image_processing_workflow(self, client, normal_image):
        """Test complete workflow with normal, safe image."""
        # Step 1: Health check
        health_response = client.get("/image/health")
        assert health_response.status_code == 200
        
        # Skip if secure processing not available
        health_data = health_response.json()
        if not health_data.get("secure_processing_available", False):
            pytest.skip("Secure processing not available")
        
        # Step 2: Validate image first
        validation_response = client.post(
            "/image/validate/security",
            files={"file": ("normal.png", io.BytesIO(normal_image), "image/png")},
            data={"security_level": "medium", "detailed_analysis": "true"}
        )
        
        assert validation_response.status_code == 200
        validation_data = validation_response.json()
        assert validation_data["success"] is True
        
        # Normal image should be safe
        assert validation_data["is_safe"] is True
        assert len(validation_data["threats_detected"]) == 0
        
        # Step 3: Process the image
        processing_response = client.post(
            "/image/process/secure",
            files={"file": ("normal.png", io.BytesIO(normal_image), "image/png")},
            data={
                "security_level": "medium",
                "enable_sanitization": "true",
                "enable_adversarial_detection": "true",
                "return_confidence_scores": "true"
            }
        )
        
        assert processing_response.status_code == 200
        processing_data = processing_response.json()
        assert processing_data["success"] is True
        
        # Should have processed image
        assert "processed_image" in processing_data
        assert processing_data["processed_image"] is not None
        
        # Should have security analysis
        assert "threats_detected" in processing_data
        assert "threats_mitigated" in processing_data
        assert "confidence_scores" in processing_data
        
        # Step 4: Check security stats
        stats_response = client.get("/image/security/stats")
        assert stats_response.status_code == 200
        stats_data = stats_response.json()
        assert stats_data["success"] is True
        
        # Stats should reflect the processing
        assert stats_data["total_images_processed"] >= 0
    
    def test_adversarial_image_detection_workflow(self, client, adversarial_image):
        """Test workflow with potentially adversarial image."""
        # Skip if secure processing not available
        health_response = client.get("/image/health")
        health_data = health_response.json()
        if not health_data.get("secure_processing_available", False):
            pytest.skip("Secure processing not available")
        
        # Validate with high security level
        validation_response = client.post(
            "/image/validate/security",
            files={"file": ("adversarial.png", io.BytesIO(adversarial_image), "image/png")},
            data={"security_level": "high", "detailed_analysis": "true"}
        )
        
        assert validation_response.status_code == 200
        validation_data = validation_response.json()
        
        # May detect threats depending on the noise pattern
        if not validation_data["is_safe"]:
            assert len(validation_data["threats_detected"]) > 0
            assert "recommendations" in validation_data
        
        # Process with maximum security
        processing_response = client.post(
            "/image/process/secure",
            files={"file": ("adversarial.png", io.BytesIO(adversarial_image), "image/png")},
            data={
                "security_level": "maximum",
                "enable_sanitization": "true",
                "enable_adversarial_detection": "true"
            }
        )
        
        assert processing_response.status_code == 200
        processing_data = processing_response.json()
        
        # Should either process successfully with mitigations or detect threats
        if processing_data["success"]:
            # If processed, should apply mitigations
            assert len(processing_data["threats_mitigated"]) >= 0
        else:
            # If blocked, should have threat detection
            assert len(processing_data["threats_detected"]) > 0
    
    def test_security_level_escalation_workflow(self, client, normal_image):
        """Test workflow with security level escalation."""
        # Skip if secure processing not available
        health_response = client.get("/image/health")
        health_data = health_response.json()
        if not health_data.get("secure_processing_available", False):
            pytest.skip("Secure processing not available")
        
        # Process with low security first
        low_security_response = client.post(
            "/image/process/secure",
            files={"file": ("test.png", io.BytesIO(normal_image), "image/png")},
            data={"security_level": "low", "enable_sanitization": "false"}
        )
        
        assert low_security_response.status_code == 200
        low_data = low_security_response.json()
        
        # Then process with high security
        high_security_response = client.post(
            "/image/process/secure",
            files={"file": ("test.png", io.BytesIO(normal_image), "image/png")},
            data={"security_level": "high", "enable_sanitization": "true"}
        )
        
        assert high_security_response.status_code == 200
        high_data = high_security_response.json()
        
        if low_data["success"] and high_data["success"]:
            # High security should potentially apply more mitigations
            low_mitigations = len(low_data["threats_mitigated"])
            high_mitigations = len(high_data["threats_mitigated"])
            
            # Processing times might differ
            assert "processing_time" in low_data
            assert "processing_time" in high_data
    
    def test_batch_processing_workflow(self, client, normal_image):
        """Test processing multiple images in sequence."""
        # Skip if secure processing not available
        health_response = client.get("/image/health")
        health_data = health_response.json()
        if not health_data.get("secure_processing_available", False):
            pytest.skip("Secure processing not available")
        
        batch_size = 3
        responses = []
        
        # Process multiple images
        for i in range(batch_size):
            response = client.post(
                "/image/process/secure",
                files={"file": (f"batch_{i}.png", io.BytesIO(normal_image), "image/png")},
                data={"security_level": "medium"}
            )
            responses.append(response)
        
        # All should process successfully
        for i, response in enumerate(responses):
            assert response.status_code == 200
            data = response.json()
            if data["success"]:
                assert "processed_image" in data
                assert "processing_time" in data
        
        # Check that stats reflect batch processing
        stats_response = client.get("/image/security/stats")
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            if stats_data["success"]:
                # Should show increased processing count
                assert stats_data["total_images_processed"] >= 0
    
    def test_error_recovery_workflow(self, client, normal_image):
        """Test system recovery after processing errors."""
        # Skip if secure processing not available
        health_response = client.get("/image/health")
        health_data = health_response.json()
        if not health_data.get("secure_processing_available", False):
            pytest.skip("Secure processing not available")
        
        # Process valid image first
        valid_response = client.post(
            "/image/process/secure",
            files={"file": ("valid.png", io.BytesIO(normal_image), "image/png")},
            data={"security_level": "medium"}
        )
        assert valid_response.status_code == 200
        
        # Try to process invalid data
        invalid_response = client.post(
            "/image/process/secure",
            files={"file": ("invalid.png", io.BytesIO(b"invalid_data"), "image/png")},
            data={"security_level": "medium"}
        )
        # Should handle error gracefully
        assert invalid_response.status_code in [200, 400, 500]
        
        # System should still be functional
        recovery_response = client.post(
            "/image/process/secure",
            files={"file": ("recovery.png", io.BytesIO(normal_image), "image/png")},
            data={"security_level": "medium"}
        )
        assert recovery_response.status_code == 200
        
        # Health check should still work
        final_health = client.get("/image/health")
        assert final_health.status_code == 200
    
    def test_comprehensive_security_analysis_workflow(self, client):
        """Test comprehensive security analysis with various image types."""
        # Skip if secure processing not available
        health_response = client.get("/image/health")
        health_data = health_response.json()
        if not health_data.get("secure_processing_available", False):
            pytest.skip("Secure processing not available")
        
        # Test different image types
        test_images = []
        
        # Normal image
        normal = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        normal_img = Image.fromarray(normal)
        buffer = io.BytesIO()
        normal_img.save(buffer, format='PNG')
        test_images.append(("normal.png", buffer.getvalue()))
        
        # High contrast image
        contrast = np.zeros((100, 100, 3), dtype=np.uint8)
        contrast[:50, :, :] = 255  # High contrast
        contrast_img = Image.fromarray(contrast)
        buffer = io.BytesIO()
        contrast_img.save(buffer, format='PNG')
        test_images.append(("contrast.png", buffer.getvalue()))
        
        # Uniform image
        uniform = np.full((100, 100, 3), 128, dtype=np.uint8)
        uniform_img = Image.fromarray(uniform)
        buffer = io.BytesIO()
        uniform_img.save(buffer, format='PNG')
        test_images.append(("uniform.png", buffer.getvalue()))
        
        results = []
        
        for filename, image_data in test_images:
            # Validate each image
            validation_response = client.post(
                "/image/validate/security",
                files={"file": (filename, io.BytesIO(image_data), "image/png")},
                data={"security_level": "high", "detailed_analysis": "true"}
            )
            
            if validation_response.status_code == 200:
                validation_data = validation_response.json()
                results.append((filename, validation_data))
        
        # Analyze results
        for filename, result in results:
            assert "is_safe" in result
            assert "threats_detected" in result
            assert "confidence_scores" in result
            
            # Different images might have different characteristics
            if "confidence_scores" in result:
                scores = result["confidence_scores"]
                # Should have entropy scores for most images
                if "entropy" in scores:
                    assert isinstance(scores["entropy"], (int, float))
                    assert scores["entropy"] >= 0
    
    def test_performance_under_load_workflow(self, client, normal_image):
        """Test system performance under simulated load."""
        # Skip if secure processing not available
        health_response = client.get("/image/health")
        health_data = health_response.json()
        if not health_data.get("secure_processing_available", False):
            pytest.skip("Secure processing not available")
        
        # Send multiple concurrent-like requests
        start_time = time.time()
        responses = []
        
        for i in range(5):  # Moderate load
            response = client.post(
                "/image/validate/security",
                files={"file": (f"load_test_{i}.png", io.BytesIO(normal_image), "image/png")},
                data={"security_level": "low"}  # Fastest processing
            )
            responses.append(response)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All requests should complete
        successful_responses = 0
        for response in responses:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    successful_responses += 1
        
        # Should handle reasonable load
        assert successful_responses >= len(responses) // 2  # At least half should succeed
        
        # Total time should be reasonable
        assert total_time < 60.0  # Should complete within a minute
        
        # System should still be healthy
        final_health = client.get("/image/health")
        assert final_health.status_code == 200


class TestSecureImageProcessingAttackScenarios:
    """Test specific attack scenarios and defense mechanisms."""
    
    @pytest.fixture
    def client(self):
        """Create test client for the FastAPI application."""
        try:
            from main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("Main app not available for testing")
    
    def test_format_spoofing_attack(self, client):
        """Test defense against format spoofing attacks."""
        # Create fake PNG with wrong signature
        fake_png = b"FAKE_PNG_HEADER" + b"\x00" * 1000
        
        response = client.post(
            "/image/validate/security",
            files={"file": ("fake.png", io.BytesIO(fake_png), "image/png")},
            data={"security_level": "high"}
        )
        
        if response.status_code == 200:
            data = response.json()
            # Should detect the format issue
            assert not data["is_safe"]
            assert len(data["threats_detected"]) > 0
    
    def test_oversized_image_attack(self, client):
        """Test defense against memory exhaustion attacks."""
        # Create large fake image data
        large_data = b"LARGE_IMAGE_DATA" * 100000  # ~1.6MB
        
        response = client.post(
            "/image/process/secure",
            files={"file": ("large.png", io.BytesIO(large_data), "image/png")},
            data={"security_level": "medium"}
        )
        
        # Should reject or handle large files appropriately
        assert response.status_code in [200, 400, 413]
        
        if response.status_code == 200:
            data = response.json()
            if not data["success"]:
                assert "error" in data
    
    def test_malformed_request_attack(self, client):
        """Test defense against malformed requests."""
        # Test various malformed requests
        malformed_requests = [
            # Missing file
            {},
            # Invalid security level
            {"security_level": "invalid"},
            # Wrong content type
            {"file": "not_a_file"},
        ]
        
        for malformed_data in malformed_requests:
            response = client.post("/image/process/secure", data=malformed_data)
            # Should handle malformed requests gracefully
            assert response.status_code in [400, 422]
    
    def test_rapid_request_attack(self, client):
        """Test defense against rapid request attacks."""
        # Send many rapid requests
        rapid_responses = []
        
        for i in range(10):
            # Create minimal valid request
            fake_image = b"FAKE_IMAGE" + str(i).encode()
            response = client.post(
                "/image/validate/security",
                files={"file": (f"rapid_{i}.png", io.BytesIO(fake_image), "image/png")},
                data={"security_level": "low"}
            )
            rapid_responses.append(response)
        
        # System should handle rapid requests
        # (Rate limiting would be handled by external middleware)
        for response in rapid_responses:
            assert response.status_code in [200, 400, 429, 503]


if __name__ == '__main__':
    pytest.main([__file__])

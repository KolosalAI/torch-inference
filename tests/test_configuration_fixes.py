#!/usr/bin/env python3
"""
Test script to verify that all configuration fixes are working correctly.
"""

import pytest
import os
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_audio_config_max_text_length():
    """Test that audio service uses configurable max text length."""
    from src.core.config import get_config
    
    config = get_config()
    assert hasattr(config.audio, 'max_text_length')
    assert config.audio.max_text_length == 5000  # Default value
    
    # Test that the value is used in audio service
    from src.models.api.audio import TTSRequest
    from pydantic import ValidationError
    
    # Should pass with text under limit
    short_text = "x" * 100
    request = TTSRequest(text=short_text)
    assert len(request.text) == 100
    
    # Should fail with text over limit (based on Pydantic validation)
    with pytest.raises(ValidationError):
        long_text = "x" * 6000  # Over the 5000 limit
        TTSRequest(text=long_text)

def test_inference_fallback_model():
    """Test that inference service uses configurable fallback model."""
    from src.core.config import get_config
    from src.services.inference import InferenceService
    
    config = get_config()
    assert hasattr(config.inference, 'fallback_model')
    assert config.inference.fallback_model == 'example'  # Default value
    
    # Test that inference service loads this config
    service = InferenceService()
    assert service.fallback_model == 'example'

def test_security_cors_configuration():
    """Test that security manager uses configurable CORS settings."""
    from src.core.config import get_config
    from src.core.security import SecurityConfig
    
    config = get_config()
    assert hasattr(config.security, 'allowed_origins')
    assert isinstance(config.security.allowed_origins, list)
    assert "*" not in config.security.allowed_origins  # No wildcards
    
    # Test that security config loads CORS settings
    security_config = SecurityConfig()
    assert isinstance(security_config.allowed_origins, list)
    assert "*" not in security_config.allowed_origins

def test_tts_validation_improvements():
    """Test that TTSRequest has proper validation."""
    from src.models.api.audio import TTSRequest
    from pydantic import ValidationError
    
    # Valid request should work
    request = TTSRequest(text="Hello world")
    assert request.text == "Hello world"
    
    # Empty text should fail
    with pytest.raises(ValidationError):
        TTSRequest(text="")
    
    # Text too long should fail
    with pytest.raises(ValidationError):
        TTSRequest(text="x" * 6000)

def test_environment_variable_configuration():
    """Test that environment variables can override test service URL."""
    # Test that we can override the service configuration
    test_host = "testhost"
    test_port = "9999"
    
    # Set environment variables first
    os.environ['TTS_SERVICE_HOST'] = test_host
    os.environ['TTS_SERVICE_PORT'] = test_port
    
    try:
        # Import the test module to verify it reads environment variables
        # Need to reload the module to pick up new environment variables
        import sys
        if 'tests.integration.test_tts_wav' in sys.modules:
            del sys.modules['tests.integration.test_tts_wav']
        
        from tests.integration.test_tts_wav import TTS_SERVICE_HOST, TTS_SERVICE_PORT, TTS_SERVICE_URL
        
        assert TTS_SERVICE_HOST == test_host
        assert TTS_SERVICE_PORT == test_port
        assert f"{test_host}:{test_port}" in TTS_SERVICE_URL
        
    finally:
        # Clean up
        if 'TTS_SERVICE_HOST' in os.environ:
            del os.environ['TTS_SERVICE_HOST']
        if 'TTS_SERVICE_PORT' in os.environ:
            del os.environ['TTS_SERVICE_PORT']

def test_configuration_structure():
    """Test that all configuration sections are properly structured."""
    from src.core.config import get_config
    
    config = get_config()
    
    # Test main sections exist
    assert hasattr(config, 'server')
    assert hasattr(config, 'security')
    assert hasattr(config, 'audio')
    assert hasattr(config, 'inference')
    
    # Test audio config
    assert hasattr(config.audio, 'max_text_length')
    assert hasattr(config.audio, 'default_tts_model')
    
    # Test inference config
    assert hasattr(config.inference, 'fallback_model')
    assert hasattr(config.inference, 'default_timeout')
    
    # Test security config
    assert hasattr(config.security, 'allowed_origins')
    assert hasattr(config.security, 'enable_cors')

def test_download_service_todo_comment():
    """Test that download service has proper TODO comments instead of placeholder text."""
    from src.services.download import DownloadService
    import inspect
    
    # Get the source code of the download service
    source = inspect.getsource(DownloadService._perform_download)
    
    # Verify that it contains proper TODO instead of placeholder comment
    assert "TODO: Implement actual download logic" in source
    assert "replace with actual implementation" not in source.lower()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

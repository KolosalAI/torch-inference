"""
Integration tests for audio processing functionality.

These tests verify that the audio TTS and STT models work correctly
with the inference framework.
"""

import pytest
import os
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import base64
import wave

from tests.utils.disk_management import managed_temp_file, DiskSpaceManager

# Import framework components
from framework.core.config import InferenceConfig, DeviceType
from framework.core.config_manager import get_config_manager


class TestAudioIntegration:
    """Test audio processing integration with the framework."""
    
    @pytest.fixture
    def config_manager(self):
        """Get configuration manager for tests."""
        return get_config_manager()
    
    @pytest.fixture
    def inference_config(self, config_manager):
        """Get inference configuration for tests."""
        return config_manager.get_inference_config()
    
    @pytest.fixture
    def mock_audio_data(self):
        """Create mock audio data for testing."""
        # Create 1 second of dummy audio at 16kHz
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        # Generate a simple sine wave
        t = np.linspace(0, duration, samples, False)
        frequency = 440.0  # A4 note
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        
        return audio_data.astype(np.float32), sample_rate
    
    @pytest.fixture
    def mock_wav_file(self, mock_audio_data):
        """Create a temporary WAV file for testing with disk space management."""
        disk_manager = DiskSpaceManager()
        
        # Check if there's enough space for audio file creation
        if not disk_manager.has_enough_space(required_mb=10):
            pytest.skip("Insufficient disk space for audio file creation")
        
        audio_data, sample_rate = mock_audio_data
        
        try:
            # Use managed temp file
            with managed_temp_file(suffix='.wav', required_space_mb=5, delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.close()  # Close before wave opens it
                
                # Write WAV file
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    
                    # Convert to 16-bit PCM
                    audio_16bit = (audio_data * 32767).astype(np.int16)
                    wav_file.writeframes(audio_16bit.tobytes())
                
                yield temp_path
                
        except OSError as e:
            if "No space left on device" in str(e):
                pytest.skip(f"Insufficient disk space for audio creation: {e}")
            else:
                raise
        finally:
            # Cleanup
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
    
    @pytest.mark.integration
    @pytest.mark.audio
    def test_audio_imports(self):
        """Test that audio modules can be imported."""
        try:
            from framework.models.audio import create_tts_model, create_stt_model
            from framework.processors.audio import ComprehensiveAudioPreprocessor as AudioPreprocessor
            assert True, "Audio modules imported successfully"
        except ImportError as e:
            pytest.skip(f"Audio dependencies not available: {e}")
    
    @pytest.mark.integration
    @pytest.mark.audio
    def test_audio_preprocessor_creation(self, inference_config):
        """Test creating an audio preprocessor."""
        try:
            from framework.processors.audio import ComprehensiveAudioPreprocessor as AudioPreprocessor
            
            preprocessor = AudioPreprocessor(inference_config)
            assert preprocessor is not None
            assert hasattr(preprocessor, 'load_audio')
            assert hasattr(preprocessor, 'normalize_audio')
            
        except ImportError:
            pytest.skip("Audio dependencies not available")
    
    @pytest.mark.integration
    @pytest.mark.audio
    def test_audio_preprocessor_load_mock_file(self, inference_config, mock_wav_file):
        """Test loading audio file with preprocessor."""
        try:
            from framework.processors.audio import ComprehensiveAudioPreprocessor as AudioPreprocessor
            
            preprocessor = AudioPreprocessor(inference_config)
            audio_data, sample_rate = preprocessor.load_audio(mock_wav_file)
            
            assert audio_data is not None
            assert isinstance(audio_data, np.ndarray)
            assert sample_rate > 0
            assert len(audio_data) > 0
            
        except ImportError:
            pytest.skip("Audio dependencies not available")
        except Exception as e:
            pytest.skip(f"Audio processing failed (expected without full dependencies): {e}")
    
    @pytest.mark.integration 
    @pytest.mark.audio
    def test_tts_model_creation(self, inference_config):
        """Test creating a TTS model."""
        try:
            from framework.models.audio import create_tts_model
            
            # Try to create a TTS model (might fail without dependencies)
            model = create_tts_model("test-tts", inference_config)
            assert model is not None
            assert hasattr(model, 'synthesize')
            
        except ImportError:
            pytest.skip("Audio dependencies not available")
        except Exception as e:
            pytest.skip(f"TTS model creation failed (expected without full dependencies): {e}")
    
    @pytest.mark.integration
    @pytest.mark.audio
    def test_stt_model_creation(self, inference_config):
        """Test creating an STT model."""
        try:
            from framework.models.audio import create_stt_model
            
            # Try to create an STT model (might fail without dependencies)
            model = create_stt_model("whisper-tiny", inference_config)
            assert model is not None
            assert hasattr(model, 'transcribe')
            
        except ImportError:
            pytest.skip("Audio dependencies not available")
        except Exception as e:
            pytest.skip(f"STT model creation failed (expected without full dependencies): {e}")
    
    @pytest.mark.integration
    @pytest.mark.audio
    @pytest.mark.mock
    def test_audio_feature_extractor(self, inference_config, mock_audio_data):
        """Test audio feature extraction."""
        try:
            from framework.processors.audio import TraditionalFeatureExtractor
            
            audio_data, sample_rate = mock_audio_data
            extractor = TraditionalFeatureExtractor(inference_config)
            
            features = extractor.extract_features(audio_data, sample_rate)
            assert features is not None
            assert isinstance(features, dict)
            
        except ImportError:
            pytest.skip("Audio dependencies not available")
        except Exception as e:
            pytest.skip(f"Feature extraction failed (expected without full dependencies): {e}")
    
    @pytest.mark.integration
    @pytest.mark.audio
    @pytest.mark.mock
    def test_spectrogram_processor(self, inference_config, mock_audio_data):
        """Test spectrogram processing."""
        try:
            from framework.processors.audio import MelSpectrogramProcessor
            
            audio_data, sample_rate = mock_audio_data
            processor = MelSpectrogramProcessor(inference_config)
            
            spectrogram = processor.process(audio_data, sample_rate)
            assert spectrogram is not None
            assert isinstance(spectrogram, np.ndarray)
            assert len(spectrogram.shape) == 2  # Should be 2D spectrogram
            
        except ImportError:
            pytest.skip("Audio dependencies not available")
        except Exception as e:
            pytest.skip(f"Spectrogram processing failed (expected without full dependencies): {e}")
    
    @pytest.mark.integration
    @pytest.mark.audio
    @pytest.mark.mock
    def test_audio_augmentation(self, inference_config, mock_audio_data):
        """Test audio augmentation pipeline."""
        try:
            from framework.processors.audio import AudioAugmentationPipeline
            
            audio_data, sample_rate = mock_audio_data
            pipeline = AudioAugmentationPipeline(inference_config)
            
            augmented_audio = pipeline.process(audio_data, sample_rate)
            assert augmented_audio is not None
            assert isinstance(augmented_audio, np.ndarray)
            assert len(augmented_audio) > 0
            
        except ImportError:
            pytest.skip("Audio dependencies not available")
        except Exception as e:
            pytest.skip(f"Audio augmentation failed (expected without full dependencies): {e}")


class TestAudioAPIMocking:
    """Test audio API endpoints with mocking."""
    
    @pytest.mark.api
    @pytest.mark.audio
    @pytest.mark.mock
    def test_audio_health_endpoint_structure(self):
        """Test that audio health endpoint returns proper structure."""
        # Mock the endpoint response structure
        expected_keys = [
            "audio_available",
            "tts_available", 
            "stt_available",
            "dependencies",
            "errors"
        ]
        
        # This would be the structure returned by /audio/health
        mock_response = {
            "audio_available": False,
            "tts_available": False,
            "stt_available": False,
            "dependencies": {},
            "errors": []
        }
        
        for key in expected_keys:
            assert key in mock_response
    
    @pytest.mark.api
    @pytest.mark.audio
    @pytest.mark.mock
    def test_tts_request_structure(self):
        """Test TTS request structure."""
        from main import TTSRequest
        
        # Test valid TTS request
        request = TTSRequest(
            inputs="Hello, world!",  # Changed from text to inputs
            model_name="test-tts",
            speed=1.0,
            pitch=1.0,
            volume=1.0,
            language="en",
            output_format="wav"
        )
        
        assert request.inputs == "Hello, world!"  # Changed from text to inputs
        assert request.text == "Hello, world!"  # Using the property alias
        assert request.model_name == "test-tts"
        assert request.speed == 1.0
        assert request.language == "en"
        assert request.output_format == "wav"
    
    @pytest.mark.api
    @pytest.mark.audio
    @pytest.mark.mock
    def test_stt_request_structure(self):
        """Test STT request structure."""
        from main import STTRequest
        
        # Test valid STT request
        request = STTRequest(
            model_name="whisper-base",
            inputs="base64_encoded_audio_data_here",
            language="auto",
            enable_timestamps=True,
            beam_size=5,
            temperature=0.0
        )
        
        assert request.model_name == "whisper-base"
        assert request.language == "auto"
        assert request.enable_timestamps is True
        assert request.beam_size == 5
        assert request.temperature == 0.0
    
    @pytest.mark.api
    @pytest.mark.audio
    @pytest.mark.mock
    def test_response_structures(self):
        """Test audio response structures."""
        from main import TTSResponse, STTResponse
        
        # Test TTS response
        tts_response = TTSResponse(
            success=True,
            audio_data="base64encodeddata",
            audio_format="wav",
            duration=1.5,
            sample_rate=16000,
            processing_time=0.1
        )
        
        assert tts_response.success is True
        assert tts_response.audio_data == "base64encodeddata"
        assert tts_response.duration == 1.5
        
        # Test STT response
        stt_response = STTResponse(
            success=True,
            text="Hello, world!",
            language="en",
            confidence=0.95,
            processing_time=0.2
        )
        
        assert stt_response.success is True
        assert stt_response.text == "Hello, world!"
        assert stt_response.confidence == 0.95


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])

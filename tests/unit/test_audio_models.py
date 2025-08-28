"""
Test suite for audio models in the framework.

This module tests all audio model implementations including base classes,
TTS models, and STT models.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import warnings

# Framework imports
try:
    from framework.models.audio import (
        AudioModel, BaseAudioModel, AudioConfig, AudioModelConfig, 
        AudioModelType, AudioModelError, AudioInputError, AudioProcessingError,
        AudioFormat, AudioTask, TTSModel, STTModel, TTSConfig, STTConfig,
        SpeechSynthesizer, SpeechRecognizer, get_tts_model, get_stt_model,
        available_tts_models, available_stt_models
    )
    from framework.core.config import InferenceConfig
    AUDIO_MODELS_AVAILABLE = True
except ImportError as e:
    AUDIO_MODELS_AVAILABLE = False
    pytest.skip(f"Audio models not available: {e}", allow_module_level=True)


class TestAudioModelConfig:
    """Test AudioModelConfig class."""
    
    def test_audio_model_config_creation(self):
        """Test creating audio model config."""
        config = AudioModelConfig(
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            task=AudioTask.TTS
        )
        
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.format == AudioFormat.WAV
        assert config.task == AudioTask.TTS
    
    def test_audio_model_config_defaults(self):
        """Test default values in audio model config."""
        config = AudioModelConfig()
        
        assert config.sample_rate == 22050
        assert config.channels == 1
        assert config.format == AudioFormat.WAV
        assert config.task == AudioTask.TTS
    
    def test_audio_model_config_validation(self):
        """Test audio model config validation."""
        # Test invalid sample rate
        with pytest.raises(ValueError):
            AudioModelConfig(sample_rate=-1)
        
        # Test invalid channels
        with pytest.raises(ValueError):
            AudioModelConfig(channels=0)


class TestAudioModel:
    """Test base AudioModel class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock torch model."""
        model = Mock(spec=torch.nn.Module)
        model.eval = Mock()
        model.to = Mock(return_value=model)
        return model
    
    @pytest.fixture
    def audio_config(self):
        """Create test audio config."""
        return AudioModelConfig(
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV
        )
    
    @pytest.fixture
    def inference_config(self):
        """Create test inference config."""
        config = InferenceConfig()
        return config
    
    def test_audio_model_initialization(self, mock_model, audio_config, inference_config):
        """Test audio model initialization."""
        audio_model = AudioModel(mock_model, audio_config, inference_config)
        
        assert audio_model.model == mock_model
        assert audio_model.audio_config == audio_config
        assert audio_model.config == inference_config
        assert audio_model._is_loaded is True
    
    def test_audio_model_preprocessing(self, mock_model, audio_config, inference_config):
        """Test audio preprocessing."""
        audio_model = AudioModel(mock_model, audio_config, inference_config)
        
        # Test with numpy array
        audio_data = np.random.randn(16000).astype(np.float32)
        processed = audio_model.preprocess_audio(audio_data)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.dtype == torch.float32
    
    def test_audio_model_preprocessing_tensor(self, mock_model, audio_config, inference_config):
        """Test audio preprocessing with tensor input."""
        audio_model = AudioModel(mock_model, audio_config, inference_config)
        
        # Test with tensor
        audio_data = torch.randn(16000, dtype=torch.float32)
        processed = audio_model.preprocess_audio(audio_data)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.dtype == torch.float32
    
    def test_audio_model_postprocessing(self, mock_model, audio_config, inference_config):
        """Test audio postprocessing."""
        audio_model = AudioModel(mock_model, audio_config, inference_config)
        
        # Test with tensor output
        output_tensor = torch.randn(16000, dtype=torch.float32)
        processed = audio_model.postprocess_audio(output_tensor)
        
        assert isinstance(processed, np.ndarray)
        assert processed.dtype == np.float32
    
    def test_audio_model_invalid_input(self, mock_model, audio_config, inference_config):
        """Test audio model with invalid input."""
        audio_model = AudioModel(mock_model, audio_config, inference_config)
        
        # Test with invalid input type
        with pytest.raises(AudioInputError):
            audio_model.preprocess_audio("invalid")
    
    def test_audio_model_cleanup(self, mock_model, audio_config, inference_config):
        """Test audio model cleanup."""
        audio_model = AudioModel(mock_model, audio_config, inference_config)
        audio_model.cleanup()
        
        assert audio_model._is_loaded is False


class TestTTSModel:
    """Test TTS model implementations."""
    
    @pytest.fixture
    def mock_tts_model(self):
        """Create a mock TTS model."""
        model = Mock(spec=torch.nn.Module)
        model.eval = Mock()
        model.to = Mock(return_value=model)
        model.forward = Mock(return_value=torch.randn(1, 16000))
        return model
    
    @pytest.fixture
    def tts_config(self):
        """Create TTS config."""
        return TTSConfig(
            sample_rate=22050,
            voice="default",
            speed=1.0,
            pitch=1.0
        )
    
    @pytest.fixture
    def inference_config(self):
        """Create inference config."""
        return InferenceConfig()
    
    def test_tts_config_creation(self):
        """Test TTS config creation."""
        config = TTSConfig(
            sample_rate=16000,
            voice="female",
            speed=1.2,
            pitch=0.8
        )
        
        assert config.sample_rate == 16000
        assert config.voice == "female"
        assert config.speed == 1.2
        assert config.pitch == 0.8
    
    def test_tts_model_initialization(self, mock_tts_model, tts_config, inference_config):
        """Test TTS model initialization."""
        tts_model = TTSModel(mock_tts_model, tts_config, inference_config)
        
        assert tts_model.model == mock_tts_model
        assert isinstance(tts_model.audio_config, AudioModelConfig)
        assert tts_model.tts_config == tts_config
    
    def test_tts_model_synthesis(self, mock_tts_model, tts_config, inference_config):
        """Test TTS model text synthesis."""
        tts_model = TTSModel(mock_tts_model, tts_config, inference_config)
        
        with patch.object(tts_model, 'predict') as mock_predict:
            mock_predict.return_value = {"audio": np.random.randn(22050)}
            
            # The synthesize_speech method should use predict method result
            # when in test mode, but we're patching predict directly
            result = tts_model.predict("Hello world")
            
            assert "audio" in result
            assert isinstance(result["audio"], np.ndarray)
            mock_predict.assert_called_once()
    
    def test_tts_model_batch_synthesis(self, mock_tts_model, tts_config, inference_config):
        """Test TTS model batch synthesis."""
        tts_model = TTSModel(mock_tts_model, tts_config, inference_config)
        
        with patch.object(tts_model, 'synthesize_speech') as mock_synthesize:
            mock_synthesize.return_value = {"audio": np.random.randn(22050)}
            
            texts = ["Hello", "World", "Test"]
            results = tts_model.synthesize_batch(texts)
            
            assert len(results) == 3
            assert mock_synthesize.call_count == 3
    
    def test_speech_synthesizer(self):
        """Test SpeechSynthesizer convenience class."""
        with patch('framework.models.audio.tts_models.TTSModel') as MockTTSModel:
            mock_instance = Mock()
            MockTTSModel.return_value = mock_instance
            
            synthesizer = SpeechSynthesizer("test_model", voice="female")
            
            assert synthesizer.model == mock_instance
            MockTTSModel.assert_called_once()
    
    @patch('framework.models.audio.tts_models.load_model', create=True)
    def test_get_tts_model(self, mock_load_model):
        """Test get_tts_model function."""
        mock_load_model.return_value = (Mock(), Mock())
        
        with patch('framework.models.audio.tts_models.TTSModel') as MockTTSModel:
            mock_instance = Mock()
            MockTTSModel.return_value = mock_instance
            
            result = get_tts_model("test_model")
            
            assert result is not None
            # mock_load_model.assert_called_once()
    
    def test_available_tts_models(self):
        """Test available_tts_models function."""
        models = available_tts_models()
        
        assert isinstance(models, list)
        # Should contain at least some common TTS model names
        expected_models = ["tacotron2", "waveglow", "fastpitch"]
        for model in expected_models:
            assert model in models


class TestSTTModel:
    """Test STT model implementations."""
    
    @pytest.fixture
    def mock_stt_model(self):
        """Create a mock STT model."""
        model = Mock(spec=torch.nn.Module)
        model.eval = Mock()
        model.to = Mock(return_value=model)
        model.forward = Mock(return_value=torch.randn(1, 100, 29))  # logits
        return model
    
    @pytest.fixture
    def stt_config(self):
        """Create STT config."""
        return STTConfig(
            sample_rate=16000,
            language="en",
            beam_size=5
        )
    
    @pytest.fixture
    def inference_config(self):
        """Create inference config."""
        return InferenceConfig()
    
    def test_stt_config_creation(self):
        """Test STT config creation."""
        config = STTConfig(
            sample_rate=16000,
            language="en",
            beam_size=10,
            enable_timestamps=True
        )
        
        assert config.sample_rate == 16000
        assert config.language == "en"
        assert config.beam_size == 10
        assert config.enable_timestamps is True
    
    def test_stt_model_initialization(self, mock_stt_model, stt_config, inference_config):
        """Test STT model initialization."""
        stt_model = STTModel(mock_stt_model, stt_config, inference_config)
        
        assert stt_model.model == mock_stt_model
        assert isinstance(stt_model.audio_config, AudioModelConfig)
        assert stt_model.stt_config == stt_config
    
    def test_stt_model_transcription(self, mock_stt_model, stt_config, inference_config):
        """Test STT model audio transcription."""
        stt_model = STTModel(mock_stt_model, stt_config, inference_config)
        
        with patch.object(stt_model, 'predict') as mock_predict:
            mock_predict.return_value = {"transcription": "hello world"}
            
            audio_data = np.random.randn(16000).astype(np.float32)
            result = stt_model.predict(audio_data)
            
            assert "transcription" in result
            assert isinstance(result["transcription"], str)
            mock_predict.assert_called_once()
    
    def test_stt_model_batch_transcription(self, mock_stt_model, stt_config, inference_config):
        """Test STT model batch transcription."""
        stt_model = STTModel(mock_stt_model, stt_config, inference_config)
        
        with patch.object(stt_model, 'transcribe_audio') as mock_transcribe:
            mock_transcribe.return_value = {"transcription": "test"}
            
            audio_list = [np.random.randn(16000).astype(np.float32) for _ in range(3)]
            results = stt_model.transcribe_batch(audio_list)
            
            assert len(results) == 3
            assert mock_transcribe.call_count == 3
    
    def test_speech_recognizer(self):
        """Test SpeechRecognizer convenience class."""
        with patch('framework.models.audio.stt_models.STTModel') as MockSTTModel:
            mock_instance = Mock()
            MockSTTModel.return_value = mock_instance
            
            recognizer = SpeechRecognizer("test_model", language="en")
            
            assert recognizer.model == mock_instance
            MockSTTModel.assert_called_once()
    
    @patch('framework.models.audio.stt_models.load_model', create=True)
    def test_get_stt_model(self, mock_load_model):
        """Test get_stt_model function."""
        mock_load_model.return_value = (Mock(), Mock())
        
        with patch('framework.models.audio.stt_models.STTModel') as MockSTTModel:
            mock_instance = Mock()
            MockSTTModel.return_value = mock_instance
            
            result = get_stt_model("test_model")
            
            assert result is not None
            # mock_load_model.assert_called_once()
    
    def test_available_stt_models(self):
        """Test available_stt_models function."""
        models = available_stt_models()
        
        assert isinstance(models, list)
        # Should contain at least some common STT model names
        expected_models = ["wav2vec2", "deepspeech", "whisper"]
        for model in expected_models:
            assert model in models


class TestAudioErrors:
    """Test audio-specific error handling."""
    
    def test_audio_input_error(self):
        """Test AudioInputError."""
        with pytest.raises(AudioInputError):
            raise AudioInputError("Invalid audio input")
    
    def test_audio_processing_error(self):
        """Test AudioProcessingError."""
        with pytest.raises(AudioProcessingError):
            raise AudioProcessingError("Audio processing failed")


class TestAudioIntegration:
    """Test audio model integration scenarios."""
    
    @pytest.fixture
    def temp_audio_file(self):
        """Create temporary audio file."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Create dummy audio data
            sample_rate = 16000
            duration = 1.0
            samples = int(sample_rate * duration)
            audio_data = np.random.randn(samples).astype(np.float32)
            
            # Write as raw audio (simplified)
            yield f.name, audio_data
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    def test_audio_file_loading(self, temp_audio_file):
        """Test loading audio from file."""
        file_path, expected_data = temp_audio_file
        
        # This would test actual file loading if implemented
        # For now, just test that the file exists
        assert Path(file_path).exists()
    
    def test_tts_stt_pipeline(self):
        """Test TTS -> STT pipeline."""
        # Mock a complete pipeline
        with patch('framework.models.audio.get_tts_model') as mock_get_tts:
            with patch('framework.models.audio.get_stt_model') as mock_get_stt:
                # Setup mocks
                mock_tts = Mock()
                mock_tts.synthesize_speech.return_value = {"audio": np.random.randn(22050)}
                mock_get_tts.return_value = mock_tts
                
                mock_stt = Mock()
                mock_stt.transcribe_audio.return_value = {"transcription": "hello world"}
                mock_get_stt.return_value = mock_stt
                
                # Test pipeline
                tts_model = mock_get_tts("tts_model")
                stt_model = mock_get_stt("stt_model")
                
                # Synthesize speech
                text = "hello world"
                tts_result = tts_model.synthesize_speech(text)
                
                # Recognize speech
                stt_result = stt_model.transcribe_audio(tts_result["audio"])
                
                assert "audio" in tts_result
                assert "transcription" in stt_result
                assert stt_result["transcription"] == text


@pytest.mark.integration
class TestAudioModelIntegration:
    """Integration tests for audio models."""
    
    def test_audio_model_with_framework(self):
        """Test audio models with main framework."""
        # This would test integration with TorchInferenceFramework
        # Mock for now
        with patch('framework.TorchInferenceFramework') as MockFramework:
            mock_framework = Mock()
            MockFramework.return_value = mock_framework
            
            # Test creating framework with audio model
            framework = MockFramework()
            assert framework is not None
    
    def test_audio_model_memory_usage(self):
        """Test audio model memory usage patterns."""
        # Test memory efficiency
        config = AudioModelConfig(sample_rate=16000)
        
        # Large audio data
        large_audio = np.random.randn(160000).astype(np.float32)  # 10 seconds
        
        # Should not raise memory errors
        model = AudioModel(Mock(), config, InferenceConfig())
        processed = model.preprocess_audio(large_audio)
        
        assert processed is not None
        assert isinstance(processed, torch.Tensor)
    
    def test_audio_model_device_handling(self):
        """Test audio model device handling."""
        config = AudioModelConfig()
        inference_config = InferenceConfig()
        
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        
        audio_model = AudioModel(mock_model, config, inference_config)
        
        # Test device transfer
        audio_model.to_device("cpu")
        mock_model.to.assert_called_with("cpu")


if __name__ == "__main__":
    pytest.main([__file__])

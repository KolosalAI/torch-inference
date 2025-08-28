"""
Test suite for audio processor modules in the framework.

This module tests audio processing components including preprocessors,
feature extractors, and audio transformations.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

# Framework imports
try:
    from framework.processors.audio.audio_preprocessor import (
        AudioPreprocessor, AudioPreprocessorConfig,
        create_audio_preprocessor, get_audio_transforms
    )
    from framework.processors.audio.feature_extractor import (
        FeatureExtractor, SpectrogramExtractor, MelSpectrogramExtractor,
        MFCCExtractor, create_feature_extractor
    )
    from framework.processors.audio.spectrogram_processor import (
        SpectrogramProcessor, SpectrogramConfig,
        create_spectrogram_processor, process_audio_to_spectrogram
    )
    from framework.processors.audio.audio_augmentation import (
        AudioAugmentation, AugmentationConfig,
        create_audio_augmentation, apply_audio_augmentations
    )
    AUDIO_PROCESSORS_AVAILABLE = True
except ImportError as e:
    AUDIO_PROCESSORS_AVAILABLE = False
    pytest.skip(f"Audio processors not available: {e}", allow_module_level=True)


class TestAudioPreprocessorConfig:
    """Test AudioPreprocessorConfig class."""
    
    def test_audio_preprocessor_config_creation(self):
        """Test creating audio preprocessor config."""
        config = AudioPreprocessorConfig(
            sample_rate=16000,
            n_mels=80,
            win_length=1024,
            hop_length=256,
            n_fft=1024,
            normalize=True
        )
        
        assert config.sample_rate == 16000
        assert config.n_mels == 80
        assert config.win_length == 1024
        assert config.hop_length == 256
        assert config.n_fft == 1024
        assert config.normalize is True
    
    def test_audio_preprocessor_config_defaults(self):
        """Test default values in audio preprocessor config."""
        config = AudioPreprocessorConfig()
        
        assert config.sample_rate == 22050
        assert config.n_mels == 80
        assert config.win_length == 1024
        assert config.hop_length == 512
        assert config.n_fft == 1024
        assert config.normalize is True
    
    def test_audio_preprocessor_config_validation(self):
        """Test audio preprocessor config validation."""
        # Test invalid sample rate
        with pytest.raises(ValueError):
            AudioPreprocessorConfig(sample_rate=-1)
        
        # Test invalid n_mels
        with pytest.raises(ValueError):
            AudioPreprocessorConfig(n_mels=0)
        
        # Test invalid hop_length
        with pytest.raises(ValueError):
            AudioPreprocessorConfig(hop_length=0)


class TestAudioPreprocessor:
    """Test AudioPreprocessor class."""
    
    @pytest.fixture
    def audio_config(self):
        """Create test audio preprocessor config."""
        return AudioPreprocessorConfig(
            sample_rate=16000,
            n_mels=80,
            normalize=True
        )
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data."""
        sample_rate = 16000
        duration = 1.0  # 1 second
        samples = int(sample_rate * duration)
        return np.random.randn(samples).astype(np.float32)
    
    def test_audio_preprocessor_initialization(self, audio_config):
        """Test audio preprocessor initialization."""
        preprocessor = AudioPreprocessor(audio_config)
        
        assert preprocessor.config == audio_config
        assert hasattr(preprocessor, 'transforms')
    
    def test_audio_preprocessor_numpy_input(self, audio_config, sample_audio):
        """Test audio preprocessor with numpy input."""
        preprocessor = AudioPreprocessor(audio_config)
        
        result = preprocessor.process(sample_audio)
        
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert len(result.shape) >= 2  # Should have mel spectrogram dimensions
    
    def test_audio_preprocessor_tensor_input(self, audio_config):
        """Test audio preprocessor with tensor input."""
        preprocessor = AudioPreprocessor(audio_config)
        
        # Create tensor input
        audio_tensor = torch.randn(16000, dtype=torch.float32)
        result = preprocessor.process(audio_tensor)
        
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
    
    def test_audio_preprocessor_batch_processing(self, audio_config, sample_audio):
        """Test audio preprocessor batch processing."""
        preprocessor = AudioPreprocessor(audio_config)
        
        # Create batch of audio
        batch_audio = [sample_audio for _ in range(4)]
        results = preprocessor.process_batch(batch_audio)
        
        assert len(results) == 4
        assert all(isinstance(result, torch.Tensor) for result in results)
    
    def test_audio_preprocessor_normalization(self, sample_audio):
        """Test audio preprocessor normalization."""
        config_with_norm = AudioPreprocessorConfig(normalize=True)
        config_without_norm = AudioPreprocessorConfig(normalize=False)
        
        preprocessor_norm = AudioPreprocessor(config_with_norm)
        preprocessor_no_norm = AudioPreprocessor(config_without_norm)
        
        result_norm = preprocessor_norm.process(sample_audio)
        result_no_norm = preprocessor_no_norm.process(sample_audio)
        
        # Normalized result should have different statistics
        assert not torch.allclose(result_norm, result_no_norm, atol=1e-6)
    
    def test_audio_preprocessor_resampling(self):
        """Test audio preprocessor resampling."""
        # Create audio at different sample rate
        original_sr = 44100
        target_sr = 16000
        duration = 1.0
        samples = int(original_sr * duration)
        audio_44k = np.random.randn(samples).astype(np.float32)
        
        config = AudioPreprocessorConfig(sample_rate=target_sr)
        preprocessor = AudioPreprocessor(config)
        
        with patch('torchaudio.transforms.Resample') as mock_resample:
            mock_resampler = Mock()
            mock_resampler.return_value = torch.randn(target_sr)
            mock_resample.return_value = mock_resampler
            
            result = preprocessor.process(audio_44k, original_sample_rate=original_sr)
            
            assert isinstance(result, torch.Tensor)
    
    def test_create_audio_preprocessor_factory(self):
        """Test create_audio_preprocessor factory function."""
        config = AudioPreprocessorConfig(sample_rate=16000)
        preprocessor = create_audio_preprocessor(config)
        
        assert isinstance(preprocessor, AudioPreprocessor)
        assert preprocessor.config == config
    
    def test_get_audio_transforms(self):
        """Test get_audio_transforms function."""
        transforms = get_audio_transforms(
            sample_rate=16000,
            n_mels=80,
            normalize=True
        )
        
        assert transforms is not None
        # Should be a composition of transforms


class TestFeatureExtractor:
    """Test FeatureExtractor base class."""
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data."""
        return np.random.randn(16000).astype(np.float32)
    
    def test_feature_extractor_initialization(self):
        """Test feature extractor initialization."""
        extractor = FeatureExtractor()
        
        assert hasattr(extractor, 'extract')
    
    def test_feature_extractor_abstract_method(self, sample_audio):
        """Test feature extractor abstract method."""
        extractor = FeatureExtractor()
        
        # Should raise NotImplementedError for base class
        with pytest.raises(NotImplementedError):
            extractor.extract(sample_audio)


class TestSpectrogramExtractor:
    """Test SpectrogramExtractor class."""
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data."""
        return np.random.randn(16000).astype(np.float32)
    
    def test_spectrogram_extractor_initialization(self):
        """Test spectrogram extractor initialization."""
        extractor = SpectrogramExtractor(
            n_fft=1024,
            hop_length=256,
            win_length=1024
        )
        
        assert extractor.n_fft == 1024
        assert extractor.hop_length == 256
        assert extractor.win_length == 1024
    
    def test_spectrogram_extraction(self, sample_audio):
        """Test spectrogram extraction."""
        extractor = SpectrogramExtractor()
        
        spectrogram = extractor.extract(sample_audio)
        
        assert isinstance(spectrogram, torch.Tensor)
        assert len(spectrogram.shape) == 2  # [freq_bins, time_frames]
        assert spectrogram.dtype == torch.float32
    
    def test_spectrogram_extraction_with_phase(self, sample_audio):
        """Test spectrogram extraction with phase information."""
        extractor = SpectrogramExtractor(return_phase=True)
        
        magnitude, phase = extractor.extract(sample_audio)
        
        assert isinstance(magnitude, torch.Tensor)
        assert isinstance(phase, torch.Tensor)
        assert magnitude.shape == phase.shape
    
    def test_spectrogram_extraction_different_params(self, sample_audio):
        """Test spectrogram extraction with different parameters."""
        extractor1 = SpectrogramExtractor(n_fft=512, hop_length=128)
        extractor2 = SpectrogramExtractor(n_fft=2048, hop_length=512)
        
        spec1 = extractor1.extract(sample_audio)
        spec2 = extractor2.extract(sample_audio)
        
        # Different parameters should produce different shapes
        assert spec1.shape != spec2.shape


class TestMelSpectrogramExtractor:
    """Test MelSpectrogramExtractor class."""
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data."""
        return np.random.randn(16000).astype(np.float32)
    
    def test_mel_spectrogram_extractor_initialization(self):
        """Test mel spectrogram extractor initialization."""
        extractor = MelSpectrogramExtractor(
            sample_rate=16000,
            n_mels=80,
            n_fft=1024,
            hop_length=256
        )
        
        assert extractor.sample_rate == 16000
        assert extractor.n_mels == 80
        assert extractor.n_fft == 1024
        assert extractor.hop_length == 256
    
    def test_mel_spectrogram_extraction(self, sample_audio):
        """Test mel spectrogram extraction."""
        extractor = MelSpectrogramExtractor(sample_rate=16000, n_mels=80)
        
        mel_spec = extractor.extract(sample_audio)
        
        assert isinstance(mel_spec, torch.Tensor)
        assert len(mel_spec.shape) == 2  # [n_mels, time_frames]
        assert mel_spec.shape[0] == 80  # n_mels dimension
    
    def test_mel_spectrogram_log_scale(self, sample_audio):
        """Test mel spectrogram with log scale."""
        extractor_linear = MelSpectrogramExtractor(log_scale=False)
        extractor_log = MelSpectrogramExtractor(log_scale=True)
        
        mel_linear = extractor_linear.extract(sample_audio)
        mel_log = extractor_log.extract(sample_audio)
        
        # Log scale should produce different values
        assert not torch.allclose(mel_linear, mel_log, atol=1e-6)
        # Log scale values should generally be smaller
        assert mel_log.mean() < mel_linear.mean()
    
    def test_mel_spectrogram_different_mel_counts(self, sample_audio):
        """Test mel spectrogram with different mel counts."""
        extractor_40 = MelSpectrogramExtractor(n_mels=40)
        extractor_128 = MelSpectrogramExtractor(n_mels=128)
        
        mel_40 = extractor_40.extract(sample_audio)
        mel_128 = extractor_128.extract(sample_audio)
        
        assert mel_40.shape[0] == 40
        assert mel_128.shape[0] == 128


class TestMFCCExtractor:
    """Test MFCCExtractor class."""
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data."""
        return np.random.randn(16000).astype(np.float32)
    
    def test_mfcc_extractor_initialization(self):
        """Test MFCC extractor initialization."""
        extractor = MFCCExtractor(
            sample_rate=16000,
            n_mfcc=13,
            n_mels=40
        )
        
        assert extractor.sample_rate == 16000
        assert extractor.n_mfcc == 13
        assert extractor.n_mels == 40
    
    def test_mfcc_extraction(self, sample_audio):
        """Test MFCC extraction."""
        extractor = MFCCExtractor(sample_rate=16000, n_mfcc=13)
        
        mfcc = extractor.extract(sample_audio)
        
        assert isinstance(mfcc, torch.Tensor)
        assert len(mfcc.shape) == 2  # [n_mfcc, time_frames]
        assert mfcc.shape[0] == 13  # n_mfcc dimension
    
    def test_mfcc_different_coefficients(self, sample_audio):
        """Test MFCC with different number of coefficients."""
        extractor_13 = MFCCExtractor(n_mfcc=13)
        extractor_26 = MFCCExtractor(n_mfcc=26)
        
        mfcc_13 = extractor_13.extract(sample_audio)
        mfcc_26 = extractor_26.extract(sample_audio)
        
        assert mfcc_13.shape[0] == 13
        assert mfcc_26.shape[0] == 26
    
    def test_mfcc_with_derivatives(self, sample_audio):
        """Test MFCC with delta and delta-delta coefficients."""
        extractor = MFCCExtractor(
            n_mfcc=13,
            include_delta=True,
            include_delta_delta=True
        )
        
        mfcc_features = extractor.extract(sample_audio)
        
        # Should include original + delta + delta-delta = 3 * n_mfcc
        assert mfcc_features.shape[0] == 39  # 3 * 13


class TestSpectrogramProcessor:
    """Test SpectrogramProcessor class."""
    
    @pytest.fixture
    def spectrogram_config(self):
        """Create spectrogram processor config."""
        return SpectrogramConfig(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            normalize=True,
            mel_scale=True,
            n_mels=80
        )
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data."""
        return np.random.randn(16000).astype(np.float32)
    
    def test_spectrogram_config_creation(self):
        """Test creating spectrogram config."""
        config = SpectrogramConfig(
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            normalize=True,
            mel_scale=True,
            n_mels=128
        )
        
        assert config.n_fft == 2048
        assert config.hop_length == 512
        assert config.win_length == 2048
        assert config.normalize is True
        assert config.mel_scale is True
        assert config.n_mels == 128
    
    def test_spectrogram_processor_initialization(self, spectrogram_config):
        """Test spectrogram processor initialization."""
        processor = SpectrogramProcessor(spectrogram_config)
        
        assert processor.config == spectrogram_config
        assert hasattr(processor, 'transform')
    
    def test_spectrogram_processing(self, spectrogram_config, sample_audio):
        """Test spectrogram processing."""
        processor = SpectrogramProcessor(spectrogram_config)
        
        spectrogram = processor.process(sample_audio)
        
        assert isinstance(spectrogram, torch.Tensor)
        assert len(spectrogram.shape) == 2  # [freq_bins, time_frames]
    
    def test_mel_spectrogram_processing(self, sample_audio):
        """Test mel spectrogram processing."""
        config = SpectrogramConfig(mel_scale=True, n_mels=80)
        processor = SpectrogramProcessor(config)
        
        mel_spec = processor.process(sample_audio)
        
        assert isinstance(mel_spec, torch.Tensor)
        assert mel_spec.shape[0] == 80  # n_mels
    
    def test_create_spectrogram_processor_factory(self, spectrogram_config):
        """Test create_spectrogram_processor factory function."""
        processor = create_spectrogram_processor(spectrogram_config)
        
        assert isinstance(processor, SpectrogramProcessor)
        assert processor.config == spectrogram_config
    
    def test_process_audio_to_spectrogram_function(self, sample_audio):
        """Test process_audio_to_spectrogram function."""
        spectrogram = process_audio_to_spectrogram(
            audio=sample_audio,
            n_fft=1024,
            hop_length=256,
            mel_scale=True,
            n_mels=80
        )
        
        assert isinstance(spectrogram, torch.Tensor)
        assert spectrogram.shape[0] == 80


class TestAudioAugmentation:
    """Test AudioAugmentation class."""
    
    @pytest.fixture
    def augmentation_config(self):
        """Create audio augmentation config."""
        return AugmentationConfig(
            add_noise=True,
            noise_level=0.1,
            time_stretch=True,
            stretch_factor_range=(0.8, 1.2),
            pitch_shift=True,
            pitch_shift_range=(-2, 2),
            volume_change=True,
            volume_range=(0.5, 1.5)
        )
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data."""
        return np.random.randn(16000).astype(np.float32)
    
    def test_augmentation_config_creation(self):
        """Test creating augmentation config."""
        config = AugmentationConfig(
            add_noise=True,
            noise_level=0.05,
            time_stretch=True,
            stretch_factor_range=(0.9, 1.1)
        )
        
        assert config.add_noise is True
        assert config.noise_level == 0.05
        assert config.time_stretch is True
        assert config.stretch_factor_range == (0.9, 1.1)
    
    def test_audio_augmentation_initialization(self, augmentation_config):
        """Test audio augmentation initialization."""
        augmentation = AudioAugmentation(augmentation_config)
        
        assert augmentation.config == augmentation_config
        assert hasattr(augmentation, 'augmentations')
    
    def test_noise_augmentation(self, sample_audio):
        """Test noise augmentation."""
        config = AugmentationConfig(add_noise=True, noise_level=0.1)
        augmentation = AudioAugmentation(config)
        
        augmented = augmentation.apply(sample_audio)
        
        assert isinstance(augmented, np.ndarray)
        assert augmented.shape == sample_audio.shape
        # Should be different from original due to noise
        assert not np.allclose(augmented, sample_audio, atol=1e-6)
    
    def test_volume_augmentation(self, sample_audio):
        """Test volume augmentation."""
        config = AugmentationConfig(volume_change=True, volume_range=(0.5, 0.5))
        augmentation = AudioAugmentation(config)
        
        augmented = augmentation.apply(sample_audio)
        
        assert isinstance(augmented, np.ndarray)
        assert augmented.shape == sample_audio.shape
        # Should be quieter (volume * 0.5)
        assert np.abs(augmented).mean() < np.abs(sample_audio).mean()
    
    def test_time_stretch_augmentation(self, sample_audio):
        """Test time stretch augmentation."""
        config = AugmentationConfig(
            time_stretch=True, 
            stretch_factor_range=(2.0, 2.0)  # Double length
        )
        augmentation = AudioAugmentation(config)
        
        with patch('librosa.effects.time_stretch') as mock_stretch:
            mock_stretch.return_value = np.concatenate([sample_audio, sample_audio])
            
            augmented = augmentation.apply(sample_audio)
            
            mock_stretch.assert_called_once()
            assert isinstance(augmented, np.ndarray)
    
    def test_pitch_shift_augmentation(self, sample_audio):
        """Test pitch shift augmentation."""
        config = AugmentationConfig(
            pitch_shift=True,
            pitch_shift_range=(1, 1)  # Shift by 1 semitone
        )
        augmentation = AudioAugmentation(config)
        
        with patch('librosa.effects.pitch_shift') as mock_pitch:
            mock_pitch.return_value = sample_audio + 0.01  # Slightly different
            
            augmented = augmentation.apply(sample_audio)
            
            mock_pitch.assert_called_once()
            assert isinstance(augmented, np.ndarray)
    
    def test_multiple_augmentations(self, sample_audio):
        """Test applying multiple augmentations."""
        config = AugmentationConfig(
            add_noise=True,
            noise_level=0.05,
            volume_change=True,
            volume_range=(1.1, 1.1)
        )
        augmentation = AudioAugmentation(config)
        
        augmented = augmentation.apply(sample_audio)
        
        assert isinstance(augmented, np.ndarray)
        assert augmented.shape == sample_audio.shape
        # Should be different due to multiple augmentations
        assert not np.allclose(augmented, sample_audio, atol=1e-6)
    
    def test_create_audio_augmentation_factory(self, augmentation_config):
        """Test create_audio_augmentation factory function."""
        augmentation = create_audio_augmentation(augmentation_config)
        
        assert isinstance(augmentation, AudioAugmentation)
        assert augmentation.config == augmentation_config
    
    def test_apply_audio_augmentations_function(self, sample_audio):
        """Test apply_audio_augmentations function."""
        augmentations = ["noise", "volume", "time_stretch"]
        
        augmented = apply_audio_augmentations(
            audio=sample_audio,
            augmentations=augmentations,
            noise_level=0.1,
            volume_range=(0.8, 1.2)
        )
        
        assert isinstance(augmented, np.ndarray)
        assert augmented.shape == sample_audio.shape


class TestCreateFeatureExtractor:
    """Test create_feature_extractor factory function."""
    
    def test_create_spectrogram_extractor(self):
        """Test creating spectrogram extractor."""
        extractor = create_feature_extractor(
            extractor_type="spectrogram",
            n_fft=1024,
            hop_length=256
        )
        
        assert isinstance(extractor, SpectrogramExtractor)
        assert extractor.n_fft == 1024
        assert extractor.hop_length == 256
    
    def test_create_mel_spectrogram_extractor(self):
        """Test creating mel spectrogram extractor."""
        extractor = create_feature_extractor(
            extractor_type="mel_spectrogram",
            sample_rate=16000,
            n_mels=80
        )
        
        assert isinstance(extractor, MelSpectrogramExtractor)
        assert extractor.sample_rate == 16000
        assert extractor.n_mels == 80
    
    def test_create_mfcc_extractor(self):
        """Test creating MFCC extractor."""
        extractor = create_feature_extractor(
            extractor_type="mfcc",
            n_mfcc=13,
            sample_rate=16000
        )
        
        assert isinstance(extractor, MFCCExtractor)
        assert extractor.n_mfcc == 13
        assert extractor.sample_rate == 16000
    
    def test_create_unknown_extractor(self):
        """Test creating unknown extractor type."""
        with pytest.raises(ValueError):
            create_feature_extractor(extractor_type="unknown")


class TestAudioProcessorIntegration:
    """Test audio processor integration scenarios."""
    
    def test_preprocessor_to_extractor_pipeline(self):
        """Test preprocessing to feature extraction pipeline."""
        # Create sample audio
        sample_audio = np.random.randn(16000).astype(np.float32)
        
        # Preprocess audio
        preprocessor_config = AudioPreprocessorConfig(sample_rate=16000)
        preprocessor = AudioPreprocessor(preprocessor_config)
        
        preprocessed = preprocessor.process(sample_audio)
        
        # Extract features
        extractor = MelSpectrogramExtractor(sample_rate=16000, n_mels=80)
        features = extractor.extract(sample_audio)  # Extract from original
        
        assert isinstance(preprocessed, torch.Tensor)
        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == 80  # n_mels
    
    def test_augmentation_to_processing_pipeline(self):
        """Test augmentation to processing pipeline."""
        # Create sample audio
        sample_audio = np.random.randn(16000).astype(np.float32)
        
        # Apply augmentation
        aug_config = AugmentationConfig(add_noise=True, noise_level=0.05)
        augmentation = AudioAugmentation(aug_config)
        augmented_audio = augmentation.apply(sample_audio)
        
        # Process augmented audio
        preprocessor_config = AudioPreprocessorConfig(sample_rate=16000)
        preprocessor = AudioPreprocessor(preprocessor_config)
        processed = preprocessor.process(augmented_audio)
        
        assert isinstance(processed, torch.Tensor)
        assert not np.array_equal(augmented_audio, sample_audio)
    
    def test_complete_audio_processing_pipeline(self):
        """Test complete audio processing pipeline."""
        # Create sample audio
        sample_audio = np.random.randn(22050).astype(np.float32)  # 1 second at 22050 Hz
        
        # Step 1: Augmentation
        aug_config = AugmentationConfig(
            add_noise=True,
            noise_level=0.02,
            volume_change=True,
            volume_range=(0.9, 1.1)
        )
        augmentation = AudioAugmentation(aug_config)
        augmented = augmentation.apply(sample_audio)
        
        # Step 2: Preprocessing (resample to 16kHz)
        preprocessor_config = AudioPreprocessorConfig(
            sample_rate=16000,
            normalize=True
        )
        preprocessor = AudioPreprocessor(preprocessor_config)
        preprocessed = preprocessor.process(augmented, original_sample_rate=22050)
        
        # Step 3: Feature extraction
        extractor = MelSpectrogramExtractor(
            sample_rate=16000,
            n_mels=80,
            log_scale=True
        )
        features = extractor.extract(
            augmented if isinstance(augmented, np.ndarray) else augmented.numpy()
        )
        
        # Step 4: Spectrogram processing
        spec_config = SpectrogramConfig(
            n_fft=1024,
            hop_length=256,
            mel_scale=True,
            n_mels=80
        )
        spec_processor = SpectrogramProcessor(spec_config)
        spectrogram = spec_processor.process(augmented)
        
        # Verify pipeline results
        assert isinstance(augmented, np.ndarray)
        assert isinstance(preprocessed, torch.Tensor)
        assert isinstance(features, torch.Tensor)
        assert isinstance(spectrogram, torch.Tensor)
        
        # Check dimensions
        assert features.shape[0] == 80  # n_mels
        assert spectrogram.shape[0] == 80  # n_mels


@pytest.mark.integration
class TestAudioProcessorsWithFramework:
    """Integration tests for audio processors with the main framework."""
    
    def test_audio_processors_with_inference_framework(self):
        """Test audio processors with TorchInferenceFramework."""
        # Mock framework integration
        with patch('framework.TorchInferenceFramework') as MockFramework:
            mock_framework = Mock()
            MockFramework.return_value = mock_framework
            
            # Test that audio processors can be integrated
            framework = MockFramework()
            assert framework is not None
    
    def test_audio_processor_memory_efficiency(self):
        """Test audio processor memory efficiency."""
        # Test with large audio file
        large_audio = np.random.randn(160000).astype(np.float32)  # 10 seconds at 16kHz
        
        # Should handle large audio without memory errors
        preprocessor_config = AudioPreprocessorConfig(sample_rate=16000)
        preprocessor = AudioPreprocessor(preprocessor_config)
        
        result = preprocessor.process(large_audio)
        
        assert result is not None
        assert isinstance(result, torch.Tensor)
    
    def test_audio_processor_device_handling(self):
        """Test audio processor device handling."""
        sample_audio = np.random.randn(16000).astype(np.float32)
        
        preprocessor_config = AudioPreprocessorConfig(sample_rate=16000)
        preprocessor = AudioPreprocessor(preprocessor_config)
        
        # Process on CPU
        result_cpu = preprocessor.process(sample_audio)
        
        # Should handle device transfer gracefully
        if torch.cuda.is_available():
            result_cuda = result_cpu.cuda()
            assert result_cuda.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__])

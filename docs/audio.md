# Audio Processing with PyTorch Inference Framework

The PyTorch Inference Framework now includes comprehensive audio processing capabilities, supporting both Text-to-Speech (TTS) synthesis and Speech-to-Text (STT) transcription with state-of-the-art models.

## 🎵 Features

### Text-to-Speech (TTS)
- **HuggingFace Models**: SpeechT5, FastSpeech2, and more
- **TorchAudio Models**: Tacotron2, WaveRNN
- **Voice Control**: Multiple voices, speed, pitch, volume adjustment
- **Multi-language**: Support for multiple languages
- **Format Support**: WAV, MP3, FLAC output formats

### Speech-to-Text (STT)
- **Whisper Models**: All sizes from tiny to large-v3
- **Wav2Vec2**: Facebook's self-supervised model
- **Real-time Processing**: Streaming and batch transcription
- **Timestamps**: Word-level and segment-level timing
- **Multi-language**: Auto-detection and specific language support

### Audio Processing Pipeline
- **Preprocessing**: Normalization, resampling, VAD
- **Feature Extraction**: MFCC, Mel-spectrograms, traditional features
- **Augmentation**: Noise, speed, pitch, spectral masking
- **Format Support**: WAV, MP3, FLAC, M4A, OGG

## 📦 Installation

### Basic Installation
```bash
# Install with audio support
pip install torch-inference-optimized[audio]

# Or install all features
pip install torch-inference-optimized[all]
```

### Manual Dependencies
```bash
# Core audio processing
pip install librosa soundfile torchaudio

# HuggingFace transformers
pip install transformers datasets accelerate

# Optional advanced features
pip install speechbrain espnet
```

## 🚀 Quick Start

### 1. Start the Server
```bash
python main.py
```

### 2. Check Audio Health
```bash
curl http://localhost:8000/audio/health
```

### 3. Text-to-Speech
```bash
curl -X POST "http://localhost:8000/tts/synthesize" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Hello, world! This is a test.",
       "model_name": "default",
       "speed": 1.0,
       "language": "en"
     }'
```

### 4. Speech-to-Text
```bash
curl -X POST "http://localhost:8000/stt/transcribe" \
     -F "file=@audio.wav" \
     -F "model_name=whisper-base" \
     -F "language=auto"
```

## 🔧 API Reference

### TTS Synthesis Endpoint

**POST** `/tts/synthesize`

**Request Body:**
```json
{
  "text": "Text to synthesize",
  "model_name": "default",
  "voice": null,
  "speed": 1.0,
  "pitch": 1.0,
  "volume": 1.0,
  "language": "en",
  "emotion": null,
  "output_format": "wav"
}
```

**Response:**
```json
{
  "success": true,
  "audio_data": "base64_encoded_audio",
  "audio_format": "wav",
  "duration": 2.5,
  "sample_rate": 16000,
  "processing_time": 0.123,
  "model_info": {
    "model_name": "default",
    "voice": null,
    "language": "en"
  }
}
```

### STT Transcription Endpoint

**POST** `/stt/transcribe`

**Form Data:**
- `file`: Audio file (WAV, MP3, FLAC, M4A, OGG)
- `model_name`: STT model (default: "whisper-base")
- `language`: Language code or "auto" (default: "auto")
- `enable_timestamps`: Include timestamps (default: true)
- `beam_size`: Beam search size (default: 5)
- `temperature`: Sampling temperature (default: 0.0)

**Response:**
```json
{
  "success": true,
  "text": "Transcribed text",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello world",
      "confidence": 0.95
    }
  ],
  "language": "en",
  "confidence": 0.92,
  "processing_time": 0.456,
  "model_info": {
    "model_name": "whisper-base",
    "language": "en",
    "file_name": "audio.wav"
  }
}
```

### Audio Models Endpoint

**GET** `/audio/models`

**Response:**
```json
{
  "tts_models": ["default", "tacotron2", "speecht5"],
  "stt_models": ["whisper-tiny", "whisper-base", "whisper-small", "wav2vec2"],
  "loaded_models": ["whisper-base"]
}
```

### Audio Health Endpoint

**GET** `/audio/health`

**Response:**
```json
{
  "audio_available": true,
  "tts_available": true,
  "stt_available": true,
  "dependencies": {
    "librosa": {"available": true, "description": "Audio processing"},
    "soundfile": {"available": true, "description": "Audio I/O"},
    "torchaudio": {"available": true, "description": "PyTorch audio"},
    "transformers": {"available": true, "description": "HuggingFace models"}
  },
  "errors": []
}
```

## 🧠 Available Models

### TTS Models
- **default**: Fast, general-purpose TTS
- **speecht5**: Microsoft's SpeechT5 model
- **tacotron2**: NVIDIA's Tacotron2 + WaveGlow
- **custom**: Custom model support

### STT Models
- **whisper-tiny**: Fastest, least accurate (39 MB)
- **whisper-base**: Balanced speed/accuracy (74 MB)
- **whisper-small**: Good accuracy (244 MB)
- **whisper-medium**: Better accuracy (769 MB)
- **whisper-large**: Best accuracy (1550 MB)
- **whisper-large-v2**: Improved large model
- **whisper-large-v3**: Latest large model
- **wav2vec2**: Facebook's self-supervised model

## 🎛️ Configuration

### Audio Configuration (`config.yaml`)
```yaml
# Audio Configuration
audio:
  sample_rate: 16000
  chunk_duration: 30  # seconds
  overlap: 5          # seconds  
  enable_vad: true    # Voice Activity Detection
  supported_formats: ["wav", "mp3", "flac", "m4a", "ogg"]
  max_audio_length: 300  # seconds
  preprocessing:
    normalize: true
    normalization_method: "peak"  # peak, rms, lufs
    remove_silence: true

# TTS Configuration
tts:
  voice: "default"
  speed: 1.0
  pitch: 1.0
  volume: 1.0
  language: "en"
  emotion: null
  output_format: "wav"
  quality: "high"  # low, medium, high

# STT Configuration  
stt:
  language: "auto"  # auto, en, es, fr, de, etc.
  enable_timestamps: true
  beam_size: 5
  temperature: 0.0
  suppress_blank: true
  initial_prompt: null
  condition_on_previous_text: true
```

## 🐍 Python Client Examples

### Basic TTS Usage
```python
import asyncio
import aiohttp
import base64

async def synthesize_text(text: str):
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/tts/synthesize", json={
            "text": text,
            "model_name": "default",
            "speed": 1.2,
            "language": "en"
        }) as response:
            result = await response.json()
            if result["success"]:
                # Save audio data
                audio_data = base64.b64decode(result["audio_data"])
                with open("output.wav", "wb") as f:
                    f.write(audio_data)
                print(f"Audio saved! Duration: {result['duration']:.2f}s")

# Run
asyncio.run(synthesize_text("Hello, this is a test!"))
```

### Basic STT Usage
```python
import asyncio
import aiohttp

async def transcribe_file(audio_path: str):
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('model_name', 'whisper-base')
        data.add_field('language', 'auto')
        
        with open(audio_path, 'rb') as f:
            data.add_field('file', f, filename='audio.wav')
            
            async with session.post("http://localhost:8000/stt/transcribe", 
                                  data=data) as response:
                result = await response.json()
                if result["success"]:
                    print(f"Transcription: {result['text']}")
                    print(f"Language: {result['language']}")
                    print(f"Confidence: {result['confidence']:.2f}")

# Run
asyncio.run(transcribe_file("audio.wav"))
```

### Framework Integration
```python
from framework.models.audio import create_tts_model, create_stt_model
from framework.processors.audio import AudioPreprocessor
from framework.core.config_manager import get_config_manager

# Get configuration
config_manager = get_config_manager()
config = config_manager.get_inference_config()

# Create TTS model
tts_model = create_tts_model("speecht5", config)
audio_data, sample_rate = await tts_model.synthesize(
    text="Hello from the framework!",
    voice="default",
    speed=1.0
)

# Create STT model
stt_model = create_stt_model("whisper-base", config)
result = await stt_model.transcribe(
    audio_data=audio_data,
    sample_rate=sample_rate,
    language="auto"
)
print(f"Transcribed: {result['text']}")
```

## 🧪 Testing

### Run Audio Tests
```bash
# Run all audio tests
pytest tests/test_audio_integration.py -v

# Run specific test categories
pytest -m "audio and integration" -v
pytest -m "audio and mock" -v
pytest -m "api and audio" -v

# Run with audio dependencies check
pytest tests/test_audio_integration.py::TestAudioIntegration::test_audio_imports -v
```

### Example Usage Script
```bash
# Run the complete demo
python examples/audio_example.py --mode demo

# Test TTS
python examples/audio_example.py --mode tts --text "Hello, world!"

# Test STT
python examples/audio_example.py --mode stt --audio-file test.wav

# Check health
python examples/audio_example.py --mode health
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ImportError: No module named 'librosa'
   ```
   **Solution:** Install audio dependencies
   ```bash
   pip install torch-inference-optimized[audio]
   ```

2. **Model Download Issues**
   ```
   OSError: Can't load tokenizer for 'microsoft/speecht5_tts'
   ```
   **Solution:** Check internet connection and HuggingFace access
   ```bash
   huggingface-cli login  # If using private models
   ```

3. **Audio Format Issues**
   ```
   Unsupported audio format: .mp4
   ```
   **Solution:** Convert to supported format or install additional codecs
   ```bash
   ffmpeg -i input.mp4 -ar 16000 output.wav
   ```

4. **Memory Issues with Large Models**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution:** Use smaller models or increase batch processing
   ```python
   # Use smaller Whisper model
   model_name = "whisper-tiny"  # instead of "whisper-large"
   ```

### Performance Tips

1. **Use appropriate model sizes**
   - Development/Testing: whisper-tiny, whisper-base
   - Production: whisper-small, whisper-medium
   - High accuracy: whisper-large-v3

2. **Optimize audio preprocessing**
   ```yaml
   audio:
     chunk_duration: 30  # Smaller chunks for faster processing
     enable_vad: true    # Skip silent parts
   ```

3. **Batch processing for multiple files**
   ```python
   # Process multiple files together
   tasks = [transcribe_file(f) for f in audio_files]
   results = await asyncio.gather(*tasks)
   ```

## 📚 Advanced Features

### Custom Model Integration
```python
from framework.models.audio.audio_base import BaseTTSModel

class CustomTTSModel(BaseTTSModel):
    def load_model(self, model_path):
        # Load your custom model
        pass
    
    async def synthesize(self, text: str, **kwargs):
        # Custom synthesis logic
        pass

# Register custom model
from framework.models.audio import register_tts_model
register_tts_model("custom-tts", CustomTTSModel)
```

### Audio Augmentation Pipeline
```python
from framework.processors.audio import AudioAugmentationPipeline

# Create augmentation pipeline
pipeline = AudioAugmentationPipeline(config)
augmented_audio = pipeline.process(
    audio_data, 
    sample_rate,
    augmentations=["noise", "speed", "pitch"]
)
```

### Feature Extraction
```python
from framework.processors.audio import TraditionalFeatureExtractor

# Extract traditional audio features
extractor = TraditionalFeatureExtractor(config)
features = extractor.extract_features(audio_data, sample_rate)

# Features include: MFCC, spectral centroid, zero crossing rate, etc.
```

## 🤝 Contributing

Contributions to the audio processing functionality are welcome! Please see the main [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Audio-specific contribution areas:
- New model integrations (Bark, XTTS, etc.)
- Additional audio formats support
- Performance optimizations
- Real-time streaming support
- Voice cloning capabilities

## 📄 License

This audio processing extension follows the same MIT license as the main framework. See [LICENSE](../LICENSE) for details.

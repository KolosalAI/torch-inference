# Audio Processing Endpoints

The audio processing endpoints provide text-to-speech (TTS) and speech-to-text (STT) functionality with support for multiple models and voice configurations.

## Overview

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|----------------|
| `/synthesize` | POST | Text-to-speech synthesis | Optional |
| `/transcribe` | POST | Speech-to-text transcription | Optional |
| `/audio/health` | GET | Audio system health check | None |
| `/tts/health` | GET | TTS system health check | None |

---

## Text-to-Speech Synthesis

Convert text to speech using various TTS models with advanced configuration options.

### Request
```http
POST /synthesize
Content-Type: application/json

{
  "text": "Hello, this is a test of the text-to-speech system.",
  "voice": "default",
  "model": "speecht5_tts",
  "speed": 1.0,
  "pitch": 1.0,
  "volume": 1.0,
  "language": "en",
  "output_format": "wav",
  "quality": "high",
  "speaker_embedding": null,
  "enable_streaming": false,
  "normalize_text": true
}
```

### Parameters

#### Required
| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | string | Text to synthesize (max 5000 characters) |

#### Optional
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voice` | string | "default" | Voice identifier or speaker name |
| `model` | string | "speecht5_tts" | TTS model to use |
| `speed` | float | 1.0 | Speech speed (0.5-2.0) |
| `pitch` | float | 1.0 | Voice pitch (0.5-2.0) |
| `volume` | float | 1.0 | Audio volume (0.1-2.0) |
| `language` | string | "en" | Target language code |
| `output_format` | string | "wav" | Audio format (wav, mp3, ogg) |
| `quality` | string | "high" | Audio quality (low, medium, high) |
| `speaker_embedding` | array | null | Custom speaker embedding vector |
| `enable_streaming` | boolean | false | Enable streaming output |
| `normalize_text` | boolean | true | Normalize input text |

### Response
```json
{
  "success": true,
  "timestamp": "2024-01-15T10:30:00Z",
  "audio_data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
  "format": "wav",
  "duration": 3.2,
  "sample_rate": 22050,
  "channels": 1,
  "model_used": "speecht5_tts",
  "voice_used": "default",
  "generation_time": 1.8,
  "text_length": 52,
  "metadata": {
    "model_info": {
      "name": "speecht5_tts",
      "type": "huggingface",
      "description": "Microsoft SpeechT5 TTS model"
    },
    "audio_info": {
      "bitrate": "352.8 kbps",
      "encoding": "PCM 16-bit",
      "size_bytes": 141120
    },
    "processing_info": {
      "gpu_used": "NVIDIA GeForce RTX 4090",
      "inference_time": 1.2,
      "preprocessing_time": 0.3,
      "postprocessing_time": 0.3
    }
  }
}
```

### Examples

#### Basic TTS
```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, welcome to our text-to-speech service!"
  }'
```

#### Advanced TTS with Custom Settings
```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test with custom voice settings.",
    "model": "bark_tts",
    "speed": 1.2,
    "pitch": 0.9,
    "volume": 1.1,
    "output_format": "mp3",
    "quality": "high"
  }'
```

#### TTS with Voice Cloning (Bark Model)
```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This text will use a specific voice.",
    "model": "bark_tts",
    "voice": "v2/en_speaker_6",
    "language": "en"
  }'
```

#### Multilingual TTS
```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hola, este es un ejemplo en español.",
    "language": "es",
    "model": "speecht5_tts"
  }'
```

---

## Speech-to-Text Transcription

Convert audio to text using speech recognition models.

### Request
```http
POST /transcribe
Content-Type: multipart/form-data

file: [audio file]
model: whisper-base
language: en
task: transcribe
```

### Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | file | Yes | - | Audio file (wav, mp3, ogg, m4a) |
| `model` | string | No | "whisper-base" | STT model to use |
| `language` | string | No | "auto" | Source language (auto-detect if not specified) |
| `task` | string | No | "transcribe" | Task type (transcribe, translate) |
| `temperature` | float | No | 0.0 | Sampling temperature (0.0-1.0) |
| `no_speech_threshold` | float | No | 0.6 | No-speech detection threshold |
| `logprob_threshold` | float | No | -1.0 | Log probability threshold |

### Response
```json
{
  "success": true,
  "timestamp": "2024-01-15T10:30:00Z",
  "transcription": "Hello, this is a test of the speech recognition system.",
  "segments": [
    {
      "start": 0.0,
      "end": 3.2,
      "text": "Hello, this is a test of the speech recognition system.",
      "confidence": 0.95,
      "no_speech_prob": 0.02
    }
  ],
  "language": "en",
  "language_probability": 0.98,
  "duration": 3.2,
  "model_used": "whisper-base",
  "processing_time": 0.8,
  "metadata": {
    "audio_info": {
      "sample_rate": 16000,
      "channels": 1,
      "format": "wav",
      "duration": 3.2,
      "size_bytes": 102400
    },
    "model_info": {
      "name": "whisper-base",
      "type": "openai-whisper",
      "parameters": "74M"
    }
  }
}
```

### Examples

#### Basic Transcription
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "model=whisper-base"
```

#### Transcription with Language Specification
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@spanish_audio.mp3" \
  -F "model=whisper-base" \
  -F "language=es"
```

#### Translation Task
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@french_audio.wav" \
  -F "model=whisper-base" \
  -F "task=translate" \
  -F "language=fr"
```

---

## Audio Health Check

Check the status and availability of audio processing systems.

### Request
```http
GET /audio/health
```

### Response
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "audio_system": {
    "status": "operational",
    "tts_available": true,
    "stt_available": true,
    "models_loaded": 3
  },
  "tts_models": [
    {
      "name": "speecht5_tts",
      "status": "ready",
      "loaded": true,
      "device": "cuda:0"
    },
    {
      "name": "bark_tts", 
      "status": "ready",
      "loaded": true,
      "device": "cuda:0"
    }
  ],
  "stt_models": [
    {
      "name": "whisper-base",
      "status": "ready",
      "loaded": true,
      "device": "cuda:0"
    }
  ],
  "supported_formats": {
    "input": ["wav", "mp3", "ogg", "m4a", "flac"],
    "output": ["wav", "mp3", "ogg"]
  },
  "system_info": {
    "audio_processing": "cuda",
    "gpu_memory_available": "8192 MB",
    "max_audio_length": 300
  }
}
```

### Example
```bash
curl http://localhost:8000/audio/health
```

---

## TTS Health Check

Detailed health check specifically for text-to-speech functionality.

### Request
```http
GET /tts/health
```

### Response
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "tts_system": {
    "status": "operational",
    "available_models": 2,
    "default_model": "speecht5_tts",
    "total_voices": 15
  },
  "models": [
    {
      "name": "speecht5_tts",
      "status": "ready",
      "loaded": true,
      "device": "cuda:0",
      "supports_voice_cloning": false,
      "supported_languages": ["en", "de", "es", "fr"],
      "max_text_length": 5000,
      "average_generation_time": 1.2
    },
    {
      "name": "bark_tts",
      "status": "ready", 
      "loaded": true,
      "device": "cuda:0",
      "supports_voice_cloning": true,
      "supported_languages": ["en", "de", "es", "fr", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"],
      "max_text_length": 1000,
      "average_generation_time": 8.5
    }
  ],
  "voice_options": {
    "speecht5_tts": ["default"],
    "bark_tts": [
      "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2",
      "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
      "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8", "v2/en_speaker_9"
    ]
  },
  "performance": {
    "requests_per_minute": 12,
    "average_response_time": 3.2,
    "error_rate": 0.02,
    "uptime": "99.8%"
  }
}
```

### Example
```bash
curl http://localhost:8000/tts/health
```

---

## Supported Models

### Text-to-Speech Models

#### SpeechT5
- **Model Name**: `speecht5_tts`
- **Source**: Microsoft
- **Quality**: Very High
- **Speed**: Medium (1-2s per sentence)
- **Languages**: English, German, Spanish, French
- **Features**: Multi-speaker, vocoder required
- **Voice Cloning**: No
- **Max Text**: 5000 characters

#### Bark
- **Model Name**: `bark_tts`
- **Source**: Suno AI
- **Quality**: Very High
- **Speed**: Slow (5-15s per sentence)
- **Languages**: 16+ languages including English, Spanish, French, Chinese, Japanese
- **Features**: Voice cloning, emotional expressions, sound effects
- **Voice Cloning**: Yes (zero-shot)
- **Max Text**: 1000 characters

#### BART (Adapted)
- **Model Name**: `bart_large_tts`
- **Source**: Facebook/Meta
- **Quality**: High
- **Speed**: Medium
- **Languages**: English (primarily)
- **Features**: Requires TTS adaptation
- **Voice Cloning**: No
- **Max Text**: 3000 characters

#### Tacotron2
- **Model Name**: `tacotron2_tts`
- **Source**: NVIDIA/TorchAudio
- **Quality**: High
- **Speed**: Fast
- **Languages**: English
- **Features**: Requires WaveGlow vocoder
- **Voice Cloning**: No
- **Max Text**: 4000 characters

#### VALL-E X
- **Model Name**: `vall_e_x`
- **Source**: Microsoft Research
- **Quality**: Very High
- **Speed**: Very Slow
- **Languages**: English, Chinese
- **Features**: Zero-shot voice cloning, experimental
- **Voice Cloning**: Yes (few-shot)
- **Max Text**: 500 characters

### Speech-to-Text Models

#### Whisper Base
- **Model Name**: `whisper-base`
- **Source**: OpenAI
- **Parameters**: 74M
- **Languages**: 99+ languages
- **Features**: Transcription, translation, timestamps
- **Accuracy**: High
- **Speed**: Fast

#### Whisper Small
- **Model Name**: `whisper-small`
- **Source**: OpenAI
- **Parameters**: 244M
- **Languages**: 99+ languages
- **Features**: Better accuracy than base
- **Accuracy**: Very High
- **Speed**: Medium

#### Whisper Medium
- **Model Name**: `whisper-medium`
- **Source**: OpenAI
- **Parameters**: 769M
- **Languages**: 99+ languages
- **Features**: Production quality
- **Accuracy**: Very High
- **Speed**: Medium-Slow

---

## Voice Configuration

### SpeechT5 Voices
- **Default Voice**: Generic English speaker
- **Configuration**: Built-in speaker embeddings
- **Customization**: Limited

### Bark Voices
Bark offers the most extensive voice options:

#### English Speakers
- `v2/en_speaker_0` - Male, young adult
- `v2/en_speaker_1` - Female, young adult  
- `v2/en_speaker_2` - Male, middle-aged
- `v2/en_speaker_3` - Female, middle-aged
- `v2/en_speaker_4` - Male, elderly
- `v2/en_speaker_5` - Female, elderly
- `v2/en_speaker_6` - Male, deep voice
- `v2/en_speaker_7` - Female, high voice
- `v2/en_speaker_8` - Male, accented
- `v2/en_speaker_9` - Female, accented

#### Other Languages
- German: `v2/de_speaker_*`
- Spanish: `v2/es_speaker_*`
- French: `v2/fr_speaker_*`
- Italian: `v2/it_speaker_*`
- Portuguese: `v2/pt_speaker_*`
- Polish: `v2/pl_speaker_*`
- Turkish: `v2/tr_speaker_*`
- Russian: `v2/ru_speaker_*`
- Dutch: `v2/nl_speaker_*`
- Czech: `v2/cs_speaker_*`
- Arabic: `v2/ar_speaker_*`
- Chinese: `v2/zh_speaker_*`
- Japanese: `v2/ja_speaker_*`
- Hungarian: `v2/hu_speaker_*`
- Korean: `v2/ko_speaker_*`

---

## Audio Formats

### Input Formats (STT)
| Format | Extension | Description |
|--------|-----------|-------------|
| WAV | `.wav` | Uncompressed, highest quality |
| MP3 | `.mp3` | Compressed, widely supported |
| OGG | `.ogg` | Open source, good compression |
| M4A | `.m4a` | Apple format, good quality |
| FLAC | `.flac` | Lossless compression |

### Output Formats (TTS)
| Format | Extension | Quality | File Size | Notes |
|--------|-----------|---------|-----------|--------|
| WAV | `.wav` | Highest | Large | Uncompressed |
| MP3 | `.mp3` | High | Medium | Most compatible |
| OGG | `.ogg` | High | Small | Open source |

---

## Performance Optimization

### TTS Performance Tips
1. **Use appropriate models** for your use case:
   - Fast response: Tacotron2
   - High quality: SpeechT5, Bark
   - Voice cloning: Bark, VALL-E X

2. **Optimize text length**:
   - Break long texts into sentences
   - Stay within model limits
   - Use streaming for long content

3. **GPU optimization**:
   - Use CUDA when available
   - Monitor GPU memory usage
   - Enable model optimization

### STT Performance Tips
1. **Audio preprocessing**:
   - Use 16kHz sample rate
   - Convert to WAV for best compatibility
   - Normalize audio levels

2. **Model selection**:
   - Whisper Base: Fast, good accuracy
   - Whisper Medium: Best balance
   - Whisper Large: Highest accuracy

3. **Language specification**:
   - Specify language when known
   - Use auto-detection sparingly
   - Consider translation tasks

---

## Error Handling

### Common TTS Errors

#### Text Too Long
```json
{
  "success": false,
  "error": "Text too long for model 'bark_tts'",
  "message": "Text length 1500 exceeds maximum 1000 characters for this model",
  "max_length": 1000,
  "current_length": 1500
}
```

#### Model Not Available
```json
{
  "success": false,
  "error": "Model not available",
  "message": "TTS model 'unknown_model' is not available or not loaded",
  "available_models": ["speecht5_tts", "bark_tts"]
}
```

#### Invalid Voice
```json
{
  "success": false,
  "error": "Invalid voice",
  "message": "Voice 'invalid_voice' not supported for model 'speecht5_tts'",
  "supported_voices": ["default"]
}
```

### Common STT Errors

#### Unsupported Format
```json
{
  "success": false,
  "error": "Unsupported audio format",
  "message": "File format '.avi' is not supported for transcription",
  "supported_formats": ["wav", "mp3", "ogg", "m4a", "flac"]
}
```

#### File Too Large
```json
{
  "success": false,
  "error": "File too large",
  "message": "Audio file exceeds maximum duration of 300 seconds",
  "max_duration": 300,
  "file_duration": 450
}
```

#### No Audio Detected
```json
{
  "success": false,
  "error": "No audio detected",
  "message": "No speech detected in the provided audio file",
  "no_speech_probability": 0.95
}
```

---

## Integration Examples

### Python Client Example
```python
import requests
import base64
import json

# TTS Example
def synthesize_text(text, voice="default", model="speecht5_tts"):
    url = "http://localhost:8000/synthesize"
    payload = {
        "text": text,
        "voice": voice,
        "model": model,
        "output_format": "wav"
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        # Decode base64 audio data
        audio_data = base64.b64decode(result["audio_data"])
        
        # Save to file
        with open("output.wav", "wb") as f:
            f.write(audio_data)
        
        return result
    else:
        return {"error": response.text}

# STT Example
def transcribe_audio(audio_file_path, model="whisper-base"):
    url = "http://localhost:8000/transcribe"
    
    with open(audio_file_path, "rb") as f:
        files = {"file": f}
        data = {"model": model}
        
        response = requests.post(url, files=files, data=data)
        
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

# Usage
tts_result = synthesize_text("Hello, this is a test!")
stt_result = transcribe_audio("audio.wav")
```

### JavaScript Client Example
```javascript
// TTS Example
async function synthesizeText(text, options = {}) {
    const response = await fetch('http://localhost:8000/synthesize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: text,
            voice: options.voice || 'default',
            model: options.model || 'speecht5_tts',
            output_format: options.format || 'wav'
        })
    });
    
    if (response.ok) {
        const result = await response.json();
        // Convert base64 to blob for browser playback
        const audioData = atob(result.audio_data);
        const bytes = new Uint8Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            bytes[i] = audioData.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(blob);
        
        return { ...result, audioUrl };
    } else {
        throw new Error(`TTS Error: ${response.statusText}`);
    }
}

// STT Example
async function transcribeAudio(audioFile, model = 'whisper-base') {
    const formData = new FormData();
    formData.append('file', audioFile);
    formData.append('model', model);
    
    const response = await fetch('http://localhost:8000/transcribe', {
        method: 'POST',
        body: formData
    });
    
    if (response.ok) {
        return await response.json();
    } else {
        throw new Error(`STT Error: ${response.statusText}`);
    }
}

// Usage
synthesizeText("Hello, this is a test!")
    .then(result => {
        console.log('TTS Result:', result);
        // Play audio
        const audio = new Audio(result.audioUrl);
        audio.play();
    })
    .catch(error => console.error(error));
```

### cURL Examples Collection
```bash
# Basic TTS
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!"}'

# High-quality TTS with Bark
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is high-quality speech synthesis.",
    "model": "bark_tts",
    "voice": "v2/en_speaker_1",
    "quality": "high",
    "output_format": "mp3"
  }'

# Basic STT
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "model=whisper-base"

# STT with language detection
curl -X POST http://localhost:8000/transcribe \
  -F "file=@multilingual.mp3" \
  -F "model=whisper-medium" \
  -F "language=auto"

# Health checks
curl http://localhost:8000/audio/health
curl http://localhost:8000/tts/health
```

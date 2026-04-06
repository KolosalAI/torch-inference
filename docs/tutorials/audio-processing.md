# Audio Processing Tutorial

Complete guide to using Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities in the PyTorch Inference Framework.

## 📚 Prerequisites

- Basic Python knowledge  
- Completed [basic usage tutorial](basic-usage.md)
- Audio libraries installed (see [installation guide](../guides/installation.md))

## 🎯 Learning Objectives

By the end of this tutorial, you will:
- ✅ Configure audio processing capabilities
- ✅ Generate speech from text (TTS)
- ✅ Transcribe speech to text (STT)
- ✅ Handle different audio formats and models
- ✅ Implement real-time audio processing
- ✅ Optimize audio processing performance

## 🔧 Setup and Configuration

### Initialize Framework with Audio

```python
# audio_setup.py
from framework import TorchInferenceFramework
from framework.core.config import AudioConfig, TTSConfig, STTConfig

# Configure TTS
tts_config = TTSConfig(
    default_model="speecht5_tts",
    default_voice="default",
    default_language="en",
    auto_download_models=True,
    supported_formats=['wav', 'mp3', 'flac'],
    default_sample_rate=16000
)

# Configure STT
stt_config = STTConfig(
    default_model="whisper-base",
    default_language="auto",
    enable_timestamps=True,
    beam_size=5,
    temperature=0.0
)

# Create audio configuration
audio_config = AudioConfig(
    enable_tts=True,
    enable_stt=True,
    tts=tts_config,
    stt=stt_config
)

# Initialize framework with audio support
framework = TorchInferenceFramework(audio_config=audio_config)

print("✅ Framework initialized with audio support")
print(f"TTS models available: {framework.list_tts_models()}")
print(f"STT models available: {framework.list_stt_models()}")
```

### Verify Audio Setup

```python
# Check audio capabilities
audio_info = framework.get_audio_info()
print("Audio Configuration:")
print(f"  TTS enabled: {audio_info['tts_enabled']}")
print(f"  STT enabled: {audio_info['stt_enabled']}")
print(f"  Supported formats: {audio_info['supported_formats']}")
print(f"  Default sample rate: {audio_info['sample_rate']} Hz")
```

## 🗣️ Text-to-Speech (TTS)

### Basic TTS Usage

```python
# Basic TTS synthesis
def basic_tts_example():
    """Basic text-to-speech example."""
    
    # Simple text synthesis
    text = "Hello! Welcome to the PyTorch Inference Framework audio tutorial."
    
    # Generate speech
    audio_result = framework.synthesize_speech(
        text=text,
        model="speecht5_tts",
        voice="default",
        language="en"
    )
    
    print(f"✅ Speech generated successfully")
    print(f"Audio duration: {audio_result['duration_seconds']:.2f} seconds")
    print(f"Sample rate: {audio_result['sample_rate']} Hz")
    print(f"Audio shape: {audio_result['audio_data'].shape}")
    
    # Save audio file
    framework.save_audio(
        audio_data=audio_result['audio_data'],
        file_path="output/hello_world.wav",
        sample_rate=audio_result['sample_rate']
    )
    
    return audio_result

# Run basic example
audio_result = basic_tts_example()
```

### Advanced TTS Configuration

```python
# Advanced TTS with custom settings
def advanced_tts_example():
    """Advanced TTS with custom configuration."""
    
    text = """
    The PyTorch Inference Framework provides state-of-the-art 
    text-to-speech capabilities with multiple models and voices.
    You can customize speed, pitch, and emotion to create 
    natural-sounding speech for any application.
    """
    
    # Custom TTS parameters
    tts_params = {
        'model': 'speecht5_tts',
        'voice': 'default',
        'language': 'en',
        'speed': 1.0,          # Speech speed (0.5 - 2.0)
        'pitch': 0.0,          # Pitch adjustment (-10 to +10)
        'volume': 1.0,         # Volume (0.0 - 2.0)
        'emotion': 'neutral',   # Emotion (neutral, happy, sad)
        'sample_rate': 22050,   # Output sample rate
        'format': 'wav'         # Output format
    }
    
    # Generate speech with custom parameters
    audio_result = framework.synthesize_speech(
        text=text,
        **tts_params
    )
    
    # Process and enhance audio
    enhanced_audio = framework.enhance_audio(
        audio_data=audio_result['audio_data'],
        noise_reduction=True,
        normalize=True,
        compress=False
    )
    
    # Save enhanced audio
    framework.save_audio(
        audio_data=enhanced_audio['audio_data'],
        file_path="output/advanced_tts.wav",
        sample_rate=tts_params['sample_rate']
    )
    
    return enhanced_audio

enhanced_result = advanced_tts_example()
```

### Multiple TTS Models

```python
# Compare different TTS models
def compare_tts_models():
    """Compare different TTS models."""
    
    text = "This is a test of different text-to-speech models."
    
    # Available TTS models
    models = [
        'speecht5_tts',    # Microsoft SpeechT5
        'bark_tts',        # Suno Bark
        'tacotron2',       # NVIDIA Tacotron2
        'fastpitch'        # NVIDIA FastPitch
    ]
    
    results = {}
    
    for model in models:
        try:
            print(f"Testing {model}...")
            
            start_time = time.time()
            audio_result = framework.synthesize_speech(
                text=text,
                model=model,
                voice="default"
            )
            end_time = time.time()
            
            results[model] = {
                'duration': audio_result['duration_seconds'],
                'synthesis_time': end_time - start_time,
                'quality_score': framework.evaluate_audio_quality(
                    audio_result['audio_data']
                ),
                'file_path': f"output/comparison_{model}.wav"
            }
            
            # Save audio
            framework.save_audio(
                audio_data=audio_result['audio_data'],
                file_path=results[model]['file_path'],
                sample_rate=audio_result['sample_rate']
            )
            
        except Exception as e:
            print(f"❌ {model} failed: {e}")
            results[model] = {'error': str(e)}
    
    # Print comparison
    print("\n📊 TTS Model Comparison:")
    for model, data in results.items():
        if 'error' not in data:
            print(f"{model}:")
            print(f"  Duration: {data['duration']:.2f}s")
            print(f"  Synthesis time: {data['synthesis_time']:.2f}s")
            print(f"  Quality score: {data['quality_score']:.3f}")
    
    return results

comparison_results = compare_tts_models()
```

### Voice Cloning

```python
# Voice cloning example (with Bark model)
def voice_cloning_example():
    """Voice cloning with Bark TTS model."""
    
    # Load reference audio for voice cloning
    reference_audio_path = "reference_voices/speaker1.wav"
    
    # Clone voice from reference
    cloned_voice = framework.clone_voice(
        reference_audio_path=reference_audio_path,
        model="bark_tts",
        voice_name="custom_speaker1"
    )
    
    print(f"✅ Voice cloned: {cloned_voice['voice_id']}")
    
    # Use cloned voice for synthesis
    text = "Hello, this is my cloned voice speaking!"
    
    cloned_speech = framework.synthesize_speech(
        text=text,
        model="bark_tts",
        voice=cloned_voice['voice_id']
    )
    
    # Save cloned speech
    framework.save_audio(
        audio_data=cloned_speech['audio_data'],
        file_path="output/cloned_voice_speech.wav",
        sample_rate=cloned_speech['sample_rate']
    )
    
    return cloned_speech

# Note: Voice cloning requires reference audio
# cloned_result = voice_cloning_example()
```

## 🎤 Speech-to-Text (STT)

### Basic STT Usage

```python
# Basic speech transcription
def basic_stt_example():
    """Basic speech-to-text example."""
    
    # Load audio file
    audio_file_path = "examples/speech_sample.wav"
    
    # Transcribe audio
    transcription_result = framework.transcribe_audio(
        audio_path=audio_file_path,
        model="whisper-base",
        language="auto"
    )
    
    print("🎤 Transcription Results:")
    print(f"Text: {transcription_result['text']}")
    print(f"Language: {transcription_result['language']}")
    print(f"Confidence: {transcription_result['confidence']:.3f}")
    print(f"Processing time: {transcription_result['processing_time_ms']:.2f} ms")
    
    # Print word-level timestamps if available
    if 'words' in transcription_result:
        print("\n📝 Word-level timestamps:")
        for word_info in transcription_result['words'][:10]:  # First 10 words
            print(f"  {word_info['word']}: "
                  f"{word_info['start']:.2f}s - {word_info['end']:.2f}s "
                  f"(confidence: {word_info['confidence']:.3f})")
    
    return transcription_result

# Run basic STT example
transcription = basic_stt_example()
```

### Advanced STT with Multiple Models

```python
# Compare different STT models
def compare_stt_models():
    """Compare different STT models."""
    
    audio_file = "examples/speech_sample.wav"
    
    # Available STT models
    models = [
        'whisper-tiny',     # Fastest, least accurate
        'whisper-base',     # Balanced
        'whisper-small',    # Better accuracy
        'whisper-medium',   # High accuracy
        'whisper-large',    # Best accuracy
        'wav2vec2',         # Facebook Wav2Vec2
        'deepspeech'        # Mozilla DeepSpeech
    ]
    
    results = {}
    
    for model in models:
        try:
            print(f"Testing {model}...")
            
            start_time = time.time()
            result = framework.transcribe_audio(
                audio_path=audio_file,
                model=model,
                language="auto",
                enable_timestamps=True
            )
            end_time = time.time()
            
            results[model] = {
                'text': result['text'],
                'confidence': result.get('confidence', 0.0),
                'processing_time': end_time - start_time,
                'word_count': len(result['text'].split()),
                'language': result.get('language', 'unknown')
            }
            
        except Exception as e:
            print(f"❌ {model} failed: {e}")
            results[model] = {'error': str(e)}
    
    # Print comparison
    print("\n📊 STT Model Comparison:")
    for model, data in results.items():
        if 'error' not in data:
            print(f"{model}:")
            print(f"  Text: {data['text'][:100]}...")
            print(f"  Confidence: {data['confidence']:.3f}")
            print(f"  Processing time: {data['processing_time']:.2f}s")
            print(f"  Words per second: {data['word_count']/data['processing_time']:.1f}")
    
    return results

stt_comparison = compare_stt_models()
```

### Real-time STT

```python
import pyaudio
import numpy as np
import threading
import queue

# Real-time speech transcription
class RealTimeSTT:
    """Real-time speech-to-text processor."""
    
    def __init__(self, framework, model="whisper-base"):
        self.framework = framework
        self.model = model
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
    def start_recording(self):
        """Start real-time recording and transcription."""
        self.is_recording = True
        
        # Open audio stream
        stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        # Start transcription thread
        transcription_thread = threading.Thread(
            target=self._transcription_worker
        )
        transcription_thread.start()
        
        print("🎤 Recording started. Speak into the microphone...")
        
        try:
            stream.start_stream()
            while self.is_recording:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping recording...")
        finally:
            self.stop_recording()
            stream.stop_stream()
            stream.close()
    
    def stop_recording(self):
        """Stop recording."""
        self.is_recording = False
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback."""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def _transcription_worker(self):
        """Worker thread for transcription."""
        audio_buffer = []
        buffer_duration = 3.0  # seconds
        buffer_size = int(self.sample_rate * buffer_duration)
        
        while self.is_recording:
            try:
                # Get audio data
                audio_chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer.extend(audio_chunk)
                
                # Process when buffer is full
                if len(audio_buffer) >= buffer_size:
                    # Convert to tensor
                    audio_tensor = torch.from_numpy(
                        np.array(audio_buffer, dtype=np.float32)
                    )
                    
                    # Transcribe
                    result = self.framework.transcribe_audio_tensor(
                        audio_tensor=audio_tensor,
                        sample_rate=self.sample_rate,
                        model=self.model
                    )
                    
                    # Print transcription
                    if result['text'].strip():
                        print(f"🗣️  {result['text']}")
                    
                    # Keep overlap for continuity
                    overlap_size = buffer_size // 4
                    audio_buffer = audio_buffer[-overlap_size:]
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Transcription error: {e}")

# Use real-time STT
def real_time_stt_example():
    """Real-time STT example."""
    
    realtime_stt = RealTimeSTT(framework)
    
    try:
        realtime_stt.start_recording()
    except KeyboardInterrupt:
        print("Recording stopped by user")
    finally:
        realtime_stt.stop_recording()

# Run real-time example (uncomment to use)
# real_time_stt_example()
```

## 🔄 Audio Format Handling

### Format Conversion

```python
# Audio format conversion utilities
def convert_audio_formats():
    """Convert between different audio formats."""
    
    # Load audio in various formats
    audio_files = {
        'wav': 'examples/sample.wav',
        'mp3': 'examples/sample.mp3',
        'flac': 'examples/sample.flac',
        'm4a': 'examples/sample.m4a'
    }
    
    for format_name, file_path in audio_files.items():
        if os.path.exists(file_path):
            print(f"Processing {format_name} file...")
            
            # Load audio
            audio_data, sample_rate = framework.load_audio(file_path)
            
            print(f"  Original format: {format_name}")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Duration: {len(audio_data) / sample_rate:.2f} seconds")
            
            # Convert to different formats
            output_formats = ['wav', 'mp3', 'flac']
            
            for output_format in output_formats:
                output_path = f"output/converted_{format_name}_to_{output_format}.{output_format}"
                
                framework.save_audio(
                    audio_data=audio_data,
                    file_path=output_path,
                    sample_rate=sample_rate,
                    format=output_format
                )
                
                print(f"  ✅ Converted to {output_format}: {output_path}")

convert_audio_formats()
```

### Audio Preprocessing

```python
# Audio preprocessing for better TTS/STT results
def preprocess_audio():
    """Preprocess audio for optimal TTS/STT performance."""
    
    audio_file = "examples/noisy_speech.wav"
    audio_data, sample_rate = framework.load_audio(audio_file)
    
    print("Original audio stats:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {len(audio_data) / sample_rate:.2f} seconds")
    print(f"  Max amplitude: {np.max(np.abs(audio_data)):.3f}")
    
    # Apply preprocessing steps
    processed_audio = framework.preprocess_audio(
        audio_data=audio_data,
        sample_rate=sample_rate,
        
        # Noise reduction
        noise_reduction=True,
        noise_reduction_strength=0.5,
        
        # Normalization
        normalize=True,
        target_lufs=-23.0,
        
        # Resampling
        target_sample_rate=16000,
        
        # Filtering
        high_pass_freq=80,    # Remove low-frequency noise
        low_pass_freq=8000,   # Remove high-frequency noise
        
        # Enhancement
        enhance_speech=True,
        auto_gain_control=True
    )
    
    print("\nProcessed audio stats:")
    print(f"  Sample rate: {processed_audio['sample_rate']} Hz")
    print(f"  Duration: {len(processed_audio['audio_data']) / processed_audio['sample_rate']:.2f} seconds")
    print(f"  Max amplitude: {np.max(np.abs(processed_audio['audio_data'])):.3f}")
    
    # Save processed audio
    framework.save_audio(
        audio_data=processed_audio['audio_data'],
        file_path="output/preprocessed_audio.wav",
        sample_rate=processed_audio['sample_rate']
    )
    
    # Compare transcription quality
    original_transcription = framework.transcribe_audio_tensor(
        audio_tensor=torch.from_numpy(audio_data),
        sample_rate=sample_rate
    )
    
    processed_transcription = framework.transcribe_audio_tensor(
        audio_tensor=torch.from_numpy(processed_audio['audio_data']),
        sample_rate=processed_audio['sample_rate']
    )
    
    print(f"\nOriginal transcription: {original_transcription['text']}")
    print(f"Original confidence: {original_transcription['confidence']:.3f}")
    print(f"\nProcessed transcription: {processed_transcription['text']}")
    print(f"Processed confidence: {processed_transcription['confidence']:.3f}")
    
    return processed_audio

processed_result = preprocess_audio()
```

## 🎛️ Audio Pipeline Integration

### TTS + STT Round-trip

```python
# Complete TTS -> STT pipeline
def tts_stt_roundtrip():
    """Test TTS and STT with round-trip processing."""
    
    original_text = "The quick brown fox jumps over the lazy dog."
    
    print(f"Original text: {original_text}")
    
    # Step 1: Text to Speech
    print("\n🗣️  Step 1: Text to Speech")
    tts_result = framework.synthesize_speech(
        text=original_text,
        model="speecht5_tts",
        voice="default"
    )
    
    # Save intermediate audio
    audio_path = "output/roundtrip_audio.wav"
    framework.save_audio(
        audio_data=tts_result['audio_data'],
        file_path=audio_path,
        sample_rate=tts_result['sample_rate']
    )
    
    print(f"✅ Speech generated: {tts_result['duration_seconds']:.2f}s")
    
    # Step 2: Speech to Text
    print("\n🎤 Step 2: Speech to Text")
    stt_result = framework.transcribe_audio(
        audio_path=audio_path,
        model="whisper-base"
    )
    
    transcribed_text = stt_result['text'].strip()
    print(f"✅ Speech transcribed: {transcribed_text}")
    
    # Compare texts
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, original_text.lower(), transcribed_text.lower()).ratio()
    
    print(f"\n📊 Round-trip Results:")
    print(f"Original:    {original_text}")
    print(f"Transcribed: {transcribed_text}")
    print(f"Similarity:  {similarity:.3f} ({similarity*100:.1f}%)")
    
    return {
        'original_text': original_text,
        'transcribed_text': transcribed_text,
        'similarity': similarity,
        'tts_time': tts_result.get('synthesis_time_ms', 0),
        'stt_time': stt_result.get('processing_time_ms', 0)
    }

roundtrip_result = tts_stt_roundtrip()
```

### Batch Audio Processing

```python
# Batch processing for multiple audio files
def batch_audio_processing():
    """Process multiple audio files in batch."""
    
    # Get all audio files
    import glob
    audio_files = glob.glob("examples/*.wav") + glob.glob("examples/*.mp3")
    
    if not audio_files:
        print("❌ No audio files found in examples directory")
        return
    
    print(f"📁 Processing {len(audio_files)} audio files...")
    
    results = []
    
    for i, audio_file in enumerate(audio_files):
        print(f"\n🎵 Processing {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
        
        try:
            # Transcribe audio
            transcription = framework.transcribe_audio(
                audio_path=audio_file,
                model="whisper-base",
                language="auto"
            )
            
            # Generate summary if text is long
            text = transcription['text']
            if len(text) > 200:
                summary = framework.summarize_text(text, max_length=50)
            else:
                summary = text
            
            # Convert back to speech with different voice
            tts_result = framework.synthesize_speech(
                text=summary,
                model="speecht5_tts",
                voice="default"
            )
            
            # Save processed audio
            output_path = f"output/processed_{i+1:03d}_{os.path.basename(audio_file)}"
            framework.save_audio(
                audio_data=tts_result['audio_data'],
                file_path=output_path,
                sample_rate=tts_result['sample_rate']
            )
            
            results.append({
                'input_file': audio_file,
                'output_file': output_path,
                'original_text': text,
                'summary': summary,
                'confidence': transcription.get('confidence', 0.0),
                'processing_time': transcription.get('processing_time_ms', 0)
            })
            
            print(f"  ✅ Processed successfully")
            print(f"  Original length: {len(text)} characters")
            print(f"  Summary length: {len(summary)} characters")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'input_file': audio_file,
                'error': str(e)
            })
    
    # Print summary
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"\n📊 Batch Processing Summary:")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if successful:
        avg_confidence = np.mean([r['confidence'] for r in successful])
        total_time = sum([r['processing_time'] for r in successful])
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  Total processing time: {total_time/1000:.2f} seconds")
    
    return results

batch_results = batch_audio_processing()
```

## 🚀 Performance Optimization

### Audio Processing Optimization

```python
# Optimize audio processing performance
def optimize_audio_performance():
    """Optimize audio processing for better performance."""
    
    # Configure for performance
    audio_config = {
        'batch_processing': True,
        'parallel_workers': 4,
        'memory_optimization': True,
        'cache_models': True,
        'use_gpu_acceleration': True,
        'streaming_processing': True
    }
    
    framework.configure_audio_processing(**audio_config)
    
    # Benchmark different configurations
    test_text = "This is a performance test for the audio processing system."
    
    configs = [
        {'model': 'speecht5_tts', 'optimization': 'none'},
        {'model': 'speecht5_tts', 'optimization': 'basic'},
        {'model': 'speecht5_tts', 'optimization': 'aggressive'},
    ]
    
    results = {}
    
    for config in configs:
        print(f"Testing {config['optimization']} optimization...")
        
        # Measure TTS performance
        start_time = time.time()
        
        for i in range(10):  # 10 iterations
            tts_result = framework.synthesize_speech(
                text=test_text,
                model=config['model'],
                optimization_level=config['optimization']
            )
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        
        results[config['optimization']] = {
            'avg_synthesis_time': avg_time,
            'throughput': len(test_text) / avg_time,  # chars per second
            'memory_usage': framework.get_memory_usage()['audio_processing_mb']
        }
        
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Throughput: {results[config['optimization']]['throughput']:.1f} chars/s")
    
    # Print comparison
    print("\n📊 Performance Comparison:")
    for optimization, metrics in results.items():
        print(f"{optimization}:")
        print(f"  Time: {metrics['avg_synthesis_time']:.3f}s")
        print(f"  Throughput: {metrics['throughput']:.1f} chars/s")
        print(f"  Memory: {metrics['memory_usage']:.1f} MB")
    
    return results

performance_results = optimize_audio_performance()
```

### Streaming Audio Processing

```python
# Streaming audio for real-time applications
async def streaming_audio_example():
    """Streaming audio processing example."""
    
    # Enable streaming mode
    framework.enable_streaming_mode()
    
    # Long text for streaming
    long_text = """
    This is a demonstration of streaming text-to-speech processing.
    The framework can generate audio in real-time as text is provided,
    allowing for interactive applications and reduced latency.
    This approach is particularly useful for chatbots, virtual assistants,
    and other real-time speech applications.
    """
    
    # Split text into chunks
    sentences = long_text.strip().split('.')
    
    print("🎵 Starting streaming TTS...")
    
    # Stream audio generation
    audio_stream = framework.stream_speech_synthesis(
        model="speecht5_tts",
        voice="default",
        chunk_size=256,  # Process in small chunks
        buffer_size=2048
    )
    
    # Process text chunks
    for i, sentence in enumerate(sentences):
        if sentence.strip():
            print(f"Processing chunk {i+1}: {sentence.strip()}...")
            
            # Send text chunk to stream
            audio_chunk = await audio_stream.synthesize_chunk(
                text=sentence.strip() + ".",
                chunk_id=i
            )
            
            # Play or save audio chunk immediately
            output_path = f"output/stream_chunk_{i:03d}.wav"
            framework.save_audio(
                audio_data=audio_chunk['audio_data'],
                file_path=output_path,
                sample_rate=audio_chunk['sample_rate']
            )
            
            print(f"  ✅ Chunk {i+1} completed ({audio_chunk['duration_seconds']:.2f}s)")
    
    # Close stream
    await audio_stream.close()
    
    print("✅ Streaming TTS completed")

# Run streaming example (uncomment to use)
# asyncio.run(streaming_audio_example())
```

## 🧪 Testing and Validation

### Audio Quality Testing

```python
# Test audio quality and accuracy
def test_audio_quality():
    """Test audio quality and transcription accuracy."""
    
    # Test sentences with different complexity
    test_sentences = [
        "Hello world, this is a simple test.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence and machine learning are transforming technology.",
        "Supercalifragilisticexpialidocious is a very long word.",
        "Speech recognition accuracy depends on audio quality and model performance."
    ]
    
    results = []
    
    for i, sentence in enumerate(test_sentences):
        print(f"\n🧪 Test {i+1}: {sentence}")
        
        # Generate speech
        tts_result = framework.synthesize_speech(
            text=sentence,
            model="speecht5_tts"
        )
        
        # Save audio
        audio_path = f"output/test_audio_{i+1}.wav"
        framework.save_audio(
            audio_data=tts_result['audio_data'],
            file_path=audio_path,
            sample_rate=tts_result['sample_rate']
        )
        
        # Transcribe back
        stt_result = framework.transcribe_audio(
            audio_path=audio_path,
            model="whisper-base"
        )
        
        # Calculate accuracy
        from difflib import SequenceMatcher
        accuracy = SequenceMatcher(
            None, 
            sentence.lower(), 
            stt_result['text'].lower()
        ).ratio()
        
        # Audio quality metrics
        quality_metrics = framework.analyze_audio_quality(
            audio_data=tts_result['audio_data'],
            sample_rate=tts_result['sample_rate']
        )
        
        test_result = {
            'original': sentence,
            'transcribed': stt_result['text'],
            'accuracy': accuracy,
            'confidence': stt_result.get('confidence', 0.0),
            'audio_quality': quality_metrics,
            'synthesis_time': tts_result.get('synthesis_time_ms', 0),
            'transcription_time': stt_result.get('processing_time_ms', 0)
        }
        
        results.append(test_result)
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Confidence: {test_result['confidence']:.3f}")
        print(f"  Audio quality: {quality_metrics['overall_score']:.3f}")
    
    # Summary statistics
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_confidence = np.mean([r['confidence'] for r in results])
    avg_quality = np.mean([r['audio_quality']['overall_score'] for r in results])
    
    print(f"\n📊 Quality Test Summary:")
    print(f"  Average accuracy: {avg_accuracy:.3f}")
    print(f"  Average confidence: {avg_confidence:.3f}")
    print(f"  Average audio quality: {avg_quality:.3f}")
    
    return results

quality_results = test_audio_quality()
```

## 🎯 Practical Applications

### Voice Assistant Demo

```python
# Simple voice assistant implementation
class VoiceAssistant:
    """Simple voice assistant using TTS and STT."""
    
    def __init__(self, framework):
        self.framework = framework
        self.conversation_history = []
        
    def listen(self, audio_path=None):
        """Listen to user input."""
        if audio_path:
            # Transcribe from file
            result = self.framework.transcribe_audio(
                audio_path=audio_path,
                model="whisper-base"
            )
        else:
            # Real-time listening (simplified)
            print("🎤 Listening... (speak now)")
            # In a real implementation, you would use microphone input
            result = {'text': 'What is the weather today?'}
        
        user_input = result['text'].strip()
        self.conversation_history.append(('user', user_input))
        
        return user_input
    
    def think(self, user_input):
        """Process user input and generate response."""
        # Simple response logic (replace with actual AI/NLP)
        responses = {
            'weather': "The weather is sunny and 72 degrees.",
            'time': f"The current time is {time.strftime('%I:%M %p')}.",
            'hello': "Hello! How can I help you today?",
            'goodbye': "Goodbye! Have a great day!",
            'default': "I'm sorry, I didn't understand that. Could you repeat?"
        }
        
        user_input_lower = user_input.lower()
        
        if 'weather' in user_input_lower:
            response = responses['weather']
        elif 'time' in user_input_lower:
            response = responses['time']
        elif any(word in user_input_lower for word in ['hello', 'hi', 'hey']):
            response = responses['hello']
        elif any(word in user_input_lower for word in ['goodbye', 'bye', 'exit']):
            response = responses['goodbye']
        else:
            response = responses['default']
        
        self.conversation_history.append(('assistant', response))
        return response
    
    def speak(self, text):
        """Convert text to speech."""
        print(f"🤖 Assistant: {text}")
        
        # Generate speech
        audio_result = self.framework.synthesize_speech(
            text=text,
            model="speecht5_tts",
            voice="default"
        )
        
        # Save audio
        timestamp = int(time.time())
        audio_path = f"output/assistant_response_{timestamp}.wav"
        
        self.framework.save_audio(
            audio_data=audio_result['audio_data'],
            file_path=audio_path,
            sample_rate=audio_result['sample_rate']
        )
        
        return audio_path
    
    def conversation_loop(self):
        """Main conversation loop."""
        print("🤖 Voice Assistant started. Say 'goodbye' to exit.")
        
        while True:
            try:
                # Listen to user
                user_input = self.listen()
                print(f"👤 User: {user_input}")
                
                # Check for exit
                if any(word in user_input.lower() for word in ['goodbye', 'bye', 'exit']):
                    response = self.think(user_input)
                    self.speak(response)
                    break
                
                # Generate response
                response = self.think(user_input)
                
                # Speak response
                self.speak(response)
                
            except KeyboardInterrupt:
                print("\n🛑 Assistant stopped.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

# Demo voice assistant
def voice_assistant_demo():
    """Demonstrate voice assistant functionality."""
    
    assistant = VoiceAssistant(framework)
    
    # Simulate conversation with audio files
    test_inputs = [
        "examples/hello.wav",  # "Hello, how are you?"
        "examples/weather.wav",  # "What's the weather like?"
        "examples/time.wav",    # "What time is it?"
        "examples/goodbye.wav"  # "Goodbye"
    ]
    
    for audio_file in test_inputs:
        if os.path.exists(audio_file):
            user_input = assistant.listen(audio_file)
            print(f"👤 User: {user_input}")
            
            response = assistant.think(user_input)
            assistant.speak(response)
            
            if 'goodbye' in user_input.lower():
                break
        else:
            print(f"⚠️  Audio file not found: {audio_file}")

# Run voice assistant demo
voice_assistant_demo()
```

## 📚 Next Steps

Now that you understand audio processing, explore these related topics:

1. **[Advanced Features Tutorial](advanced-features.md)** - Explore advanced framework capabilities
2. **[Production API Tutorial](production-api.md)** - Deploy audio APIs to production
3. **[Performance Optimization Guide](../guides/optimization.md)** - Optimize audio processing performance
4. **[Configuration Guide](../guides/configuration.md)** - Configure audio settings

## 🔍 Troubleshooting

### Common Audio Issues

**Audio Quality Problems:**
```python
# Check audio quality
quality = framework.analyze_audio_quality(audio_data, sample_rate)
if quality['overall_score'] < 0.7:
    print("⚠️  Poor audio quality detected")
    # Apply enhancement
    enhanced = framework.enhance_audio(audio_data)
```

**Model Loading Issues:**
```python
# Check available models
available_models = framework.list_available_audio_models()
print("Available TTS models:", available_models['tts'])
print("Available STT models:", available_models['stt'])
```

**Performance Issues:**
```python
# Monitor audio processing performance
stats = framework.get_audio_processing_stats()
print(f"TTS average time: {stats['tts_avg_time_ms']} ms")
print(f"STT average time: {stats['stt_avg_time_ms']} ms")
print(f"Memory usage: {stats['memory_usage_mb']} MB")
```

---

This tutorial covers comprehensive audio processing capabilities. Practice with these examples and experiment with different models and configurations!

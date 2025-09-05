#!/usr/bin/env python3
"""
Audio Processing Examples for PyTorch Inference Framework

This script demonstrates how to use the audio processing capabilities
of the torch-inference framework, including TTS synthesis and STT transcription.

Usage:
    python examples/audio_example.py --mode tts --text "Hello, world!"
    python examples/audio_example.py --mode stt --audio-file test.wav
    python examples/audio_example.py --mode demo
"""

import argparse
import asyncio
import aiohttp
import json
import base64
import tempfile
import os
from pathlib import Path
import wave
import numpy as np


class AudioExampleClient:
    """Client for demonstrating audio processing capabilities."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the audio example client."""
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def check_audio_health(self):
        """Check if audio processing is available."""
        try:
            async with self.session.get(f"{self.base_url}/audio/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print("\n🔍 Audio Health Check:")
                    print(f"  Audio Available: {health_data.get('audio_available', False)}")
                    print(f"  TTS Available: {health_data.get('tts_available', False)}")
                    print(f"  STT Available: {health_data.get('stt_available', False)}")
                    
                    dependencies = health_data.get('dependencies', {})
                    print(f"  Dependencies:")
                    for dep, info in dependencies.items():
                        status = "✅" if info.get('available') else "❌"
                        print(f"    {status} {dep}: {info.get('description', '')}")
                    
                    errors = health_data.get('errors', [])
                    if errors:
                        print(f"  Errors: {errors}")
                    
                    return health_data
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return None
    
    async def list_audio_models(self):
        """List available audio models."""
        try:
            async with self.session.get(f"{self.base_url}/audio/models") as response:
                if response.status == 200:
                    models_data = await response.json()
                    print("\n📋 Available Audio Models:")
                    
                    tts_models = models_data.get('tts_models', [])
                    if tts_models:
                        print(f"  TTS Models: {', '.join(tts_models)}")
                    
                    stt_models = models_data.get('stt_models', [])
                    if stt_models:
                        print(f"  STT Models: {', '.join(stt_models)}")
                    
                    loaded_models = models_data.get('loaded_models', [])
                    if loaded_models:
                        print(f"  Loaded Models: {', '.join(loaded_models)}")
                    
                    return models_data
                else:
                    print(f"❌ Models list failed: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Models list error: {e}")
            return None
    
    async def synthesize_speech(self, text: str, model_name: str = "default", token: str = None, **kwargs):
        """Synthesize speech from text using TTS."""
        request_data = {
            "model_name": model_name,
            "inputs": text,
            "token": token,
            **kwargs
        }
        
        print(f"\n🎙️ Synthesizing speech:")
        print(f"  Text: {text}")
        print(f"  Model: {model_name}")
        
        try:
            async with self.session.post(
                f"{self.base_url}/synthesize",
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if result.get('success'):
                        print(f"  ✅ Synthesis successful!")
                        print(f"  Duration: {result.get('duration', 'N/A'):.2f}s")
                        print(f"  Sample Rate: {result.get('sample_rate', 'N/A')} Hz")
                        print(f"  Processing Time: {result.get('processing_time', 'N/A'):.3f}s")
                        print(f"  Format: {result.get('audio_format', 'N/A')}")
                        
                        # Optionally save audio data
                        audio_data = result.get('audio_data')
                        if audio_data:
                            return self.save_audio_from_base64(audio_data, "tts_output.wav")
                        
                        return result
                    else:
                        print(f"  ❌ Synthesis failed: {result.get('error', 'Unknown error')}")
                        return None
                else:
                    error_text = await response.text()
                    print(f"  ❌ HTTP Error {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            print(f"  ❌ Synthesis error: {e}")
            return None
    
    async def transcribe_audio(self, audio_file: str, model_name: str = "whisper-base", **kwargs):
        """Transcribe audio file to text using STT."""
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return None
        
        print(f"\n🎧 Transcribing audio:")
        print(f"  File: {audio_file}")
        print(f"  Model: {model_name}")
        
        try:
            # Prepare form data
            data = aiohttp.FormData()
            data.add_field('model_name', model_name)
            for key, value in kwargs.items():
                data.add_field(key, str(value))
            
            # Add file
            with open(audio_file, 'rb') as f:
                data.add_field('file', f, filename=os.path.basename(audio_file))
                
                async with self.session.post(
                    f"{self.base_url}/stt/transcribe",
                    data=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if result.get('success'):
                            print(f"  ✅ Transcription successful!")
                            print(f"  Text: '{result.get('text', 'N/A')}'")
                            print(f"  Language: {result.get('language', 'N/A')}")
                            print(f"  Confidence: {result.get('confidence', 'N/A')}")
                            print(f"  Processing Time: {result.get('processing_time', 'N/A'):.3f}s")
                            
                            # Show segments if available
                            segments = result.get('segments', [])
                            if segments:
                                print(f"  Segments ({len(segments)}):")
                                for i, segment in enumerate(segments[:3]):  # Show first 3
                                    start = segment.get('start', 0)
                                    end = segment.get('end', 0)
                                    text = segment.get('text', '')
                                    print(f"    [{start:.1f}s - {end:.1f}s] {text}")
                                if len(segments) > 3:
                                    print(f"    ... and {len(segments) - 3} more segments")
                            
                            return result
                        else:
                            print(f"  ❌ Transcription failed: {result.get('error', 'Unknown error')}")
                            return None
                    else:
                        error_text = await response.text()
                        print(f"  ❌ HTTP Error {response.status}: {error_text}")
                        return None
                        
        except Exception as e:
            print(f"  ❌ Transcription error: {e}")
            return None
    
    def save_audio_from_base64(self, audio_base64: str, output_path: str):
        """Save base64 encoded audio data to file."""
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_base64)
            
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)
            
            print(f"  💾 Audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"  ❌ Failed to save audio: {e}")
            return None
    
    def create_test_audio(self, output_path: str = "test_audio.wav", duration: float = 2.0):
        """Create a test audio file for demonstration."""
        sample_rate = 16000
        samples = int(sample_rate * duration)
        
        # Generate a simple tone (440 Hz - A4 note)
        t = np.linspace(0, duration, samples, False)
        frequency = 440.0
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Add some variation to make it more interesting
        frequency2 = 554.37  # C#5
        audio_data += np.sin(2 * np.pi * frequency2 * t) * 0.2
        
        # Apply fade in/out
        fade_samples = int(0.1 * sample_rate)  # 0.1 second fade
        audio_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # Save as WAV file
        try:
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                # Convert to 16-bit PCM
                audio_16bit = (audio_data * 32767).astype(np.int16)
                wav_file.writeframes(audio_16bit.tobytes())
            
            print(f"📄 Test audio created: {output_path} ({duration}s)")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to create test audio: {e}")
            return None
    
    async def run_demo(self):
        """Run a complete audio processing demonstration."""
        print("🎵 PyTorch Inference Framework - Audio Processing Demo")
        print("=" * 60)
        
        # Check health
        health = await self.check_audio_health()
        if not health or not health.get('audio_available'):
            print("\n⚠️ Audio processing not fully available.")
            print("To use audio features, install dependencies:")
            print("  pip install torch-inference-optimized[audio]")
            return
        
        # List models
        await self.list_audio_models()
        
        # TTS Demo
        print("\n" + "─" * 40)
        print("🎙️ TEXT-TO-SPEECH DEMO")
        print("─" * 40)
        
        demo_texts = [
            "Hello, world! This is a text-to-speech demonstration.",
            "The PyTorch inference framework supports audio processing.",
            "Speech synthesis is working correctly!"
        ]
        
        for i, text in enumerate(demo_texts):
            result = await self.synthesize_speech(
                text=text,
                model_name="default",
                speed=1.0,
                pitch=1.0,
                language="en"
            )
            if result:
                print(f"  TTS Demo {i+1}/3: ✅")
            else:
                print(f"  TTS Demo {i+1}/3: ❌")
        
        # STT Demo
        print("\n" + "─" * 40)
        print("🎧 SPEECH-TO-TEXT DEMO")
        print("─" * 40)
        
        # Create test audio file
        test_audio = self.create_test_audio("demo_audio.wav", duration=3.0)
        if test_audio:
            result = await self.transcribe_audio(
                audio_file=test_audio,
                model_name="whisper-base",
                language="auto",
                enable_timestamps=True
            )
            if result:
                print(f"  STT Demo: ✅")
            else:
                print(f"  STT Demo: ❌")
            
            # Cleanup
            try:
                os.unlink(test_audio)
                print(f"  🗑️ Cleaned up test file: {test_audio}")
            except:
                pass
        
        print("\n🎉 Audio processing demo completed!")


async def main():
    """Main function for the audio example script."""
    parser = argparse.ArgumentParser(description="Audio Processing Examples")
    parser.add_argument("--mode", choices=["tts", "stt", "demo", "health"], 
                       default="demo", help="Example mode to run")
    parser.add_argument("--text", help="Text to synthesize (for TTS mode)")
    parser.add_argument("--audio-file", help="Audio file to transcribe (for STT mode)")
    parser.add_argument("--model", help="Model name to use")
    parser.add_argument("--base-url", default="http://localhost:8000",
                       help="Base URL of the inference server")
    parser.add_argument("--speed", type=float, default=1.0, help="TTS speed")
    parser.add_argument("--pitch", type=float, default=1.0, help="TTS pitch")
    parser.add_argument("--language", default="en", help="Language code")
    
    args = parser.parse_args()
    
    async with AudioExampleClient(args.base_url) as client:
        if args.mode == "health":
            await client.check_audio_health()
            await client.list_audio_models()
            
        elif args.mode == "tts":
            if not args.text:
                print("❌ --text is required for TTS mode")
                return
            
            await client.synthesize_speech(
                text=args.text,
                model_name=args.model or "default",
                speed=args.speed,
                pitch=args.pitch,
                language=args.language
            )
            
        elif args.mode == "stt":
            if not args.audio_file:
                print("❌ --audio-file is required for STT mode")
                return
            
            await client.transcribe_audio(
                audio_file=args.audio_file,
                model_name=args.model or "whisper-base",
                language=args.language
            )
            
        elif args.mode == "demo":
            await client.run_demo()


if __name__ == "__main__":
    asyncio.run(main())

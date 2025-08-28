#!/usr/bin/env python3
"""
Test script to check TTS output and fix WAV corruption issues.
"""

import requests
import json
import base64
import wave
import numpy as np
from pathlib import Path

def test_tts_service():
    """Test the TTS service and analyze the output."""
    
    # Test TTS synthesis
    url = 'http://localhost:8000/tts/synthesize'
    data = {
        'text': 'Hello, this is a test of the text to speech system.',
        'model_name': 'speecht5_tts',
        'output_format': 'wav'
    }

    try:
        print("Testing TTS service...")
        response = requests.post(url, json=data)
        print(f'Status: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            print(f'Success: {result["success"]}')
            
            if result['success']:
                print(f'Sample rate: {result["sample_rate"]}')
                print(f'Duration: {result["duration"]}')
                print(f'Audio format: {result["audio_format"]}')
                
                # Decode audio data
                audio_data = base64.b64decode(result['audio_data'])
                print(f'Audio data length: {len(audio_data)} bytes')
                
                # Check if data is 16-bit PCM
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                print(f'Audio samples: {len(audio_array)}')
                print(f'Audio range: {audio_array.min()} to {audio_array.max()}')
                
                # Save as WAV file with proper header
                with wave.open('test_output_fixed.wav', 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(result['sample_rate'])  
                    wav_file.writeframes(audio_data)
                
                print('Saved as test_output_fixed.wav')
                return True
            else:
                print(f'Error: {result["error"]}')
                return False
        else:
            print(f'HTTP Error: {response.text}')
            return False
            
    except Exception as e:
        print(f'Request failed: {e}')
        return False

def analyze_existing_wav():
    """Analyze the existing corrupted WAV file."""
    wav_path = Path('output_audio.wav')
    
    if wav_path.exists():
        print(f"\nAnalyzing existing WAV file: {wav_path}")
        print(f"File size: {wav_path.stat().st_size} bytes")
        
        try:
            with wave.open(str(wav_path), 'rb') as wav_file:
                print(f"Channels: {wav_file.getnchannels()}")
                print(f"Sample width: {wav_file.getsampwidth()}")
                print(f"Frame rate: {wav_file.getframerate()}")
                print(f"Frames: {wav_file.getnframes()}")
                print(f"Duration: {wav_file.getnframes() / wav_file.getframerate():.2f}s")
        except Exception as e:
            print(f"Error reading WAV file: {e}")
            
            # Try to read as raw bytes
            try:
                with open(wav_path, 'rb') as f:
                    header = f.read(44)  # WAV header is 44 bytes
                    print(f"Header bytes: {header[:12]}")
                    if header[:4] == b'RIFF':
                        print("File has RIFF header")
                    else:
                        print("File does not have RIFF header - this is the problem!")
            except Exception as e2:
                print(f"Error reading raw bytes: {e2}")
    else:
        print("output_audio.wav not found")

def main():
    """Main test function."""
    print("TTS WAV Corruption Analysis Tool")
    print("=" * 40)
    
    # First analyze existing file
    analyze_existing_wav()
    
    # Then test the service
    print("\n" + "=" * 40)
    success = test_tts_service()
    
    if success:
        print("\nTTS service test completed successfully!")
        print("Check test_output_fixed.wav for comparison")
    else:
        print("\nTTS service test failed!")

if __name__ == "__main__":
    main()

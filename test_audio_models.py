#!/usr/bin/env python3
"""
Test script for ONNX Audio Models in Rust Implementation
"""

import requests
import base64
import json
import os
from pathlib import Path

BASE_URL = "http://localhost:8080"

def test_audio_health():
    """Test audio health endpoint"""
    print("=" * 60)
    print("Testing Audio Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/audio/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"✅ Backend: {data['audio_backend']}")
        print(f"✅ Supported Formats: {', '.join(data['supported_formats'])}")
        print(f"✅ Models Available: {len(data['models_available'])} models")
        for model in data['models_available']:
            print(f"   - {model}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_tts_synthesis():
    """Test text-to-speech synthesis"""
    print("\n" + "=" * 60)
    print("Testing Text-to-Speech Synthesis")
    print("=" * 60)
    
    test_texts = [
        "Hello, this is a test of the text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "Testing with different speed and pitch parameters."
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: '{text[:50]}...'")
        
        try:
            payload = {
                "text": text,
                "model": "default",
                "speed": 1.0,
                "pitch": 1.0
            }
            
            response = requests.post(
                f"{BASE_URL}/audio/synthesize",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            audio_data = base64.b64decode(data['audio_base64'])
            
            print(f"✅ Sample Rate: {data['sample_rate']} Hz")
            print(f"✅ Duration: {data['duration_secs']:.2f} seconds")
            print(f"✅ Format: {data['format']}")
            print(f"✅ Audio Size: {len(audio_data)} bytes")
            
            # Save audio file
            output_file = f"tts_output_{i+1}.wav"
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            print(f"✅ Saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    return True

def test_tts_parameters():
    """Test TTS with different parameters"""
    print("\n" + "=" * 60)
    print("Testing TTS Parameter Variations")
    print("=" * 60)
    
    text = "This is a parameter test."
    
    test_cases = [
        {"speed": 0.8, "pitch": 1.0, "desc": "Slow speed"},
        {"speed": 1.2, "pitch": 1.0, "desc": "Fast speed"},
        {"speed": 1.0, "pitch": 0.9, "desc": "Low pitch"},
        {"speed": 1.0, "pitch": 1.1, "desc": "High pitch"},
    ]
    
    for i, params in enumerate(test_cases):
        print(f"\nTest {i+1}: {params['desc']}")
        
        try:
            payload = {
                "text": text,
                "model": "default",
                "speed": params["speed"],
                "pitch": params["pitch"]
            }
            
            response = requests.post(
                f"{BASE_URL}/audio/synthesize",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ Duration: {data['duration_secs']:.2f}s")
            print(f"✅ Speed: {params['speed']}x, Pitch: {params['pitch']}x")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    return True

def test_stt_transcription():
    """Test speech-to-text transcription"""
    print("\n" + "=" * 60)
    print("Testing Speech-to-Text Transcription")
    print("=" * 60)
    
    # Check if we have test audio files
    test_audio_files = [
        "test_audio.wav",
        "tts_output_1.wav"  # Use generated TTS output
    ]
    
    for audio_file in test_audio_files:
        if not os.path.exists(audio_file):
            print(f"⚠️  Skipping {audio_file} (not found)")
            continue
        
        print(f"\nTranscribing: {audio_file}")
        
        try:
            with open(audio_file, 'rb') as f:
                files = {'audio': f}
                data = {
                    'model': 'default',
                    'timestamps': 'true'
                }
                
                response = requests.post(
                    f"{BASE_URL}/audio/transcribe",
                    files=files,
                    data=data
                )
                response.raise_for_status()
                
                result = response.json()
                print(f"✅ Transcription: {result['text']}")
                print(f"✅ Language: {result.get('language', 'N/A')}")
                print(f"✅ Confidence: {result.get('confidence', 0):.2f}")
                
                if result.get('segments'):
                    print(f"✅ Segments: {len(result['segments'])}")
                    for seg in result['segments'][:3]:  # Show first 3
                        print(f"   [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    return True

def test_audio_validation():
    """Test audio validation"""
    print("\n" + "=" * 60)
    print("Testing Audio Validation")
    print("=" * 60)
    
    test_files = [
        ("tts_output_1.wav", True),  # Valid WAV
    ]
    
    for audio_file, should_be_valid in test_files:
        if not os.path.exists(audio_file):
            print(f"⚠️  Skipping {audio_file} (not found)")
            continue
        
        print(f"\nValidating: {audio_file}")
        
        try:
            with open(audio_file, 'rb') as f:
                files = {'audio': f}
                response = requests.post(
                    f"{BASE_URL}/audio/validate",
                    files=files
                )
                response.raise_for_status()
                
                result = response.json()
                is_valid = result['valid']
                
                if is_valid == should_be_valid:
                    print(f"✅ Valid: {is_valid}")
                    if is_valid:
                        print(f"✅ Format: {result['format']}")
                        print(f"✅ Sample Rate: {result['sample_rate']} Hz")
                        print(f"✅ Channels: {result['channels']}")
                        print(f"✅ Duration: {result['duration_secs']:.2f} seconds")
                else:
                    print(f"❌ Validation mismatch: expected {should_be_valid}, got {is_valid}")
                    return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    return True

def test_error_handling():
    """Test error handling"""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    # Test empty text
    print("\nTest 1: Empty text")
    try:
        response = requests.post(
            f"{BASE_URL}/audio/synthesize",
            json={"text": "", "model": "default"}
        )
        if response.status_code == 400:
            print("✅ Correctly rejected empty text")
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test invalid model
    print("\nTest 2: Invalid model")
    try:
        response = requests.post(
            f"{BASE_URL}/audio/synthesize",
            json={"text": "test", "model": "nonexistent"}
        )
        if response.status_code == 404:
            print("✅ Correctly rejected invalid model")
        else:
            print(f"⚠️  Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test invalid audio data
    print("\nTest 3: Invalid audio data")
    try:
        files = {'audio': ('invalid.wav', b'invalid data', 'audio/wav')}
        response = requests.post(
            f"{BASE_URL}/audio/transcribe",
            files=files,
            data={'model': 'default'}
        )
        if response.status_code == 400:
            print("✅ Correctly rejected invalid audio")
        else:
            print(f"⚠️  Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return True

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ONNX AUDIO MODELS - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    results = {
        "Audio Health Check": test_audio_health(),
        "TTS Synthesis": test_tts_synthesis(),
        "TTS Parameters": test_tts_parameters(),
        "Audio Validation": test_audio_validation(),
        "STT Transcription": test_stt_transcription(),
        "Error Handling": test_error_handling(),
    }
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    return passed == total

if __name__ == "__main__":
    import sys
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("❌ Server is not responding correctly")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print("❌ Server is not running at", BASE_URL)
        print("   Please start the server with: ./target/release/torch-inference-server")
        sys.exit(1)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)

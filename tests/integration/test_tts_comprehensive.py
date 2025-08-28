# ...existing code from test_tts_comprehensive.py will be moved here...
def test_tts_service_comprehensive():
    """Test the TTS service with multiple scenarios."""
    
    test_cases = [
        {
            "text": "Hello, this is a short test.",
            "model_name": "speecht5_tts",
            "name": "short_test"
        },
        {
            "text": "This is a longer sentence to test the text-to-speech system with more complex audio generation and better quality output.",
            "model_name": "speecht5_tts", 
            "name": "long_test"
        },
        {
            "text": "Testing different models and parameters.",
            "model_name": "default",
            "name": "default_model_test"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'='*50}")
        
        url = 'http://localhost:8000/tts/synthesize'
        data = {
            'text': test_case['text'],
            'model_name': test_case['model_name'],
            'output_format': 'wav',
            'speed': 1.0,
            'volume': 1.0
        }

        try:
            response = requests.post(url, json=data)
            print(f'Status: {response.status_code}')
            
            if response.status_code == 200:
                result = response.json()
                print(f'Success: {result["success"]}')
                
                if result['success']:
                    print(f'Sample rate: {result["sample_rate"]}')
                    print(f'Duration: {result["duration"]:.3f}s')
                    print(f'Audio format: {result["audio_format"]}')
                    
                    # Decode audio data
                    audio_data = base64.b64decode(result['audio_data'])
                    print(f'Audio data length: {len(audio_data)} bytes')
                    
                    # Check if it starts with RIFF header
                    if audio_data[:4] == b'RIFF':
                        print('✓ Audio data has proper RIFF header')
                        
                        # Save and validate WAV file
                        output_file = f'test_{test_case["name"]}.wav'
                        with open(output_file, 'wb') as f:
                            f.write(audio_data)
                        
                        # Verify WAV file integrity
                        try:
                            with wave.open(output_file, 'rb') as wav_file:
                                channels = wav_file.getnchannels()
                                sample_width = wav_file.getsampwidth()
                                frame_rate = wav_file.getframerate()
                                frames = wav_file.getnframes()
                                duration = frames / frame_rate
                                
                                print(f'✓ WAV file validation passed:')
                                print(f'  - Channels: {channels}')
                                print(f'  - Sample width: {sample_width} bytes')
                                print(f'  - Frame rate: {frame_rate} Hz')
                                print(f'  - Frames: {frames}')
                                print(f'  - Duration: {duration:.3f}s')
                                print(f'  - Saved as: {output_file}')
                                
                        except Exception as e:
                            print(f'✗ WAV file validation failed: {e}')
                    else:
                        print(f'✗ Audio data does not have RIFF header: {audio_data[:12]}')
                else:
                    print(f'✗ TTS failed: {result["error"]}')
            else:
                print(f'✗ HTTP Error {response.status_code}: {response.text}')
                
        except Exception as e:
            print(f'✗ Request failed: {e}')

def verify_wav_header_format():
    """Verify that the WAV header format is correct."""
    print(f"\n{'='*50}")
    print("WAV Header Format Verification")
    print(f"{'='*50}")
    
    # Create a simple test audio array
    sample_rate = 16000
    duration = 1.0  # 1 second
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Test the _create_wav_bytes function
    try:
        # We need to import the function - let's create a test
        test_script = '''
import sys
sys.path.insert(0, ".")
from main import _create_wav_bytes
import numpy as np

sample_rate = 16000
duration = 1.0
frequency = 440
t = np.linspace(0, duration, int(sample_rate * duration), False)
audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)

wav_bytes = _create_wav_bytes(audio_data, sample_rate)
print(f"WAV bytes length: {len(wav_bytes)}")
print(f"Header starts with RIFF: {wav_bytes[:4] == b'RIFF'}")

# Save and test
with open("header_test.wav", "wb") as f:
    f.write(wav_bytes)

import wave
with wave.open("header_test.wav", "rb") as wav_file:
    print(f"Valid WAV file - Duration: {wav_file.getnframes() / wav_file.getframerate():.2f}s")
print("✓ WAV header function works correctly!")
'''
        
        with open('test_wav_header.py', 'w') as f:
            f.write(test_script)
        
        import subprocess
        result = subprocess.run(['python', 'test_wav_header.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error testing WAV header function: {result.stderr}")
            
    except Exception as e:
        print(f"Error in WAV header verification: {e}")

def main():
    """Main test function."""
    print("Comprehensive TTS WAV Corruption Fix Test")
    print("This will test the fixed TTS service with proper WAV headers")
    
    # First verify the WAV header function
    verify_wav_header_format()
    
    # Then test the TTS service comprehensively
    test_tts_service_comprehensive()
    
    print(f"\n{'='*50}")
    print("Test Summary:")
    print("- WAV files should now have proper RIFF headers")
    print("- Audio should be playable in standard media players")
    print("- No more 'corrupted WAV' errors")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()

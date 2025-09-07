#!/usr/bin/env python3
"""
Test the TTS endpoint with correct configuration.
"""

import asyncio
import aiohttp
import time
import json

async def test_tts_endpoint():
    """Test the /synthesize endpoint correctly."""
    base_url = "http://localhost:8000"
    endpoint = "/synthesize"
    
    # Correct payload format based on TTSRequest model
    payload = {
        "model_name": "speecht5_tts",  # Use an available TTS model
        "inputs": "Hello, this is a test of the text-to-speech system.",
        "voice": "default",
        "speed": 1.0,
        "pitch": 1.0,
        "volume": 1.0,
        "language": "en",
        "output_format": "wav"
    }
    
    print(f"Testing TTS endpoint: {base_url}{endpoint}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("-" * 50)
    
    async with aiohttp.ClientSession() as session:
        start_time = time.perf_counter()
        
        try:
            async with session.post(f"{base_url}{endpoint}", json=payload) as response:
                end_time = time.perf_counter()
                
                print(f"Status: {response.status}")
                print(f"Response time: {(end_time - start_time)*1000:.1f}ms")
                
                if response.status == 200:
                    data = await response.json()
                    print("✅ Success!")
                    print(f"Audio duration: {data.get('duration', 'unknown')}s")
                    print(f"Sample rate: {data.get('sample_rate', 'unknown')} Hz")
                    print(f"Processing time: {data.get('processing_time', 'unknown')}s")
                    print(f"Audio format: {data.get('audio_format', 'unknown')}")
                    print(f"Audio data length: {len(data.get('audio_data', '')) if data.get('audio_data') else 0} chars")
                    
                    # Show model info
                    model_info = data.get('model_info', {})
                    if model_info:
                        print(f"Model info: {json.dumps(model_info, indent=2)}")
                    
                else:
                    text = await response.text()
                    print(f"❌ Error: {response.status}")
                    print(f"Response: {text}")
                    
        except Exception as e:
            print(f"❌ Exception: {e}")

async def test_multiple_models():
    """Test multiple TTS models."""
    models_to_test = ["speecht5_tts", "bark_tts", "default"]
    
    print("Testing multiple TTS models:")
    print("=" * 60)
    
    for model_name in models_to_test:
        print(f"\nTesting model: {model_name}")
        print("-" * 30)
        
        payload = {
            "model_name": model_name,
            "inputs": f"Testing {model_name} text-to-speech model.",
            "speed": 1.0,
            "language": "en"
        }
        
        async with aiohttp.ClientSession() as session:
            start_time = time.perf_counter()
            
            try:
                async with session.post("http://localhost:8000/synthesize", json=payload) as response:
                    end_time = time.perf_counter()
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ {model_name}: {(end_time - start_time)*1000:.1f}ms, "
                              f"{data.get('duration', 'unknown')}s audio")
                    else:
                        error_data = await response.json()
                        print(f"❌ {model_name}: {response.status} - {error_data.get('error', 'Unknown error')}")
                        
            except Exception as e:
                print(f"❌ {model_name}: Exception - {e}")

async def benchmark_simple():
    """Simple benchmark of the TTS endpoint."""
    print("\nRunning simple benchmark:")
    print("=" * 60)
    
    test_texts = [
        "Hello world",
        "This is a test of the text-to-speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming how we interact with technology.",
        "Speech synthesis enables computers to convert text into natural-sounding speech."
    ]
    
    model_name = "speecht5_tts"
    results = []
    
    for i, text in enumerate(test_texts):
        payload = {
            "model_name": model_name,
            "inputs": text,
            "speed": 1.0,
            "language": "en"
        }
        
        async with aiohttp.ClientSession() as session:
            start_time = time.perf_counter()
            
            try:
                async with session.post("http://localhost:8000/synthesize", json=payload) as response:
                    end_time = time.perf_counter()
                    
                    if response.status == 200:
                        data = await response.json()
                        duration = data.get('duration', 0)
                        processing_time = data.get('processing_time', 0)
                        response_time = end_time - start_time
                        
                        results.append({
                            'text_length': len(text),
                            'audio_duration': duration,
                            'processing_time': processing_time,
                            'response_time': response_time,
                            'rtf': processing_time / duration if duration > 0 else 0
                        })
                        
                        print(f"Test {i+1}: {len(text)} chars → {duration:.2f}s audio, "
                              f"RTF: {processing_time/duration:.3f}, "
                              f"Response: {response_time*1000:.1f}ms")
                    else:
                        print(f"Test {i+1}: Error {response.status}")
                        
            except Exception as e:
                print(f"Test {i+1}: Exception - {e}")
    
    if results:
        print(f"\nBenchmark Summary:")
        print(f"Tests completed: {len(results)}")
        avg_rtf = sum(r['rtf'] for r in results) / len(results)
        avg_response = sum(r['response_time'] for r in results) / len(results)
        avg_processing = sum(r['processing_time'] for r in results) / len(results)
        
        print(f"Average RTF: {avg_rtf:.3f}")
        print(f"Average response time: {avg_response*1000:.1f}ms")
        print(f"Average processing time: {avg_processing:.3f}s")

async def main():
    """Run all tests."""
    await test_tts_endpoint()
    await test_multiple_models()
    await benchmark_simple()

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
API Endpoint Examples for PyTorch Inference Framework

This script demonstrates how to use the unified API endpoints:
- /predict for all torch models and deep learning inference
- /synthesize for TTS models

Both endpoints support:
- model_name parameter to specify the model
- inputs parameter for data (batch or single)
- token parameter for authentication (optional)

Usage:
    python examples/api_endpoint_examples.py --mode predict --model example
    python examples/api_endpoint_examples.py --mode synthesize --model speecht5_tts
    python examples/api_endpoint_examples.py --mode demo
"""

import argparse
import asyncio
import aiohttp
import json
import base64
import numpy as np
from typing import List, Any, Optional


class APIClient:
    """Client for demonstrating the unified API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API client."""
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
    
    async def predict(self, model_name: str, inputs: Any, token: str = None, 
                     priority: int = 0, timeout: float = 30.0, 
                     enable_batching: bool = True) -> Optional[dict]:
        """
        Make a prediction using the unified /predict endpoint.
        
        Args:
            model_name: Name of the model to use
            inputs: Input data (single item or list for batch)
            token: Optional authentication token
            priority: Request priority (higher = processed first)
            timeout: Request timeout in seconds
            enable_batching: Enable inflight batching optimization
        """
        request_data = {
            "model_name": model_name,
            "inputs": inputs,
            "priority": priority,
            "timeout": timeout,
            "enable_batching": enable_batching
        }
        
        if token:
            request_data["token"] = token
        
        is_batch = isinstance(inputs, list) and len(inputs) > 1
        input_count = len(inputs) if isinstance(inputs, list) else 1
        
        print(f"\n🔮 Making prediction:")
        print(f"  Model: {model_name}")
        print(f"  Type: {'batch' if is_batch else 'single'}")
        print(f"  Count: {input_count}")
        print(f"  Auth: {'token' if token else 'none'}")
        
        try:
            async with self.session.post(
                f"{self.base_url}/predict",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=timeout + 5)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if result.get('success'):
                        print(f"  ✅ Prediction successful!")
                        print(f"  Processing Time: {result.get('processing_time', 'N/A'):.3f}s")
                        print(f"  Device: {result.get('model_info', {}).get('device', 'N/A')}")
                        print(f"  Path: {result.get('model_info', {}).get('processing_path', 'N/A')}")
                        
                        # Show batch info if available
                        batch_info = result.get('batch_info', {})
                        if batch_info:
                            print(f"  Batching: {batch_info.get('processed_as_batch', False)}")
                            print(f"  Concurrent: {batch_info.get('concurrent_optimization', False)}")
                        
                        return result
                    else:
                        print(f"  ❌ Prediction failed: {result.get('error', 'Unknown error')}")
                        return None
                else:
                    error_text = await response.text()
                    print(f"  ❌ HTTP Error {response.status}: {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            print(f"  ❌ Request timed out after {timeout}s")
            return None
        except Exception as e:
            print(f"  ❌ Prediction error: {e}")
            return None
    
    async def synthesize(self, model_name: str, text: str, token: str = None,
                        voice: str = None, speed: float = 1.0, pitch: float = 1.0,
                        volume: float = 1.0, language: str = "en", 
                        emotion: str = None, output_format: str = "wav") -> Optional[dict]:
        """
        Synthesize speech using the unified /synthesize endpoint.
        
        Args:
            model_name: Name of the TTS model to use
            text: Text to synthesize
            token: Optional authentication token
            voice: Voice to use for synthesis
            speed: Speech speed (0.5-2.0)
            pitch: Pitch adjustment (0.5-2.0)
            volume: Volume level (0.1-2.0)
            language: Language code
            emotion: Emotion for synthesis
            output_format: Output audio format
        """
        request_data = {
            "model_name": model_name,
            "inputs": text,
            "voice": voice,
            "speed": speed,
            "pitch": pitch,
            "volume": volume,
            "language": language,
            "emotion": emotion,
            "output_format": output_format
        }
        
        if token:
            request_data["token"] = token
        
        print(f"\n🎙️ Synthesizing speech:")
        print(f"  Model: {model_name}")
        print(f"  Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"  Length: {len(text)} chars")
        print(f"  Auth: {'token' if token else 'none'}")
        
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
                            filename = f"tts_output_{model_name}.wav"
                            self.save_audio_from_base64(audio_data, filename)
                        
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
    
    def save_audio_from_base64(self, audio_base64: str, output_path: str):
        """Save base64 encoded audio data to file."""
        try:
            audio_bytes = base64.b64decode(audio_base64)
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)
            print(f"  💾 Audio saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"  ❌ Failed to save audio: {e}")
            return None
    
    async def run_prediction_examples(self):
        """Run prediction endpoint examples."""
        print("\n" + "─" * 50)
        print("🔮 PREDICTION ENDPOINT EXAMPLES")
        print("─" * 50)
        
        # Single prediction
        print("\n1. Single Input Prediction:")
        await self.predict(
            model_name="example",
            inputs=[1, 2, 3, 4, 5]
        )
        
        # Batch prediction
        print("\n2. Batch Prediction:")
        batch_inputs = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ]
        await self.predict(
            model_name="example",
            inputs=batch_inputs
        )
        
        # Text prediction (if text model available)
        print("\n3. Text Input Prediction:")
        await self.predict(
            model_name="example",
            inputs="Hello, world! This is a test input."
        )
        
        # With authentication token
        print("\n4. Prediction with Authentication Token:")
        await self.predict(
            model_name="example",
            inputs=[1, 2, 3],
            token="dummy_token_for_demo"
        )
    
    async def run_synthesis_examples(self):
        """Run synthesis endpoint examples."""
        print("\n" + "─" * 50)
        print("🎙️ SYNTHESIS ENDPOINT EXAMPLES")
        print("─" * 50)
        
        demo_texts = [
            "Hello, world! This is a text-to-speech demonstration.",
            "The PyTorch inference framework now has unified API endpoints.",
            "You can use the synthesize endpoint for all TTS models."
        ]
        
        tts_models = ["default", "speecht5_tts", "bark"]
        
        for i, (text, model) in enumerate(zip(demo_texts, tts_models)):
            print(f"\n{i+1}. TTS with {model}:")
            await self.synthesize(
                model_name=model,
                text=text,
                speed=1.0,
                language="en"
            )
        
        # With authentication token
        print(f"\n4. TTS with Authentication Token:")
        await self.synthesize(
            model_name="default",
            text="This request includes an authentication token.",
            token="dummy_token_for_demo"
        )
    
    async def run_demo(self):
        """Run a complete demonstration of both endpoints."""
        print("🚀 PyTorch Inference Framework - Unified API Endpoints Demo")
        print("=" * 70)
        
        print("\nThis demo showcases the new unified API endpoints:")
        print("  • /predict - for all torch models and deep learning inference")
        print("  • /synthesize - for TTS models")
        print("\nBoth endpoints support:")
        print("  • model_name parameter to specify the model")
        print("  • inputs parameter for data (batch or single)")
        print("  • token parameter for authentication (optional)")
        
        # Run prediction examples
        await self.run_prediction_examples()
        
        # Run synthesis examples
        await self.run_synthesis_examples()
        
        print("\n🎉 API endpoint demo completed!")
        print("\nKey improvements:")
        print("  ✅ Unified endpoint structure")
        print("  ✅ Consistent parameter naming")
        print("  ✅ Token-based authentication support")
        print("  ✅ Batch and single input support")
        print("  ✅ Model selection via parameter")


async def main():
    """Main function for the API endpoint examples script."""
    parser = argparse.ArgumentParser(description="Unified API Endpoint Examples")
    parser.add_argument("--mode", choices=["predict", "synthesize", "demo"], 
                       default="demo", help="Example mode to run")
    parser.add_argument("--model", default="example", help="Model name to use")
    parser.add_argument("--text", help="Text for synthesis (synthesize mode)")
    parser.add_argument("--inputs", help="JSON inputs for prediction (predict mode)")
    parser.add_argument("--token", help="Authentication token")
    parser.add_argument("--base-url", default="http://localhost:8000",
                       help="Base URL of the inference server")
    
    args = parser.parse_args()
    
    async with APIClient(args.base_url) as client:
        if args.mode == "predict":
            inputs = [1, 2, 3, 4, 5]  # Default inputs
            if args.inputs:
                try:
                    inputs = json.loads(args.inputs)
                except json.JSONDecodeError:
                    inputs = args.inputs  # Use as string if not valid JSON
            
            await client.predict(
                model_name=args.model,
                inputs=inputs,
                token=args.token
            )
            
        elif args.mode == "synthesize":
            text = args.text or "Hello, world! This is a text-to-speech test."
            await client.synthesize(
                model_name=args.model,
                text=text,
                token=args.token
            )
            
        elif args.mode == "demo":
            await client.run_demo()


if __name__ == "__main__":
    asyncio.run(main())

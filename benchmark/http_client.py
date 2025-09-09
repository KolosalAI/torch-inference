"""
HTTP TTS Client for Benchmarking.

Provides HTTP client functionality for benchmarking TTS servers over REST API.
Includes support for streaming and non-streaming TTS endpoints.
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, Any, Optional, AsyncGenerator, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TTSServerConfig:
    """Configuration for TTS server connection."""
    base_url: str
    synthesize_endpoint: str = "/synthesize"
    stream_endpoint: str = "/synthesize/stream"
    timeout: float = 30.0
    headers: Optional[Dict[str, str]] = None
    auth_token: Optional[str] = None


class HTTPTTSClient:
    """
    HTTP client for TTS server benchmarking.
    
    Supports both streaming and non-streaming TTS requests
    with proper timing and error handling for benchmarking.
    """
    
    def __init__(self, config: TTSServerConfig):
        """
        Initialize HTTP TTS client.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        headers = self.config.headers or {}
        if self.config.auth_token:
            headers['Authorization'] = f'Bearer {self.config.auth_token}'
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def synthesize_text(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        sample_rate: int = 22050,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synthesize text using non-streaming endpoint.
        
        Args:
            text: Text to synthesize
            voice: Voice model to use
            speed: Speech speed multiplier
            sample_rate: Audio sample rate
            **kwargs: Additional TTS parameters
            
        Returns:
            Dictionary with audio_duration, sample_rate, etc.
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.config.base_url}{self.config.synthesize_endpoint}"
        
        # Use torch-inference server format
        payload = {
            "model_name": voice or "speecht5_tts",  # Default to speecht5_tts if no voice specified
            "inputs": text,  # torch-inference uses "inputs" not "text"
            "speed": speed,
            "language": "en",
            "output_format": "wav",
            **kwargs
        }
        
        # Add voice parameter if specified and different from model_name
        if voice and voice not in ["speecht5_tts", "bark_tts", "default"]:
            payload["voice"] = voice
        
        start_time = time.perf_counter()
        
        try:
            async with self.session.post(url, json=payload) as response:
                response.raise_for_status()
                
                # torch-inference server always returns JSON
                data = await response.json()
                
                if data.get('success', False):
                    # Parse torch-inference response format
                    audio_duration = data.get('duration', 0.0)
                    actual_sample_rate = data.get('sample_rate', sample_rate)
                    processing_time = data.get('processing_time', 0.0)
                    audio_data_b64 = data.get('audio_data', '')
                    
                    return {
                        'audio_duration': audio_duration,
                        'sample_rate': actual_sample_rate,
                        'text_tokens': len(text.split()),
                        'response_size_bytes': len(audio_data_b64) if audio_data_b64 else 0,
                        'processing_time': processing_time
                    }
                else:
                    # Handle error response
                    error_msg = data.get('error', 'Unknown error')
                    raise RuntimeError(f"TTS synthesis failed: {error_msg}")
                    
        except asyncio.TimeoutError:
            raise RuntimeError(f"TTS request timed out after {self.config.timeout}s")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"HTTP error: {e}")
        except Exception as e:
            raise RuntimeError(f"TTS synthesis failed: {e}")
    
    async def synthesize_text_streaming(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        sample_rate: int = 22050,
        first_chunk_callback: Optional[Callable[[float], None]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synthesize text using streaming endpoint.
        
        Args:
            text: Text to synthesize
            voice: Voice model to use
            speed: Speech speed multiplier
            sample_rate: Audio sample rate
            first_chunk_callback: Called with timestamp when first chunk arrives
            **kwargs: Additional TTS parameters
            
        Returns:
            Dictionary with audio_duration, sample_rate, etc.
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.config.base_url}{self.config.stream_endpoint}"
        
        # Use torch-inference server format for streaming
        payload = {
            "model_name": voice or "speecht5_tts",  # Default to speecht5_tts if no voice specified
            "inputs": text,  # torch-inference uses "inputs" not "text"
            "speed": speed,
            "language": "en",
            "output_format": "wav",
            **kwargs
        }
        
        # Add voice parameter if specified and different from model_name
        if voice and voice not in ["speecht5_tts", "bark_tts", "default"]:
            payload["voice"] = voice
        
        chunks = []
        first_chunk_time = None
        total_bytes = 0
        
        try:
            async with self.session.post(url, json=payload) as response:
                response.raise_for_status()
                
                async for chunk in response.content.iter_chunked(8192):
                    if chunk:
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter()
                            if first_chunk_callback:
                                first_chunk_callback(first_chunk_time)
                        
                        chunks.append(chunk)
                        total_bytes += len(chunk)
                
                # Estimate total audio duration
                # This is approximate - adjust based on your audio format
                bytes_per_sample = 2  # 16-bit PCM
                total_samples = total_bytes // bytes_per_sample
                audio_duration = total_samples / sample_rate
                
                return {
                    'audio_duration': audio_duration,
                    'sample_rate': sample_rate,
                    'text_tokens': len(text.split()),
                    'response_size_bytes': total_bytes,
                    'chunk_count': len(chunks),
                    'first_chunk_time': first_chunk_time
                }
                
        except asyncio.TimeoutError:
            raise RuntimeError(f"Streaming TTS request timed out after {self.config.timeout}s")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"HTTP error: {e}")
        except Exception as e:
            raise RuntimeError(f"Streaming TTS synthesis failed: {e}")


def create_http_tts_function(
    server_config: TTSServerConfig,
    streaming: bool = False,
    voice: Optional[str] = None,
    **default_kwargs
) -> Callable:
    """
    Create a TTS function for benchmarking HTTP TTS servers.
    
    Args:
        server_config: TTS server configuration
        streaming: Whether to use streaming endpoint
        voice: Default voice to use
        **default_kwargs: Default TTS parameters
        
    Returns:
        Async TTS function suitable for benchmarking
    """
    async def http_tts_function(text: str, **kwargs) -> Dict[str, Any]:
        # Merge default kwargs with call-specific kwargs
        tts_params = {**default_kwargs, **kwargs}
        if voice:
            tts_params['voice'] = voice
        
        async with HTTPTTSClient(server_config) as client:
            if streaming:
                return await client.synthesize_text_streaming(text, **tts_params)
            else:
                return await client.synthesize_text(text, **tts_params)
    
    return http_tts_function


async def test_server_connectivity(server_config: TTSServerConfig) -> bool:
    """
    Test connectivity to TTS server.
    
    Args:
        server_config: Server configuration
        
    Returns:
        True if server is reachable and responds correctly
    """
    try:
        async with HTTPTTSClient(server_config) as client:
            test_result = await client.synthesize_text(
                "Hello, this is a connectivity test.",
                voice="speecht5_tts",  # Use a valid model name
                sample_rate=22050
            )
            
            logger.info(f"Server connectivity test passed. "
                       f"Generated {test_result['audio_duration']:.2f}s of audio.")
            return True
            
    except Exception as e:
        logger.error(f"Server connectivity test failed: {e}")
        return False


def create_torch_inference_tts_function(
    base_url: str = "http://localhost:8000",
    voice: str = "speecht5_tts",
    streaming: bool = False,
    auth_token: Optional[str] = None,
    **default_params
) -> Callable:
    """
    Create TTS function specifically for torch-inference server.
    
    Args:
        base_url: Base URL of torch-inference server
        voice: Voice model to use
        streaming: Whether to use streaming
        auth_token: Authentication token if required
        **default_params: Default TTS parameters
        
    Returns:
        Async TTS function for benchmarking
    """
    config = TTSServerConfig(
        base_url=base_url,
        synthesize_endpoint="/synthesize",
        stream_endpoint="/synthesize/stream",
        auth_token=auth_token
    )
    
    return create_http_tts_function(
        config,
        streaming=streaming,
        voice=voice,
        **default_params
    )


# Example usage and demo
async def demo_http_benchmark():
    """Demonstrate HTTP TTS benchmarking."""
    from .harness import TTSBenchmarkHarness, BenchmarkConfig
    
    # Configure server (adjust URL and parameters as needed)
    server_config = TTSServerConfig(
        base_url="http://localhost:8000",
        synthesize_endpoint="/v1/audio/speech"
    )
    
    # Test connectivity first
    logger.info("Testing server connectivity...")
    if not await test_server_connectivity(server_config):
        logger.error("Cannot connect to TTS server. Please check the server is running.")
        return
    
    # Create TTS function
    tts_function = create_http_tts_function(
        server_config,
        streaming=False,
        voice="speecht5_tts",
        sample_rate=22050
    )
    
    # Configure benchmark
    config = BenchmarkConfig(
        concurrency_levels=[1, 2, 4, 8, 16, 32, 64],
        iterations_per_level=5,
        text_variations=10,
        output_dir="http_tts_benchmark_results"
    )
    
    # Run benchmark
    harness = TTSBenchmarkHarness(config)
    results = await harness.run_benchmark_async(
        tts_function,
        benchmark_name="http_tts_benchmark"
    )
    
    # Print results
    from .reporter import TTSBenchmarkReporter
    reporter = TTSBenchmarkReporter()
    reporter.print_summary_table(results, "HTTP TTS Benchmark")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    asyncio.run(demo_http_benchmark())

#!/usr/bin/env python3
"""
End-to-End TTS Benchmark Runner for torch-inference.

This script provides a complete benchmarking solution that can:
1. Auto-detect and test TTS server endpoints
2. Run comprehensive performance benchmarks
3. Generate detailed reports and visualizations
4. Compare multiple configurations
5. Save and load benchmark results

Usage:
    python benchmark.py --help
    python benchmark.py demo
    python benchmark.py server --url http://localhost:8000
    python benchmark.py compare --configs config1.json config2.json
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

# Add benchmark module to path
sys.path.insert(0, str(Path(__file__).parent))


def validate_gpu_environment() -> bool:
    """Validate that GPU environment is properly set up and warn about CPU fallback."""
    print("=" * 60)
    print("GPU ENVIRONMENT VALIDATION")
    print("=" * 60)
    
    try:
        import torch
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("❌ CUDA is not available!")
            print("   Please ensure you have:")
            print("   1. NVIDIA GPU installed")
            print("   2. CUDA drivers installed") 
            print("   3. PyTorch with CUDA support installed")
            print("   ⚠️  All benchmarks will use CPU fallback with SIGNIFICANTLY degraded performance")
            print("   ⚠️  For accurate GPU performance benchmarks, please install CUDA support")
            return False
        
        # Check GPU count
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print("❌ No CUDA devices found!")
            print("   ⚠️  All benchmarks will use CPU fallback with SIGNIFICANTLY degraded performance")
            return False
        
        print(f"✅ CUDA is available with {gpu_count} GPU(s)")
        
        # Check current GPU
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"✅ Current GPU: {gpu_name} (Device {current_device})")
        
        # Check GPU memory
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(current_device) / (1024**3)
        
        print(f"✅ GPU Memory: {allocated_memory:.2f}GB / {total_memory:.2f}GB total")
        
        if total_memory < 4.0:
            print(f"⚠️  Warning: GPU has only {total_memory:.1f}GB memory. Some models may not fit.")
        
        # Check compute capability
        props = torch.cuda.get_device_properties(current_device)
        compute_cap = f"{props.major}.{props.minor}"
        print(f"✅ Compute Capability: {compute_cap}")
        
        if props.major < 6:
            print(f"⚠️  Warning: Compute capability {compute_cap} is quite old. Performance may be limited.")
        
        print("=" * 60)
        print("🚀 GPU environment validated - ready for high-performance benchmarks!")
        print("=" * 60)
        return True
        
    except ImportError:
        print("❌ PyTorch not available!")
        print("   ⚠️  All benchmarks will use CPU fallback with SIGNIFICANTLY degraded performance")
        return False
    except Exception as e:
        print(f"❌ GPU validation error: {e}")
        print("   ⚠️  All benchmarks will use CPU fallback with SIGNIFICANTLY degraded performance")
        return False


def setup_gpu_optimizations():
    """Set up GPU optimizations for maximum performance."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠️  GPU optimizations disabled - CUDA not available")
            return
        
        print("🚀 Setting up GPU optimizations...")
        
        # Set float32 matmul precision for better performance
        torch.set_float32_matmul_precision('high')
        
        # Enable TF32 on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmark for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        print("✅ GPU optimizations configured")
        
    except Exception as e:
        print(f"⚠️  GPU optimization setup failed: {e}")
        print("   Continuing with default settings")

try:
    # TTS Benchmarking
    from benchmark.harness import TTSBenchmarkHarness, BenchmarkConfig, create_demo_tts_function
    from benchmark.http_client import (
        create_torch_inference_tts_function, 
        test_server_connectivity, 
        TTSServerConfig,
        HTTPTTSClient
    )
    from benchmark.reporter import TTSBenchmarkReporter
    from benchmark.metrics import validate_metrics_consistency, TTSMetrics
    from benchmark.tts_benchmark import TTSBenchmarkResult
    
    # Image Benchmarking
    from benchmark.harness import ImageBenchmarkHarness, generate_image_prompts, demo_image_model
    from benchmark.gpu_demo_model import gpu_demo_image_model
    from benchmark.image_benchmark import ImageBenchmarkResult
    from benchmark.image_reporter import ImageBenchmarkReporter
    from benchmark.image_metrics import ImageMetrics, validate_image_metrics_consistency
    
    # ResNet Image Classification Benchmarking
    from benchmark.resnet_image_benchmark import (
        ResNetImageBenchmarker, 
        create_resnet_classification_function,
        create_demo_resnet_function
    )
except ImportError as e:
    print(f"❌ Failed to import benchmark modules: {e}")
    print("Make sure you're running from the torch-inference directory")
    sys.exit(1)


@dataclass
class BenchmarkSession:
    """Complete benchmark session configuration and results."""
    session_id: str
    timestamp: str
    config: BenchmarkConfig
    server_config: Optional[Dict[str, Any]] = None
    results: Dict[str, Dict[int, Union[TTSBenchmarkResult, ImageBenchmarkResult]]] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        data = asdict(self)
        # Convert BenchmarkConfig to dict
        data['config'] = asdict(self.config)
        # Convert results to serializable format
        if self.results:
            serialized_results = {}
            for name, concurrency_results in self.results.items():
                serialized_results[name] = {}
                for concurrency, result in concurrency_results.items():
                    serialized_results[name][str(concurrency)] = {
                        'metrics': asdict(result.metrics),
                        'request_metrics': [asdict(req) for req in result.request_metrics],
                        'config': result.config
                    }
            data['results'] = serialized_results
        return data


class EndToEndBenchmarkRunner:
    """
    Complete end-to-end TTS and Image benchmark runner.
    
    Provides automated testing, configuration management, and comprehensive reporting
    for both TTS and Image generation models.
    """
    
    def __init__(self, output_dir: str = "benchmark_sessions"):
        """Initialize the benchmark runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logging()
        self.tts_reporter = TTSBenchmarkReporter()
        self.image_reporter = ImageBenchmarkReporter()
        self.session: Optional[BenchmarkSession] = None
        
    def _setup_logging(self) -> logging.Logger:
        """Set up comprehensive logging."""
        logger = logging.getLogger("benchmark_runner")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler
        log_file = self.output_dir / "benchmark.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def start_session(self, session_name: Optional[str] = None) -> str:
        """Start a new benchmark session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = session_name or f"benchmark_{timestamp}"
        
        self.session = BenchmarkSession(
            session_id=session_id,
            timestamp=timestamp,
            config=BenchmarkConfig(),
            metadata={
                "start_time": time.time(),
                "python_version": sys.version,
                "platform": sys.platform
            }
        )
        
        self.logger.info(f"Started benchmark session: {session_id}")
        return session_id
    
    def configure_benchmark(
        self,
        model_type: str = "tts",
        concurrency_levels: Optional[List[int]] = None,
        iterations: int = 20,
        timeout: float = 30.0,
        sample_rate: int = 22050,
        text_variations: int = 25,
        output_plots: bool = True,
        # Image-specific parameters
        image_width: int = 512,
        image_height: int = 512,
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> None:
        """Configure benchmark parameters for TTS or Image models."""
        if not self.session:
            raise RuntimeError("No active session. Call start_session() first.")
        
        self.session.config = BenchmarkConfig(
            model_type=model_type,
            concurrency_levels=concurrency_levels or [1, 2, 4, 8, 16, 32, 64],
            iterations_per_level=iterations,
            timeout_seconds=timeout,
            sample_rate=sample_rate,
            text_variations=text_variations,
            generate_plots=output_plots,
            output_dir=str(self.output_dir / self.session.session_id),
            # Image parameters
            image_width=image_width,
            image_height=image_height,
            num_images=num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        model_info = f"{model_type.upper()} model"
        if model_type == "image":
            model_info += f" ({image_width}x{image_height}, {num_inference_steps} steps)"
        
        self.logger.info(f"Configured benchmark for {model_info}: "
                        f"concurrency={self.session.config.concurrency_levels}, "
                        f"iterations={iterations}, timeout={timeout}s")
    
    def run_demo_benchmark(self) -> Dict[int, TTSBenchmarkResult]:
        """Run demo benchmark with synthetic TTS."""
        self.logger.info("Running demo benchmark with synthetic TTS")
        
        if not self.session:
            self.start_session("demo_benchmark")
            self.configure_benchmark()
        
        # Create demo TTS with realistic parameters
        demo_tts = create_demo_tts_function(
            min_audio_duration=0.5,
            max_audio_duration=4.0,
            processing_delay=0.02,  # 20ms processing
            failure_rate=0.01  # 1% failure rate
        )
        
        # Run benchmark
        harness = TTSBenchmarkHarness(self.session.config)
        results = harness.run_benchmark(
            demo_tts,
            benchmark_name="demo_synthetic_tts"
        )
        
        if not self.session.results:
            self.session.results = {}
        self.session.results["demo"] = results
        
        self._print_benchmark_summary("Demo Benchmark", results)
        return results
    
    def run_demo_image_benchmark(self) -> Dict[int, ImageBenchmarkResult]:
        """Run demo benchmark with synthetic image generation."""
        self.logger.info("Running demo image benchmark with synthetic model")
        
        if not self.session:
            self.start_session("demo_image_benchmark")
            self.configure_benchmark(model_type="image")
        
        # Use the GPU-aware demo image model
        demo_image_func = gpu_demo_image_model
        
        # Run benchmark
        harness = ImageBenchmarkHarness(self.session.config)
        results = harness.run_benchmark(
            demo_image_func,
            benchmark_name="demo_synthetic_image"
        )
        
        if not self.session.results:
            self.session.results = {}
        self.session.results["demo_image"] = results
        
        self._print_image_benchmark_summary("Demo Image Benchmark", results)
        return results
    
    def run_demo_resnet_benchmark(self) -> Dict[int, ImageBenchmarkResult]:
        """Run demo benchmark with synthetic ResNet classification."""
        self.logger.info("Running demo ResNet benchmark with synthetic model")
        
        if not self.session:
            self.start_session("demo_resnet_benchmark")
            self.configure_benchmark(model_type="image_classification")
        
        # Create demo ResNet classification function
        demo_resnet_func = create_demo_resnet_function()
        
        # Create ResNet benchmarker
        benchmarker = ResNetImageBenchmarker(
            default_width=224,
            default_height=224,
            warmup_requests=3
        )
        
        # Run benchmark
        results = benchmarker.benchmark_resnet_model(
            classification_function=demo_resnet_func,
            concurrency_levels=self.session.config.concurrency_levels,
            iterations_per_level=self.session.config.iterations_per_level
        )
        
        if not self.session.results:
            self.session.results = {}
        self.session.results["demo_resnet"] = results
        
        self._print_image_benchmark_summary("Demo ResNet Benchmark", results)
        return results
    
    async def run_resnet_server_benchmark(
        self,
        model_name: str = "resnet18",
        server_url: str = "http://localhost:8000",
        auth_token: Optional[str] = None,
        test_images_dir: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[int, ImageBenchmarkResult]:
        """Run ResNet classification benchmark against a server."""
        self.logger.info(f"Running ResNet server benchmark: {model_name}")
        
        if not self.session:
            self.start_session(f"resnet_benchmark_{model_name}")
            self.configure_benchmark(model_type="image_classification")
        
        # Check model availability first
        self.logger.info(f"Checking model availability: {model_name}")
        from benchmark.resnet_image_benchmark import check_model_availability
        
        is_available, message = check_model_availability(model_name, server_url, auth_token, auto_load=True)
        if not is_available:
            self.logger.error(f"Model availability check failed: {message}")
            raise RuntimeError(f"Cannot proceed with benchmark - {message}")
        
        self.logger.info(f"Model availability confirmed: {message}")
        
        # Store server configuration
        self.session.server_config = {
            "url": server_url,
            "model_name": model_name,
            "model_type": "image_classification",
            "auth_token": bool(auth_token),
            "top_k": top_k
        }
        
        # Create ResNet classification function with availability check disabled
        # (we already checked above)
        resnet_function = create_resnet_classification_function(
            model_name=model_name,
            base_url=server_url,
            auth_token=auth_token,
            top_k=top_k,
            check_availability=False,  # Skip check since we already did it
            auto_load=False  # Don't auto-load since we already handled it
        )
        
        # Create ResNet benchmarker
        benchmarker = ResNetImageBenchmarker(
            default_width=224,
            default_height=224,
            warmup_requests=5,
            test_images_dir=test_images_dir
        )
        
        # Run benchmark
        results = benchmarker.benchmark_resnet_model(
            classification_function=resnet_function,
            concurrency_levels=self.session.config.concurrency_levels,
            iterations_per_level=self.session.config.iterations_per_level
        )
        
        if not self.session.results:
            self.session.results = {}
        self.session.results[f"resnet_{model_name}"] = results
        
        self._print_image_benchmark_summary(f"ResNet Server Benchmark ({model_name})", results)
        return results
    
    async def run_image_model_benchmark(
        self,
        image_model_func,
        model_name: str = "custom_image_model",
        custom_prompts: Optional[List[str]] = None,
        image_params: Optional[Dict[str, Any]] = None
    ) -> Dict[int, ImageBenchmarkResult]:
        """Run benchmark on a custom image generation model."""
        self.logger.info(f"Running image model benchmark: {model_name}")
        
        if not self.session:
            self.start_session(f"image_benchmark_{model_name}")
            self.configure_benchmark(model_type="image")
        
        # Store model configuration
        self.session.server_config = {
            "model_name": model_name,
            "model_type": "image",
            "custom_prompts": bool(custom_prompts),
            "custom_params": bool(image_params)
        }
        
        # Run benchmark
        harness = ImageBenchmarkHarness(self.session.config)
        
        results = harness.run_benchmark(
            image_model_func,
            benchmark_name=f"image_{model_name}",
            test_prompts=custom_prompts,
            image_params=image_params
        )
        
        if not self.session.results:
            self.session.results = {}
        self.session.results[model_name] = results
        
        self._print_image_benchmark_summary(f"Image Model Benchmark ({model_name})", results)
        return results
    
    async def run_image_resolution_comparison(
        self,
        image_model_func,
        resolutions: List[tuple] = None,
        model_name: str = "resolution_test"
    ) -> Dict[str, Dict[int, ImageBenchmarkResult]]:
        """Run comparison benchmark across multiple image resolutions."""
        self.logger.info(f"Running image resolution comparison: {model_name}")
        
        if not self.session:
            self.start_session("image_resolution_comparison")
            self.configure_benchmark(model_type="image")
        
        if not resolutions:
            resolutions = [(256, 256), (512, 512), (768, 768), (1024, 1024)]
        
        all_results = {}
        
        for width, height in resolutions:
            resolution_name = f"{width}x{height}"
            self.logger.info(f"Benchmarking resolution: {resolution_name}")
            
            try:
                # Create temporary session config with specific resolution
                temp_config = BenchmarkConfig(
                    model_type="image",
                    concurrency_levels=self.session.config.concurrency_levels,
                    iterations_per_level=self.session.config.iterations_per_level,
                    timeout_seconds=self.session.config.timeout_seconds,
                    text_variations=self.session.config.text_variations,
                    output_dir=self.session.config.output_dir,
                    image_width=width,
                    image_height=height,
                    num_images=self.session.config.num_images,
                    num_inference_steps=self.session.config.num_inference_steps,
                    guidance_scale=self.session.config.guidance_scale
                )
                
                harness = ImageBenchmarkHarness(temp_config)
                results = harness.run_benchmark(
                    image_model_func,
                    benchmark_name=f"{model_name}_{resolution_name}",
                    image_params={
                        'width': width,
                        'height': height,
                        'num_images': self.session.config.num_images,
                        'num_inference_steps': self.session.config.num_inference_steps,
                        'guidance_scale': self.session.config.guidance_scale
                    }
                )
                all_results[resolution_name] = results
                
            except Exception as e:
                self.logger.error(f"Failed to benchmark resolution {resolution_name}: {e}")
                continue
        
        if len(all_results) > 1:
            # Generate comparison
            harness = ImageBenchmarkHarness(self.session.config)
            self.image_reporter.generate_comparison_csv(all_results)
            
            # Print comparison summary
            self._print_image_resolution_comparison(all_results)
        
        if not self.session.results:
            self.session.results = {}
        self.session.results.update(all_results)
        
        return all_results
    
    async def discover_server_endpoints(self, base_url: str) -> Dict[str, bool]:
        """Discover and test available TTS server endpoints."""
        self.logger.info(f"Discovering TTS endpoints at {base_url}")
        
        # Common TTS endpoint patterns
        endpoints_to_test = [
            "/v1/audio/speech",
            "/v1/audio/speech/stream", 
            "/synthesize",
            "/synthesize/stream",
            "/tts",
            "/tts/stream",
            "/api/v1/tts",
            "/api/tts"
        ]
        
        discovered = {}
        
        async with HTTPTTSClient(TTSServerConfig(base_url=base_url, timeout=5.0)) as client:
            for endpoint in endpoints_to_test:
                try:
                    # Test with torch-inference compatible format for /synthesize endpoint
                    if endpoint == "/synthesize":
                        test_payload = {
                            "model_name": "speecht5_tts",
                            "inputs": "Test connectivity",
                            "language": "en"
                        }
                    else:
                        # Use generic format for other endpoints
                        test_payload = {
                            "text": "Test connectivity",
                            "sample_rate": 22050
                        }
                    
                    url = f"{base_url}{endpoint}"
                    async with client.session.post(url, json=test_payload) as response:
                        if response.status == 200:
                            # For torch-inference, also check if response is successful
                            if endpoint == "/synthesize":
                                data = await response.json()
                                if data.get('success', False):
                                    discovered[endpoint] = True
                                    self.logger.info(f"Found working endpoint: {endpoint}")
                                else:
                                    discovered[endpoint] = False
                                    self.logger.debug(f"Endpoint {endpoint} returned error: {data.get('error', 'unknown')}")
                            else:
                                discovered[endpoint] = True
                                self.logger.info(f"Found working endpoint: {endpoint}")
                        else:
                            discovered[endpoint] = False
                            
                except Exception as e:
                    discovered[endpoint] = False
                    self.logger.debug(f"Endpoint {endpoint} failed: {e}")
        
        working_endpoints = [ep for ep, works in discovered.items() if works]
        if working_endpoints:
            self.logger.info(f"Discovered {len(working_endpoints)} working endpoints")
        else:
            self.logger.warning("No working TTS endpoints found")
        
        return discovered
    
    async def run_server_benchmark(
        self,
        server_url: str,
        voice: str = "default",
        streaming: bool = False,
        auth_token: Optional[str] = None,
        auto_discover: bool = True
    ) -> Dict[int, TTSBenchmarkResult]:
        """Run comprehensive server benchmark."""
        self.logger.info(f"Running server benchmark: {server_url}")
        
        if not self.session:
            self.start_session(f"server_benchmark_{voice}")
            self.configure_benchmark()
        
        # Store server configuration
        self.session.server_config = {
            "url": server_url,
            "voice": voice,
            "streaming": streaming,
            "auth_token": bool(auth_token)
        }
        
        # Auto-discover endpoints if requested
        if auto_discover:
            discovered = await self.discover_server_endpoints(server_url)
            self.session.metadata["discovered_endpoints"] = discovered
        
        # Test connectivity
        server_config = TTSServerConfig(
            base_url=server_url,
            auth_token=auth_token
        )
        
        self.logger.info("Testing server connectivity...")
        if not await test_server_connectivity(server_config):
            raise RuntimeError(f"Cannot connect to TTS server at {server_url}")
        
        self.logger.info("Server connectivity confirmed")
        
        # Create TTS function
        tts_function = create_torch_inference_tts_function(
            base_url=server_url,
            voice=voice,
            streaming=streaming,
            auth_token=auth_token,
            sample_rate=self.session.config.sample_rate
        )
        
        # Run benchmark
        harness = TTSBenchmarkHarness(self.session.config)
        benchmark_name = f"server_{voice}{'_streaming' if streaming else ''}"
        
        results = await harness.run_benchmark_async(
            tts_function,
            benchmark_name=benchmark_name
        )
        
        if not self.session.results:
            self.session.results = {}
        self.session.results[benchmark_name] = results
        
        self._print_benchmark_summary(f"Server Benchmark ({voice})", results)
        return results
    
    async def run_voice_comparison(
        self,
        server_url: str,
        voices: List[str],
        auth_token: Optional[str] = None
    ) -> Dict[str, Dict[int, TTSBenchmarkResult]]:
        """Run comparison benchmark across multiple voices."""
        self.logger.info(f"Running voice comparison: {voices}")
        
        if not self.session:
            self.start_session("voice_comparison")
            self.configure_benchmark(concurrency_levels=[1, 2, 4, 8, 16, 32, 64], iterations=100)
        
        all_results = {}
        
        for voice in voices:
            self.logger.info(f"Benchmarking voice: {voice}")
            
            try:
                results = await self.run_server_benchmark(
                    server_url=server_url,
                    voice=voice,
                    auth_token=auth_token,
                    auto_discover=False  # Skip discovery for subsequent voices
                )
                all_results[voice] = results
                
            except Exception as e:
                self.logger.error(f"Failed to benchmark voice {voice}: {e}")
                continue
        
        if len(all_results) > 1:
            # Generate comparison
            harness = TTSBenchmarkHarness(self.session.config)
            harness.compare_benchmarks(all_results, "voice_comparison")
            
            # Print comparison summary
            self._print_voice_comparison(all_results)
        
        if not self.session.results:
            self.session.results = {}
        self.session.results.update(all_results)
        
        return all_results
    
    async def run_comprehensive_benchmark(
        self,
        server_url: str,
        voices: Optional[List[str]] = None,
        test_streaming: bool = True,
        auth_token: Optional[str] = None
    ) -> Dict[str, Dict[int, TTSBenchmarkResult]]:
        """Run comprehensive benchmark covering all aspects."""
        self.logger.info("Running comprehensive benchmark suite")
        
        if not self.session:
            self.start_session("comprehensive_benchmark")
            self.configure_benchmark()
        
        all_results = {}
        
        # 1. Server discovery
        discovered = await self.discover_server_endpoints(server_url)
        self.session.metadata["discovered_endpoints"] = discovered
        
        # 2. Basic voice benchmarks
        voices = voices or ["default"]
        for voice in voices:
            self.logger.info(f"Comprehensive test for voice: {voice}")
            
            # Non-streaming benchmark
            try:
                non_streaming_results = await self.run_server_benchmark(
                    server_url=server_url,
                    voice=voice,
                    streaming=False,
                    auth_token=auth_token,
                    auto_discover=False
                )
                all_results[f"{voice}_non_streaming"] = non_streaming_results
                
            except Exception as e:
                self.logger.error(f"Non-streaming benchmark failed for {voice}: {e}")
            
            # Streaming benchmark (if requested)
            if test_streaming:
                try:
                    streaming_results = await self.run_server_benchmark(
                        server_url=server_url,
                        voice=voice,
                        streaming=True,
                        auth_token=auth_token,
                        auto_discover=False
                    )
                    all_results[f"{voice}_streaming"] = streaming_results
                    
                except Exception as e:
                    self.logger.error(f"Streaming benchmark failed for {voice}: {e}")
        
        # 3. Generate comprehensive comparison
        if len(all_results) > 1:
            harness = TTSBenchmarkHarness(self.session.config)
            harness.compare_benchmarks(all_results, "comprehensive_comparison")
            
            # Print comprehensive summary
            self._print_comprehensive_summary(all_results)
        
        if not self.session.results:
            self.session.results = {}
        self.session.results.update(all_results)
        
        return all_results
    
    def save_session(self) -> str:
        """Save the current benchmark session."""
        if not self.session:
            raise RuntimeError("No active session to save")
        
        # Add end metadata
        if self.session.metadata:
            self.session.metadata["end_time"] = time.time()
            self.session.metadata["duration"] = (
                self.session.metadata["end_time"] - self.session.metadata["start_time"]
            )
        
        # Save session data
        session_file = self.output_dir / f"{self.session.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(self.session.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Session saved to: {session_file}")
        return str(session_file)
    
    def load_session(self, session_file: str) -> BenchmarkSession:
        """Load a previous benchmark session."""
        with open(session_file, 'r') as f:
            data = json.load(f)
        
        # Reconstruct session (simplified - would need full reconstruction for results)
        session = BenchmarkSession(
            session_id=data["session_id"],
            timestamp=data["timestamp"],
            config=BenchmarkConfig(**data["config"]),
            server_config=data.get("server_config"),
            metadata=data.get("metadata")
        )
        
        self.session = session
        self.logger.info(f"Loaded session: {session.session_id}")
        return session
    
    def _print_benchmark_summary(self, title: str, results: Dict[int, TTSBenchmarkResult]):
        """Print formatted TTS benchmark summary."""
        print(f"\n{'='*80}")
        print(f"{title} - Results Summary")
        print(f"{'='*80}")
        
        self.tts_reporter.print_summary_table(results, title)
        
        # Validation warnings
        for concurrency, result in results.items():
            warnings = validate_metrics_consistency(result.metrics)
            if warnings:
                print(f"\nValidation issues for concurrency {concurrency}:")
                for warning in warnings:
                    print(f"   - {warning}")
    
    def _print_image_benchmark_summary(self, title: str, results: Dict[int, ImageBenchmarkResult]):
        """Print formatted Image benchmark summary."""
        print(f"\n{'='*80}")
        print(f"{title} - Results Summary")
        print(f"{'='*80}")
        
        self.image_reporter.print_summary_table(results, title)
        
        # Validation warnings for image metrics
        for concurrency, result in results.items():
            warnings = validate_image_metrics_consistency(result.metrics)
            if warnings:
                print(f"\nValidation issues for concurrency {concurrency}:")
                for warning in warnings:
                    print(f"   - {warning}")
    
    def _print_image_resolution_comparison(self, all_results: Dict[str, Dict[int, ImageBenchmarkResult]]):
        """Print image resolution comparison summary."""
        print(f"\n{'='*80}")
        print("IMAGE RESOLUTION COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Show results for all concurrency levels
        all_concurrencies = set()
        for results in all_results.values():
            all_concurrencies.update(results.keys())
        concurrencies = sorted(all_concurrencies)
        
        for concurrency in concurrencies:
            print(f"\nPerformance at Concurrency {concurrency}:")
            print(f"{'Resolution':<12} {'IPS':<10} {'PPS':<12} {'RPS':<10} {'TTFI p95':<12} {'Memory MB':<12} {'Success%':<10}")
            print("-" * 90)
            
            for resolution, results in all_results.items():
                if concurrency in results:
                    metrics = results[concurrency].metrics
                    print(f"{resolution:<12} "
                          f"{metrics.ips:<10.3f} "
                          f"{metrics.pps:<12.0f} "
                          f"{metrics.rps:<10.1f} "
                          f"{metrics.ttfi_p95*1000:<12.1f} "
                          f"{metrics.avg_memory_peak_mb:<12.1f} "
                          f"{metrics.success_rate:<10.1f}")
                else:
                    print(f"{resolution:<12} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
        
        # Print best performance summary
        print(f"\n{'Best Performance Summary:'}")
        print("-" * 50)
        best_ips = max((max(results.values(), key=lambda r: r.metrics.ips) for results in all_results.values()), 
                       key=lambda r: r.metrics.ips)
        best_resolution = next(res for res, results in all_results.items() 
                              if any(r.metrics.ips == best_ips.metrics.ips for r in results.values()))
        best_concurrency = next(conc for conc, r in all_results[best_resolution].items() 
                               if r.metrics.ips == best_ips.metrics.ips)
        print(f"Highest IPS: {best_ips.metrics.ips:.3f} at {best_resolution} with concurrency {best_concurrency}")
        print(f"Highest PPS: {best_ips.metrics.pps:.0f} at {best_resolution} with concurrency {best_concurrency}")
    
    def _print_voice_comparison(self, all_results: Dict[str, Dict[int, TTSBenchmarkResult]]):
        """Print voice comparison summary."""
        print(f"\n{'='*80}")
        print("VOICE COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Find common concurrency level for comparison
        concurrency = 4
        if not all(concurrency in results for results in all_results.values()):
            concurrency = 1
        
        print(f"\nPerformance at Concurrency {concurrency}:")
        print(f"{'Voice':<20} {'ASPS':<10} {'RTF':<10} {'RPS':<10} {'TTFA p95':<12} {'Success%':<10}")
        print("-" * 80)
        
        for voice, results in all_results.items():
            if concurrency in results:
                metrics = results[concurrency].metrics
                print(f"{voice:<20} "
                      f"{metrics.asps:<10.3f} "
                      f"{metrics.rtf_median:<10.3f} "
                      f"{metrics.rps:<10.1f} "
                      f"{metrics.ttfa_p95*1000:<12.1f} "
                      f"{metrics.success_rate:<10.1f}")
    
    def _print_comprehensive_summary(self, all_results: Dict[str, Dict[int, TTSBenchmarkResult]]):
        """Print comprehensive benchmark summary."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print(f"{'='*80}")
        
        # Group by voice
        voices = set()
        for key in all_results.keys():
            voice = key.split('_')[0]
            voices.add(voice)
        
        for voice in sorted(voices):
            print(f"\nVoice: {voice}")
            print("-" * 40)
            
            # Find streaming and non-streaming results
            non_streaming_key = f"{voice}_non_streaming"
            streaming_key = f"{voice}_streaming"
            
            for mode, key in [("Non-Streaming", non_streaming_key), ("Streaming", streaming_key)]:
                if key in all_results and 4 in all_results[key]:
                    metrics = all_results[key][4].metrics
                    print(f"  {mode:<15}: "
                          f"ASPS={metrics.asps:.3f}, "
                          f"RTF={metrics.rtf_median:.3f}, "
                          f"TTFA p95={metrics.ttfa_p95*1000:.1f}ms")


async def main():
    """Main CLI interface for end-to-end benchmarking."""
    parser = argparse.ArgumentParser(
        description="End-to-End TTS and Image Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # TTS Benchmarks
  python benchmark.py demo                                    # Run demo TTS benchmark
  python benchmark.py server --url http://localhost:8000     # Basic server benchmark
  python benchmark.py voices --url http://localhost:8000 --voices default premium
  python benchmark.py comprehensive --url http://localhost:8000 --voices default premium
  python benchmark.py discover --url http://localhost:8000   # Discover endpoints only
  
  # Image Benchmarks
  python benchmark.py image-demo                              # Run demo image benchmark
  python benchmark.py image-resolution --model mymodel       # Test different resolutions
  
  # ResNet Classification Benchmarks
  python benchmark.py resnet-demo                             # Run demo ResNet benchmark
  python benchmark.py resnet-server --model resnet18  # Benchmark ResNet server
        """
    )
    
    parser.add_argument("--output-dir", default="benchmark_sessions", help="Output directory")
    parser.add_argument("--session-name", help="Custom session name")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    subparsers = parser.add_subparsers(dest="command", help="Benchmark commands")
    
    # Demo benchmark
    demo_parser = subparsers.add_parser("demo", help="Run demo benchmark with synthetic TTS")
    demo_parser.add_argument("--iterations", type=int, default=100, help="Iterations per concurrency level")
    
    # Server benchmark
    server_parser = subparsers.add_parser("server", help="Benchmark TTS server")
    server_parser.add_argument("--url", required=True, help="TTS server URL")
    server_parser.add_argument("--voice", default="speecht5_tts", help="Voice model")
    server_parser.add_argument("--streaming", action="store_true", help="Test streaming endpoint")
    server_parser.add_argument("--auth-token", help="Authentication token")
    server_parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64], help="Concurrency levels")
    server_parser.add_argument("--iterations", type=int, default=100, help="Iterations per level")
    server_parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout")
    
    # Voice comparison
    voices_parser = subparsers.add_parser("voices", help="Compare multiple voice models")
    voices_parser.add_argument("--url", required=True, help="TTS server URL")
    voices_parser.add_argument("--voices", nargs="+", required=True, help="Voice models to compare")
    voices_parser.add_argument("--auth-token", help="Authentication token")
    voices_parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64], help="Concurrency levels")
    voices_parser.add_argument("--iterations", type=int, default=100, help="Iterations per level")
    
    # Comprehensive benchmark
    comprehensive_parser = subparsers.add_parser("comprehensive", help="Run comprehensive benchmark suite")
    comprehensive_parser.add_argument("--url", required=True, help="TTS server URL")
    comprehensive_parser.add_argument("--voices", nargs="+", default=["speecht5_tts"], help="Voice models to test")
    comprehensive_parser.add_argument("--no-streaming", action="store_true", help="Skip streaming tests")
    comprehensive_parser.add_argument("--auth-token", help="Authentication token")
    comprehensive_parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64], help="Concurrency levels")
    comprehensive_parser.add_argument("--iterations", type=int, default=100, help="Iterations per level")
    
    # Endpoint discovery
    discover_parser = subparsers.add_parser("discover", help="Discover available TTS endpoints")
    discover_parser.add_argument("--url", required=True, help="TTS server base URL")
    
    # Image benchmark commands
    image_demo_parser = subparsers.add_parser("image-demo", help="Run demo image benchmark")
    image_demo_parser.add_argument("--iterations", type=int, default=20, help="Iterations per concurrency level")
    image_demo_parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8], help="Concurrency levels")
    image_demo_parser.add_argument("--width", type=int, default=512, help="Image width")
    image_demo_parser.add_argument("--height", type=int, default=512, help="Image height")
    image_demo_parser.add_argument("--steps", type=int, default=20, help="Inference steps")
    
    # Image resolution comparison
    image_resolution_parser = subparsers.add_parser("image-resolution", help="Compare image generation at different resolutions")
    image_resolution_parser.add_argument("--model", default="demo", help="Model name for testing")
    image_resolution_parser.add_argument("--resolutions", nargs="+", default=["256x256", "512x512", "768x768"], help="Resolutions to test (format: WIDTHxHEIGHT)")
    image_resolution_parser.add_argument("--iterations", type=int, default=15, help="Iterations per resolution")
    image_resolution_parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4], help="Concurrency levels")
    image_resolution_parser.add_argument("--steps", type=int, default=20, help="Inference steps")
    
    # ResNet benchmark commands
    resnet_demo_parser = subparsers.add_parser("resnet-demo", help="Run demo ResNet classification benchmark")
    resnet_demo_parser.add_argument("--iterations", type=int, default=50, help="Iterations per concurrency level")
    resnet_demo_parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16], help="Concurrency levels")
    
    resnet_server_parser = subparsers.add_parser("resnet-server", help="Benchmark ResNet classification against a server")
    resnet_server_parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    resnet_server_parser.add_argument("--model", default="resnet18", help="ResNet model name")
    resnet_server_parser.add_argument("--auth-token", help="Authentication token")
    resnet_server_parser.add_argument("--test-images-dir", help="Directory containing test images")
    resnet_server_parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to return")
    resnet_server_parser.add_argument("--iterations", type=int, default=100, help="Iterations per concurrency level")
    resnet_server_parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16], help="Concurrency levels")
    resnet_server_parser.add_argument("--enforce-gpu", action="store_true", help="Enforce GPU usage and validate GPU performance")
    resnet_server_parser.add_argument("--gpu-validation", action="store_true", help="Validate GPU setup before running benchmark")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True  # This will remove existing handlers and reconfigure
    )
    
    # Create runner
    runner = EndToEndBenchmarkRunner(args.output_dir)
    
    try:
        if args.command == "demo":
            runner.start_session(args.session_name or "demo")
            runner.configure_benchmark(iterations=args.iterations)
            results = runner.run_demo_benchmark()
            
        elif args.command == "server":
            runner.start_session(args.session_name or f"server_{args.voice}")
            runner.configure_benchmark(
                concurrency_levels=args.concurrency,
                iterations=args.iterations,
                timeout=args.timeout
            )
            results = await runner.run_server_benchmark(
                server_url=args.url,
                voice=args.voice,
                streaming=args.streaming,
                auth_token=args.auth_token
            )
            
        elif args.command == "voices":
            runner.start_session(args.session_name or "voice_comparison")
            runner.configure_benchmark(
                concurrency_levels=args.concurrency,
                iterations=args.iterations
            )
            results = await runner.run_voice_comparison(
                server_url=args.url,
                voices=args.voices,
                auth_token=args.auth_token
            )
            
        elif args.command == "comprehensive":
            runner.start_session(args.session_name or "comprehensive")
            runner.configure_benchmark(
                concurrency_levels=args.concurrency,
                iterations=args.iterations
            )
            results = await runner.run_comprehensive_benchmark(
                server_url=args.url,
                voices=args.voices,
                test_streaming=not args.no_streaming,
                auth_token=args.auth_token
            )
            
        elif args.command == "discover":
            runner.start_session(args.session_name or "discovery")
            discovered = await runner.discover_server_endpoints(args.url)
            
            print(f"\nTTS Endpoint Discovery Results for {args.url}")
            print("=" * 60)
            for endpoint, works in discovered.items():
                status = "Working" if works else "Not available"
                print(f"{endpoint:<30} {status}")
            
            working_count = sum(1 for works in discovered.values() if works)
            print(f"\nSummary: {working_count}/{len(discovered)} endpoints are working")
            
        elif args.command == "image-demo":
            # Automatic GPU validation for image benchmarks
            print("🔍 Validating GPU environment for image generation benchmark...")
            gpu_available = validate_gpu_environment()
            
            if gpu_available:
                setup_gpu_optimizations()
                print("🚀 GPU detected - enabling performance optimizations")
            
            if not runner.session:
                runner.start_session(args.session_name or "image_demo")
                runner.configure_benchmark(
                    model_type="image",
                    concurrency_levels=args.concurrency,
                    iterations=args.iterations,
                    image_width=args.width,
                    image_height=args.height,
                    num_inference_steps=args.steps
                )
            results = runner.run_demo_image_benchmark()
            
        elif args.command == "image-resolution":
            if not runner.session:
                runner.start_session(args.session_name or "image_resolution_test")
                runner.configure_benchmark(
                    model_type="image",
                    concurrency_levels=args.concurrency,
                    iterations=args.iterations,
                    num_inference_steps=args.steps
                )
            
            # Parse resolutions
            resolutions = []
            for res_str in args.resolutions:
                try:
                    width, height = map(int, res_str.split('x'))
                    resolutions.append((width, height))
                except ValueError:
                    print(f"Invalid resolution format: {res_str}. Use WIDTHxHEIGHT (e.g., 512x512)")
                    return
            
            if args.model == "demo":
                # Use demo model
                results = await runner.run_image_resolution_comparison(
                    gpu_demo_image_model,
                    resolutions=resolutions,
                    model_name="demo_resolution_test"
                )
            else:
                print(f"Custom model '{args.model}' not implemented in CLI. Use demo model or implement custom model loading.")
                return
            
        elif args.command == "resnet-demo":
            # Automatic GPU validation for ResNet demo benchmark
            print("🔍 Validating GPU environment for ResNet demo benchmark...")
            gpu_available = validate_gpu_environment()
            
            if gpu_available:
                setup_gpu_optimizations()
                print("🚀 GPU detected - enabling performance optimizations")
            
            if not runner.session:
                runner.start_session(args.session_name or "resnet_demo")
                runner.configure_benchmark(
                    model_type="image_classification",
                    concurrency_levels=args.concurrency,
                    iterations=args.iterations
                )
            results = runner.run_demo_resnet_benchmark()
            
        elif args.command == "resnet-server":
            # Automatic GPU validation for ResNet server benchmarks
            print("🔍 Validating GPU environment for ResNet server benchmark...")
            gpu_available = validate_gpu_environment()
            
            # GPU validation and enforcement
            if hasattr(args, 'gpu_validation') and args.gpu_validation:
                if not gpu_available:
                    print("❌ GPU validation failed!")
                    return
                else:
                    print("✅ GPU validation passed!")
            
            if hasattr(args, 'enforce_gpu') and args.enforce_gpu:
                if not gpu_available:
                    print("❌ GPU enforcement enabled but GPU validation failed!")
                    return
                setup_gpu_optimizations()
                print("🚀 GPU enforcement enabled - maximum performance mode")
            elif gpu_available:
                # Setup GPU optimizations automatically if GPU is available
                setup_gpu_optimizations()
                print("🚀 GPU detected - enabling performance optimizations")
            
            if not runner.session:
                runner.start_session(args.session_name or f"resnet_server_{args.model}")
                runner.configure_benchmark(
                    model_type="image_classification",
                    concurrency_levels=args.concurrency,
                    iterations=args.iterations
                )
            results = await runner.run_resnet_server_benchmark(
                model_name=args.model,
                server_url=args.url,
                auth_token=args.auth_token,
                test_images_dir=args.test_images_dir,
                top_k=args.top_k
            )
            
        else:
            parser.print_help()
            return
        
        # Save session if we have results
        if hasattr(runner, 'session') and runner.session:
            session_file = runner.save_session()
            print(f"\nBenchmark session saved to: {session_file}")
        
        print("\nBenchmark completed successfully!")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        if hasattr(runner, 'session') and runner.session:
            runner.save_session()
        
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        logging.error(f"Benchmark error: {e}")
        logging.debug(traceback.format_exc())
        
        if hasattr(runner, 'session') and runner.session:
            runner.save_session()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

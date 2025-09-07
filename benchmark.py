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

try:
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
    results: Dict[str, Dict[int, TTSBenchmarkResult]] = None
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
    Complete end-to-end TTS benchmark runner.
    
    Provides automated testing, configuration management, and comprehensive reporting.
    """
    
    def __init__(self, output_dir: str = "benchmark_sessions"):
        """Initialize the benchmark runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logging()
        self.reporter = TTSBenchmarkReporter()
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
        concurrency_levels: Optional[List[int]] = None,
        iterations: int = 20,
        timeout: float = 30.0,
        sample_rate: int = 22050,
        text_variations: int = 25,
        output_plots: bool = True
    ) -> None:
        """Configure benchmark parameters."""
        if not self.session:
            raise RuntimeError("No active session. Call start_session() first.")
        
        self.session.config = BenchmarkConfig(
            concurrency_levels=concurrency_levels or [1, 2, 4, 8, 16, 32, 64],
            iterations_per_level=iterations,
            timeout_seconds=timeout,
            sample_rate=sample_rate,
            text_variations=text_variations,
            generate_plots=output_plots,
            output_dir=str(self.output_dir / self.session.session_id)
        )
        
        self.logger.info(f"Configured benchmark: "
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
        """Print formatted benchmark summary."""
        print(f"\n{'='*80}")
        print(f"{title} - Results Summary")
        print(f"{'='*80}")
        
        self.reporter.print_summary_table(results, title)
        
        # Validation warnings
        for concurrency, result in results.items():
            warnings = validate_metrics_consistency(result.metrics)
            if warnings:
                print(f"\nValidation issues for concurrency {concurrency}:")
                for warning in warnings:
                    print(f"   - {warning}")
    
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
        description="End-to-End TTS Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py demo                                    # Run demo benchmark
  python benchmark.py server --url http://localhost:8000     # Basic server benchmark
  python benchmark.py voices --url http://localhost:8000 --voices default premium
  python benchmark.py comprehensive --url http://localhost:8000 --voices default premium
  python benchmark.py discover --url http://localhost:8000   # Discover endpoints only
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
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
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

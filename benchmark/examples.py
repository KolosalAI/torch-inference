#!/usr/bin/env python3
"""
TTS Benchmark Example Scripts.

Collection of ready-to-use benchmark scripts for different TTS scenarios.
"""

import asyncio
import logging
import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from benchmark.harness import TTSBenchmarkHarness, BenchmarkConfig, create_demo_tts_function
from benchmark.http_client import create_torch_inference_tts_function, test_server_connectivity, TTSServerConfig
from benchmark.reporter import TTSBenchmarkReporter
from benchmark.metrics import validate_metrics_consistency


def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_demo_benchmark():
    """Run a demo benchmark with synthetic TTS function."""
    print("Running Demo TTS Benchmark")
    print("=" * 50)
    
    # Create demo TTS function with realistic parameters
    demo_tts = create_demo_tts_function(
        min_audio_duration=0.5,
        max_audio_duration=4.0,
        processing_delay=0.02,  # 20ms processing time
        failure_rate=0.01  # 1% failure rate
    )
    
    # Configure benchmark
    config = BenchmarkConfig(
        concurrency_levels=[1, 2, 4, 8, 16, 32, 64],
        iterations_per_level=100,
        text_variations=15,
        output_dir="demo_benchmark_results"
    )
    
    # Generate test texts
    harness = TTSBenchmarkHarness(config)
    test_texts = harness.generate_test_texts(
        count=15,
        min_length=30,
        max_length=150
    )
    
    print(f"Generated {len(test_texts)} test texts")
    print(f"Text length range: {min(len(t) for t in test_texts)}-{max(len(t) for t in test_texts)} chars")
    print()
    
    # Run benchmark
    results = harness.run_benchmark(
        demo_tts,
        test_texts=test_texts,
        benchmark_name="demo_tts_v1"
    )
    
    # Print summary
    reporter = TTSBenchmarkReporter()
    reporter.print_summary_table(results, "Demo TTS Benchmark")
    
    return results


async def run_http_server_benchmark(
    server_url: str = "http://localhost:8000",
    voice: str = "speecht5_tts",
    streaming: bool = False,
    auth_token: Optional[str] = None,
    concurrency_levels: Optional[List[int]] = None,
    iterations: int = 100
):
    """
    Run benchmark against HTTP TTS server.
    
    Args:
        server_url: TTS server URL
        voice: Voice model to use
        streaming: Whether to use streaming endpoint
        auth_token: Authentication token if required
        concurrency_levels: List of concurrency levels to test
        iterations: Iterations per concurrency level
    """
    print(f"Running HTTP TTS Server Benchmark")
    print(f"Server: {server_url}")
    print(f"Voice: {voice}")
    print(f"Streaming: {streaming}")
    print("=" * 50)
    
    # Test server connectivity first
    server_config = TTSServerConfig(
        base_url=server_url,
        auth_token=auth_token
    )
    
    print("Testing server connectivity...")
    if not await test_server_connectivity(server_config):
        print("❌ Cannot connect to TTS server. Please check:")
        print("  1. Server is running")
        print("  2. URL is correct")
        print("  3. Authentication token is valid")
        return None
    
    print("✅ Server connectivity OK")
    print()
    
    # Create TTS function
    tts_function = create_torch_inference_tts_function(
        base_url=server_url,
        voice=voice,
        streaming=streaming,
        auth_token=auth_token,
        sample_rate=22050
    )
    
    # Configure benchmark
    config = BenchmarkConfig(
        concurrency_levels=concurrency_levels or [1, 2, 4, 8, 16, 32, 64],
        iterations_per_level=iterations,
        text_variations=20,
        output_dir=f"http_benchmark_results_{voice}{'_streaming' if streaming else ''}"
    )
    
    # Run benchmark
    harness = TTSBenchmarkHarness(config)
    benchmark_name = f"http_tts_{voice}{'_streaming' if streaming else ''}"
    
    results = await harness.run_benchmark_async(
        tts_function,
        benchmark_name=benchmark_name
    )
    
    # Print summary
    reporter = TTSBenchmarkReporter()
    reporter.print_summary_table(results, f"HTTP TTS Benchmark ({voice})")
    
    return results


async def run_voice_comparison_benchmark(
    server_url: str = "http://localhost:8000",
    voices: List[str] = ["speecht5_tts"],
    auth_token: Optional[str] = None
):
    """
    Compare performance across different voice models.
    
    Args:
        server_url: TTS server URL
        voices: List of voice models to compare
        auth_token: Authentication token if required
    """
    print("Running Voice Comparison Benchmark")
    print(f"Server: {server_url}")
    print(f"Voices: {voices}")
    print("=" * 50)
    
    all_results = {}
    
    for voice in voices:
        print(f"\nBenchmarking voice: {voice}")
        print("-" * 30)
        
        results = await run_http_server_benchmark(
            server_url=server_url,
            voice=voice,
            auth_token=auth_token,
            concurrency_levels=[1, 2, 4],
            iterations=8
        )
        
        if results:
            all_results[voice] = results
    
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("VOICE COMPARISON SUMMARY")
        print("=" * 60)
        
        # Create comparison harness
        config = BenchmarkConfig(output_dir="voice_comparison_results")
        harness = TTSBenchmarkHarness(config)
        
        harness.compare_benchmarks(all_results, "voice_comparison")
        
        # Print comparison table
        print("\nPerformance Comparison (Concurrency 4):")
        print(f"{'Voice':<15} {'ASPS':<8} {'RTF':<8} {'RPS':<8} {'TTFA p95':<10}")
        print("-" * 55)
        
        for voice, results in all_results.items():
            if 4 in results:
                metrics = results[4].metrics
                print(f"{voice:<15} "
                      f"{metrics.asps:<8.3f} "
                      f"{metrics.rtf_median:<8.3f} "
                      f"{metrics.rps:<8.1f} "
                      f"{metrics.ttfa_p95*1000:<10.1f}")
    
    return all_results


def run_streaming_vs_non_streaming_benchmark(
    server_url: str = "http://localhost:8000",
    voice: str = "default",
    auth_token: Optional[str] = None
):
    """
    Compare streaming vs non-streaming performance.
    
    Args:
        server_url: TTS server URL
        voice: Voice model to use
        auth_token: Authentication token if required
    """
    async def compare():
        print("Running Streaming vs Non-Streaming Benchmark")
        print(f"Server: {server_url}")
        print(f"Voice: {voice}")
        print("=" * 50)
        
        # Benchmark non-streaming
        print("\n1. Non-Streaming TTS")
        print("-" * 30)
        non_streaming_results = await run_http_server_benchmark(
            server_url=server_url,
            voice=voice,
            streaming=False,
            auth_token=auth_token,
            concurrency_levels=[1, 2, 4],
            iterations=10
        )
        
        # Benchmark streaming
        print("\n2. Streaming TTS")
        print("-" * 30)
        streaming_results = await run_http_server_benchmark(
            server_url=server_url,
            voice=voice,
            streaming=True,
            auth_token=auth_token,
            concurrency_levels=[1, 2, 4],
            iterations=10
        )
        
        if non_streaming_results and streaming_results:
            # Generate comparison
            comparison_results = {
                "non_streaming": non_streaming_results,
                "streaming": streaming_results
            }
            
            config = BenchmarkConfig(output_dir="streaming_comparison_results")
            harness = TTSBenchmarkHarness(config)
            harness.compare_benchmarks(comparison_results, "streaming_comparison")
            
            print("\n" + "=" * 60)
            print("STREAMING vs NON-STREAMING COMPARISON")
            print("=" * 60)
            
            for concurrency in [1, 2, 4]:
                if concurrency in non_streaming_results and concurrency in streaming_results:
                    ns_metrics = non_streaming_results[concurrency].metrics
                    s_metrics = streaming_results[concurrency].metrics
                    
                    print(f"\nConcurrency {concurrency}:")
                    print(f"  Non-Streaming - ASPS: {ns_metrics.asps:.3f}, TTFA p95: {ns_metrics.ttfa_p95*1000:.1f}ms")
                    print(f"  Streaming     - ASPS: {s_metrics.asps:.3f}, TTFA p95: {s_metrics.ttfa_p95*1000:.1f}ms")
                    
                    if s_metrics.ttfa_p95 > 0 and ns_metrics.ttfa_p95 > 0:
                        ttfa_improvement = (ns_metrics.ttfa_p95 - s_metrics.ttfa_p95) / ns_metrics.ttfa_p95 * 100
                        print(f"  TTFA Improvement: {ttfa_improvement:+.1f}%")
        
        return non_streaming_results, streaming_results
    
    return asyncio.run(compare())


def main():
    """Main CLI interface for TTS benchmarks."""
    parser = argparse.ArgumentParser(description="TTS Benchmark Suite")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    subparsers = parser.add_subparsers(dest="command", help="Benchmark commands")
    
    # Demo benchmark
    demo_parser = subparsers.add_parser("demo", help="Run demo benchmark with synthetic TTS")
    
    # HTTP server benchmark
    http_parser = subparsers.add_parser("http", help="Benchmark HTTP TTS server")
    http_parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    http_parser.add_argument("--voice", default="speecht5_tts", help="Voice model")
    http_parser.add_argument("--streaming", action="store_true", help="Use streaming endpoint")
    http_parser.add_argument("--auth-token", help="Authentication token")
    http_parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8], help="Concurrency levels")
    http_parser.add_argument("--iterations", type=int, default=100, help="Iterations per level")
    
    # Voice comparison
    voices_parser = subparsers.add_parser("voices", help="Compare different voice models")
    voices_parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    voices_parser.add_argument("--voices", nargs="+", default=["speecht5_tts"], help="Voice models to compare")
    voices_parser.add_argument("--auth-token", help="Authentication token")
    
    # Streaming comparison
    streaming_parser = subparsers.add_parser("streaming", help="Compare streaming vs non-streaming")
    streaming_parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    streaming_parser.add_argument("--voice", default="speecht5_tts", help="Voice model")
    streaming_parser.add_argument("--auth-token", help="Authentication token")
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    if args.command == "demo":
        run_demo_benchmark()
    elif args.command == "http":
        asyncio.run(run_http_server_benchmark(
            server_url=args.url,
            voice=args.voice,
            streaming=args.streaming,
            auth_token=args.auth_token,
            concurrency_levels=args.concurrency,
            iterations=args.iterations
        ))
    elif args.command == "voices":
        asyncio.run(run_voice_comparison_benchmark(
            server_url=args.url,
            voices=args.voices,
            auth_token=args.auth_token
        ))
    elif args.command == "streaming":
        run_streaming_vs_non_streaming_benchmark(
            server_url=args.url,
            voice=args.voice,
            auth_token=args.auth_token
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

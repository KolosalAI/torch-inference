#!/usr/bin/env python3
"""
TTS Benchmark Management Utility.

Convenient script for managing and running TTS benchmarks with common configurations.
"""

import os
import json
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from benchmark.harness import TTSBenchmarkHarness, BenchmarkConfig
from benchmark.http_client import create_torch_inference_tts_function, test_server_connectivity, TTSServerConfig
from benchmark.reporter import TTSBenchmarkReporter


class BenchmarkManager:
    """Utility class for managing TTS benchmarks."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        """Initialize benchmark manager."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.reporter = TTSBenchmarkReporter()
    
    def list_benchmarks(self) -> List[str]:
        """List all available benchmark results."""
        benchmark_files = list(self.results_dir.glob("*_results.json"))
        return [f.stem.replace("_results", "") for f in benchmark_files]
    
    def load_benchmark(self, name: str) -> Optional[Dict]:
        """Load benchmark results by name."""
        results_file = self.results_dir / f"{name}_results.json"
        if not results_file.exists():
            return None
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def compare_benchmarks(self, benchmark_names: List[str], output_name: str = "comparison"):
        """Compare multiple benchmarks."""
        benchmark_data = {}
        
        for name in benchmark_names:
            data = self.load_benchmark(name)
            if data:
                # Convert back to benchmark results format
                from benchmark.harness import TTSBenchmarkHarness
                config = BenchmarkConfig(output_dir=str(self.results_dir))
                harness = TTSBenchmarkHarness(config)
                results = harness.load_benchmark_results(name)
                if results:
                    benchmark_data[name] = results
        
        if len(benchmark_data) > 1:
            # Generate comparison
            config = BenchmarkConfig(output_dir=str(self.results_dir))
            harness = TTSBenchmarkHarness(config)
            harness.compare_benchmarks(benchmark_data, output_name)
            print(f"Comparison saved to {self.results_dir}/{output_name}_comparison.csv")
        else:
            print("Need at least 2 benchmarks to compare")
    
    def print_benchmark_summary(self, name: str):
        """Print summary of a specific benchmark."""
        config = BenchmarkConfig(output_dir=str(self.results_dir))
        harness = TTSBenchmarkHarness(config)
        results = harness.load_benchmark_results(name)
        
        if results:
            self.reporter.print_summary_table(results, f"Benchmark: {name}")
        else:
            print(f"Benchmark '{name}' not found")
    
    def clean_old_benchmarks(self, keep_latest: int = 10):
        """Clean old benchmark files, keeping only the latest N."""
        benchmark_files = sorted(
            self.results_dir.glob("*_results.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if len(benchmark_files) > keep_latest:
            to_delete = benchmark_files[keep_latest:]
            for file in to_delete:
                # Delete associated files
                base_name = file.stem.replace("_results", "")
                associated_files = [
                    self.results_dir / f"{base_name}.csv",
                    self.results_dir / f"{base_name}_throughput.png",
                    self.results_dir / f"{base_name}_latency.png"
                ]
                
                file.unlink()
                for assoc_file in associated_files:
                    if assoc_file.exists():
                        assoc_file.unlink()
                
                print(f"Deleted: {base_name}")


async def quick_benchmark(
    server_url: str,
    voice: str = "speecht5_tts",
    concurrency: List[int] = None,
    iterations: int = 10,
    name: Optional[str] = None
):
    """Run a quick benchmark with common settings."""
    if concurrency is None:
        concurrency = [1, 2, 4, 8, 16, 32, 64]
    
    print(f"Quick benchmark: {server_url}")
    print(f"Voice: {voice}, Concurrency: {concurrency}, Iterations: {iterations}")
    
    # Test connectivity
    config = TTSServerConfig(base_url=server_url)
    if not await test_server_connectivity(config):
        print("❌ Server not accessible")
        return
    
    # Create TTS function
    tts_function = create_torch_inference_tts_function(
        base_url=server_url,
        voice=voice
    )
    
    # Configure and run benchmark
    benchmark_config = BenchmarkConfig(
        concurrency_levels=concurrency,
        iterations_per_level=iterations,
        text_variations=15,
        output_dir="quick_benchmark_results"
    )
    
    harness = TTSBenchmarkHarness(benchmark_config)
    benchmark_name = name or f"quick_{voice}"
    
    results = await harness.run_benchmark_async(
        tts_function,
        benchmark_name=benchmark_name,
        is_async=True
    )
    
    # Print results
    reporter = TTSBenchmarkReporter()
    reporter.print_summary_table(results, f"Quick Benchmark ({voice})")
    
    return results


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="TTS Benchmark Management Utility")
    parser.add_argument("--results-dir", default="benchmark_results", help="Results directory")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List benchmarks
    list_parser = subparsers.add_parser("list", help="List available benchmarks")
    
    # Show benchmark summary
    show_parser = subparsers.add_parser("show", help="Show benchmark summary")
    show_parser.add_argument("name", help="Benchmark name")
    
    # Compare benchmarks
    compare_parser = subparsers.add_parser("compare", help="Compare benchmarks")
    compare_parser.add_argument("names", nargs="+", help="Benchmark names to compare")
    compare_parser.add_argument("--output", default="comparison", help="Output name")
    
    # Quick benchmark
    quick_parser = subparsers.add_parser("quick", help="Run quick benchmark")
    quick_parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    quick_parser.add_argument("--voice", default="speecht5_tts", help="Voice model")
    quick_parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64], help="Concurrency levels")
    quick_parser.add_argument("--iterations", type=int, default=100, help="Iterations")
    quick_parser.add_argument("--name", help="Benchmark name")
    
    # Clean old benchmarks
    clean_parser = subparsers.add_parser("clean", help="Clean old benchmark files")
    clean_parser.add_argument("--keep", type=int, default=10, help="Number of benchmarks to keep")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    manager = BenchmarkManager(args.results_dir)
    
    if args.command == "list":
        benchmarks = manager.list_benchmarks()
        if benchmarks:
            print("Available benchmarks:")
            for name in sorted(benchmarks):
                print(f"  - {name}")
        else:
            print("No benchmarks found")
    
    elif args.command == "show":
        manager.print_benchmark_summary(args.name)
    
    elif args.command == "compare":
        manager.compare_benchmarks(args.names, args.output)
    
    elif args.command == "quick":
        asyncio.run(quick_benchmark(
            server_url=args.url,
            voice=args.voice,
            concurrency=args.concurrency,
            iterations=args.iterations,
            name=args.name
        ))
    
    elif args.command == "clean":
        manager.clean_old_benchmarks(args.keep)
        print(f"Cleaned old benchmarks, kept latest {args.keep}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

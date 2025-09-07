"""
TTS Benchmark Harness - Complete benchmarking system with test data generation and execution.

Provides a high-level interface for running comprehensive TTS benchmarks with
configurable test scenarios, data generation, and result analysis.
"""

import os
import json
import logging
import random
import time
from typing import List, Dict, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path

from .tts_benchmark import TTSBenchmarker, TTSBenchmarkResult
from .metrics import TTSMetrics, validate_metrics_consistency
from .reporter import TTSBenchmarkReporter

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for TTS benchmark runs."""
    # Test configuration
    concurrency_levels: List[int] = None
    iterations_per_level: int = 100
    warmup_requests: int = 3
    timeout_seconds: float = 30.0
    
    # Audio configuration
    sample_rate: int = 22050
    bit_depth: int = 16
    
    # Test data configuration
    min_text_length: int = 50
    max_text_length: int = 200
    text_variations: int = 20
    
    # Output configuration
    output_dir: str = "benchmark_results"
    save_raw_data: bool = True
    generate_plots: bool = True
    
    def __post_init__(self):
        if self.concurrency_levels is None:
            self.concurrency_levels = [1, 2, 4, 8, 16, 32, 64]


class TTSBenchmarkHarness:
    """
    High-level TTS benchmark harness providing complete benchmarking workflow.
    
    Features:
    - Automatic test data generation
    - Configurable benchmark scenarios
    - Result persistence and analysis
    - Performance regression detection
    - CSV and plot generation
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize benchmark harness.
        
        Args:
            config: Benchmark configuration (uses defaults if None)
        """
        self.config = config or BenchmarkConfig()
        self.benchmarker = TTSBenchmarker(
            sample_rate=self.config.sample_rate,
            bit_depth=self.config.bit_depth,
            warmup_requests=self.config.warmup_requests,
            timeout_seconds=self.config.timeout_seconds
        )
        self.reporter = TTSBenchmarkReporter()
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def generate_test_texts(
        self,
        count: Optional[int] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        custom_texts: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate test texts for TTS benchmarking.
        
        Args:
            count: Number of test texts to generate
            min_length: Minimum text length in characters
            max_length: Maximum text length in characters
            custom_texts: Optional list of custom test texts
            
        Returns:
            List of test texts
        """
        if custom_texts:
            return custom_texts
        
        count = count or self.config.text_variations
        min_length = min_length or self.config.min_text_length
        max_length = max_length or self.config.max_text_length
        
        # Common TTS test phrases and sentences
        base_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello world, this is a test of the text-to-speech system.",
            "Artificial intelligence is transforming the way we interact with technology.",
            "Speech synthesis has come a long way since its early days.",
            "Modern neural networks can generate incredibly realistic human speech.",
            "The weather today is sunny with a chance of scattered clouds.",
            "Machine learning algorithms continue to improve at an exponential rate.",
            "Natural language processing enables computers to understand human speech.",
            "Deep learning models require significant computational resources to train effectively.",
            "Voice assistants have become an integral part of many people's daily routines."
        ]
        
        # Generate variations by combining and extending base texts
        test_texts = []
        for i in range(count):
            if i < len(base_texts):
                text = base_texts[i]
            else:
                # Create variations by combining base texts
                text = random.choice(base_texts)
                if random.random() < 0.5:
                    text += " " + random.choice(base_texts)
            
            # Adjust length to meet requirements
            while len(text) < min_length:
                text += " " + random.choice(base_texts)
            
            if len(text) > max_length:
                # Truncate at word boundary
                words = text.split()
                truncated = []
                current_length = 0
                for word in words:
                    if current_length + len(word) + 1 > max_length:
                        break
                    truncated.append(word)
                    current_length += len(word) + 1
                text = " ".join(truncated)
            
            test_texts.append(text)
        
        logger.info(f"Generated {len(test_texts)} test texts "
                   f"(length range: {min([len(t) for t in test_texts])}-"
                   f"{max([len(t) for t in test_texts])} chars)")
        
        return test_texts
    
    def run_benchmark(
        self,
        tts_function: Callable,
        test_texts: Optional[List[str]] = None,
        benchmark_name: str = "tts_benchmark",
        is_async: bool = False,
        **tts_kwargs
    ) -> Dict[int, TTSBenchmarkResult]:
        """
        Run complete TTS benchmark with the specified function.
        
        Args:
            tts_function: TTS function to benchmark (sync or async)
            test_texts: Optional list of test texts (generated if None)
            benchmark_name: Name for this benchmark run
            is_async: Whether the TTS function is asynchronous
            **tts_kwargs: Additional arguments for TTS function
            
        Returns:
            Dictionary mapping concurrency levels to benchmark results
        """
        logger.info(f"Starting TTS benchmark: {benchmark_name}")
        
        # Generate test texts if not provided
        if test_texts is None:
            test_texts = self.generate_test_texts()
        
        # Run benchmark
        start_time = time.time()
        
        if is_async:
            import asyncio
            try:
                # Check if we're already in a running event loop
                loop = asyncio.get_running_loop()
                # We're in an event loop, so we need to await the coroutine
                # This requires making this method async, but for now we'll use a workaround
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    # Create a new event loop in a separate thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.benchmarker.benchmark_async_tts(
                                tts_function,
                                test_texts,
                                concurrency_levels=self.config.concurrency_levels,
                                iterations_per_level=self.config.iterations_per_level,
                                **tts_kwargs
                            )
                        )
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    results = future.result()
                    
            except RuntimeError:
                # No running event loop, safe to use asyncio.run()
                results = asyncio.run(
                    self.benchmarker.benchmark_async_tts(
                        tts_function,
                        test_texts,
                        concurrency_levels=self.config.concurrency_levels,
                        iterations_per_level=self.config.iterations_per_level,
                        **tts_kwargs
                    )
                )
        else:
            results = self.benchmarker.benchmark_sync_tts(
                tts_function,
                test_texts,
                concurrency_levels=self.config.concurrency_levels,
                iterations_per_level=self.config.iterations_per_level,
                **tts_kwargs
            )
        
        total_time = time.time() - start_time
        logger.info(f"Benchmark completed in {total_time:.1f} seconds")
        
        # Validate results
        self._validate_benchmark_results(results)
        
        # Save results
        if self.config.save_raw_data:
            self._save_benchmark_results(results, benchmark_name)
        
        # Generate reports
        self._generate_reports(results, benchmark_name, test_texts)
        
        return results
    
    async def run_benchmark_async(
        self,
        tts_function: Callable,
        test_texts: Optional[List[str]] = None,
        benchmark_name: str = "tts_benchmark",
        **tts_kwargs
    ) -> Dict[int, TTSBenchmarkResult]:
        """
        Run complete TTS benchmark with an async function (async version).
        
        Args:
            tts_function: Async TTS function to benchmark
            test_texts: Optional list of test texts (generated if None)
            benchmark_name: Name for this benchmark run
            **tts_kwargs: Additional arguments for TTS function
            
        Returns:
            Dictionary mapping concurrency levels to benchmark results
        """
        logger.info(f"Starting async TTS benchmark: {benchmark_name}")
        
        # Generate test texts if not provided
        if test_texts is None:
            test_texts = self.generate_test_texts()
        
        # Run benchmark
        start_time = time.time()
        
        results = await self.benchmarker.benchmark_async_tts(
            tts_function,
            test_texts,
            concurrency_levels=self.config.concurrency_levels,
            iterations_per_level=self.config.iterations_per_level,
            **tts_kwargs
        )
        
        total_time = time.time() - start_time
        logger.info(f"Async benchmark completed in {total_time:.1f} seconds")
        
        # Validate results
        self._validate_benchmark_results(results)
        
        # Save results
        if self.config.save_raw_data:
            self._save_benchmark_results(results, benchmark_name)
        
        # Generate reports
        self._generate_reports(results, benchmark_name, test_texts)
        
        return results
    
    def compare_benchmarks(
        self,
        benchmark_results: Dict[str, Dict[int, TTSBenchmarkResult]],
        output_name: str = "comparison"
    ) -> None:
        """
        Compare multiple benchmark results and generate comparison reports.
        
        Args:
            benchmark_results: Dictionary mapping benchmark names to results
            output_name: Name for comparison output files
        """
        logger.info(f"Generating benchmark comparison: {output_name}")
        
        # Generate comparison CSV
        comparison_csv = self.reporter.generate_comparison_csv(benchmark_results)
        csv_path = os.path.join(self.config.output_dir, f"{output_name}_comparison.csv")
        with open(csv_path, 'w') as f:
            f.write(comparison_csv)
        
        # Generate comparison plots if enabled
        if self.config.generate_plots:
            try:
                plot_path = os.path.join(self.config.output_dir, f"{output_name}_comparison.png")
                self.reporter.plot_benchmark_comparison(benchmark_results, plot_path)
                logger.info(f"Comparison plot saved to: {plot_path}")
            except Exception as e:
                logger.warning(f"Failed to generate comparison plot: {e}")
        
        logger.info(f"Comparison results saved to: {csv_path}")
    
    def load_benchmark_results(self, benchmark_name: str) -> Optional[Dict[int, TTSBenchmarkResult]]:
        """
        Load previously saved benchmark results.
        
        Args:
            benchmark_name: Name of the benchmark to load
            
        Returns:
            Benchmark results or None if not found
        """
        results_path = os.path.join(self.config.output_dir, f"{benchmark_name}_results.json")
        
        if not os.path.exists(results_path):
            logger.warning(f"Benchmark results not found: {results_path}")
            return None
        
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
            
            results = {}
            for concurrency_str, result_data in data.items():
                concurrency = int(concurrency_str)
                
                # Reconstruct TTSBenchmarkResult
                metrics_data = result_data['metrics']
                metrics = TTSMetrics(**metrics_data)
                
                request_metrics = []
                for req_data in result_data['request_metrics']:
                    from .metrics import TTSRequestMetrics
                    request_metrics.append(TTSRequestMetrics(**req_data))
                
                results[concurrency] = TTSBenchmarkResult(
                    metrics=metrics,
                    request_metrics=request_metrics,
                    config=result_data['config']
                )
            
            logger.info(f"Loaded benchmark results: {benchmark_name}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load benchmark results {benchmark_name}: {e}")
            return None
    
    def _validate_benchmark_results(self, results: Dict[int, TTSBenchmarkResult]) -> None:
        """Validate benchmark results and log any issues."""
        for concurrency, result in results.items():
            warnings = validate_metrics_consistency(result.metrics)
            if warnings:
                logger.warning(f"Validation issues for concurrency {concurrency}:")
                for warning in warnings:
                    logger.warning(f"  - {warning}")
    
    def _save_benchmark_results(
        self,
        results: Dict[int, TTSBenchmarkResult],
        benchmark_name: str
    ) -> None:
        """Save benchmark results to JSON file."""
        # Convert results to serializable format
        serializable_results = {}
        for concurrency, result in results.items():
            serializable_results[str(concurrency)] = {
                'metrics': asdict(result.metrics),
                'request_metrics': [asdict(req) for req in result.request_metrics],
                'config': result.config
            }
        
        results_path = os.path.join(self.config.output_dir, f"{benchmark_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to: {results_path}")
    
    def _generate_reports(
        self,
        results: Dict[int, TTSBenchmarkResult],
        benchmark_name: str,
        test_texts: List[str]
    ) -> None:
        """Generate CSV and plot reports."""
        # Generate CSV report
        csv_content = self.reporter.generate_csv_report(results, test_texts)
        csv_path = os.path.join(self.config.output_dir, f"{benchmark_name}.csv")
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        
        logger.info(f"CSV report saved to: {csv_path}")
        
        # Generate plots if enabled
        if self.config.generate_plots:
            try:
                plot_path = os.path.join(self.config.output_dir, f"{benchmark_name}_throughput.png")
                self.reporter.plot_throughput_curve(results, plot_path)
                logger.info(f"Throughput plot saved to: {plot_path}")
                
                latency_plot_path = os.path.join(self.config.output_dir, f"{benchmark_name}_latency.png")
                self.reporter.plot_latency_metrics(results, latency_plot_path)
                logger.info(f"Latency plot saved to: {latency_plot_path}")
                
            except Exception as e:
                logger.warning(f"Failed to generate plots: {e}")


def create_demo_tts_function(
    min_audio_duration: float = 1.0,
    max_audio_duration: float = 5.0,
    processing_delay: float = 0.1,
    failure_rate: float = 0.0
) -> Callable[[str], Dict[str, Any]]:
    """
    Create a demo TTS function for testing the benchmark system.
    
    Args:
        min_audio_duration: Minimum audio duration to simulate
        max_audio_duration: Maximum audio duration to simulate
        processing_delay: Simulated processing delay
        failure_rate: Probability of request failure (0.0 to 1.0)
        
    Returns:
        Demo TTS function
    """
    def demo_tts(text: str, **kwargs) -> Dict[str, Any]:
        # Simulate processing time
        if processing_delay > 0:
            time.sleep(processing_delay)
        
        # Simulate failures
        if random.random() < failure_rate:
            raise RuntimeError("Simulated TTS failure")
        
        # Simulate audio duration based on text length
        base_duration = len(text) * 0.05  # ~50ms per character
        duration = max(min_audio_duration, min(base_duration, max_audio_duration))
        
        return {
            'audio_duration': duration,
            'sample_rate': kwargs.get('sample_rate', 22050),
            'text_tokens': len(text.split())
        }
    
    return demo_tts


def run_demo_benchmark():
    """Run a demonstration benchmark with synthetic TTS function."""
    # Create demo TTS function
    demo_tts = create_demo_tts_function(
        min_audio_duration=0.5,
        max_audio_duration=3.0,
        processing_delay=0.05,
        failure_rate=0.02
    )
    
    # Configure benchmark
    config = BenchmarkConfig(
        concurrency_levels=[1, 2, 4, 8, 16, 32, 64],
        iterations_per_level=100,
        text_variations=10,
        output_dir="demo_benchmark_results"
    )
    
    # Run benchmark
    harness = TTSBenchmarkHarness(config)
    results = harness.run_benchmark(
        demo_tts,
        benchmark_name="demo_tts_benchmark"
    )
    
    # Print summary
    print("\nDemo Benchmark Results Summary:")
    print("-" * 50)
    for concurrency, result in results.items():
        metrics = result.metrics
        print(f"Concurrency {concurrency}:")
        print(f"  ASPS: {metrics.asps:.3f}")
        print(f"  RTF (median): {metrics.rtf_median:.3f}")
        print(f"  RPS: {metrics.rps:.1f}")
        print(f"  TTFA p95: {metrics.ttfa_p95*1000:.1f}ms")
        print(f"  Success rate: {metrics.success_rate:.1f}%")
        print()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    run_demo_benchmark()

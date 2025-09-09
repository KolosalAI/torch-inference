"""
Benchmark Harness - Complete benchmarking system for TTS and Image models.

Provides a high-level interface for running comprehensive benchmarks with
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

try:
    import torch
    from torch import Tensor, device as TorchDevice, dtype as TorchDtype
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    # Create dummy types for type hints when torch is not available
    class Tensor: pass
    class TorchDevice: pass
    class TorchDtype: pass

from .tts_benchmark import TTSBenchmarker, TTSBenchmarkResult
from .metrics import TTSMetrics, validate_metrics_consistency
from .reporter import TTSBenchmarkReporter
from .image_benchmark import ImageBenchmarker, ImageBenchmarkResult
from .image_metrics import ImageMetrics
from .image_reporter import ImageBenchmarkReporter

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for TTS and Image benchmark runs."""
    # Model type configuration
    model_type: str = "tts"  # "tts" or "image"
    
    # Test configuration
    concurrency_levels: List[int] = None
    iterations_per_level: int = 100
    warmup_requests: int = 3
    timeout_seconds: float = 30.0
    
    # Audio configuration (for TTS models)
    sample_rate: int = 22050
    bit_depth: int = 16
    
    # Image configuration (for Image models)
    image_width: int = 512
    image_height: int = 512
    num_images: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    
    # Test data configuration
    min_text_length: int = 50
    max_text_length: int = 200
    text_variations: int = 20
    
    # Output configuration
    output_dir: str = "benchmark_results"
    save_raw_data: bool = True
    generate_plots: bool = True
    generate_detailed_csv: bool = True  # Generate detailed CSV with individual request data
    generate_summary_csv: bool = True   # Generate summary CSV with aggregated statistics
    
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
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            f.write(comparison_csv)
        
        # Generate detailed comparison CSV if enabled
        if self.config.save_raw_data:
            try:
                detailed_comparison_csv = self.reporter.generate_detailed_comparison_csv(benchmark_results)
                detailed_csv_path = os.path.join(self.config.output_dir, f"{output_name}_detailed_comparison.csv")
                with open(detailed_csv_path, 'w', newline='', encoding='utf-8') as f:
                    f.write(detailed_comparison_csv)
                logger.info(f"Detailed comparison CSV saved to: {detailed_csv_path}")
            except Exception as e:
                logger.warning(f"Failed to generate detailed comparison CSV: {e}")
        
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
        # Generate detailed CSV report with individual request data
        if self.config.generate_detailed_csv:
            detailed_csv_content = self.reporter.generate_detailed_csv_report(results, test_texts)
            detailed_csv_path = os.path.join(self.config.output_dir, f"{benchmark_name}_detailed.csv")
            with open(detailed_csv_path, 'w', newline='', encoding='utf-8') as f:
                f.write(detailed_csv_content)
            
            logger.info(f"Detailed CSV report saved to: {detailed_csv_path}")
        
        # Generate summary CSV report with aggregated statistics
        if self.config.generate_summary_csv:
            summary_csv_content = self.reporter.generate_summary_csv_report(results, test_texts)
            summary_csv_path = os.path.join(self.config.output_dir, f"{benchmark_name}_summary.csv")
            with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
                f.write(summary_csv_content)
            
            logger.info(f"Summary CSV report saved to: {summary_csv_path}")
        
        # Generate legacy CSV report for backward compatibility
        legacy_csv_content = self.reporter.generate_csv_report(results, test_texts)
        legacy_csv_path = os.path.join(self.config.output_dir, f"{benchmark_name}.csv")
        with open(legacy_csv_path, 'w', newline='', encoding='utf-8') as f:
            f.write(legacy_csv_content)
        
        logger.info(f"Legacy CSV report saved to: {legacy_csv_path}")
        
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


def generate_image_prompts(
    count: int = 20,
    min_length: int = 20,
    max_length: int = 100,
    custom_prompts: Optional[List[str]] = None
) -> List[str]:
    """
    Generate test prompts for image model benchmarking.
    
    Args:
        count: Number of test prompts to generate
        min_length: Minimum prompt length in characters
        max_length: Maximum prompt length in characters
        custom_prompts: Optional list of custom test prompts
        
    Returns:
        List of image generation prompts
    """
    if custom_prompts:
        return custom_prompts
    
    # Common image generation test prompts
    base_prompts = [
        "A beautiful landscape with mountains and a lake",
        "A cute cat sitting on a windowsill",
        "A futuristic city skyline at sunset",
        "A peaceful forest with sunlight filtering through trees",
        "A vintage car parked on a cobblestone street",
        "A colorful abstract painting with geometric shapes",
        "A modern kitchen with sleek appliances",
        "A cozy living room with a fireplace",
        "A bustling marketplace in an ancient city",
        "A serene beach with crystal clear water",
        "A majestic eagle soaring through the sky",
        "A field of sunflowers under a blue sky",
        "A steampunk-inspired mechanical robot",
        "A magical fantasy castle on a hilltop",
        "A portrait of a wise old wizard",
        "A cyberpunk street scene with neon lights",
        "A minimalist bedroom with white walls",
        "A delicious meal on a wooden table",
        "A snow-covered mountain peak",
        "A tropical paradise with palm trees"
    ]
    
    # Style modifiers to add variety
    style_modifiers = [
        "in the style of Van Gogh",
        "photorealistic, highly detailed",
        "digital art, concept art",
        "oil painting, classical",
        "watercolor painting",
        "pencil sketch, black and white",
        "anime style, studio ghibli",
        "cartoon style, pixar",
        "impressionist painting",
        "art nouveau style",
        "surreal, dreamlike",
        "minimalist, clean lines",
        "vintage photography",
        "macro photography",
        "cinematic lighting"
    ]
    
    # Generate variations
    test_prompts = []
    for i in range(count):
        if i < len(base_prompts):
            prompt = base_prompts[i]
        else:
            prompt = random.choice(base_prompts)
        
        # Add style modifier sometimes
        if random.random() < 0.7:
            style = random.choice(style_modifiers)
            prompt += f", {style}"
        
        # Adjust length to meet requirements
        while len(prompt) < min_length:
            additional_style = random.choice(style_modifiers)
            prompt += f", {additional_style}"
        
        # Trim if too long
        if len(prompt) > max_length:
            prompt = prompt[:max_length].rsplit(',', 1)[0]
        
        test_prompts.append(prompt)
    
    return test_prompts


class ImageBenchmarkHarness:
    """
    High-level Image benchmark harness providing complete benchmarking workflow.
    
    Features:
    - Automatic test prompt generation
    - Configurable benchmark scenarios
    - Result persistence and analysis
    - Performance regression detection
    - CSV and plot generation
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize image benchmark harness.
        
        Args:
            config: Benchmark configuration (uses defaults if None)
        """
        self.config = config or BenchmarkConfig(model_type="image")
        self.benchmarker = ImageBenchmarker(
            warmup_requests=self.config.warmup_requests,
            timeout_seconds=self.config.timeout_seconds
        )
        self.reporter = ImageBenchmarkReporter()
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def generate_test_prompts(
        self,
        count: Optional[int] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        custom_prompts: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate test prompts for image benchmarking.
        
        Args:
            count: Number of test prompts to generate
            min_length: Minimum prompt length in characters
            max_length: Maximum prompt length in characters
            custom_prompts: Optional list of custom test prompts
            
        Returns:
            List of test prompts
        """
        if custom_prompts:
            return custom_prompts
        
        count = count or self.config.text_variations
        min_length = min_length or self.config.min_text_length
        max_length = max_length or self.config.max_text_length
        
        return generate_image_prompts(count, min_length, max_length)
    
    def run_benchmark(
        self,
        image_model_func: Callable,
        benchmark_name: str = "image_benchmark",
        test_prompts: Optional[List[str]] = None,
        image_params: Optional[Dict[str, Any]] = None
    ) -> Dict[int, ImageBenchmarkResult]:
        """
        Run complete image benchmark with specified configuration.
        
        Args:
            image_model_func: Function that takes (prompt, **kwargs) and returns image data
            benchmark_name: Name for this benchmark run
            test_prompts: Optional custom test prompts
            image_params: Optional image generation parameters
            
        Returns:
            Dictionary mapping concurrency levels to benchmark results
        """
        logger.info(f"Starting image benchmark: {benchmark_name}")
        logger.info(f"Configuration: {self.config}")
        
        # Generate test prompts
        if test_prompts is None:
            test_prompts = self.generate_test_prompts()
        
        logger.info(f"Generated {len(test_prompts)} test prompts")
        
        # Set up image generation parameters
        if image_params is None:
            image_params = {
                'width': self.config.image_width,
                'height': self.config.image_height,
                'num_images': self.config.num_images,
                'num_inference_steps': self.config.num_inference_steps,
                'guidance_scale': self.config.guidance_scale
            }
        
        # Run benchmarks for each concurrency level
        results = {}
        for concurrency in self.config.concurrency_levels:
            logger.info(f"Running benchmark with concurrency {concurrency}")
            
            try:
                result = self.benchmarker.benchmark_single_concurrency(
                    image_model_func,
                    test_prompts,
                    concurrency,
                    self.config.iterations_per_level,
                    **image_params
                )
                results[concurrency] = result
                
                # Log progress
                metrics = result.metrics
                logger.info(f"Concurrency {concurrency}: IPS={metrics.ips:.3f}, "
                          f"PPS={metrics.pps:.0f}, RPS={metrics.rps:.1f}, "
                          f"TTFI_p95={metrics.ttfi_p95*1000:.1f}ms")
                
            except Exception as e:
                logger.error(f"Benchmark failed for concurrency {concurrency}: {e}")
                continue
        
        if not results:
            raise RuntimeError("All benchmark runs failed")
        
        # Generate reports
        self._generate_reports(results, benchmark_name, test_prompts, image_params)
        
        logger.info(f"Image benchmark '{benchmark_name}' completed successfully")
        return results
    
    def _generate_reports(
        self,
        results: Dict[int, ImageBenchmarkResult],
        benchmark_name: str,
        test_prompts: List[str],
        image_params: Dict[str, Any]
    ) -> None:
        """Generate all configured reports and save to files."""
        
        # Generate detailed CSV if enabled
        if self.config.generate_detailed_csv:
            detailed_csv = self.reporter.generate_detailed_csv_report(results, test_prompts)
            detailed_path = os.path.join(self.config.output_dir, f"{benchmark_name}_detailed.csv")
            with open(detailed_path, 'w', newline='', encoding='utf-8') as f:
                f.write(detailed_csv)
            logger.info(f"Generated detailed CSV: {detailed_path}")
        
        # Generate summary CSV if enabled
        if self.config.generate_summary_csv:
            summary_csv = self.reporter.generate_summary_csv_report(results, test_prompts)
            summary_path = os.path.join(self.config.output_dir, f"{benchmark_name}_summary.csv")
            with open(summary_path, 'w', newline='', encoding='utf-8') as f:
                f.write(summary_csv)
            logger.info(f"Generated summary CSV: {summary_path}")
        
        # Generate plots if enabled
        if self.config.generate_plots:
            try:
                # Throughput curve plot
                throughput_path = os.path.join(self.config.output_dir, f"{benchmark_name}_throughput.png")
                self.reporter.plot_throughput_curve(results, throughput_path, f"{benchmark_name} - Throughput")
                
                # Latency metrics plot
                latency_path = os.path.join(self.config.output_dir, f"{benchmark_name}_latency.png")
                self.reporter.plot_latency_metrics(results, latency_path, f"{benchmark_name} - Latency")
                
                logger.info(f"Generated plots: {throughput_path}, {latency_path}")
            except Exception as e:
                logger.warning(f"Plot generation failed: {e}")
        
        # Save raw benchmark data if enabled
        if self.config.save_raw_data:
            raw_data = {
                "config": asdict(self.config),
                "test_prompts": test_prompts,
                "image_params": image_params,
                "results": {str(k): asdict(v) for k, v in results.items()}
            }
            raw_path = os.path.join(self.config.output_dir, f"{benchmark_name}_raw.json")
            with open(raw_path, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, indent=2, default=str)
            logger.info(f"Saved raw data: {raw_path}")
        
        # Print summary to console
        self.reporter.print_summary_table(results, benchmark_name)


# Demo image model function
def demo_image_model(
    prompt: str,
    width: int = 512,
    height: int = 512,
    num_images: int = 1,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    **kwargs
) -> Dict[str, Any]:
    """
    Demo image generation function for testing.
    
    Simulates image generation with realistic timing and GPU memory usage patterns.
    Returns a dictionary with image metadata in the expected format.
    """
    import torch
    import time
    
    # Determine device to use - prefer GPU for maximum performance
    if TORCH_AVAILABLE and torch:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logger.info(f"Using GPU for demo image generation: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using Apple MPS for demo image generation")
        else:
            device = torch.device('cpu')
            logger.warning("No GPU available - falling back to CPU (performance will be degraded)")
    else:
        device = 'cpu'  # Fallback when torch is not available
        logger.warning("PyTorch not available - using CPU fallback")
    
    start_time = time.perf_counter()
    
    try:
        # Simulate realistic image generation processing on GPU
        if TORCH_AVAILABLE and torch and hasattr(device, 'type') and device.type == 'cuda':
            _simulate_diffusion_gpu_processing(device, width, height, num_inference_steps, guidance_scale)
        else:
            # For CPU or when torch is unavailable, simulate with minimal delay
            processing_time = 0.05 * num_inference_steps * (width * height) / (512 * 512)
            time.sleep(processing_time)
        
        # Generate dummy image data for each requested image
        images = []
        for i in range(num_images):
            image_size = width * height * 3  # RGB
            dummy_data = b'dummy_image_data' * (image_size // 16)
            images.append(dummy_data)
        
        total_time = time.perf_counter() - start_time
        
        # Return in expected format
        return {
            'images': images,
            'width': width,
            'height': height,
            'num_images': num_images,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'file_size_bytes': len(images[0]) if images else 0,
            'image_format': 'PNG',
            'processing_time': total_time,
            'device': str(device)
        }
        
    except Exception as e:
        # Fallback to simple timing simulation
        base_time = 0.5
        complexity_factor = (width * height) / (512 * 512)
        steps_factor = num_inference_steps / 50
        total_time = base_time * complexity_factor * steps_factor * num_images
        actual_time = total_time * (0.8 + random.random() * 0.4)
        time.sleep(actual_time)
        
        images = []
        for i in range(num_images):
            image_size = width * height * 3
            dummy_data = b'dummy_image_data' * (image_size // 16)
            images.append(dummy_data)
        
        return {
            'images': images,
            'width': width,
            'height': height,
            'num_images': num_images,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'file_size_bytes': len(images[0]) if images else 0,
            'image_format': 'PNG',
            'error': str(e),
            'device': str(device)
        }


def _simulate_diffusion_gpu_processing(device: TorchDevice, width: int, height: int, 
                                     num_inference_steps: int, guidance_scale: float) -> None:
    """Simulate realistic diffusion model GPU processing."""
    if not TORCH_AVAILABLE or not torch:
        # Fallback when torch is not available
        time.sleep(0.01 * num_inference_steps)
        return
        
    try:
        with torch.no_grad():
            # Use appropriate dtype for the device
            dtype = torch.float16 if device.type == 'cuda' else torch.float32
            
            # Simulate latent space (typical for diffusion models)
            latent_height, latent_width = height // 8, width // 8
            latents = torch.randn(1, 4, latent_height, latent_width, device=device, dtype=dtype)
            
            # Simulate text embeddings 
            text_embeddings = torch.randn(1, 77, 768, device=device, dtype=dtype)
            uncond_embeddings = torch.randn(1, 77, 768, device=device, dtype=dtype)
            
            # Simulate U-Net processing for each diffusion step
            for step in range(min(num_inference_steps, 20)):  # Limit steps for demo
                # Simulate timestep embedding
                timestep = torch.tensor([step], device=device)
                
                # Simulate conditional and unconditional predictions
                cond_pred = _simulate_unet_forward(latents, timestep, text_embeddings, device, dtype)
                uncond_pred = _simulate_unet_forward(latents, timestep, uncond_embeddings, device, dtype)
                
                # Simulate classifier-free guidance
                guidance_factor = torch.tensor(guidance_scale, device=device, dtype=dtype)
                noise_pred = uncond_pred + guidance_factor * (cond_pred - uncond_pred)
                
                # Simulate scheduler step (update latents)
                alpha = 0.99 - (step / num_inference_steps) * 0.98
                latents = latents - 0.02 * noise_pred * alpha
                
                # Simulate some memory allocation/deallocation
                temp_tensor = torch.randn_like(latents)
                del temp_tensor
                
                # Synchronize for realistic timing
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            
            # Simulate VAE decoding (latents to image)
            decoded_image = _simulate_vae_decode(latents, device, dtype, width, height)
            
            # Cleanup
            del latents, text_embeddings, uncond_embeddings, decoded_image
            
            # Force GPU memory cleanup
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
    except Exception as e:
        # If GPU processing fails, fallback to simple delay
        import time
        time.sleep(0.01 * num_inference_steps)


def _simulate_unet_forward(latents: Tensor, timestep: Tensor, 
                          text_embeddings: Tensor, device: TorchDevice, dtype: TorchDtype) -> Tensor:
    """Simulate U-Net forward pass."""
    batch_size, channels, height, width = latents.shape
    
    # Simulate down-sampling path
    x = latents
    skip_connections = []
    
    # Simulate encoder blocks
    for i in range(4):  # 4 down-sampling blocks
        # Simulate convolution + attention
        x = torch.nn.functional.conv2d(
            x, 
            torch.randn(channels * 2, channels, 3, 3, device=device, dtype=dtype), 
            padding=1
        )
        x = torch.nn.functional.gelu(x)
        
        # Simulate cross-attention with text embeddings
        if i % 2 == 0:  # Every other block has cross-attention
            attn_output = _simulate_cross_attention(x, text_embeddings, device, dtype)
            x = x + attn_output
        
        skip_connections.append(x.clone())
        
        # Down-sample
        if i < 3:
            x = torch.nn.functional.avg_pool2d(x, 2)
        
        channels *= 2
    
    # Simulate bottleneck
    x = torch.nn.functional.conv2d(
        x,
        torch.randn(channels, channels, 3, 3, device=device, dtype=dtype),
        padding=1
    )
    
    # Simulate up-sampling path
    for i in range(4):
        # Up-sample
        if i > 0:
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        
        # Concatenate skip connection
        skip = skip_connections[-(i+1)]
        if x.shape != skip.shape:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        
        channels = x.size(1)
        
        # Simulate convolution
        x = torch.nn.functional.conv2d(
            x,
            torch.randn(channels // 2, channels, 3, 3, device=device, dtype=dtype),
            padding=1
        )
        x = torch.nn.functional.gelu(x)
    
    # Final output layer
    output = torch.nn.functional.conv2d(
        x,
        torch.randn(latents.size(1), x.size(1), 3, 3, device=device, dtype=dtype),
        padding=1
    )
    
    return output


def _simulate_cross_attention(x: Tensor, text_embeddings: Tensor, 
                            device: TorchDevice, dtype: TorchDtype) -> Tensor:
    """Simulate cross-attention mechanism."""
    batch_size, channels, height, width = x.shape
    seq_len = text_embeddings.size(1)
    embed_dim = text_embeddings.size(2)
    
    # Flatten spatial dimensions for attention
    x_flat = x.view(batch_size, channels, height * width).transpose(1, 2)
    
    # Simulate query, key, value projections
    q = torch.nn.functional.linear(x_flat, torch.randn(channels, embed_dim, device=device, dtype=dtype))
    k = text_embeddings
    v = text_embeddings
    
    # Simulate scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / (embed_dim ** 0.5)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    
    # Project back to original dimensions
    attn_output = torch.nn.functional.linear(
        attn_output, 
        torch.randn(embed_dim, channels, device=device, dtype=dtype)
    )
    
    # Reshape back to spatial format
    attn_output = attn_output.transpose(1, 2).view(batch_size, channels, height, width)
    
    return attn_output


def _simulate_vae_decode(latents: Tensor, device: TorchDevice, dtype: TorchDtype,
                        target_width: int, target_height: int) -> Tensor:
    """Simulate VAE decoder (latents to RGB image)."""
    batch_size, channels, height, width = latents.shape
    
    x = latents
    
    # Simulate up-sampling decoder blocks
    for i in range(3):  # 3 up-sampling blocks (8x total up-sampling)
        # Up-sample by 2x
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        
        # Simulate convolution
        out_channels = max(channels // 2, 3) if i < 2 else 3
        x = torch.nn.functional.conv2d(
            x,
            torch.randn(out_channels, channels, 3, 3, device=device, dtype=dtype),
            padding=1
        )
        
        if i < 2:
            x = torch.nn.functional.gelu(x)
            channels = out_channels
        else:
            # Final layer - sigmoid activation for RGB output
            x = torch.sigmoid(x)
    
    # Ensure correct output size
    if x.shape[-2:] != (target_height, target_width):
        x = torch.nn.functional.interpolate(x, size=(target_height, target_width), mode='bilinear', align_corners=False)
    
    return x


def run_demo_image_benchmark():
    """Run a demo image benchmark to showcase the system."""
    
    # Configure benchmark
    config = BenchmarkConfig(
        model_type="image",
        concurrency_levels=[1, 2, 4, 8],
        iterations_per_level=20,
        text_variations=5,
        output_dir="demo_image_benchmark_results",
        generate_detailed_csv=True,
        generate_summary_csv=True,
        generate_plots=True,
        image_width=512,
        image_height=512,
        num_images=1,
        num_inference_steps=20,
        guidance_scale=7.5
    )
    
    # Run benchmark
    harness = ImageBenchmarkHarness(config)
    results = harness.run_benchmark(
        demo_image_model,
        benchmark_name="demo_image_benchmark"
    )
    
    # Print summary
    print("\nDemo Image Benchmark Results Summary:")
    print("-" * 50)
    for concurrency, result in results.items():
        metrics = result.metrics
        print(f"Concurrency {concurrency}:")
        print(f"  IPS: {metrics.ips:.3f}")
        print(f"  PPS: {metrics.pps:.0f}")
        print(f"  RPS: {metrics.rps:.1f}")
        print(f"  TTFI p95: {metrics.ttfi_p95*1000:.1f}ms")
        print(f"  Success rate: {metrics.success_rate:.1f}%")
        print()
    
    print("CSV Files Generated:")
    print(f"  - Detailed CSV: {config.output_dir}/demo_image_benchmark_detailed.csv")
    print(f"  - Summary CSV: {config.output_dir}/demo_image_benchmark_summary.csv")


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
    
    # Configure benchmark with enhanced CSV output
    config = BenchmarkConfig(
        concurrency_levels=[1, 2, 4, 8, 16, 32, 64],
        iterations_per_level=100,
        text_variations=10,
        output_dir="demo_benchmark_results",
        generate_detailed_csv=True,   # Generate detailed CSV with individual request data
        generate_summary_csv=True,    # Generate summary CSV with aggregated statistics
        generate_plots=True
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
    
    print("CSV Files Generated:")
    print(f"  - Detailed CSV: {config.output_dir}/demo_tts_benchmark_detailed.csv")
    print(f"  - Summary CSV: {config.output_dir}/demo_tts_benchmark_summary.csv")
    print(f"  - Legacy CSV: {config.output_dir}/demo_tts_benchmark.csv")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "image":
        print("Running image model benchmark demo...")
        run_demo_image_benchmark()
    else:
        print("Running TTS model benchmark demo...")
        run_demo_benchmark()
        print("\nTo run image benchmark demo, use: python harness.py image")

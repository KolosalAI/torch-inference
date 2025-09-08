"""
Image Model Benchmark Metrics Module.

Implements the core image model benchmarking metrics including:
- IPS (Images Per Second) - the primary throughput metric
- TTFI (Time To First Image) - latency metrics for streaming/progressive generation
- RPS (Requests Per Second)
- PPS (Pixels Per Second)
- Memory usage metrics
"""

import time
import statistics
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImageRequestMetrics:
    """Metrics captured for a single image generation request."""
    request_id: str
    t_start: float  # Request start time
    t_first_image: Optional[float] = None  # Time when first image/chunk was generated
    t_end: Optional[float] = None  # Request completion time
    
    # Input characteristics
    prompt_len_chars: int = 0  # Length of input prompt in characters
    prompt_len_tokens: int = 0  # Length of input prompt in tokens (if available)
    negative_prompt_len_chars: int = 0  # Length of negative prompt
    
    # Image generation parameters
    width: int = 512  # Generated image width
    height: int = 512  # Generated image height
    num_inference_steps: int = 20  # Number of denoising steps
    guidance_scale: float = 7.5  # CFG scale
    seed: Optional[int] = None  # Random seed used
    
    # Output characteristics
    num_images: int = 1  # Number of images generated
    image_format: str = "PNG"  # Output image format
    file_size_bytes: int = 0  # Total size of generated images
    
    # Performance metrics
    memory_peak_mb: float = 0.0  # Peak memory usage in MB
    gpu_memory_mb: float = 0.0  # GPU memory usage in MB
    
    error: Optional[str] = None  # Error message if request failed
    
    @property
    def wall_time(self) -> Optional[float]:
        """Total wall-clock time for this request."""
        if self.t_end is not None:
            return self.t_end - self.t_start
        return None
    
    @property
    def ttfi(self) -> Optional[float]:
        """Time to first image (latency metric)."""
        if self.t_first_image is not None:
            return self.t_first_image - self.t_start
        return None
    
    @property
    def total_pixels(self) -> int:
        """Total pixels generated (width × height × num_images)."""
        return self.width * self.height * self.num_images
    
    @property
    def pixels_per_second(self) -> Optional[float]:
        """Pixels generated per second."""
        if self.wall_time is not None and self.wall_time > 0:
            return self.total_pixels / self.wall_time
        return None
    
    @property
    def images_per_second(self) -> Optional[float]:
        """Images generated per second."""
        if self.wall_time is not None and self.wall_time > 0:
            return self.num_images / self.wall_time
        return None
    
    @property
    def steps_per_second(self) -> Optional[float]:
        """Inference steps per second."""
        if self.wall_time is not None and self.wall_time > 0:
            return self.num_inference_steps / self.wall_time
        return None


@dataclass
class ImageMetrics:
    """Aggregated image model benchmark metrics for a complete run."""
    # Primary throughput metrics
    ips: float = 0.0  # Images Per Second
    pps: float = 0.0  # Pixels Per Second
    sps: float = 0.0  # Steps Per Second (inference steps)
    
    # Related throughput metrics
    rps: float = 0.0  # Requests Per Second
    cps: float = 0.0  # Characters Per Second (prompt processing)
    tokens_per_sec: float = 0.0  # Tokens Per Second (if available)
    
    # Latency metrics (TTFI - Time To First Image)
    ttfi_p50: float = 0.0  # 50th percentile TTFI
    ttfi_p95: float = 0.0  # 95th percentile TTFI
    ttfi_p99: float = 0.0  # 99th percentile TTFI
    ttfi_mean: float = 0.0  # Mean TTFI
    
    # Wall-clock timing
    total_wall_time: float = 0.0  # Total benchmark wall time
    min_request_start: float = 0.0  # Earliest request start time
    max_request_end: float = 0.0  # Latest request end time
    
    # Request statistics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Image statistics
    total_images: int = 0  # Sum of all generated images
    total_pixels: int = 0  # Sum of all generated pixels
    total_inference_steps: int = 0  # Sum of all inference steps
    total_prompt_chars: int = 0  # Sum of all prompt characters
    total_tokens: int = 0  # Sum of all prompt tokens (if available)
    
    # Memory statistics
    avg_memory_peak_mb: float = 0.0  # Average peak memory usage
    max_memory_peak_mb: float = 0.0  # Maximum peak memory usage
    avg_gpu_memory_mb: float = 0.0  # Average GPU memory usage
    max_gpu_memory_mb: float = 0.0  # Maximum GPU memory usage
    
    # Image properties
    avg_image_width: float = 0.0
    avg_image_height: float = 0.0
    avg_inference_steps: float = 0.0
    avg_guidance_scale: float = 0.0
    
    # Configuration
    concurrency_level: int = 1
    
    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    @property
    def avg_prompt_length(self) -> float:
        """Average prompt length in characters."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_prompt_chars / self.successful_requests
    
    @property
    def avg_images_per_request(self) -> float:
        """Average number of images per request."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_images / self.successful_requests


def compute_ips(request_metrics: List[ImageRequestMetrics]) -> float:
    """
    Compute IPS (Images Per Second) - the primary image generation throughput metric.
    
    IPS = sum(num_images_i) / T_wall
    
    Where:
    - num_images_i = number of images generated in the i-th request
    - T_wall = wall-clock time from first request start to last image completion
    
    Higher IPS is better.
    """
    if not request_metrics:
        return 0.0
    
    # Filter successful requests
    successful_requests = [r for r in request_metrics if r.error is None and r.t_end is not None]
    
    if not successful_requests:
        return 0.0
    
    # Calculate total images
    total_images = sum(r.num_images for r in successful_requests)
    
    # Calculate wall-clock time
    min_start = min(r.t_start for r in successful_requests)
    max_end = max(r.t_end for r in successful_requests)
    wall_time = max_end - min_start
    
    if wall_time <= 0:
        return 0.0
    
    return total_images / wall_time


def compute_pps(request_metrics: List[ImageRequestMetrics]) -> float:
    """
    Compute PPS (Pixels Per Second).
    
    PPS = sum(total_pixels_i) / T_wall
    
    Useful for comparing performance across different image resolutions.
    """
    if not request_metrics:
        return 0.0
    
    successful_requests = [r for r in request_metrics if r.error is None and r.t_end is not None]
    
    if not successful_requests:
        return 0.0
    
    # Calculate total pixels
    total_pixels = sum(r.total_pixels for r in successful_requests)
    
    # Calculate wall-clock time
    min_start = min(r.t_start for r in successful_requests)
    max_end = max(r.t_end for r in successful_requests)
    wall_time = max_end - min_start
    
    if wall_time <= 0:
        return 0.0
    
    return total_pixels / wall_time


def compute_sps(request_metrics: List[ImageRequestMetrics]) -> float:
    """
    Compute SPS (Steps Per Second) - inference steps throughput.
    
    SPS = sum(num_inference_steps_i) / T_wall
    
    Useful for comparing efficiency of different sampling methods.
    """
    if not request_metrics:
        return 0.0
    
    successful_requests = [r for r in request_metrics if r.error is None and r.t_end is not None]
    
    if not successful_requests:
        return 0.0
    
    # Calculate total steps
    total_steps = sum(r.num_inference_steps for r in successful_requests)
    
    # Calculate wall-clock time
    min_start = min(r.t_start for r in successful_requests)
    max_end = max(r.t_end for r in successful_requests)
    wall_time = max_end - min_start
    
    if wall_time <= 0:
        return 0.0
    
    return total_steps / wall_time


def compute_image_rps(request_metrics: List[ImageRequestMetrics]) -> float:
    """
    Compute RPS (Requests Per Second) for image generation.
    
    RPS = N / T_wall
    """
    if not request_metrics:
        return 0.0
    
    successful_requests = [r for r in request_metrics if r.error is None and r.t_end is not None]
    
    if not successful_requests:
        return 0.0
    
    # Calculate wall-clock time
    min_start = min(r.t_start for r in successful_requests)
    max_end = max(r.t_end for r in successful_requests)
    wall_time = max_end - min_start
    
    if wall_time <= 0:
        return 0.0
    
    return len(successful_requests) / wall_time


def compute_image_cps(request_metrics: List[ImageRequestMetrics]) -> float:
    """
    Compute CPS (Characters Per Second) for prompt processing.
    
    CPS = sum(prompt_len_chars + negative_prompt_len_chars) / T_wall
    """
    if not request_metrics:
        return 0.0
    
    successful_requests = [r for r in request_metrics if r.error is None and r.t_end is not None]
    
    if not successful_requests:
        return 0.0
    
    # Calculate total characters
    total_chars = sum(r.prompt_len_chars + r.negative_prompt_len_chars for r in successful_requests)
    
    # Calculate wall-clock time
    min_start = min(r.t_start for r in successful_requests)
    max_end = max(r.t_end for r in successful_requests)
    wall_time = max_end - min_start
    
    if wall_time <= 0:
        return 0.0
    
    return total_chars / wall_time


def compute_image_tokens_per_sec(request_metrics: List[ImageRequestMetrics]) -> float:
    """
    Compute tokens per second for prompt processing.
    """
    if not request_metrics:
        return 0.0
    
    successful_requests = [r for r in request_metrics if r.error is None and r.t_end is not None]
    
    if not successful_requests:
        return 0.0
    
    # Calculate total tokens
    total_tokens = sum(r.prompt_len_tokens for r in successful_requests)
    
    if total_tokens == 0:
        return 0.0
    
    # Calculate wall-clock time
    min_start = min(r.t_start for r in successful_requests)
    max_end = max(r.t_end for r in successful_requests)
    wall_time = max_end - min_start
    
    if wall_time <= 0:
        return 0.0
    
    return total_tokens / wall_time


def compute_ttfi_statistics(request_metrics: List[ImageRequestMetrics]) -> Dict[str, float]:
    """
    Compute TTFI (Time To First Image) latency statistics.
    
    Returns p50, p95, p99, and mean TTFI values.
    Important for user experience in image generation.
    """
    if not request_metrics:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    
    ttfi_values = []
    for request in request_metrics:
        if request.ttfi is not None:
            ttfi_values.append(request.ttfi)
    
    if not ttfi_values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    
    ttfi_values.sort()
    n = len(ttfi_values)
    
    p50_idx = int(0.50 * n)
    p95_idx = int(0.95 * n)
    p99_idx = int(0.99 * n)
    
    return {
        "p50": ttfi_values[min(p50_idx, n - 1)],
        "p95": ttfi_values[min(p95_idx, n - 1)],
        "p99": ttfi_values[min(p99_idx, n - 1)],
        "mean": statistics.mean(ttfi_values)
    }


def aggregate_image_metrics(
    request_metrics: List[ImageRequestMetrics],
    concurrency_level: int = 1
) -> ImageMetrics:
    """
    Aggregate individual request metrics into overall image benchmark metrics.
    
    Args:
        request_metrics: List of individual request metrics
        concurrency_level: Concurrency level used in the benchmark
        
    Returns:
        ImageMetrics object with all computed metrics
    """
    if not request_metrics:
        return ImageMetrics()
    
    # Filter successful requests
    successful_requests = [r for r in request_metrics if r.error is None and r.t_end is not None]
    failed_requests = [r for r in request_metrics if r.error is not None]
    
    # Compute primary metrics
    ips = compute_ips(request_metrics)
    pps = compute_pps(request_metrics)
    sps = compute_sps(request_metrics)
    rps = compute_image_rps(request_metrics)
    cps = compute_image_cps(request_metrics)
    tokens_per_sec = compute_image_tokens_per_sec(request_metrics)
    
    # Compute TTFI statistics
    ttfi_stats = compute_ttfi_statistics(request_metrics)
    
    # Compute timing bounds
    if successful_requests:
        min_start = min(r.t_start for r in successful_requests)
        max_end = max(r.t_end for r in successful_requests)
        total_wall_time = max_end - min_start
        
        # Image and text statistics
        total_images = sum(r.num_images for r in successful_requests)
        total_pixels = sum(r.total_pixels for r in successful_requests)
        total_inference_steps = sum(r.num_inference_steps for r in successful_requests)
        total_prompt_chars = sum(r.prompt_len_chars + r.negative_prompt_len_chars for r in successful_requests)
        total_tokens = sum(r.prompt_len_tokens for r in successful_requests)
        
        # Memory statistics
        memory_peaks = [r.memory_peak_mb for r in successful_requests if r.memory_peak_mb > 0]
        gpu_memories = [r.gpu_memory_mb for r in successful_requests if r.gpu_memory_mb > 0]
        
        avg_memory_peak_mb = statistics.mean(memory_peaks) if memory_peaks else 0.0
        max_memory_peak_mb = max(memory_peaks) if memory_peaks else 0.0
        avg_gpu_memory_mb = statistics.mean(gpu_memories) if gpu_memories else 0.0
        max_gpu_memory_mb = max(gpu_memories) if gpu_memories else 0.0
        
        # Image properties
        avg_image_width = statistics.mean([r.width for r in successful_requests])
        avg_image_height = statistics.mean([r.height for r in successful_requests])
        avg_inference_steps = statistics.mean([r.num_inference_steps for r in successful_requests])
        avg_guidance_scale = statistics.mean([r.guidance_scale for r in successful_requests])
    else:
        min_start = max_end = total_wall_time = 0.0
        total_images = total_pixels = total_inference_steps = total_prompt_chars = total_tokens = 0
        avg_memory_peak_mb = max_memory_peak_mb = avg_gpu_memory_mb = max_gpu_memory_mb = 0.0
        avg_image_width = avg_image_height = avg_inference_steps = avg_guidance_scale = 0.0
    
    return ImageMetrics(
        ips=ips,
        pps=pps,
        sps=sps,
        rps=rps,
        cps=cps,
        tokens_per_sec=tokens_per_sec,
        ttfi_p50=ttfi_stats["p50"],
        ttfi_p95=ttfi_stats["p95"],
        ttfi_p99=ttfi_stats["p99"],
        ttfi_mean=ttfi_stats["mean"],
        total_wall_time=total_wall_time,
        min_request_start=min_start,
        max_request_end=max_end,
        total_requests=len(request_metrics),
        successful_requests=len(successful_requests),
        failed_requests=len(failed_requests),
        total_images=total_images,
        total_pixels=total_pixels,
        total_inference_steps=total_inference_steps,
        total_prompt_chars=total_prompt_chars,
        total_tokens=total_tokens,
        avg_memory_peak_mb=avg_memory_peak_mb,
        max_memory_peak_mb=max_memory_peak_mb,
        avg_gpu_memory_mb=avg_gpu_memory_mb,
        max_gpu_memory_mb=max_gpu_memory_mb,
        avg_image_width=avg_image_width,
        avg_image_height=avg_image_height,
        avg_inference_steps=avg_inference_steps,
        avg_guidance_scale=avg_guidance_scale,
        concurrency_level=concurrency_level
    )


def validate_image_metrics_consistency(metrics: ImageMetrics, tolerance: float = 1e-6) -> List[str]:
    """
    Validate that computed image metrics are consistent with each other.
    
    Returns list of validation warnings/errors.
    """
    warnings = []
    
    # Check success rate
    if metrics.success_rate < 95.0:
        warnings.append(
            f"Low success rate: {metrics.success_rate:.1f}% "
            f"({metrics.failed_requests}/{metrics.total_requests} failed)"
        )
    
    # Check for reasonable values
    if metrics.ips <= 0:
        warnings.append("IPS is zero or negative - no successful image generation")
    
    if metrics.total_wall_time <= 0:
        warnings.append("Total wall time is zero or negative")
    
    # Check TTFI values are reasonable (not negative or too large)
    if metrics.ttfi_p50 < 0:
        warnings.append(f"Negative TTFI p50: {metrics.ttfi_p50:.3f}s")
    
    if metrics.ttfi_p95 > 300.0:  # More than 5 minutes seems unreasonable for image generation
        warnings.append(f"Very high TTFI p95: {metrics.ttfi_p95:.3f}s")
    
    # Check image dimensions are reasonable
    if metrics.avg_image_width < 64 or metrics.avg_image_height < 64:
        warnings.append(f"Very small image dimensions: {metrics.avg_image_width:.0f}x{metrics.avg_image_height:.0f}")
    
    if metrics.avg_image_width > 4096 or metrics.avg_image_height > 4096:
        warnings.append(f"Very large image dimensions: {metrics.avg_image_width:.0f}x{metrics.avg_image_height:.0f}")
    
    return warnings

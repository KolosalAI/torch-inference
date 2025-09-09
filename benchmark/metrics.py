"""
TTS Benchmark Metrics Module.

Implements the core TTS benchmarking metrics including:
- ASPS (Audio Seconds Per Second) - the primary throughput metric
- RTF (Real Time Factor) 
- RPS (Requests Per Second)
- CPS (Characters Per Second)
- TTFA (Time To First Audio) latency metrics
"""

import time
import statistics
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TTSRequestMetrics:
    """Metrics captured for a single TTS request."""
    request_id: str
    t_start: float  # Request start time
    t_first_audio: Optional[float] = None  # Time when first audio chunk was generated
    t_end: Optional[float] = None  # Request completion time
    text_len_chars: int = 0  # Length of input text in characters
    text_len_tokens: int = 0  # Length of input text in tokens (if available)
    audio_duration_sec: float = 0.0  # Duration of generated audio in seconds
    sample_rate: int = 22050  # Audio sample rate
    bit_depth: int = 16  # Audio bit depth
    error: Optional[str] = None  # Error message if request failed
    
    @property
    def wall_time(self) -> Optional[float]:
        """Total wall-clock time for this request."""
        if self.t_end is not None:
            return self.t_end - self.t_start
        return None
    
    @property
    def ttfa(self) -> Optional[float]:
        """Time to first audio (latency metric)."""
        if self.t_first_audio is not None:
            return self.t_first_audio - self.t_start
        return None
    
    @property
    def rtf(self) -> Optional[float]:
        """Real Time Factor for this request."""
        if self.wall_time is not None and self.audio_duration_sec > 0:
            return self.wall_time / self.audio_duration_sec
        return None


@dataclass
class TTSMetrics:
    """Aggregated TTS benchmark metrics for a complete run."""
    # Primary throughput metric
    asps: float = 0.0  # Audio Seconds Per Second
    
    # Related throughput metrics
    rtf_mean: float = 0.0  # Mean Real Time Factor
    rtf_median: float = 0.0  # Median Real Time Factor
    rps: float = 0.0  # Requests Per Second
    cps: float = 0.0  # Characters Per Second
    tokens_per_sec: float = 0.0  # Tokens Per Second (if available)
    
    # Latency metrics (TTFA - Time To First Audio)
    ttfa_p50: float = 0.0  # 50th percentile TTFA
    ttfa_p95: float = 0.0  # 95th percentile TTFA
    ttfa_p99: float = 0.0  # 99th percentile TTFA
    ttfa_mean: float = 0.0  # Mean TTFA
    
    # Wall-clock timing
    total_wall_time: float = 0.0  # Total benchmark wall time
    min_request_start: float = 0.0  # Earliest request start time
    max_request_end: float = 0.0  # Latest request end time
    
    # Request statistics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Audio statistics
    total_audio_duration: float = 0.0  # Sum of all generated audio durations
    total_text_chars: int = 0  # Sum of all input text characters
    total_tokens: int = 0  # Sum of all input tokens (if available)
    
    # Configuration
    concurrency_level: int = 1
    sample_rate: int = 22050
    bit_depth: int = 16
    
    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    @property
    def avg_text_length(self) -> float:
        """Average text length in characters."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_text_chars / self.successful_requests
    
    @property
    def avg_audio_duration(self) -> float:
        """Average audio duration in seconds."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_audio_duration / self.successful_requests


def compute_asps(request_metrics: List[TTSRequestMetrics]) -> float:
    """
    Compute ASPS (Audio Seconds Per Second) - the primary TTS throughput metric.
    
    ASPS = sum(audio_duration_i) / T_wall
    
    Where:
    - audio_duration_i = length of the i-th synthesized waveform (in seconds)
    - T_wall = wall-clock time from first request start to last audio sample produced
    
    Higher ASPS is better. ASPS = 1/RTF where RTF is Real Time Factor.
    """
    if not request_metrics:
        return 0.0
    
    # Filter successful requests
    successful_requests = [r for r in request_metrics if r.error is None and r.t_end is not None]
    
    if not successful_requests:
        return 0.0
    
    # Calculate total audio duration
    total_audio_duration = sum(r.audio_duration_sec for r in successful_requests)
    
    # Calculate wall-clock time
    min_start = min(r.t_start for r in successful_requests)
    max_end = max(r.t_end for r in successful_requests)
    wall_time = max_end - min_start
    
    if wall_time <= 0:
        return 0.0
    
    return total_audio_duration / wall_time


def compute_rtf(request_metrics: List[TTSRequestMetrics]) -> Tuple[float, float]:
    """
    Compute Real Time Factor statistics (mean and median).
    
    RTF = T_synthesis / audio_duration
    
    Where RTF < 1.0 means faster than real-time synthesis.
    Returns (mean_rtf, median_rtf).
    """
    if not request_metrics:
        return 0.0, 0.0
    
    rtf_values = []
    for request in request_metrics:
        if request.rtf is not None:
            rtf_values.append(request.rtf)
    
    if not rtf_values:
        return 0.0, 0.0
    
    mean_rtf = statistics.mean(rtf_values)
    median_rtf = statistics.median(rtf_values)
    
    return mean_rtf, median_rtf


def compute_rps(request_metrics: List[TTSRequestMetrics]) -> float:
    """
    Compute RPS (Requests Per Second).
    
    RPS = N / T_wall
    
    Only meaningful when reported alongside median input length.
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


def compute_cps(request_metrics: List[TTSRequestMetrics]) -> float:
    """
    Compute CPS (Characters Per Second).
    
    CPS = sum(text_len_chars) / T_wall
    
    Normalizes for varying prompt length.
    """
    if not request_metrics:
        return 0.0
    
    successful_requests = [r for r in request_metrics if r.error is None and r.t_end is not None]
    
    if not successful_requests:
        return 0.0
    
    # Calculate total characters
    total_chars = sum(r.text_len_chars for r in successful_requests)
    
    # Calculate wall-clock time
    min_start = min(r.t_start for r in successful_requests)
    max_end = max(r.t_end for r in successful_requests)
    wall_time = max_end - min_start
    
    if wall_time <= 0:
        return 0.0
    
    return total_chars / wall_time


def compute_tokens_per_sec(request_metrics: List[TTSRequestMetrics]) -> float:
    """
    Compute tokens per second.
    
    Similar to CPS but uses token count instead of character count.
    """
    if not request_metrics:
        return 0.0
    
    successful_requests = [r for r in request_metrics if r.error is None and r.t_end is not None]
    
    if not successful_requests:
        return 0.0
    
    # Calculate total tokens
    total_tokens = sum(r.text_len_tokens for r in successful_requests)
    
    if total_tokens == 0:
        return 0.0
    
    # Calculate wall-clock time
    min_start = min(r.t_start for r in successful_requests)
    max_end = max(r.t_end for r in successful_requests)
    wall_time = max_end - min_start
    
    if wall_time <= 0:
        return 0.0
    
    return total_tokens / wall_time


def compute_ttfa_statistics(request_metrics: List[TTSRequestMetrics]) -> Dict[str, float]:
    """
    Compute TTFA (Time To First Audio) latency statistics.
    
    Returns p50, p95, p99, and mean TTFA values.
    Essential for streaming UX evaluation.
    """
    if not request_metrics:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    
    ttfa_values = []
    for request in request_metrics:
        if request.ttfa is not None:
            ttfa_values.append(request.ttfa)
    
    if not ttfa_values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    
    ttfa_values.sort()
    n = len(ttfa_values)
    
    p50_idx = int(0.50 * n)
    p95_idx = int(0.95 * n)
    p99_idx = int(0.99 * n)
    
    return {
        "p50": ttfa_values[min(p50_idx, n - 1)],
        "p95": ttfa_values[min(p95_idx, n - 1)],
        "p99": ttfa_values[min(p99_idx, n - 1)],
        "mean": statistics.mean(ttfa_values)
    }


def aggregate_tts_metrics(
    request_metrics: List[TTSRequestMetrics],
    concurrency_level: int = 1
) -> TTSMetrics:
    """
    Aggregate individual request metrics into overall TTS benchmark metrics.
    
    Args:
        request_metrics: List of individual request metrics
        concurrency_level: Concurrency level used in the benchmark
        
    Returns:
        TTSMetrics object with all computed metrics
    """
    if not request_metrics:
        return TTSMetrics()
    
    # Filter successful requests
    successful_requests = [r for r in request_metrics if r.error is None and r.t_end is not None]
    failed_requests = [r for r in request_metrics if r.error is not None]
    
    # Compute primary metrics
    asps = compute_asps(request_metrics)
    rtf_mean, rtf_median = compute_rtf(request_metrics)
    rps = compute_rps(request_metrics)
    cps = compute_cps(request_metrics)
    tokens_per_sec = compute_tokens_per_sec(request_metrics)
    
    # Compute TTFA statistics
    ttfa_stats = compute_ttfa_statistics(request_metrics)
    
    # Compute timing bounds
    if successful_requests:
        min_start = min(r.t_start for r in successful_requests)
        max_end = max(r.t_end for r in successful_requests)
        total_wall_time = max_end - min_start
        
        # Audio and text statistics
        total_audio_duration = sum(r.audio_duration_sec for r in successful_requests)
        total_text_chars = sum(r.text_len_chars for r in successful_requests)
        total_tokens = sum(r.text_len_tokens for r in successful_requests)
        
        # Sample rate and bit depth (assume consistent across requests)
        sample_rate = successful_requests[0].sample_rate
        bit_depth = successful_requests[0].bit_depth
    else:
        min_start = max_end = total_wall_time = 0.0
        total_audio_duration = total_text_chars = total_tokens = 0
        sample_rate = 22050
        bit_depth = 16
    
    return TTSMetrics(
        asps=asps,
        rtf_mean=rtf_mean,
        rtf_median=rtf_median,
        rps=rps,
        cps=cps,
        tokens_per_sec=tokens_per_sec,
        ttfa_p50=ttfa_stats["p50"],
        ttfa_p95=ttfa_stats["p95"],
        ttfa_p99=ttfa_stats["p99"],
        ttfa_mean=ttfa_stats["mean"],
        total_wall_time=total_wall_time,
        min_request_start=min_start,
        max_request_end=max_end,
        total_requests=len(request_metrics),
        successful_requests=len(successful_requests),
        failed_requests=len(failed_requests),
        total_audio_duration=total_audio_duration,
        total_text_chars=total_text_chars,
        total_tokens=total_tokens,
        concurrency_level=concurrency_level,
        sample_rate=sample_rate,
        bit_depth=bit_depth
    )


def validate_metrics_consistency(metrics: TTSMetrics, tolerance: float = 1e-6) -> List[str]:
    """
    Validate that computed metrics are consistent with each other.
    
    Returns list of validation warnings/errors.
    """
    warnings = []
    
    # Check ASPS vs RTF relationship: ASPS = 1/RTF (approximately)
    if metrics.rtf_median > 0:
        expected_asps = 1.0 / metrics.rtf_median
        asps_diff = abs(metrics.asps - expected_asps) / expected_asps
        if asps_diff > tolerance:
            warnings.append(
                f"ASPS ({metrics.asps:.4f}) and RTF ({metrics.rtf_median:.4f}) "
                f"are inconsistent. Expected ASPS ≈ {expected_asps:.4f}"
            )
    
    # Check success rate
    if metrics.success_rate < 95.0:
        warnings.append(
            f"Low success rate: {metrics.success_rate:.1f}% "
            f"({metrics.failed_requests}/{metrics.total_requests} failed)"
        )
    
    # Check for reasonable values
    if metrics.asps <= 0:
        warnings.append("ASPS is zero or negative - no successful audio generation")
    
    if metrics.total_wall_time <= 0:
        warnings.append("Total wall time is zero or negative")
    
    # Check TTFA values are reasonable (not negative or too large)
    if metrics.ttfa_p50 < 0:
        warnings.append(f"Negative TTFA p50: {metrics.ttfa_p50:.3f}s")
    
    if metrics.ttfa_p95 > 60.0:  # More than 1 minute seems unreasonable
        warnings.append(f"Very high TTFA p95: {metrics.ttfa_p95:.3f}s")
    
    return warnings

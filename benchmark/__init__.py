"""
TTS Benchmarking Module for torch-inference server.

This module provides comprehensive benchmarking capabilities for Text-to-Speech models
with industry-standard metrics including ASPS (Audio Seconds Per Second), RTF (Real Time Factor),
and latency measurements.
"""

from .tts_benchmark import TTSBenchmarker, TTSBenchmarkResult, TTSRequestMetrics
from .harness import TTSBenchmarkHarness
from .metrics import TTSMetrics, compute_asps, compute_rtf, compute_rps, compute_cps
from .reporter import TTSBenchmarkReporter

__all__ = [
    'TTSBenchmarker',
    'TTSBenchmarkResult', 
    'TTSRequestMetrics',
    'TTSBenchmarkHarness',
    'TTSMetrics',
    'TTSBenchmarkReporter',
    'compute_asps',
    'compute_rtf', 
    'compute_rps',
    'compute_cps'
]

__version__ = "1.0.0"

"""
Benchmarking Module for torch-inference server.

This module provides comprehensive benchmarking capabilities for both Text-to-Speech 
and Image generation models with industry-standard metrics.

TTS Metrics:
- ASPS (Audio Seconds Per Second)
- RTF (Real Time Factor) 
- TTFA (Time To First Audio)
- RPS (Requests Per Second)

Image Metrics:
- IPS (Images Per Second)
- PPS (Pixels Per Second)
- SPS (Steps Per Second) 
- TTFI (Time To First Image)
"""

# TTS Benchmarking
from .tts_benchmark import TTSBenchmarker, TTSBenchmarkResult, TTSRequestMetrics
from .metrics import TTSMetrics, compute_asps, compute_rtf, compute_rps, compute_cps
from .reporter import TTSBenchmarkReporter

# Image Benchmarking  
from .image_benchmark import ImageBenchmarker, ImageBenchmarkResult, ImageRequestMetrics
from .image_metrics import ImageMetrics, compute_ips, compute_pps, compute_sps
from .image_reporter import ImageBenchmarkReporter

# Unified Harness
from .harness import TTSBenchmarkHarness, ImageBenchmarkHarness, BenchmarkConfig, generate_image_prompts

__all__ = [
    # TTS Benchmarking
    'TTSBenchmarker',
    'TTSBenchmarkResult', 
    'TTSRequestMetrics',
    'TTSBenchmarkHarness',
    'TTSMetrics',
    'TTSBenchmarkReporter',
    'compute_asps',
    'compute_rtf', 
    'compute_rps',
    'compute_cps',
    
    # Image Benchmarking
    'ImageBenchmarker',
    'ImageBenchmarkResult',
    'ImageRequestMetrics', 
    'ImageBenchmarkHarness',
    'ImageMetrics',
    'ImageBenchmarkReporter',
    'compute_ips',
    'compute_pps',
    'compute_sps',
    
    # Unified Configuration
    'BenchmarkConfig',
    'generate_image_prompts'
]

__version__ = "2.0.0"

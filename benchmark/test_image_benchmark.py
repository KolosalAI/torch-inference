#!/usr/bin/env python3
"""
Test Image Benchmark System - Simple verification tests.

This script runs basic tests to verify the image benchmarking system
is working correctly.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add benchmark module to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmark.image_benchmark import ImageBenchmarker
from benchmark.image_metrics import ImageMetrics, ImageRequestMetrics, aggregate_image_metrics
from benchmark.image_reporter import ImageBenchmarkReporter
from benchmark.harness import ImageBenchmarkHarness, BenchmarkConfig, generate_image_prompts
import time
import random


def dummy_image_model(prompt: str, **kwargs) -> bytes:
    """Simple dummy image model for testing."""
    # Simulate some processing time
    time.sleep(0.1 + random.random() * 0.1)
    
    # Return dummy image data
    width = kwargs.get('width', 512)
    height = kwargs.get('height', 512)
    image_size = width * height * 3
    return b'DUMMY_IMAGE_DATA' * (image_size // 16)


def test_image_metrics():
    """Test image metrics calculation."""
    print("Testing image metrics...")
    
    # Create test request metrics
    request_metrics = []
    for i in range(5):
        metrics = ImageRequestMetrics(
            request_id=f"req_{i}",
            prompt_len_chars=50 + i * 10,
            prompt_len_tokens=10 + i * 2,
            negative_prompt_len_chars=0,
            width=512,
            height=512,
            num_images=1,
            num_inference_steps=20,
            guidance_scale=7.5,
            seed=42,
            t_start=i * 1.0,
            t_first_image=i * 1.0 + 0.5,
            t_end=i * 1.0 + 1.0,
            memory_peak_mb=500.0 + i * 10,
            gpu_memory_mb=400.0 + i * 5,
            file_size_bytes=1024 * 50,
            image_format="PNG",
            error=None
        )
        request_metrics.append(metrics)
    
    # Calculate aggregated metrics
    metrics = aggregate_image_metrics(request_metrics, concurrency_level=1)
    
    print(f"  IPS: {metrics.ips:.3f}")
    print(f"  PPS: {metrics.pps:.0f}")
    print(f"  RPS: {metrics.rps:.1f}")
    print(f"  TTFI p95: {metrics.ttfi_p95*1000:.1f}ms")
    
    assert metrics.ips > 0, "IPS should be positive"
    assert metrics.pps > 0, "PPS should be positive"
    assert metrics.rps > 0, "RPS should be positive"
    assert metrics.total_requests == 5, "Total requests should be 5"
    
    print("✓ Image metrics test passed")


def test_image_benchmarker():
    """Test image benchmarker functionality."""
    print("Testing image benchmarker...")
    
    benchmarker = ImageBenchmarker(warmup_requests=1, timeout_seconds=10.0)
    
    test_prompts = [
        "A beautiful landscape",
        "A cute cat",
        "A modern city"
    ]
    
    result = benchmarker.benchmark_sync_image_model(
        dummy_image_model,
        test_prompts,
        concurrency_levels=[2],
        iterations_per_level=6,
        width=256,
        height=256,
        num_images=1,
        num_inference_steps=10,
        guidance_scale=5.0
    )
    
    # Get the single result (concurrency level 2)
    result = result[2]
    
    print(f"  Total requests: {len(result.request_metrics)}")
    print(f"  IPS: {result.metrics.ips:.3f}")
    print(f"  Success rate: {result.metrics.success_rate:.1f}%")
    
    assert len(result.request_metrics) == 6, "Should have 6 request metrics"
    assert result.metrics.ips > 0, "IPS should be positive"
    assert result.metrics.success_rate == 100.0, "All requests should succeed"
    
    print("✓ Image benchmarker test passed")


def test_image_reporter():
    """Test image reporter CSV generation."""
    print("Testing image reporter...")
    
    # Create dummy benchmark results
    benchmarker = ImageBenchmarker(warmup_requests=1, timeout_seconds=10.0)
    
    test_prompts = ["A test image", "Another test image"]
    
    results = {}
    for concurrency in [1, 2]:
        result = benchmarker.benchmark_sync_image_model(
            dummy_image_model,
            test_prompts,
            concurrency_levels=[concurrency],
            iterations_per_level=4,
            width=256,
            height=256,
            num_images=1,
            num_inference_steps=10,
            guidance_scale=5.0
        )
        results[concurrency] = result[concurrency]
    
    # Test reporter
    reporter = ImageBenchmarkReporter()
    
    # Test detailed CSV
    detailed_csv = reporter.generate_detailed_csv_report(results, test_prompts)
    assert "Concurrency_Level" in detailed_csv, "Detailed CSV should have header"
    assert "Images_Per_Sec" in detailed_csv, "Detailed CSV should have IPS column"
    
    # Test summary CSV
    summary_csv = reporter.generate_summary_csv_report(results, test_prompts)
    assert "IPS" in summary_csv, "Summary CSV should have IPS column"
    assert "PPS" in summary_csv, "Summary CSV should have PPS column"
    
    print("  Detailed CSV length:", len(detailed_csv.split('\n')))
    print("  Summary CSV length:", len(summary_csv.split('\n')))
    
    print("✓ Image reporter test passed")


def test_image_harness():
    """Test image benchmark harness."""
    print("Testing image benchmark harness...")
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        config = BenchmarkConfig(
            model_type="image",
            concurrency_levels=[1, 2],
            iterations_per_level=4,
            text_variations=2,
            output_dir=temp_dir,
            generate_detailed_csv=True,
            generate_summary_csv=True,
            generate_plots=False,  # Skip plots for testing
            image_width=256,
            image_height=256,
            num_images=1,
            num_inference_steps=10,
            guidance_scale=5.0
        )
        
        harness = ImageBenchmarkHarness(config)
        results = harness.run_benchmark(
            dummy_image_model,
            benchmark_name="test_image_benchmark"
        )
        
        print(f"  Benchmark results: {len(results)} concurrency levels")
        
        # Check that files were created
        detailed_csv_path = os.path.join(temp_dir, "test_image_benchmark_detailed.csv")
        summary_csv_path = os.path.join(temp_dir, "test_image_benchmark_summary.csv")
        
        assert os.path.exists(detailed_csv_path), "Detailed CSV should be created"
        assert os.path.exists(summary_csv_path), "Summary CSV should be created"
        
        print(f"  Created detailed CSV: {os.path.getsize(detailed_csv_path)} bytes")
        print(f"  Created summary CSV: {os.path.getsize(summary_csv_path)} bytes")
        
        assert len(results) == 2, "Should have results for 2 concurrency levels"
        for result in results.values():
            assert result.metrics.ips > 0, "IPS should be positive"
    
    print("✓ Image harness test passed")


def test_prompt_generation():
    """Test image prompt generation."""
    print("Testing prompt generation...")
    
    # Test basic prompt generation
    prompts = generate_image_prompts(count=5, min_length=20, max_length=100)
    
    print(f"  Generated {len(prompts)} prompts")
    for i, prompt in enumerate(prompts):
        print(f"    {i+1}: {prompt[:50]}...")
        assert len(prompt) >= 20, f"Prompt {i+1} too short: {len(prompt)}"
        assert len(prompt) <= 100, f"Prompt {i+1} too long: {len(prompt)}"
    
    # Test custom prompts
    custom_prompts = ["Custom prompt 1", "Custom prompt 2"]
    result_prompts = generate_image_prompts(custom_prompts=custom_prompts)
    assert result_prompts == custom_prompts, "Custom prompts should be returned as-is"
    
    print("✓ Prompt generation test passed")


def run_all_tests():
    """Run all tests."""
    print("Running Image Benchmark System Tests")
    print("=" * 50)
    
    try:
        test_image_metrics()
        print()
        
        test_image_benchmarker()
        print()
        
        test_image_reporter()
        print()
        
        test_image_harness()
        print()
        
        test_prompt_generation()
        print()
        
        print("=" * 50)
        print("✓ All tests passed successfully!")
        print("The image benchmark system is working correctly.")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()

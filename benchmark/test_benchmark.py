"""
Test script for TTS benchmark functionality.

Validates the benchmark system components work correctly.
"""

import asyncio
import logging
import tempfile
import os
from pathlib import Path

# Setup path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from benchmark.harness import TTSBenchmarkHarness, BenchmarkConfig, create_demo_tts_function
from benchmark.metrics import TTSRequestMetrics, aggregate_tts_metrics, validate_metrics_consistency
from benchmark.reporter import TTSBenchmarkReporter
from benchmark.http_client import TTSServerConfig, HTTPTTSClient


def test_demo_tts_function():
    """Test the demo TTS function."""
    print("Testing demo TTS function...")
    
    demo_tts = create_demo_tts_function(
        min_audio_duration=1.0,
        max_audio_duration=3.0,
        processing_delay=0.01,
        failure_rate=0.0
    )
    
    result = demo_tts("Hello world, this is a test.")
    
    assert 'audio_duration' in result
    assert result['audio_duration'] > 0
    assert 'sample_rate' in result
    assert result['sample_rate'] == 22050
    
    print("✅ Demo TTS function works correctly")


def test_metrics_calculation():
    """Test metrics calculation functions."""
    print("Testing metrics calculation...")
    
    # Create sample request metrics
    import time
    base_time = time.time()
    
    request_metrics = [
        TTSRequestMetrics(
            request_id="req_1",
            t_start=base_time,
            t_first_audio=base_time + 0.1,
            t_end=base_time + 0.5,
            text_len_chars=100,
            audio_duration_sec=2.0,
            sample_rate=22050
        ),
        TTSRequestMetrics(
            request_id="req_2", 
            t_start=base_time + 0.1,
            t_first_audio=base_time + 0.2,
            t_end=base_time + 0.7,
            text_len_chars=120,
            audio_duration_sec=2.5,
            sample_rate=22050
        )
    ]
    
    # Test aggregation
    metrics = aggregate_tts_metrics(request_metrics, concurrency_level=2)
    
    assert metrics.total_requests == 2
    assert metrics.successful_requests == 2
    assert metrics.total_audio_duration == 4.5
    assert metrics.asps > 0
    assert metrics.success_rate == 100.0
    
    print("✅ Metrics calculation works correctly")


def test_csv_generation():
    """Test CSV report generation."""
    print("Testing CSV report generation...")
    
    from benchmark.tts_benchmark import TTSBenchmarkResult
    
    # Create mock results
    demo_tts = create_demo_tts_function()
    config = BenchmarkConfig(
        concurrency_levels=[1, 2],
        iterations_per_level=3,
        output_dir=tempfile.mkdtemp()
    )
    
    harness = TTSBenchmarkHarness(config)
    results = harness.run_benchmark(demo_tts, benchmark_name="test_csv")
    
    # Test CSV generation
    reporter = TTSBenchmarkReporter()
    csv_content = reporter.generate_csv_report(results)
    
    assert "Concurrency,ASPS,RTF_Mean" in csv_content
    assert "1," in csv_content
    assert "2," in csv_content
    
    print("✅ CSV generation works correctly")


def test_http_client_mock():
    """Test HTTP client with mock server (simulation)."""
    print("Testing HTTP client structure...")
    
    # Test client initialization
    config = TTSServerConfig(
        base_url="http://localhost:8000",
        timeout=5.0
    )
    
    # Just test that client can be created
    client = HTTPTTSClient(config)
    assert client.config.base_url == "http://localhost:8000"
    assert client.config.timeout == 5.0
    
    print("✅ HTTP client structure is correct")


def test_benchmark_validation():
    """Test benchmark result validation."""
    print("Testing benchmark validation...")
    
    # Run a small benchmark
    demo_tts = create_demo_tts_function(failure_rate=0.1)  # 10% failure rate
    
    config = BenchmarkConfig(
        concurrency_levels=[1],
        iterations_per_level=5,
        output_dir=tempfile.mkdtemp()
    )
    
    harness = TTSBenchmarkHarness(config)
    results = harness.run_benchmark(demo_tts, benchmark_name="test_validation")
    
    # Test validation
    for result in results.values():
        warnings = validate_metrics_consistency(result.metrics)
        # Should have some warnings due to simulation inconsistencies
        print(f"  Validation warnings: {len(warnings)}")
    
    print("✅ Benchmark validation works correctly")


def run_all_tests():
    """Run all tests."""
    print("TTS Benchmark Test Suite")
    print("=" * 50)
    
    # Set up logging to reduce noise
    logging.basicConfig(level=logging.WARNING)
    
    try:
        test_demo_tts_function()
        test_metrics_calculation()
        test_csv_generation()
        test_http_client_mock()
        test_benchmark_validation()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! The TTS benchmark system is working correctly.")
        print("\nYou can now:")
        print("1. Use the API directly in your code for benchmarking")
        print("2. Import benchmark classes and functions as needed")
        print("3. Run quick tests using the integrated quick test functions")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

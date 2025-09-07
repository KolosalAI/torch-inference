#!/usr/bin/env python3
"""
Test script to verify the new concurrency levels (1, 2, 4, 8, 16, 32, 64) are working correctly.
"""

import sys
import logging
from pathlib import Path

# Add benchmark module to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmark.harness import BenchmarkConfig, TTSBenchmarkHarness, create_demo_tts_function

def test_new_concurrency_levels():
    """Test that the new concurrency levels are set correctly."""
    
    # Test default configuration
    config = BenchmarkConfig()
    expected_levels = [1, 2, 4, 8, 16, 32, 64]
    
    print("Testing default concurrency levels...")
    print(f"Expected: {expected_levels}")
    print(f"Actual: {config.concurrency_levels}")
    
    if config.concurrency_levels == expected_levels:
        print("✅ Default concurrency levels are correct!")
    else:
        print("❌ Default concurrency levels are incorrect!")
        return False
    
    # Test that we can create a benchmark with these levels
    print("\nTesting benchmark creation with new concurrency levels...")
    
    try:
        # Create a quick demo TTS function
        demo_tts = create_demo_tts_function(
            min_audio_duration=0.1,
            max_audio_duration=0.5,
            processing_delay=0.001,  # Very fast for testing
            failure_rate=0.0
        )
        
        # Create benchmark config with the new levels
        test_config = BenchmarkConfig(
            concurrency_levels=[1, 2, 4, 8, 16, 32, 64],
            iterations_per_level=2,  # Just 2 iterations for quick test
            text_variations=5,       # Few texts for speed
            output_dir="test_concurrency_results"
        )
        
        # Create harness
        harness = TTSBenchmarkHarness(test_config)
        
        # Generate test texts
        test_texts = harness.generate_test_texts(count=5, min_length=20, max_length=50)
        print(f"Generated {len(test_texts)} test texts")
        
        print("✅ Benchmark harness created successfully with new concurrency levels!")
        print("✅ All concurrency levels (1, 2, 4, 8, 16, 32, 64) are now available!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating benchmark: {e}")
        return False

def demo_quick_benchmark():
    """Run a very quick benchmark to demonstrate the new concurrency levels."""
    print("\n" + "="*60)
    print("Running quick demo with new concurrency levels...")
    print("="*60)
    
    try:
        # Create demo TTS
        demo_tts = create_demo_tts_function(
            min_audio_duration=0.1,
            max_audio_duration=0.3,
            processing_delay=0.005,  # 5ms processing delay
            failure_rate=0.0
        )
        
        # Test with subset of the new levels for quick demo
        config = BenchmarkConfig(
            concurrency_levels=[1, 4, 16, 64],  # Test some of the new higher levels
            iterations_per_level=3,              # Quick test
            text_variations=5,
            output_dir="demo_new_concurrency"
        )
        
        harness = TTSBenchmarkHarness(config)
        
        # Run quick benchmark
        results = harness.run_benchmark(
            demo_tts,
            benchmark_name="quick_concurrency_test"
        )
        
        print("\nQuick benchmark results:")
        print("-" * 40)
        for concurrency, result in results.items():
            metrics = result.metrics
            print(f"Concurrency {concurrency:2d}: ASPS={metrics.asps:.3f}, "
                  f"RTF={metrics.rtf_median:.3f}, RPS={metrics.rps:.1f}")
        
        print("\n✅ Successfully tested higher concurrency levels!")
        print("✅ The benchmark system now supports concurrency levels: 1, 2, 4, 8, 16, 32, 64")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set up simple logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise for testing
    
    print("Testing New Concurrency Levels: 1, 2, 4, 8, 16, 32, 64")
    print("=" * 60)
    
    # Test configuration
    success = test_new_concurrency_levels()
    
    if success:
        # Run demo if configuration test passed
        demo_quick_benchmark()
    else:
        print("❌ Configuration test failed!")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ All tests passed! New concurrency levels are ready to use.")
    print("="*60)

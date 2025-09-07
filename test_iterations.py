#!/usr/bin/env python3
"""
Test script to verify that the default iterations are now set to 100.
"""

import sys
from pathlib import Path

# Add benchmark module to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmark.harness import BenchmarkConfig

def test_default_iterations():
    """Test that the default iterations are set to 100."""
    
    # Test default configuration
    config = BenchmarkConfig()
    expected_iterations = 100
    
    print("Testing default iteration count...")
    print(f"Expected: {expected_iterations}")
    print(f"Actual: {config.iterations_per_level}")
    
    if config.iterations_per_level == expected_iterations:
        print("✅ Default iterations are correctly set to 100!")
        return True
    else:
        print("❌ Default iterations are incorrect!")
        return False

if __name__ == "__main__":
    print("Testing Default Iterations: 100")
    print("=" * 40)
    
    success = test_default_iterations()
    
    if success:
        print("\n✅ All tests passed! Default iterations are now 100.")
        print("\nThis means:")
        print("- Running 'python benchmark.py demo' will use 100 iterations per concurrency level")
        print("- Running 'python benchmark.py server --url <url>' will use 100 iterations per concurrency level") 
        print("- Running 'python benchmark.py voices --url <url> --voices <voices>' will use 100 iterations per concurrency level")
        print("- All benchmark configurations now default to 100 iterations for better statistical accuracy")
    else:
        print("❌ Test failed!")
        sys.exit(1)

#!/usr/bin/env python3
"""
ResNet Benchmark Usage Examples.

This script demonstrates how to use the ResNet image classification benchmark
with different configurations and scenarios.
"""

import logging
import asyncio
from pathlib import Path
import sys

# Add benchmark module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.resnet_image_benchmark import (
    ResNetImageBenchmarker, 
    create_resnet_classification_function,
    create_demo_resnet_function,
    run_resnet_benchmark_example
)


def example_demo_resnet_benchmark():
    """Run a demo ResNet benchmark example."""
    print("=" * 60)
    print("Example 1: Demo ResNet Classification Benchmark")
    print("=" * 60)
    
    # Create ResNet benchmarker
    benchmarker = ResNetImageBenchmarker(
        default_width=224,
        default_height=224,
        warmup_requests=3
    )
    
    # Create demo classification function
    demo_classifier = create_demo_resnet_function()
    
    # Run benchmark with moderate settings
    results = benchmarker.benchmark_resnet_model(
        classification_function=demo_classifier,
        concurrency_levels=[1, 2, 4, 8],
        iterations_per_level=25
    )
    
    print("\nDemo ResNet Benchmark Results:")
    print("-" * 50)
    for concurrency, result in results.items():
        metrics = result.metrics
        print(f"Concurrency {concurrency:2d}: "
              f"Classifications/sec={metrics.ips:7.2f}, "
              f"RPS={metrics.rps:7.1f}, "
              f"Latency p95={metrics.ttfi_p95*1000:6.1f}ms, "
              f"Memory={metrics.avg_memory_peak_mb:6.1f}MB")
    
    return results


def example_server_resnet_benchmark():
    """Example of benchmarking against a real ResNet server."""
    print("=" * 60)
    print("Example 2: Server ResNet Classification Benchmark")
    print("=" * 60)
    
    # Configuration for server benchmark
    server_url = "http://localhost:8000"
    model_name = "torchvision_resnet18"
    
    print(f"Benchmarking ResNet model: {model_name}")
    print(f"Server URL: {server_url}")
    print(f"Note: This requires a running torch-inference server with ResNet model loaded")
    
    # Create ResNet classification function for server
    resnet_function = create_resnet_classification_function(
        model_name=model_name,
        base_url=server_url,
        top_k=5
    )
    
    # Create ResNet benchmarker
    benchmarker = ResNetImageBenchmarker(
        default_width=224,
        default_height=224,
        warmup_requests=5
    )
    
    # Test single classification first
    print("\nTesting single classification...")
    test_images = benchmarker._generate_test_images(1)
    try:
        result = resnet_function(test_images[0])
        if result.get('success', False):
            print("✓ Server connection successful!")
            print(f"  Top prediction: {result.get('predictions', [{}])[0]}")
            
            # Run full benchmark
            print("\nRunning full benchmark...")
            results = benchmarker.benchmark_resnet_model(
                classification_function=resnet_function,
                concurrency_levels=[1, 2, 4],
                iterations_per_level=20
            )
            
            print("\nServer ResNet Benchmark Results:")
            print("-" * 50)
            for concurrency, result in results.items():
                metrics = result.metrics
                print(f"Concurrency {concurrency:2d}: "
                      f"Classifications/sec={metrics.ips:7.2f}, "
                      f"RPS={metrics.rps:7.1f}, "
                      f"Latency p95={metrics.ttfi_p95*1000:6.1f}ms")
            
            return results
        else:
            print("✗ Server connection failed!")
            print(f"  Error: {result.get('error', 'Unknown error')}")
            print("  Falling back to demo benchmark...")
            return example_demo_resnet_benchmark()
            
    except Exception as e:
        print(f"✗ Connection error: {e}")
        print("  Make sure the torch-inference server is running")
        print("  Falling back to demo benchmark...")
        return example_demo_resnet_benchmark()


def example_performance_comparison():
    """Compare ResNet performance at different concurrency levels."""
    print("=" * 60)
    print("Example 3: ResNet Performance Scaling Analysis")
    print("=" * 60)
    
    benchmarker = ResNetImageBenchmarker(
        default_width=224,
        default_height=224,
        warmup_requests=3
    )
    
    demo_classifier = create_demo_resnet_function()
    
    # Test wider range of concurrency levels
    concurrency_levels = [1, 2, 4, 8, 16, 32]
    print(f"Testing concurrency levels: {concurrency_levels}")
    
    results = benchmarker.benchmark_resnet_model(
        classification_function=demo_classifier,
        concurrency_levels=concurrency_levels,
        iterations_per_level=30
    )
    
    print("\nPerformance Scaling Analysis:")
    print("-" * 70)
    print(f"{'Concurrency':<12} {'Class/sec':<12} {'Speedup':<10} {'Efficiency':<12} {'Latency p95':<12}")
    print("-" * 70)
    
    baseline_throughput = results[1].metrics.ips
    
    for concurrency, result in results.items():
        metrics = result.metrics
        speedup = metrics.ips / baseline_throughput
        efficiency = speedup / concurrency * 100
        
        print(f"{concurrency:<12} "
              f"{metrics.ips:<12.2f} "
              f"{speedup:<10.2f}x "
              f"{efficiency:<12.1f}% "
              f"{metrics.ttfi_p95*1000:<12.1f}ms")
    
    # Find optimal concurrency
    best_efficiency = max(results.items(), key=lambda x: x[1].metrics.ips / x[0])
    optimal_concurrency = best_efficiency[0]
    optimal_throughput = best_efficiency[1].metrics.ips
    
    print("-" * 70)
    print(f"Optimal Configuration:")
    print(f"  Concurrency: {optimal_concurrency}")
    print(f"  Throughput: {optimal_throughput:.2f} classifications/sec")
    print(f"  Speedup: {optimal_throughput/baseline_throughput:.2f}x")
    
    return results


def example_batch_size_comparison():
    """Compare different batch processing approaches."""
    print("=" * 60)
    print("Example 4: Batch Size Impact Analysis")
    print("=" * 60)
    
    # This example shows how you might adapt the benchmark for batch processing
    # Note: This would require modifications to support actual batch processing
    
    benchmarker = ResNetImageBenchmarker(
        default_width=224,
        default_height=224,
        warmup_requests=3
    )
    
    # Simulate different batch sizes by adjusting processing time
    def create_batch_classifier(batch_size):
        def batch_classify(image_data, **kwargs):
            import time
            import random
            
            # Simulate batch processing efficiency
            base_time = 0.05  # 50ms base
            batch_efficiency = min(1.0, batch_size * 0.8)  # Efficiency improves with batch size
            processing_time = base_time / batch_efficiency
            
            time.sleep(processing_time)
            
            return {
                'success': True,
                'predictions': [
                    {'class': f'class_{random.randint(0, 999)}', 'confidence': random.uniform(0.7, 0.95)}
                    for _ in range(batch_size)
                ],
                'processing_time': processing_time,
                'batch_size': batch_size
            }
        return batch_classify
    
    batch_sizes = [1, 4, 8, 16, 32]
    print(f"Testing batch sizes: {batch_sizes}")
    
    all_results = {}
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size {batch_size}...")
        batch_classifier = create_batch_classifier(batch_size)
        
        # Run smaller benchmark for batch comparison
        results = benchmarker.benchmark_single_concurrency_classification(
            classification_function=batch_classifier,
            concurrency=4,
            iterations=20
        )
        
        all_results[batch_size] = results
    
    print("\nBatch Size Comparison:")
    print("-" * 60)
    print(f"{'Batch Size':<12} {'Total/sec':<12} {'Per Item/sec':<15} {'Latency':<12}")
    print("-" * 60)
    
    for batch_size, result in all_results.items():
        metrics = result.metrics
        total_per_sec = metrics.ips
        per_item_per_sec = total_per_sec * batch_size
        latency_ms = metrics.ttfi_p95 * 1000
        
        print(f"{batch_size:<12} "
              f"{total_per_sec:<12.2f} "
              f"{per_item_per_sec:<15.2f} "
              f"{latency_ms:<12.1f}ms")
    
    return all_results


async def run_all_examples():
    """Run all ResNet benchmark examples."""
    print("ResNet Image Classification Benchmarking Examples")
    print("=" * 60)
    print("This script demonstrates various ResNet benchmarking scenarios")
    print()
    
    try:
        # Example 1: Basic demo benchmark
        example_demo_resnet_benchmark()
        print("\n" + "="*60 + "\n")
        
        # Example 2: Server benchmark (with fallback)
        example_server_resnet_benchmark()
        print("\n" + "="*60 + "\n")
        
        # Example 3: Performance scaling
        example_performance_comparison()
        print("\n" + "="*60 + "\n")
        
        # Example 4: Batch size comparison
        example_batch_size_comparison()
        
        print("\n" + "="*60)
        print("All ResNet benchmark examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        raise


def quick_resnet_test():
    """Quick test to verify ResNet benchmark functionality."""
    print("Quick ResNet Benchmark Test")
    print("-" * 30)
    
    benchmarker = ResNetImageBenchmarker(warmup_requests=1)
    demo_classifier = create_demo_resnet_function()
    
    # Quick single-concurrency test
    result = benchmarker.benchmark_single_concurrency_classification(
        classification_function=demo_classifier,
        concurrency=1,
        iterations=5
    )
    
    metrics = result.metrics
    print(f"✓ ResNet benchmark working!")
    print(f"  Classifications/sec: {metrics.ips:.2f}")
    print(f"  Average latency: {metrics.ttfi_p50*1000:.1f}ms")
    print(f"  Success rate: {metrics.success_rate:.1f}%")
    
    return result


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_resnet_test()
        elif sys.argv[1] == "demo":
            example_demo_resnet_benchmark()
        elif sys.argv[1] == "server":
            example_server_resnet_benchmark()
        elif sys.argv[1] == "scaling":
            example_performance_comparison()
        elif sys.argv[1] == "batch":
            example_batch_size_comparison()
        else:
            print("Usage: python resnet_examples.py [quick|demo|server|scaling|batch|all]")
    else:
        # Run all examples
        asyncio.run(run_all_examples())

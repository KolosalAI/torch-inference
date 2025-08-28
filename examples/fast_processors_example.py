"""
Example usage of the optimized fast processors.

This example demonstrates how to use the newly optimized processors
for maximum performance in your PyTorch inference workflows.
"""

import torch
import numpy as np
from framework.core.config import InferenceConfig, ConfigFactory
from framework.processors import (
    create_fast_default_pipelines,
    setup_fast_processors,
    FastProcessorFactory,
    benchmark_processor_performance,
    optimize_for_production_throughput
)


def basic_fast_processor_example():
    """Basic example of using fast processors."""
    print("=== Basic Fast Processor Example ===")
    
    # Create a configuration
    config = ConfigFactory.create_classification_config(num_classes=1000)
    
    # Enable optimizations in the config
    config.performance.enable_async = True
    config.performance.max_workers = 4
    config.cache.enable_caching = True
    config.batch.adaptive_batching = True
    
    # Create fast processors (recommended way)
    preprocessor, postprocessor = create_fast_default_pipelines(config)
    
    # Test with sample data
    sample_image = np.random.rand(224, 224, 3) * 255
    sample_image = sample_image.astype(np.uint8)
    
    # Preprocess
    preprocessed = preprocessor.preprocess(sample_image)
    print(f"Preprocessed shape: {preprocessed.data.shape}")
    print(f"Processing time: {preprocessed.processing_time:.4f}s")
    
    # Simulate model output
    model_output = torch.randn(1, 1000)
    
    # Postprocess
    from framework.processors.postprocessor import OutputType
    result = postprocessor.auto_postprocess(model_output)
    print(f"Predicted class: {result.predictions.predicted_class}")
    print(f"Confidence: {result.predictions.confidence:.4f}")


def production_optimization_example():
    """Example of production-ready optimization."""
    print("\n=== Production Optimization Example ===")
    
    # Create configuration optimized for production
    config = ConfigFactory.create_classification_config(num_classes=1000)
    
    # Create production-ready processors
    preprocessor, postprocessor = FastProcessorFactory.create_production_ready(config)
    
    # Test batch processing (optimized for throughput)
    batch_images = [
        np.random.rand(224, 224, 3) * 255 for _ in range(8)
    ]
    
    # Process batch
    import time
    start_time = time.time()
    batch_results = preprocessor.preprocess_batch(batch_images)
    batch_time = time.time() - start_time
    
    print(f"Processed {len(batch_images)} images in {batch_time:.4f}s")
    print(f"Throughput: {len(batch_images)/batch_time:.2f} images/sec")
    
    # Show performance statistics
    stats = preprocessor.get_performance_stats()
    print(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
    print(f"Average processing time: {stats.get('avg_processing_time', 0):.4f}s")


def real_time_optimization_example():
    """Example of real-time latency optimization."""
    print("\n=== Real-Time Optimization Example ===")
    
    config = ConfigFactory.create_classification_config(num_classes=1000)
    
    # Create real-time processors (optimized for latency)
    preprocessor, postprocessor = FastProcessorFactory.create_real_time(config)
    
    # Test single-item processing (optimized for latency)
    test_image = np.random.rand(224, 224, 3) * 255
    
    # Measure latency over multiple runs
    latencies = []
    for _ in range(10):
        start = time.perf_counter()
        result = preprocessor.preprocess(test_image)
        end = time.perf_counter()
        latencies.append(end - start)
    
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    print(f"Average latency: {avg_latency*1000:.2f}ms")
    print(f"Min latency: {min_latency*1000:.2f}ms")
    print(f"Max latency: {max_latency*1000:.2f}ms")


def memory_optimization_example():
    """Example of memory-efficient processing."""
    print("\n=== Memory Optimization Example ===")
    
    config = ConfigFactory.create_classification_config(num_classes=1000)
    
    # Create memory-efficient processors
    preprocessor, postprocessor = FastProcessorFactory.create_memory_efficient(config)
    
    # Process with memory monitoring
    import psutil
    process = psutil.Process()
    
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Process multiple items
    for i in range(50):
        test_data = np.random.rand(224, 224, 3) * 255
        result = preprocessor.preprocess(test_data)
        
        if i % 10 == 0:
            current_memory = process.memory_info().rss / 1024 / 1024
            print(f"Memory after {i+1} items: {current_memory:.2f} MB")
    
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"Final memory usage: {final_memory:.2f} MB")
    print(f"Memory increase: {final_memory - initial_memory:.2f} MB")


def benchmarking_example():
    """Example of benchmarking processor performance."""
    print("\n=== Benchmarking Example ===")
    
    config = ConfigFactory.create_classification_config(num_classes=1000)
    
    # Create different processor configurations for comparison
    configs = {
        "auto_optimized": FastProcessorFactory.create_auto_optimized(config)[0],
        "production": FastProcessorFactory.create_production_ready(config)[0],
        "real_time": FastProcessorFactory.create_real_time(config)[0],
        "memory_efficient": FastProcessorFactory.create_memory_efficient(config)[0]
    }
    
    # Create test data
    test_inputs = [np.random.rand(224, 224, 3) * 255 for _ in range(20)]
    
    # Benchmark each configuration
    for name, preprocessor in configs.items():
        print(f"\nBenchmarking {name}:")
        results = benchmark_processor_performance(
            preprocessor, test_inputs, num_iterations=5
        )
        
        timing = results['timing']
        print(f"  Throughput: {timing['throughput_items_per_second']:.2f} items/sec")
        print(f"  Average time: {timing['avg_time_seconds']*1000:.2f}ms")
        print(f"  Min time: {timing['min_time_seconds']*1000:.2f}ms")


def custom_optimization_example():
    """Example of custom optimization configuration."""
    print("\n=== Custom Optimization Example ===")
    
    from framework.processors.performance_config import ProcessorPerformanceConfig, ProcessorOptimizer
    
    # Create custom performance configuration
    perf_config = ProcessorPerformanceConfig(
        optimization_mode="aggressive",
        enable_parallel_processing=True,
        max_parallel_workers=8,
        enable_batch_processing=True,
        max_batch_size=64,
        enable_memory_pooling=True,
        enable_torch_compile=False,  # Keep disabled for stability
        enable_tensor_optimizations=True
    )
    
    # Create base config and processors
    config = ConfigFactory.create_classification_config(num_classes=1000)
    preprocessor, postprocessor = create_fast_default_pipelines(config)
    
    # Apply custom optimizations
    optimizer = ProcessorOptimizer(perf_config)
    optimizer.optimize_preprocessor_pipeline(preprocessor)
    optimizer.optimize_postprocessor_pipeline(postprocessor)
    
    # Test the optimized processors
    test_data = [np.random.rand(224, 224, 3) * 255 for _ in range(16)]
    
    start_time = time.time()
    results = preprocessor.preprocess_batch(test_data)
    end_time = time.time()
    
    print(f"Custom optimization results:")
    print(f"  Processed {len(test_data)} items in {end_time - start_time:.4f}s")
    print(f"  Throughput: {len(test_data)/(end_time - start_time):.2f} items/sec")
    
    # Get optimization report
    report = optimizer.get_optimization_report()
    print(f"  Optimizations applied: {report['total_optimizations']}")


def async_processing_example():
    """Example of asynchronous processing."""
    print("\n=== Async Processing Example ===")
    
    import asyncio
    
    async def async_processing_demo():
        config = ConfigFactory.create_classification_config(num_classes=1000)
        config.performance.enable_async = True
        
        preprocessor, postprocessor = create_fast_default_pipelines(config)
        
        # Create test data
        test_images = [np.random.rand(224, 224, 3) * 255 for _ in range(10)]
        
        # Process asynchronously
        start_time = time.time()
        
        # Create async tasks
        tasks = [preprocessor.preprocess_async(img) for img in test_images]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        print(f"Async processed {len(test_images)} images in {end_time - start_time:.4f}s")
        print(f"Async throughput: {len(test_images)/(end_time - start_time):.2f} images/sec")
        
        # Compare with synchronous processing
        start_time = time.time()
        sync_results = preprocessor.preprocess_batch(test_images)
        end_time = time.time()
        
        print(f"Sync processed {len(test_images)} images in {end_time - start_time:.4f}s")
        print(f"Sync throughput: {len(test_images)/(end_time - start_time):.2f} images/sec")
    
    # Run the async demo
    asyncio.run(async_processing_demo())


if __name__ == "__main__":
    import time
    
    print("Fast Processors Performance Examples")
    print("=====================================")
    
    try:
        basic_fast_processor_example()
        production_optimization_example()
        real_time_optimization_example()
        memory_optimization_example()
        benchmarking_example()
        custom_optimization_example()
        async_processing_example()
        
        print("\n=== All examples completed successfully! ===")
        print("\nKey takeaways:")
        print("- Use create_fast_default_pipelines() for general purpose")
        print("- Use FastProcessorFactory for specific optimization targets")
        print("- Use benchmark_processor_performance() to measure improvements")
        print("- Enable async processing for better throughput")
        print("- Configure optimization mode based on your use case")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

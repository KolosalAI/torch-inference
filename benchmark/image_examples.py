#!/usr/bin/env python3
"""
Image Model Benchmark Examples - Demonstrate image model benchmarking capabilities.

This script provides examples of how to use the image benchmarking system
with different configurations and scenarios.
"""

import logging
import random
import time
from typing import Any, Dict

from benchmark.harness import ImageBenchmarkHarness, BenchmarkConfig, generate_image_prompts


def simple_image_model(
    prompt: str,
    width: int = 512,
    height: int = 512,
    num_images: int = 1,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    **kwargs
) -> bytes:
    """
    Simple demo image generation function.
    
    Simulates a basic diffusion model with realistic timing patterns.
    """
    # Simulate processing time based on complexity
    base_time = 0.8
    resolution_factor = (width * height) / (512 * 512)
    steps_factor = num_inference_steps / 50
    images_factor = num_images
    
    total_time = base_time * resolution_factor * steps_factor * images_factor
    
    # Add some variation
    actual_time = total_time * (0.7 + random.random() * 0.6)
    time.sleep(actual_time)
    
    # Return dummy image data
    image_size = width * height * 3  # RGB
    return b'PNG_DUMMY_DATA' * (image_size // 15)


def fast_image_model(
    prompt: str,
    width: int = 512,
    height: int = 512,
    num_images: int = 1,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    **kwargs
) -> bytes:
    """
    Fast demo image generation function.
    
    Simulates a faster model optimized for speed.
    """
    # Much faster processing
    base_time = 0.2
    resolution_factor = (width * height) / (512 * 512)
    steps_factor = num_inference_steps / 20
    
    total_time = base_time * resolution_factor * steps_factor * num_images
    actual_time = total_time * (0.8 + random.random() * 0.4)
    time.sleep(actual_time)
    
    image_size = width * height * 3
    return b'FAST_PNG_DATA' * (image_size // 15)


def quality_image_model(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    num_images: int = 1,
    num_inference_steps: int = 100,
    guidance_scale: float = 10.0,
    **kwargs
) -> bytes:
    """
    High-quality demo image generation function.
    
    Simulates a high-quality model with longer processing times.
    """
    # Higher quality = longer processing
    base_time = 2.0
    resolution_factor = (width * height) / (512 * 512)
    steps_factor = num_inference_steps / 50
    
    total_time = base_time * resolution_factor * steps_factor * num_images
    actual_time = total_time * (0.9 + random.random() * 0.2)
    time.sleep(actual_time)
    
    image_size = width * height * 3
    return b'QUALITY_PNG_DATA' * (image_size // 15)


def example_basic_benchmark():
    """Run a basic image model benchmark."""
    print("=" * 60)
    print("Example 1: Basic Image Model Benchmark")
    print("=" * 60)
    
    # Configure benchmark for quick demo
    config = BenchmarkConfig(
        model_type="image",
        concurrency_levels=[1, 2, 4],
        iterations_per_level=10,
        text_variations=3,
        output_dir="examples/basic_image_benchmark",
        generate_detailed_csv=True,
        generate_summary_csv=True,
        generate_plots=True,
        image_width=512,
        image_height=512,
        num_images=1,
        num_inference_steps=25,
        guidance_scale=7.5
    )
    
    # Create test prompts
    test_prompts = [
        "A beautiful landscape with mountains and a sunset",
        "A cute cat sitting in a garden, photorealistic",
        "A futuristic city skyline, cyberpunk style"
    ]
    
    # Run benchmark
    harness = ImageBenchmarkHarness(config)
    results = harness.run_benchmark(
        simple_image_model,
        benchmark_name="basic_image_example",
        test_prompts=test_prompts
    )
    
    return results


def example_speed_comparison():
    """Compare different image model speeds."""
    print("=" * 60)
    print("Example 2: Speed Comparison Benchmark")
    print("=" * 60)
    
    # Test fast model
    config_fast = BenchmarkConfig(
        model_type="image",
        concurrency_levels=[1, 2, 4, 8],
        iterations_per_level=15,
        text_variations=5,
        output_dir="examples/speed_comparison",
        generate_detailed_csv=True,
        generate_summary_csv=True,
        generate_plots=True,
        image_width=512,
        image_height=512,
        num_images=1,
        num_inference_steps=20,
        guidance_scale=5.0
    )
    
    harness_fast = ImageBenchmarkHarness(config_fast)
    fast_results = harness_fast.run_benchmark(
        fast_image_model,
        benchmark_name="fast_model_comparison"
    )
    
    # Test standard model
    config_standard = BenchmarkConfig(
        model_type="image",
        concurrency_levels=[1, 2, 4, 8],
        iterations_per_level=15,
        text_variations=5,
        output_dir="examples/speed_comparison",
        generate_detailed_csv=True,
        generate_summary_csv=True,
        generate_plots=True,
        image_width=512,
        image_height=512,
        num_images=1,
        num_inference_steps=50,
        guidance_scale=7.5
    )
    
    harness_standard = ImageBenchmarkHarness(config_standard)
    standard_results = harness_standard.run_benchmark(
        simple_image_model,
        benchmark_name="standard_model_comparison"
    )
    
    # Print comparison
    print("\nSpeed Comparison Results:")
    print("-" * 40)
    print("Fast Model (20 steps):")
    for conc, result in fast_results.items():
        metrics = result.metrics
        print(f"  Concurrency {conc}: IPS={metrics.ips:.3f}, RPS={metrics.rps:.1f}")
    
    print("\nStandard Model (50 steps):")
    for conc, result in standard_results.items():
        metrics = result.metrics
        print(f"  Concurrency {conc}: IPS={metrics.ips:.3f}, RPS={metrics.rps:.1f}")
    
    return fast_results, standard_results


def example_resolution_benchmark():
    """Benchmark different image resolutions."""
    print("=" * 60)
    print("Example 3: Resolution Impact Benchmark")
    print("=" * 60)
    
    resolutions = [
        (256, 256, "256x256"),
        (512, 512, "512x512"),
        (768, 768, "768x768"),
        (1024, 1024, "1024x1024")
    ]
    
    results_by_resolution = {}
    
    for width, height, name in resolutions:
        print(f"\nTesting {name} resolution...")
        
        config = BenchmarkConfig(
            model_type="image",
            concurrency_levels=[1, 2, 4],
            iterations_per_level=8,
            text_variations=3,
            output_dir=f"examples/resolution_benchmark",
            generate_detailed_csv=True,
            generate_summary_csv=True,
            generate_plots=False,  # Skip plots for this example
            image_width=width,
            image_height=height,
            num_images=1,
            num_inference_steps=30,
            guidance_scale=7.5
        )
        
        harness = ImageBenchmarkHarness(config)
        results = harness.run_benchmark(
            simple_image_model,
            benchmark_name=f"resolution_{name}",
            image_params={
                'width': width,
                'height': height,
                'num_images': 1,
                'num_inference_steps': 30,
                'guidance_scale': 7.5
            }
        )
        
        results_by_resolution[name] = results
    
    # Print resolution comparison
    print("\nResolution Impact Summary:")
    print("-" * 50)
    for res_name, results in results_by_resolution.items():
        best_result = max(results.values(), key=lambda r: r.metrics.ips)
        metrics = best_result.metrics
        print(f"{res_name:>10}: Best IPS={metrics.ips:.3f}, PPS={metrics.pps:.0f}")
    
    return results_by_resolution


def example_custom_prompts():
    """Example with custom image generation prompts."""
    print("=" * 60)
    print("Example 4: Custom Prompts Benchmark")
    print("=" * 60)
    
    # Custom art-focused prompts
    art_prompts = [
        "A serene Japanese garden with cherry blossoms, traditional ink painting style",
        "Abstract geometric patterns in vibrant colors, modern digital art",
        "Portrait of a renaissance nobleman, oil painting, chiaroscuro lighting",
        "Surreal landscape with floating islands, concept art for fantasy game",
        "Minimalist architecture, concrete and glass, brutalist style, black and white"
    ]
    
    config = BenchmarkConfig(
        model_type="image",
        concurrency_levels=[1, 2, 4],
        iterations_per_level=12,
        output_dir="examples/custom_prompts",
        generate_detailed_csv=True,
        generate_summary_csv=True,
        generate_plots=True,
        image_width=768,
        image_height=768,
        num_images=1,
        num_inference_steps=40,
        guidance_scale=8.0
    )
    
    harness = ImageBenchmarkHarness(config)
    results = harness.run_benchmark(
        simple_image_model,
        benchmark_name="custom_art_prompts",
        test_prompts=art_prompts
    )
    
    return results


def run_all_examples():
    """Run all image benchmark examples."""
    print("Image Model Benchmarking Examples")
    print("=" * 60)
    print("This script demonstrates various image benchmarking scenarios")
    print()
    
    try:
        # Example 1: Basic benchmark
        example_basic_benchmark()
        print("\n" + "="*60 + "\n")
        
        # Example 2: Speed comparison
        example_speed_comparison()
        print("\n" + "="*60 + "\n")
        
        # Example 3: Resolution impact
        example_resolution_benchmark()
        print("\n" + "="*60 + "\n")
        
        # Example 4: Custom prompts
        example_custom_prompts()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("Check the 'examples/' directory for generated CSV files and plots.")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    run_all_examples()

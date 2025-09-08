# Image Model Benchmarking System

The Image Model Benchmarking System provides comprehensive performance testing for image generation models with standardized metrics and detailed reporting.

## Overview

This system extends the existing TTS benchmark framework to support image models with metrics specifically designed for image generation workloads:

### Key Metrics

- **IPS (Images Per Second)** - Primary throughput metric for image generation
- **PPS (Pixels Per Second)** - Throughput considering image resolution  
- **SPS (Steps Per Second)** - Inference steps throughput for diffusion models
- **TTFI (Time To First Image)** - Latency metrics with percentile analysis
- **RPS (Requests Per Second)** - Standard request throughput
- **Memory Usage** - Peak and GPU memory monitoring

## Quick Start

### Basic Image Benchmark

```python
from benchmark import ImageBenchmarkHarness, BenchmarkConfig

# Configure benchmark
config = BenchmarkConfig(
    model_type="image",
    concurrency_levels=[1, 2, 4, 8],
    iterations_per_level=50,
    image_width=512,
    image_height=512,
    num_inference_steps=20,
    generate_detailed_csv=True,
    generate_summary_csv=True
)

# Your image generation function
def my_image_model(prompt, **kwargs):
    # Your image generation logic here
    # Return Dict[str, Any] with image info
    pass

# Run benchmark
harness = ImageBenchmarkHarness(config)
results = harness.run_benchmark(my_image_model, "my_image_benchmark")
```

### Command Line Demo

```bash
# Run TTS benchmark demo
python -m benchmark.harness

# Run Image benchmark demo  
python -m benchmark.harness image

# Run image benchmark examples
python -m benchmark.image_examples

# Run tests
python -m benchmark.test_image_benchmark
```

## Image Model Function Interface

Your image generation function should have this signature:

```python
def image_model_function(
    prompt: str,
    width: int = 512,
    height: int = 512,
    num_images: int = 1,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    **kwargs
) -> Union[Dict[str, Any], bytes]:
    """
    Expected return format (Dict):
    {
        'images': [image_array1, image_array2, ...],  # PIL Images or numpy arrays
        'width': int,
        'height': int,
        'num_images': int,
        'num_inference_steps': int,
        'guidance_scale': float,
        'seed': int (optional),
        'format': str (optional, default 'PNG')
    }
    
    Or return raw image bytes for simple cases.
    """
    pass
```

## Configuration Options

### BenchmarkConfig for Image Models

```python
config = BenchmarkConfig(
    # Model type
    model_type="image",                    # "tts" or "image"
    
    # Test parameters
    concurrency_levels=[1, 2, 4, 8, 16],  # Concurrency levels to test
    iterations_per_level=100,              # Iterations per concurrency level
    warmup_requests=3,                     # Warmup requests
    timeout_seconds=30.0,                  # Request timeout
    
    # Image generation parameters
    image_width=512,                       # Default image width
    image_height=512,                      # Default image height
    num_images=1,                          # Images per request
    num_inference_steps=50,                # Diffusion model steps
    guidance_scale=7.5,                    # CFG scale
    
    # Test data
    min_text_length=20,                    # Min prompt length
    max_text_length=100,                   # Max prompt length
    text_variations=20,                    # Number of test prompts
    
    # Output configuration
    output_dir="benchmark_results",        # Output directory
    generate_detailed_csv=True,            # Individual request data
    generate_summary_csv=True,             # Aggregated statistics
    generate_plots=True,                   # Performance plots
    save_raw_data=True                     # Raw JSON data
)
```

## Output Files

The benchmark generates several output files:

### CSV Reports

1. **Detailed CSV** (`*_detailed.csv`)
   - Individual request metrics
   - 35+ columns with comprehensive data
   - Includes prompt text, timing, image parameters
   - Suitable for detailed analysis

2. **Summary CSV** (`*_summary.csv`)
   - Aggregated statistics per concurrency level
   - Performance summary with percentiles
   - Best performance analysis
   - Test configuration details

### Visualizations

1. **Throughput Plot** (`*_throughput.png`)
   - IPS, PPS, and RPS vs concurrency
   - Performance scaling visualization

2. **Latency Plot** (`*_latency.png`)
   - TTFI percentiles (p50, p95, p99)
   - Latency analysis across concurrency levels

### Raw Data

1. **Raw JSON** (`*_raw.json`)
   - Complete benchmark data
   - Configuration and results
   - For custom analysis and comparison

## Advanced Usage

### Custom Test Prompts

```python
custom_prompts = [
    "A beautiful landscape with mountains, photorealistic",
    "Abstract art with geometric patterns, vibrant colors",
    "Portrait of a wizard, fantasy art style",
    "Cyberpunk city at night, neon lights"
]

results = harness.run_benchmark(
    my_image_model,
    benchmark_name="custom_prompts",
    test_prompts=custom_prompts
)
```

### Custom Image Parameters

```python
image_params = {
    'width': 1024,
    'height': 1024,
    'num_images': 4,
    'num_inference_steps': 100,
    'guidance_scale': 12.0
}

results = harness.run_benchmark(
    my_image_model,
    benchmark_name="high_quality",
    image_params=image_params
)
```

### Multiple Benchmark Comparison

```python
# Run multiple benchmarks
fast_config = BenchmarkConfig(num_inference_steps=20)
quality_config = BenchmarkConfig(num_inference_steps=100)

harness_fast = ImageBenchmarkHarness(fast_config)
harness_quality = ImageBenchmarkHarness(quality_config)

fast_results = harness_fast.run_benchmark(my_model, "fast_model")
quality_results = harness_quality.run_benchmark(my_model, "quality_model")

# Compare results
from benchmark import ImageBenchmarkReporter
reporter = ImageBenchmarkReporter()

comparison_csv = reporter.generate_comparison_csv({
    "fast": fast_results,
    "quality": quality_results
})
```

## Understanding Metrics

### Primary Throughput Metrics

- **IPS (Images Per Second)**: Images generated per second across all concurrent requests
- **PPS (Pixels Per Second)**: Total pixels generated per second (accounts for resolution)
- **SPS (Steps Per Second)**: Inference steps executed per second (for diffusion models)

### Latency Metrics

- **TTFI (Time To First Image)**: Time from request start to first image generation
- **TTFI p50/p95/p99**: 50th, 95th, and 99th percentile latencies
- **Wall Time**: Total time from request start to completion

### Resource Metrics

- **Memory Peak**: Maximum memory usage during generation
- **GPU Memory**: GPU memory utilization (if available)

## Performance Analysis Tips

1. **Concurrency Scaling**: Look for the optimal concurrency level where throughput peaks
2. **Memory Usage**: Monitor memory growth with concurrency to avoid OOM issues
3. **Latency vs Throughput**: Balance between high throughput and acceptable latency
4. **Resolution Impact**: Test different resolutions to understand performance scaling
5. **Steps vs Quality**: Evaluate inference steps vs generation speed tradeoffs

## Integration Examples

### With Diffusion Models

```python
def stable_diffusion_benchmark(prompt, **kwargs):
    # Initialize your model
    pipe = StableDiffusionPipeline.from_pretrained("model_name")
    
    # Generate images
    images = pipe(
        prompt=prompt,
        width=kwargs.get('width', 512),
        height=kwargs.get('height', 512),
        num_images_per_prompt=kwargs.get('num_images', 1),
        num_inference_steps=kwargs.get('num_inference_steps', 50),
        guidance_scale=kwargs.get('guidance_scale', 7.5)
    ).images
    
    return {
        'images': images,
        'width': images[0].width,
        'height': images[0].height,
        'num_images': len(images),
        'format': 'PNG'
    }
```

### With API Services

```python
def api_image_service_benchmark(prompt, **kwargs):
    import requests
    
    response = requests.post('http://your-api/generate', json={
        'prompt': prompt,
        'width': kwargs.get('width', 512),
        'height': kwargs.get('height', 512),
        'steps': kwargs.get('num_inference_steps', 50)
    })
    
    # Return image data
    return response.content  # Returns bytes
```

## Error Handling

The benchmark system handles common errors gracefully:

- **Timeout errors**: Requests exceeding timeout are marked as failed
- **Generation failures**: Failed requests are tracked in metrics
- **Memory errors**: System continues with available requests
- **Invalid outputs**: Warns about unexpected return types but continues

## Best Practices

1. **Warmup**: Always include warmup requests for accurate measurements
2. **Test Variety**: Use diverse prompts to get representative performance
3. **Resource Monitoring**: Monitor system resources during benchmarks
4. **Reproducibility**: Use fixed seeds when possible for consistent results
5. **Documentation**: Document your model configuration and benchmark settings

## Troubleshooting

### Common Issues

1. **"Image function returned unexpected type"**: Check your function returns Dict or bytes
2. **Memory errors**: Reduce concurrency levels or image resolution
3. **Timeout errors**: Increase timeout_seconds for slow models
4. **Empty results**: Check that your function doesn't raise exceptions

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This comprehensive system provides standardized benchmarking for image generation models with detailed analysis and reporting capabilities.

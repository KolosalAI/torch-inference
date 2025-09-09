# Benchmark Suite for torch-inference

A comprehensive benchmarking system for AI models including Text-to-Speech (TTS) and Image Classification models. The system implements industry-standard metrics and provides high-precision performance measurement capabilities.

## Overview

This benchmark suite provides:
- **Multi-model support**: TTS, Image Classification (ResNet), and other AI models
- **Industry-standard metrics**: ASPS, RTF, RPS, CPS, TTFA (TTS) / IPS, Classifications/sec (Image)
- **Multiple concurrency levels**: Test scalability from 1 to N concurrent requests
- **HTTP client support**: Benchmark remote AI model servers via REST API
- **Streaming support**: Measure streaming vs non-streaming performance (TTS)
- **Comprehensive reporting**: CSV outputs, comparison tables, and plots
- **Quick testing functions**: Built-in validation and stress testing
- **Statistical rigor**: Proper warmup, timing, and percentile analysis

## Core Metrics

### TTS Metrics

#### Primary Metric: ASPS (Audio Seconds Per Second)
**ASPS** is the cleanest throughput metric for TTS:
```
ASPS = sum(audio_duration_i) / T_wall
```
- **Higher ASPS is better**
- **Relation to RTF**: `ASPS = 1/RTF`

#### Supporting TTS Metrics
- **RTF (Real Time Factor)**: `T_synthesis / audio_duration` (lower is better)
- **RPS (Requests/sec)**: `N / T_wall` (report with median input length)
- **CPS (Characters/sec)**: `sum(chars) / T_wall` (normalizes for varying prompt length)
- **TTFA (Time To First Audio)**: Latency metric; essential for streaming UX (p50/p95/p99)

### Image Classification Metrics

#### Primary Metric: IPS (Images Per Second)
**IPS** is the primary throughput metric for image classification:
```
IPS = N_images / T_wall
```
- **Higher IPS is better**

#### Supporting Image Metrics
- **Classifications/sec**: Similar to IPS for classification tasks
- **RPS (Requests/sec)**: Request throughput
- **Latency (TTFI)**: Time to First Image result (p50/p95/p99)
- **Success Rate**: Percentage of successful classifications

## Quick Start

### 1. Quick Testing Functions

The benchmark system includes built-in quick test functions for immediate validation:

```bash
# Quick ResNet classification test (5 iterations)
python -c "from benchmark.resnet_image_benchmark import quick_resnet_test; quick_resnet_test()"

# Quick stress test (30-second load test)
python -c "from benchmark.resnet_image_benchmark import quick_stress_test; quick_stress_test()"
```

### 2. Image Classification Benchmark

```python
from benchmark.resnet_image_benchmark import ResNetImageBenchmarker, create_demo_resnet_function

# Create benchmarker
benchmarker = ResNetImageBenchmarker(
    default_width=224,
    default_height=224,
    warmup_requests=5
)

# Create demo classification function (or use your own)
demo_classifier = create_demo_resnet_function()

# Run benchmark
results = benchmarker.benchmark_resnet_model(
    classification_function=demo_classifier,
    concurrency_levels=[1, 2, 4, 8],
    iterations_per_level=50
)
```

### 3. TTS Benchmark

```python
# Example TTS HTTP benchmark setup
from benchmark.http_client import create_torch_inference_tts_function
from benchmark.harness import TTSBenchmarkHarness, BenchmarkConfig

config = BenchmarkConfig(
    concurrency_levels=[1, 2, 4],
    iterations_per_level=50,
    output_dir="my_benchmark_results"
)

tts_function = create_torch_inference_tts_function(
    base_url="http://your-tts-server:8000",
    voice="premium_voice",
    streaming=False
)

harness = TTSBenchmarkHarness(config)
results = harness.run_benchmark(tts_function)
```

### 4. HTTP Server Benchmark (ResNet)

```python
from benchmark.resnet_image_benchmark import create_resnet_classification_function, ResNetImageBenchmarker

# Create classification function that calls your server
resnet_function = create_resnet_classification_function(
    model_name="resnet18",
    base_url="http://localhost:8000",
    top_k=5
)

# Run benchmark
benchmarker = ResNetImageBenchmarker()
results = benchmarker.benchmark_resnet_model(
    classification_function=resnet_function,
    concurrency_levels=[1, 2, 4, 8, 16],
    iterations_per_level=50
)
```

### 5. Stress Testing

```python
from benchmark.resnet_image_benchmark import ResNetImageBenchmarker, create_demo_resnet_function

benchmarker = ResNetImageBenchmarker(monitor_memory=True)
demo_classifier = create_demo_resnet_function()

# Run comprehensive stress test
stress_results = benchmarker.stress_test_resnet_model(
    classification_function=demo_classifier,
    duration_minutes=5,
    max_concurrency=64,
    ramp_up_seconds=30
)
```

## API Usage

### TTS Benchmarking

```python
from benchmark.harness import TTSBenchmarkHarness, BenchmarkConfig

# Configure benchmark
config = BenchmarkConfig(
    concurrency_levels=[1, 2, 4, 8, 16, 32, 64],
    iterations_per_level=20,
    sample_rate=22050,
    output_dir="my_benchmark_results"
)

# Create harness
harness = TTSBenchmarkHarness(config)

# Define your TTS function
def my_tts_function(text: str, **kwargs) -> dict:
    # Your TTS implementation
    audio_duration = synthesize_speech(text)
    return {
        'audio_duration': audio_duration,
        'sample_rate': 22050,
        'text_tokens': len(text.split())
    }

# Run benchmark
results = harness.run_benchmark(my_tts_function, benchmark_name="my_tts")

# Print results
from benchmark.reporter import TTSBenchmarkReporter
reporter = TTSBenchmarkReporter()
reporter.print_summary_table(results)
```

### Image Classification Benchmarking

```python
from benchmark.resnet_image_benchmark import ResNetImageBenchmarker

# Create benchmarker
benchmarker = ResNetImageBenchmarker(
    default_width=224,
    default_height=224,
    warmup_requests=5,
    monitor_memory=True
)

# Define your classification function
def my_classifier(image_data: bytes, **kwargs) -> dict:
    # Your classification implementation
    predictions = classify_image(image_data)
    return {
        'success': True,
        'predictions': predictions,
        'processing_time': time.time() - start_time
    }

# Run benchmark
results = benchmarker.benchmark_resnet_model(
    classification_function=my_classifier,
    concurrency_levels=[1, 2, 4, 8],
    iterations_per_level=50
)

# Print results
for concurrency, result in results.items():
    metrics = result.metrics
    print(f"Concurrency {concurrency}: {metrics.ips:.2f} classifications/sec")
```

### Async TTS Functions

```python
import asyncio

async def my_async_tts_function(text: str, **kwargs) -> dict:
    # Your async TTS implementation
    audio_duration = await async_synthesize_speech(text)
    return {
        'audio_duration': audio_duration,
        'sample_rate': 22050
    }

# Run async benchmark
results = asyncio.run(harness.run_benchmark(
    my_async_tts_function,
    benchmark_name="async_tts",
    is_async=True
))
```

### HTTP Servers (TTS and Image)

```python
from benchmark.http_client import create_torch_inference_tts_function
from benchmark.resnet_image_benchmark import create_resnet_classification_function

# TTS HTTP server
tts_function = create_torch_inference_tts_function(
    base_url="http://localhost:8000",
    voice="default",
    streaming=False,
    auth_token="your-token"  # if needed
)

# Image classification HTTP server
classifier_function = create_resnet_classification_function(
    model_name="resnet18",
    base_url="http://localhost:8000",
    top_k=5,
    auth_token="your-token"  # if needed
)

# Benchmark them
import asyncio
tts_results = asyncio.run(harness.run_benchmark(
    tts_function,
    benchmark_name="http_tts",
    is_async=True
))

image_results = benchmarker.benchmark_resnet_model(
    classification_function=classifier_function,
    concurrency_levels=[1, 2, 4, 8],
    iterations_per_level=50
)
```

## Output and Reports

### Enhanced CSV Reports

The benchmark system now generates multiple types of CSV reports for comprehensive analysis:

#### 1. Detailed CSV Report (`*_detailed.csv`)
Contains individual request/iteration data with all benchmark variables:

```csv
Concurrency_Level,Request_ID,Iteration,Text_Input,Text_Length_Chars,Text_Length_Tokens,Start_Time_Sec,First_Audio_Time_Sec,End_Time_Sec,Wall_Time_Sec,TTFA_Sec,Audio_Duration_Sec,RTF,Sample_Rate,Bit_Depth,Success,Error_Message
1,req_0,0,"The quick brown fox jumps over the lazy dog.",43,9,1234567890.123456,1234567890.145678,1234567890.234567,0.111111,0.022222,1.500000,0.074074,22050,16,True,
1,req_1,1,"Hello world, this is a test...",28,7,1234567890.250000,1234567890.268000,1234567890.320000,0.070000,0.018000,1.200000,0.058333,22050,16,True,
```

#### 2. Summary CSV Report (`*_summary.csv`)
Contains aggregated statistics and performance metrics:

```csv
Concurrency,ASPS,RTF_Mean,RTF_Median,RPS,CPS,TTFA_p95_ms,Success_Rate_%
1,2.450,0.408,0.408,12.5,875.2,45.2,100.0
2,4.320,0.462,0.462,22.1,1547.8,52.1,100.0
4,7.890,0.507,0.507,38.9,2724.1,58.7,100.0
8,12.450,0.643,0.643,58.2,4074.8,67.3,98.8
```

#### 3. Legacy CSV Report (`*.csv`)
Backward-compatible format for existing analysis tools.

#### Configuration Options
Control CSV generation through `BenchmarkConfig`:

```python
config = BenchmarkConfig(
    generate_detailed_csv=True,    # Individual request data
    generate_summary_csv=True,     # Aggregated statistics
    save_raw_data=True            # Enable detailed comparison CSVs
)
```

### Performance Plots

When matplotlib is available, the system generates:
- **Throughput curves**: ASPS vs concurrency
- **Latency plots**: TTFA percentiles
- **Comparison charts**: Multiple benchmark comparison

### Console Output

```
TTS Benchmark Results Summary
================================================================================
Conc ASPS     RTF      RPS      CPS      TTFA p95   Success 
--------------------------------------------------------------------------------
1    2.450    0.408    12.5     875      45.2       100.0%
2    4.320    0.462    22.1     1548     52.1       100.0%
4    7.890    0.507    38.9     2724     58.7       100.0%
8    12.450   0.643    58.2     4075     67.3       98.8%
================================================================================
Best Performance:
  Highest ASPS: 12.450 at concurrency 8
  Lowest RTF: 0.408 at concurrency 1
  Highest RPS: 58.2 at concurrency 8
  Lowest TTFA p95: 45.2ms at concurrency 1
```

## Configuration

### TTS BenchmarkConfig Options

```python
from benchmark.harness import BenchmarkConfig

config = BenchmarkConfig(
    # Test configuration
    concurrency_levels=[1, 2, 4, 8, 16, 32, 64],  # Concurrency levels to test
    iterations_per_level=20,               # Requests per level
    warmup_requests=5,                     # Warmup requests
    timeout_seconds=30.0,                  # Request timeout
    
    # Audio configuration (TTS)
    sample_rate=22050,                     # Expected sample rate
    bit_depth=16,                          # Expected bit depth
    
    # Test data
    min_text_length=50,                    # Min chars per test text
    max_text_length=200,                   # Max chars per test text
    text_variations=20,                    # Number of test texts
    
    # Output
    output_dir="benchmark_results",        # Output directory
    save_raw_data=True,                    # Save detailed JSON
    generate_plots=True                    # Generate matplotlib plots
)
```

### ResNet Image Benchmarker Options

```python
from benchmark.resnet_image_benchmark import ResNetImageBenchmarker

benchmarker = ResNetImageBenchmarker(
    default_width=224,                     # Default image width
    default_height=224,                    # Default image height
    warmup_requests=5,                     # Warmup requests
    timeout_seconds=30.0,                  # Request timeout
    monitor_memory=True,                   # Monitor memory usage
    test_images_dir=None                   # Optional: directory with test images
)
```

### Custom Test Texts

```python
custom_texts = [
    "Hello world, this is a test.",
    "The quick brown fox jumps over the lazy dog.",
    "Your custom TTS test phrases here."
]

results = harness.run_benchmark(
    tts_function,
    test_texts=custom_texts,
    benchmark_name="custom_texts"
)
```

## Advanced Usage

### Streaming TTS Support

For streaming TTS where you need to capture Time To First Audio (TTFA):

```python
from benchmark.tts_benchmark import create_streaming_tts_wrapper

def streaming_tts_generator(text: str, **kwargs):
    for chunk in your_streaming_tts(text, **kwargs):
        yield chunk

# Wrap for proper TTFA measurement
wrapped_tts = create_streaming_tts_wrapper(
    streaming_tts_generator,
    chunk_callback=lambda req_id, timestamp: print(f"First chunk for {req_id}")
)
```

### Custom Metrics Validation

```python
from benchmark.metrics import validate_metrics_consistency

# Validate benchmark results
warnings = validate_metrics_consistency(result.metrics)
for warning in warnings:
    print(f"⚠️  {warning}")
```

### Benchmark Comparison

```python
# Run multiple benchmarks
results_v1 = harness.run_benchmark(tts_v1, benchmark_name="model_v1")
results_v2 = harness.run_benchmark(tts_v2, benchmark_name="model_v2")

# Compare them
comparison_data = {
    "model_v1": results_v1,
    "model_v2": results_v2
}

harness.compare_benchmarks(comparison_data, "model_comparison")
```

## Implementation Notes

### For TTS Function Authors

Your TTS function should return a dictionary with:

```python
{
    'audio_duration': 2.5,      # Required: audio length in seconds
    'sample_rate': 22050,       # Recommended: actual sample rate
    'text_tokens': 42,          # Optional: token count for tokens/sec metric
    'response_size_bytes': 1024 # Optional: for bandwidth analysis
}
```

### For Image Classification Function Authors

Your classification function should return a dictionary with:

```python
{
    'success': True,            # Required: whether classification succeeded
    'predictions': [            # Required: list of predictions
        {
            'class': 'golden_retriever',
            'confidence': 0.95,
            'class_id': 207
        }
    ],
    'processing_time': 0.05,    # Optional: processing time in seconds
    'model_name': 'resnet18'    # Optional: model identifier
}
```

### Error Handling

The benchmark system gracefully handles:
- Request timeouts
- Network errors  
- TTS synthesis failures
- Image classification failures
- Model loading errors
- Partial failures in concurrent tests

Failed requests are tracked and reported in success rate metrics.

### Timing Precision

The system uses `time.perf_counter()` for high-precision timing and properly handles:
- CUDA synchronization for GPU models
- Async/await timing boundaries
- Warmup iterations to avoid cold-start bias

## Requirements

### Core Requirements
- Python 3.8+
- `aiohttp` (for HTTP client)
- `dataclasses` (Python 3.7+ or backport)

### Optional Requirements
- `matplotlib` (for plots)
- `numpy` (for enhanced plotting)

### Installation

```bash
# Core functionality
pip install aiohttp

# With plotting support
pip install aiohttp matplotlib numpy
```

## Available Benchmarkers

### Core Benchmarkers

1. **TTSBenchmarkHarness**: Main TTS benchmarking system
   - Location: `benchmark.harness.TTSBenchmarkHarness`
   - Supports: Async/sync TTS functions, streaming, HTTP clients

2. **ImageBenchmarker**: General image model benchmarking
   - Location: `benchmark.image_benchmark.ImageBenchmarker`
   - Supports: Image generation models, diffusion models

3. **ResNetImageBenchmarker**: Specialized for image classification
   - Location: `benchmark.resnet_image_benchmark.ResNetImageBenchmarker`
   - Supports: Classification models, ResNet variants, stress testing
   - Includes: Built-in quick test functions

### Quick Test Functions

The ResNet benchmarker includes built-in validation functions:

```python
from benchmark.resnet_image_benchmark import quick_resnet_test, quick_stress_test

# Fast validation (5 iterations)
quick_resnet_test()

# Stress test (30 seconds, up to 16 concurrent requests)
quick_stress_test()
```

Command line usage:
```bash
# Quick ResNet benchmark test
python -c "from benchmark.resnet_image_benchmark import quick_resnet_test; quick_resnet_test()"

# Quick stress test
python -c "from benchmark.resnet_image_benchmark import quick_stress_test; quick_stress_test()"
```

## Best Practices

### General
1. **Use sufficient iterations** (≥10) for statistical significance
2. **Warm up your models** before timing measurements
3. **Monitor resource usage** during high-concurrency tests
4. **Compare steady-state performance** after initial startup costs
5. **Run across multiple concurrency levels** to find optimal throughput

### TTS Specific
1. **Keep audio settings consistent** across tests (sample rate, bit depth)
2. **Report ASPS as primary metric** with RTF and latency as supporting metrics
3. **Test both streaming and non-streaming** modes when available

### Image Classification Specific
1. **Use consistent image dimensions** across tests (e.g., 224x224 for ResNet)
2. **Report IPS (Images Per Second) as primary metric**
3. **Test with diverse image content** to avoid overfitting to specific patterns
4. **Use quick test functions** for initial validation before full benchmarks

### Stress Testing
1. **Start with quick stress tests** (30 seconds) before longer runs
2. **Monitor memory usage** to detect memory leaks
3. **Gradually increase load** with proper ramp-up periods
4. **Validate system stability** under sustained load

## Module Overview

### Core Modules

- **`harness.py`**: Main TTS benchmark harness and configuration
- **`tts_benchmark.py`**: TTS-specific benchmarking utilities
- **`image_benchmark.py`**: General image model benchmarking
- **`resnet_image_benchmark.py`**: ResNet classification benchmarking with quick tests
- **`http_client.py`**: HTTP client for remote model servers

### Metrics and Reporting

- **`metrics.py`**: TTS metrics calculation and validation
- **`image_metrics.py`**: Image benchmarking metrics
- **`reporter.py`**: TTS benchmark reporting
- **`image_reporter.py`**: Image benchmark reporting

### Testing

- **`test_benchmark.py`**: TTS benchmark system tests
- **`test_image_benchmark.py`**: Image benchmark system tests

## Contributing

To extend this benchmark suite:

1. **New metrics**: Add to appropriate metrics modules (`metrics.py`, `image_metrics.py`)
2. **New model types**: Create new benchmarker classes following existing patterns
3. **New clients**: Implement in style of `http_client.py`
4. **New reports**: Extend reporting modules with additional visualizations
5. **New quick tests**: Add to existing benchmarker classes or create new ones

The modular design makes it easy to add support for new AI model types, metrics, or analysis approaches.

### Adding New Model Types

To add support for a new model type (e.g., LLM benchmarking):

1. Create a new benchmarker class (e.g., `LLMBenchmarker`)
2. Define appropriate metrics for your model type
3. Implement model-specific test functions
4. Add quick test functions for validation
5. Create appropriate reporter classes for visualization

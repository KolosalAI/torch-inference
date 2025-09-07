# TTS Benchmark Suite for torch-inference

A comprehensive benchmarking system for Text-to-Speech (TTS) models implementing industry-standard metrics including **ASPS (Audio Seconds Per Second)**, the primary throughput metric for TTS systems.

## Overview

This benchmark suite provides:
- **Industry-standard metrics**: ASPS, RTF, RPS, CPS, and TTFA
- **Multiple concurrency levels**: Test scalability from 1 to N concurrent requests
- **HTTP client support**: Benchmark remote TTS servers via REST API
- **Streaming support**: Measure streaming vs non-streaming performance
- **Comprehensive reporting**: CSV outputs, comparison tables, and plots
- **Statistical rigor**: Proper warmup, timing, and percentile analysis

## Core Metrics

### Primary Metric: ASPS (Audio Seconds Per Second)

**ASPS** is the cleanest throughput metric for TTS:

```
ASPS = sum(audio_duration_i) / T_wall
```

Where:
- `audio_duration_i` = length of the i-th synthesized waveform (in seconds)
- `T_wall` = wall-clock time from first request start to last audio sample produced
- **Higher ASPS is better**
- **Relation to RTF**: `ASPS = 1/RTF`

### Supportive Metrics

- **RTF (Real Time Factor)**: `T_synthesis / audio_duration` (lower is better)
- **RPS (Requests/sec)**: `N / T_wall` (report with median input length)
- **CPS (Characters/sec)**: `sum(chars) / T_wall` (normalizes for varying prompt length)
- **TTFA (Time To First Audio)**: Latency metric; essential for streaming UX (p50/p95/p99)

## Quick Start

### 1. Demo Benchmark (Synthetic TTS)

```python
from benchmark.examples import run_demo_benchmark

# Run demo with synthetic TTS function
results = run_demo_benchmark()
```

Or via CLI:
```bash
python benchmark/examples.py demo
```

### 2. HTTP Server Benchmark

```python
import asyncio
from benchmark.examples import run_http_server_benchmark

# Benchmark your TTS server
results = asyncio.run(run_http_server_benchmark(
    server_url="http://localhost:8000",
    voice="default",
    streaming=False
))
```

Or via CLI:
```bash
# Basic benchmark
python benchmark/examples.py http --url http://localhost:8000

# With custom settings
python benchmark/examples.py http \
    --url http://your-tts-server:8000 \
    --voice "premium_voice" \
    --streaming \
    --concurrency 1 2 4 8 16 \
    --iterations 20
```

### 3. Voice Model Comparison

```python
import asyncio
from benchmark.examples import run_voice_comparison_benchmark

results = asyncio.run(run_voice_comparison_benchmark(
    server_url="http://localhost:8000",
    voices=["default", "premium", "fast"]
))
```

Or via CLI:
```bash
python benchmark/examples.py voices \
    --url http://localhost:8000 \
    --voices default premium fast
```

## API Usage

### Basic Benchmarking

```python
from benchmark import TTSBenchmarker, BenchmarkConfig
from benchmark.harness import TTSBenchmarkHarness

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

### HTTP TTS Servers

```python
from benchmark.http_client import create_torch_inference_tts_function

# Create TTS function for your server
tts_function = create_torch_inference_tts_function(
    base_url="http://localhost:8000",
    voice="default",
    streaming=False,
    auth_token="your-token"  # if needed
)

# Benchmark it
results = asyncio.run(harness.run_benchmark(
    tts_function,
    benchmark_name="http_tts",
    is_async=True
))
```

## Output and Reports

### CSV Reports

Each benchmark run generates a comprehensive CSV report:

```csv
Concurrency,ASPS,RTF_Mean,RTF_Median,RPS,CPS,TTFA_p95_ms,Success_Rate_%
1,2.450,0.408,0.408,12.5,875.2,45.2,100.0
2,4.320,0.462,0.462,22.1,1547.8,52.1,100.0
4,7.890,0.507,0.507,38.9,2724.1,58.7,100.0
8,12.450,0.643,0.643,58.2,4074.8,67.3,98.8
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

### BenchmarkConfig Options

```python
config = BenchmarkConfig(
    # Test configuration
    concurrency_levels=[1, 2, 4, 8, 16, 32, 64],  # Concurrency levels to test
    iterations_per_level=20,               # Requests per level
    warmup_requests=5,                     # Warmup requests
    timeout_seconds=30.0,                  # Request timeout
    
    # Audio configuration
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

### Error Handling

The benchmark system gracefully handles:
- Request timeouts
- Network errors  
- TTS synthesis failures
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

## Examples Directory

The `examples.py` script provides ready-to-use benchmark scenarios:

1. **Demo Benchmark**: Synthetic TTS for testing the system
2. **HTTP Server**: Benchmark any REST API TTS server
3. **Voice Comparison**: Compare multiple voice models
4. **Streaming vs Non-Streaming**: Performance comparison

Run any example:
```bash
python benchmark/examples.py <command> --help
```

## Best Practices

1. **Keep audio settings consistent** across tests (sample rate, bit depth)
2. **Run across multiple concurrency levels** to find optimal throughput
3. **Report ASPS as primary metric** with RTF and latency as supporting metrics
4. **Use sufficient iterations** (≥10) for statistical significance
5. **Warm up your models** before timing measurements
6. **Monitor resource usage** during high-concurrency tests
7. **Compare steady-state performance** after initial startup costs

## Contributing

To extend this benchmark suite:

1. **New metrics**: Add to `metrics.py` and update aggregation
2. **New clients**: Implement in style of `http_client.py`
3. **New reports**: Extend `reporter.py` with additional visualizations
4. **New scenarios**: Add to `examples.py` or create new example scripts

The modular design makes it easy to add support for new TTS systems, metrics, or analysis approaches.

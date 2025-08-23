# Performance Testing Tools

A comprehensive performance testing tool for the PyTorch Inference Server with **automatic model downloading** and detailed performance analysis.

## 🚀 Quick Start

### One-Command Testing (Auto-Download)
```bash
# Test ResNet-50 (auto-downloads if needed)
python tools/performance_test.py --model resnet50 --duration 30 --concurrency 4

# Test BERT (auto-downloads if needed) 
python tools/performance_test.py --model bert --duration 60 --concurrency 2

# Test MobileNet with specific requests
python tools/performance_test.py --model mobilenet --requests 1000 --batch-size 8
```

### Quick Demo
```bash
# Run interactive demo with multiple models
python tools/quick_test_demo.py
```

## ✨ Key Features

- **🔄 Auto-Download**: Just specify model name - downloads automatically
- **🌐 Multiple Sources**: TorchVision, Hugging Face models supported
- **🧠 Smart Input**: Auto-generates appropriate input for each model type
- **📊 Rich Reports**: Detailed latency, throughput, and error analysis
- **⚡ Concurrent Testing**: Multi-threaded performance evaluation
- **📦 Batch Support**: Test with different batch sizes
- **🎯 Multiple Formats**: Text, JSON, CSV output

## 📋 Available Models (Auto-Download)

### Vision Models (TorchVision)
```bash
# Small Models (~14-100MB)
--model mobilenet          # MobileNet V2 (~14MB)
--model mobilenet_v3_small # MobileNet V3 Small (~10MB)
--model resnet18           # ResNet-18 (~45MB)

# Medium Models (~100-300MB)  
--model resnet50           # ResNet-50 (~98MB)
--model efficientnet_b0    # EfficientNet B0 (~20MB)
--model densenet121        # DenseNet-121 (~30MB)

# Large Models (300MB+)
--model resnet152          # ResNet-152 (~230MB)
--model densenet169        # DenseNet-169 (~55MB)
--model vgg16              # VGG-16 (~528MB)
```

### Text Models (Hugging Face)
```bash
# Small Models (~250-400MB)
--model distilbert         # DistilBERT Base (~268MB)
--model electra-small-discriminator  # ELECTRA Small (~55MB)

# Medium Models (~400-600MB)
--model bert               # BERT Base (~440MB)  
--model roberta-base       # RoBERTa Base (~500MB)
--model albert-base-v2     # ALBERT Base (~45MB)

# Large Models (1GB+)
--model bert-large-uncased # BERT Large (~1.3GB)
--model roberta-large      # RoBERTa Large (~1.4GB)
```

### Simple Aliases
```bash
--model resnet    # → resnet50
--model bert      # → bert-base-uncased  
--model mobilenet # → mobilenet_v2
--model roberta   # → roberta-base
```

Ready to test? Try: `python tools/performance_test.py --model resnet50 --duration 30`

## Features

- **Model Download Integration**: Automatically download predefined models (Small, Medium, Large)
- **Concurrency Testing**: Test with multiple concurrent threads
- **Batch Processing**: Test with various batch sizes
- **Multiple Output Formats**: Text, JSON, CSV reports
- **Comprehensive Metrics**: Latency percentiles, throughput, error analysis
- **Model-specific Input**: Automatically adapts input data based on model type

## Installation

The tool requires the following dependencies:
```bash
pip install requests numpy aiohttp
```

## Usage

### Basic Usage

```bash
# Test the default example model
python tools/performance_test.py --model example --duration 30

# Test with concurrency
python tools/performance_test.py --model example --duration 30 --concurrency 4

# Test with specific number of requests
python tools/performance_test.py --model example --requests 1000 --batch-size 8
```

### Model Management

```bash
# List available models on server
python tools/performance_test.py --list-models

# List downloadable models
python tools/performance_test.py --list-downloadable

# Download a specific model
python tools/performance_test.py --download mobilenet_v2_small
```

### Advanced Testing

```bash
# Download and test a medium-sized model
python tools/performance_test.py --model resnet50_medium --duration 60 --concurrency 8

# Test with custom input and output format
python tools/performance_test.py --model bert_medium --requests 500 --format json --output results.json

# Stress test with high concurrency
python tools/performance_test.py --model mobilenet_v2_small --duration 120 --concurrency 16 --batch-size 4
```

## Available Models

### Small Models (< 300MB)
- **mobilenet_v2_small**: MobileNet V2 - Efficient CNN (~14MB)
- **distilbert_small**: DistilBERT - Compact BERT variant (~268MB)

### Medium Models (300MB - 500MB)  
- **resnet50_medium**: ResNet-50 - Standard CNN (~98MB)
- **bert_medium**: BERT Base - Standard transformer (~440MB)

### Large Models (> 500MB)
- **resnet152_large**: ResNet-152 - Deep CNN (~230MB)
- **roberta_large**: RoBERTa Large - Large transformer (~1.3GB)

## Output Formats

### Text Format (Default)
```
🎯 PERFORMANCE TEST REPORT - MOBILENET_V2_SMALL
================================================================================

📊 TEST SUMMARY:
   • Test Duration: 30.05s
   • Total Requests: 1,234
   • Successful: 1,230 (99.7%)
   • Failed: 4 (0.3%)
   • Concurrency: 4 threads
   • Batch Size: 1

⚡ THROUGHPUT METRICS:
   • Requests/second: 40.93
   • Predictions/second: 40.93

⏱️  LATENCY METRICS (seconds):
   • Min: 0.0124s
   • Max: 0.2456s
   • Average: 0.0978s
   • Median: 0.0945s
   • 95th percentile: 0.1567s
   • 99th percentile: 0.1889s
```

### JSON Format
```bash
python tools/performance_test.py --model resnet50_medium --duration 30 --format json
```

### CSV Format
```bash
python tools/performance_test.py --model bert_medium --requests 100 --format csv --output results.csv
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | Model to test | `example` |
| `--duration`, `-d` | Test duration in seconds | `30` |
| `--requests`, `-r` | Number of requests to send | - |
| `--concurrency`, `-c` | Number of concurrent threads | `1` |
| `--batch-size`, `-b` | Batch size for each request | `1` |
| `--warmup` | Number of warmup requests | `10` |
| `--timeout` | Request timeout in seconds | `30.0` |
| `--url` | Server URL | `http://localhost:8000` |
| `--input-data` | Custom input data | `42` |
| `--priority` | Request priority | `0` |
| `--format`, `-f` | Output format (text/json/csv) | `text` |
| `--output`, `-o` | Output file | stdout |
| `--verbose`, `-v` | Verbose output | `false` |

## Performance Metrics

The tool provides comprehensive performance analysis:

- **Throughput**: Requests per second, Predictions per second
- **Latency**: Min, Max, Average, Median, 95th, 99th percentiles
- **Error Analysis**: Error rate, Error breakdown by type
- **Resource Usage**: Data transfer, Memory usage (when available)
- **Processing Time**: Server-side vs network overhead

## Examples

### 1. Quick Model Comparison
```bash
# Test small model
python tools/performance_test.py --model mobilenet_v2_small --duration 60 --concurrency 4

# Test medium model
python tools/performance_test.py --model resnet50_medium --duration 60 --concurrency 4

# Test large model  
python tools/performance_test.py --model resnet152_large --duration 60 --concurrency 4
```

### 2. Batch Size Analysis
```bash
# Test different batch sizes
for batch in 1 2 4 8 16; do
    python tools/performance_test.py --model resnet50_medium --requests 100 --batch-size $batch --format csv >> batch_analysis.csv
done
```

### 3. Concurrency Analysis
```bash
# Test different concurrency levels
for conc in 1 2 4 8 16; do
    python tools/performance_test.py --model mobilenet_v2_small --duration 30 --concurrency $conc --format json > "concurrency_${conc}.json"
done
```

### 4. Load Testing
```bash
# High-load stress test
python tools/performance_test.py --model distilbert_small --duration 300 --concurrency 20 --batch-size 8 --verbose
```

## Troubleshooting

### Server Not Running
```
❌ Server health check failed: HTTPConnectionPool(host='localhost', port=8000): Max retries exceeded
```
**Solution**: Start the PyTorch inference server first:
```bash
python main.py
```

### Model Download Fails
```
❌ Download failed (HTTP 500): Internal server error
```
**Solution**: Check server logs and ensure required dependencies (transformers, torchvision) are installed.

### Out of Memory
```
❌ CUDA out of memory
```
**Solution**: Reduce batch size or concurrency:
```bash
python tools/performance_test.py --model resnet152_large --batch-size 1 --concurrency 1
```

## Performance Tips

1. **Start Small**: Begin testing with small models and low concurrency
2. **Monitor Resources**: Watch CPU/GPU/Memory usage during tests
3. **Warmup**: Always include warmup requests for accurate measurements
4. **Baseline**: Establish baseline performance with the example model
5. **Gradual Scaling**: Increase concurrency gradually to find optimal settings

## Integration with CI/CD

Example GitHub Actions workflow:
```yaml
- name: Performance Test
  run: |
    python main.py &
    sleep 10  # Wait for server to start
    python tools/performance_test.py --model mobilenet_v2_small --requests 100 --format json > perf_results.json
    # Add performance regression checks here
```

## Contributing

To add new test models:
1. Add model configuration to `MODEL_CONFIGS` in `performance_test.py`
2. Ensure the model is available via the server's download endpoints
3. Test with various input types and batch sizes
4. Update documentation

## License

This tool is part of the PyTorch Inference Framework and follows the same license terms.

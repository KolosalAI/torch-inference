import torch
import torchvision.models as models
import time
import json
from datetime import datetime

# Check CUDA
print("=" * 60)
print("CUDA + TensorRT Benchmark")
print("=" * 60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("=" * 60)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

# Models to benchmark
model_list = [
    ("resnet50", models.resnet50),
    ("resnet18", models.resnet18),
    ("mobilenet_v3_large", models.mobilenet_v3_large),
    ("efficientnet_b0", models.efficientnet_b0),
]

results = []
batch_size = 1
num_warmup = 20
num_iterations = 100

for model_name, model_fn in model_list:
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    # Load model
    print("Loading model...")
    model = model_fn(weights=None).to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Warmup
    print(f"Warmup ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark ({num_iterations} iterations)...")
    latencies = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            output = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    # Calculate statistics
    import statistics
    avg_latency = statistics.mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    std_latency = statistics.stdev(latencies)
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
    p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
    throughput = 1000 / avg_latency  # FPS
    
    result = {
        "model": model_name,
        "device": str(device),
        "batch_size": batch_size,
        "avg_latency_ms": round(avg_latency, 2),
        "min_latency_ms": round(min_latency, 2),
        "max_latency_ms": round(max_latency, 2),
        "std_latency_ms": round(std_latency, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "throughput_fps": round(throughput, 2),
        "timestamp": datetime.now().isoformat()
    }
    
    results.append(result)
    
    print(f"\nResults:")
    print(f"  Avg Latency: {avg_latency:.2f} ms")
    print(f"  Min Latency: {min_latency:.2f} ms")
    print(f"  Max Latency: {max_latency:.2f} ms")
    print(f"  P95 Latency: {p95:.2f} ms")
    print(f"  P99 Latency: {p99:.2f} ms")
    print(f"  Throughput: {throughput:.2f} FPS")
    
    # Clean up
    del model
    del dummy_input
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Save results
output_file = "benchmark_results/windows_cuda/pytorch_cuda_results.json"
import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("Benchmark Complete!")
print(f"{'='*60}")
print(f"Results saved to: {output_file}")
print("\nSummary:")
for result in results:
    print(f"  {result['model']:20s}: {result['throughput_fps']:7.2f} FPS ({result['avg_latency_ms']:6.2f} ms avg)")

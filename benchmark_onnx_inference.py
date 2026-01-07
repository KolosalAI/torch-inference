#!/usr/bin/env python3
"""
ONNX Runtime Inference Benchmark
Tests actual inference performance with CUDA execution provider
"""

import onnxruntime as ort
import numpy as np
import time
import os
import sys

def download_resnet18_onnx():
    """Download ResNet-18 ONNX model if not exists"""
    import urllib.request
    
    model_path = "models/benchmark/resnet18.onnx"
    os.makedirs("models/benchmark", exist_ok=True)
    
    if not os.path.exists(model_path):
        print("Downloading ResNet-18 ONNX model...")
        url = "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx"
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"Downloaded to {model_path}")
        except Exception as e:
            print(f"Failed to download: {e}")
            print("Please download manually from: {url}")
            return None
    
    return model_path

def get_execution_providers():
    """Get available execution providers"""
    providers = ort.get_available_providers()
    print(f"\nAvailable providers: {providers}")
    return providers

def benchmark_inference(model_path, provider, num_warmup=10, num_iterations=100, batch_size=1):
    """Benchmark inference with given provider"""
    
    print(f"\n{'='*60}")
    print(f"Provider: {provider}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}")
    
    try:
        # Create session with specific provider
        if provider == "CUDAExecutionProvider":
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(model_path, sess_options, providers=[provider])
        elif provider == "TensorrtExecutionProvider":
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            trt_options = {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./tensorrt_cache",
            }
            session = ort.InferenceSession(
                model_path, 
                sess_options, 
                providers=[(provider, trt_options), "CUDAExecutionProvider"]
            )
        else:
            session = ort.InferenceSession(model_path, providers=[provider])
        
        # Get input shape
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape
        
        # Replace batch dimension with actual batch size
        input_shape = [batch_size if isinstance(d, str) or d is None else d for d in input_shape]
        
        print(f"Input: {input_name}, shape: {input_shape}")
        
        # Create random input
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        print(f"Warming up ({num_warmup} iterations)...")
        for _ in range(num_warmup):
            session.run(None, {input_name: input_data})
        
        # Benchmark
        print(f"Benchmarking ({num_iterations} iterations)...")
        
        latencies = []
        start_total = time.perf_counter()
        
        for _ in range(num_iterations):
            start = time.perf_counter()
            session.run(None, {input_name: input_data})
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
        
        end_total = time.perf_counter()
        
        # Calculate statistics
        total_time = end_total - start_total
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        p50_latency = np.percentile(latencies, 50)
        p90_latency = np.percentile(latencies, 90)
        p99_latency = np.percentile(latencies, 99)
        
        # Throughput in images per second
        throughput = num_iterations / total_time * batch_size
        
        print(f"\nResults:")
        print(f"  Throughput: {throughput:.2f} images/sec")
        print(f"  Average latency: {avg_latency:.2f} ms")
        print(f"  Min latency: {min_latency:.2f} ms")
        print(f"  Max latency: {max_latency:.2f} ms")
        print(f"  P50 latency: {p50_latency:.2f} ms")
        print(f"  P90 latency: {p90_latency:.2f} ms")
        print(f"  P99 latency: {p99_latency:.2f} ms")
        
        return {
            "provider": provider,
            "batch_size": batch_size,
            "throughput": throughput,
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "p50_latency": p50_latency,
            "p90_latency": p90_latency,
            "p99_latency": p99_latency,
        }
        
    except Exception as e:
        print(f"Error with {provider}: {e}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='Test CUDA only (skip TensorRT)')
    parser.add_argument('--cpu', action='store_true', help='Test CPU only')
    args = parser.parse_args()

    print("="*60)
    print("ONNX Runtime Inference Benchmark")
    print("="*60)
    
    # Print ONNX Runtime version
    print(f"\nONNX Runtime version: {ort.__version__}")
    
    # Get available providers
    providers = get_execution_providers()
    
    # Download/check for ResNet-18 model
    model_path = download_resnet18_onnx()
    if not model_path:
        print("No model available for benchmarking")
        return
    
    print(f"\nModel: {model_path}")
    
    # Benchmark with different providers
    results = []
    
    # Test providers in order of preference
    test_providers = []
    if args.cuda:
        if "CUDAExecutionProvider" in providers:
            test_providers.append("CUDAExecutionProvider")
    elif args.cpu:
        test_providers.append("CPUExecutionProvider")
    else:
        if "TensorrtExecutionProvider" in providers:
            test_providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in providers:
            test_providers.append("CUDAExecutionProvider")
        test_providers.append("CPUExecutionProvider")
    
    # Batch sizes to test
    batch_sizes = [1, 4, 8, 16, 32, 64]
    
    for provider in test_providers:
        for batch_size in batch_sizes:
            result = benchmark_inference(
                model_path, 
                provider, 
                num_warmup=100,
                num_iterations=500,
                batch_size=batch_size
            )
            if result:
                results.append(result)
    
    # Print summary
    if results:
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(f"{'Provider':<25} {'Batch':>6} {'Throughput':>12} {'Avg Lat':>10} {'P99 Lat':>10}")
        print("-"*80)
        for r in results:
            print(f"{r['provider']:<25} {r['batch_size']:>6} {r['throughput']:>10.1f}/s {r['avg_latency']:>8.2f}ms {r['p99_latency']:>8.2f}ms")

if __name__ == "__main__":
    main()

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torchvision.models as models

# Import your engine's configuration and class.
from modules.core.engine import EngineConfig, InferenceEngine

@dataclass
class BenchmarkConfig:
    num_inputs: int = 4             # Number of inference requests
    warmup_runs: int = 8
    input_channels: int = 4
    input_height: int = 224
    input_width: int = 224
    batch_size: int = 64                # Batch size for synchronous inference
    use_tensorrt: bool = False
    enable_dynamic_batching: bool = False
    profile: bool = False
    async_mode: bool = True
    sync_mode: bool = True
    max_concurrent: int = 16           # Maximum number of concurrent async requests
    log_file: Optional[str] = "benchmark.log"
    debug_mode: bool = True

def parse_args():
    parser = argparse.ArgumentParser(description="Inference Engine Benchmark")
    parser.add_argument("--num-inputs", type=int, default=10000, help="Number of inference requests")
    parser.add_argument("--input-channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--input-height", type=int, default=224, help="Input image height")
    parser.add_argument("--input-width", type=int, default=224, help="Input image width")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for synchronous inference")
    parser.add_argument("--use-tensorrt", action="store_true", help="Enable TensorRT optimization")
    parser.add_argument("--no-async", action="store_false", dest="async_mode", help="Disable async benchmarking")
    parser.add_argument("--no-sync", action="store_false", dest="sync_mode", help="Disable sync benchmarking")
    parser.add_argument("--profile", action="store_true", help="Run profiling")
    parser.add_argument("--max-concurrent", type=int, default=256, help="Max number of concurrent async requests")
    parser.add_argument("--log-file", type=str, default="benchmark.log", help="Log file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()

async def benchmark_async(engine, inputs, max_concurrent: int):
    """
    Benchmark asynchronous inference throughput with concurrency control.
    The semaphore limits the number of concurrently running async tasks.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def sem_task(x):
        async with semaphore:
            return await engine.run_inference_async(x)
    
    start_time = time.perf_counter()
    tasks = [sem_task(x) for x in inputs]
    await asyncio.gather(*tasks, return_exceptions=False)
    duration = time.perf_counter() - start_time
    throughput = len(inputs) / duration
    seconds_per_pred = duration / len(inputs)
    return throughput, seconds_per_pred, duration

def benchmark_sync(engine, inputs):
    """
    Benchmark synchronous batch inference throughput.
    Splits the inputs into batches and processes each batch sequentially.
    """
    start_time = time.perf_counter()
    
    # Create batches by stacking inputs according to the engine's batch size
    batches = [
        torch.stack(inputs[i:i+engine.config.batch_size])
        for i in range(0, len(inputs), engine.config.batch_size)
    ]
    
    for batch in batches:
        engine.run_batch_inference(batch)
    
    duration = time.perf_counter() - start_time
    throughput = len(inputs) / duration
    seconds_per_pred = duration / len(inputs)
    return throughput, seconds_per_pred, duration

async def main(benchmark_config: BenchmarkConfig):
    # Use a larger model: ResNet-50 (or change to any other model as needed)
    model = models.resnet18(pretrained=True)
    # Optionally modify the final layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model = model.to("cuda")
    
    # Set up the engine configuration
    engine_config = EngineConfig(
        input_shape=[1, benchmark_config.input_channels, benchmark_config.input_height, benchmark_config.input_width],
        batch_size=benchmark_config.batch_size,
        use_tensorrt=benchmark_config.use_tensorrt,
        enable_dynamic_batching=benchmark_config.enable_dynamic_batching,
        log_file=benchmark_config.log_file
    )
    
    engine = InferenceEngine(model=model, config=engine_config)
    
    # Generate test inputs (random images) with shape [C, H, W]
    input_shape = (benchmark_config.input_channels, benchmark_config.input_height, benchmark_config.input_width)
    inputs = [torch.randn(*input_shape, device="cuda") for _ in range(benchmark_config.num_inputs)]
    
    # Warmup: Run a few inferences to prepare the model/engine
    warmup_input = inputs[0]
    for _ in range(benchmark_config.warmup_runs):
        await engine.run_inference_async(warmup_input)
    
    results = {}
    
    # Asynchronous Benchmark (run only once)
    if benchmark_config.async_mode:
        throughput, sec_per_pred, duration = await benchmark_async(engine, inputs, benchmark_config.max_concurrent)
        results["async_throughput"] = throughput
        results["async_sec_per_pred"] = sec_per_pred
        print("=== Asynchronous Inference ===")
        print(f"Total Duration: {duration:.4f} seconds")
        print(f"Throughput: {throughput:.2f} predictions/s")
        print(f"Seconds per Prediction: {sec_per_pred:.6f} s/pred")
    
    # Synchronous Benchmark (run only once)
    if benchmark_config.sync_mode:
        throughput, sec_per_pred, duration = benchmark_sync(engine, inputs)
        results["sync_throughput"] = throughput
        results["sync_sec_per_pred"] = sec_per_pred
        print("\n=== Synchronous Inference ===")
        print(f"Total Duration: {duration:.4f} seconds")
        print(f"Throughput: {throughput:.2f} predictions/s")
        print(f"Seconds per Prediction: {sec_per_pred:.6f} s/pred")
    
    # Profiling (if available and enabled)
    if benchmark_config.profile and hasattr(engine, 'profile_inference'):
        profile_input = inputs[0]
        profile_metrics = engine.profile_inference(profile_input)
        results["profile"] = profile_metrics
        print("\n=== Profile Metrics ===")
        print(profile_metrics)
    
    engine.cleanup()
    return results

if __name__ == "__main__":
    args = parse_args()
    
    benchmark_config = BenchmarkConfig(
        num_inputs=args.num_inputs,
        warmup_runs=10,
        input_channels=args.input_channels,
        input_height=args.input_height,
        input_width=args.input_width,
        batch_size=args.batch_size,
        use_tensorrt=args.use_tensorrt,
        enable_dynamic_batching=True,
        profile=args.profile,
        async_mode=args.async_mode,
        sync_mode=args.sync_mode,
        max_concurrent=args.max_concurrent,
        log_file=args.log_file,
        debug_mode=args.debug
    )
    
    try:
        results = asyncio.run(main(benchmark_config))
    except KeyboardInterrupt:
        print("Benchmark interrupted!")
        exit(1)

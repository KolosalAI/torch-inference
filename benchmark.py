import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torchvision.models as models

# Import your engine's configuration and class.
from modules.core.inference_engine import EngineConfig, InferenceEngine

@dataclass
class BenchmarkConfig:
    num_inputs: int = 10000           # Larger number of inference requests
    warmup_runs: int = 10
    input_channels: int = 3            # ResNet expects 3-channel images
    input_height: int = 224
    input_width: int = 224
    batch_size: int = 64               # Batch size for synchronous inference
    use_tensorrt: bool = True
    enable_dynamic_batching: bool = True
    profile: bool = True
    async_mode: bool = True
    sync_mode: bool = True
    max_concurrent: int = 256          # Maximum number of concurrent async requests
    log_file: Optional[str] = "benchmark.log"
    debug_mode: bool = True

async def benchmark_async(engine, inputs, max_concurrent: int):
    """
    Benchmark asynchronous inference throughput with concurrency control.
    The semaphore limits the number of concurrently running async tasks.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    logger = logging.getLogger(__name__)

    async def sem_task(x):
        async with semaphore:
            return await engine.run_inference_async(x)

    start_time = time.perf_counter()
    tasks = [sem_task(x) for x in inputs]
    await asyncio.gather(*tasks, return_exceptions=False)
    duration = time.perf_counter() - start_time
    throughput = len(inputs) / duration
    seconds_per_pred = duration / len(inputs)
    logger.debug("Asynchronous benchmark completed in %.4f seconds", duration)
    return throughput, seconds_per_pred, duration

def benchmark_sync(engine, inputs):
    """
    Benchmark synchronous batch inference throughput.
    Splits the inputs into batches and processes each batch sequentially.
    """
    logger = logging.getLogger(__name__)
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
    logger.debug("Synchronous benchmark completed in %.4f seconds", duration)
    return throughput, seconds_per_pred, duration

async def main(benchmark_config: BenchmarkConfig):
    logger = logging.getLogger(__name__)
    logger.info("Starting benchmark with configuration: %s", benchmark_config)

    # Use ResNet-50 (or change to any other model as needed)
    model = models.resnet50(pretrained=True)
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
        log_file=benchmark_config.log_file,
        autoscale_interval=0
    )

    engine = InferenceEngine(model=model, config=engine_config)

    # Generate test inputs (random images) with shape [C, H, W]
    input_shape = (benchmark_config.input_channels, benchmark_config.input_height, benchmark_config.input_width)
    inputs = [torch.randn(*input_shape, device="cuda") for _ in range(benchmark_config.num_inputs)]
    logger.debug("Generated %d test inputs with shape %s", benchmark_config.num_inputs, input_shape)

    # Warmup: Run a few inferences to prepare the model/engine
    warmup_input = inputs[0]
    logger.info("Warming up the model with %d runs", benchmark_config.warmup_runs)
    for _ in range(benchmark_config.warmup_runs):
        await engine.run_inference_async(warmup_input)

    results = {}

    # Asynchronous Benchmark
    if benchmark_config.async_mode:
        throughput, sec_per_pred, duration = await benchmark_async(engine, inputs, benchmark_config.max_concurrent)
        results["async_throughput"] = throughput
        results["async_sec_per_pred"] = sec_per_pred
        logger.info("=== Asynchronous Inference ===")
        logger.info("Total Duration: %.4f seconds", duration)
        logger.info("Throughput: %.2f predictions/s", throughput)
        logger.info("Seconds per Prediction: %.6f s/pred", sec_per_pred)

    # Synchronous Benchmark
    if benchmark_config.sync_mode:
        throughput, sec_per_pred, duration = benchmark_sync(engine, inputs)
        results["sync_throughput"] = throughput
        results["sync_sec_per_pred"] = sec_per_pred
        logger.info("=== Synchronous Inference ===")
        logger.info("Total Duration: %.4f seconds", duration)
        logger.info("Throughput: %.2f predictions/s", throughput)
        logger.info("Seconds per Prediction: %.6f s/pred", sec_per_pred)

    # Profiling (if available and enabled)
    if benchmark_config.profile and hasattr(engine, 'profile_inference'):
        profile_input = inputs[0]
        profile_metrics = engine.profile_inference(profile_input)
        results["profile"] = profile_metrics
        logger.info("=== Profile Metrics ===")
        logger.info("Profile Metrics: %s", profile_metrics)

    await engine.cleanup()
    logger.info("Engine cleanup completed.")

    return results

if __name__ == "__main__":
    # Directly create a benchmark configuration with all options enabled
    benchmark_config = BenchmarkConfig(
        num_inputs=2048*4,           # Larger number of test inputs
        warmup_runs=10,
        input_channels=3,
        input_height=224,
        input_width=224,
        batch_size=64,
        use_tensorrt=True,
        enable_dynamic_batching=False,
        profile=True,
        async_mode=True,
        sync_mode=True,
        max_concurrent=256,
        log_file="benchmark.log",
        debug_mode=True
    )

    # Configure logging: set level based on debug_mode and add both console and file handlers.
    handlers = [logging.StreamHandler()]
    if benchmark_config.log_file:
        handlers.append(logging.FileHandler(benchmark_config.log_file))
    logging.basicConfig(
        level=logging.DEBUG if benchmark_config.debug_mode else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    try:
        asyncio.run(main(benchmark_config))
    except KeyboardInterrupt:
        logging.warning("Benchmark interrupted!")
        exit(1)

#!/usr/bin/env python3

import asyncio
import logging
import time
from statistics import mean, median
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as models

from modules.core.inference_engine import InferenceEngine, EngineConfig

@dataclass
class BenchmarkConfig:
    num_inputs: int = 4
    warmup_runs: int = 8
    input_channels: int = 3
    input_height: int = 224
    input_width: int = 224
    batch_size: int = 64
    async_mode: bool = True
    sync_mode: bool = True
    max_concurrent: int = 16
    log_file: str = "benchmark.log"
    debug_mode: bool = False

def setup_logging(log_file: str, debug_mode: bool):
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Logging initialized. Log file: %s", log_file)

def create_resnet18_model(pretrained: bool = False) -> nn.Module:
    logging.info("Creating ResNet-18 model with pretrained=%s", pretrained)
    try:
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)

        # Disable torch.compile
        # try:
        #     model = torch.compile(model)
        #     logging.info("Model compiled with torch.compile()")
        # except Exception as e:
        #     logging.warning("torch.compile not available or failed: %s", str(e))

        logging.info("Model created successfully.")
        return model
    except Exception as e:
        logging.error("Failed to create model: %s", str(e))
        raise


async def warmup(engine, warmup_runs: int, input_data: torch.Tensor):
    logging.info("Starting warmup with %d runs", warmup_runs)
    for i in range(warmup_runs):
        try:
            await engine.infer(input_data)
        except Exception as e:
            logging.error("Warmup failed on run %d: %s", i, str(e))
            raise
    logging.info("Warmup complete.")

async def async_infer_requests(engine, total_requests: int, concurrency: int, input_data: torch.Tensor):
    latencies = []
    semaphore = asyncio.Semaphore(concurrency)

    async def infer_one():
        async with semaphore:
            try:
                start = time.perf_counter()
                await engine.infer(input_data)
                end = time.perf_counter()
                return (end - start)
            except Exception as e:
                logging.error("Inference failed: %s", str(e))
                return None

    # Use chunking to avoid thousands of tasks at once
    CHUNK_SIZE = concurrency  # Or you can tune this further
    requests_done = 0

    while requests_done < total_requests:
        batch_size = min(CHUNK_SIZE, total_requests - requests_done)
        tasks = [asyncio.create_task(infer_one()) for _ in range(batch_size)]
        latencies_chunk = await asyncio.gather(*tasks)
        latencies.extend([lat for lat in latencies_chunk if lat is not None])
        requests_done += batch_size

    return latencies

def sync_infer_requests(engine, total_requests: int, input_data: torch.Tensor):
    latencies = []
    for i in range(total_requests):
        try:
            start = time.perf_counter()
            if asyncio.iscoroutinefunction(engine.infer):
                asyncio.run(engine.infer(input_data))
            else:
                engine.infer(input_data)
            end = time.perf_counter()
            latencies.append(end - start)
        except Exception as e:
            logging.error("Sync inference failed on request %d: %s", i, str(e))
    return latencies

def log_benchmark_stats(latencies, total_time, mode):
    n = len(latencies)
    if n == 0:
        logging.warning("No latencies to log for %s mode", mode)
        return
    avg = mean(latencies) * 1000.0
    med = median(latencies) * 1000.0
    throughput = n / total_time
    msg = (
        f"{mode} benchmark results:\n"
        f"  Requests: {n}\n"
        f"  Total time: {total_time:.4f} s\n"
        f"  Throughput: {throughput:.2f} req/s\n"
        f"  Avg latency: {avg:.2f} ms\n"
        f"  Median latency: {med:.2f} ms"
    )
    logging.info(msg)
    print(msg)

async def benchmark(config: BenchmarkConfig):
    setup_logging(config.log_file, config.debug_mode)
    logging.info("===== Starting Benchmark =====")

    try:
        # Create the model
        model = create_resnet18_model(pretrained=False)

        # Move model to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        logging.info("Using device: %s", device)

        # Engine config
        engine_cfg = EngineConfig(
            debug_mode=config.debug_mode,
            batch_size=config.batch_size,
            async_mode=config.async_mode,
            warmup_runs=0
        )

        # Instantiate engine
        engine = InferenceEngine(
            model=model,
            device=device,
            config=engine_cfg
        )

        # Create a single input tensor to reuse
        input_shape = (config.input_channels, config.input_height, config.input_width)
        input_data = torch.randn(input_shape, dtype=torch.float32)
        if torch.cuda.is_available():
            input_data = input_data.pin_memory().to(device, non_blocking=True)

        # Warmup
        await warmup(engine, config.warmup_runs, input_data)

        # Async benchmark
        if config.async_mode:
            start_time = time.perf_counter()
            async_latencies = await async_infer_requests(
                engine,
                config.num_inputs,
                config.max_concurrent,
                input_data
            )
            total_time = time.perf_counter() - start_time
            log_benchmark_stats(async_latencies, total_time, "Async")

        # Sync benchmark
        if config.sync_mode:
            logging.info("Switching to synchronous benchmark...")
            engine_cfg_sync = EngineConfig(
                debug_mode=config.debug_mode,
                batch_size=config.batch_size,
                async_mode=False,
                warmup_runs=0
            )
            engine_sync = InferenceEngine(
                model=model,
                device=device,
                config=engine_cfg_sync
            )
            # Warmup for sync
            await warmup(engine_sync, config.warmup_runs, input_data)
            logging.info("Running synchronous benchmark...")
            start_time = time.perf_counter()
            sync_latencies = sync_infer_requests(
                engine_sync,
                config.num_inputs,
                input_data
            )
            total_time = time.perf_counter() - start_time
            log_benchmark_stats(sync_latencies, total_time, "Sync")

            # Shutdown sync
            await engine_sync.shutdown()

        # Shutdown async
        await engine.shutdown()
        logging.info("Benchmark complete. Engines shut down.")
    except Exception as e:
        logging.error("Benchmark failed: %s", str(e))
        raise

if __name__ == "__main__":
    config = BenchmarkConfig(
        num_inputs=8192,          # Large number for demonstration
        warmup_runs=10,
        input_channels=3,
        input_height=224,
        input_width=224,
        batch_size=64,
        async_mode=True,
        sync_mode=True,
        max_concurrent=256,
        log_file="benchmark.log",
        debug_mode=False  # Turn off debug logs by default
    )

    asyncio.run(benchmark(config))

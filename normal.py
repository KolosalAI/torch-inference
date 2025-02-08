import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torchvision.models as models
import torchvision
from packaging import version  # Used to check the torchvision version

# -----------------------------------------------------------------------------
# Configuration Dataclass
# -----------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    num_inputs: int = 4             # Number of inference requests for timing
    warmup_runs: int = 8
    input_channels: int = 3         # Use 3 channels by default
    input_height: int = 224
    input_width: int = 224
    batch_size: int = 64            # Batch size for inference
    async_mode: bool = True
    sync_mode: bool = True
    max_concurrent: int = 16        # Maximum number of concurrent asynchronous requests
    log_file: Optional[str] = "benchmark.log"
    debug_mode: bool = True

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------

def setup_logging(log_file: Optional[str] = None, debug_mode: bool = False) -> logging.Logger:
    """
    Configures logging for the application.
    """
    logger = logging.getLogger("InferenceSystem")
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Clear existing handlers.
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    # Console handler.
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (if a log_file is provided).
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

# -----------------------------------------------------------------------------
# Model and Preprocessing Functions
# -----------------------------------------------------------------------------

def load_model(device: torch.device, logger: logging.Logger, input_channels: int) -> torch.nn.Module:
    """
    Loads a pre-trained ResNet50 model from Torchvision and adapts its first convolution
    layer if the expected number of input channels is different from 3.
    """
    logger.info("Loading pre-trained ResNet50 model from Torchvision")
    try:
        # Use the new weights API if available (Torchvision >= 0.13), else fallback.
        if version.parse(torchvision.__version__) >= version.parse("0.13"):
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        else:
            model = models.resnet50(pretrained=True)

        if input_channels != 3:
            old_conv = model.conv1
            new_conv = torch.nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None)
            )
            if input_channels >= 3:
                # Copy the weights for the first 3 channels.
                new_conv.weight.data[:, :3, :, :] = old_conv.weight.data
                if input_channels > 3:
                    # For extra channels, initialize them as the mean of the first 3 channels,
                    # and repeat to cover all extra channels.
                    extra_channels = input_channels - 3
                    mean_weights = old_conv.weight.data.mean(dim=1, keepdim=True)
                    new_conv.weight.data[:, 3:input_channels, :, :] = mean_weights.repeat(1, extra_channels, 1, 1)
            else:
                new_conv.weight.data = old_conv.weight.data[:, :input_channels, :, :]
            model.conv1 = new_conv
            logger.info("Adjusted model.conv1 to accept %d input channels", input_channels)

        model.to(device)
        model.eval()

        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        logger.info("Model loaded successfully on %s", device)
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise
    return model

def generate_synthetic_input(b_config: BenchmarkConfig, device: torch.device,
                             logger: logging.Logger) -> torch.Tensor:
    """
    Generates a synthetic input tensor based on the benchmark configuration.
    """
    logger.info("Generating synthetic input: batch_size=%d, channels=%d, height=%d, width=%d",
                b_config.batch_size, b_config.input_channels, b_config.input_height, b_config.input_width)
    try:
        synthetic_input = torch.randn(
            b_config.batch_size,
            b_config.input_channels,
            b_config.input_height,
            b_config.input_width,
            device=device
        )
    except Exception as e:
        logger.exception("Error generating synthetic input: %s", e)
        raise
    return synthetic_input

# -----------------------------------------------------------------------------
# Inference Functions
# -----------------------------------------------------------------------------

def predict(input_batch: torch.Tensor, model: torch.nn.Module, device: torch.device,
            logger: logging.Logger) -> torch.Tensor:
    """
    Runs inference on a single input batch and returns the computed probabilities
    for the first sample in the batch.
    """
    try:
        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
    except Exception as e:
        logger.exception("Error during model inference: %s", e)
        raise
    return probabilities

# -----------------------------------------------------------------------------
# Benchmarking Functions
# -----------------------------------------------------------------------------

def synchronous_speed_test(model: torch.nn.Module, input_batch: torch.Tensor,
                           warmup_runs: int, test_runs: int,
                           device: torch.device, logger: logging.Logger
                           ) -> Tuple[float, float, float]:
    """
    Runs synchronous inference speed test.
    Returns:
        - Total duration (seconds)
        - Throughput (predictions per second)
        - Seconds per prediction
    """
    logger.info("Starting synchronous speed test: warmup_runs=%d, test_runs=%d", warmup_runs, test_runs)
    try:
        # Warmup runs.
        for _ in range(warmup_runs):
            _ = predict(input_batch, model, device, logger)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(test_runs):
            _ = predict(input_batch, model, device, logger)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start_time
        throughput = test_runs / elapsed
        sec_per_pred = elapsed / test_runs
        logger.info("Synchronous speed test completed in %.4f seconds", elapsed)

        # Free any unused GPU memory.
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    except Exception as e:
        logger.exception("Synchronous speed test failed: %s", e)
        raise
    return elapsed, throughput, sec_per_pred

def asynchronous_speed_test(model: torch.nn.Module, input_batch: torch.Tensor,
                            warmup_runs: int, test_runs: int,
                            max_concurrent: int, device: torch.device,
                            logger: logging.Logger) -> Tuple[float, float, float]:
    """
    Runs asynchronous inference speed test using a ThreadPoolExecutor.
    Returns:
        - Total duration (seconds)
        - Throughput (predictions per second)
        - Seconds per prediction
    """
    logger.info("Starting asynchronous speed test: warmup_runs=%d, test_runs=%d", warmup_runs, test_runs)
    try:
        # Warmup runs.
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            warmup_futures = [
                executor.submit(predict, input_batch, model, device, logger)
                for _ in range(warmup_runs)
            ]
            for future in as_completed(warmup_futures):
                _ = future.result()
        if device.type == 'cuda':
            torch.cuda.synchronize()
    except Exception as e:
        logger.exception("Asynchronous warmup failed: %s", e)
        raise

    start_time = time.time()
    try:
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [
                executor.submit(predict, input_batch, model, device, logger)
                for _ in range(test_runs)
            ]
            for future in as_completed(futures):
                _ = future.result()
        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start_time
        throughput = test_runs / elapsed
        sec_per_pred = elapsed / test_runs
        logger.info("Asynchronous speed test completed in %.4f seconds", elapsed)

        # Free any unused GPU memory.
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    except Exception as e:
        logger.exception("Asynchronous speed test failed: %s", e)
        raise
    return elapsed, throughput, sec_per_pred

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main() -> None:
    # Update configuration as requested.
    bench_config = BenchmarkConfig(
        num_inputs=2048 * 4,           # Larger number of test inputs
        warmup_runs=10,
        input_channels=3,
        input_height=224,
        input_width=224,
        batch_size=64,
        async_mode=True,
        sync_mode=True,
        max_concurrent=256,
        log_file="benchmark.log",
        debug_mode=True
    )
    # Note: The parameters 'use_tensorrt', 'enable_dynamic_batching', and 'profile' are
    # not added because no new variables should be introduced.

    logger = setup_logging(bench_config.log_file, bench_config.debug_mode)
    logger.info("Starting Inference System (Pre-trained ResNet50)")
    logger.info("Benchmark mode enabled with configuration: %s", bench_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # If running on GPU, reduce the number of concurrent asynchronous tasks to avoid GPU memory exhaustion.
    if device.type == 'cuda':
        safe_max_concurrent = 16
        if bench_config.max_concurrent > safe_max_concurrent:
            logger.warning(
                "Reducing max_concurrent from %d to %d to avoid GPU memory exhaustion",
                bench_config.max_concurrent, safe_max_concurrent
            )
            bench_config.max_concurrent = safe_max_concurrent

    try:
        model = load_model(device, logger, bench_config.input_channels)
    except Exception:
        logger.error("Exiting due to model load failure.")
        sys.exit(1)

    try:
        input_batch = generate_synthetic_input(bench_config, device, logger)
    except Exception:
        logger.error("Exiting due to synthetic input generation failure.")
        sys.exit(1)

    test_runs = bench_config.num_inputs

    if bench_config.async_mode:
        try:
            async_duration, async_throughput, async_sec_per_pred = asynchronous_speed_test(
                model, input_batch, bench_config.warmup_runs, test_runs,
                bench_config.max_concurrent, device, logger
            )
            logger.info("\n=== Asynchronous Inference ===")
            logger.info("Total Duration: %.4f seconds", async_duration)
            logger.info("Throughput: %.2f predictions/s", async_throughput)
            logger.info("Seconds per Prediction: %.6f s/pred", async_sec_per_pred)
        except Exception:
            logger.error("Exiting due to asynchronous speed test failure.")
            sys.exit(1)

    if bench_config.sync_mode:
        try:
            sync_duration, sync_throughput, sync_sec_per_pred = synchronous_speed_test(
                model, input_batch, bench_config.warmup_runs, test_runs,
                device, logger
            )
            logger.info("\n=== Synchronous Inference ===")
            logger.info("Total Duration: %.4f seconds", sync_duration)
            logger.info("Throughput: %.2f predictions/s", sync_throughput)
            logger.info("Seconds per Prediction: %.6f s/pred", sync_sec_per_pred)
        except Exception:
            logger.error("Exiting due to synchronous speed test failure.")
            sys.exit(1)

    logger.info("Inference System completed successfully.")

if __name__ == '__main__':
    main()

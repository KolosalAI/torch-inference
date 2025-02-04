import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# -----------------------------------------------------------------------------
# Configuration Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    num_inputs: int = 4             # Number of inference requests for timing
    warmup_runs: int = 8
    input_channels: int = 4
    input_height: int = 224
    input_width: int = 224
    batch_size: int = 64           # Batch size for inference
    use_tensorrt: bool = False
    enable_dynamic_batching: bool = False
    profile: bool = False
    async_mode: bool = True
    sync_mode: bool = True
    max_concurrent: int = 16       # Maximum number of concurrent async requests
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

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (if log_file provided)
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
    Loads a pre-trained ResNet18 model from Torchvision and adapts its first convolution
    layer if the expected input channels differ from 3.
    """
    logger.info("Loading pre-trained ResNet18 model from Torchvision")
    try:
        model = models.resnet18(pretrained=True)
        # If the synthetic input has a different number of channels, modify the first conv layer.
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
            # Copy pretrained weights for the first 3 channels.
            new_conv.weight.data[:, :3, :, :] = old_conv.weight.data
            # For the extra channel(s), initialize as the mean of the first 3 channels.
            new_conv.weight.data[:, 3:input_channels, :, :] = old_conv.weight.data.mean(dim=1, keepdim=True)
            model.conv1 = new_conv
            logger.info("Adjusted model.conv1 to accept %d input channels", input_channels)

        model.to(device)
        model.eval()
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
    Runs inference on a single input batch.
    """
    input_batch = input_batch.to(device)
    logger.debug("Running inference...")
    try:
        with torch.no_grad():
            output = model(input_batch)
            # Compute probabilities from the first example in the batch.
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
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = predict(input_batch, model, device, logger)
        # Timed runs
        start_time = time.time()
        with torch.no_grad():
            for _ in range(test_runs):
                _ = predict(input_batch, model, device, logger)
        elapsed = time.time() - start_time
        throughput = test_runs / elapsed
        sec_per_pred = elapsed / test_runs
        logger.info("Synchronous speed test completed in %.4f seconds", elapsed)
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
        # Warmup runs using asynchronous calls.
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            warmup_futures = [executor.submit(predict, input_batch, model, device, logger)
                              for _ in range(warmup_runs)]
            for future in as_completed(warmup_futures):
                _ = future.result()
    except Exception as e:
        logger.exception("Asynchronous warmup failed: %s", e)
        raise

    start_time = time.time()
    try:
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(predict, input_batch, model, device, logger)
                       for _ in range(test_runs)]
            for future in as_completed(futures):
                _ = future.result()
        elapsed = time.time() - start_time
        throughput = test_runs / elapsed
        sec_per_pred = elapsed / test_runs
        logger.info("Asynchronous speed test completed in %.4f seconds", elapsed)
    except Exception as e:
        logger.exception("Asynchronous speed test failed: %s", e)
        raise
    return elapsed, throughput, sec_per_pred

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main() -> None:
    # In this version we perform benchmark tests only.
    benchmark_mode = True

    # Create benchmark configuration using the provided settings.
    bench_config = BenchmarkConfig()
    
    # Setup logging.
    log_file = bench_config.log_file if benchmark_mode else "inference_system.log"
    logger = setup_logging(log_file=log_file, debug_mode=bench_config.debug_mode)
    logger.info("Starting Inference System (Pre-trained ResNet18)")
    logger.info("Benchmark mode: %s", benchmark_mode)

    # Choose the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load a pre-trained ResNet18 model and adapt it for the configured number of input channels.
    try:
        model = load_model(device, logger, bench_config.input_channels)
    except Exception:
        logger.error("Exiting due to model load failure.")
        sys.exit(1)

    # Generate synthetic input for benchmarking.
    try:
        input_batch = generate_synthetic_input(bench_config, device, logger)
    except Exception:
        logger.error("Exiting due to synthetic input generation failure.")
        sys.exit(1)

    # For benchmarking, use bench_config.num_inputs as the number of measured inferences.
    test_runs = bench_config.num_inputs

    # Run asynchronous speed test (if enabled).
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

    # Run synchronous speed test (if enabled).
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

import logging
import time
import torch
import torchvision.models as models
import asyncio

# Updated module imports according to the project structure.
from modules.core.engine import InferenceEngine
from modules.core.preprocessor import TensorRTPreprocessor
from modules.core.postprocessor import ClassificationPostprocessor
from modules.utils.config import Config
from modules.utils.logger import setup_logging

# Setup logging using the custom logger.
logger = setup_logging()

async def main():
    """
    Main entry point for running inference on a random input.
    
    This function:
      - Loads configuration parameters.
      - Initializes a TensorRTPreprocessor, InferenceEngine, and ClassificationPostprocessor.
      - Uses a pretrained torchvision model (ResNet-18) for inference.
      - Generates a random input tensor simulating a batch of images.
      - Runs inference and measures performance (inference time and throughput).
      - Post-processes and displays the results.
    """
    # Load configuration settings.
    config = Config()

    # Initialize the TensorRT preprocessor.
    preprocessor = TensorRTPreprocessor(
        image_size=config.IMAGE_SIZE,  # e.g. (224, 224)
        trt_fp16=config.TRT_FP16,       # Boolean flag for FP16 conversion
        device=config.DEVICE
    )

    # Use a pretrained model from torchvision.
    model = models.resnet18(pretrained=True)
    # Optionally convert model to half precision if TRT_FP16 is enabled.
    if config.TRT_FP16 and (
        (isinstance(config.DEVICE, str) and config.DEVICE.lower() == "cuda") or
        (isinstance(config.DEVICE, torch.device) and config.DEVICE.type == "cuda")
    ):
        model = model.half()

    # Initialize the inference engine with the loaded model.
    engine = InferenceEngine(model, config.DEVICE)

    # Initialize the postprocessor.
    postprocessor = ClassificationPostprocessor()

    # Generate a random input batch (simulate images).
    batch_size = 4
    random_input = torch.randn((batch_size, 3, *config.IMAGE_SIZE), dtype=torch.float32).to(config.DEVICE)
    logger.info(f"Generated random input tensor of shape: {random_input.shape}")

    # Ensure any asynchronous GPU operations are complete before timing.
    if (isinstance(config.DEVICE, str) and config.DEVICE.lower() == "cuda") or \
       (isinstance(config.DEVICE, torch.device) and config.DEVICE.type == "cuda"):
        torch.cuda.synchronize()

    start_time = time.time()
    output = engine(random_input)
    if (isinstance(config.DEVICE, str) and config.DEVICE.lower() == "cuda") or \
       (isinstance(config.DEVICE, torch.device) and config.DEVICE.type == "cuda"):
        torch.cuda.synchronize()  # Synchronize to capture accurate timing.
    end_time = time.time()

    inference_time = end_time - start_time  # Total time taken for inference.
    pred_per_second = batch_size / inference_time if inference_time > 0 else float('inf')
    seconds_per_pred = inference_time / batch_size if batch_size > 0 else float('inf')

    logger.info(f"Inference completed in {inference_time:.6f} seconds.")
    logger.info(f"Predictions per second: {pred_per_second:.2f} pred/sec")
    logger.info(f"Seconds per prediction: {seconds_per_pred:.6f} sec/pred")

    # Run post-processing on the engine output.
    logger.info("Post-processing results...")
    results = postprocessor(output)

    # Log each result.
    for i, result in enumerate(results):
        logger.info(f"Result {i}: {result}")

if __name__ == "__main__":
    # Run the async main function, which creates and uses a running event loop.
    asyncio.run(main())

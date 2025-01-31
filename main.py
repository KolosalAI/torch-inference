import logging
import time
import torch
from modules.core.preprocessor import TensorRTPreprocessor
from modules.core.engine import InferenceEngine
from modules.core.postprocessor import PostProcessor
from modules.utils.logger import setup_logging
from modules.utils.config import Config

# Setup logging
logger = setup_logging()

def main():
    """
    Main function to initialize preprocessing, inference, and post-processing.
    This function uses randomly generated tensors instead of images and
    measures performance (predictions per second).
    """

    # Load configuration
    config = Config()

    # Initialize components
    preprocessor = TensorRTPreprocessor(
        image_size=config.IMAGE_SIZE,  # Used for shaping the random tensor
        trt_fp16=config.TRT_FP16,
        device=config.DEVICE
    )

    engine = InferenceEngine(
        model_path=config.MODEL_PATH,
        device=config.DEVICE
    )

    postprocessor = PostProcessor()

    # Simulate random input batch instead of images
    batch_size = 4
    random_input = torch.randn((batch_size, 3, *config.IMAGE_SIZE), dtype=torch.float32).to(config.DEVICE)

    logger.info(f"Generated random input tensor of shape: {random_input.shape}")

    # Measure inference time
    start_time = time.time()
    output = engine(random_input)
    end_time = time.time()

    inference_time = end_time - start_time  # Total time taken for inference
    pred_per_second = batch_size / inference_time if inference_time > 0 else float('inf')
    seconds_per_pred = inference_time / batch_size if batch_size > 0 else float('inf')

    logger.info(f"Inference completed in {inference_time:.6f} seconds.")
    logger.info(f"Predictions per second: {pred_per_second:.2f} pred/sec")
    logger.info(f"Seconds per prediction: {seconds_per_pred:.6f} sec/pred")

    # Post-processing
    logger.info("Post-processing results...")
    results = postprocessor(output)

    # Display results
    for i, result in enumerate(results):
        logger.info(f"Result {i}: {result}")

if __name__ == "__main__":
    main()

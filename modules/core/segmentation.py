import cv2
import numpy as np
import torch
import asyncio
from typing import Dict, Tuple, Union, Any, Optional
import sys
import os
import time
import logging

# Add the parent directory to sys.path so that imports like "core.engine" work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Local imports
from utils.config import SEGMENTATION_CONFIG, EngineConfig
from utils.logger import get_logger
from core.engine import InferenceEngine
from core.preprocessor import ImagePreprocessor  # Enhanced preprocessor implementation
from core.postprocessor import BasePostprocessor

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)  # Set logger to debug mode

class SegmentationPreprocessor(ImagePreprocessor):
    """
    Segmentation-specific preprocessor that leverages the enhanced ImagePreprocessor.
    Supports batched tensor input, PIL Images, numpy arrays, and file paths.
    In addition to the standard preprocessing, it center-crops (to match the target aspect ratio)
    and then resizes the input image to the model's expected input size.
    """
    def __init__(self):
        # Preprocess mean and std from the configuration:
        mean = SEGMENTATION_CONFIG.get("mean", [0.485, 0.456, 0.406])
        std = SEGMENTATION_CONFIG.get("std", [0.229, 0.224, 0.225])
        # Ensure they are lists. If not, convert numeric types into a one-element list.
        if not isinstance(mean, list):
            mean = [mean]
        if not isinstance(std, list):
            std = [std]
            
        super().__init__(
            image_size=SEGMENTATION_CONFIG["input_size"],
            mean=mean,
            std=std,
            convert_grayscale=False,  # For 3-channel images
            trt_fp16=SEGMENTATION_CONFIG.get("use_fp16", False),
            async_preproc=SEGMENTATION_CONFIG.get("async_preproc", False),
            compiled_module_cache=SEGMENTATION_CONFIG.get("compiled_module_cache", None),
            additional_transforms=SEGMENTATION_CONFIG.get("additional_transforms", [])
        )
        logger.debug("SegmentationPreprocessor initialized with image_size: " + str(SEGMENTATION_CONFIG["input_size"]))


    def _crop_and_resize(self, img: np.ndarray) -> np.ndarray:
        """
        Center-crop the input image to match the target aspect ratio and then resize it to the
        desired input dimensions.
        """
        logger.debug("Cropping and resizing image with shape: %s", img.shape)
        # Determine desired height and width.
        desired_size = self.image_size
        if isinstance(desired_size, int):
            desired_h, desired_w = desired_size, desired_size
        else:
            desired_h, desired_w = desired_size

        target_aspect = desired_w / desired_h
        h, w = img.shape[:2]
        current_aspect = w / h
        logger.debug("Current aspect ratio: %.4f, Target aspect ratio: %.4f", current_aspect, target_aspect)

        if current_aspect > target_aspect:
            # Image is too wide: crop width.
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            img = img[:, start_x:start_x + new_w]
            logger.debug("Cropped width: new_w=%d, start_x=%d", new_w, start_x)
        elif current_aspect < target_aspect:
            # Image is too tall: crop height.
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            img = img[start_y:start_y + new_h, :]
            logger.debug("Cropped height: new_h=%d, start_y=%d", new_h, start_y)
        else:
            logger.debug("No cropping needed for image.")

        # Resize to target dimensions.
        resized = cv2.resize(img, (desired_w, desired_h))
        logger.debug("Resized image to: (%d, %d)", desired_w, desired_h)
        return resized

    def __call__(self, inputs: Any) -> torch.Tensor:
        """
        Overrides the base call to support:
          - Batched tensor input (4D): splits into a list of images.
          - Single image tensor (3D): wraps into a list and squeezes later.
          - List/tuple of images: processed directly.
          - Other types: wrapped into a list.
        """
        logger.debug("Preprocessing inputs of type: %s", type(inputs))
        # Convert inputs into a list of images.
        if isinstance(inputs, torch.Tensor):
            if inputs.ndim == 4:
                images_list = [inputs[i] for i in range(inputs.size(0))]
                logger.debug("Received batched tensor with %d images", len(images_list))
            elif inputs.ndim == 3:
                images_list = [inputs]
                logger.debug("Received single image tensor")
            else:
                raise ValueError("Tensor input must be 3D (single image) or 4D (batched images)")
        elif isinstance(inputs, (list, tuple)):
            images_list = list(inputs)
            logger.debug("Received list/tuple of images with length: %d", len(images_list))
        else:
            images_list = [inputs]
            logger.debug("Wrapped single input of type: %s into list", type(inputs))

        processed_images = []
        for idx, img in enumerate(images_list):
            logger.debug("Processing image %d of type: %s", idx, type(img))
            # Convert string inputs (file paths) to numpy arrays.
            if isinstance(img, str):
                np_img = cv2.imread(img)
                if np_img is None:
                    logger.error("Failed to load image from path: %s", img)
                    raise ValueError(f"Failed to load image: {img}")
                # Convert from BGR to RGB.
                np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                logger.debug("Loaded image from path: %s with shape: %s", img, np_img.shape)
            elif isinstance(img, np.ndarray):
                np_img = img
                logger.debug("Input is numpy array with shape: %s", np_img.shape)
            elif isinstance(img, torch.Tensor):
                # Assume channel-first (C, H, W) format.
                np_img = img.detach().cpu().numpy().transpose(1, 2, 0)
                logger.debug("Converted torch tensor to numpy array with shape: %s", np_img.shape)
            else:
                # Assume it is a PIL image.
                np_img = np.array(img)
                logger.debug("Converted PIL image to numpy array with shape: %s", np_img.shape)

            # Apply center-crop and resize.
            np_img = self._crop_and_resize(np_img)
            processed_images.append(np_img)

        logger.debug("Completed preprocessing of %d images", len(processed_images))
        # Pass the list of preprocessed images to the base preprocessor.
        return super().__call__(processed_images)


class SegmentationPostprocessor(BasePostprocessor):
    """
    Postprocessor that converts raw model outputs into a segmentation mask and contours.
    Applies a threshold, finds contours with OpenCV, and filters small regions.
    """
    def __init__(self, threshold: float = 0.5, min_contour_area: int = 100):
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.contour_mode = cv2.RETR_EXTERNAL
        self.contour_method = cv2.CHAIN_APPROX_SIMPLE
        logger.debug(f"SegmentationPostprocessor initialized with threshold: {threshold:.2f} and min_contour_area: {min_contour_area}")


    def __call__(self, outputs: torch.Tensor) -> Tuple[np.ndarray, list]:
        """
        Process raw model outputs.
        """
        logger.debug("Postprocessing outputs of type: %s", type(outputs))
        # If multiple outputs are provided, assume the segmentation mask is the first element.
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
            logger.debug("Multiple outputs detected, using the first element as mask")

        # Convert to numpy array, squeeze extra dimensions, and apply threshold.
        mask_np = outputs.squeeze().cpu().numpy()
        logger.debug("Mask numpy shape after squeeze: %s", mask_np.shape)
        mask = ((mask_np > self.threshold) * 255).astype(np.uint8)
        logger.debug("Applied threshold: %f, unique values in mask: %s", self.threshold, np.unique(mask))

        # Find contours using OpenCV.
        contours, _ = cv2.findContours(mask, self.contour_mode, self.contour_method)
        logger.debug("Found %d total contours", len(contours))

        # Filter out small contours based on area.
        filtered_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
        logger.debug("Filtered contours count: %d", len(filtered_contours))

        return mask, filtered_contours


class SegmentationModel:
    """
    Segmentation model wrapper that integrates preprocessing, inference, and postprocessing.
    Supports both asynchronous and synchronous inference, along with benchmarking and cleanup.
    """
    def __init__(self, model: torch.nn.Module, config: Optional[EngineConfig] = None):
        self.config = config or self._create_default_config()
        # Use a default device ("cpu") if config does not provide one.
        self.device = torch.device(getattr(self.config, "device", "cpu"))
        logger.debug("SegmentationModel will use device: " + str(self.device))

        # Initialize components.
        self.preprocessor = SegmentationPreprocessor()
        self.postprocessor = SegmentationPostprocessor(
            threshold=SEGMENTATION_CONFIG["threshold"],
            min_contour_area=SEGMENTATION_CONFIG["min_contour_area"]
        )

        # Configure inference engine.
        self.engine = InferenceEngine(
            model=model,
            device=self.device,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_fp16=SEGMENTATION_CONFIG.get("use_fp16", False),
            use_tensorrt=SEGMENTATION_CONFIG.get("use_tensorrt", False),
            config=self.config
        )
        logger.debug("InferenceEngine initialized.")

        # If the engine supports dedicated batch processing, start its coroutine.
        if hasattr(self.engine, '_process_batches') and asyncio.iscoroutinefunction(self.engine._process_batches):
            self.engine._batch_task = asyncio.create_task(self.engine._process_batches())
            logger.debug("Started batch processing coroutine.")

        logger.info("Segmentation model initialized on %s", self.device)

    def _create_default_config(self) -> EngineConfig:
        """
        Create a default engine configuration optimized for segmentation.
        """
        default_config = EngineConfig(
            num_workers=4,
            batch_size=8,
            max_batch_size=32,
            enable_dynamic_batching=True,
            pid_kp=0.6,
            pid_ki=0.15,
            pid_kd=0.1,
            use_multigpu=SEGMENTATION_CONFIG.get("use_multigpu", False),
            guard_enabled=True,
            guard_confidence_threshold=0.7,
            guard_variance_threshold=0.03,
            num_classes=1
        )
        logger.debug(f"Created default engine configuration: {default_config}")
        return default_config

    async def process_image_async(self, image: Union[str, np.ndarray], timeout: Optional[float] = None) -> Dict:
        """
        Asynchronously process an image through the segmentation pipeline.
        """
        start_time = time.perf_counter()
        logger.debug("Starting asynchronous processing for image: %s", image)
        try:
            # If a timeout is specified, wrap the inference coroutine.
            if timeout:
                result = await asyncio.wait_for(self.engine.run_inference_async(image), timeout=timeout)
                logger.debug("Inference completed with timeout: %.2fs", timeout)
            else:
                result = await self.engine.run_inference_async(image)
                logger.debug("Inference completed without timeout")
            processing_time = time.perf_counter() - start_time
            logger.info("Inference completed in %.4fs", processing_time)
            return {
                "mask": result[0],
                "contours": result[1],
                "processing_time": processing_time
            }
        except Exception as e:
            logger.error("Segmentation failed: %s", e, exc_info=True)
            raise

    def process_image(self, image: Union[str, np.ndarray], timeout: Optional[float] = None) -> Dict:
        """
        Synchronous interface to process an image.
        """
        logger.debug("Starting synchronous processing for image: %s", image)
        try:
            # If there is no running loop, we can safely run asyncio.run.
            asyncio.get_running_loop()
        except RuntimeError:
            result = asyncio.run(self.process_image_async(image, timeout=timeout))
            logger.debug("Synchronous processing result: %s", result)
            return result
        else:
            logger.error("Called synchronous process_image from an async context")
            raise RuntimeError("Use the async interface 'process_image_async' in async contexts")

    def benchmark_segmentation(self, image: Union[str, np.ndarray], iterations: int = 10) -> Tuple[float, float]:
        """
        Run a benchmark on the segmentation pipeline.
        """
        logger.debug("Starting benchmark with %d iterations", iterations)
        times = []
        for i in range(iterations):
            start = time.perf_counter()
            # Run synchronous processing.
            self.process_image(image)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            logger.debug("Iteration %d: %.4fs", i+1, elapsed)
        mean_time = np.mean(times)
        std_time = np.std(times)
        logger.info("Benchmark: %d iterations, mean=%.4fs, std=%.4fs", iterations, mean_time, std_time)
        return mean_time, std_time

    async def shutdown(self):
        """
        Cleanly shutdown the segmentation model, including cancelling pending tasks and releasing resources.
        """
        logger.debug("Shutting down SegmentationModel")
        try:
            if hasattr(self.engine, '_batch_task'):
                self.engine._batch_task.cancel()
                await self.engine._batch_task
                logger.debug("Batch processing task cancelled.")
        except asyncio.CancelledError:
            logger.debug("Batch processing coroutine cancelled during shutdown.")
        finally:
            # Cancel any pending tasks in the preprocessor if supported.
            if hasattr(self.preprocessor, "cancel_pending_tasks"):
                self.preprocessor.cancel_pending_tasks()
                logger.debug("Cancelled pending tasks in preprocessor.")
            await self.engine.cleanup()
            logger.info("Segmentation model shutdown complete.")


async def main():
    """
    Example usage of the segmentation pipeline. Adjust the import as needed for your project.
    """
    logger.debug("Starting main function for segmentation example.")
    # Import your model-creation function (adjust the path accordingly)
    from models.segmentation_model import create_segmentation_model

    # Create a sample model.
    model = create_segmentation_model()
    logger.debug("Segmentation model created.")

    # Initialize the segmentation system.
    segmenter = SegmentationModel(model)

    try:
        # Process a sample image asynchronously (with optional timeout).
        result = await segmenter.process_image_async("sample.jpg", timeout=5.0)

        logger.debug("Received result: %s", result)
        print(f"Processing time: {result['processing_time']:.4f}s")
        print(f"Detected {len(result['contours'])} contours")

        # Visualize the segmentation mask.
        cv2.imshow("Segmentation Mask", result["mask"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    finally:
        await segmenter.shutdown()
        logger.debug("Main function shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())

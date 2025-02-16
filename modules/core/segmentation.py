import cv2
import numpy as np
import torch
import asyncio
from typing import Dict, Tuple, Union, Any, Optional
import sys
import os
import time

# Add the parent directory to sys.path so that imports like "core.engine" work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Local imports
from utils.config import SEGMENTATION_CONFIG
from utils.logger import get_logger
from core.engine import InferenceEngine, EngineConfig
from core.preprocessor import ImagePreprocessor  # Enhanced preprocessor implementation
from core.postprocessor import BasePostprocessor

logger = get_logger(__name__)


class SegmentationPreprocessor(ImagePreprocessor):
    """
    Segmentation-specific preprocessor that leverages the enhanced ImagePreprocessor.
    Supports batched tensor input, PIL Images, numpy arrays, and file paths.
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
            convert_grayscale=False,  # Corrected to False for 3-channel images
            trt_fp16=SEGMENTATION_CONFIG.get("use_fp16", False),
            async_preproc=SEGMENTATION_CONFIG.get("async_preproc", False),
            compiled_module_cache=SEGMENTATION_CONFIG.get("compiled_module_cache", None),
            additional_transforms=SEGMENTATION_CONFIG.get("additional_transforms", [])
        )

    def __call__(self, inputs: Any) -> torch.Tensor:
        """
        Overrides the base call to support:
          - Batched tensor input (4D): splits into a list of images.
          - Single image tensor (3D): wraps into a list and squeezes later.
          - List/tuple of images: processed directly.
          - Other types: wrapped into a list.
        """
        if isinstance(inputs, torch.Tensor):
            if inputs.ndim == 4:
                # Batched tensor input: process each image separately.
                inputs_list = [inputs[i] for i in range(inputs.size(0))]
                processed = super().__call__(inputs_list)
                return processed
            elif inputs.ndim == 3:
                # Single image tensor.
                return super().__call__([inputs]).squeeze(0)
            else:
                raise ValueError("Tensor input must be 3D (single image) or 4D (batched images)")
        elif isinstance(inputs, (list, tuple)):
            return super().__call__(inputs)
        else:
            return super().__call__([inputs])


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

    def __call__(self, outputs: torch.Tensor) -> Tuple[np.ndarray, list]:
        """
        Process raw model outputs.

        Args:
            outputs (torch.Tensor): Model output (or a tuple/list with the mask as first element).

        Returns:
            Tuple[np.ndarray, list]: A tuple containing the binary mask (uint8) and a list of filtered contours.
        """
        # If multiple outputs are provided, assume the segmentation mask is the first element.
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        # Convert to numpy array, squeeze extra dimensions, and apply threshold.
        mask_np = outputs.squeeze().cpu().numpy()
        mask = ((mask_np > self.threshold) * 255).astype(np.uint8)

        # Find contours using OpenCV.
        contours, _ = cv2.findContours(mask, self.contour_mode, self.contour_method)

        # Filter out small contours based on area.
        filtered_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]

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

        # If the engine supports dedicated batch processing, start its coroutine.
        if hasattr(self.engine, '_process_batches') and asyncio.iscoroutinefunction(self.engine._process_batches):
            self.engine._batch_task = asyncio.create_task(self.engine._process_batches())

        logger.info(f"Segmentation model initialized on {self.device}")

    def _create_default_config(self) -> EngineConfig:
        """
        Create a default engine configuration optimized for segmentation.
        """
        return EngineConfig(
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

    async def process_image_async(self, image: Union[str, np.ndarray], timeout: Optional[float] = None) -> Dict:
        """
        Asynchronously process an image through the segmentation pipeline.

        Args:
            image (Union[str, np.ndarray]): Input image (file path or numpy array).
            timeout (float, optional): Maximum allowed inference time in seconds.

        Returns:
            Dict: Dictionary containing the segmentation mask, detected contours, and processing time.
        """
        start_time = time.perf_counter()
        try:
            # If a timeout is specified, wrap the inference coroutine.
            if timeout:
                result = await asyncio.wait_for(self.engine.run_inference_async(image), timeout=timeout)
            else:
                result = await self.engine.run_inference_async(image)
            processing_time = time.perf_counter() - start_time
            logger.info(f"Inference completed in {processing_time:.4f}s")
            return {
                "mask": result[0],
                "contours": result[1],
                "processing_time": processing_time
            }
        except Exception as e:
            logger.error(f"Segmentation failed: {e}", exc_info=True)
            raise

    def process_image(self, image: Union[str, np.ndarray], timeout: Optional[float] = None) -> Dict:
        """
        Synchronous interface to process an image.
        If called from an asynchronous context, a RuntimeError is raised.

        Args:
            image (Union[str, np.ndarray]): Input image.
            timeout (float, optional): Maximum allowed inference time in seconds.

        Returns:
            Dict: Inference results.

        Raises:
            RuntimeError: When called from an async context.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.process_image_async(image, timeout=timeout))
        else:
            raise RuntimeError("Use the async interface 'process_image_async' in async contexts")

    def benchmark_segmentation(self, image: Union[str, np.ndarray], iterations: int = 10) -> Tuple[float, float]:
        """
        Run a benchmark on the segmentation pipeline.

        Args:
            image (Union[str, np.ndarray]): Input image.
            iterations (int): Number of iterations.

        Returns:
            Tuple[float, float]: Mean and standard deviation of processing times.
        """
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            # Run synchronous processing.
            self.process_image(image)
            times.append(time.perf_counter() - start)
        mean_time = np.mean(times)
        std_time = np.std(times)
        logger.info(f"Benchmark: {iterations} iterations, mean={mean_time:.4f}s, std={std_time:.4f}s")
        return mean_time, std_time

    async def shutdown(self):
        """
        Cleanly shutdown the segmentation model, including cancelling pending tasks and releasing resources.
        """
        try:
            if hasattr(self.engine, '_batch_task'):
                self.engine._batch_task.cancel()
                await self.engine._batch_task
        except asyncio.CancelledError:
            logger.debug("Batch processing task cancelled.")
        finally:
            # Cancel any pending tasks in the preprocessor if supported.
            if hasattr(self.preprocessor, "cancel_pending_tasks"):
                self.preprocessor.cancel_pending_tasks()
            await self.engine.cleanup()
            logger.info("Segmentation model shutdown complete.")


async def main():
    """
    Example usage of the segmentation pipeline. Adjust the import as needed for your project.
    """
    # Import your model-creation function (adjust the path accordingly)
    from models.segmentation_model import create_segmentation_model

    # Create a sample model.
    model = create_segmentation_model()

    # Initialize the segmentation system.
    segmenter = SegmentationModel(model)

    try:
        # Process a sample image asynchronously (with optional timeout).
        result = await segmenter.process_image_async("sample.jpg", timeout=5.0)

        print(f"Processing time: {result['processing_time']:.4f}s")
        print(f"Detected {len(result['contours'])} contours")

        # Visualize the segmentation mask.
        cv2.imshow("Segmentation Mask", result["mask"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    finally:
        await segmenter.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
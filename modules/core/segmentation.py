import cv2
import numpy as np
import torch
import asyncio
from typing import Dict, Tuple, Union, Any
import sys
import os
import time

# Add the parent directory to sys.path so that imports like "core.engine" work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Local imports
from utils.config import SEGMENTATION_CONFIG
from utils.logger import get_logger
from core.engine import InferenceEngine, EngineConfig
from core.preprocessor import ImagePreprocessor  # New enhanced preprocessor implementation
from core.postprocessor import BasePostprocessor

logger = get_logger(__name__)

class SegmentationPreprocessor(ImagePreprocessor):
    """
    Segmentation-specific preprocessor with optimized image handling.
    Inherits from the enhanced ImagePreprocessor that now supports:
      - Asynchronous processing
      - Compiled module caching
      - Additional (custom) transforms
    """
    def __init__(self):
        super().__init__(
            image_size=SEGMENTATION_CONFIG["input_size"],
            mean=SEGMENTATION_CONFIG["mean"],
            std=SEGMENTATION_CONFIG["std"],
            device=SEGMENTATION_CONFIG.get("device", "cuda"),
            convert_grayscale=True,
            trt_fp16=SEGMENTATION_CONFIG.get("use_fp16", False),
            async_preproc=SEGMENTATION_CONFIG.get("async_preproc", False),
            compiled_module_cache=SEGMENTATION_CONFIG.get("compiled_module_cache", None),
            additional_transforms=SEGMENTATION_CONFIG.get("additional_transforms", [])
        )

    def __call__(self, inputs: Any) -> torch.Tensor:
        # If the input is a torch.Tensor
        if isinstance(inputs, torch.Tensor):
            # If already batched: shape [batch, C, H, W]
            if inputs.ndim == 4:
                processed_images = []
                for i in range(inputs.size(0)):
                    # Process each image (shape: [C, H, W])
                    img = inputs[i]
                    # Call the parent's __call__ method with a single image wrapped in a list
                    processed_img = super().__call__([img])
                    # The parent's method returns a batched tensor [1, C, H, W]; remove the extra batch dim.
                    processed_images.append(processed_img.squeeze(0))
                # Stack back into a single 4D tensor: [batch, C, H, W]
                return torch.stack(processed_images, dim=0)
            elif inputs.ndim == 3:
                # Single unbatched image: shape [C, H, W]
                return super().__call__([inputs]).squeeze(0)
            else:
                raise ValueError("Tensor input must be 3D (single image) or 4D (batched images)")
        elif isinstance(inputs, (list, tuple)):
            # Assume it's a list of unbatched images.
            return super().__call__(inputs)
        else:
            # For other input types, wrap them in a list.
            return super().__call__([inputs])

class SegmentationPostprocessor(BasePostprocessor):
    def __init__(self, threshold=0.5, min_contour_area=100):
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.contour_mode = cv2.RETR_EXTERNAL
        self.contour_method = cv2.CHAIN_APPROX_SIMPLE

    def __call__(self, outputs: torch.Tensor) -> Tuple[np.ndarray, list]:
        """Enhanced output handling with batch support"""
        # Handle multi-output models: assume the first element is the segmentation mask.
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
            
        # Convert to numpy, squeeze extra dimensions, and threshold.
        mask_np = outputs.squeeze().cpu().numpy()
        mask = ((mask_np > self.threshold) * 255).astype(np.uint8)
        
        # Find contours with OpenCV.
        contours, _ = cv2.findContours(
            mask, 
            self.contour_mode, 
            self.contour_method
        )
        
        # Filter out small contours.
        areas = [cv2.contourArea(c) for c in contours]
        filtered_contours = [c for c, a in zip(contours, areas) 
                             if a > self.min_contour_area]

        return mask, filtered_contours


class SegmentationModel:
    def __init__(self, model: torch.nn.Module, config: EngineConfig = None):
        self.config = config or self._create_default_config()
        self.device = torch.device(self.config.device)

        # Initialize components with enhanced preprocessing.
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

        # Initialize asynchronous batch processing if available.
        self._init_async_processing()
        
        logger.info(f"Segmentation model initialized on {self.device}")

    def _create_default_config(self) -> EngineConfig:
        """Segmentation-optimized engine configuration"""
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

    def _init_async_processing(self):
        """Initialize async batch processing if needed."""
        if hasattr(self.engine, '_process_batches') and \
           asyncio.iscoroutinefunction(self.engine._process_batches):
            self.engine._batch_task = asyncio.create_task(
                self.engine._process_batches()
            )

    async def process_image_async(self, image: Union[str, np.ndarray]) -> Dict:
        """Enhanced async processing with timing metrics."""
        start_time = time.perf_counter()
        
        try:
            result = await self.engine.run_inference_async(image)
            return {
                "mask": result[0],
                "contours": result[1],
                "processing_time": time.perf_counter() - start_time
            }
        except Exception as e:
            logger.error(f"Segmentation failed: {str(e)}", exc_info=True)
            raise

    def process_image(self, image: Union[str, np.ndarray]) -> Dict:
        """
        Synchronous interface to process an image.
        Raises a RuntimeError if called from within an async context.
        """
        try:
            # This call will succeed only if there is a running loop.
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop; safe to use asyncio.run().
            return asyncio.run(self.process_image_async(image))
        else:
            # There is an active event loop.
            raise RuntimeError("Use the async interface 'process_image_async' in async contexts")

    async def shutdown(self):
        """Enhanced cleanup with error handling."""
        try:
            if hasattr(self.engine, '_batch_task'):
                self.engine._batch_task.cancel()
                await self.engine._batch_task
        except asyncio.CancelledError:
            pass
        finally:
            await self.engine.cleanup()
            logger.info("Segmentation model shutdown complete")


async def main():
    # Example usage: adjust the import as needed.
    from models.segmentation_model import create_segmentation_model

    # Create sample model.
    model = create_segmentation_model()

    # Initialize segmentation system.
    segmenter = SegmentationModel(model)

    try:
        # Process sample image asynchronously.
        result = await segmenter.process_image_async("sample.jpg")

        print(f"Processing time: {result['processing_time']:.4f}s")
        print(f"Detected {len(result['contours'])} contours")

        # Visualize results.
        cv2.imshow("Segmentation Mask", result["mask"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    finally:
        await segmenter.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

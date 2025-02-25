import cv2
import numpy as np
import torch
import asyncio
from typing import Dict, Tuple, Union, Any, Optional
import time
import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Local imports from your project
from utils.config import SEGMENTATION_CONFIG, EngineConfig
from utils.logger import get_logger
from core.engine import InferenceEngine
from core.preprocessor import ImagePreprocessor
from core.postprocessor import BasePostprocessor
from core.pid import PIDController

# Import the downloader function for fetching models from Hugging Face Hub or a custom URL
from models.downloader import download_model

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)


################################################################################
# Segmentation Preprocessor
################################################################################
class SegmentationPreprocessor(ImagePreprocessor):
    """
    Preprocessor for segmentation tasks that converts various image inputs into
    a 4D tensor [B, C, H, W]. It performs center-cropping, resizing, and normalization.
    """
    def __init__(self):
        mean = SEGMENTATION_CONFIG.get("mean", [0.485, 0.456, 0.406])
        std  = SEGMENTATION_CONFIG.get("std",  [0.229, 0.224, 0.225])
        mean = mean if isinstance(mean, list) else [mean]
        std  = std if isinstance(std, list) else [std]
        super().__init__(
            image_size=SEGMENTATION_CONFIG["input_size"],
            mean=mean,
            std=std,
            convert_grayscale=False,
            trt_fp16=SEGMENTATION_CONFIG.get("use_fp16", False),
            async_preproc=SEGMENTATION_CONFIG.get("async_preproc", False),
            compiled_module_cache=SEGMENTATION_CONFIG.get("compiled_module_cache", None),
            additional_transforms=SEGMENTATION_CONFIG.get("additional_transforms", [])
        )
        logger.debug(f"Initialized SegmentationPreprocessor with image_size: {self.image_size}")
        self.mean = mean
        self.std = std

    def _crop_and_resize(self, img: np.ndarray) -> np.ndarray:
        """Center-crop and resize the image to the desired dimensions."""
        desired_size = self.image_size
        if isinstance(desired_size, int):
            desired_h, desired_w = desired_size, desired_size
        else:
            desired_h, desired_w = desired_size

        target_aspect = desired_w / desired_h
        h, w = img.shape[:2]
        current_aspect = w / h

        if current_aspect > target_aspect:
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            img = img[:, start_x:start_x + new_w]
        elif current_aspect < target_aspect:
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            img = img[start_y:start_y + new_h, :]

        return cv2.resize(img, (desired_w, desired_h))

    def __call__(self, inputs: Any) -> torch.Tensor:
        """
        Convert the input (file path, numpy array, PIL image, or tensor) into a 
        4D tensor [B, C, H, W] with proper cropping, resizing, and normalization.
        """
        if isinstance(inputs, torch.Tensor):
            if inputs.ndim == 4:
                images_list = [inputs[i] for i in range(inputs.size(0))]
            elif inputs.ndim == 3:
                images_list = [inputs]
            else:
                raise ValueError("Tensor input must be 3D or 4D for segmentation")
        elif isinstance(inputs, (list, tuple)):
            images_list = list(inputs)
        else:
            images_list = [inputs]

        processed_tensors = []
        for img in images_list:
            if isinstance(img, str):
                np_img = cv2.imread(img)
                if np_img is None:
                    raise ValueError(f"Failed to load image: {img}")
                np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
            elif isinstance(img, np.ndarray):
                np_img = img
            elif isinstance(img, torch.Tensor):
                np_img = img.detach().cpu().numpy().transpose(1, 2, 0)
            else:
                np_img = np.array(img)

            if np_img.ndim != 3 or np_img.shape[-1] != 3:
                raise ValueError("Each image must be an RGB image with 3 channels.")

            np_img = self._crop_and_resize(np_img)
            np_img = np.transpose(np_img, (2, 0, 1))
            img_tensor = torch.from_numpy(np_img).float()
            for c in range(3):
                img_tensor[c] = (img_tensor[c] - self.mean[c]) / self.std[c]
            processed_tensors.append(img_tensor)

        return torch.stack(processed_tensors, dim=0)


################################################################################
# Segmentation Postprocessor
################################################################################
class SegmentationPostprocessor(BasePostprocessor):
    """
    Postprocessor that converts the model output into a binary mask and extracts contours.
    
    For a standard segmentation model, it applies an optional sigmoid, thresholding,
    and contour extraction. For YOLO outputs, it attempts to extract instance masks
    and combine them into a binary mask.
    """
    def __init__(self, threshold: float = 0.5, min_contour_area: int = 100, apply_sigmoid: bool = False):
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.apply_sigmoid = apply_sigmoid
        self.contour_mode = cv2.RETR_EXTERNAL
        self.contour_method = cv2.CHAIN_APPROX_SIMPLE
        logger.debug(f"Initialized SegmentationPostprocessor: threshold={threshold}, "
                     f"min_contour_area={min_contour_area}, apply_sigmoid={apply_sigmoid}")

    def __call__(self, outputs: Any) -> Tuple[np.ndarray, list]:
        # Case 1: Standard segmentation tensor output
        if isinstance(outputs, torch.Tensor):
            if outputs.dim() == 4 and outputs.size(0) == 1:
                outputs = outputs.squeeze(0)
            elif outputs.dim() == 3 and outputs.size(0) == 1:
                outputs = outputs.squeeze(0)
            mask_np = outputs.detach().cpu().numpy()
            if self.apply_sigmoid:
                mask_np = 1 / (1 + np.exp(-mask_np))
            mask = ((mask_np > self.threshold) * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask, self.contour_mode, self.contour_method)
            filtered_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
            return mask, filtered_contours

        # Case 2: YOLO returns a list of Results objects (if available)
        elif isinstance(outputs, list) and len(outputs) > 0:
            first_res = outputs[0]
            if hasattr(first_res, "masks") and first_res.masks is not None:
                instance_masks = first_res.masks.data  # Expected shape: [N, H, W]
                if len(instance_masks) == 0:
                    return np.zeros((1, 1), dtype=np.uint8), []
                combined_mask = torch.zeros_like(instance_masks[0], dtype=torch.uint8)
                for imask in instance_masks:
                    mask_bool = imask > self.threshold
                    combined_mask = torch.maximum(combined_mask, mask_bool.to(torch.uint8) * 255)
                combined_mask_np = combined_mask.detach().cpu().numpy()
                contours, _ = cv2.findContours(combined_mask_np, self.contour_mode, self.contour_method)
                filtered_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
                return combined_mask_np, filtered_contours
            else:
                logger.warning("YOLO output has no masks. Returning empty mask/contours.")
                return np.zeros((1, 1), dtype=np.uint8), []
        else:
            logger.warning("Postprocessor received unrecognized output, returning empty results.")
            return np.zeros((1, 1), dtype=np.uint8), []


################################################################################
# Segmentation Model Pipeline
################################################################################
class SegmentationModel:
    """
    Manages the segmentation pipeline by integrating preprocessing, inference,
    and postprocessing. Provides both asynchronous and synchronous interfaces.
    """
    def __init__(self, model: torch.nn.Module, config: Optional[EngineConfig] = None):
        self.config = config or self._create_default_config()
        if SEGMENTATION_CONFIG.get("use_multigpu", False) and torch.cuda.device_count() > 1:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            devices = [self.config.device]
        logger.debug(f"SegmentationModel devices: {devices}")

        self.preprocessor = SegmentationPreprocessor()
        self.postprocessor = SegmentationPostprocessor(
            threshold=SEGMENTATION_CONFIG["threshold"],
            min_contour_area=SEGMENTATION_CONFIG["min_contour_area"],
            apply_sigmoid=SEGMENTATION_CONFIG.get("apply_sigmoid", False)
        )

        self.engine = InferenceEngine(
            model=model,
            device=devices,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_fp16=SEGMENTATION_CONFIG.get("use_fp16", False),
            use_tensorrt=SEGMENTATION_CONFIG.get("use_tensorrt", False),
            config=self.config
        )

    def _create_default_config(self) -> EngineConfig:
        pid = PIDController(kp=0.6, ki=0.15, kd=0.1, setpoint=50.0)
        return EngineConfig(
            num_workers=4,
            queue_size=32,
            batch_size=4,
            min_batch_size=1,
            max_batch_size=8,
            warmup_runs=2,
            timeout=5.0,
            batch_wait_timeout=0.01,
            autoscale_interval=1.0,
            debug_mode=True,
            pid_kp=0.6,
            pid_ki=0.15,
            pid_kd=0.1,
            guard_enabled=True,
            guard_confidence_threshold=0.7,
            guard_variance_threshold=0.03,
            num_classes=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            async_mode=True
        )

    async def process_image_async(self, image: Union[str, np.ndarray], timeout: Optional[float] = None) -> Dict:
        """
        Asynchronously process a single image (given as a file path or numpy array).
        Returns a dict with keys: "mask", "contours", and "processing_time".
        """
        start_time = time.perf_counter()
        try:
            coro = self.engine.run_inference_async(image)
            result = await asyncio.wait_for(coro, timeout) if timeout is not None else await coro
            processing_time = time.perf_counter() - start_time
            return {
                "mask": result[0],
                "contours": result[1],
                "processing_time": processing_time
            }
        except asyncio.TimeoutError:
            logger.error(f"Inference timed out after {timeout}s for image.")
            raise
        except Exception as e:
            logger.error(f"Segmentation failed: {e}", exc_info=True)
            raise

    def process_image(self, image: Union[str, np.ndarray], timeout: Optional[float] = None) -> Dict:
        """
        Synchronously process a single image by creating an event loop if needed.
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError("Use 'process_image_async' in an existing async context.")
        except RuntimeError:
            return asyncio.run(self.process_image_async(image, timeout=timeout))

    def benchmark_segmentation(self, image: Union[str, np.ndarray], iterations: int = 10) -> Tuple[float, float]:
        """
        Benchmark the synchronous segmentation pipeline.
        Returns the mean and standard deviation of processing times.
        """
        times = []
        for _ in range(iterations):
            result = self.process_image(image)
            times.append(result["processing_time"])
        mean_time = np.mean(times)
        std_time = np.std(times)
        logger.info(f"Benchmark: {iterations} iterations, mean={mean_time:.4f}s, std={std_time:.4f}s")
        return mean_time, std_time

    async def shutdown(self):
        """
        Asynchronously clean up engine resources.
        """
        if hasattr(self.preprocessor, "cancel_pending_tasks"):
            self.preprocessor.cancel_pending_tasks()
        await self.engine.cleanup()
        logger.info("SegmentationModel shutdown complete")


################################################################################
# Example main: Demonstrate usage with a YOLO segmentation model downloaded from a custom URL
################################################################################
async def main():
    try:
        # Download the YOLO11 model checkpoint from a custom URL.
        model_checkpoint = download_model(
            custom_url="https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11s.pt"
        )
    except Exception as e:
        logger.error("Failed to download model checkpoint from custom URL. "
                     "Please verify the URL.", exc_info=True)
        return

    # Load the model checkpoint. This assumes the checkpoint is a PyTorch state dict.
    try:
        model = torch.load(model_checkpoint, map_location="cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        logger.error("Failed to load the model checkpoint. Check the file format.", exc_info=True)
        return

    # Initialize the segmentation pipeline
    segmenter = SegmentationModel(model)
    try:
        # List of example images (local paths or URLs)
        images = [
            "https://ultralytics.com/images/zidane.jpg",
            "https://ultralytics.com/images/zidane.jpg"
        ]
        tasks = [segmenter.process_image_async(img, timeout=5.0) for img in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Image {idx+1} failed: {result}")
            else:
                print(f"Image {idx+1}: Processing time: {result['processing_time']:.4f}s, Contours: {len(result['contours'])}")
    finally:
        await segmenter.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

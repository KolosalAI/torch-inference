import cv2
import numpy as np
import torch
import asyncio
from typing import Dict, Tuple, Union, Any, Optional, List
import time
import logging
import os
import sys
from pathlib import Path
import requests
from io import BytesIO

# Configure proper path for relative imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Local imports from your project
try:
    from utils.config import SEGMENTATION_CONFIG, EngineConfig
    from utils.logger import get_logger
    from modules.core.inference_engine import InferenceEngine
    from core.preprocessor import ImagePreprocessor
    from core.postprocessor import BasePostprocessor
    from core.pid import PIDController
    from models.downloader import download_model
except ImportError as e:
    # Provide fallback definitions if imports fail
    print(f"Import error: {e}. Using fallback definitions.")
    
    # Fallback logger
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    # Fallback config
    SEGMENTATION_CONFIG = {
        "input_size": (640, 640),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "threshold": 0.5,
        "min_contour_area": 100,
        "apply_sigmoid": False,
        "use_fp16": False,
        "use_tensorrt": False,
        "async_preproc": True,
    }
    
    class EngineConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

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
        std = SEGMENTATION_CONFIG.get("std", [0.229, 0.224, 0.225])
        mean = mean if isinstance(mean, list) else [mean]
        std = std if isinstance(std, list) else [std]
        
        # Handle possible inheritance issues
        try:
            super().__init__(
                image_size=SEGMENTATION_CONFIG.get("input_size", (640, 640)),
                mean=mean,
                std=std,
                convert_grayscale=False,
                trt_fp16=SEGMENTATION_CONFIG.get("use_fp16", False),
                async_preproc=SEGMENTATION_CONFIG.get("async_preproc", False),
                compiled_module_cache=SEGMENTATION_CONFIG.get("compiled_module_cache", None),
                additional_transforms=SEGMENTATION_CONFIG.get("additional_transforms", [])
            )
        except (TypeError, AttributeError) as e:
            logger.warning(f"Failed to initialize parent class: {e}. Using standalone implementation.")
            self.image_size = SEGMENTATION_CONFIG.get("input_size", (640, 640))
            self.async_preproc = SEGMENTATION_CONFIG.get("async_preproc", False)
            self.pending_tasks = []
            
        logger.debug(f"Initialized SegmentationPreprocessor with image_size: {self.image_size}")
        self.mean = mean
        self.std = std

    def _crop_and_resize(self, img: np.ndarray) -> np.ndarray:
        """Center-crop and resize the image to the desired dimensions."""
        if img is None or img.size == 0:
            raise ValueError("Empty or invalid image provided")
            
        desired_size = self.image_size
        if isinstance(desired_size, int):
            desired_h, desired_w = desired_size, desired_size
        else:
            desired_h, desired_w = desired_size

        # Handle potential shape issues
        if len(img.shape) < 2:
            raise ValueError(f"Invalid image shape: {img.shape}")
            
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions: {h}x{w}")
            
        target_aspect = desired_w / desired_h
        current_aspect = w / h

        if current_aspect > target_aspect:
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            img = img[:, max(0, start_x):min(w, start_x + new_w)]
        elif current_aspect < target_aspect:
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            img = img[max(0, start_y):min(h, start_y + new_h), :]

        return cv2.resize(img, (desired_w, desired_h), interpolation=cv2.INTER_AREA)

    async def preprocess_async(self, input_data: Any) -> torch.Tensor:
        """Asynchronous preprocessing implementation"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.__call__, input_data)
        
    def cancel_pending_tasks(self):
        """Cancel any pending preprocessing tasks"""
        if hasattr(self, 'pending_tasks'):
            for task in self.pending_tasks:
                if not task.done():
                    task.cancel()
            self.pending_tasks = []

    def __call__(self, inputs: Any) -> torch.Tensor:
        """
        Convert the input (file path, numpy array, PIL image, or tensor) into a 
        4D tensor [B, C, H, W] with proper cropping, resizing, and normalization.
        """
        if inputs is None:
            raise ValueError("Input cannot be None")
            
        if isinstance(inputs, torch.Tensor):
            if inputs.ndim == 4:
                images_list = [inputs[i] for i in range(inputs.size(0))]
            elif inputs.ndim == 3:
                images_list = [inputs]
            else:
                raise ValueError(f"Tensor input must be 3D or 4D for segmentation, got shape {inputs.shape}")
        elif isinstance(inputs, (list, tuple)):
            images_list = list(inputs)
        else:
            images_list = [inputs]

        processed_tensors = []
        for img in images_list:
            try:
                if isinstance(img, str):
                    # Handle URLs vs local file paths
                    if img.startswith(('http://', 'https://')):
                        response = requests.get(img, timeout=10)
                        response.raise_for_status()
                        img_array = np.frombuffer(response.content, dtype=np.uint8)
                        np_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    else:
                        np_img = cv2.imread(img)
                    
                    if np_img is None:
                        raise ValueError(f"Failed to load image: {img}")
                    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                elif isinstance(img, np.ndarray):
                    np_img = img.copy()  # Create a copy to avoid modifying the original
                    # Convert BGR to RGB if needed
                    if np_img.ndim == 3 and np_img.shape[2] == 3:
                        if np.max(np_img) > 1.0 and np_img.dtype != np.uint8:
                            np_img = np_img.astype(np.uint8)
                elif isinstance(img, torch.Tensor):
                    # Handle different tensor formats
                    if img.ndim == 3 and img.shape[0] in [1, 3]:  # CHW format
                        np_img = img.detach().cpu().numpy().transpose(1, 2, 0)
                    elif img.ndim == 3 and img.shape[2] in [1, 3]:  # HWC format
                        np_img = img.detach().cpu().numpy()
                    else:
                        raise ValueError(f"Unsupported tensor shape: {img.shape}")
                else:
                    np_img = np.array(img)

                # Ensure we have an RGB image
                if np_img.ndim == 2:  # Grayscale
                    np_img = np.stack([np_img] * 3, axis=2)
                elif np_img.ndim == 3 and np_img.shape[2] == 1:  # Single channel
                    np_img = np.concatenate([np_img] * 3, axis=2)
                elif np_img.ndim == 3 and np_img.shape[2] == 4:  # RGBA
                    np_img = np_img[:, :, :3]
                elif np_img.ndim != 3 or np_img.shape[2] != 3:
                    raise ValueError(f"Unsupported image format with shape {np_img.shape}")

                # Apply preprocessing
                np_img = self._crop_and_resize(np_img)
                
                # Convert to float and normalize
                if np_img.dtype != np.float32:
                    np_img = np_img.astype(np.float32)
                    if np_img.max() > 1.0:
                        np_img /= 255.0
                
                # Change to CHW format
                np_img = np.transpose(np_img, (2, 0, 1))
                img_tensor = torch.from_numpy(np_img).float()
                
                # Apply normalization
                for c in range(min(3, img_tensor.shape[0])):
                    img_tensor[c] = (img_tensor[c] - self.mean[min(c, len(self.mean)-1)]) / self.std[min(c, len(self.std)-1)]
                
                processed_tensors.append(img_tensor)
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                raise

        if not processed_tensors:
            raise ValueError("No valid images to process")
            
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
        try:
            super().__init__()
        except (TypeError, AttributeError) as e:
            logger.warning(f"Failed to initialize parent class: {e}")
            
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.apply_sigmoid = apply_sigmoid
        self.contour_mode = cv2.RETR_EXTERNAL
        self.contour_method = cv2.CHAIN_APPROX_SIMPLE
        logger.debug(f"Initialized SegmentationPostprocessor: threshold={threshold}, "
                     f"min_contour_area={min_contour_area}, apply_sigmoid={apply_sigmoid}")

    def process_yolo_output(self, outputs) -> Tuple[np.ndarray, List]:
        """Process YOLO-specific outputs to extract masks and contours"""
        try:
            # Try to get masks from YOLO format output
            if hasattr(outputs[0], "masks") and outputs[0].masks is not None:
                instance_masks = outputs[0].masks.data
                if len(instance_masks) == 0:
                    return np.zeros((640, 640), dtype=np.uint8), []
                    
                # Combine all masks
                combined_mask = torch.zeros_like(instance_masks[0], dtype=torch.uint8)
                for imask in instance_masks:
                    mask_bool = imask > self.threshold
                    combined_mask = torch.maximum(combined_mask, mask_bool.to(torch.uint8) * 255)
                
                # Convert to numpy and find contours
                combined_mask_np = combined_mask.detach().cpu().numpy()
                if combined_mask_np.size == 0:
                    return np.zeros((640, 640), dtype=np.uint8), []
                    
                contours, _ = cv2.findContours(combined_mask_np, self.contour_mode, self.contour_method)
                filtered_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
                return combined_mask_np, filtered_contours
            elif isinstance(outputs, dict) and "masks" in outputs:
                # Alternative format
                masks = outputs["masks"]
                if isinstance(masks, torch.Tensor):
                    mask_np = masks.detach().cpu().numpy()
                else:
                    mask_np = np.asarray(masks)
                
                binary_mask = ((mask_np > self.threshold) * 255).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, self.contour_mode, self.contour_method)
                filtered_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
                return binary_mask, filtered_contours
            else:
                logger.warning("YOLO output has no recognizable mask format")
                return np.zeros((640, 640), dtype=np.uint8), []
        except Exception as e:
            logger.error(f"Error processing YOLO output: {e}")
            return np.zeros((640, 640), dtype=np.uint8), []

    def __call__(self, outputs: Any) -> Tuple[np.ndarray, list]:
        """Process model outputs to produce masks and contours"""
        try:
            # Case 1: Standard segmentation tensor output
            if isinstance(outputs, torch.Tensor):
                if outputs.dim() == 4:
                    # Handle output shape [B, C, H, W]
                    if outputs.size(0) == 1:  # Single batch
                        mask_logits = outputs[0]
                        if outputs.size(1) > 1:  # Multiple classes
                            # Take argmax for multi-class segmentation
                            mask_logits = torch.argmax(mask_logits, dim=0).float()
                    else:
                        logger.warning(f"Expected batch size 1, got {outputs.size(0)}. Using first result.")
                        mask_logits = outputs[0]
                elif outputs.dim() == 3:
                    # Could be [C, H, W] or [1, H, W]
                    mask_logits = outputs
                elif outputs.dim() == 2:
                    # Already [H, W]
                    mask_logits = outputs
                else:
                    raise ValueError(f"Unsupported tensor shape: {outputs.shape}")
                
                # Convert to numpy and apply sigmoid if needed
                mask_np = mask_logits.detach().cpu().numpy()
                if self.apply_sigmoid:
                    mask_np = 1 / (1 + np.exp(-mask_np))
                
                # Create binary mask
                if mask_np.ndim > 2:  # Handle multi-dimensional output
                    if mask_np.shape[0] == 1:
                        mask_np = mask_np[0]  # Take first channel if shape is [1, H, W]
                    elif mask_np.shape[0] > 1:
                        mask_np = np.argmax(mask_np, axis=0)  # Multi-class segmentation
                
                mask = ((mask_np > self.threshold) * 255).astype(np.uint8)
                
                # Find and filter contours
                contours, _ = cv2.findContours(mask, self.contour_mode, self.contour_method)
                filtered_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
                return mask, filtered_contours

            # Case 2: YOLO or other structured outputs
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                return self.process_yolo_output(outputs)
            
            # Case 3: Dictionary output format
            elif isinstance(outputs, dict):
                if "masks" in outputs:
                    masks = outputs["masks"]
                    if isinstance(masks, torch.Tensor):
                        return self.__call__(masks)  # Process the mask tensor
                    elif isinstance(masks, np.ndarray):
                        binary_mask = ((masks > self.threshold) * 255).astype(np.uint8)
                        contours, _ = cv2.findContours(binary_mask, self.contour_mode, self.contour_method)
                        filtered_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
                        return binary_mask, filtered_contours
                
                # Handle other dictionary formats
                logger.warning("Unrecognized dictionary output format")
                return np.zeros((640, 640), dtype=np.uint8), []
            else:
                logger.warning(f"Unrecognized output type: {type(outputs)}")
                return np.zeros((640, 640), dtype=np.uint8), []
                
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
            return np.zeros((640, 640), dtype=np.uint8), []


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
        self.model = model
        
        # Check for multi-GPU setup
        if SEGMENTATION_CONFIG.get("use_multigpu", False) and torch.cuda.device_count() > 1:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            # When using multiple GPUs, wrap the model in DataParallel
            if not isinstance(model, torch.nn.DataParallel):
                self.model = torch.nn.DataParallel(model)
        else:
            devices = [self.config.device]
            
        logger.debug(f"SegmentationModel devices: {devices}")

        self.preprocessor = SegmentationPreprocessor()
        self.postprocessor = SegmentationPostprocessor(
            threshold=SEGMENTATION_CONFIG.get("threshold", 0.5),
            min_contour_area=SEGMENTATION_CONFIG.get("min_contour_area", 100),
            apply_sigmoid=SEGMENTATION_CONFIG.get("apply_sigmoid", False)
        )

        # Try to initialize the engine with proper error handling
        try:
            self.engine = InferenceEngine(
                model=self.model,
                device=devices,
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                use_fp16=SEGMENTATION_CONFIG.get("use_fp16", False),
                use_tensorrt=SEGMENTATION_CONFIG.get("use_tensorrt", False),
                config=self.config
            )
        except (ImportError, NameError, AttributeError) as e:
            logger.error(f"Failed to initialize InferenceEngine: {e}")
            logger.warning("Falling back to direct model inference")
            self.engine = None
            # Move model to appropriate device
            self.device = torch.device(self.config.device if self.config.device else 
                                       ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model.to(self.device)
            self.model.eval()

    def _create_default_config(self) -> EngineConfig:
        """Create a default configuration for the inference engine"""
        try:
            pid = PIDController(kp=0.6, ki=0.15, kd=0.1, setpoint=50.0)
        except (ImportError, NameError, AttributeError):
            # Fallback if PIDController is not available
            pid = None
            
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

    async def _direct_inference_async(self, image: Union[str, np.ndarray]) -> Tuple[np.ndarray, List]:
        """Fallback method for direct inference when engine isn't available"""
        with torch.no_grad():
            # Preprocess
            if hasattr(self.preprocessor, 'preprocess_async'):
                input_tensor = await self.preprocessor.preprocess_async(image)
            else:
                input_tensor = self.preprocessor(image)
                
            # Move to device
            input_tensor = input_tensor.to(self.device)
            
            # Run inference
            output = self.model(input_tensor)
            
            # Postprocess
            mask, contours = self.postprocessor(output)
            
            return mask, contours

    async def process_image_async(self, image: Union[str, np.ndarray], timeout: Optional[float] = None) -> Dict:
        """
        Asynchronously process a single image (given as a file path or numpy array).
        Returns a dict with keys: "mask", "contours", and "processing_time".
        """
        start_time = time.perf_counter()
        try:
            if self.engine is not None:
                # Use the inference engine if available
                coro = self.engine.run_inference_async(image)
                result = await asyncio.wait_for(coro, timeout) if timeout is not None else await coro
            else:
                # Fallback to direct inference
                coro = self._direct_inference_async(image)
                result = await asyncio.wait_for(coro, timeout) if timeout is not None else await coro
                
            processing_time = time.perf_counter() - start_time
            
            if isinstance(result, tuple) and len(result) == 2:
                mask, contours = result
            else:
                logger.warning(f"Unexpected result format: {type(result)}")
                mask, contours = self.postprocessor(result)  # Try to extract mask/contours
                
            return {
                "mask": mask,
                "contours": contours,
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
        Synchronously process a single image.
        """
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an event loop, create a new task
                future = asyncio.run_coroutine_threadsafe(
                    self.process_image_async(image, timeout), loop)
                return future.result(timeout if timeout is not None else 30)
            else:
                # Event loop exists but not running, use it
                return loop.run_until_complete(
                    self.process_image_async(image, timeout))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.process_image_async(image, timeout))

    def benchmark_segmentation(self, image: Union[str, np.ndarray], iterations: int = 10) -> Tuple[float, float]:
        """
        Benchmark the synchronous segmentation pipeline.
        Returns the mean and standard deviation of processing times.
        """
        times = []
        for i in range(iterations):
            try:
                result = self.process_image(image)
                times.append(result["processing_time"])
                logger.debug(f"Iteration {i+1}/{iterations}: {result['processing_time']:.4f}s")
            except Exception as e:
                logger.error(f"Benchmark iteration {i+1} failed: {e}")
                
        if not times:
            raise RuntimeError("All benchmark iterations failed")
            
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
            
        if self.engine is not None and hasattr(self.engine, "cleanup"):
            try:
                await self.engine.cleanup()
            except Exception as e:
                logger.error(f"Error during engine cleanup: {e}")
                
        logger.info("SegmentationModel shutdown complete")


################################################################################
# Load image helper function
################################################################################
async def load_image_async(image_path: str) -> np.ndarray:
    """
    Asynchronously load an image from a URL or local path
    """
    try:
        if image_path.startswith(('http://', 'https://')):
            loop = asyncio.get_running_loop()
            
            def _fetch_image():
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to decode image from URL: {image_path}")
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            return await loop.run_in_executor(None, _fetch_image)
        else:
            def _read_local_image():
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Failed to read local image: {image_path}")
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _read_local_image)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise


################################################################################
# Example main: Demonstrate usage with a YOLO segmentation model
################################################################################
async def main():
    # Set up error handling for the entire process
    try:
        logger.info("Starting segmentation model pipeline example")
        
        # Try to download the model
        try:
            model_path = download_model(
                custom_url="https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11s.pt"
            )
            logger.info(f"Downloaded model to {model_path}")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            # Use a fallback URL if available, or raise the exception
            raise
            
        # Load the model safely
        try:
            if torch.cuda.is_available():
                map_location = torch.device("cuda")
            else:
                map_location = torch.device("cpu")
                
            model = torch.load(model_path, map_location=map_location)
            
            # Handle different model formats
            if isinstance(model, dict):
                if "model" in model:
                    model = model["model"]
                elif "state_dict" in model:
                    # This is just a state dict, need the model class
                    raise ValueError("Downloaded file contains only state_dict, model class is required")
                    
            # Make sure we have a properly initialized model
            if not hasattr(model, "forward"):
                # If this is an Ultralytics YOLO model
                if hasattr(model, "predict") or hasattr(model, "__call__"):
                    logger.info("Loaded Ultralytics YOLO model")
                else:
                    raise ValueError(f"Loaded object is not a valid PyTorch model: {type(model)}")
            
            logger.info(f"Successfully loaded model of type: {type(model).__name__}")
            
        except Exception as e:
            logger.error(f"Failed to load the model: {e}", exc_info=True)
            raise

        # Initialize the segmentation pipeline
        segmenter = SegmentationModel(model)
        
        try:
            # List of example images (local paths or URLs)
            images = [
                "https://ultralytics.com/images/zidane.jpg",
                "https://ultralytics.com/images/bus.jpg"
            ]
            
            logger.info(f"Processing {len(images)} images...")
            
            # Process images asynchronously
            tasks = [segmenter.process_image_async(img, timeout=10.0) for img in images]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Display results
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Image {idx+1} ({images[idx]}) processing failed: {result}")
                else:
                    num_contours = len(result["contours"])
                    proc_time = result["processing_time"]
                    logger.info(f"Image {idx+1}: Processing time: {proc_time:.4f}s, Contours: {num_contours}")
                    
                    # Log information about the largest contours
                    if num_contours > 0:
                        contours_by_area = sorted(result["contours"], key=cv2.contourArea, reverse=True)
                        for i, contour in enumerate(contours_by_area[:3]):  # Top 3 largest
                            area = cv2.contourArea(contour)
                            logger.info(f"  Contour {i+1}: Area = {area:.1f} pixels")
            
            # Run a benchmark on the first image if it was successful
            if isinstance(results[0], dict):
                logger.info("Running benchmark...")
                mean_time, std_time = segmenter.benchmark_segmentation(images[0], iterations=5)
                logger.info(f"Benchmark complete: {mean_time:.4f}s ± {std_time:.4f}s")
            
        except Exception as e:
            logger.error(f"Error during image processing: {e}", exc_info=True)
        finally:
            # Ensure we clean up resources properly
            await segmenter.shutdown()
            logger.info("Pipeline shutdown complete")
            
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        sys.exit(1)


################################################################################
# Example of saving segmentation results
################################################################################
async def save_segmentation_results(segmenter: SegmentationModel, image_path: str, output_dir: str):
    """
    Process an image and save the segmentation results to disk
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load image
        image = await load_image_async(image_path)
        
        # Get filename without extension
        if image_path.startswith(('http://', 'https://')):
            filename = image_path.split('/')[-1].split('.')[0]
        else:
            filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Process the image
        result = await segmenter.process_image_async(image)
        
        # Get results
        mask = result["mask"]
        contours = result["contours"]
        
        # Create visualization image
        visualization = image.copy()
        
        # Draw contours on the image
        cv2.drawContours(visualization, contours, -1, (0, 255, 0), 2)
        
        # Create colored mask for overlay
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [0, 0, 255]  # Red mask
        
        # Create semi-transparent overlay
        alpha = 0.5
        overlay = cv2.addWeighted(visualization, 1, colored_mask, alpha, 0)
        
        # Save original image
        cv2.imwrite(os.path.join(output_dir, f"{filename}_original.jpg"), 
                   cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Save mask
        cv2.imwrite(os.path.join(output_dir, f"{filename}_mask.png"), mask)
        
        # Save visualization with contours
        cv2.imwrite(os.path.join(output_dir, f"{filename}_contours.jpg"), 
                   cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        
        # Save overlay
        cv2.imwrite(os.path.join(output_dir, f"{filename}_overlay.jpg"), 
                   cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        logger.info(f"Saved segmentation results for {filename} to {output_dir}")
        
        return {
            "image_path": image_path,
            "mask_path": os.path.join(output_dir, f"{filename}_mask.png"),
            "contours_path": os.path.join(output_dir, f"{filename}_contours.jpg"),
            "overlay_path": os.path.join(output_dir, f"{filename}_overlay.jpg"),
            "num_contours": len(contours)
        }
        
    except Exception as e:
        logger.error(f"Error saving segmentation results for {image_path}: {e}")
        raise


################################################################################
# Batch processing example
################################################################################
async def batch_process_images(model_path: str, image_paths: List[str], output_dir: str):
    """
    Batch process multiple images with the segmentation model
    """
    try:
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path, map_location=device)
        
        # Initialize segmenter
        segmenter = SegmentationModel(model)
        
        # Process images
        results = []
        for img_path in image_paths:
            try:
                result = await save_segmentation_results(segmenter, img_path, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                results.append({"image_path": img_path, "error": str(e)})
        
        # Generate summary report
        report_path = os.path.join(output_dir, "segmentation_report.txt")
        with open(report_path, "w") as f:
            f.write("Segmentation Batch Processing Report\n")
            f.write("===================================\n\n")
            f.write(f"Total images processed: {len(image_paths)}\n")
            f.write(f"Successful: {sum(1 for r in results if 'error' not in r)}\n")
            f.write(f"Failed: {sum(1 for r in results if 'error' in r)}\n\n")
            
            f.write("Details:\n")
            for i, result in enumerate(results):
                f.write(f"Image {i+1}: {result['image_path']}\n")
                if "error" in result:
                    f.write(f"  Status: FAILED\n")
                    f.write(f"  Error: {result['error']}\n")
                else:
                    f.write(f"  Status: SUCCESS\n")
                    f.write(f"  Contours found: {result['num_contours']}\n")
                    f.write(f"  Mask saved to: {result['mask_path']}\n")
                f.write("\n")
        
        logger.info(f"Batch processing complete. Report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise
    finally:
        if 'segmenter' in locals():
            await segmenter.shutdown()


################################################################################
# Command-line interface example
################################################################################
def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description="Run segmentation on images")
    
    parser.add_argument("--model", type=str, default="",
                        help="Path to model file or URL to download")
    
    parser.add_argument("--images", type=str, nargs="+", 
                        help="Paths or URLs to images to process")
    
    parser.add_argument("--output_dir", type=str, default="segmentation_results",
                        help="Directory to save results")
    
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Segmentation threshold")
    
    parser.add_argument("--min_contour_area", type=int, default=100,
                        help="Minimum contour area to keep")
    
    parser.add_argument("--device", type=str, default="",
                        help="Device to use (cuda or cpu)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Override config with command line args
    if args.threshold:
        SEGMENTATION_CONFIG["threshold"] = args.threshold
    
    if args.min_contour_area:
        SEGMENTATION_CONFIG["min_contour_area"] = args.min_contour_area
    
    if args.device:
        # Override device in config
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set CUDA device
    if device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else "0"
    
    # If specific model provided, use it
    if args.model:
        model_path = args.model
    else:
        # Default to YOLO11
        model_path = download_model(
            custom_url="https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11s.pt"
        )
    
    # If images provided, process them
    if args.images:
        asyncio.run(batch_process_images(model_path, args.images, args.output_dir))
    else:
        # Run the default demo
        asyncio.run(main())

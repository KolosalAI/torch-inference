import cv2
import numpy as np
import torch
import asyncio
from typing import Dict, Tuple, Union, Any, Optional, List, Protocol, Callable, TypeVar, cast
import time
import logging
import os
import sys
from pathlib import Path
import requests
from io import BytesIO
from dataclasses import dataclass, field, asdict
import functools
import concurrent.futures
from PIL import Image
from contextlib import asynccontextmanager, contextmanager
from tqdm import tqdm

# Configure proper path for relative imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("segmentation.log", mode="a")
    ]
)

# Type variable for generic functions
T = TypeVar('T')

# Local imports with proper error handling
try:
    from utils.config import SEGMENTATION_CONFIG, EngineConfig
    from utils.logger import get_logger
    from modules.core.inference_engine import InferenceEngine
    from modules.core.preprocessor import ImagePreprocessor
    from modules.core.postprocessor import BasePostprocessor
    from models.downloader import download_model
except ImportError as e:
    # Fallback definitions with improved structure
    @dataclass
    class EngineConfig:
        num_workers: int = 4
        queue_size: int = 32
        batch_size: int = 4
        min_batch_size: int = 1
        max_batch_size: int = 8
        warmup_runs: int = 2
        timeout: float = 5.0
        batch_wait_timeout: float = 0.01
        autoscale_interval: float = 1.0
        debug_mode: bool = True
        pid_kp: float = 0.6
        pid_ki: float = 0.15
        pid_kd: float = 0.1
        guard_enabled: bool = True
        guard_confidence_threshold: float = 0.7
        guard_variance_threshold: float = 0.03
        num_classes: int = 1
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        async_mode: bool = True
    
    # Fallback config with dataclass for better structure and type safety
    @dataclass
    class SegmentationConfigClass:
        input_size: Tuple[int, int] = (640, 640)
        mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
        std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
        threshold: float = 0.5
        min_contour_area: int = 100
        apply_sigmoid: bool = False
        use_fp16: bool = False
        use_tensorrt: bool = False
        async_preproc: bool = True
        use_multigpu: bool = False
        compiled_module_cache: Optional[str] = None
        additional_transforms: List[Any] = field(default_factory=list)
        
    SEGMENTATION_CONFIG = SegmentationConfigClass()
    
    # Fallback logger with enhanced capabilities
    def get_logger(name, level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Add file handler for persistent logs
            file_handler = logging.FileHandler(f"{name.replace('.', '_')}.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger
    
    # Protocol definitions for missing classes
    class ImagePreprocessor(Protocol):
        def __call__(self, inputs: Any) -> torch.Tensor: ...
        async def preprocess_async(self, input_data: Any) -> torch.Tensor: ...
        
    class BasePostprocessor(Protocol):
        def __call__(self, outputs: Any) -> Any: ...
        
    class InferenceEngine(Protocol):
        async def run_inference_async(self, inputs: Any) -> Any: ...
        async def cleanup(self) -> None: ...
        
    # Simple model downloader fallback
    def download_model(custom_url=None, model_name=None, cache_dir=None):
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "ml_models"
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not custom_url:
            raise ValueError("URL must be provided when using fallback downloader")
            
        model_name = model_name or custom_url.split("/")[-1]
        model_path = cache_dir / model_name
        
        if model_path.exists():
            print(f"Using cached model: {model_path}")
            return str(model_path)
            
        print(f"Downloading model from {custom_url}")
        response = requests.get(custom_url, stream=True)
        response.raise_for_status()
        
        # Get total size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(model_path, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=model_name
        ) as t:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
                
        return str(model_path)

# Configure logger for this module
logger = get_logger(__name__)
logger.setLevel(logging.INFO)

# Memory cache for preprocessed images
image_cache = {}
MAX_CACHE_SIZE = 50  # Limit cache to prevent memory issues


def cache_result(max_size=100):
    """Decorator for caching function results with LRU behavior"""
    cache = {}
    order = []
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from arguments
            key = str(args) + str(kwargs)
            
            if key in cache:
                # Move to most recently used
                order.remove(key)
                order.append(key)
                return cache[key]
            
            result = func(*args, **kwargs)
            
            # Add to cache
            cache[key] = result
            order.append(key)
            
            # Trim cache if needed
            if len(cache) > max_size:
                oldest_key = order.pop(0)
                del cache[oldest_key]
                
            return result
        return wrapper
    return decorator


################################################################################
# PID Controller for dynamic batch size adjustment
################################################################################
class PIDController:
    """PID controller for dynamic batch size adjustment based on latency"""
    def __init__(self, kp=0.6, ki=0.1, kd=0.05, setpoint=30.0, min_value=1, max_value=16):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.min_value = min_value
        self.max_value = max_value
        self.prev_error = 0
        self.integral = 0
        self.last_value = min_value
        
    def update(self, current_value, dt=1.0):
        """Update controller with current measurement"""
        error = self.setpoint - current_value
        self.integral += error * dt
        # Anti-windup: limit integral term
        self.integral = max(-50, min(50, self.integral))
        
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        
        # Apply output to current batch size
        new_value = self.last_value + output
        # Clamp to limits
        new_value = max(self.min_value, min(self.max_value, round(new_value)))
        self.last_value = new_value
        
        return int(new_value)
        
    def reset(self):
        """Reset controller state"""
        self.prev_error = 0
        self.integral = 0
        self.last_value = self.min_value


################################################################################
# Enhanced Segmentation Preprocessor
################################################################################
class SegmentationPreprocessor:
    """
    Enhanced preprocessor for segmentation tasks with caching, hardware acceleration,
    and optimized image processing.
    """
    def __init__(self):
        # Extract configuration with improved defaults
        config = SEGMENTATION_CONFIG
        self.image_size = getattr(config, "input_size", (640, 640))
        self.mean = getattr(config, "mean", [0.485, 0.456, 0.406])
        self.std = getattr(config, "std", [0.229, 0.224, 0.225])
        self.async_preproc = getattr(config, "async_preproc", True)
        self.use_fp16 = getattr(config, "use_fp16", False)
        
        # Ensure lists for mean and std
        if not isinstance(self.mean, list):
            self.mean = [self.mean] * 3
        if not isinstance(self.std, list):
            self.std = [self.std] * 3
            
        # Initialize executor for parallel preprocessing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.pending_tasks = []
        
        # Try to use torch transforms for better performance
        try:
            import torchvision.transforms as T
            self.transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
                T.Resize(self.image_size, antialias=True)
            ])
            self.use_torch_transforms = True
            logger.info("Using torchvision transforms for preprocessing")
        except (ImportError, AttributeError):
            self.use_torch_transforms = False
            logger.info("Falling back to OpenCV for preprocessing")
            
        logger.debug(f"Initialized SegmentationPreprocessor with image_size: {self.image_size}")

    def _crop_and_resize(self, img: np.ndarray) -> np.ndarray:
        """Center-crop and resize the image while preserving aspect ratio"""
        if img is None or img.size == 0:
            raise ValueError("Empty or invalid image provided")
            
        # Unpack image size
        if isinstance(self.image_size, int):
            desired_h, desired_w = self.image_size, self.image_size
        else:
            desired_h, desired_w = self.image_size

        # Validate image dimensions
        if len(img.shape) < 2:
            raise ValueError(f"Invalid image shape: {img.shape}")
            
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions: {h}x{w}")
        
        # Calculate target aspect ratio
        target_aspect = desired_w / desired_h
        current_aspect = w / h

        # Center crop to match target aspect ratio
        if abs(current_aspect - target_aspect) > 0.01:  # Only crop if aspect ratios differ significantly
            if current_aspect > target_aspect:
                new_w = int(h * target_aspect)
                start_x = (w - new_w) // 2
                img = img[:, max(0, start_x):min(w, start_x + new_w)]
            else:
                new_h = int(w / target_aspect)
                start_y = (h - new_h) // 2
                img = img[max(0, start_y):min(h, start_y + new_h), :]

        # Resize with area interpolation for downsampling, bilinear for upsampling
        if h > desired_h or w > desired_w:
            interp = cv2.INTER_AREA  # Better quality when downsizing
        else:
            interp = cv2.INTER_LINEAR
            
        return cv2.resize(img, (desired_w, desired_h), interpolation=interp)

    async def preprocess_async(self, input_data: Any) -> torch.Tensor:
        """Asynchronous preprocessing with better concurrency handling"""
        loop = asyncio.get_running_loop()
        try:
            # If input is a list, process in parallel
            if isinstance(input_data, (list, tuple)) and len(input_data) > 1:
                futures = [loop.run_in_executor(self.executor, self._process_single, item) 
                           for item in input_data]
                tensors = await asyncio.gather(*futures)
                return torch.stack(tensors, dim=0)
            else:
                # Process single input
                return await loop.run_in_executor(self.executor, self.__call__, input_data)
        except asyncio.CancelledError:
            logger.debug("Preprocessing task was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in async preprocessing: {e}")
            raise
        
    def cancel_pending_tasks(self):
        """Cancel any pending preprocessing tasks"""
        for task in self.pending_tasks:
            if not task.done():
                task.cancel()
        self.pending_tasks = []
        
    def _process_single(self, input_data: Any) -> torch.Tensor:
        """Process a single image input with caching"""
        # Generate cache key for input
        if isinstance(input_data, str):
            cache_key = input_data
        elif isinstance(input_data, np.ndarray):
            # Hash the array content for caching
            cache_key = hash(input_data.tobytes())
        elif isinstance(input_data, torch.Tensor):
            cache_key = hash(input_data.detach().cpu().numpy().tobytes())
        else:
            # Non-cacheable input
            return self._preprocess_image(input_data)
            
        # Check cache
        if cache_key in image_cache:
            return image_cache[cache_key]
            
        # Process and cache result
        result = self._preprocess_image(input_data)
        
        # Add to cache with LRU mechanism
        if len(image_cache) >= MAX_CACHE_SIZE:
            # Remove oldest item (simple approach)
            image_cache.pop(next(iter(image_cache)))
        image_cache[cache_key] = result
        
        return result

    def _preprocess_image(self, img: Any) -> torch.Tensor:
        """Core image preprocessing with optimized paths for different input types"""
        # Handle different input types
        if isinstance(img, str):
            if img.startswith(('http://', 'https://')):
                try:
                    response = requests.get(img, timeout=10)
                    response.raise_for_status()
                    
                    # Try using PIL for better image format support
                    if self.use_torch_transforms:
                        pil_img = Image.open(BytesIO(response.content)).convert('RGB')
                        return self.transforms(pil_img).unsqueeze(0)
                    else:
                        img_array = np.frombuffer(response.content, dtype=np.uint8)
                        np_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if np_img is None:
                            raise ValueError(f"Failed to decode image from URL: {img}")
                        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    logger.error(f"Error loading image from URL '{img}': {e}")
                    raise
            else:
                # Local file path
                if self.use_torch_transforms:
                    try:
                        pil_img = Image.open(img).convert('RGB')
                        return self.transforms(pil_img).unsqueeze(0)
                    except Exception:
                        # Fall back to OpenCV if PIL fails
                        np_img = cv2.imread(img)
                        if np_img is None:
                            raise ValueError(f"Failed to load image: {img}")
                        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                else:
                    np_img = cv2.imread(img)
                    if np_img is None:
                        raise ValueError(f"Failed to load image: {img}")
                    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        elif isinstance(img, np.ndarray):
            np_img = img.copy()
            # Convert BGR to RGB if needed
            if np_img.ndim == 3 and np_img.shape[2] == 3:
                # Check if image might be in BGR format (OpenCV default)
                if np_img.dtype == np.uint8:
                    # Heuristic: in natural images, the blue channel typically has 
                    # less intensity than red. If blue > red, it might be BGR.
                    b, g, r = np.mean(np_img[:,:,0]), np.mean(np_img[:,:,1]), np.mean(np_img[:,:,2])
                    if b > r:  # Likely BGR
                        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        elif isinstance(img, torch.Tensor):
            # Handle different tensor formats
            if img.ndim == 4:  # Batch of images, take the first one
                img = img[0]
                
            if img.ndim == 3:
                if img.shape[0] in [1, 3]:  # CHW format
                    # If already normalized and in right format, return directly
                    if img.shape[1:] == self.image_size:
                        channel_means = [img[c].mean().item() for c in range(min(3, img.shape[0]))]
                        # Check if likely already normalized
                        if all(-1 < cm < 1 for cm in channel_means):
                            return img.unsqueeze(0)
                    
                    np_img = img.detach().cpu().numpy().transpose(1, 2, 0)
                elif img.shape[2] in [1, 3]:  # HWC format
                    np_img = img.detach().cpu().numpy()
                else:
                    raise ValueError(f"Unsupported tensor shape: {img.shape}")
            else:
                raise ValueError(f"Unsupported tensor dimensions: {img.ndim}")
        elif hasattr(img, 'convert'):  # PIL Image
            if self.use_torch_transforms:
                return self.transforms(img).unsqueeze(0)
            np_img = np.array(img.convert('RGB'))
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        # Ensure we have an RGB image with correct dimensions
        if np_img.ndim == 2:  # Grayscale
            np_img = np.stack([np_img] * 3, axis=2)
        elif np_img.ndim == 3 and np_img.shape[2] == 1:  # Single channel
            np_img = np.concatenate([np_img] * 3, axis=2)
        elif np_img.ndim == 3 and np_img.shape[2] == 4:  # RGBA
            np_img = np_img[:, :, :3]
        elif np_img.ndim != 3 or np_img.shape[2] != 3:
            raise ValueError(f"Unsupported image format with shape {np_img.shape}")

        # Apply preprocessing - crop and resize
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
        
        # Convert to half precision if requested
        if self.use_fp16:
            img_tensor = img_tensor.half()
            
        return img_tensor.unsqueeze(0)  # Add batch dimension

    def __call__(self, inputs: Any) -> torch.Tensor:
        """Process inputs and return normalized tensor ready for model input"""
        if inputs is None:
            raise ValueError("Input cannot be None")
            
        # Handle different input types
        if isinstance(inputs, torch.Tensor):
            if inputs.ndim == 4:  # Already batched
                return inputs  # Assume already preprocessed
            elif inputs.ndim == 3:  # Single image tensor
                return self._process_single(inputs)
            else:
                raise ValueError(f"Tensor input must be 3D or 4D, got shape {inputs.shape}")
        elif isinstance(inputs, (list, tuple)):
            # Process each input and batch together
            tensors = [self._process_single(img) for img in inputs]
            return torch.cat(tensors, dim=0)
        else:
            # Single input - process and add batch dimension
            return self._process_single(inputs)


################################################################################
# Enhanced Segmentation Postprocessor
################################################################################
class SegmentationPostprocessor:
    """
    Enhanced postprocessor for segmentation outputs with optimized contour extraction,
    support for various model architectures, and visualization features.
    """
    def __init__(self, threshold: float = 0.5, min_contour_area: int = 100, 
                 apply_sigmoid: bool = False, max_contours: int = 100):
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.apply_sigmoid = apply_sigmoid
        self.max_contours = max_contours
        self.contour_mode = cv2.RETR_EXTERNAL
        self.contour_method = cv2.CHAIN_APPROX_SIMPLE
        
        logger.debug(f"Initialized SegmentationPostprocessor: threshold={threshold}, "
                     f"min_contour_area={min_contour_area}, apply_sigmoid={apply_sigmoid}")

    def process_yolo_output(self, outputs) -> Tuple[np.ndarray, List]:
        """Process YOLO-specific outputs with improved error handling"""
        try:
            # Handle ultralytics YOLO format results
            if hasattr(outputs, "masks") and outputs.masks is not None:
                # Direct YOLO result object
                masks = outputs.masks.data
                return self._process_masks(masks)
            
            # Handle list of results - get first item
            if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                first_output = outputs[0]
                
                # Check for masks attribute
                if hasattr(first_output, "masks") and first_output.masks is not None:
                    masks = first_output.masks.data
                    return self._process_masks(masks)
                    
                # Check for pred_masks attribute (SAM model format)
                if hasattr(first_output, "pred_masks") and first_output.pred_masks is not None:
                    masks = first_output.pred_masks
                    return self._process_masks(masks)
            
            # Handle dictionary format
            if isinstance(outputs, dict) and "masks" in outputs:
                masks = outputs["masks"]
                return self._process_masks(masks)
                
            # If no known format is detected, return empty result
            logger.warning(f"Unrecognized YOLO output format: {type(outputs)}")
            height, width = getattr(SEGMENTATION_CONFIG, "input_size", (640, 640))
            return np.zeros((height, width), dtype=np.uint8), []
            
        except Exception as e:
            logger.error(f"Error processing YOLO output: {e}")
            height, width = getattr(SEGMENTATION_CONFIG, "input_size", (640, 640))
            return np.zeros((height, width), dtype=np.uint8), []
            
    def _process_masks(self, masks) -> Tuple[np.ndarray, List]:
        """Process mask data into binary mask and contours"""
        if isinstance(masks, torch.Tensor):
            if len(masks) == 0:
                height, width = getattr(SEGMENTATION_CONFIG, "input_size", (640, 640))
                return np.zeros((height, width), dtype=np.uint8), []
                
            # Combine all instance masks into one binary mask
            combined_mask = torch.zeros_like(masks[0], dtype=torch.uint8)
            for mask in masks:
                mask_bool = mask > self.threshold
                combined_mask = torch.logical_or(combined_mask, mask_bool).to(torch.uint8) * 255
            
            # Convert to numpy and find contours
            combined_mask_np = combined_mask.detach().cpu().numpy()
        else:
            # Handle numpy array masks
            combined_mask_np = np.asarray(masks)
            if combined_mask_np.max() <= 1.0 and combined_mask_np.dtype != np.uint8:
                combined_mask_np = (combined_mask_np > self.threshold).astype(np.uint8) * 255
        
        # Ensure mask has the right shape and type
        if combined_mask_np.ndim > 2:
            if combined_mask_np.shape[0] == 1:
                combined_mask_np = combined_mask_np[0]
            else:
                combined_mask_np = np.max(combined_mask_np, axis=0)
        
        if combined_mask_np.size == 0:
            height, width = getattr(SEGMENTATION_CONFIG, "input_size", (640, 640))
            return np.zeros((height, width), dtype=np.uint8), []
            
        # Ensure we have the right dtype
        if combined_mask_np.dtype != np.uint8:
            combined_mask_np = combined_mask_np.astype(np.uint8)
            
        # Find contours
        contours, _ = cv2.findContours(combined_mask_np, self.contour_mode, self.contour_method)
        
        # Filter contours by area and limit the number
        if contours:
            filtered_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
            
            # Sort by area (largest first) and limit the number
            if len(filtered_contours) > self.max_contours:
                filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:self.max_contours]
        else:
            filtered_contours = []
            
        return combined_mask_np, filtered_contours

    def __call__(self, outputs: Any) -> Tuple[np.ndarray, list]:
        """Process model outputs to produce masks and contours with support for various formats"""
        try:
            # Case 1: Standard segmentation tensor output
            if isinstance(outputs, torch.Tensor):
                # Handle tensor output with better dimension checking
                if outputs.dim() == 4:  # [B, C, H, W]
                    if outputs.size(0) == 1:  # Single batch
                        mask_logits = outputs[0]
                        if outputs.size(1) > 1:  # Multi-class
                            mask_logits = torch.argmax(mask_logits, dim=0).float()
                    else:
                        logger.warning(f"Expected batch size 1, got {outputs.size(0)}. Using first result.")
                        mask_logits = outputs[0]
                elif outputs.dim() == 3:  # [C, H, W]
                    mask_logits = outputs
                    if outputs.size(0) > 1:  # Multi-class
                        mask_logits = torch.argmax(mask_logits, dim=0).float()
                elif outputs.dim() == 2:  # [H, W]
                    mask_logits = outputs
                else:
                    raise ValueError(f"Unsupported tensor shape: {outputs.shape}")
                
                # Convert to numpy and apply sigmoid if needed
                mask_np = mask_logits.detach().cpu().numpy()
                if self.apply_sigmoid:
                    mask_np = 1 / (1 + np.exp(-mask_np))
                
                # Create binary mask
                if mask_np.ndim > 2:  # Multi-dimensional
                    if mask_np.shape[0] == 1:
                        mask_np = mask_np[0]  # [1, H, W] -> [H, W]
                
                # Threshold and convert to uint8
                mask = (mask_np > self.threshold).astype(np.uint8) * 255
                
                # Find and filter contours
                contours, _ = cv2.findContours(mask, self.contour_mode, self.contour_method)
                filtered_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
                
                if filtered_contours and len(filtered_contours) > self.max_contours:
                    filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:self.max_contours]
                
                return mask, filtered_contours
            
            # Case 2: YOLO or other structured object detection outputs
            elif isinstance(outputs, (list, tuple)) or hasattr(outputs, "masks"):
                return self.process_yolo_output(outputs)
            
            # Case 3: Dictionary output format
            elif isinstance(outputs, dict):
                if "masks" in outputs:
                    return self._process_masks(outputs["masks"])
                elif "segmentation" in outputs:
                    return self._process_masks(outputs["segmentation"])
                elif "predictions" in outputs:
                    return self._process_masks(outputs["predictions"])
                
                # Handle other dictionary formats
                logger.warning("Unrecognized dictionary output format")
                height, width = getattr(SEGMENTATION_CONFIG, "input_size", (640, 640))
                return np.zeros((height, width), dtype=np.uint8), []
            else:
                logger.warning(f"Unrecognized output type: {type(outputs)}")
                height, width = getattr(SEGMENTATION_CONFIG, "input_size", (640, 640))
                return np.zeros((height, width), dtype=np.uint8), []
                
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}", exc_info=True)
            height, width = getattr(SEGMENTATION_CONFIG, "input_size", (640, 640))
            return np.zeros((height, width), dtype=np.uint8), []
    
    def create_visualization(self, image: np.ndarray, mask: np.ndarray, 
                            contours: List = None, alpha: float = 0.5, 
                            mask_color: Tuple[int, int, int] = (0, 0, 255),
                            contour_color: Tuple[int, int, int] = (0, 255, 0),
                            contour_thickness: int = 2) -> np.ndarray:
        """Create a visualization with mask overlay and contours"""
        # Ensure image is in RGB format and uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
        # Create a copy for visualization
        viz = image.copy()
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        if mask.ndim == 2:
            mask_binary = mask > 0
            colored_mask[mask_binary] = mask_color
        else:
            # Handle multi-channel masks
            mask_binary = np.max(mask, axis=2) > 0
            colored_mask[mask_binary] = mask_color
        
        # Create overlay with mask
        overlay = cv2.addWeighted(viz, 1, colored_mask, alpha, 0)
        
        # Draw contours if provided
        if contours:
            cv2.drawContours(overlay, contours, -1, contour_color, contour_thickness)
        
        return overlay


################################################################################
# Optimized Segmentation Model Pipeline
################################################################################
class SegmentationModel:
    """
    Enhanced segmentation pipeline integrating preprocessing, inference, and postprocessing
    with optimized batch handling, caching, and progress tracking.
    """
    def __init__(self, model: torch.nn.Module, config: Optional[EngineConfig] = None):
        self.config = config or self._create_default_config()
        self.model = model
        
        # Check for multi-GPU setup and optimize accordingly
        if hasattr(SEGMENTATION_CONFIG, "use_multigpu") and getattr(SEGMENTATION_CONFIG, "use_multigpu") and torch.cuda.device_count() > 1:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            logger.info(f"Using multi-GPU setup with {len(devices)} devices")
            
            # When using multiple GPUs, wrap the model in DataParallel if not already
            if not isinstance(model, torch.nn.DataParallel) and not isinstance(model, torch.nn.parallel.DistributedDataParallel):
                self.model = torch.nn.DataParallel(model)
        else:
            devices = [self.config.device]
            
        logger.debug(f"SegmentationModel using devices: {devices}")

        # Initialize components with enhanced error trapping
        self.preprocessor = SegmentationPreprocessor()
        self.postprocessor = SegmentationPostprocessor(
            threshold=getattr(SEGMENTATION_CONFIG, "threshold", 0.5),
            min_contour_area=getattr(SEGMENTATION_CONFIG, "min_contour_area", 100),
            apply_sigmoid=getattr(SEGMENTATION_CONFIG, "apply_sigmoid", False),
            max_contours=getattr(SEGMENTATION_CONFIG, "max_contours", 100)
        )
        
        # Initialize batch size controller if supported
        self.batch_controller = PIDController(
            kp=getattr(self.config, "pid_kp", 0.6),
            ki=getattr(self.config, "pid_ki", 0.15),
            kd=getattr(self.config, "pid_kd", 0.1),
            setpoint=30.0,  # Target latency in ms
            min_value=getattr(self.config, "min_batch_size", 1),
            max_value=getattr(self.config, "max_batch_size", 8)
        )

        # Try to initialize the inference engine with proper error handling
        try:
            self.engine = InferenceEngine(
                model=self.model,
                device=devices,
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                use_fp16=getattr(SEGMENTATION_CONFIG, "use_fp16", False),
                use_tensorrt=getattr(SEGMENTATION_CONFIG, "use_tensorrt", False),
                config=self.config
            )
            self.use_engine = True
            logger.info("Using InferenceEngine for optimized inference")
        except (ImportError, NameError, AttributeError) as e:
            logger.warning(f"Failed to initialize InferenceEngine: {e}")
            logger.info("Falling back to direct model inference")
            self.engine = None
            self.use_engine = False
            
            # Move model to appropriate device
            self.device = torch.device(self.config.device if self.config.device else 
                                     ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize thread pool for parallel processing
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=getattr(self.config, "num_workers", 4)
            )
        
        # Counters for stats tracking
        self.processed_images = 0
        self.total_processing_time = 0
        self.start_time = time.time()

    def _create_default_config(self) -> EngineConfig:
        """Create optimized default configuration for the inference engine"""
        # Determine optimal defaults based on hardware
        num_cpus = os.cpu_count() or 4
        has_cuda = torch.cuda.is_available()
        batch_size = 4 if has_cuda else 1
        
        return EngineConfig(
            num_workers=min(num_cpus - 1, 8),  # Leave one CPU core free
            queue_size=32,
            batch_size=batch_size,
            min_batch_size=1,
            max_batch_size=16 if has_cuda else 2,
            warmup_runs=2,
            timeout=5.0,
            batch_wait_timeout=0.01,
            autoscale_interval=1.0,
            debug_mode=False,  # Set to False for production
            pid_kp=0.6,
            pid_ki=0.15,
            pid_kd=0.1,
            guard_enabled=True,
            guard_confidence_threshold=0.7,
            guard_variance_threshold=0.03,
            num_classes=1,
            device="cuda" if has_cuda else "cpu",
            async_mode=True
        )

    @contextmanager
    def _torch_inference_context(self):
        """Context manager for configuring torch inference settings"""
        # Save previous settings
        prev_grad = torch.is_grad_enabled()
        try:
            # Configure for inference
            torch.set_grad_enabled(False)
            if hasattr(torch.backends, 'cudnn'):
                prev_cudnn_enabled = torch.backends.cudnn.enabled
                prev_cudnn_benchmark = torch.backends.cudnn.benchmark
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
            
            # Use torch.compile if available (PyTorch 2.0+)
            if hasattr(torch, 'compile') and getattr(SEGMENTATION_CONFIG, "use_torch_compile", False):
                if not hasattr(self, '_compiled_model'):
                    try:
                        logger.info("Compiling model with torch.compile...")
                        self._compiled_model = torch.compile(
                            self.model, 
                            mode="reduce-overhead", 
                            fullgraph=False
                        )
                    except Exception as e:
                        logger.warning(f"Failed to compile model: {e}")
                        self._compiled_model = self.model
                model_to_use = self._compiled_model
            else:
                model_to_use = self.model
                
            yield model_to_use
            
        finally:
            # Restore settings
            torch.set_grad_enabled(prev_grad)
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.enabled = prev_cudnn_enabled
                torch.backends.cudnn.benchmark = prev_cudnn_benchmark

    async def _direct_inference_async(self, image: Union[str, np.ndarray, List]) -> Tuple[np.ndarray, List]:
        """Optimized fallback method for direct async inference when engine isn't available"""
        loop = asyncio.get_running_loop()
        
        async def process_batch(batch):
            def _run_inference():
                with self._torch_inference_context() as model:
                    with torch.no_grad():
                        input_tensor = self.preprocessor(batch)
                        input_tensor = input_tensor.to(self.device)
                        
                        try:
                            if hasattr(model, 'predict'):
                                # YOLO-style model
                                output = model.predict(input_tensor)
                            else:
                                # Standard PyTorch model
                                output = model(input_tensor)
                            
                            results = []
                            for i in range(len(batch)):
                                if isinstance(output, (list, tuple)) and len(output) > 0:
                                    # Handle YOLO list results
                                    batch_output = output[i] if i < len(output) else output[0]
                                elif isinstance(output, torch.Tensor) and output.dim() == 4:
                                    # Handle tensor [B,C,H,W]
                                    batch_output = output[i].unsqueeze(0)
                                else:
                                    batch_output = output
                                
                                mask, contours = self.postprocessor(batch_output)
                                results.append((mask, contours))
                            
                            return results
                        except Exception as e:
                            logger.error(f"Inference error: {e}", exc_info=True)
                            raise
            
            try:
                return await loop.run_in_executor(self.executor, _run_inference)
            except Exception as e:
                logger.error(f"Async inference failed: {e}")
                raise
        
        # Handle different input types
        if isinstance(image, (list, tuple)):
            # Process in batches for better efficiency
            batch_size = self.batch_controller.last_value
            results = []
            
            for i in range(0, len(image), batch_size):
                batch = image[i:i+batch_size]
                start_time = time.perf_counter()
                batch_results = await process_batch(batch)
                elapsed = time.perf_counter() - start_time
                
                # Update batch size based on latency
                new_batch_size = self.batch_controller.update(elapsed * 1000)
                if new_batch_size != batch_size:
                    logger.debug(f"Adjusted batch size: {batch_size} -> {new_batch_size}")
                    batch_size = new_batch_size
                
                results.extend(batch_results)
            
            return results
        else:
            # Single image
            results = await process_batch([image])
            return results[0]

    async def process_image_async(self, image: Union[str, np.ndarray, List], 
                                 timeout: Optional[float] = None) -> Dict:
        """
        Asynchronously process image(s) with optimized handling of lists and batches.
        Returns a dict with keys: "mask", "contours", and "processing_time".
        """
        start_time = time.perf_counter()
        try:
            # Handle batch processing when a list is provided
            is_batch = isinstance(image, (list, tuple)) and len(image) > 1
            
            if self.use_engine:
                # Use the inference engine if available
                if is_batch:
                    # Process batch through the engine
                    coro = asyncio.gather(*[self.engine.run_inference_async(img) for img in image])
                else:
                    coro = self.engine.run_inference_async(image)
                
                result = await asyncio.wait_for(coro, timeout) if timeout is not None else await coro
            else:
                # Fallback to direct inference
                coro = self._direct_inference_async(image)
                result = await asyncio.wait_for(coro, timeout) if timeout is not None else await coro
                
            processing_time = time.perf_counter() - start_time
            
            # Update stats
            self.processed_images += 1 if not is_batch else len(image)
            self.total_processing_time += processing_time
            
            if is_batch:
                # Return list of results for batch processing
                batch_results = []
                for item in result:
                    if isinstance(item, tuple) and len(item) == 2:
                        mask, contours = item
                    else:
                        mask, contours = self.postprocessor(item)
                    
                    batch_results.append({
                        "mask": mask,
                        "contours": contours,
                        "processing_time": processing_time / len(image)  # Approximate per-image time
                    })
                return batch_results
            else:
                # Handle single image result
                if isinstance(result, tuple) and len(result) == 2:
                    mask, contours = result
                else:
                    mask, contours = self.postprocessor(result)
                    
                return {
                    "mask": mask,
                    "contours": contours,
                    "processing_time": processing_time
                }
        except asyncio.TimeoutError:
            logger.error(f"Inference timed out after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Segmentation failed: {e}", exc_info=True)
            raise

    def process_image(self, image: Union[str, np.ndarray, List], timeout: Optional[float] = None) -> Dict:
        """
        Synchronously process single image or batch of images
        """
        try:
            # Try to get current event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Check if we're already in a running event loop
            if loop.is_running():
                # We're in an event loop, use run_coroutine_threadsafe
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

    def batch_process(self, images: List[Union[str, np.ndarray]], 
                     timeout: Optional[float] = None,
                     show_progress: bool = True) -> List[Dict]:
        """
        Process a batch of images with optional progress bar
        """
        if not images:
            return []
        
        total = len(images)
        results = []
        
        with tqdm(total=total, desc="Processing images", disable=not show_progress) as pbar:
            # Process in optimal batch sizes
            batch_size = self.config.batch_size
            for i in range(0, total, batch_size):
                batch = images[i:i+batch_size]
                try:
                    batch_results = self.process_image(batch, timeout)
                    results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size}: {e}")
                    # Add error placeholders for failed images
                    for _ in range(len(batch)):
                        results.append({"error": str(e)})
                pbar.update(len(batch))
        
        return results

    def get_performance_stats(self) -> Dict:
        """Get performance statistics for the model"""
        runtime = time.time() - self.start_time
        
        # Avoid division by zero
        if self.processed_images == 0:
            return {
                "total_images": 0,
                "average_time_per_image": 0,
                "images_per_second": 0,
                "total_runtime_seconds": runtime
            }
        
        return {
            "total_images": self.processed_images,
            "average_time_per_image": self.total_processing_time / self.processed_images,
            "images_per_second": self.processed_images / runtime if runtime > 0 else 0,
            "total_runtime_seconds": runtime
        }

    def benchmark_segmentation(self, image: Union[str, np.ndarray], iterations: int = 10,
                              warmup: int = 2, show_progress: bool = True) -> Dict:
        """
        Enhanced benchmark for the segmentation pipeline with warmup iterations.
        Returns detailed performance metrics.
        """
        # Preload image to eliminate I/O from benchmark
        if isinstance(image, str):
            if image.startswith(('http://', 'https://')):
                response = requests.get(image, timeout=10)
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                preloaded_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                preloaded_image = cv2.cvtColor(preloaded_image, cv2.COLOR_BGR2RGB)
            else:
                preloaded_image = cv2.imread(image)
                preloaded_image = cv2.cvtColor(preloaded_image, cv2.COLOR_BGR2RGB)
        else:
            preloaded_image = image.copy()
        
        # Perform warmup iterations (not included in results)
        logger.info(f"Running {warmup} warmup iterations...")
        for _ in range(warmup):
            try:
                self.process_image(preloaded_image)
            except Exception as e:
                logger.error(f"Warmup failed: {e}")
        
        # Run benchmark iterations
        logger.info(f"Running {iterations} benchmark iterations...")
        times = []
        cpu_percents = []
        memory_usages = []
        
        # Try to get system metrics if psutil is available
        try:
            import psutil
            process = psutil.Process()
            can_measure_resources = True
        except ImportError:
            can_measure_resources = False
        
        with tqdm(total=iterations, desc="Benchmarking", disable=not show_progress) as pbar:
            for i in range(iterations):
                try:
                    if can_measure_resources:
                        cpu_before = process.cpu_percent()
                        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
                    
                    start = time.perf_counter()
                    result = self.process_image(preloaded_image)
                    elapsed = time.perf_counter() - start
                    
                    if can_measure_resources:
                        cpu_percents.append(process.cpu_percent() - cpu_before)
                        memory_usages.append(process.memory_info().rss / (1024 * 1024) - mem_before)
                    
                    times.append(elapsed)
                    pbar.update(1)
                    pbar.set_postfix({"time": f"{elapsed:.4f}s"})
                    
                except Exception as e:
                    logger.error(f"Benchmark iteration {i+1} failed: {e}")
                
        if not times:
            raise RuntimeError("All benchmark iterations failed")
            
        # Calculate statistics
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        median_time = np.median(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1.0 / mean_time
        
        logger.info(f"Benchmark results: {iterations} iterations")
        logger.info(f"  Mean: {mean_time:.4f}s ± {std_time:.4f}s")
        logger.info(f"  Median: {median_time:.4f}s")
        logger.info(f"  Min/Max: {min_time:.4f}s / {max_time:.4f}s")
        logger.info(f"  FPS: {fps:.2f}")
        
        # Add resource usage if available
        resource_metrics = {}
        if can_measure_resources and cpu_percents and memory_usages:
            avg_cpu = np.mean(cpu_percents)
            avg_memory = np.mean(memory_usages)
            logger.info(f"  CPU usage: {avg_cpu:.1f}%")
            logger.info(f"  Memory delta: {avg_memory:.1f} MB")
            resource_metrics = {
                "avg_cpu_percent": avg_cpu,
                "avg_memory_delta_mb": avg_memory
            }
        
        return {
            "mean_time": mean_time,
            "std_time": std_time,
            "median_time": median_time,
            "min_time": min_time,
            "max_time": max_time,
            "fps": fps,
            "iterations": iterations,
            "device": str(self.device if hasattr(self, 'device') else self.config.device),
            **resource_metrics
        }

    async def shutdown(self):
        """
        Clean up resources properly and safely
        """
        logger.info("Shutting down SegmentationModel...")
        
        if hasattr(self.preprocessor, "cancel_pending_tasks"):
            self.preprocessor.cancel_pending_tasks()
            
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
            
        if self.use_engine and hasattr(self.engine, "cleanup"):
            try:
                await self.engine.cleanup()
            except Exception as e:
                logger.error(f"Error during engine cleanup: {e}")
        
        # Clear memory cache
        global image_cache
        image_cache.clear()
        
        # Print performance summary
        stats = self.get_performance_stats()
        if stats["total_images"] > 0:
            logger.info(f"Session summary: Processed {stats['total_images']} images at " 
                       f"{stats['images_per_second']:.2f} images/sec")
                
        logger.info("SegmentationModel shutdown complete")


################################################################################
# Enhanced Image Loading and I/O Functions
################################################################################
@asynccontextmanager
async def get_aiohttp_session():
    """Get aiohttp session with proper cleanup"""
    try:
        import aiohttp
        session = aiohttp.ClientSession()
        try:
            yield session
        finally:
            await session.close()
    except ImportError:
        # Fallback to requests
        class DummySession:
            async def get(self, url, **kwargs):
                return await asyncio.to_thread(requests.get, url, **kwargs)
        yield DummySession()

@cache_result(max_size=MAX_CACHE_SIZE)
async def load_image_async(image_path: str) -> np.ndarray:
    """
    Enhanced asynchronous image loader with better error handling and caching
    """
    try:
        if image_path.startswith(('http://', 'https://')):
            try:
                # Try with aiohttp for better async performance
                async with get_aiohttp_session() as session:
                    async with session.get(image_path, timeout=10) as response:
                        if response.status != 200:
                            raise ValueError(f"Failed to fetch image from URL: {image_path}, status: {response.status}")
                        content = await response.read()
                        img_array = np.frombuffer(content, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is None:
                            raise ValueError(f"Failed to decode image from URL: {image_path}")
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except ImportError:
                # Fallback to requests in thread pool
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
            # Local file
            loop = asyncio.get_running_loop()
            
            def _read_local_image():
                # Try to use PIL first for better format support
                try:
                    from PIL import Image
                    img = Image.open(image_path).convert('RGB')
                    return np.array(img)
                except (ImportError, IOError):
                    # Fall back to OpenCV
                    img = cv2.imread(image_path)
                    if img is None:
                        raise ValueError(f"Failed to read local image: {image_path}")
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
            return await loop.run_in_executor(None, _read_local_image)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise


################################################################################
# Enhanced Result Saving
################################################################################
async def save_segmentation_results(segmenter: SegmentationModel, 
                                   image_path: str, 
                                   output_dir: str,
                                   visualization_alpha: float = 0.5,
                                   mask_color: Tuple[int, int, int] = (0, 0, 255),
                                   contour_color: Tuple[int, int, int] = (0, 255, 0)) -> Dict:
    """
    Process an image and save the segmentation results with enhanced visualizations
    """
    # Create output directory with parents
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load image
        image = await load_image_async(image_path)
        
        # Get filename without extension
        if image_path.startswith(('http://', 'https://')):
            filename = image_path.split('/')[-1].split('.')[0]
        else:
            filename = Path(image_path).stem
        
        # Process the image
        result = await segmenter.process_image_async(image)
        
        # Get results
        mask = result["mask"]
        contours = result["contours"]
        
        # Create output paths
        original_path = os.path.join(output_dir, f"{filename}_original.jpg")
        mask_path = os.path.join(output_dir, f"{filename}_mask.png")
        contours_path = os.path.join(output_dir, f"{filename}_contours.jpg")
        overlay_path = os.path.join(output_dir, f"{filename}_overlay.jpg")
        
        # Use thread pool for faster I/O operations
        loop = asyncio.get_running_loop()
        
        # Create visualization
        overlay = segmenter.postprocessor.create_visualization(
            image, mask, contours, 
            alpha=visualization_alpha,
            mask_color=mask_color,
            contour_color=contour_color
        )
        
        # Save all outputs in parallel
        await asyncio.gather(
            loop.run_in_executor(None, cv2.imwrite, original_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)),
            loop.run_in_executor(None, cv2.imwrite, mask_path, mask),
            loop.run_in_executor(None, cv2.imwrite, contours_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        )
        
        logger.info(f"Saved segmentation results for {filename} to {output_dir}")
        
        # Compute additional metrics
        mask_coverage = np.count_nonzero(mask) / mask.size * 100
        largest_contour_area = max([cv2.contourArea(c) for c in contours]) if contours else 0
        
        return {
            "image_path": image_path,
            "mask_path": mask_path,
            "contours_path": contours_path, 
            "overlay_path": overlay_path,
            "num_contours": len(contours),
            "mask_coverage_percent": mask_coverage,
            "largest_contour_area": largest_contour_area,
            "processing_time": result["processing_time"]
        }
        
    except Exception as e:
        logger.error(f"Error saving segmentation results for {image_path}: {e}")
        return {
            "image_path": image_path,
            "error": str(e),
            "status": "failed"
        }


################################################################################
# Batch Processing with Enhanced Error Handling and Reporting
################################################################################
async def batch_process_images(model_path: str, image_paths: List[str], output_dir: str,
                              show_progress: bool = True, timeout: Optional[float] = None):
    """
    Enhanced batch processing with progress reporting, better error handling,
    and optimized concurrency
    """
    # Config validation with defaults
    if not model_path:
        raise ValueError("Model path must be provided")
    if not image_paths:
        raise ValueError("No images provided for processing")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model with better error handling
        start_time = time.time()
        logger.info(f"Loading model from: {model_path}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = torch.load(model_path, map_location=device)
            
            # Handle different model formats
            if isinstance(model, dict):
                if "model" in model:
                    model = model["model"]
                elif "state_dict" in model:
                    # This requires the model class
                    raise ValueError("Downloaded file contains only state_dict, model class is required")
            
            # Check if this is a valid model
            if not hasattr(model, "forward") and not hasattr(model, "predict") and not hasattr(model, "__call__"):
                raise ValueError(f"Loaded object is not a valid model: {type(model)}")
                
            logger.info(f"Successfully loaded model: {type(model).__name__}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Initialize segmenter
        segmenter = SegmentationModel(model)
        model_load_time = time.time() - start_time
        logger.info(f"Model loaded in {model_load_time:.2f} seconds")
        
        # Process images with progress bar
        total = len(image_paths)
        results = []
        failed = 0
        successful = 0
        
        # Create progress bar for better visibility
        with tqdm(total=total, desc="Processing images", disable=not show_progress) as pbar:
            # Determine optimal batch size for concurrent processing
            # but limit to avoid excessive memory usage
            concurrency = min(10, os.cpu_count() or 4)
            
            # Process in batches for better memory management
            for i in range(0, total, concurrency):
                batch = image_paths[i:i+concurrency]
                batch_tasks = []
                
                # Create tasks for concurrent processing
                for img_path in batch:
                    task = save_segmentation_results(segmenter, img_path, output_dir)
                    batch_tasks.append(task)
                
                # Process batch concurrently
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Handle results and update progress
                for img_path, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing {img_path}: {result}")
                        results.append({
                            "image_path": img_path, 
                            "error": str(result),
                            "status": "failed"
                        })
                        failed += 1
                    else:
                        results.append(result)
                        successful += 1
                    pbar.update(1)
        
        # Generate detailed summary report
        report_path = output_dir / "segmentation_report.md"
        total_time = sum(r.get("processing_time", 0) for r in results if isinstance(r, dict) and "processing_time" in r)
        
        with open(report_path, "w") as f:
            f.write("# Segmentation Batch Processing Report\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Model:** {Path(model_path).name}\n")
            f.write(f"**Device:** {device}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Total images processed:** {total}\n")
            f.write(f"- **Successful:** {successful}\n")
            f.write(f"- **Failed:** {failed}\n")
            f.write(f"- **Success rate:** {successful/total*100:.1f}%\n")
            f.write(f"- **Total processing time:** {total_time:.2f} seconds\n")
            if successful > 0:
                f.write(f"- **Average time per image:** {total_time/successful:.4f} seconds\n\n")
            
            # Add mask coverage statistics
            if successful > 0:
                coverages = [r.get("mask_coverage_percent", 0) for r in results 
                             if isinstance(r, dict) and "mask_coverage_percent" in r]
                if coverages:
                    avg_coverage = sum(coverages) / len(coverages)
                    f.write(f"- **Average mask coverage:** {avg_coverage:.1f}%\n\n")
            
            # Add performance metrics from segmenter
            f.write("## Performance Metrics\n\n")
            perf_stats = segmenter.get_performance_stats()
            f.write(f"- **Images per second:** {perf_stats['images_per_second']:.2f}\n")
            f.write(f"- **Total runtime:** {perf_stats['total_runtime_seconds']:.2f} seconds\n\n")
            
            # Detailed results table
            f.write("## Detailed Results\n\n")
            f.write("| Image | Status | Processing Time | Contours | Coverage |\n")
            f.write("|-------|--------|----------------|----------|----------|\n")
            
            for r in results:
                img_name = Path(r["image_path"]).name
                if "error" in r:
                    f.write(f"| {img_name} | ❌ Failed | - | - | - |\n")
                else:
                    time_ms = r.get("processing_time", 0) * 1000
                    contours = r.get("num_contours", 0)
                    coverage = r.get("mask_coverage_percent", 0)
                    f.write(f"| {img_name} | ✅ Success | {time_ms:.1f} ms | {contours} | {coverage:.1f}% |\n")
        
        logger.info(f"Batch processing complete: {successful}/{total} successful")
        logger.info(f"Report saved to {report_path}")
        
        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "report_path": str(report_path),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info=True)
        raise
    finally:
        if 'segmenter' in locals():
            await segmenter.shutdown()


################################################################################
# Main Demo Function
################################################################################
async def main():
    """Enhanced demo function with better error handling and example usage"""
    # Set up error handling for the entire process
    try:
        logger.info("Starting segmentation model demo")
        
        # Try to download a demo model if not provided
        try:
            model_path = download_model(
                custom_url="https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n-seg.pt",
                model_name="yolov8n-seg.pt"
            )
            logger.info(f"Downloaded model to {model_path}")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            # Use a fallback URL or local model
            model_path = download_model(
                custom_url="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt",
                model_name="yolov8n-seg.pt"
            )
            
        # Load the model with better error handling
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = torch.load(model_path, map_location=device)
            
            # Handle different model formats
            if isinstance(model, dict):
                if "model" in model:
                    model = model["model"]
                elif "state_dict" in model:
                    # This is just a state dict, need the model class
                    raise ValueError("Downloaded file contains only state_dict, model class is required")
                    
            logger.info(f"Successfully loaded model of type: {type(model).__name__}")
            
        except Exception as e:
            logger.error(f"Failed to load the model: {e}", exc_info=True)
            raise

        # Initialize the segmentation pipeline
        segmenter = SegmentationModel(model)
        
        try:
            # Example images for demonstration
            images = [
                "https://ultralytics.com/images/zidane.jpg",
                "https://ultralytics.com/images/bus.jpg",
                "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg"
            ]
            
            logger.info(f"Processing {len(images)} example images...")
            
            # Create demo output directory
            demo_dir = Path("segmentation_demo")
            demo_dir.mkdir(exist_ok=True)
            
            # Process images and save results
            tasks = []
            for img in images:
                task = save_segmentation_results(segmenter, img, str(demo_dir))
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Display results summary
            print("\nSegmentation Results Summary:")
            print("----------------------------")
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Image {idx+1}: ❌ Failed - {result}")
                else:
                    print(f"Image {idx+1}: ✅ Success - {Path(images[idx]).name}")
                    print(f"  • Contours found: {result['num_contours']}")
                    print(f"  • Processing time: {result['processing_time']*1000:.1f} ms")
                    print(f"  • Mask coverage: {result.get('mask_coverage_percent', 0):.1f}%")
                    print(f"  • Results saved to: {demo_dir}")
                print()
            
            # Run a quick benchmark
            if isinstance(results[0], dict):
                print("\nRunning benchmark...")
                benchmark_results = segmenter.benchmark_segmentation(
                    images[0], iterations=5, warmup=2)
                
                print("\nBenchmark Results:")
                print("----------------")
                print(f"• Average processing time: {benchmark_results['mean_time']*1000:.1f} ms")
                print(f"• Frames per second: {benchmark_results['fps']:.2f}")
                print(f"• Device: {benchmark_results['device']}")
                if "avg_cpu_percent" in benchmark_results:
                    print(f"• CPU usage: {benchmark_results['avg_cpu_percent']:.1f}%")
                print()
            
        except Exception as e:
            logger.error(f"Error during image processing: {e}", exc_info=True)
        finally:
            # Ensure we clean up resources properly
            await segmenter.shutdown()
            logger.info("Demo completed")
            
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        sys.exit(1)


################################################################################
# Command-line Interface with Argument Groups
################################################################################
def parse_args():
    """Enhanced command line argument parser with argument groups"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Advanced Segmentation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create argument groups for better organization
    input_group = parser.add_argument_group("Input Options")
    model_group = parser.add_argument_group("Model Options")
    output_group = parser.add_argument_group("Output Options")
    perf_group = parser.add_argument_group("Performance Options")
    
    # Input options
    input_group.add_argument("--images", type=str, nargs="+", 
                        help="Paths or URLs to images to process")
    input_group.add_argument("--image_dir", type=str,
                        help="Directory containing images to process")
    input_group.add_argument("--image_extensions", type=str, nargs="+",
                        default=["jpg", "jpeg", "png"],
                        help="File extensions to process when using --image_dir")
    
    # Model options
    model_group.add_argument("--model", type=str, default="",
                        help="Path to model file or URL to download")
    model_group.add_argument("--threshold", type=float, default=0.5,
                        help="Segmentation threshold")
    model_group.add_argument("--min_contour_area", type=int, default=100,
                        help="Minimum contour area to keep")
    model_group.add_argument("--apply_sigmoid", action="store_true",
                        help="Apply sigmoid activation to model output")
    
    # Output options
    output_group.add_argument("--output_dir", type=str, default="segmentation_results",
                        help="Directory to save results")
    output_group.add_argument("--no_progress", action="store_true",
                        help="Disable progress bar")
    output_group.add_argument("--mask_color", type=int, nargs=3, default=[0, 0, 255],
                        help="RGB color for mask visualization")
    output_group.add_argument("--contour_color", type=int, nargs=3, default=[0, 255, 0],
                        help="RGB color for contour visualization")
    
    # Performance options
    perf_group.add_argument("--device", type=str, default="",
                        help="Device to use (cuda, cuda:0, cpu)")
    perf_group.add_argument("--batch_size", type=int, default=0,
                        help="Processing batch size (0 for auto)")
    perf_group.add_argument("--use_fp16", action="store_true",
                        help="Use half precision (FP16) for inference")
    perf_group.add_argument("--timeout", type=float, default=30.0,
                        help="Timeout in seconds for each inference")
    perf_group.add_argument("--benchmark", action="store_true",
                        help="Run benchmark on the first image")
    perf_group.add_argument("--benchmark_iterations", type=int, default=10,
                        help="Number of iterations for benchmark")
    
    return parser.parse_args()


def get_images_from_dir(directory: str, extensions: List[str]) -> List[str]:
    """Get all image paths from a directory with specified extensions"""
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory not found: {directory}")
        
    # Normalize extensions (remove dots, convert to lowercase)
    extensions = [ext.lower().lstrip('.') for ext in extensions]
    
    # Find all matching files
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(directory.glob(f"*.{ext}")))
        image_paths.extend(list(directory.glob(f"*.{ext.upper()}")))
    
    return [str(p) for p in sorted(image_paths)]


if __name__ == "__main__":
    args = parse_args()
    
    # Override config with command line args
    if hasattr(SEGMENTATION_CONFIG, "threshold"):
        SEGMENTATION_CONFIG.threshold = args.threshold
    else:
        SEGMENTATION_CONFIG = {**SEGMENTATION_CONFIG, "threshold": args.threshold}
    
    if hasattr(SEGMENTATION_CONFIG, "min_contour_area"):
        SEGMENTATION_CONFIG.min_contour_area = args.min_contour_area
    else:
        SEGMENTATION_CONFIG = {**SEGMENTATION_CONFIG, "min_contour_area": args.min_contour_area}
        
    if hasattr(SEGMENTATION_CONFIG, "apply_sigmoid"):
        SEGMENTATION_CONFIG.apply_sigmoid = args.apply_sigmoid
    else:
        SEGMENTATION_CONFIG = {**SEGMENTATION_CONFIG, "apply_sigmoid": args.apply_sigmoid}
        
    if hasattr(SEGMENTATION_CONFIG, "use_fp16"):
        SEGMENTATION_CONFIG.use_fp16 = args.use_fp16
    else:
        SEGMENTATION_CONFIG = {**SEGMENTATION_CONFIG, "use_fp16": args.use_fp16}
    
    # Set device
    if args.device:
        device = args.device
        # Override device in config
        if hasattr(SEGMENTATION_CONFIG, "device"):
            SEGMENTATION_CONFIG.device = device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set CUDA visible devices if needed
    if device.startswith("cuda") and ":" in device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
    
    # If specific model provided, use it
    if args.model:
        model_path = args.model
    else:
        # Default to YOLOv8 segmentation model
        model_path = download_model(
            custom_url="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt",
            model_name="yolov8n-seg.pt"
        )
    
    # Get images to process
    image_paths = []
    if args.images:
        image_paths.extend(args.images)
    
    if args.image_dir:
        dir_images = get_images_from_dir(args.image_dir, args.image_extensions)
        image_paths.extend(dir_images)
        print(f"Found {len(dir_images)} images in directory {args.image_dir}")
    
    # Process images if provided
    if image_paths:
        # Run batch processing
        asyncio.run(batch_process_images(
            model_path, 
            image_paths, 
            args.output_dir,
            show_progress=not args.no_progress,
            timeout=args.timeout
        ))
        
        # Run benchmark if requested
        if args.benchmark and image_paths:
            print("\nRunning benchmark...")
            # Load model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = torch.load(model_path, map_location=device)
            segmenter = SegmentationModel(model)
            
            try:
                results = segmenter.benchmark_segmentation(
                    image_paths[0], 
                    iterations=args.benchmark_iterations,
                    warmup=2,
                    show_progress=not args.no_progress
                )
                
                print("\nBenchmark Results:")
                print("----------------")
                print(f"• Average processing time: {results['mean_time']*1000:.1f} ms")
                print(f"• Median processing time: {results['median_time']*1000:.1f} ms")
                print(f"• Frames per second: {results['fps']:.2f}")
                print(f"• Device: {results['device']}")
                if "avg_cpu_percent" in results:
                    print(f"• CPU usage: {results['avg_cpu_percent']:.1f}%")
                print()
            finally:
                asyncio.run(segmenter.shutdown())
    else:
        # Run the default demo
        asyncio.run(main())
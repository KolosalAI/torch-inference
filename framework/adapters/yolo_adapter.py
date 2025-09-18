"""
YOLO model adapter for object detection models.

This module provides adapters for YOLO (You Only Look Once) object detection models,
including YOLOv5, YOLOv8, YOLOv10, and other YOLO variants.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

from ..core.base_model import BaseModel, ModelMetadata, ModelLoadError, ModelInferenceError
from ..core.config import InferenceConfig

logger = logging.getLogger(__name__)


class YOLOModelAdapter(BaseModel):
    """Adapter for YOLO object detection models."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.model_path: Optional[Path] = None
        self.class_names: List[str] = []
        self.yolo_variant: str = "unknown"
        self.input_size: Tuple[int, int] = (640, 640)
        self.confidence_threshold: float = 0.25
        self.iou_threshold: float = 0.45
        self.max_detections: int = 300
        
        # Override default config values for YOLO
        if hasattr(config, 'postprocessing'):
            self.confidence_threshold = getattr(config.postprocessing, 'threshold', 0.25)
            self.iou_threshold = getattr(config.postprocessing, 'nms_threshold', 0.45)
            self.max_detections = getattr(config.postprocessing, 'max_detections', 300)
        
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load YOLO model from various sources."""
        try:
            model_path = Path(model_path) if isinstance(model_path, str) else model_path
            self.model_path = model_path
            
            self.logger.info(f"Loading YOLO model from {model_path}")
            self.logger.info(f"Target device: {self.device}")
            
            # Determine YOLO variant and load appropriately
            if self._is_ultralytics_model(model_path):
                self._load_ultralytics_model(model_path)
            elif self._is_yolov5_model(model_path):
                self._load_yolov5_model(model_path)
            elif self._is_pytorch_model(model_path):
                self._load_pytorch_yolo_model(model_path)
            else:
                raise ModelLoadError(f"Unsupported YOLO model format: {model_path}")
            
            # Move to target device
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
                self.logger.info(f"YOLO model moved to device: {self.device}")
            
            # Set metadata
            self.metadata = ModelMetadata(
                name=model_path.stem if hasattr(model_path, 'stem') else str(model_path),
                version="1.0",
                model_type="yolo",
                input_shape=(3,) + self.input_size,
                output_shape=self._get_output_shape(),
                description=f"YOLO {self.yolo_variant} object detection model"
            )
            
            self._is_loaded = True
            self.logger.info(f"Successfully loaded YOLO model: {self.metadata.name} ({self.yolo_variant})")
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise ModelLoadError(f"Failed to load YOLO model: {e}") from e
    
    def _is_ultralytics_model(self, model_path: Path) -> bool:
        """Check if model is an Ultralytics YOLO model."""
        # Check for Ultralytics YOLO (YOLOv8+)
        try:
            import ultralytics
            # Try to load as ultralytics model
            if str(model_path).endswith('.pt'):
                return True
            # Also support model names like 'yolov8n.pt', 'yolov8s.pt', etc.
            if any(variant in str(model_path).lower() for variant in ['yolov8', 'yolov9', 'yolov10', 'yolo11']):
                return True
        except ImportError:
            pass
        return False
    
    def _is_yolov5_model(self, model_path: Path) -> bool:
        """Check if model is a YOLOv5 model."""
        try:
            # Check for YOLOv5 indicators
            if 'yolov5' in str(model_path).lower():
                return True
            # Could also check model structure if it's a .pt file
        except Exception:
            pass
        return False
    
    def _is_pytorch_model(self, model_path: Path) -> bool:
        """Check if model is a standard PyTorch model."""
        return model_path.suffix in ['.pt', '.pth', '.torchscript']
    
    def _load_ultralytics_model(self, model_path: Path) -> None:
        """Load Ultralytics YOLO model (YOLOv8+)."""
        try:
            from ultralytics import YOLO
            
            # Load model
            self.model = YOLO(str(model_path))
            self.yolo_variant = "ultralytics"
            
            # Extract model info
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                self.class_names = list(self.model.model.names.values())
            elif hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            else:
                # Default COCO classes
                self.class_names = self._get_default_coco_classes()
            
            # Get input size
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'yaml'):
                yaml_dict = self.model.model.yaml
                if isinstance(yaml_dict, dict) and 'imgsz' in yaml_dict:
                    size = yaml_dict['imgsz']
                    self.input_size = (size, size) if isinstance(size, int) else tuple(size[:2])
            
            self.logger.info(f"Loaded Ultralytics YOLO model with {len(self.class_names)} classes")
            
        except ImportError:
            raise ModelLoadError("ultralytics package required for YOLOv8+ models. Install with: pip install ultralytics")
        except Exception as e:
            raise ModelLoadError(f"Failed to load Ultralytics YOLO model: {e}")
    
    def _load_yolov5_model(self, model_path: Path) -> None:
        """Load YOLOv5 model."""
        try:
            # Try YOLOv5 from torch.hub first
            try:
                import yolov5
                self.model = yolov5.load(str(model_path))
                self.yolo_variant = "yolov5"
            except ImportError:
                # Fallback: load as torch hub model
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path))
                self.yolo_variant = "yolov5_hub"
            
            # Extract class names
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            else:
                self.class_names = self._get_default_coco_classes()
            
            self.logger.info(f"Loaded YOLOv5 model with {len(self.class_names)} classes")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load YOLOv5 model: {e}")
    
    def _load_pytorch_yolo_model(self, model_path: Path) -> None:
        """Load generic PyTorch YOLO model."""
        try:
            # Load as standard PyTorch model
            if model_path.suffix == '.torchscript':
                self.model = torch.jit.load(str(model_path), map_location=self.device)
            else:
                checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
                
                if isinstance(checkpoint, nn.Module):
                    self.model = checkpoint
                elif isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        self.model = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        raise ModelLoadError("State dict found but no model architecture provided for YOLO model")
                    else:
                        raise ModelLoadError("Unsupported PyTorch YOLO model format")
                else:
                    raise ModelLoadError(f"Unsupported checkpoint format: {type(checkpoint)}")
            
            self.yolo_variant = "pytorch"
            self.class_names = self._get_default_coco_classes()  # Default to COCO
            
            self.logger.info(f"Loaded PyTorch YOLO model")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load PyTorch YOLO model: {e}")
    
    def _get_default_coco_classes(self) -> List[str]:
        """Get default COCO class names."""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """Preprocess inputs for YOLO model."""
        try:
            # Handle different input types
            if isinstance(inputs, torch.Tensor):
                # Already a tensor, ensure correct format
                image_tensor = inputs
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            elif isinstance(inputs, (str, Path)):
                # Image file path
                image_tensor = self._load_image_from_path(inputs)
            elif isinstance(inputs, Image.Image):
                # PIL Image
                image_tensor = self._pil_to_tensor(inputs)
            elif isinstance(inputs, np.ndarray):
                # Numpy array
                image_tensor = self._numpy_to_tensor(inputs)
            else:
                raise ValueError(f"Unsupported input type: {type(inputs)}")
            
            # Resize to model input size
            image_tensor = self._resize_image(image_tensor)
            
            # Normalize to [0, 1] range
            if image_tensor.dtype == torch.uint8:
                image_tensor = image_tensor.float() / 255.0
            elif image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            
            # Move to device
            image_tensor = image_tensor.to(self.device)
            
            return image_tensor
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise ModelInferenceError(f"Preprocessing failed: {e}") from e
    
    def _load_image_from_path(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Load image from file path."""
        try:
            # Use PIL for better format support
            image = Image.open(image_path).convert('RGB')
            return self._pil_to_tensor(image)
        except Exception as e:
            # Fallback to OpenCV
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return self._numpy_to_tensor(image)
            except Exception as e2:
                raise ValueError(f"Failed to load image {image_path}: {e2}")
    
    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor."""
        image_array = np.array(image)
        return self._numpy_to_tensor(image_array)
    
    def _numpy_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor."""
        # Ensure correct shape (H, W, C) -> (C, H, W)
        if image.ndim == 3 and image.shape[2] == 3:
            image = np.transpose(image, (2, 0, 1))
        elif image.ndim == 2:
            # Grayscale to RGB
            image = np.stack([image] * 3, axis=0)
        
        tensor = torch.from_numpy(image.copy())
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def _resize_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Resize image tensor to model input size."""
        if image_tensor.shape[-2:] != self.input_size:
            # Use F.interpolate for resizing
            import torch.nn.functional as F
            image_tensor = F.interpolate(
                image_tensor, 
                size=self.input_size, 
                mode='bilinear', 
                align_corners=False
            )
        return image_tensor
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run forward pass through YOLO model."""
        if not self._is_loaded:
            raise ModelLoadError("Model not loaded")
        
        try:
            model = self.get_model_for_inference()
            
            # Ensure inputs are on the correct device
            inputs = inputs.to(self.device)
            
            # Handle different YOLO variants
            if self.yolo_variant == "ultralytics":
                # Ultralytics YOLO
                results = model(inputs, verbose=False)
                # Extract raw predictions
                if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                    # YOLOv8+ format
                    outputs = results[0].boxes.data  # [N, 6] format: [x1, y1, x2, y2, conf, cls]
                else:
                    # Fallback: try to get raw tensor
                    outputs = results[0]
                    if not isinstance(outputs, torch.Tensor):
                        # Convert to tensor if needed
                        outputs = torch.tensor(outputs, device=self.device)
            
            elif self.yolo_variant in ["yolov5", "yolov5_hub"]:
                # YOLOv5
                with torch.no_grad():
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Take the first output (predictions)
            
            elif self.yolo_variant == "pytorch":
                # Generic PyTorch YOLO
                with torch.no_grad():
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
            
            else:
                raise ModelInferenceError(f"Unsupported YOLO variant: {self.yolo_variant}")
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"Forward pass failed: {e}")
            raise ModelInferenceError(f"Forward pass failed: {e}") from e
    
    def postprocess(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Postprocess YOLO model outputs."""
        try:
            # Handle different output formats
            if self.yolo_variant == "ultralytics":
                return self._postprocess_ultralytics(outputs)
            elif self.yolo_variant in ["yolov5", "yolov5_hub"]:
                return self._postprocess_yolov5(outputs)
            elif self.yolo_variant == "pytorch":
                return self._postprocess_pytorch(outputs)
            else:
                raise ModelInferenceError(f"Unsupported YOLO variant: {self.yolo_variant}")
                
        except Exception as e:
            self.logger.error(f"Postprocessing failed: {e}")
            raise ModelInferenceError(f"Postprocessing failed: {e}") from e
    
    def _postprocess_ultralytics(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Postprocess Ultralytics YOLO outputs."""
        detections = []
        
        if outputs.dim() == 2 and outputs.shape[1] >= 6:
            # Format: [x1, y1, x2, y2, conf, cls, ...]
            for detection in outputs:
                if detection[4] >= self.confidence_threshold:  # confidence check
                    x1, y1, x2, y2, conf, cls = detection[:6]
                    
                    # Ensure coordinates are within bounds
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    
                    class_id = int(cls)
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class_id': class_id,
                        'class_name': class_name
                    })
        
        # Apply NMS if needed
        if len(detections) > 1:
            detections = self._apply_nms(detections)
        
        return {
            'detections': detections[:self.max_detections],
            'num_detections': len(detections),
            'input_size': self.input_size,
            'model_type': 'yolo'
        }
    
    def _postprocess_yolov5(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Postprocess YOLOv5 outputs."""
        detections = []
        
        # YOLOv5 output format: [batch, detections, 5+num_classes]
        # where 5 = [x, y, w, h, objectness]
        if outputs.dim() == 3:
            batch_outputs = outputs[0]  # Take first batch
            
            for detection in batch_outputs:
                objectness = detection[4]
                if objectness >= self.confidence_threshold:
                    # Get class scores
                    class_scores = detection[5:]
                    class_conf, class_id = torch.max(class_scores, 0)
                    
                    total_conf = objectness * class_conf
                    if total_conf >= self.confidence_threshold:
                        # Convert center format to corner format
                        x_center, y_center, width, height = detection[:4]
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        
                        class_id = int(class_id)
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(total_conf),
                            'class_id': class_id,
                            'class_name': class_name
                        })
        
        # Apply NMS
        if len(detections) > 1:
            detections = self._apply_nms(detections)
        
        return {
            'detections': detections[:self.max_detections],
            'num_detections': len(detections),
            'input_size': self.input_size,
            'model_type': 'yolo'
        }
    
    def _postprocess_pytorch(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Postprocess generic PyTorch YOLO outputs."""
        # This is a generic handler - actual format depends on the specific model
        # We'll try to handle common formats
        
        detections = []
        
        # Try to detect output format
        if outputs.dim() == 3 and outputs.shape[2] >= 5:
            # Likely format: [batch, detections, features]
            batch_outputs = outputs[0]  # Take first batch
            
            for detection in batch_outputs:
                if len(detection) >= 5:
                    # Assume format: [x1, y1, x2, y2, conf, cls] or [x, y, w, h, conf, cls]
                    if len(detection) >= 6:
                        # With class
                        if detection[0] < 1.0 and detection[1] < 1.0:
                            # Normalized coordinates
                            x1, y1, x2, y2 = detection[:4] * torch.tensor([self.input_size[1], self.input_size[0], self.input_size[1], self.input_size[0]])
                        else:
                            x1, y1, x2, y2 = detection[:4]
                        
                        conf = detection[4]
                        class_id = int(detection[5])
                    else:
                        # Without class (assume class 0)
                        if detection[0] < 1.0 and detection[1] < 1.0:
                            x1, y1, x2, y2 = detection[:4] * torch.tensor([self.input_size[1], self.input_size[0], self.input_size[1], self.input_size[0]])
                        else:
                            x1, y1, x2, y2 = detection[:4]
                        
                        conf = detection[4]
                        class_id = 0
                    
                    if conf >= self.confidence_threshold:
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class_id': class_id,
                            'class_name': class_name
                        })
        
        return {
            'detections': detections[:self.max_detections],
            'num_detections': len(detections),
            'input_size': self.input_size,
            'model_type': 'yolo'
        }
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if len(detections) <= 1:
            return detections
        
        try:
            # Convert to tensors for NMS
            boxes = torch.tensor([det['bbox'] for det in detections])
            scores = torch.tensor([det['confidence'] for det in detections])
            
            # Apply NMS
            from torchvision.ops import nms
            keep_indices = nms(boxes, scores, self.iou_threshold)
            
            # Return filtered detections
            return [detections[i] for i in keep_indices.tolist()]
            
        except Exception as e:
            self.logger.warning(f"NMS failed, returning original detections: {e}")
            return detections
    
    def _get_output_shape(self) -> Tuple[int, ...]:
        """Get model output shape."""
        # YOLO output shape varies by model, use a generic shape
        return (self.max_detections, 6)  # [x1, y1, x2, y2, conf, cls]
    
    def _create_dummy_input(self) -> torch.Tensor:
        """Create dummy input for warmup."""
        return torch.randn(1, 3, *self.input_size, device=self.device, dtype=torch.float32)
    
    def predict_batch(self, inputs_list: List[Any]) -> List[Dict[str, Any]]:
        """
        Batch prediction for YOLO models.
        
        Args:
            inputs_list: List of images
            
        Returns:
            List of detection results
        """
        if not inputs_list:
            return []
        
        try:
            # Preprocess all inputs
            preprocessed_inputs = []
            for inp in inputs_list:
                processed = self.preprocess(inp)
                if processed.dim() == 4 and processed.shape[0] == 1:
                    processed = processed.squeeze(0)  # Remove batch dim for stacking
                preprocessed_inputs.append(processed)
            
            # Stack into batch
            batch_tensor = torch.stack(preprocessed_inputs, dim=0)
            
            # Forward pass
            with torch.no_grad():
                batch_outputs = self.forward(batch_tensor)
            
            # Handle batch outputs based on YOLO variant
            results = []
            if self.yolo_variant == "ultralytics":
                # Ultralytics handles batches internally
                if hasattr(batch_outputs, '__len__') and len(batch_outputs) == len(inputs_list):
                    for output in batch_outputs:
                        result = self.postprocess(output)
                        results.append(result)
                else:
                    # Single output for batch
                    result = self.postprocess(batch_outputs)
                    results = [result] * len(inputs_list)
            else:
                # For other variants, split batch outputs
                if batch_outputs.dim() >= 1:
                    batch_size = batch_outputs.shape[0]
                    for i in range(min(batch_size, len(inputs_list))):
                        output = batch_outputs[i:i+1]  # Keep batch dimension
                        result = self.postprocess(output)
                        results.append(result)
                
                # Fill missing results if needed
                while len(results) < len(inputs_list):
                    results.append({'detections': [], 'num_detections': 0, 'input_size': self.input_size, 'model_type': 'yolo'})
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Batch processing failed: {e}, falling back to individual processing")
            # Fallback to individual processing
            return [self.predict(inp) for inp in inputs_list]


class YOLOv5Adapter(YOLOModelAdapter):
    """Specialized adapter for YOLOv5 models."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.yolo_variant = "yolov5"
        
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load YOLOv5 model specifically."""
        try:
            # Force YOLOv5 loading
            self._load_yolov5_model(Path(model_path))
            
            # Set metadata
            self.metadata = ModelMetadata(
                name=f"yolov5_{Path(model_path).stem}",
                version="1.0",
                model_type="yolov5",
                input_shape=(3,) + self.input_size,
                output_shape=self._get_output_shape(),
                description="YOLOv5 object detection model"
            )
            
            self._is_loaded = True
            self.logger.info(f"Successfully loaded YOLOv5 model")
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv5 model: {e}")
            raise ModelLoadError(f"Failed to load YOLOv5 model: {e}") from e


class YOLOv8Adapter(YOLOModelAdapter):
    """Specialized adapter for YOLOv8+ models (Ultralytics)."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.yolo_variant = "ultralytics"
        
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load YOLOv8+ model specifically."""
        try:
            # Force Ultralytics loading
            self._load_ultralytics_model(Path(model_path))
            
            # Set metadata
            self.metadata = ModelMetadata(
                name=f"yolov8_{Path(model_path).stem}",
                version="1.0",
                model_type="yolov8",
                input_shape=(3,) + self.input_size,
                output_shape=self._get_output_shape(),
                description="YOLOv8+ object detection model"
            )
            
            self._is_loaded = True
            self.logger.info(f"Successfully loaded YOLOv8+ model")
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv8+ model: {e}")
            raise ModelLoadError(f"Failed to load YOLOv8+ model: {e}") from e
"""
Generic postprocessor framework for various output types.

This module provides a flexible postprocessing system that can handle
different model outputs (classification, detection, segmentation, etc.)
and convert them to user-friendly formats.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import numpy as np
import torch
import time
from dataclasses import dataclass
from enum import Enum

from ..core.config import InferenceConfig, ModelType


logger = logging.getLogger(__name__)


class OutputType(Enum):
    """Types of model outputs."""
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    REGRESSION = "regression"
    EMBEDDING = "embedding"
    CUSTOM = "custom"


@dataclass
class PostprocessingResult:
    """Result of postprocessing operation."""
    predictions: Any
    confidence_scores: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format for compatibility."""
        # Convert predictions to dict if they have a to_dict method
        if hasattr(self.predictions, 'to_dict'):
            predictions_data = self.predictions.to_dict()
        else:
            predictions_data = self.predictions
            
        result = {
            "predictions": predictions_data,
            "processing_time": self.processing_time,
            "metadata": self.metadata.copy() if self.metadata else {}
        }
        
        if self.confidence_scores is not None:
            result["confidence_scores"] = self.confidence_scores
            
        # Add compatibility fields based on prediction type
        if hasattr(self.predictions, 'predicted_class'):
            result["predicted_class"] = self.predictions.predicted_class
            result["confidence"] = self.predictions.confidence
        
        return result


@dataclass
@dataclass
class ClassificationResult:
    """Classification result."""
    predicted_class: int
    class_name: Optional[str]
    confidence: float
    top_k_classes: Optional[List[Tuple[int, str, float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        # For single predictions, return the predicted class as a regular Python type
        # instead of a tensor to ensure JSON serialization works correctly
        
        result = {
            "predictions": self.predicted_class,  # Return as int for compatibility
            "confidence": self.confidence,
            "predicted_class": self.predicted_class,
        }
        
        if self.class_name is not None:
            result["class_name"] = self.class_name
            
        if self.top_k_classes is not None:
            result["top_k_classes"] = self.top_k_classes
            
        return result


@dataclass
class DetectionResult:
    """Object detection result."""
    boxes: List[Tuple[float, float, float, float]]  # [x1, y1, x2, y2]
    classes: List[int]
    class_names: Optional[List[str]]
    confidences: List[float]
    masks: Optional[np.ndarray] = None  # For instance segmentation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "boxes": self.boxes,
            "classes": self.classes,
            "confidences": self.confidences,
            "predictions": torch.tensor(self.classes)  # For compatibility
        }
        
        if self.class_names is not None:
            result["class_names"] = self.class_names
            
        if self.masks is not None:
            result["masks"] = self.masks
            
        return result


@dataclass
class SegmentationResult:
    """Segmentation result."""
    mask: np.ndarray
    contours: List[np.ndarray]
    area_pixels: int
    coverage_percentage: float
    largest_contour_area: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "mask": self.mask,
            "contours": self.contours,
            "area_pixels": self.area_pixels,
            "coverage_percentage": self.coverage_percentage,
            "largest_contour_area": self.largest_contour_area,
            "predictions": self.mask  # For compatibility
        }


class PostprocessingError(Exception):
    """Exception raised during postprocessing."""
    pass


class BasePostprocessor(ABC):
    """
    Abstract base class for all postprocessors.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def supports_output_type(self, output_type: OutputType) -> bool:
        """Check if this postprocessor supports the given output type."""
        pass
    
    @abstractmethod
    def postprocess(self, outputs: torch.Tensor, **kwargs) -> Union[Dict[str, Any], PostprocessingResult]:
        """Postprocess model outputs."""
        pass
    
    def validate_outputs(self, outputs: torch.Tensor) -> bool:
        """Validate model outputs."""
        return outputs is not None and isinstance(outputs, torch.Tensor)


class CustomPostprocessor(BasePostprocessor):
    """
    Custom postprocessor for generic/unknown output types.
    """
    
    def supports_output_type(self, output_type: OutputType) -> bool:
        """Supports custom output type."""
        return output_type == OutputType.CUSTOM
    
    def postprocess(self, outputs: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Simple postprocessing for custom outputs."""
        # Convert tensor to numpy for easier handling
        outputs_np = outputs.detach().cpu().numpy()
        
        # For compatibility with old tests expecting dict, create a simple result
        return {
            "predictions": outputs_np.tolist(),
            "raw_output": outputs_np.tolist(),
            "shape": outputs.shape,
            "prediction": "custom_result",
            "metadata": {
                "output_type": "custom",
                "shape": list(outputs.shape),
                "dtype": str(outputs.dtype)
            }
        }
    
    def validate_outputs(self, outputs: torch.Tensor) -> bool:
        """Validate custom outputs (always accepts)."""
        return isinstance(outputs, torch.Tensor)


class ClassificationPostprocessor(BasePostprocessor):
    """Postprocessor for classification outputs."""
    
    def __init__(self, config: InferenceConfig, class_names: Optional[List[str]] = None):
        super().__init__(config)
        self.class_names = class_names
        self.apply_softmax = config.postprocessing.apply_softmax
        self.top_k = config.custom_params.get("top_k", 5)
    
    def supports_output_type(self, output_type: OutputType) -> bool:
        """Check if this postprocessor supports the given output type."""
        return output_type == OutputType.CLASSIFICATION
    
    def postprocess(self, outputs: torch.Tensor, **kwargs) -> PostprocessingResult:
        """Postprocess classification outputs."""
        start_time = time.time()
        
        try:
            # Handle batch dimension
            if outputs.dim() == 2:
                # Batch of classifications (batch_size, num_classes)
                batch_size = outputs.size(0)
                
                if batch_size == 1:
                    # Single sample in batch
                    logits = outputs[0]
                    
                    # Apply softmax if requested
                    if self.apply_softmax:
                        probabilities = torch.softmax(logits, dim=0)
                    else:
                        probabilities = logits
                    
                    # Get top prediction
                    predicted_class_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_class_idx].item()
                    
                    # Get class name if available
                    predicted_class_name = None
                    if self.class_names and 0 <= predicted_class_idx < len(self.class_names):
                        predicted_class_name = self.class_names[predicted_class_idx]
                    
                    # Get top-k predictions
                    top_k_classes = None
                    if self.top_k > 0:
                        top_k_indices = torch.topk(probabilities, min(self.top_k, len(probabilities))).indices
                        top_k_classes = []
                        
                        for idx in top_k_indices:
                            idx = idx.item()
                            class_name = self.class_names[idx] if self.class_names else None
                            top_k_classes.append((idx, class_name, probabilities[idx].item()))
                    
                    # Create result
                    result = ClassificationResult(
                        predicted_class=predicted_class_idx,
                        class_name=predicted_class_name,
                        confidence=confidence,
                        top_k_classes=top_k_classes
                    )
                    
                    processing_time = time.time() - start_time
                    
                    return PostprocessingResult(
                        predictions=result,
                        confidence_scores=[confidence],
                        metadata={
                            "output_type": "classification",
                            "num_classes": len(probabilities),
                            "top_k": self.top_k
                        },
                        processing_time=processing_time
                    )
                else:
                    # Multiple samples in batch - return dict format for compatibility
                    # Apply softmax if requested
                    if self.apply_softmax:
                        probabilities = torch.softmax(outputs, dim=1)
                    else:
                        probabilities = outputs
                    
                    # Get predictions for entire batch
                    predicted_classes = torch.argmax(probabilities, dim=1)
                    confidences = torch.max(probabilities, dim=1).values
                    
                    processing_time = time.time() - start_time
                    
                    # Return dict format for batch compatibility
                    result = {
                        "predictions": predicted_classes.tolist(),  # Convert tensor to list
                        "confidence": confidences.mean().item(),  # Average confidence
                        "confidences": confidences.tolist(),  # Convert tensor to list
                        "probabilities": probabilities.tolist(),  # Convert tensor to list
                        "metadata": {
                            "output_type": "classification",
                            "batch_size": batch_size,
                            "num_classes": outputs.size(1),
                            "processing_time": processing_time
                        }
                    }
                    return result
                    
            elif outputs.dim() == 1:
                # Single prediction without batch dimension
                logits = outputs
                
                # Apply softmax if requested
                if self.apply_softmax:
                    probabilities = torch.softmax(logits, dim=0)
                else:
                    probabilities = logits
                
                # Get top prediction
                predicted_class_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class_idx].item()
                
                # Get class name if available
                predicted_class_name = None
                if self.class_names and 0 <= predicted_class_idx < len(self.class_names):
                    predicted_class_name = self.class_names[predicted_class_idx]
                
                # Get top-k predictions
                top_k_classes = None
                if self.top_k > 0:
                    top_k_indices = torch.topk(probabilities, min(self.top_k, len(probabilities))).indices
                    top_k_classes = []
                    
                    for idx in top_k_indices:
                        idx = idx.item()
                        class_name = self.class_names[idx] if self.class_names else None
                        top_k_classes.append((idx, class_name, probabilities[idx].item()))
                
                # Create result
                result = ClassificationResult(
                    predicted_class=predicted_class_idx,
                    class_name=predicted_class_name,
                    confidence=confidence,
                    top_k_classes=top_k_classes
                )
                
                processing_time = time.time() - start_time
                
                return PostprocessingResult(
                    predictions=result,
                    confidence_scores=[confidence],
                    metadata={
                        "output_type": "classification",
                        "num_classes": len(probabilities),
                        "top_k": self.top_k
                    },
                    processing_time=processing_time
                )
            else:
                raise ValueError(f"Unexpected output shape: {outputs.shape}")
            
        except Exception as e:
            self.logger.error(f"Classification postprocessing failed: {e}")
            raise PostprocessingError(f"Classification postprocessing failed: {e}") from e


class DetectionPostprocessor(BasePostprocessor):
    """Postprocessor for object detection outputs."""
    
    def __init__(self, config: InferenceConfig, class_names: Optional[List[str]] = None):
        super().__init__(config)
        self.class_names = class_names
        self.confidence_threshold = config.postprocessing.threshold
        self.nms_threshold = config.postprocessing.nms_threshold
        self.max_detections = config.postprocessing.max_detections
    
    def supports_output_type(self, output_type: OutputType) -> bool:
        """Check if this postprocessor supports the given output type."""
        return output_type == OutputType.DETECTION
    
    def postprocess(self, outputs: torch.Tensor, **kwargs) -> PostprocessingResult:
        """Postprocess detection outputs."""
        start_time = time.time()
        
        try:
            # Handle different output formats
            if self._is_yolo_format(outputs):
                result = self._postprocess_yolo(outputs, **kwargs)
            else:
                result = self._postprocess_generic(outputs, **kwargs)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Detection postprocessing failed: {e}")
            raise PostprocessingError(f"Detection postprocessing failed: {e}") from e
    
    def _is_yolo_format(self, outputs: torch.Tensor) -> bool:
        """Check if outputs are in YOLO format."""
        # YOLO typically outputs [batch, num_anchors, 5+num_classes] or similar
        return outputs.dim() == 3 and outputs.size(-1) > 5
    
    def _postprocess_yolo(self, outputs: torch.Tensor, **kwargs) -> PostprocessingResult:
        """Postprocess YOLO-style outputs."""
        # Handle batch dimension
        if outputs.size(0) == 1:
            detections = outputs[0]  # Remove batch dimension
        else:
            detections = outputs[0]  # Take first in batch
        
        # Filter by confidence
        obj_conf = detections[:, 4]  # Objectness confidence
        conf_mask = obj_conf > self.confidence_threshold
        detections = detections[conf_mask]
        
        if len(detections) == 0:
            return self._empty_detection_result()
        
        # Extract boxes, confidences, and classes
        boxes = detections[:, :4]  # x_center, y_center, width, height
        confidences = detections[:, 4]
        class_probs = detections[:, 5:]
        
        # Convert to corner format and get final confidences and classes
        final_boxes = []
        final_confidences = []
        final_classes = []
        
        for i, (box, obj_conf, class_prob) in enumerate(zip(boxes, confidences, class_probs)):
            # Get best class
            class_conf, class_idx = torch.max(class_prob, dim=0)
            final_conf = obj_conf * class_conf
            
            if final_conf > self.confidence_threshold:
                # Convert from center format to corner format
                x_center, y_center, width, height = box
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                final_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                final_confidences.append(final_conf.item())
                final_classes.append(class_idx.item())
        
        # Apply NMS
        if len(final_boxes) > 1:
            final_boxes, final_confidences, final_classes = self._apply_nms(
                final_boxes, final_confidences, final_classes
            )
        
        # Limit number of detections
        if len(final_boxes) > self.max_detections:
            # Sort by confidence and take top detections
            sorted_indices = sorted(range(len(final_confidences)), 
                                  key=lambda i: final_confidences[i], reverse=True)
            sorted_indices = sorted_indices[:self.max_detections]
            
            final_boxes = [final_boxes[i] for i in sorted_indices]
            final_confidences = [final_confidences[i] for i in sorted_indices]
            final_classes = [final_classes[i] for i in sorted_indices]
        
        # Get class names
        class_names = None
        if self.class_names:
            class_names = [self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}" 
                          for cls in final_classes]
        
        detection_result = DetectionResult(
            boxes=final_boxes,
            classes=final_classes,
            class_names=class_names,
            confidences=final_confidences
        )
        
        return PostprocessingResult(
            predictions=detection_result,
            confidence_scores=final_confidences,
            metadata={
                "output_type": "detection",
                "num_detections": len(final_boxes),
                "confidence_threshold": self.confidence_threshold,
                "nms_threshold": self.nms_threshold
            }
        )
    
    def _postprocess_generic(self, outputs: torch.Tensor, **kwargs) -> PostprocessingResult:
        """Postprocess generic detection outputs."""
        # Placeholder for generic detection postprocessing
        # This would be implemented based on specific model requirements
        return self._empty_detection_result()
    
    def _apply_nms(self, boxes: List[List[float]], confidences: List[float], 
                   classes: List[int]) -> Tuple[List[List[float]], List[float], List[int]]:
        """Apply Non-Maximum Suppression."""
        try:
            import torchvision.ops as ops
            
            # Convert to tensors
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            confidences_tensor = torch.tensor(confidences, dtype=torch.float32)
            
            # Apply NMS
            keep_indices = ops.nms(boxes_tensor, confidences_tensor, self.nms_threshold)
            
            # Filter results
            filtered_boxes = [boxes[i] for i in keep_indices]
            filtered_confidences = [confidences[i] for i in keep_indices]
            filtered_classes = [classes[i] for i in keep_indices]
            
            return filtered_boxes, filtered_confidences, filtered_classes
            
        except ImportError:
            # Fallback: simple NMS implementation
            self.logger.warning("torchvision not available, using simple NMS fallback")
            return self._simple_nms(boxes, confidences, classes)
    
    def _simple_nms(self, boxes: List[List[float]], confidences: List[float], 
                    classes: List[int]) -> Tuple[List[List[float]], List[float], List[int]]:
        """Simple NMS implementation."""
        # Sort by confidence
        indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
        
        keep = []
        while indices:
            current = indices.pop(0)
            keep.append(current)
            
            # Remove boxes with high IoU
            remaining = []
            for idx in indices:
                if self._calculate_iou(boxes[current], boxes[idx]) < self.nms_threshold:
                    remaining.append(idx)
            indices = remaining
        
        return ([boxes[i] for i in keep], 
                [confidences[i] for i in keep], 
                [classes[i] for i in keep])
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU)."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate union
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _empty_detection_result(self) -> PostprocessingResult:
        """Create empty detection result."""
        detection_result = DetectionResult(
            boxes=[],
            classes=[],
            class_names=[],
            confidences=[]
        )
        
        return PostprocessingResult(
            predictions=detection_result,
            confidence_scores=[],
            metadata={"output_type": "detection", "num_detections": 0}
        )


class SegmentationPostprocessor(BasePostprocessor):
    """Postprocessor for segmentation outputs."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.threshold = config.postprocessing.threshold
        self.apply_sigmoid = config.postprocessing.apply_sigmoid
        self.min_contour_area = config.custom_params.get("min_contour_area", 100)
        self.max_contours = config.custom_params.get("max_contours", 100)
    
    def supports_output_type(self, output_type: OutputType) -> bool:
        """Check if this postprocessor supports the given output type."""
        return output_type == OutputType.SEGMENTATION
    
    def postprocess(self, outputs: torch.Tensor, **kwargs) -> PostprocessingResult:
        """Postprocess segmentation outputs."""
        start_time = time.time()
        
        try:
            # Handle different output formats
            if hasattr(outputs, 'masks') and outputs.masks is not None:
                # YOLO-style segmentation
                result = self._postprocess_yolo_segmentation(outputs, **kwargs)
            else:
                # Generic segmentation tensor
                result = self._postprocess_generic_segmentation(outputs, **kwargs)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Segmentation postprocessing failed: {e}")
            raise PostprocessingError(f"Segmentation postprocessing failed: {e}") from e
    
    def _postprocess_yolo_segmentation(self, outputs: Any, **kwargs) -> PostprocessingResult:
        """Postprocess YOLO segmentation outputs."""
        try:
            # Extract masks from YOLO output
            if hasattr(outputs, 'masks') and outputs.masks is not None:
                masks = outputs.masks.data
            else:
                masks = outputs[0].masks.data if isinstance(outputs, list) else None
            
            if masks is None or len(masks) == 0:
                return self._empty_segmentation_result()
            
            # Combine all masks
            combined_mask = self._combine_masks(masks)
            
            # Find contours
            contours = self._find_contours(combined_mask)
            
            # Calculate metrics
            area_pixels = int(np.count_nonzero(combined_mask))
            total_pixels = combined_mask.size
            coverage_percentage = (area_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            largest_contour_area = 0
            if contours:
                try:
                    import cv2
                    largest_contour_area = max(cv2.contourArea(c) for c in contours)
                except ImportError:
                    largest_contour_area = len(contours[0]) if contours else 0
            
            segmentation_result = SegmentationResult(
                mask=combined_mask,
                contours=contours,
                area_pixels=area_pixels,
                coverage_percentage=coverage_percentage,
                largest_contour_area=largest_contour_area
            )
            
            return PostprocessingResult(
                predictions=segmentation_result,
                metadata={
                    "output_type": "segmentation",
                    "num_contours": len(contours),
                    "mask_shape": combined_mask.shape,
                    "coverage_percentage": coverage_percentage
                }
            )
            
        except Exception as e:
            self.logger.error(f"YOLO segmentation postprocessing failed: {e}")
            return self._empty_segmentation_result()
    
    def _postprocess_generic_segmentation(self, outputs: torch.Tensor, **kwargs) -> PostprocessingResult:
        """Postprocess generic segmentation outputs."""
        # Handle batch dimension
        if outputs.dim() == 4 and outputs.size(0) == 1:
            mask_logits = outputs[0]
        elif outputs.dim() == 3:
            mask_logits = outputs
        elif outputs.dim() == 2:
            mask_logits = outputs.unsqueeze(0)  # Add channel dimension
        else:
            raise ValueError(f"Unexpected output shape: {outputs.shape}")
        
        # Handle multi-class segmentation
        if mask_logits.size(0) > 1:
            # Take argmax across classes
            mask_logits = torch.argmax(mask_logits, dim=0).float()
        else:
            mask_logits = mask_logits[0]
        
        # Convert to numpy
        mask_np = mask_logits.detach().cpu().numpy()
        
        # Apply sigmoid if requested
        if self.apply_sigmoid:
            mask_np = 1 / (1 + np.exp(-mask_np))
        
        # Threshold to create binary mask
        binary_mask = (mask_np > self.threshold).astype(np.uint8) * 255
        
        # Find contours
        contours = self._find_contours(binary_mask)
        
        # Calculate metrics
        area_pixels = int(np.count_nonzero(binary_mask))
        total_pixels = binary_mask.size
        coverage_percentage = (area_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        largest_contour_area = 0
        if contours:
            try:
                import cv2
                largest_contour_area = max(cv2.contourArea(c) for c in contours)
            except ImportError:
                largest_contour_area = len(contours[0]) if contours else 0
        
        segmentation_result = SegmentationResult(
            mask=binary_mask,
            contours=contours,
            area_pixels=area_pixels,
            coverage_percentage=coverage_percentage,
            largest_contour_area=largest_contour_area
        )
        
        return PostprocessingResult(
            predictions=segmentation_result,
            metadata={
                "output_type": "segmentation",
                "num_contours": len(contours),
                "mask_shape": binary_mask.shape,
                "coverage_percentage": coverage_percentage
            }
        )
    
    def _combine_masks(self, masks: torch.Tensor) -> np.ndarray:
        """Combine multiple masks into one binary mask."""
        if len(masks) == 0:
            return np.zeros((100, 100), dtype=np.uint8)  # Default empty mask
        
        # Combine all masks with OR operation
        combined_mask = torch.zeros_like(masks[0], dtype=torch.uint8)
        
        for mask in masks:
            mask_binary = (mask > self.threshold).to(torch.uint8)
            combined_mask = torch.logical_or(combined_mask, mask_binary).to(torch.uint8)
        
        combined_mask_np = (combined_mask * 255).detach().cpu().numpy().astype(np.uint8)
        return combined_mask_np
    
    def _find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Find contours in the binary mask."""
        try:
            import cv2
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by area
            filtered_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
            
            # Limit number of contours
            if len(filtered_contours) > self.max_contours:
                filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
                filtered_contours = filtered_contours[:self.max_contours]
            
            return filtered_contours
            
        except ImportError:
            self.logger.warning("OpenCV not available, returning empty contours")
            return []
    
    def _empty_segmentation_result(self) -> PostprocessingResult:
        """Create empty segmentation result."""
        segmentation_result = SegmentationResult(
            mask=np.zeros((100, 100), dtype=np.uint8),
            contours=[],
            area_pixels=0,
            coverage_percentage=0.0,
            largest_contour_area=0.0
        )
        
        return PostprocessingResult(
            predictions=segmentation_result,
            metadata={"output_type": "segmentation", "num_contours": 0}
        )


class PostprocessorPipeline:
    """
    Pipeline for managing multiple postprocessors.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.postprocessors: Dict[OutputType, BasePostprocessor] = {}
        self.logger = logging.getLogger(f"{__name__}.PostprocessorPipeline")
        
        # Add default postprocessors
        self._add_default_postprocessors()
        
    def _add_default_postprocessors(self) -> None:
        """Add default postprocessors for each output type."""
        # Add classification postprocessor
        classification_processor = ClassificationPostprocessor(self.config)
        self.add_postprocessor(OutputType.CLASSIFICATION, classification_processor)
        
        # Add detection postprocessor
        detection_processor = DetectionPostprocessor(self.config)
        self.add_postprocessor(OutputType.DETECTION, detection_processor)
        
        # Add segmentation postprocessor
        segmentation_processor = SegmentationPostprocessor(self.config)
        self.add_postprocessor(OutputType.SEGMENTATION, segmentation_processor)
        
        # Add custom postprocessor for unknown types
        custom_processor = CustomPostprocessor(self.config)
        self.add_postprocessor(OutputType.CUSTOM, custom_processor)
    
    def add_postprocessor(self, output_type: OutputType, postprocessor: BasePostprocessor) -> None:
        """Add a postprocessor for a specific output type."""
        self.postprocessors[output_type] = postprocessor
        self.logger.info(f"Added postprocessor for {output_type.value}: {postprocessor.__class__.__name__}")
    
    def postprocess(self, outputs, output_type: OutputType, **kwargs) -> Union[Dict[str, Any], PostprocessingResult]:
        """Postprocess outputs using the appropriate postprocessor."""
        if output_type not in self.postprocessors:
            raise PostprocessingError(f"No postprocessor found for output type: {output_type}")
        
        postprocessor = self.postprocessors[output_type]
        
        # Extract tensor from outputs if needed
        tensor_outputs = outputs
        if hasattr(outputs, 'logits'):
            tensor_outputs = outputs.logits
        elif hasattr(outputs, 'last_hidden_state'):
            tensor_outputs = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            tensor_outputs = outputs[0]
            if hasattr(tensor_outputs, 'logits'):
                tensor_outputs = tensor_outputs.logits
        
        # Validate outputs
        if isinstance(tensor_outputs, torch.Tensor) and not postprocessor.validate_outputs(tensor_outputs):
            raise PostprocessingError("Output validation failed")
        
        return postprocessor.postprocess(tensor_outputs, **kwargs)
    
    def detect_output_type(self, outputs) -> OutputType:
        """Detect the type of model outputs."""
        # Handle different output types
        
        # Check if it's a Hugging Face output object
        if hasattr(outputs, 'logits'):
            # Extract logits tensor for analysis
            tensor_outputs = outputs.logits
        elif hasattr(outputs, 'last_hidden_state'):
            tensor_outputs = outputs.last_hidden_state
        elif isinstance(outputs, torch.Tensor):
            tensor_outputs = outputs
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            # Take the first tensor from tuple/list outputs
            tensor_outputs = outputs[0]
            if hasattr(tensor_outputs, 'logits'):
                tensor_outputs = tensor_outputs.logits
        else:
            # If we can't determine the type, assume custom
            return OutputType.CUSTOM
        
        # Now analyze the tensor
        if not isinstance(tensor_outputs, torch.Tensor):
            return OutputType.CUSTOM
            
        # Simple heuristics for output type detection
        if tensor_outputs.dim() == 2 and tensor_outputs.size(1) > 1:
            # Likely classification (batch_size, num_classes)
            return OutputType.CLASSIFICATION
        elif tensor_outputs.dim() == 3 and tensor_outputs.size(-1) > 5:
            # Likely detection (batch_size, num_detections, [x, y, w, h, conf, classes...])
            return OutputType.DETECTION
        elif tensor_outputs.dim() >= 3 and min(tensor_outputs.shape[-2:]) > 10:
            # Likely segmentation (has spatial dimensions)
            return OutputType.SEGMENTATION
        else:
            return OutputType.CUSTOM
    
    def auto_postprocess(self, outputs, **kwargs) -> Union[Dict[str, Any], PostprocessingResult]:
        """Automatically detect output type and postprocess."""
        output_type = self.detect_output_type(outputs)
        return self.postprocess(outputs, output_type, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get postprocessing statistics."""
        return {
            "num_postprocessors": len(self.postprocessors),
            "supported_output_types": [ot.value for ot in self.postprocessors.keys()]
        }


def create_default_postprocessing_pipeline(config: InferenceConfig, 
                                         class_names: Optional[List[str]] = None) -> PostprocessorPipeline:
    """Create a default postprocessing pipeline with common postprocessors."""
    pipeline = PostprocessorPipeline(config)
    
    # Add common postprocessors
    pipeline.add_postprocessor(OutputType.CLASSIFICATION, 
                              ClassificationPostprocessor(config, class_names))
    pipeline.add_postprocessor(OutputType.DETECTION, 
                              DetectionPostprocessor(config, class_names))
    pipeline.add_postprocessor(OutputType.SEGMENTATION, 
                              SegmentationPostprocessor(config))
    
    return pipeline

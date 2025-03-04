import logging
import threading
import queue
import time
import os
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
from pathlib import Path
import traceback

import torch
import torch.nn.functional as F
import cv2
import tensorrt as trt
import numpy as np
from cuda import cudart

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------
# Error Handling
# ---------------------------
class InferenceError(Exception):
    """Base class for inference errors."""
    pass

class ModelLoadError(InferenceError):
    """Error raised when model loading fails."""
    pass

class InputValidationError(InferenceError):
    """Error raised when input validation fails."""
    pass

class ProcessingError(InferenceError):
    """Error raised during processing."""
    pass

# ---------------------------
# Utility Functions
# ---------------------------
def validate_tensor(tensor: torch.Tensor, expected_shape: Optional[Tuple] = None,
                   min_dims: int = 1, max_dims: int = 6) -> None:
    """Validate tensor properties."""
    if not isinstance(tensor, torch.Tensor):
        raise InputValidationError(f"Expected torch.Tensor, got {type(tensor)}")
    
    if tensor.dim() < min_dims or tensor.dim() > max_dims:
        raise InputValidationError(
            f"Tensor has {tensor.dim()} dimensions, expected between {min_dims} and {max_dims}"
        )
    
    if expected_shape and any(a != b for a, b in zip(tensor.shape, expected_shape) 
                           if b is not None):
        raise InputValidationError(
            f"Tensor shape {tensor.shape} does not match expected shape {expected_shape}"
        )

@contextmanager
def cuda_error_handling():
    """Context manager for CUDA error handling."""
    try:
        yield
    except RuntimeError as e:
        if "CUDA" in str(e):
            logger.error(f"CUDA error: {e}")
            # Clear CUDA cache and reset device
            torch.cuda.empty_cache()
            current_device = torch.cuda.current_device()
            torch.cuda.device(current_device).empty_cache()
            # Reraise as our custom error
            raise ProcessingError(f"CUDA error occurred: {str(e)}")
        raise

def get_available_memory():
    """Get available GPU memory in bytes."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return torch.cuda.mem_get_info()[0]  # Free memory
    return 0

# ---------------------------
# Base and Postprocessor Classes
# ---------------------------
class BasePostprocessor:
    """
    Enhanced base postprocessor with validation and error handling.
    """
    def __init__(self):
        self.initialized = True
    
    def validate_input(self, outputs: Any) -> None:
        """Validate input to the postprocessor."""
        pass
    
    def __call__(self, outputs: Any) -> Any:
        """Process model outputs with validation."""
        try:
            self.validate_input(outputs)
            return self.process(outputs)
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
            logger.debug(traceback.format_exc())
            raise ProcessingError(f"Postprocessing error: {str(e)}")
    
    def process(self, outputs: Any) -> Any:
        """Main processing method to be implemented by subclasses."""
        return outputs

class ClassificationPostprocessor(BasePostprocessor):
    """
    Enhanced postprocessor for classification models with visualization capability,
    class mapping, and confidence normalization.
    """
    def __init__(
        self,
        top_k: int = 5,
        class_labels: Optional[List[str]] = None,
        class_label_map: Optional[Dict[int, str]] = None,
        apply_softmax: bool = True,
        multi_label: bool = False,
        threshold: float = 0.5,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
        enable_cache: bool = False,
        cache_size: int = 128
    ):
        super().__init__()
        self.top_k = max(1, top_k)  # Ensure top_k is at least 1
        
        # Class label handling
        self.class_labels = class_labels
        self.class_label_map = class_label_map or {}
        
        # Processing options
        self.apply_softmax = apply_softmax
        self.multi_label = multi_label
        self.threshold = max(0.0, min(1.0, threshold))  # Clamp between 0 and 1
        self.temperature = max(0.01, temperature)  # Prevent division by zero
        
        # Device management
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Result caching for repeated inputs (optional)
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.cache = {}
        self.cache_keys = []
    
    def validate_input(self, outputs: Any) -> None:
        """Validate the input tensor."""
        if not isinstance(outputs, torch.Tensor):
            raise InputValidationError(f"Expected torch.Tensor, got {type(outputs)}")
        
        if outputs.dim() != 2:
            raise InputValidationError(
                f"Expected 2D tensor [batch_size, num_classes], got shape {outputs.shape}"
            )
    
    def get_class_label(self, index: int) -> str:
        """Get class label from index using available label sources."""
        if self.class_labels and 0 <= index < len(self.class_labels):
            return self.class_labels[index]
        elif index in self.class_label_map:
            return self.class_label_map[index]
        return f"Class_{index}"
    
    def process(self, outputs: torch.Tensor) -> List[List[Tuple[str, float]]]:
        """Process classification outputs with optimized batch processing."""
        # Check cache first if enabled
        if self.enable_cache:
            cache_key = hash(outputs.cpu().numpy().tobytes())
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Move tensor to correct device once
        with cuda_error_handling():
            outputs = outputs.to(self.device)
            
            # Apply temperature scaling if needed
            if self.temperature != 1.0:
                outputs = outputs / self.temperature
            
            if self.apply_softmax:
                # Apply softmax in-place when possible
                if not outputs.requires_grad:
                    outputs = F.softmax(outputs, dim=1, inplace=True)
                else:
                    outputs = F.softmax(outputs, dim=1)
            
            results = []
            batch_size = outputs.size(0)
            
            if self.multi_label:
                # Multi-label classification (threshold-based)
                mask = outputs >= self.threshold
                
                for i in range(batch_size):
                    sample_mask = mask[i]
                    if not sample_mask.any():
                        # If no class exceeds threshold, take the highest one
                        max_idx = outputs[i].argmax().item()
                        result = [(self.get_class_label(max_idx), outputs[i, max_idx].item())]
                    else:
                        # Get indices and probabilities where mask is True
                        indices = torch.nonzero(sample_mask, as_tuple=True)[0]
                        probs = outputs[i, sample_mask]
                        
                        # Sort by probability (highest first)
                        sorted_idx = torch.argsort(probs, descending=True)
                        indices = indices[sorted_idx]
                        probs = probs[sorted_idx]
                        
                        # Create result list
                        result = [
                            (self.get_class_label(idx.item()), prob.item())
                            for idx, prob in zip(indices, probs)
                        ]
                    
                    results.append(result)
            else:
                # Single-label classification (top-k based)
                k = min(self.top_k, outputs.size(1))
                top_probs, top_idxs = torch.topk(outputs, k, dim=1)
                
                for i in range(batch_size):
                    result = [
                        (self.get_class_label(idx.item()), prob.item())
                        for idx, prob in zip(top_idxs[i], top_probs[i])
                    ]
                    results.append(result)
            
            # Update cache if enabled
            if self.enable_cache:
                self.cache[cache_key] = results
                self.cache_keys.append(cache_key)
                
                # Limit cache size
                if len(self.cache_keys) > self.cache_size:
                    old_key = self.cache_keys.pop(0)
                    self.cache.pop(old_key, None)
            
            return results
    
    def visualize(self, results: List[Tuple[str, float]], 
                 image: Optional[np.ndarray] = None,
                 max_display: int = 5,
                 font_scale: float = 0.6,
                 thickness: int = 1) -> np.ndarray:
        """
        Visualize classification results on an image.
        
        Args:
            results: Classification results from the postprocessor
            image: Optional image to visualize on (creates blank if None)
            max_display: Maximum number of classes to display
            font_scale: Font scale for text
            thickness: Line thickness
            
        Returns:
            Visualization image with classification results
        """
        # Create image if not provided
        if image is None:
            image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Make a copy to avoid modifying original
        vis_img = image.copy()
        
        # Get dimensions
        h, w = vis_img.shape[:2]
        padding = 10
        box_h = 30
        
        # Create semi-transparent overlay
        overlay = vis_img.copy()
        cv2.rectangle(overlay, (0, 0), (250, min(max_display+1, len(results))*box_h + 40), 
                     (20, 20, 20), -1)
        
        # Add title
        cv2.putText(overlay, "Classification Results", (padding, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale*1.2, (255, 255, 255), thickness+1)
        
        # Add results
        for i, (label, prob) in enumerate(results[:max_display]):
            y = 30 + (i+1)*box_h
            # Format probability as percentage
            prob_str = f"{prob*100:.1f}%"
            cv2.putText(overlay, f"{label}: {prob_str}", (padding, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Blend overlay with original image
        alpha = 0.7
        vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1-alpha, 0)
        
        return vis_img
    
    def serialize_results(self, results: List[List[Tuple[str, float]]]) -> Dict[str, Any]:
        """Convert results to a JSON-serializable format."""
        return {
            "classification_results": [
                [{"label": label, "probability": float(prob)} for label, prob in sample_result]
                for sample_result in results
            ],
            "multi_label": self.multi_label,
            "threshold": self.threshold if self.multi_label else None,
            "top_k": self.top_k if not self.multi_label else None
        }

class DetectionPostprocessor(BasePostprocessor):
    """
    Enhanced object detection postprocessor with visualization capability,
    class filtering, and improved NMS implementations.
    """
    def __init__(
        self,
        score_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        max_detections: int = 100,
        class_labels: Optional[List[str]] = None,
        class_map: Optional[Dict[int, str]] = None,
        filter_classes: Optional[Set[int]] = None,
        device: Optional[torch.device] = None,
        soft_nms: bool = False,
        soft_nms_sigma: float = 0.5
    ):
        super().__init__()
        self.score_threshold = max(0.0, min(1.0, score_threshold))
        self.nms_iou_threshold = max(0.0, min(1.0, nms_iou_threshold))
        self.max_detections = max(1, max_detections)
        
        # Class label management
        self.class_labels = class_labels
        self.class_map = class_map or {}
        self.filter_classes = filter_classes
        
        # Device management
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Advanced NMS options
        self.soft_nms = soft_nms
        self.soft_nms_sigma = soft_nms_sigma
        
        # Check for optimized NMS implementation
        self.use_torchvision_nms = self._check_torchvision_nms()
    
    def _check_torchvision_nms(self) -> bool:
        """Check if torchvision's CUDA NMS implementation is available."""
        try:
            if torch.cuda.is_available() and hasattr(torch.ops, 'torchvision') and hasattr(torch.ops.torchvision, 'nms'):
                return True
        except (ImportError, AttributeError):
            pass
        logger.info("Torchvision CUDA NMS not available, using custom NMS implementation")
        return False
    
    def validate_input(self, outputs: Any) -> None:
        """Validate detection outputs."""
        if not isinstance(outputs, list):
            raise InputValidationError(f"Expected list of dicts, got {type(outputs)}")
        
        for i, item in enumerate(outputs):
            if not isinstance(item, dict):
                raise InputValidationError(f"Expected dict at index {i}, got {type(item)}")
            
            required_keys = ["boxes", "scores", "labels"]
            for key in required_keys:
                if key not in item:
                    raise InputValidationError(f"Missing required key '{key}' in detection output")
    
    def get_class_label(self, class_id: int) -> str:
        """Get class label from class ID."""
        if self.class_labels and 0 <= class_id < len(self.class_labels):
            return self.class_labels[class_id]
        elif class_id in self.class_map:
            return self.class_map[class_id]
        return f"Class_{class_id}"
    
    def _box_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between boxes. Boxes in [x1, y1, x2, y2] format."""
        # Calculate box areas
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        # Get coordinates of intersection
        lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
        
        # Calculate intersection area (handle non-overlapping case)
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        
        # Calculate IoU
        union = area1[:, None] + area2 - inter
        iou = inter / union.clamp(min=1e-6)
        
        return iou
    
    def _soft_nms(self, boxes: torch.Tensor, scores: torch.Tensor, sigma: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Soft-NMS implementation that decays scores based on IoU overlap.
        Returns indices, updated scores, and weights.
        """
        N = boxes.shape[0]
        indices = torch.arange(N, device=boxes.device)
        
        if N == 0:
            return indices, scores, torch.ones_like(scores)
        
        # Sort boxes by score
        _, order = scores.sort(descending=True)
        
        keep_indices = []
        keep_scores = []
        
        # Initialize weights for all boxes
        weights = torch.ones_like(scores)
        
        for i in range(N):
            idx = order[i].item()
            
            # If below threshold after previous decays, skip
            if scores[idx] < self.score_threshold:
                continue
            
            keep_indices.append(idx)
            keep_scores.append(scores[idx].item())
            
            # Get IoU of the current box with remaining boxes
            remaining = order[i+1:]
            if len(remaining) == 0:
                break
                
            # Calculate IoU between current box and remaining boxes
            current_box = boxes[idx:idx+1]
            other_boxes = boxes[remaining]
            ious = self._box_iou(current_box, other_boxes).squeeze(0)
            
            # Apply Gaussian penalty to scores based on IoU
            decay = torch.exp(-(ious * ious) / sigma)
            
            # Update scores of remaining boxes
            scores[remaining] *= decay
            weights[remaining] *= decay
        
        return (
            torch.tensor(keep_indices, device=boxes.device),
            torch.tensor(keep_scores, device=boxes.device),
            weights[keep_indices]
        )
    
    def _hard_nms(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Standard NMS implementation."""
        if len(boxes) == 0:
            return torch.zeros(0, dtype=torch.int64, device=boxes.device)
        
        # Use torchvision NMS if available
        if self.use_torchvision_nms:
            return torch.ops.torchvision.nms(boxes, scores, self.nms_iou_threshold)
        
        # Otherwise use our custom implementation
        _, order = scores.sort(descending=True)
        keep = []
        
        while order.numel() > 0:
            if len(keep) >= self.max_detections:
                break
                
            # Pick the box with highest score
            i = order[0].item()
            keep.append(i)
            
            # If only one box left, finish
            if order.numel() == 1:
                break
                
            # Get IoU of remaining boxes with the selected box
            remaining = order[1:]
            current_box = boxes[i:i+1]
            other_boxes = boxes[remaining]
            ious = self._box_iou(current_box, other_boxes).squeeze(0)
            
            # Keep boxes with IoU less than threshold
            mask = ious <= self.nms_iou_threshold
            order = remaining[mask]
        
        return torch.tensor(keep, dtype=torch.int64, device=boxes.device)
    
    def process(self, outputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, Any]]:
        """Process detection outputs with improved NMS and class filtering."""
        processed_results = []
        
        with cuda_error_handling():
            for per_image_output in outputs:
                boxes = per_image_output["boxes"].to(self.device)
                scores = per_image_output["scores"].to(self.device)
                labels = per_image_output["labels"].to(self.device)
                
                # Initial confidence filtering
                conf_mask = scores >= self.score_threshold
                
                # Class filtering if specified
                if self.filter_classes is not None:
                    class_mask = torch.zeros_like(conf_mask, dtype=torch.bool)
                    for class_id in self.filter_classes:
                        class_mask = class_mask | (labels == class_id)
                    conf_mask = conf_mask & class_mask
                
                # Apply filters
                boxes = boxes[conf_mask]
                scores = scores[conf_mask]
                labels = labels[conf_mask]
                
                result = {"boxes": [], "scores": [], "labels": [], "class_names": []}
                
                # Process each class separately for better NMS
                unique_labels = labels.unique()
                for class_id in unique_labels:
                    class_mask = labels == class_id
                    class_boxes = boxes[class_mask]
                    class_scores = scores[class_mask]
                    
                    # Skip if no boxes for this class
                    if len(class_boxes) == 0:
                        continue
                    
                    # Apply NMS
                    if self.soft_nms:
                        keep_idx, updated_scores, weights = self._soft_nms(
                            class_boxes, class_scores, self.soft_nms_sigma)
                        # Use updated scores
                        keep_boxes = class_boxes[keep_idx]
                        keep_scores = updated_scores
                        keep_labels = torch.full_like(keep_idx, class_id)
                    else:
                        keep_idx = self._hard_nms(class_boxes, class_scores)
                        # Limit by max detections
                        if len(keep_idx) > self.max_detections:
                            keep_idx = keep_idx[:self.max_detections]
                        keep_boxes = class_boxes[keep_idx]
                        keep_scores = class_scores[keep_idx]
                        keep_labels = torch.full_like(keep_idx, class_id)
                    
                    # Add to results
                    result["boxes"].extend(keep_boxes.cpu().tolist())
                    result["scores"].extend(keep_scores.cpu().tolist())
                    result["labels"].extend(keep_labels.cpu().tolist())
                    result["class_names"].extend(
                        [self.get_class_label(class_id.item())] * len(keep_idx)
                    )
                
                # If we have more than max_detections after combining all classes,
                # keep only the highest scoring ones
                if len(result["scores"]) > self.max_detections:
                    # Sort by score
                    combined_scores = torch.tensor(result["scores"])
                    _, idx = combined_scores.sort(descending=True)
                    idx = idx[:self.max_detections].tolist()
                    
                    # Filter results
                    result["boxes"] = [result["boxes"][i] for i in idx]
                    result["scores"] = [result["scores"][i] for i in idx]
                    result["labels"] = [result["labels"][i] for i in idx]
                    result["class_names"] = [result["class_names"][i] for i in idx]
                
                processed_results.append(result)
        
        return processed_results
    
    def visualize(self, image: np.ndarray, detections: Dict[str, List],
                 line_thickness: int = 2, font_scale: float = 0.6,
                 draw_scores: bool = True) -> np.ndarray:
        """
        Visualize detection results on an image.
        
        Args:
            image: The image to visualize on
            detections: Detection results from the postprocessor
            line_thickness: Line thickness for bounding boxes
            font_scale: Font scale for text
            draw_scores: Whether to draw confidence scores
            
        Returns:
            Image with visualization
        """
        vis_img = image.copy()
        
        # Generate colors for classes (consistent colors for same classes)
        classes = list(set(detections["labels"]))
        colors = {}
        for cls_id in classes:
            # Generate deterministic color based on class id
            hue = (cls_id * 0.1) % 1.0
            rgb = cv2.cvtColor(np.uint8([[[hue * 180, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
            colors[cls_id] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        
        # Draw boxes
        for box, score, label, class_name in zip(
            detections["boxes"], detections["scores"], 
            detections["labels"], detections["class_names"]
        ):
            # Get coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Get color for this class
            color = colors.get(label, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, line_thickness)
            
            # Prepare label text
            label_text = class_name
            if draw_scores:
                label_text += f" {score:.2f}"
            
            # Calculate text size for background rectangle
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)
            
            # Draw label background
            cv2.rectangle(
                vis_img, 
                (x1, y1 - text_size[1] - 5), 
                (x1 + text_size[0], y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                vis_img, 
                label_text, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (255, 255, 255), 
                line_thickness//2
            )
        
        return vis_img
    
    def serialize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert results to a JSON-serializable format."""
        serialized = []
        for img_result in results:
            detections = []
            for i in range(len(img_result["boxes"])):
                detections.append({
                    "bbox": img_result["boxes"][i],
                    "score": float(img_result["scores"][i]),
                    "class_id": int(img_result["labels"][i]),
                    "class_name": img_result["class_names"][i]
                })
            serialized.append({"detections": detections})
        
        return {
            "detection_results": serialized,
            "threshold": self.score_threshold,
            "nms_threshold": self.nms_iou_threshold
        }

class SegmentationPostprocessor(BasePostprocessor):
    """
    Enhanced segmentation postprocessor with support for instance segmentation,
    boundary extraction, and visualization.
    """
    def __init__(
        self,
        threshold: float = 0.5,
        min_contour_area: int = 100,
        device: Optional[torch.device] = None,
        multi_class: bool = False,
        class_labels: Optional[List[str]] = None,
        refinement_iterations: int = 0,
        morphology_size: int = 3
    ):
        super().__init__()
        self.threshold = max(0.0, min(1.0, threshold))
        self.min_contour_area = max(0, min_contour_area)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_class = multi_class
        self.class_labels = class_labels
        self.refinement_iterations = refinement_iterations
        self.morphology_size = morphology_size
        
        # Create structuring element for morphology operations
        self.kernel = np.ones((morphology_size, morphology_size), np.uint8)
    
    def validate_input(self, outputs: Any) -> None:
        """Validate segmentation outputs."""
        if not isinstance(outputs, torch.Tensor):
            raise InputValidationError(f"Expected torch.Tensor, got {type(outputs)}")
        
        if self.multi_class:
            # Multi-class: [batch, num_classes, height, width]
            if outputs.dim() != 4:
                raise InputValidationError(
                    f"Expected 4D tensor for multi-class, got shape {outputs.shape}"
                )
        else:
            # Binary: [batch, 1, height, width]
            if outputs.dim() not in (3, 4):
                raise InputValidationError(
                    f"Expected 3D or 4D tensor for binary, got shape {outputs.shape}"
                )
    
    def get_class_label(self, class_id: int) -> str:
        """Get class label from class ID."""
        if self.class_labels and 0 <= class_id < len(self.class_labels):
            return self.class_labels[class_id]
        return f"Class_{class_id}"
    
    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to refine the mask."""
        if self.refinement_iterations <= 0:
            return mask
        
        refined_mask = mask.copy()
        for _ in range(self.refinement_iterations):
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, self.kernel)
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, self.kernel)
        
        return refined_mask
    
    def _extract_instances(self, binary_mask: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Extract instance information from binary mask."""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # Instance color map for visualization
        instance_map = np.zeros_like(binary_mask, dtype=np.uint8)
        
        # Skip first label which is background (label 0)
        instances = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Skip small regions
            if area < self.min_contour_area:
                continue
                
            # Extract instance mask and properties
            instance_mask = (labels == i).astype(np.uint8)
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]
            
            # Find contours for boundary representation
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Only include largest contour
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate additional metrics
                perimeter = cv2.arcLength(largest_contour, True)
                circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
                
                instances.append({
                    "id": i,
                    "area": int(area),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "centroid": (float(cx), float(cy)),
                    "contour": largest_contour.tolist(),
                    "perimeter": float(perimeter),
                    "circularity": float(circularity)
                })
                
                # Add to instance visualization map with unique color
                instance_map[instance_mask > 0] = i
        
        return instance_map, instances
    
    def process(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """
        Process segmentation outputs with support for multi-class segmentation.
        Returns masks, instance information, and boundaries.
        """
        with cuda_error_handling():
            # Move to correct device
            outputs = outputs.to(self.device)
            
            # Handle different input formats
            if outputs.dim() == 4:  # [batch, channels, height, width]
                if outputs.size(0) > 1:
                    logger.warning("Batch size > 1, processing only first item in batch")
                
                # Extract first item in batch
                outputs = outputs[0]
            
            if self.multi_class:
                # For multi-class, channels represent different classes
                num_classes = outputs.size(0)
                height, width = outputs.size(1), outputs.size(2)
                
                # Apply softmax across class dimension
                if num_classes > 1:
                    probs = F.softmax(outputs, dim=0)
                else:
                    probs = outputs
                
                # Get predicted class for each pixel (argmax across classes)
                class_mask = torch.argmax(probs, dim=0).cpu().numpy()
                
                # Init result containers
                all_binary_masks = {}
                all_instances = {}
                all_instance_maps = {}
                
                # Process each class separately
                for class_id in range(num_classes):
                    # Skip background class (0) if more than one class
                    if class_id == 0 and num_classes > 1:
                        continue
                        
                    # Create binary mask for this class
                    binary_mask = (class_mask == class_id).astype(np.uint8) * 255
                    
                    # Refine mask
                    binary_mask = self._refine_mask(binary_mask)
                    
                    # Extract instances
                    instance_map, instances = self._extract_instances(binary_mask)
                    
                    if instances:  # Only add if we have instances
                        class_name = self.get_class_label(class_id)
                        all_binary_masks[class_id] = binary_mask
                        all_instances[class_id] = instances
                        all_instance_maps[class_id] = instance_map
                
                result = {
                    "class_masks": all_binary_masks,
                    "instances": all_instances,
                    "instance_maps": all_instance_maps,
                    "class_names": {class_id: self.get_class_label(class_id) 
                                   for class_id in all_binary_masks.keys()},
                    "multi_class": True
                }
                
            else:
                # Handle binary segmentation case
                if outputs.dim() == 3:
                    outputs = outputs.unsqueeze(0)  # Add channel dimension
                
                # Ensure single channel output
                outputs = outputs.squeeze(0)
                
                # Apply threshold
                mask_np = outputs.cpu().numpy()
                binary_mask = (mask_np > self.threshold).astype(np.uint8) * 255
                
                # Refine mask
                binary_mask = self._refine_mask(binary_mask)
                
                # Extract instances
                instance_map, instances = self._extract_instances(binary_mask)
                
                result = {
                    "mask": binary_mask,
                    "instances": instances,
                    "instance_map": instance_map,
                    "multi_class": False
                }
        
        return result
    
    def visualize(self, image: np.ndarray, result: Dict[str, Any],
                 alpha: float = 0.5, draw_contours: bool = True,
                 draw_centroids: bool = True) -> np.ndarray:
        """
        Visualize segmentation results on an image.
        
        Args:
            image: The image to visualize on
            result: Segmentation results from the postprocessor
            alpha: Opacity of mask overlay
            draw_contours: Whether to draw instance contours
            draw_centroids: Whether to draw instance centroids
            
        Returns:
            Image with visualization
        """
        vis_img = image.copy()
        
        if result["multi_class"]:
            # Create a colored overlay for multi-class segmentation
            overlay = np.zeros_like(vis_img)
            
            # Process each class
            for class_id, mask in result["class_masks"].items():
                class_name = result["class_names"][class_id]
                
                # Generate a deterministic color for this class
                hue = (class_id * 30) % 180
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
                color = (int(color[0]), int(color[1]), int(color[2]))
                
                # Apply color to mask areas in overlay
                mask_bool = mask > 0
                overlay[mask_bool] = color
                
                # Draw contours if requested
                if draw_contours and class_id in result["instances"]:
                    for instance in result["instances"][class_id]:
                        contour = np.array(instance["contour"], dtype=np.int32)
                        cv2.drawContours(vis_img, [contour], 0, color, 2)
                        
                        if draw_centroids:
                            cx, cy = map(int, instance["centroid"])
                            cv2.circle(vis_img, (cx, cy), 4, color, -1)
                            cv2.putText(vis_img, class_name, (cx + 5, cy), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Blend overlay with original image
            vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1-alpha, 0)
            
        else:
            # Binary segmentation
            mask = result["mask"]
            
            # Create colored overlay for binary mask
            overlay = np.zeros_like(vis_img)
            overlay[mask > 0] = (0, 255, 0)  # Green for binary mask
            
            # Blend overlay with original image
            vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1-alpha, 0)
            
            # Draw contours if requested
            if draw_contours and "instances" in result:
                for instance in result["instances"]:
                    contour = np.array(instance["contour"], dtype=np.int32)
                    cv2.drawContours(vis_img, [contour], 0, (0, 0, 255), 2)
                    
                    if draw_centroids:
                        cx, cy = map(int, instance["centroid"])
                        cv2.circle(vis_img, (cx, cy), 4, (255, 0, 0), -1)
        
        return vis_img
    
    def serialize_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert results to a JSON-serializable format."""
        serialized = {
            "threshold": self.threshold,
            "min_contour_area": self.min_contour_area,
            "multi_class": result["multi_class"]
        }
        
        if result["multi_class"]:
            # Convert masks to base64 or summarize
            serialized["classes"] = []
            for class_id, instances in result["instances"].items():
                class_info = {
                    "class_id": int(class_id),
                    "class_name": result["class_names"][class_id],
                    "instance_count": len(instances),
                    "instances": []
                }
                
                # Include non-contour instance information
                for instance in instances:
                    # Copy instance but exclude the large contour array
                    instance_data = {
                        k: v for k, v in instance.items() if k != "contour"
                    }
                    # Add simplified contour for API responses (fewer points)
                    if "contour" in instance:
                        contour = np.array(instance["contour"], dtype=np.int32)
                        # Simplify contour to reduce size
                        epsilon = 0.005 * cv2.arcLength(contour, True)
                        simplified = cv2.approxPolyDP(contour, epsilon, True)
                        instance_data["simplified_contour"] = simplified.tolist()
                    
                    class_info["instances"].append(instance_data)
                
                serialized["classes"].append(class_info)
        else:
            # Handle binary case
            instance_count = len(result["instances"])
            serialized["instance_count"] = instance_count
            
            # Include simplified instances
            serialized["instances"] = []
            for instance in result["instances"]:
                # Copy instance but exclude the large contour array
                instance_data = {
                    k: v for k, v in instance.items() if k != "contour"
                }
                
                # Add simplified contour
                if "contour" in instance:
                    contour = np.array(instance["contour"], dtype=np.int32)
                    # Simplify contour to reduce size
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    simplified = cv2.approxPolyDP(contour, epsilon, True)
                    instance_data["simplified_contour"] = simplified.tolist()
                
                serialized["instances"].append(instance_data)
        
        return serialized

# ---------------------------
# TensorRT Inference Engine Data Structures
# ---------------------------
@dataclass
class InferenceRequest:
    """Data structure for holding inference requests."""
    id: str  # Unique request ID
    input: Union[torch.Tensor, np.ndarray]
    callback: Callable[[Dict[str, Any]], None]
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher numbers = higher priority
    timestamp: float = field(default_factory=time.time)

class PriorityQueue(queue.PriorityQueue):
    """Priority queue for inference requests based on priority and timestamp."""
    def put(self, item: InferenceRequest) -> None:
        # Convert to tuple with negative priority (for max-first) and timestamp
        entry = (-item.priority, item.timestamp, item)
        super().put(entry)
    
    def get(self, *args, **kwargs) -> InferenceRequest:
        _, _, item = super().get(*args, **kwargs)
        return item

class TRTOutputBinding:
    """Manages TensorRT output bindings with proper memory management."""
    def __init__(self, name: str, index: int, shape: Tuple[int, ...], dtype: trt.DataType):
        self.name = name
        self.index = index
        self.shape = shape
        self.dtype = dtype
        self.tensor = None
        
        # Map TensorRT data types to PyTorch data types
        dtype_map = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT8: torch.int8,
            trt.DataType.INT32: torch.int32,
            trt.DataType.BOOL: torch.bool
        }
        self.torch_dtype = dtype_map.get(dtype, torch.float32)
    
    def allocate(self, batch_size: int) -> torch.Tensor:
        """Allocate output tensor with the right shape and data type."""
        # Use first dim as batch size if shape[0] is -1 (dynamic)
        shape = list(self.shape)
        if shape[0] == -1 or shape[0] == trt.OptProfileSelector.MAX:
            shape[0] = batch_size
            
        # Create tensor on CUDA device
        self.tensor = torch.empty(shape, dtype=self.torch_dtype, device='cuda')
        return self.tensor
    
    def deallocate(self) -> None:
        """Free CUDA memory by removing the tensor reference."""
        self.tensor = None

# ---------------------------
# Optimized TensorRT Inference Engine
# ---------------------------
class TensorRTInferenceEngine:
    """
    Highly optimized TensorRT inference engine with enhanced features:
    - Dynamic batching with priorities
    - Autoscaling based on GPU memory
    - Multiple inference streams
    - Health monitoring with error recovery
    - Profiling and metrics collection
    """
    def __init__(
        self,
        engine_path: str,
        max_batch_size: int = 32,
        min_batch_size: int = 1,
        timeout: float = 0.05,
        dynamic_batching: bool = True,
        num_streams: int = 1,
        auto_optimize: bool = True,
        enable_profiling: bool = False,
        warmup_iterations: int = 3
    ):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.engine_path = engine_path
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.timeout = timeout
        self.dynamic_batching = dynamic_batching
        self.num_streams = max(1, min(num_streams, 4))  # Limit reasonable number of streams
        self.auto_optimize = auto_optimize
        self.enable_profiling = enable_profiling
        self.warmup_iterations = warmup_iterations
        
        # Operational state
        self.running = False
        self.health_status = {"status": "initializing", "last_error": None, "error_count": 0}
        
        # Request queues and synchronization
        self.request_queue = PriorityQueue()
        self.lock = threading.Lock()
        
        # Streams, events, and threads
        self.streams = []
        self.events = []
        self.worker_threads = []
        
        # Performance metrics
        self.metrics = {
            "inference_count": 0,
            "batch_count": 0,
            "total_latency": 0,
            "batch_sizes": [],
            "error_count": 0,
            "profiling_data": {},
            "start_time": time.time()
        }
        
        # Initialize engine and resources
        try:
            self._initialize_engine()
            self._initialize_streams()
            
            # Start processing threads
            self.running = True
            for i in range(self.num_streams):
                thread = threading.Thread(
                    target=self._process_requests, 
                    args=(i,),
                    daemon=True
                )
                self.worker_threads.append(thread)
                thread.start()
            
            # Start health monitoring thread
            self.health_thread = threading.Thread(
                target=self._monitor_health, 
                daemon=True
            )
            self.health_thread.start()
            
            # Perform warmup
            if self.warmup_iterations > 0:
                self._warmup()
                
            self.health_status["status"] = "ready"
            logger.info(f"TensorRT inference engine initialized with {self.num_streams} streams")
            
        except Exception as e:
            logger.error(f"Failed to initialize TensorRT engine: {e}")
            logger.debug(traceback.format_exc())
            self.health_status["status"] = "error"
            self.health_status["last_error"] = str(e)
            self.health_status["error_count"] += 1
            raise ModelLoadError(f"Failed to initialize TensorRT engine: {e}")
    
    def _initialize_engine(self) -> None:
        """Initialize TensorRT engine and execution context."""
        # Check if engine file exists
        if not os.path.exists(self.engine_path):
            raise ModelLoadError(f"Engine file not found: {self.engine_path}")
        
        try:
            # Load engine from file
            with open(self.engine_path, "rb") as f:
                engine_data = f.read()
                
            with trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(engine_data)
                
            if self.engine is None:
                raise ModelLoadError("Failed to deserialize TensorRT engine")
                
            # Configure profiling if enabled
            if self.enable_profiling:
                self.trt_profiler = trt.Profiler()
            else:
                self.trt_profiler = None
                
            # Create execution contexts
            self.contexts = []
            for _ in range(self.num_streams):
                context = self.engine.create_execution_context()
                if context is None:
                    raise ModelLoadError("Failed to create TensorRT execution context")
                self.contexts.append(context)
                
            # Analyze engine bindings
            self._analyze_bindings()
            
        except Exception as e:
            logger.error(f"Error in engine initialization: {e}")
            logger.debug(traceback.format_exc())
            raise ModelLoadError(f"Failed to initialize TensorRT engine: {e}")
    
    def _analyze_bindings(self) -> None:
        """Analyze engine bindings to extract input/output information."""
        self.binding_names = []
        self.input_bindings = {}
        self.output_bindings = {}
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            self.binding_names.append(name)
            
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            
            if self.engine.binding_is_input(i):
                self.input_bindings[name] = {
                    "index": i,
                    "shape": shape,
                    "dtype": dtype
                }
                logger.info(f"Input binding: {name}, shape: {shape}, dtype: {dtype}")
            else:
                self.output_bindings[name] = TRTOutputBinding(name, i, shape, dtype)
                logger.info(f"Output binding: {name}, shape: {shape}, dtype: {dtype}")
                
        if not self.input_bindings:
            raise ModelLoadError("No input bindings found in the engine")
            
        if not self.output_bindings:
            raise ModelLoadError("No output bindings found in the engine")
    
    def _initialize_streams(self) -> None:
        """Initialize CUDA streams and events for asynchronous processing."""
        try:
            for _ in range(self.num_streams):
                # Create CUDA stream
                stream = torch.cuda.Stream()
                self.streams.append(stream)
                
                # Create CUDA event for synchronization
                start_event = torch.cuda.Event(enable_timing=self.enable_profiling)
                end_event = torch.cuda.Event(enable_timing=self.enable_profiling)
                self.events.append((start_event, end_event))
                
            logger.info(f"Created {self.num_streams} CUDA streams for inference")
            
        except Exception as e:
            logger.error(f"Error initializing CUDA streams: {e}")
            raise
    
    def _warmup(self) -> None:
        """Warm up the engine to initialize and optimize performance."""
        logger.info(f"Warming up TensorRT engine with {self.warmup_iterations} iterations")
        try:
            # Create sample input
            for name, binding in self.input_bindings.items():
                # Determine input shape for warmup
                shape = list(binding["shape"])
                
                # Replace dynamic dimensions with actual sizes
                for i, dim in enumerate(shape):
                    if dim == -1:
                        if i == 0:
                            # Batch dimension
                            shape[i] = self.min_batch_size
                        else:
                            # Other dynamic dimensions, use a reasonable size
                            shape[i] = 16
                
                # Create dummy input tensor
                dummy_input = torch.zeros(shape, device="cuda")
                
                # Run warmup iterations
                for i in range(self.warmup_iterations):
                    event = threading.Event()
                    results = {}
                    
                    def callback(output):
                        nonlocal results
                        results = output
                        event.set()
                    
                    # Submit inference request
                    self.infer(
                        dummy_input,
                        callback,
                        {"warmup": True, "iteration": i}
                    )
                    
                    # Wait for completion with timeout
                    if not event.wait(timeout=5.0):
                        logger.warning(f"Warmup iteration {i} timed out")
                    
                logger.info("Engine warmup completed successfully")
                break  # Just use the first input binding
                
        except Exception as e:
            logger.error(f"Error during engine warmup: {e}")
            self.health_status["last_error"] = str(e)
    
    def _monitor_health(self) -> None:
        """Monitor engine health and perform recovery if needed."""
        error_threshold = 5  # Number of consecutive errors before recovery
        check_interval = 10  # Seconds between health checks
        
        while self.running:
            try:
                time.sleep(check_interval)
                
                # Check error count
                if self.health_status["error_count"] >= error_threshold:
                    logger.warning(f"Error threshold reached ({error_threshold} errors), attempting recovery")
                    self._attempt_recovery()
                
                # Check GPU memory status if auto-optimize is enabled
                if self.auto_optimize:
                    free_memory = get_available_memory()
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    usage_ratio = 1.0 - (free_memory / total_memory)
                    
                    # Log if memory usage is high
                    if usage_ratio > 0.9:
                        logger.warning(f"High GPU memory usage: {usage_ratio:.1%}")
                        
                        # Force CUDA cache clearing
                        torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
    
    def _attempt_recovery(self) -> None:
        """Attempt to recover from error state."""
        logger.info("Attempting inference engine recovery")
        
        try:
            # Reset error count
            self.health_status["error_count"] = 0
            self.health_status["status"] = "recovering"
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Re-create contexts if needed
            with self.lock:
                for i, context in enumerate(self.contexts):
                    try:
                        # Test if context is valid
                        binding_idx = next(iter(self.input_bindings.values()))["index"]
                        shape = context.get_binding_shape(binding_idx)
                        
                        # If we got this far, context is probably okay
                        logger.debug(f"Context {i} appears valid, binding shape: {shape}")
                        
                    except Exception:
                        logger.warning(f"Re-creating execution context {i}")
                        # Try to create a new context
                        self.contexts[i] = self.engine.create_execution_context()
            
            # Update status
            self.health_status["status"] = "ready"
            logger.info("Recovery completed successfully")
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            self.health_status["status"] = "error"
            self.health_status["last_error"] = str(e)
    
    def _process_requests(self, stream_idx: int) -> None:
        """Process requests from the queue using assigned stream."""
        context = self.contexts[stream_idx]
        stream = self.streams[stream_idx]
        start_event, end_event = self.events[stream_idx]
        
        logger.info(f"Worker thread {stream_idx} started")
        
        while self.running:
            try:
                # Collect batch of requests
                batch = self._collect_batch(stream_idx)
                
                # Skip if no requests
                if not batch:
                    continue
                
                # Process batch
                self._infer_batch(batch, context, stream, start_event, end_event)
                
            except queue.Empty:
                # No requests, just continue
                continue
                
            except Exception as e:
                logger.error(f"Error in worker thread {stream_idx}: {e}")
                logger.debug(traceback.format_exc())
                
                # Update health status
                with self.lock:
                    self.health_status["last_error"] = str(e)
                    self.health_status["error_count"] += 1
                    self.metrics["error_count"] += 1
                
                # Notify clients about the error
                for req in batch:
                    try:
                        req.callback({"error": str(e), "request_id": req.id})
                    except Exception as cb_err:
                        logger.error(f"Error in callback: {cb_err}")
    
    def _collect_batch(self, stream_idx: int) -> List[InferenceRequest]:
        """Collect a batch of requests from the queue."""
        if not self.dynamic_batching:
            # In non-dynamic mode, wait for exactly max_batch_size or timeout
            reqs = []
            try:
                timeout = self.timeout
                # Get first request (blocking)
                first_req = self.request_queue.get(timeout=timeout)
                reqs.append(first_req)
                
                # Try to get more until max_batch_size (non-blocking)
                remaining = self.max_batch_size - 1
                while remaining > 0:
                    try:
                        req = self.request_queue.get_nowait()
                        reqs.append(req)
                        remaining -= 1
                    except queue.Empty:
                        break
                        
                return reqs
                
            except queue.Empty:
                return []
        
        # Dynamic batching mode
        reqs = []
        max_wait = self.timeout
        
        try:
            # Wait for first request (blocking with timeout)
            start_time = time.time()
            first_req = self.request_queue.get(timeout=max_wait)
            reqs.append(first_req)
            
            # How long we waited for the first request
            waited = time.time() - start_time
            
            # Shorter timeout for collecting batch
            batch_collect_timeout = min(0.001, max(0, max_wait - waited))
            
            # Try to collect more requests for the batch, up to max_batch_size
            while len(reqs) < self.max_batch_size:
                try:
                    # Shorter timeout for subsequent requests
                    req = self.request_queue.get(timeout=batch_collect_timeout)
                    reqs.append(req)
                except queue.Empty:
                    break
            
            return reqs
            
        except queue.Empty:
            return []
    
    def _prepare_inputs(self, batch: List[InferenceRequest]) -> Dict[str, torch.Tensor]:
        """Prepare input tensors for inference."""
        inputs = {}
        batch_size = len(batch)
        
        for name, binding in self.input_bindings.items():
            first_input = batch[0].input
            input_is_tensor = isinstance(first_input, torch.Tensor)
            
            if input_is_tensor:
                # Create tensor list from batch
                input_list = []
                for req in batch:
                    tensor = req.input
                    if not isinstance(tensor, torch.Tensor):
                        tensor = torch.tensor(tensor)
                    
                    # Move to GPU if not already there
                    if tensor.device.type != "cuda":
                        tensor = tensor.to("cuda")
                    
                    input_list.append(tensor)
                
                # Stack tensors into batch
                try:
                    stacked_input = torch.stack(input_list)
                    inputs[name] = stacked_input
                    
                except Exception as e:
                    # Handle case where tensors have different shapes
                    logger.error(f"Error stacking tensors: {e}")
                    raise InputValidationError(f"Cannot batch tensors of different shapes: {e}")
                
            else:
                # Handle numpy array input
                try:
                    if isinstance(first_input, np.ndarray):
                        # Stack numpy arrays and convert to tensor
                        np_batch = np.stack([req.input for req in batch])
                        inputs[name] = torch.from_numpy(np_batch).to("cuda")
                    else:
                        raise InputValidationError(f"Unsupported input type: {type(first_input)}")
                except Exception as e:
                    logger.error(f"Error processing inputs: {e}")
                    raise InputValidationError(f"Failed to prepare inputs: {e}")
        
        return inputs
    
    def _prepare_outputs(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Prepare output tensors for inference results."""
        outputs = {}
        
        for name, binding in self.output_bindings.items():
            # Allocate tensor for output
            outputs[name] = binding.allocate(batch_size)
            
        return outputs
    
    def _infer_batch(self, batch: List[InferenceRequest], context, stream, start_event, end_event) -> None:
        """Execute inference on a batch of inputs."""
        batch_size = len(batch)
        
        # Skip empty batches
        if batch_size == 0:
            return
        
        # Record batch size for metrics
        with self.lock:
            self.metrics["batch_sizes"].append(batch_size)
        
        # Record start time for profiling
        # Record start time for profiling
        batch_start_time = time.time()
        
        try:
            with torch.cuda.stream(stream):
                # Record CUDA events if profiling
                if self.enable_profiling:
                    start_event.record()
                
                # Prepare inputs and outputs
                inputs = self._prepare_inputs(batch)
                outputs = self._prepare_outputs(batch_size)
                
                # Set input shapes in execution context
                for name, tensor in inputs.items():
                    binding_idx = self.input_bindings[name]["index"]
                    if context.get_binding_shape(binding_idx)[0] != tensor.shape[0]:
                        context.set_binding_shape(binding_idx, tensor.shape)
                
                # Prepare bindings list in correct order
                bindings = []
                for i in range(self.engine.num_bindings):
                    if self.engine.binding_is_input(i):
                        # Find the corresponding input tensor
                        name = self.engine.get_binding_name(i)
                        bindings.append(inputs[name].data_ptr())
                    else:
                        # Find the corresponding output tensor
                        name = self.engine.get_binding_name(i)
                        bindings.append(outputs[name].data_ptr())
                
                # Execute inference
                if not context.execute_async_v2(
                    bindings=bindings,
                    stream_handle=stream.cuda_stream
                ):
                    raise RuntimeError("TensorRT execution failed")
                
                # Record end event if profiling
                if self.enable_profiling:
                    end_event.record()
                
                # Process results and send to callbacks
                self._process_batch_results(batch, outputs, stream)
                
                # Update metrics
                with self.lock:
                    self.metrics["inference_count"] += batch_size
                    self.metrics["batch_count"] += 1
                    
                    if self.enable_profiling:
                        # Calculate latency
                        stream.synchronize()
                        latency = start_event.elapsed_time(end_event)
                        self.metrics["total_latency"] += latency
                        
                        # Store detailed profiling data
                        per_sample_latency = latency / batch_size
                        self.metrics["profiling_data"].setdefault("latencies", []).append({
                            "batch_size": batch_size, 
                            "latency_ms": latency,
                            "per_sample_ms": per_sample_latency,
                            "timestamp": time.time()
                        })
                
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            logger.debug(traceback.format_exc())
            
            # Update error metrics
            with self.lock:
                self.health_status["last_error"] = str(e)
                self.health_status["error_count"] += 1
                self.metrics["error_count"] += 1
            
            # Notify callbacks about error
            for req in batch:
                try:
                    req.callback({
                        "error": str(e),
                        "request_id": req.id,
                        "timestamp": time.time()
                    })
                except Exception as cb_err:
                    logger.error(f"Error in callback: {cb_err}")
    
    def _process_batch_results(self, batch: List[InferenceRequest], outputs: Dict[str, torch.Tensor], stream) -> None:
        """Process inference results and deliver to callbacks."""
        # Wait for the stream to complete
        stream.synchronize()
        
        for i, req in enumerate(batch):
            try:
                # Extract individual result from the batch for each output
                result = {
                    name: tensor[i].cpu().clone() for name, tensor in outputs.items()
                }
                
                # Add metadata to result
                result["request_id"] = req.id
                result["timestamp"] = time.time()
                
                if req.metadata:
                    result["metadata"] = req.metadata
                
                # Deliver result through callback
                req.callback(result)
                
            except Exception as e:
                logger.error(f"Error processing result for request {req.id}: {e}")
                req.callback({
                    "error": f"Result processing error: {str(e)}",
                    "request_id": req.id,
                    "timestamp": time.time()
                })
    
    def infer(self, input_data: Union[torch.Tensor, np.ndarray], 
              callback: Callable[[Dict[str, Any]], None],
              metadata: Optional[Dict[str, Any]] = None,
              priority: int = 0) -> str:
        """
        Submit a single inference request.
        
        Args:
            input_data: Input tensor or numpy array
            callback: Function to call with results
            metadata: Optional metadata to include with request
            priority: Request priority (higher = processed sooner)
            
        Returns:
            request_id: Unique ID for the request
        """
        if not self.running:
            raise RuntimeError("Inference engine is not running")
        
        # Validate input
        if not isinstance(input_data, (torch.Tensor, np.ndarray)):
            raise InputValidationError(f"Input must be tensor or numpy array, got {type(input_data)}")
        
        # Create request ID
        request_id = f"req_{time.time()}_{id(input_data)}"
        
        # Create and enqueue request
        request = InferenceRequest(
            id=request_id,
            input=input_data,
            callback=callback,
            metadata=metadata or {},
            priority=priority
        )
        
        # Add to queue
        self.request_queue.put(request)
        
        return request_id
    
    def batch_infer(self, inputs: List[Union[torch.Tensor, np.ndarray]],
                  callbacks: List[Callable[[Dict[str, Any]], None]],
                  metadata: Optional[List[Dict[str, Any]]] = None,
                  priority: int = 0) -> List[str]:
        """
        Submit a batch of inference requests.
        
        Args:
            inputs: List of input tensors or numpy arrays
            callbacks: List of callback functions
            metadata: Optional list of metadata dicts
            priority: Priority for all requests in batch
            
        Returns:
            List of request IDs
        """
        if not self.running:
            raise RuntimeError("Inference engine is not running")
        
        if len(inputs) != len(callbacks):
            raise ValueError(f"Number of inputs ({len(inputs)}) must match number of callbacks ({len(callbacks)})")
        
        if metadata and len(metadata) != len(inputs):
            raise ValueError(f"If provided, metadata length ({len(metadata)}) must match inputs ({len(inputs)})")
        
        request_ids = []
        
        # Create and submit individual requests
        for i, (input_data, callback) in enumerate(zip(inputs, callbacks)):
            meta = metadata[i] if metadata else None
            request_id = self.infer(input_data, callback, meta, priority)
            request_ids.append(request_id)
        
        return request_ids
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics."""
        with self.lock:
            # Calculate derived metrics
            uptime = time.time() - self.metrics["start_time"]
            metrics = dict(self.metrics)  # Create a copy
            
            # Calculate average batch size
            batch_sizes = metrics.get("batch_sizes", [])
            avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
            
            # Calculate throughput
            inference_count = metrics["inference_count"]
            throughput = inference_count / uptime if uptime > 0 else 0
            
            # Calculate average latency
            avg_latency = (metrics["total_latency"] / metrics["batch_count"]) if metrics["batch_count"] > 0 else 0
            
            # Add derived metrics
            metrics.update({
                "uptime": uptime,
                "avg_batch_size": avg_batch_size,
                "throughput": throughput,
                "avg_latency": avg_latency,
                "health_status": self.health_status["status"]
            })
            
            # Limit size of batch_sizes array to prevent memory growth
            if len(batch_sizes) > 1000:
                metrics["batch_sizes"] = batch_sizes[-1000:]
            
            return metrics
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the loaded engine."""
        info = {
            "engine_path": self.engine_path,
            "max_batch_size": self.max_batch_size,
            "min_batch_size": self.min_batch_size,
            "dynamic_batching": self.dynamic_batching,
            "num_streams": self.num_streams,
            "input_bindings": {},
            "output_bindings": {},
            "health_status": self.health_status["status"]
        }
        
        # Add input binding info
        for name, binding in self.input_bindings.items():
            info["input_bindings"][name] = {
                "shape": list(binding["shape"]),
                "index": binding["index"],
            }
        
        # Add output binding info
        for name, binding in self.output_bindings.items():
            info["output_bindings"][name] = {
                "shape": list(binding.shape),
                "index": binding.index,
            }
        
        return info
    
    def clear_queue(self) -> int:
        """
        Clear all pending requests from the queue.
        Returns the number of cleared requests.
        """
        count = 0
        with self.lock:
            # Create an empty queue
            old_queue = self.request_queue
            self.request_queue = PriorityQueue()
            
            # Count and notify callbacks about cancellation
            try:
                while True:
                    req = old_queue.get_nowait()
                    count += 1
                    try:
                        req.callback({
                            "error": "Request cancelled - queue cleared",
                            "request_id": req.id,
                            "timestamp": time.time()
                        })
                    except Exception:
                        pass
            except queue.Empty:
                pass
        
        logger.info(f"Cleared {count} pending requests from queue")
        return count
    
    def shutdown(self) -> None:
        """Gracefully shut down the inference engine."""
        logger.info("Shutting down TensorRT inference engine...")
        
        # Set running flag to False to stop threads
        with self.lock:
            self.running = False
        
        # Clear request queue
        self.clear_queue()
        
        # Wait for threads to finish
        for i, thread in enumerate(self.worker_threads):
            if thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning(f"Worker thread {i} did not terminate gracefully")
        
        # Wait for health monitoring thread
        if hasattr(self, 'health_thread') and self.health_thread.is_alive():
            self.health_thread.join(timeout=2.0)
        
        # Clean up CUDA resources
        if hasattr(self, 'contexts'):
            for context in self.contexts:
                del context
        
        if hasattr(self, 'engine'):
            del self.engine
        
        # Force CUDA garbage collection
        torch.cuda.empty_cache()
        
        logger.info("TensorRT inference engine shut down successfully")
    
    def __enter__(self):
        """Support for context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        self.shutdown()

# ---------------------------
# Utility Factory Class
# ---------------------------
class InferenceFactory:
    """Factory class to create and manage inference components."""
    
    @staticmethod
    def create_postprocessor(
        task_type: str, 
        **kwargs
    ) -> BasePostprocessor:
        """
        Create a postprocessor for the specified task type.
        
        Args:
            task_type: Type of task (classification, detection, segmentation)
            **kwargs: Additional arguments for the postprocessor
            
        Returns:
            Appropriate postprocessor instance
        """
        task_type = task_type.lower()
        
        if task_type == "classification":
            return ClassificationPostprocessor(**kwargs)
            
        elif task_type in ("detection", "object_detection"):
            return DetectionPostprocessor(**kwargs)
            
        elif task_type in ("segmentation", "semantic_segmentation"):
            return SegmentationPostprocessor(**kwargs)
            
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    @staticmethod
    def create_engine(
        engine_path: str,
        **kwargs
    ) -> TensorRTInferenceEngine:
        """
        Create and initialize a TensorRT inference engine.
        
        Args:
            engine_path: Path to TensorRT engine file
            **kwargs: Additional arguments for the engine
            
        Returns:
            Initialized TensorRTInferenceEngine
        """
        return TensorRTInferenceEngine(engine_path, **kwargs)
    
    @staticmethod
    def create_pipeline(
        engine_path: str,
        task_type: str,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        postprocessor_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[TensorRTInferenceEngine, BasePostprocessor]:
        """
        Create a complete inference pipeline with engine and postprocessor.
        
        Args:
            engine_path: Path to TensorRT engine file
            task_type: Type of task for postprocessor
            engine_kwargs: Additional arguments for the engine
            postprocessor_kwargs: Additional arguments for the postprocessor
            
        Returns:
            Tuple of (engine, postprocessor)
        """
        engine_kwargs = engine_kwargs or {}
        postprocessor_kwargs = postprocessor_kwargs or {}
        
        # Create components
        engine = InferenceFactory.create_engine(engine_path, **engine_kwargs)
        postprocessor = InferenceFactory.create_postprocessor(task_type, **postprocessor_kwargs)
        
        return engine, postprocessor

# ---------------------------
# Example Usage
# ---------------------------
def example_classification():
    """Example of classification inference."""
    # Initialize engine and postprocessor
    engine_path = "/path/to/classification_engine.trt"
    
    try:
        # Create pipeline
        engine, postprocessor = InferenceFactory.create_pipeline(
            engine_path=engine_path,
            task_type="classification",
            engine_kwargs={
                "max_batch_size": 16,
                "dynamic_batching": True,
                "num_streams": 2
            },
            postprocessor_kwargs={
                "class_labels": ["cat", "dog", "bird"],
                "top_k": 3
            }
        )
        
        # Create dummy input
        dummy_input = torch.randn(3, 224, 224)
        
        # Define callback
        def callback(result):
            if "error" in result:
                print(f"Error: {result['error']}")
                return
                
            # Get classification output
            logits = result["output0"]
            
            # Apply postprocessing
            classifications = postprocessor(logits)
            print(f"Classifications: {classifications}")
            
            # Optionally visualize
            if hasattr(postprocessor, "visualize") and "input_image" in result["metadata"]:
                image = result["metadata"]["input_image"]
                vis_img = postprocessor.visualize(classifications[0], image)
                cv2.imwrite("classification_result.jpg", vis_img)
        
        # Submit inference request
        engine.infer(
            dummy_input, 
            callback,
            {"input_image": np.zeros((224, 224, 3), dtype=np.uint8)}
        )
        
        # Wait for completion
        time.sleep(1)
        
        # Clean up
        engine.shutdown()
        
    except Exception as e:
        logger.error(f"Error in classification example: {e}")

def example_detection():
    """Example of object detection inference."""
    # Initialize engine and postprocessor
    engine_path = "/path/to/detection_engine.trt"
    
    try:
        # Create pipeline
        engine, postprocessor = InferenceFactory.create_pipeline(
            engine_path=engine_path,
            task_type="detection",
            engine_kwargs={
                "max_batch_size": 8,
                "dynamic_batching": True
            },
            postprocessor_kwargs={
                "class_labels": ["person", "car", "bicycle"],
                "score_threshold": 0.3,
                "nms_iou_threshold": 0.45
            }
        )
        
        # Create dummy input
        dummy_input = torch.randn(3, 640, 640)
        
        # Define callback
        def callback(result):
            if "error" in result:
                print(f"Error: {result['error']}")
                return
                
            # Get detection outputs
            # Example: Result has boxes, scores and labels
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["labels"]
            
            # Format as expected by postprocessor
            detection_output = [{
                "boxes": boxes,
                "scores": scores,
                "labels": labels
            }]
            
            # Apply postprocessing
            detections = postprocessor(detection_output)
            print(f"Detected {len(detections[0]['boxes'])} objects")
            
            # Optionally visualize
            if hasattr(postprocessor, "visualize") and "input_image" in result["metadata"]:
                image = result["metadata"]["input_image"]
                vis_img = postprocessor.visualize(image, detections[0])
                cv2.imwrite("detection_result.jpg", vis_img)
        
        # Submit inference request
        engine.infer(
            dummy_input, 
            callback,
            {"input_image": np.zeros((640, 640, 3), dtype=np.uint8)}
        )
        
        # Wait for completion
        time.sleep(1)
        
        # Clean up
        engine.shutdown()
        
    except Exception as e:
        logger.error(f"Error in detection example: {e}")

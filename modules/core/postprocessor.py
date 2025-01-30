"""
core/postprocessor.py

Postprocessing module for PyTorch model outputs.

Includes:
    - BasePostprocessor: A no-op base class.
    - ClassificationPostprocessor: Converts logits to probabilities, extracts top-K classes.
    - DetectionPostprocessor: Applies NMS and thresholding to bounding boxes, labels, and scores.
      (useful for object detection models).

Key Optimization Features:
    - Vectorized PyTorch operations (e.g., softmax, max, topk).
    - Optional GPU usage for NMS, which can significantly speed up large batch detection tasks.
    - Batch-friendly design (process multiple samples at once if feasible).
    - Parameterizable thresholds for filtering out low-confidence predictions.
"""

import logging
from typing import Any, Dict, List, Tuple, Union, Optional

import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BasePostprocessor:
    """
    A base postprocessor that performs no additional processing on the model outputs.
    It can serve as a pass-through or a foundation for more specialized postprocessors.
    """

    def __call__(self, outputs: Any) -> Any:
        """
        Handle raw model outputs with no further transformation.
        
        Args:
            outputs (Any): Raw outputs from a PyTorch model (logits, bounding boxes, etc.).

        Returns:
            Any: Post-processed outputs (here, returned as-is).
        """
        return outputs


class ClassificationPostprocessor(BasePostprocessor):
    """
    A postprocessor for classification tasks. 
    It converts logits to probabilities (softmax), then retrieves the top-K classes and scores.

    If you have a known mapping from class indices to labels (like ImageNet classes),
    you can inject or load that to produce human-readable labels.
    """

    def __init__(
        self,
        top_k: int = 5,
        class_labels: Optional[List[str]] = None,
        apply_softmax: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            top_k (int): Number of top predictions to return.
            class_labels (List[str], optional): A list of string labels for each class index.
            apply_softmax (bool): Whether to apply softmax to model outputs.
            device (torch.device, optional): Device on which postprocessing operations should run.
                                             If None, it uses the same device as the outputs' tensor.
        """
        super().__init__()
        self.top_k = top_k
        self.class_labels = class_labels  # e.g., ["cat", "dog", "car", ...]
        self.apply_softmax = apply_softmax
        self.device = device

    def __call__(self, outputs: torch.Tensor) -> List[List[Tuple[str, float]]]:
        """
        Convert model outputs into a list of top-K label-probability pairs.

        Args:
            outputs (torch.Tensor): Raw model outputs (batch_size x num_classes).

        Returns:
            List[List[Tuple[str, float]]]:
                A nested list where each element corresponds to one sample's top-K results,
                e.g., [[("cat", 0.9), ("dog", 0.05), ...], [...], ...].
        """
        if self.device is not None:
            outputs = outputs.to(self.device)

        # 1. (Optional) Softmax to get probabilities
        if self.apply_softmax:
            outputs = torch.nn.functional.softmax(outputs, dim=1)

        # 2. top-K extraction
        top_probs, top_idxs = outputs.topk(self.top_k, dim=1)

        # 3. Convert to CPU for indexing / label mapping if needed
        top_probs = top_probs.cpu().numpy()
        top_idxs = top_idxs.cpu().numpy()

        results = []
        batch_size = outputs.shape[0]

        for i in range(batch_size):
            sample_results = []
            for j in range(self.top_k):
                class_idx = top_idxs[i, j]
                prob = float(top_probs[i, j])

                if self.class_labels and class_idx < len(self.class_labels):
                    label = self.class_labels[class_idx]
                else:
                    # fallback to numeric index if no labels provided
                    label = f"Class_{class_idx}"

                sample_results.append((label, prob))

            results.append(sample_results)

        return results


class DetectionPostprocessor(BasePostprocessor):
    """
    A postprocessor for object detection models.
    Typically expects model outputs in the format (for each image):
        [
            {
                "boxes": Tensor[N,4],
                "labels": Tensor[N],
                "scores": Tensor[N],
                ...
            },
            ...
        ]

    Applies:
        - Non-Maximum Suppression (NMS) per image
        - Confidence threshold filtering
    """

    def __init__(
        self,
        score_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        max_detections: int = 100,
        use_fast_nms: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            score_threshold (float): Minimum confidence score to keep a detection.
            nms_iou_threshold (float): IoU threshold for NMS (0.5 is typical).
            max_detections (int): Maximum number of detections to keep after NMS.
            use_fast_nms (bool): Whether to use built-in GPU-accelerated NMS if available.
            device (torch.device, optional): Device to run postprocessing on. None means auto-detect.
        """
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections
        self.use_fast_nms = use_fast_nms
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def __call__(self, outputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, List]]: 
        """
        Perform NMS and filtering on object detection outputs.

        Args:
            outputs (List[Dict[str, torch.Tensor]]):
                A list of dicts, where each dict corresponds to one image.
                Each dict should have keys "boxes", "labels", "scores".

        Returns:
            List[Dict[str, List]]:
                A list of dicts with the same length as 'outputs'. Each dict has filtered
                "boxes", "labels", and "scores" in Python list format (instead of Tensors).
        """
        processed_results = []

        for per_image_output in outputs:
            boxes = per_image_output["boxes"].to(self.device)
            scores = per_image_output["scores"].to(self.device)
            labels = per_image_output["labels"].to(self.device)

            # 1. Filter out low-confidence detections
            high_conf_indices = (scores >= self.score_threshold).nonzero(as_tuple=True)[0]
            boxes = boxes[high_conf_indices]
            scores = scores[high_conf_indices]
            labels = labels[high_conf_indices]

            # 2. NMS
            keep_indices = self._nms(boxes, scores)
            # Limit the number of detections
            keep_indices = keep_indices[: self.max_detections]

            # 3. Gather final boxes/scores/labels
            final_boxes = boxes[keep_indices].cpu().numpy().tolist()
            final_scores = scores[keep_indices].cpu().numpy().tolist()
            final_labels = labels[keep_indices].cpu().numpy().tolist()

            processed_results.append({
                "boxes": final_boxes,
                "scores": final_scores,
                "labels": final_labels,
            })

        return processed_results

    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Non-Maximum Suppression utility. If `use_fast_nms` is True and GPU is available,
        it uses the built-in Torch NMS on GPU (significantly faster on large detection sets).

        Args:
            boxes (torch.Tensor): [N,4]
            scores (torch.Tensor): [N]

        Returns:
            torch.Tensor: The indices of the boxes that remain after NMS.
        """
        # Torch's built-in NMS is typically GPU-accelerated if the data is on CUDA.
        # nms(boxes, scores, iou_threshold) returns indices of the kept boxes
        keep_indices = torch.ops.torchvision.nms(boxes, scores, self.nms_iou_threshold)
        return keep_indices

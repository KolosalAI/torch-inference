import logging
from typing import Any, Dict, List, Tuple, Union, Optional
import threading
import queue
import torch
import tensorrt as trt

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
        self.class_labels = class_labels
        self.apply_softmax = apply_softmax
        self.device = device

    def __call__(self, outputs: torch.Tensor) -> List[List[Tuple[str, float]]]:
        """
        Convert model outputs into a list of top-K label-probability pairs.

        Args:
            outputs (torch.Tensor): Raw model outputs (batch_size x num_classes).

        Returns:
            List[List[Tuple[str, float]]]:
                A nested list where each element corresponds to one sample's top-K results.
        """
        if self.device is not None:
            outputs = outputs.to(self.device)

        if self.apply_softmax:
            outputs = torch.nn.functional.softmax(outputs, dim=1)

        top_probs, top_idxs = outputs.topk(self.top_k, dim=1)
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
                    label = f"Class_{class_idx}"

                sample_results.append((label, prob))

            results.append(sample_results)

        return results


class DetectionPostprocessor(BasePostprocessor):
    """
    A postprocessor for object detection models.
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
            nms_iou_threshold (float): IoU threshold for NMS.
            max_detections (int): Maximum number of detections to keep after NMS.
            use_fast_nms (bool): Whether to use built-in GPU-accelerated NMS if available.
            device (torch.device, optional): Device to run postprocessing on.
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

        Returns:
            List[Dict[str, List]]:
                A list of dicts with filtered "boxes", "labels", and "scores".
        """
        processed_results = []

        for per_image_output in outputs:
            boxes = per_image_output["boxes"].to(self.device)
            scores = per_image_output["scores"].to(self.device)
            labels = per_image_output["labels"].to(self.device)

            high_conf_indices = (scores >= self.score_threshold).nonzero(as_tuple=True)[0]
            boxes = boxes[high_conf_indices]
            scores = scores[high_conf_indices]
            labels = labels[high_conf_indices]

            keep_indices = self._nms(boxes, scores)
            keep_indices = keep_indices[: self.max_detections]

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
        Non-Maximum Suppression utility.

        Args:
            boxes (torch.Tensor): [N,4]
            scores (torch.Tensor): [N]

        Returns:
            torch.Tensor: The indices of the boxes that remain after NMS.
        """
        keep_indices = torch.ops.torchvision.nms(boxes, scores, self.nms_iou_threshold)
        return keep_indices


class TensorRTInferenceEngine:
    """
    A class to handle TensorRT inference with inflight batching and concurrent request processing.
    """

    def __init__(self, engine_path: str, max_batch_size: int = 32):
        """
        Args:
            engine_path (str): Path to the TensorRT engine file.
            max_batch_size (int): Maximum batch size for inference.
        """
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.max_batch_size = max_batch_size
        self.request_queue = queue.Queue()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._process_requests, daemon=True)
        self.thread.start()

    def _process_requests(self):
        """
        Process requests from the queue in batches.
        """
        while True:
            batch = []
            while len(batch) < self.max_batch_size and not self.request_queue.empty():
                batch.append(self.request_queue.get())

            if batch:
                self._infer_batch(batch)

    def _infer_batch(self, batch: List[Dict[str, Any]]):
        """
        Perform inference on a batch of requests.

        Args:
            batch (List[Dict[str, Any]]): A list of requests to process.
        """
        # Prepare inputs and outputs for TensorRT
        inputs = self._prepare_inputs(batch)
        outputs = self._prepare_outputs()

        # Perform inference
        self.context.execute_async_v2(bindings=inputs + outputs, stream_handle=torch.cuda.current_stream().cuda_stream)

        # Postprocess outputs
        results = self._postprocess_outputs(outputs)

        # Notify requesters
        for i, result in enumerate(results):
            batch[i]["callback"](result)

    def _prepare_inputs(self, batch: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """
        Prepare inputs for TensorRT inference.

        Args:
            batch (List[Dict[str, Any]]): A list of requests.

        Returns:
            List[torch.Tensor]: A list of input tensors.
        """
        # Implement input preparation logic here
        pass

    def _prepare_outputs(self) -> List[torch.Tensor]:
        """
        Prepare outputs for TensorRT inference.

        Returns:
            List[torch.Tensor]: A list of output tensors.
        """
        # Implement output preparation logic here
        pass

    def _postprocess_outputs(self, outputs: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """
        Postprocess TensorRT outputs.

        Args:
            outputs (List[torch.Tensor]): A list of output tensors.

        Returns:
            List[Dict[str, Any]]: A list of postprocessed results.
        """
        # Implement postprocessing logic here
        pass

    def infer(self, request: Dict[str, Any], callback: callable):
        """
        Add a request to the inference queue.

        Args:
            request (Dict[str, Any]): The request to process.
            callback (callable): A callback function to handle the result.
        """
        with self.lock:
            self.request_queue.put({"request": request, "callback": callback})
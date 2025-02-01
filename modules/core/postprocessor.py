import logging
from typing import Any, Dict, List, Optional
import threading
import queue
import torch
import torch.nn.functional as F
import tensorrt as trt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------
# (Unchanged) Postprocessor Classes
# ---------------------------
class BasePostprocessor:
    """
    A base postprocessor that performs no additional processing on the model outputs.
    """
    def __call__(self, outputs: Any) -> Any:
        return outputs

class ClassificationPostprocessor(BasePostprocessor):
    """
    Postprocessor for classification models, supporting multi-label classification.
    """
    def __init__(
        self, top_k: int = 5, class_labels: Optional[List[str]] = None, apply_softmax: bool = True,
        multi_label: bool = False, threshold: float = 0.5, device: Optional[torch.device] = None
    ):
        super().__init__()
        self.top_k = top_k
        self.class_labels = class_labels
        self.apply_softmax = apply_softmax
        self.multi_label = multi_label
        self.threshold = threshold
        self.device = device

    def __call__(self, outputs: torch.Tensor) -> List[List[tuple]]:
        if self.device:
            outputs = outputs.to(self.device)

        if self.apply_softmax:
            outputs = F.softmax(outputs, dim=1)

        if self.multi_label:
            results = []
            for sample in outputs:
                result = [(self.class_labels[i], float(prob)) for i, prob in enumerate(sample) if prob >= self.threshold]
                results.append(result)
            return results

        top_probs, top_idxs = outputs.topk(self.top_k, dim=1)
        return [[(self.class_labels[i] if self.class_labels else f'Class_{i}', float(prob))
                 for i, prob in zip(indices, probs)]
                for probs, indices in zip(top_probs, top_idxs)]

class DetectionPostprocessor(BasePostprocessor):
    """
    Postprocessor for object detection models with zero-copy operations and batch-wise optimizations.
    """
    def __init__(self, score_threshold: float = 0.5, nms_iou_threshold: float = 0.5, max_detections: int = 100,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, outputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, List]]:
        processed_results = []
        for per_image_output in outputs:
            boxes = per_image_output["boxes"].to(self.device)
            scores = per_image_output["scores"].to(self.device)
            labels = per_image_output["labels"].to(self.device)
            keep = scores >= self.score_threshold
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep = torch.ops.torchvision.nms(boxes, scores, self.nms_iou_threshold)[:self.max_detections]
            processed_results.append({
                "boxes": boxes[keep].cpu().tolist(),
                "scores": scores[keep].cpu().tolist(),
                "labels": labels[keep].cpu().tolist()
            })
        return processed_results

class SegmentationPostprocessor(BasePostprocessor):
    """
    Postprocessor for segmentation models that generates masks efficiently.
    """
    def __init__(self, threshold: float = 0.5, device: Optional[torch.device] = None):
        super().__init__()
        self.threshold = threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, outputs: torch.Tensor) -> List[torch.Tensor]:
        if self.device:
            outputs = outputs.to(self.device)
        masks = (outputs > self.threshold).byte().cpu()
        return [mask.numpy() for mask in masks]

# ---------------------------
# Updated TensorRTInferenceEngine
# ---------------------------
class TensorRTInferenceEngine:
    """
    TensorRT inference engine supporting dynamic batching, autoscaling, and streaming.
    """
    def __init__(self,
                 engine_path: str,
                 max_batch_size: int = 32,
                 min_batch_size: int = 1,
                 timeout: float = 0.05,
                 dynamic_batching: bool = True):
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.timeout = timeout
        self.dynamic_batching = dynamic_batching
        self.request_queue = queue.Queue()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._process_requests, daemon=True)
        self.thread.start()

    def _process_requests(self):
        """Process requests from the queue in dynamic batches."""
        while True:
            batch = []
            try:
                # Wait for the first request with a timeout.
                req = self.request_queue.get(timeout=self.timeout)
                batch.append(req)
            except queue.Empty:
                continue

            if self.dynamic_batching:
                # Attempt to collect additional requests quickly.
                while len(batch) < self.max_batch_size:
                    try:
                        req = self.request_queue.get(timeout=0.005)
                        batch.append(req)
                    except queue.Empty:
                        break
            else:
                # If dynamic batching is disabled, wait until reaching max_batch_size.
                while len(batch) < self.max_batch_size:
                    req = self.request_queue.get()
                    batch.append(req)
            self._infer_batch(batch)

    def _infer_batch(self, batch: List[Dict[str, Any]]):
        """Perform inference on a batch with zero-copy execution."""
        actual_batch_size = len(batch)
        inputs = self._prepare_inputs(batch)  # returns a list with one batched tensor
        outputs = self._prepare_outputs(actual_batch_size)  # prepare outputs sized to actual batch
        # Build bindings from input and output tensors (using data_ptr() for zero-copy)
        bindings = [inp.data_ptr() for inp in inputs] + [out.data_ptr() for out in outputs]
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream)
        results = self._postprocess_outputs(outputs, actual_batch_size)
        for i, result in enumerate(results):
            batch[i]["callback"](result)

    def _prepare_inputs(self, batch: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """Prepare and stack inputs for TensorRT inference (zero-copy)."""
        input_list = []
        for req in batch:
            inp = req["input"]
            if not isinstance(inp, torch.Tensor):
                inp = torch.tensor(inp, device="cuda")
            else:
                inp = inp.to("cuda")
            input_list.append(inp)
        # Stack the inputs along the first dimension to form a batched tensor.
        batch_tensor = torch.stack(input_list)
        return [batch_tensor]

    def _prepare_outputs(self, actual_batch_size: int) -> List[torch.Tensor]:
        """
        Prepare output tensors for inference (zero-copy).
        (Adjust the output shapes as needed for your model.)
        """
        # Example output shapes; update these to match your engine's outputs.
        output_shapes = [(1000,), (1, 28, 28)]
        return [torch.empty((actual_batch_size, *shape), device="cuda") for shape in output_shapes]

    def _postprocess_outputs(self, outputs: List[torch.Tensor], actual_batch_size: int) -> List[Dict[str, Any]]:
        """
        Postprocess results by splitting the output tensors per sample.
        Returns a list of dictionaries, one per sample.
        """
        sample_results = []
        for i in range(actual_batch_size):
            sample_result = {}
            for j, output in enumerate(outputs):
                sample_result[f"output{j}"] = output[i].cpu().numpy()
            sample_results.append(sample_result)
        return sample_results

    def infer(self, request: Any, callback: callable):
        """
        Add a request to the inference queue with batch-wise optimization.
        'request' should contain the input data. 'callback' will be called with the inference result.
        """
        with self.lock:
            self.request_queue.put({"input": request, "callback": callback})

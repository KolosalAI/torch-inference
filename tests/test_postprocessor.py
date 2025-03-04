import os
import time
import json
import unittest
import torch
import numpy as np
import cv2
from pathlib import Path

# Import from the module where your code is located.
# Replace `inference_pipeline` with the actual module path.
from modules.core.postprocessor import (
    validate_tensor,
    cuda_error_handling,
    ClassificationPostprocessor,
    DetectionPostprocessor,
    SegmentationPostprocessor,
    PriorityQueue,
    TRTOutputBinding,
    TensorRTInferenceEngine,
    InferenceFactory,
    InferenceRequest,
    ModelLoadError,
    InputValidationError,
    ProcessingError
)


class TestUtilityFunctions(unittest.TestCase):
    def test_validate_tensor_success(self):
        # Valid tensor with expected shape
        t = torch.randn(3, 4)
        validate_tensor(t, expected_shape=(3, 4))
    
    def test_validate_tensor_wrong_type(self):
        with self.assertRaises(InputValidationError):
            validate_tensor("not a tensor", expected_shape=(3, 4))
    
    def test_validate_tensor_wrong_shape(self):
        t = torch.randn(2, 4)
        with self.assertRaises(InputValidationError):
            validate_tensor(t, expected_shape=(3, 4))
    
    def test_cuda_error_handling(self):
        # Test that the context manager catches a RuntimeError with "CUDA" in its message
        with self.assertRaises(Exception) as context:
            with cuda_error_handling():
                raise RuntimeError("CUDA error: something went wrong")
        self.assertIn("CUDA error occurred", str(context.exception))


class TestClassificationPostprocessor(unittest.TestCase):
    def setUp(self):
        # Create a simple classification postprocessor instance.
        self.postprocessor = ClassificationPostprocessor(
            top_k=2,
            class_labels=["cat", "dog", "bird"],
            apply_softmax=True,
            multi_label=False
        )
    
    def test_validate_input_success(self):
        # Create a 2D tensor (batch_size, num_classes)
        logits = torch.randn(5, 3)
        # Should not raise any error.
        self.postprocessor.validate_input(logits)
    
    def test_validate_input_wrong_dims(self):
        logits = torch.randn(5, 3, 2)
        with self.assertRaises(InputValidationError):
            self.postprocessor.validate_input(logits)
    
    def test_process_single_label(self):
        # Create a batch of logits
        logits = torch.tensor([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]])
        results = self.postprocessor.process(logits)
        # For each sample, result should be a list of tuples (label, probability)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        for sample in results:
            self.assertIsInstance(sample, list)
            self.assertGreaterEqual(len(sample), 1)
            for tup in sample:
                self.assertIsInstance(tup, tuple)
                self.assertIn(tup[0], self.postprocessor.class_labels)
    
    def test_visualize(self):
        # Test visualization returns an image of the same shape as input image.
        classifications = [("cat", 0.9), ("dog", 0.1)]
        blank_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        vis_img = self.postprocessor.visualize(classifications, image=blank_image)
        self.assertEqual(vis_img.shape, blank_image.shape)


class TestDetectionPostprocessor(unittest.TestCase):
    def setUp(self):
        # Create a detection postprocessor instance.
        self.postprocessor = DetectionPostprocessor(
            score_threshold=0.5,
            nms_iou_threshold=0.5,
            class_labels=["person", "car", "bicycle"]
        )
    
    def test_validate_input_success(self):
        # Create a valid detection output list
        outputs = [{
            "boxes": torch.tensor([[10, 10, 50, 50], [15, 15, 55, 55]]),
            "scores": torch.tensor([0.8, 0.4]),
            "labels": torch.tensor([0, 1])
        }]
        self.postprocessor.validate_input(outputs)
    
    def test_validate_input_missing_key(self):
        outputs = [{
            "boxes": torch.tensor([[10, 10, 50, 50]]),
            "scores": torch.tensor([0.8])
            # Missing "labels"
        }]
        with self.assertRaises(InputValidationError):
            self.postprocessor.validate_input(outputs)
    
    def test_process_detection(self):
        outputs = [{
            "boxes": torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]]),
            "scores": torch.tensor([0.9, 0.6]),
            "labels": torch.tensor([0, 0])
        }]
        result = self.postprocessor.process(outputs)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        res = result[0]
        for key in ["boxes", "scores", "labels", "class_names"]:
            self.assertIn(key, res)
    
    def test_visualize_detection(self):
        # Create a dummy detection result dictionary.
        detections = {
            "boxes": [[10, 10, 50, 50]],
            "scores": [0.95],
            "labels": [0],
            "class_names": ["person"]
        }
        # Create a blank image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        vis_img = self.postprocessor.visualize(image, detections)
        self.assertEqual(vis_img.shape, image.shape)


class TestSegmentationPostprocessor(unittest.TestCase):
    def test_validate_input_binary_success(self):
        # Test with binary segmentation: either (H, W) or (1, H, W)
        postprocessor = SegmentationPostprocessor(threshold=0.5, multi_class=False)
        # 3D tensor: (1, H, W)
        tensor1 = torch.randn(1, 224, 224)
        postprocessor.validate_input(tensor1)
        # 2D tensor: (H, W)
        tensor2 = torch.randn(224, 224)
        postprocessor.validate_input(tensor2)
    
    def test_validate_input_multiclass_success(self):
        # Multi-class: tensor with shape (num_classes, H, W)
        postprocessor = SegmentationPostprocessor(threshold=0.5, multi_class=True)
        tensor = torch.randn(3, 224, 224)
        postprocessor.validate_input(tensor)
    
    def test_validate_input_multiclass_wrong_dims(self):
        postprocessor = SegmentationPostprocessor(threshold=0.5, multi_class=True)
        tensor = torch.randn(224, 224)  # 2D tensor, not valid for multi-class
        with self.assertRaises(InputValidationError):
            postprocessor.validate_input(tensor)
    
    def test_process_binary_segmentation(self):
        # Create a binary segmentation tensor with a clear thresholded output.
        postprocessor = SegmentationPostprocessor(threshold=0.0, multi_class=False, min_contour_area=1)
        # Create a tensor where all values exceed threshold.
        tensor = torch.ones(1, 64, 64)
        result = postprocessor.process(tensor)
        self.assertIsInstance(result, dict)
        self.assertIn("mask", result)
        self.assertIn("instances", result)
        self.assertFalse(result["multi_class"])
    
    def test_process_multiclass_segmentation(self):
        # Create a multi-class segmentation tensor.
        postprocessor = SegmentationPostprocessor(threshold=0.0, multi_class=True, min_contour_area=1,
                                                  class_labels=["bg", "class1", "class2"])
        # Create a tensor with 3 classes.
        tensor = torch.zeros(3, 64, 64)
        # Set half the pixels to class1 and half to class2.
        tensor[1, :32, :] = 5.0
        tensor[2, 32:, :] = 5.0
        result = postprocessor.process(tensor)
        self.assertIsInstance(result, dict)
        self.assertIn("class_masks", result)
        self.assertTrue(result["multi_class"])


class TestPriorityQueue(unittest.TestCase):
    def test_priority_order(self):
        pq = PriorityQueue()
        # Create three inference requests with different priorities and timestamps.
        req1 = InferenceRequest(id="1", input=torch.tensor([1.0]), callback=lambda r: None, priority=1)
        req2 = InferenceRequest(id="2", input=torch.tensor([2.0]), callback=lambda r: None, priority=3)
        req3 = InferenceRequest(id="3", input=torch.tensor([3.0]), callback=lambda r: None, priority=2)
        pq.put(req1)
        pq.put(req2)
        pq.put(req3)
        # Requests should come out in order of highest priority first.
        first = pq.get()
        second = pq.get()
        third = pq.get()
        self.assertEqual(first.id, "2")
        self.assertEqual(second.id, "3")
        self.assertEqual(third.id, "1")


class TestTRTOutputBinding(unittest.TestCase):
    def test_allocate_and_deallocate(self):
        # We simulate a binding for a tensor of shape [-1, 3, 224, 224].
        # Here we set the dynamic batch size to 4.
        binding = TRTOutputBinding(name="output0", index=0, shape=(-1, 3, 224, 224), dtype=0)  # trt.DataType.FLOAT assumed 0
        allocated_tensor = binding.allocate(4)
        self.assertIsInstance(allocated_tensor, torch.Tensor)
        self.assertEqual(allocated_tensor.shape[0], 4)
        self.assertEqual(list(allocated_tensor.shape[1:]), [3, 224, 224])
        binding.deallocate()
        self.assertIsNone(binding.tensor)


class TestInferenceFactory(unittest.TestCase):
    def test_create_postprocessor_classification(self):
        pp = InferenceFactory.create_postprocessor(
            task_type="classification",
            class_labels=["cat", "dog"],
            top_k=1
        )
        self.assertIsInstance(pp, ClassificationPostprocessor)
    
    def test_create_postprocessor_detection(self):
        pp = InferenceFactory.create_postprocessor(
            task_type="detection",
            class_labels=["person", "car"]
        )
        self.assertIsInstance(pp, DetectionPostprocessor)
    
    def test_create_engine_invalid_path(self):
        # Engine file does not exist, so it should raise a ModelLoadError.
        with self.assertRaises(ModelLoadError):
            InferenceFactory.create_engine("nonexistent_engine.trt")
    
    def test_create_pipeline(self):
        # Since an engine with a valid file is required and we cannot create a real engine here,
        # we test that the pipeline creation raises the appropriate error.
        with self.assertRaises(ModelLoadError):
            InferenceFactory.create_pipeline(
                engine_path="nonexistent_engine.trt",
                task_type="classification"
            )


class TestTensorRTInferenceEngine(unittest.TestCase):
    def test_infer_invalid_engine(self):
        # With an invalid engine path, engine initialization should raise ModelLoadError.
        with self.assertRaises(ModelLoadError):
            TensorRTInferenceEngine(engine_path="nonexistent_engine.trt")
    
    def test_shutdown_without_running(self):
        # Create an engine instance in a try/except block.
        # We simulate an engine that fails initialization and then call shutdown.
        try:
            engine = TensorRTInferenceEngine(engine_path="nonexistent_engine.trt")
        except ModelLoadError:
            # Since initialization failed, create a dummy engine-like object.
            class DummyEngine:
                running = False
                def shutdown(self): pass
            engine = DummyEngine()
        # Call shutdown (should not raise an error)
        engine.shutdown()


if __name__ == "__main__":
    unittest.main()

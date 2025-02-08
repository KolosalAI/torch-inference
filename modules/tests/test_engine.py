import sys
import os
# Add the parent directory to sys.path so that imports like "core.engine" work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import unittest
import torch
import torch.nn as nn

# Import the engine classes from the core module.
from core.engine import InferenceEngine, EngineConfig

# ------------------------------------------------------------------------------
# Dummy Implementations for Testing
# ------------------------------------------------------------------------------

class DummyPIDController:
    """A simple PID controller that returns a proportional adjustment."""
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint

    def update(self, value, dt):
        # For testing, simply return a proportional adjustment.
        return self.kp * (self.setpoint - value)

# Override EngineConfig.__post_init__ to use DummyPIDController.
def dummy_post_init(self):
    self.pid_controller = DummyPIDController(self.pid_kp, self.pid_ki, self.pid_kd, 50.0)
    # Validate augmentation types.
    valid_augmentations = {"noise", "dropout", "flip"}
    if (invalid := set(self.guard_augmentation_types) - valid_augmentations):
        raise ValueError(f"Invalid augmentation types: {invalid}")
EngineConfig.__post_init__ = dummy_post_init

# Dummy Preprocessor and Postprocessor classes.
class DummyPreprocessor:
    def __call__(self, x):
        # If input is already a tensor, return it; otherwise, convert it.
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, dtype=torch.float32)

class DummyPostprocessor:
    def __call__(self, x):
        return x

# ------------------------------------------------------------------------------
# Dummy Models for Testing
# ------------------------------------------------------------------------------

class DummyModelSafe(nn.Module):
    """
    A dummy model that produces a dominant logit for class 0.
    This ensures that after softmax the confidence is high and the guard passes.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        batch_size = x.shape[0]
        # Set all logits to a low value.
        logits = torch.full((batch_size, self.num_classes), -10.0, device=x.device)
        # Set the first logit high so that softmax produces near-1 confidence for class 0.
        logits[:, 0] = 10.0
        return logits

class DummyModelUnsafe(nn.Module):
    """
    A dummy model that produces uniform logits.
    With softmax, this results in low confidence per class, causing the guard to fail.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.zeros((batch_size, self.num_classes), device=x.device)

class DummyModelIdentity(nn.Module):
    """
    A dummy model that returns zeros.
    This model is used for profiling tests.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.zeros((batch_size, self.num_classes), device=x.device)

# ------------------------------------------------------------------------------
# Test Cases for InferenceEngine
# ------------------------------------------------------------------------------

class TestInferenceEngine(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create a common EngineConfig for testing.
        # We force CPU usage to bypass CUDA/TensorRT code paths.
        self.config = EngineConfig(
            num_workers=1,
            queue_size=10,
            batch_size=2,
            min_batch_size=1,
            max_batch_size=4,
            warmup_runs=1,
            timeout=0.01,
            autoscale_interval=0.1,
            queue_size_threshold_high=80.0,
            queue_size_threshold_low=20.0,
            enable_dynamic_batching=True,
            debug_mode=True,
            use_multigpu=False,
            log_file="test_engine.log",  # Provide a valid log file path.
            executor_type="thread",
            enable_trt=False,
            use_tensorrt=False,
            num_classes=10,
            guard_enabled=True,
            guard_num_augmentations=2,
            guard_noise_level_range=(0.005, 0.005),  # Fixed noise level for determinism.
            guard_dropout_rate=0.0,
            guard_flip_prob=0.0,
            guard_confidence_threshold=0.5,
            guard_variance_threshold=0.05,
            guard_input_range=(0.0, 1.0),
            guard_augmentation_types=["noise", "dropout", "flip"]
        )
        self.device = torch.device("cpu")
        self.preprocessor = DummyPreprocessor()
        self.postprocessor = DummyPostprocessor()

    async def asyncTearDown(self):
        # (If needed, add any tearDown code common for all tests.)
        pass

    async def test_run_inference_async_safe(self):
        """
        Test asynchronous inference when the guard passes.
        The dummy model produces a dominant logit, so the engine returns the model's output.
        """
        safe_model = DummyModelSafe(num_classes=self.config.num_classes)
        engine = InferenceEngine(
            model=safe_model,
            device="cpu",
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_fp16=False,
            use_tensorrt=False,
            config=self.config
        )
        dummy_input = torch.randn(10)
        result = await engine.run_inference_async(dummy_input)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, self.config.num_classes))
        self.assertGreater(result[0, 0].item(), 0)
        await engine.cleanup()

    async def test_run_inference_async_unsafe(self):
        """
        Test asynchronous inference when the guard fails.
        The dummy model produces uniform logits so the guard returns a default response.
        """
        unsafe_model = DummyModelUnsafe(num_classes=self.config.num_classes)
        engine = InferenceEngine(
            model=unsafe_model,
            device="cpu",
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_fp16=False,
            use_tensorrt=False,
            config=self.config
        )
        dummy_input = torch.randn(10)
        result = await engine.run_inference_async(dummy_input)
        # The default response is a 1D tensor with uniform probabilities.
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (self.config.num_classes,))
        expected_prob = 1.0 / self.config.num_classes
        for val in result:
            self.assertAlmostEqual(val.item(), expected_prob, places=5)
        await engine.cleanup()

    async def test_run_batch_inference(self):
        """
        Test synchronous batch inference.
        The engine should stack a list of input tensors and produce a batched output.
        """
        safe_model = DummyModelSafe(num_classes=self.config.num_classes)
        engine = InferenceEngine(
            model=safe_model,
            device="cpu",
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_fp16=False,
            use_tensorrt=False,
            config=self.config
        )
        # Determine the expected input shape.
        if engine.input_shape is None:
            input_shape = (10,)
        else:
            input_shape = tuple(engine.input_shape)
        # Create a list of 4 dummy inputs.
        batch_inputs = [torch.randn(*input_shape) for _ in range(4)]
        output = engine.run_batch_inference(batch_inputs)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (4, self.config.num_classes))
        await engine.cleanup()

    async def test_profile_inference(self):
        """
        Test that the profiling interface returns expected metric keys.
        """
        identity_model = DummyModelIdentity(num_classes=self.config.num_classes)
        engine = InferenceEngine(
            model=identity_model,
            device="cpu",
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_fp16=False,
            use_tensorrt=False,
            config=self.config
        )
        if engine.input_shape is None:
            input_shape = (10,)
        else:
            input_shape = tuple(engine.input_shape)
        # Create a dummy input with a batch dimension.
        dummy_input = torch.randn(*(1,) + input_shape)
        metrics = engine.profile_inference(dummy_input)
        for key in ["preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"]:
            self.assertIn(key, metrics)
        await engine.cleanup()

    async def test_dynamic_batch_size(self):
        """
        Test that the dynamic batch size computed is within the configured limits.
        """
        safe_model = DummyModelSafe(num_classes=self.config.num_classes)
        engine = InferenceEngine(
            model=safe_model,
            device="cpu",
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_fp16=False,
            use_tensorrt=False,
            config=self.config
        )
        if engine.input_shape is None:
            sample_shape = (10,)
        else:
            sample_shape = tuple(engine.input_shape)
        sample_tensor = torch.randn(*sample_shape)
        new_batch_size = engine.dynamic_batch_size(sample_tensor)
        self.assertIsInstance(new_batch_size, int)
        self.assertGreaterEqual(new_batch_size, self.config.min_batch_size)
        self.assertLessEqual(new_batch_size, self.config.max_batch_size)
        await engine.cleanup()

if __name__ == "__main__":
    unittest.main()

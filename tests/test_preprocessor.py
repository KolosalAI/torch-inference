import os
import tempfile
import time
import threading
import unittest
import numpy as np
from PIL import Image
import torch

# Import the ImagePreprocessor from the corresponding module path.
# Adjust the import statement below to match your project structure.
from modules.core.preprocessor import ImagePreprocessor


class TestImagePreprocessor(unittest.TestCase):
    def setUp(self):
        # Create a preprocessor instance using CPU for deterministic behavior.
        self.preproc = ImagePreprocessor(
            image_size=(224, 224),
            mean=[0.485],
            std=[0.229],
            async_preproc=True,
            max_batch_size=4,
            max_queue_size=10,
            max_cache_size=50,
            num_cuda_streams=1,
            fallback_to_cpu=True,
            compiled_module_cache=None,  # disable TRT cache file to simplify testing
            device="cpu"
        )

    def tearDown(self):
        # Cleanup resources and remove any temporary files created.
        self.preproc.cleanup()
        self.preproc = None

    def test_single_input_pil(self):
        # Create a simple red image
        img = Image.new("RGB", (300, 300), color="red")
        result = self.preproc(img)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1:], self.preproc.image_size)

    def test_single_input_numpy(self):
        # Create a random numpy image array with shape (300,300,3)
        np_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        result = self.preproc(np_img)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1:], self.preproc.image_size)

    def test_single_input_tensor(self):
        # Create a torch tensor simulating an image
        tensor_img = torch.rand(3, 300, 300)
        result = self.preproc(tensor_img)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1:], self.preproc.image_size)

    def test_file_path_input(self):
        # Create a temporary image file using PIL and pass its path.
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            img = Image.new("RGB", (300, 300), color="blue")
            img.save(tmp_path)
            result = self.preproc(tmp_path)
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.shape[0], 3)
            self.assertEqual(result.shape[1:], self.preproc.image_size)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_batch_processing(self):
        # Create a list of images with different colors
        colors = ["red", "green", "blue"]
        images = [Image.new("RGB", (300, 300), color=c) for c in colors]
        batch = self.preproc(images)
        self.assertIsInstance(batch, torch.Tensor)
        self.assertEqual(batch.shape[0], len(images))
        self.assertEqual(batch.shape[1], 3)
        self.assertEqual(batch.shape[2:], self.preproc.image_size)

    def test_exceed_batch_size(self):
        # Create more images than max_batch_size and expect a ValueError.
        images = [Image.new("RGB", (300, 300), color="red") for _ in range(self.preproc.max_batch_size + 1)]
        with self.assertRaises(ValueError):
            _ = self.preproc(images)

    def test_async_preprocessing(self):
        # Use a threading event to wait for the async callback.
        event = threading.Event()
        results = []

        def callback(result):
            results.append(result)
            event.set()

        # Use a numpy image as input.
        np_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        self.preproc.preprocess_async(np_img, callback, timeout=2.0)
        # Wait for the callback (timeout after 5 seconds to prevent hanging)
        event.wait(5)
        self.assertTrue(event.is_set(), "Async callback was not called.")
        self.assertTrue(results[0] is not None)
        self.assertIsInstance(results[0], torch.Tensor)
        self.assertEqual(results[0].shape[0], 3)

    def test_benchmark(self):
        # Use a numpy image for benchmarking.
        np_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        mean_time, std_time, metrics = self.preproc.benchmark(np_img, iterations=3, warmup=1)
        self.assertIsInstance(mean_time, float)
        self.assertIsInstance(std_time, float)
        self.assertIsInstance(metrics, dict)
        # Check that at least one metric is available
        self.assertTrue(any(k in metrics for k in ["processing_time", "io_time", "gpu_time"]))

    def test_config_save_load_from_config(self):
        # Save the configuration to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            config_path = tmp.name
        try:
            self.preproc.save_config(config_path)
            loaded_config = ImagePreprocessor.load_config(config_path)
            self.assertIn("image_size", loaded_config)
            # Create a new preprocessor from the loaded config with device override
            new_preproc = ImagePreprocessor.from_config(config_path, device="cpu")
            self.assertEqual(str(new_preproc.device), "cpu")
            new_preproc.cleanup()
        finally:
            if os.path.exists(config_path):
                os.remove(config_path)

    def test_cancel_pending_tasks(self):
        # Enqueue several async tasks and then cancel them.
        # We do not wait for the callback so they remain pending.
        for _ in range(3):
            self.preproc.preprocess_async(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8),
                                            lambda x: None, timeout=1.0)
        # Cancel pending tasks
        self.preproc.cancel_pending_tasks()
        # Check that the internal queue is cleared (accessing the queue size)
        self.assertEqual(self.preproc.queue.qsize(), 0)

    def test_unsupported_input(self):
        # Pass an unsupported input type (e.g., integer) and expect a ValueError.
        with self.assertRaises(ValueError):
            _ = self.preproc(12345)

    def test_cleanup_and_del(self):
        # Test cleanup by calling cleanup and then deleting the instance.
        # After cleanup, re-calling cleanup should not cause error.
        self.preproc.cleanup()
        try:
            self.preproc.cleanup()
        except Exception as e:
            self.fail(f"Cleanup raised an exception on second call: {e}")
        # Test __del__ indirectly by deleting the instance
        del self.preproc  # This should not raise any exceptions


if __name__ == "__main__":
    unittest.main()

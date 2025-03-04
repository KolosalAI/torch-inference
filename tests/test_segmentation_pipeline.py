import unittest
import asyncio
import os
import tempfile
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from modules.pipelines.segmentations import (
    SegmentationModel,
    SegmentationPreprocessor,
    SegmentationPostprocessor,
    load_image_async,
    save_segmentation_results,
    batch_process_images,
    get_images_from_dir,
)

class TestSegmentationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up a dummy model for testing
        cls.dummy_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 1, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )
        cls.test_image_url = "https://via.placeholder.com/640x480.png"
        cls.test_image_path = str(Path(__file__).parent / "test_image.jpg")
        cls.test_output_dir = tempfile.mkdtemp()

        # Create a test image
        img = Image.new('RGB', (640, 480), color='red')
        img.save(cls.test_image_path)

    @classmethod
    def tearDownClass(cls):
        # Clean up test files
        if os.path.exists(cls.test_image_path):
            os.remove(cls.test_image_path)
        if os.path.exists(cls.test_output_dir):
            for root, dirs, files in os.walk(cls.test_output_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(cls.test_output_dir)

    def test_segmentation_preprocessor(self):
        """Test the SegmentationPreprocessor class"""
        preprocessor = SegmentationPreprocessor()

        # Test with numpy array
        img_np = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = preprocessor(img_np)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))

        # Test with file path
        tensor = preprocessor(self.test_image_path)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))

        # Test with invalid input
        with self.assertRaises(ValueError):
            preprocessor(None)

    def test_segmentation_postprocessor(self):
        """Test the SegmentationPostprocessor class"""
        postprocessor = SegmentationPostprocessor()

        # Test with tensor input
        output_tensor = torch.rand(1, 1, 640, 640)
        mask, contours = postprocessor(output_tensor)
        self.assertIsInstance(mask, np.ndarray)
        self.assertIsInstance(contours, list)

        # Test with invalid input
        with self.assertRaises(ValueError):
            postprocessor("invalid input")

    def test_segmentation_model(self):
        """Test the SegmentationModel class"""
        model = SegmentationModel(self.dummy_model)

        # Test process_image with numpy array
        img_np = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = model.process_image(img_np)
        self.assertIsInstance(result, dict)
        self.assertIn("mask", result)
        self.assertIn("contours", result)

        # Test process_image with file path
        result = model.process_image(self.test_image_path)
        self.assertIsInstance(result, dict)

        # Test benchmark_segmentation
        mean_time, std_time = model.benchmark_segmentation(self.test_image_path, iterations=2)
        self.assertIsInstance(mean_time, float)
        self.assertIsInstance(std_time, float)

    def test_load_image_async(self):
        """Test the load_image_async function"""
        # Test with local file
        img = asyncio.run(load_image_async(self.test_image_path))
        self.assertIsInstance(img, np.ndarray)

        # Test with invalid file
        with self.assertRaises(ValueError):
            asyncio.run(load_image_async("invalid_path.jpg"))

    def test_save_segmentation_results(self):
        """Test the save_segmentation_results function"""
        model = SegmentationModel(self.dummy_model)
        result = asyncio.run(save_segmentation_results(
            model, self.test_image_path, self.test_output_dir
        ))
        self.assertIsInstance(result, dict)
        self.assertIn("mask_path", result)
        self.assertTrue(os.path.exists(result["mask_path"]))

    def test_batch_process_images(self):
        """Test the batch_process_images function"""
        model_path = "dummy_model.pt"
        torch.save(self.dummy_model, model_path)

        try:
            result = asyncio.run(batch_process_images(
                model_path, [self.test_image_path], self.test_output_dir
            ))
            self.assertIsInstance(result, dict)
            self.assertIn("total", result)
            self.assertIn("successful", result)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_get_images_from_dir(self):
        """Test the get_images_from_dir function"""
        test_dir = tempfile.mkdtemp()
        try:
            # Create test images
            img1 = Path(test_dir) / "test1.jpg"
            img2 = Path(test_dir) / "test2.png"
            Image.new('RGB', (100, 100)).save(img1)
            Image.new('RGB', (100, 100)).save(img2)

            # Test with valid extensions
            images = get_images_from_dir(test_dir, ["jpg", "png"])
            self.assertEqual(len(images), 2)

            # Test with invalid extensions
            images = get_images_from_dir(test_dir, ["gif"])
            self.assertEqual(len(images), 0)
        finally:
            if os.path.exists(test_dir):
                for root, dirs, files in os.walk(test_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(test_dir)

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        model = SegmentationModel(self.dummy_model)

        # Test empty input
        with self.assertRaises(ValueError):
            model.process_image("")

        # Test invalid image URL
        with self.assertRaises(ValueError):
            asyncio.run(load_image_async("http://invalid.url/image.jpg"))

        # Test batch processing with empty list
        with self.assertRaises(ValueError):
            asyncio.run(batch_process_images("dummy_model.pt", [], self.test_output_dir))

        # Test get_images_from_dir with non-existent directory
        with self.assertRaises(ValueError):
            get_images_from_dir("non_existent_dir", ["jpg"])

if __name__ == "__main__":
    unittest.main()

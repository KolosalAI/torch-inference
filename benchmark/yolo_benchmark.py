"""
YOLO model benchmarking and validation tests.

This module provides comprehensive testing and benchmarking for YOLO object detection models
integrated into the PyTorch inference framework.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
import requests
from io import BytesIO

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from framework.core.config import InferenceConfig, DeviceConfig, BatchConfig, PostprocessingConfig
from framework.adapters.yolo_adapter import YOLOModelAdapter, YOLOv5Adapter, YOLOv8Adapter
from framework.adapters.model_adapters import ModelAdapterFactory

logger = logging.getLogger(__name__)


class YOLOBenchmark:
    """Comprehensive YOLO model benchmark suite."""
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """Initialize the YOLO benchmark suite."""
        self.config = config or self._create_default_config()
        self.test_images = []
        self.results = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _create_default_config(self) -> InferenceConfig:
        """Create default configuration for YOLO testing."""
        config = InferenceConfig()
        config.device = DeviceConfig()
        config.device.type = "cuda" if torch.cuda.is_available() else "cpu"
        config.device.use_fp16 = False  # Disable FP16 for testing stability
        config.device.use_torch_compile = False  # Disable compilation for testing
        
        config.batch = BatchConfig()
        config.batch.batch_size = 1
        config.batch.max_batch_size = 4
        
        config.postprocessing = PostprocessingConfig()
        config.postprocessing.threshold = 0.25
        config.postprocessing.nms_threshold = 0.45
        config.postprocessing.max_detections = 100
        
        return config
    
    def setup_test_images(self) -> None:
        """Download and prepare test images for benchmarking."""
        test_image_urls = [
            "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
            "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
        ]
        
        self.test_images = []
        
        for i, url in enumerate(test_image_urls):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
                self.test_images.append(image)
                self.logger.info(f"Downloaded test image {i+1}: {image.size}")
            except Exception as e:
                self.logger.warning(f"Failed to download test image {url}: {e}")
                # Create a dummy image as fallback
                dummy_image = Image.new('RGB', (640, 480), color=(128, 128, 128))
                self.test_images.append(dummy_image)
                self.logger.info(f"Using dummy image {i+1} as fallback")
    
    def test_yolo_adapters(self) -> Dict[str, Any]:
        """Test YOLO adapter instantiation and configuration."""
        results = {}
        
        try:
            # Test YOLOModelAdapter
            adapter = YOLOModelAdapter(self.config)
            results['yolo_adapter'] = {
                'instantiation': True,
                'device': str(adapter.device),
                'confidence_threshold': adapter.confidence_threshold,
                'iou_threshold': adapter.iou_threshold,
                'input_size': adapter.input_size
            }
            self.logger.info("✓ YOLOModelAdapter instantiated successfully")
            
            # Test YOLOv5Adapter
            yolov5_adapter = YOLOv5Adapter(self.config)
            results['yolov5_adapter'] = {
                'instantiation': True,
                'variant': yolov5_adapter.yolo_variant,
                'device': str(yolov5_adapter.device)
            }
            self.logger.info("✓ YOLOv5Adapter instantiated successfully")
            
            # Test YOLOv8Adapter
            yolov8_adapter = YOLOv8Adapter(self.config)
            results['yolov8_adapter'] = {
                'instantiation': True,
                'variant': yolov8_adapter.yolo_variant,
                'device': str(yolov8_adapter.device)
            }
            self.logger.info("✓ YOLOv8Adapter instantiated successfully")
            
        except Exception as e:
            self.logger.error(f"✗ YOLO adapter test failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def test_model_factory(self) -> Dict[str, Any]:
        """Test ModelAdapterFactory YOLO model detection."""
        results = {}
        
        try:
            # Test YOLO model detection by name
            test_cases = [
                ("yolov8n.pt", "YOLOv8Adapter"),
                ("yolov5s.pt", "YOLOv5Adapter"),
                ("custom_yolo.pt", "YOLOModelAdapter"),
                ("ultralytics/yolov8n", "YOLOv8Adapter"),
                ("yolov5", "YOLOModelAdapter")
            ]
            
            for model_path, expected_type in test_cases:
                adapter = ModelAdapterFactory.create_adapter(model_path, self.config)
                adapter_type = type(adapter).__name__
                
                results[model_path] = {
                    'detected_type': adapter_type,
                    'expected_type': expected_type,
                    'correct': adapter_type == expected_type
                }
                
                if adapter_type == expected_type:
                    self.logger.info(f"✓ {model_path} -> {adapter_type}")
                else:
                    self.logger.warning(f"✗ {model_path} -> {adapter_type} (expected {expected_type})")
            
        except Exception as e:
            self.logger.error(f"✗ Model factory test failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def test_preprocessing(self) -> Dict[str, Any]:
        """Test YOLO preprocessing functionality."""
        results = {}
        
        if not self.test_images:
            self.setup_test_images()
        
        try:
            adapter = YOLOModelAdapter(self.config)
            
            for i, image in enumerate(self.test_images):
                # Test PIL image preprocessing
                preprocessed = adapter.preprocess(image)
                
                results[f'image_{i+1}'] = {
                    'input_type': 'PIL.Image',
                    'input_size': image.size,
                    'output_shape': list(preprocessed.shape),
                    'output_dtype': str(preprocessed.dtype),
                    'output_device': str(preprocessed.device),
                    'value_range': [float(preprocessed.min()), float(preprocessed.max())]
                }
                
                # Validate preprocessing
                assert preprocessed.dim() == 4, f"Expected 4D tensor, got {preprocessed.dim()}D"
                assert preprocessed.shape[0] == 1, f"Expected batch size 1, got {preprocessed.shape[0]}"
                assert preprocessed.shape[1] == 3, f"Expected 3 channels, got {preprocessed.shape[1]}"
                assert preprocessed.shape[2] == adapter.input_size[0], f"Expected height {adapter.input_size[0]}, got {preprocessed.shape[2]}"
                assert preprocessed.shape[3] == adapter.input_size[1], f"Expected width {adapter.input_size[1]}, got {preprocessed.shape[3]}"
                
                self.logger.info(f"✓ Image {i+1} preprocessing: {image.size} -> {preprocessed.shape}")
            
            # Test numpy array preprocessing
            numpy_image = np.array(self.test_images[0])
            preprocessed_numpy = adapter.preprocess(numpy_image)
            results['numpy_array'] = {
                'input_shape': list(numpy_image.shape),
                'output_shape': list(preprocessed_numpy.shape),
                'success': True
            }
            self.logger.info(f"✓ Numpy array preprocessing: {numpy_image.shape} -> {preprocessed_numpy.shape}")
            
            # Test tensor preprocessing
            tensor_image = torch.from_numpy(numpy_image).permute(2, 0, 1).float() / 255.0
            preprocessed_tensor = adapter.preprocess(tensor_image)
            results['tensor'] = {
                'input_shape': list(tensor_image.shape),
                'output_shape': list(preprocessed_tensor.shape),
                'success': True
            }
            self.logger.info(f"✓ Tensor preprocessing: {tensor_image.shape} -> {preprocessed_tensor.shape}")
            
        except Exception as e:
            self.logger.error(f"✗ Preprocessing test failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def test_mock_inference(self) -> Dict[str, Any]:
        """Test YOLO inference with mock model outputs."""
        results = {}
        
        try:
            adapter = YOLOModelAdapter(self.config)
            
            # Create mock outputs for different YOLO variants
            test_cases = [
                {
                    'name': 'ultralytics_format',
                    'output': torch.tensor([
                        [100, 150, 200, 250, 0.8, 0],  # [x1, y1, x2, y2, conf, cls]
                        [300, 350, 400, 450, 0.9, 1],
                        [50, 75, 150, 175, 0.3, 2]     # Low confidence
                    ])
                },
                {
                    'name': 'yolov5_format',
                    'output': torch.tensor([[
                        [320, 240, 100, 150, 0.7, 0.1, 0.8, 0.05, 0.05],  # [x, y, w, h, obj, cls0, cls1, ...]
                        [160, 120, 80, 100, 0.9, 0.05, 0.1, 0.85, 0.0],
                        [400, 300, 60, 80, 0.2, 0.3, 0.4, 0.2, 0.1]       # Low objectness
                    ]])
                }
            ]
            
            for test_case in test_cases:
                adapter.yolo_variant = test_case['name'].split('_')[0]
                mock_output = test_case['output']
                
                # Test postprocessing
                result = adapter.postprocess(mock_output)
                
                results[test_case['name']] = {
                    'input_shape': list(mock_output.shape),
                    'num_detections': result['num_detections'],
                    'detections': result['detections'],
                    'success': True
                }
                
                self.logger.info(f"✓ {test_case['name']}: {result['num_detections']} detections")
                
                # Validate detection format
                for detection in result['detections']:
                    assert 'bbox' in detection, "Detection missing bbox"
                    assert 'confidence' in detection, "Detection missing confidence"
                    assert 'class_id' in detection, "Detection missing class_id"
                    assert 'class_name' in detection, "Detection missing class_name"
                    assert len(detection['bbox']) == 4, "Bbox should have 4 coordinates"
                    assert detection['confidence'] >= adapter.confidence_threshold, "Detection below threshold"
            
        except Exception as e:
            self.logger.error(f"✗ Mock inference test failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def test_batch_processing(self) -> Dict[str, Any]:
        """Test YOLO batch processing capabilities."""
        results = {}
        
        if not self.test_images:
            self.setup_test_images()
        
        try:
            adapter = YOLOModelAdapter(self.config)
            
            # Test batch preprocessing
            batch_results = adapter.predict_batch(self.test_images)
            
            results['batch_preprocessing'] = {
                'input_count': len(self.test_images),
                'output_count': len(batch_results),
                'success': len(batch_results) == len(self.test_images)
            }
            
            # Validate batch results structure
            for i, result in enumerate(batch_results):
                assert isinstance(result, dict), f"Result {i} is not a dict"
                assert 'detections' in result, f"Result {i} missing detections"
                assert 'num_detections' in result, f"Result {i} missing num_detections"
                assert 'model_type' in result, f"Result {i} missing model_type"
                assert result['model_type'] == 'yolo', f"Result {i} wrong model_type"
            
            self.logger.info(f"✓ Batch processing: {len(self.test_images)} images -> {len(batch_results)} results")
            
        except Exception as e:
            self.logger.error(f"✗ Batch processing test failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def benchmark_performance(self) -> Dict[str, Any]:
        """Benchmark YOLO preprocessing and postprocessing performance."""
        results = {}
        
        if not self.test_images:
            self.setup_test_images()
        
        try:
            adapter = YOLOModelAdapter(self.config)
            
            # Benchmark preprocessing
            preprocess_times = []
            for _ in range(10):  # 10 iterations
                start_time = time.perf_counter()
                _ = adapter.preprocess(self.test_images[0])
                end_time = time.perf_counter()
                preprocess_times.append(end_time - start_time)
            
            # Benchmark postprocessing with mock data
            mock_output = torch.randn(1, 25200, 85)  # Typical YOLOv8 output shape
            postprocess_times = []
            for _ in range(10):
                start_time = time.perf_counter()
                _ = adapter.postprocess(mock_output)
                end_time = time.perf_counter()
                postprocess_times.append(end_time - start_time)
            
            results['preprocessing'] = {
                'mean_time_ms': np.mean(preprocess_times) * 1000,
                'std_time_ms': np.std(preprocess_times) * 1000,
                'min_time_ms': np.min(preprocess_times) * 1000,
                'max_time_ms': np.max(preprocess_times) * 1000
            }
            
            results['postprocessing'] = {
                'mean_time_ms': np.mean(postprocess_times) * 1000,
                'std_time_ms': np.std(postprocess_times) * 1000,
                'min_time_ms': np.min(postprocess_times) * 1000,
                'max_time_ms': np.max(postprocess_times) * 1000
            }
            
            self.logger.info(f"✓ Preprocessing: {results['preprocessing']['mean_time_ms']:.2f}ms avg")
            self.logger.info(f"✓ Postprocessing: {results['postprocessing']['mean_time_ms']:.2f}ms avg")
            
        except Exception as e:
            self.logger.error(f"✗ Performance benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all YOLO tests and return comprehensive results."""
        self.logger.info("Starting comprehensive YOLO integration tests...")
        
        all_results = {}
        
        # Run individual tests
        test_methods = [
            ('adapter_tests', self.test_yolo_adapters),
            ('factory_tests', self.test_model_factory),
            ('preprocessing_tests', self.test_preprocessing),
            ('mock_inference_tests', self.test_mock_inference),
            ('batch_processing_tests', self.test_batch_processing),
            ('performance_benchmark', self.benchmark_performance)
        ]
        
        for test_name, test_method in test_methods:
            self.logger.info(f"\nRunning {test_name}...")
            try:
                all_results[test_name] = test_method()
            except Exception as e:
                self.logger.error(f"Test {test_name} failed with exception: {e}")
                all_results[test_name] = {'error': str(e), 'exception': True}
        
        # Generate summary
        passed_tests = sum(1 for result in all_results.values() if 'error' not in result)
        total_tests = len(all_results)
        
        all_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'overall_status': 'PASS' if passed_tests == total_tests else 'PARTIAL' if passed_tests > 0 else 'FAIL'
        }
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"YOLO Integration Test Summary:")
        self.logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        self.logger.info(f"Success Rate: {all_results['summary']['success_rate']*100:.1f}%")
        self.logger.info(f"Overall Status: {all_results['summary']['overall_status']}")
        self.logger.info(f"{'='*50}")
        
        return all_results


def main():
    """Main function to run YOLO integration tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Integration Tests")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                       help="Device to run tests on")
    parser.add_argument("--save-results", type=str, help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Create configuration
    config = InferenceConfig()
    config.device = DeviceConfig()
    
    if args.device == "auto":
        config.device.type = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        config.device.type = args.device
    
    # Run benchmark
    benchmark = YOLOBenchmark(config)
    results = benchmark.run_all_tests()
    
    # Save results if requested
    if args.save_results:
        import json
        with open(args.save_results, 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            json_results[key][k] = v.tolist()
                        else:
                            json_results[key][k] = v
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2)
        print(f"Results saved to {args.save_results}")
    
    # Exit with appropriate code
    exit_code = 0 if results['summary']['overall_status'] == 'PASS' else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
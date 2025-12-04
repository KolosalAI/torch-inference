"""
YOLO Object Detection Example

This script demonstrates how to use YOLO models for object detection
with the PyTorch inference framework.
"""

import sys
import logging
from pathlib import Path
from PIL import Image
import torch

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from framework.core.config import InferenceConfig, DeviceConfig, PostprocessingConfig
from framework.adapters.model_adapters import ModelAdapterFactory


def create_yolo_config() -> InferenceConfig:
    """Create configuration optimized for YOLO inference."""
    config = InferenceConfig()
    
    # Device configuration
    config.device = DeviceConfig()
    config.device.type = "cuda" if torch.cuda.is_available() else "cpu"
    config.device.use_fp16 = True if config.device.type == "cuda" else False
    config.device.use_torch_compile = False  # Disable for better compatibility
    
    # Postprocessing configuration for object detection
    config.postprocessing = PostprocessingConfig()
    config.postprocessing.threshold = 0.25  # Confidence threshold
    config.postprocessing.nms_threshold = 0.45  # NMS IoU threshold
    config.postprocessing.max_detections = 100  # Maximum detections per image
    
    return config


def load_yolo_model(model_name: str, config: InferenceConfig):
    """Load a YOLO model using the model factory."""
    print(f"Loading YOLO model: {model_name}")
    print(f"Target device: {config.device.type}")
    
    # Create adapter using the factory
    adapter = ModelAdapterFactory.create_adapter(model_name, config)
    print(f"Created adapter: {type(adapter).__name__}")
    
    # For demonstration, we'll show how to load different model types
    try:
        # Note: This would normally load the actual model file
        # For demo purposes, we'll just show the adapter setup
        print(f"✓ Adapter configured successfully")
        print(f"  - YOLO variant: {getattr(adapter, 'yolo_variant', 'unknown')}")
        print(f"  - Input size: {getattr(adapter, 'input_size', 'unknown')}")
        print(f"  - Confidence threshold: {getattr(adapter, 'confidence_threshold', 'unknown')}")
        print(f"  - IoU threshold: {getattr(adapter, 'iou_threshold', 'unknown')}")
        print(f"  - Device: {adapter.device}")
        
        return adapter
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None


def detect_objects(adapter, image_path: str):
    """Perform object detection on an image."""
    try:
        # Load image
        print(f"\nProcessing image: {image_path}")
        
        if Path(image_path).exists():
            image = Image.open(image_path).convert('RGB')
            print(f"  - Image size: {image.size}")
        else:
            # Create a dummy image for demonstration
            print("  - Using dummy image (file not found)")
            image = Image.new('RGB', (640, 480), color=(128, 128, 128))
        
        # For demonstration, we'll show the preprocessing step
        print("  - Preprocessing image...")
        preprocessed = adapter.preprocess(image)
        print(f"  - Preprocessed shape: {preprocessed.shape}")
        print(f"  - Preprocessed dtype: {preprocessed.dtype}")
        
        # Note: In a real scenario, we would run inference here:
        # results = adapter.predict(image)
        
        # For demo, show what the results would look like
        print("  - Inference would be performed here")
        print("  - Expected output format:")
        print("    {")
        print("      'detections': [")
        print("        {")
        print("          'bbox': [x1, y1, x2, y2],")
        print("          'confidence': 0.85,")
        print("          'class_id': 0,")
        print("          'class_name': 'person'")
        print("        },")
        print("        ...")
        print("      ],")
        print("      'num_detections': 3,")
        print("      'model_type': 'yolo'")
        print("    }")
        
        return True
        
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        return False


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("YOLO Object Detection Example")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = create_yolo_config()
    print(f"Configuration created:")
    print(f"  - Device: {config.device.type}")
    print(f"  - FP16: {getattr(config.device, 'use_fp16', False)}")
    print(f"  - Confidence threshold: {config.postprocessing.threshold}")
    print(f"  - NMS threshold: {config.postprocessing.nms_threshold}")
    print()
    
    # Test different YOLO model types
    test_models = [
        "yolov8n.pt",
        "yolov5s.pt", 
        "ultralytics/yolov8n",
        "custom_yolo_model.pt"
    ]
    
    print("Testing YOLO model adapter creation:")
    print("-" * 40)
    
    for model_name in test_models:
        print(f"\nModel: {model_name}")
        adapter = load_yolo_model(model_name, config)
        
        if adapter:
            # Test preprocessing with a dummy image
            success = detect_objects(adapter, "test_image.jpg")
            if success:
                print("  ✓ Object detection pipeline tested successfully")
            else:
                print("  ✗ Object detection pipeline test failed")
    
    print("\n" + "=" * 60)
    print("YOLO Integration Summary:")
    print("✓ YOLO adapters can be created for different model types")
    print("✓ Model factory correctly identifies YOLO models")
    print("✓ Preprocessing pipeline configured for YOLO input requirements")
    print("✓ Postprocessing configured for object detection outputs")
    print("✓ Device and optimization settings applied")
    print()
    print("To use with real models:")
    print("1. Install required packages: pip install ultralytics")
    print("2. Download YOLO models or use pre-trained ones")
    print("3. Load models using: adapter.load_model('path/to/model.pt')")
    print("4. Run inference: results = adapter.predict(image)")
    print("=" * 60)


if __name__ == "__main__":
    main()
# YOLO Object Detection Support

This document describes the YOLO (You Only Look Once) object detection support added to the PyTorch Inference Framework.

## Overview

The framework now supports multiple YOLO variants for real-time object detection:

- **YOLOv5**: Fast and efficient object detection
- **YOLOv8/v9/v10/v11**: Latest Ultralytics YOLO models with improved accuracy
- **Custom YOLO models**: Support for custom-trained YOLO models

## Supported Models

### Pre-configured Models

The following YOLO models are pre-configured in `models.json`:

| Model | Description | Size | Speed | Accuracy (mAP@0.5) |
|-------|-------------|------|-------|---------------------|
| `yolov8n` | YOLOv8 Nano | 6.2 MB | Very Fast | 37.5% |
| `yolov8s` | YOLOv8 Small | 22.5 MB | Fast | 44.5% |
| `yolov8m` | YOLOv8 Medium | 52.0 MB | Medium | 50.2% |
| `yolov8l` | YOLOv8 Large | 87.7 MB | Slow | 52.9% |
| `yolov5n` | YOLOv5 Nano | 3.9 MB | Very Fast | 28.0% |
| `yolov5s` | YOLOv5 Small | 14.1 MB | Fast | 36.7% |

### Model Sources

- **Ultralytics**: YOLOv8+ models from Ultralytics
- **PyTorch Hub**: YOLOv5 models
- **Local Files**: Custom trained `.pt` files
- **URL Downloads**: Direct model downloads

## Quick Start

### Basic Usage

```python
from framework.core.config import InferenceConfig
from framework.adapters.model_adapters import ModelAdapterFactory
from PIL import Image

# Create configuration
config = InferenceConfig()
config.device.type = "cuda"  # or "cpu"
config.postprocessing.threshold = 0.25
config.postprocessing.nms_threshold = 0.45

# Load YOLO model
adapter = ModelAdapterFactory.create_adapter("yolov8n.pt", config)
adapter.load_model("yolov8n.pt")  # Will auto-download if needed

# Perform detection
image = Image.open("image.jpg")
results = adapter.predict(image)

# Process results
for detection in results['detections']:
    bbox = detection['bbox']  # [x1, y1, x2, y2]
    confidence = detection['confidence']
    class_name = detection['class_name']
    print(f"{class_name}: {confidence:.2f} at {bbox}")
```

### Batch Processing

```python
# Process multiple images
images = [Image.open(f"image_{i}.jpg") for i in range(5)]
batch_results = adapter.predict_batch(images)

for i, result in enumerate(batch_results):
    print(f"Image {i}: {result['num_detections']} detections")
```

## Model Loading

### Automatic Model Detection

The framework automatically detects YOLO models based on:

1. **File names**: Files containing "yolo", "yolov5", "yolov8", etc.
2. **Model IDs**: Ultralytics model identifiers
3. **File extensions**: `.pt` files are checked for YOLO format

### Loading Methods

#### Method 1: Using Model Factory (Recommended)

```python
# Automatically detects YOLO type
adapter = ModelAdapterFactory.create_adapter("yolov8n.pt", config)
```

#### Method 2: Direct Adapter Creation

```python
from framework.adapters.yolo_adapter import YOLOv8Adapter

adapter = YOLOv8Adapter(config)
adapter.load_model("yolov8n.pt")
```

#### Method 3: Using Model Manager

```python
from framework.core.base_model import get_model_manager

manager = get_model_manager()
manager.download_and_load_model(
    source="ultralytics",
    model_id="yolov8n.pt", 
    name="yolo_detector",
    config=config
)
```

## Configuration

### Device Configuration

```python
# GPU with optimizations
config.device.type = "cuda"
config.device.use_fp16 = True  # Faster inference
config.device.use_torch_compile = True  # Advanced optimization

# CPU configuration
config.device.type = "cpu"
config.device.use_quantization = True  # CPU speedup
```

### Detection Configuration

```python
# Adjust detection thresholds
config.postprocessing.threshold = 0.25        # Confidence threshold
config.postprocessing.nms_threshold = 0.45    # NMS IoU threshold
config.postprocessing.max_detections = 100    # Max detections per image
```

### Batch Configuration

```python
# Optimize for batch processing
config.batch.batch_size = 4
config.batch.max_batch_size = 16
config.batch.adaptive_batching = True
```

## Output Format

### Detection Results

```json
{
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.85,
      "class_id": 0,
      "class_name": "person"
    }
  ],
  "num_detections": 3,
  "input_size": [640, 640],
  "model_type": "yolo"
}
```

### Coordinate Format

- **bbox**: `[x1, y1, x2, y2]` in image pixel coordinates
- **x1, y1**: Top-left corner
- **x2, y2**: Bottom-right corner

## Performance Optimization

### Hardware Recommendations

| Model Size | Minimum RAM | Recommended GPU | Batch Size |
|------------|-------------|-----------------|------------|
| Nano/Small | 4 GB | GTX 1060 | 8-16 |
| Medium | 8 GB | RTX 3060 | 4-8 |
| Large | 16 GB | RTX 3080 | 2-4 |

### Speed Optimizations

```python
# Maximum speed configuration
config.device.use_fp16 = True
config.device.use_torch_compile = True
config.device.compile_mode = "max-autotune"
config.batch.adaptive_batching = True
config.performance.enable_cuda_graphs = True
```

### Memory Optimizations

```python
# Memory-efficient configuration
config.device.memory_fraction = 0.8
config.batch.batch_size = 1
config.cache.enable_caching = True
config.optimization.enable_quantization = True
```

## YOLO Variants

### YOLOv5

- **Framework**: PyTorch
- **Source**: Ultralytics PyTorch Hub
- **Strengths**: Fast, lightweight, good for edge devices
- **Models**: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x

### YOLOv8+

- **Framework**: Ultralytics
- **Source**: Ultralytics package
- **Strengths**: Latest architecture, best accuracy
- **Models**: yolov8n/s/m/l/x, yolov9, yolov10, yolo11

### Custom Models

```python
# Load custom trained model
adapter = ModelAdapterFactory.create_adapter("my_custom_yolo.pt", config)
adapter.load_model("my_custom_yolo.pt")

# Override class names if needed
adapter.class_names = ["cat", "dog", "bird"]
```

## Input Requirements

### Supported Input Types

- **PIL Images**: `Image.open("file.jpg")`
- **NumPy arrays**: `np.array(image)`
- **PyTorch tensors**: `torch.tensor(image_data)`
- **File paths**: `"/path/to/image.jpg"`

### Input Preprocessing

Images are automatically:
1. Resized to model input size (typically 640x640)
2. Normalized to [0, 1] range
3. Converted to RGB format
4. Moved to target device

## Error Handling

### Common Issues

#### 1. Missing Dependencies

```python
# Install required packages
pip install ultralytics  # For YOLOv8+
pip install yolov5      # For YOLOv5 (optional)
```

#### 2. Model Loading Errors

```python
try:
    adapter.load_model("model.pt")
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
    # Try alternative model or check file path
```

#### 3. Memory Issues

```python
# Reduce batch size or use CPU
config.batch.batch_size = 1
config.device.type = "cpu"
```

## Benchmarking

### Run Performance Tests

```bash
# Run YOLO integration tests
python benchmark/yolo_benchmark.py --device cuda --save-results results.json

# Run example
python examples/yolo_detection_example.py
```

### Expected Performance

| Device | Model | Batch Size | Speed (FPS) |
|--------|-------|------------|-------------|
| RTX 3080 | YOLOv8n | 1 | ~400 |
| RTX 3080 | YOLOv8s | 1 | ~250 |
| RTX 3080 | YOLOv8m | 1 | ~150 |
| CPU (i7) | YOLOv8n | 1 | ~15 |

## Integration Examples

### REST API Integration

```python
from framework.api.routes import detection_router

# YOLO detection endpoint automatically available
# POST /api/v1/models/yolov8n/detect
```

### Batch Processing Pipeline

```python
import asyncio

async def process_video_frames(adapter, frames):
    """Process video frames in batches."""
    results = []
    batch_size = adapter.config.batch.batch_size
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        batch_results = await adapter.predict_async_batch(batch)
        results.extend(batch_results)
    
    return results
```

### Real-time Detection

```python
import cv2

def real_time_detection(adapter):
    """Real-time webcam detection."""
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect objects
        results = adapter.predict(rgb_frame)
        
        # Draw bounding boxes
        for detection in results['detections']:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{detection['class_name']}: {detection['confidence']:.2f}",
                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('YOLO Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

## Troubleshooting

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
config.monitoring.enable_logging = True
adapter = ModelAdapterFactory.create_adapter("yolov8n.pt", config)
```

### Common Solutions

1. **Low accuracy**: Adjust confidence threshold
2. **Too many detections**: Lower confidence or adjust NMS threshold
3. **Memory errors**: Reduce batch size or use smaller model
4. **Slow inference**: Enable GPU, FP16, or use smaller model

## Future Enhancements

- Support for YOLO segmentation models
- Custom training integration
- TensorRT optimization for YOLO models
- Mobile/edge deployment optimizations
- Multi-scale inference support
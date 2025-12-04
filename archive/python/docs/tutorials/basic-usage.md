# Basic Usage Tutorial

Learn the fundamentals of using the PyTorch Inference Framework through practical examples.

## 📚 Prerequisites

- Basic Python knowledge
- Familiarity with PyTorch concepts
- Completed [installation](../guides/installation.md)

## 🎯 Learning Objectives

By the end of this tutorial, you will:
- ✅ Understand framework initialization
- ✅ Load and use models for inference
- ✅ Handle different input types and formats
- ✅ Process outputs and responses
- ✅ Implement basic error handling

## 🚀 Getting Started

### 1. Framework Initialization

```python
# basic_usage.py
from framework import TorchInferenceFramework
import torch

# Create framework instance with default settings
framework = TorchInferenceFramework()

# Verify initialization
print(f"Framework initialized: {framework.is_ready()}")
print(f"Device: {framework.device}")
print(f"Optimizations: {framework.list_optimizations()}")
```

**Output:**
```
Framework initialized: True
Device: cuda:0
Optimizations: ['torch_compile', 'memory_pooling', 'cuda_graphs']
```

### 2. Loading Your First Model

```python
# Load a pre-trained model
model_path = "path/to/your/model.pth"

# Method 1: Load PyTorch model
model = framework.load_model(
    model_path=model_path,
    model_type="pytorch",
    optimize=True  # Apply automatic optimizations
)

# Method 2: Load with custom configuration
from framework.core.config import ModelConfig

model_config = ModelConfig(
    model_path=model_path,
    batch_size=4,
    use_fp16=True,
    enable_dynamic_batching=True
)

model = framework.load_model_with_config(model_config)

print(f"Model loaded: {model.model_id}")
print(f"Model device: {model.device}")
print(f"Optimizations applied: {model.optimizations}")
```

### 3. Basic Inference

```python
# Prepare input data
input_data = torch.randn(1, 3, 224, 224)  # Example image tensor

# Perform inference
result = framework.predict(
    model_id=model.model_id,
    input_data=input_data
)

print(f"Prediction result: {result}")
print(f"Output shape: {result['output'].shape}")
print(f"Inference time: {result['inference_time_ms']} ms")
```

**Example Output:**
```python
{
    'output': tensor([[0.1234, -0.5678, 0.9012, ...]]),
    'inference_time_ms': 23.45,
    'model_id': 'my_model_v1',
    'batch_size': 1,
    'device': 'cuda:0'
}
```

## 🔄 Working with Different Input Types

### Tensor Inputs

```python
# Single tensor
single_tensor = torch.randn(1, 100)
result = framework.predict("text_classifier", single_tensor)

# Multiple tensors
input_dict = {
    'input_ids': torch.tensor([[101, 2054, 2003, 102]]),
    'attention_mask': torch.tensor([[1, 1, 1, 1]])
}
result = framework.predict("bert_model", input_dict)

# Batch of tensors
batch_tensors = torch.randn(4, 3, 224, 224)
result = framework.predict("image_classifier", batch_tensors)
```

### NumPy Arrays

```python
import numpy as np

# Convert NumPy to tensor automatically
numpy_input = np.random.randn(1, 784)
result = framework.predict("mnist_classifier", numpy_input)

# Multiple NumPy arrays
numpy_dict = {
    'features': np.random.randn(1, 100),
    'metadata': np.array([[1, 0, 1]])
}
result = framework.predict("multi_input_model", numpy_dict)
```

### Lists and Raw Data

```python
# List input (automatically converted)
list_input = [[1.0, 2.0, 3.0, 4.0]]
result = framework.predict("regression_model", list_input)

# Text input (for NLP models)
text_input = "Hello, how are you today?"
result = framework.predict("sentiment_analyzer", text_input)

# Image path (automatically loaded)
image_path = "path/to/image.jpg"
result = framework.predict("image_classifier", image_path)
```

## 🎛️ Batch Processing

### Static Batching

```python
# Prepare batch data
batch_size = 4
batch_input = torch.randn(batch_size, 3, 224, 224)

# Process batch
batch_result = framework.predict_batch(
    model_id="image_classifier",
    batch_input=batch_input,
    batch_size=batch_size
)

print(f"Batch processed: {len(batch_result['outputs'])} items")
for i, output in enumerate(batch_result['outputs']):
    print(f"Item {i}: {output.shape}")
```

### Dynamic Batching

```python
# Enable dynamic batching
framework.enable_dynamic_batching(
    max_batch_size=8,
    timeout_ms=50
)

# Send individual requests (automatically batched)
import asyncio

async def process_request(data):
    return await framework.predict_async("model_id", data)

# Multiple concurrent requests get automatically batched
tasks = [
    process_request(torch.randn(1, 100)) 
    for _ in range(10)
]

results = await asyncio.gather(*tasks)
print(f"Processed {len(results)} requests with automatic batching")
```

## 🎯 Model Management

### Multiple Models

```python
# Load multiple models
models = {}

# Classification model
models['classifier'] = framework.load_model(
    "models/classifier.pth",
    model_type="classification"
)

# Regression model
models['regressor'] = framework.load_model(
    "models/regressor.pth",
    model_type="regression"
)

# NLP model
models['nlp'] = framework.load_model(
    "models/bert.pth",
    model_type="transformer"
)

# Use different models
classification_result = framework.predict(
    models['classifier'].model_id, 
    image_data
)

regression_result = framework.predict(
    models['regressor'].model_id, 
    numerical_data
)

nlp_result = framework.predict(
    models['nlp'].model_id, 
    text_data
)
```

### Model Information

```python
# Get model information
model_info = framework.get_model_info("classifier")
print(f"Model info: {model_info}")

# List all loaded models
all_models = framework.list_models()
for model_id, info in all_models.items():
    print(f"{model_id}: {info['model_type']} on {info['device']}")

# Model statistics
stats = framework.get_model_stats("classifier")
print(f"Inference count: {stats['inference_count']}")
print(f"Average time: {stats['avg_inference_time_ms']} ms")
print(f"Memory usage: {stats['memory_usage_mb']} MB")
```

## ⚡ Performance Optimization

### Automatic Optimization

```python
# Enable all optimizations
optimized_model = framework.load_model(
    "model.pth",
    optimize=True,
    optimization_level="aggressive"
)

# Check applied optimizations
print("Applied optimizations:")
for opt in optimized_model.optimizations:
    print(f"  - {opt}")
```

### Custom Optimization

```python
from framework.core.config import OptimizationConfig

# Configure specific optimizations
opt_config = OptimizationConfig(
    use_fp16=True,
    use_torch_compile=True,
    compile_mode="max-autotune",
    enable_cuda_graphs=True,
    memory_pooling=True
)

# Apply to model
framework.apply_optimizations("model_id", opt_config)
```

### Benchmarking

```python
# Benchmark model performance
benchmark_results = framework.benchmark_model(
    model_id="classifier",
    input_shapes=[(1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224)],
    num_iterations=100
)

print("Benchmark Results:")
for batch_size, results in benchmark_results.items():
    print(f"Batch size {batch_size}:")
    print(f"  Average time: {results['avg_time_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput_fps']:.1f} FPS")
    print(f"  Memory usage: {results['memory_mb']:.1f} MB")
```

## 🔍 Error Handling

### Basic Error Handling

```python
try:
    # Load model
    model = framework.load_model("path/to/model.pth")
    
    # Perform inference
    result = framework.predict(model.model_id, input_data)
    
except FileNotFoundError:
    print("❌ Model file not found")
except ValueError as e:
    print(f"❌ Invalid input: {e}")
except RuntimeError as e:
    print(f"❌ Runtime error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
```

### Graceful Fallbacks

```python
# Try optimized inference with fallback
def safe_predict(model_id, input_data):
    try:
        # Try with optimizations
        return framework.predict(model_id, input_data)
    except Exception as e:
        print(f"⚠️  Optimization failed: {e}")
        
        try:
            # Fallback to standard inference
            return framework.predict(
                model_id, 
                input_data, 
                use_optimization=False
            )
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return None

result = safe_predict("model_id", input_data)
if result:
    print("✅ Inference successful")
```

### Input Validation

```python
def validate_input(input_data, expected_shape):
    """Validate input data before inference."""
    if isinstance(input_data, torch.Tensor):
        if input_data.shape[1:] != expected_shape[1:]:
            raise ValueError(
                f"Expected shape {expected_shape}, got {input_data.shape}"
            )
    elif isinstance(input_data, np.ndarray):
        input_data = torch.from_numpy(input_data).float()
    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}")
    
    return input_data

# Use validation
try:
    validated_input = validate_input(input_data, (1, 3, 224, 224))
    result = framework.predict("model_id", validated_input)
except ValueError as e:
    print(f"❌ Input validation failed: {e}")
```

## 📊 Monitoring and Logging

### Basic Monitoring

```python
# Enable monitoring
framework.enable_monitoring()

# Get performance metrics
metrics = framework.get_metrics()
print(f"Total predictions: {metrics['total_predictions']}")
print(f"Average latency: {metrics['avg_latency_ms']} ms")
print(f"Error rate: {metrics['error_rate']*100:.2f}%")

# Model-specific metrics
model_metrics = framework.get_model_metrics("classifier")
print(f"Model throughput: {model_metrics['throughput_fps']} FPS")
```

### Custom Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log predictions
def logged_predict(model_id, input_data):
    logger.info(f"Starting prediction with model {model_id}")
    
    start_time = time.time()
    result = framework.predict(model_id, input_data)
    end_time = time.time()
    
    logger.info(
        f"Prediction completed in {(end_time - start_time)*1000:.2f} ms"
    )
    
    return result
```

## 🎯 Practical Examples

### Example 1: Image Classification

```python
# Image classification pipeline
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load image classifier
classifier = framework.load_model(
    "models/resnet50_classifier.pth",
    model_type="classification"
)

# Preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Classify image
image_path = "examples/cat.jpg"
input_tensor = preprocess_image(image_path)

result = framework.predict(classifier.model_id, input_tensor)
probabilities = torch.softmax(result['output'], dim=1)

# Get top 5 predictions
top5_probs, top5_indices = torch.topk(probabilities, 5)

print("Top 5 predictions:")
for i in range(5):
    class_id = top5_indices[0][i].item()
    probability = top5_probs[0][i].item()
    print(f"  {class_id}: {probability:.3f}")
```

### Example 2: Text Sentiment Analysis

```python
# Text sentiment analysis
from transformers import AutoTokenizer

# Load sentiment model and tokenizer
sentiment_model = framework.load_model(
    "models/bert_sentiment.pth",
    model_type="transformer"
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def analyze_sentiment(text):
    # Tokenize text
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Predict sentiment
    result = framework.predict(sentiment_model.model_id, {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask']
    })
    
    # Process output
    logits = result['output']
    probabilities = torch.softmax(logits, dim=1)
    
    sentiment_labels = ['negative', 'neutral', 'positive']
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()
    
    return {
        'sentiment': sentiment_labels[predicted_class],
        'confidence': confidence,
        'all_scores': {
            label: prob.item() 
            for label, prob in zip(sentiment_labels, probabilities[0])
        }
    }

# Analyze text
text = "I love using this framework! It's so fast and easy."
sentiment = analyze_sentiment(text)

print(f"Text: {text}")
print(f"Sentiment: {sentiment['sentiment']}")
print(f"Confidence: {sentiment['confidence']:.3f}")
```

### Example 3: Batch Processing Pipeline

```python
# Batch processing with progress tracking
from tqdm import tqdm
import os

def process_image_batch(image_dir, batch_size=8):
    """Process all images in a directory."""
    
    # Get all image files
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(image_files), batch_size)):
        batch_files = image_files[i:i + batch_size]
        
        # Prepare batch
        batch_tensors = []
        for file in batch_files:
            image_path = os.path.join(image_dir, file)
            tensor = preprocess_image(image_path)
            batch_tensors.append(tensor)
        
        # Stack into batch
        batch_input = torch.cat(batch_tensors, dim=0)
        
        # Process batch
        batch_result = framework.predict(
            classifier.model_id, 
            batch_input
        )
        
        # Store results
        for j, file in enumerate(batch_files):
            output = batch_result['output'][j]
            probabilities = torch.softmax(output, dim=0)
            top_class = torch.argmax(probabilities).item()
            confidence = probabilities[top_class].item()
            
            results.append({
                'file': file,
                'predicted_class': top_class,
                'confidence': confidence
            })
    
    return results

# Process directory
image_directory = "path/to/images"
batch_results = process_image_batch(image_directory)

# Print summary
print(f"Processed {len(batch_results)} images")
for result in batch_results[:5]:  # Show first 5
    print(f"  {result['file']}: class {result['predicted_class']} "
          f"(confidence: {result['confidence']:.3f})")
```

## 🔄 Asynchronous Processing

### Async Predictions

```python
import asyncio

async def async_pipeline():
    """Async processing pipeline."""
    
    # Enable async mode
    framework.enable_async_mode()
    
    # Multiple async predictions
    tasks = []
    
    for i in range(10):
        input_data = torch.randn(1, 3, 224, 224)
        task = framework.predict_async("classifier", input_data)
        tasks.append(task)
    
    # Wait for all predictions
    results = await asyncio.gather(*tasks)
    
    print(f"Completed {len(results)} async predictions")
    
    return results

# Run async pipeline
results = asyncio.run(async_pipeline())
```

## 🧪 Testing Your Implementation

### Unit Tests

```python
# test_basic_usage.py
import unittest
import torch

class TestBasicUsage(unittest.TestCase):
    
    def setUp(self):
        self.framework = TorchInferenceFramework()
        
    def test_framework_initialization(self):
        """Test framework initialization."""
        self.assertTrue(self.framework.is_ready())
        self.assertIsNotNone(self.framework.device)
    
    def test_model_loading(self):
        """Test model loading."""
        # Create dummy model
        model = torch.nn.Linear(10, 1)
        torch.save(model.state_dict(), "test_model.pth")
        
        # Load with framework
        loaded_model = self.framework.load_model("test_model.pth")
        self.assertIsNotNone(loaded_model)
        
    def test_prediction(self):
        """Test prediction."""
        # Create and load test model
        model = torch.nn.Linear(10, 1)
        torch.save(model.state_dict(), "test_model.pth")
        loaded_model = self.framework.load_model("test_model.pth")
        
        # Test prediction
        input_data = torch.randn(1, 10)
        result = self.framework.predict(loaded_model.model_id, input_data)
        
        self.assertIn('output', result)
        self.assertIn('inference_time_ms', result)
        
    def tearDown(self):
        # Cleanup
        self.framework.cleanup()

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

```python
# test_integration.py
def test_full_pipeline():
    """Test complete inference pipeline."""
    
    # Initialize framework
    framework = TorchInferenceFramework()
    
    # Load model
    model = framework.load_model("path/to/test_model.pth")
    
    # Prepare test data
    test_inputs = [
        torch.randn(1, 3, 224, 224),
        torch.randn(2, 3, 224, 224),
        torch.randn(4, 3, 224, 224)
    ]
    
    # Test different batch sizes
    for i, input_data in enumerate(test_inputs):
        result = framework.predict(model.model_id, input_data)
        
        assert 'output' in result
        assert result['output'].shape[0] == input_data.shape[0]
        assert result['inference_time_ms'] > 0
        
        print(f"✅ Test {i+1} passed")
    
    print("✅ All integration tests passed")

test_full_pipeline()
```

## 📚 Next Steps

Now that you understand the basics, explore these advanced topics:

1. **[Optimization Guide](optimization.md)** - Learn advanced optimization techniques
2. **[Audio Processing Tutorial](audio-processing.md)** - Work with TTS and STT
3. **[Production API Tutorial](production-api.md)** - Deploy as a production API
4. **[Advanced Features Tutorial](advanced-features.md)** - Explore advanced capabilities

## 🔍 Troubleshooting

### Common Issues

**Model Loading Issues:**
```python
# Check model file
import os
if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")

# Check device compatibility
if torch.cuda.is_available():
    print("✅ CUDA available")
else:
    print("⚠️  CUDA not available, using CPU")
```

**Memory Issues:**
```python
# Monitor memory usage
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    print(f"GPU memory used: {torch.cuda.memory_allocated() // 1024**3} GB")

# Reduce batch size if needed
framework.update_config({'batch': {'batch_size': 1}})
```

**Performance Issues:**
```python
# Enable profiling
framework.enable_profiling()

# Check optimization status
model_info = framework.get_model_info("model_id")
print(f"Optimizations: {model_info['optimizations']}")

# Benchmark different configurations
results = framework.benchmark_model("model_id")
```

---

This tutorial covers the essential concepts for using the PyTorch Inference Framework. Practice with these examples and gradually explore more advanced features!

# 📊 Examples and Tutorials

This guide provides comprehensive examples for using the PyTorch Inference Framework across different scenarios and use cases.

## 📁 Example Structure

The `examples/` directory contains:

```
examples/
├── README.md                     # This guide
├── basic_usage.py               # Simple synchronous inference
├── async_processing.py          # High-throughput async inference
├── fastapi_server.py            # Production REST API
├── custom_models.py             # Integrating custom models
├── tensorrt_optimization.py     # TensorRT optimization
├── onnx_optimization.py         # ONNX Runtime optimization
├── quantization_examples.py     # Model quantization
├── performance_tuning.py        # Advanced performance optimization
├── docker_deployment.py         # Docker containerization
├── monitoring_setup.py          # Production monitoring
├── config_example.py            # Configuration management
├── config_modification_examples.py # Dynamic configuration
└── download_test_models.py      # Download models for testing
```

## 🚀 Basic Examples

### 1. Simple Inference (`basic_usage.py`)

```python
#!/usr/bin/env python3
"""
Basic PyTorch Inference Framework Usage Example

This example demonstrates simple synchronous inference patterns
using the framework with different model types.
"""

import torch
import numpy as np
from pathlib import Path
import logging

from framework import create_pytorch_framework, TorchInferenceFramework
from framework.core.config import InferenceConfig, DeviceConfig, BatchConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_model(model_type="linear"):
    """Create sample models for demonstration"""
    
    if model_type == "linear":
        # Simple linear classifier
        model = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )
    elif model_type == "cnn":
        # Simple CNN for image classification
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 10)
        )
    elif model_type == "complex":
        # More complex model
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 16, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Initialize weights
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    model.eval()
    return model

def example_1_quick_start():
    """Example 1: Quick start with minimal setup"""
    print("\n=== Example 1: Quick Start ===")
    
    # Create and save a simple model
    model = create_sample_model("linear")
    model_path = "models/simple_linear.pt"
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    # Initialize framework with minimal configuration
    framework = create_pytorch_framework(
        model_path=model_path,
        device="cpu"  # Use CPU for compatibility
    )
    
    # Create sample input (batch of 5 samples)
    input_data = torch.randn(5, 784)
    
    # Run inference
    result = framework.predict(input_data)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Prediction (first sample): {result[0]}")
    print(f"Predicted classes: {torch.argmax(result, dim=1)}")
    
    return framework

def example_2_custom_configuration():
    """Example 2: Custom configuration"""
    print("\n=== Example 2: Custom Configuration ===")
    
    # Create more complex model
    model = create_sample_model("cnn")
    model_path = "models/simple_cnn.pt"
    torch.save(model.state_dict(), model_path)
    
    # Custom configuration
    config = InferenceConfig(
        model_path=model_path,
        device=DeviceConfig(
            device_type="cpu",
            use_fp16=False  # FP16 not supported on CPU
        ),
        batch=BatchConfig(
            batch_size=8,
            max_batch_size=16
        )
    )
    
    # Initialize framework with configuration
    framework = TorchInferenceFramework(config=config)
    framework.initialize()
    
    # Create sample image input
    input_data = torch.randn(3, 3, 32, 32)  # 3 RGB images, 32x32
    
    # Run inference
    result = framework.predict(input_data)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Configuration used:")
    print(f"  Device: {config.device.device_type}")
    print(f"  Batch size: {config.batch.batch_size}")
    
    framework.cleanup()
    return result

def example_3_batch_processing():
    """Example 3: Batch processing"""
    print("\n=== Example 3: Batch Processing ===")
    
    # Use existing model
    model_path = "models/simple_linear.pt"
    
    framework = create_pytorch_framework(
        model_path=model_path,
        device="cpu",
        batch_size=16  # Process in batches of 16
    )
    
    # Create larger dataset
    num_samples = 100
    all_inputs = [torch.randn(1, 784) for _ in range(num_samples)]
    
    # Process as batch
    print(f"Processing {num_samples} samples in batches...")
    all_results = framework.predict_batch(all_inputs)
    
    print(f"Processed {len(all_results)} predictions")
    print(f"First prediction shape: {all_results[0].shape}")
    
    # Calculate accuracy on dummy labels
    dummy_labels = torch.randint(0, 10, (num_samples,))
    predicted_classes = torch.cat([torch.argmax(r, dim=1) for r in all_results])
    accuracy = (predicted_classes == dummy_labels).float().mean()
    print(f"Dummy accuracy: {accuracy:.2%}")
    
    return all_results

def example_4_performance_monitoring():
    """Example 4: Performance monitoring"""
    print("\n=== Example 4: Performance Monitoring ===")
    
    from framework import create_monitored_framework
    import time
    
    # Create framework with monitoring
    framework = create_monitored_framework(
        model_path="models/simple_linear.pt",
        enable_detailed_metrics=True
    )
    
    # Run multiple predictions for statistics
    input_data = torch.randn(1, 784)
    
    print("Running performance test...")
    for i in range(20):
        result = framework.predict(input_data)
        if i % 5 == 0:
            print(f"Completed {i+1}/20 predictions")
    
    # Get performance metrics
    metrics = framework.get_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Average latency: {metrics.get('latency', {}).get('avg_ms', 0):.2f}ms")
    print(f"  Total predictions: {metrics.get('predictions', {}).get('count', 0)}")
    print(f"  Throughput: {metrics.get('throughput', {}).get('requests_per_second', 0):.1f} req/s")
    
    return metrics

def example_5_error_handling():
    """Example 5: Error handling and validation"""
    print("\n=== Example 5: Error Handling ===")
    
    framework = create_pytorch_framework(
        model_path="models/simple_linear.pt",
        device="cpu"
    )
    
    # Test with correct input
    correct_input = torch.randn(2, 784)
    try:
        result = framework.predict(correct_input)
        print(f"✅ Correct input processed: {result.shape}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test with incorrect input shape
    wrong_input = torch.randn(2, 100)  # Wrong feature size
    try:
        result = framework.predict(wrong_input)
        print(f"⚠️  Wrong input somehow worked: {result.shape}")
    except Exception as e:
        print(f"✅ Correctly caught error: {type(e).__name__}: {e}")
    
    # Test with invalid input type
    try:
        result = framework.predict("invalid_input")
        print(f"⚠️  String input somehow worked")
    except Exception as e:
        print(f"✅ Correctly caught type error: {type(e).__name__}")
    
    # Test framework health
    health = framework.get_health_status()
    print(f"\nFramework health: {health.get('status', 'unknown')}")

def main():
    """Run all basic examples"""
    print("🚀 PyTorch Inference Framework - Basic Examples")
    print("=" * 50)
    
    try:
        # Run examples in sequence
        example_1_quick_start()
        example_2_custom_configuration() 
        example_3_batch_processing()
        example_4_performance_monitoring()
        example_5_error_handling()
        
        print("\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("  - Try async_processing.py for high-performance async inference")
        print("  - See fastapi_server.py for REST API deployment")
        print("  - Check optimization examples for performance tuning")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        logger.exception("Example execution failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
```

### 2. Async Processing (`async_processing.py`)

```python
#!/usr/bin/env python3
"""
Async Processing Example

Demonstrates high-throughput async inference with dynamic batching,
concurrent processing, and performance optimization.
"""

import asyncio
import torch
import time
import random
from pathlib import Path
import logging
from typing import List

from framework import create_async_framework
from framework.core.config import InferenceConfig, BatchConfig, DeviceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def example_1_basic_async():
    """Example 1: Basic async inference"""
    print("\n=== Example 1: Basic Async Inference ===")
    
    # Create async framework
    framework = await create_async_framework(
        model_path="models/simple_linear.pt",
        batch_size=4,
        max_batch_delay=0.05  # 50ms max batching delay
    )
    
    # Single async prediction
    input_data = torch.randn(1, 784)
    result = await framework.predict_async(input_data)
    
    print(f"Async prediction shape: {result.shape}")
    print(f"Async prediction: {result}")
    
    await framework.close()
    return result

async def example_2_concurrent_processing():
    """Example 2: Concurrent request processing"""
    print("\n=== Example 2: Concurrent Processing ===")
    
    framework = await create_async_framework(
        model_path="models/simple_linear.pt",
        batch_size=8,
        max_batch_size=16,
        max_batch_delay=0.1
    )
    
    # Create multiple concurrent requests
    num_requests = 20
    concurrent_inputs = [torch.randn(1, 784) for _ in range(num_requests)]
    
    print(f"Processing {num_requests} concurrent requests...")
    
    # Submit all requests at once
    start_time = time.time()
    tasks = [framework.predict_async(inp) for inp in concurrent_inputs]
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    print(f"✅ Processed {len(results)} requests in {end_time - start_time:.3f}s")
    print(f"Average latency per request: {(end_time - start_time) / num_requests * 1000:.1f}ms")
    print(f"Throughput: {num_requests / (end_time - start_time):.1f} req/s")
    
    await framework.close()
    return results

async def example_3_streaming_processing():
    """Example 3: Streaming request processing"""
    print("\n=== Example 3: Streaming Processing ===")
    
    framework = await create_async_framework(
        model_path="models/simple_linear.pt",
        batch_size=4,
        adaptive_batching=True  # Enable adaptive batching
    )
    
    async def request_generator():
        """Generate requests at varying intervals"""
        for i in range(30):
            # Simulate varying request rates
            await asyncio.sleep(random.uniform(0.01, 0.1))
            yield torch.randn(1, 784), i
    
    async def process_streaming_requests():
        """Process requests as they arrive"""
        results = []
        async for input_data, request_id in request_generator():
            result = await framework.predict_async(input_data)
            results.append((request_id, result))
            
            if len(results) % 10 == 0:
                print(f"Processed {len(results)} streaming requests...")
        
        return results
    
    print("Processing streaming requests...")
    start_time = time.time()
    streaming_results = await process_streaming_requests()
    end_time = time.time()
    
    print(f"✅ Processed {len(streaming_results)} streaming requests")
    print(f"Total time: {end_time - start_time:.3f}s")
    print(f"Average throughput: {len(streaming_results) / (end_time - start_time):.1f} req/s")
    
    await framework.close()
    return streaming_results

async def example_4_batch_optimization():
    """Example 4: Batch size optimization"""
    print("\n=== Example 4: Batch Size Optimization ===")
    
    # Test different batch configurations
    batch_configs = [
        {"batch_size": 1, "max_batch_size": 1},      # No batching
        {"batch_size": 4, "max_batch_size": 8},      # Small batches  
        {"batch_size": 8, "max_batch_size": 16},     # Medium batches
        {"batch_size": 16, "max_batch_size": 32},    # Large batches
    ]
    
    test_requests = [torch.randn(1, 784) for _ in range(50)]
    results = {}
    
    for config in batch_configs:
        print(f"\nTesting batch config: {config}")
        
        framework = await create_async_framework(
            model_path="models/simple_linear.pt",
            **config,
            max_batch_delay=0.05
        )
        
        # Benchmark this configuration
        start_time = time.time()
        tasks = [framework.predict_async(inp) for inp in test_requests]
        batch_results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = len(test_requests) / total_time
        
        results[str(config)] = {
            "total_time": total_time,
            "throughput": throughput,
            "avg_latency": total_time / len(test_requests) * 1000
        }
        
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} req/s") 
        print(f"  Avg latency: {total_time / len(test_requests) * 1000:.1f}ms")
        
        await framework.close()
    
    # Find best configuration
    best_config = max(results.items(), key=lambda x: x[1]["throughput"])
    print(f"\n🏆 Best configuration: {best_config[0]}")
    print(f"   Throughput: {best_config[1]['throughput']:.1f} req/s")
    
    return results

async def example_5_error_handling_async():
    """Example 5: Async error handling"""
    print("\n=== Example 5: Async Error Handling ===")
    
    framework = await create_async_framework(
        model_path="models/simple_linear.pt",
        batch_size=4
    )
    
    # Mix of valid and invalid requests
    requests = [
        torch.randn(1, 784),      # Valid
        torch.randn(1, 100),      # Invalid shape
        torch.randn(1, 784),      # Valid
        "invalid_input",          # Invalid type
        torch.randn(1, 784),      # Valid
    ]
    
    async def safe_predict(inp, request_id):
        """Safely handle prediction with error catching"""
        try:
            result = await framework.predict_async(inp)
            return {"id": request_id, "status": "success", "result": result}
        except Exception as e:
            return {"id": request_id, "status": "error", "error": str(e)}
    
    # Process all requests with error handling
    print("Processing mixed valid/invalid requests...")
    tasks = [safe_predict(inp, i) for i, inp in enumerate(requests)]
    results = await asyncio.gather(*tasks)
    
    # Analyze results
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    
    print(f"✅ Successful requests: {len(successful)}")
    print(f"❌ Failed requests: {len(failed)}")
    
    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        if result["status"] == "success":
            print(f"  {status_icon} Request {result['id']}: Success")
        else:
            print(f"  {status_icon} Request {result['id']}: {result['error']}")
    
    await framework.close()
    return results

async def example_6_performance_monitoring():
    """Example 6: Performance monitoring in async context"""
    print("\n=== Example 6: Performance Monitoring ===")
    
    from framework import create_monitored_framework
    
    # Create monitored async framework
    framework = await create_monitored_framework(
        model_path="models/simple_linear.pt",
        batch_size=8,
        enable_detailed_metrics=True,
        async_mode=True
    )
    
    # Run sustained load test
    print("Running sustained load test...")
    test_duration = 5  # seconds
    
    async def sustained_load():
        """Generate sustained load"""
        results = []
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            input_data = torch.randn(1, 784)
            result = await framework.predict_async(input_data)
            results.append(result)
            
            # Small delay between requests
            await asyncio.sleep(0.01)
        
        return results
    
    load_results = await sustained_load()
    
    # Get detailed metrics
    metrics = await framework.get_metrics_async()
    
    print(f"\nSustained Load Results:")
    print(f"  Duration: {test_duration}s")
    print(f"  Total requests: {len(load_results)}")
    print(f"  Average throughput: {len(load_results) / test_duration:.1f} req/s")
    
    if metrics:
        print(f"  Average latency: {metrics.get('latency', {}).get('avg_ms', 0):.2f}ms")
        print(f"  95th percentile: {metrics.get('latency', {}).get('p95_ms', 0):.2f}ms")
        print(f"  Batch efficiency: {metrics.get('batching', {}).get('efficiency', 0):.1%}")
    
    await framework.close()
    return metrics

async def main():
    """Run all async examples"""
    print("🚀 PyTorch Inference Framework - Async Processing Examples")
    print("=" * 60)
    
    # Ensure model exists
    Path("models").mkdir(exist_ok=True)
    if not Path("models/simple_linear.pt").exists():
        print("Creating sample model...")
        model = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )
        model.eval()
        torch.save(model.state_dict(), "models/simple_linear.pt")
    
    try:
        # Run async examples
        await example_1_basic_async()
        await example_2_concurrent_processing()
        await example_3_streaming_processing()
        await example_4_batch_optimization()
        await example_5_error_handling_async()
        await example_6_performance_monitoring()
        
        print("\n🎉 All async examples completed successfully!")
        print("\nKey takeaways:")
        print("  - Async processing enables high throughput")
        print("  - Dynamic batching improves efficiency")
        print("  - Proper error handling is essential")
        print("  - Monitor performance for optimization")
        
    except Exception as e:
        print(f"\n❌ Async example failed: {e}")
        logger.exception("Async example execution failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
```

### 3. FastAPI Server (`fastapi_server.py`)

```python
#!/usr/bin/env python3
"""
FastAPI Production Server Example

Production-ready REST API server with the PyTorch Inference Framework.
Includes authentication, monitoring, error handling, and documentation.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import logging
import time
import asyncio
from pathlib import Path

from framework import create_optimized_framework, create_monitored_framework
from framework.core.config_manager import get_config_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Pydantic models
class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: List[float] = Field(..., description="Model predictions")
    predicted_class: int = Field(..., description="Predicted class index")
    confidence: float = Field(..., description="Confidence score")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_info: Dict[str, Any] = Field(..., description="Model information")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    timestamp: float = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    predictions_served: int = Field(..., description="Total predictions served")
    uptime_seconds: float = Field(..., description="Service uptime")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    inputs: List[List[float]] = Field(..., description="Batch of input vectors")
    return_probabilities: bool = Field(True, description="Return probability distributions")

class ConfigResponse(BaseModel):
    """Response model for configuration info"""
    environment: str = Field(..., description="Current environment")
    device: str = Field(..., description="Compute device")
    batch_size: int = Field(..., description="Batch size")
    optimization_enabled: bool = Field(..., description="Whether optimization is enabled")
    features: Dict[str, bool] = Field(..., description="Available features")

# Global state
app_state = {
    "framework": None,
    "startup_time": time.time(),
    "prediction_count": 0,
    "config_manager": None
}

# Create FastAPI app
app = FastAPI(
    title="PyTorch Inference API",
    description="Production-ready PyTorch inference API with optimization and monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token verification (replace with proper auth in production)"""
    if not credentials:
        return None
    
    # Simple token check (use proper JWT validation in production)
    if credentials.credentials == "demo-token-12345":
        return {"user": "demo_user", "scope": "read_write"}
    
    return None

def get_current_user(user = Depends(verify_token)):
    """Get current authenticated user (optional)"""
    return user

@app.on_event("startup")
async def startup_event():
    """Initialize the inference framework on startup"""
    logger.info("Starting PyTorch Inference API...")
    
    try:
        # Get configuration
        config_manager = get_config_manager()
        app_state["config_manager"] = config_manager
        
        # Initialize framework
        model_path = "models/simple_linear.pt"
        
        # Ensure model exists
        Path("models").mkdir(exist_ok=True)
        if not Path(model_path).exists():
            logger.info("Creating sample model...")
            model = torch.nn.Sequential(
                torch.nn.Linear(784, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 10)
            )
            model.eval()
            torch.save(model.state_dict(), model_path)
        
        # Create optimized and monitored framework
        app_state["framework"] = await create_monitored_framework(
            model_path=model_path,
            optimization_level="balanced",
            enable_detailed_metrics=True,
            async_mode=True
        )
        
        logger.info("✅ Framework initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize framework: {e}")
        raise

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down PyTorch Inference API...")
    
    if app_state["framework"]:
        await app_state["framework"].close()
    
    logger.info("✅ Shutdown complete")

@app.get("/", response_model=Dict[str, Any])
async def root():
    """API information and status"""
    config_manager = app_state.get("config_manager")
    uptime = time.time() - app_state["startup_time"]
    
    return {
        "message": "PyTorch Inference API",
        "version": "1.0.0",
        "status": "healthy",
        "uptime_seconds": uptime,
        "environment": config_manager.environment if config_manager else "unknown",
        "endpoints": {
            "POST /predict": "Single prediction",
            "POST /predict/batch": "Batch prediction",
            "POST /predict/image": "Image classification",
            "GET /health": "Health check",
            "GET /config": "Configuration info",
            "GET /metrics": "Performance metrics",
            "GET /docs": "API documentation"
        },
        "authentication": {
            "required": False,
            "demo_token": "demo-token-12345"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    input_data: List[float],
    user = Depends(get_current_user)
):
    """Single prediction endpoint"""
    try:
        start_time = time.time()
        
        # Validate input
        if len(input_data) != 784:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 784 features, got {len(input_data)}"
            )
        
        # Convert to tensor
        input_tensor = torch.tensor(input_data).float().unsqueeze(0)
        
        # Run inference
        framework = app_state["framework"]
        if not framework:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        result = await framework.predict_async(input_tensor)
        
        # Process results
        probabilities = F.softmax(result, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
        
        processing_time = (time.time() - start_time) * 1000
        app_state["prediction_count"] += 1
        
        return PredictionResponse(
            prediction=probabilities.tolist(),
            predicted_class=predicted_class,
            confidence=confidence,
            processing_time_ms=processing_time,
            model_info={
                "model_type": "linear_classifier",
                "num_classes": 10,
                "input_features": 784
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(
    request: BatchPredictionRequest,
    user = Depends(get_current_user)
):
    """Batch prediction endpoint"""
    try:
        start_time = time.time()
        
        # Validate inputs
        if not request.inputs:
            raise HTTPException(status_code=400, detail="No inputs provided")
        
        if len(request.inputs) > 100:
            raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
        
        # Validate input dimensions
        for i, inp in enumerate(request.inputs):
            if len(inp) != 784:
                raise HTTPException(
                    status_code=400,
                    detail=f"Input {i}: expected 784 features, got {len(inp)}"
                )
        
        # Convert to tensor
        input_tensor = torch.tensor(request.inputs).float()
        
        # Run batch inference
        framework = app_state["framework"]
        if not framework:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        result = await framework.predict_async(input_tensor)
        
        # Process results
        if request.return_probabilities:
            probabilities = F.softmax(result, dim=1)
            predictions = probabilities.tolist()
        else:
            predictions = torch.argmax(result, dim=1).tolist()
        
        processing_time = (time.time() - start_time) * 1000
        app_state["prediction_count"] += len(request.inputs)
        
        return {
            "predictions": predictions,
            "batch_size": len(request.inputs),
            "processing_time_ms": processing_time,
            "predictions_per_second": len(request.inputs) / (processing_time / 1000)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    user = Depends(get_current_user)
):
    """Image classification endpoint"""
    try:
        start_time = time.time()
        
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess image
        input_tensor = image_transform(image).unsqueeze(0)
        
        # For demo, we'll flatten the image to match our linear model
        # In practice, you'd use a CNN model
        flattened_input = input_tensor.view(1, -1)
        
        # Pad or truncate to 784 features
        if flattened_input.shape[1] > 784:
            flattened_input = flattened_input[:, :784]
        elif flattened_input.shape[1] < 784:
            padding = torch.zeros(1, 784 - flattened_input.shape[1])
            flattened_input = torch.cat([flattened_input, padding], dim=1)
        
        # Run inference
        framework = app_state["framework"]
        if not framework:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        result = await framework.predict_async(flattened_input)
        
        # Process results
        probabilities = F.softmax(result, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
        
        processing_time = (time.time() - start_time) * 1000
        app_state["prediction_count"] += 1
        
        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "processing_time_ms": processing_time,
            "image_info": {
                "size": image.size,
                "mode": image.mode,
                "format": image.format
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    framework = app_state["framework"]
    uptime = time.time() - app_state["startup_time"]
    
    # Get performance metrics if available
    performance_metrics = None
    if framework:
        try:
            performance_metrics = await framework.get_metrics_async()
        except:
            pass  # Metrics not available
    
    return HealthResponse(
        status="healthy" if framework else "degraded",
        timestamp=time.time(),
        version="1.0.0",
        model_loaded=framework is not None,
        predictions_served=app_state["prediction_count"],
        uptime_seconds=uptime,
        performance_metrics=performance_metrics
    )

@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get configuration information"""
    config_manager = app_state.get("config_manager")
    
    if not config_manager:
        raise HTTPException(status_code=503, detail="Configuration not available")
    
    inference_config = config_manager.get_inference_config()
    
    return ConfigResponse(
        environment=config_manager.environment,
        device=str(inference_config.device.device_type),
        batch_size=inference_config.batch.batch_size,
        optimization_enabled=any([
            inference_config.optimization.enable_tensorrt,
            inference_config.optimization.enable_quantization,
            inference_config.optimization.enable_jit
        ]),
        features={
            "async_processing": True,
            "batch_processing": True,
            "image_processing": True,
            "monitoring": True,
            "authentication": False  # Demo only
        }
    )

@app.get("/metrics")
async def get_metrics(user = Depends(get_current_user)):
    """Get performance metrics (requires authentication in production)"""
    framework = app_state["framework"]
    
    if not framework:
        raise HTTPException(status_code=503, detail="Framework not available")
    
    try:
        metrics = await framework.get_metrics_async()
        return {
            "metrics": metrics,
            "total_predictions": app_state["prediction_count"],
            "uptime_seconds": time.time() - app_state["startup_time"],
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An error occurred"
        }
    )

# Development server
if __name__ == "__main__":
    import uvicorn
    
    # Start server without banner prints
    logger.info("Starting PyTorch Inference API Server")
    logger.info("API Documentation available at: http://localhost:8000/docs")
    logger.info("Health Check available at: http://localhost:8000/health")
    logger.info("Configuration available at: http://localhost:8000/config")
    logger.info("Demo Token: demo-token-12345")
    
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

## 🎯 Running the Examples

### Setup

```bash
# Ensure you're in the project root
cd torch-inference

# Install dependencies
uv sync --extra dev

# Create models directory
mkdir -p models

# Download test models (optional)
uv run python examples/download_test_models.py
```

### Running Individual Examples

```bash
# Basic usage patterns
uv run python examples/basic_usage.py

# High-performance async processing
uv run python examples/async_processing.py

# Production REST API server
uv run python examples/fastapi_server.py

# Test the API (in another terminal)
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"input_data": [0.1, 0.2, ...]}'  # 784 numbers
```

### Example Output

When running `basic_usage.py`:

```
🚀 PyTorch Inference Framework - Basic Examples
==================================================

=== Example 1: Quick Start ===
Input shape: torch.Size([5, 784])
Output shape: torch.Size([5, 10])
Prediction (first sample): tensor([-0.2435,  0.1234, ...])
Predicted classes: tensor([7, 2, 1, 9, 4])

=== Example 2: Custom Configuration ===
Input shape: torch.Size([3, 3, 32, 32])
Output shape: torch.Size([3, 10])
Configuration used:
  Device: cpu
  Batch size: 8

...

🎉 All examples completed successfully!
```

## 🚀 Advanced Examples

For more advanced examples, see:

- **[Performance Optimization](optimization-guide.md)** - TensorRT, ONNX, quantization
- **[Deployment Guide](deployment.md)** - Docker, Kubernetes, scaling
- **[Monitoring Guide](monitoring.md)** - Production monitoring setup
- **[API Reference](api.md)** - Complete API documentation

## 💡 Tips for Using Examples

1. **Start Simple**: Begin with `basic_usage.py` to understand core concepts
2. **Progress Gradually**: Move to async examples for production workloads
3. **Customize**: Adapt examples to your specific models and use cases
4. **Monitor Performance**: Use the monitoring examples to optimize your setup
5. **Handle Errors**: Learn from error handling examples for robust applications

---

*Ready to build your own application? Use these examples as starting points and refer to the [API Reference](api.md) for detailed documentation.*

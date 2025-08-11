"""
FastAPI server for PyTorch Inference Framework

This module provides a REST API interface for the PyTorch inference framework,
enabling remote inference requests with features like:
- Synchronous and asynchronous inference
- Batch processing
- Performance monitoring
- Health checks
- Model management
"""

import os
import sys
import logging
import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel as PydanticBaseModel, Field
import torch
import numpy as np
from pathlib import Path

# Get the absolute path of the current file (main.py)
project_root = os.path.dirname(os.path.abspath(__file__))

# Insert the project root at the beginning of sys.path if it's not already there.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import framework components
from framework.core.config import InferenceConfig, DeviceConfig, BatchConfig, PerformanceConfig, DeviceType
from framework.core.base_model import BaseModel, ModelManager, get_model_manager
from framework.core.inference_engine import InferenceEngine, create_inference_engine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
inference_engine: Optional[InferenceEngine] = None
model_manager: ModelManager = get_model_manager()

# Pydantic models for API
class InferenceRequest(PydanticBaseModel):
    """Request model for inference."""
    inputs: Any = Field(..., description="Input data for inference")
    priority: int = Field(default=0, description="Request priority (higher = processed first)")
    timeout: Optional[float] = Field(default=None, description="Request timeout in seconds")

class BatchInferenceRequest(PydanticBaseModel):
    """Request model for batch inference."""
    inputs: List[Any] = Field(..., description="List of input data for batch inference")
    priority: int = Field(default=0, description="Request priority (higher = processed first)")
    timeout: Optional[float] = Field(default=None, description="Request timeout in seconds")

class InferenceResponse(PydanticBaseModel):
    """Response model for inference."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    model_info: Optional[Dict[str, Any]] = None

class BatchInferenceResponse(PydanticBaseModel):
    """Response model for batch inference."""
    success: bool
    results: List[Any] = []
    error: Optional[str] = None
    processing_time: Optional[float] = None
    batch_size: int = 0

class HealthResponse(PydanticBaseModel):
    """Health check response."""
    healthy: bool
    checks: Dict[str, Any]
    timestamp: float
    engine_stats: Optional[Dict[str, Any]] = None

class StatsResponse(PydanticBaseModel):
    """Engine statistics response."""
    stats: Dict[str, Any]
    performance_report: Dict[str, Any]

# Simple example model for demonstration
class ExampleModel(BaseModel):
    """Example model implementation for demonstration."""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.model_name = "ExampleModel"
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load example model (dummy implementation)."""
        # Create a simple dummy model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True
        self.logger.info(f"Loaded example model on device: {self.device}")
    
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """Preprocess inputs."""
        if isinstance(inputs, list) and all(isinstance(x, (int, float)) for x in inputs):
            # List of numbers
            tensor_input = torch.tensor(inputs, dtype=torch.float32)
            if tensor_input.dim() == 1:
                tensor_input = tensor_input.unsqueeze(0)
            return tensor_input.to(self.device)
        elif isinstance(inputs, (int, float)):
            # Single number, pad to size 10
            padded_inputs = [float(inputs)] + [0.0] * 9
            return torch.tensor(padded_inputs, dtype=torch.float32).unsqueeze(0).to(self.device)
        elif isinstance(inputs, str):
            # Convert string to numeric (hash-based)
            numeric_input = [float(hash(inputs) % 1000) / 1000] + [0.0] * 9
            return torch.tensor(numeric_input, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            # Default: try to convert to tensor
            try:
                if hasattr(inputs, '__iter__') and not isinstance(inputs, str):
                    tensor_input = torch.tensor(list(inputs), dtype=torch.float32)
                else:
                    tensor_input = torch.tensor([inputs], dtype=torch.float32)
                
                # Pad or truncate to size 10
                if tensor_input.numel() > 10:
                    tensor_input = tensor_input[:10]
                elif tensor_input.numel() < 10:
                    padding = torch.zeros(10 - tensor_input.numel())
                    tensor_input = torch.cat([tensor_input.flatten(), padding])
                
                return tensor_input.unsqueeze(0).to(self.device)
            except Exception:
                # Fallback: random input
                return torch.randn(1, 10, device=self.device)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        model = self.get_model_for_inference()
        return model(inputs)
    
    def postprocess(self, outputs: torch.Tensor) -> Any:
        """Postprocess model outputs."""
        # Return as a simple prediction value
        result = outputs.cpu().numpy().squeeze()
        if result.shape == ():
            return float(result)
        return result.tolist()
    
    def _create_dummy_input(self) -> torch.Tensor:
        """Create dummy input for warmup."""
        return torch.randn(1, 10, device=self.device)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting PyTorch Inference API Server...")
    
    # Initialize model and engine
    await initialize_inference_engine()
    
    yield
    
    # Cleanup
    logger.info("Shutting down PyTorch Inference API Server...")
    await cleanup_inference_engine()

# Create FastAPI app
app = FastAPI(
    title="PyTorch Inference Framework API",
    description="REST API for PyTorch model inference with optimization and monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_inference_engine():
    """Initialize the inference engine with example model."""
    global inference_engine
    
    try:
        # Create configuration
        config = InferenceConfig(
            device=DeviceConfig(
                device_type=DeviceType.CUDA if torch.cuda.is_available() else DeviceType.CPU,
                use_fp16=False,  # Disable for example model
                use_torch_compile=False  # Disable for stability
            ),
            batch=BatchConfig(
                batch_size=4,
                max_batch_size=16,
                min_batch_size=1,
                timeout_seconds=5.0
            ),
            performance=PerformanceConfig(
                warmup_iterations=3,
                max_workers=4
            )
        )
        
        # Create and load example model
        example_model = ExampleModel(config)
        example_model.load_model("example")  # Dummy path
        example_model.optimize_for_inference()
        
        # Register model
        model_manager.register_model("example", example_model)
        
        # Create inference engine
        inference_engine = create_inference_engine(example_model, config)
        await inference_engine.start()
        
        # Warmup
        example_model.warmup(3)
        
        logger.info("Inference engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise

async def cleanup_inference_engine():
    """Cleanup inference engine."""
    global inference_engine
    
    if inference_engine:
        await inference_engine.stop()
        inference_engine = None
    
    model_manager.cleanup_all()

# API Routes

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "PyTorch Inference Framework API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "inference": "/predict",
            "batch_inference": "/predict/batch",
            "health": "/health",
            "stats": "/stats",
            "models": "/models"
        }
    }

@app.post("/predict")
async def predict(request: InferenceRequest) -> InferenceResponse:
    """Single prediction endpoint."""
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not available")
    
    try:
        import time
        start_time = time.time()
        
        result = await inference_engine.predict(
            inputs=request.inputs,
            priority=request.priority,
            timeout=request.timeout
        )
        
        processing_time = time.time() - start_time
        
        return InferenceResponse(
            success=True,
            result=result,
            processing_time=processing_time,
            model_info={"model": "example", "device": str(inference_engine.device)}
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return InferenceResponse(
            success=False,
            error=str(e)
        )

@app.post("/predict/batch")
async def predict_batch(request: BatchInferenceRequest) -> BatchInferenceResponse:
    """Batch prediction endpoint."""
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not available")
    
    try:
        import time
        start_time = time.time()
        
        results = await inference_engine.predict_batch(
            inputs_list=request.inputs,
            priority=request.priority,
            timeout=request.timeout
        )
        
        processing_time = time.time() - start_time
        
        return BatchInferenceResponse(
            success=True,
            results=results,
            processing_time=processing_time,
            batch_size=len(request.inputs)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return BatchInferenceResponse(
            success=False,
            error=str(e),
            batch_size=len(request.inputs)
        )

@app.get("/health")
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    if not inference_engine:
        return HealthResponse(
            healthy=False,
            checks={"inference_engine": False},
            timestamp=time.time()
        )
    
    try:
        health_status = await inference_engine.health_check()
        engine_stats = inference_engine.get_stats()
        
        return HealthResponse(
            healthy=health_status["healthy"],
            checks=health_status["checks"],
            timestamp=health_status["timestamp"],
            engine_stats=engine_stats
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            healthy=False,
            checks={"error": str(e)},
            timestamp=time.time()
        )

@app.get("/stats")
async def get_stats() -> StatsResponse:
    """Get engine statistics."""
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not available")
    
    try:
        stats = inference_engine.get_stats()
        performance_report = inference_engine.get_performance_report()
        
        return StatsResponse(
            stats=stats,
            performance_report=performance_report
        )
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models."""
    try:
        models = model_manager.list_models()
        model_info = {}
        
        for model_name in models:
            model = model_manager.get_model(model_name)
            model_info[model_name] = model.model_info
        
        return {
            "models": models,
            "model_info": model_info,
            "total_models": len(models)
        }
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Example usage endpoints
@app.post("/examples/simple")
async def simple_example(data: Dict[str, Any]):
    """Simple example endpoint for testing."""
    try:
        # Extract input from data
        input_value = data.get("input", 42)
        
        request = InferenceRequest(inputs=input_value)
        response = await predict(request)
        
        return {
            "example": "simple_prediction",
            "input": input_value,
            "response": response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/examples/batch")
async def batch_example():
    """Batch example endpoint for testing."""
    try:
        # Create example batch
        example_inputs = [1, 2, 3, 4, 5]
        
        request = BatchInferenceRequest(inputs=example_inputs)
        response = await predict_batch(request)
        
        return {
            "example": "batch_prediction",
            "input_count": len(example_inputs),
            "response": response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )

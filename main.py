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
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request, Response
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
from framework.core.config_manager import get_config_manager, ConfigManager
from framework.core.base_model import BaseModel, ModelManager, get_model_manager
from framework.core.inference_engine import InferenceEngine, create_inference_engine

# Initialize configuration manager
config_manager = get_config_manager()

# Setup logging with configuration
server_config = config_manager.get_server_config()
logging.basicConfig(level=getattr(logging, server_config['log_level']))
logger = logging.getLogger(__name__)

# Global variables
inference_engine: Optional[InferenceEngine] = None
model_manager: ModelManager = get_model_manager()

def print_api_endpoints():
    """Print all available API endpoints at startup"""
    endpoints = [
        ("GET", "/", "Root endpoint - API information"),
        ("POST", "/predict", "Single prediction endpoint"),
        ("POST", "/predict/batch", "Batch prediction endpoint"),
        ("GET", "/health", "Health check endpoint"),
        ("GET", "/stats", "Engine statistics endpoint"),
        ("GET", "/config", "Configuration information endpoint"),
        ("GET", "/models", "List available models"),
        ("POST", "/models/download", "Download a model from source"),
        ("GET", "/models/available", "List available models for download"),
        ("GET", "/models/download/{model_name}/info", "Get download info for a model"),
        ("DELETE", "/models/download/{model_name}", "Remove model from cache"),
        ("GET", "/models/cache/info", "Get model cache information"),
        ("POST", "/examples/simple", "Simple prediction example"),
        ("POST", "/examples/batch", "Batch prediction example"),
    ]
    
    print("\n" + "="*80)
    print("  PYTORCH INFERENCE FRAMEWORK - API ENDPOINTS")
    print("="*80)
    
    for method, endpoint, description in endpoints:
        print(f"  {method:<7} {endpoint:<40} - {description}")
    
    print("="*80)
    print(f"  Total Endpoints: {len(endpoints)}")
    print(f"  Documentation: http://localhost:8000/docs")
    print(f"  Health Check: http://localhost:8000/health")
    print("="*80 + "\n")

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
    logger.info("[SERVER] Starting PyTorch Inference API Server...")
    logger.info("[SERVER] Initializing inference engine...")
    
    # Initialize model and engine
    await initialize_inference_engine()
    
    logger.info("[SERVER] Server startup complete - Ready to accept requests")
    
    yield
    
    # Cleanup
    logger.info("[SERVER] Shutting down PyTorch Inference API Server...")
    await cleanup_inference_engine()
    logger.info("[SERVER] Server shutdown complete")

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

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all API requests and responses"""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    url = str(request.url)
    
    # Get user agent and other headers
    user_agent = request.headers.get("user-agent", "unknown")
    content_type = request.headers.get("content-type", "")
    
    # Log request
    logger.info(f"[API REQUEST] {method} {url} - Client: {client_ip} - User-Agent: {user_agent[:100]}")
    
    # Call the endpoint
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(f"[API RESPONSE] {method} {url} - Status: {response.status_code} - Time: {process_time:.3f}s - Client: {client_ip}")
    
    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

async def initialize_inference_engine():
    """Initialize the inference engine with example model."""
    global inference_engine
    
    try:
        # Get configuration from config manager
        config = config_manager.get_inference_config()
        
        logger.info(f"Initializing inference engine with configuration:")
        logger.info(f"  Device: {config.device.device_type.value}")
        logger.info(f"  Batch size: {config.batch.batch_size}")
        logger.info(f"  FP16: {config.device.use_fp16}")
        
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
        example_model.warmup(config.performance.warmup_iterations)
        
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
    logger.info("[ENDPOINT] Root endpoint accessed - returning API information")
    
    response_data = {
        "message": "PyTorch Inference Framework API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "environment": config_manager.environment,
        "endpoints": {
            "inference": "/predict",
            "batch_inference": "/predict/batch",
            "health": "/health",
            "stats": "/stats",
            "models": "/models",
            "config": "/config",
            "model_downloads": {
                "download": "/models/download",
                "available": "/models/available", 
                "cache_info": "/models/cache/info",
                "remove": "/models/download/{model_name}"
            }
        }
    }
    
    logger.debug(f"[ENDPOINT] Root endpoint response: {response_data}")
    return response_data@app.post("/predict")
async def predict(request: InferenceRequest) -> InferenceResponse:
    """Single prediction endpoint."""
    logger.info(f"[ENDPOINT] Single prediction requested - Priority: {request.priority}, Timeout: {request.timeout}")
    
    if not inference_engine:
        logger.error("[ENDPOINT] Prediction failed - Inference engine not available")
        raise HTTPException(status_code=503, detail="Inference engine not available")
    
    try:
        start_time = time.time()
        
        logger.debug(f"[ENDPOINT] Processing prediction with inputs type: {type(request.inputs)}")
        
        result = await inference_engine.predict(
            inputs=request.inputs,
            priority=request.priority,
            timeout=request.timeout
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"[ENDPOINT] Prediction completed successfully - Processing time: {processing_time:.3f}s")
        
        response = InferenceResponse(
            success=True,
            result=result,
            processing_time=processing_time,
            model_info={"model": "example", "device": str(inference_engine.device)}
        )
        
        logger.debug(f"[ENDPOINT] Prediction response generated - Success: {response.success}")
        return response
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Prediction failed with error: {e}")
        return InferenceResponse(
            success=False,
            error=str(e)
        )

@app.post("/predict/batch")
async def predict_batch(request: BatchInferenceRequest) -> BatchInferenceResponse:
    """Batch prediction endpoint."""
    batch_size = len(request.inputs)
    logger.info(f"[ENDPOINT] Batch prediction requested - Batch size: {batch_size}, Priority: {request.priority}, Timeout: {request.timeout}")
    
    if not inference_engine:
        logger.error("[ENDPOINT] Batch prediction failed - Inference engine not available")
        raise HTTPException(status_code=503, detail="Inference engine not available")
    
    try:
        start_time = time.time()
        
        logger.debug(f"[ENDPOINT] Processing batch prediction with {batch_size} inputs")
        
        results = await inference_engine.predict_batch(
            inputs_list=request.inputs,
            priority=request.priority,
            timeout=request.timeout
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"[ENDPOINT] Batch prediction completed successfully - Batch size: {batch_size}, Processing time: {processing_time:.3f}s")
        
        response = BatchInferenceResponse(
            success=True,
            results=results,
            processing_time=processing_time,
            batch_size=batch_size
        )
        
        logger.debug(f"[ENDPOINT] Batch prediction response generated - Success: {response.success}, Results count: {len(results)}")
        return response
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Batch prediction failed with error: {e}")
        return BatchInferenceResponse(
            success=False,
            error=str(e),
            batch_size=batch_size
        )

@app.get("/health")
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    logger.info("[ENDPOINT] Health check requested")
    
    if not inference_engine:
        logger.warning("[ENDPOINT] Health check - Inference engine not available")
        response = HealthResponse(
            healthy=False,
            checks={"inference_engine": False},
            timestamp=time.time()
        )
        logger.debug(f"[ENDPOINT] Health check response: unhealthy - {response.checks}")
        return response
    
    try:
        health_status = await inference_engine.health_check()
        engine_stats = inference_engine.get_stats()
        
        logger.info(f"[ENDPOINT] Health check completed - Healthy: {health_status['healthy']}")
        logger.debug(f"[ENDPOINT] Health check details - Checks: {health_status['checks']}")
        
        response = HealthResponse(
            healthy=health_status["healthy"],
            checks=health_status["checks"],
            timestamp=health_status["timestamp"],
            engine_stats=engine_stats
        )
        
        return response
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Health check failed with error: {e}")
        return HealthResponse(
            healthy=False,
            checks={"error": str(e)},
            timestamp=time.time()
        )

@app.get("/stats")
async def get_stats() -> StatsResponse:
    """Get engine statistics."""
    logger.info("[ENDPOINT] Statistics requested")
    
    if not inference_engine:
        logger.error("[ENDPOINT] Statistics failed - Inference engine not available")
        raise HTTPException(status_code=503, detail="Inference engine not available")
    
    try:
        stats = inference_engine.get_stats()
        performance_report = inference_engine.get_performance_report()
        
        logger.info("[ENDPOINT] Statistics retrieved successfully")
        logger.debug(f"[ENDPOINT] Stats summary - Requests processed: {stats.get('requests_processed', 'unknown')}")
        
        response = StatsResponse(
            stats=stats,
            performance_report=performance_report
        )
        
        return response
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Statistics failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    """Get current configuration information."""
    logger.info("[ENDPOINT] Configuration information requested")
    
    try:
        config_data = {
            "configuration": config_manager.export_config(),
            "inference_config": {
                "device_type": config_manager.get('DEVICE', 'auto', 'device.type'),
                "batch_size": config_manager.get('BATCH_SIZE', 4, 'batch.batch_size'),
                "use_fp16": config_manager.get('USE_FP16', False, 'device.use_fp16'),
                "enable_profiling": config_manager.get('ENABLE_PROFILING', False, 'performance.enable_profiling')
            },
            "server_config": server_config
        }
        
        logger.info("[ENDPOINT] Configuration information retrieved successfully")
        logger.debug(f"[ENDPOINT] Config summary - Environment: {config_manager.environment}, Device: {config_data['inference_config']['device_type']}")
        
        return config_data
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Configuration retrieval failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models."""
    logger.info("[ENDPOINT] Model list requested")
    
    try:
        models = model_manager.list_models()
        model_info = {}
        
        for model_name in models:
            model = model_manager.get_model(model_name)
            model_info[model_name] = model.model_info
        
        logger.info(f"[ENDPOINT] Model list retrieved successfully - Total models: {len(models)}")
        logger.debug(f"[ENDPOINT] Available models: {models}")
        
        return {
            "models": models,
            "model_info": model_info,
            "total_models": len(models)
        }
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Model list retrieval failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model download endpoints
@app.post("/models/download")
async def download_model_endpoint(
    source: str,
    model_id: str,
    name: str,
    task: str = "classification",
    pretrained: bool = True,
    background_tasks: BackgroundTasks = None
):
    """Download a model from a source."""
    logger.info(f"[ENDPOINT] Model download requested - Name: {name}, Source: {source}, Model ID: {model_id}, Task: {task}")
    
    try:
        # Validate source
        valid_sources = ["pytorch_hub", "torchvision", "huggingface", "url"]
        if source not in valid_sources:
            logger.error(f"[ENDPOINT] Model download failed - Invalid source: {source}")
            raise HTTPException(status_code=400, detail=f"Invalid source. Must be one of: {valid_sources}")
        
        logger.debug(f"[ENDPOINT] Model download validation passed - Source: {source}, Background: {background_tasks is not None}")
        
        # Start download in background
        if background_tasks:
            background_tasks.add_task(
                model_manager.download_and_load_model,
                source, model_id, name, None, task=task, pretrained=pretrained
            )
            
            logger.info(f"[ENDPOINT] Model download started in background - Name: {name}")
            
            return {
                "message": f"Started downloading model '{name}' from {source}",
                "model_name": name,
                "source": source,
                "model_id": model_id,
                "status": "downloading"
            }
        else:
            # Download synchronously
            logger.info(f"[ENDPOINT] Starting synchronous model download - Name: {name}")
            model_manager.download_and_load_model(
                source, model_id, name, None, task=task, pretrained=pretrained
            )
            
            logger.info(f"[ENDPOINT] Model download completed successfully - Name: {name}")
            
            return {
                "message": f"Successfully downloaded and loaded model '{name}'",
                "model_name": name,
                "source": source,
                "model_id": model_id,
                "status": "completed"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ENDPOINT] Model download failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/available")
async def list_available_downloads():
    """List available models that can be downloaded."""
    logger.info("[ENDPOINT] Available downloads list requested")
    
    try:
        available_models = model_manager.list_available_downloads()
        
        logger.info(f"[ENDPOINT] Available downloads retrieved successfully - Total available: {len(available_models)}")
        logger.debug(f"[ENDPOINT] Available models: {list(available_models.keys())}")
        
        return {
            "available_models": {
                name: {
                    "name": info.name,
                    "source": info.source,
                    "model_id": info.model_id,
                    "task": info.task,
                    "description": info.description,
                    "size_mb": info.size_mb,
                    "tags": info.tags
                } for name, info in available_models.items()
            },
            "total_available": len(available_models)
        }
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Available downloads retrieval failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/download/{model_name}/info")
async def get_download_info(model_name: str):
    """Get information about a downloadable model."""
    logger.info(f"[ENDPOINT] Download info requested for model: {model_name}")
    
    try:
        info = model_manager.get_download_info(model_name)
        
        if info is None:
            logger.warning(f"[ENDPOINT] Model not found: {model_name}")
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
        
        logger.info(f"[ENDPOINT] Download info retrieved successfully for model: {model_name}")
        logger.debug(f"[ENDPOINT] Model info: {info}")
        
        return {
            "model_name": model_name,
            "info": info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ENDPOINT] Download info retrieval failed for model {model_name} with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/download/{model_name}")
async def remove_downloaded_model(model_name: str):
    """Remove a downloaded model from cache."""
    logger.info(f"[ENDPOINT] Model removal requested for: {model_name}")
    
    try:
        downloader = model_manager.get_downloader()
        
        if not downloader.is_model_cached(model_name):
            logger.warning(f"[ENDPOINT] Model not found in cache: {model_name}")
            raise HTTPException(status_code=404, detail=f"Model not found in cache: {model_name}")
        
        logger.debug(f"[ENDPOINT] Removing model from cache: {model_name}")
        success = downloader.remove_model(model_name)
        
        if success:
            logger.info(f"[ENDPOINT] Model removed successfully from cache: {model_name}")
            return {
                "message": f"Successfully removed model from cache: {model_name}",
                "model_name": model_name
            }
        else:
            logger.error(f"[ENDPOINT] Failed to remove model from cache: {model_name}")
            raise HTTPException(status_code=500, detail=f"Failed to remove model: {model_name}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ENDPOINT] Model removal failed for {model_name} with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/cache/info")
async def get_cache_info():
    """Get information about the model cache."""
    logger.info("[ENDPOINT] Cache info requested")
    
    try:
        downloader = model_manager.get_downloader()
        
        cache_info = {
            "cache_directory": str(downloader.cache_dir),
            "total_models": len(downloader.registry),
            "total_size_mb": downloader.get_cache_size(),
            "models": list(downloader.registry.keys())
        }
        
        logger.info(f"[ENDPOINT] Cache info retrieved successfully - Total models: {cache_info['total_models']}, Size: {cache_info['total_size_mb']} MB")
        logger.debug(f"[ENDPOINT] Cached models: {cache_info['models']}")
        
        return cache_info
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Cache info retrieval failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Example usage endpoints
@app.post("/examples/simple")
async def simple_example(data: Dict[str, Any]):
    """Simple example endpoint for testing."""
    input_value = data.get("input", 42)
    logger.info(f"[ENDPOINT] Simple example requested with input: {input_value}")
    
    try:
        logger.debug("[ENDPOINT] Processing simple example prediction")
        
        request = InferenceRequest(inputs=input_value)
        response = await predict(request)
        
        logger.info(f"[ENDPOINT] Simple example completed - Success: {response.success}")
        
        result = {
            "example": "simple_prediction",
            "input": input_value,
            "response": response
        }
        
        return result
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Simple example failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/examples/batch")
async def batch_example():
    """Batch example endpoint for testing."""
    example_inputs = [1, 2, 3, 4, 5]
    logger.info(f"[ENDPOINT] Batch example requested with {len(example_inputs)} inputs")
    
    try:
        logger.debug(f"[ENDPOINT] Processing batch example with inputs: {example_inputs}")
        
        request = BatchInferenceRequest(inputs=example_inputs)
        response = await predict_batch(request)
        
        logger.info(f"[ENDPOINT] Batch example completed - Success: {response.success}")
        
        result = {
            "example": "batch_prediction",
            "input_count": len(example_inputs),
            "response": response
        }
        
        return result
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Batch example failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Print all available endpoints first
    print_api_endpoints()
    
    # Get server configuration
    server_config = config_manager.get_server_config()
    
    # Log startup information without banners
    logger.info("Initializing PyTorch Inference Framework API Server")
    logger.info(f"Server configuration - Host: {server_config['host']}, Port: {server_config['port']}")
    logger.info(f"Environment: {config_manager.environment}")
    logger.info(f"Log level: {server_config['log_level']}")
    logger.info(f"Reload mode: {server_config['reload']}")
    
    # Start the FastAPI server with configuration
    logger.info("Starting server...")
    uvicorn.run(
        "main:app",
        host=server_config['host'],
        port=server_config['port'],
        reload=server_config['reload'],
        log_level=server_config['log_level'].lower()
    )

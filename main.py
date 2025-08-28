"""
FastAPI server for PyTorch Inference Framework

This module provides a REST API interface for the PyTorch inference framework,
enabling remote inference requests with features like:
- Synchronous and asynchronous inference
- Batch processing
- Performance monitoring
- Health checks
- Model management
- Security mitigations for known vulnerabilities
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
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel as PydanticBaseModel, Field
import torch
import numpy as np
from pathlib import Path
import tempfile
import io
import base64

# Print GPU information at startup
print("\n" + "="*60)
print("  GPU DETECTION AND SYSTEM INFO")
print("="*60)

# Enable TensorFloat32 for better performance on modern GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    gpu_count = torch.cuda.device_count()
    memory_total = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)  # GB
    memory_allocated = torch.cuda.memory_allocated(current_device) / (1024**3)  # GB
    memory_reserved = torch.cuda.memory_reserved(current_device) / (1024**3)  # GB
    
    print(f"✓ CUDA Available: Yes")
    print(f"  Current GPU: {gpu_name} (Device {current_device})")
    print(f"  Total GPUs: {gpu_count}")
    print(f"  GPU Memory: {memory_allocated:.2f}GB / {memory_total:.2f}GB (Reserved: {memory_reserved:.2f}GB)")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  PyTorch Version: {torch.__version__}")
    
    # List all available GPUs
    if gpu_count > 1:
        print(f"  Available GPUs:")
        for i in range(gpu_count):
            gpu_name_i = torch.cuda.get_device_name(i)
            memory_total_i = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"    GPU {i}: {gpu_name_i} ({memory_total_i:.1f}GB)")
else:
    print(f"✗ CUDA Available: No")
    # Check for other device types
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"✓ Apple MPS Available: Yes")
        print(f"  Running on Apple Silicon GPU")
    else:
        print(f"✗ Apple MPS Available: No")
    
    print(f"  Running on CPU")
    print(f"  PyTorch Version: {torch.__version__}")

print("="*60 + "\n")

# Get the absolute path of the current file (main.py)
project_root = os.path.dirname(os.path.abspath(__file__))

# Insert the project root at the beginning of sys.path if it's not already there.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Debug information
print(f"Project root: {project_root}")
print(f"Python executable: {sys.executable}")
print(f"Project root in sys.path: {project_root in sys.path}")

# Make sure we can import the framework modules
framework_available = False
try:
    # Test basic framework import
    import framework
    print("✓ Framework module imported successfully")
    framework_available = True
except ImportError as e:
    print(f"✗ Failed to import framework module: {e}")
    framework_available = False
    
    # Try to fix the import by adding the project root again
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        try:
            import framework
            print("✓ Framework module imported successfully after path fix")
            framework_available = True
        except ImportError as e2:
            print(f"✗ Still failed to import framework module: {e2}")
            framework_available = False
except RuntimeError as e:
    print(f"✗ Runtime error during framework import (likely TorchVision compatibility issue): {e}")
    framework_available = False

print(f"Framework availability: {framework_available}")

# Import and initialize security mitigations from framework
try:
    from framework.core.security import (
        initialize_security_mitigations,
        PyTorchSecurityMitigation,
        ECDSASecurityMitigation
    )
    
    # Initialize security mitigations immediately
    pytorch_security, ecdsa_security = initialize_security_mitigations()
    
    print("Security mitigations initialized successfully from framework")
    
except Exception as e:
    print(f"Warning: Failed to initialize security mitigations: {e}")
    # Create dummy security mitigation for fallback
    class DummySecurityMitigation:
        def secure_context(self):
            import contextlib
            return contextlib.nullcontext()
        
        def secure_timing_context(self):
            import contextlib
            return contextlib.nullcontext()
        
        def cleanup_resources(self):
            pass
        
        @staticmethod
        def secure_model_load(model_path, map_location=None):
            import torch
            return torch.load(model_path, map_location=map_location, weights_only=True)
        
        @staticmethod
        def secure_tensor_operation(operation_func, *args, **kwargs):
            return operation_func(*args, **kwargs)
    
    pytorch_security = DummySecurityMitigation()
    ecdsa_security = DummySecurityMitigation()

# Import framework components with error handling
if framework_available:
    try:
        from framework.core.config import InferenceConfig, DeviceConfig, BatchConfig, PerformanceConfig, DeviceType
        print("✓ Config imports successful")
    except (ImportError, RuntimeError) as e:
        print(f"✗ Config import failed: {e}")
        framework_available = False
else:
    print("ℹ Using fallback imports due to framework unavailability")

if not framework_available:
    # Create minimal dummy classes
    class DeviceType:
        CPU = "cpu"
        CUDA = "cuda"
    
    class InferenceConfig:
        def __init__(self):
            self.device = type('obj', (object,), {'device_type': DeviceType.CPU, 'device_id': 0})()
            self.batch = type('obj', (object,), {'batch_size': 1, 'max_batch_size': 8})()
            self.performance = type('obj', (object,), {'warmup_iterations': 3})()

if framework_available:
    try:
        from framework.core.config_manager import get_config_manager, ConfigManager
        print("✓ Config manager imports successful")
    except (ImportError, RuntimeError) as e:
        print(f"✗ Config manager import failed: {e}")
        framework_available = False

if not framework_available:
    # Create minimal dummy config manager
    class ConfigManager:
        def __init__(self):
            self.environment = "development"
        
        def get_config_manager(self):
            return self
        
        def get_inference_config(self):
            return InferenceConfig()
        
        def get_server_config(self):
            return {
                'host': '0.0.0.0',
                'port': 8000,
                'log_level': 'INFO',
                'reload': False
            }
    
    def get_config_manager():
        return ConfigManager()

if framework_available:
    try:
        from framework.core.base_model import BaseModel, ModelManager, get_model_manager
        print("✓ Base model imports successful")
    except (ImportError, RuntimeError) as e:
        print(f"✗ Base model import failed: {e}")
        framework_available = False

if not framework_available:
    # Create minimal dummy base model
    class BaseModel:
        def __init__(self, config):
            self.config = config
            self.device = torch.device('cpu')
            self.model = None
            self._is_loaded = False
            self.logger = logging.getLogger(__name__)
            self.model_name = "DummyModel"
        
        def load_model(self, model_path):
            self._is_loaded = True
        
        def preprocess(self, inputs):
            return inputs
        
        def forward(self, inputs):
            return inputs
        
        def postprocess(self, outputs):
            return outputs
        
        def predict(self, inputs):
            return {"result": "dummy_prediction"}
        
        @property
        def is_loaded(self):
            return self._is_loaded
        
        @property
        def model_info(self):
            return {"model_name": self.model_name, "device": str(self.device)}
        
        def warmup(self, iterations=3):
            pass
        
        def optimize_for_inference(self):
            pass
        
        def cleanup(self):
            pass
    
    class ModelManager:
        def __init__(self):
            self.models = {}
        
        def register_model(self, name, model):
            self.models[name] = model
        
        def get_model(self, name):
            return self.models.get(name)
        
        def list_models(self):
            return list(self.models.keys())
        
        def cleanup_all(self):
            pass
    
    def get_model_manager():
        return ModelManager()

if framework_available:
    try:
        from framework.core.inference_engine import InferenceEngine, create_inference_engine
        print("✓ Inference engine imports successful")
    except (ImportError, RuntimeError) as e:
        print(f"✗ Inference engine import failed: {e}")
        framework_available = False

if not framework_available:
    # Create minimal dummy inference engine
    class InferenceEngine:
        def __init__(self, model, config):
            self.model = model
            self.config = config
            self.device = torch.device('cpu')
            self.stats = {"requests_processed": 0}
        
        async def start(self):
            pass
        
        async def stop(self):
            pass
        
        async def predict(self, inputs, priority=0, timeout=None):
            self.stats["requests_processed"] += 1
            return self.model.predict(inputs)
        
        async def predict_batch(self, inputs_list, priority=0, timeout=None):
            results = []
            for inputs in inputs_list:
                results.append(await self.predict(inputs, priority, timeout))
            return results
        
        def get_stats(self):
            return self.stats
        
        def get_performance_report(self):
            return {"performance": "dummy_report"}
        
        async def health_check(self):
            return {
                "healthy": True,
                "checks": {"model": True, "engine": True},
                "timestamp": time.time()
            }
    
    def create_inference_engine(model, config):
        return InferenceEngine(model, config)

if framework_available:
    try:
        from framework.core.gpu_manager import GPUManager, auto_configure_device
        print("✓ GPU manager imports successful")
    except (ImportError, RuntimeError) as e:
        print(f"✗ GPU manager import failed: {e}")
        # GPU manager failure is non-critical, we have GPU detection code already

if framework_available:
    try:
        from framework.autoscaling import Autoscaler, AutoscalerConfig, ZeroScalingConfig, ModelLoaderConfig
        print("✓ Autoscaling imports successful")
    except (ImportError, RuntimeError) as e:
        print(f"✗ Autoscaling import failed: {e}")
        framework_available = False

if not framework_available:
    # Create minimal dummy autoscaler
    class AutoscalerConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ZeroScalingConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ModelLoaderConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class Autoscaler:
        def __init__(self, config, model_manager):
            self.config = config
            self.model_manager = model_manager
        
        async def start(self):
            pass
        
        async def stop(self):
            pass
        
        async def predict(self, model_name, inputs, priority=0, timeout=None):
            model = self.model_manager.get_model(model_name)
            if model:
                return model.predict(inputs)
            return {"error": "Model not found"}
        
        def get_stats(self):
            return {"autoscaler": "dummy_stats"}
        
        def get_health_status(self):
            return {"healthy": True, "timestamp": time.time()}

# Initialize configuration manager
config_manager = get_config_manager()

# Setup logging with configuration
server_config = config_manager.get_server_config()
logging.basicConfig(level=getattr(logging, server_config['log_level']))
logger = logging.getLogger(__name__)

# Global variables
inference_engine: Optional[InferenceEngine] = None
model_manager: ModelManager = get_model_manager()
autoscaler: Optional[Autoscaler] = None

def print_api_endpoints():
    """Print all available API endpoints at startup"""
    endpoints = [
        ("GET", "/", "Root endpoint - API information"),
        ("POST", "/{model_name}/predict", "Model-specific prediction endpoint with inflight batching"),
        ("GET", "/health", "Health check endpoint"),
        ("GET", "/stats", "Engine statistics endpoint"),
        ("GET", "/config", "Configuration information endpoint"),
        ("GET", "/models", "List available models"),
        ("POST", "/models/download", "Download a model from source"),
        ("GET", "/models/available", "List available models for download"),
        ("GET", "/models/download/{model_name}/info", "Get download info for a model"),
        ("DELETE", "/models/download/{model_name}", "Remove model from cache"),
        ("GET", "/models/cache/info", "Get model cache information"),
        # GPU detection endpoints
        ("GET", "/gpu/detect", "Detect available GPUs"),
        ("GET", "/gpu/best", "Get best GPU for inference"),
        ("GET", "/gpu/config", "Get GPU-optimized configuration"),
        ("GET", "/gpu/report", "Get comprehensive GPU report"),
        # New autoscaling endpoints
        ("GET", "/autoscaler/stats", "Get autoscaler statistics"),
        ("GET", "/autoscaler/health", "Get autoscaler health status"),
        ("POST", "/autoscaler/scale", "Scale a model to target instances"),
        ("POST", "/autoscaler/load", "Load a model with autoscaling"),
        ("DELETE", "/autoscaler/unload", "Unload a model"),
        ("GET", "/autoscaler/metrics", "Get detailed autoscaling metrics"),
        # Audio processing endpoints
        ("POST", "/tts/synthesize", "Text-to-Speech synthesis"),
        ("POST", "/stt/transcribe", "Speech-to-Text transcription"),
        ("GET", "/audio/models", "List available audio models"),
        ("GET", "/audio/health", "Audio processing health check"),
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
    """Request model for model-specific inference with inflight batching support."""
    inputs: Union[Any, List[Any]] = Field(..., description="Input data for inference (single item or list for batch)")
    priority: int = Field(default=0, description="Request priority (higher = processed first)")
    timeout: Optional[float] = Field(default=None, description="Request timeout in seconds")
    enable_batching: bool = Field(default=True, description="Enable inflight batching optimization")

class InferenceResponse(PydanticBaseModel):
    """Response model for inference."""
    success: bool
    result: Union[Any, List[Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    model_info: Optional[Dict[str, Any]] = None
    batch_info: Optional[Dict[str, Any]] = None

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

# Audio-specific Pydantic models
class TTSRequest(PydanticBaseModel):
    """Request model for Text-to-Speech synthesis."""
    text: str = Field(..., description="Text to synthesize")
    model_name: str = Field(default="default", description="TTS model to use")
    voice: Optional[str] = Field(default=None, description="Voice to use for synthesis")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed (0.5-2.0)")
    pitch: float = Field(default=1.0, ge=0.5, le=2.0, description="Pitch adjustment (0.5-2.0)")
    volume: float = Field(default=1.0, ge=0.1, le=2.0, description="Volume level (0.1-2.0)")
    language: str = Field(default="en", description="Language code (e.g., 'en', 'es', 'fr')")
    emotion: Optional[str] = Field(default=None, description="Emotion for synthesis")
    output_format: str = Field(default="wav", pattern="^(wav|mp3|flac)$", description="Output audio format")

class STTRequest(PydanticBaseModel):
    """Request model for Speech-to-Text transcription."""
    model_name: str = Field(default="whisper-base", description="STT model to use")
    language: str = Field(default="auto", description="Language code or 'auto' for detection")
    enable_timestamps: bool = Field(default=True, description="Include word-level timestamps")
    beam_size: int = Field(default=5, ge=1, le=10, description="Beam search size")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")
    suppress_blank: bool = Field(default=True, description="Suppress blank outputs")
    initial_prompt: Optional[str] = Field(default=None, description="Initial prompt for context")

class TTSResponse(PydanticBaseModel):
    """Response model for TTS synthesis."""
    success: bool
    audio_data: Optional[str] = Field(default=None, description="Base64 encoded audio data")
    audio_format: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    processing_time: Optional[float] = None
    model_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class STTResponse(PydanticBaseModel):
    """Response model for STT transcription."""
    success: bool
    text: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = Field(default=None, description="Transcription segments with timestamps")
    language: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    model_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

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
        
        # Log device information for debugging
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device)
            memory_allocated = torch.cuda.memory_allocated(self.device) / (1024**2)  # MB
            memory_total = torch.cuda.get_device_properties(self.device).total_memory / (1024**2)  # MB
            self.logger.info(f"GPU: {gpu_name}, Memory: {memory_allocated:.1f}/{memory_total:.1f} MB")
        elif self.device.type == 'mps':
            self.logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            self.logger.info("Using CPU for inference")
    
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """Preprocess inputs."""
        if isinstance(inputs, list) and all(isinstance(x, (int, float)) for x in inputs):
            # List of numbers - pad or truncate to size 10
            input_list = list(inputs)
            if len(input_list) > 10:
                input_list = input_list[:10]
            elif len(input_list) < 10:
                input_list = input_list + [0.0] * (10 - len(input_list))
            
            tensor_input = torch.tensor(input_list, dtype=torch.float32).unsqueeze(0)
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
                    # Convert to numpy array first to avoid the warning
                    np_array = np.array(list(inputs), dtype=np.float32)
                    tensor_input = torch.from_numpy(np_array)
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
    
    def get_model_for_inference(self):
        """Get the model for inference (required by the forward method)."""
        return self.model
    
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
    """Initialize the inference engine with example model and autoscaler."""
    global inference_engine, autoscaler
    
    try:
        # Get configuration from config manager
        config = config_manager.get_inference_config()
        
        # Apply performance optimizations
        if framework_available:
            try:
                from framework.optimizers.performance_optimizer import optimize_for_inference
                
                # Apply performance optimizations to the model
                try:
                    optimized_model, optimized_device_config = optimize_for_inference(example_model.model, config)
                    example_model.model = optimized_model
                    example_model.device = optimized_device_config.get_torch_device()
                    config.device = optimized_device_config
                    logger.info("Performance optimizations applied successfully")
                except Exception as e:
                    logger.warning(f"Performance optimization failed, using default: {e}")
            except (ImportError, RuntimeError) as e:
                logger.warning(f"Performance optimizer not available: {e}")
        else:
            logger.info("Using basic setup without framework optimizations")
        
        logger.info(f"Initializing optimized inference engine with configuration:")
        logger.info(f"  Device: {config.device.device_type.value}")
        logger.info(f"  Device ID: {config.device.device_id}")
        logger.info(f"  Torch device: {config.device.get_torch_device()}")
        logger.info(f"  Batch size: {config.batch.batch_size}")
        logger.info(f"  FP16: {config.device.use_fp16}")
        logger.info(f"  TensorRT: {config.device.use_tensorrt}")
        logger.info(f"  Torch compile: {config.device.use_torch_compile}")
        
        # Log CUDA availability and GPU info if available
        if torch.cuda.is_available():
            logger.info(f"CUDA available: True")
            logger.info(f"CUDA devices: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                memory_total = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)  # GB
                logger.info(f"Current GPU: {gpu_name} ({memory_total:.1f} GB)")
        else:
            logger.info("CUDA available: False")
        
        # Check MPS availability
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS available: True")
        
        # Create and load example model
        example_model = ExampleModel(config)
        example_model.load_model("example")  # Dummy path
        
        example_model.optimize_for_inference()
        
        # Register model
        model_manager.register_model("example", example_model)
        
        # Initialize autoscaler
        autoscaler_config = AutoscalerConfig(
            enable_zero_scaling=True,
            enable_dynamic_loading=True,
            zero_scaling=ZeroScalingConfig(
                enabled=True,
                scale_to_zero_delay=300.0,  # 5 minutes
                max_loaded_models=5,
                preload_popular_models=True
            ),
            model_loading=ModelLoaderConfig(
                max_instances_per_model=3,
                min_instances_per_model=1,
                enable_model_caching=True,
                prefetch_popular_models=True
            )
        )
        
        autoscaler = Autoscaler(autoscaler_config, model_manager)
        await autoscaler.start()
        
        # Create ultra-fast inference engine for optimal performance
        if framework_available:
            try:
                from framework.core.inference_engine import create_ultra_fast_inference_engine
                inference_engine = create_ultra_fast_inference_engine(example_model, config)
                logger.info("Using enhanced InferenceEngine with ultra-fast optimizations")
            except Exception as e:
                logger.warning(f"Failed to create ultra-fast inference engine, trying hybrid engine: {e}")
                try:
                    from framework.core.inference_engine import create_hybrid_inference_engine
                    inference_engine = create_hybrid_inference_engine(example_model, config)
                    logger.info("Using enhanced InferenceEngine with hybrid optimizations")
                except Exception as e2:
                    logger.warning(f"Failed to create fast inference engine, using standard: {e2}")
                    inference_engine = create_inference_engine(example_model, config)
        else:
            inference_engine = create_inference_engine(example_model, config)
            logger.info("Using basic inference engine")
        
        await inference_engine.start()
        
        # Enhanced warmup with different batch sizes
        logger.info("Performing enhanced warmup for stable performance...")
        example_model.warmup(config.performance.warmup_iterations * 2)  # Double warmup iterations
        
        # Additional warmup for different batch sizes
        with torch.no_grad():
            for batch_size in [1, 2, 4, 8]:
                if batch_size <= config.batch.max_batch_size:
                    try:
                        dummy_input = torch.randn(batch_size, 10, device=example_model.device)
                        for _ in range(3):
                            _ = example_model.model(dummy_input)
                        logger.debug(f"Warmup completed for batch size: {batch_size}")
                    except Exception as e:
                        logger.debug(f"Warmup failed for batch size {batch_size}: {e}")
        
        logger.info("Inference engine and autoscaler initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise

async def cleanup_inference_engine():
    """Cleanup inference engine and autoscaler."""
    global inference_engine, autoscaler
    
    if autoscaler:
        await autoscaler.stop()
        autoscaler = None
    
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
            },
            "audio": {
                "tts_synthesize": "/tts/synthesize",
                "stt_transcribe": "/stt/transcribe",
                "models": "/audio/models",
                "health": "/audio/health"
            }
        }
    }
    
    logger.debug(f"[ENDPOINT] Root endpoint response: {response_data}")
    return response_data

@app.post("/predict")
async def predict(request: InferenceRequest) -> InferenceResponse:
    """
    General prediction endpoint using default model.
    
    Performs inference using the default 'example' model.
    """
    logger.info("[ENDPOINT] General predict endpoint accessed")
    
    # Use the existing predict_model function with default model
    return await predict_model("example", request)

@app.post("/predict/batch")
async def predict_batch(request: InferenceRequest) -> InferenceResponse:
    """
    Batch prediction endpoint.
    
    Optimized for batch processing with multiple inputs.
    """
    logger.info("[ENDPOINT] Batch predict endpoint accessed")
    
    # Ensure inputs is a list for batch processing
    if not isinstance(request.inputs, list):
        request.inputs = [request.inputs]
    
    # Use the existing predict_model function with default model
    return await predict_model("example", request)

@app.post("/{model_name}/predict")
async def predict_model(model_name: str, request: InferenceRequest) -> InferenceResponse:
    """
    Ultra-optimized model-specific prediction endpoint.
    
    Optimized for:
    - Ultra-low latency (<600ms target)
    - High concurrent request throughput
    - Minimal processing overhead
    - Robust error handling
    """
    # Determine if this is a batch request or single request
    is_batch_input = isinstance(request.inputs, list) and len(request.inputs) > 1
    input_count = len(request.inputs) if isinstance(request.inputs, list) else 1
    
    logger.debug(f"[ENDPOINT] Optimized prediction - Model: {model_name}, "
                f"Type: {'batch' if is_batch_input else 'single'}, Count: {input_count}")
    
    # Check if model exists or use default
    if model_name not in model_manager.list_models():
        if model_name != "example":
            logger.debug(f"[ENDPOINT] Model '{model_name}' not found, using 'example'")
            model_name = "example"
    
    if not autoscaler and not inference_engine:
        logger.error("[ENDPOINT] No inference services available")
        raise HTTPException(status_code=503, detail="Inference services not available")
    
    try:
        start_time = time.perf_counter()
        
        # Optimized request handling based on type
        if is_batch_input and len(request.inputs) <= 8:  # Small batches - process as batch
            logger.debug(f"[ENDPOINT] Processing small batch ({input_count} items)")
            
            if autoscaler:
                # Process batch through autoscaler
                tasks = [autoscaler.predict(model_name, input_item, priority=request.priority, timeout=request.timeout)
                        for input_item in request.inputs]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle any exceptions in results
                final_results = []
                for result in results:
                    if isinstance(result, Exception):
                        final_results.append({"error": str(result)})
                    else:
                        final_results.append(result)
                result = final_results
            else:
                # Use inference engine batch processing
                result = await inference_engine.predict_batch(
                    inputs_list=request.inputs,
                    priority=request.priority,
                    timeout=request.timeout or 1.0
                )
        
        elif is_batch_input:  # Large batches - process concurrently
            logger.debug(f"[ENDPOINT] Processing large batch ({input_count} items) concurrently")
            
            # Split into smaller chunks for better performance
            chunk_size = 4
            chunks = [request.inputs[i:i + chunk_size] for i in range(0, len(request.inputs), chunk_size)]
            
            # Process chunks concurrently
            chunk_tasks = []
            for chunk in chunks:
                if autoscaler:
                    task_batch = [autoscaler.predict(model_name, item, priority=request.priority, timeout=request.timeout)
                                 for item in chunk]
                    chunk_tasks.append(asyncio.gather(*task_batch, return_exceptions=True))
                else:
                    chunk_tasks.append(inference_engine.predict_batch(chunk, request.priority, request.timeout or 1.0))
            
            # Gather all chunk results
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            # Flatten results
            result = []
            for chunk_result in chunk_results:
                if isinstance(chunk_result, Exception):
                    result.extend([{"error": str(chunk_result)}] * chunk_size)
                else:
                    result.extend(chunk_result)
        
        else:  # Single input - fastest path
            logger.debug(f"[ENDPOINT] Processing single input (fastest path)")
            
            if autoscaler:
                result = await autoscaler.predict(model_name, request.inputs, priority=request.priority, timeout=request.timeout)
            else:
                result = await inference_engine.predict(
                    inputs=request.inputs,
                    priority=request.priority,
                    timeout=request.timeout or 1.0
                )
        
        processing_time = time.perf_counter() - start_time
        
        logger.debug(f"[ENDPOINT] Prediction completed - Model: {model_name}, "
                   f"Type: {'batch' if is_batch_input else 'single'}, "
                   f"Time: {processing_time*1000:.1f}ms")
        
        return InferenceResponse(
            success=True,
            result=result,
            processing_time=processing_time,
            model_info={
                "model": model_name, 
                "device": str(inference_engine.device) if inference_engine else "autoscaler",
                "input_type": "batch" if is_batch_input else "single",
                "input_count": input_count,
                "processing_path": "optimized"
            },
            batch_info={
                "inflight_batching_enabled": request.enable_batching,
                "processed_as_batch": is_batch_input,
                "concurrent_optimization": is_batch_input and input_count > 8
            }
        )
        
    except asyncio.TimeoutError:
        logger.warning(f"[ENDPOINT] Prediction timeout for model {model_name}")
        return InferenceResponse(
            success=False,
            error="Request timed out",
            model_info={"model": model_name},
            processing_time=time.perf_counter() - start_time if 'start_time' in locals() else 0
        )
    except Exception as e:
        logger.error(f"[ENDPOINT] Prediction failed for model {model_name}: {e}")
        return InferenceResponse(
            success=False,
            error=str(e),
            model_info={"model": model_name},
            processing_time=time.perf_counter() - start_time if 'start_time' in locals() else 0
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

# GPU Detection endpoints
@app.get("/gpu/detect")
async def detect_gpus_endpoint(include_benchmarks: bool = False):
    """Detect available GPUs."""
    logger.info(f"[ENDPOINT] GPU detection requested - Include benchmarks: {include_benchmarks}")
    
    try:
        gpu_manager = GPUManager()
        gpus, _ = gpu_manager.detect_and_configure(force_refresh=True)
        
        gpu_list = []
        for gpu in gpus:
            gpu_info = {
                "id": gpu.id,
                "name": gpu.name,
                "vendor": gpu.vendor.value,
                "architecture": gpu.architecture.value,
                "device_id": gpu.device_id,
                "memory_mb": gpu.memory.total_mb,
                "available_memory_mb": gpu.memory.available_mb,
                "pytorch_support": gpu.pytorch_support,
                "suitable_for_inference": gpu.is_suitable_for_inference(),
                "recommended_precisions": gpu.get_recommended_precision(),
                "supported_accelerators": [acc.value for acc in gpu.supported_accelerators]
            }
            
            if gpu.compute_capability:
                gpu_info["compute_capability"] = {
                    "major": gpu.compute_capability.major,
                    "minor": gpu.compute_capability.minor,
                    "version": gpu.compute_capability.version,
                    "supports_fp16": gpu.compute_capability.supports_fp16,
                    "supports_int8": gpu.compute_capability.supports_int8,
                    "supports_tensor_cores": gpu.compute_capability.supports_tensor_cores,
                    "supports_tf32": gpu.compute_capability.supports_tf32
                }
            
            if include_benchmarks and gpu.benchmark_results:
                gpu_info["benchmark_results"] = gpu.benchmark_results
            
            gpu_list.append(gpu_info)
        
        logger.info(f"[ENDPOINT] GPU detection completed - Found {len(gpus)} GPU(s)")
        
        return {
            "gpus": gpu_list,
            "total_gpus": len(gpus),
            "suitable_gpus": sum(1 for gpu in gpus if gpu.is_suitable_for_inference()),
            "include_benchmarks": include_benchmarks
        }
        
    except Exception as e:
        logger.error(f"[ENDPOINT] GPU detection failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpu/best")
async def get_best_gpu_endpoint():
    """Get the best GPU for inference."""
    logger.info("[ENDPOINT] Best GPU request")
    
    try:
        gpu_manager = GPUManager()
        best_gpu = gpu_manager.get_best_gpu_info()
        
        if not best_gpu:
            logger.info("[ENDPOINT] No suitable GPU found")
            return {
                "best_gpu": None,
                "message": "No suitable GPU found for inference"
            }
        
        gpu_info = {
            "id": best_gpu.id,
            "name": best_gpu.name,
            "vendor": best_gpu.vendor.value,
            "architecture": best_gpu.architecture.value,
            "device_id": best_gpu.device_id,
            "memory_mb": best_gpu.memory.total_mb,
            "available_memory_mb": best_gpu.memory.available_mb,
            "pytorch_support": best_gpu.pytorch_support,
            "recommended_precisions": best_gpu.get_recommended_precision(),
            "estimated_max_batch_size": best_gpu.estimate_max_batch_size()
        }
        
        if best_gpu.compute_capability:
            gpu_info["compute_capability"] = {
                "version": best_gpu.compute_capability.version,
                "supports_tensor_cores": best_gpu.compute_capability.supports_tensor_cores,
                "supports_fp16": best_gpu.compute_capability.supports_fp16
            }
        
        logger.info(f"[ENDPOINT] Best GPU: {best_gpu.name}")
        
        return {
            "best_gpu": gpu_info,
            "message": f"Best GPU for inference: {best_gpu.name}"
        }
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Best GPU detection failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpu/config")
async def get_gpu_config_endpoint():
    """Get GPU-optimized configuration."""
    logger.info("[ENDPOINT] GPU configuration request")
    
    try:
        gpu_manager = GPUManager()
        gpus, device_config = gpu_manager.detect_and_configure()
        
        memory_rec = gpu_manager.get_memory_recommendations()
        optimization_rec = gpu_manager.get_optimization_recommendations()
        
        config_info = {
            "device_config": {
                "device_type": device_config.device_type.value,
                "device_id": device_config.device_id,
                "use_fp16": device_config.use_fp16,
                "use_int8": device_config.use_int8,
                "use_tensorrt": device_config.use_tensorrt,
                "use_torch_compile": device_config.use_torch_compile,
                "compile_mode": device_config.compile_mode
            },
            "memory_recommendations": memory_rec,
            "optimization_recommendations": optimization_rec,
            "pytorch_device": str(device_config.get_torch_device())
        }
        
        logger.info(f"[ENDPOINT] GPU configuration generated for: {device_config.device_type.value}")
        
        return config_info
        
    except Exception as e:
        logger.error(f"[ENDPOINT] GPU configuration failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpu/report")
async def get_gpu_report_endpoint(format: str = "json"):
    """Get comprehensive GPU report."""
    logger.info(f"[ENDPOINT] GPU report requested - Format: {format}")
    
    try:
        gpu_manager = GPUManager()
        
        if format.lower() == "text":
            report = gpu_manager.generate_full_report()
            return Response(content=report, media_type="text/plain")
        else:
            # JSON format
            gpus, device_config = gpu_manager.detect_and_configure()
            memory_rec = gpu_manager.get_memory_recommendations()
            optimization_rec = gpu_manager.get_optimization_recommendations()
            
            gpu_list = []
            for gpu in gpus:
                gpu_info = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "vendor": gpu.vendor.value,
                    "architecture": gpu.architecture.value,
                    "device_id": gpu.device_id,
                    "memory_mb": gpu.memory.total_mb,
                    "available_memory_mb": gpu.memory.available_mb,
                    "pytorch_support": gpu.pytorch_support,
                    "suitable_for_inference": gpu.is_suitable_for_inference(),
                    "recommended_precisions": gpu.get_recommended_precision(),
                    "estimated_max_batch_size": gpu.estimate_max_batch_size()
                }
                
                if gpu.compute_capability:
                    gpu_info["compute_capability"] = {
                        "major": gpu.compute_capability.major,
                        "minor": gpu.compute_capability.minor,
                        "version": gpu.compute_capability.version,
                        "supports_fp16": gpu.compute_capability.supports_fp16,
                        "supports_int8": gpu.compute_capability.supports_int8,
                        "supports_tensor_cores": gpu.compute_capability.supports_tensor_cores
                    }
                
                if gpu.benchmark_results:
                    gpu_info["benchmark_results"] = gpu.benchmark_results
                
                gpu_list.append(gpu_info)
            
            report = {
                "summary": {
                    "total_gpus": len(gpus),
                    "suitable_gpus": sum(1 for gpu in gpus if gpu.is_suitable_for_inference()),
                    "best_gpu": gpu_manager.get_best_gpu_info().name if gpu_manager.get_best_gpu_info() else None
                },
                "gpus": gpu_list,
                "device_config": {
                    "device_type": device_config.device_type.value,
                    "device_id": device_config.device_id,
                    "use_fp16": device_config.use_fp16,
                    "use_int8": device_config.use_int8,
                    "use_tensorrt": device_config.use_tensorrt,
                    "use_torch_compile": device_config.use_torch_compile
                },
                "recommendations": {
                    "memory": memory_rec,
                    "optimization": optimization_rec
                }
            }
            
            logger.info(f"[ENDPOINT] GPU report generated - {len(gpus)} GPU(s)")
            
            return report
        
    except Exception as e:
        logger.error(f"[ENDPOINT] GPU report failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Autoscaler endpoints
@app.get("/autoscaler/stats")
async def get_autoscaler_stats():
    """Get autoscaler statistics."""
    logger.info("[ENDPOINT] Autoscaler stats requested")
    
    if not autoscaler:
        logger.error("[ENDPOINT] Autoscaler stats failed - Autoscaler not available")
        raise HTTPException(status_code=503, detail="Autoscaler not available")
    
    try:
        stats = autoscaler.get_stats()
        logger.info("[ENDPOINT] Autoscaler stats retrieved successfully")
        return stats
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Autoscaler stats failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/autoscaler/health")
async def get_autoscaler_health():
    """Get autoscaler health status."""
    logger.info("[ENDPOINT] Autoscaler health check requested")
    
    if not autoscaler:
        return {
            "healthy": False,
            "error": "Autoscaler not available",
            "timestamp": time.time()
        }
    
    try:
        health = autoscaler.get_health_status()
        logger.info(f"[ENDPOINT] Autoscaler health check completed - Healthy: {health['healthy']}")
        return health
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Autoscaler health check failed with error: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "timestamp": time.time()
        }


@app.post("/autoscaler/scale")
async def scale_model(model_name: str, target_instances: int):
    """Scale a model to target number of instances."""
    logger.info(f"[ENDPOINT] Model scaling requested - Model: {model_name}, Target: {target_instances}")
    
    if not autoscaler:
        logger.error("[ENDPOINT] Model scaling failed - Autoscaler not available")
        raise HTTPException(status_code=503, detail="Autoscaler not available")
    
    if target_instances < 0 or target_instances > 10:
        raise HTTPException(status_code=400, detail="Target instances must be between 0 and 10")
    
    try:
        success = await autoscaler.scale_model(model_name, target_instances)
        
        if success:
            logger.info(f"[ENDPOINT] Model scaling completed successfully - {model_name} to {target_instances} instances")
            return {
                "success": True,
                "message": f"Successfully scaled {model_name} to {target_instances} instances",
                "model_name": model_name,
                "target_instances": target_instances
            }
        else:
            logger.error(f"[ENDPOINT] Model scaling failed - {model_name}")
            return {
                "success": False,
                "message": f"Failed to scale {model_name}",
                "model_name": model_name,
                "target_instances": target_instances
            }
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Model scaling failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/autoscaler/load")
async def load_model_autoscaler(model_name: str, version: str = "v1"):
    """Load a model through the autoscaler."""
    logger.info(f"[ENDPOINT] Model loading requested via autoscaler - Model: {model_name}, Version: {version}")
    
    if not autoscaler:
        logger.error("[ENDPOINT] Model loading failed - Autoscaler not available")
        raise HTTPException(status_code=503, detail="Autoscaler not available")
    
    try:
        success = await autoscaler.load_model(model_name, version)
        
        if success:
            logger.info(f"[ENDPOINT] Model loading completed successfully - {model_name}:{version}")
            return {
                "success": True,
                "message": f"Successfully loaded {model_name}:{version}",
                "model_name": model_name,
                "version": version
            }
        else:
            logger.error(f"[ENDPOINT] Model loading failed - {model_name}:{version}")
            return {
                "success": False,
                "message": f"Failed to load {model_name}:{version}",
                "model_name": model_name,
                "version": version
            }
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Model loading failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/autoscaler/unload")
async def unload_model_autoscaler(model_name: str, version: Optional[str] = None):
    """Unload a model through the autoscaler."""
    version_str = f":{version}" if version else ""
    logger.info(f"[ENDPOINT] Model unloading requested via autoscaler - Model: {model_name}{version_str}")
    
    if not autoscaler:
        logger.error("[ENDPOINT] Model unloading failed - Autoscaler not available")
        raise HTTPException(status_code=503, detail="Autoscaler not available")
    
    try:
        success = await autoscaler.unload_model(model_name, version)
        
        if success:
            logger.info(f"[ENDPOINT] Model unloading completed successfully - {model_name}{version_str}")
            return {
                "success": True,
                "message": f"Successfully unloaded {model_name}{version_str}",
                "model_name": model_name,
                "version": version
            }
        else:
            logger.error(f"[ENDPOINT] Model unloading failed - {model_name}{version_str}")
            return {
                "success": False,
                "message": f"Failed to unload {model_name}{version_str}",
                "model_name": model_name,
                "version": version
            }
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Model unloading failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/autoscaler/metrics")
async def get_autoscaler_metrics(window_seconds: Optional[int] = None):
    """Get detailed autoscaling metrics."""
    logger.info(f"[ENDPOINT] Autoscaler metrics requested - Window: {window_seconds}s")
    
    if not autoscaler:
        logger.error("[ENDPOINT] Autoscaler metrics failed - Autoscaler not available")
        raise HTTPException(status_code=503, detail="Autoscaler not available")
    
    try:
        stats = autoscaler.get_stats()
        
        # Add metrics from metrics collector if available
        if hasattr(autoscaler, 'metrics_collector') and autoscaler.metrics_collector:
            metrics_summary = autoscaler.metrics_collector.get_summary(window_seconds)
            stats['detailed_metrics'] = metrics_summary
        
        logger.info("[ENDPOINT] Autoscaler metrics retrieved successfully")
        return {
            "metrics": stats,
            "window_seconds": window_seconds,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Autoscaler metrics failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Audio Processing Endpoints

@app.post("/tts/synthesize", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest) -> TTSResponse:
    """
    Text-to-Speech synthesis endpoint.
    
    Converts text input to speech audio using the specified TTS model.
    Returns base64 encoded audio data.
    """
    logger.info(f"[ENDPOINT] TTS synthesis requested - Model: {request.model_name}, Text length: {len(request.text)}")
    
    try:
        start_time = time.perf_counter()
        
        # Import audio modules dynamically
        try:
            from framework.models.audio import create_tts_model
            from framework.processors.audio import AudioPreprocessor
        except ImportError as e:
            logger.error(f"[ENDPOINT] Audio modules not available: {e}")
            raise HTTPException(
                status_code=503, 
                detail="Audio processing not available. Install audio dependencies."
            )
        
        # Validate text length
        if len(request.text) > 5000:  # 5000 character limit
            raise HTTPException(
                status_code=400,
                detail="Text too long. Maximum 5000 characters allowed."
            )
        
        # Get or create TTS model
        try:
            if request.model_name not in model_manager.list_models():
                logger.info(f"[ENDPOINT] Loading TTS model: {request.model_name}")
                config = config_manager.get_inference_config()
                tts_model = create_tts_model(request.model_name, config)
                model_manager.register_model(request.model_name, tts_model)
            else:
                tts_model = model_manager.get_model(request.model_name)
        except Exception as e:
            logger.error(f"[ENDPOINT] Failed to load TTS model {request.model_name}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load TTS model: {str(e)}"
            )
        
        # Prepare synthesis parameters
        synthesis_params = {
            "voice": request.voice,
            "speed": request.speed,
            "pitch": request.pitch,
            "volume": request.volume,
            "language": request.language,
            "emotion": request.emotion
        }
        
        # Perform TTS synthesis
        try:
            audio_data, sample_rate = await tts_model.synthesize(
                text=request.text,
                **{k: v for k, v in synthesis_params.items() if v is not None}
            )
            
            # Convert audio to requested format
            if request.output_format != "wav":
                # For now, we'll keep WAV format and note the requested format
                # Additional format conversion can be implemented here
                pass
            
            # Calculate duration
            duration = len(audio_data) / sample_rate if sample_rate > 0 else None
            
            # Encode audio as base64
            if isinstance(audio_data, np.ndarray):
                # Convert numpy array to bytes
                audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            else:
                audio_bytes = audio_data
            
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            processing_time = time.perf_counter() - start_time
            
            logger.info(f"[ENDPOINT] TTS synthesis completed - Duration: {duration:.2f}s, "
                       f"Processing time: {processing_time:.3f}s")
            
            return TTSResponse(
                success=True,
                audio_data=audio_base64,
                audio_format=request.output_format,
                duration=duration,
                sample_rate=sample_rate,
                processing_time=processing_time,
                model_info={
                    "model_name": request.model_name,
                    "voice": request.voice,
                    "language": request.language
                }
            )
            
        except Exception as e:
            logger.error(f"[ENDPOINT] TTS synthesis failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"TTS synthesis failed: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ENDPOINT] TTS endpoint failed with unexpected error: {e}")
        return TTSResponse(
            success=False,
            error=str(e),
            processing_time=time.perf_counter() - start_time if 'start_time' in locals() else None
        )


@app.post("/stt/transcribe", response_model=STTResponse)
async def speech_to_text(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model_name: str = Form(default="whisper-base"),
    language: str = Form(default="auto"),
    enable_timestamps: bool = Form(default=True),
    beam_size: int = Form(default=5),
    temperature: float = Form(default=0.0),
    suppress_blank: bool = Form(default=True),
    initial_prompt: Optional[str] = Form(default=None)
) -> STTResponse:
    """
    Speech-to-Text transcription endpoint.
    
    Transcribes uploaded audio file to text using the specified STT model.
    Supports various audio formats and returns text with optional timestamps.
    """
    logger.info(f"[ENDPOINT] STT transcription requested - Model: {model_name}, "
               f"File: {file.filename}, Size: {file.size if hasattr(file, 'size') else 'unknown'}")
    
    try:
        start_time = time.perf_counter()
        
        # Import audio modules dynamically
        try:
            from framework.models.audio import create_stt_model
            from framework.processors.audio import AudioPreprocessor
        except ImportError as e:
            logger.error(f"[ENDPOINT] Audio modules not available: {e}")
            raise HTTPException(
                status_code=503,
                detail="Audio processing not available. Install audio dependencies."
            )
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file extension
        allowed_extensions = config_manager.get_config().get("security", {}).get("allowed_extensions", [])
        audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in audio_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {file_ext}. Supported: {audio_extensions}"
            )
        
        # Check file size
        max_size_mb = config_manager.get_config().get("security", {}).get("max_file_size_mb", 100)
        if hasattr(file, 'size') and file.size > max_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {max_size_mb}MB"
            )
        
        # Get or create STT model
        try:
            if model_name not in model_manager.list_models():
                logger.info(f"[ENDPOINT] Loading STT model: {model_name}")
                config = config_manager.get_inference_config()
                stt_model = create_stt_model(model_name, config)
                model_manager.register_model(model_name, stt_model)
            else:
                stt_model = model_manager.get_model(model_name)
        except Exception as e:
            logger.error(f"[ENDPOINT] Failed to load STT model {model_name}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load STT model: {str(e)}"
            )
        
        # Read and preprocess audio
        try:
            # Read file content
            audio_content = await file.read()
            
            # Save to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
                temp_file.write(audio_content)
                temp_file_path = temp_file.name
            
            # Load and preprocess audio
            config = config_manager.get_inference_config()
            audio_processor = AudioPreprocessor(config)
            audio_data, sample_rate = audio_processor.load_audio(temp_file_path)
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
        except Exception as e:
            logger.error(f"[ENDPOINT] Audio preprocessing failed: {e}")
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process audio file: {str(e)}"
            )
        
        # Prepare transcription parameters
        transcription_params = {
            "language": language if language != "auto" else None,
            "enable_timestamps": enable_timestamps,
            "beam_size": beam_size,
            "temperature": temperature,
            "suppress_blank": suppress_blank,
            "initial_prompt": initial_prompt
        }
        
        # Perform STT transcription
        try:
            result = await stt_model.transcribe(
                audio_data=audio_data,
                sample_rate=sample_rate,
                **{k: v for k, v in transcription_params.items() if v is not None}
            )
            
            # Extract results
            text = result.get("text", "")
            segments = result.get("segments", [])
            detected_language = result.get("language")
            confidence = result.get("confidence")
            
            processing_time = time.perf_counter() - start_time
            
            logger.info(f"[ENDPOINT] STT transcription completed - Text length: {len(text)}, "
                       f"Language: {detected_language}, Processing time: {processing_time:.3f}s")
            
            return STTResponse(
                success=True,
                text=text,
                segments=segments if enable_timestamps else None,
                language=detected_language,
                confidence=confidence,
                processing_time=processing_time,
                model_info={
                    "model_name": model_name,
                    "language": detected_language or language,
                    "file_name": file.filename
                }
            )
            
        except Exception as e:
            logger.error(f"[ENDPOINT] STT transcription failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"STT transcription failed: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ENDPOINT] STT endpoint failed with unexpected error: {e}")
        return STTResponse(
            success=False,
            error=str(e),
            processing_time=time.perf_counter() - start_time if 'start_time' in locals() else None
        )


@app.get("/audio/models")
async def list_audio_models():
    """List available audio models (TTS and STT)."""
    logger.info("[ENDPOINT] Audio models list requested")
    
    try:
        from framework.models.audio import get_available_tts_models, get_available_stt_models
        
        tts_models = get_available_tts_models()
        stt_models = get_available_stt_models()
        
        return {
            "tts_models": tts_models,
            "stt_models": stt_models,
            "loaded_models": [name for name in model_manager.list_models() 
                            if any(audio_type in name.lower() for audio_type in ['tts', 'stt', 'whisper', 'tacotron', 'wav2vec'])]
        }
        
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Audio models not available. Install audio dependencies."
        )
    except Exception as e:
        logger.error(f"[ENDPOINT] Failed to list audio models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/health")
async def audio_health_check():
    """Check audio processing health and capabilities."""
    logger.info("[ENDPOINT] Audio health check requested")
    
    health_status = {
        "audio_available": False,
        "tts_available": False,
        "stt_available": False,
        "dependencies": {},
        "errors": []
    }
    
    # Check audio dependencies
    dependencies_to_check = [
        ("librosa", "Audio processing"),
        ("soundfile", "Audio I/O"),
        ("torchaudio", "PyTorch audio"),
        ("transformers", "HuggingFace models")
    ]
    
    for dep_name, dep_desc in dependencies_to_check:
        try:
            __import__(dep_name)
            health_status["dependencies"][dep_name] = {"available": True, "description": dep_desc}
        except ImportError as e:
            health_status["dependencies"][dep_name] = {
                "available": False, 
                "description": dep_desc,
                "error": str(e)
            }
            health_status["errors"].append(f"{dep_name}: {str(e)}")
    
    # Check if audio modules can be imported
    try:
        from framework.models.audio import create_tts_model, create_stt_model
        health_status["tts_available"] = True
        health_status["stt_available"] = True
        health_status["audio_available"] = True
    except ImportError as e:
        health_status["errors"].append(f"Audio modules: {str(e)}")
    
    return health_status

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

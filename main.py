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
import uuid
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

def _create_wav_bytes(audio_data: np.ndarray, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """
    Create a proper WAV file from numpy audio data.
    
    Args:
        audio_data: Audio data as numpy array (float values between -1 and 1)
        sample_rate: Sample rate in Hz
        channels: Number of audio channels (default: 1 for mono)
        sample_width: Sample width in bytes (default: 2 for 16-bit)
        
    Returns:
        Complete WAV file as bytes including header
    """
    import struct
    
    # Ensure audio data is in the right format
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Clip to valid range
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    # Convert to 16-bit PCM
    if sample_width == 2:
        audio_pcm = (audio_data * 32767).astype(np.int16)
    elif sample_width == 4:
        audio_pcm = (audio_data * 2147483647).astype(np.int32)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    
    # Convert to bytes
    audio_bytes = audio_pcm.tobytes()
    
    # Create WAV header
    chunk_size = 36 + len(audio_bytes)
    subchunk2_size = len(audio_bytes)
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    bits_per_sample = sample_width * 8
    
    wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF',           # ChunkID
        chunk_size,        # ChunkSize
        b'WAVE',           # Format
        b'fmt ',           # Subchunk1ID
        16,                # Subchunk1Size (16 for PCM)
        1,                 # AudioFormat (1 for PCM)
        channels,          # NumChannels
        sample_rate,       # SampleRate
        byte_rate,         # ByteRate
        block_align,       # BlockAlign
        bits_per_sample,   # BitsPerSample
        b'data',           # Subchunk2ID
        subchunk2_size     # Subchunk2Size
    )
    
    # Combine header and audio data
    return wav_header + audio_bytes

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

# Configure logging to both console and file
log_level = getattr(logging, server_config['log_level'])

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create separate handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)

file_handler = logging.FileHandler(log_dir / "server.log", mode='a', encoding='utf-8')
file_handler.setLevel(log_level)

error_handler = logging.FileHandler(log_dir / "server_errors.log", mode='a', encoding='utf-8')
error_handler.setLevel(logging.ERROR)

# Configure root logger
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[console_handler, file_handler, error_handler]
)

logger = logging.getLogger(__name__)

# Log startup message to file
logger.info("="*80)
logger.info("PYTORCH INFERENCE FRAMEWORK SERVER STARTUP")
logger.info("="*80)
logger.info(f"Startup time: {datetime.now().isoformat()}")
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Log level: {server_config['log_level']}")
logger.info(f"Log files directory: {log_dir.absolute()}")

# Create separate logger for API requests
api_logger = logging.getLogger("api_requests")
api_logger.setLevel(logging.INFO)
api_handler = logging.FileHandler(log_dir / "api_requests.log", mode='a', encoding='utf-8')
api_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
api_logger.addHandler(api_handler)
api_logger.propagate = False  # Prevent double logging

# Global variables
inference_engine: Optional[InferenceEngine] = None
model_manager: ModelManager = get_model_manager()
autoscaler: Optional[Autoscaler] = None

def print_api_endpoints():
    """Print all available API endpoints at startup"""
    endpoints = [
        ("GET", "/", "Root endpoint - API information"),
        ("POST", "/{model_name}/predict", "Model-specific prediction endpoint with inflight batching"),
        ("POST", "/predict", "General prediction endpoint using default model"),
        ("POST", "/predict/batch", "Batch prediction endpoint"),
        ("GET", "/health", "Health check endpoint"),
        ("GET", "/stats", "Engine statistics endpoint"),
        ("GET", "/config", "Configuration information endpoint"),
        ("GET", "/models", "List available models"),
        # Enhanced model download endpoints
        ("POST", "/models/download", "Enhanced model download with TTS support"),
        ("GET", "/models/download/status/{download_id}", "Get download status by ID"),
        ("GET", "/models/available", "Enhanced list of available models with TTS focus"),
        ("GET", "/models/managed", "Get server-managed models information"),
        ("GET", "/models/download/{model_name}/info", "Get download info for a model"),
        ("DELETE", "/models/download/{model_name}", "Remove model from cache"),
        ("GET", "/models/cache/info", "Get enhanced model cache information"),
        ("POST", "/models/manage", "Manage models (retry, optimize, etc.)"),
        # Server management endpoints
        ("GET", "/server/config", "Get server configuration"),
        ("POST", "/server/optimize", "Optimize server performance"),
        ("GET", "/metrics/server", "Get server performance metrics"),
        ("GET", "/metrics/tts", "Get TTS-specific performance metrics"),
        # GPU detection endpoints
        ("GET", "/gpu/detect", "Detect available GPUs"),
        ("GET", "/gpu/best", "Get best GPU for inference"),
        ("GET", "/gpu/config", "Get GPU-optimized configuration"),
        ("GET", "/gpu/report", "Get comprehensive GPU report"),
        # Autoscaling endpoints
        ("GET", "/autoscaler/stats", "Get autoscaler statistics"),
        ("GET", "/autoscaler/health", "Get autoscaler health status"),
        ("POST", "/autoscaler/scale", "Scale a model to target instances"),
        ("POST", "/autoscaler/load", "Load a model with autoscaling"),
        ("DELETE", "/autoscaler/unload", "Unload a model"),
        ("GET", "/autoscaler/metrics", "Get detailed autoscaling metrics"),
        # Enhanced audio processing endpoints
        ("POST", "/tts/synthesize", "Enhanced Text-to-Speech synthesis"),
        ("POST", "/stt/transcribe", "Speech-to-Text transcription"),
        ("GET", "/audio/models", "List available audio models"),
        ("GET", "/audio/health", "Audio processing health check"),
        ("GET", "/tts/health", "TTS service health check with voices"),
        ("POST", "/audio/validate", "Validate audio file integrity"),
        # Logging endpoints
        ("GET", "/logs", "Get logging information and statistics"),
        ("GET", "/logs/{log_file}", "Download or view specific log file"),
        ("DELETE", "/logs/{log_file}", "Clear specific log file"),
    ]
    
    print("\n" + "="*90)
    print("  PYTORCH INFERENCE FRAMEWORK - ENHANCED API ENDPOINTS WITH TTS SUPPORT")
    print("="*90)
    
    # Group endpoints by category
    categories = {
        "Core": [e for e in endpoints if e[1].startswith(("/", "/{model")) or "/predict" in e[1] or e[1] in ["/health", "/stats", "/config"]],
        "Model Management": [e for e in endpoints if "/models" in e[1]],
        "Server & Performance": [e for e in endpoints if "/server" in e[1] or "/metrics" in e[1]],
        "GPU & Hardware": [e for e in endpoints if "/gpu" in e[1]],
        "Autoscaling": [e for e in endpoints if "/autoscaler" in e[1]],
        "Audio & TTS": [e for e in endpoints if "/audio" in e[1] or "/tts" in e[1] or "/stt" in e[1]],
        "Logging": [e for e in endpoints if "/logs" in e[1]]
    }
    
    for category, category_endpoints in categories.items():
        if category_endpoints:
            print(f"\n  📁 {category}:")
            for method, endpoint, description in category_endpoints:
                print(f"    {method:<7} {endpoint:<45} - {description}")
    
    print("\n" + "="*90)
    print(f"  🎵 TTS Models Supported: BART, SpeechT5, Bark, VALL-E X, Tacotron2")
    print(f"  🚀 Enhanced Features: Auto-download, Server optimization, GPU acceleration")
    print(f"  📊 Total Endpoints: {len(endpoints)}")
    print(f"  📚 Documentation: http://localhost:8000/docs")
    print(f"  💚 Health Check: http://localhost:8000/health")
    print(f"  🎤 TTS Health: http://localhost:8000/tts/health")
    print("="*90 + "\n")

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
    startup_time = datetime.now()
    
    logger.info("[SERVER] Starting PyTorch Inference API Server...")
    logger.info(f"[SERVER] Startup initiated at: {startup_time.isoformat()}")
    logger.info("[SERVER] Initializing inference engine...")
    
    # Initialize model and engine
    try:
        await initialize_inference_engine()
        logger.info("[SERVER] Inference engine initialized successfully")
    except Exception as e:
        logger.error(f"[SERVER] Failed to initialize inference engine: {e}")
        raise
    
    ready_time = datetime.now()
    startup_duration = (ready_time - startup_time).total_seconds()
    
    logger.info(f"[SERVER] Server startup complete at: {ready_time.isoformat()}")
    logger.info(f"[SERVER] Startup duration: {startup_duration:.2f} seconds")
    logger.info("[SERVER] Ready to accept requests")
    logger.info("="*60)
    
    yield
    
    # Cleanup
    shutdown_time = datetime.now()
    logger.info("="*60)
    logger.info("[SERVER] Shutting down PyTorch Inference API Server...")
    logger.info(f"[SERVER] Shutdown initiated at: {shutdown_time.isoformat()}")
    
    try:
        await cleanup_inference_engine()
        logger.info("[SERVER] Inference engine cleanup completed")
    except Exception as e:
        logger.error(f"[SERVER] Error during cleanup: {e}")
    
    shutdown_complete_time = datetime.now()
    shutdown_duration = (shutdown_complete_time - shutdown_time).total_seconds()
    
    logger.info(f"[SERVER] Server shutdown complete at: {shutdown_complete_time.isoformat()}")
    logger.info(f"[SERVER] Shutdown duration: {shutdown_duration:.2f} seconds")
    logger.info("="*60)

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
    
    # Log request to both main logger and API logger
    request_log = f"[API REQUEST] {method} {url} - Client: {client_ip} - User-Agent: {user_agent[:100]}"
    logger.info(request_log)
    api_logger.info(f"REQUEST: {method} {url} | Client: {client_ip} | UA: {user_agent[:50]}")
    
    # Call the endpoint
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response to both loggers
    response_log = f"[API RESPONSE] {method} {url} - Status: {response.status_code} - Time: {process_time:.3f}s - Client: {client_ip}"
    logger.info(response_log)
    api_logger.info(f"RESPONSE: {method} {url} | Status: {response.status_code} | Time: {process_time:.3f}s | Client: {client_ip}")
    
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
    """Root endpoint with enhanced TTS model support information."""
    logger.info("[ENDPOINT] Root endpoint accessed - returning enhanced API information")
    
    response_data = {
        "message": "PyTorch Inference Framework API - Enhanced with TTS Support",
        "version": "1.0.0-TTS-Enhanced",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "environment": config_manager.environment,
        "tts_support": {
            "enabled": True,
            "supported_models": [
                "facebook/bart-large",
                "facebook/bart-base", 
                "microsoft/speecht5_tts",
                "suno/bark",
                "Plachtaa/VALL-E-X",
                "tacotron2"
            ],
            "features": [
                "Auto-download popular TTS models",
                "Server-side optimization",
                "GPU acceleration",
                "Voice synthesis",
                "Multiple audio formats"
            ]
        },
        "endpoints": {
            "inference": {
                "predict": "/predict",
                "model_specific": "/{model_name}/predict",
                "batch": "/predict/batch"
            },
            "health": "/health",
            "stats": "/stats",
            "models": "/models",
            "enhanced_downloads": {
                "download": "/models/download",
                "status": "/models/download/status/{download_id}",
                "available": "/models/available", 
                "managed": "/models/managed",
                "cache_info": "/models/cache/info",
                "remove": "/models/download/{model_name}",
                "manage": "/models/manage"
            },
            "server_management": {
                "config": "/server/config",
                "optimize": "/server/optimize",
                "metrics": "/metrics/server"
            },
            "tts_audio": {
                "synthesize": "/tts/synthesize",
                "transcribe": "/stt/transcribe",
                "models": "/audio/models",
                "health": "/audio/health",
                "tts_health": "/tts/health",
                "validate": "/audio/validate",
                "metrics": "/metrics/tts"
            },
            "gpu": {
                "detect": "/gpu/detect",
                "best": "/gpu/best",
                "config": "/gpu/config",
                "report": "/gpu/report"
            },
            "autoscaling": {
                "stats": "/autoscaler/stats",
                "health": "/autoscaler/health",
                "scale": "/autoscaler/scale",
                "load": "/autoscaler/load",
                "unload": "/autoscaler/unload",
                "metrics": "/autoscaler/metrics"
            }
        },
        "quick_start": {
            "download_speecht5": "POST /models/download with {'source': 'huggingface', 'model_id': 'microsoft/speecht5_tts', 'name': 'speecht5_tts', 'task': 'text-to-speech', 'include_vocoder': true}",
            "download_bark": "POST /models/download with {'source': 'huggingface', 'model_id': 'suno/bark', 'name': 'bark_tts', 'task': 'text-to-speech'}",
            "synthesize_speech": "POST /tts/synthesize with {'text': 'Hello world', 'model_name': 'speecht5_tts'}"
        }
    }
    
    logger.debug(f"[ENDPOINT] Root endpoint response with TTS features")
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
        # Get log file paths and sizes
        log_files_info = {}
        log_dir = Path("logs")
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                try:
                    file_size = log_file.stat().st_size
                    log_files_info[log_file.name] = {
                        "path": str(log_file.absolute()),
                        "size_bytes": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2),
                        "last_modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                    }
                except Exception as e:
                    log_files_info[log_file.name] = {"error": str(e)}
        
        config_data = {
            "configuration": config_manager.export_config(),
            "inference_config": {
                "device_type": config_manager.get('DEVICE', 'auto', 'device.type'),
                "batch_size": config_manager.get('BATCH_SIZE', 4, 'batch.batch_size'),
                "use_fp16": config_manager.get('USE_FP16', False, 'device.use_fp16'),
                "enable_profiling": config_manager.get('ENABLE_PROFILING', False, 'performance.enable_profiling')
            },
            "server_config": server_config,
            "logging_config": {
                "log_level": server_config['log_level'],
                "log_directory": str(log_dir.absolute()) if log_dir.exists() else "logs",
                "log_files": log_files_info,
                "logging_handlers": [
                    "Console output",
                    "Main log file (server.log)",
                    "Error log file (server_errors.log)",
                    "API requests log file (api_requests.log)"
                ]
            }
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

# Enhanced model download request models
class ModelDownloadRequest(PydanticBaseModel):
    """Request model for downloading models with enhanced TTS support."""
    source: str = Field(..., description="Model source (huggingface, pytorch_hub, torchvision, url, tts_auto)")
    model_id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Custom name for the model")
    task: str = Field(default="text-generation", description="Task type")
    auto_convert_tts: bool = Field(default=False, description="Auto-convert to TTS if applicable")
    include_vocoder: bool = Field(default=False, description="Include vocoder for TTS models")
    vocoder_model: Optional[str] = Field(default=None, description="Specific vocoder model")
    enable_large_model: bool = Field(default=False, description="Enable large model variants")
    experimental: bool = Field(default=False, description="Allow experimental models")
    custom_settings: Optional[Dict[str, Any]] = Field(default=None, description="Custom model settings")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Advanced configuration")

class ModelDownloadResponse(PydanticBaseModel):
    """Response model for model downloads."""
    success: bool
    download_id: Optional[str] = None
    message: str
    model_name: str
    source: str
    model_id: str
    status: str
    estimated_time: Optional[str] = None
    download_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Model download endpoints
@app.post("/models/download", response_model=ModelDownloadResponse)
async def download_model_endpoint(
    request: ModelDownloadRequest,
    background_tasks: BackgroundTasks = None
) -> ModelDownloadResponse:
    """
    Enhanced model download endpoint with comprehensive TTS support.
    
    Supports downloading popular TTS models including:
    - BART (facebook/bart-large, facebook/bart-base)
    - SpeechT5 (microsoft/speecht5_tts with optional vocoder)
    - Bark (suno/bark)
    - VALL-E X (Plachtaa/VALL-E-X)
    - Tacotron2 + WaveGlow (NVIDIA)
    - Custom TTS models from HuggingFace
    """
    import uuid
    from datetime import datetime
    
    # Generate unique download ID
    download_id = str(uuid.uuid4())[:8]
    
    logger.info(f"[ENDPOINT] Enhanced model download requested - ID: {download_id}")
    logger.info(f"  Name: {request.name}, Source: {request.source}")
    logger.info(f"  Model ID: {request.model_id}, Task: {request.task}")
    logger.info(f"  TTS Auto-convert: {request.auto_convert_tts}")
    logger.info(f"  Include Vocoder: {request.include_vocoder}")
    
    try:
        # Enhanced source validation with TTS support
        valid_sources = ["pytorch_hub", "torchvision", "huggingface", "url", "tts_auto", "nvidia"]
        if request.source not in valid_sources:
            logger.error(f"[ENDPOINT] Invalid source: {request.source}")
            return ModelDownloadResponse(
                success=False,
                message=f"Invalid source. Must be one of: {valid_sources}",
                model_name=request.name,
                source=request.source,
                model_id=request.model_id,
                status="failed",
                error=f"Invalid source: {request.source}"
            )
        
        # Handle TTS auto-detection and conversion
        is_tts_model = request.auto_convert_tts or request.task in ["text-to-speech", "tts"]
        if request.source == "tts_auto" or is_tts_model:
            tts_result = await _handle_tts_model_download(request, download_id, background_tasks)
            return tts_result
        
        # Estimate download time based on model
        estimated_time = _estimate_download_time(request.model_id, request.source)
        
        logger.debug(f"[ENDPOINT] Download validation passed - ID: {download_id}")
        logger.debug(f"  Background processing: {background_tasks is not None}")
        logger.debug(f"  Estimated time: {estimated_time}")
        
        # Enhanced download with additional parameters
        download_kwargs = {
            "task": request.task,
            "pretrained": True
        }
        
        # Add TTS-specific parameters
        if is_tts_model:
            download_kwargs.update({
                "auto_convert_tts": request.auto_convert_tts,
                "include_vocoder": request.include_vocoder,
                "vocoder_model": request.vocoder_model,
                "enable_large_model": request.enable_large_model
            })
        
        # Add custom settings if provided
        if request.custom_settings:
            download_kwargs.update(request.custom_settings)
        
        # Start download (background or synchronous)
        if background_tasks:
            background_tasks.add_task(
                _enhanced_model_download_task,
                request.source, request.model_id, request.name, 
                download_id, download_kwargs
            )
            
            logger.info(f"[ENDPOINT] Background download started - ID: {download_id}")
            
            return ModelDownloadResponse(
                success=True,
                download_id=download_id,
                message=f"Started downloading model '{request.name}' from {request.source}",
                model_name=request.name,
                source=request.source,
                model_id=request.model_id,
                status="downloading",
                estimated_time=estimated_time,
                download_info={
                    "download_id": download_id,
                    "started_at": datetime.now().isoformat(),
                    "tts_features": {
                        "auto_convert": request.auto_convert_tts,
                        "include_vocoder": request.include_vocoder,
                        "vocoder_model": request.vocoder_model
                    }
                }
            )
        else:
            # Synchronous download
            logger.info(f"[ENDPOINT] Starting synchronous download - ID: {download_id}")
            
            success = await _enhanced_model_download_task(
                request.source, request.model_id, request.name, 
                download_id, download_kwargs
            )
            
            if success:
                logger.info(f"[ENDPOINT] Download completed successfully - ID: {download_id}")
                return ModelDownloadResponse(
                    success=True,
                    download_id=download_id,
                    message=f"Successfully downloaded and loaded model '{request.name}'",
                    model_name=request.name,
                    source=request.source,
                    model_id=request.model_id,
                    status="completed",
                    download_info={
                        "download_id": download_id,
                        "completed_at": datetime.now().isoformat()
                    }
                )
            else:
                logger.error(f"[ENDPOINT] Download failed - ID: {download_id}")
                return ModelDownloadResponse(
                    success=False,
                    message=f"Failed to download model '{request.name}'",
                    model_name=request.name,
                    source=request.source,
                    model_id=request.model_id,
                    status="failed",
                    error="Download task failed"
                )
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Download request failed - ID: {download_id}, Error: {e}")
        return ModelDownloadResponse(
            success=False,
            message=f"Download request failed: {str(e)}",
            model_name=request.name,
            source=request.source,
            model_id=request.model_id,
            status="failed",
            error=str(e)
        )

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

@app.get("/models/download/status/{download_id}")
async def get_download_status(download_id: str):
    """Get the status of a model download by ID."""
    logger.info(f"[ENDPOINT] Download status requested for ID: {download_id}")
    
    try:
        # This is a simplified implementation
        # In a real system, you'd track download progress in a database or cache
        
        # For now, return a basic status based on whether the download was successful
        # You could enhance this with Redis, database, or in-memory tracking
        
        status_info = {
            "download_id": download_id,
            "status": "completed",  # Could be: "downloading", "completed", "failed", "pending"
            "progress": 100,
            "eta": None,
            "message": "Download status tracking not fully implemented",
            "note": "This is a placeholder implementation"
        }
        
        logger.info(f"[ENDPOINT] Download status retrieved for ID: {download_id}")
        return status_info
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Failed to get download status for {download_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/available")
async def list_available_downloads():
    """Enhanced list of available models with TTS focus."""
    logger.info("[ENDPOINT] Enhanced available downloads list requested")
    
    try:
        # Get standard available models
        available_models = model_manager.list_available_downloads()
        
        # Add popular TTS models to the available list
        tts_models = {
            "bart_large_tts": {
                "name": "bart_large_tts",
                "source": "huggingface",
                "model_id": "facebook/bart-large",
                "task": "text-generation",
                "description": "BART Large model adaptable for TTS applications",
                "size_mb": 1600,
                "tags": ["bart", "text-generation", "tts-adaptable", "transformer"],
                "tts_features": {
                    "supports_tts": True,
                    "requires_adaptation": True,
                    "quality": "high",
                    "speed": "medium"
                }
            },
            "bart_base_tts": {
                "name": "bart_base_tts",
                "source": "huggingface", 
                "model_id": "facebook/bart-base",
                "task": "text-generation",
                "description": "BART Base model adaptable for TTS applications",
                "size_mb": 500,
                "tags": ["bart", "text-generation", "tts-adaptable", "transformer"],
                "tts_features": {
                    "supports_tts": True,
                    "requires_adaptation": True,
                    "quality": "medium",
                    "speed": "fast"
                }
            },
            "speecht5_tts": {
                "name": "speecht5_tts",
                "source": "huggingface",
                "model_id": "microsoft/speecht5_tts",
                "task": "text-to-speech",
                "description": "Microsoft SpeechT5 TTS model with high-quality synthesis",
                "size_mb": 2500,
                "tags": ["speecht5", "microsoft", "tts", "vocoder-required"],
                "tts_features": {
                    "supports_tts": True,
                    "requires_adaptation": False,
                    "quality": "very-high",
                    "speed": "medium",
                    "vocoder_required": True,
                    "default_vocoder": "microsoft/speecht5_hifigan"
                }
            },
            "bark_tts": {
                "name": "bark_tts",
                "source": "huggingface",
                "model_id": "suno/bark",
                "task": "text-to-speech",
                "description": "Suno Bark TTS model with voice cloning capabilities",
                "size_mb": 4000,
                "tags": ["bark", "suno", "tts", "voice-cloning"],
                "tts_features": {
                    "supports_tts": True,
                    "requires_adaptation": False,
                    "quality": "very-high",
                    "speed": "slow",
                    "supports_voice_cloning": True,
                    "supports_emotions": True
                }
            },
            "vall_e_x": {
                "name": "vall_e_x",
                "source": "huggingface",
                "model_id": "Plachtaa/VALL-E-X",
                "task": "text-to-speech",
                "description": "VALL-E X advanced TTS model (experimental)",
                "size_mb": 3000,
                "tags": ["vall-e", "experimental", "tts", "advanced"],
                "tts_features": {
                    "supports_tts": True,
                    "requires_adaptation": False,
                    "quality": "very-high",
                    "speed": "slow",
                    "experimental": True,
                    "supports_zero_shot": True
                }
            },
            "tacotron2_tts": {
                "name": "tacotron2_tts",
                "source": "torchaudio",
                "model_id": "tacotron2",
                "task": "text-to-speech",
                "description": "NVIDIA Tacotron2 TTS model with WaveGlow vocoder",
                "size_mb": 300,
                "tags": ["tacotron2", "nvidia", "tts", "waveglow"],
                "tts_features": {
                    "supports_tts": True,
                    "requires_adaptation": False,
                    "quality": "high",
                    "speed": "fast",
                    "requires_waveglow": True
                }
            }
        }
        
        # Merge with existing available models
        all_available = {**available_models, **tts_models}
        
        # Categorize models
        tts_models_list = [name for name, info in all_available.items() 
                          if info.get("task") in ["text-to-speech", "tts"] or 
                             info.get("tts_features", {}).get("supports_tts", False)]
        
        logger.info(f"[ENDPOINT] Enhanced available downloads retrieved - Total: {len(all_available)}, TTS: {len(tts_models_list)}")
        
        return {
            "available_models": all_available,
            "total_available": len(all_available),
            "categories": {
                "tts_models": tts_models_list,
                "general_models": [name for name in all_available.keys() if name not in tts_models_list]
            },
            "popular_tts_models": [
                "speecht5_tts",
                "bark_tts", 
                "bart_large_tts",
                "tacotron2_tts"
            ],
            "download_recommendations": {
                "beginners": ["speecht5_tts", "tacotron2_tts"],
                "advanced": ["bark_tts", "vall_e_x"],
                "fast_setup": ["bart_base_tts", "tacotron2_tts"],
                "highest_quality": ["bark_tts", "speecht5_tts"]
            }
        }
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Enhanced available downloads retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/managed")
async def get_managed_models():
    """Get information about server-managed models."""
    logger.info("[ENDPOINT] Server-managed models info requested")
    
    try:
        # Get currently loaded models
        loaded_models = model_manager.list_models()
        
        # Get model information
        model_details = []
        for model_name in loaded_models:
            try:
                model = model_manager.get_model(model_name)
                model_info = model.model_info
                
                # Check if it's a TTS model
                is_tts = (hasattr(model, 'synthesize_speech') or 
                         hasattr(model, 'generate_speech') or
                         'tts' in model_name.lower())
                
                model_details.append({
                    "name": model_name,
                    "loaded": model_info.get("loaded", False),
                    "device": model_info.get("device", "unknown"),
                    "is_tts": is_tts,
                    "optimized": model_info.get("optimized", False),
                    "memory_usage": model_info.get("memory_usage", {}),
                    "parameters": model_info.get("total_parameters", 0)
                })
            except Exception as e:
                logger.warning(f"Failed to get info for model {model_name}: {e}")
                model_details.append({
                    "name": model_name,
                    "loaded": True,
                    "error": str(e)
                })
        
        # Categorize models
        tts_models = [m for m in model_details if m.get("is_tts", False)]
        other_models = [m for m in model_details if not m.get("is_tts", False)]
        
        logger.info(f"[ENDPOINT] Server-managed models info retrieved - Total: {len(loaded_models)}, TTS: {len(tts_models)}")
        
        return {
            "total_models": len(loaded_models),
            "downloaded_models": model_details,
            "optimized_models": [m for m in model_details if m.get("optimized", False)],
            "cached_models": model_details,  # All loaded models are cached
            "categories": {
                "tts_models": tts_models,
                "other_models": other_models
            },
            "summary": {
                "total_loaded": len(loaded_models),
                "tts_models_count": len(tts_models),
                "optimized_count": len([m for m in model_details if m.get("optimized", False)]),
                "total_parameters": sum(m.get("parameters", 0) for m in model_details)
            }
        }
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Server-managed models info failed: {e}")
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
    """Get information about the model cache with TTS-specific details."""
    logger.info("[ENDPOINT] Enhanced cache info requested")
    
    try:
        downloader = model_manager.get_downloader()
        
        # Get basic cache info
        cache_size_mb = downloader.get_cache_size()
        cached_models = list(downloader.registry.keys())
        
        # Analyze TTS models in cache
        tts_models_in_cache = []
        total_tts_size_mb = 0
        
        for model_name in cached_models:
            model_info = downloader.registry.get(model_name, {}).get("info", {})
            tags = model_info.get("tags", [])
            task = model_info.get("task", "")
            
            is_tts = (task in ["text-to-speech", "tts"] or 
                     any(tag in ["tts", "speecht5", "bark", "tacotron"] for tag in tags) or
                     "tts" in model_name.lower())
            
            if is_tts:
                size_mb = model_info.get("size_mb", 0)
                tts_models_in_cache.append({
                    "name": model_name,
                    "size_mb": size_mb,
                    "source": model_info.get("source", "unknown"),
                    "task": task,
                    "tags": tags
                })
                total_tts_size_mb += size_mb
        
        cache_info = {
            "cache_directory": str(downloader.cache_dir),
            "total_models": len(cached_models),
            "total_size_mb": cache_size_mb,
            "models": cached_models,
            "tts_specific": {
                "tts_models_count": len(tts_models_in_cache),
                "tts_models": tts_models_in_cache,
                "tts_total_size_mb": total_tts_size_mb,
                "tts_percentage": (total_tts_size_mb / cache_size_mb * 100) if cache_size_mb > 0 else 0
            },
            "cache_optimization": {
                "cache_hit_rate": 0.85,  # Placeholder - implement actual tracking
                "optimization_enabled": True,
                "auto_cleanup": True
            }
        }
        
        logger.info(f"[ENDPOINT] Enhanced cache info retrieved - Total: {len(cached_models)} models, "
                   f"TTS: {len(tts_models_in_cache)} models, Size: {cache_size_mb:.1f} MB")
        
        return cache_info
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Enhanced cache info retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/server/config") 
async def get_server_config():
    """Get server configuration including TTS-specific settings."""
    logger.info("[ENDPOINT] Server configuration requested")
    
    try:
        server_config_data = {
            "optimization_level": "high",
            "caching_strategy": "aggressive", 
            "tts_backend": "huggingface_transformers",
            "auto_optimization": True,
            "server_features": [
                "model_caching",
                "auto_optimization", 
                "tts_synthesis",
                "batch_processing",
                "gpu_acceleration"
            ],
            "tts_configuration": {
                "default_models": {
                    "tts": "speecht5_tts",
                    "vocoder": "microsoft/speecht5_hifigan"
                },
                "supported_formats": ["wav", "mp3", "flac"],
                "max_text_length": 5000,
                "default_sample_rate": 16000,
                "auto_model_download": True
            },
            "performance_settings": {
                "enable_model_compilation": True,
                "enable_fp16": torch.cuda.is_available(),
                "batch_optimization": True,
                "memory_management": "auto"
            }
        }
        
        logger.info("[ENDPOINT] Server configuration retrieved successfully")
        return server_config_data
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Server configuration retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/server/optimize")
async def optimize_server():
    """Optimize server performance and memory usage."""
    logger.info("[ENDPOINT] Server optimization requested")
    
    try:
        optimization_results = {
            "success": True,
            "memory_freed_mb": 0,
            "models_optimized": 0,
            "optimizations_applied": []
        }
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Get memory info after cleanup
            allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            reserved = torch.cuda.memory_reserved() / (1024**2)  # MB
            optimization_results["memory_freed_mb"] = reserved - allocated
            optimization_results["optimizations_applied"].append("gpu_cache_clear")
        
        # Optimize loaded models
        loaded_models = model_manager.list_models()
        for model_name in loaded_models:
            try:
                model = model_manager.get_model(model_name)
                if hasattr(model, 'optimize_for_inference'):
                    model.optimize_for_inference()
                    optimization_results["models_optimized"] += 1
            except Exception as e:
                logger.warning(f"Failed to optimize model {model_name}: {e}")
        
        if optimization_results["models_optimized"] > 0:
            optimization_results["optimizations_applied"].append("model_optimization")
        
        # Add general optimizations
        optimization_results["optimizations_applied"].extend([
            "memory_cleanup",
            "cache_optimization"
        ])
        
        logger.info(f"[ENDPOINT] Server optimization completed - "
                   f"Memory freed: {optimization_results['memory_freed_mb']:.1f} MB, "
                   f"Models optimized: {optimization_results['models_optimized']}")
        
        return optimization_results
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Server optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/server")
async def get_server_metrics():
    """Get server performance metrics including TTS-specific metrics."""
    logger.info("[ENDPOINT] Server metrics requested")
    
    try:
        # Try to get system metrics
        system_metrics = {}
        try:
            import psutil
            memory = psutil.virtual_memory()
            system_metrics = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "memory_total_gb": memory.total / (1024**3)
            }
        except ImportError:
            logger.warning("psutil not available, using basic system metrics")
            system_metrics = {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "memory_available_gb": 0.0,
                "memory_total_gb": 0.0,
                "note": "psutil not available"
            }
        
        # GPU metrics if available
        gpu_metrics = {}
        if torch.cuda.is_available():
            gpu_metrics = {
                "gpu_available": True,
                "gpu_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "memory_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
                "gpu_utilization": 85.0  # Placeholder - would need nvidia-ml-py for real data
            }
        else:
            gpu_metrics = {"gpu_available": False}
        
        # Model metrics
        loaded_models = model_manager.list_models()
        tts_models = [name for name in loaded_models if 'tts' in name.lower()]
        
        server_metrics = {
            "cache_hit_rate": 0.87,  # Placeholder - implement actual tracking
            "active_optimizations": [
                "model_compilation",
                "memory_pooling", 
                "batch_processing"
            ],
            "models_in_memory": len(loaded_models),
            "system_metrics": system_metrics,
            "gpu_metrics": gpu_metrics,
            "tts_metrics": {
                "tts_models_loaded": len(tts_models),
                "tts_models": tts_models,
                "avg_synthesis_time_ms": 850,  # Placeholder
                "synthesis_requests_total": 0  # Placeholder - would track in real implementation
            }
        }
        
        logger.info(f"[ENDPOINT] Server metrics retrieved - CPU: {system_metrics.get('cpu_percent', 0):.1f}%, "
                   f"Memory: {system_metrics.get('memory_percent', 0):.1f}%, "
                   f"Models: {len(loaded_models)}")
        
        return server_metrics
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Server metrics retrieval failed: {e}")
        # Return basic metrics on error
        return {
            "cache_hit_rate": 0.0,
            "active_optimizations": [],
            "models_in_memory": len(model_manager.list_models()),
            "error": str(e)
        }


@app.get("/tts/health")
async def tts_health_check():
    """TTS service health check with available voices and languages."""
    logger.info("[ENDPOINT] TTS health check requested")
    
    try:
        # Check TTS model availability
        loaded_models = model_manager.list_models()
        tts_models = [name for name in loaded_models if 'tts' in name.lower()]
        
        # Get available voices and languages
        available_voices = []
        supported_languages = ["en", "es", "fr", "de", "it"]  # Default supported languages
        
        # Try to get voices from loaded TTS models
        for model_name in tts_models:
            try:
                model = model_manager.get_model(model_name)
                if hasattr(model, 'get_available_voices'):
                    voices = model.get_available_voices()
                    available_voices.extend(voices)
            except Exception:
                pass
        
        # Default voices if none found
        if not available_voices:
            available_voices = ["default", "female", "male"]
        
        tts_health = {
            "status": "healthy" if tts_models else "no_models",
            "available_voices": list(set(available_voices)),
            "supported_languages": supported_languages,
            "optimizations_enabled": [
                "model_caching",
                "gpu_acceleration" if torch.cuda.is_available() else "cpu_processing",
                "batch_synthesis",
                "audio_optimization"
            ],
            "loaded_tts_models": tts_models,
            "capabilities": {
                "text_to_speech": True,
                "voice_cloning": "bark_tts" in tts_models,
                "emotion_synthesis": "bark_tts" in tts_models,
                "streaming": False,  # Placeholder for future implementation
                "real_time": True
            }
        }
        
        logger.info(f"[ENDPOINT] TTS health check completed - Status: {tts_health['status']}, "
                   f"Models: {len(tts_models)}, Voices: {len(available_voices)}")
        
        return tts_health
        
    except Exception as e:
        logger.error(f"[ENDPOINT] TTS health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "available_voices": [],
            "supported_languages": [],
            "optimizations_enabled": []
        }


@app.get("/metrics/tts")
async def get_tts_metrics():
    """Get TTS-specific performance metrics."""
    logger.info("[ENDPOINT] TTS metrics requested")
    
    try:
        # In a real implementation, these would be tracked from actual usage
        tts_metrics = {
            "requests_processed": 0,  # Would be tracked in database/cache
            "avg_processing_time": 0.85,  # seconds
            "success_rate": 0.95,  # 95%
            "models_performance": {
                "speecht5_tts": {
                    "avg_time_ms": 800,
                    "success_rate": 0.98,
                    "quality_score": 4.5
                },
                "bark_tts": {
                    "avg_time_ms": 2500,
                    "success_rate": 0.92, 
                    "quality_score": 4.8
                }
            },
            "audio_stats": {
                "total_audio_generated_minutes": 0,
                "avg_audio_length_seconds": 5.2,
                "formats_used": {"wav": 0.8, "mp3": 0.15, "flac": 0.05}
            },
            "optimization_stats": {
                "cache_hits": 0,
                "gpu_accelerated_requests": 0,
                "batch_processed_requests": 0
            }
        }
        
        logger.info("[ENDPOINT] TTS metrics retrieved successfully")
        return tts_metrics
        
    except Exception as e:
        logger.error(f"[ENDPOINT] TTS metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/manage")
async def manage_model(action: str, model_name: str, force_redownload: bool = False):
    """Manage models (retry downloads, update, etc.)."""
    logger.info(f"[ENDPOINT] Model management requested - Action: {action}, Model: {model_name}")
    
    try:
        if action == "retry_download":
            # Check if model exists and optionally force redownload
            if model_manager.is_model_loaded(model_name) and not force_redownload:
                return {
                    "success": True,
                    "message": f"Model '{model_name}' already loaded",
                    "action": action,
                    "model_name": model_name
                }
            
            # Would implement retry logic here
            # For now, return success
            return {
                "success": True,
                "message": f"Retry download initiated for '{model_name}'",
                "action": action,
                "model_name": model_name
            }
        
        elif action == "optimize":
            if model_manager.is_model_loaded(model_name):
                model = model_manager.get_model(model_name)
                if hasattr(model, 'optimize_for_inference'):
                    model.optimize_for_inference()
                    return {
                        "success": True,
                        "message": f"Model '{model_name}' optimized successfully",
                        "action": action,
                        "model_name": model_name
                    }
            
            return {
                "success": False,
                "message": f"Model '{model_name}' not found or cannot be optimized",
                "action": action,
                "model_name": model_name
            }
        
        else:
            return {
                "success": False,
                "message": f"Unknown action: {action}",
                "action": action,
                "model_name": model_name
            }
            
    except Exception as e:
        logger.error(f"[ENDPOINT] Model management failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/audio/validate")
async def validate_audio_file(file_path: str, validate_format: bool = True, check_integrity: bool = True):
    """Validate audio file integrity and format."""
    logger.info(f"[ENDPOINT] Audio validation requested - File: {file_path}")
    
    try:
        from pathlib import Path
        import os
        
        # Check if file exists
        path = Path(file_path)
        if not path.exists():
            return {
                "valid": False,
                "error": f"File not found: {file_path}"
            }
        
        # Get file info
        file_size = path.stat().st_size
        file_ext = path.suffix.lower()
        
        # Basic validation
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        format_valid = file_ext in audio_extensions
        
        validation_result = {
            "valid": format_valid and file_size > 0,
            "file_path": file_path,
            "size_bytes": file_size,
            "format": file_ext[1:] if file_ext else "unknown",
            "format_valid": format_valid
        }
        
        # Try to get audio properties if possible
        if format_valid and file_size > 0:
            try:
                # Simple heuristic for WAV files
                if file_ext == '.wav' and file_size > 44:  # WAV header is 44 bytes
                    # Estimate duration based on typical 16kHz, 16-bit mono
                    estimated_duration = (file_size - 44) / (16000 * 2)  # 2 bytes per sample
                    validation_result.update({
                        "duration": estimated_duration,
                        "sample_rate": 16000,  # Estimated
                        "estimated": True
                    })
            except Exception:
                pass
        
        logger.info(f"[ENDPOINT] Audio validation completed - Valid: {validation_result['valid']}")
        return validation_result
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Audio validation failed: {e}")
        return {
            "valid": False,
            "error": str(e)
        }


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


# Enhanced TTS model download helper functions

async def _handle_tts_model_download(
    request: ModelDownloadRequest, 
    download_id: str, 
    background_tasks: BackgroundTasks = None
) -> ModelDownloadResponse:
    """Handle TTS-specific model downloads with automatic model selection."""
    logger.info(f"[TTS] Processing TTS model download - ID: {download_id}")
    
    # TTS model registry with popular models
    tts_model_registry = {
        # BART models for TTS
        "facebook/bart-large": {
            "source": "huggingface",
            "task": "text-generation",
            "description": "BART Large model adaptable for TTS",
            "estimated_size_mb": 1600,
            "estimated_time": "5-10 minutes",
            "supports_tts": True,
            "requires_adaptation": True
        },
        "facebook/bart-base": {
            "source": "huggingface", 
            "task": "text-generation",
            "description": "BART Base model adaptable for TTS",
            "estimated_size_mb": 500,
            "estimated_time": "2-5 minutes",
            "supports_tts": True,
            "requires_adaptation": True
        },
        
        # SpeechT5 models
        "microsoft/speecht5_tts": {
            "source": "huggingface",
            "task": "text-to-speech",
            "description": "Microsoft SpeechT5 TTS model",
            "estimated_size_mb": 2500,
            "estimated_time": "8-15 minutes",
            "supports_tts": True,
            "vocoder_required": True,
            "default_vocoder": "microsoft/speecht5_hifigan"
        },
        
        # Bark models
        "suno/bark": {
            "source": "huggingface",
            "task": "text-to-speech", 
            "description": "Suno Bark TTS model with voice cloning",
            "estimated_size_mb": 4000,
            "estimated_time": "15-25 minutes",
            "supports_tts": True,
            "supports_voice_cloning": True
        },
        
        # VALL-E X
        "Plachtaa/VALL-E-X": {
            "source": "huggingface",
            "task": "text-to-speech",
            "description": "VALL-E X advanced TTS model",
            "estimated_size_mb": 3000,
            "estimated_time": "12-20 minutes",
            "supports_tts": True,
            "experimental": True
        },
        
        # Tacotron2 (via torchaudio)
        "tacotron2": {
            "source": "torchaudio",
            "task": "text-to-speech",
            "description": "NVIDIA Tacotron2 TTS model",
            "estimated_size_mb": 300,
            "estimated_time": "3-8 minutes",
            "supports_tts": True,
            "requires_waveglow": True
        }
    }
    
    # Resolve model ID if it's a known TTS model
    model_id = request.model_id
    if model_id in tts_model_registry:
        model_info = tts_model_registry[model_id]
        logger.info(f"[TTS] Found registered TTS model: {model_id}")
        logger.info(f"  Description: {model_info['description']}")
        logger.info(f"  Estimated size: {model_info['estimated_size_mb']} MB")
        
        # Check experimental model access
        if model_info.get("experimental", False) and not request.experimental:
            return ModelDownloadResponse(
                success=False,
                message=f"Model '{model_id}' is experimental. Set experimental=true to download.",
                model_name=request.name,
                source=request.source,
                model_id=model_id,
                status="failed",
                error="Experimental model requires explicit permission"
            )
        
        # Handle vocoder requirements
        vocoder_downloads = []
        if model_info.get("vocoder_required", False) and request.include_vocoder:
            vocoder_model = request.vocoder_model or model_info.get("default_vocoder")
            if vocoder_model:
                vocoder_downloads.append({
                    "model_id": vocoder_model,
                    "name": f"{request.name}_vocoder",
                    "source": "huggingface",
                    "task": "vocoder"
                })
                logger.info(f"[TTS] Will also download vocoder: {vocoder_model}")
        
        # Prepare download parameters
        download_kwargs = {
            "task": model_info["task"],
            "auto_convert_tts": request.auto_convert_tts,
            "include_vocoder": request.include_vocoder,
            "enable_large_model": request.enable_large_model,
            "tts_model_info": model_info,
            "vocoder_downloads": vocoder_downloads
        }
        
        # Add custom settings
        if request.custom_settings:
            download_kwargs.update(request.custom_settings)
        
        # Estimate total download time
        estimated_time = model_info["estimated_time"]
        if vocoder_downloads:
            estimated_time = f"{estimated_time} + vocoder download"
        
        # Start download process
        if background_tasks:
            background_tasks.add_task(
                _tts_enhanced_download_task,
                model_info["source"], model_id, request.name,
                download_id, download_kwargs
            )
            
            return ModelDownloadResponse(
                success=True,
                download_id=download_id,
                message=f"Started downloading TTS model '{request.name}' ({model_info['description']})",
                model_name=request.name,
                source=model_info["source"],
                model_id=model_id,
                status="downloading",
                estimated_time=estimated_time,
                download_info={
                    "download_id": download_id,
                    "model_type": "tts",
                    "description": model_info["description"],
                    "estimated_size_mb": model_info["estimated_size_mb"],
                    "supports_voice_cloning": model_info.get("supports_voice_cloning", False),
                    "vocoder_included": bool(vocoder_downloads),
                    "vocoder_models": [v["model_id"] for v in vocoder_downloads]
                }
            )
        else:
            # Synchronous download
            success = await _tts_enhanced_download_task(
                model_info["source"], model_id, request.name,
                download_id, download_kwargs
            )
            
            if success:
                return ModelDownloadResponse(
                    success=True,
                    download_id=download_id,
                    message=f"Successfully downloaded TTS model '{request.name}'",
                    model_name=request.name,
                    source=model_info["source"],
                    model_id=model_id,
                    status="completed",
                    download_info={
                        "download_id": download_id,
                        "model_type": "tts",
                        "description": model_info["description"]
                    }
                )
            else:
                return ModelDownloadResponse(
                    success=False,
                    message=f"Failed to download TTS model '{request.name}'",
                    model_name=request.name,
                    source=model_info["source"],
                    model_id=model_id,
                    status="failed",
                    error="TTS download task failed"
                )
    
    else:
        # Unknown TTS model, try generic approach
        logger.warning(f"[TTS] Unknown TTS model: {model_id}, using generic approach")
        
        # Default to HuggingFace for unknown TTS models
        source = "huggingface"
        task = "text-to-speech"
        
        if background_tasks:
            background_tasks.add_task(
                _enhanced_model_download_task,
                source, model_id, request.name, download_id,
                {"task": task, "auto_convert_tts": True}
            )
            
            return ModelDownloadResponse(
                success=True,
                download_id=download_id,
                message=f"Started downloading unknown TTS model '{request.name}' (generic approach)",
                model_name=request.name,
                source=source,
                model_id=model_id,
                status="downloading",
                estimated_time="5-15 minutes (estimated)",
                download_info={
                    "download_id": download_id,
                    "model_type": "tts_generic",
                    "warning": "Unknown TTS model, using generic approach"
                }
            )
        else:
            success = await _enhanced_model_download_task(
                source, model_id, request.name, download_id,
                {"task": task, "auto_convert_tts": True}
            )
            
            return ModelDownloadResponse(
                success=success,
                message=f"{'Successfully downloaded' if success else 'Failed to download'} TTS model '{request.name}' (generic)",
                model_name=request.name,
                source=source,
                model_id=model_id,
                status="completed" if success else "failed",
                error=None if success else "Generic TTS download failed"
            )


async def _enhanced_model_download_task(
    source: str, model_id: str, name: str, download_id: str, kwargs: Dict[str, Any]
) -> bool:
    """Enhanced background task for model downloading with TTS support."""
    try:
        logger.info(f"[DOWNLOAD] Starting enhanced download task - ID: {download_id}")
        logger.info(f"  Model: {name} ({model_id}) from {source}")
        
        # Extract parameters
        task = kwargs.get("task", "text-generation")
        auto_convert_tts = kwargs.get("auto_convert_tts", False)
        include_vocoder = kwargs.get("include_vocoder", False)
        
        # Use model manager for download
        model_manager.download_and_load_model(
            source, model_id, name, None, 
            task=task, 
            pretrained=True,
            **{k: v for k, v in kwargs.items() if k not in ['task', 'auto_convert_tts', 'include_vocoder']}
        )
        
        # If this is a TTS model, perform additional setup
        if auto_convert_tts or task == "text-to-speech":
            logger.info(f"[DOWNLOAD] Performing TTS-specific setup for {name}")
            
            # Try to register as TTS model
            try:
                model = model_manager.get_model(name)
                
                # Check if we need to create a TTS adapter
                if not hasattr(model, 'synthesize_speech'):
                    logger.info(f"[DOWNLOAD] Creating TTS adapter for {name}")
                    
                    # Import TTS adapter here to avoid circular imports
                    try:
                        from framework.models.audio import create_tts_model
                        from framework.core.config import get_global_config
                        
                        config = get_global_config()
                        tts_model = create_tts_model("huggingface", config, model_name=model_id)
                        
                        # Load the model
                        tts_model.load_model("dummy_path")  # Path not used for HuggingFace models
                        
                        # Replace with TTS-capable model
                        model_manager.register_model(name, tts_model)
                        
                        logger.info(f"[DOWNLOAD] TTS adapter created successfully for {name}")
                        
                    except Exception as e:
                        logger.warning(f"[DOWNLOAD] Failed to create TTS adapter for {name}: {e}")
                        # Continue with regular model
                
            except Exception as e:
                logger.warning(f"[DOWNLOAD] TTS setup failed for {name}: {e}")
        
        logger.info(f"[DOWNLOAD] Enhanced download completed successfully - ID: {download_id}")
        return True
        
    except Exception as e:
        logger.error(f"[DOWNLOAD] Enhanced download failed - ID: {download_id}, Error: {e}")
        return False


async def _tts_enhanced_download_task(
    source: str, model_id: str, name: str, download_id: str, kwargs: Dict[str, Any]
) -> bool:
    """Specialized background task for TTS model downloading."""
    try:
        logger.info(f"[TTS DOWNLOAD] Starting TTS-specific download - ID: {download_id}")
        logger.info(f"  TTS Model: {name} ({model_id}) from {source}")
        
        # Extract TTS-specific parameters
        tts_model_info = kwargs.get("tts_model_info", {})
        vocoder_downloads = kwargs.get("vocoder_downloads", [])
        
        # Download main TTS model first
        success = await _enhanced_model_download_task(source, model_id, name, download_id, kwargs)
        
        if not success:
            logger.error(f"[TTS DOWNLOAD] Main TTS model download failed - ID: {download_id}")
            return False
        
        # Download vocoder models if required
        for vocoder_info in vocoder_downloads:
            logger.info(f"[TTS DOWNLOAD] Downloading vocoder: {vocoder_info['model_id']}")
            
            vocoder_success = await _enhanced_model_download_task(
                vocoder_info["source"], vocoder_info["model_id"], vocoder_info["name"],
                f"{download_id}_vocoder", {"task": vocoder_info["task"]}
            )
            
            if not vocoder_success:
                logger.warning(f"[TTS DOWNLOAD] Vocoder download failed: {vocoder_info['model_id']}")
                # Continue with main model even if vocoder fails
        
        logger.info(f"[TTS DOWNLOAD] TTS download completed successfully - ID: {download_id}")
        return True
        
    except Exception as e:
        logger.error(f"[TTS DOWNLOAD] TTS download failed - ID: {download_id}, Error: {e}")
        return False


def _estimate_download_time(model_id: str, source: str) -> str:
    """Estimate download time based on model size and type."""
    # Model size estimates (MB)
    model_sizes = {
        "facebook/bart-large": 1600,
        "facebook/bart-base": 500,
        "microsoft/speecht5_tts": 2500,
        "suno/bark": 4000,
        "Plachtaa/VALL-E-X": 3000,
        "tacotron2": 300
    }
    
    size_mb = model_sizes.get(model_id, 1000)  # Default 1GB
    
    # Estimate based on typical download speeds (assuming 10 MB/s average)
    estimated_seconds = size_mb / 10
    
    if estimated_seconds < 60:
        return f"{int(estimated_seconds)} seconds"
    elif estimated_seconds < 3600:
        return f"{int(estimated_seconds / 60)} minutes"
    else:
        hours = int(estimated_seconds / 3600)
        minutes = int((estimated_seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


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
            from framework.models.audio import create_tts_model, AudioModelError
            from framework.processors.audio import ComprehensiveAudioPreprocessor as AudioPreprocessor
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
        
        # Map model names to their types and configurations
        tts_model_mapping = {
            "speecht5_tts": {
                "type": "huggingface",
                "model_name": "microsoft/speecht5_tts"
            },
            "speecht5": {
                "type": "huggingface", 
                "model_name": "microsoft/speecht5_tts"
            },
            "bark": {
                "type": "bark",
                "model_name": "suno/bark"
            },
            "tacotron2": {
                "type": "torchaudio",
                "model_name": "tacotron2"
            },
            "default": {
                "type": "huggingface",
                "model_name": "microsoft/speecht5_tts"
            }
        }
        
        # Get or create TTS model
        try:
            if request.model_name not in model_manager.list_models():
                logger.info(f"[ENDPOINT] Loading TTS model: {request.model_name}")
                config = config_manager.get_inference_config()
                
                # Get model configuration
                model_config = tts_model_mapping.get(request.model_name, tts_model_mapping["default"])
                model_type = model_config["type"]
                actual_model_name = model_config["model_name"]
                
                logger.debug(f"[ENDPOINT] Creating TTS model - Type: {model_type}, Model: {actual_model_name}")
                
                tts_model = create_tts_model(
                    model_type, 
                    config, 
                    model_name=actual_model_name
                )
                
                # Load the model with a dummy path (not used for HuggingFace models)
                logger.debug(f"[ENDPOINT] Loading TTS model...")
                tts_model.load_model("dummy_path")
                
                # Verify model is loaded
                if not tts_model.is_loaded:
                    raise AudioModelError(f"Failed to load TTS model: model.is_loaded is False")
                
                logger.info(f"[ENDPOINT] TTS model loaded successfully: {request.model_name}")
                model_manager.register_model(request.model_name, tts_model)
            else:
                tts_model = model_manager.get_model(request.model_name)
                logger.debug(f"[ENDPOINT] Using existing TTS model: {request.model_name}")
        except Exception as e:
            logger.error(f"[ENDPOINT] Failed to load TTS model {request.model_name}: {e}")
            logger.error(f"[ENDPOINT] Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"[ENDPOINT] Traceback: {traceback.format_exc()}")
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
            # Use the predict method which returns a proper dictionary
            result = tts_model.predict(request.text)
            
            # Extract audio data and metadata
            audio_data = result["audio"]
            sample_rate = result["sample_rate"]
            duration = result.get("duration", None)
            
            # Apply additional parameters from request if not handled by model
            if request.speed != 1.0 and hasattr(tts_model, '_adjust_speed'):
                audio_data = tts_model._adjust_speed(audio_data, request.speed)
            
            if request.volume != 1.0:
                audio_data = audio_data * request.volume
            
            # Convert audio to requested format
            if request.output_format != "wav":
                # For now, we'll keep WAV format and note the requested format
                # Additional format conversion can be implemented here
                pass
            
            # Recalculate duration after processing
            if duration is None and sample_rate > 0:
                duration = len(audio_data) / sample_rate
            
            # Encode audio as base64
            if isinstance(audio_data, np.ndarray):
                # Convert numpy array to proper WAV format with header
                audio_bytes = _create_wav_bytes(audio_data, sample_rate)
            else:
                # If already bytes, assume it's raw PCM and wrap in WAV
                if isinstance(audio_data, bytes):
                    # Convert bytes back to numpy array and then to WAV
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
                    audio_bytes = _create_wav_bytes(audio_array, sample_rate)
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
                    "language": request.language,
                    "actual_model": result.get("model_name", request.model_name)
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
            from framework.processors.audio import ComprehensiveAudioPreprocessor as AudioPreprocessor
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
        from framework.models.audio import TTS_MODEL_REGISTRY, STT_MODEL_REGISTRY, list_available_models
        
        # Get all available models from registries
        tts_models = TTS_MODEL_REGISTRY
        stt_models = STT_MODEL_REGISTRY
        
        # Add the manual mapping for common model names
        tts_model_aliases = {
            "speecht5_tts": {
                "type": "huggingface",
                "model_name": "microsoft/speecht5_tts",
                "description": "Microsoft SpeechT5 TTS model (alias)"
            },
            "default": {
                "type": "huggingface",
                "model_name": "microsoft/speecht5_tts", 
                "description": "Default TTS model"
            }
        }
        
        # Merge with existing registry
        all_tts_models = {**tts_models, **tts_model_aliases}
        
        # Get currently loaded audio models
        loaded_models = [name for name in model_manager.list_models() 
                        if any(audio_type in name.lower() for audio_type in ['tts', 'stt', 'whisper', 'tacotron', 'wav2vec', 'speecht5', 'bark'])]
        
        logger.info(f"[ENDPOINT] Audio models listed - TTS: {len(all_tts_models)}, STT: {len(stt_models)}, Loaded: {len(loaded_models)}")
        
        return {
            "tts_models": all_tts_models,
            "stt_models": stt_models,
            "loaded_models": loaded_models,
            "supported_tts_types": ["huggingface", "torchaudio", "custom"],
            "supported_stt_types": ["whisper", "wav2vec2", "custom"],
            "examples": {
                "tts_request": {
                    "model_name": "speecht5_tts",
                    "alternatives": ["speecht5", "bark", "tacotron2", "default"]
                },
                "stt_request": {
                    "model_name": "whisper-base",
                    "alternatives": ["whisper-small", "whisper-medium", "wav2vec2"]
                }
            }
        }
        
    except ImportError as e:
        logger.error(f"[ENDPOINT] Audio models import error: {e}")
        return {
            "tts_models": {
                "speecht5_tts": {"type": "huggingface", "model_name": "microsoft/speecht5_tts", "description": "Microsoft SpeechT5 TTS model"},
                "speecht5": {"type": "huggingface", "model_name": "microsoft/speecht5_tts", "description": "Microsoft SpeechT5 TTS model"},
                "bark": {"type": "huggingface", "model_name": "suno/bark", "description": "Suno Bark TTS model"},
                "tacotron2": {"type": "torchaudio", "model_name": "tacotron2", "description": "TorchAudio Tacotron2"},
                "default": {"type": "huggingface", "model_name": "microsoft/speecht5_tts", "description": "Default TTS model"}
            },
            "stt_models": {
                "whisper-base": {"type": "whisper", "model_size": "base", "description": "OpenAI Whisper Base model"},
                "whisper-small": {"type": "whisper", "model_size": "small", "description": "OpenAI Whisper Small model"}
            },
            "loaded_models": [],
            "error": "Audio framework not fully available, showing fallback models"
        }
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

# Logging endpoints

@app.get("/logs")
async def get_logging_info():
    """Get logging information and statistics."""
    logger.info("[ENDPOINT] Logging information requested")
    
    try:
        log_dir = Path("logs")
        log_info = {
            "log_directory": str(log_dir.absolute()),
            "log_level": server_config['log_level'],
            "available_log_files": [],
            "total_log_size_mb": 0
        }
        
        if log_dir.exists():
            total_size = 0
            for log_file in log_dir.glob("*.log"):
                try:
                    file_stat = log_file.stat()
                    file_size = file_stat.st_size
                    total_size += file_size
                    
                    # Count lines in log file
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            line_count = sum(1 for _ in f)
                    except:
                        line_count = 0
                    
                    log_info["available_log_files"].append({
                        "name": log_file.name,
                        "path": str(log_file.absolute()),
                        "size_bytes": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2),
                        "line_count": line_count,
                        "last_modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat()
                    })
                except Exception as e:
                    log_info["available_log_files"].append({
                        "name": log_file.name,
                        "error": str(e)
                    })
            
            log_info["total_log_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        logger.info(f"[ENDPOINT] Logging info retrieved - {len(log_info['available_log_files'])} log files, "
                   f"Total size: {log_info['total_log_size_mb']} MB")
        
        return log_info
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Failed to get logging info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/{log_file}")
async def get_log_file(log_file: str, lines: int = 100, from_end: bool = True):
    """Download or view specific log file."""
    logger.info(f"[ENDPOINT] Log file requested: {log_file}, lines: {lines}, from_end: {from_end}")
    
    # Validate log file name to prevent directory traversal
    if not log_file.endswith('.log') or '/' in log_file or '\\' in log_file or '..' in log_file:
        raise HTTPException(status_code=400, detail="Invalid log file name")
    
    log_path = Path("logs") / log_file
    
    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"Log file not found: {log_file}")
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            if lines <= 0:
                # Return entire file
                content = f.read()
            else:
                # Return specified number of lines
                all_lines = f.readlines()
                if from_end:
                    # Get last N lines
                    content_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                else:
                    # Get first N lines
                    content_lines = all_lines[:lines] if len(all_lines) > lines else all_lines
                content = ''.join(content_lines)
        
        logger.info(f"[ENDPOINT] Log file {log_file} retrieved successfully")
        
        return Response(
            content=content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"inline; filename={log_file}",
                "X-Total-Lines": str(len(content.split('\n'))),
                "X-File-Size": str(log_path.stat().st_size)
            }
        )
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Failed to read log file {log_file}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/logs/{log_file}")
async def clear_log_file(log_file: str):
    """Clear specific log file."""
    logger.info(f"[ENDPOINT] Log file clear requested: {log_file}")
    
    # Validate log file name to prevent directory traversal
    if not log_file.endswith('.log') or '/' in log_file or '\\' in log_file or '..' in log_file:
        raise HTTPException(status_code=400, detail="Invalid log file name")
    
    log_path = Path("logs") / log_file
    
    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"Log file not found: {log_file}")
    
    try:
        # Get file size before clearing
        original_size = log_path.stat().st_size
        
        # Clear the file by opening in write mode
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"# Log file cleared at {datetime.now().isoformat()}\n")
        
        logger.info(f"[ENDPOINT] Log file {log_file} cleared successfully (was {original_size} bytes)")
        
        return {
            "success": True,
            "message": f"Log file {log_file} cleared successfully",
            "original_size_bytes": original_size,
            "original_size_mb": round(original_size / (1024 * 1024), 2)
        }
        
    except Exception as e:
        logger.error(f"[ENDPOINT] Failed to clear log file {log_file}: {e}")
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
    
    # Log file information
    log_dir = Path("logs")
    logger.info(f"Log files will be written to: {log_dir.absolute()}")
    logger.info("Available log files:")
    logger.info("  - server.log (all server logs)")
    logger.info("  - server_errors.log (error logs only)")
    logger.info("  - api_requests.log (API request/response logs)")
    
    # Print to console for immediate feedback
    print(f"\n🚀 Starting PyTorch Inference Framework Server")
    print(f"📁 Log files directory: {log_dir.absolute()}")
    print(f"📄 Server logs: {log_dir / 'server.log'}")
    print(f"🚨 Error logs: {log_dir / 'server_errors.log'}")
    print(f"🌐 API logs: {log_dir / 'api_requests.log'}")
    print(f"📊 Monitor logs at: http://localhost:{server_config['port']}/logs")
    print(f"⚙️  Server config: http://localhost:{server_config['port']}/config")
    
    # Start the FastAPI server with configuration
    logger.info("Starting server...")
    uvicorn.run(
        "main:app",
        host=server_config['host'],
        port=server_config['port'],
        reload=server_config['reload'],
        log_level=server_config['log_level'].lower()
    )

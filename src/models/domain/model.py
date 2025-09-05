"""
Domain model for ML models.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ModelStatus(Enum):
    """Model loading status."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


class ModelType(Enum):
    """Model type classifications."""
    TEXT_GENERATION = "text_generation"
    TEXT_TO_SPEECH = "text_to_speech"
    SPEECH_TO_TEXT = "speech_to_text"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    CUSTOM = "custom"


class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    model_type: ModelType
    device_type: DeviceType
    device_id: Optional[int] = None
    batch_size: int = 1
    max_batch_size: int = 8
    use_fp16: bool = False
    use_tensorrt: bool = False
    use_torch_compile: bool = False
    warmup_iterations: int = 3
    timeout: float = 30.0
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetadata:
    """Model metadata information."""
    name: str
    version: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    parameters: Optional[int] = None  # Number of parameters
    model_size_mb: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    source: Optional[str] = None  # huggingface, pytorch_hub, etc.
    model_id: Optional[str] = None  # Original model identifier


@dataclass
class ModelInfo:
    """Complete model information."""
    config: ModelConfig
    metadata: ModelMetadata
    status: ModelStatus = ModelStatus.NOT_LOADED
    loaded_at: Optional[datetime] = None
    model_path: Optional[Path] = None
    memory_usage_mb: Optional[float] = None
    inference_count: int = 0
    last_inference_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class ModelPerformanceMetrics:
    """Model performance tracking."""
    model_name: str
    inference_count: int = 0
    total_inference_time: float = 0.0
    average_inference_time: float = 0.0
    min_inference_time: float = float('inf')
    max_inference_time: float = 0.0
    successful_inferences: int = 0
    failed_inferences: int = 0
    memory_usage_mb: float = 0.0
    gpu_utilization: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_inference(self, inference_time: float, success: bool, memory_usage: Optional[float] = None):
        """Add an inference to the metrics."""
        self.inference_count += 1
        
        if success:
            self.successful_inferences += 1
            self.total_inference_time += inference_time
            self.average_inference_time = self.total_inference_time / self.successful_inferences
            self.min_inference_time = min(self.min_inference_time, inference_time)
            self.max_inference_time = max(self.max_inference_time, inference_time)
        else:
            self.failed_inferences += 1
        
        if memory_usage is not None:
            self.memory_usage_mb = memory_usage
        
        self.last_updated = datetime.now()


@dataclass
class ModelRegistry:
    """Model registry for tracking all models."""
    models: Dict[str, ModelInfo] = field(default_factory=dict)
    performance_metrics: Dict[str, ModelPerformanceMetrics] = field(default_factory=dict)
    
    def register_model(self, model_info: ModelInfo):
        """Register a new model."""
        self.models[model_info.config.name] = model_info
        self.performance_metrics[model_info.config.name] = ModelPerformanceMetrics(
            model_name=model_info.config.name
        )
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model info by name."""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())
    
    def get_loaded_models(self) -> List[ModelInfo]:
        """Get all currently loaded models."""
        return [
            model for model in self.models.values()
            if model.status == ModelStatus.LOADED
        ]
    
    def update_model_status(self, name: str, status: ModelStatus, error_message: Optional[str] = None):
        """Update model status."""
        if name in self.models:
            self.models[name].status = status
            if status == ModelStatus.LOADED:
                self.models[name].loaded_at = datetime.now()
                self.models[name].error_message = None
            elif status == ModelStatus.ERROR:
                self.models[name].error_message = error_message

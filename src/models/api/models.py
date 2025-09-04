"""
Model management API models.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about a model."""
    name: str = Field(..., description="Model name")
    source: str = Field(..., description="Model source (huggingface, pytorch_hub, etc.)")
    model_id: str = Field(..., description="Model identifier")
    task: str = Field(..., description="Task type")
    description: Optional[str] = Field(None, description="Model description")
    size_mb: Optional[float] = Field(None, description="Model size in MB")
    tags: Optional[List[str]] = Field(default=None, description="Model tags")
    loaded: bool = Field(default=False, description="Whether model is loaded")


class ModelDownloadRequest(BaseModel):
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


class ModelDownloadResponse(BaseModel):
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


class ModelListResponse(BaseModel):
    """Response model for listing models."""
    models: List[str] = Field(..., description="List of model names")
    model_info: Dict[str, ModelInfo] = Field(..., description="Detailed model information")
    total_models: int = Field(..., description="Total number of models")


class ModelCacheInfo(BaseModel):
    """Information about model cache."""
    cache_directory: str = Field(..., description="Cache directory path")
    total_models: int = Field(..., description="Total cached models")
    total_size_mb: float = Field(..., description="Total cache size in MB")
    models: List[str] = Field(..., description="List of cached models")


class ModelManagementRequest(BaseModel):
    """Request model for model management operations."""
    action: str = Field(..., description="Management action (retry_download, optimize, etc.)")
    model_name: str = Field(..., description="Model name to manage")
    force_redownload: bool = Field(default=False, description="Force redownload if exists")


class ModelManagementResponse(BaseModel):
    """Response model for model management operations."""
    success: bool = Field(..., description="Whether operation succeeded")
    message: str = Field(..., description="Operation result message")
    action: str = Field(..., description="Action that was performed")
    model_name: str = Field(..., description="Model name that was managed")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_name: str = Field(..., description="Model name")
    model_info: ModelInfo = Field(..., description="Detailed model information")
    status: str = Field(..., description="Model status")
    device: Optional[str] = Field(None, description="Device model is loaded on")
    memory_usage: Optional[int] = Field(None, description="Memory usage in bytes")


class ModelLoadRequest(BaseModel):
    """Request model for loading a model."""
    device: Optional[str] = Field(None, description="Device to load model on (cpu, cuda, auto)")
    force_reload: bool = Field(default=False, description="Force reload even if already loaded")


class ModelUnloadRequest(BaseModel):
    """Request model for unloading a model."""
    force: bool = Field(default=False, description="Force unload even if model is in use")
    immediate: bool = Field(default=False, description="Unload immediately vs in background")

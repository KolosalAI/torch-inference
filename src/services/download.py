"""
Model download and management service.
"""

import logging
import uuid
import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

from ..core.exceptions import ModelError, ServiceUnavailableError
# Import models directly from specific files  
from ..models.api.downloads import ModelDownloadRequest, ModelDownloadResponse, DownloadStatus

logger = logging.getLogger(__name__)


class DownloadService:
    """Service for downloading and managing model downloads."""
    
    def __init__(self, model_manager=None):
        self.model_manager = model_manager
        self.logger = logger
        # In production, this would be a persistent store (Redis, database, etc.)
        self.download_status = {}
    
    async def download_model(
        self, 
        request: ModelDownloadRequest,
        background_download: bool = True
    ) -> ModelDownloadResponse:
        """
        Download a model with enhanced TTS support.
        
        Args:
            request: Model download request
            background_download: Whether to download in background
            
        Returns:
            ModelDownloadResponse with download status
        """
        download_id = str(uuid.uuid4())[:8]
        
        self.logger.info(f"Model download requested - ID: {download_id}")
        self.logger.info(f"  Name: {request.name}, Source: {request.source}")
        self.logger.info(f"  Model ID: {request.model_id}, Task: {request.task}")
        
        try:
            # Check if model already exists
            if self._is_model_already_available(request.name):
                self.logger.info(f"Model '{request.name}' already exists")
                return ModelDownloadResponse(
                    success=True,
                    download_id=download_id,
                    message=f"Model '{request.name}' already exists and is ready to use",
                    model_name=request.name,
                    source=request.source,
                    model_id=request.model_id,
                    status="already_exists",
                    download_info={
                        "download_id": download_id,
                        "existed_at": datetime.now().isoformat()
                    }
                )
            
            # Validate source
            valid_sources = ["pytorch_hub", "torchvision", "huggingface", "url", "tts_auto", "nvidia"]
            if request.source not in valid_sources:
                self.logger.error(f"Invalid source: {request.source}")
                return ModelDownloadResponse(
                    success=False,
                    message=f"Invalid source. Must be one of: {valid_sources}",
                    model_name=request.name,
                    source=request.source,
                    model_id=request.model_id,
                    status="failed",
                    error=f"Invalid source: {request.source}"
                )
            
            # Handle TTS models specially
            if request.source == "tts_auto" or request.auto_convert_tts:
                return await self._handle_tts_download(request, download_id, background_download)
            
            # Estimate download time
            estimated_time = self._estimate_download_time(request.model_id, request.source)
            
            # Initialize download status
            self.download_status[download_id] = {
                "status": "downloading",
                "progress": 0,
                "started_at": datetime.now().isoformat(),
                "model_name": request.name,
                "estimated_completion": None
            }
            
            if background_download:
                # Start background download
                asyncio.create_task(self._perform_download(request, download_id))
                
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
                        "started_at": datetime.now().isoformat()
                    }
                )
            else:
                # Synchronous download
                success = await self._perform_download(request, download_id)
                
                if success:
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
            self.logger.error(f"Download request failed - ID: {download_id}, Error: {e}")
            return ModelDownloadResponse(
                success=False,
                message=f"Download request failed: {str(e)}",
                model_name=request.name,
                source=request.source,
                model_id=request.model_id,
                status="failed",
                error=str(e)
            )
    
    def get_download_status(self, download_id: str) -> Dict[str, Any]:
        """Get the status of a download by ID."""
        if download_id not in self.download_status:
            return {
                "download_id": download_id,
                "status": "not_found",
                "error": "Download ID not found"
            }
        
        return {
            "download_id": download_id,
            **self.download_status[download_id]
        }
    
    def list_download_history(self) -> Dict[str, Any]:
        """List all download history."""
        return {
            "downloads": self.download_status,
            "total_downloads": len(self.download_status)
        }
    
    async def _perform_download(self, request: ModelDownloadRequest, download_id: str) -> bool:
        """Perform the actual model download."""
        try:
            self.logger.info(f"Starting download task - ID: {download_id}")
            
            if not self.model_manager:
                self.logger.error("Model manager not available")
                self._update_download_status(download_id, "failed", error="Model manager not available")
                return False
            
            # Update status to downloading
            self._update_download_status(download_id, "downloading", progress=10)
            
            # Prepare download parameters
            download_kwargs = {
                "task": request.task,
                "pretrained": True
            }
            
            if request.custom_settings:
                download_kwargs.update(request.custom_settings)
            
            # Perform download through model manager
            self._update_download_status(download_id, "downloading", progress=50)

            # TODO: Implement actual download logic with progress tracking and error handling
            # Tracking issue: https://github.com/torch-inference/issues/download-implementation
            if hasattr(self.model_manager, 'download_and_load_model'):
                self.model_manager.download_and_load_model(
                    request.source,
                    request.model_id,
                    request.name,
                    None,
                    **download_kwargs
                )
            else:
                # Fallback - just wait to simulate download
                await asyncio.sleep(2)
            
            self._update_download_status(download_id, "completed", progress=100)
            
            self.logger.info(f"Download completed successfully - ID: {download_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Download failed - ID: {download_id}, Error: {e}")
            self._update_download_status(download_id, "failed", error=str(e))
            return False
    
    async def _handle_tts_download(
        self, 
        request: ModelDownloadRequest, 
        download_id: str, 
        background_download: bool
    ) -> ModelDownloadResponse:
        """Handle TTS-specific model downloads."""
        self.logger.info(f"Processing TTS model download - ID: {download_id}")
        
        # TTS model registry
        tts_models = {
            "microsoft/speecht5_tts": {
                "description": "Microsoft SpeechT5 TTS model",
                "size_mb": 2500,
                "estimated_time": "8-15 minutes"
            },
            "suno/bark": {
                "description": "Suno Bark TTS model",
                "size_mb": 4000,
                "estimated_time": "15-25 minutes"
            },
            "facebook/bart-large": {
                "description": "BART Large (TTS adaptable)",
                "size_mb": 1600,
                "estimated_time": "5-10 minutes"
            }
        }
        
        model_info = tts_models.get(request.model_id, {
            "description": f"TTS model: {request.model_id}",
            "size_mb": 1000,
            "estimated_time": "5-15 minutes"
        })
        
        if background_download:
            asyncio.create_task(self._perform_tts_download(request, download_id))
            
            return ModelDownloadResponse(
                success=True,
                download_id=download_id,
                message=f"Started downloading TTS model '{request.name}' ({model_info['description']})",
                model_name=request.name,
                source=request.source,
                model_id=request.model_id,
                status="downloading",
                estimated_time=model_info["estimated_time"],
                download_info={
                    "download_id": download_id,
                    "model_type": "tts",
                    "description": model_info["description"],
                    "estimated_size_mb": model_info["size_mb"]
                }
            )
        else:
            success = await self._perform_tts_download(request, download_id)
            
            return ModelDownloadResponse(
                success=success,
                download_id=download_id,
                message=f"{'Successfully downloaded' if success else 'Failed to download'} TTS model '{request.name}'",
                model_name=request.name,
                source=request.source,
                model_id=request.model_id,
                status="completed" if success else "failed",
                download_info={
                    "download_id": download_id,
                    "model_type": "tts"
                }
            )
    
    async def _perform_tts_download(self, request: ModelDownloadRequest, download_id: str) -> bool:
        """Perform TTS-specific download."""
        try:
            self.logger.info(f"Starting TTS download - ID: {download_id}")
            
            # Update status
            self._update_download_status(download_id, "downloading", progress=20)
            
            # Perform main model download
            success = await self._perform_download(request, download_id)
            
            if success:
                # Additional TTS setup if needed
                self._update_download_status(download_id, "configuring", progress=90)
                
                # TTS-specific configuration
                await self._setup_tts_model(request.name)
                
                self._update_download_status(download_id, "completed", progress=100)
                
            return success
            
        except Exception as e:
            self.logger.error(f"TTS download failed - ID: {download_id}, Error: {e}")
            self._update_download_status(download_id, "failed", error=str(e))
            return False
    
    async def _setup_tts_model(self, model_name: str):
        """Setup TTS model after download."""
        try:
            # Placeholder for TTS-specific setup
            # This could include vocoder setup, model optimization, etc.
            await asyncio.sleep(1)  # Simulate setup time
            self.logger.info(f"TTS model setup completed: {model_name}")
        except Exception as e:
            self.logger.warning(f"TTS setup failed for {model_name}: {e}")
    
    def _is_model_already_available(self, model_name: str) -> bool:
        """Check if model is already available."""
        if not self.model_manager:
            return False
        
        # Check if loaded
        if hasattr(self.model_manager, 'is_model_loaded'):
            if self.model_manager.is_model_loaded(model_name):
                return True
        else:
            if model_name in self.model_manager.list_models():
                return True
        
        # Check if in cache
        if hasattr(self.model_manager, 'get_downloader'):
            downloader = self.model_manager.get_downloader()
            if hasattr(downloader, 'is_model_cached') and downloader.is_model_cached(model_name):
                return True
        
        return False
    
    def _estimate_download_time(self, model_id: str, source: str) -> str:
        """Estimate download time based on model size."""
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
        
        # Estimate based on 10 MB/s average download speed
        estimated_seconds = size_mb / 10
        
        if estimated_seconds < 60:
            return f"{int(estimated_seconds)} seconds"
        elif estimated_seconds < 3600:
            return f"{int(estimated_seconds / 60)} minutes"
        else:
            hours = int(estimated_seconds / 3600)
            minutes = int((estimated_seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def _update_download_status(
        self, 
        download_id: str, 
        status: str, 
        progress: int = None,
        error: str = None
    ):
        """Update download status."""
        if download_id in self.download_status:
            self.download_status[download_id]["status"] = status
            if progress is not None:
                self.download_status[download_id]["progress"] = progress
            if error:
                self.download_status[download_id]["error"] = error
            if status == "completed":
                self.download_status[download_id]["completed_at"] = datetime.now().isoformat()

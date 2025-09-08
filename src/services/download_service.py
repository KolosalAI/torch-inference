"""
Download service for model and resource management.

This module provides functionality for downloading and managing
models and other resources with proper error handling and validation.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import asyncio
import os
import hashlib
import aiohttp
import aiofiles
from pathlib import Path

from ..core.exceptions import ValidationError, ServiceUnavailableError
from ..core.config import get_config

logger = logging.getLogger(__name__)


class BasicDownloadService:
    """
    Basic implementation of download service for models and resources.
    
    Provides functionality for downloading models, validating downloads,
    and managing local model storage.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the download service.
        
        Args:
            base_path: Base directory for downloads (defaults to config value)
        """
        self.base_path = base_path or self._get_default_base_path()
        self._downloads_in_progress = set()
        self._download_history = []
        self._initialized = False
        
        logger.debug(f"BasicDownloadService initialized with base_path: {self.base_path}")
    
    def _get_default_base_path(self) -> str:
        """Get default base path for downloads."""
        try:
            config = get_config()
            if hasattr(config, 'models') and hasattr(config.models, 'download_path'):
                return config.models.download_path
            elif hasattr(config, 'storage') and hasattr(config.storage, 'models_path'):
                return config.storage.models_path
            else:
                return "./models"
        except Exception as e:
            logger.warning(f"Failed to get config download path: {e}")
            return "./models"
    
    def initialize(self):
        """Initialize the download service."""
        if not self._initialized:
            # Ensure base directory exists
            os.makedirs(self.base_path, exist_ok=True)
            self._initialized = True
            logger.debug("BasicDownloadService initialization completed")
    
    async def download_model(self, model_info: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Download a model from a remote source.
        
        Args:
            model_info: Dictionary containing model download information
            **kwargs: Additional download parameters
        
        Returns:
            Dict containing download result information
        """
        try:
            self.initialize()
            
            # Validate model info
            if not isinstance(model_info, dict):
                raise ValidationError(
                    field="model_info",
                    details="model_info must be a dictionary"
                )
            
            required_fields = ["name", "url"]
            for field in required_fields:
                if field not in model_info:
                    raise ValidationError(
                        field=field,
                        details=f"Required field '{field}' missing from model_info"
                    )
            
            model_name = model_info["name"]
            model_url = model_info["url"]
            
            # Check if download is already in progress
            if model_name in self._downloads_in_progress:
                raise ServiceUnavailableError(
                    service="download",
                    details=f"Download already in progress for model '{model_name}'"
                )
            
            # Start download
            self._downloads_in_progress.add(model_name)
            
            try:
                download_result = await self._download_file(
                    url=model_url,
                    model_name=model_name,
                    model_info=model_info,
                    **kwargs
                )
                
                # Record download in history
                self._download_history.append({
                    "model_name": model_name,
                    "url": model_url,
                    "timestamp": datetime.utcnow().isoformat(),
                    "success": download_result.get("success", False),
                    "file_path": download_result.get("file_path"),
                    "file_size": download_result.get("file_size", 0)
                })
                
                return download_result
                
            finally:
                self._downloads_in_progress.discard(model_name)
            
        except Exception as e:
            logger.error(f"Model download failed for {model_info.get('name', 'unknown')}: {e}")
            raise
    
    async def _download_file(self, url: str, model_name: str, model_info: Dict[str, Any], 
                           **kwargs) -> Dict[str, Any]:
        """Internal method to perform the actual file download."""
        try:
            # Determine local file path
            filename = model_info.get("filename", f"{model_name}.bin")
            local_path = os.path.join(self.base_path, filename)
            
            # Check if file already exists
            if os.path.exists(local_path) and not kwargs.get("force_download", False):
                file_size = os.path.getsize(local_path)
                logger.info(f"Model file already exists: {local_path} ({file_size} bytes)")
                
                return {
                    "success": True,
                    "file_path": local_path,
                    "file_size": file_size,
                    "already_existed": True,
                    "model_name": model_name,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Download the file
            logger.info(f"Downloading model {model_name} from {url}")
            
            timeout = aiohttp.ClientTimeout(total=kwargs.get("timeout", 3600))  # 1 hour default
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ServiceUnavailableError(
                            service="download",
                            details=f"Download failed with status {response.status}: {response.reason}"
                        )
                    
                    # Get content length if available
                    content_length = response.headers.get("Content-Length")
                    total_size = int(content_length) if content_length else None
                    
                    # Download with progress tracking
                    downloaded_size = 0
                    hash_md5 = hashlib.md5()
                    
                    async with aiofiles.open(local_path, 'wb') as file:
                        async for chunk in response.content.iter_chunked(8192):
                            await file.write(chunk)
                            downloaded_size += len(chunk)
                            hash_md5.update(chunk)
                            
                            # Log progress periodically
                            if total_size and downloaded_size % (1024 * 1024 * 10) == 0:  # Every 10MB
                                progress = (downloaded_size / total_size) * 100
                                logger.debug(f"Download progress for {model_name}: {progress:.1f}%")
            
            # Verify download
            file_hash = hash_md5.hexdigest()
            expected_hash = model_info.get("md5_hash")
            
            if expected_hash and file_hash != expected_hash:
                os.remove(local_path)  # Remove corrupted file
                raise ValidationError(
                    field="file_hash",
                    details=f"Downloaded file hash mismatch (expected: {expected_hash}, got: {file_hash})"
                )
            
            logger.info(f"Successfully downloaded {model_name} to {local_path} ({downloaded_size} bytes)")
            
            return {
                "success": True,
                "file_path": local_path,
                "file_size": downloaded_size,
                "file_hash": file_hash,
                "model_name": model_name,
                "timestamp": datetime.utcnow().isoformat(),
                "already_existed": False
            }
            
        except Exception as e:
            # Clean up partial download
            if 'local_path' in locals() and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up partial download: {cleanup_error}")
            
            raise ServiceUnavailableError(
                service="download",
                details=f"File download failed: {e}",
                cause=e
            )
    
    def list_downloaded_models(self) -> List[Dict[str, Any]]:
        """List all downloaded models."""
        try:
            self.initialize()
            
            models = []
            
            if not os.path.exists(self.base_path):
                return models
            
            for filename in os.listdir(self.base_path):
                file_path = os.path.join(self.base_path, filename)
                
                if os.path.isfile(file_path):
                    stat_info = os.stat(file_path)
                    
                    models.append({
                        "filename": filename,
                        "file_path": file_path,
                        "file_size": stat_info.st_size,
                        "modified_time": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                        "created_time": datetime.fromtimestamp(stat_info.st_ctime).isoformat()
                    })
            
            return sorted(models, key=lambda x: x["modified_time"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list downloaded models: {e}")
            return []
    
    def get_model_file_path(self, model_name: str) -> Optional[str]:
        """Get the local file path for a downloaded model."""
        try:
            models = self.list_downloaded_models()
            
            # Try exact filename match first
            for model in models:
                if model["filename"] == f"{model_name}.bin":
                    return model["file_path"]
            
            # Try partial name match
            for model in models:
                if model_name in model["filename"]:
                    return model["file_path"]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get model file path for {model_name}: {e}")
            return None
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a model is already downloaded."""
        return self.get_model_file_path(model_name) is not None
    
    def get_download_status(self, model_name: str) -> Dict[str, Any]:
        """Get download status for a specific model."""
        try:
            status = {
                "model_name": model_name,
                "downloaded": False,
                "downloading": model_name in self._downloads_in_progress,
                "file_path": None,
                "file_size": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            file_path = self.get_model_file_path(model_name)
            if file_path:
                status.update({
                    "downloaded": True,
                    "file_path": file_path,
                    "file_size": os.path.getsize(file_path)
                })
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get download status for {model_name}: {e}")
            return {
                "model_name": model_name,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_download_history(self) -> List[Dict[str, Any]]:
        """Get download history."""
        return list(self._download_history)
    
    def delete_downloaded_model(self, model_name: str) -> bool:
        """Delete a downloaded model file."""
        try:
            file_path = self.get_model_file_path(model_name)
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted downloaded model: {file_path}")
                return True
            else:
                logger.warning(f"Model file not found for deletion: {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False
    
    def cleanup_partial_downloads(self) -> int:
        """Clean up any partial or corrupted downloads."""
        try:
            self.initialize()
            
            cleaned_count = 0
            
            if not os.path.exists(self.base_path):
                return cleaned_count
            
            for filename in os.listdir(self.base_path):
                file_path = os.path.join(self.base_path, filename)
                
                # Check for temporary files or very small files that might be corrupted
                if (filename.endswith('.tmp') or 
                    filename.endswith('.part') or 
                    (os.path.isfile(file_path) and os.path.getsize(file_path) < 1024)):  # Less than 1KB
                    
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                        logger.info(f"Cleaned up partial download: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to clean up {file_path}: {e}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup partial downloads: {e}")
            return 0
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            downloaded_models = self.list_downloaded_models()
            total_size = sum(model["file_size"] for model in downloaded_models)
            
            return {
                "total_models": len(downloaded_models),
                "total_size_bytes": total_size,
                "downloads_in_progress": len(self._downloads_in_progress),
                "download_history_count": len(self._download_history),
                "base_path": self.base_path,
                "service": "basic_download_service",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get service stats: {e}")
            return {"error": str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the download service."""
        try:
            health = {
                "healthy": True,
                "service": "basic_download_service",
                "initialized": self._initialized,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check if base path is accessible
            try:
                os.makedirs(self.base_path, exist_ok=True)
                # Try to write a test file
                test_file = os.path.join(self.base_path, '.health_check')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                health["base_path_writable"] = True
            except Exception as e:
                health["healthy"] = False
                health["base_path_writable"] = False
                health["base_path_error"] = str(e)
            
            # Add basic stats
            health.update(self.get_service_stats())
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "service": "basic_download_service",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # Legacy compatibility methods
    
    def download_model_sync(self, model_info: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Synchronous model download for backward compatibility."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.download_model(model_info, **kwargs))
        except RuntimeError:
            return asyncio.run(self.download_model(model_info, **kwargs))
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status (legacy method)."""
        return {
            "status": "running" if self._initialized else "not_initialized",
            "stats": self.get_service_stats(),
            "health": self.get_health_status()
        }


# Convenience function for creating the service
def create_download_service(base_path: Optional[str] = None) -> BasicDownloadService:
    """Create a download service instance."""
    return BasicDownloadService(base_path)

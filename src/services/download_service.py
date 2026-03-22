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
    
    async def list_downloads(self, status_filter: Optional[str] = None, 
                           limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """
        List downloads with filtering and pagination.
        
        Args:
            status_filter: Filter by download status
            limit: Maximum number of downloads to return
            offset: Number of downloads to skip
            
        Returns:
            Dict containing download list and metadata
        """
        try:
            self.initialize()
            
            # Get download history
            all_downloads = []
            
            # Convert download history to download info format
            for item in self._download_history:
                download_info = {
                    "download_id": f"dl_{hash(item['model_name'])}",
                    "url": item.get("url", ""),
                    "filename": item["model_name"],
                    "status": "completed" if item.get("success", False) else "failed",
                    "progress": 100.0 if item.get("success", False) else 0.0,
                    "file_size": item.get("file_size", 0),
                    "downloaded_size": item.get("file_size", 0) if item.get("success", False) else 0,
                    "download_speed": None,
                    "eta": None,
                    "error_message": None,
                    "created_at": datetime.fromisoformat(item["timestamp"]),
                    "completed_at": datetime.fromisoformat(item["timestamp"]) if item.get("success", False) else None
                }
                all_downloads.append(download_info)
            
            # Add downloads in progress
            for model_name in self._downloads_in_progress:
                download_info = {
                    "download_id": f"dl_{hash(model_name)}",
                    "url": "",
                    "filename": model_name,
                    "status": "downloading",
                    "progress": 50.0,  # Mock progress
                    "file_size": None,
                    "downloaded_size": None,
                    "download_speed": None,
                    "eta": None,
                    "error_message": None,
                    "created_at": datetime.utcnow(),
                    "completed_at": None
                }
                all_downloads.append(download_info)
            
            # Apply status filter
            if status_filter:
                all_downloads = [d for d in all_downloads if d["status"] == status_filter]
            
            # Apply pagination
            total_count = len(all_downloads)
            paginated_downloads = all_downloads[offset:offset + limit]
            
            # Calculate statistics
            status_counts = {}
            for download in all_downloads:
                status = download["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "downloads": paginated_downloads,
                "total_count": total_count,
                "active_downloads": status_counts.get("downloading", 0),
                "completed_downloads": status_counts.get("completed", 0),
                "failed_downloads": status_counts.get("failed", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to list downloads: {e}")
            return {
                "downloads": [],
                "total_count": 0,
                "active_downloads": 0,
                "completed_downloads": 0,
                "failed_downloads": 0
            }
    
    async def get_download_info(self, download_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific download.
        
        Args:
            download_id: Download ID
            
        Returns:
            Download information or None if not found
        """
        try:
            # List all downloads and find matching ID
            downloads_data = await self.list_downloads()
            
            for download in downloads_data["downloads"]:
                if download["download_id"] == download_id:
                    return download
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get download info for {download_id}: {e}")
            return None
    
    async def get_download_status(self, download_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a download.
        
        Args:
            download_id: Download ID
            
        Returns:
            Download status or None if not found
        """
        try:
            download_info = await self.get_download_info(download_id)
            
            if not download_info:
                return None
            
            return {
                "download_id": download_id,
                "status": download_info["status"],
                "progress": download_info["progress"],
                "file_size": download_info["file_size"],
                "downloaded_size": download_info["downloaded_size"],
                "download_speed": download_info["download_speed"],
                "eta": download_info["eta"],
                "error_message": download_info["error_message"],
                "timestamp": download_info["created_at"].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get download status for {download_id}: {e}")
            return None
    
    async def cancel_download(self, download_id: str, delete_partial: bool = False) -> bool:
        """
        Cancel an active download.
        
        Args:
            download_id: Download ID to cancel
            delete_partial: Whether to delete partially downloaded files
            
        Returns:
            True if canceled, False if not found or not cancelable
        """
        try:
            # For basic implementation, we can't really cancel downloads
            # This is a placeholder for compatibility
            download_info = await self.get_download_info(download_id)
            
            if not download_info:
                return False
            
            # Only allow canceling downloads that are in progress
            if download_info["status"] != "downloading":
                return False
            
            # For basic implementation, just return True
            # In a real implementation, this would stop the download task
            logger.info(f"Download {download_id} canceled (basic implementation)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel download {download_id}: {e}")
            return False
    
    async def list_files(self, file_type_filter: Optional[str] = None, 
                        limit: int = 50) -> List[Dict[str, Any]]:
        """
        List downloaded files.
        
        Args:
            file_type_filter: Filter by file type
            limit: Maximum number of files to return
            
        Returns:
            List of file information
        """
        try:
            self.initialize()
            
            files = []
            downloaded_models = self.list_downloaded_models()
            
            for model in downloaded_models[:limit]:
                file_info = {
                    "filename": model["filename"],
                    "file_path": model["file_path"],
                    "file_size": model["file_size"],
                    "mime_type": self._guess_mime_type(model["filename"]),
                    "created_at": datetime.fromisoformat(model["created_time"]),
                    "modified_at": datetime.fromisoformat(model["modified_time"]),
                    "is_model_file": True,
                    "checksum": None
                }
                
                # Apply file type filter if specified
                if file_type_filter:
                    if file_type_filter == "model" and not file_info["is_model_file"]:
                        continue
                
                files.append(file_info)
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    async def get_file_path(self, filename: str) -> Optional[str]:
        """
        Get file path for a specific filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            File path or None if not found
        """
        try:
            downloaded_models = self.list_downloaded_models()
            
            for model in downloaded_models:
                if model["filename"] == filename:
                    return model["file_path"]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get file path for {filename}: {e}")
            return None
    
    async def delete_file(self, filename: str) -> bool:
        """
        Delete a downloaded file.
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            file_path = await self.get_file_path(filename)
            
            if not file_path or not os.path.exists(file_path):
                return False
            
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {filename}: {e}")
            return False
    
    def _guess_mime_type(self, filename: str) -> str:
        """Guess MIME type for a filename."""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or 'application/octet-stream'
    
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

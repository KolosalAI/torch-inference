"""
Disk space management utilities for tests.
"""
import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Optional, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DiskSpaceManager:
    """Manages disk space for test operations."""
    
    def __init__(self, min_free_space_mb: int = 1024):
        """
        Initialize disk space manager.
        
        Args:
            min_free_space_mb: Minimum free space in MB required for operations
        """
        self.min_free_space_mb = min_free_space_mb
        self.min_free_space_bytes = min_free_space_mb * 1024 * 1024
    
    def get_free_space(self, path: Union[str, Path] = None) -> int:
        """
        Get free space in bytes for given path.
        
        Args:
            path: Path to check (defaults to current directory)
            
        Returns:
            Free space in bytes
        """
        if path is None:
            path = os.getcwd()
        
        try:
            if os.name == 'nt':  # Windows
                import shutil
                return shutil.disk_usage(path).free
            else:  # Unix/Linux
                statvfs = os.statvfs(path)
                return statvfs.f_frsize * statvfs.f_bavail
        except Exception as e:
            logger.warning(f"Could not get disk space for {path}: {e}")
            return 0
    
    def has_enough_space(self, path: Union[str, Path] = None, required_mb: Optional[int] = None) -> bool:
        """
        Check if there's enough free space.
        
        Args:
            path: Path to check (defaults to current directory)
            required_mb: Required space in MB (defaults to min_free_space_mb)
            
        Returns:
            True if enough space is available
        """
        if required_mb is None:
            required_mb = self.min_free_space_mb
        
        required_bytes = required_mb * 1024 * 1024
        free_space = self.get_free_space(path)
        
        return free_space >= required_bytes
    
    def cleanup_temp_files(self, pattern: str = "pytest-of-*") -> int:
        """
        Clean up temporary files matching pattern.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            Number of bytes freed
        """
        temp_dir = Path(tempfile.gettempdir())
        bytes_freed = 0
        
        try:
            for item in temp_dir.glob(pattern):
                if item.is_dir():
                    try:
                        size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                        shutil.rmtree(item, ignore_errors=True)
                        bytes_freed += size
                        logger.info(f"Cleaned up {item}, freed {size / 1024 / 1024:.1f} MB")
                    except Exception as e:
                        logger.warning(f"Could not clean up {item}: {e}")
                elif item.is_file():
                    try:
                        size = item.stat().st_size
                        item.unlink()
                        bytes_freed += size
                        logger.info(f"Cleaned up {item}, freed {size / 1024 / 1024:.1f} MB")
                    except Exception as e:
                        logger.warning(f"Could not clean up {item}: {e}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        
        return bytes_freed
    
    def ensure_space_available(self, required_mb: int, path: Union[str, Path] = None) -> bool:
        """
        Ensure enough space is available, cleaning up if necessary.
        
        Args:
            required_mb: Required space in MB
            path: Path to check
            
        Returns:
            True if space is available after cleanup
        """
        if self.has_enough_space(path, required_mb):
            return True
        
        logger.info(f"Insufficient disk space, attempting cleanup...")
        bytes_freed = self.cleanup_temp_files()
        logger.info(f"Cleanup freed {bytes_freed / 1024 / 1024:.1f} MB")
        
        return self.has_enough_space(path, required_mb)


@contextmanager
def managed_temp_file(suffix: str = '', prefix: str = 'tmp', dir: str = None, 
                     required_space_mb: int = 100, delete: bool = True):
    """
    Context manager for temporary files with disk space management.
    
    Args:
        suffix: File suffix
        prefix: File prefix
        dir: Directory for temp file
        required_space_mb: Required free space in MB
        delete: Whether to delete file on exit
    """
    manager = DiskSpaceManager()
    
    if not manager.ensure_space_available(required_space_mb, dir):
        raise OSError(f"Insufficient disk space: need {required_space_mb}MB")
    
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix, prefix=prefix, dir=dir, delete=False
        )
        yield temp_file
    finally:
        if temp_file is not None:
            temp_file.close()
            if delete and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logger.warning(f"Could not delete temp file {temp_file.name}: {e}")


@contextmanager
def managed_temp_dir(suffix: str = '', prefix: str = 'tmp', dir: str = None,
                    required_space_mb: int = 100):
    """
    Context manager for temporary directories with disk space management.
    
    Args:
        suffix: Directory suffix
        prefix: Directory prefix
        dir: Parent directory
        required_space_mb: Required free space in MB
    """
    manager = DiskSpaceManager()
    
    if not manager.ensure_space_available(required_space_mb, dir):
        raise OSError(f"Insufficient disk space: need {required_space_mb}MB")
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        yield Path(temp_dir)
    finally:
        if temp_dir is not None and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Could not delete temp dir {temp_dir}: {e}")


def safe_torch_save(obj, path: Union[str, Path], required_space_mb: int = 500):
    """
    Safely save PyTorch object with disk space checking.
    
    Args:
        obj: Object to save
        path: Path to save to
        required_space_mb: Required free space in MB
    """
    import torch
    
    manager = DiskSpaceManager()
    path = Path(path)
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check disk space
    if not manager.ensure_space_available(required_space_mb, path.parent):
        raise OSError(f"Insufficient disk space for saving model: need {required_space_mb}MB")
    
    try:
        torch.save(obj, path)
        logger.info(f"Successfully saved model to {path}")
    except Exception as e:
        # Clean up partial file if it exists
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        raise OSError(f"Failed to save model: {e}") from e


# Global instance
disk_manager = DiskSpaceManager()
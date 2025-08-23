"""
Model downloader for PyTorch models.

This module provides functionality to download models from various sources:
- PyTorch Hub
- Hugging Face Hub  
- Torchvision models
- Custom URLs
- Local caching and management
"""

import os
import json
import hashlib
import requests
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from urllib.parse import urlparse
from dataclasses import dataclass, asdict
import logging
import torch
import torch.nn as nn

# Try to import optional dependencies
try:
    import torchvision.models as tv_models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoConfig,
        AutoModelForSequenceClassification,
        AutoModelForImageClassification
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)


def _get_project_root() -> Path:
    """Get the project root directory."""
    current_path = Path(__file__).resolve()
    # Go up from framework/core/model_downloader.py to project root
    return current_path.parent.parent.parent


def _load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = _get_project_root() / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}")
    return {}


def _get_default_cache_dir() -> Path:
    """Get the default cache directory from config or use project models/ directory."""
    config = _load_config()
    
    # Try to get cache dir from config
    cache_dir = config.get("models", {}).get("download", {}).get("cache_dir")
    
    if cache_dir:
        # If relative path, make it relative to project root
        cache_path = Path(cache_dir)
        if not cache_path.is_absolute():
            cache_path = _get_project_root() / cache_path
        return cache_path
    else:
        # Default to project models/ directory
        return _get_project_root() / "models"


@dataclass
class ModelInfo:
    """Information about a downloadable model."""
    name: str
    source: str  # 'pytorch_hub', 'huggingface', 'torchvision', 'url'
    model_id: str  # Model identifier (e.g., 'pytorch/vision:v0.10.0', 'bert-base-uncased')
    task: str  # 'classification', 'detection', 'segmentation', 'text-classification', etc.
    description: str = ""
    size_mb: Optional[float] = None
    license: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ModelDownloader:
    """Downloads and manages PyTorch models from various sources."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model downloader.
        
        Args:
            cache_dir: Directory to cache downloaded models. If None, uses config or default cache.
            config: Configuration dictionary. If None, loads from config.yaml.
        """
        if config is None:
            config = _load_config()
        
        self.config = config.get("models", {}).get("download", {})
        
        if cache_dir is None:
            cache_dir = _get_default_cache_dir()
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry file location - check config first
        registry_file = self.config.get("registry_file", "model_registry.json")
        registry_path = Path(registry_file)
        
        if registry_path.is_absolute():
            self.registry_file = registry_path
        else:
            # If relative, make it relative to cache_dir, but handle the case where
            # registry_file might already include the cache directory path
            if registry_file.startswith("models/"):
                # Remove the models/ prefix since we're already in the cache_dir
                registry_file = registry_file[7:]  # Remove "models/"
            self.registry_file = self.cache_dir / registry_file
        
        self.registry = self._load_registry()
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load source configurations
        self.sources_config = self.config.get("sources", {})
        
        self.logger.info(f"Model downloader initialized with cache dir: {self.cache_dir}")
        self.logger.info(f"Registry file: {self.registry_file}")
        self.logger.info(f"Auto-download enabled: {self.config.get('auto_download', True)}")
    
    def _is_source_enabled(self, source: str) -> bool:
        """Check if a source is enabled in configuration."""
        return self.sources_config.get(source, {}).get("enabled", True)
    
    def _validate_source_config(self, source: str) -> None:
        """Validate that a source is enabled and available."""
        if not self._is_source_enabled(source):
            raise ValueError(f"Source '{source}' is disabled in configuration")
        
        # Check for required dependencies
        if source == "torchvision" and not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision not available. Install with: pip install torchvision")
        elif source == "huggingface" and not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available. Install with: pip install transformers")
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load the model registry from disk."""
        if not self.registry_file.exists():
            return {}
        
        try:
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load registry: {e}")
            return {}
    
    def _save_registry(self) -> None:
        """Save the model registry to disk."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save registry: {e}")
    
    def _register_model(self, model_name: str, model_info: Dict[str, Any]) -> None:
        """Register a downloaded model in the registry."""
        self.registry[model_name] = model_info
        self._save_registry()
    
    def _get_model_cache_dir(self, model_name: str) -> Path:
        """Get cache directory for a specific model."""
        # Create a safe filename from model name
        safe_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in model_name)
        return self.cache_dir / safe_name
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def download_pytorch_hub_model(
        self, 
        repo_or_dir: str, 
        model: str,
        model_name: Optional[str] = None,
        pretrained: bool = True,
        **kwargs
    ) -> Tuple[Path, ModelInfo]:
        """
        Download a model from PyTorch Hub.
        
        Args:
            repo_or_dir: Repository name (e.g., 'pytorch/vision:v0.10.0')
            model: Model name (e.g., 'resnet18')
            model_name: Custom name for the model. If None, uses model name.
            pretrained: Whether to load pretrained weights
            **kwargs: Additional arguments to pass to torch.hub.load
            
        Returns:
            Tuple of (model_path, model_info)
        """
        # Validate source configuration
        self._validate_source_config("pytorch_hub")
        
        if model_name is None:
            model_name = f"{repo_or_dir.replace('/', '_').replace(':', '_')}_{model}"
        
        self.logger.info(f"Downloading PyTorch Hub model: {repo_or_dir}/{model}")
        
        # Check if already cached
        if self.is_model_cached(model_name):
            cached_path = Path(self.registry[model_name]["path"])
            if cached_path.exists():
                self.logger.info(f"Using cached model: {model_name}")
                return cached_path, ModelInfo(**self.registry[model_name]["info"])
        
        try:
            # Filter out non-PyTorch Hub parameters
            # PyTorch Hub models don't accept 'task' as a parameter
            hub_kwargs = {k: v for k, v in kwargs.items() if k not in ['task']}
            
            # Download model from PyTorch Hub
            hub_model = torch.hub.load(
                repo_or_dir, 
                model, 
                pretrained=pretrained,
                **hub_kwargs
            )
            
            # Create model cache directory
            model_dir = self._get_model_cache_dir(model_name)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the full model
            model_path = model_dir / "model.pt"
            torch.save(hub_model, model_path)
            
            # Save state dict separately
            state_dict_path = model_dir / "state_dict.pt" 
            torch.save(hub_model.state_dict(), state_dict_path)
            
            # Create model info
            task = kwargs.get('task', 'classification')  # Get task from kwargs, default to classification
            model_info = ModelInfo(
                name=model_name,
                source="pytorch_hub",
                model_id=f"{repo_or_dir}/{model}",
                task=task,
                description=f"PyTorch Hub model: {repo_or_dir}/{model}",
                size_mb=model_path.stat().st_size / (1024 * 1024),
                tags=["pytorch_hub", "pretrained" if pretrained else "random"]
            )
            
            # Save model info
            info_path = model_dir / "model_info.json"
            with open(info_path, 'w') as f:
                json.dump(asdict(model_info), f, indent=2)
            
            # Register model
            self._register_model(model_name, {
                "path": str(model_path),
                "state_dict_path": str(state_dict_path),
                "info_path": str(info_path),
                "info": asdict(model_info)
            })
            
            self.logger.info(f"Successfully downloaded {model_name} ({model_info.size_mb:.1f} MB)")
            return model_path, model_info
            
        except Exception as e:
            self.logger.error(f"Failed to download PyTorch Hub model {repo_or_dir}/{model}: {e}")
            raise
    
    def download_torchvision_model(
        self,
        model_name: str,
        pretrained: bool = True,
        custom_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[Path, ModelInfo]:
        """
        Download a model from torchvision.
        
        Args:
            model_name: Name of the torchvision model (e.g., 'resnet18', 'vgg16')
            pretrained: Whether to load pretrained weights
            custom_name: Custom name for the model. If None, uses model_name.
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            Tuple of (model_path, model_info)
        """
        # Validate source configuration
        self._validate_source_config("torchvision")
        
        if custom_name is None:
            custom_name = f"torchvision_{model_name}"
        
        self.logger.info(f"Downloading torchvision model: {model_name}")
        
        # Check if already cached
        if self.is_model_cached(custom_name):
            cached_path = Path(self.registry[custom_name]["path"])
            if cached_path.exists():
                self.logger.info(f"Using cached model: {custom_name}")
                return cached_path, ModelInfo(**self.registry[custom_name]["info"])
        
        try:
            # Get model constructor
            model_fn = getattr(tv_models, model_name)
            
            # Filter out parameters that torchvision models don't accept
            # Torchvision models don't accept 'task' as a parameter
            tv_kwargs = {k: v for k, v in kwargs.items() if k not in ['task']}
            
            # Create model - use 'weights' instead of deprecated 'pretrained'
            if pretrained:
                # For newer torchvision versions, use weights="DEFAULT" or weights="IMAGENET1K_V1"
                try:
                    model = model_fn(weights="DEFAULT", **tv_kwargs)
                except TypeError:
                    # Fallback for older versions or models that still use pretrained
                    model = model_fn(pretrained=pretrained, **tv_kwargs)
            else:
                try:
                    model = model_fn(weights=None, **tv_kwargs)
                except TypeError:
                    # Fallback for older versions
                    model = model_fn(pretrained=pretrained, **tv_kwargs)
            
            # Create model cache directory
            model_dir = self._get_model_cache_dir(custom_name)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the full model
            model_path = model_dir / "model.pt"
            torch.save(model, model_path)
            
            # Save state dict separately
            state_dict_path = model_dir / "state_dict.pt"
            torch.save(model.state_dict(), state_dict_path)
            
            # Create model info
            task = kwargs.get('task', 'image-classification')  # Get task from kwargs, default to image-classification
            model_info = ModelInfo(
                name=custom_name,
                source="torchvision",
                model_id=model_name,
                task=task,
                description=f"Torchvision model: {model_name}",
                size_mb=model_path.stat().st_size / (1024 * 1024),
                tags=["torchvision", "pretrained" if pretrained else "random", "vision"]
            )
            
            # Save model info
            info_path = model_dir / "model_info.json"
            with open(info_path, 'w') as f:
                json.dump(asdict(model_info), f, indent=2)
            
            # Register model
            self._register_model(custom_name, {
                "path": str(model_path),
                "state_dict_path": str(state_dict_path), 
                "info_path": str(info_path),
                "info": asdict(model_info)
            })
            
            self.logger.info(f"Successfully downloaded {custom_name} ({model_info.size_mb:.1f} MB)")
            return model_path, model_info
            
        except Exception as e:
            self.logger.error(f"Failed to download torchvision model {model_name}: {e}")
            raise
    
    def download_huggingface_model(
        self,
        model_id: str,
        task: str = "feature-extraction",
        custom_name: Optional[str] = None
    ) -> Tuple[Path, ModelInfo]:
        """
        Download a model from Hugging Face Hub.
        
        Args:
            model_id: Hugging Face model identifier (e.g., 'bert-base-uncased')
            task: Task type for the model
            custom_name: Custom name for the model. If None, uses model_id.
            
        Returns:
            Tuple of (model_path, model_info)
        """
        # Validate source configuration
        self._validate_source_config("huggingface")
        
        if custom_name is None:
            custom_name = model_id.replace('/', '_')
        
        self.logger.info(f"Downloading Hugging Face model: {model_id}")
        
        # Check if already cached
        if self.is_model_cached(custom_name):
            cached_path = Path(self.registry[custom_name]["path"])
            if cached_path.exists():
                self.logger.info(f"Using cached model: {custom_name}")
                return cached_path, ModelInfo(**self.registry[custom_name]["info"])
        
        try:
            # Create model cache directory
            model_dir = self._get_model_cache_dir(custom_name)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Choose appropriate model class based on task
            if task == "text-classification":
                model = AutoModelForSequenceClassification.from_pretrained(model_id)
            elif task == "image-classification":
                model = AutoModelForImageClassification.from_pretrained(model_id)
            else:
                model = AutoModel.from_pretrained(model_id)
            
            # Try to load tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                tokenizer.save_pretrained(model_dir)
            except:
                self.logger.warning(f"Could not load tokenizer for {model_id}")
            
            # Save the model
            model_path = model_dir / "pytorch_model.pt"
            torch.save(model.state_dict(), model_path)
            
            # Save full model
            full_model_path = model_dir / "model.pt"
            torch.save(model, full_model_path)
            
            # Save config
            config = AutoConfig.from_pretrained(model_id)
            config.save_pretrained(model_dir)
            
            # Create model info
            model_info = ModelInfo(
                name=custom_name,
                source="huggingface",
                model_id=model_id,
                task=task,
                description=f"Hugging Face model: {model_id}",
                size_mb=full_model_path.stat().st_size / (1024 * 1024),
                tags=["huggingface", "transformers", task]
            )
            
            # Save model info
            info_path = model_dir / "model_info.json"
            with open(info_path, 'w') as f:
                json.dump(asdict(model_info), f, indent=2)
            
            # Register model
            self._register_model(custom_name, {
                "path": str(full_model_path),
                "state_dict_path": str(model_path),
                "info_path": str(info_path),
                "config_path": str(model_dir / "config.json"),
                "info": asdict(model_info)
            })
            
            self.logger.info(f"Successfully downloaded {custom_name} ({model_info.size_mb:.1f} MB)")
            return full_model_path, model_info
            
        except Exception as e:
            self.logger.error(f"Failed to download Hugging Face model {model_id}: {e}")
            raise
    
    def download_from_url(
        self,
        url: str,
        model_name: str,
        task: str = "custom",
        description: str = "",
        expected_hash: Optional[str] = None
    ) -> Tuple[Path, ModelInfo]:
        """
        Download a model from a URL.
        
        Args:
            url: URL to download the model from
            model_name: Name to assign to the model
            task: Task type for the model
            description: Description of the model
            expected_hash: Expected SHA256 hash of the file for validation
            
        Returns:
            Tuple of (model_path, model_info)
        """
        # Validate source configuration
        self._validate_source_config("url")
        
        self.logger.info(f"Downloading model from URL: {url}")
        
        # Check if already cached
        if self.is_model_cached(model_name):
            cached_path = Path(self.registry[model_name]["path"])
            if cached_path.exists():
                self.logger.info(f"Using cached model: {model_name}")
                return cached_path, ModelInfo(**self.registry[model_name]["info"])
        
        try:
            # Create model cache directory
            model_dir = self._get_model_cache_dir(model_name)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine filename from URL
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename or not any(filename.endswith(ext) for ext in ['.pt', '.pth', '.onnx']):
                filename = "model.pt"
            
            model_path = model_dir / filename
            
            # Get URL configuration
            url_config = self.sources_config.get("url", {})
            verify_ssl = url_config.get("verify_ssl", True)
            timeout = url_config.get("timeout_seconds", 300)
            
            # Download file
            response = requests.get(url, stream=True, verify=verify_ssl, timeout=timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\rDownloading {model_name}: {progress:.1f}%", end="", flush=True)
            
            print()  # New line after progress
            
            # Verify hash if provided
            if expected_hash:
                actual_hash = self._compute_file_hash(model_path)
                if actual_hash != expected_hash:
                    model_path.unlink()  # Remove corrupted file
                    raise ValueError(f"Hash mismatch. Expected: {expected_hash}, Got: {actual_hash}")
            
            # Create model info
            model_info = ModelInfo(
                name=model_name,
                source="url",
                model_id=url,
                task=task,
                description=description or f"Model downloaded from {url}",
                size_mb=model_path.stat().st_size / (1024 * 1024),
                tags=["url", task]
            )
            
            # Save model info
            info_path = model_dir / "model_info.json"
            with open(info_path, 'w') as f:
                json.dump(asdict(model_info), f, indent=2)
            
            # Register model
            self._register_model(model_name, {
                "path": str(model_path),
                "info_path": str(info_path),
                "info": asdict(model_info)
            })
            
            self.logger.info(f"Successfully downloaded {model_name} ({model_info.size_mb:.1f} MB)")
            return model_path, model_info
            
        except Exception as e:
            self.logger.error(f"Failed to download model from {url}: {e}")
            raise
    
    def list_available_models(self) -> Dict[str, ModelInfo]:
        """List all available (cached) models."""
        models = {}
        for model_name, model_data in self.registry.items():
            try:
                models[model_name] = ModelInfo(**model_data["info"])
            except Exception as e:
                self.logger.warning(f"Failed to load model info for {model_name}: {e}")
        return models
    
    def is_model_cached(self, model_name: str) -> bool:
        """Check if a model is already cached."""
        return model_name in self.registry
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get the path to a cached model."""
        if not self.is_model_cached(model_name):
            return None
        
        path_str = self.registry[model_name]["path"]
        path = Path(path_str)
        
        if path.exists():
            return path
        else:
            # Remove from registry if file doesn't exist
            self.logger.warning(f"Model file not found, removing from registry: {model_name}")
            del self.registry[model_name]
            self._save_registry()
            return None
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get info about a cached model."""
        if not self.is_model_cached(model_name):
            return None
        
        try:
            return ModelInfo(**self.registry[model_name]["info"])
        except Exception as e:
            self.logger.warning(f"Failed to load model info for {model_name}: {e}")
            return None
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model from cache."""
        if not self.is_model_cached(model_name):
            return False
        
        try:
            # Get model directory
            model_dir = self._get_model_cache_dir(model_name)
            
            # Remove directory and all contents
            import shutil
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Remove from registry
            del self.registry[model_name]
            self._save_registry()
            
            self.logger.info(f"Removed model from cache: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove model {model_name}: {e}")
            return False
    
    def clear_cache(self) -> int:
        """Clear all cached models. Returns number of models removed."""
        count = 0
        for model_name in list(self.registry.keys()):
            if self.remove_model(model_name):
                count += 1
        return count
    
    def get_cache_size(self) -> float:
        """Get total size of cache in MB."""
        total_size = 0
        for model_name, model_data in self.registry.items():
            try:
                path = Path(model_data["path"])
                if path.exists():
                    total_size += path.stat().st_size
            except:
                pass
        return total_size / (1024 * 1024)
    
    def auto_download_model(self, model_identifier: str, **kwargs) -> Tuple[Path, ModelInfo]:
        """
        Automatically download a model based on identifier patterns.
        
        This method attempts to determine the source and download the model automatically.
        
        Args:
            model_identifier: Model identifier that can be:
                - 'torchvision:model_name' (e.g., 'torchvision:resnet18')
                - 'huggingface:model_id' (e.g., 'huggingface:bert-base-uncased')
                - 'pytorch_hub:repo/model' (e.g., 'pytorch_hub:pytorch/vision/resnet50')
                - URL starting with 'http' or 'https'
                - Model name to search in available sources
            **kwargs: Additional arguments for download
            
        Returns:
            Tuple of (model_path, model_info)
        """
        if not self.config.get("auto_download", True):
            raise ValueError("Auto-download is disabled in configuration")
        
        self.logger.info(f"Auto-downloading model: {model_identifier}")
        
        # Parse identifier to determine source
        if model_identifier.startswith(("http://", "https://")):
            # URL download
            model_name = kwargs.get("model_name") or os.path.basename(urlparse(model_identifier).path).split('.')[0]
            return self.download_from_url(model_identifier, model_name, **kwargs)
            
        elif ":" in model_identifier:
            # Explicit source specification
            source, model_id = model_identifier.split(":", 1)
            
            if source == "torchvision":
                return self.download_torchvision_model(model_id, **kwargs)
            elif source == "huggingface":
                return self.download_huggingface_model(model_id, **kwargs)
            elif source == "pytorch_hub":
                # Split repo/model
                if "/" not in model_id:
                    raise ValueError("PyTorch Hub models require repo/model format")
                parts = model_id.rsplit("/", 1)
                repo = parts[0]
                model = parts[1]
                return self.download_pytorch_hub_model(repo, model, **kwargs)
            else:
                raise ValueError(f"Unknown source: {source}")
        else:
            # Try to auto-detect source
            return self._auto_detect_and_download(model_identifier, **kwargs)
    
    def _auto_detect_and_download(self, model_name: str, **kwargs) -> Tuple[Path, ModelInfo]:
        """
        Automatically detect source and download model.
        
        Tries sources in order: torchvision, pytorch_hub, huggingface
        """
        # First try torchvision
        if self._is_source_enabled("torchvision") and TORCHVISION_AVAILABLE:
            try:
                # Check if it's a valid torchvision model
                import torchvision.models as tv_models
                if hasattr(tv_models, model_name):
                    self.logger.info(f"Auto-detected torchvision model: {model_name}")
                    return self.download_torchvision_model(model_name, **kwargs)
            except Exception as e:
                self.logger.debug(f"Failed to download as torchvision model: {e}")
        
        # Try pytorch hub common patterns
        if self._is_source_enabled("pytorch_hub"):
            common_repos = [
                "pytorch/vision",
                "pytorch/fairseq", 
                "pytorch/audio",
                "facebookresearch/detr"
            ]
            
            for repo in common_repos:
                try:
                    self.logger.info(f"Trying PyTorch Hub: {repo}/{model_name}")
                    return self.download_pytorch_hub_model(repo, model_name, **kwargs)
                except Exception as e:
                    self.logger.debug(f"Failed to download from {repo}: {e}")
        
        # Finally try huggingface if it looks like a model identifier
        if self._is_source_enabled("huggingface") and TRANSFORMERS_AVAILABLE:
            try:
                self.logger.info(f"Trying Hugging Face model: {model_name}")
                return self.download_huggingface_model(model_name, **kwargs)
            except Exception as e:
                self.logger.debug(f"Failed to download as Hugging Face model: {e}")
        
        raise ValueError(f"Could not auto-detect source for model: {model_name}. "
                        f"Please specify source explicitly (e.g., 'torchvision:{model_name}')")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self.config
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update the configuration."""
        self.config.update(config)
        self.sources_config = self.config.get("sources", {})


# Global downloader instance
_global_downloader: Optional[ModelDownloader] = None


def get_model_downloader(cache_dir: Optional[Union[str, Path]] = None) -> ModelDownloader:
    """Get the global model downloader instance."""
    global _global_downloader
    if _global_downloader is None:
        _global_downloader = ModelDownloader(cache_dir)
    return _global_downloader


def download_model(
    source: str,
    model_id: str,
    model_name: Optional[str] = None,
    **kwargs
) -> Tuple[Path, ModelInfo]:
    """
    Convenient function to download a model from any source.
    
    Args:
        source: Source type ('pytorch_hub', 'torchvision', 'huggingface', 'url')
        model_id: Model identifier (depends on source)
        model_name: Custom name for the model
        **kwargs: Additional arguments specific to each source
        
    Returns:
        Tuple of (model_path, model_info)
    """
    downloader = get_model_downloader()
    
    if source == "pytorch_hub":
        # model_id should be "repo/model" format
        if "/" not in model_id:
            raise ValueError("PyTorch Hub model_id should be in format 'repo/model'")
        parts = model_id.split("/", 1)
        if ":" in parts[0]:
            repo = parts[0]
            model = parts[1]
        else:
            repo = parts[0]
            model = parts[1]
        return downloader.download_pytorch_hub_model(repo, model, model_name, **kwargs)
    
    elif source == "torchvision":
        return downloader.download_torchvision_model(model_id, custom_name=model_name, **kwargs)
    
    elif source == "huggingface":
        return downloader.download_huggingface_model(model_id, custom_name=model_name, **kwargs)
    
    elif source == "url":
        if model_name is None:
            model_name = os.path.basename(urlparse(model_id).path).split('.')[0]
        return downloader.download_from_url(model_id, model_name, **kwargs)
    
    else:
        raise ValueError(f"Unsupported source: {source}")


def list_available_models() -> Dict[str, ModelInfo]:
    """List all available (cached) models."""
    downloader = get_model_downloader()
    return downloader.list_available_models()


def auto_download_model(model_identifier: str, **kwargs) -> Tuple[Path, ModelInfo]:
    """
    Automatically download a model to the project models/ directory.
    
    This is a convenience function that uses the global model downloader
    with automatic source detection.
    
    Args:
        model_identifier: Model identifier. Can be:
            - 'torchvision:model_name' (e.g., 'torchvision:resnet18')
            - 'huggingface:model_id' (e.g., 'huggingface:bert-base-uncased')  
            - 'pytorch_hub:repo/model' (e.g., 'pytorch_hub:pytorch/vision/resnet50')
            - URL starting with 'http' or 'https'
            - Model name for auto-detection
        **kwargs: Additional arguments for download
        
    Returns:
        Tuple of (model_path, model_info)
        
    Examples:
        >>> # Download a torchvision model
        >>> path, info = auto_download_model("torchvision:resnet18")
        
        >>> # Download a Hugging Face model
        >>> path, info = auto_download_model("huggingface:bert-base-uncased")
        
        >>> # Auto-detect source
        >>> path, info = auto_download_model("resnet18")
        
        >>> # Download from URL
        >>> path, info = auto_download_model("https://example.com/model.pt", model_name="custom_model")
    """
    downloader = get_model_downloader()
    return downloader.auto_download_model(model_identifier, **kwargs)


def ensure_model_available(model_identifier: str, **kwargs) -> Tuple[Path, ModelInfo]:
    """
    Ensure a model is available, downloading it if necessary.
    
    This function checks if the model is already cached and downloads it
    if not available.
    
    Args:
        model_identifier: Model identifier (see auto_download_model)
        **kwargs: Additional arguments for download
        
    Returns:
        Tuple of (model_path, model_info)
    """
    downloader = get_model_downloader()
    
    # Try to extract model name for checking cache
    custom_name = kwargs.get('custom_name') or kwargs.get('model_name')
    
    # For simple cases, try to extract name from identifier
    if not custom_name:
        if ":" in model_identifier and not model_identifier.startswith(("http://", "https://")):
            source, model_id = model_identifier.split(":", 1)
            if source == "torchvision":
                custom_name = f"torchvision_{model_id}"
            elif source == "huggingface":
                custom_name = model_id.replace('/', '_')
            elif source == "pytorch_hub":
                custom_name = f"{model_id.replace('/', '_').replace(':', '_')}"
    
    # Check if already cached
    if custom_name and downloader.is_model_cached(custom_name):
        model_path = downloader.get_model_path(custom_name)
        model_info = downloader.get_model_info(custom_name)
        if model_path and model_info:
            return model_path, model_info
    
    # Download if not cached
    return auto_download_model(model_identifier, **kwargs)

#!/usr/bin/env python3
"""
Automatic model downloader for the torch-inference framework.

This script provides convenient functions to automatically download models
to the project's models/ directory based on configuration.
"""

import sys
import argparse
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the framework to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from framework.core.model_downloader import get_model_downloader, ModelInfo

logger = logging.getLogger(__name__)


def download_model_auto(
    model_identifier: str, 
    force: bool = False,
    **kwargs
) -> tuple[Path, ModelInfo]:
    """
    Automatically download a model to the project models/ directory.
    
    Args:
        model_identifier: Model identifier (see auto_download_model docs)
        force: Force re-download even if cached
        **kwargs: Additional download parameters
        
    Returns:
        Tuple of (model_path, model_info)
    """
    downloader = get_model_downloader()
    
    # Extract custom name if provided
    custom_name = kwargs.get('custom_name') or kwargs.get('model_name')
    
    # Check if already cached (unless forcing)
    if not force and custom_name and downloader.is_model_cached(custom_name):
        model_path = downloader.get_model_path(custom_name)
        model_info = downloader.get_model_info(custom_name)
        if model_path and model_info:
            print(f"✅ Model already cached: {custom_name}")
            print(f"   Path: {model_path}")
            return model_path, model_info
    
    # Download the model
    try:
        model_path, model_info = downloader.auto_download_model(model_identifier, **kwargs)
        print(f"✅ Successfully downloaded: {model_info.name}")
        print(f"   Path: {model_path}")
        print(f"   Size: {model_info.size_mb:.1f} MB")
        print(f"   Source: {model_info.source}")
        return model_path, model_info
    except Exception as e:
        print(f"❌ Failed to download model '{model_identifier}': {e}")
        raise


def download_multiple_models(
    model_identifiers: List[str],
    force: bool = False,
    continue_on_error: bool = True
) -> Dict[str, tuple[Optional[Path], Optional[ModelInfo], Optional[str]]]:
    """
    Download multiple models automatically.
    
    Args:
        model_identifiers: List of model identifiers
        force: Force re-download even if cached
        continue_on_error: Continue downloading other models if one fails
        
    Returns:
        Dict mapping model_identifier to (path, info, error) tuple
    """
    results = {}
    
    for identifier in model_identifiers:
        try:
            model_path, model_info = download_model_auto(identifier, force=force)
            results[identifier] = (model_path, model_info, None)
        except Exception as e:
            error_msg = str(e)
            results[identifier] = (None, None, error_msg)
            
            if continue_on_error:
                print(f"⚠️  Continuing with next model after error...")
                continue
            else:
                print(f"❌ Stopping due to error")
                break
    
    return results


def download_preset_models(preset: str = "basic") -> Dict[str, tuple[Optional[Path], Optional[ModelInfo], Optional[str]]]:
    """
    Download a preset collection of models.
    
    Args:
        preset: Preset name ('basic', 'vision', 'nlp', 'all')
        
    Returns:
        Results dictionary
    """
    presets = {
        "basic": [
            "torchvision:resnet18",
            "torchvision:mobilenet_v2"
        ],
        "vision": [
            "torchvision:resnet18",
            "torchvision:resnet50",
            "torchvision:mobilenet_v2",
            "torchvision:efficientnet_b0",
            "torchvision:vgg16"
        ],
        "nlp": [
            "huggingface:bert-base-uncased",
            "huggingface:distilbert-base-uncased",
            "huggingface:roberta-base"
        ],
        "all": [
            "torchvision:resnet18",
            "torchvision:resnet50",
            "torchvision:mobilenet_v2",
            "huggingface:bert-base-uncased",
            "huggingface:distilbert-base-uncased"
        ]
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    print(f"📦 Downloading preset '{preset}' models...")
    return download_multiple_models(presets[preset])


def main():
    """CLI interface for automatic model downloading."""
    parser = argparse.ArgumentParser(
        description="Automatically download models to the project models/ directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a torchvision model
  python -m framework.scripts.auto_download torchvision:resnet18
  
  # Download a Hugging Face model
  python -m framework.scripts.auto_download huggingface:bert-base-uncased
  
  # Download from URL
  python -m framework.scripts.auto_download https://example.com/model.pt
  
  # Download preset collection
  python -m framework.scripts.auto_download --preset basic
  
  # Download multiple models
  python -m framework.scripts.auto_download torchvision:resnet18 torchvision:mobilenet_v2
  
  # Force re-download
  python -m framework.scripts.auto_download torchvision:resnet18 --force
        """
    )
    
    parser.add_argument(
        'models', 
        nargs='*',
        help='Model identifiers to download'
    )
    
    parser.add_argument(
        '--preset',
        choices=['basic', 'vision', 'nlp', 'all'],
        help='Download a preset collection of models'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if cached'
    )
    
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop on first error instead of continuing'
    )
    
    parser.add_argument(
        '--list-config',
        action='store_true',
        help='Show current configuration'
    )
    
    args = parser.parse_args()
    
    if args.list_config:
        downloader = get_model_downloader()
        config = downloader.get_config()
        print("📋 Current Model Download Configuration:")
        print("=" * 50)
        print(f"Cache directory: {downloader.cache_dir}")
        print(f"Auto-download enabled: {config.get('auto_download', True)}")
        print(f"Registry file: {downloader.registry_file}")
        print("\nEnabled sources:")
        sources = config.get('sources', {})
        for source, source_config in sources.items():
            enabled = source_config.get('enabled', True)
            status = "✅" if enabled else "❌"
            print(f"  {status} {source}")
        return 0
    
    if args.preset:
        if args.models:
            print("⚠️  Cannot specify both --preset and individual models")
            return 1
        
        try:
            results = download_preset_models(args.preset)
            
            # Print summary
            successful = sum(1 for _, _, error in results.values() if error is None)
            failed = len(results) - successful
            
            print(f"\n📊 Download Summary:")
            print(f"   ✅ Successful: {successful}")
            print(f"   ❌ Failed: {failed}")
            
            if failed > 0:
                print(f"\n❌ Failed downloads:")
                for identifier, (_, _, error) in results.items():
                    if error:
                        print(f"   • {identifier}: {error}")
                        
            return 1 if failed > 0 else 0
            
        except Exception as e:
            print(f"❌ Failed to download preset: {e}")
            return 1
    
    if not args.models:
        parser.print_help()
        return 1
    
    try:
        if len(args.models) == 1:
            # Single model download
            download_model_auto(args.models[0], force=args.force)
        else:
            # Multiple model download
            results = download_multiple_models(
                args.models, 
                force=args.force, 
                continue_on_error=not args.stop_on_error
            )
            
            # Print summary
            successful = sum(1 for _, _, error in results.values() if error is None)
            failed = len(results) - successful
            
            print(f"\n📊 Download Summary:")
            print(f"   ✅ Successful: {successful}")
            print(f"   ❌ Failed: {failed}")
            
            if failed > 0:
                print(f"\n❌ Failed downloads:")
                for identifier, (_, _, error) in results.items():
                    if error:
                        print(f"   • {identifier}: {error}")
                        
            return 1 if failed > 0 else 0
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


class AutoDownloader:
    """
    Automatic model downloader with intelligent source detection.
    """
    
    def __init__(self):
        self.downloader = get_model_downloader()
    
    def auto_download(self, model_identifier: str, 
                     model_name: Optional[str] = None,
                     **kwargs) -> tuple[Path, ModelInfo]:
        """
        Automatically download a model with source detection.
        
        Args:
            model_identifier: Model identifier
            model_name: Custom name for the model
            **kwargs: Additional arguments for downloading
            
        Returns:
            Tuple of (model_path, model_info)
        """
        return download_model_auto(model_identifier, **kwargs)
    
    def suggest_alternatives(self, model_identifier: str) -> List[str]:
        """
        Suggest alternative model identifiers if download fails.
        
        Args:
            model_identifier: Original model identifier
            
        Returns:
            List of suggested alternatives
        """
        suggestions = []
        
        # Generate suggestions for different sources
        suggestions.extend([
            f"torchvision:{model_identifier}",
            f"huggingface:{model_identifier}",
            f"pytorch:{model_identifier}",
        ])
        
        # Common model name variations
        if '_' in model_identifier:
            suggestions.append(model_identifier.replace('_', '-'))
        if '-' in model_identifier:
            suggestions.append(model_identifier.replace('-', '_'))
        
        # Remove duplicates and original
        suggestions = list(set(suggestions))
        if model_identifier in suggestions:
            suggestions.remove(model_identifier)
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def validate_identifier(self, model_identifier: str) -> bool:
        """
        Validate if a model identifier is potentially valid.
        
        Args:
            model_identifier: Model identifier to validate
            
        Returns:
            True if identifier appears valid
        """
        if not model_identifier or not isinstance(model_identifier, str):
            return False
        
        # Basic validation
        if len(model_identifier.strip()) < 3:
            return False
        
        return True


class SourceDetector:
    """Detect the source of a model identifier."""
    
    def __init__(self):
        self.source_patterns = {
            'huggingface': [
                r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$',  # org/model-name
                r'^huggingface:',
                r'^hf:',
            ],
            'torchvision': [
                r'^(resnet|alexnet|vgg|squeezenet|densenet|inception|googlenet|shufflenet|mobilenet|resnext|wide_resnet|mnasnet|efficientnet|regnet|convnext)',
                r'^torchvision:',
                r'^tv:',
            ],
            'pytorch_hub': [
                r'^[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+$',  # repo:model:version
                r'^pytorch:',
                r'^hub:',
            ],
            'url': [
                r'^https?://',
                r'^ftp://',
            ],
            'file': [
                r'^[/\\]',  # Unix and Windows absolute paths
                r'^\w:',  # Windows drive paths
                r'^\.',  # Relative paths starting with .
                r'\.pth$',
                r'\.pt$',
                r'\.onnx$',
            ]
        }
    
    def detect_source(self, model_identifier: str) -> str:
        """Detect the source of a model identifier."""
        import re
        
        model_id = model_identifier.lower().strip()
        
        # Check file patterns first (most specific)
        if any(re.match(pattern, model_id) for pattern in self.source_patterns['file']):
            return 'file'
        
        # Check URL patterns
        if any(re.match(pattern, model_id) for pattern in self.source_patterns['url']):
            return 'url'
        
        # Check other source patterns
        for source, patterns in self.source_patterns.items():
            if source in ['file', 'url']:  # Already checked
                continue
            for pattern in patterns:
                if re.match(pattern, model_id):
                    return source
        
        # Default fallback
        return 'pytorch_hub'


class ModelRegistry:
    """Registry for downloaded models."""
    
    def __init__(self, registry_file: str = None):
        self.registry_file = registry_file or Path.home() / '.torch_inference' / 'model_registry.json'
        self.registry_file = Path(self.registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}
    
    def _save_registry(self):
        """Save registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self._registry, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save model registry: {e}")
    
    def register_model(self, model_id: str, model_path: str, metadata: Dict[str, Any] = None):
        """Register a downloaded model."""
        self._registry[model_id] = {
            'path': str(model_path),
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        self._save_registry()
    
    def get_model_path(self, model_id: str) -> Optional[str]:
        """Get path for a registered model."""
        if model_id in self._registry:
            return self._registry[model_id]['path']
        return None
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self._registry.keys())
    
    def remove_model(self, model_id: str):
        """Remove a model from registry."""
        if model_id in self._registry:
            del self._registry[model_id]
            self._save_registry()
    
    def clear(self):
        """Clear all models from registry."""
        self._registry.clear()
        self._save_registry()


# Global registry instance
_model_registry = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


# Convenience functions
def auto_download_model(model_identifier: str, **kwargs) -> str:
    """Auto-download a model based on identifier."""
    downloader = AutoDownloader()
    path, info = downloader.auto_download(model_identifier, **kwargs)
    return model_identifier  # Return the identifier for backward compatibility


def detect_model_source(model_identifier: str) -> str:
    """Detect the source of a model identifier."""
    detector = SourceDetector()
    return detector.detect_source(model_identifier)


def register_model(model_id: str, model_path: str, metadata: Dict[str, Any] = None):
    """Register a downloaded model."""
    registry = get_model_registry()
    registry.register_model(model_id, model_path, metadata)


def get_registered_models() -> List[str]:
    """Get list of registered models."""
    registry = get_model_registry()
    return registry.list_models()


def cleanup_downloaded_models():
    """Clean up downloaded models."""
    registry = get_model_registry()
    registry.clear()

"""
Command line interface for downloading PyTorch models.

This script provides a convenient CLI for downloading models from various sources
and managing the local model cache.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add the framework to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from framework.core.model_downloader import (
    get_model_downloader, 
    download_model,
    list_available_models,
    auto_download_model,
    ModelInfo
)


def parse_model_identifier(model_identifier: str) -> dict:
    """
    Parse a model identifier string into its components.
    
    Args:
        model_identifier: Model identifier in format 'source:model_id' or just 'model_id'
        
    Returns:
        Dictionary with parsed components
    """
    if ':' in model_identifier:
        source, model_id = model_identifier.split(':', 1)
        return {
            'source': source.strip(),
            'model_id': model_id.strip(),
            'full_identifier': model_identifier
        }
    else:
        return {
            'source': 'auto',  # Auto-detect source
            'model_id': model_identifier.strip(),
            'full_identifier': model_identifier
        }


def download_command(args):
    """Handle model download command."""
    try:
        print(f"📥 Downloading model: {args.model_id}")
        print(f"   Source: {args.source}")
        
        # Prepare kwargs based on source
        kwargs = {}
        if hasattr(args, 'pretrained'):
            kwargs['pretrained'] = args.pretrained
        
        # Only pass task for sources that support it
        if args.source in ['huggingface', 'url'] and hasattr(args, 'task'):
            kwargs['task'] = args.task
        
        # Download model
        model_path, model_info = download_model(
            source=args.source,
            model_id=args.model_id,
            model_name=args.name,
            **kwargs
        )
        
        print(f"✅ Successfully downloaded model:")
        print(f"   Name: {model_info.name}")
        print(f"   Path: {model_path}")
        print(f"   Size: {model_info.size_mb:.1f} MB")
        print(f"   Task: {model_info.task}")
        print(f"   Description: {model_info.description}")
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return 1
    
    return 0


def auto_download_command(args):
    """Handle auto model download command."""
    try:
        print(f"📥 Auto-downloading model: {args.model_identifier}")
        
        # Prepare kwargs
        kwargs = {}
        if hasattr(args, 'task') and args.task:
            kwargs['task'] = args.task
        if hasattr(args, 'name') and args.name:
            kwargs['custom_name'] = args.name
        
        # Download model
        model_path, model_info = auto_download_model(args.model_identifier, **kwargs)
        
        print(f"✅ Successfully downloaded model:")
        print(f"   Name: {model_info.name}")
        print(f"   Path: {model_path}")
        print(f"   Size: {model_info.size_mb:.1f} MB")
        print(f"   Source: {model_info.source}")
        print(f"   Task: {model_info.task}")
        print(f"   Description: {model_info.description}")
        
    except Exception as e:
        print(f"❌ Failed to auto-download model: {e}")
        return 1
    
    return 0


def list_command(args):
    """Handle list models command."""
    try:
        models = list_available_models()
        
        if not models:
            print("No models found in cache.")
            return 0
        
        print("📚 Available Models:")
        print("=" * 80)
        
        total_size = 0
        for name, info in models.items():
            print(f"📦 {name}")
            print(f"   Source: {info.source}")
            print(f"   Task: {info.task}")
            print(f"   Size: {info.size_mb:.1f} MB")
            print(f"   Description: {info.description}")
            if info.tags:
                print(f"   Tags: {', '.join(info.tags)}")
            print()
            total_size += info.size_mb or 0
        
        print(f"Total models: {len(models)}")
        print(f"Total cache size: {total_size:.1f} MB")
        
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        return 1
    
    return 0


def info_command(args):
    """Handle model info command."""
    try:
        downloader = get_model_downloader()
        info = downloader.get_model_info(args.name)
        
        if info is None:
            print(f"❌ Model not found: {args.name}")
            return 1
        
        print(f"📋 Model Information: {args.name}")
        print("=" * 50)
        print(f"Name: {info.name}")
        print(f"Source: {info.source}")
        print(f"Model ID: {info.model_id}")
        print(f"Task: {info.task}")
        print(f"Size: {info.size_mb:.1f} MB")
        print(f"Description: {info.description}")
        if info.license:
            print(f"License: {info.license}")
        if info.tags:
            print(f"Tags: {', '.join(info.tags)}")
        
        # Show path
        model_path = downloader.get_model_path(args.name)
        if model_path:
            print(f"Path: {model_path}")
        
    except Exception as e:
        print(f"❌ Failed to get model info: {e}")
        return 1
    
    return 0


def remove_command(args):
    """Handle remove model command."""
    try:
        downloader = get_model_downloader()
        
        if args.name == "all":
            if input("⚠️  Remove ALL cached models? (y/N): ").lower() != 'y':
                print("Cancelled.")
                return 0
            
            count = downloader.clear_cache()
            print(f"✅ Removed {count} models from cache.")
        else:
            if not downloader.is_model_cached(args.name):
                print(f"❌ Model not found: {args.name}")
                return 1
            
            if downloader.remove_model(args.name):
                print(f"✅ Removed model: {args.name}")
            else:
                print(f"❌ Failed to remove model: {args.name}")
                return 1
        
    except Exception as e:
        print(f"❌ Failed to remove model: {e}")
        return 1
    
    return 0


def clean_command(args):
    """Handle cache cleaning command."""
    try:
        downloader = get_model_downloader()
        
        if args.force:
            # Clear all cache
            print("🧹 Cleaning all cached models...")
            cleared_count = downloader.clear_cache()
            print(f"✅ Cleared {cleared_count} cached models")
        else:
            # Clean only unused/old models
            print("🧹 Cleaning unused cached models...")
            
            # Get cache stats before cleaning
            original_size = downloader.get_cache_size()
            original_count = len(downloader.registry)
            
            # Perform selective cleaning (this is a simplified version)
            # In a real implementation, you'd want more sophisticated logic
            cleared_count = downloader.clean_old_models() if hasattr(downloader, 'clean_old_models') else 0
            
            # Get cache stats after cleaning
            new_size = downloader.get_cache_size()
            new_count = len(downloader.registry)
            
            print(f"✅ Cleaned cache:")
            print(f"   Models: {original_count} → {new_count} ({original_count - new_count} removed)")
            print(f"   Size: {original_size:.1f} MB → {new_size:.1f} MB ({original_size - new_size:.1f} MB freed)")
        
    except Exception as e:
        print(f"❌ Failed to clean cache: {e}")
        return 1
    
    return 0


def cache_command(args):
    """Handle cache management command."""
    try:
        downloader = get_model_downloader()
        
        print("💾 Cache Information:")
        print("=" * 30)
        print(f"Cache directory: {downloader.cache_dir}")
        print(f"Total models: {len(downloader.registry)}")
        print(f"Total size: {downloader.get_cache_size():.1f} MB")
        
    except Exception as e:
        print(f"❌ Failed to get cache info: {e}")
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download and manage PyTorch models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a torchvision model
  python -m framework.scripts.download_models download torchvision resnet18
  
  # Download a PyTorch Hub model
  python -m framework.scripts.download_models download pytorch_hub "pytorch/vision:v0.10.0/resnet50"
  
  # Download a Hugging Face model
  python -m framework.scripts.download_models download huggingface bert-base-uncased --task text-classification
  
  # Download from URL
  python -m framework.scripts.download_models download url "https://example.com/model.pt" --name my_model
  
  # Auto-download with source detection
  python -m framework.scripts.download_models auto torchvision:resnet18
  python -m framework.scripts.download_models auto huggingface:bert-base-uncased
  python -m framework.scripts.download_models auto resnet18  # Auto-detects torchvision
  
  # List all models
  python -m framework.scripts.download_models list
  
  # Get model info
  python -m framework.scripts.download_models info resnet18
  
  # Remove a model
  python -m framework.scripts.download_models remove resnet18
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download a model')
    download_parser.add_argument('source', choices=['pytorch_hub', 'torchvision', 'huggingface', 'url'],
                                help='Model source')
    download_parser.add_argument('model_id', help='Model identifier')
    download_parser.add_argument('--name', help='Custom name for the model')
    download_parser.add_argument('--pretrained', action='store_true', default=True,
                                help='Download pretrained weights (default: True)')
    download_parser.add_argument('--task', default='classification',
                                help='Task type (default: classification)')
    download_parser.set_defaults(func=download_command)
    
    # Auto-download command
    auto_parser = subparsers.add_parser('auto', help='Auto-download a model with source detection')
    auto_parser.add_argument('model_identifier', 
                            help='Model identifier (torchvision:resnet18, huggingface:bert-base-uncased, etc.)')
    auto_parser.add_argument('--name', help='Custom name for the model')
    auto_parser.add_argument('--task', help='Task type')
    auto_parser.set_defaults(func=auto_download_command)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available models')
    list_parser.set_defaults(func=list_command)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get model information')
    info_parser.add_argument('name', help='Model name')
    info_parser.set_defaults(func=info_command)
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove a model from cache')
    remove_parser.add_argument('name', help='Model name (use "all" to remove all models)')
    remove_parser.set_defaults(func=remove_command)
    
    # Cache command
    cache_parser = subparsers.add_parser('cache', help='Show cache information')
    cache_parser.set_defaults(func=cache_command)
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean cached models')
    clean_parser.add_argument('--force', action='store_true', 
                             help='Force clean all cached models (default: clean only unused)')
    clean_parser.set_defaults(func=clean_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    # Execute command
    return args.func(args)


class ModelDownloadScript:
    """Script wrapper for model downloading functionality."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the model download script.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = cache_dir
        from framework.core.model_downloader import get_model_downloader
        self.downloader = get_model_downloader(cache_dir)
    
    def download_model(self, model_name: str, **kwargs) -> Path:
        """Download a model and return its path."""
        return download_model(model_name, **kwargs)
    
    def list_models(self) -> dict:
        """List all available models."""
        return list_available_models()
    
    def run_cli(self, args: Optional[list] = None) -> int:
        """Run the CLI interface."""
        import sys
        if args:
            sys.argv = ["download_models"] + args
        return main()


if __name__ == "__main__":
    exit(main())

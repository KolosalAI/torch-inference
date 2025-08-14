"""
Example demonstrating the model downloader functionality.

This script shows how to download models from various sources and use them
with the PyTorch inference framework.
"""

import sys
from pathlib import Path

# Add the framework to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from framework.core.model_downloader import get_model_downloader, download_model
from framework.core.config import get_global_config
from framework.core.base_model import get_model_manager


def download_torchvision_models():
    """Download some popular torchvision models."""
    print("🔥 Downloading Torchvision Models")
    print("=" * 40)
    
    models_to_download = [
        ("resnet18", "ResNet-18 - Lightweight residual network"),
        ("mobilenet_v2", "MobileNet v2 - Mobile-optimized model"),
        ("efficientnet_b0", "EfficientNet B0 - Efficient architecture")
    ]
    
    for model_name, description in models_to_download:
        try:
            print(f"\n📥 Downloading {model_name}...")
            print(f"   Description: {description}")
            
            model_path, model_info = download_model(
                source="torchvision",
                model_id=model_name,
                pretrained=True
            )
            
            print(f"   ✅ Downloaded successfully!")
            print(f"   📁 Path: {model_path}")
            print(f"   📏 Size: {model_info.size_mb:.1f} MB")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def download_pytorch_hub_models():
    """Download some PyTorch Hub models."""
    print("\n🚀 Downloading PyTorch Hub Models")
    print("=" * 40)
    
    models_to_download = [
        ("pytorch/vision", "resnet50", "ResNet-50 from PyTorch Hub"),
        ("pytorch/vision", "vgg16", "VGG-16 from PyTorch Hub")
    ]
    
    for repo, model, description in models_to_download:
        try:
            print(f"\n📥 Downloading {repo}/{model}...")
            print(f"   Description: {description}")
            
            model_path, model_info = download_model(
                source="pytorch_hub", 
                model_id=f"{repo}/{model}",
                pretrained=True
            )
            
            print(f"   ✅ Downloaded successfully!")
            print(f"   📁 Path: {model_path}")
            print(f"   📏 Size: {model_info.size_mb:.1f} MB")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def download_huggingface_models():
    """Download some Hugging Face models (if transformers is available)."""
    try:
        from transformers import AutoModel
        transformers_available = True
    except ImportError:
        transformers_available = False
    
    if not transformers_available:
        print("\n🤗 Hugging Face Models - Skipped (transformers not installed)")
        print("   Install with: pip install transformers")
        return
    
    print("\n🤗 Downloading Hugging Face Models")
    print("=" * 40)
    
    models_to_download = [
        ("distilbert-base-uncased", "text-classification", "DistilBERT for text classification"),
        ("microsoft/DialoGPT-small", "text-generation", "DialoGPT for conversation")
    ]
    
    for model_id, task, description in models_to_download:
        try:
            print(f"\n📥 Downloading {model_id}...")
            print(f"   Description: {description}")
            print(f"   Task: {task}")
            
            model_path, model_info = download_model(
                source="huggingface",
                model_id=model_id,
                task=task
            )
            
            print(f"   ✅ Downloaded successfully!")
            print(f"   📁 Path: {model_path}")
            print(f"   📏 Size: {model_info.size_mb:.1f} MB")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def demonstrate_model_loading():
    """Demonstrate loading a downloaded model into the framework."""
    print("\n⚡ Loading Models into Framework")
    print("=" * 40)
    
    try:
        # Get model manager
        model_manager = get_model_manager()
        
        # Try to download and load a torchvision model directly
        print("📥 Downloading and loading ResNet-18...")
        
        model_manager.download_and_load_model(
            source="torchvision",
            model_id="resnet18", 
            name="resnet18_example"
        )
        
        print("✅ Model loaded successfully!")
        
        # List loaded models
        loaded_models = model_manager.list_models()
        print(f"📚 Loaded models: {loaded_models}")
        
        # Get model and run a prediction
        if "resnet18_example" in loaded_models:
            model = model_manager.get_model("resnet18_example")
            print(f"📋 Model info: {model.model_info}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")


def list_cached_models():
    """List all cached models."""
    print("\n📚 Cached Models Summary")
    print("=" * 40)
    
    try:
        downloader = get_model_downloader()
        models = downloader.list_available_models()
        
        if not models:
            print("No models in cache.")
            return
        
        total_size = 0
        for name, info in models.items():
            print(f"📦 {name}")
            print(f"   Source: {info.source}")
            print(f"   Task: {info.task}")
            print(f"   Size: {info.size_mb:.1f} MB")
            if info.tags:
                print(f"   Tags: {', '.join(info.tags[:3])}{'...' if len(info.tags) > 3 else ''}")
            total_size += info.size_mb or 0
            print()
        
        print(f"Total: {len(models)} models, {total_size:.1f} MB")
        
    except Exception as e:
        print(f"❌ Failed to list models: {e}")


def main():
    """Main example function."""
    print("🧠 PyTorch Model Downloader Example")
    print("=" * 50)
    print("This example demonstrates downloading models from various sources.")
    print()
    
    # Download from different sources
    download_torchvision_models()
    download_pytorch_hub_models()
    download_huggingface_models()
    
    # Demonstrate model loading
    demonstrate_model_loading()
    
    # Show cached models summary
    list_cached_models()
    
    print("\n🎉 Example completed!")
    print("You can now use these models with the PyTorch inference framework.")
    print("\nTip: Use the CLI tool to manage models:")
    print("  python -m framework.scripts.download_models list")


if __name__ == "__main__":
    main()

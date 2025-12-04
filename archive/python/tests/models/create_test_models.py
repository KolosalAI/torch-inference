#!/usr/bin/env python3
"""
Script to download real models for testing the torch-inference framework.

This script downloads lightweight, well-known models from Hugging Face and 
other platforms for use in testing. The models are chosen to be:
1. Small in size for fast CI/CD
2. Representative of common model types
3. Publicly available and license-friendly
4. Cover different architectures and use cases
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as tv_models
from pathlib import Path
from typing import Dict, Any, Optional
import json
import requests
from urllib.parse import urlparse
import hashlib

# Add the framework to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
        AutoModelForImageClassification, AutoConfig
    )
    HF_AVAILABLE = True
except ImportError:
    print("Transformers not available. Install with: pip install transformers")
    HF_AVAILABLE = False

try:
    from torchvision import models as tv_models
    TORCHVISION_AVAILABLE = True
except ImportError:
    print("Torchvision not available. Install with: pip install torchvision")
    TORCHVISION_AVAILABLE = False


class ModelDownloader:
    """Handles downloading and caching of test models."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True)
        
        # Model registry
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.registry_file = models_dir / "model_registry.json"
        
        # Load existing registry
        self.load_registry()
    
    def load_registry(self):
        """Load model registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.model_registry = json.load(f)
    
    def save_registry(self):
        """Save model registry to disk."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.model_registry, f, indent=2)
    
    def register_model(self, model_id: str, model_info: Dict[str, Any]):
        """Register a model in the registry."""
        self.model_registry[model_id] = model_info
        self.save_registry()
    
    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Get path to a registered model."""
        if model_id in self.model_registry:
            path = Path(self.model_registry[model_id]["path"])
            if path.exists():
                return path
        return None
    
    def download_huggingface_model(self, model_name: str, model_id: str, 
                                 task: str = "feature-extraction") -> Path:
        """Download a model from Hugging Face."""
        if not HF_AVAILABLE:
            raise ImportError("transformers library required for Hugging Face models")
        
        print(f"Downloading HuggingFace model: {model_name}")
        
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)
        
        try:
            # Choose appropriate AutoModel class based on task
            if task == "text-classification":
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, cache_dir=model_dir / "cache"
                )
            elif task == "image-classification":
                model = AutoModelForImageClassification.from_pretrained(
                    model_name, cache_dir=model_dir / "cache"
                )
            else:
                model = AutoModel.from_pretrained(
                    model_name, cache_dir=model_dir / "cache"
                )
            
            # Save the model
            model_path = model_dir / "pytorch_model.pt"
            torch.save(model.state_dict(), model_path)
            
            # Also save the full model for easier loading
            full_model_path = model_dir / "model.pt" 
            torch.save(model, full_model_path)
            
            # Save config
            config = AutoConfig.from_pretrained(model_name)
            config_path = model_dir / "config.json"
            config.save_pretrained(model_dir)
            
            # Try to get tokenizer if available
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.save_pretrained(model_dir)
            except:
                pass
            
            # Register model
            model_info = {
                "path": str(model_path),
                "full_model_path": str(full_model_path),
                "config_path": str(config_path),
                "source": "huggingface",
                "model_name": model_name,
                "task": task,
                "size_mb": model_path.stat().st_size / (1024 * 1024),
                "architecture": config.architectures[0] if hasattr(config, 'architectures') and config.architectures else "unknown"
            }
            
            self.register_model(model_id, model_info)
            
            print(f"✓ Downloaded {model_name} -> {model_path}")
            return model_path
            
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
            raise
    
    def download_torchvision_model(self, model_name: str, model_id: str, 
                                 pretrained: bool = True) -> Path:
        """Download a model from torchvision."""
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision library required for torchvision models")
        
        print(f"Downloading torchvision model: {model_name}")
        
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)
        
        try:
            # Get model from torchvision
            model_fn = getattr(tv_models, model_name)
            model = model_fn(pretrained=pretrained)
            
            # Save the model
            model_path = model_dir / "model.pt"
            torch.save(model, model_path)
            
            # Save state dict separately
            state_dict_path = model_dir / "pytorch_model.pt"
            torch.save(model.state_dict(), state_dict_path)
            
            # Create a simple config
            config = {
                "model_name": model_name,
                "architecture": model.__class__.__name__,
                "pretrained": pretrained,
                "num_classes": getattr(model, 'num_classes', getattr(model, 'fc', {}).out_features if hasattr(getattr(model, 'fc', {}), 'out_features') else 1000)
            }
            
            config_path = model_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Register model
            model_info = {
                "path": str(state_dict_path),
                "full_model_path": str(model_path),
                "config_path": str(config_path),
                "source": "torchvision",
                "model_name": model_name,
                "task": "image-classification",
                "size_mb": model_path.stat().st_size / (1024 * 1024),
                "architecture": model.__class__.__name__,
                "pretrained": pretrained
            }
            
            self.register_model(model_id, model_info)
            
            print(f"✓ Downloaded {model_name} -> {model_path}")
            return model_path
            
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
            raise
    
    def create_custom_model(self, model_id: str, model_type: str) -> Path:
        """Create custom models for specific test scenarios."""
        print(f"Creating custom {model_type} model: {model_id}")
        
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)
        
        if model_type == "simple_cnn":
            model = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 10)
            )
        
        elif model_type == "simple_transformer":
            model = nn.Sequential(
                nn.Embedding(1000, 128),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True),
                    num_layers=2
                ),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(128, 2)
            )
        
        elif model_type == "simple_rnn":
            model = nn.Sequential(
                nn.Embedding(1000, 64),
                nn.LSTM(64, 32, batch_first=True),
                nn.Lambda(lambda x: x[0][:, -1, :]),  # Take last output
                nn.Linear(32, 2)
            )
        
        else:  # simple_linear
            model = nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
        
        # Save the model
        model_path = model_dir / "model.pt"
        torch.save(model, model_path)
        
        # Save state dict
        state_dict_path = model_dir / "pytorch_model.pt"
        torch.save(model.state_dict(), state_dict_path)
        
        # Create config
        config = {
            "model_type": model_type,
            "architecture": "Custom",
            "parameters": sum(p.numel() for p in model.parameters())
        }
        
        config_path = model_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Register model
        model_info = {
            "path": str(state_dict_path),
            "full_model_path": str(model_path),
            "config_path": str(config_path),
            "source": "custom",
            "model_name": model_id,
            "task": "classification",
            "size_mb": model_path.stat().st_size / (1024 * 1024),
            "architecture": "Custom",
            "model_type": model_type
        }
        
        self.register_model(model_id, model_info)
        
        print(f"✓ Created {model_type} -> {model_path}")
        return model_path


def download_all_test_models(models_dir: Path):
    """Download all models needed for testing."""
    downloader = ModelDownloader(models_dir)
    
    print("🚀 Starting model downloads...")
    print("=" * 50)
    
    models_to_download = []
    
    # Hugging Face Models (small, fast models)
    if HF_AVAILABLE:
        hf_models = [
            {
                "name": "distilbert-base-uncased-finetuned-sst-2-english",
                "id": "distilbert_sentiment",
                "task": "text-classification"
            },
            {
                "name": "microsoft/DialoGPT-small", 
                "id": "dialogpt_small",
                "task": "text-generation"
            },
            {
                "name": "google/vit-base-patch16-224",
                "id": "vit_base", 
                "task": "image-classification"
            },
            {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "id": "sentence_transformer",
                "task": "feature-extraction"
            }
        ]
        
        print(f"\n📥 Downloading {len(hf_models)} Hugging Face models...")
        for model_info in hf_models:
            try:
                if not downloader.get_model_path(model_info["id"]):
                    downloader.download_huggingface_model(
                        model_info["name"], 
                        model_info["id"], 
                        model_info["task"]
                    )
                    models_to_download.append(model_info["id"])
                else:
                    print(f"✓ {model_info['id']} already exists")
            except Exception as e:
                print(f"⚠️  Skipping {model_info['name']}: {e}")
    
    # Torchvision Models (lightweight versions)
    if TORCHVISION_AVAILABLE:
        tv_models_list = [
            {
                "name": "mobilenet_v2",
                "id": "mobilenet_v2"
            },
            {
                "name": "resnet18", 
                "id": "resnet18"
            },
            {
                "name": "efficientnet_b0",
                "id": "efficientnet_b0"  
            }
        ]
        
        print(f"\n📥 Downloading {len(tv_models_list)} torchvision models...")
        for model_info in tv_models_list:
            try:
                if not downloader.get_model_path(model_info["id"]):
                    downloader.download_torchvision_model(
                        model_info["name"],
                        model_info["id"],
                        pretrained=True
                    )
                    models_to_download.append(model_info["id"])
                else:
                    print(f"✓ {model_info['id']} already exists")
            except Exception as e:
                print(f"⚠️  Skipping {model_info['name']}: {e}")
    
    # Custom Models for specific test scenarios
    custom_models = [
        {"id": "simple_cnn", "type": "simple_cnn"},
        {"id": "simple_transformer", "type": "simple_transformer"},
        {"id": "simple_rnn", "type": "simple_rnn"},
        {"id": "simple_linear", "type": "simple_linear"}
    ]
    
    print(f"\n🛠️  Creating {len(custom_models)} custom models...")
    for model_info in custom_models:
        try:
            if not downloader.get_model_path(model_info["id"]):
                downloader.create_custom_model(
                    model_info["id"],
                    model_info["type"]
                )
                models_to_download.append(model_info["id"])
            else:
                print(f"✓ {model_info['id']} already exists")
        except Exception as e:
            print(f"⚠️  Failed to create {model_info['id']}: {e}")
    
    print("\n" + "=" * 50)
    print("📊 Model Download Summary:")
    print("=" * 50)
    
    # Print registry summary
    total_size_mb = 0
    for model_id, model_info in downloader.model_registry.items():
        size_mb = model_info.get("size_mb", 0)
        total_size_mb += size_mb
        status = "✓ Downloaded" if model_id in models_to_download else "✓ Existing"
        print(f"{status:<15} {model_id:<25} ({size_mb:.1f} MB) - {model_info['source']}")
    
    print(f"\n📈 Total models: {len(downloader.model_registry)}")
    print(f"💾 Total size: {total_size_mb:.1f} MB")
    print(f"📁 Models directory: {models_dir}")
    
    # Create a summary file
    summary = {
        "total_models": len(downloader.model_registry),
        "total_size_mb": total_size_mb,
        "models_by_source": {},
        "download_timestamp": str(torch.utils.data.get_worker_info())
    }
    
    for model_info in downloader.model_registry.values():
        source = model_info["source"]
        if source not in summary["models_by_source"]:
            summary["models_by_source"][source] = 0
        summary["models_by_source"][source] += 1
    
    summary_file = models_dir / "download_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Summary saved to: {summary_file}")
    
    return downloader


def verify_models(models_dir: Path):
    """Verify that downloaded models can be loaded."""
    print("\n🔍 Verifying downloaded models...")
    
    registry_file = models_dir / "model_registry.json"
    if not registry_file.exists():
        print("❌ No model registry found")
        return False
    
    with open(registry_file, 'r') as f:
        registry = json.load(f)
    
    verified_count = 0
    failed_count = 0
    
    for model_id, model_info in registry.items():
        try:
            model_path = Path(model_info["full_model_path"])
            if model_path.exists():
                # Try to load the model (use weights_only=False for test models)
                model = torch.load(model_path, map_location='cpu', weights_only=False)
                print(f"✓ {model_id:<25} - Verified")
                verified_count += 1
            else:
                print(f"✗ {model_id:<25} - File not found")
                failed_count += 1
        except Exception as e:
            print(f"✗ {model_id:<25} - Load failed: {e}")
            failed_count += 1
    
    print(f"\n📊 Verification Results:")
    print(f"✓ Verified: {verified_count}")
    print(f"✗ Failed: {failed_count}")
    
    return failed_count == 0


def main():
    """Main function to download and verify models."""
    models_dir = Path(__file__).parent
    
    print("🧠 Torch Inference Framework - Test Model Downloader")
    print("=" * 60)
    
    try:
        # Download models
        downloader = download_all_test_models(models_dir)
        
        # Verify models
        success = verify_models(models_dir)
        
        if success:
            print("\n🎉 All models downloaded and verified successfully!")
            print("\n💡 You can now run tests with real models:")
            print("   python ../run_tests.py all")
        else:
            print("\n⚠️  Some models failed verification. Check the output above.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n❌ Download interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

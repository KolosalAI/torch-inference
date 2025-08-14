"""
Model loader utilities for tests using real downloaded models.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import warnings

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class TestModelLoader:
    """Loads real models for testing the torch-inference framework."""
    
    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            models_dir = Path(__file__).parent
        
        self.models_dir = Path(models_dir)
        self.registry_file = self.models_dir / "model_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the model registry."""
        if not self.registry_file.exists():
            warnings.warn(
                f"Model registry not found at {self.registry_file}. "
                "Run 'python tests/models/create_test_models.py' to download models."
            )
            return {}
        
        try:
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load model registry: {e}")
            return {}
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with their metadata."""
        return self.registry.copy()
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        return self.registry.get(model_id)
    
    def load_model(self, model_id: str, device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load a model by ID.
        
        Args:
            model_id: Model identifier from registry
            device: Device to load model on
            
        Returns:
            Tuple of (model, model_info)
        """
        if model_id not in self.registry:
            available_models = ", ".join(self.registry.keys())
            raise ValueError(
                f"Model '{model_id}' not found in registry. "
                f"Available models: {available_models}"
            )
        
        model_info = self.registry[model_id]
        
        try:
            # Load the full model
            model_path = Path(model_info["full_model_path"])
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Use weights_only=False for test models (trusted source)
            model = torch.load(model_path, map_location=device, weights_only=False)
            model.eval()
            
            return model, model_info
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_id}': {e}")
    
    def load_model_for_task(self, task: str, device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load a model suitable for a specific task.
        
        Args:
            task: Task type (e.g., "text-classification", "image-classification")
            device: Device to load model on
            
        Returns:
            Tuple of (model, model_info)
        """
        # Find models suitable for the task
        suitable_models = [
            model_id for model_id, info in self.registry.items()
            if info.get("task") == task
        ]
        
        if not suitable_models:
            raise ValueError(f"No models found for task '{task}'")
        
        # Prefer smaller models for faster testing
        model_id = min(suitable_models, key=lambda x: self.registry[x].get("size_mb", float('inf')))
        
        return self.load_model(model_id, device)
    
    def load_classification_model(self, device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
        """Load a model suitable for classification tasks."""
        # Try image classification first, then text classification
        for task in ["image-classification", "text-classification", "classification"]:
            try:
                return self.load_model_for_task(task, device)
            except ValueError:
                continue
        
        # Fallback to any available model
        if self.registry:
            model_id = next(iter(self.registry.keys()))
            return self.load_model(model_id, device)
        
        raise ValueError("No classification models available")
    
    def load_lightweight_model(self, device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
        """Load the smallest available model for fast testing."""
        if not self.registry:
            raise ValueError("No models available")
        
        # Find the smallest model
        model_id = min(
            self.registry.keys(), 
            key=lambda x: self.registry[x].get("size_mb", float('inf'))
        )
        
        return self.load_model(model_id, device)
    
    def load_models_by_source(self, source: str, device: str = "cpu") -> Dict[str, Tuple[nn.Module, Dict[str, Any]]]:
        """
        Load all models from a specific source.
        
        Args:
            source: Source type ("huggingface", "torchvision", "custom")
            device: Device to load models on
            
        Returns:
            Dictionary mapping model_id to (model, model_info)
        """
        models = {}
        
        for model_id, model_info in self.registry.items():
            if model_info.get("source") == source:
                try:
                    model, info = self.load_model(model_id, device)
                    models[model_id] = (model, info)
                except Exception as e:
                    warnings.warn(f"Failed to load {model_id}: {e}")
        
        return models
    
    def create_sample_input(self, model_id: str, batch_size: int = 1) -> torch.Tensor:
        """
        Create sample input data for a model.
        
        Args:
            model_id: Model identifier
            batch_size: Batch size for input
            
        Returns:
            Sample input tensor
        """
        if model_id not in self.registry:
            raise ValueError(f"Model '{model_id}' not found")
        
        model_info = self.registry[model_id]
        task = model_info.get("task", "classification")
        source = model_info.get("source", "custom")
        
        # Create appropriate sample inputs based on model type
        if task == "image-classification" or "vit" in model_id.lower() or "resnet" in model_id.lower():
            # Image input (batch_size, channels, height, width)
            return torch.randn(batch_size, 3, 224, 224)
        
        elif task == "text-classification" or "bert" in model_id.lower() or "gpt" in model_id.lower():
            # Text input (sequence of token IDs)
            seq_length = 128
            vocab_size = 30522  # Common BERT vocab size
            return torch.randint(0, vocab_size, (batch_size, seq_length))
        
        elif "transformer" in model_id.lower():
            # Transformer input
            return torch.randint(0, 1000, (batch_size, 64))
        
        elif "rnn" in model_id.lower() or "lstm" in model_id.lower():
            # RNN input
            return torch.randint(0, 1000, (batch_size, 32))
        
        elif "cnn" in model_id.lower():
            # CNN input (smaller image)
            return torch.randn(batch_size, 3, 64, 64)
        
        else:
            # Default to linear input
            return torch.randn(batch_size, 784)
    
    def get_expected_output_shape(self, model_id: str, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Get expected output shape for a model given input shape.
        
        Args:
            model_id: Model identifier
            input_shape: Input tensor shape
            
        Returns:
            Expected output shape
        """
        if model_id not in self.registry:
            raise ValueError(f"Model '{model_id}' not found")
        
        model_info = self.registry[model_id]
        
        # Load config if available
        config_path = Path(model_info.get("config_path", ""))
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                if "num_classes" in config:
                    return (input_shape[0], config["num_classes"])
                elif "num_labels" in config:
                    return (input_shape[0], config["num_labels"])
            except:
                pass
        
        # Default assumptions based on model type
        batch_size = input_shape[0]
        
        if "classification" in model_info.get("task", ""):
            return (batch_size, 10)  # Assume 10 classes
        
        # Default to same as input for other tasks
        return input_shape
    
    def verify_model_loading(self, model_id: str) -> bool:
        """
        Verify that a model can be loaded successfully.
        
        Args:
            model_id: Model identifier to verify
            
        Returns:
            True if model loads successfully, False otherwise
        """
        try:
            model, model_info = self.load_model(model_id)
            
            # Create sample input and test forward pass
            sample_input = self.create_sample_input(model_id)
            
            with torch.no_grad():
                output = model(sample_input)
            
            print(f"✓ {model_id}: Input {tuple(sample_input.shape)} -> Output {tuple(output.shape)}")
            return True
            
        except Exception as e:
            print(f"✗ {model_id}: {e}")
            return False
    
    def verify_all_models(self) -> Dict[str, bool]:
        """
        Verify all models in the registry.
        
        Returns:
            Dictionary mapping model_id to verification result (True/False)
        """
        results = {}
        
        print("🔍 Verifying all models...")
        print("-" * 50)
        
        for model_id in self.registry.keys():
            results[model_id] = self.verify_model_loading(model_id)
        
        successful = sum(results.values())
        total = len(results)
        
        print("-" * 50)
        print(f"📊 Verification Results: {successful}/{total} models passed")
        
        return results


# Global instance for easy access in tests
_test_model_loader = None


def get_test_model_loader() -> TestModelLoader:
    """Get the global test model loader instance."""
    global _test_model_loader
    if _test_model_loader is None:
        _test_model_loader = TestModelLoader()
    return _test_model_loader


def load_test_model(model_id: str, device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
    """Convenience function to load a test model."""
    loader = get_test_model_loader()
    return loader.load_model(model_id, device)


def create_test_input(model_id: str, batch_size: int = 1) -> torch.Tensor:
    """Convenience function to create test input for a model."""
    loader = get_test_model_loader()
    return loader.create_sample_input(model_id, batch_size)


def list_test_models() -> Dict[str, Dict[str, Any]]:
    """Convenience function to list available test models."""
    loader = get_test_model_loader()
    return loader.list_available_models()


# Model categories for easy access
MODEL_CATEGORIES = {
    "lightweight": ["simple_linear", "simple_cnn"],
    "image": ["mobilenet_v2", "resnet18", "efficientnet_b0", "vit_base"],
    "text": ["distilbert_sentiment", "sentence_transformer"],
    "custom": ["simple_cnn", "simple_transformer", "simple_rnn", "simple_linear"]
}


def get_models_by_category(category: str) -> Dict[str, Dict[str, Any]]:
    """Get models by category."""
    loader = get_test_model_loader()
    all_models = loader.list_available_models()
    
    if category not in MODEL_CATEGORIES:
        available_categories = ", ".join(MODEL_CATEGORIES.keys())
        raise ValueError(f"Unknown category '{category}'. Available: {available_categories}")
    
    category_models = {}
    for model_id in MODEL_CATEGORIES[category]:
        if model_id in all_models:
            category_models[model_id] = all_models[model_id]
    
    return category_models

#!/usr/bin/env python3
"""Debug script for the failing test case."""

import torch
from tests.models.model_loader import TestModelLoader
from framework.adapters.model_adapters import PyTorchModelAdapter
from framework.core.config import InferenceConfig, DeviceConfig, DeviceType

def debug_prediction_issue():
    """Debug the prediction issue with batch processing."""
    
    # Load test model
    loader = TestModelLoader()
    model, model_info = loader.load_lightweight_model(device="cpu")
    print(f"Loaded model: {model_info}")
    print(f"Model architecture: {model}")
    
    # Find model ID
    available_models = loader.list_available_models()
    model_id = None
    for mid, info in available_models.items():
        if info["size_mb"] == model_info["size_mb"]:
            model_id = mid
            break
    
    print(f"Model ID: {model_id}")
    
    # Create sample inputs with batch size 2
    sample_input = loader.create_sample_input(model_id, batch_size=2)
    print(f"Sample input shape: {sample_input.shape}")
    print(f"Sample input dtype: {sample_input.dtype}")
    
    # Test model forward pass directly
    with torch.no_grad():
        raw_output = model(sample_input)
        print(f"Raw model output shape: {raw_output.shape}")
        print(f"Raw model output: {raw_output}")
    
    # Create PyTorchModelAdapter
    config = InferenceConfig(device=DeviceConfig(device_type=DeviceType.CPU))
    adapter = PyTorchModelAdapter(config)
    adapter.model = model
    adapter._is_loaded = True
    
    # Test prediction through adapter
    try:
        result = adapter.predict(sample_input)
        print(f"Adapter result type: {type(result)}")
        print(f"Adapter result: {result}")
        
        if isinstance(result, dict):
            print("Prediction keys:", result.keys())
            if "predictions" in result:
                print(f"Predictions type: {type(result['predictions'])}")
                if hasattr(result['predictions'], 'shape'):
                    print(f"Predictions shape: {result['predictions'].shape}")
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_prediction_issue()

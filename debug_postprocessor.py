#!/usr/bin/env python3
"""Debug the postprocessor issue."""

import torch
from tests.models.model_loader import TestModelLoader
from framework.processors.postprocessor import create_default_postprocessing_pipeline
from framework.core.config import InferenceConfig, DeviceConfig, DeviceType

def debug_postprocessor():
    """Debug the postprocessor issue."""
    
    # Create sample model output
    batch_outputs = torch.tensor([
        [ 0.2088, -0.4266,  0.0405,  0.0335,  0.1769, -0.0620, -0.3346, -0.2386,  0.1325, -0.1128],
        [ 0.2066, -0.4253,  0.0376,  0.0368,  0.1731, -0.0618, -0.3329, -0.2382,  0.1292, -0.1116]
    ])
    
    print(f"Test output shape: {batch_outputs.shape}")
    print(f"Test output: {batch_outputs}")
    
    # Create config and postprocessor
    config = InferenceConfig(device=DeviceConfig(device_type=DeviceType.CPU))
    pipeline = create_default_postprocessing_pipeline(config)
    
    # Test detection
    output_type = pipeline.detect_output_type(batch_outputs)
    print(f"Detected output type: {output_type}")
    
    # Test postprocessing
    result = pipeline.auto_postprocess(batch_outputs)
    print(f"Postprocess result type: {type(result)}")
    print(f"Postprocess result: {result}")
    
    if isinstance(result, dict):
        print("Result keys:", result.keys())
        if "predictions" in result:
            print(f"Predictions type: {type(result['predictions'])}")
            if hasattr(result['predictions'], 'shape'):
                print(f"Predictions shape: {result['predictions'].shape}")

if __name__ == "__main__":
    debug_postprocessor()

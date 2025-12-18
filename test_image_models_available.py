#!/usr/bin/env python3
"""
Test image classification models that can be downloaded from HuggingFace
and used for inference.
"""

import json
import os
from pathlib import Path

# Read the model registry
with open('model_registry.json', 'r') as f:
    registry = json.load(f)

# Get all image classification models
image_models = []
for model_id, model_data in registry['models'].items():
    if model_data.get('task') == 'Image Classification':
        image_models.append({
            'id': model_id,
            'name': model_data['name'],
            'architecture': model_data['architecture'],
            'repo_id': model_data.get('repo_id', model_data.get('url', '')),
            'size': model_data['size_estimate'],
            'accuracy': model_data.get('accuracy', 'N/A'),
            'note': model_data.get('note', '')
        })

# Sort by size (smallest first for testing)
def parse_size(size_str):
    size_str = size_str.replace('~', '').strip()
    if 'GB' in size_str:
        return float(size_str.replace('GB', '').strip()) * 1024
    elif 'MB' in size_str:
        return float(size_str.replace('MB', '').strip())
    return 0

image_models.sort(key=lambda x: parse_size(x['size']))

print("=" * 80)
print("SOTA IMAGE CLASSIFICATION MODELS")
print("=" * 80)
print()

for idx, model in enumerate(image_models, 1):
    print(f"{idx}. {model['name']}")
    print(f"   ID: {model['id']}")
    print(f"   Architecture: {model['architecture']}")
    print(f"   Repo: {model['repo_id']}")
    print(f"   Size: {model['size']}")
    print(f"   Accuracy: {model['accuracy']}")
    if model['note']:
        print(f"   Note: {model['note']}")
    print()

# Test if we can access these models via timm
print("=" * 80)
print("TESTING MODEL AVAILABILITY (via timm)")
print("=" * 80)
print()

try:
    import timm
    print(f"✓ timm version: {timm.__version__}")
    print()
    
    # Extract model names from repo_ids
    for model in image_models[:3]:  # Test first 3 models
        repo_id = model['repo_id']
        # timm repo format: timm/model_name
        if 'timm/' in repo_id:
            model_name = repo_id.split('/')[-1]
            
            print(f"Testing: {model['name']}")
            print(f"  Model ID: {model_name}")
            
            # Check if model is available
            available_models = timm.list_models(f"*{model_name.split('.')[0]}*", pretrained=True)
            if available_models:
                print(f"  ✓ Available in timm: {available_models[0]}")
                
                # Try to create model (without loading weights)
                try:
                    m = timm.create_model(available_models[0], pretrained=False, num_classes=1000)
                    print(f"  ✓ Model can be created")
                    print(f"  ✓ Parameters: {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M")
                except Exception as e:
                    print(f"  ✗ Error creating model: {e}")
            else:
                print(f"  ✗ Not found in timm")
            print()
            
except ImportError:
    print("✗ timm not installed. Install with: pip install timm")
    print()
    print("Alternative: Use transformers library")
    try:
        import transformers
        print(f"✓ transformers version: {transformers.__version__}")
    except ImportError:
        print("✗ transformers not installed. Install with: pip install transformers")

print()
print("=" * 80)
print("RECOMMENDATION FOR TESTING")
print("=" * 80)
print()
print("For actual inference testing, use Python with either:")
print("1. timm library (for Vision Transformer models)")
print("   pip install timm torch torchvision")
print()
print("2. transformers library (for HuggingFace models)")
print("   pip install transformers torch torchvision")
print()
print("Smallest model for testing:")
print(f"  - {image_models[0]['name']} ({image_models[0]['size']})")
print(f"  - {image_models[0]['repo_id']}")

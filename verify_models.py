#!/usr/bin/env python3
"""
Verify test model setup for torch-inference framework.
"""

import sys
from pathlib import Path

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tests.models.model_loader import TestModelLoader
    
    def main():
        print("🧠 Torch Inference Framework - Model Verification")
        print("=" * 55)
        
        # Initialize model loader
        models_dir = Path(__file__).parent / "tests" / "models"
        loader = TestModelLoader(models_dir)
        
        # Check if we have any models
        available_models = loader.list_available_models()
        
        if not available_models:
            print("❌ No models found!")
            print("\n💡 To download test models, run:")
            print("   python tests/models/create_test_models.py")
            return 1
        
        print(f"✓ Found {len(available_models)} models")
        print()
        
        # Verify each model
        verification_results = loader.verify_all_models()
        
        # Summary
        successful = sum(verification_results.values())
        total = len(verification_results)
        
        print(f"\n📊 Verification Summary:")
        print(f"✓ Working models: {successful}/{total}")
        
        if successful == total:
            print("\n🎉 All models verified successfully!")
            print("\n💡 You can now run tests with:")
            print("   python run_tests.py all")
            return 0
        elif successful > 0:
            print(f"\n⚠️  {total - successful} models failed verification")
            print("Tests will still work with available models")
            return 0
        else:
            print("\n❌ No models verified successfully")
            print("Please check model downloads")
            return 1

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n💡 Make sure you've installed required dependencies:")
    print("   uv add transformers torch torchvision")
    sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())

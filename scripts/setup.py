#!/usr/bin/env python3
"""
Environment setup script for PyTorch Inference Framework.
"""

import os
import sys
import subprocess
from pathlib import Path
import platform
import shutil


def run_command(command: str, cwd=None):
    """Run a shell command and print the output."""
    print(f"🔄 Running: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True,
            cwd=cwd
        )
        if result.stdout.strip():
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {command}")
        print(f"Error: {e.stderr}")
        return False


def setup_python_environment():
    """Set up Python environment and dependencies."""
    print("🐍 Setting up Python environment...")
    
    # Check Python version
    version = sys.version_info
    if version < (3, 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    
    # Install/upgrade pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip"):
        return False
    
    # Install requirements
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        if not run_command(f"{sys.executable} -m pip install -r requirements.txt"):
            return False
    else:
        print("⚠️  requirements.txt not found, installing basic dependencies...")
        basic_deps = [
            "fastapi[all]",
            "torch",
            "torchvision",
            "torchaudio",
            "transformers",
            "uvicorn[standard]",
            "pydantic",
            "numpy",
            "psutil",
            "pyyaml",
            "pytest",
            "pytest-asyncio"
        ]
        
        for dep in basic_deps:
            if not run_command(f"{sys.executable} -m pip install {dep}"):
                print(f"⚠️  Failed to install {dep}, continuing...")
    
    return True


def setup_directories():
    """Create necessary directories."""
    print("📁 Setting up directories...")
    
    directories = [
        "logs",
        "models",
        "cache",
        "calibration_cache",
        "kernel_cache",
        "data",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created/verified: {directory}/")
    
    return True


def check_gpu_support():
    """Check for GPU support."""
    print("🔍 Checking GPU support...")
    
    try:
        import torch
        
        # CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {name}")
        else:
            print("❌ CUDA not available")
        
        # MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("✅ Apple MPS available")
        
        return True
        
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check GPU support")
        return False


def create_config_files():
    """Create configuration files if they don't exist."""
    print("⚙️  Checking configuration files...")
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Check for existing configs
    existing_configs = list(config_dir.glob("*.yaml"))
    if existing_configs:
        print(f"✅ Found existing config files: {[f.name for f in existing_configs]}")
    else:
        print("⚠️  No config files found. Please create config files manually.")
    
    return True


def run_tests():
    """Run basic tests to verify setup."""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        import torch
        import fastapi
        import uvicorn
        print("✅ Core imports successful")
        
        # Test basic PyTorch operation
        x = torch.randn(2, 2)
        y = x + 1
        print("✅ Basic PyTorch operations working")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 PyTorch Inference Framework - Environment Setup")
    print("=" * 60)
    
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🖥️  Platform: {platform.platform()}")
    print(f"🐍 Python: {sys.version}")
    print("=" * 60)
    
    steps = [
        ("Python Environment", setup_python_environment),
        ("Directories", setup_directories),
        ("GPU Support", check_gpu_support),
        ("Configuration", create_config_files),
        ("Basic Tests", run_tests),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}")
        print("-" * 40)
        
        if not step_func():
            failed_steps.append(step_name)
            print(f"❌ {step_name} setup failed!")
        else:
            print(f"✅ {step_name} setup completed!")
    
    print("\n" + "=" * 60)
    
    if failed_steps:
        print(f"⚠️  Setup completed with issues in: {', '.join(failed_steps)}")
        print("Please review the errors above and fix them manually.")
        return 1
    else:
        print("🎉 Environment setup completed successfully!")
        print("\n📚 Next steps:")
        print("   1. Review configuration files in config/")
        print("   2. Run: python main.py")
        print("   3. Visit: http://localhost:8000")
        return 0


if __name__ == "__main__":
    sys.exit(main())

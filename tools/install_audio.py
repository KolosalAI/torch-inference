#!/usr/bin/env python3
"""
Audio Dependencies Installation Script

This script helps install and verify audio processing dependencies
for the PyTorch Inference Framework.

Usage:
    python tools/install_audio.py
    python tools/install_audio.py --check-only
    python tools/install_audio.py --force-reinstall
"""

import subprocess
import sys
import importlib
import argparse
from pathlib import Path


class AudioInstaller:
    """Installer for audio processing dependencies."""
    
    def __init__(self):
        """Initialize the audio installer."""
        self.dependencies = {
            "librosa": {
                "package": "librosa>=0.10.0",
                "description": "Audio analysis and processing library",
                "import_name": "librosa",
                "critical": True
            },
            "soundfile": {
                "package": "soundfile>=0.12.0", 
                "description": "Audio file I/O library",
                "import_name": "soundfile",
                "critical": True
            },
            "transformers": {
                "package": "transformers>=4.30.0",
                "description": "HuggingFace Transformers for audio models",
                "import_name": "transformers",
                "critical": True
            },
            "datasets": {
                "package": "datasets>=2.10.0",
                "description": "HuggingFace Datasets for audio processing",
                "import_name": "datasets",
                "critical": False
            },
            "accelerate": {
                "package": "accelerate>=0.20.0",
                "description": "HuggingFace Accelerate for model acceleration",
                "import_name": "accelerate",
                "critical": False
            },
            "speechbrain": {
                "package": "speechbrain>=0.5.0",
                "description": "SpeechBrain for advanced speech processing",
                "import_name": "speechbrain",
                "critical": False
            }
        }
        
        self.status = {}
    
    def check_dependency(self, name: str, info: dict) -> bool:
        """Check if a dependency is available."""
        try:
            importlib.import_module(info["import_name"])
            print(f"✅ {name}: Available")
            return True
        except ImportError:
            print(f"❌ {name}: Not available")
            return False
    
    def check_all_dependencies(self) -> dict:
        """Check all audio dependencies."""
        print("🔍 Checking audio dependencies...")
        print("-" * 50)
        
        all_available = True
        critical_available = True
        
        for name, info in self.dependencies.items():
            is_available = self.check_dependency(name, info)
            self.status[name] = is_available
            
            if not is_available:
                all_available = False
                if info["critical"]:
                    critical_available = False
        
        print("-" * 50)
        
        if all_available:
            print("🎉 All audio dependencies are available!")
        elif critical_available:
            print("⚠️ Critical dependencies available, but some optional packages missing.")
        else:
            print("❌ Critical audio dependencies are missing.")
        
        return {
            "all_available": all_available,
            "critical_available": critical_available,
            "status": self.status
        }
    
    def install_dependency(self, package: str) -> bool:
        """Install a single dependency."""
        try:
            print(f"📦 Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            print(f"✅ Successfully installed {package}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    def install_all_dependencies(self, force_reinstall: bool = False) -> bool:
        """Install all audio dependencies."""
        print("📦 Installing audio dependencies...")
        print("-" * 50)
        
        install_args = [sys.executable, "-m", "pip", "install"]
        if force_reinstall:
            install_args.append("--force-reinstall")
        
        # Install critical dependencies first
        critical_packages = [
            info["package"] for name, info in self.dependencies.items() 
            if info["critical"]
        ]
        
        if critical_packages:
            print("Installing critical dependencies...")
            try:
                subprocess.check_call(install_args + critical_packages)
                print("✅ Critical dependencies installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install critical dependencies: {e}")
                return False
        
        # Install optional dependencies
        optional_packages = [
            info["package"] for name, info in self.dependencies.items() 
            if not info["critical"]
        ]
        
        if optional_packages:
            print("Installing optional dependencies...")
            for package in optional_packages:
                try:
                    subprocess.check_call(install_args + [package])
                    print(f"✅ Installed {package}")
                except subprocess.CalledProcessError as e:
                    print(f"⚠️ Failed to install optional package {package}: {e}")
        
        print("-" * 50)
        print("🎉 Audio dependencies installation completed!")
        return True
    
    def install_torch_audio_extra(self) -> bool:
        """Install PyTorch audio framework with audio extras."""
        try:
            print("📦 Installing torch-inference-optimized[audio]...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch-inference-optimized[audio]"
            ])
            print("✅ Successfully installed torch-inference-optimized[audio]")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install torch-inference-optimized[audio]: {e}")
            return False
    
    def verify_installation(self):
        """Verify that the installation was successful."""
        print("\n🔍 Verifying installation...")
        print("-" * 50)
        
        # Check dependencies again
        results = self.check_all_dependencies()
        
        # Try to import framework audio modules
        try:
            print("\n🧪 Testing framework audio imports...")
            from framework.models.audio import create_tts_model, create_stt_model
            from framework.processors.audio import AudioPreprocessor
            print("✅ Framework audio modules imported successfully")
            
            # Test basic functionality
            print("\n🧪 Testing basic audio functionality...")
            from framework.core.config_manager import get_config_manager
            config = get_config_manager().get_inference_config()
            
            # Test preprocessor creation
            preprocessor = AudioPreprocessor(config)
            print("✅ Audio preprocessor created successfully")
            
            print("\n🎉 Audio functionality verification completed!")
            
        except ImportError as e:
            print(f"❌ Framework audio import failed: {e}")
            print("💡 Try running: python tools/install_audio.py")
        except Exception as e:
            print(f"⚠️ Audio functionality test failed: {e}")
            print("💡 This might be expected if audio dependencies are not fully installed")
    
    def show_installation_help(self):
        """Show helpful installation information."""
        print("\n💡 INSTALLATION HELP")
        print("=" * 50)
        print("1. Install all audio dependencies:")
        print("   python tools/install_audio.py")
        print()
        print("2. Install using pip extras:")
        print("   pip install torch-inference-optimized[audio]")
        print()
        print("3. Install individual packages:")
        for name, info in self.dependencies.items():
            critical = " (CRITICAL)" if info["critical"] else " (optional)"
            print(f"   pip install {info['package']}{critical}")
        print()
        print("4. Check current status:")
        print("   python tools/install_audio.py --check-only")
        print()
        print("5. Force reinstall:")
        print("   python tools/install_audio.py --force-reinstall")
        print("=" * 50)


def main():
    """Main function for the audio installer."""
    parser = argparse.ArgumentParser(description="Install audio dependencies")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check dependencies, don't install")
    parser.add_argument("--force-reinstall", action="store_true",
                       help="Force reinstall all dependencies")
    parser.add_argument("--help-info", action="store_true",
                       help="Show installation help information")
    parser.add_argument("--verify", action="store_true",
                       help="Verify installation after installing")
    
    args = parser.parse_args()
    
    installer = AudioInstaller()
    
    if args.help_info:
        installer.show_installation_help()
        return
    
    print("🎵 PyTorch Inference Framework - Audio Dependencies Installer")
    print("=" * 60)
    
    # Check current status
    results = installer.check_all_dependencies()
    
    if args.check_only:
        print("\n📋 Dependency Status Summary:")
        for name, status in results["status"].items():
            status_text = "✅ Available" if status else "❌ Missing"
            critical = " (CRITICAL)" if installer.dependencies[name]["critical"] else ""
            print(f"  {name}: {status_text}{critical}")
        
        if not results["critical_available"]:
            print("\n💡 To install missing dependencies:")
            print("   python tools/install_audio.py")
        
        return
    
    # Install dependencies if needed
    if not results["all_available"]:
        print(f"\n📦 Installing missing dependencies...")
        success = installer.install_all_dependencies(args.force_reinstall)
        
        if success and args.verify:
            installer.verify_installation()
    else:
        print("\n🎉 All dependencies are already available!")
        if args.verify:
            installer.verify_installation()


if __name__ == "__main__":
    main()

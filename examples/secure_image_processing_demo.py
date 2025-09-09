"""
Example script demonstrating secure image preprocessing and postprocessing.

This script shows how to:
1. Set up secure image processing with attack prevention
2. Process images with security validation
3. Handle different types of image attacks
4. Monitor security events and statistics
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
import tempfile
import logging

# Add the framework to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from framework.processors.image.secure_image_processor import (
        SecurityLevel, SecurityConfig, SecurityReport,
        SecureImagePreprocessor, SecureImagePostprocessor,
        create_secure_image_processor, validate_image_security
    )
    from framework.security.security import SecurityManager
    from framework.core.config import InferenceConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the framework is properly installed and in the Python path.")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_example_image(width: int = 224, height: int = 224, 
                        image_type: str = "normal") -> np.ndarray:
    """Create different types of example images for testing."""
    
    if image_type == "normal":
        # Create a normal-looking image with some structure
        image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
        
        # Add some patterns to make it look more natural
        for i in range(0, height, 20):
            for j in range(0, width, 20):
                color = np.random.randint(0, 255, 3)
                image[i:i+10, j:j+10] = color
        
    elif image_type == "adversarial":
        # Create an image that might trigger adversarial detection
        # High-frequency noise pattern
        base = np.random.randint(100, 150, (height, width, 3), dtype=np.uint8)
        noise = np.random.randint(-50, 50, (height, width, 3), dtype=np.int16)
        image = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    elif image_type == "steganographic":
        # Create an image that might trigger steganography detection
        # Unusual pattern in LSBs
        image = np.random.randint(0, 254, (height, width, 3), dtype=np.uint8)
        # Set LSBs to follow a pattern
        for i in range(height):
            for j in range(width):
                for c in range(3):
                    # Create a checkerboard pattern in LSBs
                    if (i + j) % 2 == 0:
                        image[i, j, c] = (image[i, j, c] & 0xFE) | 1  # Set LSB to 1
                    else:
                        image[i, j, c] = image[i, j, c] & 0xFE  # Set LSB to 0
    
    elif image_type == "low_entropy":
        # Create an image with very low entropy (uniform colors)
        color = np.random.randint(0, 255, 3)
        image = np.full((height, width, 3), color, dtype=np.uint8)
        
    elif image_type == "high_gradient":
        # Create an image with high gradients (might trigger adversarial detection)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                # Create strong gradients
                image[i, j] = [(i * 255) // height, (j * 255) // width, 128]
        
    else:
        # Default to normal
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    return image


def save_image_to_temp_file(image: np.ndarray, format_ext: str = ".png") -> str:
    """Save image to a temporary file."""
    try:
        from PIL import Image
        
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=format_ext)
        os.close(temp_fd)
        
        # Save image
        pil_image = Image.fromarray(image)
        pil_image.save(temp_path)
        
        return temp_path
        
    except ImportError:
        logger.warning("PIL not available, cannot save test images")
        return None


def demonstrate_secure_preprocessing():
    """Demonstrate secure image preprocessing capabilities."""
    print("\n" + "="*60)
    print("SECURE IMAGE PREPROCESSING DEMONSTRATION")
    print("="*60)
    
    # Create secure image processor with high security
    security_config = SecurityConfig(
        security_level=SecurityLevel.HIGH,
        enable_adversarial_detection=True,
        enable_steganography_detection=True,
        enable_statistical_analysis=True,
        max_image_size_mb=10.0,
        max_image_dimensions=(1024, 1024)
    )
    
    # Initialize security manager (optional)
    try:
        from framework.security.config import SecurityConfig as FrameworkSecurityConfig
        framework_security_config = FrameworkSecurityConfig()
        security_manager = SecurityManager(framework_security_config)
        logger.info("Security manager initialized")
    except Exception as e:
        logger.warning(f"Could not initialize security manager: {e}")
        security_manager = None
    
    # Create secure preprocessor
    secure_preprocessor = SecureImagePreprocessor(
        target_size=(224, 224),
        security_config=security_config,
        security_manager=security_manager
    )
    
    # Test different types of images
    test_cases = [
        ("normal", "Normal image"),
        ("adversarial", "Image with adversarial-like patterns"),
        ("steganographic", "Image with steganographic-like patterns"),
        ("low_entropy", "Low entropy image"),
        ("high_gradient", "High gradient image")
    ]
    
    results = []
    
    for image_type, description in test_cases:
        print(f"\n📸 Testing: {description}")
        print("-" * 40)
        
        try:
            # Create test image
            test_image = create_example_image(224, 224, image_type)
            
            # Process with secure preprocessor
            processed_tensor, security_report = secure_preprocessor.process_secure(
                test_image, 
                client_id=f"test_client_{image_type}",
                enable_sanitization=True
            )
            
            # Display results
            print(f"✅ Processing successful")
            print(f"   📊 Is safe: {security_report.is_safe}")
            print(f"   🔍 Threats detected: {[t.value for t in security_report.threats_detected]}")
            print(f"   ⚠️  Threat level: {security_report.threat_level.value}")
            print(f"   🎯 Confidence: {security_report.confidence_score:.2f}")
            print(f"   🛡️  Sanitizations applied: {security_report.sanitization_applied}")
            print(f"   ⏱️  Processing time: {security_report.processing_time:.3f}s")
            print(f"   📏 Output shape: {processed_tensor.shape}")
            
            results.append({
                'type': image_type,
                'success': True,
                'report': security_report
            })
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            results.append({
                'type': image_type,
                'success': False,
                'error': str(e)
            })
    
    # Display summary statistics
    print(f"\n📈 PROCESSING SUMMARY")
    print("-" * 40)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    
    if successful:
        threats_detected = sum(1 for r in successful if r['report'].threats_detected)
        sanitized = sum(1 for r in successful if r['report'].sanitization_applied)
        
        print(f"🔍 Threats detected: {threats_detected}/{len(successful)}")
        print(f"🛡️  Images sanitized: {sanitized}/{len(successful)}")
    
    # Display security statistics
    security_stats = secure_preprocessor.get_security_stats()
    print(f"\n🔒 SECURITY STATISTICS")
    print("-" * 40)
    for key, value in security_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    return results


def demonstrate_file_validation():
    """Demonstrate file-based image validation."""
    print("\n" + "="*60)
    print("FILE VALIDATION DEMONSTRATION")
    print("="*60)
    
    # Create test images and save to files
    test_images = [
        ("normal", create_example_image(256, 256, "normal")),
        ("suspicious", create_example_image(256, 256, "adversarial")),
        ("low_entropy", create_example_image(128, 128, "low_entropy"))
    ]
    
    temp_files = []
    
    try:
        # Save test images to temporary files
        for name, image in test_images:
            temp_path = save_image_to_temp_file(image, ".png")
            if temp_path:
                temp_files.append((name, temp_path))
                print(f"📁 Created test file: {name} -> {temp_path}")
        
        if not temp_files:
            print("⚠️  Could not create test files (PIL not available)")
            return
        
        # Validate each file
        for name, file_path in temp_files:
            print(f"\n🔍 Validating: {name}")
            print("-" * 30)
            
            try:
                # Validate file security
                security_report = validate_image_security(file_path, SecurityLevel.HIGH)
                
                print(f"✅ Validation completed")
                print(f"   📊 Is safe: {security_report.is_safe}")
                print(f"   🔍 Threats: {[t.value for t in security_report.threats_detected]}")
                print(f"   ⚠️  Threat level: {security_report.threat_level.value}")
                print(f"   🎯 Confidence: {security_report.confidence_score:.2f}")
                print(f"   📝 Format validation: {security_report.format_validation}")
                print(f"   📈 Statistical analysis: {len(security_report.statistical_analysis)} metrics")
                
                if security_report.recommended_actions:
                    print(f"   💡 Recommendations: {security_report.recommended_actions}")
                
            except Exception as e:
                print(f"❌ Validation failed: {e}")
    
    finally:
        # Clean up temporary files
        for _, temp_path in temp_files:
            try:
                os.unlink(temp_path)
                print(f"🗑️  Cleaned up: {temp_path}")
            except Exception as e:
                print(f"⚠️  Could not clean up {temp_path}: {e}")


def demonstrate_security_levels():
    """Demonstrate different security levels."""
    print("\n" + "="*60)
    print("SECURITY LEVELS DEMONSTRATION")
    print("="*60)
    
    # Create a test image
    test_image = create_example_image(224, 224, "adversarial")
    
    # Test different security levels
    security_levels = [
        SecurityLevel.LOW,
        SecurityLevel.MEDIUM,
        SecurityLevel.HIGH,
        SecurityLevel.PARANOID
    ]
    
    for level in security_levels:
        print(f"\n🔒 Testing Security Level: {level.value.upper()}")
        print("-" * 40)
        
        try:
            # Create preprocessor with specific security level
            secure_preprocessor = create_secure_image_processor(
                target_size=(224, 224),
                security_level=level
            )
            
            # Process image
            start_time = time.time()
            processed_tensor, security_report = secure_preprocessor.process_secure(
                test_image,
                client_id=f"test_{level.value}",
                enable_sanitization=True
            )
            processing_time = time.time() - start_time
            
            print(f"✅ Processing completed")
            print(f"   ⏱️  Total time: {processing_time:.3f}s")
            print(f"   🔍 Threats detected: {len(security_report.threats_detected)}")
            print(f"   🛡️  Sanitizations: {len(security_report.sanitization_applied)}")
            print(f"   📊 Is safe: {security_report.is_safe}")
            print(f"   🎯 Confidence: {security_report.confidence_score:.2f}")
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")


def main():
    """Main demonstration function."""
    print("🛡️  SECURE IMAGE PROCESSING DEMONSTRATION")
    print("🔒 Framework for Attack Prevention and Security")
    print("")
    
    try:
        # Check dependencies
        required_packages = []
        
        try:
            import torch
            print("✅ PyTorch available")
        except ImportError:
            required_packages.append("torch")
        
        try:
            from PIL import Image
            print("✅ PIL available")
        except ImportError:
            print("⚠️  PIL not available (some features will be limited)")
        
        try:
            import cv2
            print("✅ OpenCV available")
        except ImportError:
            print("⚠️  OpenCV not available (using fallback implementations)")
        
        try:
            import numpy as np
            print("✅ NumPy available")
        except ImportError:
            required_packages.append("numpy")
        
        if required_packages:
            print(f"❌ Missing required packages: {required_packages}")
            print("Please install them and try again.")
            return
        
        # Run demonstrations
        print("\n🚀 Starting demonstrations...")
        
        # Import time here to avoid issues
        import time
        
        # Demonstrate secure preprocessing
        results = demonstrate_secure_preprocessing()
        
        # Demonstrate file validation
        demonstrate_file_validation()
        
        # Demonstrate security levels
        demonstrate_security_levels()
        
        print("\n" + "="*60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("• 🔍 Adversarial attack detection")
        print("• 🕵️  Steganography detection")
        print("• 📊 Statistical anomaly analysis")
        print("• 🛡️  Automatic image sanitization")
        print("• 📁 File format validation")
        print("• 🔒 Multiple security levels")
        print("• 📈 Security monitoring and statistics")
        print("\nSecurity Measures Include:")
        print("• 🚫 Format exploit prevention")
        print("• 🔎 Metadata analysis")
        print("• 📏 Size and dimension validation")
        print("• 🎲 Noise injection defense")
        print("• 🌀 Gaussian blur defense")
        print("• 📦 Bit depth reduction")
        print("• 🔄 JPEG compression simulation")
        print("• ⏱️  Timing attack prevention")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

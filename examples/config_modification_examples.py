#!/usr/bin/env python3
"""
Configuration Modification Examples

This script demonstrates how to modify configuration through environment variables
and see the effects on the application behavior.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from framework.core.config_manager import ConfigManager


def test_configuration_changes():
    """Test different configuration scenarios."""
    
    print("🔧 Configuration Modification Examples")
    print("=" * 50)
    
    # Example 1: Default configuration
    print("\n1. Default Configuration (development):")
    config_manager = ConfigManager(environment='development')
    inference_config = config_manager.get_inference_config()
    print(f"   Device: {inference_config.device.device_type.value}")
    print(f"   Batch size: {inference_config.batch.batch_size}")
    print(f"   Use FP16: {inference_config.device.use_fp16}")
    print(f"   Log level: {inference_config.performance.log_level}")
    
    # Example 2: Production configuration
    print("\n2. Production Configuration:")
    config_manager_prod = ConfigManager(environment='production')
    inference_config_prod = config_manager_prod.get_inference_config()
    print(f"   Device: {inference_config_prod.device.device_type.value}")
    print(f"   Batch size: {inference_config_prod.batch.batch_size}")
    print(f"   Use FP16: {inference_config_prod.device.use_fp16}")
    print(f"   Log level: {inference_config_prod.performance.log_level}")
    
    # Example 3: Environment variable overrides
    print("\n3. Environment Variable Override Example:")
    print("   Setting environment variables...")
    
    # Set some environment variables to override configuration
    os.environ['DEVICE'] = 'cuda'
    os.environ['BATCH_SIZE'] = '8'
    os.environ['USE_FP16'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    # Create new config manager to pick up changes
    config_manager_override = ConfigManager(environment='development')
    inference_config_override = config_manager_override.get_inference_config()
    
    print(f"   Device: {inference_config_override.device.device_type.value}")
    print(f"   Batch size: {inference_config_override.batch.batch_size}")
    print(f"   Use FP16: {inference_config_override.device.use_fp16}")
    print(f"   Log level: {inference_config_override.performance.log_level}")
    
    # Clean up environment variables
    for key in ['DEVICE', 'BATCH_SIZE', 'USE_FP16', 'LOG_LEVEL']:
        if key in os.environ:
            del os.environ[key]
    
    # Example 4: Server configuration changes
    print("\n4. Server Configuration Examples:")
    
    environments = ['development', 'staging', 'production']
    for env in environments:
        config_mgr = ConfigManager(environment=env)
        server_config = config_mgr.get_server_config()
        print(f"   {env.upper()}:")
        print(f"     Host: {server_config['host']}")
        print(f"     Port: {server_config['port']}")
        print(f"     Reload: {server_config['reload']}")
        print(f"     Workers: {server_config['workers']}")
    
    print("\n✅ Configuration modification examples completed!")
    
    print("\n📝 How to modify configuration:")
    print("   1. Edit .env file to change environment variables")
    print("   2. Edit config.yaml to change base configuration")
    print("   3. Set ENVIRONMENT variable to change environment")
    print("   4. Override individual values with environment variables")
    
    print("\n🚀 Examples:")
    print("   # To use CUDA with larger batches:")
    print("   echo 'DEVICE=cuda' >> .env")
    print("   echo 'BATCH_SIZE=16' >> .env")
    print("   echo 'USE_FP16=true' >> .env")
    print()
    print("   # To run in production mode:")
    print("   echo 'ENVIRONMENT=production' >> .env")
    print()
    print("   # To enable debug logging:")
    print("   echo 'LOG_LEVEL=DEBUG' >> .env")


if __name__ == "__main__":
    test_configuration_changes()

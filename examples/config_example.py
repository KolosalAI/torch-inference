#!/usr/bin/env python3
"""
Configuration Example for PyTorch Inference Framework

This script demonstrates how to use the new configuration management system
with .env files and config.yaml files.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from framework.core.config_manager import get_config_manager, ConfigManager


def main():
    """Demonstrate configuration management."""
    print("🔧 PyTorch Inference Framework - Configuration Example")
    print("=" * 60)
    
    # 1. Default configuration (development environment)
    print("\n1. Loading default configuration...")
    config_manager = get_config_manager()
    
    print(f"Environment: {config_manager.environment}")
    print(f"Config files:")
    print(f"  - .env file: {config_manager.env_file}")
    print(f"  - YAML file: {config_manager.config_file}")
    
    # 2. Server configuration
    print("\n2. Server Configuration:")
    server_config = config_manager.get_server_config()
    for key, value in server_config.items():
        print(f"  {key}: {value}")
    
    # 3. Inference configuration
    print("\n3. Inference Configuration:")
    inference_config = config_manager.get_inference_config()
    print(f"  Device type: {inference_config.device.device_type.value}")
    print(f"  Device ID: {inference_config.device.device_id}")
    print(f"  Use FP16: {inference_config.device.use_fp16}")
    print(f"  Batch size: {inference_config.batch.batch_size}")
    print(f"  Max batch size: {inference_config.batch.max_batch_size}")
    print(f"  Input size: {inference_config.preprocessing.input_size}")
    print(f"  Warmup iterations: {inference_config.performance.warmup_iterations}")
    
    # 4. Configuration precedence example
    print("\n4. Configuration Precedence Example:")
    print("   Environment Variable -> YAML Config -> Default Value")
    
    # Test with a sample configuration key
    batch_size = config_manager.get('BATCH_SIZE', 1, 'batch.batch_size')
    print(f"  Batch size: {batch_size}")
    
    device_type = config_manager.get('DEVICE', 'cpu', 'device.type')
    print(f"  Device type: {device_type}")
    
    log_level = config_manager.get('LOG_LEVEL', 'INFO', 'server.log_level')
    print(f"  Log level: {log_level}")
    
    # 5. Environment-specific configuration
    print("\n5. Environment-specific Configuration:")
    
    # Test different environments
    for env in ['development', 'staging', 'production']:
        print(f"\n   {env.upper()} Environment:")
        env_config_manager = ConfigManager(environment=env)
        env_server_config = env_config_manager.get_server_config()
        print(f"     Reload: {env_server_config['reload']}")
        print(f"     Log level: {env_server_config['log_level']}")
        print(f"     Workers: {env_server_config['workers']}")
    
    # 6. Enterprise configuration (if available)
    print("\n6. Enterprise Configuration:")
    enterprise_config = config_manager.get_enterprise_config()
    if enterprise_config:
        print(f"  Environment: {enterprise_config.environment}")
        print(f"  Auth provider: {enterprise_config.auth.provider.value}")
        print(f"  Rate limiting: {enterprise_config.security.enable_rate_limiting}")
    else:
        print("  Enterprise features disabled or not available")
    
    # 7. Configuration export
    print("\n7. Configuration Export:")
    exported_config = config_manager.export_config()
    print(f"  Environment: {exported_config['environment']}")
    print(f"  Env file: {exported_config['env_file']}")
    print(f"  Config file: {exported_config['config_file']}")
    
    print("\n✅ Configuration example completed!")
    print("\n💡 Tips:")
    print("   - Modify .env file to override environment variables")
    print("   - Modify config.yaml to change base configuration")
    print("   - Set ENVIRONMENT=production to use production settings")
    print("   - Check /config endpoint when running the server")


if __name__ == "__main__":
    main()

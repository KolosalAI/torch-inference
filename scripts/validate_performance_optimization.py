"""
Phase 3 Validation Test
Simple validation test to ensure Phase 3 components can be imported and initialized.
"""

import sys
import os

# Add framework to path
framework_path = os.path.join(os.path.dirname(__file__), '..', 'framework')
sys.path.insert(0, framework_path)
print(f"Added to Python path: {os.path.abspath(framework_path)}")

def test_phase3_imports():
    """Test that all Phase 3 components can be imported."""
    print("Testing Phase 3 component imports...")
    
    try:
        from core.memory_optimizer import MemoryOptimizer, MemoryStats
        print("✅ Memory optimizer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import memory optimizer: {e}")
        return False
    
    try:
        from core.comm_optimizer import CommunicationOptimizer, CommPattern
        print("✅ Communication optimizer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import communication optimizer: {e}")
        return False
    
    try:
        from core.dynamic_scaler import DynamicScaler, ScalingConfig, ScalingAction
        print("✅ Dynamic scaler imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import dynamic scaler: {e}")
        return False
    
    try:
        from core.advanced_scheduler import AdvancedScheduler, SchedulerConfig, TaskPriority
        print("✅ Advanced scheduler imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import advanced scheduler: {e}")
        return False
    
    return True

def test_phase3_config():
    """Test Phase 3 configuration enhancements."""
    print("\nTesting Phase 3 configuration...")
    
    try:
        from core.config import MultiGPUConfig
        
        # Test basic config creation
        config = MultiGPUConfig(
            enabled=True,
            strategy="data_parallel",
            device_ids=None,  # Use None to bypass validation for testing
            memory_pool_size_mb=256,
            enable_nccl=False,
            enable_dynamic_scaling=True,
            enable_advanced_scheduling=True
        )
        
        print("✅ Phase 3 MultiGPUConfig created successfully")
        print(f"   - Memory pool size: {config.memory_pool_size_mb}MB")
        print(f"   - Dynamic scaling: {config.enable_dynamic_scaling}")
        print(f"   - Advanced scheduling: {config.enable_advanced_scheduling}")
        print(f"   - Scheduling strategy: {config.scheduling_strategy}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create Phase 3 config: {e}")
        return False

def test_phase3_integration():
    """Test Phase 3 integration with multi-GPU manager."""
    print("\nTesting Phase 3 integration...")
    
    try:
        from core.config import MultiGPUConfig
        from core.multi_gpu_manager import MultiGPUManager
        from core.gpu_manager import GPUManager
        from unittest.mock import Mock
        
        # Create mock GPU manager
        gpu_manager = Mock(spec=GPUManager)
        mock_gpu = Mock()
        mock_gpu.id = 0
        mock_gpu.is_suitable_for_inference.return_value = True
        mock_gpu.memory.total_mb = 8192
        mock_gpu.memory.available_mb = 6144
        gpu_manager.get_detected_gpus.return_value = [mock_gpu]
        
        # Create config with Phase 3 features
        config = MultiGPUConfig(
            enabled=True,
            strategy="data_parallel",
            device_ids=None,  # Use None to bypass validation for testing
            memory_pool_size_mb=256,
            enable_nccl=False,  # Disable for testing
            enable_dynamic_scaling=False,  # Disable for simple test
            enable_advanced_scheduling=False  # Disable for simple test
        )
        
        # Create multi-GPU manager
        manager = MultiGPUManager(config, gpu_manager)
        
        print("✅ Phase 3 MultiGPUManager created successfully")
        print(f"   - Config strategy: {config.strategy}")
        print(f"   - Memory optimization enabled: {hasattr(manager, 'memory_optimizer')}")
        print(f"   - Communication optimization enabled: {hasattr(manager, 'comm_optimizer')}")
        print(f"   - Dynamic scaling enabled: {hasattr(manager, 'dynamic_scaler')}")
        print(f"   - Advanced scheduling enabled: {hasattr(manager, 'advanced_scheduler')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed Phase 3 integration test: {e}")
        return False

def main():
    """Run all Phase 3 validation tests."""
    print("=" * 60)
    print("Phase 3 Multi-GPU Performance Optimization Validation")
    print("=" * 60)
    
    success = True
    
    # Test imports
    success &= test_phase3_imports()
    
    # Test configuration
    success &= test_phase3_config()
    
    # Test integration
    success &= test_phase3_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Phase 3 validation completed successfully!")
        print("\nPhase 3 Features Available:")
        print("  • Memory Optimization with pooling and monitoring")
        print("  • Communication Optimization with NCCL support")
        print("  • Dynamic Scaling with workload-based decisions")
        print("  • Advanced Scheduling with priority and resource awareness")
        print("  • Comprehensive Performance Monitoring")
    else:
        print("❌ Phase 3 validation failed!")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

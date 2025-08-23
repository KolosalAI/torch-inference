"""
Production configuration examples for optimized torch-inference server.

This file provides example configurations for different deployment scenarios
to achieve optimal performance with the enhanced inference engine.
"""

from framework.core.config import InferenceConfig, DeviceConfig, BatchConfig, PerformanceConfig, DeviceType
from framework.core.inference_engine import (
    InferenceEngine, 
    create_ultra_fast_inference_engine, 
    create_hybrid_inference_engine,
    EngineConfig
)

# ============================================================================
# HIGH THROUGHPUT CONFIGURATION
# Optimized for maximum concurrent request handling
# ============================================================================

def get_high_throughput_config():
    """Configuration for high-throughput scenarios (1000+ req/s)."""
    return InferenceConfig(
        device=DeviceConfig(
            device_type=DeviceType.CUDA,  # Use GPU for best performance
            device_id=0,
            use_fp16=True,               # Enable FP16 for faster inference
            use_tensorrt=True,           # Enable TensorRT optimization
            use_torch_compile=True       # Enable torch.compile
        ),
        batch=BatchConfig(
            batch_size=8,                # Larger batches for throughput
            max_batch_size=16,           # Allow even larger batches under load
            min_batch_size=1,            # Single requests for low latency
            queue_size=2000              # Large queue for high load
        ),
        performance=PerformanceConfig(
            max_workers=16,              # Many workers for concurrency
            warmup_iterations=10,        # Thorough warmup
            enable_profiling=False       # Disable profiling in production
        )
    )

# ============================================================================
# LOW LATENCY CONFIGURATION  
# Optimized for minimal response time
# ============================================================================

def get_low_latency_config():
    """Configuration for low-latency scenarios (<20ms response time)."""
    return InferenceConfig(
        device=DeviceConfig(
            device_type=DeviceType.CUDA,
            device_id=0,
            use_fp16=True,               # FP16 for speed
            use_tensorrt=False,          # Skip TensorRT to avoid startup delay
            use_torch_compile=False      # Skip compilation for faster startup
        ),
        batch=BatchConfig(
            batch_size=1,                # Prefer single requests
            max_batch_size=4,            # Small max batch
            min_batch_size=1,
            queue_size=100               # Small queue for quick processing
        ),
        performance=PerformanceConfig(
            max_workers=8,               # Moderate workers to avoid overhead
            warmup_iterations=3,         # Quick warmup
            enable_profiling=False
        )
    )

# ============================================================================
# BALANCED CONFIGURATION
# Good balance between latency and throughput
# ============================================================================

def get_balanced_config():
    """Configuration balanced for both latency and throughput."""
    return InferenceConfig(
        device=DeviceConfig(
            device_type=DeviceType.CUDA,
            device_id=0,
            use_fp16=True,
            use_tensorrt=True,
            use_torch_compile=True
        ),
        batch=BatchConfig(
            batch_size=4,                # Moderate batch size
            max_batch_size=8,
            min_batch_size=1,
            queue_size=500               # Moderate queue size
        ),
        performance=PerformanceConfig(
            max_workers=8,
            warmup_iterations=5,
            enable_profiling=False
        )
    )

# ============================================================================
# CPU-ONLY CONFIGURATION
# For deployments without GPU
# ============================================================================

def get_cpu_config():
    """Configuration for CPU-only deployments."""
    return InferenceConfig(
        device=DeviceConfig(
            device_type=DeviceType.CPU,
            device_id=0,
            use_fp16=False,              # No FP16 on CPU
            use_tensorrt=False,          # No TensorRT on CPU
            use_torch_compile=True       # torch.compile can help on CPU
        ),
        batch=BatchConfig(
            batch_size=2,                # Smaller batches for CPU
            max_batch_size=4,
            min_batch_size=1,
            queue_size=200
        ),
        performance=PerformanceConfig(
            max_workers=4,               # Limited workers for CPU
            warmup_iterations=3,
            enable_profiling=False
        )
    )

# ============================================================================
# ENGINE SELECTION GUIDE
# ============================================================================

def create_engine_for_workload(model, workload_type: str):
    """
    Create the optimal engine for different workload types.
    
    Args:
        model: The loaded model instance
        workload_type: One of 'high_throughput', 'low_latency', 'balanced', 'cpu'
    
    Returns:
        Configured inference engine
    """
    
    if workload_type == 'high_throughput':
        config = get_high_throughput_config()
        # Use enhanced InferenceEngine with ultra-fast optimizations for maximum throughput
        return create_ultra_fast_inference_engine(model, config)
    
    elif workload_type == 'low_latency':
        config = get_low_latency_config()
        # Use enhanced InferenceEngine with ultra-fast optimizations for lowest latency
        return create_ultra_fast_inference_engine(model, config)
    
    elif workload_type == 'balanced':
        config = get_balanced_config()
        # Use enhanced InferenceEngine with hybrid optimizations for balanced performance
        return create_hybrid_inference_engine(model, config)
    
    elif workload_type == 'cpu':
        config = get_cpu_config()
        # Use enhanced InferenceEngine with hybrid optimizations for CPU workloads
        return create_hybrid_inference_engine(model, config)
    
    else:
        raise ValueError(f"Unknown workload type: {workload_type}")

# ============================================================================
# DEPLOYMENT EXAMPLES
# ============================================================================

async def deploy_high_performance_server(model):
    """Example deployment for high-performance production server."""
    
    # Create high-throughput engine
    engine = create_engine_for_workload(model, 'high_throughput')
    
    # Start the engine
    await engine.start()
    
    print("🚀 High-performance server deployed!")
    print("Expected performance:")
    print("  - Throughput: 1000+ req/s")
    print("  - Latency: <50ms average")
    print("  - Concurrency: 50+ concurrent requests")
    
    return engine

async def deploy_realtime_server(model):
    """Example deployment for real-time applications."""
    
    # Create low-latency engine
    engine = create_engine_for_workload(model, 'low_latency')
    
    # Start the engine
    await engine.start()
    
    print("⚡ Real-time server deployed!")
    print("Expected performance:")
    print("  - Latency: <20ms average")
    print("  - Throughput: 500+ req/s")
    print("  - Response time: Sub-50ms P99")
    
    return engine

# ============================================================================
# MONITORING AND TUNING
# ============================================================================

def monitor_engine_performance(engine):
    """Get performance metrics for monitoring."""
    stats = engine.get_stats()
    
    metrics = {
        'requests_processed': stats.get('requests_processed', 0),
        'average_latency_ms': stats.get('average_processing_time', 0) * 1000,
        'error_rate': stats.get('errors', 0) / max(1, stats.get('requests_processed', 1)),
        'queue_size': stats.get('queue_size', 0),
        'throughput_rps': stats.get('requests_processed', 0) / max(1, stats.get('total_processing_time', 1))
    }
    
    return metrics

def tune_for_latency(engine):
    """Tune engine configuration for lower latency."""
    # These would be dynamic adjustments in a real implementation
    recommendations = [
        "Reduce batch_size to 1-2",
        "Decrease max_workers to 4-8", 
        "Enable FP16 precision",
        "Disable TensorRT compilation",
        "Use smaller queue_size (100-200)"
    ]
    return recommendations

def tune_for_throughput(engine):
    """Tune engine configuration for higher throughput."""
    recommendations = [
        "Increase batch_size to 8-16",
        "Increase max_workers to 12-16",
        "Enable TensorRT compilation", 
        "Use larger queue_size (1000-2000)",
        "Enable torch.compile optimization"
    ]
    return recommendations

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("🔧 Torch-Inference Production Configuration Examples")
    print("=" * 60)
    
    print("\n1. High Throughput Config:")
    high_config = get_high_throughput_config()
    print(f"   - Batch size: {high_config.batch.batch_size}")
    print(f"   - Max workers: {high_config.performance.max_workers}")
    print(f"   - Queue size: {high_config.batch.queue_size}")
    
    print("\n2. Low Latency Config:")
    low_config = get_low_latency_config()
    print(f"   - Batch size: {low_config.batch.batch_size}")
    print(f"   - Max workers: {low_config.performance.max_workers}")
    print(f"   - Queue size: {low_config.batch.queue_size}")
    
    print("\n3. Balanced Config:")
    balanced_config = get_balanced_config()
    print(f"   - Batch size: {balanced_config.batch.batch_size}")
    print(f"   - Max workers: {balanced_config.performance.max_workers}")
    print(f"   - Queue size: {balanced_config.batch.queue_size}")
    
    print("\n4. CPU Config:")
    cpu_config = get_cpu_config()
    print(f"   - Batch size: {cpu_config.batch.batch_size}")
    print(f"   - Max workers: {cpu_config.performance.max_workers}")
    print(f"   - Queue size: {cpu_config.batch.queue_size}")
    
    print("\n📈 Performance Expectations:")
    print("High Throughput: 1000+ req/s, 50ms latency")
    print("Low Latency:     500+ req/s, 20ms latency") 
    print("Balanced:        200+ req/s, 30ms latency")
    print("CPU:             50+ req/s, 100ms latency")

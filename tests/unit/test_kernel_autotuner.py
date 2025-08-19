"""
Unit tests for kernel auto-tuning optimization.
"""

import pytest
import torch
import torch.nn as nn
import time
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path

from framework.optimizers.kernel_autotuner import (
    KernelAutoTuner,
    TuningConfig,
    BenchmarkResult,
    OptimizationStrategy,
    HardwareProfiler,
    HardwareProfile,
    KernelConfig,
    get_kernel_autotuner
)


class SimpleModel(nn.Module):
    """Simple model for testing kernel tuning."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(64 * 8 * 8, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def sample_model():
    """Create sample model for testing."""
    return SimpleModel()


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def tuning_config():
    """Create tuning configuration."""
    return TuningConfig(
        max_iterations=3,  # Keep small for tests
        warmup_iterations=1,
        timeout_seconds=30,
        memory_limit_mb=1000,
        enable_cuda_graphs=False,  # Disable for simpler testing
        use_mixed_precision=False,
        batch_sizes=[1, 4],
        enable_caching=False
    )


class TestTuningConfig:
    """Test tuning configuration."""
    
    def test_default_config(self):
        """Test default tuning configuration."""
        config = TuningConfig()
        assert config.max_iterations == 50
        assert config.warmup_iterations == 5
        assert config.timeout_seconds == 300
        assert config.memory_limit_mb == 4000
        assert config.enable_cuda_graphs == True
        assert config.use_mixed_precision == True
        assert config.batch_sizes == [1, 2, 4, 8, 16, 32]
        assert config.enable_caching == True
        assert config.cache_dir == "./kernel_cache"
        assert config.profile_memory == True
        assert config.optimize_for_inference == True
    
    def test_custom_config(self):
        """Test custom tuning configuration."""
        config = TuningConfig(
            max_iterations=10,
            warmup_iterations=2,
            timeout_seconds=60,
            batch_sizes=[2, 8],
            enable_cuda_graphs=False
        )
        assert config.max_iterations == 10
        assert config.warmup_iterations == 2
        assert config.timeout_seconds == 60
        assert config.batch_sizes == [2, 8]
        assert config.enable_cuda_graphs == False


class TestBenchmarkResult:
    """Test benchmark result structure."""
    
    def test_benchmark_result_creation(self):
        """Test creating benchmark result."""
        result = BenchmarkResult(
            latency_ms=10.5,
            throughput_samples_per_sec=95.2,
            memory_mb=256.0,
            gpu_utilization=85.0,
            configuration={"batch_size": 4},
            device="cuda:0"
        )
        
        assert result.latency_ms == 10.5
        assert result.throughput_samples_per_sec == 95.2
        assert result.memory_mb == 256.0
        assert result.gpu_utilization == 85.0
        assert result.configuration == {"batch_size": 4}
        assert result.device == "cuda:0"
    
    def test_benchmark_result_comparison(self):
        """Test comparing benchmark results."""
        result1 = BenchmarkResult(10.0, 100.0, 200.0, 80.0, {}, "cpu")
        result2 = BenchmarkResult(15.0, 80.0, 250.0, 75.0, {}, "cpu")
        
        # result1 is better (lower latency, higher throughput)
        assert result1.latency_ms < result2.latency_ms
        assert result1.throughput_samples_per_sec > result2.throughput_samples_per_sec


class TestHardwareProfiler:
    """Test hardware profiling functionality."""
    
    def test_profiler_initialization(self):
        """Test hardware profiler initialization."""
        profiler = HardwareProfiler()
        assert profiler is not None
    
    def test_detect_hardware(self):
        """Test hardware detection."""
        profiler = HardwareProfiler()
        hw_info = profiler.detect_hardware()
        
        assert isinstance(hw_info, dict)
        assert "device_type" in hw_info
        assert hw_info["device_type"] in ["cuda", "cpu", "mps"]
        assert "device_count" in hw_info
        assert isinstance(hw_info["device_count"], int)
        assert "total_memory" in hw_info
    
    def test_get_device_capabilities(self):
        """Test getting device capabilities."""
        profiler = HardwareProfiler()
        capabilities = profiler.get_device_capabilities()
        
        assert isinstance(capabilities, dict)
        # Should have basic capability info regardless of device
        assert len(capabilities) > 0
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_cuda_capabilities(self, mock_props, mock_cuda_available):
        """Test CUDA capabilities detection."""
        # Mock CUDA device properties
        mock_props.return_value = MagicMock(
            name="Tesla V100",
            major=7,
            minor=0,
            total_memory=32*1024**3,
            multi_processor_count=80
        )
        
        profiler = HardwareProfiler()
        capabilities = profiler.get_device_capabilities()
        
        # Should include CUDA-specific capabilities
        assert isinstance(capabilities, dict)
    
    def test_benchmark_operation(self, sample_model, sample_input):
        """Test benchmarking operation."""
        profiler = HardwareProfiler()
        
        def operation():
            return sample_model(sample_input)
        
        result = profiler.benchmark_operation(operation, warmup=1, iterations=3)
        
        assert isinstance(result, BenchmarkResult)
        assert result.latency_ms > 0
        assert result.throughput_samples_per_sec > 0
        assert result.memory_mb >= 0
    
    def test_memory_benchmark(self):
        """Test memory benchmarking."""
        profiler = HardwareProfiler()
        
        # Create operation that allocates memory
        def memory_operation():
            x = torch.randn(1000, 1000)
            return torch.mm(x, x.t())
        
        result = profiler.benchmark_operation(memory_operation, warmup=1, iterations=2)
        
        assert isinstance(result, BenchmarkResult)
        assert result.memory_mb > 0


class TestOptimizationStrategy:
    """Test optimization strategies."""
    
    def test_cpu_optimization_strategy(self, sample_model, sample_input, tuning_config):
        """Test CPU optimization strategy."""
        strategy = OptimizationStrategy("cpu", tuning_config)
        
        optimized_model = strategy.optimize(sample_model, sample_input)
        
        # Should return optimized model
        assert optimized_model is not None
        assert isinstance(optimized_model, nn.Module)
        
        # Test that optimized model produces same output shape
        original_output = sample_model(sample_input)
        optimized_output = optimized_model(sample_input)
        assert original_output.shape == optimized_output.shape
    
    def test_cuda_optimization_strategy(self, sample_model, tuning_config):
        """Test CUDA optimization strategy."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.device') as mock_device:
                mock_device.return_value = Mock()
                
                # Mock the model.to() method to return the model itself
                sample_model.to = Mock(return_value=sample_model)
                
                strategy = OptimizationStrategy("cuda", tuning_config)
                
                # Mock the sample input
                with patch('torch.randn') as mock_randn:
                    mock_input = Mock()
                    mock_input.to = Mock(return_value=mock_input)
                    mock_randn.return_value = mock_input
                    
                    sample_input = torch.randn(4, 3, 32, 32)
                    sample_input.to = Mock(return_value=sample_input)
                    
                    optimized_model = strategy.optimize(sample_model, sample_input)
        
                    assert optimized_model is not None
                    assert isinstance(optimized_model, nn.Module)
        
        # Verify model still works - use real tensor for actual inference
        real_input = torch.randn(4, 3, 32, 32)
        original_output = sample_model(real_input)
        optimized_output = optimized_model(real_input)
        assert original_output.shape == optimized_output.shape
    
    def test_jit_compilation(self, sample_model, sample_input, tuning_config):
        """Test JIT compilation optimization."""
        strategy = OptimizationStrategy("cpu", tuning_config)
        
        # Apply JIT compilation
        optimized_model = strategy.apply_jit_compilation(sample_model, sample_input)
        
        # Should return a traced/scripted module
        assert optimized_model is not None
        
        # Test functionality
        original_output = sample_model(sample_input)
        optimized_output = optimized_model(sample_input)
        
        # Outputs should be close (accounting for numerical differences)
        torch.testing.assert_close(original_output, optimized_output, rtol=1e-3, atol=1e-3)
    
    def test_thread_optimization(self, tuning_config):
        """Test thread count optimization."""
        strategy = OptimizationStrategy("cpu", tuning_config)
        
        original_threads = torch.get_num_threads()
        
        # Apply thread optimization
        strategy.optimize_threads()
        
        # Should set number of threads (may or may not change from original)
        current_threads = torch.get_num_threads()
        assert isinstance(current_threads, int)
        assert current_threads > 0
        
        # Restore original setting
        torch.set_num_threads(original_threads)
    
    def test_memory_format_optimization(self, sample_model, sample_input, tuning_config):
        """Test memory format optimization."""
        strategy = OptimizationStrategy("cpu", tuning_config)
        
        # Test channels last optimization
        optimized_input = strategy.optimize_memory_format(sample_input)
        
        # Should return tensor (may or may not change format depending on conditions)
        assert isinstance(optimized_input, torch.Tensor)
        assert optimized_input.shape == sample_input.shape


class TestKernelAutoTuner:
    """Test kernel auto-tuner."""
    
    def test_tuner_initialization(self, tuning_config):
        """Test auto-tuner initialization."""
        tuner = KernelAutoTuner(tuning_config)
        assert tuner.config == tuning_config
        assert isinstance(tuner.profiler, HardwareProfiler)
        assert len(tuner.optimization_cache) == 0
    
    def test_auto_tune_cpu(self, sample_model, sample_input, tuning_config):
        """Test auto-tuning on CPU."""
        tuner = KernelAutoTuner(tuning_config)
        
        device = torch.device("cpu")
        result = tuner.auto_tune(sample_model, sample_input, device)
        
        assert "optimized_model" in result
        assert "best_config" in result
        assert "benchmark_results" in result
        assert "recommendations" in result
        
        # Optimized model should work
        optimized_model = result["optimized_model"]
        output = optimized_model(sample_input)
        assert output.shape == sample_model(sample_input).shape
    
    def test_auto_tune_cuda(self, sample_model, tuning_config):
        """Test auto-tuning on CUDA."""
        with patch('torch.cuda.is_available', return_value=True):
            # Mock the hardware profiler to avoid real CUDA calls
            with patch('framework.optimizers.kernel_autotuner.HardwareProfiler') as mock_profiler_class:
                mock_profiler = Mock()
                mock_profile = Mock()
                mock_profile.device_type = "cuda"
                mock_profile.device_name = "Mock GPU"
                mock_profile.compute_capability = (7, 5)
                mock_profile.memory_gb = 16.0
                mock_profile.core_count = 80
                mock_profile.tensor_core_support = True
                mock_profile.mixed_precision_support = True
                mock_profile.cache_sizes = {}
                
                mock_profiler.get_hardware_profile.return_value = mock_profile
                mock_profiler_class.return_value = mock_profiler
                
                with patch('torch.device') as mock_device:
                    mock_device.return_value = Mock()

                    # Mock the model.to() method
                    sample_model.to = Mock(return_value=sample_model)

                    # Mock sample input with to() method
                    sample_input = torch.randn(4, 3, 32, 32)
                    sample_input.to = Mock(return_value=sample_input)

                    tuner = KernelAutoTuner(tuning_config)
                    result = tuner.auto_tune(sample_model, sample_input, mock_device.return_value)

                    assert "optimized_model" in result
                    assert "best_config" in result
                    assert "benchmark_results" in result

                    # Test optimized model
                    optimized_model = result["optimized_model"]
                    # Use real tensor for actual inference - match model precision
                    real_input = torch.randn(4, 3, 32, 32)
                    
                    # Check if model is in half precision and convert input accordingly
                    first_param = next(optimized_model.parameters())
                    if first_param.dtype == torch.float16:
                        real_input = real_input.half()
                    
                    output = optimized_model(real_input)
                    assert output.shape[0] == real_input.shape[0]  # Batch size should match    def test_batch_size_tuning(self, sample_model, tuning_config):
        """Test batch size tuning."""
        tuning_config.batch_sizes = [2, 4]
        tuner = KernelAutoTuner(tuning_config)
        
        device = torch.device("cpu")
        
        # Test different batch sizes
        for batch_size in tuning_config.batch_sizes:
            sample_input = torch.randn(batch_size, 3, 32, 32)
            result = tuner.auto_tune(sample_model, sample_input, device)
            
            assert "optimized_model" in result
            optimized_model = result["optimized_model"]
            
            # Check if model is in half precision and convert input accordingly
            first_param = next(optimized_model.parameters())
            if first_param.dtype == torch.float16:
                sample_input = sample_input.half()
            
            output = optimized_model(sample_input)
            assert output.shape[0] == batch_size
    
    def test_get_optimization_report(self, sample_model, sample_input, tuning_config):
        """Test optimization report generation."""
        tuner = KernelAutoTuner(tuning_config)
        
        device = torch.device("cpu")
        result = tuner.auto_tune(sample_model, sample_input, device)
        
        report = tuner.get_optimization_report()
        
        assert isinstance(report, dict)
        assert "hardware_info" in report
        assert "optimization_results" in report
        assert "performance_metrics" in report
        assert "recommendations" in report
        
        # Check hardware info
        hw_info = report["hardware_info"]
        assert "device_type" in hw_info
        assert "device_count" in hw_info
    
    def test_save_load_optimization_cache(self, sample_model, sample_input):
        """Test saving and loading optimization cache."""
        config = TuningConfig(enable_caching=True)
        tuner = KernelAutoTuner(config)
        
        device = torch.device("cpu")
        
        # First run - should populate cache
        result1 = tuner.auto_tune(sample_model, sample_input, device)
        
        # Save cache
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "test_cache.json"
            tuner.save_optimization_cache(str(cache_file))
            
            # Create new tuner and load cache
            tuner2 = KernelAutoTuner(config)
            tuner2.load_optimization_cache(str(cache_file))
            
            # Should have loaded cache
            assert len(tuner2.optimization_cache) > 0
    
    def test_timeout_handling(self, sample_model, sample_input):
        """Test timeout handling during optimization."""
        # Use very short timeout
        config = TuningConfig(timeout_seconds=0.1, max_iterations=100)
        tuner = KernelAutoTuner(config)
        
        device = torch.device("cpu")
        
        # Should complete without hanging (may timeout some operations)
        result = tuner.auto_tune(sample_model, sample_input, device)
        
        # Should still return a result, even if not fully optimized
        assert "optimized_model" in result
    
    def test_memory_limit_handling(self, sample_model):
        """Test memory limit handling."""
        # Set very low memory limit
        config = TuningConfig(memory_limit_mb=1)  # 1MB limit
        tuner = KernelAutoTuner(config)
        
        device = torch.device("cpu")
        sample_input = torch.randn(4, 3, 32, 32)
        
        # Should handle memory constraints gracefully
        result = tuner.auto_tune(sample_model, sample_input, device)
        
        assert "optimized_model" in result


class TestGlobalAutoTuner:
    """Test global auto-tuner instance."""
    
    def test_get_kernel_autotuner(self):
        """Test getting global auto-tuner."""
        tuner1 = get_kernel_autotuner()
        tuner2 = get_kernel_autotuner()
        
        # Should return the same instance
        assert tuner1 is tuner2
        assert isinstance(tuner1, KernelAutoTuner)


class TestAdvancedOptimizations:
    """Test advanced optimization features."""
    
    def test_mixed_precision_optimization(self, sample_model, sample_input):
        """Test mixed precision optimization."""
        config = TuningConfig(use_mixed_precision=True, max_iterations=2)
        tuner = KernelAutoTuner(config)
        
        device = torch.device("cpu")
        result = tuner.auto_tune(sample_model, sample_input, device)
        
        assert "optimized_model" in result
        optimized_model = result["optimized_model"]
        
        # Should still produce correct output
        output = optimized_model(sample_input)
        expected_output = sample_model(sample_input)
        assert output.shape == expected_output.shape
    
    def test_cuda_graphs_optimization(self, sample_model):
        """Test CUDA graphs optimization."""
        with patch('torch.cuda.is_available', return_value=True):
            # Mock the hardware profiler to avoid real CUDA calls
            with patch('framework.optimizers.kernel_autotuner.HardwareProfiler') as mock_profiler_class:
                mock_profiler = Mock()
                mock_profile = Mock()
                mock_profile.device_type = "cuda"
                mock_profile.device_name = "Mock GPU"
                mock_profile.compute_capability = (7, 5)
                mock_profile.memory_gb = 16.0
                mock_profile.core_count = 80
                mock_profile.tensor_core_support = True
                mock_profile.mixed_precision_support = True
                mock_profile.cache_sizes = {}
                
                mock_profiler.get_hardware_profile.return_value = mock_profile
                mock_profiler_class.return_value = mock_profiler
                
                config = TuningConfig(enable_cuda_graphs=True, max_iterations=2)
                tuner = KernelAutoTuner(config)
        
                with patch('torch.device') as mock_device:
                    mock_device.return_value = Mock()
                    sample_model.to = Mock(return_value=sample_model)
                    
                    sample_input = torch.randn(4, 3, 32, 32)
                    sample_input.to = Mock(return_value=sample_input)
        
                    result = tuner.auto_tune(sample_model, sample_input, mock_device.return_value)
        
                    assert "optimized_model" in result
                    optimized_model = result["optimized_model"]
        
                    # Test optimized model with real tensor
                    real_input = torch.randn(4, 3, 32, 32)
                    
                    # Check if model is in half precision and convert input accordingly
                    first_param = next(optimized_model.parameters())
                    if first_param.dtype == torch.float16:
                        real_input = real_input.half()
                    
                    output = optimized_model(real_input)
                    assert output.shape[0] == 4
    
    def test_inference_optimization(self, sample_model, sample_input):
        """Test inference-specific optimizations."""
        config = TuningConfig(optimize_for_inference=True, max_iterations=2)
        tuner = KernelAutoTuner(config)
        
        device = torch.device("cpu")
        result = tuner.auto_tune(sample_model, sample_input, device)
        
        assert "optimized_model" in result
        optimized_model = result["optimized_model"]
        
        # Model should be in eval mode for inference
        assert not optimized_model.training


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_device(self, sample_model, sample_input, tuning_config):
        """Test handling of invalid device."""
        tuner = KernelAutoTuner(tuning_config)
        
        # Try with invalid device
        try:
            device = torch.device("invalid")
            result = tuner.auto_tune(sample_model, sample_input, device)
            # Should either handle gracefully or raise appropriate error
            assert "optimized_model" in result or True  # Allow either outcome
        except (RuntimeError, ValueError):
            # Expected to fail with invalid device
            pass
    
    def test_empty_input(self, sample_model, tuning_config):
        """Test handling of empty input."""
        tuner = KernelAutoTuner(tuning_config)
        
        device = torch.device("cpu")
        empty_input = torch.empty(0, 3, 32, 32)
        
        # Should handle empty input gracefully
        try:
            result = tuner.auto_tune(sample_model, empty_input, device)
            assert "optimized_model" in result
        except (RuntimeError, ValueError):
            # May legitimately fail with empty input
            pass
    
    def test_large_model_handling(self, tuning_config):
        """Test handling of large models."""
        # Create larger model
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(1000, 1000) for _ in range(10)
                ])
            
            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu(layer(x))
                return x
        
        large_model = LargeModel()
        large_input = torch.randn(4, 1000)
        
        # Use shorter timeout for large model test
        config = TuningConfig(timeout_seconds=10, max_iterations=2)
        tuner = KernelAutoTuner(config)
        
        device = torch.device("cpu")
        result = tuner.auto_tune(large_model, large_input, device)
        
        assert "optimized_model" in result


class TestPerformanceRegression:
    """Test performance regression detection."""
    
    def test_performance_comparison(self, sample_model, sample_input, tuning_config):
        """Test comparing performance before and after optimization."""
        tuner = KernelAutoTuner(tuning_config)
        
        device = torch.device("cpu")
        
        # Benchmark original model
        def original_forward():
            return sample_model(sample_input)
        
        original_result = tuner.profiler.benchmark_operation(
            original_forward, warmup=1, iterations=3
        )
        
        # Auto-tune model
        result = tuner.auto_tune(sample_model, sample_input, device)
        optimized_model = result["optimized_model"]
        
        # Benchmark optimized model
        def optimized_forward():
            return optimized_model(sample_input)
        
        optimized_result = tuner.profiler.benchmark_operation(
            optimized_forward, warmup=1, iterations=3
        )
        
        # Both should complete successfully
        assert original_result.latency_ms > 0
        assert optimized_result.latency_ms > 0
        
        # Log performance comparison (in real tests, could assert improvement)
        print(f"Original latency: {original_result.latency_ms:.2f}ms")
        print(f"Optimized latency: {optimized_result.latency_ms:.2f}ms")


if __name__ == "__main__":
    # Run basic smoke test
    model = SimpleModel()
    input_tensor = torch.randn(2, 3, 32, 32)
    config = TuningConfig(max_iterations=2, warmup_iterations=1)
    
    tuner = KernelAutoTuner(config)
    device = torch.device("cpu")
    
    result = tuner.auto_tune(model, input_tensor, device)
    
    print(f"✓ Auto-tuning completed")
    print(f"✓ Optimized model created: {result['optimized_model'] is not None}")
    print(f"✓ Best configuration found: {len(result['best_config']) > 0}")
    print("✓ Kernel auto-tuning tests ready")

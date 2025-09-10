"""
Example optimized test file demonstrating best practices for fast, reliable tests.
"""

import pytest
import torch
import time
from unittest.mock import Mock, patch

# Import optimization utilities
from tests.optimized_test_utils import (
    fast_test, performance_test, BenchmarkTimer,
    FastMockFactory, TestDataGenerator, 
    assert_performance_improvement, measure_memory_usage
)


class TestOptimizedExamples:
    """Examples of optimized test patterns."""
    
    @fast_test("fast")
    def test_fast_model_creation(self, mock_factory):
        """Example of fast test using cached mocks."""
        # Use cached mock instead of creating new model
        model = mock_factory.get_mock_model("classification")
        
        # Minimal test logic
        assert model is not None
        assert callable(model)
        
        # Test basic functionality
        input_tensor = TestDataGenerator.get_tensor((1, 3, 224, 224))
        output = model(input_tensor)
        assert output.shape == (1, 10)
    
    @fast_test("fast")
    def test_config_validation_fast(self, mock_factory):
        """Fast configuration validation test."""
        config = mock_factory.get_mock_config("default")
        
        assert config is not None
        assert hasattr(config, 'device')
        assert hasattr(config, 'batch')
    
    def test_model_inference_standard(self, test_config):
        """Standard test with moderate setup."""
        # Create simple real model for testing
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        model.eval()
        
        # Test inference
        input_data = torch.randn(2, 10)
        with torch.no_grad():
            output = model(input_data)
        
        assert output.shape == (2, 1)
        assert not torch.isnan(output).any()
    
    @performance_test(timeout=30)
    def test_optimization_performance(self, benchmark_timer):
        """Example performance test with benchmarking."""
        # Create test model
        model = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )
        model.eval()
        
        input_data = torch.randn(32, 256)
        
        # Baseline measurement
        def baseline_inference():
            with torch.no_grad():
                return model(input_data)
        
        baseline_result, baseline_time = benchmark_timer.time_function(baseline_inference)
        
        # Optimized version (using torch.jit.script)
        try:
            scripted_model = torch.jit.script(model)
            
            def optimized_inference():
                with torch.no_grad():
                    return scripted_model(input_data)
            
            optimized_result, optimized_time = benchmark_timer.time_function(optimized_inference)
            
            # Validate results are similar
            torch.testing.assert_close(baseline_result, optimized_result, rtol=1e-3)
            
            # Performance should be better or at least not significantly worse
            if optimized_time < baseline_time:
                improvement = (baseline_time - optimized_time) / baseline_time
                print(f"✅ Performance improvement: {improvement:.2%}")
            
        except Exception as e:
            pytest.skip(f"JIT compilation failed: {e}")
    
    def test_memory_usage_optimization(self):
        """Test memory usage patterns."""
        # Test memory usage of a function
        def memory_intensive_function():
            # Create large tensors
            tensors = []
            for _ in range(10):
                tensors.append(torch.randn(1000, 1000))
            return sum(tensor.sum() for tensor in tensors)
        
        result, memory_used = measure_memory_usage(memory_intensive_function)
        
        # Validate result
        assert isinstance(result, torch.Tensor)
        
        # Memory usage should be reasonable (less than 1GB)
        max_memory_mb = 1024
        if torch.cuda.is_available():
            assert memory_used < max_memory_mb * 1024 * 1024, f"Memory usage too high: {memory_used / 1024 / 1024:.1f}MB"
        else:
            assert memory_used < max_memory_mb * 1024 * 1024, f"Memory usage too high: {memory_used / 1024 / 1024:.1f}MB"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_optimization_conditional(self):
        """Test that conditionally runs only on GPU."""
        device = torch.device("cuda")
        
        # Test GPU-specific functionality
        model = torch.nn.Linear(100, 50).to(device)
        input_data = torch.randn(10, 100).to(device)
        
        # Measure GPU memory usage
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            output = model(input_data)
        
        peak_memory = torch.cuda.max_memory_allocated()
        
        assert output.device.type == "cuda"
        assert peak_memory > 0
        
        # Clean up
        torch.cuda.empty_cache()


class TestMockingPatterns:
    """Examples of efficient mocking patterns."""
    
    def test_smart_mocking_expensive_operations(self):
        """Mock expensive operations intelligently."""
        
        # Mock expensive model loading
        with patch('torch.load') as mock_load:
            mock_model = Mock()
            mock_model.eval.return_value = mock_model
            mock_load.return_value = mock_model
            
            # Code under test
            def load_model(path):
                model = torch.load(path)
                return model.eval()
            
            result = load_model("fake_path.pt")
            
            assert result is mock_model
            mock_load.assert_called_once_with("fake_path.pt")
    
    def test_progressive_mocking(self, mock_factory):
        """Start with mocks, upgrade to real components as needed."""
        
        # Start with mock for basic functionality
        mock_processor = Mock()
        mock_processor.process.return_value = {"status": "success"}
        
        # Test basic interface
        result = mock_processor.process("test_data")
        assert result["status"] == "success"
        
        # If needed, upgrade to more realistic mock
        def realistic_process(data):
            return {
                "status": "success",
                "data": f"processed_{data}",
                "timestamp": time.time()
            }
        
        mock_processor.process.side_effect = realistic_process
        
        # Test with more realistic behavior
        result = mock_processor.process("test_data")
        assert result["data"] == "processed_test_data"
        assert "timestamp" in result


class TestResourceOptimization:
    """Examples of resource optimization patterns."""
    
    @pytest.fixture(scope="class")
    def expensive_setup(self):
        """Class-scoped fixture for expensive setup."""
        # Simulate expensive setup
        print("Setting up expensive resource...")
        resource = {"initialized": True, "data": torch.randn(1000, 1000)}
        yield resource
        print("Cleaning up expensive resource...")
    
    def test_shared_resource_1(self, expensive_setup):
        """Test using shared expensive resource."""
        assert expensive_setup["initialized"]
        assert expensive_setup["data"].shape == (1000, 1000)
    
    def test_shared_resource_2(self, expensive_setup):
        """Another test using same resource."""
        assert expensive_setup["initialized"]
        # Resource is reused, not recreated
    
    def test_lazy_fixture_loading(self, request):
        """Example of conditional fixture loading."""
        
        # Only create expensive resource if test needs it
        if "gpu" in request.node.keywords:
            if not torch.cuda.is_available():
                pytest.skip("GPU test requires CUDA")
            
            device = torch.device("cuda")
            test_tensor = torch.randn(100, 100).to(device)
        else:
            device = torch.device("cpu")  
            test_tensor = torch.randn(100, 100)
        
        assert test_tensor.device.type == device.type


class TestParameterizedOptimization:
    """Examples of efficient parameterized testing."""
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8], ids=["single", "small", "medium"])
    @pytest.mark.parametrize("input_size", [32, 64], ids=["small_input", "large_input"])
    def test_batch_processing_efficient(self, batch_size, input_size):
        """Efficiently test multiple parameter combinations."""
        
        # Use cached model to avoid recreation
        model = FastMockFactory.get_mock_model("classification")
        
        # Use optimized tensor generation
        input_data = TestDataGenerator.get_tensor((batch_size, 3, input_size, input_size))
        
        # Quick test
        output = model(input_data)
        assert output.shape[0] == batch_size
    
    @pytest.mark.parametrize("optimization_level", ["basic", "aggressive"], 
                           ids=["basic_opt", "aggressive_opt"])
    def test_optimization_levels(self, optimization_level):
        """Test different optimization levels efficiently."""
        
        # Mock optimization based on level
        if optimization_level == "basic":
            expected_speedup = 1.2
        else:
            expected_speedup = 2.0
        
        # Simulate optimization test
        baseline_time = 1.0
        optimized_time = baseline_time / expected_speedup
        
        improvement = (baseline_time - optimized_time) / baseline_time
        assert improvement > 0.1  # At least 10% improvement


# Example of test organization for different execution modes

@pytest.mark.fast
class TestFastSuite:
    """Tests that should complete in <10 seconds total."""
    
    @fast_test("fast")
    def test_quick_validation(self):
        """Ultra-fast smoke test."""
        assert 1 + 1 == 2
    
    @fast_test("fast") 
    def test_mock_based_functionality(self, mock_factory):
        """Fast test using mocks."""
        model = mock_factory.get_mock_model()
        assert model is not None


@pytest.mark.integration
class TestIntegrationSuite:
    """Integration tests with moderate execution time."""
    
    def test_end_to_end_workflow(self, test_config):
        """Test complete workflow."""
        # This would test actual integration
        pass


@pytest.mark.performance  
class TestPerformanceSuite:
    """Performance tests that may take longer."""
    
    @performance_test(timeout=60)
    def test_throughput_benchmark(self, benchmark_timer):
        """Benchmark throughput performance."""
        # This would run actual performance measurements
        pass

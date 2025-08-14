# 🧪 Testing Documentation

This guide covers the comprehensive test suite for the PyTorch Inference Framework, including test structure, running tests, and contributing new tests.

## 📊 Test Overview

The framework includes a robust test suite with:
- **2000+ test cases** across all modules
- **90%+ code coverage** on critical paths
- **Unit, integration, and performance tests**
- **Mock implementations** for optional dependencies
- **CI/CD integration** with automated testing

## 🏗️ Test Structure

```
tests/
├── conftest.py                          # Shared fixtures and configuration
├── __init__.py                          # Package initialization
├── README.md                            # Testing documentation
├── fixtures/                            # Test data and fixtures
├── unit/                                # Unit tests for individual modules
│   ├── test_config.py                   # Configuration system (200+ tests)
│   ├── test_base_model.py               # Model management (250+ tests)  
│   ├── test_inference_engine.py         # Async inference (300+ tests)
│   ├── test_optimizers.py               # Optimization modules (350+ tests)
│   ├── test_adapters.py                 # Model adapters (200+ tests)
│   ├── test_enterprise.py               # Enterprise features (250+ tests)
│   ├── test_utils.py                    # Utility modules (150+ tests)
│   └── test_framework.py                # Main framework (400+ tests)
├── integration/                         # Integration tests
│   └── test_framework_integration.py    # End-to-end workflows (600+ tests)
├── models/                              # Test model utilities
│   ├── create_test_models.py            # Model download/creation
│   ├── model_loader.py                  # Model loading utilities
│   └── README.md                        # Model documentation
└── run_tests.py                         # Test runner script
```

## 🚀 Running Tests

### Quick Start

```bash
# Install test dependencies
uv sync --extra dev

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=framework

# Run specific test categories
uv run pytest -m unit              # Unit tests only
uv run pytest -m integration       # Integration tests only
uv run pytest -m "not slow"        # Skip slow tests
```

### Using Test Runner Script

```bash
# Run all tests
python run_tests.py all

# Run only unit tests
python run_tests.py unit

# Run only integration tests  
python run_tests.py integration

# Run with coverage reporting
python run_tests.py coverage

# Run specific test file
python run_tests.py unit --test-file test_config.py

# Verbose output
python run_tests.py all --verbose

# Performance benchmarks
python run_tests.py performance
```

### Using Helper Scripts

**Windows:**
```cmd
test.bat install-dev           # Install dependencies
test.bat test                  # Run all tests
test.bat coverage             # Run with coverage
test.bat lint                 # Run code quality checks
```

**Unix/Linux/macOS:**
```bash
make install-dev              # Install dependencies
make test                     # Run all tests  
make coverage                 # Run with coverage
make lint                     # Run code quality checks
```

## 🏷️ Test Markers

Tests are categorized using pytest markers:

### Performance Markers
- `unit` - Fast, isolated unit tests
- `integration` - End-to-end integration tests
- `slow` - Tests taking >5 seconds
- `benchmark` - Performance benchmarks

### Technology Markers
- `gpu` - Tests requiring GPU/CUDA
- `tensorrt` - Tests requiring TensorRT
- `onnx` - Tests requiring ONNX runtime
- `enterprise` - Enterprise feature tests

### Functional Markers
- `smoke` - Quick validation tests
- `regression` - Regression tests
- `security` - Security-related tests
- `api` - API endpoint tests
- `model` - Tests requiring real models
- `mock` - Tests using only mock objects

### Running Specific Categories

```bash
# Run only fast tests
uv run pytest -m "not slow and not gpu"

# Run GPU tests
uv run pytest -m gpu

# Run enterprise tests
uv run pytest -m enterprise

# Run benchmarks only
uv run pytest -m benchmark --benchmark-only

# Combine markers
uv run pytest -m "unit and not slow"
```

## 📋 Test Categories

### Unit Tests (`tests/unit/`)

#### Configuration Tests (`test_config.py`)
Tests the configuration management system:

```python
class TestDeviceConfig:
    """Test device configuration validation and conversion"""
    
    def test_device_detection(self):
        """Test automatic device detection"""
        config = DeviceConfig(device_type=DeviceType.AUTO)
        assert config.get_resolved_device() in ["cpu", "cuda", "mps"]
    
    def test_invalid_device_handling(self):
        """Test handling of invalid device specifications"""
        with pytest.raises(ValueError):
            DeviceConfig(device_type="invalid_device")

class TestConfigManager:
    """Test configuration manager functionality"""
    
    def test_environment_variable_override(self, monkeypatch):
        """Test environment variable precedence"""
        monkeypatch.setenv("BATCH_SIZE", "32")
        config_manager = ConfigManager()
        assert config_manager.get("BATCH_SIZE", default=4) == 32
    
    def test_yaml_configuration_loading(self, tmp_path):
        """Test YAML configuration file parsing"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
        batch:
          batch_size: 16
        """)
        
        config_manager = ConfigManager(config_file=config_file)
        inference_config = config_manager.get_inference_config()
        assert inference_config.batch.batch_size == 16
```

#### Model Management Tests (`test_base_model.py`)
Tests model loading and management:

```python
class TestBaseModel:
    """Test base model abstract class"""
    
    def test_prediction_interface(self, simple_model):
        """Test model prediction interface"""
        model = MockModel(simple_model)
        result = model.predict(torch.randn(1, 10))
        assert result is not None
        assert isinstance(result, torch.Tensor)

class TestModelManager:
    """Test model manager functionality"""
    
    def test_model_registration(self, model_manager, simple_model):
        """Test model registration and retrieval"""
        model_manager.register_model("test_model", simple_model)
        retrieved_model = model_manager.get_model("test_model")
        assert retrieved_model is not None
    
    def test_memory_usage_tracking(self, model_manager, complex_model):
        """Test memory usage monitoring"""
        initial_memory = model_manager.get_memory_usage()
        model_manager.register_model("memory_test", complex_model)
        final_memory = model_manager.get_memory_usage()
        assert final_memory > initial_memory
```

#### Inference Engine Tests (`test_inference_engine.py`)
Tests async inference capabilities:

```python
class TestInferenceEngine:
    """Test async inference engine"""
    
    @pytest.mark.asyncio
    async def test_async_prediction(self, inference_engine, sample_input):
        """Test basic async prediction"""
        result = await inference_engine.predict_async(sample_input)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, inference_engine):
        """Test handling multiple concurrent requests"""
        inputs = [torch.randn(1, 10) for _ in range(10)]
        tasks = [inference_engine.predict_async(inp) for inp in inputs]
        results = await asyncio.gather(*tasks)
        assert len(results) == 10
        assert all(r is not None for r in results)
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, inference_engine):
        """Test dynamic batching functionality"""
        batch_inputs = [torch.randn(1, 10) for _ in range(5)]
        
        # Submit requests close together for batching
        start_time = time.time()
        tasks = [inference_engine.predict_async(inp) for inp in batch_inputs]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Should be faster than individual requests due to batching
        assert len(results) == 5
        assert end_time - start_time < 1.0  # Should be fast due to batching
```

#### Optimizer Tests (`test_optimizers.py`)
Tests all optimization modules:

```python
class TestTensorRTOptimizer:
    """Test TensorRT optimization (with mocks for CI)"""
    
    def test_tensorrt_optimization_mock(self, simple_model):
        """Test TensorRT optimization with mock"""
        optimizer = MockTensorRTOptimizer()
        optimized_model = optimizer.optimize(simple_model)
        assert optimized_model is not None
        assert optimizer.get_optimization_info()["speedup"] > 1.0
    
    @pytest.mark.gpu
    @pytest.mark.tensorrt
    def test_real_tensorrt_optimization(self, simple_model):
        """Test real TensorRT optimization (requires GPU)"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            import tensorrt
            optimizer = TensorRTOptimizer()
            optimized_model = optimizer.optimize(simple_model)
            assert optimized_model is not None
        except ImportError:
            pytest.skip("TensorRT not available")

class TestQuantizationOptimizer:
    """Test quantization optimization"""
    
    def test_dynamic_quantization(self, simple_model):
        """Test dynamic quantization"""
        optimizer = QuantizationOptimizer()
        quantized_model = optimizer.quantize_dynamic(simple_model)
        
        # Check model size reduction
        original_size = sum(p.numel() * p.element_size() for p in simple_model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
        
        # Should be smaller (though exact reduction depends on model)
        assert quantized_size <= original_size
    
    def test_quantization_accuracy(self, simple_model):
        """Test quantization maintains reasonable accuracy"""
        optimizer = QuantizationOptimizer()
        quantized_model = optimizer.quantize_dynamic(simple_model)
        
        # Test inputs
        test_input = torch.randn(10, 10)
        original_output = simple_model(test_input)
        quantized_output = quantized_model(test_input)
        
        # Should maintain similar outputs
        mse = torch.nn.functional.mse_loss(original_output, quantized_output)
        assert mse < 0.1  # Allow some quantization error
```

### Integration Tests (`tests/integration/`)

#### End-to-End Tests (`test_framework_integration.py`)
Tests complete workflows:

```python
class TestFrameworkIntegration:
    """Test complete framework integration"""
    
    @pytest.mark.asyncio
    async def test_complete_async_workflow(self, temp_model_dir):
        """Test complete async inference workflow"""
        # Create and save a test model
        model = torch.nn.Linear(10, 5)
        model_path = temp_model_dir / "test_model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Initialize framework
        config = InferenceConfig(
            model_path=str(model_path),
            device=DeviceConfig(device_type=DeviceType.CPU),
            batch=BatchConfig(batch_size=4, max_batch_size=8)
        )
        
        framework = TorchInferenceFramework(config=config)
        await framework.initialize()
        
        # Test predictions
        test_inputs = [torch.randn(1, 10) for _ in range(10)]
        
        # Single predictions
        result = await framework.predict_async(test_inputs[0])
        assert result is not None
        assert result.shape == (1, 5)
        
        # Batch predictions
        batch_results = await framework.predict_batch_async(test_inputs[:5])
        assert len(batch_results) == 5
        
        # Concurrent predictions
        tasks = [framework.predict_async(inp) for inp in test_inputs]
        concurrent_results = await asyncio.gather(*tasks)
        assert len(concurrent_results) == 10
        
        # Cleanup
        await framework.cleanup()
    
    def test_optimization_pipeline(self, temp_model_dir):
        """Test optimization pipeline integration"""
        # Create test model
        model = create_test_model("resnet_like")
        model_path = temp_model_dir / "optimization_test.pt"
        torch.save(model.state_dict(), model_path)
        
        # Test different optimization configurations
        optimizations = [
            {"enable_jit": True},
            {"enable_quantization": True},
            {"enable_jit": True, "enable_quantization": True},
        ]
        
        results = {}
        
        for i, opt_config in enumerate(optimizations):
            config = InferenceConfig(
                model_path=str(model_path),
                optimization=OptimizationConfig(**opt_config)
            )
            
            framework = TorchInferenceFramework(config=config)
            framework.initialize()
            
            # Benchmark performance
            test_input = torch.randn(4, 3, 224, 224)
            
            start_time = time.time()
            for _ in range(10):
                result = framework.predict(test_input)
            end_time = time.time()
            
            results[f"config_{i}"] = {
                "time": end_time - start_time,
                "config": opt_config,
                "output_shape": result.shape
            }
            
            framework.cleanup()
        
        # Verify all configurations worked
        assert len(results) == len(optimizations)
        for result in results.values():
            assert result["output_shape"] is not None
            assert result["time"] > 0
```

## 🛠️ Test Fixtures and Utilities

### Core Fixtures (`conftest.py`)

```python
@pytest.fixture
def simple_model():
    """Simple linear model for testing"""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )

@pytest.fixture
def complex_model():
    """More complex model for performance testing"""
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 10)
    )

@pytest.fixture
def inference_config():
    """Standard inference configuration for testing"""
    return InferenceConfig(
        device=DeviceConfig(device_type=DeviceType.CPU),
        batch=BatchConfig(batch_size=4, max_batch_size=8),
        optimization=OptimizationConfig(enable_jit=False)
    )

@pytest.fixture
async def inference_engine(simple_model, inference_config):
    """Configured inference engine for testing"""
    engine = InferenceEngine(inference_config)
    await engine.initialize()
    engine.load_model(simple_model, "test_model")
    yield engine
    await engine.cleanup()

@pytest.fixture
def temp_model_dir(tmp_path):
    """Temporary directory for model files"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir

@pytest.fixture
def mock_model_manager():
    """Pre-configured model manager with mock models"""
    manager = ModelManager()
    
    # Add some mock models
    for i in range(3):
        mock_model = MockModel(torch.nn.Linear(10, 5))
        manager.register_model(f"mock_model_{i}", mock_model)
    
    return manager
```

### Mock Classes

```python
class MockModel:
    """Realistic model behavior for testing"""
    
    def __init__(self, pytorch_model):
        self.model = pytorch_model
        self.model.eval()
        self.predict_count = 0
        self.total_inference_time = 0
    
    def predict(self, input_tensor):
        """Mock prediction with timing"""
        start_time = time.time()
        with torch.no_grad():
            result = self.model(input_tensor)
        end_time = time.time()
        
        self.predict_count += 1
        self.total_inference_time += (end_time - start_time)
        
        return result
    
    def get_statistics(self):
        """Get mock model statistics"""
        avg_time = self.total_inference_time / max(self.predict_count, 1)
        return {
            "predict_count": self.predict_count,
            "average_inference_time": avg_time,
            "total_time": self.total_inference_time
        }

class MockTensorRTOptimizer:
    """Mock TensorRT optimizer for testing"""
    
    def optimize(self, model):
        """Mock optimization that wraps model"""
        return OptimizedModelWrapper(model, speedup=3.5)
    
    def get_optimization_info(self):
        return {
            "optimizer": "TensorRT",
            "speedup": 3.5,
            "memory_reduction": 0.6
        }
```

## ⚡ Performance Testing

### Benchmark Tests

```python
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    def test_inference_latency_benchmark(self, benchmark, simple_model):
        """Benchmark inference latency"""
        test_input = torch.randn(1, 10)
        
        def inference():
            with torch.no_grad():
                return simple_model(test_input)
        
        result = benchmark(inference)
        assert result is not None
    
    def test_batch_throughput_benchmark(self, benchmark, simple_model):
        """Benchmark batch processing throughput"""
        batch_input = torch.randn(16, 10)
        
        def batch_inference():
            with torch.no_grad():
                return simple_model(batch_input)
        
        result = benchmark(batch_inference)
        assert result.shape == (16, 5)
    
    @pytest.mark.gpu
    def test_gpu_optimization_benchmark(self, benchmark):
        """Benchmark GPU optimization performance"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = torch.nn.Linear(1000, 1000).cuda()
        input_tensor = torch.randn(100, 1000).cuda()
        
        def gpu_inference():
            with torch.no_grad():
                return model(input_tensor)
        
        result = benchmark(gpu_inference)
        assert result.device.type == 'cuda'
```

### Running Benchmarks

```bash
# Run benchmark tests only
uv run pytest -m benchmark --benchmark-only

# Save benchmark results
uv run pytest -m benchmark --benchmark-json=benchmark.json

# Compare benchmarks
uv run pytest -m benchmark --benchmark-compare=benchmark.json

# Sort benchmarks by mean time
uv run pytest -m benchmark --benchmark-sort=mean
```

## 🔧 Test Configuration

### pytest.ini Configuration

```ini
[tool:pytest]
minversion = 6.0
addopts = -ra -q --strict-markers --tb=short
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*

markers =
    unit: Fast, isolated unit tests
    integration: Slower end-to-end tests
    slow: Tests taking >5 seconds
    gpu: Tests requiring GPU/CUDA
    tensorrt: Tests requiring TensorRT
    onnx: Tests requiring ONNX runtime
    enterprise: Enterprise feature tests
    benchmark: Performance benchmarks
    smoke: Quick validation tests
    regression: Regression tests
    security: Security-related tests
    api: API endpoint tests
    model: Tests requiring real models
    mock: Tests using only mock objects

filterwarnings =
    ignore::DeprecationWarning:torch.*
    ignore::UserWarning:transformers.*
    error::RuntimeWarning
    error::ImportWarning

asyncio_mode = auto
asyncio_default_fixture_loop_scope = function

timeout = 300
timeout_method = thread

log_cli = true
log_cli_level = INFO
log_cli_format = %(asctime)s [%(levelname)8s] %(name)s: %(message)s
log_cli_date_format = %Y-%m-%d %H:%M:%S

junit_family = xunit2
```

### Coverage Configuration

```ini
[tool:coverage:run]
source = framework
omit = 
    tests/*
    framework/__pycache__/*
    */__pycache__/*
    setup.py
    */site-packages/*

[tool:coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    @abstractmethod

[tool:coverage:html]
directory = htmlcov
```

## 🚨 Debugging Tests

### Common Debugging Commands

```bash
# Stop on first failure
uv run pytest -x

# Run only failed tests from last run
uv run pytest --lf

# Run failed tests first
uv run pytest --ff

# Verbose debugging output
uv run pytest -vvv --tb=long --showlocals

# Debug specific test
uv run pytest tests/unit/test_config.py::TestDeviceConfig::test_device_detection -vvv

# Run with pdb on failure
uv run pytest --pdb

# Run tests matching pattern
uv run pytest -k "test_async"
```

### Test Environment Variables

```bash
# Set test environment
export ENVIRONMENT=test
export DEVICE=cpu
export LOG_LEVEL=DEBUG
export BATCH_SIZE=1
export TEST_TIMEOUT=600

# Run tests with environment
uv run pytest tests/
```

## 📊 Test Metrics and Reporting

### Coverage Reporting

```bash
# Generate HTML coverage report
uv run pytest --cov=framework --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### JUnit XML for CI

```bash
# Generate JUnit XML for CI systems
uv run pytest --junitxml=junit.xml

# CI-friendly output
uv run pytest --tb=short --junit-xml=junit.xml
```

### Test Statistics

The test runner provides detailed statistics:

```bash
python run_tests.py all --stats

# Example output:
# Test Results Summary:
# =====================
# Total Tests: 2,147
# Passed: 2,142 (99.8%)
# Failed: 0 (0.0%)
# Skipped: 5 (0.2%)
# Errors: 0 (0.0%)
# 
# Coverage: 94.2%
# Duration: 45.7 seconds
# 
# Test Categories:
# - Unit Tests: 1,547 (72.0%)
# - Integration Tests: 600 (28.0%)
# 
# Performance Tests: 25
# Average Inference Time: 2.3ms
# Memory Usage: 245MB peak
```

## 🔄 Continuous Integration

### GitHub Actions Integration

The test suite integrates with CI/CD:

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v3
    
    - name: Install dependencies
      run: uv sync --extra dev
    
    - name: Run tests
      run: uv run pytest --cov=framework --junit-xml=junit.xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml
```

## 📝 Contributing Tests

### Writing New Tests

1. **Follow naming conventions**: `test_<functionality>`
2. **Use existing fixtures**: Leverage `conftest.py` fixtures
3. **Include docstrings**: Explain test purpose clearly
4. **Test edge cases**: Include error conditions
5. **Use appropriate markers**: Categorize tests properly

### Test Review Checklist

- [ ] Tests cover new functionality completely
- [ ] Proper error case testing included
- [ ] No external dependencies without mocks
- [ ] Tests are deterministic and reproducible
- [ ] Performance impact is acceptable
- [ ] Documentation updated if needed
- [ ] Appropriate test markers applied

### Example New Test

```python
class TestNewFeature:
    """Test new feature functionality"""
    
    def test_basic_functionality(self, fixture):
        """Test basic feature operation"""
        # Arrange
        feature = NewFeature(config)
        
        # Act
        result = feature.execute(input_data)
        
        # Assert
        assert result is not None
        assert isinstance(result, ExpectedType)
    
    def test_error_handling(self, fixture):
        """Test error handling in edge cases"""
        feature = NewFeature(config)
        
        with pytest.raises(ExpectedError):
            feature.execute(invalid_input)
    
    @pytest.mark.slow
    def test_performance_requirements(self, fixture):
        """Test performance meets requirements"""
        feature = NewFeature(config)
        
        start_time = time.time()
        result = feature.execute(large_input)
        end_time = time.time()
        
        assert end_time - start_time < 1.0  # Must complete in 1s
        assert result.quality_score > 0.95  # Must maintain quality
```

## 🔗 Related Documentation

- **[Configuration Guide](configuration.md)** - Test configuration
- **[API Reference](api.md)** - Testing APIs
- **[Contributing Guide](contributing.md)** - Development workflow
- **[Troubleshooting Guide](troubleshooting.md)** - Common test issues

---

*Ready to contribute tests? Check out the [Contributing Guide](contributing.md) for development setup and workflow.*

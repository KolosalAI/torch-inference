# Advanced Optimization Tests

This directory contains comprehensive tests for the advanced optimization features implemented in the PyTorch inference framework, including the new **Enhanced Optimization Suite** (Vulkan, Numba, Enhanced JIT).

## Test Structure

### Unit Tests (`tests/unit/`)
- `test_int8_calibration.py` - Tests for INT8 calibration toolkit
- `test_kernel_autotuner.py` - Tests for kernel auto-tuning optimization
- `test_advanced_fusion.py` - Tests for advanced layer fusion
- `test_memory_optimizer.py` - Tests for enhanced memory optimization
- **`test_enhanced_optimizers.py`** - **NEW**: Tests for Vulkan, Numba, and Enhanced JIT optimizers
- **`test_framework.py`** - **UPDATED**: Tests for enhanced framework optimization methods

### Integration Tests (`tests/integration/`)
- `test_advanced_optimizations.py` - Integration tests for combined optimization usage
- **`test_enhanced_optimization_integration.py`** - **NEW**: End-to-end tests for enhanced optimization pipeline

### Performance Tests (`tests/performance/`)
- `test_optimization_benchmarks.py` - Performance benchmarks and regression tests

### Test Utilities
- **`test_enhanced_optimizers.py`** - **NEW**: Test runner script for enhanced optimization features
- **`ENHANCED_OPTIMIZATION_TESTS.md`** - **NEW**: Detailed documentation for enhanced optimization tests

## Features Tested

### 🆕 Enhanced Optimization Suite
- ✅ **Vulkan GPU Acceleration** - Cross-platform compute acceleration
  - Device detection and capability assessment
  - SPIR-V shader compilation and optimization
  - Compute context management
  - Tensor operation acceleration
  - Fallback mechanisms when unavailable
- ✅ **Numba JIT Compilation** - CPU/CUDA numerical optimization
  - Function JIT compilation for mathematical operations
  - CUDA kernel generation and optimization
  - CPU parallelization and vectorization
  - Target-specific optimization (CPU/CUDA/auto)
  - Performance benchmarking and validation
- ✅ **Enhanced JIT Optimizer** - Multi-backend JIT compilation
  - Automatic strategy selection based on hardware
  - Performance benchmarking across backends
  - Integration with TorchScript, Vulkan, and Numba
  - Fallback chain for maximum compatibility
  - Dynamic optimization based on model characteristics
- ✅ **Performance Optimizer** - Comprehensive optimization pipeline
  - Multiple optimization levels (conservative, balanced, aggressive)
  - Automatic optimization selection and application
  - Integration with all other optimizers
  - Performance monitoring and reporting

### 🔧 Utility Functions and Framework Integration
- ✅ **Optimization Recommendations** - Intelligent optimizer selection
  - Hardware-aware recommendations
  - Model-size specific suggestions
  - Target-specific optimization strategies
  - Automatic fallback and compatibility checking
- ✅ **Framework Integration** - Seamless integration with TorchInferenceFramework
  - `apply_automatic_optimizations()` method
  - `get_optimization_recommendations()` method
  - `get_available_optimizers()` method
  - Real-time optimization during model loading
  - Performance monitoring with optimization metrics

### 1. INT8 Calibration Toolkit
- ✅ Multiple calibration algorithms (entropy, percentile, KL-divergence, minmax)
- ✅ Activation statistics collection and analysis
- ✅ Quantization parameter generation
- ✅ Calibration quality validation
- ✅ Comprehensive reporting
- ✅ Caching and optimization
- ✅ CUDA support
- ✅ Error handling and edge cases

### 2. Kernel Auto-Tuner
- ✅ Hardware detection and profiling
- ✅ Multiple optimization strategies
- ✅ Batch size optimization
- ✅ JIT compilation optimization
- ✅ Thread count optimization
- ✅ Memory format optimization
- ✅ Performance benchmarking
- ✅ CUDA graphs support
- ✅ Mixed precision optimization
- ✅ Timeout and resource constraint handling

### 3. Advanced Layer Fusion
- ✅ FX-based graph analysis and tracing
- ✅ Multiple fusion patterns (Conv+BN, Conv+BN+ReLU, attention, residual)
- ✅ Custom pattern definition and matching
- ✅ Numerical validation
- ✅ Training mode preservation
- ✅ Optimization levels
- ✅ Fallback mechanisms
- ✅ Performance measurement

### 4. Enhanced Memory Optimizer
- ✅ Advanced memory pool with fragmentation prevention
- ✅ Background cleanup and defragmentation
- ✅ Memory usage monitoring and profiling
- ✅ Gradient checkpointing optimization
- ✅ Memory-efficient attention
- ✅ Automatic garbage collection
- ✅ Batch size optimization
- ✅ CUDA memory pool integration
- ✅ Concurrent access handling
- ✅ Memory leak detection

### 5. Integration Testing
- ✅ Sequential optimization pipeline
- ✅ Parallel optimization reporting
- ✅ Compatibility between different optimizations
- ✅ Rollback and error recovery
- ✅ Resource constraint handling
- ✅ End-to-end performance measurement

## Running Tests

### Prerequisites
```bash
pip install pytest torch torchvision psutil
# Optional for enhanced optimizers (graceful fallback if not available):
pip install vulkan numba
```

### 🆕 Run Enhanced Optimization Tests
```bash
# Quick test of enhanced optimization features
python test_enhanced_optimizers.py --basic

# Full enhanced optimization test suite
python test_enhanced_optimizers.py --all

# Performance benchmarks for enhanced optimizers
python test_enhanced_optimizers.py --performance
```

### Run All Tests
```bash
# From repository root
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/ -v

# Enhanced optimizer tests specifically
pytest tests/unit/test_enhanced_optimizers.py -v

# Framework integration tests
pytest tests/unit/test_framework.py::TestEnhancedOptimizationMethods -v

# Integration tests only
pytest tests/integration/ -v

# Enhanced optimization integration tests
pytest tests/integration/test_enhanced_optimization_integration.py -v

# Performance benchmarks
pytest tests/performance/ -v
```

### Run Specific Features
```bash
# Vulkan optimizer tests
pytest tests/unit/test_enhanced_optimizers.py::TestVulkanOptimizer -v

# Numba optimizer tests
pytest tests/unit/test_enhanced_optimizers.py::TestNumbaOptimizer -v

# Enhanced JIT tests
pytest tests/unit/test_enhanced_optimizers.py::TestEnhancedJITOptimizer -v

# Framework optimization methods
pytest tests/unit/test_framework.py::TestEnhancedOptimizationMethods -v

# INT8 calibration tests
pytest tests/unit/test_int8_calibration.py -v

# Kernel auto-tuning tests
pytest tests/unit/test_kernel_autotuner.py -v

# Advanced fusion tests
pytest tests/unit/test_advanced_fusion.py -v

# Memory optimization tests
pytest tests/unit/test_memory_optimizer.py -v
```

### CUDA Tests
```bash
# Run CUDA-specific tests (requires CUDA)
pytest tests/ -v -k "cuda"
```

### Performance Benchmarks
```bash
# Run performance benchmarks for enhanced optimizers
pytest tests/unit/test_enhanced_optimizers.py::TestOptimizerPerformance -v --performance

# Run baseline performance benchmarks
pytest tests/performance/test_optimization_benchmarks.py::TestBaseBenchmarks::test_baseline_performance -v -s
```

## Test Configuration

### Pytest Configuration
Tests use the existing `pytest.ini` configuration with the following key settings:
- Warnings filtered for common PyTorch/optimization warnings
- Test discovery from `tests/` directory
- Verbose output by default for optimization tests
- **NEW**: `--performance` flag for running performance-sensitive tests

### Test Fixtures
Common fixtures are available across all tests:
- `sample_model` - Standard models for testing
- `sample_input` - Standard input tensors
- `calibration_data` - Calibration datasets
- `profiler` - Performance measurement utilities
- **NEW Enhanced Fixtures**:
  - `enhanced_config` - Configuration with all optimizations enabled
  - `optimization_test_model` - Models suitable for optimization testing
  - `mock_vulkan_optimizer`, `mock_numba_optimizer` - Mock optimizers for testing
  - `framework_with_enhanced_model` - Framework with loaded model for testing

## Expected Behaviors

### 🆕 Enhanced Optimization Behaviors
- **Graceful Fallback**: When Vulkan/Numba unavailable, gracefully fall back to standard optimizations
- **Automatic Selection**: Framework automatically selects best optimizations based on hardware
- **Performance Validation**: Optimizations never significantly degrade performance
- **Error Recovery**: Failed optimizations don't break the framework functionality
- **Hardware Awareness**: Recommendations adapt to available hardware (CPU/GPU/specialized)

### Graceful Degradation
All optimization modules are designed to fail gracefully:
- If FX tracing fails, fallback to eager mode
- If CUDA is unavailable, run on CPU
- If optimization fails, return original model
- If calibration fails, use fallback quantization
- **NEW**: If Vulkan/Numba unavailable, use standard optimizations

### Performance Expectations
Benchmark tests validate that optimizations provide measurable improvements:
- **Enhanced Optimizations**: 20-60% potential speedup depending on hardware and model
- Memory optimization: Reduced memory usage and potential latency improvements
- Kernel tuning: Hardware-specific performance improvements
- Layer fusion: Reduced operator count and improved inference speed
- INT8 calibration: Maintained accuracy with potential speed improvements

### Error Handling
Tests verify proper error handling for:
- Invalid inputs and configurations
- Resource constraints (memory, timeout)
- Hardware limitations
- Unsupported operations
- Concurrent access
- **NEW**: Missing optional dependencies (Vulkan, Numba)
- **NEW**: Optimization failures and recovery

## Test Coverage

The test suite provides comprehensive coverage of:
- ✅ Core functionality of all optimization modules
- ✅ **NEW**: Enhanced optimization features (Vulkan, Numba, Enhanced JIT)
- ✅ **NEW**: Framework integration methods
- ✅ **NEW**: Automatic optimization selection and application
- ✅ Configuration and parameter validation
- ✅ Error conditions and edge cases
- ✅ Integration scenarios
- ✅ Performance characteristics
- ✅ Hardware-specific behaviors (CPU/CUDA/Vulkan)
- ✅ Memory management and leak detection
- ✅ Concurrent usage patterns
- ✅ **NEW**: Cross-platform compatibility
- ✅ **NEW**: Multi-backend optimization strategies

### 📊 Enhanced Test Metrics
- **65 new test methods** for enhanced optimization features
- **100% fallback coverage** when dependencies unavailable
- **Real-world scenario testing** (edge device, GPU server, production)
- **Performance regression protection** with automated benchmarking
- **Compatibility matrix validation** across different optimization combinations

## Continuous Integration

Tests are designed to run in CI environments:
- Tests skip CUDA functionality when CUDA is unavailable
- **NEW**: Tests skip Vulkan/Numba when not available (graceful fallback)
- Performance tests use appropriate timeouts
- Memory tests use conservative limits
- Integration tests handle optimization failures gracefully
- **NEW**: Mock-based testing ensures consistent CI results

## Troubleshooting

### Common Issues

**Enhanced Optimizer Import Errors**
- Expected when Vulkan SDK or Numba not installed
- Tests automatically skip and use fallback behavior
- Framework continues to work with standard optimizations

**FX Tracing Failures**
- Expected for complex models with control flow
- Tests skip gracefully when FX tracing is not supported

**CUDA Memory Issues**
- Tests use conservative memory allocations
- CUDA tests skip when CUDA is unavailable

**Performance Variations**
- Benchmark results may vary across hardware
- Tests use relative improvements rather than absolute values
- **NEW**: Enhanced optimizers tested with mock backends for consistency

**Import Errors**
- Ensure all dependencies are installed
- Check PyTorch version compatibility
- **NEW**: Enhanced optimizers are optional and gracefully degrade

### Debug Mode
Run tests with additional debugging:
```bash
pytest tests/ -v -s --tb=long

# Debug enhanced optimizers specifically
python test_enhanced_optimizers.py --basic -v
```

## Extending Tests

### Adding New Test Cases
1. Follow existing test patterns and naming conventions
2. Use appropriate fixtures for common setup
3. Include both positive and negative test cases
4. Add performance tests for new optimizations
5. Test error handling and edge cases
6. **NEW**: Include availability detection and fallback testing

### Performance Benchmarks
1. Use `PerformanceProfiler` for consistent measurements
2. Include baseline measurements for comparison
3. Test across different model sizes and complexities
4. Measure both latency and memory usage
5. Save results for regression detection
6. **NEW**: Include optimization overhead measurements

## Results and Reporting

Tests generate various outputs:
- Performance benchmark results in JSON format
- Optimization reports with detailed metrics
- Memory usage and fragmentation statistics
- Hardware profiling information
- **NEW**: Enhanced optimization selection reports
- **NEW**: Cross-platform compatibility matrices

These outputs can be used for:
- Performance regression detection
- Optimization effectiveness analysis
- Hardware-specific tuning
- Production deployment validation
- **NEW**: Optimization strategy comparison
- **NEW**: Hardware capability assessment

## 🎯 Quick Start for Enhanced Optimizations

```bash
# Test basic availability and functionality
python test_enhanced_optimizers.py --basic

# Run comprehensive enhanced optimization tests
python test_enhanced_optimizers.py --all

# Validate framework integration
pytest tests/unit/test_framework.py::TestEnhancedOptimizationMethods -v

# Test real-world scenarios
pytest tests/integration/test_enhanced_optimization_integration.py::TestRealWorldOptimizationScenarios -v
```

For detailed information about the enhanced optimization tests, see [`ENHANCED_OPTIMIZATION_TESTS.md`](ENHANCED_OPTIMIZATION_TESTS.md).

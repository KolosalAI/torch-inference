# Advanced Optimization Tests

This directory contains comprehensive tests for the advanced optimization features implemented in the PyTorch inference framework.

## Test Structure

### Unit Tests (`tests/unit/`)
- `test_int8_calibration.py` - Tests for INT8 calibration toolkit
- `test_kernel_autotuner.py` - Tests for kernel auto-tuning optimization
- `test_advanced_fusion.py` - Tests for advanced layer fusion
- `test_memory_optimizer.py` - Tests for enhanced memory optimization

### Integration Tests (`tests/integration/`)
- `test_advanced_optimizations.py` - Integration tests for combined optimization usage

### Performance Tests (`tests/performance/`)
- `test_optimization_benchmarks.py` - Performance benchmarks and regression tests

## Features Tested

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

# Integration tests only
pytest tests/integration/ -v

# Performance benchmarks
pytest tests/performance/ -v
```

### Run Specific Features
```bash
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
# Run performance benchmarks and save results
pytest tests/performance/test_optimization_benchmarks.py::TestBaseBenchmarks::test_baseline_performance -v -s
```

## Test Configuration

### Pytest Configuration
Tests use the existing `pytest.ini` configuration with the following key settings:
- Warnings filtered for common PyTorch/optimization warnings
- Test discovery from `tests/` directory
- Verbose output by default for optimization tests

### Test Fixtures
Common fixtures are available across all tests:
- `sample_model` - Standard models for testing
- `sample_input` - Standard input tensors
- `calibration_data` - Calibration datasets
- `profiler` - Performance measurement utilities

## Expected Behaviors

### Graceful Degradation
All optimization modules are designed to fail gracefully:
- If FX tracing fails, fallback to eager mode
- If CUDA is unavailable, run on CPU
- If optimization fails, return original model
- If calibration fails, use fallback quantization

### Performance Expectations
Benchmark tests validate that optimizations provide measurable improvements:
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

## Test Coverage

The test suite provides comprehensive coverage of:
- ✅ Core functionality of all optimization modules
- ✅ Configuration and parameter validation
- ✅ Error conditions and edge cases
- ✅ Integration scenarios
- ✅ Performance characteristics
- ✅ Hardware-specific behaviors (CPU/CUDA)
- ✅ Memory management and leak detection
- ✅ Concurrent usage patterns

## Continuous Integration

Tests are designed to run in CI environments:
- Tests skip CUDA functionality when CUDA is unavailable
- Performance tests use appropriate timeouts
- Memory tests use conservative limits
- Integration tests handle optimization failures gracefully

## Troubleshooting

### Common Issues

**FX Tracing Failures**
- Expected for complex models with control flow
- Tests skip gracefully when FX tracing is not supported

**CUDA Memory Issues**
- Tests use conservative memory allocations
- CUDA tests skip when CUDA is unavailable

**Performance Variations**
- Benchmark results may vary across hardware
- Tests use relative improvements rather than absolute values

**Import Errors**
- Ensure all dependencies are installed
- Check PyTorch version compatibility

### Debug Mode
Run tests with additional debugging:
```bash
pytest tests/ -v -s --tb=long
```

## Extending Tests

### Adding New Test Cases
1. Follow existing test patterns and naming conventions
2. Use appropriate fixtures for common setup
3. Include both positive and negative test cases
4. Add performance tests for new optimizations
5. Test error handling and edge cases

### Performance Benchmarks
1. Use `PerformanceProfiler` for consistent measurements
2. Include baseline measurements for comparison
3. Test across different model sizes and complexities
4. Measure both latency and memory usage
5. Save results for regression detection

## Results and Reporting

Tests generate various outputs:
- Performance benchmark results in JSON format
- Optimization reports with detailed metrics
- Memory usage and fragmentation statistics
- Hardware profiling information

These outputs can be used for:
- Performance regression detection
- Optimization effectiveness analysis
- Hardware-specific tuning
- Production deployment validation

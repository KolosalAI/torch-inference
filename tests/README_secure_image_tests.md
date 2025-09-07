# Secure Image Processing Tests

This directory contains comprehensive tests for the secure image processing system, which provides robust protection against various image-based attacks while maintaining high performance.

## Test Structure

### Unit Tests (`tests/unit/`)

- **`test_secure_image_processor.py`** - Tests for core security components
  - SecurityLevel configuration validation
  - SecureImageValidator threat detection capabilities
  - SecureImageSanitizer defense mechanisms  
  - SecureImagePreprocessor end-to-end processing
  - Error handling and edge cases

- **`test_secure_image_model.py`** - Tests for secure image model wrapper
  - Model initialization and configuration
  - Security level management
  - Image processing workflows
  - Statistics tracking and reporting

### Integration Tests (`tests/integration/`)

- **`test_secure_image_api.py`** - Tests for FastAPI endpoints
  - `/image/process/secure` - Secure image processing endpoint
  - `/image/validate/security` - Security validation endpoint  
  - `/image/security/stats` - Security statistics endpoint
  - `/image/models` - Available models endpoint
  - `/image/health` - Health check endpoint
  - Authentication and authorization
  - Error handling and response validation

### End-to-End Tests (`tests/integration/end_to_end/`)

- **`test_secure_image_e2e.py`** - Complete workflow testing
  - Normal image processing workflows
  - Adversarial attack detection and mitigation
  - Security level escalation scenarios
  - Batch processing capabilities
  - Error recovery mechanisms
  - Performance under load testing
  - Comprehensive security analysis

### Performance Tests (`tests/performance/`)

- **`test_secure_image_performance.py`** - Performance and scalability testing
  - Processing performance across security levels
  - Memory usage and leak detection
  - Concurrent processing capabilities
  - Large batch processing
  - API response time measurement
  - Resource utilization monitoring

## Test Fixtures and Utilities

### Test Image Generation (`tests/conftest.py`)

The test suite includes a comprehensive image generation system for creating test images with specific characteristics:

- **Normal Images** - Natural-looking test images for baseline testing
- **High Entropy Images** - Random noise patterns to test entropy analysis
- **Low Entropy Images** - Uniform patterns to test edge cases
- **Adversarial Patterns** - High-frequency noise simulating adversarial attacks
- **Steganography Patterns** - LSB modifications simulating hidden data
- **Text Images** - Images with embedded text content

### Performance Tracking

- **PerformanceTracker** - Monitors execution time and memory usage
- **Memory leak detection** - Validates system stability under load
- **Concurrent processing metrics** - Measures scalability characteristics

### Mock Components

- **Mock Security Manager** - Simulates enterprise security framework
- **Mock PyTorch Models** - Lightweight model substitutes for testing
- **Mock Image Processors** - Isolated component testing

## Security Test Coverage

### Threat Detection Testing

- **Adversarial Attack Detection**
  - High-frequency noise patterns
  - Pixel-level perturbations  
  - Statistical anomaly detection
  - Confidence score validation

- **Steganography Detection**
  - LSB (Least Significant Bit) modifications
  - Pattern analysis in pixel data
  - Entropy-based detection methods
  - Hidden data indicators

- **Format Exploitation Protection**
  - File header validation
  - Format spoofing attempts
  - Malformed image handling
  - Buffer overflow protection

- **Metadata Security**
  - EXIF data scanning
  - Suspicious metadata detection
  - Privacy information removal
  - Embedded script detection

### Defense Mechanism Testing

- **Image Sanitization**
  - Noise removal and filtering
  - Pixel value normalization
  - Metadata stripping
  - Format standardization

- **Security Level Enforcement**
  - Low security (basic validation)
  - Medium security (threat detection)
  - High security (active mitigation)
  - Maximum security (strict filtering)

- **Rate Limiting and DOS Protection**
  - Request frequency limits
  - Resource usage monitoring
  - Graceful degradation
  - Error recovery

## Running Tests

### Quick Start

```bash
# Run all secure image tests
python tests/run_secure_image_tests.py all

# Run only unit tests (fastest)
python tests/run_secure_image_tests.py unit

# Run with coverage reporting
python tests/run_secure_image_tests.py all --coverage

# Run performance tests
python tests/run_secure_image_tests.py performance
```

### Using pytest directly

```bash
# Unit tests only
pytest tests/unit/test_secure_image_processor.py tests/unit/test_secure_image_model.py -v

# Integration tests
pytest tests/integration/test_secure_image_api.py -v

# End-to-end tests
pytest tests/integration/end_to_end/test_secure_image_e2e.py -v

# Performance tests (requires --performance flag)
pytest tests/performance/test_secure_image_performance.py -m performance

# Security-focused tests
pytest -m "security or secure_image" -v

# Quick smoke tests
pytest -m smoke -x --tb=line
```

### Test Categories and Markers

Tests are organized using pytest markers:

- `@pytest.mark.security` - Security-focused tests
- `@pytest.mark.secure_image` - Secure image processing tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.adversarial_detection` - Adversarial attack tests
- `@pytest.mark.steganography_detection` - Steganography tests

## Test Configuration

### Environment Variables

```bash
# Enable debug logging for tests
export LOG_LEVEL=DEBUG

# Configure test timeouts
export PYTEST_TIMEOUT=30

# Enable real model loading (if available)
export USE_REAL_MODELS=true

# Configure security settings
export SECURITY_LEVEL=high
export ENABLE_THREAT_DETECTION=true
```

### Test Data

Test images are generated dynamically during test execution to ensure:
- Consistent test conditions
- No dependency on external files
- Controlled image characteristics
- Efficient memory usage

### Mock Configuration

Tests use extensive mocking to:
- Isolate components under test
- Simulate various attack scenarios
- Control external dependencies
- Ensure reproducible results

## Performance Benchmarks

### Expected Performance Characteristics

- **Unit Tests**: < 1 second per test
- **Integration Tests**: < 5 seconds per test  
- **End-to-End Tests**: < 30 seconds per test
- **Performance Tests**: < 60 seconds per test

### Memory Usage Targets

- **Small Images (224x224)**: < 50 MB peak memory
- **Large Images (1024x1024)**: < 200 MB peak memory
- **Batch Processing**: < 10 MB per image
- **Concurrent Processing**: Linear scaling with thread count

### Throughput Expectations

- **Low Security**: > 100 images/second
- **Medium Security**: > 50 images/second  
- **High Security**: > 20 images/second
- **Maximum Security**: > 10 images/second

## Continuous Integration

### GitHub Actions Integration

```yaml
- name: Run Secure Image Tests
  run: |
    python tests/run_secure_image_tests.py all --coverage --html-report
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
    
- name: Upload Test Results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: reports/
```

### Local Development

```bash
# Pre-commit hook for secure image tests
python tests/run_secure_image_tests.py smoke

# Full validation before pushing
python tests/run_secure_image_tests.py all --coverage
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure project root is in Python path
2. **Missing Dependencies**: Install PIL, numpy, pytest
3. **Timeout Errors**: Increase timeout for slow systems
4. **Memory Issues**: Reduce batch sizes for limited memory
5. **Permission Errors**: Check file system permissions for temp directories

### Debug Mode

```bash
# Run with verbose output and no capture
pytest tests/unit/test_secure_image_processor.py -v -s --tb=long

# Enable debug logging
pytest tests/unit/test_secure_image_processor.py --log-cli-level=DEBUG

# Run single test method
pytest tests/unit/test_secure_image_processor.py::TestSecureImageValidator::test_basic_validation -v
```

### Performance Issues

```bash
# Profile test execution
pytest tests/performance/test_secure_image_performance.py --profile

# Monitor memory usage
pytest tests/unit/test_secure_image_processor.py --memory-profile

# Check for resource leaks
pytest tests/unit/test_secure_image_processor.py --strict
```

## Contributing

### Adding New Tests

1. **Unit Tests**: Add to appropriate test class in `test_secure_image_processor.py` or `test_secure_image_model.py`
2. **Integration Tests**: Add to `test_secure_image_api.py` for API testing
3. **End-to-End Tests**: Add to `test_secure_image_e2e.py` for workflow testing
4. **Performance Tests**: Add to `test_secure_image_performance.py` for performance validation

### Test Guidelines

- Use descriptive test names that explain the scenario
- Include both positive and negative test cases
- Test error conditions and edge cases
- Mock external dependencies appropriately
- Include performance assertions where relevant
- Document complex test scenarios

### Code Coverage

Target coverage levels:
- **Unit Tests**: > 90%
- **Integration Tests**: > 80%
- **Overall**: > 85%

```bash
# Generate coverage report
python tests/run_secure_image_tests.py all --coverage

# View detailed coverage
open htmlcov_secure_image/index.html
```

## Security Considerations

### Test Data Security

- No real sensitive images in test data
- Generated test images use non-sensitive patterns
- Temporary files are cleaned up after tests
- No persistent storage of test images

### Attack Simulation

- Tests simulate attacks without using real malicious content
- Adversarial patterns are mathematically generated
- Steganography tests use benign hidden data
- No actual malware or exploits in test suite

### Validation Scope

Tests validate:
- Threat detection accuracy
- False positive rates
- Performance under attack
- System resilience
- Recovery mechanisms
- Security boundary enforcement

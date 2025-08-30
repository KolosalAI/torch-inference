# Test Timeout Configuration

This document explains how the 15-minute test timeout is configured and how to use it.

## Overview

The torch-inference framework is configured with multiple timeout mechanisms to prevent tests from running indefinitely:

1. **Individual test timeout**: 120 seconds (2 minutes) per test
2. **Global suite timeout**: 900 seconds (15 minutes) for the entire test suite

## Configuration Files

### pytest.ini
- Individual test timeout: `timeout = 120`
- Timeout method: `timeout_method = thread`
- Includes timeout options in `addopts`

### pyproject.toml
- Backup timeout configuration
- Individual test timeout: `timeout = 120`
- Timeout method: `timeout_method = "thread"`

## Usage

### Method 1: Python Script (Recommended)
```bash
# Run all tests with 15-minute timeout
uv run python run_tests_timeout.py

# Run with custom timeout (10 minutes)
uv run python run_tests_timeout.py --timeout 10

# Run specific tests with timeout
uv run python run_tests_timeout.py tests/unit/

# Run with coverage and timeout
uv run python run_tests_timeout.py --cov=framework --cov-report=html
```

### Method 2: PowerShell Script
```powershell
# Run with default 15-minute timeout
powershell -ExecutionPolicy Bypass -File run_tests_with_timeout.ps1

# Run with custom arguments
powershell -ExecutionPolicy Bypass -File run_tests_with_timeout.ps1 "tests/unit/ -v"
```

### Method 3: Makefile Commands
```bash
# Run all tests with timeout
make test-timeout

# Run unit tests with timeout
make test-unit-timeout

# Run integration tests with timeout
make test-integration-timeout

# Run CI tests with timeout and reports
make ci-test
```

### Method 4: Direct pytest (no global timeout)
```bash
# Individual test timeouts only (120 seconds per test)
uv run pytest

# With custom individual timeout
uv run pytest --timeout=60
```

## Timeout Behavior

### Individual Test Timeout (120 seconds)
- Each test method has a maximum execution time of 2 minutes
- Prevents individual tests from hanging
- Uses thread-based timeout method for compatibility

### Global Suite Timeout (15 minutes)
- Prevents the entire test suite from running longer than 15 minutes
- Gracefully terminates the pytest process
- Returns exit code 124 (standard timeout code)

### Exit Codes
- `0`: All tests passed
- `1`: Tests failed
- `124`: Timeout occurred
- `130`: User interrupted (Ctrl+C)

## Configuration Customization

### Change Individual Test Timeout
Edit `pytest.ini` or `pyproject.toml`:
```ini
timeout = 60  # 1 minute per test
```

### Change Global Timeout
Edit the timeout scripts or use command line:
```bash
# 10-minute timeout
uv run python run_tests_timeout.py --timeout 10

# 30-minute timeout
uv run python run_tests_timeout.py --timeout 30
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Run tests with timeout
  run: uv run python run_tests_timeout.py --cov=framework --cov-report=xml
  timeout-minutes: 15
```

### Azure DevOps
```yaml
- task: PythonScript@0
  displayName: 'Run tests with timeout'
  inputs:
    scriptSource: 'filePath'
    scriptPath: 'run_tests_timeout.py'
    arguments: '--cov=framework --cov-report=xml'
  timeoutInMinutes: 15
```

## Troubleshooting

### Tests Still Running Too Long
1. Check if individual tests are taking more than 120 seconds
2. Use `--durations=20` to see slowest tests
3. Consider marking slow tests with `@pytest.mark.slow`
4. Reduce timeout for faster feedback: `--timeout 60`

### Timeout Not Working
1. Ensure `pytest-timeout` is installed: `uv add pytest-timeout`
2. Check that timeout scripts have execution permissions
3. Verify PowerShell execution policy allows script execution

### False Timeouts
1. Increase timeout for complex test suites: `--timeout 20`
2. Use selective test execution: `pytest tests/unit/`
3. Consider running tests in smaller batches

## Best Practices

1. **Use timeout scripts**: Always use `test-timeout` commands for consistent behavior
2. **Monitor test duration**: Use `--durations=10` to identify slow tests
3. **Fail fast**: Use `--maxfail=5` to stop after several failures
4. **Parallel execution**: Use `-n auto` for faster test runs
5. **Mark slow tests**: Use `@pytest.mark.slow` for tests that need more time

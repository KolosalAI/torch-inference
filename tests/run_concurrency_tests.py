"""
Test runner script for concurrency optimization modules

This script provides convenient commands to run different test suites:
- Unit tests for individual components
- Integration tests for component interaction
- Performance benchmarking tests
- Complete test suite with coverage
"""

import subprocess
import sys
import os
import argparse
import time
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    
    elapsed = time.time() - start_time
    print(f"Command completed in {elapsed:.2f} seconds")
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0


def run_unit_tests(verbose=False):
    """Run unit tests for concurrency optimization modules"""
    print("🧪 Running unit tests for concurrency optimization modules...")
    
    test_files = [
        "tests/unit/test_concurrency_manager.py",
        "tests/unit/test_async_handler.py", 
        "tests/unit/test_batch_processor.py",
        "tests/unit/test_performance_optimizer.py",
        "tests/unit/test_optimization_integration.py"
    ]
    
    cmd = ["python", "-m", "pytest"] + test_files
    
    if verbose:
        cmd.extend(["-v", "--tb=long"])
    else:
        cmd.extend(["-q"])
    
    cmd.extend([
        "--asyncio-mode=auto",
        "-m", "unit"
    ])
    
    return run_command(cmd)


def run_integration_tests(verbose=False):
    """Run integration tests"""
    print("🔗 Running integration tests...")
    
    cmd = [
        "python", "-m", "pytest",
        "tests/integration/test_concurrency_optimization_integration.py"
    ]
    
    if verbose:
        cmd.extend(["-v", "--tb=long"])
    else:
        cmd.extend(["-q"])
    
    cmd.extend([
        "--asyncio-mode=auto", 
        "-m", "integration"
    ])
    
    return run_command(cmd)


def run_performance_tests(verbose=False):
    """Run performance benchmarking tests"""
    print("⚡ Running performance tests...")
    
    # Performance tests are marked with @pytest.mark.slow or have 'performance' in name
    cmd = [
        "python", "-m", "pytest",
        "tests/",
        "-k", "performance or load or benchmark"
    ]
    
    if verbose:
        cmd.extend(["-v", "--tb=long"])
    else:
        cmd.extend(["-q"])
    
    cmd.extend([
        "--asyncio-mode=auto",
        "--timeout=60"  # Longer timeout for performance tests
    ])
    
    return run_command(cmd)


def run_specific_component_tests(component, verbose=False):
    """Run tests for a specific component"""
    component_files = {
        "concurrency": "tests/unit/test_concurrency_manager.py",
        "async": "tests/unit/test_async_handler.py",
        "batch": "tests/unit/test_batch_processor.py", 
        "performance": "tests/unit/test_performance_optimizer.py",
        "integration": "tests/unit/test_optimization_integration.py"
    }
    
    if component not in component_files:
        print(f"❌ Unknown component: {component}")
        print(f"Available components: {', '.join(component_files.keys())}")
        return False
    
    print(f"🎯 Running tests for {component} component...")
    
    cmd = ["python", "-m", "pytest", component_files[component]]
    
    if verbose:
        cmd.extend(["-v", "--tb=long"])
    else:
        cmd.extend(["-q"])
    
    cmd.extend(["--asyncio-mode=auto"])
    
    return run_command(cmd)


def run_all_tests_with_coverage():
    """Run all tests with coverage reporting"""
    print("📊 Running all tests with coverage...")
    
    cmd = [
        "python", "-m", "pytest",
        "tests/unit/",
        "tests/integration/",
        "--cov=framework.core",
        "--cov-report=html",
        "--cov-report=term",
        "--cov-report=xml",
        "--asyncio-mode=auto",
        "-v"
    ]
    
    success = run_command(cmd)
    
    if success:
        print("✅ Coverage report generated in htmlcov/")
        print("📄 Coverage XML report: coverage.xml")
    
    return success


def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality"""
    print("💨 Running quick smoke test...")
    
    # Run a subset of fast tests from each component
    cmd = [
        "python", "-m", "pytest",
        "tests/unit/",
        "-k", "test_initialization or test_config or test_basic",
        "--asyncio-mode=auto",
        "-q"
    ]
    
    return run_command(cmd)


def validate_test_environment():
    """Validate that the test environment is properly set up"""
    print("🔍 Validating test environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check required packages
    required_packages = [
        "pytest",
        "pytest-asyncio", 
        "pytest-mock",
        "aiohttp",
        "psutil",
        "torch",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    # Check test files exist
    test_files = [
        "tests/unit/test_concurrency_manager.py",
        "tests/unit/test_async_handler.py",
        "tests/unit/test_batch_processor.py", 
        "tests/unit/test_performance_optimizer.py",
        "tests/unit/test_optimization_integration.py",
        "tests/integration/test_concurrency_optimization_integration.py"
    ]
    
    missing_files = []
    for test_file in test_files:
        if not Path(test_file).exists():
            missing_files.append(test_file)
    
    if missing_files:
        print(f"❌ Missing test files: {', '.join(missing_files)}")
        return False
    
    print("✅ Test environment validation passed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test runner for concurrency optimization modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --unit                    # Run unit tests
  python run_tests.py --integration             # Run integration tests
  python run_tests.py --component concurrency   # Test specific component
  python run_tests.py --performance             # Run performance tests  
  python run_tests.py --coverage                # Run all tests with coverage
  python run_tests.py --smoke                   # Quick smoke test
        """
    )
    
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--coverage", action="store_true", help="Run all tests with coverage")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    parser.add_argument("--component", choices=["concurrency", "async", "batch", "performance", "integration"], 
                       help="Run tests for specific component")
    parser.add_argument("--validate", action="store_true", help="Validate test environment")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--all", action="store_true", help="Run all test suites")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return 1
    
    # Validate environment first
    if args.validate or args.all:
        if not validate_test_environment():
            return 1
    
    success = True
    
    if args.smoke:
        success &= run_quick_smoke_test()
    
    if args.unit or args.all:
        success &= run_unit_tests(args.verbose)
    
    if args.integration or args.all:
        success &= run_integration_tests(args.verbose)
    
    if args.performance:
        success &= run_performance_tests(args.verbose)
    
    if args.component:
        success &= run_specific_component_tests(args.component, args.verbose)
    
    if args.coverage:
        success &= run_all_tests_with_coverage()
    
    if success:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Test runner script for secure image processing tests.

This script provides convenient commands to run different categories
of secure image processing tests with appropriate configurations.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return the exit code."""
    print(f"\n{'='*60}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n❌ Test execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error running command: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Run secure image processing tests")
    parser.add_argument(
        "test_type",
        choices=[
            "unit", "integration", "e2e", "performance", "security",
            "all", "fast", "full", "smoke"
        ],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true", 
        help="Run tests in parallel (requires pytest-xdist)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--html-report", 
        action="store_true", 
        help="Generate HTML test report"
    )
    parser.add_argument(
        "--benchmark", 
        action="store_true", 
        help="Run performance benchmarks"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    # Test path mappings
    test_paths = {
        "unit": [
            "tests/unit/test_secure_image_processor.py",
            "tests/unit/test_secure_image_model.py"
        ],
        "integration": [
            "tests/integration/test_secure_image_api.py"
        ],
        "e2e": [
            "tests/integration/end_to_end/test_secure_image_e2e.py"
        ],
        "performance": [
            "tests/performance/test_secure_image_performance.py"
        ],
        "security": [
            "tests/unit/test_secure_image_processor.py",
            "tests/unit/test_secure_image_model.py",
            "tests/integration/test_secure_image_api.py"
        ],
        "all": [
            "tests/unit/test_secure_image_processor.py",
            "tests/unit/test_secure_image_model.py", 
            "tests/integration/test_secure_image_api.py",
            "tests/integration/end_to_end/test_secure_image_e2e.py",
            "tests/performance/test_secure_image_performance.py"
        ],
        "fast": [
            "tests/unit/test_secure_image_processor.py",
            "tests/unit/test_secure_image_model.py"
        ],
        "full": [
            "tests/unit/test_secure_image_processor.py",
            "tests/unit/test_secure_image_model.py", 
            "tests/integration/test_secure_image_api.py",
            "tests/integration/end_to_end/test_secure_image_e2e.py"
        ],
        "smoke": [
            "tests/integration/test_secure_image_api.py::TestSecureImageAPI::test_image_health_endpoint",
            "tests/unit/test_secure_image_processor.py::TestSecureImageValidator::test_validate_image_security_success"
        ]
    }
    
    # Get test paths for the specified test type
    paths = test_paths.get(args.test_type, [])
    if not paths:
        print(f"❌ Unknown test type: {args.test_type}")
        return 1
    
    # Build pytest command
    cmd = base_cmd + paths
    
    # Add options based on arguments
    if args.verbose:
        cmd.extend(["-v", "-s"])
    else:
        cmd.extend(["--tb=short"])
    
    if args.parallel:
        cmd.extend(["-n", "auto", "--dist=worksteal"])
    
    if args.coverage:
        cmd.extend([
            "--cov=framework.processors.image",
            "--cov=framework.models.secure_image_model",
            "--cov-report=html:htmlcov_secure_image",
            "--cov-report=term-missing"
        ])
    
    if args.html_report:
        cmd.extend(["--html=reports/secure_image_tests.html", "--self-contained-html"])
    
    if args.benchmark:
        cmd.extend(["--benchmark-only", "--benchmark-autosave"])
    
    # Add markers based on test type
    if args.test_type == "performance":
        cmd.extend(["-m", "performance"])
    elif args.test_type == "security":
        cmd.extend(["-m", "security or secure_image"])
    elif args.test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif args.test_type == "smoke":
        cmd.extend(["-m", "smoke"])
    
    # Add timeout for longer test suites
    if args.test_type in ["all", "full", "performance"]:
        cmd.extend(["--timeout=30"])
    
    # Run the tests
    description_map = {
        "unit": "Unit Tests for Secure Image Processing",
        "integration": "Integration Tests for Secure Image API",
        "e2e": "End-to-End Tests for Complete Workflow",
        "performance": "Performance Tests for Secure Image Processing",
        "security": "Security-Focused Tests",
        "all": "Complete Test Suite",
        "fast": "Fast Unit Tests Only",
        "full": "Full Test Suite (excluding performance)",
        "smoke": "Smoke Tests for Quick Validation"
    }
    
    description = description_map.get(args.test_type, "Secure Image Processing Tests")
    exit_code = run_command(cmd, description)
    
    # Print summary
    print(f"\n{'='*60}")
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code: {exit_code}")
    
    if args.coverage and exit_code == 0:
        print("📊 Coverage report generated in htmlcov_secure_image/")
    
    if args.html_report and exit_code == 0:
        print("📋 HTML test report generated: reports/secure_image_tests.html")
    
    print(f"{'='*60}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

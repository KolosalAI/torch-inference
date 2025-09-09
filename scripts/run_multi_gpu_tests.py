#!/usr/bin/env python3
"""
Multi-GPU Test Runner Script

This script provides convenient commands for running multi-GPU tests
with different configurations and environments.
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class MultiGPUTestRunner:
    """Test runner for multi-GPU functionality."""
    
    def __init__(self):
        self.project_root = project_root
        self.test_commands = {
            "unit": self._get_unit_test_command(),
            "integration": self._get_integration_test_command(),
            "performance": self._get_performance_test_command(),
            "stress": self._get_stress_test_command(),
            "all": self._get_all_test_command(),
            "smoke": self._get_smoke_test_command(),
        }
    
    def _get_unit_test_command(self) -> List[str]:
        """Get command for unit tests."""
        return [
            "python", "-m", "pytest",
            "tests/unit/test_multi_gpu.py",
            "-v",
            "--tb=short",
            "-m", "not slow",
            "--disable-warnings"
        ]
    
    def _get_integration_test_command(self) -> List[str]:
        """Get command for integration tests."""
        return [
            "python", "-m", "pytest", 
            "tests/integration/test_multi_gpu_integration.py",
            "-v",
            "--tb=short",
            "-m", "integration"
        ]
    
    def _get_performance_test_command(self) -> List[str]:
        """Get command for performance tests."""
        return [
            "python", "-m", "pytest",
            "tests/integration/test_multi_gpu_integration.py",
            "-v",
            "--tb=short",
            "-m", "performance",
            "--timeout=30"
        ]
    
    def _get_stress_test_command(self) -> List[str]:
        """Get command for stress tests."""
        return [
            "python", "-m", "pytest",
            "tests/integration/test_multi_gpu_integration.py",
            "-v", 
            "--tb=short",
            "-m", "slow",
            "--timeout=60"
        ]
    
    def _get_all_test_command(self) -> List[str]:
        """Get command for all multi-GPU tests."""
        return [
            "python", "-m", "pytest",
            "tests/unit/test_multi_gpu.py",
            "tests/integration/test_multi_gpu_integration.py",
            "-v",
            "--tb=short"
        ]
    
    def _get_smoke_test_command(self) -> List[str]:
        """Get command for smoke tests."""
        return [
            "python", "-m", "pytest",
            "tests/unit/test_multi_gpu.py::TestMultiGPUConfig::test_valid_config",
            "tests/unit/test_multi_gpu.py::TestDevicePool::test_device_pool_initialization",
            "tests/integration/test_multi_gpu_integration.py::TestMultiGPUEndToEnd::test_multi_gpu_configuration_detection",
            "-v",
            "--tb=line",
            "-x"  # Stop on first failure
        ]
    
    def run_tests(self, test_type: str, coverage: bool = False, parallel: bool = False, 
                  verbose: bool = True, mock_gpu: bool = True) -> int:
        """
        Run tests with specified configuration.
        
        Args:
            test_type: Type of tests to run ('unit', 'integration', 'performance', 'stress', 'all', 'smoke')
            coverage: Whether to include coverage reporting
            parallel: Whether to run tests in parallel
            verbose: Whether to use verbose output
            mock_gpu: Whether to mock GPU availability (useful for CI/CD)
            
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        if test_type not in self.test_commands:
            print(f"Error: Unknown test type '{test_type}'")
            print(f"Available types: {', '.join(self.test_commands.keys())}")
            return 1
        
        # Start with base command
        cmd = self.test_commands[test_type].copy()
        
        # Add coverage options
        if coverage:
            cmd.extend([
                "--cov=framework.core.multi_gpu_manager",
                "--cov=framework.core.multi_gpu_strategies", 
                "--cov=framework.core.config",
                "--cov-report=html:htmlcov/multi_gpu",
                "--cov-report=term-missing:skip-covered",
                "--cov-fail-under=85"
            ])
        
        # Add parallel execution
        if parallel:
            cmd.extend(["-n", "auto", "--dist=loadfile"])
        
        # Configure verbosity
        if not verbose:
            cmd = [arg for arg in cmd if arg != "-v"]
            cmd.append("-q")
        
        # Set environment variables
        env = os.environ.copy()
        
        # Mock GPU environment for CI/CD
        if mock_gpu:
            env.update({
                "TORCH_INFERENCE_MOCK_GPU": "true",
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",  # Mock 4 GPUs
                "TORCH_INFERENCE_TEST_MODE": "true"
            })
        
        # Set working directory
        os.chdir(self.project_root)
        
        print(f"Running {test_type} tests...")
        print(f"Command: {' '.join(cmd)}")
        print(f"Working directory: {self.project_root}")
        
        if mock_gpu:
            print("Note: Running with mocked GPU environment")
        
        # Run the command
        try:
            result = subprocess.run(cmd, env=env, cwd=self.project_root)
            return result.returncode
        except KeyboardInterrupt:
            print("\nTest execution interrupted by user")
            return 130
        except Exception as e:
            print(f"Error running tests: {e}")
            return 1
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate test environment setup."""
        validation = {
            "python_path": sys.executable,
            "working_directory": str(self.project_root),
            "pytest_available": False,
            "torch_available": False,
            "test_files_exist": False,
            "errors": []
        }
        
        # Check pytest availability
        try:
            import pytest
            validation["pytest_available"] = True
            validation["pytest_version"] = pytest.__version__
        except ImportError:
            validation["errors"].append("pytest not available")
        
        # Check torch availability  
        try:
            import torch
            validation["torch_available"] = True
            validation["torch_version"] = torch.__version__
        except ImportError:
            validation["errors"].append("torch not available")
        
        # Check test files exist
        unit_test_file = self.project_root / "tests" / "unit" / "test_multi_gpu.py"
        integration_test_file = self.project_root / "tests" / "integration" / "test_multi_gpu_integration.py"
        
        if unit_test_file.exists() and integration_test_file.exists():
            validation["test_files_exist"] = True
        else:
            validation["errors"].append("test files missing")
        
        # Check framework imports
        try:
            from framework.core.config import MultiGPUConfig
            from framework.core.multi_gpu_manager import MultiGPUManager
            validation["framework_available"] = True
        except ImportError as e:
            validation["framework_available"] = False
            validation["errors"].append(f"framework import error: {e}")
        
        return validation
    
    def print_validation_report(self, validation: Dict[str, Any]):
        """Print environment validation report."""
        print("\n" + "="*60)
        print("MULTI-GPU TEST ENVIRONMENT VALIDATION")
        print("="*60)
        
        print(f"Python: {validation['python_path']}")
        print(f"Working Directory: {validation['working_directory']}")
        
        status_items = [
            ("Pytest Available", validation["pytest_available"]),
            ("PyTorch Available", validation["torch_available"]), 
            ("Test Files Exist", validation["test_files_exist"]),
            ("Framework Available", validation.get("framework_available", False))
        ]
        
        for name, status in status_items:
            status_str = "✅ YES" if status else "❌ NO"
            print(f"{name}: {status_str}")
            
            # Print versions if available
            if status:
                if name == "Pytest Available" and "pytest_version" in validation:
                    print(f"  Version: {validation['pytest_version']}")
                elif name == "PyTorch Available" and "torch_version" in validation:
                    print(f"  Version: {validation['torch_version']}")
        
        if validation["errors"]:
            print("\nErrors:")
            for error in validation["errors"]:
                print(f"  ❌ {error}")
        else:
            print("\n✅ Environment validation passed!")
        
        print("="*60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU Test Runner for torch-inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_multi_gpu_tests.py unit                    # Run unit tests
  python run_multi_gpu_tests.py integration --coverage # Integration with coverage
  python run_multi_gpu_tests.py all --parallel         # All tests in parallel
  python run_multi_gpu_tests.py smoke                  # Quick smoke tests
  python run_multi_gpu_tests.py validate               # Validate environment
        """
    )
    
    parser.add_argument(
        "command",
        choices=["unit", "integration", "performance", "stress", "all", "smoke", "validate"],
        help="Test command to run"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Include coverage reporting"
    )
    
    parser.add_argument(
        "--parallel", "-p", 
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce verbosity"
    )
    
    parser.add_argument(
        "--real-gpu",
        action="store_true",
        help="Use real GPU environment (default: mocked for CI/CD)"
    )
    
    args = parser.parse_args()
    
    runner = MultiGPUTestRunner()
    
    # Handle validation command
    if args.command == "validate":
        validation = runner.validate_environment()
        runner.print_validation_report(validation)
        return 0 if not validation["errors"] else 1
    
    # Handle test commands
    exit_code = runner.run_tests(
        test_type=args.command,
        coverage=args.coverage,
        parallel=args.parallel,
        verbose=not args.quiet,
        mock_gpu=not args.real_gpu
    )
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

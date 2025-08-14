"""
Test runner script for server endpoint tests.
This script provides an easy way to run the server endpoint tests.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_server_endpoint_tests():
    """Run server endpoint tests."""
    print("🚀 Running PyTorch Inference Framework Server Endpoint Tests")
    print("=" * 70)
    
    # Set the working directory to the project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Test commands to run
    test_commands = [
        {
            "name": "Basic Server Endpoint Tests",
            "cmd": [
                sys.executable, "-m", "pytest", 
                "tests/integration/test_server_endpoints.py::TestServerEndpoints",
                "-k", "not (performance or stress)",
                "-v", "--tb=line", "--maxfail=3"
            ],
            "description": "Tests all basic server endpoints including health, predict, models, etc."
        },
        {
            "name": "Performance Tests",
            "cmd": [
                sys.executable, "-m", "pytest",
                "tests/integration/test_server_endpoints.py::TestServerPerformance",
                "-v", "--tb=line", "--maxfail=2"
            ],
            "description": "Tests server performance characteristics"
        }
    ]
    
    results = {}
    
    for test_group in test_commands:
        print(f"\n📋 {test_group['name']}")
        print(f"   {test_group['description']}")
        print("-" * 70)
        
        try:
            result = subprocess.run(
                test_group["cmd"],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per test group
            )
            
            # Parse results
            output = result.stdout
            if "failed" in output.lower() and "passed" in output.lower():
                # Extract test results
                lines = output.split('\n')
                for line in lines:
                    if 'failed' in line and 'passed' in line:
                        results[test_group["name"]] = line.strip()
                        break
                else:
                    results[test_group["name"]] = f"Exit Code: {result.returncode}"
            elif "passed" in output.lower():
                lines = output.split('\n')
                for line in lines:
                    if 'passed' in line and ('warning' in line or '=' in line):
                        results[test_group["name"]] = line.strip()
                        break
                else:
                    results[test_group["name"]] = f"Exit Code: {result.returncode}"
            else:
                results[test_group["name"]] = f"Exit Code: {result.returncode}"
            
            if result.returncode == 0:
                print(f"✅ PASSED: {results[test_group['name']]}")
            else:
                print(f"❌ FAILED: {results[test_group['name']]}")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
            
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT: {test_group['name']} timed out after 2 minutes")
            results[test_group["name"]] = "TIMEOUT"
        except Exception as e:
            print(f"💥 ERROR: {test_group['name']} failed with error: {e}")
            results[test_group["name"]] = f"ERROR: {e}"
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    total_groups = len(test_commands)
    passed_groups = 0
    
    for test_group_name, result in results.items():
        if "passed" in result.lower() or "warning" in result.lower():
            status = "✅ PASSED"
            passed_groups += 1
        elif "timeout" in result.lower():
            status = "⏰ TIMEOUT"
        else:
            status = "❌ FAILED"
        
        print(f"{status:<12} {test_group_name}")
        print(f"             {result}")
    
    print("-" * 70)
    print(f"📈 Overall: {passed_groups}/{total_groups} test groups passed")
    
    if passed_groups == total_groups:
        print("🎉 All server endpoint tests completed successfully!")
        return 0
    elif passed_groups > 0:
        print("⚠️  Some tests passed, but there were issues with others.")
        return 1
    else:
        print("🚨 All test groups failed!")
        return 2


if __name__ == "__main__":
    exit_code = run_server_endpoint_tests()
    sys.exit(exit_code)

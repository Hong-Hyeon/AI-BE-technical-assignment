#!/usr/bin/env python3
"""
Test runner for the AI-BE Technical Assignment backend.

This script runs all unit tests for the analyze functionality and provides
a comprehensive summary of test results.

Usage:
    python run_tests.py [options]

Options:
    --verbose, -v    : Show verbose output
    --coverage, -c   : Generate coverage report
    --html          : Generate HTML coverage report
    --specific TEST : Run specific test file or test case
    --marker MARKER : Run tests with specific marker
    --parallel, -p  : Run tests in parallel
    --help, -h      : Show this help message
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_FILES = [
    "tests/test_analysis_router.py",
    "tests/test_analysis_schema.py", 
    "tests/test_analysis_integration.py"
]

def get_docker_compatible_pytest_args() -> List[str]:
    """Get pytest arguments compatible with Docker environment."""
    base_args = [
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker handling
        "--disable-warnings",  # Disable warnings for cleaner output
        "--no-header",  # Reduce output noise
        "--maxfail=5",  # Stop after 5 failures to save time
        "-ra",  # Show summary for all except passed
    ]
    
    # Try to add forked option if pytest-forked is available
    try:
        import pytest_forked
        base_args.append("--forked")
        print("✅ Using --forked option for better test isolation")
    except ImportError:
        print("ℹ️ pytest-forked not available, using standard execution")
    
    return base_args


PYTEST_BASE_ARGS = get_docker_compatible_pytest_args()


def print_banner(title: str, char: str = "=", width: int = 80):
    """Print a formatted banner."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def print_section(title: str, char: str = "-", width: int = 60):
    """Print a section header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    print_section("Checking Dependencies")
    
    required_packages = ["pytest", "pytest-asyncio", "fastapi", "pydantic"]
    missing_packages = []
    
    for package in required_packages:
        try:
            # Handle package name variations for import
            import_name = package.replace("-", "_")
            if package == "pytest-asyncio":
                import_name = "pytest_asyncio"
            
            __import__(import_name)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        print("Or install test dependencies: pip install -r test-requirements.txt")
        return False
    
    print("✅ All dependencies are available")
    return True


def setup_environment():
    """Set up test environment variables."""
    print_section("Setting up Test Environment")
    
    # Set test environment variables
    test_env_vars = {
        "ENVIRONMENT": "testing",
        "DEBUG": "false",
        "OPENAI_API_KEY": "test-key-for-testing",
        "DATABASE_URL": "sqlite:///test.db",
        "LOG_LEVEL": "WARNING",  # Reduce log noise during tests
        "PYTHONPATH": str(project_root),  # Ensure Python path is set
        "PYTEST_CURRENT_TEST": "",  # Clear any existing pytest state
        "PYTHONDONTWRITEBYTECODE": "1",  # Prevent .pyc files in Docker
        "PYTHONUNBUFFERED": "1"  # Ensure output is not buffered
    }
    
    for key, value in test_env_vars.items():
        os.environ[key] = value
        print(f"✅ Set {key}={value}")
    
    # Clean up any existing test artifacts
    cleanup_test_artifacts()
    
    print("✅ Test environment configured")


def cleanup_test_artifacts():
    """Clean up test artifacts to prevent conflicts."""
    artifacts_to_clean = [
        "test.db",
        ".pytest_cache",
        "__pycache__",
        "*.pyc"
    ]
    
    for artifact in artifacts_to_clean:
        artifact_path = project_root / artifact
        if artifact_path.exists():
            try:
                if artifact_path.is_file():
                    artifact_path.unlink()
                elif artifact_path.is_dir():
                    import shutil
                    shutil.rmtree(artifact_path, ignore_errors=True)
            except Exception:
                pass  # Ignore cleanup errors


def run_pytest(args: List[str], test_files: Optional[List[str]] = None, retry_on_failure: bool = True) -> Dict:
    """Run pytest with given arguments and return results."""
    cmd = ["python", "-m", "pytest"] + args
    
    if test_files:
        cmd.extend(test_files)
    else:
        cmd.extend(TEST_FILES)
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True,
            cwd=project_root
        )
        
        duration = time.time() - start_time
        
        # If tests failed and retry is enabled, try once more
        # This helps with Docker environment occasional race conditions
        if result.returncode != 0 and retry_on_failure:
            print("\n⚠️ Some tests failed. Retrying once to handle potential Docker race conditions...")
            print("=" * 60)
            
            retry_start = time.time()
            retry_result = subprocess.run(
                cmd,
                capture_output=False,
                text=True, 
                cwd=project_root
            )
            retry_duration = time.time() - retry_start
            
            if retry_result.returncode == 0:
                print("\n✅ Retry successful! All tests passed on second attempt.")
                return {
                    "returncode": retry_result.returncode,
                    "duration": duration + retry_duration,
                    "success": True,
                    "retry_attempted": True
                }
            else:
                print("\n❌ Retry also failed. There are genuine test failures.")
        
        return {
            "returncode": result.returncode,
            "duration": duration,
            "success": result.returncode == 0,
            "retry_attempted": False
        }
        
    except FileNotFoundError:
        print("❌ pytest not found. Install with: pip install pytest")
        return {
            "returncode": 1,
            "duration": 0,
            "success": False,
            "error": "pytest not found",
            "retry_attempted": False
        }
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return {
            "returncode": 1,
            "duration": 0,
            "success": False,
            "error": str(e),
            "retry_attempted": False
        }


def run_coverage_report(html: bool = False) -> Dict:
    """Generate coverage report."""
    print_section("Generating Coverage Report")
    
    # Install coverage if needed
    try:
        import coverage
        print("✅ Coverage package available")
    except ImportError:
        print("Installing coverage package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "coverage"], check=True)
    
    # Run tests with coverage
    coverage_args = [
        "--cov=routers",
        "--cov=schema", 
        "--cov=models",
        "--cov=workflows",
        "--cov-report=term-missing",
    ]
    
    if html:
        coverage_args.append("--cov-report=html:htmlcov")
        print("HTML coverage report will be generated in 'htmlcov/' directory")
    
    pytest_args = PYTEST_BASE_ARGS + coverage_args
    result = run_pytest(pytest_args)
    
    if html and result["success"]:
        html_path = project_root / "htmlcov" / "index.html"
        if html_path.exists():
            print(f"✅ HTML coverage report generated: {html_path}")
        else:
            print("⚠️ HTML coverage report not found")
    
    return result


def run_specific_tests(test_pattern: str) -> Dict:
    """Run specific tests matching the pattern."""
    print_section(f"Running Specific Tests: {test_pattern}")
    
    pytest_args = PYTEST_BASE_ARGS + ["-k", test_pattern, "-v"]
    return run_pytest(pytest_args)


def run_with_marker(marker: str) -> Dict:
    """Run tests with specific marker."""
    print_section(f"Running Tests with Marker: {marker}")
    
    pytest_args = PYTEST_BASE_ARGS + ["-m", marker, "-v"]
    return run_pytest(pytest_args)


def run_parallel_tests() -> Dict:
    """Run tests in parallel using pytest-xdist."""
    print_section("Running Tests in Parallel")
    
    try:
        import xdist
        print("✅ pytest-xdist available")
    except ImportError:
        print("Installing pytest-xdist for parallel execution...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest-xdist"], check=True)
    
    pytest_args = PYTEST_BASE_ARGS + ["-n", "auto", "-v"]
    return run_pytest(pytest_args)


def validate_test_files() -> bool:
    """Validate that all test files exist."""
    print_section("Validating Test Files")
    
    missing_files = []
    for test_file in TEST_FILES:
        file_path = project_root / test_file
        if file_path.exists():
            print(f"✅ {test_file}")
        else:
            print(f"❌ {test_file} - Not found")
            missing_files.append(test_file)
    
    if missing_files:
        print(f"\n❌ Missing test files: {missing_files}")
        return False
    
    print("✅ All test files found")
    return True


def print_summary(results: Dict):
    """Print test execution summary."""
    print_banner("TEST EXECUTION SUMMARY", "=", 80)
    
    if results["success"]:
        status_icon = "✅"
        status_text = "PASSED"
    else:
        status_icon = "❌" 
        status_text = "FAILED"
    
    print(f"\n{status_icon} Test Status: {status_text}")
    print(f"⏱️  Execution Time: {results['duration']:.2f} seconds")
    print(f"🔧 Return Code: {results['returncode']}")
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
    
    print("\n" + "=" * 80)
    
    if results["success"]:
        print("🎉 All tests passed! The analyze functionality is working correctly.")
        print("\nNext steps:")
        print("  - Run with --coverage to see test coverage")
        print("  - Run with --html to generate HTML coverage report")
        print("  - Deploy your changes with confidence!")
    else:
        print("❌ Some tests failed. Please review the output above.")
        print("\nDebugging tips:")
        print("  - Check the test output for specific failure details")
        print("  - Run with --verbose for more detailed output")
        print("  - Run specific failing tests with --specific <test_name>")
        print("  - Ensure all dependencies are properly mocked")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run analysis functionality tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true", 
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report (requires --coverage)"
    )
    
    parser.add_argument(
        "--specific",
        type=str,
        help="Run specific test file or test case"
    )
    
    parser.add_argument(
        "--marker",
        type=str,
        help="Run tests with specific marker"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    
    args = parser.parse_args()
    
    print_banner("AI-BE Technical Assignment - Test Runner", "=", 80)
    print("Testing the analyze functionality with comprehensive unit tests")
    print(f"Project Root: {project_root}")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    # Step 2: Validate test files
    if not validate_test_files():
        print("\n❌ Test file validation failed. Please ensure all test files exist.")
        sys.exit(1)
    
    # Step 3: Set up environment
    setup_environment()
    
    # Step 4: Run tests based on arguments
    try:
        if args.coverage or args.html:
            results = run_coverage_report(html=args.html)
        elif args.specific:
            results = run_specific_tests(args.specific)
        elif args.marker:
            results = run_with_marker(args.marker)
        elif args.parallel:
            results = run_parallel_tests()
        else:
            # Default test run
            print_section("Running All Analysis Tests")
            pytest_args = PYTEST_BASE_ARGS.copy()
            
            if args.verbose:
                pytest_args.append("-v")
            
            results = run_pytest(pytest_args)
        
        # Step 5: Print summary
        print_summary(results)
        
        # Exit with appropriate code
        sys.exit(0 if results["success"] else 1)
        
    except KeyboardInterrupt:
        print("\n\n❌ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
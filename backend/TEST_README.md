# Testing Guide for Analyze Functionality

This directory contains comprehensive unit tests for the talent analysis API functionality. The tests are designed to run independently with proper mocking of external dependencies.

## Overview

The test suite covers:
- **Router functionality**: API endpoint behavior, error handling, response structure
- **Schema validation**: Request/response models, field validation, serialization
- **Integration testing**: End-to-end workflow testing with mocked dependencies
- **Metadata exclusion**: Verification that metadata field is excluded from responses

## Test Files

- `tests/test_analysis_router.py` - Unit tests for the AnalysisRouter class
- `tests/test_analysis_schema.py` - Schema validation tests for AnalyzeRequest/Response
- `tests/test_analysis_integration.py` - Integration tests for complete workflow
- `run_tests.py` - Comprehensive test runner script

## Quick Start

### 1. Install Test Dependencies

```bash
# Install test-specific dependencies
pip install -r test-requirements.txt

# Or install core testing packages only
pip install pytest pytest-asyncio fastapi httpx
```

### 2. Run All Tests

```bash
# Run all tests with summary
python run_tests.py

# Run with verbose output
python run_tests.py --verbose

# Run with coverage report
python run_tests.py --coverage

# Run with HTML coverage report
python run_tests.py --coverage --html
```

### 3. Run Specific Tests

```bash
# Run specific test file
python run_tests.py --specific test_analysis_router

# Run specific test method
python run_tests.py --specific "test_analyze_talent_get_success"

# Run tests with specific marker
python run_tests.py --marker asyncio
```

## Test Runner Options

The `run_tests.py` script provides various options:

| Option | Description |
|--------|-------------|
| `--verbose, -v` | Show verbose test output |
| `--coverage, -c` | Generate coverage report |
| `--html` | Generate HTML coverage report (with --coverage) |
| `--specific TEST` | Run specific test file or test case |
| `--marker MARKER` | Run tests with specific pytest marker |
| `--parallel, -p` | Run tests in parallel for speed |
| `--help, -h` | Show detailed help message |

## Example Usage

```bash
# Basic test run
./run_tests.py

# Comprehensive test with coverage
./run_tests.py --coverage --html

# Debug specific failing test
./run_tests.py --specific "test_analyze_talent_get_not_found" --verbose

# Fast parallel execution
./run_tests.py --parallel

# Test schema validation only
./run_tests.py --specific test_analysis_schema
```

## Key Features Tested

### ✅ Metadata Exclusion
- Verifies that the `metadata` field is removed from AnalyzeResponse
- Tests API response structure consistency
- Validates OpenAPI schema excludes metadata

### ✅ Error Handling
- Tests 404 errors for non-existent talent data
- Tests 500 errors for internal processing failures
- Validates proper HTTP status codes and error messages

### ✅ API Functionality
- Tests both GET and POST endpoints
- Validates parameter handling and defaults
- Tests different LLM provider configurations

### ✅ Response Structure
- Ensures consistent response format
- Validates required fields are present
- Tests JSON serialization/deserialization

### ✅ Integration Workflow
- Tests complete analysis workflow with mocked dependencies
- Validates factory manager integration
- Tests concurrent request handling

## Mock Dependencies

The tests use comprehensive mocking for:
- **LLM Factory Manager**: Mock LLM model creation and responses
- **Workflow Execution**: Mock talent analysis workflow
- **Data Sources**: Mock talent data retrieval
- **External APIs**: Mock OpenAI/Anthropic API calls

This ensures tests run:
- ⚡ **Fast**: No external API calls
- 🔒 **Isolated**: No dependencies on external services
- 🎯 **Focused**: Tests only the analyze functionality logic
- 🔄 **Repeatable**: Consistent results every time

## Test Output

The test runner provides comprehensive output:

```
================================================================================
             AI-BE Technical Assignment - Test Runner
================================================================================

------------------------------------------------------------
 Checking Dependencies
------------------------------------------------------------
✅ pytest
✅ pytest-asyncio
✅ fastapi
✅ pydantic
✅ All dependencies are available

------------------------------------------------------------
 Running All Analysis Tests
------------------------------------------------------------
======================== test session starts =========================
tests/test_analysis_router.py ........                        [100%]
tests/test_analysis_schema.py ..........                      [100%]
tests/test_analysis_integration.py ......                     [100%]

======================== 24 passed in 2.34s ==========================

================================================================================
                        TEST EXECUTION SUMMARY
================================================================================

✅ Test Status: PASSED
⏱️  Execution Time: 2.34 seconds
🔧 Return Code: 0

🎉 All tests passed! The analyze functionality is working correctly.
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the backend directory
2. **Missing Dependencies**: Install test requirements with `pip install -r test-requirements.txt`
3. **Path Issues**: The test runner automatically sets up the Python path

### Debug Failed Tests

```bash
# Run with maximum verbosity
python run_tests.py --verbose

# Run only failing tests
python run_tests.py --specific "failing_test_name"

# Check specific test file
python -m pytest tests/test_analysis_router.py -v
```

### Environment Variables

The test runner automatically sets up test environment variables:
- `ENVIRONMENT=test`
- `DEBUG=false`
- `OPENAI_API_KEY=test-key-for-testing`
- `LOG_LEVEL=WARNING`

## Contributing

When adding new tests:
1. Follow the existing naming convention: `test_*_<functionality>`
2. Use proper fixtures and mocking
3. Include both positive and negative test cases
4. Add docstrings explaining what each test verifies
5. Run the full test suite to ensure no regressions

## Coverage Goals

The test suite aims for high coverage of:
- ✅ All API endpoints (GET/POST /analyze)
- ✅ All error scenarios (404, 500)
- ✅ Schema validation edge cases
- ✅ Response structure verification
- ✅ Integration workflow paths

Run `python run_tests.py --coverage --html` to see detailed coverage reports. 
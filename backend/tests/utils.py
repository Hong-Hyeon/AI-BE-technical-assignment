"""
Test utilities and helper functions.
"""

import asyncio
from functools import wraps
from typing import Any, Callable


def async_test(func: Callable) -> Callable:
    """Decorator to run async test functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


class TestHelper:
    """Helper class for common test operations."""
    
    @staticmethod
    def assert_analysis_result_valid(result):
        """Assert that analysis result is valid."""
        assert result is not None
        assert hasattr(result, 'experience_tags')
        assert hasattr(result, 'processing_time')
        assert hasattr(result, 'timestamp')
    
    @staticmethod
    def assert_response_success(response):
        """Assert that API response indicates success."""
        assert response.success is True
        assert response.message is not None 
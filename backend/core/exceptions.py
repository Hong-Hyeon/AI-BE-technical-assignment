"""
Standardized exception handling system for the AI BE Technical Assignment.

This module provides:
- Custom exception classes with structured error information
- Error codes and categories
- Standardized error response formatting
- Exception middleware integration
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime


class ErrorCategory(Enum):
    """Error categories for better error classification."""
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    SYSTEM = "system"
    WORKFLOW = "workflow"


class ErrorCode(Enum):
    """Standardized error codes."""
    # Validation errors (1000-1999)
    INVALID_INPUT = ("INVALID_INPUT", 1001, "Invalid input provided")
    MISSING_REQUIRED_FIELD = ("MISSING_REQUIRED_FIELD", 1002, "Required field is missing")
    INVALID_FORMAT = ("INVALID_FORMAT", 1003, "Invalid format")
    
    # Business logic errors (2000-2999)
    ANALYSIS_FAILED = ("ANALYSIS_FAILED", 2001, "Talent analysis processing failed")
    WORKFLOW_EXECUTION_ERROR = ("WORKFLOW_EXECUTION_ERROR", 2002, "Workflow execution error")
    PROMPT_GENERATION_FAILED = ("PROMPT_GENERATION_FAILED", 2003, "Failed to generate prompt")
    
    # External service errors (3000-3999)
    LLM_SERVICE_ERROR = ("LLM_SERVICE_ERROR", 3001, "LLM service error")
    VECTOR_SEARCH_ERROR = ("VECTOR_SEARCH_ERROR", 3002, "Vector search service error")
    API_TIMEOUT = ("API_TIMEOUT", 3003, "API request timeout")
    
    # Database errors (4000-4999)
    DATABASE_CONNECTION_ERROR = ("DATABASE_CONNECTION_ERROR", 4001, "Database connection failed")
    QUERY_EXECUTION_ERROR = ("QUERY_EXECUTION_ERROR", 4002, "Database query execution failed")
    
    # Authentication/Authorization errors (5000-5999)
    UNAUTHORIZED = ("UNAUTHORIZED", 5001, "Unauthorized access")
    FORBIDDEN = ("FORBIDDEN", 5002, "Access forbidden")
    
    # Rate limiting errors (6000-6999)
    RATE_LIMIT_EXCEEDED = ("RATE_LIMIT_EXCEEDED", 6001, "Rate limit exceeded")
    
    # System errors (7000-7999)
    INTERNAL_SERVER_ERROR = ("INTERNAL_SERVER_ERROR", 7001, "Internal server error")
    SERVICE_UNAVAILABLE = ("SERVICE_UNAVAILABLE", 7002, "Service temporarily unavailable")
    
    def __init__(self, code: str, number: int, message: str):
        self.code = code
        self.number = number
        self.message = message


@dataclass
class ErrorDetail:
    """Detailed error information."""
    field: Optional[str] = None
    message: str = ""
    code: Optional[str] = None
    value: Optional[Any] = None


@dataclass
class ErrorResponse:
    """Standardized error response format."""
    success: bool = False
    error_code: str = ""
    error_number: int = 0
    message: str = ""
    details: List[ErrorDetail] = None
    category: str = ""
    timestamp: str = ""
    trace_id: Optional[str] = None
    path: Optional[str] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = []
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['details'] = [asdict(detail) for detail in self.details]
        return result


class BaseAPIException(Exception):
    """Base exception class for all API-related exceptions."""
    
    def __init__(
        self,
        error_code: ErrorCode,
        category: ErrorCategory,
        message: Optional[str] = None,
        details: Optional[List[ErrorDetail]] = None,
        trace_id: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        self.error_code = error_code
        self.category = category
        self.message = message or error_code.message
        self.details = details or []
        self.trace_id = trace_id
        self.original_exception = original_exception
        
        super().__init__(self.message)
    
    def to_error_response(self, path: Optional[str] = None) -> ErrorResponse:
        """Convert exception to standardized error response."""
        return ErrorResponse(
            error_code=self.error_code.code,
            error_number=self.error_code.number,
            message=self.message,
            details=self.details,
            category=self.category.value,
            trace_id=self.trace_id,
            path=path
        )


class ValidationException(BaseAPIException):
    """Exception for validation errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[List[ErrorDetail]] = None,
        error_code: ErrorCode = ErrorCode.INVALID_INPUT,
        trace_id: Optional[str] = None
    ):
        super().__init__(
            error_code=error_code,
            category=ErrorCategory.VALIDATION,
            message=message,
            details=details,
            trace_id=trace_id
        )


class BusinessLogicException(BaseAPIException):
    """Exception for business logic errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.ANALYSIS_FAILED,
        details: Optional[List[ErrorDetail]] = None,
        trace_id: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(
            error_code=error_code,
            category=ErrorCategory.BUSINESS_LOGIC,
            message=message,
            details=details,
            trace_id=trace_id,
            original_exception=original_exception
        )


class ExternalServiceException(BaseAPIException):
    """Exception for external service errors."""
    
    def __init__(
        self,
        service_name: str,
        message: str,
        error_code: ErrorCode = ErrorCode.LLM_SERVICE_ERROR,
        details: Optional[List[ErrorDetail]] = None,
        trace_id: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        self.service_name = service_name
        super().__init__(
            error_code=error_code,
            category=ErrorCategory.EXTERNAL_SERVICE,
            message=f"{service_name}: {message}",
            details=details,
            trace_id=trace_id,
            original_exception=original_exception
        )


class WorkflowException(BaseAPIException):
    """Exception for workflow execution errors."""
    
    def __init__(
        self,
        workflow_name: str,
        node_name: Optional[str] = None,
        message: str = "",
        error_code: ErrorCode = ErrorCode.WORKFLOW_EXECUTION_ERROR,
        details: Optional[List[ErrorDetail]] = None,
        trace_id: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        self.workflow_name = workflow_name
        self.node_name = node_name
        
        formatted_message = f"Workflow '{workflow_name}'"
        if node_name:
            formatted_message += f" at node '{node_name}'"
        if message:
            formatted_message += f": {message}"
        
        super().__init__(
            error_code=error_code,
            category=ErrorCategory.WORKFLOW,
            message=formatted_message,
            details=details,
            trace_id=trace_id,
            original_exception=original_exception
        )


class DatabaseException(BaseAPIException):
    """Exception for database-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.DATABASE_CONNECTION_ERROR,
        details: Optional[List[ErrorDetail]] = None,
        trace_id: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(
            error_code=error_code,
            category=ErrorCategory.DATABASE,
            message=message,
            details=details,
            trace_id=trace_id,
            original_exception=original_exception
        )


def create_validation_error(field: str, message: str, value: Any = None) -> ErrorDetail:
    """Helper function to create validation error details."""
    return ErrorDetail(
        field=field,
        message=message,
        code=ErrorCode.INVALID_FORMAT.code,
        value=value
    )


def handle_unexpected_error(
    error: Exception,
    context: str = "",
    trace_id: Optional[str] = None
) -> BaseAPIException:
    """Handle unexpected errors by wrapping them in a standardized exception."""
    message = "Unexpected error"
    if context:
        message += f" in {context}"
    message += f": {str(error)}"
    
    details = [
        ErrorDetail(
            field="traceback",
            message=traceback.format_exc(),
            code="TRACEBACK"
        )
    ]
    
    return BaseAPIException(
        error_code=ErrorCode.INTERNAL_SERVER_ERROR,
        category=ErrorCategory.SYSTEM,
        message=message,
        details=details,
        trace_id=trace_id,
        original_exception=error
    ) 
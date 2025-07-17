"""
Middleware for handling exceptions and providing standardized error responses.

This module provides:
- Exception handling middleware for FastAPI
- HTTP status code mapping for different exception types
- Request tracing and logging integration
"""

import uuid
from typing import Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging

from .exceptions import (
    BaseAPIException,
    ValidationException,
    BusinessLogicException,
    ExternalServiceException,
    WorkflowException,
    DatabaseException,
    ErrorCategory,
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    handle_unexpected_error
)

logger = logging.getLogger('api')


class ExceptionHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware to handle exceptions and provide standardized error responses."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate trace ID for request tracking
        trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id
        
        try:
            response = await call_next(request)
            return response
        except BaseAPIException as e:
            # Handle our custom exceptions
            error_response = e.to_error_response(path=str(request.url))
            error_response.trace_id = trace_id
            
            status_code = self._get_status_code_for_exception(e)
            
            # Log the error
            self._log_exception(e, request, trace_id, status_code)
            
            return JSONResponse(
                status_code=status_code,
                content=error_response.to_dict()
            )
        except RequestValidationError as e:
            # Handle FastAPI validation errors
            validation_exception = self._convert_validation_error(e, trace_id)
            error_response = validation_exception.to_error_response(path=str(request.url))
            
            self._log_exception(validation_exception, request, trace_id, 422)
            
            return JSONResponse(
                status_code=422,
                content=error_response.to_dict()
            )
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            error_response = ErrorResponse(
                error_code="HTTP_EXCEPTION",
                error_number=e.status_code,
                message=e.detail,
                category="http",
                trace_id=trace_id,
                path=str(request.url)
            )
            
            logger.warning(
                f"HTTP Exception: {e.status_code} - {e.detail}",
                extra={
                    'trace_id': trace_id,
                    'path': str(request.url),
                    'method': request.method,
                    'status_code': e.status_code
                }
            )
            
            return JSONResponse(
                status_code=e.status_code,
                content=error_response.to_dict()
            )
        except Exception as e:
            # Handle unexpected exceptions
            unexpected_exception = handle_unexpected_error(e, "request processing", trace_id)
            error_response = unexpected_exception.to_error_response(path=str(request.url))
            
            self._log_exception(unexpected_exception, request, trace_id, 500)
            
            return JSONResponse(
                status_code=500,
                content=error_response.to_dict()
            )
    
    def _get_status_code_for_exception(self, exception: BaseAPIException) -> int:
        """Map exception categories to HTTP status codes."""
        status_mapping = {
            ErrorCategory.VALIDATION: 400,
            ErrorCategory.BUSINESS_LOGIC: 422,
            ErrorCategory.EXTERNAL_SERVICE: 502,
            ErrorCategory.DATABASE: 503,
            ErrorCategory.AUTHENTICATION: 401,
            ErrorCategory.AUTHORIZATION: 403,
            ErrorCategory.RATE_LIMIT: 429,
            ErrorCategory.SYSTEM: 500,
            ErrorCategory.WORKFLOW: 422
        }
        
        return status_mapping.get(exception.category, 500)
    
    def _convert_validation_error(
        self, 
        validation_error: RequestValidationError, 
        trace_id: str
    ) -> ValidationException:
        """Convert FastAPI validation error to our custom validation exception."""
        details = []
        
        for error in validation_error.errors():
            field_path = ".".join(str(x) for x in error['loc'])
            details.append(
                ErrorDetail(
                    field=field_path,
                    message=error['msg'],
                    code=error['type'],
                    value=error.get('input')
                )
            )
        
        return ValidationException(
            message="Request validation failed",
            details=details,
            error_code=ErrorCode.INVALID_INPUT,
            trace_id=trace_id
        )
    
    def _log_exception(
        self, 
        exception: BaseAPIException, 
        request: Request, 
        trace_id: str, 
        status_code: int
    ):
        """Log exception with appropriate level based on severity."""
        log_data = {
            'trace_id': trace_id,
            'path': str(request.url),
            'method': request.method,
            'status_code': status_code,
            'error_code': exception.error_code.code,
            'error_category': exception.category.value,
            'client_ip': request.client.host if request.client else None
        }
        
        if status_code >= 500:
            logger.error(
                f"Server Error [{exception.error_code.code}]: {exception.message}",
                extra=log_data,
                exc_info=exception.original_exception
            )
        elif status_code >= 400:
            logger.warning(
                f"Client Error [{exception.error_code.code}]: {exception.message}",
                extra=log_data
            )
        else:
            logger.info(
                f"Exception [{exception.error_code.code}]: {exception.message}",
                extra=log_data
            )


def create_error_response(
    error_code: ErrorCode,
    category: ErrorCategory,
    message: Optional[str] = None,
    details: Optional[list] = None,
    trace_id: Optional[str] = None,
    path: Optional[str] = None
) -> ErrorResponse:
    """Helper function to create error responses."""
    return ErrorResponse(
        error_code=error_code.code,
        error_number=error_code.number,
        message=message or error_code.message,
        details=details or [],
        category=category.value,
        trace_id=trace_id,
        path=path
    )


# Utility functions for common error scenarios
def raise_validation_error(message: str, field: Optional[str] = None, value: Optional[str] = None):
    """Raise a validation error with optional field details."""
    details = []
    if field:
        details.append(ErrorDetail(field=field, message=message, value=value))
    
    raise ValidationException(message=message, details=details)


def raise_business_error(message: str, error_code: ErrorCode = ErrorCode.ANALYSIS_FAILED):
    """Raise a business logic error."""
    raise BusinessLogicException(message=message, error_code=error_code)


def raise_external_service_error(service_name: str, message: str, error_code: ErrorCode = ErrorCode.LLM_SERVICE_ERROR):
    """Raise an external service error."""
    raise ExternalServiceException(service_name=service_name, message=message, error_code=error_code)


def raise_workflow_error(workflow_name: str, node_name: Optional[str] = None, message: str = ""):
    """Raise a workflow execution error."""
    raise WorkflowException(workflow_name=workflow_name, node_name=node_name, message=message)


def raise_database_error(message: str, error_code: ErrorCode = ErrorCode.DATABASE_CONNECTION_ERROR):
    """Raise a database error."""
    raise DatabaseException(message=message, error_code=error_code) 
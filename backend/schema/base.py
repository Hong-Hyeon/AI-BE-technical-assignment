"""
Base schema classes for the AI BE Technical Assignment.

This module provides:
- Base request/response models
- Common validation patterns
- Standardized response formats
- Pagination and filtering schemas
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Generic, TypeVar, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    class Config:
        # Allow population by field name and alias
        allow_population_by_field_name = True
        # Use enum values instead of names
        use_enum_values = True
        # Validate assignment
        validate_assignment = True
        # Allow extra fields in some cases
        extra = "forbid"
        # JSON encoders for custom types
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class TimestampMixin(BaseModel):
    """Mixin for adding timestamp fields."""
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)


class BaseRequest(BaseSchema):
    """Base request schema."""
    trace_id: Optional[str] = Field(None, description="Request trace ID for debugging")
    
    class Config(BaseSchema.Config):
        extra = "allow"  # Allow extra fields in requests


class BaseResponse(BaseSchema):
    """Base response schema."""
    success: bool = Field(True, description="Whether the request was successful")
    message: str = Field("", description="Response message")
    trace_id: Optional[str] = Field(None, description="Request trace ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


# Generic types for pagination
T = TypeVar('T')


class PaginationParams(BaseSchema):
    """Pagination parameters."""
    page: int = Field(1, ge=1, description="Page number (1-based)")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.page_size


class SortOrder(str, Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"


class SortParams(BaseSchema):
    """Sorting parameters."""
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: SortOrder = Field(SortOrder.ASC, description="Sort order")


class FilterParams(BaseSchema):
    """Base filtering parameters."""
    search: Optional[str] = Field(None, description="Search query")
    
    @validator('search')
    def validate_search(cls, v):
        if v is not None and len(v.strip()) < 2:
            raise ValueError('Search query must be at least 2 characters')
        return v.strip() if v else None


class PaginatedResponse(BaseResponse, Generic[T]):
    """Paginated response schema."""
    data: List[T] = Field(default_factory=list, description="Response data")
    pagination: Dict[str, Any] = Field(default_factory=dict, description="Pagination metadata")
    
    @classmethod
    def create(
        cls,
        data: List[T],
        page: int,
        page_size: int,
        total_count: int,
        message: str = "Success",
        trace_id: Optional[str] = None
    ) -> 'PaginatedResponse[T]':
        """Create a paginated response."""
        total_pages = (total_count + page_size - 1) // page_size
        
        pagination = {
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_previous": page > 1
        }
        
        return cls(
            data=data,
            pagination=pagination,
            message=message,
            trace_id=trace_id
        )


class DataResponse(BaseResponse, Generic[T]):
    """Single data item response schema."""
    data: Optional[T] = Field(None, description="Response data")
    
    @classmethod
    def success(
        cls,
        data: T,
        message: str = "Success",
        trace_id: Optional[str] = None
    ) -> 'DataResponse[T]':
        """Create a successful data response."""
        return cls(
            data=data,
            success=True,
            message=message,
            trace_id=trace_id
        )
    
    @classmethod
    def error(
        cls,
        message: str,
        trace_id: Optional[str] = None
    ) -> 'DataResponse[T]':
        """Create an error response."""
        return cls(
            data=None,
            success=False,
            message=message,
            trace_id=trace_id
        )


class HealthCheckResponse(BaseResponse):
    """Health check response schema."""
    status: str = Field("healthy", description="Service status")
    version: str = Field("", description="Application version")
    environment: str = Field("", description="Environment name")
    uptime: float = Field(0.0, description="Uptime in seconds")
    services: Dict[str, str] = Field(default_factory=dict, description="Service status")


class ValidationErrorDetail(BaseSchema):
    """Validation error detail schema."""
    field: str = Field(..., description="Field with validation error")
    message: str = Field(..., description="Validation error message")
    value: Optional[Any] = Field(None, description="Invalid value")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseResponse):
    """Error response schema."""
    success: bool = Field(False, description="Always false for errors")
    error_code: str = Field("", description="Error code")
    error_number: int = Field(0, description="Error number")
    category: str = Field("", description="Error category")
    details: List[ValidationErrorDetail] = Field(default_factory=list, description="Error details")
    path: Optional[str] = Field(None, description="Request path")


class StatusEnum(str, Enum):
    """Common status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PriorityEnum(str, Enum):
    """Priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProviderEnum(str, Enum):
    """Service provider enumeration."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


class AnalysisTypeEnum(str, Enum):
    """Analysis type enumeration."""
    EDUCATION = "education"
    EXPERIENCE = "experience"
    SKILLS = "skills"
    FULL = "full"
    CUSTOM = "custom"


class FileMetadata(BaseSchema):
    """File metadata schema."""
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., ge=0, description="File size in bytes")
    content_type: str = Field(..., description="MIME content type")
    checksum: Optional[str] = Field(None, description="File checksum")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('size')
    def validate_size(cls, v):
        # 16MB limit
        max_size = 16 * 1024 * 1024
        if v > max_size:
            raise ValueError(f'File size cannot exceed {max_size} bytes')
        return v


class ApiKeyMixin(BaseSchema):
    """Mixin for API key validation."""
    api_key: Optional[str] = Field(None, description="API key for authentication")
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if v and len(v) < 10:
            raise ValueError('API key must be at least 10 characters')
        return v


class ConfigurationMixin(BaseSchema):
    """Mixin for configuration parameters."""
    timeout: Optional[int] = Field(30, ge=1, le=300, description="Timeout in seconds")
    max_retries: Optional[int] = Field(3, ge=0, le=10, description="Maximum retry attempts")
    
    @validator('timeout')
    def validate_timeout(cls, v):
        if v and (v < 1 or v > 300):
            raise ValueError('Timeout must be between 1 and 300 seconds')
        return v 
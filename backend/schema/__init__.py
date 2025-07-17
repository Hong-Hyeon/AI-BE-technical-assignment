"""
Schema module for the AI BE Technical Assignment.

This module provides comprehensive schema definitions for:
- Request/Response models
- Validation and serialization
- Base classes and mixins
- Domain-specific schemas

Usage:
    from schema import AnalyzeRequest, AnalyzeResponse
    from schema.base import BaseRequest, BaseResponse
    from schema.workflow import WorkflowExecutionRequest
"""

# Base schemas and common utilities
from .base import (
    BaseSchema,
    BaseRequest,
    BaseResponse,
    TimestampMixin,
    PaginationParams,
    SortParams,
    FilterParams,
    PaginatedResponse,
    DataResponse,
    HealthCheckResponse,
    ValidationErrorDetail,
    ErrorResponse,
    StatusEnum,
    PriorityEnum,
    ProviderEnum,
    AnalysisTypeEnum,
    FileMetadata,
    ApiKeyMixin,
    ConfigurationMixin,
    SortOrder
)

# Analysis schemas
from .analysis import (
    LLMModelEnum,
    AnalysisProgressEnum,
    LLMConfiguration,
    VectorSearchConfiguration,
    AnalysisOptions,
    AnalyzeRequest,
    AnalysisProgress,
    AnalysisMetadata,
    AnalyzeResponse,
    VectorSearchRequest,
    VectorSearchResult,
    VectorSearchResponse,
    AnalysisStatusRequest,
    AnalysisStatusResponse,
    AnalysisListRequest,
    AnalysisListResponse,
    BulkAnalysisRequest,
    BulkAnalysisResponse
)

# Workflow schemas
from .workflow import (
    WorkflowNodeType,
    WorkflowExecutionStatus,
    NodeExecutionStatus,
    WorkflowNodeConfig,
    WorkflowConfig,
    NodeExecutionResult,
    WorkflowExecutionProgress,
    WorkflowExecutionRequest,
    WorkflowExecutionResponse,
    WorkflowStatusRequest,
    WorkflowStatusResponse,
    WorkflowCancelRequest,
    WorkflowCancelResponse,
    WorkflowListRequest,
    WorkflowExecutionSummary,
    WorkflowListResponse,
    NodeMetrics,
    WorkflowMetrics,
    WorkflowMetricsRequest,
    WorkflowMetricsResponse
)

# Organized exports by category
__all__ = [
    # Base schemas
    "BaseSchema",
    "BaseRequest", 
    "BaseResponse",
    "TimestampMixin",
    "PaginationParams",
    "SortParams",
    "FilterParams", 
    "PaginatedResponse",
    "DataResponse",
    "HealthCheckResponse",
    "ValidationErrorDetail",
    "ErrorResponse",
    "FileMetadata",
    "ApiKeyMixin",
    "ConfigurationMixin",
    
    # Enums
    "StatusEnum",
    "PriorityEnum",
    "ProviderEnum",
    "AnalysisTypeEnum",
    "SortOrder",
    "LLMModelEnum",
    "AnalysisProgressEnum",
    "WorkflowNodeType",
    "WorkflowExecutionStatus", 
    "NodeExecutionStatus",
    
    # Analysis schemas
    "LLMConfiguration",
    "VectorSearchConfiguration",
    "AnalysisOptions",
    "AnalyzeRequest",
    "AnalysisProgress",
    "AnalysisMetadata", 
    "AnalyzeResponse",
    "VectorSearchRequest",
    "VectorSearchResult",
    "VectorSearchResponse",
    "AnalysisStatusRequest",
    "AnalysisStatusResponse",
    "AnalysisListRequest",
    "AnalysisListResponse",
    "BulkAnalysisRequest",
    "BulkAnalysisResponse",
    
    # Workflow schemas
    "WorkflowNodeConfig",
    "WorkflowConfig",
    "NodeExecutionResult",
    "WorkflowExecutionProgress",
    "WorkflowExecutionRequest",
    "WorkflowExecutionResponse",
    "WorkflowStatusRequest", 
    "WorkflowStatusResponse",
    "WorkflowCancelRequest",
    "WorkflowCancelResponse",
    "WorkflowListRequest",
    "WorkflowExecutionSummary",
    "WorkflowListResponse",
    "NodeMetrics",
    "WorkflowMetrics",
    "WorkflowMetricsRequest",
    "WorkflowMetricsResponse"
]

# Convenience imports for common use cases
REQUEST_SCHEMAS = [
    AnalyzeRequest,
    VectorSearchRequest,
    AnalysisStatusRequest,
    AnalysisListRequest,
    BulkAnalysisRequest,
    WorkflowExecutionRequest,
    WorkflowStatusRequest,
    WorkflowCancelRequest,
    WorkflowListRequest,
    WorkflowMetricsRequest
]

RESPONSE_SCHEMAS = [
    AnalyzeResponse,
    VectorSearchResponse,
    AnalysisStatusResponse,
    AnalysisListResponse,
    BulkAnalysisResponse,
    WorkflowExecutionResponse,
    WorkflowStatusResponse,
    WorkflowCancelResponse,
    WorkflowListResponse,
    WorkflowMetricsResponse
]

CONFIG_SCHEMAS = [
    LLMConfiguration,
    VectorSearchConfiguration,
    AnalysisOptions,
    WorkflowConfig,
    WorkflowNodeConfig
]

ENUM_SCHEMAS = [
    StatusEnum,
    PriorityEnum,
    ProviderEnum,
    AnalysisTypeEnum,
    LLMModelEnum,
    AnalysisProgressEnum,
    WorkflowNodeType,
    WorkflowExecutionStatus,
    NodeExecutionStatus,
    SortOrder
] 
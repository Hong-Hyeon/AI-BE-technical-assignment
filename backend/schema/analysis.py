"""
Analysis-related schema models for the AI BE Technical Assignment.

This module provides:
- Request/Response models for talent analysis
- Validation for analysis parameters
- Vector search schemas
- LLM provider configurations
- Analysis result schemas
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import Field, validator
from enum import Enum

from .base import (
    BaseRequest,
    BaseResponse,
    BaseSchema,
    DataResponse,
    PaginatedResponse,
    StatusEnum,
    ProviderEnum,
    AnalysisTypeEnum,
    ConfigurationMixin,
    TimestampMixin
)
from models.talent import AnalysisResult


class LLMModelEnum(str, Enum):
    """Supported LLM models."""
    # OpenAI models
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    
    # Anthropic models
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    
    # Google models
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"


class AnalysisProgressEnum(str, Enum):
    """Analysis progress stages."""
    INITIALIZING = "initializing"
    LOADING_DATA = "loading_data"
    PROCESSING_EDUCATION = "processing_education"
    PROCESSING_EXPERIENCE = "processing_experience"
    PROCESSING_SKILLS = "processing_skills"
    GENERATING_SUMMARY = "generating_summary"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


class LLMConfiguration(BaseSchema, ConfigurationMixin):
    """LLM configuration for analysis."""
    provider: ProviderEnum = Field(ProviderEnum.OPENAI, description="LLM provider")
    model: LLMModelEnum = Field(LLMModelEnum.GPT_4O_MINI, description="LLM model")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="LLM temperature")
    max_tokens: int = Field(1500, ge=100, le=4000, description="Maximum tokens")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Temperature must be between 0.0 and 1.0')
        return v
    
    @validator('model', pre=True)
    def validate_model_provider_compatibility(cls, v, values):
        """Validate model is compatible with provider."""
        provider = values.get('provider')
        if provider == ProviderEnum.OPENAI and not v.startswith(('gpt-', 'text-')):
            raise ValueError(f'Model {v} is not compatible with OpenAI provider')
        elif provider == ProviderEnum.ANTHROPIC and not v.startswith('claude-'):
            raise ValueError(f'Model {v} is not compatible with Anthropic provider')
        elif provider == ProviderEnum.GOOGLE and not v.startswith('gemini-'):
            raise ValueError(f'Model {v} is not compatible with Google provider')
        return v


class VectorSearchConfiguration(BaseSchema, ConfigurationMixin):
    """Vector search configuration."""
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    max_results: int = Field(10, ge=1, le=100, description="Maximum search results")
    index_name: Optional[str] = Field(None, description="Custom index name")
    
    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Similarity threshold must be between 0.0 and 1.0')
        return v


class AnalysisOptions(BaseSchema):
    """Options for customizing analysis behavior."""
    include_detailed_reasoning: bool = Field(True, description="Include detailed reasoning in results")
    include_confidence_scores: bool = Field(True, description="Include confidence scores")
    include_source_references: bool = Field(False, description="Include source document references")
    custom_prompt_additions: Optional[str] = Field(None, description="Additional prompt instructions")
    analysis_depth: str = Field("standard", regex="^(quick|standard|detailed)$", description="Analysis depth")
    
    @validator('custom_prompt_additions')
    def validate_custom_prompt(cls, v):
        if v and len(v) > 500:
            raise ValueError('Custom prompt additions cannot exceed 500 characters')
        return v


class AnalyzeRequest(BaseRequest, ConfigurationMixin):
    """Request model for talent analysis."""
    talent_id: str = Field(..., description="Unique talent identifier")
    analysis_type: AnalysisTypeEnum = Field(AnalysisTypeEnum.FULL, description="Type of analysis to perform")
    
    # LLM Configuration
    llm_config: LLMConfiguration = Field(default_factory=LLMConfiguration, description="LLM configuration")
    
    # Vector Search Configuration
    vector_config: Optional[VectorSearchConfiguration] = Field(None, description="Vector search configuration")
    
    # Analysis Options
    options: AnalysisOptions = Field(default_factory=AnalysisOptions, description="Analysis options")
    
    # Additional context
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for analysis")
    priority: str = Field("medium", regex="^(low|medium|high|critical)$", description="Analysis priority")
    
    @validator('talent_id')
    def validate_talent_id(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('Talent ID must be at least 3 characters')
        return v.strip()
    
    @validator('context')
    def validate_context(cls, v):
        if v and len(str(v)) > 2000:
            raise ValueError('Context data cannot exceed 2000 characters when serialized')
        return v


class AnalysisProgress(BaseSchema, TimestampMixin):
    """Analysis progress tracking."""
    stage: AnalysisProgressEnum = Field(..., description="Current analysis stage")
    progress_percentage: int = Field(0, ge=0, le=100, description="Progress percentage")
    current_task: str = Field("", description="Current task description")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class AnalysisMetadata(BaseSchema, TimestampMixin):
    """Metadata about the analysis process."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    talent_id: str = Field(..., description="Talent identifier")
    analysis_type: AnalysisTypeEnum = Field(..., description="Type of analysis performed")
    status: StatusEnum = Field(StatusEnum.PENDING, description="Analysis status")
    
    # Timing information
    started_at: Optional[datetime] = Field(None, description="Analysis start time")
    completed_at: Optional[datetime] = Field(None, description="Analysis completion time")
    duration_seconds: Optional[float] = Field(None, description="Analysis duration in seconds")
    
    # Configuration used
    llm_config: LLMConfiguration = Field(..., description="LLM configuration used")
    vector_config: Optional[VectorSearchConfiguration] = Field(None, description="Vector search configuration used")
    options: AnalysisOptions = Field(..., description="Analysis options used")
    
    # Resource usage
    tokens_used: Optional[int] = Field(None, description="Total tokens consumed")
    api_calls_made: Optional[int] = Field(None, description="Number of API calls made")
    cost_estimate: Optional[float] = Field(None, description="Estimated cost in USD")


class AnalyzeResponse(BaseResponse):
    """Response model for talent analysis."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    result: Optional[AnalysisResult] = Field(None, description="Analysis result")
    metadata: AnalysisMetadata = Field(..., description="Analysis metadata")
    progress: Optional[AnalysisProgress] = Field(None, description="Current progress if still running")
    
    @classmethod
    def create_async_response(
        cls,
        analysis_id: str,
        metadata: AnalysisMetadata,
        message: str = "Analysis started successfully",
        trace_id: Optional[str] = None
    ) -> 'AnalyzeResponse':
        """Create response for asynchronous analysis."""
        return cls(
            analysis_id=analysis_id,
            result=None,
            metadata=metadata,
            progress=AnalysisProgress(
                stage=AnalysisProgressEnum.INITIALIZING,
                progress_percentage=0,
                current_task="Starting analysis"
            ),
            message=message,
            trace_id=trace_id
        )
    
    @classmethod
    def create_completed_response(
        cls,
        analysis_id: str,
        result: AnalysisResult,
        metadata: AnalysisMetadata,
        message: str = "Analysis completed successfully",
        trace_id: Optional[str] = None
    ) -> 'AnalyzeResponse':
        """Create response for completed analysis."""
        return cls(
            analysis_id=analysis_id,
            result=result,
            metadata=metadata,
            progress=AnalysisProgress(
                stage=AnalysisProgressEnum.COMPLETED,
                progress_percentage=100,
                current_task="Analysis completed"
            ),
            message=message,
            trace_id=trace_id
        )


class VectorSearchRequest(BaseRequest):
    """Request model for vector search."""
    query: str = Field(..., min_length=2, description="Search query")
    documents: List[Dict[str, Any]] = Field(..., min_items=1, description="Documents to search")
    config: VectorSearchConfiguration = Field(default_factory=VectorSearchConfiguration, description="Search configuration")
    
    @validator('query')
    def validate_query(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Query must be at least 2 characters')
        return v.strip()
    
    @validator('documents')
    def validate_documents(cls, v):
        if not v:
            raise ValueError('At least one document is required')
        if len(v) > 1000:
            raise ValueError('Cannot search more than 1000 documents at once')
        return v


class VectorSearchResult(BaseSchema):
    """Single vector search result."""
    document_id: str = Field(..., description="Document identifier")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    content: Dict[str, Any] = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")


class VectorSearchResponse(BaseResponse):
    """Response model for vector search."""
    query: str = Field(..., description="Original search query")
    results: List[VectorSearchResult] = Field(default_factory=list, description="Search results")
    total_documents_searched: int = Field(0, description="Total documents searched")
    search_time_ms: float = Field(0.0, description="Search time in milliseconds")
    
    @classmethod
    def create_results(
        cls,
        query: str,
        results: List[VectorSearchResult],
        total_searched: int,
        search_time_ms: float,
        trace_id: Optional[str] = None
    ) -> 'VectorSearchResponse':
        """Create search results response."""
        return cls(
            query=query,
            results=results,
            total_documents_searched=total_searched,
            search_time_ms=search_time_ms,
            message=f"Found {len(results)} results",
            trace_id=trace_id
        )


class AnalysisStatusRequest(BaseRequest):
    """Request model for checking analysis status."""
    analysis_id: str = Field(..., description="Analysis identifier to check")
    
    @validator('analysis_id')
    def validate_analysis_id(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError('Analysis ID must be at least 10 characters')
        return v.strip()


class AnalysisStatusResponse(BaseResponse):
    """Response model for analysis status."""
    analysis_id: str = Field(..., description="Analysis identifier")
    status: StatusEnum = Field(..., description="Current status")
    progress: AnalysisProgress = Field(..., description="Current progress")
    metadata: AnalysisMetadata = Field(..., description="Analysis metadata")
    result: Optional[AnalysisResult] = Field(None, description="Result if completed")


class AnalysisListRequest(BaseRequest):
    """Request model for listing analyses."""
    talent_id: Optional[str] = Field(None, description="Filter by talent ID")
    status: Optional[StatusEnum] = Field(None, description="Filter by status")
    analysis_type: Optional[AnalysisTypeEnum] = Field(None, description="Filter by analysis type")
    date_from: Optional[datetime] = Field(None, description="Filter from date")
    date_to: Optional[datetime] = Field(None, description="Filter to date")
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    
    @validator('date_from', 'date_to')
    def validate_dates(cls, v):
        if v and v > datetime.utcnow():
            raise ValueError('Date cannot be in the future')
        return v


class AnalysisListResponse(PaginatedResponse[AnalysisMetadata]):
    """Response model for analysis list."""
    pass


class BulkAnalysisRequest(BaseRequest):
    """Request model for bulk analysis."""
    talent_ids: List[str] = Field(..., min_items=1, max_items=100, description="List of talent IDs")
    analysis_type: AnalysisTypeEnum = Field(AnalysisTypeEnum.FULL, description="Type of analysis")
    llm_config: LLMConfiguration = Field(default_factory=LLMConfiguration, description="LLM configuration")
    options: AnalysisOptions = Field(default_factory=AnalysisOptions, description="Analysis options")
    
    @validator('talent_ids')
    def validate_talent_ids(cls, v):
        if len(set(v)) != len(v):
            raise ValueError('Duplicate talent IDs are not allowed')
        for talent_id in v:
            if not talent_id or len(talent_id.strip()) < 3:
                raise ValueError('All talent IDs must be at least 3 characters')
        return [tid.strip() for tid in v]


class BulkAnalysisResponse(BaseResponse):
    """Response model for bulk analysis."""
    batch_id: str = Field(..., description="Batch identifier")
    total_analyses: int = Field(..., description="Total number of analyses requested")
    analyses: List[AnalysisMetadata] = Field(..., description="Individual analysis metadata")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")
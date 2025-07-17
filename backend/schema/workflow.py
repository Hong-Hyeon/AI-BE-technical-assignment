"""
Workflow-related schema models for LangGraph workflow management.

This module provides:
- Workflow configuration schemas
- Execution tracking and monitoring
- Node-level progress and error handling
- Workflow result and artifact schemas
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import Field, field_validator
from enum import Enum

from .base import (
    BaseRequest,
    BaseResponse,
    BaseSchema,
    DataResponse,
    StatusEnum,
    TimestampMixin,
    ConfigurationMixin
)


class WorkflowNodeType(str, Enum):
    """Types of workflow nodes."""
    EDUCATION_ANALYSIS = "education_analysis"
    EXPERIENCE_ANALYSIS = "experience_analysis"
    POSITION_ANALYSIS = "position_analysis"
    AGGREGATION = "aggregation"
    VECTOR_SEARCH = "vector_search"
    LLM_CALL = "llm_call"
    DATA_TRANSFORMATION = "data_transformation"
    VALIDATION = "validation"
    OUTPUT_FORMATTING = "output_formatting"


class WorkflowExecutionStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RETRYING = "retrying"


class NodeExecutionStatus(str, Enum):
    """Individual node execution status."""
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class WorkflowNodeConfig(ConfigurationMixin):
    """Configuration for a workflow node."""
    node_id: str = Field(..., description="Unique node identifier")
    node_type: WorkflowNodeType = Field(..., description="Type of node")
    enabled: bool = Field(True, description="Whether node is enabled")
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list, description="Node dependencies")
    
    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict, description="Node-specific configuration")
    
    # Error handling
    retry_on_failure: bool = Field(True, description="Retry on failure")
    continue_on_failure: bool = Field(False, description="Continue workflow on node failure")
    
    @field_validator('node_id')
    def validate_node_id(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Node ID must be at least 2 characters')
        return v.strip()


class WorkflowConfig(ConfigurationMixin):
    """Configuration for workflow execution."""
    workflow_id: str = Field(..., description="Unique workflow identifier")
    workflow_name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    
    # Nodes configuration
    nodes: List[WorkflowNodeConfig] = Field(..., min_items=1, description="Workflow nodes")
    
    # Execution settings
    parallel_execution: bool = Field(True, description="Enable parallel node execution")
    fail_fast: bool = Field(False, description="Stop on first failure")
    
    # Resource limits
    max_execution_time: int = Field(3600, ge=60, le=7200, description="Max execution time in seconds")
    memory_limit_mb: Optional[int] = Field(None, description="Memory limit in MB")
    
    @field_validator('workflow_id')
    def validate_workflow_id(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('Workflow ID must be at least 3 characters')
        return v.strip()
    
    @field_validator('nodes')
    def validate_nodes(cls, v):
        if not v:
            raise ValueError('At least one node is required')
        
        # Check for duplicate node IDs
        node_ids = [node.node_id for node in v]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError('Duplicate node IDs are not allowed')
        
        # Validate dependencies
        for node in v:
            for dep in node.depends_on:
                if dep not in node_ids:
                    raise ValueError(f'Node {node.node_id} depends on non-existent node {dep}')
        
        return v


class NodeExecutionResult(TimestampMixin):
    """Result of a single node execution."""
    node_id: str = Field(..., description="Node identifier")
    execution_id: str = Field(..., description="Execution identifier")
    status: NodeExecutionStatus = Field(..., description="Execution status")
    
    # Timing
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    duration_seconds: Optional[float] = Field(None, description="Execution duration")
    
    # Results
    output: Optional[Dict[str, Any]] = Field(None, description="Node output data")
    artifacts: Dict[str, Any] = Field(default_factory=dict, description="Generated artifacts")
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_code: Optional[str] = Field(None, description="Error code if failed")
    retry_count: int = Field(0, description="Number of retries attempted")
    
    # Resource usage
    memory_used_mb: Optional[float] = Field(None, description="Memory used in MB")
    cpu_time_seconds: Optional[float] = Field(None, description="CPU time used")


class WorkflowExecutionProgress(TimestampMixin):
    """Progress tracking for workflow execution."""
    execution_id: str = Field(..., description="Execution identifier")
    workflow_id: str = Field(..., description="Workflow identifier")
    status: WorkflowExecutionStatus = Field(..., description="Overall execution status")
    
    # Progress tracking
    total_nodes: int = Field(0, description="Total number of nodes")
    completed_nodes: int = Field(0, description="Number of completed nodes")
    failed_nodes: int = Field(0, description="Number of failed nodes")
    running_nodes: int = Field(0, description="Number of currently running nodes")
    
    # Timing
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    # Current stage
    current_stage: str = Field("", description="Current execution stage")
    current_nodes: List[str] = Field(default_factory=list, description="Currently executing nodes")
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_nodes == 0:
            return 0.0
        return (self.completed_nodes / self.total_nodes) * 100


class WorkflowExecutionRequest(BaseRequest, ConfigurationMixin):
    """Request to execute a workflow."""
    workflow_config: WorkflowConfig = Field(..., description="Workflow configuration")
    input_data: Dict[str, Any] = Field(..., description="Input data for workflow")
    
    # Execution options
    execution_mode: str = Field("async", pattern="^(sync|async)$", description="Execution mode")
    priority: str = Field("medium", pattern="^(low|medium|high|critical)$", description="Execution priority")
    
    # Monitoring options
    enable_detailed_logging: bool = Field(True, description="Enable detailed logging")
    enable_metrics_collection: bool = Field(True, description="Enable metrics collection")
    
    # Callback configuration
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")
    
    @field_validator('input_data')
    def validate_input_data(cls, v):
        if len(str(v)) > 100000:  # 100KB limit
            raise ValueError('Input data cannot exceed 100KB when serialized')
        return v


class WorkflowExecutionResponse(BaseResponse):
    """Response for workflow execution request."""
    execution_id: str = Field(..., description="Unique execution identifier")
    workflow_id: str = Field(..., description="Workflow identifier")
    status: WorkflowExecutionStatus = Field(..., description="Initial execution status")
    progress: WorkflowExecutionProgress = Field(..., description="Execution progress")
    
    # Results (for sync execution)
    result: Optional[Dict[str, Any]] = Field(None, description="Workflow result if completed")
    node_results: List[NodeExecutionResult] = Field(default_factory=list, description="Individual node results")
    
    @classmethod
    def create_async_response(
        cls,
        execution_id: str,
        workflow_id: str,
        progress: WorkflowExecutionProgress,
        message: str = "Workflow execution started",
        trace_id: Optional[str] = None
    ) -> 'WorkflowExecutionResponse':
        """Create response for async workflow execution."""
        return cls(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowExecutionStatus.RUNNING,
            progress=progress,
            message=message,
            trace_id=trace_id
        )


class WorkflowStatusRequest(BaseRequest):
    """Request to check workflow execution status."""
    execution_id: str = Field(..., description="Execution identifier")
    include_node_details: bool = Field(False, description="Include detailed node results")
    
    @field_validator('execution_id')
    def validate_execution_id(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError('Execution ID must be at least 10 characters')
        return v.strip()


class WorkflowStatusResponse(BaseResponse):
    """Response for workflow status check."""
    execution_id: str = Field(..., description="Execution identifier")
    workflow_id: str = Field(..., description="Workflow identifier")
    status: WorkflowExecutionStatus = Field(..., description="Current execution status")
    progress: WorkflowExecutionProgress = Field(..., description="Current progress")
    
    # Results
    result: Optional[Dict[str, Any]] = Field(None, description="Final result if completed")
    node_results: List[NodeExecutionResult] = Field(default_factory=list, description="Node execution results")
    
    # Error information
    error_summary: Optional[str] = Field(None, description="Error summary if failed")
    failed_nodes: List[str] = Field(default_factory=list, description="List of failed node IDs")


class WorkflowCancelRequest(BaseRequest):
    """Request to cancel workflow execution."""
    execution_id: str = Field(..., description="Execution identifier")
    reason: Optional[str] = Field(None, description="Cancellation reason")
    
    @field_validator('execution_id')
    def validate_execution_id(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError('Execution ID must be at least 10 characters')
        return v.strip()


class WorkflowCancelResponse(BaseResponse):
    """Response for workflow cancellation."""
    execution_id: str = Field(..., description="Execution identifier")
    cancelled_at: datetime = Field(default_factory=datetime.utcnow, description="Cancellation timestamp")
    final_status: WorkflowExecutionStatus = Field(..., description="Final execution status")


class WorkflowListRequest(BaseRequest):
    """Request to list workflow executions."""
    workflow_id: Optional[str] = Field(None, description="Filter by workflow ID")
    status: Optional[WorkflowExecutionStatus] = Field(None, description="Filter by status")
    date_from: Optional[datetime] = Field(None, description="Filter from date")
    date_to: Optional[datetime] = Field(None, description="Filter to date")
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    
    @field_validator('date_from', 'date_to')
    def validate_dates(cls, v):
        if v and v > datetime.utcnow():
            raise ValueError('Date cannot be in the future')
        return v


class WorkflowExecutionSummary(TimestampMixin):
    """Summary of workflow execution."""
    execution_id: str = Field(..., description="Execution identifier")
    workflow_id: str = Field(..., description="Workflow identifier")
    workflow_name: str = Field(..., description="Workflow name")
    status: WorkflowExecutionStatus = Field(..., description="Execution status")
    
    # Timing
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    duration_seconds: Optional[float] = Field(None, description="Total duration")
    
    # Progress
    total_nodes: int = Field(0, description="Total nodes")
    completed_nodes: int = Field(0, description="Completed nodes")
    failed_nodes: int = Field(0, description="Failed nodes")
    
    # Resource usage
    total_tokens_used: Optional[int] = Field(None, description="Total tokens consumed")
    estimated_cost: Optional[float] = Field(None, description="Estimated cost in USD")


class WorkflowListResponse(BaseResponse):
    """Response for workflow list request."""
    executions: List[WorkflowExecutionSummary] = Field(default_factory=list, description="Workflow executions")
    total_count: int = Field(0, description="Total number of executions")
    page: int = Field(1, description="Current page")
    page_size: int = Field(20, description="Page size")
    total_pages: int = Field(0, description="Total pages")


class NodeMetrics(BaseSchema):
    """Metrics for a workflow node."""
    node_id: str = Field(..., description="Node identifier")
    node_type: WorkflowNodeType = Field(..., description="Node type")
    
    # Execution metrics
    total_executions: int = Field(0, description="Total executions")
    successful_executions: int = Field(0, description="Successful executions")
    failed_executions: int = Field(0, description="Failed executions")
    
    # Performance metrics
    avg_duration_seconds: Optional[float] = Field(None, description="Average execution duration")
    min_duration_seconds: Optional[float] = Field(None, description="Minimum execution duration")
    max_duration_seconds: Optional[float] = Field(None, description="Maximum execution duration")
    
    # Resource metrics
    avg_memory_mb: Optional[float] = Field(None, description="Average memory usage")
    avg_tokens_used: Optional[float] = Field(None, description="Average tokens used")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100


class WorkflowMetrics(BaseSchema):
    """Aggregated metrics for a workflow."""
    workflow_id: str = Field(..., description="Workflow identifier")
    workflow_name: str = Field(..., description="Workflow name")
    
    # Overall metrics
    total_executions: int = Field(0, description="Total executions")
    successful_executions: int = Field(0, description="Successful executions")
    failed_executions: int = Field(0, description="Failed executions")
    
    # Performance metrics
    avg_duration_seconds: Optional[float] = Field(None, description="Average workflow duration")
    avg_total_cost: Optional[float] = Field(None, description="Average total cost")
    
    # Node metrics
    node_metrics: List[NodeMetrics] = Field(default_factory=list, description="Per-node metrics")
    
    # Time period
    metrics_period_start: datetime = Field(..., description="Metrics period start")
    metrics_period_end: datetime = Field(..., description="Metrics period end")
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100


class WorkflowMetricsRequest(BaseRequest):
    """Request for workflow metrics."""
    workflow_id: Optional[str] = Field(None, description="Specific workflow ID (optional)")
    date_from: datetime = Field(..., description="Metrics period start")
    date_to: datetime = Field(..., description="Metrics period end")
    include_node_metrics: bool = Field(True, description="Include per-node metrics")
    
    @field_validator('date_from', 'date_to')
    def validate_dates(cls, v):
        if v > datetime.utcnow():
            raise ValueError('Date cannot be in the future')
        return v
    
    @field_validator('date_to')
    def validate_date_range(cls, v, values):
        date_from = values.get('date_from')
        if date_from and v <= date_from:
            raise ValueError('End date must be after start date')
        return v


class WorkflowMetricsResponse(BaseResponse):
    """Response for workflow metrics."""
    metrics: List[WorkflowMetrics] = Field(default_factory=list, description="Workflow metrics")
    period_start: datetime = Field(..., description="Metrics period start")
    period_end: datetime = Field(..., description="Metrics period end") 
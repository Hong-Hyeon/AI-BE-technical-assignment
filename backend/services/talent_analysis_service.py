"""
Talent Analysis Service for centralizing business logic.

This service provides:
- Centralized talent analysis orchestration
- Business logic coordination
- Clean interface for routers
- Error handling and logging integration
"""

import asyncio
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from models.talent import AnalysisResult
from schema import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalysisMetadata,
    AnalysisProgress,
    AnalysisProgressEnum,
    StatusEnum,
    LLMConfiguration,
    AnalysisOptions
)
from core.exceptions import (
    BusinessLogicException,
    ValidationException,
    WorkflowException,
    ErrorCode
)
from config import get_settings
from workflows.talent_analysis_refactored import create_talent_analysis_workflow
from factories.llm_factory import LLMFactory
import logging

logger = logging.getLogger('api')


class TalentAnalysisService:
    """
    Service for managing talent analysis business logic.
    
    This service:
    - Orchestrates the entire analysis process
    - Manages analysis state and progress
    - Coordinates between workflows and data sources
    - Provides business logic abstractions
    """
    
    def __init__(self):
        """Initialize talent analysis service."""
        self.settings = get_settings()
        self.llm_factory = LLMFactory()
        self._active_analyses: Dict[str, Dict[str, Any]] = {}
    
    async def start_analysis(
        self, 
        request: AnalyzeRequest,
        trace_id: Optional[str] = None
    ) -> AnalyzeResponse:
        """
        Start a talent analysis process.
        
        Args:
            request: Analysis request with configuration
            trace_id: Optional trace ID for request tracking
            
        Returns:
            Analysis response with initial status or completed result
            
        Raises:
            ValidationException: If request validation fails
            BusinessLogicException: If analysis business logic fails
        """
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        try:
            # Validate request
            await self._validate_analysis_request(request)
            
            # Create analysis metadata
            metadata = self._create_analysis_metadata(analysis_id, request)
            
            # Determine execution mode (sync vs async)
            if self._should_run_async(request):
                # Start async analysis
                return await self._start_async_analysis(
                    analysis_id, request, metadata, trace_id
                )
            else:
                # Run sync analysis
                return await self._run_sync_analysis(
                    analysis_id, request, metadata, trace_id
                )
                
        except Exception as e:
            logger.error(
                f"Failed to start analysis for talent {request.talent_id}: {str(e)}",
                extra={'trace_id': trace_id, 'talent_id': request.talent_id}
            )
            
            # Update metadata with error
            if analysis_id in self._active_analyses:
                self._active_analyses[analysis_id]['metadata'].status = StatusEnum.FAILED
            
            if isinstance(e, (ValidationException, BusinessLogicException)):
                raise
            else:
                raise BusinessLogicException(
                    f"Failed to start analysis: {str(e)}",
                    original_exception=e
                )
    
    async def get_analysis_status(
        self, 
        analysis_id: str,
        trace_id: Optional[str] = None
    ) -> AnalyzeResponse:
        """
        Get the status of an ongoing or completed analysis.
        
        Args:
            analysis_id: Analysis identifier
            trace_id: Optional trace ID for request tracking
            
        Returns:
            Analysis response with current status
            
        Raises:
            ValidationException: If analysis ID is invalid
            BusinessLogicException: If analysis is not found
        """
        if not analysis_id or len(analysis_id) < 10:
            raise ValidationException("Invalid analysis ID format")
        
        analysis_data = self._active_analyses.get(analysis_id)
        if not analysis_data:
            raise BusinessLogicException(
                f"Analysis {analysis_id} not found",
                error_code=ErrorCode.ANALYSIS_FAILED
            )
        
        metadata = analysis_data['metadata']
        progress = analysis_data.get('progress')
        result = analysis_data.get('result')
        
        return AnalyzeResponse(
            analysis_id=analysis_id,
            result=result,
            metadata=metadata,
            progress=progress,
            message=self._get_status_message(metadata.status),
            trace_id=trace_id
        )
    
    async def list_analyses(
        self,
        talent_id: Optional[str] = None,
        status: Optional[StatusEnum] = None,
        limit: int = 20
    ) -> List[AnalysisMetadata]:
        """
        List analyses with optional filtering.
        
        Args:
            talent_id: Filter by talent ID
            status: Filter by status
            limit: Maximum number of results
            
        Returns:
            List of analysis metadata
        """
        analyses = []
        
        for analysis_id, data in self._active_analyses.items():
            metadata = data['metadata']
            
            # Apply filters
            if talent_id and metadata.talent_id != talent_id:
                continue
            if status and metadata.status != status:
                continue
            
            analyses.append(metadata)
            
            if len(analyses) >= limit:
                break
        
        # Sort by creation time (newest first)
        analyses.sort(key=lambda x: x.created_at, reverse=True)
        return analyses
    
    async def _validate_analysis_request(self, request: AnalyzeRequest):
        """Validate analysis request."""
        # Basic validation
        if not request.talent_id or len(request.talent_id.strip()) < 3:
            raise ValidationException(
                "Talent ID must be at least 3 characters",
                details=[{"field": "talent_id", "message": "Too short", "value": request.talent_id}]
            )
        
        # Validate LLM configuration
        llm_config = request.llm_config
        if llm_config.temperature < 0 or llm_config.temperature > 1:
            raise ValidationException(
                "Temperature must be between 0 and 1",
                details=[{"field": "llm_config.temperature", "message": "Out of range", "value": llm_config.temperature}]
            )
        
        # Validate analysis options
        if request.options.custom_prompt_additions and len(request.options.custom_prompt_additions) > 500:
            raise ValidationException(
                "Custom prompt additions cannot exceed 500 characters",
                details=[{"field": "options.custom_prompt_additions", "message": "Too long"}]
            )
    
    def _create_analysis_metadata(
        self, 
        analysis_id: str, 
        request: AnalyzeRequest
    ) -> AnalysisMetadata:
        """Create analysis metadata from request."""
        return AnalysisMetadata(
            analysis_id=analysis_id,
            talent_id=request.talent_id,
            analysis_type=request.analysis_type,
            status=StatusEnum.PENDING,
            llm_config=request.llm_config,
            vector_config=request.vector_config,
            options=request.options
        )
    
    def _should_run_async(self, request: AnalyzeRequest) -> bool:
        """Determine if analysis should run asynchronously."""
        # Run async if:
        # - Analysis type is FULL (more comprehensive)
        # - Custom configurations are complex
        # - System is configured for async by default
        
        if request.analysis_type.value == "full":
            return True
        
        if request.options.analysis_depth == "detailed":
            return True
        
        return self.settings.is_production  # Default to async in production
    
    async def _start_async_analysis(
        self,
        analysis_id: str,
        request: AnalyzeRequest,
        metadata: AnalysisMetadata,
        trace_id: Optional[str]
    ) -> AnalyzeResponse:
        """Start asynchronous analysis."""
        # Initialize progress tracking
        progress = AnalysisProgress(
            stage=AnalysisProgressEnum.INITIALIZING,
            progress_percentage=0,
            current_task="Starting analysis"
        )
        
        # Store analysis state
        self._active_analyses[analysis_id] = {
            'metadata': metadata,
            'progress': progress,
            'result': None,
            'request': request,
            'trace_id': trace_id
        }
        
        # Start background task
        asyncio.create_task(self._run_background_analysis(analysis_id))
        
        return AnalyzeResponse.create_async_response(
            analysis_id=analysis_id,
            metadata=metadata,
            message="Analysis started successfully",
            trace_id=trace_id
        )
    
    async def _run_sync_analysis(
        self,
        analysis_id: str,
        request: AnalyzeRequest,
        metadata: AnalysisMetadata,
        trace_id: Optional[str]
    ) -> AnalyzeResponse:
        """Run synchronous analysis."""
        try:
            # Update metadata
            metadata.status = StatusEnum.IN_PROGRESS
            metadata.started_at = datetime.utcnow()
            
            # Create and execute workflow
            workflow = create_talent_analysis_workflow()
            llm_model = self.llm_factory.create_llm(request.llm_config)
            
            result = await workflow.run_analysis(request.talent_id, llm_model)
            
            # Update metadata
            metadata.status = StatusEnum.COMPLETED
            metadata.completed_at = datetime.utcnow()
            metadata.duration_seconds = (metadata.completed_at - metadata.started_at).total_seconds()
            
            return AnalyzeResponse.create_completed_response(
                analysis_id=analysis_id,
                result=result,
                metadata=metadata,
                message="Analysis completed successfully",
                trace_id=trace_id
            )
            
        except Exception as e:
            metadata.status = StatusEnum.FAILED
            metadata.completed_at = datetime.utcnow()
            
            logger.error(f"Sync analysis failed: {str(e)}", exc_info=True)
            
            raise BusinessLogicException(
                f"Analysis execution failed: {str(e)}",
                error_code=ErrorCode.ANALYSIS_FAILED,
                original_exception=e
            )
    
    async def _run_background_analysis(self, analysis_id: str):
        """Run analysis in background for async execution."""
        analysis_data = self._active_analyses.get(analysis_id)
        if not analysis_data:
            return
        
        metadata = analysis_data['metadata']
        request = analysis_data['request']
        trace_id = analysis_data['trace_id']
        
        try:
            # Update status
            metadata.status = StatusEnum.IN_PROGRESS
            metadata.started_at = datetime.utcnow()
            
            # Update progress
            progress = AnalysisProgress(
                stage=AnalysisProgressEnum.LOADING_DATA,
                progress_percentage=10,
                current_task="Loading talent data"
            )
            analysis_data['progress'] = progress
            
            # Create and execute workflow
            workflow = create_talent_analysis_workflow()
            llm_model = self.llm_factory.create_llm(request.llm_config)
            
            # Execute with progress updates
            result = await self._execute_with_progress_updates(
                workflow, request.talent_id, llm_model, analysis_id
            )
            
            # Update final state
            metadata.status = StatusEnum.COMPLETED
            metadata.completed_at = datetime.utcnow()
            metadata.duration_seconds = (metadata.completed_at - metadata.started_at).total_seconds()
            
            analysis_data['result'] = result
            analysis_data['progress'] = AnalysisProgress(
                stage=AnalysisProgressEnum.COMPLETED,
                progress_percentage=100,
                current_task="Analysis completed"
            )
            
            logger.info(
                f"Background analysis {analysis_id} completed successfully",
                extra={'trace_id': trace_id, 'analysis_id': analysis_id}
            )
            
        except Exception as e:
            # Update error state
            metadata.status = StatusEnum.FAILED
            metadata.completed_at = datetime.utcnow()
            
            analysis_data['progress'] = AnalysisProgress(
                stage=AnalysisProgressEnum.FAILED,
                progress_percentage=0,
                current_task="Analysis failed",
                error_message=str(e)
            )
            
            logger.error(
                f"Background analysis {analysis_id} failed: {str(e)}",
                extra={'trace_id': trace_id, 'analysis_id': analysis_id},
                exc_info=True
            )
    
    async def _execute_with_progress_updates(
        self,
        workflow,
        talent_id: str,
        llm_model,
        analysis_id: str
    ) -> AnalysisResult:
        """Execute workflow with progress updates."""
        analysis_data = self._active_analyses[analysis_id]
        
        # Update progress through stages
        stages = [
            (AnalysisProgressEnum.PROCESSING_EDUCATION, 30, "Analyzing education"),
            (AnalysisProgressEnum.PROCESSING_EXPERIENCE, 50, "Analyzing experience"),
            (AnalysisProgressEnum.PROCESSING_SKILLS, 70, "Processing skills"),
            (AnalysisProgressEnum.GENERATING_SUMMARY, 90, "Generating summary")
        ]
        
        for stage, percentage, task in stages:
            analysis_data['progress'] = AnalysisProgress(
                stage=stage,
                progress_percentage=percentage,
                current_task=task
            )
            await asyncio.sleep(0.1)  # Allow other tasks to run
        
        # Execute the actual workflow
        result = await workflow.run_analysis(talent_id, llm_model)
        
        return result
    
    def _get_status_message(self, status: StatusEnum) -> str:
        """Get human-readable status message."""
        messages = {
            StatusEnum.PENDING: "Analysis is pending",
            StatusEnum.IN_PROGRESS: "Analysis is in progress",
            StatusEnum.COMPLETED: "Analysis completed successfully",
            StatusEnum.FAILED: "Analysis failed",
            StatusEnum.CANCELLED: "Analysis was cancelled"
        }
        return messages.get(status, "Unknown status")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        total_analyses = len(self._active_analyses)
        status_counts = {}
        
        for data in self._active_analyses.values():
            status = data['metadata'].status
            status_counts[status.value] = status_counts.get(status.value, 0) + 1
        
        return {
            "total_analyses": total_analyses,
            "status_distribution": status_counts,
            "active_analyses": len([
                d for d in self._active_analyses.values() 
                if d['metadata'].status == StatusEnum.IN_PROGRESS
            ])
        } 
"""
Base workflow class for LangGraph workflows.

This module provides:
- BaseWorkflow abstract class
- Common workflow functionality
- Error handling and logging integration
- State management utilities
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph
from config.logging_config import WorkflowLogger
from core.exceptions import WorkflowException, ErrorCode
from .state import TalentAnalysisState, get_state_summary, validate_state


class BaseWorkflow(ABC):
    """
    Base class for all LangGraph workflows.
    
    Provides common functionality like logging, error handling,
    state validation, and workflow execution patterns.
    """
    
    def __init__(self, workflow_name: str):
        """
        Initialize base workflow.
        
        Args:
            workflow_name: Name of the workflow for logging
        """
        self.workflow_name = workflow_name
        self.logger = WorkflowLogger(workflow_name)
        self.graph = None
        self._build_time = None
    
    @abstractmethod
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Must be implemented by subclasses to define the specific
        workflow structure, nodes, and edges.
        
        Returns:
            Compiled StateGraph instance
        """
        pass
    
    def build_graph(self) -> StateGraph:
        """
        Build and compile the workflow graph with timing and logging.
        
        Returns:
            Compiled StateGraph instance
        """
        if self.graph is not None:
            return self.graph
        
        build_start = time.time()
        self.logger.logger.info(f"🔧 Building {self.workflow_name} workflow graph...")
        
        try:
            self.graph = self._build_graph()
            self._build_time = time.time() - build_start
            
            self.logger.logger.info(
                f"✅ {self.workflow_name} workflow graph built successfully in {self._build_time:.2f}s"
            )
            
            return self.graph
            
        except Exception as e:
            build_time = time.time() - build_start
            self.logger.logger.error(
                f"❌ Failed to build {self.workflow_name} workflow graph after {build_time:.2f}s: {str(e)}"
            )
            raise WorkflowException(
                workflow_name=self.workflow_name,
                message=f"Failed to build workflow graph: {str(e)}",
                original_exception=e
            )
    
    async def execute_node_safely(
        self,
        node_name: str,
        node_func,
        state: TalentAnalysisState,
        required_fields: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Execute a workflow node with error handling and logging.
        
        Args:
            node_name: Name of the node for logging
            node_func: Node function to execute
            state: Current workflow state
            required_fields: List of required state fields
            
        Returns:
            Node execution result
            
        Raises:
            WorkflowException: If node execution fails
        """
        node_start_time = time.time()
        
        # Validate required fields
        if required_fields and not validate_state(state, required_fields):
            missing_fields = [f for f in required_fields if f not in state or state[f] is None]
            raise WorkflowException(
                workflow_name=self.workflow_name,
                node_name=node_name,
                message=f"Missing required state fields: {missing_fields}"
            )
        
        # Log node start with state summary
        state_summary = get_state_summary(state)
        self.logger.log_node_start(node_name, state_summary)
        
        try:
            # Execute the node
            result = await node_func(state)
            
            # Log successful completion
            node_duration = time.time() - node_start_time
            self.logger.log_node_complete(
                node_name,
                node_duration,
                {"result_keys": list(result.keys()) if isinstance(result, dict) else None}
            )
            
            return result
            
        except Exception as e:
            # Log node error
            node_duration = time.time() - node_start_time
            self.logger.log_node_error(node_name, e, state_summary)
            
            # Wrap in workflow exception
            raise WorkflowException(
                workflow_name=self.workflow_name,
                node_name=node_name,
                message=f"Node execution failed: {str(e)}",
                original_exception=e
            )
    
    def log_workflow_metrics(
        self,
        total_duration: float,
        state: TalentAnalysisState,
        success: bool = True,
        error: Optional[Exception] = None
    ):
        """
        Log workflow execution metrics.
        
        Args:
            total_duration: Total workflow execution time
            state: Final workflow state
            success: Whether workflow completed successfully
            error: Exception if workflow failed
        """
        metrics = {
            "workflow_name": self.workflow_name,
            "total_duration": total_duration,
            "build_time": self._build_time,
            "success": success,
            "talent_id": state.get("talent_id", "unknown")
        }
        
        # Add success metrics
        if success:
            analysis_result = state.get("analysis_result")
            if analysis_result:
                metrics.update({
                    "experience_tags_count": len(analysis_result.experience_tags),
                    "processing_time": analysis_result.processing_time
                })
        
        # Add error metrics
        if error:
            metrics.update({
                "error_type": type(error).__name__,
                "error_message": str(error)
            })
        
        # Log based on success/failure
        if success:
            self.logger.logger.info(
                f"📊 {self.workflow_name} metrics: {metrics}"
            )
        else:
            self.logger.logger.error(
                f"📊 {self.workflow_name} failed metrics: {metrics}"
            )
    
    async def run_workflow(
        self,
        initial_state: TalentAnalysisState,
        **kwargs
    ) -> TalentAnalysisState:
        """
        Run the complete workflow with comprehensive logging and error handling.
        
        Args:
            initial_state: Initial workflow state
            **kwargs: Additional arguments for workflow execution
            
        Returns:
            Final workflow state
            
        Raises:
            WorkflowException: If workflow execution fails
        """
        workflow_start_time = time.time()
        talent_id = initial_state.get("talent_id", "unknown")
        
        # Ensure graph is built
        if self.graph is None:
            self.build_graph()
        
        # Log workflow start
        self.logger.log_workflow_start(talent_id, get_state_summary(initial_state))
        
        try:
            # Execute the workflow
            self.logger.logger.info(f"🔄 Starting {self.workflow_name} execution...")
            final_state = await self.graph.ainvoke(initial_state, **kwargs)
            
            # Calculate metrics
            total_duration = time.time() - workflow_start_time
            
            # Log success
            result_summary = {}
            analysis_result = final_state.get("analysis_result")
            if analysis_result:
                result_summary = {
                    "experience_tags_count": len(analysis_result.experience_tags),
                    "processing_time": analysis_result.processing_time
                }
            
            self.logger.log_workflow_complete(talent_id, total_duration, result_summary)
            self.log_workflow_metrics(total_duration, final_state, success=True)
            
            return final_state
            
        except Exception as e:
            # Calculate metrics for failed execution
            total_duration = time.time() - workflow_start_time
            
            # Log failure
            self.logger.logger.error(
                f"💥 {self.workflow_name} failed for talent_id {talent_id} "
                f"after {total_duration:.2f}s: {str(e)}",
                exc_info=True
            )
            
            self.log_workflow_metrics(total_duration, initial_state, success=False, error=e)
            
            # Wrap in workflow exception if not already
            if isinstance(e, WorkflowException):
                raise
            else:
                raise WorkflowException(
                    workflow_name=self.workflow_name,
                    message=f"Workflow execution failed: {str(e)}",
                    original_exception=e
                )
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """
        Get information about the workflow.
        
        Returns:
            Dictionary with workflow metadata
        """
        return {
            "workflow_name": self.workflow_name,
            "graph_built": self.graph is not None,
            "build_time": self._build_time,
            "logger_name": self.logger.logger.name
        } 
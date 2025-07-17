"""
Refactored LangGraph workflow for talent analysis using modular nodes.

This module provides the new TalentAnalysisWorkflow that uses the
decomposed node architecture for better maintainability and testing.
"""

import time
from typing import Any

from langgraph.graph import StateGraph, END
from models.talent import AnalysisResult

from .base import BaseWorkflow, TalentAnalysisState, create_initial_state
from .nodes import (
    DataLoaderNode,
    ContextPreprocessorNode,
    VectorSearchNode,
    EducationAnalysisNode,
    PositionAnalysisNode,
    AggregationNode
)


class TalentAnalysisWorkflow(BaseWorkflow):
    """
    Refactored LangGraph workflow for talent analysis using modular nodes.
    
    This workflow:
    - Uses decomposed node architecture
    - Inherits from BaseWorkflow for common functionality
    - Provides better error handling and logging
    - Supports parallel execution of analysis nodes
    """
    
    def __init__(self):
        """Initialize the talent analysis workflow."""
        super().__init__("TalentAnalysis")
        
        # Initialize node instances
        self.data_loader = DataLoaderNode(self.logger)
        self.context_preprocessor = ContextPreprocessorNode(self.logger)
        self.vector_search = VectorSearchNode(self.logger)
        self.education_analysis = EducationAnalysisNode(self.logger)
        self.position_analysis = PositionAnalysisNode(self.logger)
        self.aggregation = AggregationNode(self.logger)
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow with integrated preprocessing and vector search.
        
        Returns:
            Compiled StateGraph instance
        """
        workflow = StateGraph(TalentAnalysisState)
        
        # Add nodes with proper async execution
        workflow.add_node("load_talent_data", self._load_talent_data_wrapper)
        workflow.add_node("preprocess_context", self._preprocess_context_wrapper)
        workflow.add_node("vector_search", self._vector_search_wrapper)
        workflow.add_node("analyze_education", self._analyze_education_wrapper)
        workflow.add_node("analyze_positions", self._analyze_positions_wrapper)
        workflow.add_node("aggregate_results", self._aggregate_results_wrapper)
        
        # Sequential preprocessing flow
        workflow.set_entry_point("load_talent_data")
        workflow.add_edge("load_talent_data", "preprocess_context")
        workflow.add_edge("preprocess_context", "vector_search")
        
        # Parallel analysis after preprocessing
        workflow.add_edge("vector_search", "analyze_education")
        workflow.add_edge("vector_search", "analyze_positions")
        
        # Final aggregation (waits for both analysis nodes)
        workflow.add_edge("analyze_education", "aggregate_results")
        workflow.add_edge("analyze_positions", "aggregate_results")
        workflow.add_edge("aggregate_results", END)
        
        return workflow.compile()
    
    # Node wrapper methods for compatibility with LangGraph
    
    async def _load_talent_data_wrapper(self, state: TalentAnalysisState) -> TalentAnalysisState:
        """Wrapper for data loader node."""
        result = await self.data_loader.execute(state)
        return {**state, **result}
    
    async def _preprocess_context_wrapper(self, state: TalentAnalysisState) -> TalentAnalysisState:
        """Wrapper for context preprocessor node."""
        result = await self.context_preprocessor.execute(state)
        return {**state, **result}
    
    async def _vector_search_wrapper(self, state: TalentAnalysisState) -> TalentAnalysisState:
        """Wrapper for vector search node."""
        result = await self.vector_search.execute(state)
        return {**state, **result}
    
    async def _analyze_education_wrapper(self, state: TalentAnalysisState) -> TalentAnalysisState:
        """Wrapper for education analysis node."""
        result = await self.education_analysis.execute(state)
        return {**state, **result}
    
    async def _analyze_positions_wrapper(self, state: TalentAnalysisState) -> TalentAnalysisState:
        """Wrapper for position analysis node."""
        result = await self.position_analysis.execute(state)
        return {**state, **result}
    
    async def _aggregate_results_wrapper(self, state: TalentAnalysisState) -> TalentAnalysisState:
        """Wrapper for aggregation node."""
        result = await self.aggregation.execute(state)
        return {**state, **result}
    
    async def run_analysis(
        self,
        talent_id: str,
        llm_model: Any
    ) -> AnalysisResult:
        """
        Run the complete talent analysis workflow.
        
        Args:
            talent_id: Identifier for the talent to analyze
            llm_model: LLM model instance for analysis
            
        Returns:
            AnalysisResult with experience tags and metadata
            
        Raises:
            WorkflowException: If workflow execution fails
        """
        # Extract the LangChain ChatOpenAI model if it's wrapped
        if hasattr(llm_model, 'get_langchain_model'):
            langchain_model = llm_model.get_langchain_model()
            self.logger.logger.debug("Extracted LangChain model from wrapper")
        else:
            langchain_model = llm_model
            self.logger.logger.debug("Using model directly")
        
        # Create initial state
        initial_state = create_initial_state(talent_id, langchain_model)
        
        # Run the workflow using base class method
        final_state = await self.run_workflow(initial_state)
        
        # Return the analysis result
        return final_state["analysis_result"]
    
    def get_node_info(self) -> dict:
        """
        Get information about the workflow nodes.
        
        Returns:
            Dictionary with node information
        """
        return {
            "workflow_name": self.workflow_name,
            "nodes": [
                "load_talent_data",
                "preprocess_context", 
                "vector_search",
                "analyze_education",
                "analyze_positions",
                "aggregate_results"
            ],
            "parallel_nodes": ["analyze_education", "analyze_positions"],
            "sequential_nodes": ["load_talent_data", "preprocess_context", "vector_search"],
            "final_node": "aggregate_results"
        }


# Create a singleton instance for backward compatibility
talent_analysis_workflow = TalentAnalysisWorkflow()


# Factory function for creating new instances
def create_talent_analysis_workflow() -> TalentAnalysisWorkflow:
    """
    Create a new talent analysis workflow instance.
    
    Returns:
        New TalentAnalysisWorkflow instance
    """
    return TalentAnalysisWorkflow()


# Utility function for quick analysis
async def analyze_talent(talent_id: str, llm_model: Any) -> AnalysisResult:
    """
    Quick utility function to analyze a talent.
    
    Args:
        talent_id: Identifier for the talent to analyze
        llm_model: LLM model instance for analysis
        
    Returns:
        AnalysisResult with experience tags and metadata
    """
    workflow = create_talent_analysis_workflow()
    return await workflow.run_analysis(talent_id, llm_model) 
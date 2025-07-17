"""
State definitions for talent analysis workflows.

This module provides:
- TalentAnalysisState TypedDict for LangGraph
- State merging utilities
- State validation functions
"""

from typing import Dict, Any, List, TypedDict, Annotated

from models.talent import TalentData, AnalysisResult


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two dictionaries for LangGraph state.
    
    This function is used as a reducer for Annotated state fields
    to handle concurrent updates from parallel nodes.
    
    Args:
        left: Left dictionary (existing state)
        right: Right dictionary (new state)
        
    Returns:
        Merged dictionary with right taking precedence
    """
    if not left:
        return right
    if not right:
        return left
    return {**left, **right}


class TalentAnalysisState(TypedDict):
    """
    State for talent analysis workflow.
    
    This TypedDict defines the complete state that flows through
    the LangGraph workflow, including input data, intermediate
    results, and final analysis output.
    """
    # Input data
    talent_id: str
    llm_model: Any  # ChatOpenAI or compatible model
    
    # Preprocessing results
    talent_data: TalentData
    preprocessed_context: Annotated[Dict[str, Any], merge_dicts]
    vector_search_results: Annotated[Dict[str, Any], merge_dicts]
    
    # Analysis results - using Annotated to handle concurrent updates
    education_analysis: Annotated[Dict[str, Any], merge_dicts]
    position_analysis: Annotated[Dict[str, Any], merge_dicts]
    
    # Final result
    analysis_result: AnalysisResult
    processing_start_time: float


def validate_state(state: TalentAnalysisState, required_fields: List[str]) -> bool:
    """
    Validate that required fields are present in the state.
    
    Args:
        state: Current workflow state
        required_fields: List of field names that must be present
        
    Returns:
        True if all required fields are present and not None
    """
    for field in required_fields:
        if field not in state or state[field] is None:
            return False
    return True


def get_state_summary(state: TalentAnalysisState) -> Dict[str, Any]:
    """
    Get a summary of the current state for logging/debugging.
    
    Args:
        state: Current workflow state
        
    Returns:
        Summary dictionary with key state information
    """
    summary = {
        "talent_id": state.get("talent_id", "unknown"),
        "has_talent_data": state.get("talent_data") is not None,
        "has_preprocessed_context": bool(state.get("preprocessed_context")),
        "has_vector_results": bool(state.get("vector_search_results")),
        "has_education_analysis": bool(state.get("education_analysis")),
        "has_position_analysis": bool(state.get("position_analysis")),
        "has_final_result": state.get("analysis_result") is not None
    }
    
    # Add counts if data is available
    talent_data = state.get("talent_data")
    if talent_data:
        summary.update({
            "education_count": len(talent_data.educations),
            "position_count": len(talent_data.positions),
            "skills_count": len(talent_data.skills)
        })
    
    return summary


def create_initial_state(talent_id: str, llm_model: Any) -> TalentAnalysisState:
    """
    Create initial state for talent analysis workflow.
    
    Args:
        talent_id: Identifier for the talent to analyze
        llm_model: LLM model instance to use for analysis
        
    Returns:
        Initial state with required fields set
    """
    import time
    
    return TalentAnalysisState(
        talent_id=talent_id,
        llm_model=llm_model,
        talent_data=None,  # Will be filled by load_talent_data node
        preprocessed_context={},
        vector_search_results={},
        education_analysis={},
        position_analysis={},
        analysis_result=None,
        processing_start_time=time.time()
    ) 
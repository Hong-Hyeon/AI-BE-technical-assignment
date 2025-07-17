"""
Base module for workflow components.

This module provides:
- Common state definitions
- Utility functions
- Base workflow classes
- Shared components for LangGraph workflows
"""

from .state import TalentAnalysisState, merge_dicts, create_initial_state
from .base_workflow import BaseWorkflow

__all__ = [
    "TalentAnalysisState",
    "merge_dicts", 
    "create_initial_state",
    "BaseWorkflow"
] 
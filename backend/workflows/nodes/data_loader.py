"""
Data Loader Node for talent analysis workflow.

This module provides the DataLoaderNode class responsible for
loading talent data from various data sources with proper
error handling and logging.
"""

import time
from typing import Dict, Any

from config.logging_config import WorkflowLogger
from core.exceptions import WorkflowException, ErrorCode
from ..base.state import TalentAnalysisState


class DataLoaderNode:
    """
    Node responsible for loading talent data from data sources.
    
    This node:
    - Initializes data source manager
    - Loads talent data by ID
    - Validates data integrity
    - Logs data transformation metrics
    """
    
    def __init__(self, logger: WorkflowLogger):
        """
        Initialize data loader node.
        
        Args:
            logger: Workflow logger instance
        """
        self.logger = logger
    
    async def execute(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """
        Load talent data from data source.
        
        Args:
            state: Current workflow state containing talent_id
            
        Returns:
            Dictionary with loaded talent_data
            
        Raises:
            WorkflowException: If data loading fails
        """
        node_start_time = time.time()
        talent_id = state["talent_id"]
        
        self.logger.log_node_start("load_talent_data", {"talent_id": talent_id})
        
        try:
            # Import here to avoid circular dependencies
            from factories.data_source_factory import DefaultDataSourceFactory, DataSourceManager
            
            # Initialize data source manager
            self.logger.logger.debug("Initializing data source manager...")
            data_factory = DefaultDataSourceFactory("./example_datas")
            data_manager = DataSourceManager(data_factory)
            
            # Load talent data
            self.logger.logger.debug(f"Loading talent data for ID: {talent_id}")
            talent_data = data_manager.get_talent_data(talent_id)
            
            # Validate loaded data
            self._validate_talent_data(talent_data, talent_id)
            
            # Log data transformation info
            data_summary = {
                "education_count": len(talent_data.educations),
                "position_count": len(talent_data.positions),
                "skills_count": len(talent_data.skills),
                "full_name": f"{talent_data.first_name} {talent_data.last_name}"
            }
            
            self.logger.log_data_transformation(
                "talent_data_loading",
                {"talent_id": talent_id},
                data_summary
            )
            
            # Log successful completion
            node_duration = time.time() - node_start_time
                         self.logger.log_node_complete(
                 "load_talent_data", 
                 node_duration,
                 {
                     "data_loaded": f"{len(talent_data.educations)} educations, "
                                    f"{len(talent_data.positions)} positions, "
                                    f"{len(talent_data.skills)} skills",
                     "talent_name": f"{talent_data.first_name} {talent_data.last_name}"
                 }
             )
            
            return {"talent_data": talent_data}
            
        except Exception as e:
            # Log error with context
            error_context = {
                "talent_id": talent_id,
                "error_type": type(e).__name__,
                "node_duration": time.time() - node_start_time
            }
            self.logger.log_node_error("load_talent_data", e, error_context)
            
            # Wrap in workflow exception
            raise WorkflowException(
                workflow_name="TalentAnalysis",
                node_name="load_talent_data",
                message=f"Failed to load talent data for ID {talent_id}: {str(e)}",
                original_exception=e
            )
    
    def _validate_talent_data(self, talent_data, talent_id: str):
        """
        Validate the loaded talent data.
        
        Args:
            talent_data: Loaded talent data object
            talent_id: Original talent ID for error context
            
        Raises:
            WorkflowException: If validation fails
        """
        if not talent_data:
            raise WorkflowException(
                workflow_name="TalentAnalysis",
                node_name="load_talent_data",
                message=f"No data found for talent ID: {talent_id}"
            )
        
        # Check required fields
        required_fields = ['first_name', 'last_name', 'educations', 'positions']
        for field in required_fields:
            if not hasattr(talent_data, field):
                raise WorkflowException(
                    workflow_name="TalentAnalysis",
                    node_name="load_talent_data",
                    message=f"Talent data missing required field: {field}"
                )
        
        # Log validation results
        validation_summary = {
            "has_name": bool(talent_data.first_name and talent_data.last_name),
            "has_educations": len(talent_data.educations) > 0,
            "has_positions": len(talent_data.positions) > 0,
            "has_skills": len(getattr(talent_data, 'skills', [])) > 0,
            "has_headline": bool(getattr(talent_data, 'headline', '')),
            "has_summary": bool(getattr(talent_data, 'summary', ''))
        }
        
        self.logger.logger.debug(f"Talent data validation: {validation_summary}")
        
        # Warn about missing optional data
        if not validation_summary['has_educations']:
            self.logger.logger.warning(f"No education data found for talent {talent_id}")
        
        if not validation_summary['has_positions']:
            self.logger.logger.warning(f"No position data found for talent {talent_id}")
        
        if not validation_summary['has_skills']:
            self.logger.logger.warning(f"No skills data found for talent {talent_id}")


# Factory function for easy instantiation
def create_data_loader_node(logger: WorkflowLogger) -> DataLoaderNode:
    """
    Create a data loader node instance.
    
    Args:
        logger: Workflow logger instance
        
    Returns:
        Configured DataLoaderNode instance
    """
    return DataLoaderNode(logger) 
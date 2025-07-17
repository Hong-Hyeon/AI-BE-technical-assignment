"""
Workflow Service for workflow management and execution.
"""

from typing import Dict, Any, List, Optional
from workflows.talent_analysis_refactored import create_talent_analysis_workflow
from core.exceptions import WorkflowException
import logging

logger = logging.getLogger('workflow')


class WorkflowService:
    """Service for managing workflow execution and coordination."""
    
    def __init__(self):
        self._workflow_instances = {}
    
    async def execute_talent_analysis(self, talent_id: str, llm_model) -> Any:
        """Execute talent analysis workflow."""
        workflow = create_talent_analysis_workflow()
        return await workflow.run_analysis(talent_id, llm_model)
    
    def get_workflow_info(self, workflow_name: str) -> Dict[str, Any]:
        """Get workflow information."""
        if workflow_name == "talent_analysis":
            workflow = create_talent_analysis_workflow()
            return workflow.get_node_info()
        return {}
    
    def list_available_workflows(self) -> List[str]:
        """List available workflows."""
        return ["talent_analysis"] 
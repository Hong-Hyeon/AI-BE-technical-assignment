"""
Education Analysis Node for talent analysis workflow.

This module provides the EducationAnalysisNode class responsible for
analyzing education background using LLM with proper error handling.
"""

import time
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage
from config.logging_config import WorkflowLogger
from core.exceptions import WorkflowException, ExternalServiceException, ErrorCode
from ..base.state import TalentAnalysisState


class EducationAnalysisNode:
    """
    Node responsible for analyzing education background in parallel.
    
    This node:
    - Analyzes each education record using LLM
    - Generates education summaries
    - Handles LLM failures gracefully
    - Logs analysis metrics
    """
    
    def __init__(self, logger: WorkflowLogger):
        """
        Initialize education analysis node.
        
        Args:
            logger: Workflow logger instance
        """
        self.logger = logger
    
    async def execute(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """
        Analyze education background in parallel.
        
        Args:
            state: Current workflow state containing talent_data and llm_model
            
        Returns:
            Dictionary with education_analysis results
        """
        node_start_time = time.time()
        talent_data = state["talent_data"]
        llm_model = state["llm_model"]
        talent_id = state["talent_id"]
        
        education_count = len(talent_data.educations)
        schools = [edu.school_name for edu in talent_data.educations]
        
        self.logger.log_node_start(
            "analyze_education", 
            {
                "talent_id": talent_id,
                "educations_to_analyze": education_count,
                "schools": schools
            }
        )
        
        education_analysis = []
        successful_analyses = 0
        
        try:
            # Import prompt factory here to avoid circular dependencies
            from factories.prompt_factory import get_education_prompt
            
            for i, education in enumerate(talent_data.educations):
                self.logger.logger.debug(
                    f"Analyzing education {i + 1}/{education_count}: {education.school_name}"
                )
                
                # Generate prompt for this education
                prompt = get_education_prompt(
                    school_name=education.school_name,
                    degree_name=education.degree_name,
                    field_of_study=education.field_of_study,
                    start_end_date=education.start_end_date
                )
                
                # Analyze this education
                analysis_result = await self._analyze_single_education(
                    llm_model, education, prompt, i + 1, education_count
                )
                
                education_analysis.append(analysis_result)
                
                if analysis_result.get("success", False):
                    successful_analyses += 1
            
            # Generate summary
            summary = self._create_education_summary(education_analysis)
            
            # Log data transformation
            self.logger.log_data_transformation(
                "education_analysis",
                {"input_educations": education_count},
                {
                    "successful_analyses": successful_analyses,
                    "failed_analyses": education_count - successful_analyses,
                    "success_rate": successful_analyses / education_count if education_count > 0 else 0
                }
            )
            
            # Create result
            result = {
                "education_analysis": {
                    "results": education_analysis,
                    "summary": summary,
                    "metadata": {
                        "total_educations": education_count,
                        "successful_analyses": successful_analyses,
                        "failed_analyses": education_count - successful_analyses,
                        "schools_analyzed": [result["school"] for result in education_analysis if result.get("success")]
                    }
                }
            }
            
            # Log successful completion
            node_duration = time.time() - node_start_time
            self.logger.log_node_complete(
                "analyze_education", 
                node_duration,
                {
                    "educations_analyzed": education_count,
                    "successful": successful_analyses,
                    "success_rate": f"{successful_analyses}/{education_count}"
                }
            )
            
            return result
            
        except Exception as e:
            # Log error with context
            error_context = {
                "talent_id": talent_id,
                "education_count": education_count,
                "error_type": type(e).__name__,
                "node_duration": time.time() - node_start_time
            }
            self.logger.log_node_error("analyze_education", e, error_context)
            
            # Return partial results if available
            if education_analysis:
                self.logger.logger.warning(
                    f"Education analysis partially failed, returning {len(education_analysis)} results"
                )
                return {
                    "education_analysis": {
                        "results": education_analysis,
                        "summary": self._create_education_summary(education_analysis),
                        "metadata": {
                            "total_educations": education_count,
                            "successful_analyses": successful_analyses,
                            "error": str(e)
                        }
                    }
                }
            else:
                # Return empty results
                return {
                    "education_analysis": {
                        "results": [],
                        "summary": "교육 분석 실패",
                        "metadata": {
                            "total_educations": education_count,
                            "successful_analyses": 0,
                            "error": str(e)
                        }
                    }
                }
    
    async def _analyze_single_education(
        self, 
        llm_model, 
        education, 
        prompt: str, 
        current: int, 
        total: int
    ) -> Dict[str, Any]:
        """
        Analyze a single education record.
        
        Args:
            llm_model: LLM model instance
            education: Education object to analyze
            prompt: Generated prompt for this education
            current: Current education number (for logging)
            total: Total number of educations
            
        Returns:
            Analysis result with success status
        """
        try:
            llm_start_time = time.time()
            response = await llm_model.ainvoke([HumanMessage(content=prompt)])
            llm_time = time.time() - llm_start_time
            
            # Log LLM call
            self.logger.log_llm_call(
                "analyze_education",
                getattr(llm_model, 'model_name', 'unknown'),
                len(prompt),
                len(response.content),
                llm_time
            )
            
            self.logger.logger.debug(f"✅ Completed analysis for {education.school_name}")
            
            return {
                "school": education.school_name,
                "degree": education.degree_name,
                "field": education.field_of_study,
                "analysis": response.content,
                "success": True,
                "llm_time": llm_time
            }
            
        except Exception as e:
            self.logger.logger.error(
                f"❌ Failed to analyze {education.school_name}: {str(e)}"
            )
            return {
                "school": education.school_name,
                "degree": education.degree_name,
                "field": education.field_of_study,
                "analysis": f"분석 실패: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def _create_education_summary(self, education_analysis: List[Dict]) -> str:
        """
        Create a summary of education analysis results.
        
        Args:
            education_analysis: List of education analysis results
            
        Returns:
            Summary string
        """
        if not education_analysis:
            return "교육 정보 없음"
        
        successful_results = [result for result in education_analysis if result.get("success", False)]
        
        if not successful_results:
            return "교육 분석 실패"
        
        summaries = []
        for result in successful_results:
            school = result["school"]
            analysis = result["analysis"]
            summaries.append(f"- {school}: {analysis}")
        
        return "\n".join(summaries)


# Factory function for easy instantiation
def create_education_analysis_node(logger: WorkflowLogger) -> EducationAnalysisNode:
    """
    Create an education analysis node instance.
    
    Args:
        logger: Workflow logger instance
        
    Returns:
        Configured EducationAnalysisNode instance
    """
    return EducationAnalysisNode(logger) 
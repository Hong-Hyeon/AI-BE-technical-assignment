"""
Aggregation Node for talent analysis workflow.
"""

import time
import json
from typing import Dict, Any, List
from datetime import datetime
from langchain_core.messages import HumanMessage
from config.logging_config import WorkflowLogger
from core.exceptions import WorkflowException
from models.talent import AnalysisResult, ExperienceTag
from ..base.state import TalentAnalysisState


class AggregationNode:
    """Node responsible for aggregating education and position analyses into final summary."""
    
    def __init__(self, logger: WorkflowLogger):
        self.logger = logger
    
    async def execute(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """Aggregate education and position analyses into final summary."""
        node_start_time = time.time()
        talent_data = state["talent_data"]
        education_analysis = state.get("education_analysis", {})
        position_analysis = state.get("position_analysis", {})
        vector_context = state.get("vector_search_results", {})
        llm_model = state["llm_model"]
        processing_start_time = state["processing_start_time"]
        talent_id = state["talent_id"]
        
        context_items = len(vector_context.get("companies", [])) + len(vector_context.get("news", []))
        
        self.logger.log_node_start("aggregate_results", {
            "talent_id": talent_id,
            "has_education_analysis": bool(education_analysis),
            "has_position_analysis": bool(position_analysis),
            "vector_context_items": context_items
        })
        
        try:
            from factories.prompt_factory import get_aggregation_prompt
            
            # Create comprehensive summary prompt
            prompt = get_aggregation_prompt({
                "first_name": talent_data.first_name,
                "last_name": talent_data.last_name,
                "headline": getattr(talent_data, 'headline', ''),
                "summary": getattr(talent_data, 'summary', ''),
                "skills": ', '.join(getattr(talent_data, 'skills', [])),
                "education_analysis": education_analysis.get('summary', '교육 정보 없음'),
                "position_analysis": position_analysis.get('summary', '경력 정보 없음'),
                "companies_count": len(vector_context.get('companies', [])),
                "news_count": len(vector_context.get('news', []))
            })
            
            # Generate experience tags using LLM
            experience_tags = await self._generate_experience_tags(llm_model, prompt)
            
            # Create final analysis result
            processing_time = time.time() - processing_start_time
            
            self.logger.logger.debug(f"Creating final analysis result with {len(experience_tags)} tags...")
            analysis_result = AnalysisResult(
                talent_id=f"{talent_data.first_name}_{talent_data.last_name}",
                experience_tags=experience_tags,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={
                    "education_analysis": education_analysis,
                    "position_analysis": position_analysis,
                    "vector_search_results": vector_context,
                    "search_queries_used": vector_context.get("search_queries", [])
                }
            )
            
            # Log final data transformation
            self.logger.log_data_transformation(
                "final_aggregation",
                {
                    "education_results": len(education_analysis.get("results", [])),
                    "position_results": len(position_analysis.get("results", [])),
                    "vector_items": context_items
                },
                {
                    "experience_tags_generated": len(experience_tags),
                    "processing_time_seconds": processing_time
                }
            )
            
            node_duration = time.time() - node_start_time
            self.logger.log_node_complete("aggregate_results", node_duration, {
                "tags_generated": len(experience_tags),
                "total_processing_time": processing_time
            })
            
            return {"analysis_result": analysis_result}
            
        except Exception as e:
            error_context = {
                "talent_id": talent_id,
                "error_type": type(e).__name__,
                "node_duration": time.time() - node_start_time
            }
            self.logger.log_node_error("aggregate_results", e, error_context)
            
            # Create fallback result
            processing_time = time.time() - processing_start_time
            fallback_tags = [
                ExperienceTag(
                    tag="분석 실패",
                    confidence=0.1,
                    reasoning=f"집계 중 오류 발생: {str(e)}"
                )
            ]
            
            analysis_result = AnalysisResult(
                talent_id=f"{talent_data.first_name}_{talent_data.last_name}",
                experience_tags=fallback_tags,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={
                    "error": str(e),
                    "education_analysis": education_analysis,
                    "position_analysis": position_analysis
                }
            )
            
            return {"analysis_result": analysis_result}
    
    async def _generate_experience_tags(self, llm_model, prompt: str) -> List[ExperienceTag]:
        """Generate experience tags using LLM."""
        try:
            self.logger.logger.debug("Invoking LLM for final experience tag generation...")
            llm_start_time = time.time()
            response = await llm_model.ainvoke([HumanMessage(content=prompt)])
            llm_time = time.time() - llm_start_time
            
            # Log LLM call
            self.logger.log_llm_call(
                "aggregate_results",
                getattr(llm_model, 'model_name', 'unknown'),
                len(prompt),
                len(response.content),
                llm_time
            )
            
            self.logger.logger.debug("Parsing experience tags from LLM response...")
            experience_tags = self._parse_experience_tags(response.content)
            
            return experience_tags
            
        except Exception as e:
            self.logger.logger.error(f"❌ LLM failed during aggregation: {str(e)}")
            # Return fallback tags
            return [
                ExperienceTag(
                    tag="분석 실패",
                    confidence=0.1,
                    reasoning=f"LLM 분석 중 오류 발생: {str(e)}"
                )
            ]
    
    def _parse_experience_tags(self, llm_response: str) -> List[ExperienceTag]:
        """Parse LLM response into ExperienceTag objects."""
        try:
            # Try to extract JSON from the response
            start_idx = llm_response.find('[')
            end_idx = llm_response.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = llm_response[start_idx:end_idx]
                tags_data = json.loads(json_str)
                
                return [
                    ExperienceTag(
                        tag=tag["tag"],
                        confidence=tag["confidence"],
                        reasoning=tag["reasoning"]
                    )
                    for tag in tags_data
                ]
        except (json.JSONDecodeError, KeyError):
            # Fallback parsing or default tags
            pass
        
        # Return default tags if parsing fails
        return [
            ExperienceTag(
                tag="분석 완료",
                confidence=0.7,
                reasoning="LLM 응답 파싱 실패, 수동 검토 필요"
            )
        ]


def create_aggregation_node(logger: WorkflowLogger) -> AggregationNode:
    return AggregationNode(logger) 
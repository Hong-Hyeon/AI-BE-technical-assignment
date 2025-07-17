"""
Position Analysis Node for talent analysis workflow.
"""

import time
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
from config.logging_config import WorkflowLogger
from core.exceptions import WorkflowException
from ..base.state import TalentAnalysisState


class PositionAnalysisNode:
    """Node responsible for analyzing work positions with company context."""
    
    def __init__(self, logger: WorkflowLogger):
        self.logger = logger
    
    async def execute(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """Analyze work positions with company context and vector search results."""
        node_start_time = time.time()
        talent_data = state["talent_data"]
        preprocessed_context = state.get("preprocessed_context", {})
        llm_model = state["llm_model"]
        talent_id = state["talent_id"]
        
        position_count = len(talent_data.positions)
        companies = [pos.company_name for pos in talent_data.positions]
        
        self.logger.log_node_start("analyze_positions", {
            "talent_id": talent_id,
            "positions_to_analyze": position_count,
            "companies": companies
        })
        
        position_analysis = []
        successful_analyses = 0
        
        try:
            from factories.prompt_factory import get_position_prompt, talent_analysis_prompt_factory
            
            for i, position in enumerate(talent_data.positions):
                self.logger.logger.debug(
                    f"Analyzing position {i + 1}/{position_count}: {position.company_name} - {position.title}"
                )
                
                # Get context data
                company_data = preprocessed_context.get("companies", {}).get(position.company_name, {})
                news_data = preprocessed_context.get("news", {}).get(position.company_name, {})
                
                # Prepare context templates
                company_context = talent_analysis_prompt_factory.get_company_context_template(company_data)
                news_context = talent_analysis_prompt_factory.get_news_context_template(news_data)
                
                # Generate prompt
                prompt = get_position_prompt({
                    "title": position.title,
                    "company_name": position.company_name,
                    "description": position.description,
                    "start_end_date": position.start_end_date,
                    "company_location": position.company_location,
                    "company_context": company_context,
                    "news_context": news_context
                })
                
                # Analyze this position
                analysis_result = await self._analyze_single_position(
                    llm_model, position, prompt, i + 1, position_count
                )
                
                position_analysis.append(analysis_result)
                
                if analysis_result.get("success", False):
                    successful_analyses += 1
            
            # Create summary
            summary = self._create_position_summary(position_analysis)
            
            # Log data transformation
            self.logger.log_data_transformation(
                "position_analysis",
                {"input_positions": position_count},
                {
                    "successful_analyses": successful_analyses,
                    "failed_analyses": position_count - successful_analyses,
                    "success_rate": successful_analyses / position_count if position_count > 0 else 0
                }
            )
            
            result = {
                "position_analysis": {
                    "results": position_analysis,
                    "summary": summary,
                    "metadata": {
                        "total_positions": position_count,
                        "successful_analyses": successful_analyses,
                        "companies_analyzed": [r["company"] for r in position_analysis if r.get("success")]
                    }
                }
            }
            
            node_duration = time.time() - node_start_time
            self.logger.log_node_complete("analyze_positions", node_duration, {
                "positions_analyzed": position_count,
                "successful": successful_analyses,
                "success_rate": f"{successful_analyses}/{position_count}"
            })
            
            return result
            
        except Exception as e:
            error_context = {
                "talent_id": talent_id,
                "position_count": position_count,
                "error_type": type(e).__name__,
                "node_duration": time.time() - node_start_time
            }
            self.logger.log_node_error("analyze_positions", e, error_context)
            
            return {
                "position_analysis": {
                    "results": position_analysis,
                    "summary": self._create_position_summary(position_analysis),
                    "metadata": {
                        "total_positions": position_count,
                        "successful_analyses": successful_analyses,
                        "error": str(e)
                    }
                }
            }
    
    async def _analyze_single_position(self, llm_model, position, prompt: str, current: int, total: int) -> Dict[str, Any]:
        """Analyze a single position record."""
        try:
            llm_start_time = time.time()
            response = await llm_model.ainvoke([HumanMessage(content=prompt)])
            llm_time = time.time() - llm_start_time
            
            self.logger.log_llm_call(
                "analyze_positions",
                getattr(llm_model, 'model_name', 'unknown'),
                len(prompt),
                len(response.content),
                llm_time
            )
            
            self.logger.logger.debug(f"✅ Completed analysis for {position.company_name}")
            
            return {
                "company": position.company_name,
                "position": position.title,
                "analysis": response.content,
                "success": True,
                "llm_time": llm_time
            }
            
        except Exception as e:
            self.logger.logger.error(f"❌ Failed to analyze {position.company_name}: {str(e)}")
            return {
                "company": position.company_name,
                "position": position.title,
                "analysis": f"분석 실패: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def _create_position_summary(self, position_analysis: List[Dict]) -> str:
        """Create a summary of position analysis results."""
        if not position_analysis:
            return "경력 정보 없음"
        
        successful_results = [result for result in position_analysis if result.get("success", False)]
        
        if not successful_results:
            return "경력 분석 실패"
        
        summaries = []
        for result in successful_results:
            company = result["company"]
            position = result["position"]
            analysis = result["analysis"]
            summaries.append(f"- {company} ({position}): {analysis}")
        
        return "\n".join(summaries)


def create_position_analysis_node(logger: WorkflowLogger) -> PositionAnalysisNode:
    return PositionAnalysisNode(logger) 
"""
LangGraph workflow for talent analysis with integrated preprocessing and vector search.
"""
import asyncio
from typing import Dict, Any, List, TypedDict, Annotated
from datetime import datetime
import time
import json

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from models.talent import TalentData, AnalysisResult, ExperienceTag
from db.session import SessionLocal
from models.company import Company, CompanyNews
from factories.vector_search_factory import VectorSearchManager
from factories.prompt_factory import (
    talent_analysis_prompt_factory,
    get_education_prompt,
    get_position_prompt,
    get_aggregation_prompt,
    PromptCategory
)
from config.logging_config import WorkflowLogger


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries for LangGraph state."""
    if not left:
        return right
    if not right:
        return left
    return {**left, **right}


class TalentAnalysisState(TypedDict):
    """State for talent analysis workflow."""
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


class TalentAnalysisWorkflow:
    """LangGraph workflow for talent analysis."""
    
    def __init__(self):
        self.logger = WorkflowLogger("TalentAnalysis")
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with integrated preprocessing and vector search."""
        workflow = StateGraph(TalentAnalysisState)
        
        # Add nodes
        workflow.add_node("load_talent_data", self._load_talent_data)
        workflow.add_node("preprocess_context", self._preprocess_context)
        workflow.add_node("vector_search", self._vector_search)
        workflow.add_node("analyze_education", self._analyze_education)
        workflow.add_node("analyze_positions", self._analyze_positions)
        workflow.add_node("aggregate_results", self._aggregate_results)
        
        # Sequential preprocessing flow
        workflow.set_entry_point("load_talent_data")
        workflow.add_edge("load_talent_data", "preprocess_context")
        workflow.add_edge("preprocess_context", "vector_search")
        
        # Parallel analysis after preprocessing
        workflow.add_edge("vector_search", "analyze_education")
        workflow.add_edge("vector_search", "analyze_positions")
        
        # Final aggregation
        workflow.add_edge("analyze_education", "aggregate_results")
        workflow.add_edge("analyze_positions", "aggregate_results")
        workflow.add_edge("aggregate_results", END)
        
        return workflow.compile()
    
    async def _load_talent_data(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """Load talent data from data source."""
        node_start_time = time.time()
        self.logger.log_node_start("load_talent_data", {"talent_id": state["talent_id"]})
        
        from factories.data_source_factory import DefaultDataSourceFactory, DataSourceManager
        
        try:
            # Initialize data source manager
            self.logger.logger.debug("Initializing data source manager...")
            data_factory = DefaultDataSourceFactory("./example_datas")
            data_manager = DataSourceManager(data_factory)
            
            # Load talent data
            self.logger.logger.debug(f"Loading talent data for ID: {state['talent_id']}")
            talent_data = data_manager.get_talent_data(state["talent_id"])
            
            # Log data transformation info
            self.logger.log_data_transformation(
                "talent_data_loading",
                {"talent_id": state["talent_id"]},
                {
                    "education_count": len(talent_data.educations),
                    "position_count": len(talent_data.positions),
                    "skills_count": len(talent_data.skills)
                }
            )
            
            node_duration = time.time() - node_start_time
            self.logger.log_node_complete(
                "load_talent_data", 
                node_duration,
                {"data_loaded": f"{len(talent_data.educations)} educations, {len(talent_data.positions)} positions"}
            )
            
            return {"talent_data": talent_data}
            
        except Exception as e:
            self.logger.log_node_error("load_talent_data", e, {"talent_id": state["talent_id"]})
            raise ValueError(f"Failed to load talent data: {str(e)}")
    
    async def _preprocess_context(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """Preprocess context data from database."""
        node_start_time = time.time()
        talent_data = state["talent_data"]
        
        self.logger.log_node_start(
            "preprocess_context", 
            {"talent_id": state["talent_id"], "positions_to_process": len(talent_data.positions)}
        )
        
        from factories.data_source_factory import DefaultDataSourceFactory, DataSourceManager
        
        company_data = {}
        news_data = {}
        
        try:
            # Initialize data source manager
            self.logger.logger.debug("Initializing data source manager for context preprocessing...")
            data_factory = DefaultDataSourceFactory("./example_datas")
            data_manager = DataSourceManager(data_factory)
            
            # Collect company and news data for each position
            for i, position in enumerate(talent_data.positions):
                self.logger.logger.debug(f"Processing position {i + 1}/{len(talent_data.positions)}: {position.company_name}")
                
                try:
                    company_info = data_manager.get_company_data(position.company_name)
                    company_data[position.company_name] = company_info
                    self.logger.logger.debug(f"✅ Found company data for {position.company_name}")
                    
                    # Get news data
                    try:
                        news_info = data_manager.get_news_data(position.company_name)
                        news_data[position.company_name] = news_info
                        self.logger.logger.debug(f"✅ Found news data for {position.company_name}")
                    except ValueError:
                        self.logger.logger.debug(f"⚠️ No news data available for {position.company_name}")
                        pass  # No news data available
                except Exception as e:
                    self.logger.logger.debug(f"⚠️ No company data available for {position.company_name}: {str(e)}")
                    pass  # No company data available
            
            preprocessed_context = {
                "companies": company_data,
                "news": news_data
            }
            
            # Log data transformation info
            self.logger.log_data_transformation(
                "context_preprocessing",
                {"positions_processed": len(talent_data.positions)},
                {
                    "companies_found": len(company_data),
                    "news_datasets_found": len(news_data),
                    "company_names": list(company_data.keys())
                }
            )
            
            node_duration = time.time() - node_start_time
            self.logger.log_node_complete(
                "preprocess_context", 
                node_duration,
                {"companies_loaded": len(company_data), "news_datasets": len(news_data)}
            )
            
            return {"preprocessed_context": preprocessed_context}
            
        except Exception as e:
            self.logger.log_node_error("preprocess_context", e, {"talent_id": state["talent_id"]})
            self.logger.logger.warning(f"Context preprocessing warning: {e}")
            return {"preprocessed_context": {"companies": {}, "news": {}}}
    
    async def _vector_search(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """Perform vector search for semantically relevant context."""
        node_start_time = time.time()
        talent_data = state["talent_data"]
        
        self.logger.log_node_start(
            "vector_search", 
            {
                "talent_id": state["talent_id"],
                "educations": len(talent_data.educations),
                "positions": len(talent_data.positions),
                "skills": len(talent_data.skills)
            }
        )
        
        try:
            # Initialize vector search manager
            search_start_time = time.time()
            self.logger.logger.debug("Initializing vector search manager...")
            vector_search_manager = VectorSearchManager()
            
            # Convert talent data to dict for vector search
            self.logger.logger.debug("Converting talent data for vector search...")
            talent_dict = {
                "educations": [edu.dict() for edu in talent_data.educations],
                "positions": [pos.dict() for pos in talent_data.positions],
                "skills": talent_data.skills
            }
            
            # Log input data transformation
            self.logger.log_data_transformation(
                "vector_search_input",
                {"raw_talent_data": f"{len(talent_data.educations)} educations, {len(talent_data.positions)} positions"},
                {"search_ready_data": "Converted to dict format for vector search"}
            )
            
            # Perform vector search
            self.logger.logger.debug("Performing vector search for talent context...")
            vector_results = vector_search_manager.search_talent_context(talent_dict)
            search_time = time.time() - search_start_time
            
            # Log vector search results
            results_count = len(vector_results.get("companies", [])) + len(vector_results.get("news", []))
            self.logger.log_vector_search(
                {
                    "search_queries": vector_results.get("search_queries", []),
                    "input_data_types": list(talent_dict.keys())
                },
                results_count,
                search_time
            )
            
            # Close resources
            vector_search_manager.close()
            
            node_duration = time.time() - node_start_time
            self.logger.log_node_complete(
                "vector_search", 
                node_duration,
                {
                    "companies_found": len(vector_results.get("companies", [])),
                    "news_found": len(vector_results.get("news", [])),
                    "queries_executed": len(vector_results.get("search_queries", []))
                }
            )
            
            return {"vector_search_results": vector_results}
            
        except Exception as e:
            self.logger.log_node_error("vector_search", e, {"talent_id": state["talent_id"]})
            self.logger.logger.warning(f"Vector search warning: {e}")
            return {"vector_search_results": {"companies": [], "news": [], "search_queries": []}}
    
    async def _analyze_education(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """Analyze education background in parallel."""
        node_start_time = time.time()
        talent_data = state["talent_data"]
        llm_model = state["llm_model"]
        
        self.logger.log_node_start(
            "analyze_education", 
            {
                "talent_id": state["talent_id"],
                "educations_to_analyze": len(talent_data.educations),
                "schools": [edu.school_name for edu in talent_data.educations]
            }
        )
        
        education_analysis = []
        
        for i, education in enumerate(talent_data.educations):
            self.logger.logger.debug(f"Analyzing education {i + 1}/{len(talent_data.educations)}: {education.school_name}")
            
            prompt = get_education_prompt({
                "school_name": education.school_name,
                "degree_name": education.degree_name,
                "field_of_study": education.field_of_study,
                "start_end_date": education.start_end_date
            })
            
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
                
                education_analysis.append({
                    "school": education.school_name,
                    "analysis": response.content
                })
                
                self.logger.logger.debug(f"✅ Completed analysis for {education.school_name}")
                
            except Exception as e:
                self.logger.logger.error(f"❌ Failed to analyze {education.school_name}: {str(e)}")
                education_analysis.append({
                    "school": education.school_name,
                    "analysis": f"분석 실패: {str(e)}"
                })
        
        # Log data transformation
        successful_analyses = len([a for a in education_analysis if "분석 실패" not in a["analysis"]])
        self.logger.log_data_transformation(
            "education_analysis",
            {"input_educations": len(talent_data.educations)},
            {"successful_analyses": successful_analyses, "failed_analyses": len(education_analysis) - successful_analyses}
        )
        
        # Return only the specific key this node is responsible for
        result = {
            "education_analysis": {
                "results": education_analysis,
                "summary": self._summarize_education_analysis(education_analysis)
            }
        }
        
        node_duration = time.time() - node_start_time
        self.logger.log_node_complete(
            "analyze_education", 
            node_duration,
            {"educations_analyzed": len(education_analysis), "successful": successful_analyses}
        )
        
        return result
    
    async def _analyze_positions(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """Analyze work positions with company context and vector search results."""
        node_start_time = time.time()
        talent_data = state["talent_data"]
        preprocessed_context = state.get("preprocessed_context", {})
        llm_model = state["llm_model"]
        
        self.logger.log_node_start(
            "analyze_positions", 
            {
                "talent_id": state["talent_id"],
                "positions_to_analyze": len(talent_data.positions),
                "companies": [pos.company_name for pos in talent_data.positions]
            }
        )
        
        position_analysis = []
        
        for i, position in enumerate(talent_data.positions):
            self.logger.logger.debug(f"Analyzing position {i + 1}/{len(talent_data.positions)}: {position.company_name} - {position.title}")
            
            # Get company and news data from preprocessing
            company_data = preprocessed_context.get("companies", {}).get(position.company_name, {})
            news_data = preprocessed_context.get("news", {}).get(position.company_name, {})
            
            # Log context availability
            context_info = {
                "has_company_data": bool(company_data),
                "has_news_data": bool(news_data)
            }
            self.logger.logger.debug(f"Context for {position.company_name}: {context_info}")
            
            # Prepare context templates
            company_context = talent_analysis_prompt_factory.get_company_context_template(company_data)
            news_context = talent_analysis_prompt_factory.get_news_context_template(news_data)
            
            # Use the position analysis prompt from factory
            prompt = get_position_prompt({
                "title": position.title,
                "company_name": position.company_name,
                "description": position.description,
                "start_end_date": position.start_end_date,
                "company_location": position.company_location,
                "company_context": company_context,
                "news_context": news_context
            })
            
            try:
                llm_start_time = time.time()
                response = await llm_model.ainvoke([HumanMessage(content=prompt)])
                llm_time = time.time() - llm_start_time
                
                # Log LLM call
                self.logger.log_llm_call(
                    "analyze_positions",
                    getattr(llm_model, 'model_name', 'unknown'),
                    len(prompt),
                    len(response.content),
                    llm_time
                )
                
                position_analysis.append({
                    "company": position.company_name,
                    "position": position.title,
                    "analysis": response.content
                })
                
                self.logger.logger.debug(f"✅ Completed analysis for {position.company_name}")
                
            except Exception as e:
                self.logger.logger.error(f"❌ Failed to analyze {position.company_name}: {str(e)}")
                position_analysis.append({
                    "company": position.company_name,
                    "position": position.title,
                    "analysis": f"분석 실패: {str(e)}"
                })
        
        # Log data transformation
        successful_analyses = len([a for a in position_analysis if "분석 실패" not in a["analysis"]])
        self.logger.log_data_transformation(
            "position_analysis",
            {"input_positions": len(talent_data.positions)},
            {"successful_analyses": successful_analyses, "failed_analyses": len(position_analysis) - successful_analyses}
        )
        
        # Return only the specific key this node is responsible for
        result = {
            "position_analysis": {
                "results": position_analysis,
                "summary": self._summarize_position_analysis(position_analysis)
            }
        }
        
        node_duration = time.time() - node_start_time
        self.logger.log_node_complete(
            "analyze_positions", 
            node_duration,
            {"positions_analyzed": len(position_analysis), "successful": successful_analyses}
        )
        
        return result
    
    async def _aggregate_results(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """Aggregate education and position analyses into final summary."""
        node_start_time = time.time()
        talent_data = state["talent_data"]
        education_analysis = state.get("education_analysis", {})
        position_analysis = state.get("position_analysis", {})
        vector_context = state.get("vector_search_results", {})
        llm_model = state["llm_model"]
        processing_start_time = state["processing_start_time"]
        
        self.logger.log_node_start(
            "aggregate_results", 
            {
                "talent_id": state["talent_id"],
                "has_education_analysis": bool(education_analysis),
                "has_position_analysis": bool(position_analysis),
                "vector_context_items": len(vector_context.get("companies", [])) + len(vector_context.get("news", []))
            }
        )
        
        # Both analyses should be complete when this node executes due to LangGraph dependencies
        self.logger.logger.debug("Aggregating education and position analyses...")
        
        # Create comprehensive summary prompt with vector search context
        prompt = get_aggregation_prompt({
            "first_name": talent_data.first_name,
            "last_name": talent_data.last_name,
            "headline": talent_data.headline,
            "summary": talent_data.summary,
            "skills": ', '.join(talent_data.skills),
            "education_analysis": education_analysis.get('summary', '교육 정보 없음'),
            "position_analysis": position_analysis.get('summary', '경력 정보 없음'),
            "companies_count": len(vector_context.get('companies', [])),
            "news_count": len(vector_context.get('news', []))
        })
        
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
            
        except Exception as e:
            self.logger.logger.error(f"❌ LLM failed during aggregation: {str(e)}")
            # Fallback tags if LLM fails
            experience_tags = [
                ExperienceTag(
                    tag="분석 실패",
                    confidence=0.1,
                    reasoning=f"LLM 분석 중 오류 발생: {str(e)}"
                )
            ]
        
        # Create final analysis result
        processing_time = time.time() - processing_start_time
        
        self.logger.logger.debug(f"Creating final analysis result with {len(experience_tags)} tags...")
        analysis_result = AnalysisResult(
            talent_id=f"{talent_data.first_name}_{talent_data.last_name}",
            experience_tags=experience_tags,
            processing_time=processing_time,
            timestamp=datetime.now(),
            # metadata={
            #     "education_analysis": education_analysis,
            #     "position_analysis": position_analysis,
            #     "vector_search_results": vector_context,
            #     "search_queries_used": vector_context.get("search_queries", [])
            # }
        )
        
        # Log final data transformation
        self.logger.log_data_transformation(
            "final_aggregation",
            {
                "education_results": len(education_analysis.get("results", [])),
                "position_results": len(position_analysis.get("results", [])),
                "vector_items": len(vector_context.get("companies", [])) + len(vector_context.get("news", []))
            },
            {
                "experience_tags_generated": len(experience_tags),
                "processing_time_seconds": processing_time
            }
        )
        
        node_duration = time.time() - node_start_time
        self.logger.log_node_complete(
            "aggregate_results", 
            node_duration,
            {
                "tags_generated": len(experience_tags),
                "total_processing_time": processing_time
            }
        )
        
        # Return only the analysis result
        return {"analysis_result": analysis_result}
    
    def _summarize_education_analysis(self, education_analysis: List[Dict]) -> str:
        """Summarize education analysis results."""
        if not education_analysis:
            return "교육 정보 없음"
        
        summaries = []
        for edu in education_analysis:
            summaries.append(f"- {edu['school']}: {edu['analysis']}")
        
        return "\n".join(summaries)
    
    def _summarize_position_analysis(self, position_analysis: List[Dict]) -> str:
        """Summarize position analysis results."""
        if not position_analysis:
            return "경력 정보 없음"
        
        summaries = []
        for pos in position_analysis:
            summaries.append(f"- {pos['company']} ({pos['position']}): {pos['analysis']}")
        
        return "\n".join(summaries)
    
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
            # Fallback: parse text manually or return default tags
            pass
        
        # Fallback parsing or default tags
        return [
            ExperienceTag(
                tag="분석 완료",
                confidence=0.7,
                reasoning="LLM 응답 파싱 실패, 수동 검토 필요"
            )
        ]
    
    async def run_analysis(
        self,
        talent_id: str,
        llm_model
    ) -> AnalysisResult:
        """Run the complete talent analysis workflow with integrated preprocessing and vector search."""
        workflow_start_time = time.time()
        
        # Log workflow start
        self.logger.log_workflow_start(
            talent_id, 
            {
                "llm_model": getattr(llm_model, 'model_name', str(type(llm_model))),
                "workflow_nodes": ["load_talent_data", "preprocess_context", "vector_search", "analyze_education", "analyze_positions", "aggregate_results"]
            }
        )
        
        try:
            # Extract the LangChain ChatOpenAI model if it's wrapped
            if hasattr(llm_model, 'get_langchain_model'):
                langchain_model = llm_model.get_langchain_model()
                self.logger.logger.debug("Extracted LangChain model from wrapper")
            else:
                langchain_model = llm_model
                self.logger.logger.debug("Using model directly")
                
            initial_state = TalentAnalysisState(
                talent_id=talent_id,
                llm_model=langchain_model,
                talent_data=None,  # Will be filled by load_talent_data node
                preprocessed_context={},
                vector_search_results={},
                education_analysis={},
                position_analysis={},
                analysis_result=None,
                processing_start_time=time.time()
            )
            
            self.logger.logger.info("🔄 Starting LangGraph workflow execution...")
            
            # Run the workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            # Calculate total workflow time
            total_duration = time.time() - workflow_start_time
            analysis_result = final_state["analysis_result"]
            
            # Log workflow completion with summary
            result_summary = {
                "experience_tags_count": len(analysis_result.experience_tags),
                "top_tags": [tag.tag for tag in analysis_result.experience_tags[:3]],  # Top 3 tags
                "processing_time": analysis_result.processing_time,
                "workflow_total_time": total_duration
            }
            
            self.logger.log_workflow_complete(talent_id, total_duration, result_summary)
            
            return analysis_result
            
        except Exception as e:
            total_duration = time.time() - workflow_start_time
            self.logger.logger.error(f"💥 Workflow failed for talent_id {talent_id} after {total_duration:.2f}s: {str(e)}", exc_info=True)
            raise


# Create a singleton instance
talent_analysis_workflow = TalentAnalysisWorkflow() 
"""
LangGraph workflow for talent analysis with integrated preprocessing and vector search.
"""
import asyncio
from typing import Dict, Any, List, TypedDict, Annotated
from datetime import datetime
import time

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from models.talent import TalentData, AnalysisResult, ExperienceTag
from db.session import SessionLocal
from models.company import Company, CompanyNews
from factories.vector_search_factory import VectorSearchManager


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
        from factories.data_source_factory import DefaultDataSourceFactory, DataSourceManager
        
        try:
            # Initialize data source manager
            data_factory = DefaultDataSourceFactory("./example_datas")
            data_manager = DataSourceManager(data_factory)
            
            # Load talent data
            talent_data = data_manager.get_talent_data(state["talent_id"])
            
            return {"talent_data": talent_data}
            
        except Exception as e:
            raise ValueError(f"Failed to load talent data: {str(e)}")
    
    async def _preprocess_context(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """Preprocess context data from database."""
        from factories.data_source_factory import DefaultDataSourceFactory, DataSourceManager
        
        talent_data = state["talent_data"]
        company_data = {}
        news_data = {}
        
        try:
            # Initialize data source manager
            data_factory = DefaultDataSourceFactory("./example_datas")
            data_manager = DataSourceManager(data_factory)
            
            # Collect company and news data for each position
            for position in talent_data.positions:
                try:
                    company_info = data_manager.get_company_data(position.company_name)
                    company_data[position.company_name] = company_info
                    
                    # Get news data
                    try:
                        news_info = data_manager.get_news_data(position.company_name)
                        news_data[position.company_name] = news_info
                    except ValueError:
                        pass  # No news data available
                except Exception:
                    pass  # No company data available
            
            preprocessed_context = {
                "companies": company_data,
                "news": news_data
            }
            
            return {"preprocessed_context": preprocessed_context}
            
        except Exception as e:
            print(f"Context preprocessing warning: {e}")
            return {"preprocessed_context": {"companies": {}, "news": {}}}
    
    async def _vector_search(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """Perform vector search for semantically relevant context."""
        talent_data = state["talent_data"]
        
        try:
            # Initialize vector search manager
            vector_search_manager = VectorSearchManager()
            
            # Convert talent data to dict for vector search
            talent_dict = {
                "educations": [edu.dict() for edu in talent_data.educations],
                "positions": [pos.dict() for pos in talent_data.positions],
                "skills": talent_data.skills
            }
            
            # Perform vector search
            vector_results = vector_search_manager.search_talent_context(talent_dict)
            
            # Close resources
            vector_search_manager.close()
            
            return {"vector_search_results": vector_results}
            
        except Exception as e:
            print(f"Vector search warning: {e}")
            return {"vector_search_results": {"companies": [], "news": [], "search_queries": []}}
    
    async def _analyze_education(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """Analyze education background in parallel."""
        talent_data = state["talent_data"]
        llm_model = state["llm_model"]
        
        education_analysis = []
        
        for education in talent_data.educations:
            prompt = f"""
            다음 교육 정보를 분석하여 대학교의 등급을 분류해주세요:
            
            학교명: {education.school_name}
            학위: {education.degree_name}
            전공: {education.field_of_study}
            기간: {education.start_end_date}
            
            대학교를 다음 기준으로 분류해주세요:
            - 상위권: SKY(서울대, 연세대, 고려대), KAIST, POSTECH 등 최상위 대학
            - 중위권: 지방 국립대, 주요 사립대 (성균관대, 한양대, 중앙대, 경희대 등)
            - 하위권: 기타 대학
            
            응답 형식:
            {{
                "tier": "상위권/중위권/하위권",
                "confidence": 0.9,
                "reasoning": "분류 근거"
            }}
            """
            
            try:
                response = await llm_model.ainvoke([HumanMessage(content=prompt)])
                education_analysis.append({
                    "school": education.school_name,
                    "analysis": response.content
                })
            except Exception as e:
                education_analysis.append({
                    "school": education.school_name,
                    "analysis": f"분석 실패: {str(e)}"
                })
        
        # Return only the specific key this node is responsible for
        return {
            "education_analysis": {
                "results": education_analysis,
                "summary": self._summarize_education_analysis(education_analysis)
            }
        }
    
    async def _analyze_positions(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """Analyze work positions with company context and vector search results."""
        talent_data = state["talent_data"]
        preprocessed_context = state.get("preprocessed_context", {})
        llm_model = state["llm_model"]
        
        position_analysis = []
        
        for position in talent_data.positions:
            # Get company and news data from preprocessing
            company_data = preprocessed_context.get("companies", {}).get(position.company_name, {})
            news_data = preprocessed_context.get("news", {}).get(position.company_name, {})
            
            # Use the existing position analysis prompt (temporarily)
            prompt = self._build_position_analysis_prompt(position, company_data, news_data)
            
            try:
                response = await llm_model.ainvoke([HumanMessage(content=prompt)])
                position_analysis.append({
                    "company": position.company_name,
                    "position": position.title,
                    "analysis": response.content
                })
            except Exception as e:
                position_analysis.append({
                    "company": position.company_name,
                    "position": position.title,
                    "analysis": f"분석 실패: {str(e)}"
                })
        
        # Return only the specific key this node is responsible for
        return {
            "position_analysis": {
                "results": position_analysis,
                "summary": self._summarize_position_analysis(position_analysis)
            }
        }
    
    async def _aggregate_results(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """Aggregate education and position analyses into final summary."""
        talent_data = state["talent_data"]
        education_analysis = state.get("education_analysis", {})
        position_analysis = state.get("position_analysis", {})
        vector_context = state.get("vector_search_results", {})
        llm_model = state["llm_model"]
        processing_start_time = state["processing_start_time"]
        
        # Both analyses should be complete when this node executes due to LangGraph dependencies
        
        # Create comprehensive summary prompt with vector search context
        prompt = f"""
        다음 인재의 교육과 경력을 종합적으로 분석하여 핵심 태그와 요약을 생성해주세요:
        
        인재 정보:
        - 이름: {talent_data.first_name} {talent_data.last_name}
        - 헤드라인: {talent_data.headline}
        - 요약: {talent_data.summary}
        - 스킬: {', '.join(talent_data.skills)}
        
        교육 분석:
        {education_analysis.get('summary', '교육 정보 없음')}
        
        경력 분석:
        {position_analysis.get('summary', '경력 정보 없음')}
        
        벡터 검색으로 발견된 관련 컨텍스트:
        - 검색된 관련 회사 수: {len(vector_context.get('companies', []))}
        - 검색된 관련 뉴스 수: {len(vector_context.get('news', []))}
        
        다음 형식으로 핵심 경험 태그들을 생성해주세요:
        
        예시 태그들:
        - 상위권대학교 (구체적 학교명)
        - 성장기스타트업 경험 (회사명과 성장 지표)
        - 리더십 (구체적 직책/역할)
        - 대용량데이터처리경험 (관련 회사/프로젝트)
        - IPO (관련 회사와 시기)
        - M&A 경험 (관련 거래)
        - 신규 투자 유치 경험 (회사명과 역할)
        
        각 태그에 대해 다음 JSON 형식으로 응답해주세요:
        [
            {{
                "tag": "태그명",
                "confidence": 0.9,
                "reasoning": "이 태그를 부여한 구체적인 근거"
            }}
        ]
        """
        
        try:
            response = await llm_model.ainvoke([HumanMessage(content=prompt)])
            experience_tags = self._parse_experience_tags(response.content)
        except Exception as e:
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
        
        # Return only the analysis result
        return {"analysis_result": analysis_result}
    
    def _build_position_analysis_prompt(self, position, company_data, news_data) -> str:
        """Build detailed prompt for position analysis."""
        prompt = f"""
        다음 경력을 회사의 성장 단계와 시장 상황을 고려하여 분석해주세요:
        
        포지션 정보:
        - 직책: {position.title}
        - 회사: {position.company_name}
        - 설명: {position.description}
        - 기간: {position.start_end_date}
        - 위치: {position.company_location}
        """
        
        # Add company context if available
        if company_data:
            prompt += f"""
        
        회사 정보:
        - 회사명: {getattr(company_data, 'name', 'N/A')}
        - 산업: {getattr(company_data, 'industry', 'N/A')}
        - 설립년도: {getattr(company_data, 'founded_year', 'N/A')}
        - 직원수: {getattr(company_data, 'employee_count', 'N/A')}
        - 펀딩 규모: {getattr(company_data, 'funding_amount', 'N/A')}
        """
        
        # Add news context if available
        if news_data and news_data.get('news_data'):
            prompt += f"""
        
        관련 뉴스 (최근 {news_data.get('news_count', 0)}건):
        """
            for news in news_data['news_data'][:5]:  # Limit to recent 5 news
                prompt += f"- {getattr(news, 'title', '')} ({getattr(news, 'date', '')})\n"
        
        prompt += """
        
        다음 관점에서 분석해주세요:
        1. 회사의 성장 단계 (스타트업/성장기/성숙기)
        2. 시장에서의 위치와 경쟁력
        3. 해당 시기의 회사 상황 (투자, IPO, M&A 등)
        4. 직책의 중요도와 리더십 역할
        5. 기술적/비즈니스적 임팩트
        
        응답 형식:
        {{
            "company_stage": "성장 단계",
            "leadership_role": "리더십 여부와 수준",
            "market_timing": "시장 타이밍 분석",
            "key_achievements": "주요 성과 추정",
            "confidence": 0.8
        }}
        """
        
        return prompt
    
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
            import json
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
        # Extract the LangChain ChatOpenAI model if it's wrapped
        if hasattr(llm_model, 'get_langchain_model'):
            langchain_model = llm_model.get_langchain_model()
        else:
            langchain_model = llm_model
            
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
        
        # Run the workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        return final_state["analysis_result"]


# Create a singleton instance
talent_analysis_workflow = TalentAnalysisWorkflow() 
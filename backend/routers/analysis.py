"""
Analysis router for talent analysis endpoints.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from .base import BaseRouter
from models.talent import AnalysisResult


class AnalyzeRequest(BaseModel):
    """Request model for talent analysis."""
    talent_id: str
    analyzer_type: str = "default"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"


class AnalyzeResponse(BaseModel):
    """Response model for talent analysis."""
    result: AnalysisResult
    success: bool
    message: str


class VectorSearchRequest(BaseModel):
    """Request model for vector search."""
    query: str
    documents: List[Dict[str, Any]]


class AnalysisRouter(BaseRouter):
    """Router for talent analysis endpoints."""
    
    def get_router(self) -> APIRouter:
        """Get analysis router."""
        router = APIRouter(prefix="/analyze", tags=["analysis"])
        
        @router.get("/{talent_id}")
        async def analyze_talent_get(
            talent_id: str,
            analyzer_type: str = "default",
            llm_provider: str = "openai",
            llm_model: str = "gpt-4o-mini",
            enrich_context: bool = True
        ):
            """
            GET 엔드포인트로 인재 데이터를 분석합니다.
            
            Factory 패턴을 활용한 확장 가능한 구조:
            - DataSourceFactory: 다양한 데이터 소스 지원
            - LLMFactory: 다양한 LLM 모델 지원
            - ProcessorFactory: 벡터 검색 및 컨텍스트 강화
            - ExperienceAnalyzerFactory: 다양한 분석 로직 지원
            """
            self.check_managers()
            
            try:
                # 1. 인재 데이터 로드 (DataSourceFactory 활용)
                talent_data = self.data_source_manager.get_talent_data(talent_id)
                
                # 2. 컨텍스트 데이터 수집 (회사 정보, 뉴스 등)
                company_data = {}
                news_data = {}
                
                for position in talent_data.positions:
                    try:
                        company_info = self.data_source_manager.get_company_data(position.company_name)
                        company_data[position.company_name] = company_info
                        
                        # 뉴스 데이터도 수집
                        try:
                            news_info = self.data_source_manager.get_news_data(position.company_name)
                            news_data[position.company_name] = news_info
                        except ValueError:
                            pass
                    except ValueError:
                        # 회사 데이터가 없는 경우 무시
                        pass
                
                # 3. 컨텍스트 강화 (ProcessorFactory 활용)
                if enrich_context:
                    enriched_data = self.processor_manager.process_data("context_enrichment", {
                        "talent_data": talent_data.dict(),
                        "company_data": company_data,
                        "news_data": news_data
                    })
                    context = enriched_data.get("context", {})
                else:
                    context = {
                        "companies": company_data,
                        "news": news_data
                    }
                
                # 4. LLM 모델 생성 (LLMFactory 활용)
                llm_model_instance = self.llm_factory_manager.create_llm(llm_provider, llm_model)
                
                # 5. 분석기 생성 및 실행 (ExperienceAnalyzerFactory 활용)
                from factories.experience_analyzer_factory import DefaultExperienceAnalyzerFactory
                analyzer_factory = DefaultExperienceAnalyzerFactory(llm_model_instance)
                from factories.experience_analyzer_factory import ExperienceAnalyzerManager
                analyzer_manager = ExperienceAnalyzerManager(analyzer_factory)
                
                result = analyzer_manager.analyze_talent(talent_data, context, analyzer_type)
                
                return AnalyzeResponse(
                    result=result,
                    success=True,
                    message=f"Successfully analyzed talent {talent_id}"
                )
                
            except ValueError as e:
                raise HTTPException(status_code=404, detail=f"Data not found: {str(e)}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        @router.post("/", response_model=AnalyzeResponse)
        async def analyze_talent_post(request: AnalyzeRequest):
            """
            POST 엔드포인트로 인재 데이터를 분석합니다.
            
            더 세밀한 제어가 필요한 경우 사용합니다.
            """
            return await analyze_talent_get(
                talent_id=request.talent_id,
                analyzer_type=request.analyzer_type,
                llm_provider=request.llm_provider,
                llm_model=request.llm_model
            )
        
        return router 
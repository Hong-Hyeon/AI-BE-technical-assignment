"""
Analysis router for talent analysis endpoints.
"""
from fastapi import APIRouter, HTTPException
from .base import BaseRouter
from schema.analysis import AnalyzeRequest, AnalyzeResponse


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
                # LangGraph 워크플로우를 통한 통합 분석 실행
                # (전처리, 벡터 검색, 교육/경력 분석이 모두 포함됨)
                from workflows.talent_analysis import talent_analysis_workflow
                
                # 4. LLM 모델 생성 (LLMFactory 활용)
                llm_model_instance = self.llm_factory_manager.create_llm(llm_provider, llm_model)
                
                # 5. 통합 워크플로우 실행 (전처리 + 벡터 검색 + 분석)
                result = await talent_analysis_workflow.run_analysis(
                    talent_id=talent_id,
                    llm_model=llm_model_instance
                )
                
                return AnalyzeResponse(
                    result=result,
                    success=True,
                    message=f"Successfully analyzed talent {talent_id} with vector search enhancement"
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
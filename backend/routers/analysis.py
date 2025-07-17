"""
Analysis router for talent analysis endpoints.
"""
import time
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from .base import BaseRouter
from schema.analysis import (
    AnalyzeRequest, AnalyzeResponse, LLMModelEnum
)
from schema.base import ProviderEnum
from config.logging_config import get_api_logger


class AnalysisRouter(BaseRouter):
    """Router for talent analysis endpoints."""
    
    def __init__(self):
        super().__init__()
        self.logger = get_api_logger()
    
    def get_router(self) -> APIRouter:
        """Get analysis router."""
        router = APIRouter(prefix="/analyze", tags=["analysis"])
        
        @router.get("/{talent_id}")
        async def analyze_talent_get(
            talent_id: str,
            request: Request,
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
            api_start_time = time.time()
            client_ip = request.client.host if request.client else "unknown"
            
            # Log API request
            self.logger.info(f"🔄 Analysis API called - talent_id: {talent_id} | "
                             f"llm: {llm_provider}/{llm_model} | "
                             f"analyzer: {analyzer_type} | "
                             f"client_ip: {client_ip}")
            
            self.check_managers()
            
            try:
                # LangGraph 워크플로우를 통한 통합 분석 실행
                # (전처리, 벡터 검색, 교육/경력 분석이 모두 포함됨)
                self.logger.debug(f"Importing talent analysis workflow for {talent_id}")
                from workflows.talent_analysis import talent_analysis_workflow
                
                # 4. LLM 모델 생성 (LLMFactory 활용)
                self.logger.debug(f"Creating LLM model: {llm_provider}/{llm_model}")
                llm_model_instance = self.llm_factory_manager.create_llm(llm_provider, llm_model)
                
                # 5. 통합 워크플로우 실행 (전처리 + 벡터 검색 + 분석)
                self.logger.info(f"🚀 Starting workflow execution for talent_id: {talent_id}")
                workflow_start_time = time.time()
                
                result = await talent_analysis_workflow.run_analysis(
                    talent_id=talent_id,
                    llm_model=llm_model_instance
                )
                
                workflow_duration = time.time() - workflow_start_time
                api_duration = time.time() - api_start_time
                
                # Log successful completion
                self.logger.info(f"✅ Analysis completed for {talent_id} | "
                                 f"workflow_time: {workflow_duration:.2f}s | "
                                 f"total_api_time: {api_duration:.2f}s | "
                                 f"tags_generated: {len(result.experience_tags)}")
                
                # Create simplified response without metadata
                analysis_id = str(uuid.uuid4())
                
                response = AnalyzeResponse(
                    analysis_id=analysis_id,
                    result=result,
                    success=True,
                    message=f"Successfully analyzed talent {talent_id} with vector search enhancement"
                )
                
                # Log response summary
                self.logger.debug(f"Response for {talent_id}: success={response.success}, "
                                  f"tags_count={len(result.experience_tags)}")
                
                return response
                
            except ValueError as e:
                api_duration = time.time() - api_start_time
                error_msg = f"Data not found: {str(e)}"
                self.logger.warning(f"⚠️ Data not found for talent_id: {talent_id} | "
                                    f"duration: {api_duration:.2f}s | error: {error_msg}")
                raise HTTPException(status_code=404, detail=error_msg)
                
            except Exception as e:
                api_duration = time.time() - api_start_time
                error_msg = f"Analysis failed: {str(e)}"
                self.logger.error(f"❌ Analysis failed for talent_id: {talent_id} | "
                                  f"duration: {api_duration:.2f}s | "
                                  f"error: {error_msg}", exc_info=True)
                raise HTTPException(status_code=500, detail=error_msg)
        
        @router.post("/", response_model=AnalyzeResponse)
        async def analyze_talent_post(request_data: AnalyzeRequest, request: Request):
            """
            POST 엔드포인트로 인재 데이터를 분석합니다.
            
            더 세밀한 제어가 필요한 경우 사용합니다.
            """
            client_ip = request.client.host if request.client else "unknown"
            
            # Log POST request
            self.logger.info(f"🔄 Analysis POST API called - talent_id: {request_data.talent_id} | "
                             f"client_ip: {client_ip}")
            
            return await analyze_talent_get(
                talent_id=request_data.talent_id,
                request=request,
                analyzer_type="default",  # Use default since AnalyzeRequest doesn't have this field
                llm_provider=request_data.llm_config.provider.value,
                llm_model=request_data.llm_config.model.value
            )
        
        return router 
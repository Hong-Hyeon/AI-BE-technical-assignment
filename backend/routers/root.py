"""
Root router for basic system information.
"""
from fastapi import APIRouter
from .base import BaseRouter


class RootRouter(BaseRouter):
    """Router for root endpoints."""
    
    def get_router(self) -> APIRouter:
        """Get root router."""
        router = APIRouter(tags=["root"])
        
        @router.get("/")
        async def root():
            """Root endpoint with system information."""
            return {
                "message": "SearchRight AI API is running",
                "description": "Factory Pattern을 활용한 LLM 기반 인재 경험 분석 시스템",
                "version": "1.0.0",
                "architecture": {
                    "pattern": "Factory Pattern",
                    "components": [
                        "DataSourceFactory",
                        "LLMFactory", 
                        "ProcessorFactory",
                        "ExperienceAnalyzerFactory"
                    ]
                }
            }
        
        return router 
"""
Health and status router.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from .base import BaseRouter


class FactoryStatusResponse(BaseModel):
    """Response model for factory status."""
    llm_providers: Dict[str, Any]
    data_sources: Dict[str, Any]
    processors: Dict[str, Any]
    analyzers: Dict[str, Any]


class HealthRouter(BaseRouter):
    """Router for health and status endpoints."""
    
    def get_router(self) -> APIRouter:
        """Get health router."""
        router = APIRouter(prefix="/health", tags=["health"])
        
        @router.get("/")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "factories_initialized": all([
                    self.llm_factory_manager is not None,
                    self.data_source_manager is not None,
                    self.experience_analyzer_manager is not None,
                    self.processor_manager is not None
                ])
            }
        
        @router.get("/factories", response_model=FactoryStatusResponse)
        async def get_factory_status():
            """Factory 상태 정보를 반환합니다."""
            self.check_managers()
            
            return FactoryStatusResponse(
                llm_providers={
                    "available_providers": self.llm_factory_manager.get_supported_providers(),
                    "openai_models": self.llm_factory_manager.get_supported_models("openai")
                },
                data_sources={
                    "available_sources": self.data_source_manager.factory.get_supported_sources(),
                },
                processors={
                    "available_processors": self.processor_manager.factory.get_supported_processors(),
                },
                analyzers={
                    "available_analyzers": self.experience_analyzer_manager.factory.get_supported_analyzers(),
                }
            )
        
        return router 
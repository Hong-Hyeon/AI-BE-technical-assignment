"""
Vector search router for vector search endpoints.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from .base import BaseRouter


class VectorSearchRequest(BaseModel):
    """Request model for vector search."""
    query: str
    documents: List[Dict[str, Any]]


class VectorSearchResponse(BaseModel):
    """Response model for vector search."""
    success: bool
    query: str
    similar_documents: List[Dict[str, Any]]
    total_documents: int


class VectorSearchRouter(BaseRouter):
    """Router for vector search endpoints."""
    
    def get_router(self) -> APIRouter:
        """Get vector search router."""
        router = APIRouter(prefix="/vector-search", tags=["vector-search"])
        
        @router.post("/", response_model=VectorSearchResponse)
        async def vector_search(request: VectorSearchRequest):
            """
            벡터 검색을 수행합니다.
            
            Factory 패턴의 ProcessorFactory를 활용한 벡터 검색 기능
            """
            self.check_managers()
            
            try:
                result = self.processor_manager.process_data("vector_search", {
                    "query": request.query,
                    "documents": request.documents
                })
                
                return VectorSearchResponse(
                    success=True,
                    query=request.query,
                    similar_documents=result["similar_documents"],
                    total_documents=len(request.documents)
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")
        
        return router 
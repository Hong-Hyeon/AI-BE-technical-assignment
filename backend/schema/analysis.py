from pydantic import BaseModel
from typing import Dict, Any, List
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
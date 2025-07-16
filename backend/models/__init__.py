"""
Models package for the SearchRight AI API.
"""
from .talent import (
    TalentData,
    Position,
    AnalysisResult
)

from .company import (
    # SQLAlchemy Models
    Base,
    Company,
    CompanyNews,
    
    # Pydantic Schemas
    CompanyBase,
    CompanyCreate,
    CompanyUpdate,
    CompanyResponse,
    CompanyWithNewsResponse,
    
    CompanyNewsBase,
    CompanyNewsCreate,
    CompanyNewsUpdate,
    CompanyNewsResponse,
    
    # Utility Schemas
    CompanyNewsListResponse,
    CompanySearchResponse
)

__all__ = [
    # Talent Models
    "TalentData",
    "Position", 
    "AnalysisResult",
    
    # SQLAlchemy Models
    "Base",
    "Company",
    "CompanyNews",
    
    # Company Pydantic Schemas
    "CompanyBase",
    "CompanyCreate", 
    "CompanyUpdate",
    "CompanyResponse",
    "CompanyWithNewsResponse",
    
    "CompanyNewsBase",
    "CompanyNewsCreate",
    "CompanyNewsUpdate", 
    "CompanyNewsResponse",
    
    # Utility Schemas
    "CompanyNewsListResponse",
    "CompanySearchResponse"
] 
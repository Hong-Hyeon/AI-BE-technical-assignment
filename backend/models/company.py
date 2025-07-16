"""
Company and Company News models for SQLAlchemy and Pydantic.
"""
from sqlalchemy import Column, Integer, String, Date, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import date
from pgvector.sqlalchemy import Vector

Base = declarative_base()


# SQLAlchemy Models
class Company(Base):
    """Company SQLAlchemy model."""
    __tablename__ = "company"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    data = Column(JSON, nullable=False)
    embedding = Column(Vector(1536), nullable=True)  # OpenAI text-embedding-3-small dimension

    # Relationship
    news = relationship("CompanyNews", back_populates="company", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Company(id={self.id}, name='{self.name}')>"


class CompanyNews(Base):
    """Company News SQLAlchemy model."""
    __tablename__ = "company_news"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("public.company.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(1000), nullable=False)
    original_link = Column(Text, nullable=True)
    news_date = Column(Date, nullable=False)
    embedding = Column(Vector(1536), nullable=True)  # OpenAI text-embedding-3-small dimension

    # Relationship
    company = relationship("Company", back_populates="news")

    def __repr__(self):
        return f"<CompanyNews(id={self.id}, company_id={self.company_id}, title='{self.title[:50]}...')>"


# Pydantic Schemas
class CompanyNewsBase(BaseModel):
    """Base Pydantic schema for company news."""
    title: str = Field(..., max_length=1000, description="뉴스 제목")
    original_link: Optional[str] = Field(None, description="원본 링크")
    news_date: date = Field(..., description="뉴스 날짜")

    class Config:
        from_attributes = True


class CompanyNewsCreate(CompanyNewsBase):
    """Pydantic schema for creating company news."""
    company_id: int = Field(..., description="회사 ID")


class CompanyNewsUpdate(BaseModel):
    """Pydantic schema for updating company news."""
    title: Optional[str] = Field(None, max_length=1000, description="뉴스 제목")
    original_link: Optional[str] = Field(None, description="원본 링크")
    news_date: Optional[date] = Field(None, description="뉴스 날짜")

    class Config:
        from_attributes = True


class CompanyNewsResponse(CompanyNewsBase):
    """Pydantic schema for company news response."""
    id: int = Field(..., description="뉴스 ID")
    company_id: int = Field(..., description="회사 ID")

    class Config:
        from_attributes = True


class CompanyBase(BaseModel):
    """Base Pydantic schema for company."""
    name: str = Field(..., max_length=255, description="회사명")
    data: Dict[str, Any] = Field(..., description="회사 데이터 (JSON)")

    class Config:
        from_attributes = True


class CompanyCreate(CompanyBase):
    """Pydantic schema for creating company."""
    pass


class CompanyUpdate(BaseModel):
    """Pydantic schema for updating company."""
    name: Optional[str] = Field(None, max_length=255, description="회사명")
    data: Optional[Dict[str, Any]] = Field(None, description="회사 데이터 (JSON)")

    class Config:
        from_attributes = True


class CompanyResponse(CompanyBase):
    """Pydantic schema for company response."""
    id: int = Field(..., description="회사 ID")
    news: List[CompanyNewsResponse] = Field(default=[], description="회사 관련 뉴스 목록")

    class Config:
        from_attributes = True


class CompanyWithNewsResponse(CompanyBase):
    """Pydantic schema for company with news response."""
    id: int = Field(..., description="회사 ID")
    news: List[CompanyNewsResponse] = Field(default=[], description="회사 관련 뉴스 목록")

    class Config:
        from_attributes = True


# Additional utility schemas
class CompanyNewsListResponse(BaseModel):
    """Pydantic schema for company news list response."""
    company_id: int = Field(..., description="회사 ID")
    company_name: str = Field(..., description="회사명")
    news: List[CompanyNewsResponse] = Field(default=[], description="뉴스 목록")
    total_count: int = Field(..., description="총 뉴스 개수")

    class Config:
        from_attributes = True


class CompanySearchResponse(BaseModel):
    """Pydantic schema for company search response."""
    companies: List[CompanyResponse] = Field(default=[], description="회사 목록")
    total_count: int = Field(..., description="총 회사 개수")
    page: int = Field(..., description="현재 페이지")
    page_size: int = Field(..., description="페이지 크기")

    class Config:
        from_attributes = True 
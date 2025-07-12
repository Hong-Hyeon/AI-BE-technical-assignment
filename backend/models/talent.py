"""
Talent data models for the LLM-based experience analysis system.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DateInfo(BaseModel):
    """Date information for positions and education."""
    year: int
    month: Optional[int] = None


class StartEndDate(BaseModel):
    """Start and end date for positions and education."""
    start: Optional[DateInfo] = None
    end: Optional[DateInfo] = None


class Position(BaseModel):
    """Work position information."""
    title: str
    company_name: str = Field(alias="companyName")
    description: str
    start_end_date: Optional[StartEndDate] = Field(alias="startEndDate")
    company_location: Optional[str] = Field(alias="companyLocation")
    company_logo: Optional[str] = Field(alias="companyLogo", default="")


class Education(BaseModel):
    """Education information."""
    school_name: str = Field(alias="schoolName")
    degree_name: str = Field(alias="degreeName")
    field_of_study: str = Field(alias="fieldOfStudy")
    start_end_date: Optional[str] = Field(alias="startEndDate")
    description: Optional[str] = ""
    grade: Optional[str] = ""


class TalentData(BaseModel):
    """Complete talent data structure."""
    first_name: str = Field(alias="firstName")
    last_name: str = Field(alias="lastName")
    summary: str
    headline: str
    skills: List[str]
    positions: List[Position]
    educations: List[Education]
    industry_name: str = Field(alias="industryName")
    linkedin_url: Optional[str] = Field(alias="linkedinUrl")
    website: Optional[List[str]] = []
    photo_url: Optional[str] = Field(alias="photoUrl", default="")
    projects: Optional[List[Dict[str, Any]]] = []
    recommendations: Optional[List[Dict[str, Any]]] = []

    class Config:
        validate_by_name = True


class ExperienceTag(BaseModel):
    """Experience tag with confidence score."""
    tag: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class AnalysisResult(BaseModel):
    """Result of talent analysis."""
    talent_id: str
    experience_tags: List[ExperienceTag]
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict) 
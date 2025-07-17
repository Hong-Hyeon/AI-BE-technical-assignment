"""
Test fixtures for generating test data.
"""

from typing import List
from datetime import datetime
from models.talent import TalentData, Education, Position, AnalysisResult, ExperienceTag
from schema import AnalyzeRequest, LLMConfiguration, AnalysisOptions, AnalysisTypeEnum
from core.container import create_test_container


def create_test_talent_data(talent_id: str = "test_talent_001") -> TalentData:
    """Create test talent data."""
    return TalentData(
        first_name="John",
        last_name="Doe",
        headline="Software Engineer",
        summary="Experienced software engineer with expertise in AI/ML",
        educations=[
            Education(
                school_name="Stanford University",
                degree_name="Master of Science",
                field_of_study="Computer Science",
                start_end_date="2018-2020"
            )
        ],
        positions=[
            Position(
                title="Senior Software Engineer",
                company_name="TechCorp",
                description="Developed AI-powered applications",
                start_end_date="2020-Present",
                company_location="San Francisco, CA"
            )
        ],
        skills=["Python", "Machine Learning", "FastAPI", "PostgreSQL"]
    )


def create_test_analysis_request(talent_id: str = "test_talent_001") -> AnalyzeRequest:
    """Create test analysis request."""
    return AnalyzeRequest(
        talent_id=talent_id,
        analysis_type=AnalysisTypeEnum.FULL,
        llm_config=LLMConfiguration(),
        options=AnalysisOptions()
    )


def create_test_analysis_result(talent_id: str = "test_talent_001") -> AnalysisResult:
    """Create test analysis result."""
    return AnalysisResult(
        talent_id=talent_id,
        experience_tags=[
            ExperienceTag(
                tag="AI/ML Expert",
                confidence=0.9,
                reasoning="Strong background in machine learning"
            )
        ],
        processing_time=1.5,
        timestamp=datetime.now(),
        metadata={}
    ) 
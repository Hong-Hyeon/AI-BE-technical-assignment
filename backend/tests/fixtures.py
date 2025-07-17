"""
Test fixtures for generating test data.
"""

from typing import List
from datetime import datetime
from models.talent import TalentData, Education, Position, AnalysisResult, ExperienceTag
from schema import AnalyzeRequest, LLMConfiguration, AnalysisOptions, AnalysisTypeEnum
from core.container import create_test_container


def create_test_talent_data(talent_id: str = "test_talent_001") -> TalentData:
    """Create test talent data with Docker-compatible validation."""
    try:
        # Create education with all required fields explicitly
        education = Education(
            schoolName="Stanford University",
            degreeName="Master of Science",
            fieldOfStudy="Computer Science",
            startEndDate="2018-2020",
            description="",  # Provide default for optional field
            grade=""  # Provide default for optional field
        )
        
        # Create position with all required fields explicitly
        position = Position(
            title="Senior Software Engineer",
            companyName="TechCorp", 
            description="Developed AI-powered applications",
            startEndDate=None,  # Explicit None for optional field
            companyLocation="San Francisco, CA",
            companyLogo=""  # Provide default for optional field
        )
        
        # Create TalentData with all fields to avoid validation issues
        return TalentData(
            firstName="John",
            lastName="Doe",
            headline="Software Engineer",
            summary="Experienced software engineer with expertise in AI/ML",
            educations=[education],
            positions=[position],
            skills=["Python", "Machine Learning", "FastAPI", "PostgreSQL"],
            industryName="Technology",
            linkedinUrl="https://linkedin.com/in/johndoe",
            website=[],  # Provide default empty list
            photoUrl="",  # Provide default empty string
            projects=[],  # Provide default empty list
            recommendations=[]  # Provide default empty list
        )
    except Exception as e:
        # Fallback for any validation issues in Docker
        print(f"Warning: TalentData creation failed with {e}, using model_validate")
        return TalentData.model_validate({
            "firstName": "John",
            "lastName": "Doe", 
            "headline": "Software Engineer",
            "summary": "Experienced software engineer",
            "educations": [],
            "positions": [],
            "skills": ["Python"],
            "industryName": "Technology",
            "linkedinUrl": "https://linkedin.com/in/johndoe"
        })


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
        # metadata={}
    ) 
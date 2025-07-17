"""
Experience Analyzer factory with centralized prompt management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

from models.talent import TalentData, AnalysisResult, ExperienceTag
from models.company import Company
from db.session import SessionLocal
from langchain_core.messages import HumanMessage
from factories.prompt_factory import get_experience_tagging_prompt
import json


class ExperienceAnalyzer(ABC):
    """Abstract base class for experience analyzers."""
    
    @abstractmethod
    async def analyze(self, talent_data: TalentData, context: Dict[str, Any]) -> AnalysisResult:
        """Analyze talent data and return experience tags."""
        pass


class DefaultExperienceAnalyzer(ExperienceAnalyzer):
    """Default implementation of experience analyzer using centralized prompts."""
    
    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.experience_tags = [
            "상위권대학교", "중위권대학교", "하위권대학교",
            "대규모 회사 경험", "중견기업 경험", "스타트업 경험", "성장기스타트업 경험",
            "리더십", "시니어 경력", "주니어 경력",
            "대용량데이터처리경험", "AI/ML 경험", "클라우드 경험",
            "IPO", "M&A 경험", "신규 투자 유치 경험",
            "해외 경험", "글로벌 기업 경험",
            "기술 특허", "오픈소스 기여", "연구 경험"
        ]
    
    async def analyze(self, talent_data: TalentData, context: Dict[str, Any]) -> AnalysisResult:
        """Analyze talent data using centralized prompt factory."""
        
        # Extract information for prompt variables
        positions_info = self._extract_positions_info(talent_data)
        education_info = self._extract_education_info(talent_data)
        company_context = self._extract_company_context(context)
        
        # Get prompt from factory
        prompt = get_experience_tagging_prompt({
            "first_name": talent_data.first_name,
            "last_name": talent_data.last_name,
            "summary": talent_data.summary,
            "headline": talent_data.headline,
            "skills": ', '.join(talent_data.skills),
            "industry_name": talent_data.industry_name,
            "positions_info": positions_info,
            "education_info": education_info,
            "company_context": company_context,
            "experience_tags": ', '.join(self.experience_tags)
        })
        
        try:
            # Call LLM with the rendered prompt
            response = await self.llm_model.ainvoke([HumanMessage(content=prompt)])
            
            # Parse experience tags from response
            experience_tags = self._parse_experience_tags(response.content)
            
            return AnalysisResult(
                talent_id=f"{talent_data.first_name}_{talent_data.last_name}",
                experience_tags=experience_tags,
                processing_time=0.0,  # Will be calculated by caller
                timestamp=None,  # Will be set by caller
                metadata={"analyzer_type": "default", "prompt_factory_used": True}
            )
            
        except Exception as e:
            # Return fallback result on error
            return AnalysisResult(
                talent_id=f"{talent_data.first_name}_{talent_data.last_name}",
                experience_tags=[
                    ExperienceTag(
                        tag="분석 실패",
                        confidence=0.1,
                        reasoning=f"분석 중 오류 발생: {str(e)}"
                    )
                ],
                processing_time=0.0,
                timestamp=None,
                metadata={"analyzer_type": "default", "error": str(e)}
            )
    
    def _extract_positions_info(self, talent_data: TalentData) -> str:
        """Extract position information as formatted string."""
        if not talent_data.positions:
            return "경력 정보 없음"
        
        positions_text = []
        for position in talent_data.positions:
            pos_text = f"""
**{position.title}** at **{position.company_name}**
- 기간: {position.start_end_date}
- 위치: {position.company_location}
- 설명: {position.description}
"""
            positions_text.append(pos_text)
        
        return "\n".join(positions_text)
    
    def _extract_education_info(self, talent_data: TalentData) -> str:
        """Extract education information as formatted string."""
        if not talent_data.educations:
            return "교육 정보 없음"
        
        education_text = []
        for education in talent_data.educations:
            edu_text = f"""
**{education.degree_name}** in **{education.field_of_study}**
- 학교: {education.school_name}
- 기간: {education.start_end_date}
"""
            education_text.append(edu_text)
        
        return "\n".join(education_text)
    
    def _extract_company_context(self, context: Dict[str, Any]) -> str:
        """Extract company context information."""
        if not context:
            return "회사 컨텍스트 정보 없음"
        
        context_text = []
        
        # Add company information
        companies = context.get("companies", {})
        for company_name, company_data in companies.items():
            if hasattr(company_data, 'name'):
                comp_text = f"""
**{company_data.name}**
- 산업: {getattr(company_data, 'industry', 'N/A')}
- 설립년도: {getattr(company_data, 'founded_year', 'N/A')}
- 직원수: {getattr(company_data, 'employee_count', 'N/A')}
- 펀딩: {getattr(company_data, 'funding_amount', 'N/A')}
"""
                context_text.append(comp_text)
        
        # Add news information
        news = context.get("news", {})
        for company_name, news_data in news.items():
            if news_data and news_data.get("news_data"):
                news_text = f"\n**{company_name} 관련 뉴스:**"
                for news_item in news_data["news_data"][:3]:  # Top 3 news
                    news_text += f"\n- {getattr(news_item, 'title', 'N/A')}"
                context_text.append(news_text)
        
        return "\n".join(context_text) if context_text else "회사 컨텍스트 정보 없음"
    
    def _parse_experience_tags(self, llm_response: str) -> List[ExperienceTag]:
        """Parse LLM response into ExperienceTag objects."""
        try:
            # Try to extract JSON from the response
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = llm_response[start_idx:end_idx]
                response_data = json.loads(json_str)
                
                # Extract experience_tags array
                tags_data = response_data.get("experience_tags", [])
                
                return [
                    ExperienceTag(
                        tag=tag["tag"],
                        confidence=tag["confidence"],
                        reasoning=tag["reasoning"]
                    )
                    for tag in tags_data
                    if tag.get("confidence", 0) >= 0.6  # Only include high-confidence tags
                ]
                
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: try to parse array format
            try:
                start_idx = llm_response.find('[')
                end_idx = llm_response.rfind(']') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = llm_response[start_idx:end_idx]
                    tags_data = json.loads(json_str)
                    
                    return [
                        ExperienceTag(
                            tag=tag["tag"],
                            confidence=tag["confidence"],
                            reasoning=tag["reasoning"]
                        )
                        for tag in tags_data
                        if tag.get("confidence", 0) >= 0.6
                    ]
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        
        # Fallback parsing or default tags
        return [
            ExperienceTag(
                tag="분석 완료",
                confidence=0.7,
                reasoning="LLM 응답 파싱 실패, 수동 검토 필요"
            )
        ]


class ExperienceAnalyzerManager:
    """Manager for experience analyzer instances."""
    
    def __init__(self, factory):
        self.factory = factory
    
    def get_analyzer(self, analyzer_type: str = "default"):
        """Get analyzer instance."""
        return self.factory.create_analyzer(analyzer_type)


class DefaultExperienceAnalyzerFactory:
    """Factory for creating experience analyzer instances."""
    
    def __init__(self, llm_model):
        self.llm_model = llm_model
    
    def create_analyzer(self, analyzer_type: str = "default") -> ExperienceAnalyzer:
        """Create analyzer instance."""
        if analyzer_type == "default":
            return DefaultExperienceAnalyzer(self.llm_model)
        else:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}")


# Export classes
__all__ = [
    "ExperienceAnalyzer",
    "DefaultExperienceAnalyzer", 
    "ExperienceAnalyzerManager",
    "DefaultExperienceAnalyzerFactory"
] 
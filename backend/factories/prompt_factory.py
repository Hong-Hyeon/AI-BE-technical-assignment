"""
Prompt Factory for managing and organizing all prompts used in the talent analysis system.

This factory provides centralized prompt management with support for:
- Categorization by use case (education, position, aggregation, etc.)
- Versioning for prompt evolution
- Template variables and substitution
- Prompt validation and formatting
- Easy maintenance and consistency
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import logging

from config.logging_config import get_factory_logger


class PromptCategory(Enum):
    """Categories for different types of prompts."""
    EDUCATION_ANALYSIS = "education_analysis"
    POSITION_ANALYSIS = "position_analysis"
    AGGREGATION = "aggregation"
    EXPERIENCE_TAGGING = "experience_tagging"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    VALIDATION = "validation"


class PromptVersion(Enum):
    """Version control for prompts."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"
    LATEST = "latest"


@dataclass
class PromptMetadata:
    """Metadata for prompt templates."""
    category: PromptCategory
    version: PromptVersion
    name: str
    description: str
    created_date: datetime
    last_modified: datetime
    author: str
    required_variables: List[str]
    optional_variables: List[str]
    output_format: str
    language: str = "Korean"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "category": self.category.value,
            "version": self.version.value,
            "name": self.name,
            "description": self.description,
            "created_date": self.created_date.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "author": self.author,
            "required_variables": self.required_variables,
            "optional_variables": self.optional_variables,
            "output_format": self.output_format,
            "language": self.language
        }


@dataclass
class PromptTemplate:
    """A complete prompt template with metadata and content."""
    metadata: PromptMetadata
    template: str
    examples: Optional[List[Dict[str, Any]]] = None
    validation_rules: Optional[List[str]] = None
    
    def render(self, variables: Dict[str, Any]) -> str:
        """Render the template with provided variables."""
        # Validate required variables
        missing_vars = set(self.metadata.required_variables) - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Apply template substitution
        try:
            return self.template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Template variable not provided: {e}")
    
    def validate_output(self, output: str) -> bool:
        """Validate output against defined rules."""
        if not self.validation_rules:
            return True
        
        # Add validation logic here if needed
        return True


class BasePromptFactory(ABC):
    """Abstract base class for prompt factories."""
    
    @abstractmethod
    def get_prompt(self, category: PromptCategory, name: str, version: PromptVersion = PromptVersion.LATEST) -> PromptTemplate:
        """Get a specific prompt template."""
        pass
    
    @abstractmethod
    def list_prompts(self, category: Optional[PromptCategory] = None) -> List[PromptMetadata]:
        """List available prompts."""
        pass
    
    @abstractmethod
    def register_prompt(self, prompt: PromptTemplate) -> None:
        """Register a new prompt template."""
        pass


class TalentAnalysisPromptFactory(BasePromptFactory):
    """Factory for talent analysis prompts with comprehensive prompt management."""
    
    def __init__(self):
        self.logger = get_factory_logger()
        self._prompts: Dict[str, PromptTemplate] = {}
        self._initialize_prompts()
        self.logger.info("TalentAnalysisPromptFactory initialized with all prompt templates")
    
    def _make_key(self, category: PromptCategory, name: str, version: PromptVersion) -> str:
        """Create a unique key for prompt storage."""
        return f"{category.value}:{name}:{version.value}"
    
    def _initialize_prompts(self) -> None:
        """Initialize all prompt templates."""
        self._register_education_prompts()
        self._register_position_prompts()
        self._register_aggregation_prompts()
        self._register_experience_tagging_prompts()
        
        self.logger.info(f"Initialized {len(self._prompts)} prompt templates across {len(PromptCategory)} categories")
    
    def _register_education_prompts(self) -> None:
        """Register education analysis prompts."""
        
        # Education Analysis v1.0
        education_template = """다음 교육 정보를 분석하여 대학교의 등급을 분류해주세요:

학교명: {school_name}
학위: {degree_name}
전공: {field_of_study}
기간: {start_end_date}

대학교를 다음 기준으로 분류해주세요:
- 상위권: SKY(서울대, 연세대, 고려대), KAIST, POSTECH 등 최상위 대학
- 중위권: 지방 국립대, 주요 사립대 (성균관대, 한양대, 중앙대, 경희대 등)
- 하위권: 기타 대학

응답 형식:
{{
    "tier": "상위권/중위권/하위권",
    "confidence": 0.9,
    "reasoning": "분류 근거"
}}"""

        education_metadata = PromptMetadata(
            category=PromptCategory.EDUCATION_ANALYSIS,
            version=PromptVersion.V1_0,
            name="university_tier_classification",
            description="한국 대학교 등급 분류를 위한 프롬프트",
            created_date=datetime.now(),
            last_modified=datetime.now(),
            author="AI-BE-technical-assignment",
            required_variables=["school_name", "degree_name", "field_of_study", "start_end_date"],
            optional_variables=[],
            output_format="JSON with tier, confidence, reasoning fields",
            language="Korean"
        )
        
        education_prompt = PromptTemplate(
            metadata=education_metadata,
            template=education_template,
            examples=[
                {
                    "input": {"school_name": "연세대학교", "degree_name": "학사", "field_of_study": "컴퓨터과학", "start_end_date": "2018-2022"},
                    "output": '{"tier": "상위권", "confidence": 0.95, "reasoning": "연세대학교는 SKY 대학 중 하나로 최상위 대학에 해당"}'
                }
            ],
            validation_rules=["Output must be valid JSON", "tier must be one of: 상위권, 중위권, 하위권"]
        )
        
        self.register_prompt(education_prompt)
    
    def _register_position_prompts(self) -> None:
        """Register position analysis prompts."""
        
        # Position Analysis v1.0
        position_template = """다음 경력을 회사의 성장 단계와 시장 상황을 고려하여 분석해주세요:

포지션 정보:
- 직책: {title}
- 회사: {company_name}
- 설명: {description}
- 기간: {start_end_date}
- 위치: {company_location}

{company_context}

{news_context}

다음 관점에서 분석해주세요:
1. 회사의 성장 단계 (스타트업/성장기/성숙기)
2. 시장에서의 위치와 경쟁력
3. 해당 시기의 회사 상황 (투자, IPO, M&A 등)
4. 직책의 중요도와 리더십 역할
5. 기술적/비즈니스적 임팩트

응답 형식:
{{
    "company_stage": "성장 단계",
    "leadership_role": "리더십 여부와 수준",
    "market_timing": "시장 타이밍 분석",
    "key_achievements": "주요 성과 추정",
    "confidence": 0.8
}}"""

        position_metadata = PromptMetadata(
            category=PromptCategory.POSITION_ANALYSIS,
            version=PromptVersion.V1_0,
            name="career_position_analysis",
            description="경력 포지션 분석을 위한 컨텍스트 기반 프롬프트",
            created_date=datetime.now(),
            last_modified=datetime.now(),
            author="AI-BE-technical-assignment",
            required_variables=["title", "company_name", "description", "start_end_date", "company_location"],
            optional_variables=["company_context", "news_context"],
            output_format="JSON with company_stage, leadership_role, market_timing, key_achievements, confidence fields",
            language="Korean"
        )
        
        position_prompt = PromptTemplate(
            metadata=position_metadata,
            template=position_template,
            validation_rules=["Output must be valid JSON", "confidence must be between 0.0 and 1.0"]
        )
        
        self.register_prompt(position_prompt)
    
    def _register_aggregation_prompts(self) -> None:
        """Register aggregation and summarization prompts."""
        
        # Experience Tag Generation v1.0
        aggregation_template = """다음 인재의 교육과 경력을 종합적으로 분석하여 핵심 태그와 요약을 생성해주세요:

인재 정보:
- 이름: {first_name} {last_name}
- 헤드라인: {headline}
- 요약: {summary}
- 스킬: {skills}

교육 분석:
{education_analysis}

경력 분석:
{position_analysis}

벡터 검색으로 발견된 관련 컨텍스트:
- 검색된 관련 회사 수: {companies_count}
- 검색된 관련 뉴스 수: {news_count}

다음 형식으로 핵심 경험 태그들을 생성해주세요:

예시 태그들:
- 상위권대학교 (구체적 학교명)
- 성장기스타트업 경험 (회사명과 성장 지표)
- 대규모 회사 경험 (회사명과 직책/역할)
- 리더십 (구체적 직책/역할)
- 대용량데이터처리경험 (관련 회사/프로젝트)
- IPO (관련 회사와 시기)
- M&A 경험 (관련 거래)
- 신규 투자 유치 경험 (회사명과 역할)

각 태그에 대해 다음 JSON 형식으로 응답해주세요:
[
    {{
        "tag": "태그명",
        "confidence": 0.9,
        "reasoning": "이 태그를 부여한 구체적인 근거"
    }}
]"""

        aggregation_metadata = PromptMetadata(
            category=PromptCategory.AGGREGATION,
            version=PromptVersion.V1_0,
            name="experience_tag_generation",
            description="교육과 경력 분석을 종합하여 경험 태그를 생성하는 프롬프트",
            created_date=datetime.now(),
            last_modified=datetime.now(),
            author="AI-BE-technical-assignment",
            required_variables=[
                "first_name", "last_name", "headline", "summary", "skills",
                "education_analysis", "position_analysis", "companies_count", "news_count"
            ],
            optional_variables=[],
            output_format="JSON array with tag, confidence, reasoning objects",
            language="Korean"
        )
        
        aggregation_prompt = PromptTemplate(
            metadata=aggregation_metadata,
            template=aggregation_template,
            validation_rules=["Output must be valid JSON array", "Each tag must have confidence between 0.0 and 1.0"]
        )
        
        self.register_prompt(aggregation_prompt)
    
    def _register_experience_tagging_prompts(self) -> None:
        """Register comprehensive experience tagging prompts."""
        
        # Experience Tagging Analysis v1.0 (from experience_analyzer_factory.py)
        experience_template = """당신은 인재의 경력 데이터를 분석하여 경험 태그를 추론하는 전문가입니다.

## 분석 대상 인재 정보:
**이름**: {first_name} {last_name}
**요약**: {summary}
**헤드라인**: {headline}
**스킬**: {skills}
**산업**: {industry_name}

## 경력 정보:
{positions_info}

## 교육 정보:
{education_info}

## 회사 컨텍스트 정보:
{company_context}

## 가능한 경험 태그:
{experience_tags}

## 분석 지침:
1. 위의 정보를 바탕으로 해당 인재가 가지고 있을 것으로 추론되는 경험 태그를 식별하세요.
2. 각 태그에 대해 0.0-1.0 사이의 신뢰도 점수를 부여하세요.
3. 각 태그를 선택한 이유를 명확하게 설명하세요.
4. 신뢰도가 0.6 이상인 태그만 포함하세요.

## 응답 형식 (JSON):
{{
    "experience_tags": [
        {{
            "tag": "태그명",
            "confidence": 0.8,
            "reasoning": "선택 이유 설명"
        }}
    ]
}}

분석을 시작하세요:"""

        experience_metadata = PromptMetadata(
            category=PromptCategory.EXPERIENCE_TAGGING,
            version=PromptVersion.V1_0,
            name="comprehensive_experience_analysis",
            description="종합적인 경험 태그 분석을 위한 상세 프롬프트",
            created_date=datetime.now(),
            last_modified=datetime.now(),
            author="AI-BE-technical-assignment",
            required_variables=[
                "first_name", "last_name", "summary", "headline", "skills", "industry_name",
                "positions_info", "education_info", "company_context", "experience_tags"
            ],
            optional_variables=[],
            output_format="JSON with experience_tags array containing tag, confidence, reasoning objects",
            language="Korean"
        )
        
        experience_prompt = PromptTemplate(
            metadata=experience_metadata,
            template=experience_template,
            validation_rules=[
                "Output must be valid JSON",
                "confidence values must be between 0.0 and 1.0",
                "Only include tags with confidence >= 0.6"
            ]
        )
        
        self.register_prompt(experience_prompt)
    
    def get_prompt(self, category: PromptCategory, name: str, version: PromptVersion = PromptVersion.LATEST) -> PromptTemplate:
        """Get a specific prompt template."""
        if version == PromptVersion.LATEST:
            # Find the latest version for this category and name
            matching_keys = [k for k in self._prompts.keys() if k.startswith(f"{category.value}:{name}:")]
            if not matching_keys:
                raise ValueError(f"No prompt found for category '{category.value}' and name '{name}'")
            
            # Sort by version and get the latest
            latest_key = sorted(matching_keys)[-1]
            prompt = self._prompts[latest_key]
        else:
            key = self._make_key(category, name, version)
            if key not in self._prompts:
                raise ValueError(f"Prompt not found: {key}")
            prompt = self._prompts[key]
        
        self.logger.debug(f"Retrieved prompt: {category.value}:{name}:{prompt.metadata.version.value}")
        return prompt
    
    def list_prompts(self, category: Optional[PromptCategory] = None) -> List[PromptMetadata]:
        """List available prompts."""
        prompts = []
        for prompt in self._prompts.values():
            if category is None or prompt.metadata.category == category:
                prompts.append(prompt.metadata)
        
        # Sort by category and name
        prompts.sort(key=lambda p: (p.category.value, p.name, p.version.value))
        return prompts
    
    def register_prompt(self, prompt: PromptTemplate) -> None:
        """Register a new prompt template."""
        key = self._make_key(prompt.metadata.category, prompt.metadata.name, prompt.metadata.version)
        self._prompts[key] = prompt
        
        self.logger.debug(f"Registered prompt: {key}")
    
    def get_company_context_template(self, company_data: Any) -> str:
        """Generate company context template section."""
        if not company_data:
            return ""
        
        return f"""
회사 정보:
- 회사명: {getattr(company_data, 'name', 'N/A')}
- 산업: {getattr(company_data, 'industry', 'N/A')}
- 설립년도: {getattr(company_data, 'founded_year', 'N/A')}
- 직원수: {getattr(company_data, 'employee_count', 'N/A')}
- 펀딩 규모: {getattr(company_data, 'funding_amount', 'N/A')}"""
    
    def get_news_context_template(self, news_data: Dict[str, Any]) -> str:
        """Generate news context template section."""
        if not news_data or not news_data.get('news_data'):
            return ""
        
        context = f"""
관련 뉴스 (최근 {news_data.get('news_count', 0)}건):"""
        
        for news in news_data['news_data'][:5]:  # Limit to recent 5 news
            context += f"\n- {getattr(news, 'title', '')} ({getattr(news, 'date', '')})"
        
        return context
    
    def export_prompts_to_json(self, filepath: str) -> None:
        """Export all prompts to JSON file for backup/sharing."""
        export_data = {
            "export_date": datetime.now().isoformat(),
            "total_prompts": len(self._prompts),
            "prompts": {
                key: {
                    "metadata": prompt.metadata.to_dict(),
                    "template": prompt.template,
                    "examples": prompt.examples,
                    "validation_rules": prompt.validation_rules
                }
                for key, prompt in self._prompts.items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Exported {len(self._prompts)} prompts to {filepath}")
    
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered prompts."""
        category_counts = {}
        version_counts = {}
        
        for prompt in self._prompts.values():
            category = prompt.metadata.category.value
            version = prompt.metadata.version.value
            
            category_counts[category] = category_counts.get(category, 0) + 1
            version_counts[version] = version_counts.get(version, 0) + 1
        
        return {
            "total_prompts": len(self._prompts),
            "categories": category_counts,
            "versions": version_counts,
            "last_modified": max(prompt.metadata.last_modified for prompt in self._prompts.values()).isoformat()
        }


# Create a singleton instance
talent_analysis_prompt_factory = TalentAnalysisPromptFactory()


# Convenience functions for easy access
def get_education_prompt(variables: Dict[str, Any]) -> str:
    """Get rendered education analysis prompt."""
    prompt_template = talent_analysis_prompt_factory.get_prompt(
        PromptCategory.EDUCATION_ANALYSIS, 
        "university_tier_classification"
    )
    return prompt_template.render(variables)


def get_position_prompt(variables: Dict[str, Any]) -> str:
    """Get rendered position analysis prompt."""
    prompt_template = talent_analysis_prompt_factory.get_prompt(
        PromptCategory.POSITION_ANALYSIS, 
        "career_position_analysis"
    )
    return prompt_template.render(variables)


def get_aggregation_prompt(variables: Dict[str, Any]) -> str:
    """Get rendered experience tag generation prompt."""
    prompt_template = talent_analysis_prompt_factory.get_prompt(
        PromptCategory.AGGREGATION, 
        "experience_tag_generation"
    )
    return prompt_template.render(variables)


def get_experience_tagging_prompt(variables: Dict[str, Any]) -> str:
    """Get rendered comprehensive experience analysis prompt."""
    prompt_template = talent_analysis_prompt_factory.get_prompt(
        PromptCategory.EXPERIENCE_TAGGING, 
        "comprehensive_experience_analysis"
    )
    return prompt_template.render(variables)


# Export convenience functions
__all__ = [
    "TalentAnalysisPromptFactory",
    "PromptCategory",
    "PromptVersion",
    "PromptTemplate",
    "PromptMetadata",
    "talent_analysis_prompt_factory",
    "get_education_prompt",
    "get_position_prompt", 
    "get_aggregation_prompt",
    "get_experience_tagging_prompt"
] 
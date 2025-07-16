"""
Experience analyzer factory for talent analysis.
"""
import json
import time
from typing import Dict, Any, List
from factories.base import ExperienceAnalyzerFactory, ExperienceAnalyzer
from models.talent import TalentData, AnalysisResult, ExperienceTag
from factories.llm_factory import LLMModel


class DefaultExperienceAnalyzer(ExperienceAnalyzer):
    """Default implementation of experience analyzer."""
    
    def __init__(self, llm_model: LLMModel):
        self.llm_model = llm_model
    
    def analyze(self, talent_data: TalentData, context: Dict[str, Any]) -> AnalysisResult:
        """Analyze talent data and return experience tags."""
        start_time = time.time()
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(talent_data, context)
        
        # Generate analysis using LLM
        response = self.llm_model.generate(prompt, temperature=0.3, max_tokens=2000)
        
        # Parse response and extract experience tags
        experience_tags = self._parse_analysis_response(response)
        
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            talent_id=f"{talent_data.first_name}_{talent_data.last_name}",
            experience_tags=experience_tags,
            processing_time=processing_time,
            metadata={
                "model_used": getattr(self.llm_model, 'model_name', 'unknown'),
                "context_sources": list(context.keys()),
                "prompt_length": len(prompt),
                "response_length": len(response)
            }
        )
    
    def _build_analysis_prompt(self, talent_data: TalentData, context: Dict[str, Any]) -> str:
        """Build analysis prompt for LLM."""
        
        # Extract key information
        positions_info = self._extract_positions_info(talent_data)
        education_info = self._extract_education_info(talent_data)
        company_context = self._extract_company_context(context)
        
        prompt = f"""
당신은 인재의 경력 데이터를 분석하여 경험 태그를 추론하는 전문가입니다.

## 분석 대상 인재 정보:
**이름**: {talent_data.first_name} {talent_data.last_name}
**요약**: {talent_data.summary}
**헤드라인**: {talent_data.headline}
**스킬**: {', '.join(talent_data.skills)}
**산업**: {talent_data.industry_name}

## 경력 정보:
{positions_info}

## 교육 정보:
{education_info}

## 회사 컨텍스트 정보:
{company_context}

## 가능한 경험 태그:
{', '.join(self.experience_tags)}

## 분석 지침:
1. 위의 정보를 바탕으로 해당 인재가 가지고 있을 것으로 추론되는 경험 태그를 식별하세요.
2. 각 태그에 대해 0.0-1.0 사이의 신뢰도 점수를 부여하세요.
3. 각 태그를 선택한 이유를 명확하게 설명하세요.
4. 신뢰도가 0.6 이상인 태그만 포함하세요.
5. 다음 기준을 참고하세요:
   - 상위권대학교: 서울대, 연세대, 고려대, KAIST 등
   - 대규모 회사 경험: 삼성, LG, SK, 네이버, 카카오 등
   - 성장기스타트업 경험: 토스, 쿠팡, 배달의민족 등
   - 리더십: 팀장, 리드, CTO, CPO, Director 등
   - 대용량데이터처리경험: 빅데이터, AI, ML 관련 경험
   - IPO: 상장 관련 경험
   - M&A: 인수합병 관련 경험
   - 신규 투자 유치 경험: 시리즈 A, B, C 등 투자 유치

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

분석을 시작하세요:
"""
        return prompt
    
    def _extract_positions_info(self, talent_data: TalentData) -> str:
        """Extract and format positions information."""
        positions_info = []
        
        for i, position in enumerate(talent_data.positions, 1):
            start_date = ""
            end_date = ""
            
            if position.start_end_date:
                if position.start_end_date.start:
                    start_date = f"{position.start_end_date.start.year}년 {position.start_end_date.start.month}월"
                if position.start_end_date.end:
                    end_date = f"{position.start_end_date.end.year}년 {position.start_end_date.end.month}월"
            
            duration = f"{start_date} ~ {end_date}" if start_date or end_date else "기간 정보 없음"
            
            position_info = f"""
**경력 {i}**: {position.title} at {position.company_name}
- 기간: {duration}
- 위치: {position.company_location or '정보 없음'}
- 설명: {position.description}
"""
            positions_info.append(position_info)
        
        return "\n".join(positions_info)
    
    def _extract_education_info(self, talent_data: TalentData) -> str:
        """Extract and format education information."""
        education_info = []
        
        for i, education in enumerate(talent_data.educations, 1):
            edu_info = f"""
**교육 {i}**: {education.degree_name} in {education.field_of_study}
- 학교: {education.school_name}
- 기간: {education.start_end_date}
- 설명: {education.description}
"""
            education_info.append(edu_info)
        
        return "\n".join(education_info) if education_info else "교육 정보 없음"
    
    def _extract_company_context(self, context: Dict[str, Any]) -> str:
        """Extract and format company context information."""
        company_contexts = []
        
        for company_name, company_data in context.items():
            if company_name.startswith("company_"):
                company_info = f"""
**{company_name}**:
- 직원 수 변화: {company_data.get('employee_count_changes', '정보 없음')}
- 투자 정보: {company_data.get('investment_rounds', '정보 없음')}
- 주요 뉴스: {company_data.get('major_news', '정보 없음')}
"""
                company_contexts.append(company_info)
        
        return "\n".join(company_contexts) if company_contexts else "회사 컨텍스트 정보 없음"
    
    def _parse_analysis_response(self, response: str) -> List[ExperienceTag]:
        """Parse LLM response and extract experience tags."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                parsed_response = json.loads(json_str)
                
                experience_tags = []
                for tag_data in parsed_response.get('experience_tags', []):
                    tag = ExperienceTag(
                        tag=tag_data['tag'],
                        confidence=tag_data['confidence'],
                        reasoning=tag_data['reasoning']
                    )
                    experience_tags.append(tag)
                
                return experience_tags
            else:
                raise ValueError("No valid JSON found in response")
                
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: try to parse manually
            return self._fallback_parse_response(response)
    
    def _fallback_parse_response(self, response: str) -> List[ExperienceTag]:
        """Fallback parser for when JSON parsing fails."""
        experience_tags = []
        
        # Simple pattern matching for common tags
        for tag in self.experience_tags:
            if tag in response:
                # Simple confidence estimation based on context
                confidence = 0.7 if "확실" in response or "명확" in response else 0.6
                reasoning = f"텍스트에서 '{tag}' 관련 언급 발견"
                
                experience_tags.append(ExperienceTag(
                    tag=tag,
                    confidence=confidence,
                    reasoning=reasoning
                ))
        
        return experience_tags


class DefaultExperienceAnalyzerFactory(ExperienceAnalyzerFactory):
    """Default factory for experience analyzers."""
    
    SUPPORTED_ANALYZERS = ["default", "comprehensive", "fast"]
    
    def __init__(self, llm_model: LLMModel):
        self.llm_model = llm_model
    
    def create_analyzer(self, analyzer_type: str) -> ExperienceAnalyzer:
        """Create an experience analyzer."""
        if analyzer_type not in self.SUPPORTED_ANALYZERS:
            raise ValueError(f"Unsupported analyzer type: {analyzer_type}")
        
        # For now, return the default analyzer
        # In the future, we can implement different analyzer types
        return DefaultExperienceAnalyzer(self.llm_model)
    
    def get_supported_analyzers(self) -> List[str]:
        """Get list of supported analyzer types."""
        return self.SUPPORTED_ANALYZERS.copy()


class ExperienceAnalyzerManager:
    """Manager for experience analyzers."""
    
    def __init__(self, factory: ExperienceAnalyzerFactory):
        self.factory = factory
        self.analyzers: Dict[str, ExperienceAnalyzer] = {}
    
    def get_analyzer(self, analyzer_type: str = "default") -> ExperienceAnalyzer:
        """Get or create an analyzer instance."""
        if analyzer_type not in self.analyzers:
            self.analyzers[analyzer_type] = self.factory.create_analyzer(analyzer_type)
        
        return self.analyzers[analyzer_type]
    
    def analyze_talent(self, talent_data: TalentData, context: Dict[str, Any],
                       analyzer_type: str = "default") -> AnalysisResult:
        """Analyze talent using specified analyzer."""
        analyzer = self.get_analyzer(analyzer_type)
        return analyzer.analyze(talent_data, context) 
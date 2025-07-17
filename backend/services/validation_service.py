"""
Validation Service for input validation and business rules.
"""

from typing import List, Any, Optional
from schema import AnalyzeRequest, ValidationErrorDetail
from core.exceptions import ValidationException
import re


class ValidationService:
    """Service for input validation and business rule enforcement."""
    
    def __init__(self):
        self.talent_id_pattern = re.compile(r'^[a-zA-Z0-9_-]{3,50}$')
    
    async def validate_analysis_request(self, request: AnalyzeRequest) -> List[ValidationErrorDetail]:
        """Validate analysis request and return any errors."""
        errors = []
        
        # Validate talent ID
        if not self._is_valid_talent_id(request.talent_id):
            errors.append(ValidationErrorDetail(
                field="talent_id",
                message="Invalid talent ID format",
                value=request.talent_id,
                code="INVALID_FORMAT"
            ))
        
        # Validate LLM configuration
        llm_errors = self._validate_llm_config(request.llm_config)
        errors.extend(llm_errors)
        
        # Validate analysis options
        option_errors = self._validate_analysis_options(request.options)
        errors.extend(option_errors)
        
        return errors
    
    def _is_valid_talent_id(self, talent_id: str) -> bool:
        """Check if talent ID is valid."""
        return bool(self.talent_id_pattern.match(talent_id))
    
    def _validate_llm_config(self, config) -> List[ValidationErrorDetail]:
        """Validate LLM configuration."""
        errors = []
        
        if config.temperature < 0 or config.temperature > 1:
            errors.append(ValidationErrorDetail(
                field="llm_config.temperature",
                message="Temperature must be between 0 and 1",
                value=config.temperature,
                code="OUT_OF_RANGE"
            ))
        
        if config.max_tokens < 100 or config.max_tokens > 4000:
            errors.append(ValidationErrorDetail(
                field="llm_config.max_tokens",
                message="Max tokens must be between 100 and 4000",
                value=config.max_tokens,
                code="OUT_OF_RANGE"
            ))
        
        return errors
    
    def _validate_analysis_options(self, options) -> List[ValidationErrorDetail]:
        """Validate analysis options."""
        errors = []
        
        if options.custom_prompt_additions and len(options.custom_prompt_additions) > 500:
            errors.append(ValidationErrorDetail(
                field="options.custom_prompt_additions",
                message="Custom prompt additions cannot exceed 500 characters",
                value=len(options.custom_prompt_additions),
                code="TOO_LONG"
            ))
        
        return errors 
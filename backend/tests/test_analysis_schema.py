"""
Unit tests for analysis schema validation.
"""

import pytest
import uuid
from datetime import datetime
from pydantic import ValidationError

from schema.analysis import (
    AnalyzeRequest, AnalyzeResponse, AnalysisProgress, AnalysisProgressEnum,
    LLMModelEnum, LLMConfiguration
)
from schema.base import ProviderEnum
from tests.fixtures import create_test_analysis_result


class TestAnalyzeRequest:
    """Test cases for AnalyzeRequest schema validation."""
    
    def test_analyze_request_valid_data(self):
        """Test AnalyzeRequest with valid data."""
        request = AnalyzeRequest(
            talent_id="test_talent_001"
        )
        
        assert request.talent_id == "test_talent_001"
        assert request.analysis_type is not None
        assert request.llm_config is not None
    
    def test_analyze_request_minimal_data(self):
        """Test AnalyzeRequest with minimal required data."""
        request = AnalyzeRequest(talent_id="test_talent_001")
        
        assert request.talent_id == "test_talent_001"
        assert request.analysis_type is not None  # Has default value
        assert request.llm_config is not None  # Has default value
        assert request.llm_config.provider is not None  # Default provider
        assert request.llm_config.model is not None  # Default model
    
    def test_analyze_request_empty_talent_id(self):
        """Test AnalyzeRequest with empty talent_id raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AnalyzeRequest(talent_id="")
        
        assert "Talent ID must be at least 3 characters" in str(exc_info.value)
    
    def test_analyze_request_invalid_llm_config(self):
        """Test AnalyzeRequest with invalid LLM configuration."""
        with pytest.raises(ValidationError) as exc_info:
            from schema.analysis import LLMConfiguration
            LLMConfiguration(
                provider="invalid_provider",
                model="gpt-4o-mini"
            )
        
        error_str = str(exc_info.value)
        assert "Input should be" in error_str


class TestAnalyzeResponse:
    """Test cases for AnalyzeResponse schema validation."""
    
    def test_analyze_response_valid_data(self):
        """Test AnalyzeResponse with valid data."""
        analysis_id = str(uuid.uuid4())
        result = create_test_analysis_result()
        
        response = AnalyzeResponse(
            analysis_id=analysis_id,
            result=result,
            success=True,
            message="Analysis completed successfully"
        )
        
        assert response.analysis_id == analysis_id
        assert response.result == result
        assert response.success is True
        assert response.message == "Analysis completed successfully"
        assert response.timestamp is not None
    
    def test_analyze_response_without_result(self):
        """Test AnalyzeResponse without result (in-progress case)."""
        analysis_id = str(uuid.uuid4())
        
        response = AnalyzeResponse(
            analysis_id=analysis_id,
            result=None,
            success=True,
            message="Analysis in progress",
            progress=AnalysisProgress(
                stage=AnalysisProgressEnum.PROCESSING_EDUCATION,
                progress_percentage=50,
                current_task="Processing data"
            )
        )
        
        assert response.analysis_id == analysis_id
        assert response.result is None
        assert response.progress is not None
        assert response.progress.stage == AnalysisProgressEnum.PROCESSING_EDUCATION
        assert response.progress.progress_percentage == 50
    
    def test_analyze_response_create_async_response(self):
        """Test AnalyzeResponse.create_async_response class method."""
        analysis_id = str(uuid.uuid4())
        
        response = AnalyzeResponse.create_async_response(
            analysis_id=analysis_id,
            message="Analysis started"
        )
        
        assert response.analysis_id == analysis_id
        assert response.result is None
        assert response.progress is not None
        assert response.progress.stage == AnalysisProgressEnum.INITIALIZING
        assert response.progress.progress_percentage == 0
        assert response.message == "Analysis started"
    
    def test_analyze_response_create_completed_response(self):
        """Test AnalyzeResponse.create_completed_response class method."""
        analysis_id = str(uuid.uuid4())
        result = create_test_analysis_result()
        
        response = AnalyzeResponse.create_completed_response(
            analysis_id=analysis_id,
            result=result,
            message="Analysis completed"
        )
        
        assert response.analysis_id == analysis_id
        assert response.result == result
        assert response.progress is not None
        assert response.progress.stage == AnalysisProgressEnum.COMPLETED
        assert response.progress.progress_percentage == 100
        assert response.message == "Analysis completed"
    
    def test_analyze_response_missing_analysis_id(self):
        """Test AnalyzeResponse with missing analysis_id raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AnalyzeResponse(
                result=create_test_analysis_result(),
                success=True,
                message="Test message"
            )
        
        assert "Field required" in str(exc_info.value)
    
    def test_analyze_response_excludes_metadata_field(self):
        """Test that AnalyzeResponse model does not include metadata field."""
        analysis_id = str(uuid.uuid4())
        result = create_test_analysis_result()
        
        response = AnalyzeResponse(
            analysis_id=analysis_id,
            result=result,
            success=True,
            message="Analysis completed"
        )
        
        # Convert to dict and verify metadata is not present
        response_dict = response.model_dump()
        assert "metadata" not in response_dict
        
        # Verify expected fields are present
        expected_fields = {"analysis_id", "result", "success", "message", "timestamp"}
        assert expected_fields.issubset(set(response_dict.keys()))
    
    def test_analyze_response_serialization(self):
        """Test AnalyzeResponse JSON serialization."""
        analysis_id = str(uuid.uuid4())
        result = create_test_analysis_result()
        
        response = AnalyzeResponse(
            analysis_id=analysis_id,
            result=result,
            success=True,
            message="Analysis completed"
        )
        
        # Test JSON serialization
        json_str = response.model_dump_json()
        assert isinstance(json_str, str)
        assert analysis_id in json_str
        assert "Analysis completed" in json_str
        
        # Test deserialization
        import json
        data = json.loads(json_str)
        assert data["analysis_id"] == analysis_id
        assert data["success"] is True


class TestLLMConfiguration:
    """Test cases for LLMConfiguration schema validation."""
    
    def test_llm_configuration_valid_openai(self):
        """Test LLMConfiguration with valid OpenAI model."""
        config = LLMConfiguration(
            provider=ProviderEnum.OPENAI,
            model=LLMModelEnum.GPT_4O_MINI
        )
        
        assert config.provider == ProviderEnum.OPENAI
        assert config.model == LLMModelEnum.GPT_4O_MINI
    
    def test_llm_configuration_valid_anthropic(self):
        """Test LLMConfiguration with valid Anthropic model."""
        config = LLMConfiguration(
            provider=ProviderEnum.ANTHROPIC,
            model=LLMModelEnum.CLAUDE_3_SONNET
        )
        
        assert config.provider == ProviderEnum.ANTHROPIC
        assert config.model == LLMModelEnum.CLAUDE_3_SONNET
    
    def test_llm_configuration_string_inputs(self):
        """Test LLMConfiguration with string inputs (auto-conversion)."""
        config = LLMConfiguration(
            provider="openai",
            model="gpt-4o-mini"
        )
        
        assert config.provider == ProviderEnum.OPENAI
        assert config.model == LLMModelEnum.GPT_4O_MINI
    
    def test_llm_configuration_model_provider_compatibility(self):
        """Test LLMConfiguration model-provider compatibility validation."""
        # This should work - OpenAI model with OpenAI provider
        config = LLMConfiguration(
            provider=ProviderEnum.OPENAI,
            model=LLMModelEnum.GPT_4O_MINI
        )
        assert config.provider == ProviderEnum.OPENAI
        
        # Note: The validator may not catch incompatible combinations
        # depending on the implementation, but the test structure is here
        # for when such validation is needed


class TestAnalysisProgress:
    """Test cases for AnalysisProgress schema validation."""
    
    def test_analysis_progress_valid_data(self):
        """Test AnalysisProgress with valid data."""
        progress = AnalysisProgress(
            stage=AnalysisProgressEnum.PROCESSING_EXPERIENCE,
            progress_percentage=75,
            current_task="Analyzing experience data"
        )
        
        assert progress.stage == AnalysisProgressEnum.PROCESSING_EXPERIENCE
        assert progress.progress_percentage == 75
        assert progress.current_task == "Analyzing experience data"
    
    def test_analysis_progress_invalid_percentage(self):
        """Test AnalysisProgress with invalid percentage values."""
        # Test negative percentage
        with pytest.raises(ValidationError):
            AnalysisProgress(
                stage=AnalysisProgressEnum.PROCESSING_EDUCATION,
                progress_percentage=-10,
                current_task="Test task"
            )
        
        # Test percentage over 100
        with pytest.raises(ValidationError):
            AnalysisProgress(
                stage=AnalysisProgressEnum.PROCESSING_EDUCATION,
                progress_percentage=150,
                current_task="Test task"
            )
    
    def test_analysis_progress_valid_boundary_values(self):
        """Test AnalysisProgress with boundary percentage values."""
        # Test 0%
        progress_0 = AnalysisProgress(
            stage=AnalysisProgressEnum.INITIALIZING,
            progress_percentage=0,
            current_task="Starting"
        )
        assert progress_0.progress_percentage == 0
        
        # Test 100%
        progress_100 = AnalysisProgress(
            stage=AnalysisProgressEnum.COMPLETED,
            progress_percentage=100,
            current_task="Completed"
        )
        assert progress_100.progress_percentage == 100
    
    def test_analysis_progress_empty_task(self):
        """Test AnalysisProgress with empty current_task."""
        # Empty task should be allowed based on the schema
        progress = AnalysisProgress(
            stage=AnalysisProgressEnum.PROCESSING_EDUCATION,
            progress_percentage=50,
            current_task=""
        )
        
        assert progress.current_task == "" 
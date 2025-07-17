"""
Unit tests for the analysis router.
"""

import pytest
import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from routers.analysis import AnalysisRouter
from tests.mocks import MockLLMModel, create_mock_llm_model
from tests.fixtures import create_test_talent_data, create_test_analysis_result
from models.talent import AnalysisResult, ExperienceTag
from schema.analysis import AnalyzeResponse


class TestAnalysisRouter:
    """Test cases for the AnalysisRouter class."""
    
    @pytest.fixture
    def router(self):
        """Create analysis router instance."""
        return AnalysisRouter()
    
    @pytest.fixture
    def mock_managers(self):
        """Mock factory managers."""
        return {
            'llm_factory_manager': MagicMock(),
            'data_source_manager': MagicMock(),
            'experience_analyzer_manager': MagicMock(),
            'processor_manager': MagicMock()
        }
    
    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request object."""
        request = MagicMock()
        request.client.host = "127.0.0.1"
        return request
    
    def test_router_initialization(self, router):
        """Test router initialization."""
        assert router is not None
        assert hasattr(router, 'logger')
        assert hasattr(router, 'get_router')
    
    def test_get_router_returns_api_router(self, router):
        """Test that get_router returns a FastAPI APIRouter."""
        api_router = router.get_router()
        assert api_router is not None
        assert hasattr(api_router, 'prefix')
        assert api_router.prefix == "/analyze"
        assert "analysis" in api_router.tags
    
    @pytest.mark.asyncio
    async def test_analyze_talent_get_success(self, router, mock_managers, mock_request):
        """Test successful talent analysis via GET endpoint."""
        # Setup
        router.set_managers(**mock_managers)
        
        # Mock workflow result
        mock_result = create_test_analysis_result()
        
        # Mock LLM creation
        mock_llm = create_mock_llm_model()
        mock_managers['llm_factory_manager'].create_llm.return_value = mock_llm
        
        # Mock workflow execution
        with patch('workflows.talent_analysis.talent_analysis_workflow') as mock_workflow:
            mock_workflow.run_analysis = AsyncMock(return_value=mock_result)
            
            # Get router and execute endpoint
            api_router = router.get_router()
            
            # Find the analyze_talent_get function
            analyze_func = None
            for route in api_router.routes:
                if hasattr(route, 'endpoint') and route.endpoint.__name__ == 'analyze_talent_get':
                    analyze_func = route.endpoint
                    break
            
            assert analyze_func is not None, "analyze_talent_get endpoint not found"
            
            # Execute the endpoint
            response = await analyze_func(
                talent_id="test_talent_001",
                request=mock_request,
                analyzer_type="default",
                llm_provider="openai",
                llm_model="gpt-4o-mini",
                enrich_context=True
            )
            
            # Assertions
            assert isinstance(response, AnalyzeResponse)
            assert response.success is True
            assert response.analysis_id is not None
            assert response.result is not None
            assert len(response.result.experience_tags) > 0
            assert "Successfully analyzed talent test_talent_001" in response.message
            
            # Verify workflow was called
            mock_workflow.run_analysis.assert_called_once_with(
                talent_id="test_talent_001",
                llm_model=mock_llm
            )
    
    @pytest.mark.asyncio
    async def test_analyze_talent_get_not_found(self, router, mock_managers, mock_request):
        """Test analysis when talent data is not found."""
        # Setup
        router.set_managers(**mock_managers)
        
        # Mock LLM creation
        mock_llm = create_mock_llm_model()
        mock_managers['llm_factory_manager'].create_llm.return_value = mock_llm
        
        # Mock workflow to raise ValueError (data not found)
        with patch('workflows.talent_analysis.talent_analysis_workflow') as mock_workflow:
            mock_workflow.run_analysis = AsyncMock(side_effect=ValueError("Talent not found"))
            
            api_router = router.get_router()
            
            # Find the analyze_talent_get function
            analyze_func = None
            for route in api_router.routes:
                if hasattr(route, 'endpoint') and route.endpoint.__name__ == 'analyze_talent_get':
                    analyze_func = route.endpoint
                    break
            
            # Execute and expect HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await analyze_func(
                    talent_id="nonexistent_talent",
                    request=mock_request,
                    analyzer_type="default",
                    llm_provider="openai",
                    llm_model="gpt-4o-mini",
                    enrich_context=True
                )
            
            assert exc_info.value.status_code == 404
            assert "Data not found" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_analyze_talent_get_internal_error(self, router, mock_managers, mock_request):
        """Test analysis with internal server error."""
        # Setup
        router.set_managers(**mock_managers)
        
        # Mock LLM creation
        mock_llm = create_mock_llm_model()
        mock_managers['llm_factory_manager'].create_llm.return_value = mock_llm
        
        # Mock workflow to raise general exception
        with patch('workflows.talent_analysis.talent_analysis_workflow') as mock_workflow:
            mock_workflow.run_analysis = AsyncMock(side_effect=Exception("Internal error"))
            
            api_router = router.get_router()
            
            # Find the analyze_talent_get function
            analyze_func = None
            for route in api_router.routes:
                if hasattr(route, 'endpoint') and route.endpoint.__name__ == 'analyze_talent_get':
                    analyze_func = route.endpoint
                    break
            
            # Execute and expect HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await analyze_func(
                    talent_id="test_talent_001",
                    request=mock_request,
                    analyzer_type="default",
                    llm_provider="openai",
                    llm_model="gpt-4o-mini",
                    enrich_context=True
                )
            
            assert exc_info.value.status_code == 500
            assert "Analysis failed" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_analyze_talent_post_delegates_to_get(self, router, mock_managers, mock_request):
        """Test that POST endpoint correctly delegates to GET endpoint."""
        # Setup
        router.set_managers(**mock_managers)
        
        # Mock result
        mock_result = create_test_analysis_result()
        
        # Mock LLM creation
        mock_llm = create_mock_llm_model()
        mock_managers['llm_factory_manager'].create_llm.return_value = mock_llm
        
        # Mock workflow
        with patch('workflows.talent_analysis.talent_analysis_workflow') as mock_workflow:
            mock_workflow.run_analysis = AsyncMock(return_value=mock_result)
            
            api_router = router.get_router()
            
            # Find the analyze_talent_post function
            analyze_post_func = None
            for route in api_router.routes:
                if hasattr(route, 'endpoint') and route.endpoint.__name__ == 'analyze_talent_post':
                    analyze_post_func = route.endpoint
                    break
            
            assert analyze_post_func is not None, "analyze_talent_post endpoint not found"
            
            # Create request data
            from schema.analysis import AnalyzeRequest
            request_data = AnalyzeRequest(
                talent_id="test_talent_001"
            )
            
            # Execute the endpoint
            response = await analyze_post_func(
                request_data=request_data,
                request=mock_request
            )
            
            # Assertions
            assert isinstance(response, AnalyzeResponse)
            assert response.success is True
            assert response.analysis_id is not None
            assert response.result is not None
    
    def test_check_managers_raises_error_when_not_set(self, router):
        """Test that check_managers raises error when managers are not initialized."""
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            router.check_managers()
        
        assert exc_info.value.status_code == 503
        assert "Factories not initialized" in str(exc_info.value.detail)
    
    def test_check_managers_passes_when_set(self, router, mock_managers):
        """Test that check_managers passes when managers are properly set."""
        router.set_managers(**mock_managers)
        
        # Should not raise any exception
        router.check_managers()
    
    @pytest.mark.asyncio
    async def test_response_structure_excludes_metadata(self, router, mock_managers, mock_request):
        """Test that the response structure does not include metadata field."""
        # Setup
        router.set_managers(**mock_managers)
        
        # Mock result
        mock_result = create_test_analysis_result()
        
        # Mock LLM creation
        mock_llm = create_mock_llm_model()
        mock_managers['llm_factory_manager'].create_llm.return_value = mock_llm
        
        # Mock workflow
        with patch('workflows.talent_analysis.talent_analysis_workflow') as mock_workflow:
            mock_workflow.run_analysis = AsyncMock(return_value=mock_result)
            
            api_router = router.get_router()
            
            # Find the analyze_talent_get function
            analyze_func = None
            for route in api_router.routes:
                if hasattr(route, 'endpoint') and route.endpoint.__name__ == 'analyze_talent_get':
                    analyze_func = route.endpoint
                    break
            
            # Execute the endpoint
            response = await analyze_func(
                talent_id="test_talent_001",
                request=mock_request,
                analyzer_type="default",
                llm_provider="openai",
                llm_model="gpt-4o-mini",
                enrich_context=True
            )
            
            # Convert to dict to check structure
            response_dict = response.model_dump()
            
            # Verify metadata is not in response
            assert "metadata" not in response_dict
            
            # Verify expected fields are present
            assert "analysis_id" in response_dict
            assert "result" in response_dict
            assert "success" in response_dict
            assert "message" in response_dict
            assert "timestamp" in response_dict 
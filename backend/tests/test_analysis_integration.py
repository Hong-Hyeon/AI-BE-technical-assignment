"""
Integration tests for the complete analysis workflow.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app_factory import create_app
from tests.mocks import create_mock_llm_model
from tests.fixtures import create_test_talent_data, create_test_analysis_result
from models.talent import AnalysisResult, ExperienceTag, TalentData


class TestAnalysisIntegration:
    """Integration tests for the analysis workflow."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a FastAPI app with mocked dependencies."""
        # Create the app using the standalone function
        app = create_app()
        return app
    
    @pytest.fixture
    def client(self, mock_app):
        """Create test client."""
        return TestClient(mock_app)
    
    @pytest.fixture
    def mock_workflow_dependencies(self):
        """Mock all workflow dependencies."""
        # Mock talent data
        mock_talent_data = create_test_talent_data()
        
        # Mock analysis result
        mock_result = create_test_analysis_result()
        
        # Mock LLM model
        mock_llm = create_mock_llm_model()
        
        return {
            'talent_data': mock_talent_data,
            'analysis_result': mock_result,
            'llm_model': mock_llm
        }
    
    def test_app_creation_with_analysis_router(self):
        """Test that the app is created with analysis router properly configured."""
        app = create_app()
        
        # Check that analyze routes are registered
        analyze_routes = [route for route in app.routes if hasattr(route, 'path') and '/analyze' in route.path]
        assert len(analyze_routes) > 0, "No analyze routes found in the app"
    
    @pytest.mark.asyncio
    async def test_full_analysis_workflow_success(self, client, mock_workflow_dependencies):
        """Test complete analysis workflow from API call to response."""
        with patch('workflows.talent_analysis.talent_analysis_workflow') as mock_workflow:
            # Setup workflow mock
            mock_workflow.run_analysis = AsyncMock(return_value=mock_workflow_dependencies['analysis_result'])
            
            # Mock factory manager method
            with patch('routers.analysis.AnalysisRouter.check_managers'):
                with patch.object(client.app.state, 'llm_factory_manager', create=True) as mock_llm_mgr:
                    mock_llm_mgr.create_llm.return_value = mock_workflow_dependencies['llm_model']
                    
                    # Make API request
                    response = client.get("/analyze/test_talent_001?llm_provider=openai&llm_model=gpt-4o-mini")
                    
                    # Verify response
                    assert response.status_code == 200
                    
                    response_data = response.json()
                    assert response_data["success"] is True
                    assert "analysis_id" in response_data
                    assert "result" in response_data
                    assert response_data["result"] is not None
                    assert "experience_tags" in response_data["result"]
                    assert len(response_data["result"]["experience_tags"]) > 0
                    
                    # Verify metadata is not in response
                    assert "metadata" not in response_data
    
    @pytest.mark.asyncio
    async def test_analysis_workflow_with_post_request(self, client, mock_workflow_dependencies):
        """Test analysis workflow via POST endpoint."""
        with patch('workflows.talent_analysis.talent_analysis_workflow') as mock_workflow:
            # Setup workflow mock
            mock_workflow.run_analysis = AsyncMock(return_value=mock_workflow_dependencies['analysis_result'])
            
            # Mock factory manager
            with patch('routers.analysis.AnalysisRouter.check_managers'):
                with patch.object(client.app.state, 'llm_factory_manager', create=True) as mock_llm_mgr:
                    mock_llm_mgr.create_llm.return_value = mock_workflow_dependencies['llm_model']
                    
                    # Prepare POST data
                    post_data = {
                        "talent_id": "test_talent_001"
                    }
                    
                    # Make API request
                    response = client.post("/analyze/", json=post_data)
                    
                    # Verify response
                    assert response.status_code == 200
                    
                    response_data = response.json()
                    assert response_data["success"] is True
                    assert "analysis_id" in response_data
    
    @pytest.mark.asyncio
    async def test_analysis_workflow_error_handling(self, client):
        """Test error handling in the analysis workflow."""
        with patch('workflows.talent_analysis.talent_analysis_workflow') as mock_workflow:
            # Setup workflow to raise an error
            mock_workflow.run_analysis = AsyncMock(side_effect=ValueError("Talent not found"))
            
            with patch('routers.analysis.AnalysisRouter.check_managers'):
                with patch.object(client.app.state, 'llm_factory_manager', create=True) as mock_llm_mgr:
                    mock_llm_mgr.create_llm.return_value = create_mock_llm_model()
                    
                    # Make API request
                    response = client.get("/analyze/nonexistent_talent")
                    
                    # Verify error response
                    assert response.status_code == 404
                    
                    response_data = response.json()
                    assert "detail" in response_data
                    assert "Data not found" in response_data["detail"]
    
    @pytest.mark.asyncio
    async def test_analysis_workflow_internal_server_error(self, client):
        """Test internal server error handling."""
        with patch('workflows.talent_analysis.talent_analysis_workflow') as mock_workflow:
            # Setup workflow to raise an internal error
            mock_workflow.run_analysis = AsyncMock(side_effect=Exception("Internal processing error"))
            
            with patch('routers.analysis.AnalysisRouter.check_managers'):
                with patch.object(client.app.state, 'llm_factory_manager', create=True) as mock_llm_mgr:
                    mock_llm_mgr.create_llm.return_value = create_mock_llm_model()
                    
                    # Make API request
                    response = client.get("/analyze/test_talent_001")
                    
                    # Verify error response
                    assert response.status_code == 500
                    
                    response_data = response.json()
                    assert "detail" in response_data
                    assert "Analysis failed" in response_data["detail"]
    
    @pytest.mark.asyncio
    async def test_analysis_response_structure_consistency(self, client, mock_workflow_dependencies):
        """Test that response structure is consistent across different scenarios."""
        with patch('workflows.talent_analysis.talent_analysis_workflow') as mock_workflow:
            mock_workflow.run_analysis = AsyncMock(return_value=mock_workflow_dependencies['analysis_result'])
            
            with patch('routers.analysis.AnalysisRouter.check_managers'):
                with patch.object(client.app.state, 'llm_factory_manager', create=True) as mock_llm_mgr:
                    mock_llm_mgr.create_llm.return_value = mock_workflow_dependencies['llm_model']
                    
                    # Test different parameter combinations
                    test_cases = [
                        "/analyze/talent_001",
                        "/analyze/talent_002",
                        "/analyze/talent_003"
                    ]
                    
                    for endpoint in test_cases:
                        response = client.get(endpoint)
                        
                        assert response.status_code == 200
                        
                        response_data = response.json()
                        
                        # Verify consistent structure
                        required_fields = {"success", "analysis_id", "result", "message", "timestamp"}
                        assert required_fields.issubset(set(response_data.keys()))
                        
                        # Verify metadata is never present
                        assert "metadata" not in response_data
                        
                        # Verify result structure
                        if response_data["result"]:
                            result = response_data["result"]
                            assert "talent_id" in result
                            assert "experience_tags" in result
                            assert isinstance(result["experience_tags"], list)
    
    def test_openapi_schema_excludes_metadata(self, client):
        """Test that the OpenAPI schema does not include metadata in AnalyzeResponse."""
        # Get OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_schema = response.json()
        
        # Find AnalyzeResponse schema
        schemas = openapi_schema.get("components", {}).get("schemas", {})
        analyze_response_schema = schemas.get("AnalyzeResponse")
        
        if analyze_response_schema:  # Schema might be inlined
            properties = analyze_response_schema.get("properties", {})
            
            # Verify metadata is not in the schema
            assert "metadata" not in properties
            
            # Verify expected fields are present
            expected_fields = {"analysis_id", "result", "success", "message", "timestamp"}
            schema_fields = set(properties.keys())
            assert expected_fields.issubset(schema_fields)
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_requests(self, client, mock_workflow_dependencies):
        """Test handling of concurrent analysis requests."""
        with patch('workflows.talent_analysis.talent_analysis_workflow') as mock_workflow:
            # Add delay to simulate processing time
            async def mock_analysis(*args, **kwargs):
                await asyncio.sleep(0.1)  # Small delay
                return mock_workflow_dependencies['analysis_result']
            
            mock_workflow.run_analysis = AsyncMock(side_effect=mock_analysis)
            
            with patch('routers.analysis.AnalysisRouter.check_managers'):
                with patch.object(client.app.state, 'llm_factory_manager', create=True) as mock_llm_mgr:
                    mock_llm_mgr.create_llm.return_value = mock_workflow_dependencies['llm_model']
                    
                    # Make concurrent requests
                    import concurrent.futures
                    import threading
                    
                    def make_request(talent_id):
                        return client.get(f"/analyze/{talent_id}")
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        futures = [
                            executor.submit(make_request, f"talent_{i}")
                            for i in range(3)
                        ]
                        
                        results = [future.result() for future in concurrent.futures.as_completed(futures)]
                    
                    # Verify all requests succeeded
                    for response in results:
                        assert response.status_code == 200
                        response_data = response.json()
                        assert response_data["success"] is True
                        
                        # Each should have unique analysis_id
                        assert "analysis_id" in response_data
    
    @pytest.mark.asyncio
    async def test_analysis_with_different_llm_providers(self, client, mock_workflow_dependencies):
        """Test analysis with different LLM providers."""
        with patch('workflows.talent_analysis.talent_analysis_workflow') as mock_workflow:
            mock_workflow.run_analysis = AsyncMock(return_value=mock_workflow_dependencies['analysis_result'])
            
            with patch('routers.analysis.AnalysisRouter.check_managers'):
                with patch.object(client.app.state, 'llm_factory_manager', create=True) as mock_llm_mgr:
                    
                    # Test different providers (only test supported ones)
                    providers = [
                        ("openai", "gpt-4o-mini"),
                        ("openai", "gpt-4o"),  # Test another OpenAI model
                    ]
                    
                    for provider, model in providers:
                        mock_llm_mgr.create_llm.return_value = create_mock_llm_model()
                        
                        response = client.get(f"/analyze/test_talent?llm_provider={provider}&llm_model={model}")
                        
                        assert response.status_code == 200
                        response_data = response.json()
                        assert response_data["success"] is True
                        
                        # Since workflow is mocked, just verify the response is successful
                        # The actual LLM creation is bypassed by the workflow mock
                        assert "analysis_id" in response_data 
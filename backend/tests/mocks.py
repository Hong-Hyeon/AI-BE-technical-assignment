"""
Mock objects for testing.
"""

from unittest.mock import AsyncMock, MagicMock
from typing import Any, Dict
from langchain_core.messages import AIMessage


class MockLLMModel:
    """Mock LLM model for testing."""
    
    def __init__(self, response_content: str = "Test analysis response"):
        self.response_content = response_content
        self.model_name = "test-model"
        self.call_count = 0
    
    async def ainvoke(self, messages):
        """Mock async invoke method."""
        self.call_count += 1
        return AIMessage(content=self.response_content)
    
    def get_langchain_model(self):
        """Return self for compatibility."""
        return self


def create_mock_llm_model(response: str = "Mock analysis response") -> MockLLMModel:
    """Create a mock LLM model."""
    return MockLLMModel(response)


class MockDataManager:
    """Mock data manager for testing."""
    
    def __init__(self):
        self.talent_data = None
        self.company_data = {}
        self.news_data = {}
    
    def get_talent_data(self, talent_id: str):
        """Mock get talent data."""
        if self.talent_data:
            return self.talent_data
        raise ValueError(f"No talent data for {talent_id}")
    
    def get_company_data(self, company_name: str):
        """Mock get company data."""
        return self.company_data.get(company_name, {})
    
    def get_news_data(self, company_name: str):
        """Mock get news data."""
        return self.news_data.get(company_name, {})


class MockVectorSearchManager:
    """Mock vector search manager for testing."""
    
    def __init__(self):
        self.search_results = {
            "companies": [],
            "news": [],
            "search_queries": ["test query"]
        }
    
    def search_talent_context(self, talent_dict: Dict[str, Any]):
        """Mock search method."""
        return self.search_results
    
    def close(self):
        """Mock close method."""
        pass 
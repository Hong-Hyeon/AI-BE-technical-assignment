"""
Data Source Service for managing data operations.
"""

from typing import Dict, Any, Optional
from factories.data_source_factory import DefaultDataSourceFactory, DataSourceManager
from core.exceptions import BusinessLogicException
import logging

logger = logging.getLogger('data')


class DataSourceService:
    """Service for managing data source operations and caching."""
    
    def __init__(self, data_path: str = "./example_datas"):
        self.data_path = data_path
        self._data_manager = None
    
    def _get_data_manager(self) -> DataSourceManager:
        """Get or create data manager instance."""
        if self._data_manager is None:
            factory = DefaultDataSourceFactory(self.data_path)
            self._data_manager = DataSourceManager(factory)
        return self._data_manager
    
    async def get_talent_data(self, talent_id: str):
        """Get talent data by ID."""
        try:
            manager = self._get_data_manager()
            return manager.get_talent_data(talent_id)
        except Exception as e:
            raise BusinessLogicException(f"Failed to load talent data: {str(e)}")
    
    async def get_company_data(self, company_name: str):
        """Get company data by name."""
        try:
            manager = self._get_data_manager()
            return manager.get_company_data(company_name)
        except Exception as e:
            logger.warning(f"Company data not found for {company_name}: {e}")
            return None
    
    async def get_news_data(self, company_name: str):
        """Get news data by company name."""
        try:
            manager = self._get_data_manager()
            return manager.get_news_data(company_name)
        except Exception as e:
            logger.warning(f"News data not found for {company_name}: {e}")
            return None 
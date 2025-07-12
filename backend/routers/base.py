"""
Base router with common dependencies and utilities.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from factories.llm_factory import LLMFactoryManager
from factories.data_source_factory import DataSourceManager
from factories.experience_analyzer_factory import ExperienceAnalyzerManager
from factories.processor_factory import ProcessorManager


class BaseRouter:
    """Base router class with common dependencies."""
    
    def __init__(self):
        self.llm_factory_manager: Optional[LLMFactoryManager] = None
        self.data_source_manager: Optional[DataSourceManager] = None
        self.experience_analyzer_manager: Optional[ExperienceAnalyzerManager] = None
        self.processor_manager: Optional[ProcessorManager] = None
    
    def set_managers(
        self,
        llm_factory_manager: LLMFactoryManager,
        data_source_manager: DataSourceManager,
        experience_analyzer_manager: ExperienceAnalyzerManager,
        processor_manager: ProcessorManager
    ):
        """Set factory managers."""
        self.llm_factory_manager = llm_factory_manager
        self.data_source_manager = data_source_manager
        self.experience_analyzer_manager = experience_analyzer_manager
        self.processor_manager = processor_manager
    
    def check_managers(self):
        """Check if all managers are initialized."""
        if not all([
            self.llm_factory_manager,
            self.data_source_manager,
            self.experience_analyzer_manager,
            self.processor_manager
        ]):
            raise HTTPException(status_code=503, detail="Factories not initialized")
    
    def get_router(self) -> APIRouter:
        """Get the router instance. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement get_router") 
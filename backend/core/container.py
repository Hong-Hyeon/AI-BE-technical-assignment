"""
Dependency Injection Container for the AI BE Technical Assignment.

This module provides:
- Centralized dependency management
- Service lifecycle management
- Configuration injection
- Clean separation of concerns
"""

from typing import Dict, Any, Optional, TypeVar, Type, Callable
from functools import lru_cache
import logging

from config import get_settings, Settings
from services import (
    TalentAnalysisService,
    WorkflowService,
    DataSourceService,
    ValidationService
)
from factories.llm_factory import LLMFactory
from factories.prompt_factory import TalentAnalysisPromptFactory

logger = logging.getLogger('container')

T = TypeVar('T')


class ServiceContainer:
    """
    Dependency injection container for managing services and their dependencies.
    
    This container:
    - Manages service instances and lifecycles
    - Handles dependency injection
    - Provides service discovery
    - Supports singleton and transient services
    """
    
    def __init__(self):
        """Initialize the service container."""
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._initialized = False
    
    def register_singleton(self, service_type: Type[T], factory: Callable[[], T], name: Optional[str] = None):
        """
        Register a singleton service.
        
        Args:
            service_type: Service class type
            factory: Factory function to create the service
            name: Optional service name (defaults to class name)
        """
        service_name = name or service_type.__name__
        self._factories[service_name] = factory
        logger.debug(f"Registered singleton service: {service_name}")
    
    def register_transient(self, service_type: Type[T], factory: Callable[[], T], name: Optional[str] = None):
        """
        Register a transient service (new instance each time).
        
        Args:
            service_type: Service class type
            factory: Factory function to create the service
            name: Optional service name (defaults to class name)
        """
        service_name = name or service_type.__name__
        self._services[service_name] = factory
        logger.debug(f"Registered transient service: {service_name}")
    
    def register_instance(self, service_type: Type[T], instance: T, name: Optional[str] = None):
        """
        Register an existing service instance.
        
        Args:
            service_type: Service class type
            instance: Service instance
            name: Optional service name (defaults to class name)
        """
        service_name = name or service_type.__name__
        self._singletons[service_name] = instance
        logger.debug(f"Registered service instance: {service_name}")
    
    def get(self, service_type: Type[T], name: Optional[str] = None) -> T:
        """
        Get a service instance.
        
        Args:
            service_type: Service class type
            name: Optional service name (defaults to class name)
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service is not registered
        """
        service_name = name or service_type.__name__
        
        # Check if already instantiated singleton
        if service_name in self._singletons:
            return self._singletons[service_name]
        
        # Check if singleton factory exists
        if service_name in self._factories:
            instance = self._factories[service_name]()
            self._singletons[service_name] = instance
            logger.debug(f"Created singleton service: {service_name}")
            return instance
        
        # Check if transient factory exists
        if service_name in self._services:
            instance = self._services[service_name]()
            logger.debug(f"Created transient service: {service_name}")
            return instance
        
        raise ValueError(f"Service {service_name} is not registered")
    
    def get_all_services(self) -> Dict[str, str]:
        """
        Get information about all registered services.
        
        Returns:
            Dictionary mapping service names to their types
        """
        services = {}
        
        for name in self._singletons.keys():
            services[name] = "singleton (instantiated)"
        
        for name in self._factories.keys():
            if name not in self._singletons:
                services[name] = "singleton (factory)"
        
        for name in self._services.keys():
            services[name] = "transient"
        
        return services
    
    def clear_singletons(self):
        """Clear all singleton instances (for testing/reinitialization)."""
        self._singletons.clear()
        logger.debug("Cleared all singleton instances")
    
    def is_registered(self, service_type: Type[T], name: Optional[str] = None) -> bool:
        """
        Check if a service is registered.
        
        Args:
            service_type: Service class type
            name: Optional service name
            
        Returns:
            True if service is registered
        """
        service_name = name or service_type.__name__
        return (
            service_name in self._singletons or 
            service_name in self._factories or 
            service_name in self._services
        )


# Global container instance
_container = ServiceContainer()


def get_container() -> ServiceContainer:
    """Get the global service container."""
    if not _container._initialized:
        _setup_default_services()
        _container._initialized = True
    return _container


def _setup_default_services():
    """Setup default services in the container."""
    logger.info("Setting up default services in container...")
    
    # Register configuration
    _container.register_singleton(Settings, get_settings)
    
    # Register factories
    _container.register_singleton(LLMFactory, lambda: LLMFactory())
    _container.register_singleton(
        TalentAnalysisPromptFactory,
        lambda: TalentAnalysisPromptFactory()
    )
    
    # Register core services
    _container.register_singleton(
        ValidationService,
        lambda: ValidationService()
    )
    
    _container.register_singleton(
        DataSourceService,
        lambda: DataSourceService()
    )
    
    _container.register_singleton(
        WorkflowService,
        lambda: WorkflowService()
    )
    
    _container.register_singleton(
        TalentAnalysisService,
        lambda: TalentAnalysisService()
    )
    
    logger.info("Default services registered successfully")


# Convenience functions for common services
@lru_cache()
def get_talent_analysis_service() -> TalentAnalysisService:
    """Get talent analysis service instance."""
    return get_container().get(TalentAnalysisService)


@lru_cache()
def get_workflow_service() -> WorkflowService:
    """Get workflow service instance."""
    return get_container().get(WorkflowService)


@lru_cache()
def get_data_source_service() -> DataSourceService:
    """Get data source service instance."""
    return get_container().get(DataSourceService)


@lru_cache()
def get_validation_service() -> ValidationService:
    """Get validation service instance."""
    return get_container().get(ValidationService)


@lru_cache()
def get_llm_factory() -> LLMFactory:
    """Get LLM factory instance."""
    return get_container().get(LLMFactory)


# Context manager for testing
class ServiceContainerContext:
    """Context manager for service container testing."""
    
    def __init__(self):
        self.original_container = None
    
    def __enter__(self) -> ServiceContainer:
        """Enter context with clean container."""
        global _container
        self.original_container = _container
        _container = ServiceContainer()
        return _container
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original container."""
        global _container
        _container = self.original_container


def create_test_container() -> ServiceContainerContext:
    """Create a test container context."""
    return ServiceContainerContext() 
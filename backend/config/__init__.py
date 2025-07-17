"""
Configuration module for the AI BE Technical Assignment.

This module provides centralized configuration management with:
- Environment-specific settings
- Type-safe configuration
- Validation and defaults
"""

from .settings import (
    Settings,
    Environment,
    LogLevel,
    DatabaseSettings,
    LLMSettings,
    VectorSearchSettings,
    RedisSettings,
    SecuritySettings,
    get_settings,
    create_settings,
    get_development_settings,
    get_production_settings,
    get_testing_settings
)

from .logging_config import (
    setup_logging,
    WorkflowLogger,
    get_workflow_logger,
    get_api_logger,
    get_factory_logger
)

__all__ = [
    "Settings",
    "Environment", 
    "LogLevel",
    "DatabaseSettings",
    "LLMSettings", 
    "VectorSearchSettings",
    "RedisSettings",
    "SecuritySettings",
    "get_settings",
    "create_settings",
    "get_development_settings",
    "get_production_settings", 
    "get_testing_settings",
    "setup_logging",
    "WorkflowLogger",
    "get_workflow_logger",
    "get_api_logger",
    "get_factory_logger"
] 
"""
Configuration management system for the AI BE Technical Assignment.

This module provides:
- Environment-specific configuration (dev/staging/prod)
- Pydantic-based settings validation
- Centralized configuration access
- Type-safe configuration management
"""

import os
from enum import Enum
from typing import List, Optional, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Available log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(default="ai_be_assignment", description="Database name")
    username: str = Field(default="postgres", description="Database username")
    password: str = Field(default="", description="Database password")
    
    # Connection pool settings
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Maximum overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, description="Pool recycle time in seconds")
    
    # Query settings
    query_timeout: int = Field(default=30, description="Query timeout in seconds")
    
    @property
    def url(self) -> str:
        """Generate database URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_url(self) -> str:
        """Generate async database URL."""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    model_config = {
        "env_prefix": "DB_"
    }


class LLMSettings(BaseSettings):
    """LLM service configuration settings."""
    
    # OpenAI settings
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model name")
    openai_max_tokens: int = Field(default=1500, description="Maximum tokens for OpenAI")
    openai_temperature: float = Field(default=0.7, description="Temperature for OpenAI")
    openai_timeout: int = Field(default=30, description="OpenAI API timeout")
    
    # Rate limiting
    requests_per_minute: int = Field(default=60, description="Rate limit per minute")
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent requests")
    
    # Retry settings
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Retry delay in seconds")
    
    @field_validator('openai_temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Temperature must be between 0 and 1')
        return v
    
    model_config = {
        "env_prefix": "LLM_"
    }


class VectorSearchSettings(BaseSettings):
    """Vector search configuration settings."""
    
    # Elasticsearch settings
    elasticsearch_url: str = Field(default="http://localhost:9200", description="Elasticsearch URL")
    elasticsearch_index: str = Field(default="talent_profiles", description="Elasticsearch index name")
    elasticsearch_timeout: int = Field(default=30, description="Elasticsearch timeout")
    
    # Search settings
    max_search_results: int = Field(default=100, description="Maximum search results")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    
    # Embedding settings - Updated to match actual implementation
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    embedding_dimensions: int = Field(default=1536, description="Embedding dimensions")
    batch_size: int = Field(default=50, description="Embedding batch size for optimization")
    cache_enabled: bool = Field(default=True, description="Enable embedding cache")
    cache_ttl_hours: int = Field(default=24, description="Cache TTL in hours")
    
    # Performance settings
    max_concurrent_requests: int = Field(default=5, description="Max concurrent embedding requests")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    
    @field_validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Similarity threshold must be between 0 and 1')
        return v
    
    @field_validator('batch_size')
    def validate_batch_size(cls, v):
        if v < 1 or v > 100:
            raise ValueError('Batch size must be between 1 and 100')
        return v
    
    model_config = {
        "env_prefix": "VECTOR_"
    }


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: str = Field(default="", description="Redis password")
    database: int = Field(default=0, description="Redis database number")
    
    # Connection settings
    connection_timeout: int = Field(default=5, description="Connection timeout")
    socket_timeout: int = Field(default=5, description="Socket timeout")
    max_connections: int = Field(default=10, description="Maximum connections")
    
    # Cache settings
    default_ttl: int = Field(default=3600, description="Default TTL in seconds")
    
    @property
    def url(self) -> str:
        """Generate Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.database}"
    
    model_config = {
        "env_prefix": "REDIS_"
    }


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    # JWT settings
    secret_key: str = Field(default="", description="Secret key for JWT")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiration")
    
    # CORS settings
    allow_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    allow_credentials: bool = Field(default=True, description="Allow credentials")
    allow_methods: List[str] = Field(default=["*"], description="Allowed HTTP methods")
    allow_headers: List[str] = Field(default=["*"], description="Allowed headers")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per window")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    model_config = {
        "env_prefix": "SECURITY_"
    }


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application settings
    app_name: str = Field(default="AI BE Technical Assignment", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    debug: bool = Field(default=True, description="Debug mode")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")
    
    # Logging settings
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    log_file: bool = Field(default=True, description="Enable file logging")
    log_rotation: bool = Field(default=True, description="Enable log rotation")
    log_retention_days: int = Field(default=30, description="Log retention in days")
    
    # Performance settings
    request_timeout: int = Field(default=60, description="Request timeout in seconds")
    max_request_size: int = Field(default=16 * 1024 * 1024, description="Max request size in bytes")
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    
    # Feature flags
    enable_caching: bool = Field(default=True, description="Enable caching")
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    enable_tracing: bool = Field(default=True, description="Enable request tracing")
    
    # Sub-configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_search: VectorSearchSettings = Field(default_factory=VectorSearchSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    
    @field_validator('environment', mode='before')
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @field_validator('log_level', mode='before')
    def validate_log_level(cls, v):
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING
    
    def get_database_url(self, async_db: bool = False) -> str:
        """Get database URL."""
        return self.database.async_url if async_db else self.database.url
    
    def get_redis_url(self) -> str:
        """Get Redis URL."""
        return self.redis.url
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"  # Allow extra environment variables to be ignored
    }
    
    @classmethod
    def customise_sources(cls, init_settings, env_settings, file_secret_settings):
        """Customize settings sources priority."""
        return (
            init_settings,
            env_settings,
            file_secret_settings,
        )


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()


# Environment-specific configurations
def get_development_settings() -> Settings:
    """Get development-specific settings."""
    settings = get_settings()
    settings.debug = True
    settings.log_level = LogLevel.DEBUG
    settings.database.pool_size = 2
    settings.llm.max_retries = 1
    return settings


def get_production_settings() -> Settings:
    """Get production-specific settings."""
    settings = get_settings()
    settings.debug = False
    settings.log_level = LogLevel.INFO
    settings.database.pool_size = 20
    settings.llm.max_retries = 5
    settings.security.allow_origins = []  # Configure specific origins
    return settings


def get_testing_settings() -> Settings:
    """Get testing-specific settings."""
    settings = get_settings()
    settings.debug = True
    settings.log_level = LogLevel.DEBUG
    settings.database.database = "test_" + settings.database.database
    settings.enable_caching = False
    return settings


# Configuration factory
def create_settings(environment: Optional[str] = None) -> Settings:
    """Create settings based on environment."""
    if environment:
        os.environ["ENVIRONMENT"] = environment
    
    env = Environment(os.getenv("ENVIRONMENT", Environment.DEVELOPMENT).lower())
    
    if env == Environment.DEVELOPMENT:
        return get_development_settings()
    elif env == Environment.PRODUCTION:
        return get_production_settings()
    elif env == Environment.TESTING:
        return get_testing_settings()
    else:
        return get_settings() 
# Configuration Management

This directory contains the configuration management system for the AI BE Technical Assignment.

## Overview

The configuration system provides:
- Environment-specific settings (development, staging, production, testing)
- Type-safe configuration with Pydantic validation
- Centralized configuration access
- Environment variable support with validation

## Files

- `settings.py` - Main configuration classes and settings management
- `logging_config.py` - Logging configuration and setup
- `__init__.py` - Configuration module exports

## Environment Variables

### Application Settings
```bash
ENVIRONMENT=development          # Application environment
APP_NAME="AI BE Technical Assignment"
APP_VERSION="1.0.0"
DEBUG=true
HOST=0.0.0.0
PORT=8000
WORKERS=1
```

### Database Settings
```bash
DB_HOST=localhost
DB_PORT=5432
DB_DATABASE=ai_be_assignment
DB_USERNAME=postgres
DB_PASSWORD=
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
DB_QUERY_TIMEOUT=30
```

### LLM Settings (OpenAI)
```bash
LLM_OPENAI_API_KEY=your_openai_api_key_here
LLM_OPENAI_MODEL=gpt-3.5-turbo
LLM_OPENAI_MAX_TOKENS=1500
LLM_OPENAI_TEMPERATURE=0.7
LLM_OPENAI_TIMEOUT=30
LLM_REQUESTS_PER_MINUTE=60
LLM_MAX_CONCURRENT_REQUESTS=10
LLM_MAX_RETRIES=3
LLM_RETRY_DELAY=1.0
```

### Vector Search Settings (Elasticsearch)
```bash
VECTOR_ELASTICSEARCH_URL=http://localhost:9200
VECTOR_ELASTICSEARCH_INDEX=talent_profiles
VECTOR_ELASTICSEARCH_TIMEOUT=30
VECTOR_MAX_SEARCH_RESULTS=100
VECTOR_SIMILARITY_THRESHOLD=0.7
VECTOR_EMBEDDING_MODEL=text-embedding-ada-002
VECTOR_EMBEDDING_DIMENSIONS=1536
```

### Redis Settings
```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DATABASE=0
REDIS_CONNECTION_TIMEOUT=5
REDIS_SOCKET_TIMEOUT=5
REDIS_MAX_CONNECTIONS=10
REDIS_DEFAULT_TTL=3600
```

### Security Settings
```bash
SECURITY_SECRET_KEY=your_secret_key_here_change_in_production
SECURITY_ALGORITHM=HS256
SECURITY_ACCESS_TOKEN_EXPIRE_MINUTES=30
SECURITY_REFRESH_TOKEN_EXPIRE_DAYS=7
SECURITY_RATE_LIMIT_REQUESTS=100
SECURITY_RATE_LIMIT_WINDOW=60
```

### Logging Settings
```bash
LOG_LEVEL=INFO
LOG_FILE=true
LOG_ROTATION=true
LOG_RETENTION_DAYS=30
```

### Feature Flags
```bash
ENABLE_CACHING=true
ENABLE_RATE_LIMITING=true
ENABLE_TRACING=true
ENABLE_METRICS=true
METRICS_PORT=9090
```

## Usage

### Basic Usage

```python
from config import get_settings

settings = get_settings()
print(f"Database URL: {settings.get_database_url()}")
print(f"Environment: {settings.environment}")
```

### Environment-Specific Settings

```python
from config import (
    create_settings,
    get_development_settings,
    get_production_settings,
    get_testing_settings
)

# Auto-detect environment
settings = create_settings()

# Specific environment
dev_settings = get_development_settings()
prod_settings = get_production_settings()
test_settings = get_testing_settings()
```

### Configuration in FastAPI

```python
from fastapi import FastAPI, Depends
from config import get_settings, Settings

app = FastAPI()

@app.get("/config")
def get_config(settings: Settings = Depends(get_settings)):
    return {
        "app_name": settings.app_name,
        "environment": settings.environment,
        "debug": settings.debug
    }
```

## Environment Files

Create a `.env` file in the project root with your environment-specific values:

```bash
# .env
ENVIRONMENT=development
LLM_OPENAI_API_KEY=sk-your-actual-key-here
DB_PASSWORD=your_db_password
SECURITY_SECRET_KEY=your-secret-key-here
```

## Environment-Specific Behavior

### Development
- Debug mode enabled
- Verbose logging (DEBUG level)
- Smaller connection pools
- Fewer retry attempts

### Production
- Debug mode disabled
- Standard logging (INFO level)
- Larger connection pools
- More retry attempts
- CORS origins should be configured

### Testing
- Debug mode enabled
- Test database prefix
- Caching disabled
- Verbose logging

## Validation

All configuration values are validated using Pydantic:
- Type checking
- Range validation (e.g., temperature between 0-1)
- Required field validation
- Custom validators for complex rules

## Caching

Settings are cached using `@lru_cache()` for performance. Clear the cache if you need to reload settings:

```python
from config import get_settings

get_settings.cache_clear()
```

## Integration with Other Components

The configuration system integrates with:
- Exception handling (error categories and codes)
- Logging system (log levels and configuration)
- Database connections (connection strings and pool settings)
- LLM services (API keys and parameters)
- Vector search (Elasticsearch configuration)
- Caching (Redis configuration)
- Security (JWT and CORS settings) 
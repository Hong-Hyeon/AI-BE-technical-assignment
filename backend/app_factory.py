"""
Application factory for managing FastAPI app creation and factory initialization.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

# Core infrastructure imports
from core.middleware import ExceptionHandlingMiddleware
from core.container import get_container
from config import setup_logging, get_settings

# Factory imports
from factories.llm_factory import LLMFactoryManager
from factories.data_source_factory import DefaultDataSourceFactory, DataSourceManager
from factories.experience_analyzer_factory import DefaultExperienceAnalyzerFactory, ExperienceAnalyzerManager
from factories.processor_factory import DefaultProcessorFactory, ProcessorManager

# Router imports
from routers.root import RootRouter
from routers.health import HealthRouter
from routers.analysis import AnalysisRouter
from routers.vector_search import VectorSearchRouter


class AppFactory:
    """Factory for creating and configuring FastAPI application with modern architecture."""
    
    def __init__(self):
        # Legacy factory managers (for backward compatibility)
        self.llm_factory_manager: LLMFactoryManager = None
        self.data_source_manager: DataSourceManager = None
        self.experience_analyzer_manager: ExperienceAnalyzerManager = None
        self.processor_manager: ProcessorManager = None
        
        # Modern infrastructure
        self.container = None
        self.settings = None
        self.routers = []
        self.logger = None
    
    def initialize_logging(self):
        """Initialize logging system."""
        # Setup logging configuration
        self.settings = get_settings()
        setup_logging(self.settings.log_level)
        
        # Get logger
        from config.logging_config import get_factory_logger
        self.logger = get_factory_logger()
    
    def initialize_container(self):
        """Initialize dependency injection container."""
        self.logger.info("🔧 Initializing dependency injection container...")
        self.container = get_container()
        
        # Log registered services
        services = self.container.get_all_services()
        self.logger.info(f"✅ Container initialized with {len(services)} services")
        for service_name, service_type in services.items():
            self.logger.debug(f"  - {service_name}: {service_type}")
    
    def initialize_factories(self):
        """Initialize all factory managers (legacy support)."""
        if not self.logger:
            self.initialize_logging()
        
        self.logger.info("🚀 Starting SearchRight AI API...")
        self.logger.info(f"Environment: {self.settings.environment}")
        self.logger.info(f"Debug mode: {self.settings.debug}")
        self.logger.info(f"OpenAI API Key: {os.getenv('OPENAI_API_KEY', 'Not set')[:10]}...")
        self.logger.info(f"Current working directory: {os.getcwd()}")
        
        try:
            # Initialize modern container first
            self.initialize_container()
            
            # Initialize legacy LLM Factory Manager (for backward compatibility)
            self.logger.info("Initializing LLM Factory Manager...")
            self.llm_factory_manager = LLMFactoryManager()
            self.logger.info("✅ LLM Factory Manager initialized")
            
            # Initialize Data Source Manager
            self.logger.info("Initializing Data Source Manager...")
            data_factory = DefaultDataSourceFactory("./example_datas")
            self.data_source_manager = DataSourceManager(data_factory)
            self.logger.info("✅ Data Source Manager initialized")
            
            # Initialize Experience Analyzer Manager
            self.logger.info("Initializing Experience Analyzer Manager...")
            # Create default LLM model for the analyzer factory
            default_llm = self.llm_factory_manager.create_llm("openai", "gpt-3.5-turbo")
            analyzer_factory = DefaultExperienceAnalyzerFactory(default_llm)
            self.experience_analyzer_manager = ExperienceAnalyzerManager(analyzer_factory)
            self.logger.info("✅ Experience Analyzer Manager initialized")
            
            # Initialize Processor Manager
            self.logger.info("Initializing Processor Manager...")
            # Use the same LLM model for processor factory
            processor_factory = DefaultProcessorFactory(default_llm)
            self.processor_manager = ProcessorManager(processor_factory)
            self.logger.info("✅ Processor Manager initialized")
            
            self.logger.info("🎉 All factory managers initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize factories: {str(e)}")
            raise
    
    def initialize_routers(self):
        """Initialize all routers with dependency injection."""
        self.logger.info("🛠️ Initializing routers...")
        
        try:
            # Initialize routers
            root_router = RootRouter()
            health_router = HealthRouter()
            analysis_router = AnalysisRouter()
            vector_search_router = VectorSearchRouter()
            
            # Set managers for routers that inherit from BaseRouter
            if hasattr(analysis_router, 'set_managers'):
                analysis_router.set_managers(
                    self.llm_factory_manager,
                    self.data_source_manager,
                    self.experience_analyzer_manager,
                    self.processor_manager
                )
                
            if hasattr(vector_search_router, 'set_managers'):
                vector_search_router.set_managers(
                    self.llm_factory_manager,
                    self.data_source_manager,
                    self.experience_analyzer_manager,
                    self.processor_manager
                )
            
            self.routers = [
                root_router,
                health_router,
                analysis_router,
                vector_search_router
            ]
            
            self.logger.info(f"✅ {len(self.routers)} routers initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize routers: {str(e)}")
            raise
    
    def create_middleware(self, app: FastAPI):
        """Add middleware to the FastAPI application."""
        self.logger.info("🔧 Setting up middleware...")
        
        # Add exception handling middleware (first to catch all exceptions)
        app.add_middleware(ExceptionHandlingMiddleware)
        self.logger.info("✅ Exception handling middleware added")
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.security.allow_origins,
            allow_credentials=self.settings.security.allow_credentials,
            allow_methods=self.settings.security.allow_methods,
            allow_headers=self.settings.security.allow_headers,
        )
        self.logger.info("✅ CORS middleware added")
    
    def register_routes(self, app: FastAPI):
        """Register all routes with the FastAPI application."""
        self.logger.info("🛣️ Registering routes...")
        
        for router in self.routers:
            if hasattr(router, 'get_router'):
                router_instance = router.get_router()
                app.include_router(router_instance)
                self.logger.info(f"✅ {router.__class__.__name__} routes registered")
        
        self.logger.info("🎯 All routes registered successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    app_factory = AppFactory()
    app_factory.initialize_factories()
    app.state.app_factory = app_factory
    
    # Log startup completion
    logger = app_factory.logger
    logger.info("🚀 Application startup completed!")
    logger.info(f"📊 Container services: {len(app_factory.container.get_all_services())}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Application shutdown initiated...")
    
    # Clean up container
    if hasattr(app_factory, 'container'):
        app_factory.container.clear_singletons()
        logger.info("✅ Container cleaned up")
    
    logger.info("👋 Application shutdown completed!")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application with modern architecture.
    
    Returns:
        Configured FastAPI application
    """
    # Load environment variables
    load_dotenv()
    
    # Create temporary factory to get settings
    temp_factory = AppFactory()
    temp_factory.initialize_logging()
    settings = temp_factory.settings
    
    # Create FastAPI app with enhanced configuration
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered talent analysis system with LangGraph workflows",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        debug=settings.debug
    )
    
    # Create app factory and initialize components
    app_factory = AppFactory()
    app_factory.initialize_factories()
    app_factory.initialize_routers()
    
    # Add middleware
    app_factory.create_middleware(app)
    
    # Register routes
    app_factory.register_routes(app)
    
    # Store factory in app state
    app.state.app_factory = app_factory
    
    # Log application creation
    logger = app_factory.logger
    logger.info("🎉 FastAPI application created successfully!")
    logger.info(f"📋 App configuration:")
    logger.info(f"  - Name: {settings.app_name}")
    logger.info(f"  - Version: {settings.app_version}")
    logger.info(f"  - Environment: {settings.environment}")
    logger.info(f"  - Debug: {settings.debug}")
    logger.info(f"  - Routers: {len(app_factory.routers)}")
    logger.info(f"  - Services: {len(app_factory.container.get_all_services())}")
    
    return app


# Create the FastAPI application instance 
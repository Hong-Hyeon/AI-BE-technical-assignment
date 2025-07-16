"""
Application factory for managing FastAPI app creation and factory initialization.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

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
    """Factory for creating and configuring FastAPI application."""
    
    def __init__(self):
        self.llm_factory_manager: LLMFactoryManager = None
        self.data_source_manager: DataSourceManager = None
        self.experience_analyzer_manager: ExperienceAnalyzerManager = None
        self.processor_manager: ProcessorManager = None
        self.routers = []
    
    def initialize_factories(self):
        """Initialize all factory managers."""
        print("🚀 Starting SearchRight AI API...")
        print(f"OpenAI API Key: {os.getenv('OPENAI_API_KEY', 'Not set')[:10]}...")
        print(f"Current working directory: {os.getcwd()}")
        
        try:
            # Initialize LLM Factory Manager
            self.llm_factory_manager = LLMFactoryManager()
            print("✅ LLM Factory Manager initialized")
            
            # Initialize Data Source Manager
            data_source_factory = DefaultDataSourceFactory("./example_datas")
            self.data_source_manager = DataSourceManager(data_source_factory)
            print("✅ Data Source Manager initialized")
            
            # Initialize Processor Manager
            default_llm = self.llm_factory_manager.create_llm("openai", "gpt-4o-mini")
            processor_factory = DefaultProcessorFactory(default_llm)
            self.processor_manager = ProcessorManager(processor_factory)
            print("✅ Processor Manager initialized")
            
            # Initialize Experience Analyzer Manager
            analyzer_factory = DefaultExperienceAnalyzerFactory(default_llm)
            self.experience_analyzer_manager = ExperienceAnalyzerManager(analyzer_factory)
            print("✅ Experience Analyzer Manager initialized")
            
            print("🎉 All factories initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize factories: {e}")
            raise
    
    def create_routers(self):
        """Create and configure all routers."""
        # Create router instances
        root_router = RootRouter()
        health_router = HealthRouter()
        analysis_router = AnalysisRouter()
        vector_search_router = VectorSearchRouter()
        
        # Set managers for all routers
        for router in [root_router, health_router, analysis_router, vector_search_router]:
            router.set_managers(
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
    
    def create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Application lifespan management."""
            yield
            print("🔄 Shutting down SearchRight AI API...")
        
        app = FastAPI(
            title="SearchRight AI API",
            description="SearchRight Backend 과제",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize factories and create routers
        self.initialize_factories()
        self.create_routers()
        
        # Include all routers
        for router in self.routers:
            app.include_router(router.get_router())
        
        return app


def create_app() -> FastAPI:
    """Factory function to create FastAPI application."""
    load_dotenv()
    factory = AppFactory()
    return factory.create_app() 
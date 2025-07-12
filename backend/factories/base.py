"""
Base factory interfaces for the talent analysis system.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from models.talent import TalentData, AnalysisResult


class DataSourceFactory(ABC):
    """Abstract factory for data source handling."""
    
    @abstractmethod
    def create_data_loader(self, source_type: str) -> 'DataLoader':
        """Create a data loader for the specified source type."""
        pass
    
    @abstractmethod
    def get_supported_sources(self) -> List[str]:
        """Get list of supported data source types."""
        pass


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(self, identifier: str) -> Dict[str, Any]:
        """Load data by identifier."""
        pass


class ProcessorFactory(ABC):
    """Abstract factory for data processing."""
    
    @abstractmethod
    def create_processor(self, processor_type: str) -> 'DataProcessor':
        """Create a data processor for the specified type."""
        pass
    
    @abstractmethod
    def get_supported_processors(self) -> List[str]:
        """Get list of supported processor types."""
        pass


class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return processed result."""
        pass


class LLMFactory(ABC):
    """Abstract factory for LLM models."""
    
    @abstractmethod
    def create_llm(self, model_type: str, **kwargs) -> 'LLMModel':
        """Create an LLM model instance."""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Get list of supported model types."""
        pass


class LLMModel(ABC):
    """Abstract base class for LLM models."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from the model."""
        pass
    
    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for the input text."""
        pass


class ExperienceAnalyzerFactory(ABC):
    """Abstract factory for experience analysis."""
    
    @abstractmethod
    def create_analyzer(self, analyzer_type: str) -> 'ExperienceAnalyzer':
        """Create an experience analyzer."""
        pass
    
    @abstractmethod
    def get_supported_analyzers(self) -> List[str]:
        """Get list of supported analyzer types."""
        pass


class ExperienceAnalyzer(ABC):
    """Abstract base class for experience analyzers."""
    
    @abstractmethod
    def analyze(self, talent_data: TalentData, context: Dict[str, Any]) -> AnalysisResult:
        """Analyze talent data and return experience tags."""
        pass 
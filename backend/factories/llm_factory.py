"""
OpenAI-based LLM factory implementation.
"""
import openai
from typing import List, Dict, Any, Optional
from factories.base import LLMFactory, LLMModel
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


class LangChainOpenAIModel(LLMModel):
    """LangChain OpenAI model wrapper."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.chat_model = ChatOpenAI(
            model=model_name,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=0.1
        )
        self.embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using LangChain ChatOpenAI."""
        try:
            from langchain_core.messages import HumanMessage
            response = self.chat_model.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using LangChain OpenAIEmbeddings."""
        try:
            return self.embeddings_model.embed_query(text)
        except Exception as e:
            raise Exception(f"OpenAI embeddings error: {str(e)}")
    
    def get_langchain_model(self):
        """Get the underlying LangChain ChatOpenAI model for direct use."""
        return self.chat_model


class OpenAIFactory(LLMFactory):
    """Factory for creating OpenAI LLM models."""
    
    SUPPORTED_MODELS = {
        "gpt-4": "gpt-4",
        "gpt-4-turbo": "gpt-4-turbo-preview",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini"
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
    
    def create_llm(self, model_type: str, **kwargs) -> LangChainOpenAIModel:
        """Create OpenAI LLM model instance."""
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_name = self.SUPPORTED_MODELS[model_type]
        return LangChainOpenAIModel(model_name, self.api_key)
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported OpenAI models."""
        return list(self.SUPPORTED_MODELS.keys())


class LLMFactoryManager:
    """Manager for multiple LLM factories."""
    
    def __init__(self):
        self.factories: Dict[str, LLMFactory] = {}
        self.register_factory("openai", OpenAIFactory(
            api_key=os.getenv("OPENAI_API_KEY")
        ))
    
    def register_factory(self, name: str, factory: LLMFactory):
        """Register a new LLM factory."""
        self.factories[name] = factory
    
    def create_llm(self, provider: str, model_type: str, **kwargs) -> LLMModel:
        """Create LLM model from specified provider."""
        if provider not in self.factories:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return self.factories[provider].create_llm(model_type, **kwargs)
    
    def get_supported_providers(self) -> List[str]:
        """Get list of supported providers."""
        return list(self.factories.keys())
    
    def get_supported_models(self, provider: str) -> List[str]:
        """Get supported models for a provider."""
        if provider not in self.factories:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return self.factories[provider].get_supported_models() 
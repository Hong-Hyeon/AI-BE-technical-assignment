"""
OpenAI-based LLM factory implementation.
"""
import openai
from typing import List, Dict, Any, Optional
from factories.base import LLMFactory, LLMModel
import os
from dotenv import load_dotenv

load_dotenv()


class OpenAIModel(LLMModel):
    """OpenAI LLM model implementation."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000),
                top_p=kwargs.get("top_p", 1.0),
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"OpenAI embeddings error: {str(e)}")


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
    
    def create_llm(self, model_type: str, **kwargs) -> OpenAIModel:
        """Create OpenAI LLM model instance."""
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_name = self.SUPPORTED_MODELS[model_type]
        return OpenAIModel(model_name, self.api_key)
    
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
"""
Processor factory for vector search and data processing.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from factories.base import ProcessorFactory, DataProcessor
from factories.llm_factory import LLMModel
import json


class VectorSearchProcessor(DataProcessor):
    """Vector search processor for finding similar content."""
    
    def __init__(self, llm_model: LLMModel):
        self.llm_model = llm_model
        self.embedding_cache: Dict[str, List[float]] = {}
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data using vector search."""
        query = data.get("query", "")
        documents = data.get("documents", [])
        
        if not query or not documents:
            return {"similar_documents": [], "query_embedding": None}
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for doc in documents:
            doc_embedding = self._get_embedding(doc.get("content", ""))
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append({
                "document": doc,
                "similarity": similarity
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "similar_documents": similarities[:5],  # Top 5
            "query_embedding": query_embedding
        }
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with caching."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = self.llm_model.get_embeddings(text)
        self.embedding_cache[text] = embedding
        return embedding
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class ContextEnrichmentProcessor(DataProcessor):
    """Processor for enriching talent data with context."""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich talent data with additional context."""
        talent_data = data.get("talent_data", {})
        company_data = data.get("company_data", {})
        news_data = data.get("news_data", {})
        
        enriched_data = {
            "talent": talent_data,
            "context": {
                "companies": self._extract_company_context(company_data),
                "news": self._extract_news_context(news_data),
                "market_trends": self._extract_market_trends(news_data)
            }
        }
        
        return enriched_data
    
    def _extract_company_context(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant company context."""
        context = {}
        
        for company_name, data in company_data.items():
            if isinstance(data, dict):
                context[company_name] = {
                    "employee_count": data.get("employee_count", "정보 없음"),
                    "investment_rounds": data.get("investment_rounds", []),
                    "major_events": data.get("major_events", [])
                }
        
        return context
    
    def _extract_news_context(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant news context."""
        context = {}
        
        for company_name, data in news_data.items():
            if isinstance(data, dict) and "news_data" in data:
                recent_news = data["news_data"][:5]  # Recent 5 news
                context[company_name] = {
                    "recent_news": recent_news,
                    "news_count": data.get("news_count", 0)
                }
        
        return context
    
    def _extract_market_trends(self, news_data: Dict[str, Any]) -> List[str]:
        """Extract market trends from news data."""
        trends = []
        
        # Simple trend extraction based on news keywords
        keywords = ["투자", "IPO", "M&A", "성장", "확장", "기술혁신"]
        
        for company_data in news_data.values():
            if isinstance(company_data, dict) and "news_data" in company_data:
                for news in company_data["news_data"]:
                    if isinstance(news, dict) and "title" in news:
                        title = news["title"]
                        for keyword in keywords:
                            if keyword in title:
                                trends.append(f"{keyword}: {title}")
        
        return trends[:10]  # Top 10 trends


class DefaultProcessorFactory(ProcessorFactory):
    """Default implementation of processor factory."""
    
    SUPPORTED_PROCESSORS = ["vector_search", "context_enrichment"]
    
    def __init__(self, llm_model: LLMModel):
        self.llm_model = llm_model
        self.processors = {
            "vector_search": VectorSearchProcessor(llm_model),
            "context_enrichment": ContextEnrichmentProcessor()
        }
    
    def create_processor(self, processor_type: str) -> DataProcessor:
        """Create a data processor for the specified type."""
        if processor_type not in self.SUPPORTED_PROCESSORS:
            raise ValueError(f"Unsupported processor type: {processor_type}")
        
        return self.processors[processor_type]
    
    def get_supported_processors(self) -> List[str]:
        """Get list of supported processor types."""
        return self.SUPPORTED_PROCESSORS.copy()


class ProcessorManager:
    """Manager for data processors."""
    
    def __init__(self, factory: ProcessorFactory):
        self.factory = factory
        self.processors: Dict[str, DataProcessor] = {}
    
    def get_processor(self, processor_type: str) -> DataProcessor:
        """Get or create a processor instance."""
        if processor_type not in self.processors:
            self.processors[processor_type] = self.factory.create_processor(processor_type)
        
        return self.processors[processor_type]
    
    def process_data(self, processor_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data using specified processor."""
        processor = self.get_processor(processor_type)
        return processor.process(data) 
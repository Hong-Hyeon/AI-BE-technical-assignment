"""
Batch embedding service for efficient embedding generation.

This service optimizes embedding generation by:
- Batching multiple texts into single API calls
- Intelligent caching with TTL
- Cost tracking and budget management
- Async processing for better performance
"""
import asyncio
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from langchain_openai import OpenAIEmbeddings
from config.logging_config import get_api_logger


@dataclass
class EmbeddingRequest:
    """Single embedding request."""
    id: str
    text: str
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Embedding result with metadata."""
    id: str
    text: str
    embedding: List[float]
    model: str
    dimensions: int
    tokens_used: int
    created_at: datetime
    cache_hit: bool = False


@dataclass
class EmbeddingBatch:
    """Batch of embedding requests."""
    id: str
    requests: List[EmbeddingRequest]
    created_at: datetime
    priority: int = 1
    
    @property
    def size(self) -> int:
        return len(self.requests)
    
    @property
    def total_text_length(self) -> int:
        return sum(len(req.text) for req in self.requests)


class EmbeddingCache:
    """Smart cache for embeddings with TTL and size limits."""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.cache: Dict[str, Tuple[List[float], datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache if available and not expired."""
        key = self._generate_key(text, model)
        
        if key in self.cache:
            embedding, created_at = self.cache[key]
            
            # Check if expired
            if datetime.now() - created_at < self.ttl:
                self.hit_count += 1
                return embedding
            else:
                # Remove expired entry
                del self.cache[key]
        
        self.miss_count += 1
        return None
    
    def set(self, text: str, model: str, embedding: List[float]):
        """Store embedding in cache."""
        # Check size limit
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        key = self._generate_key(text, model)
        self.cache[key] = (embedding, datetime.now())
    
    def _evict_oldest(self):
        """Remove oldest cache entries."""
        if not self.cache:
            return
        
        # Remove 20% of oldest entries
        entries = list(self.cache.items())
        entries.sort(key=lambda x: x[1][1])  # Sort by creation time
        
        remove_count = max(1, len(entries) // 5)
        for key, _ in entries[:remove_count]:
            del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "ttl_hours": self.ttl.total_seconds() / 3600
        }
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0


class BatchEmbeddingService:
    """Service for efficient batch embedding generation."""
    
    def __init__(
        self, 
        model: str = "text-embedding-3-small",
        max_batch_size: int = 100,  # OpenAI's max batch size
        cache_enabled: bool = True
    ):
        self.logger = get_api_logger()
        self.model = model
        self.max_batch_size = max_batch_size
        
        # Initialize OpenAI embeddings
        self.embeddings_model = OpenAIEmbeddings(model=model)
        
        # Initialize cache
        self.cache = EmbeddingCache() if cache_enabled else None
        
        # Metrics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.batch_count = 0
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: 1 token ≈ 4 characters for English/Korean
        return max(1, len(text) // 4)
    
    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost for embedding generation."""
        # text-embedding-3-small: $0.00002 per 1K tokens
        return (tokens / 1000) * 0.00002
    
    async def get_embedding_single(
        self, 
        text: str, 
        use_cache: bool = True
    ) -> EmbeddingResult:
        """Get embedding for a single text."""
        # Check cache first
        if use_cache and self.cache:
            cached_embedding = self.cache.get(text, self.model)
            if cached_embedding:
                self.logger.debug(f"Cache hit for text: {text[:50]}...")
                return EmbeddingResult(
                    id="single",
                    text=text,
                    embedding=cached_embedding,
                    model=self.model,
                    dimensions=len(cached_embedding),
                    tokens_used=0,  # No API call made
                    created_at=datetime.now(),
                    cache_hit=True
                )
        
        # Generate embedding
        start_time = time.time()
        try:
            embedding = await asyncio.to_thread(
                self.embeddings_model.embed_query, text
            )
            
            # Update metrics
            tokens_used = self.estimate_tokens(text)
            self.total_requests += 1
            self.total_tokens += tokens_used
            self.total_cost += self.estimate_cost(tokens_used)
            
            # Cache result
            if self.cache:
                self.cache.set(text, self.model, embedding)
            
            processing_time = time.time() - start_time
            self.logger.debug(
                f"Generated embedding for text (len={len(text)}) | "
                f"Time: {processing_time:.3f}s | Tokens: {tokens_used}"
            )
            
            return EmbeddingResult(
                id="single",
                text=text,
                embedding=embedding,
                model=self.model,
                dimensions=len(embedding),
                tokens_used=tokens_used,
                created_at=datetime.now(),
                cache_hit=False
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    async def get_embeddings_batch(
        self, 
        requests: List[EmbeddingRequest],
        use_cache: bool = True
    ) -> List[EmbeddingResult]:
        """Get embeddings for multiple texts in batch."""
        if not requests:
            return []
        
        self.logger.info(f"🔄 Processing batch of {len(requests)} embedding requests")
        
        # Separate cached and non-cached requests
        cached_results = []
        uncached_requests = []
        
        if use_cache and self.cache:
            for req in requests:
                cached_embedding = self.cache.get(req.text, self.model)
                if cached_embedding:
                    cached_results.append(EmbeddingResult(
                        id=req.id,
                        text=req.text,
                        embedding=cached_embedding,
                        model=self.model,
                        dimensions=len(cached_embedding),
                        tokens_used=0,
                        created_at=datetime.now(),
                        cache_hit=True
                    ))
                else:
                    uncached_requests.append(req)
        else:
            uncached_requests = requests
        
        # Process uncached requests in batches
        batch_results = []
        if uncached_requests:
            batch_results = await self._process_uncached_batch(uncached_requests)
        
        # Combine results
        all_results = cached_results + batch_results
        
        # Sort by original request order
        request_order = {req.id: i for i, req in enumerate(requests)}
        all_results.sort(key=lambda x: request_order.get(x.id, 999))
        
        self.logger.info(
            f"✅ Batch processing complete | "
            f"Total: {len(all_results)} | Cached: {len(cached_results)} | "
            f"Generated: {len(batch_results)}"
        )
        
        return all_results
    
    async def _process_uncached_batch(
        self, 
        requests: List[EmbeddingRequest]
    ) -> List[EmbeddingResult]:
        """Process requests that are not in cache."""
        if not requests:
            return []
        
        # Split into manageable batches
        batches = [
            requests[i:i + self.max_batch_size] 
            for i in range(0, len(requests), self.max_batch_size)
        ]
        
        all_results = []
        for i, batch in enumerate(batches):
            self.logger.debug(f"Processing batch {i + 1}/{len(batches)} ({len(batch)} items)")
            
            # Extract texts
            texts = [req.text for req in batch]
            
            start_time = time.time()
            try:
                # Generate embeddings in batch
                embeddings = await asyncio.to_thread(
                    self.embeddings_model.embed_documents, texts
                )
                
                # Create results
                batch_results = []
                total_tokens = 0
                
                for req, embedding in zip(batch, embeddings):
                    tokens_used = self.estimate_tokens(req.text)
                    total_tokens += tokens_used
                    
                    result = EmbeddingResult(
                        id=req.id,
                        text=req.text,
                        embedding=embedding,
                        model=self.model,
                        dimensions=len(embedding),
                        tokens_used=tokens_used,
                        created_at=datetime.now(),
                        cache_hit=False
                    )
                    
                    batch_results.append(result)
                    
                    # Cache the result
                    if self.cache:
                        self.cache.set(req.text, self.model, embedding)
                
                # Update metrics
                self.total_requests += len(batch)
                self.total_tokens += total_tokens
                self.total_cost += self.estimate_cost(total_tokens)
                self.batch_count += 1
                
                processing_time = time.time() - start_time
                self.logger.debug(
                    f"Batch {i + 1} completed | "
                    f"Time: {processing_time:.3f}s | "
                    f"Tokens: {total_tokens} | "
                    f"Cost: ${self.estimate_cost(total_tokens):.6f}"
                )
                
                all_results.extend(batch_results)
                
            except Exception as e:
                self.logger.error(f"Batch {i + 1} failed: {str(e)}")
                # Create error results
                for req in batch:
                    all_results.append(EmbeddingResult(
                        id=req.id,
                        text=req.text,
                        embedding=[0.0] * 1536,  # Default dimension
                        model=self.model,
                        dimensions=1536,
                        tokens_used=0,
                        created_at=datetime.now(),
                        cache_hit=False
                    ))
        
        return all_results
    
    def create_batch_from_texts(
        self, 
        texts: List[str], 
        priority: int = 1
    ) -> EmbeddingBatch:
        """Create batch from list of texts."""
        requests = [
            EmbeddingRequest(
                id=f"text_{i}",
                text=text,
                priority=priority
            )
            for i, text in enumerate(texts)
        ]
        
        return EmbeddingBatch(
            id=f"batch_{int(time.time())}",
            requests=requests,
            created_at=datetime.now(),
            priority=priority
        )
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        stats = {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "batch_count": self.batch_count,
            "model": self.model,
            "max_batch_size": self.max_batch_size
        }
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats
    
    async def optimize_vector_search_queries(
        self, 
        queries: List[str]
    ) -> Dict[str, List[float]]:
        """
        Optimize embedding generation for vector search queries.
        
        Returns mapping of query -> embedding.
        """
        if not queries:
            return {}
        
        self.logger.info(f"🔍 Optimizing embeddings for {len(queries)} search queries")
        
        # Create batch requests
        requests = [
            EmbeddingRequest(
                id=f"query_{i}",
                text=query,
                priority=2,  # Higher priority for search queries
                metadata={"type": "search_query", "index": i}
            )
            for i, query in enumerate(queries)
        ]
        
        # Process batch
        results = await self.get_embeddings_batch(requests, use_cache=True)
        
        # Create mapping
        query_embeddings = {}
        for result in results:
            original_index = result.id.split("_")[1]
            query = queries[int(original_index)]
            query_embeddings[query] = result.embedding
        
        # Log optimization results
        cache_hits = sum(1 for r in results if r.cache_hit)
        
        # Calculate cost savings (only if cache exists)
        cost_saved = 0.0
        if self.cache:
            cached_queries = [q for q in queries if self.cache.get(q, self.model)]
            cost_saved = self.estimate_cost(
                sum(self.estimate_tokens(q) for q in cached_queries)
            ) * len(cached_queries)
        
        self.logger.info(
            f"✅ Query embedding optimization complete | "
            f"Cache hits: {cache_hits}/{len(results)} | "
            f"Cost saved: ${cost_saved:.6f}"
        )
        
        return query_embeddings 
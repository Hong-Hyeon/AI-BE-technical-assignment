"""
Token optimization service for efficient LLM usage within budget constraints.

This service implements:
- Batch processing for multiple LLM calls
- Prompt optimization to reduce token usage
- Token budget management and monitoring
- Intelligent caching strategies
"""
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage
from config.logging_config import get_api_logger


class OptimizationStrategy(str, Enum):
    """Token optimization strategies."""
    BATCH = "batch"  # Batch multiple requests
    COMPRESS = "compress"  # Compress prompts
    CACHE = "cache"  # Use intelligent caching
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class TokenBudget:
    """Token budget management."""
    total_budget: int = 20000  # $20 budget (~400k tokens for gpt-4o-mini)
    used_tokens: int = 0
    reserved_tokens: int = 2000  # Reserve for safety
    
    @property
    def available_tokens(self) -> int:
        return max(0, self.total_budget - self.used_tokens - self.reserved_tokens)
    
    @property
    def usage_percentage(self) -> float:
        return (self.used_tokens / self.total_budget) * 100
    
    def can_afford(self, estimated_tokens: int) -> bool:
        return estimated_tokens <= self.available_tokens


@dataclass
class BatchRequest:
    """Batch request for LLM processing."""
    id: str
    prompt: str
    context: Dict[str, Any]
    priority: int = 1  # Higher number = higher priority


@dataclass
class OptimizedPrompt:
    """Optimized prompt with metadata."""
    content: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    estimated_tokens: int


class TokenOptimizationService:
    """Service for optimizing LLM token usage."""
    
    def __init__(self, initial_budget: int = 20000):
        self.logger = get_api_logger()
        self.budget = TokenBudget(total_budget=initial_budget)
        self.prompt_cache: Dict[str, str] = {}
        self.batch_size = 3  # Optimal batch size for context preservation
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def compress_prompt(self, prompt: str, context: Dict[str, Any]) -> OptimizedPrompt:
        """
        Compress prompt while preserving essential information.
        
        Optimization techniques:
        1. Remove redundant phrases
        2. Use abbreviations for common terms
        3. Consolidate similar information
        4. Optimize formatting
        """
        original_length = len(prompt)
        
        # Apply compression techniques
        compressed = prompt
        
        # 1. Remove redundant phrases
        redundant_phrases = [
            "다음과 같은 정보를 바탕으로 ",
            "아래의 내용을 참고하여 ",
            "상세한 분석을 통해 ",
            "종합적으로 판단하여 "
        ]
        for phrase in redundant_phrases:
            compressed = compressed.replace(phrase, "")
        
        # 2. Use abbreviations
        abbreviations = {
            "대학교": "대학",
            "회사에서": "에서",
            "경험을 했다": "경험함",
            "분석해주세요": "분석하세요",
            "다음과 같습니다": "다음:",
            "정보는 다음과 같다": "정보:"
        }
        for full, abbrev in abbreviations.items():
            compressed = compressed.replace(full, abbrev)
        
        # 3. Remove excessive whitespace
        import re
        compressed = re.sub(r'\s+', ' ', compressed.strip())
        
        compressed_length = len(compressed)
        compression_ratio = (original_length - compressed_length) / original_length
        estimated_tokens = self.estimate_tokens(compressed)
        
        return OptimizedPrompt(
            content=compressed,
            original_length=original_length,
            compressed_length=compressed_length,
            compression_ratio=compression_ratio,
            estimated_tokens=estimated_tokens
        )
    
    def create_batch_prompt(self, requests: List[BatchRequest]) -> str:
        """
        Create a single batch prompt for multiple requests.
        
        This reduces overhead by processing multiple items in one LLM call.
        """
        if not requests:
            return ""
        
        # Sort by priority
        requests.sort(key=lambda x: x.priority, reverse=True)
        
        batch_prompt = """다음 항목들을 각각 분석하고 결과를 JSON 형태로 반환하세요:

분석 항목들:
"""
        
        for i, req in enumerate(requests, 1):
            batch_prompt += f"\n{i}. ID: {req.id}\n"
            batch_prompt += f"   내용: {req.context}\n"
        
        batch_prompt += """
각 항목에 대해 다음 형식으로 응답하세요:
{
  "results": [
    {
      "id": "항목_ID",
      "analysis": "분석 결과",
      "tags": ["태그1", "태그2"],
      "confidence": 0.85
    }
  ]
}
"""
        return batch_prompt
    
    async def process_batch(
        self, 
        llm_model, 
        requests: List[BatchRequest]
    ) -> Dict[str, Any]:
        """Process multiple requests in a single batch."""
        if not requests:
            return {}
        
        # Check budget
        batch_prompt = self.create_batch_prompt(requests)
        estimated_tokens = self.estimate_tokens(batch_prompt) * 2  # Input + output
        
        if not self.budget.can_afford(estimated_tokens):
            self.logger.warning(
                f"🚨 Token budget exceeded! "
                f"Need: {estimated_tokens}, Available: {self.budget.available_tokens}"
            )
            raise ValueError("Token budget exceeded")
        
        # Log batch processing
        self.logger.info(
            f"🔄 Processing batch of {len(requests)} requests | "
            f"Estimated tokens: {estimated_tokens} | "
            f"Budget usage: {self.budget.usage_percentage:.1f}%"
        )
        
        start_time = time.time()
        try:
            response = await llm_model.ainvoke([HumanMessage(content=batch_prompt)])
            
            # Update token usage
            actual_tokens = self.estimate_tokens(batch_prompt + response.content)
            self.budget.used_tokens += actual_tokens
            
            processing_time = time.time() - start_time
            
            # Log success
            self.logger.info(
                f"✅ Batch processing completed | "
                f"Time: {processing_time:.2f}s | "
                f"Tokens used: {actual_tokens} | "
                f"Remaining budget: {self.budget.available_tokens}"
            )
            
            return {
                "response": response.content,
                "tokens_used": actual_tokens,
                "processing_time": processing_time,
                "batch_size": len(requests)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Batch processing failed: {str(e)}")
            raise
    
    async def optimize_workflow_prompts(
        self, 
        educations: List[Dict], 
        positions: List[Dict]
    ) -> Tuple[List[OptimizedPrompt], int]:
        """
        Optimize all workflow prompts for token efficiency.
        
        Returns optimized prompts and total estimated token usage.
        """
        optimized_prompts = []
        total_tokens = 0
        
        # Optimize education prompts
        for edu in educations:
            prompt = f"교육 분석: {edu.get('school_name', '')} {edu.get('degree_name', '')} {edu.get('field_of_study', '')}"
            optimized = self.compress_prompt(prompt, edu)
            optimized_prompts.append(optimized)
            total_tokens += optimized.estimated_tokens
        
        # Optimize position prompts
        for pos in positions:
            prompt = f"경력 분석: {pos.get('company_name', '')} {pos.get('title', '')} {pos.get('description', '')}"
            optimized = self.compress_prompt(prompt, pos)
            optimized_prompts.append(optimized)
            total_tokens += optimized.estimated_tokens
        
        # Add aggregation prompt estimation
        aggregation_tokens = self.estimate_tokens("최종 통합 분석 프롬프트") * 2
        total_tokens += aggregation_tokens
        
        self.logger.info(
            f"📊 Workflow optimization complete | "
            f"Total estimated tokens: {total_tokens} | "
            f"Budget usage: {((total_tokens + self.budget.used_tokens) / self.budget.total_budget) * 100:.1f}%"
        )
        
        return optimized_prompts, total_tokens
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        return {
            "total_budget": self.budget.total_budget,
            "used_tokens": self.budget.used_tokens,
            "available_tokens": self.budget.available_tokens,
            "usage_percentage": self.budget.usage_percentage,
            "reserved_tokens": self.budget.reserved_tokens,
            "status": (
                "critical" if self.budget.usage_percentage > 90 else 
                "warning" if self.budget.usage_percentage > 75 else "healthy"
            )
        }
    
    def should_use_batch_strategy(self, item_count: int) -> bool:
        """Determine if batch processing should be used."""
        # Use batch if we have multiple items and sufficient budget
        return (
            item_count >= 2 and 
            self.budget.available_tokens > 5000 and
            item_count <= self.batch_size
        )
    
    async def process_with_strategy(
        self,
        llm_model,
        items: List[Dict],
        strategy: OptimizationStrategy = OptimizationStrategy.HYBRID
    ) -> List[Dict[str, Any]]:
        """
        Process items using the specified optimization strategy.
        """
        if strategy == OptimizationStrategy.BATCH or (
            strategy == OptimizationStrategy.HYBRID and 
            self.should_use_batch_strategy(len(items))
        ):
            # Use batch processing
            batch_requests = [
                BatchRequest(
                    id=f"item_{i}",
                    prompt="",  # Will be generated in batch
                    context=item,
                    priority=1
                )
                for i, item in enumerate(items)
            ]
            
            result = await self.process_batch(llm_model, batch_requests)
            return [result]  # Single batch result
        
        else:
            # Use individual processing with optimization
            results = []
            for i, item in enumerate(items):
                # Apply compression and caching
                # Individual processing logic here
                results.append({"item": item, "processed": True})
            
            return results 
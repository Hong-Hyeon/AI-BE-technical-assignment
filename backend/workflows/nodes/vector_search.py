"""
Vector Search Node for talent analysis workflow.

This module provides the VectorSearchNode class responsible for
performing vector search operations to find semantically relevant
context with proper error handling and logging.
"""

import time
from typing import Dict, Any

from config.logging_config import WorkflowLogger
from core.exceptions import WorkflowException, ExternalServiceException, ErrorCode
from ..base.state import TalentAnalysisState


class VectorSearchNode:
    """
    Node responsible for performing vector search for semantically relevant context.
    
    This node:
    - Converts talent data to searchable format
    - Performs vector search operations
    - Logs search metrics and results
    - Handles search failures gracefully
    """
    
    def __init__(self, logger: WorkflowLogger):
        """
        Initialize vector search node.
        
        Args:
            logger: Workflow logger instance
        """
        self.logger = logger
    
    async def execute(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """
        Perform vector search for semantically relevant context.
        
        Args:
            state: Current workflow state containing talent_data
            
        Returns:
            Dictionary with vector_search_results
        """
        node_start_time = time.time()
        talent_data = state["talent_data"]
        talent_id = state["talent_id"]
        
        search_input_summary = {
            "educations": len(talent_data.educations),
            "positions": len(talent_data.positions),
            "skills": len(talent_data.skills)
        }
        
        self.logger.log_node_start("vector_search", search_input_summary)
        
        try:
            # Import here to avoid circular dependencies
            from factories.vector_search_factory import VectorSearchManager
            
            # Initialize vector search manager
            search_start_time = time.time()
            self.logger.logger.debug("Initializing vector search manager...")
            vector_search_manager = VectorSearchManager()
            
            # Convert talent data to searchable format
            self.logger.logger.debug("Converting talent data for vector search...")
            talent_dict = self._convert_talent_data_for_search(talent_data)
            
            # Log input data transformation
            self.logger.log_data_transformation(
                "vector_search_input",
                {"raw_talent_data": f"{len(talent_data.educations)} educations, {len(talent_data.positions)} positions"},
                {"search_ready_data": "Converted to dict format for vector search"}
            )
            
            # Perform vector search
            self.logger.logger.debug("Performing vector search for talent context...")
            vector_results = vector_search_manager.search_talent_context(talent_dict)
            search_time = time.time() - search_start_time
            
            # Validate and enrich results
            enriched_results = self._enrich_search_results(vector_results, talent_dict)
            
            # Log vector search results
            results_count = len(enriched_results.get("companies", [])) + len(enriched_results.get("news", []))
            self.logger.log_vector_search(
                {
                    "search_queries": enriched_results.get("search_queries", []),
                    "input_data_types": list(talent_dict.keys())
                },
                results_count,
                search_time
            )
            
            # Close resources
            vector_search_manager.close()
            
            # Log successful completion
            node_duration = time.time() - node_start_time
            self.logger.log_node_complete(
                "vector_search", 
                node_duration,
                {
                    "companies_found": len(enriched_results.get("companies", [])),
                    "news_found": len(enriched_results.get("news", [])),
                    "queries_executed": len(enriched_results.get("search_queries", [])),
                    "search_time": search_time
                }
            )
            
            return {"vector_search_results": enriched_results}
            
        except Exception as e:
            # Log error with context
            error_context = {
                "talent_id": talent_id,
                "error_type": type(e).__name__,
                "node_duration": time.time() - node_start_time,
                "search_input": search_input_summary
            }
            self.logger.log_node_error("vector_search", e, error_context)
            
            # Return empty results instead of failing (graceful degradation)
            self.logger.logger.warning(f"Vector search failed, returning empty results: {e}")
            empty_results = {
                "companies": [],
                "news": [],
                "search_queries": [],
                "metadata": {
                    "search_time": 0,
                    "error": str(e),
                    "fallback": True
                }
            }
            return {"vector_search_results": empty_results}
    
    def _convert_talent_data_for_search(self, talent_data) -> Dict[str, Any]:
        """
        Convert talent data to format suitable for vector search.
        
        Args:
            talent_data: TalentData object
            
        Returns:
            Dictionary suitable for vector search
        """
        try:
            talent_dict = {
                "educations": [edu.dict() for edu in talent_data.educations],
                "positions": [pos.dict() for pos in talent_data.positions],
                "skills": talent_data.skills if hasattr(talent_data, 'skills') else []
            }
            
            # Add optional fields if available
            if hasattr(talent_data, 'headline') and talent_data.headline:
                talent_dict["headline"] = talent_data.headline
            
            if hasattr(talent_data, 'summary') and talent_data.summary:
                talent_dict["summary"] = talent_data.summary
            
            return talent_dict
            
        except Exception as e:
            self.logger.logger.error(f"Failed to convert talent data for search: {e}")
            # Return minimal searchable data
            return {
                "educations": [],
                "positions": [],
                "skills": []
            }
    
    def _enrich_search_results(self, results: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich search results with metadata and validation.
        
        Args:
            results: Raw search results from vector search
            input_data: Original input data for context
            
        Returns:
            Enriched results with metadata
        """
        enriched = {
            "companies": results.get("companies", []),
            "news": results.get("news", []),
            "search_queries": results.get("search_queries", []),
            "metadata": {
                "original_input_size": {
                    "educations": len(input_data.get("educations", [])),
                    "positions": len(input_data.get("positions", [])),
                    "skills": len(input_data.get("skills", []))
                },
                "results_found": {
                    "companies": len(results.get("companies", [])),
                    "news": len(results.get("news", []))
                },
                "search_success": True
            }
        }
        
        # Add quality metrics
        total_results = len(enriched["companies"]) + len(enriched["news"])
        total_queries = len(enriched["search_queries"])
        
        enriched["metadata"]["quality_metrics"] = {
            "total_results": total_results,
            "total_queries": total_queries,
            "avg_results_per_query": total_results / total_queries if total_queries > 0 else 0,
            "has_company_context": len(enriched["companies"]) > 0,
            "has_news_context": len(enriched["news"]) > 0
        }
        
        return enriched
    
    def _validate_search_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate search results structure and content.
        
        Args:
            results: Search results to validate
            
        Returns:
            Validation summary
        """
        validation = {
            "valid_structure": all(
                key in results for key in ["companies", "news", "search_queries"]
            ),
            "has_results": len(results.get("companies", [])) > 0 or len(results.get("news", [])) > 0,
            "companies_count": len(results.get("companies", [])),
            "news_count": len(results.get("news", [])),
            "queries_count": len(results.get("search_queries", []))
        }
        
        return validation


# Factory function for easy instantiation
def create_vector_search_node(logger: WorkflowLogger) -> VectorSearchNode:
    """
    Create a vector search node instance.
    
    Args:
        logger: Workflow logger instance
        
    Returns:
        Configured VectorSearchNode instance
    """
    return VectorSearchNode(logger) 
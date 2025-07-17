"""
Context Preprocessor Node for talent analysis workflow.

This module provides the ContextPreprocessorNode class responsible for
preprocessing context data (company and news information) with proper
error handling and logging.
"""

import time
from typing import Dict, Any

from config.logging_config import WorkflowLogger
from core.exceptions import WorkflowException, ExternalServiceException, ErrorCode
from ..base.state import TalentAnalysisState


class ContextPreprocessorNode:
    """
    Node responsible for preprocessing context data from database.
    
    This node:
    - Loads company data for each position
    - Loads news data for each company
    - Handles missing data gracefully
    - Logs preprocessing metrics
    """
    
    def __init__(self, logger: WorkflowLogger):
        """
        Initialize context preprocessor node.
        
        Args:
            logger: Workflow logger instance
        """
        self.logger = logger
    
    async def execute(self, state: TalentAnalysisState) -> Dict[str, Any]:
        """
        Preprocess context data from database.
        
        Args:
            state: Current workflow state containing talent_data
            
        Returns:
            Dictionary with preprocessed_context containing companies and news
            
        Raises:
            WorkflowException: If critical preprocessing fails
        """
        node_start_time = time.time()
        talent_data = state["talent_data"]
        talent_id = state["talent_id"]
        
        positions_count = len(talent_data.positions)
        
        self.logger.log_node_start(
            "preprocess_context", 
            {
                "talent_id": talent_id,
                "positions_to_process": positions_count,
                "companies": [pos.company_name for pos in talent_data.positions]
            }
        )
        
        try:
            # Import here to avoid circular dependencies
            from factories.data_source_factory import DefaultDataSourceFactory, DataSourceManager
            
            # Initialize data source manager
            self.logger.logger.debug("Initializing data source manager for context preprocessing...")
            data_factory = DefaultDataSourceFactory("./example_datas")
            data_manager = DataSourceManager(data_factory)
            
            # Process each position to collect company and news data
            company_data = {}
            news_data = {}
            processing_results = {
                "companies_found": 0,
                "news_found": 0,
                "companies_missing": 0,
                "news_missing": 0,
                "processing_errors": []
            }
            
            for i, position in enumerate(talent_data.positions):
                company_name = position.company_name
                self.logger.logger.debug(
                    f"Processing position {i + 1}/{positions_count}: {company_name}"
                )
                
                # Process company data
                company_result = await self._process_company_data(
                    data_manager, company_name
                )
                if company_result['success']:
                    company_data[company_name] = company_result['data']
                    processing_results["companies_found"] += 1
                else:
                    processing_results["companies_missing"] += 1
                    if company_result.get('error'):
                        processing_results["processing_errors"].append(
                            f"Company {company_name}: {company_result['error']}"
                        )
                
                # Process news data
                news_result = await self._process_news_data(
                    data_manager, company_name
                )
                if news_result['success']:
                    news_data[company_name] = news_result['data']
                    processing_results["news_found"] += 1
                else:
                    processing_results["news_missing"] += 1
            
            # Create preprocessed context
            preprocessed_context = {
                "companies": company_data,
                "news": news_data,
                "metadata": {
                    "processed_at": time.time(),
                    "positions_processed": positions_count,
                    "processing_results": processing_results
                }
            }
            
            # Log data transformation info
            self.logger.log_data_transformation(
                "context_preprocessing",
                {
                    "positions_processed": positions_count,
                    "unique_companies": len(set(pos.company_name for pos in talent_data.positions))
                },
                {
                    "companies_found": processing_results["companies_found"],
                    "news_datasets_found": processing_results["news_found"],
                    "company_names": list(company_data.keys()),
                    "success_rate": {
                        "companies": processing_results["companies_found"] / positions_count if positions_count > 0 else 0,
                        "news": processing_results["news_found"] / positions_count if positions_count > 0 else 0
                    }
                }
            )
            
            # Log successful completion
            node_duration = time.time() - node_start_time
            self.logger.log_node_complete(
                "preprocess_context", 
                node_duration,
                {
                    "companies_loaded": len(company_data),
                    "news_datasets": len(news_data),
                    "success_rates": f"Companies: {processing_results['companies_found']}/{positions_count}, "
                                   f"News: {processing_results['news_found']}/{positions_count}"
                }
            )
            
            return {"preprocessed_context": preprocessed_context}
            
        except Exception as e:
            # Log error with context
            error_context = {
                "talent_id": talent_id,
                "positions_processed": positions_count,
                "error_type": type(e).__name__,
                "node_duration": time.time() - node_start_time
            }
            self.logger.log_node_error("preprocess_context", e, error_context)
            
            # Return empty context instead of failing (graceful degradation)
            self.logger.logger.warning(
                f"Context preprocessing failed, returning empty context: {e}"
            )
            return {
                "preprocessed_context": {
                    "companies": {},
                    "news": {},
                    "metadata": {
                        "processed_at": time.time(),
                        "positions_processed": 0,
                        "error": str(e)
                    }
                }
            }
    
    async def _process_company_data(
        self, 
        data_manager, 
        company_name: str
    ) -> Dict[str, Any]:
        """
        Process company data for a single company.
        
        Args:
            data_manager: Data manager instance
            company_name: Name of the company to process
            
        Returns:
            Dictionary with success status and data/error
        """
        try:
            company_info = data_manager.get_company_data(company_name)
            self.logger.logger.debug(f"✅ Found company data for {company_name}")
            
            return {
                "success": True,
                "data": company_info,
                "company_name": company_name
            }
            
        except Exception as e:
            self.logger.logger.debug(
                f"⚠️ No company data available for {company_name}: {str(e)}"
            )
            return {
                "success": False,
                "error": str(e),
                "company_name": company_name
            }
    
    async def _process_news_data(
        self, 
        data_manager, 
        company_name: str
    ) -> Dict[str, Any]:
        """
        Process news data for a single company.
        
        Args:
            data_manager: Data manager instance
            company_name: Name of the company to process
            
        Returns:
            Dictionary with success status and data/error
        """
        try:
            news_info = data_manager.get_news_data(company_name)
            self.logger.logger.debug(f"✅ Found news data for {company_name}")
            
            return {
                "success": True,
                "data": news_info,
                "company_name": company_name
            }
            
        except Exception as e:
            self.logger.logger.debug(
                f"⚠️ No news data available for {company_name}: {str(e)}"
            )
            return {
                "success": False,
                "error": str(e),
                "company_name": company_name
            }
    
    def _validate_context_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the preprocessed context data.
        
        Args:
            context: Preprocessed context dictionary
            
        Returns:
            Validation summary
        """
        validation = {
            "has_companies": len(context.get("companies", {})) > 0,
            "has_news": len(context.get("news", {})) > 0,
            "company_count": len(context.get("companies", {})),
            "news_count": len(context.get("news", {})),
            "valid_structure": all(
                key in context for key in ["companies", "news"]
            )
        }
        
        return validation


# Factory function for easy instantiation
def create_context_preprocessor_node(logger: WorkflowLogger) -> ContextPreprocessorNode:
    """
    Create a context preprocessor node instance.
    
    Args:
        logger: Workflow logger instance
        
    Returns:
        Configured ContextPreprocessorNode instance
    """
    return ContextPreprocessorNode(logger) 
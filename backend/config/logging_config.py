"""
Logging configuration for the AI-BE-technical-assignment project.
Provides comprehensive logging setup for tracking LangGraph workflow execution.
"""
import logging
import logging.config
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def get_log_directory() -> str:
    """Get the logs directory path."""
    # Use the project root's logs directory
    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    return str(log_dir)


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration dictionary."""
    log_dir = get_log_directory()
    
    # Create timestamped log files
    timestamp = datetime.now().strftime("%Y%m%d")
    
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(asctime)s | %(levelname)-8s | %(message)s",
                "datefmt": "%H:%M:%S"
            },
            "workflow": {
                "format": "%(asctime)s | WORKFLOW | %(levelname)-8s | %(message)s",
                "datefmt": "%H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": sys.stdout
            },
            "file_all": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": os.path.join(log_dir, f"app_{timestamp}.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "file_workflow": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "workflow",
                "filename": os.path.join(log_dir, f"workflow_{timestamp}.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "file_api": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": os.path.join(log_dir, f"api_{timestamp}.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "file_error": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": os.path.join(log_dir, f"error_{timestamp}.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 10,
                "encoding": "utf8"
            }
        },
        "loggers": {
            "": {  # Root logger
                "level": "INFO",
                "handlers": ["console", "file_all", "file_error"]
            },
            "workflow": {
                "level": "DEBUG",
                "handlers": ["console", "file_workflow", "file_error"],
                "propagate": False
            },
            "api": {
                "level": "INFO",
                "handlers": ["console", "file_api", "file_error"],
                "propagate": False
            },
            "factories": {
                "level": "INFO",
                "handlers": ["console", "file_all", "file_error"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file_api"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["file_api"],
                "propagate": False
            }
        }
    }


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration for the application."""
    config = get_logging_config()
    
    # Adjust log level if specified
    if log_level.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        config["loggers"][""]["level"] = log_level.upper()
        config["loggers"]["workflow"]["level"] = log_level.upper()
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Log startup message
    logger = logging.getLogger("app")
    logger.info("=" * 60)
    logger.info("🚀 AI-BE Technical Assignment - Logging Initialized")
    logger.info(f"📝 Log Level: {log_level.upper()}")
    logger.info(f"📁 Log Directory: {get_log_directory()}")
    logger.info("=" * 60)


def get_workflow_logger() -> logging.Logger:
    """Get logger specifically for workflow operations."""
    return logging.getLogger("workflow")


def get_api_logger() -> logging.Logger:
    """Get logger specifically for API operations."""
    return logging.getLogger("api")


def get_factory_logger() -> logging.Logger:
    """Get logger specifically for factory operations."""
    return logging.getLogger("factories")


class WorkflowLogger:
    """Utility class for structured workflow logging."""
    
    def __init__(self, workflow_name: str = "TalentAnalysis"):
        self.logger = get_workflow_logger()
        self.workflow_name = workflow_name
    
    def log_node_start(self, node_name: str, state_info: Dict[str, Any] = None):
        """Log the start of a workflow node."""
        msg = f"🔄 [{self.workflow_name}] Starting node: {node_name}"
        if state_info:
            msg += f" | State: {state_info}"
        self.logger.info(msg)
    
    def log_node_complete(self, node_name: str, duration: float = None, result_info: Dict[str, Any] = None):
        """Log the completion of a workflow node."""
        msg = f"✅ [{self.workflow_name}] Completed node: {node_name}"
        if duration:
            msg += f" | Duration: {duration:.2f}s"
        if result_info:
            msg += f" | Result: {result_info}"
        self.logger.info(msg)
    
    def log_node_error(self, node_name: str, error: Exception, state_info: Dict[str, Any] = None):
        """Log an error in a workflow node."""
        msg = f"❌ [{self.workflow_name}] Error in node: {node_name} | Error: {str(error)}"
        if state_info:
            msg += f" | State: {state_info}"
        self.logger.error(msg, exc_info=True)
    
    def log_workflow_start(self, talent_id: str, params: Dict[str, Any] = None):
        """Log the start of the entire workflow."""
        msg = f"🚀 [{self.workflow_name}] Starting workflow for talent_id: {talent_id}"
        if params:
            msg += f" | Parameters: {params}"
        self.logger.info(msg)
    
    def log_workflow_complete(self, talent_id: str, total_duration: float, result_summary: Dict[str, Any] = None):
        """Log the completion of the entire workflow."""
        msg = f"🎉 [{self.workflow_name}] Workflow completed for talent_id: {talent_id} | Total Duration: {total_duration:.2f}s"
        if result_summary:
            msg += f" | Summary: {result_summary}"
        self.logger.info(msg)
    
    def log_data_transformation(self, step: str, input_info: Dict[str, Any], output_info: Dict[str, Any]):
        """Log data transformation steps."""
        msg = f"🔄 [{self.workflow_name}] Data transformation: {step}"
        msg += f" | Input: {input_info} | Output: {output_info}"
        self.logger.debug(msg)
    
    def log_vector_search(self, query_info: Dict[str, Any], results_count: int, search_time: float):
        """Log vector search operations."""
        msg = f"🔍 [{self.workflow_name}] Vector search completed"
        msg += f" | Query: {query_info} | Results: {results_count} | Time: {search_time:.2f}s"
        self.logger.info(msg)
    
    def log_llm_call(self, node_name: str, model_name: str, prompt_length: int, response_length: int, call_time: float):
        """Log LLM API calls."""
        msg = f"🤖 [{self.workflow_name}] LLM call in {node_name}"
        msg += f" | Model: {model_name} | Prompt: {prompt_length} chars | Response: {response_length} chars | Time: {call_time:.2f}s"
        self.logger.info(msg)


# Export convenience functions
__all__ = [
    "setup_logging",
    "get_workflow_logger", 
    "get_api_logger",
    "get_factory_logger",
    "WorkflowLogger"
] 
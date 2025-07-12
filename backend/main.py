"""
Main application entry point.

This module is responsible solely for starting the FastAPI server.
All routing logic has been separated into dedicated router modules.
"""
import uvicorn
from app_factory import create_app

# Create FastAPI application using the factory pattern
app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
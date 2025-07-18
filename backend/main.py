"""
Main application entry point.

This module is responsible solely for starting the FastAPI server.
All routing logic has been separated into dedicated router modules.
"""
import uvicorn
from app_factory import create_app

from db import session

# Create FastAPI application using the factory pattern  
app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
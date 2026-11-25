"""
Minor Project Runner Script
Starts the Uvicorn web server hosting the FastAPI application.
Run locally: python run.py
Cloud platforms automatically inject $PORT.
"""

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    is_dev = os.environ.get("ENVIRONMENT", "development").lower() == "development"
    
    print("=" * 65)
    print(" AI Research Paper Summarizer & Key Insight Extractor")
    print(" College Minor Project - ECE")
    print("=" * 65)
    print(f" Server starting on http://{host}:{port}")
    print(f" Open your browser at: http://localhost:{port}")
    print(" Press Ctrl+C to stop the server.")
    print("=" * 65)
    
    uvicorn.run("app:app", host=host, port=port, reload=is_dev)

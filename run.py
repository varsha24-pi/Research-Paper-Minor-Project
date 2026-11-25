"""
Minor Project Runner Script
Starts the Uvicorn web server hosting the FastAPI application.
Run using: python run.py
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 65)
    print(" AI Research Paper Summarizer & Key Insight Extractor")
    print(" College Minor Project - CSE")
    print("=" * 65)
    print(" Server starting at: http://127.0.0.1:8000")
    print(" Open your browser and navigate to: http://127.0.0.1:8000")
    print(" Press Ctrl+C to stop the server.")
    print("=" * 65)
    
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

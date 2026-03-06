"""
Main entry point for the Retrieval Experiment Platform.
Run with: uvicorn main:app --reload
Or for the UI: streamlit run ui/dashboard.py
"""
import logging
import uvicorn
from api.routes import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
    from config import settings
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug)

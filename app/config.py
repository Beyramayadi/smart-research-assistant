# app/config.py
from pydantic import BaseModel
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"       # where PDFs live
VECTOR_DIR = BASE_DIR / ".chroma"  # Chroma persistence dir
DATA_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

class Settings(BaseModel):
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model name
    collection_name: str = "research_docs"     # Chroma collection name

settings = Settings()

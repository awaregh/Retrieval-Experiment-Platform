from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    # API
    app_name: str = "Retrieval Experiment Platform"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "sqlite:///./retrieval_lab.db"

    # Vector Store
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_prefix: str = "retrieval_exp"

    # Embedding defaults
    default_embedding_model: str = "all-MiniLM-L6-v2"
    openai_api_key: Optional[str] = None
    embedding_batch_size: int = 32

    # Chunking defaults
    default_chunk_size: int = 500
    default_chunk_overlap: int = 50

    # Retrieval defaults
    default_top_k: int = 10
    default_similarity_threshold: float = 0.0

    # Experiment tracking
    experiments_dir: str = "./experiments_data"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

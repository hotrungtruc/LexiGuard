"""
Core System Configuration using Pydantic BaseSettings
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # LLM Settings (Google Gemini)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # Qdrant Vector DB
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "legal_statutes_corpus"

    # Neo4j Legal Knowledge Graph
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "lexiguard_secure_password"

    # Langfuse
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "http://localhost:3000"

    # Embedding & Reranker
    DENSE_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    # Hugging Face & Local VLM Settings (Google PaliGemma / Qwen2-VL)
    HF_TOKEN: str = ""
    HF_USERNAME: str = "trungtruc1706"
    LOCAL_VLM_MODEL: str = "google/paligemma2-3b-pt-448"
    LOCAL_VLM_FALLBACK_MODEL: str = "Qwen/Qwen2-VL-2B-Instruct"
    LOCAL_VLM_QUANTIZATION: str = "4bit"
    USE_LOCAL_VLM: bool = True

    # API Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

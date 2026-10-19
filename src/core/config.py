"""
Core System Configuration using Pydantic BaseSettings
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # LLM Settings (Google Gemini & Ollama Dual-LLM)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    FAST_LLM_MODEL: str = "qwen2.5:3b-instruct-q4_K_M"
    REASONING_LLM_MODEL: str = "qwen2.5:7b-instruct-q4_K_M"
    OLLAMA_NUM_THREADS: int = 8

    # Qdrant Vector DB
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_PREFER_GRPC: bool = False
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "legal_statutes_corpus"
    QDRANT_TIMEOUT: float = 10.0

    # Neo4j Legal Knowledge Graph
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "lexiguard_secure_password"
    NEO4J_DATABASE: str = "neo4j"
    NEO4J_MAX_CONNECTION_LIFETIME: int = 3600
    NEO4J_MAX_CONNECTION_POOL_SIZE: int = 50
    NEO4J_CONNECTION_TIMEOUT: float = 30.0

    # Langfuse
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "http://localhost:3000"

    # Embedding & Reranker
    DENSE_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    # Hugging Face Settings
    HF_TOKEN: str = ""
    HF_USERNAME: str = "trungtruc1706"

    # OCR Quality Assessment Gate Thresholds (Phase 1 vs Phase 2 Branching)
    OCR_QUALITY_MIN_BLUR: float = 30.0
    OCR_QUALITY_MIN_CONTRAST: float = 20.0
    OCR_QUALITY_MIN_WIDTH: int = 500
    OCR_QUALITY_MIN_HEIGHT: int = 600

    # API Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()


# ==============================================================================
# ASYNCHRONOUS DATABASE CLIENT FACTORIES & LIFECYCLE MANAGEMENT
# ==============================================================================

from typing import Any, Dict, Tuple
from loguru import logger
from neo4j import AsyncDriver, AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient


def get_async_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    prefer_grpc: Optional[bool] = None,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> AsyncQdrantClient:
    """
    Factory function to initialize and return an AsyncQdrantClient instance.
    Supports in-memory mode (':memory:') for isolated testing and development.
    """
    target_url = url or settings.QDRANT_URL
    if target_url == ":memory:":
        logger.debug("Initializing in-memory AsyncQdrantClient for local testing")
        return AsyncQdrantClient(":memory:", **kwargs)

    logger.debug(f"Initializing AsyncQdrantClient connected to {target_url}")
    return AsyncQdrantClient(
        url=target_url,
        api_key=api_key or settings.QDRANT_API_KEY,
        prefer_grpc=prefer_grpc if prefer_grpc is not None else settings.QDRANT_PREFER_GRPC,
        timeout=timeout or settings.QDRANT_TIMEOUT,
        **kwargs,
    )


def get_async_neo4j_driver(
    uri: Optional[str] = None,
    auth: Optional[Tuple[str, str]] = None,
    max_connection_lifetime: Optional[int] = None,
    max_connection_pool_size: Optional[int] = None,
    connection_timeout: Optional[float] = None,
    **kwargs: Any,
) -> AsyncDriver:
    """
    Factory function to initialize and return an AsyncDriver instance via AsyncGraphDatabase.
    Utilizes connection pooling and configured timeouts for production stability.
    """
    target_uri = uri or settings.NEO4J_URI
    auth_credentials = auth or (settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
    logger.debug(f"Initializing AsyncGraphDatabase driver for {target_uri}")
    return AsyncGraphDatabase.driver(
        target_uri,
        auth=auth_credentials,
        max_connection_lifetime=max_connection_lifetime or settings.NEO4J_MAX_CONNECTION_LIFETIME,
        max_connection_pool_size=max_connection_pool_size or settings.NEO4J_MAX_CONNECTION_POOL_SIZE,
        connection_timeout=connection_timeout or settings.NEO4J_CONNECTION_TIMEOUT,
        **kwargs,
    )


async def check_async_qdrant_connection(
    client: Optional[AsyncQdrantClient] = None,
) -> bool:
    """
    Asynchronously verifies network connectivity to Qdrant Vector DB.
    Returns True if healthy, False otherwise.
    """
    should_close = False
    if client is None:
        client = get_async_qdrant_client()
        should_close = True

    try:
        collections = await client.get_collections()
        logger.info(
            f"Successfully connected to Qdrant. Found {len(collections.collections)} collections."
        )
        return True
    except Exception as exc:
        logger.warning(f"AsyncQdrantClient connection check failed: {exc}")
        return False
    finally:
        if should_close:
            await client.close()


async def check_async_neo4j_connection(
    driver: Optional[AsyncDriver] = None,
) -> bool:
    """
    Asynchronously verifies network connectivity to Neo4j Knowledge Graph.
    Returns True if healthy, False otherwise.
    """
    should_close = False
    if driver is None:
        driver = get_async_neo4j_driver()
        should_close = True

    try:
        await driver.verify_connectivity()
        logger.info("Successfully connected to Neo4j Knowledge Graph.")
        return True
    except Exception as exc:
        logger.warning(f"AsyncGraphDatabase connection check failed: {exc}")
        return False
    finally:
        if should_close:
            await driver.close()


class AsyncConnectionManager:
    """
    Singleton connection manager for asynchronous database instances.
    Coordinates initialization, health monitoring, and graceful resource disposal.
    """

    def __init__(self):
        self._qdrant_client: Optional[AsyncQdrantClient] = None
        self._neo4j_driver: Optional[AsyncDriver] = None

    def get_qdrant_client(self, **kwargs: Any) -> AsyncQdrantClient:
        """Get or initialize singleton AsyncQdrantClient."""
        if self._qdrant_client is None:
            self._qdrant_client = get_async_qdrant_client(**kwargs)
        return self._qdrant_client

    def get_neo4j_driver(self, **kwargs: Any) -> AsyncDriver:
        """Get or initialize singleton AsyncGraphDatabase driver."""
        if self._neo4j_driver is None:
            self._neo4j_driver = get_async_neo4j_driver(**kwargs)
        return self._neo4j_driver

    async def check_health(self) -> Dict[str, bool]:
        """Check asynchronous connectivity for both databases."""
        qdrant_ok = await check_async_qdrant_connection(self.get_qdrant_client())
        neo4j_ok = await check_async_neo4j_connection(self.get_neo4j_driver())
        return {
            "qdrant": qdrant_ok,
            "neo4j": neo4j_ok,
        }

    async def close(self) -> None:
        """Gracefully release all connection pools."""
        if self._qdrant_client is not None:
            await self._qdrant_client.close()
            self._qdrant_client = None
            logger.info("Closed AsyncQdrantClient connection pool.")

        if self._neo4j_driver is not None:
            await self._neo4j_driver.close()
            self._neo4j_driver = None
            logger.info("Closed AsyncGraphDatabase driver pool.")


async_connection_manager = AsyncConnectionManager()

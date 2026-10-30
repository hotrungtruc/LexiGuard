"""
Core System Configuration using Pydantic BaseSettings
"""

import asyncio
import os

# Prevent pandas/pyarrow timezone initialization from causing native access violation on Windows
os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")

from typing import Any, Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.hardware import HardwareProfile, hardware_profile


class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Hardware & Compute Optimization (Centralized Profile)
    OLLAMA_NUM_THREADS: int = Field(default_factory=lambda: hardware_profile.optimal_ollama_threads)
    OCR_NUM_THREADS: int = Field(default_factory=lambda: hardware_profile.optimal_ocr_threads)
    DEVICE: str = Field(default_factory=lambda: hardware_profile.optimal_device)

    # LLM Settings (Google Gemini & Ollama Dual-LLM)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    FAST_LLM_MODEL: str = "qwen2.5:3b-instruct-q4_K_M"
    REASONING_LLM_MODEL: str = "qwen2.5:7b-instruct-q4_K_M"

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

    @property
    def hardware(self) -> HardwareProfile:
        """Centralized hardware capabilities and compute profile."""
        return hardware_profile

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()


# ==============================================================================
# ASYNCHRONOUS DATABASE CLIENT FACTORIES & LIFECYCLE MANAGEMENT
# ==============================================================================

from typing import Tuple

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
        max_connection_pool_size=max_connection_pool_size
        or settings.NEO4J_MAX_CONNECTION_POOL_SIZE,
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
        if not circuit_breaker.is_available("qdrant"):
            return False
        client = get_async_qdrant_client()
        should_close = True

    try:
        collections = await asyncio.wait_for(client.get_collections(), timeout=2.0)
        logger.info(
            f"Successfully connected to Qdrant. Found {len(collections.collections)} collections."
        )
        circuit_breaker.record_success("qdrant")
        return True
    except Exception as exc:
        logger.warning(f"AsyncQdrantClient connection check failed: {exc}")
        circuit_breaker.record_failure("qdrant", exc)
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
        if not circuit_breaker.is_available("neo4j"):
            return False
        driver = get_async_neo4j_driver()
        should_close = True

    try:
        await asyncio.wait_for(driver.verify_connectivity(), timeout=2.0)
        logger.info("Successfully connected to Neo4j Knowledge Graph.")
        circuit_breaker.record_success("neo4j")
        return True
    except Exception as exc:
        logger.warning(f"AsyncGraphDatabase connection check failed: {exc}")
        circuit_breaker.record_failure("neo4j", exc)
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
        self._qdrant_loop: Optional[Any] = None
        self._neo4j_loop: Optional[Any] = None

    def get_qdrant_client(self, **kwargs: Any) -> AsyncQdrantClient:
        """Get or initialize singleton AsyncQdrantClient."""
        try:
            curr_loop = asyncio.get_running_loop()
        except RuntimeError:
            curr_loop = None

        if self._qdrant_client is None or (curr_loop is not None and self._qdrant_loop != curr_loop):
            self._qdrant_client = get_async_qdrant_client(**kwargs)
            self._qdrant_loop = curr_loop
        return self._qdrant_client

    def get_neo4j_driver(self, **kwargs: Any) -> AsyncDriver:
        """Get or initialize singleton AsyncGraphDatabase driver."""
        try:
            curr_loop = asyncio.get_running_loop()
        except RuntimeError:
            curr_loop = None

        if self._neo4j_driver is None or (curr_loop is not None and self._neo4j_loop != curr_loop):
            self._neo4j_driver = get_async_neo4j_driver(**kwargs)
            self._neo4j_loop = curr_loop
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


class DatabaseCircuitBreaker:
    """
    Resilient circuit breaker for external database services (Qdrant & Neo4j).
    Prevents cascading TCP connection timeouts when databases are unreachable,
    enabling instant fallback to local offline knowledge repositories.
    """

    def __init__(self, recovery_timeout: float = 60.0):
        self.recovery_timeout = recovery_timeout
        self._states: Dict[str, Dict[str, Any]] = {}

    def is_available(self, service: str) -> bool:
        """Checks if a service is healthy or ready for a probe."""
        info = self._states.get(service)
        if not info:
            return True
        state = info.get("state", "CLOSED")
        if state == "CLOSED":
            return True
        elif state == "OPEN":
            import time

            now = time.monotonic()
            if now - info.get("last_failure", 0.0) >= self.recovery_timeout:
                info["state"] = "HALF_OPEN"
                logger.info(
                    f"Circuit breaker for '{service}' transitioned to HALF_OPEN (probing connection)."
                )
                return True
            return False
        elif state == "HALF_OPEN":
            return True
        return True

    def record_success(self, service: str) -> None:
        """Records a successful operation, resetting the circuit breaker."""
        if service in self._states:
            if self._states[service].get("state") != "CLOSED":
                logger.info(
                    f"Circuit breaker for '{service}' transitioned to CLOSED (service recovered)."
                )
            self._states[service] = {"state": "CLOSED", "last_failure": 0.0, "failures": 0}

    def record_failure(self, service: str, error: Optional[Exception] = None) -> None:
        """Records a failure and trips the circuit breaker to OPEN."""
        import time

        now = time.monotonic()
        prev = self._states.get(service, {})
        failures = prev.get("failures", 0) + 1
        self._states[service] = {
            "state": "OPEN",
            "last_failure": now,
            "failures": failures,
            "last_error": str(error) if error else "",
        }
        logger.warning(
            f"Circuit breaker for '{service}' TRIPPED to OPEN (failures={failures}). "
            f"Bypassing live calls for {self.recovery_timeout}s. Reason: {error}"
        )

    def reset(self, service: Optional[str] = None) -> None:
        """Manually reset circuit breaker state."""
        if service:
            self._states.pop(service, None)
        else:
            self._states.clear()


circuit_breaker = DatabaseCircuitBreaker()
async_connection_manager = AsyncConnectionManager()


def run_coroutine_sync(coro_fn, *args, **kwargs):
    """
    Safely executes an async coroutine factory from synchronous context.
    Eliminates 'event loop already running' crashes by isolating execution inside a dedicated worker thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(coro_fn(*args, **kwargs))).result()
    else:
        return asyncio.run(coro_fn(*args, **kwargs))

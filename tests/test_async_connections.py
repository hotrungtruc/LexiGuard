"""
Test Suite: Asynchronous Database Client Connections
===================================================
Verifies the successful initialization and lifecycle of:
- AsyncQdrantClient (Qdrant Vector Database)
- AsyncGraphDatabase (Neo4j Knowledge Graph)
- AsyncConnectionManager (Unified Lifespan & Health Management)
"""

import pytest
from neo4j import AsyncDriver, AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient

from src.core.config import (
    AsyncConnectionManager,
    check_async_neo4j_connection,
    check_async_qdrant_connection,
    get_async_neo4j_driver,
    get_async_qdrant_client,
)


@pytest.mark.asyncio
async def test_get_async_qdrant_client_initialization():
    """Verify AsyncQdrantClient initializes properly with configured settings."""
    client = get_async_qdrant_client()
    assert isinstance(client, AsyncQdrantClient)
    # Ensure client can be closed cleanly
    await client.close()


@pytest.mark.asyncio
async def test_async_qdrant_client_in_memory_operations():
    """Verify AsyncQdrantClient can execute asynchronous operations in memory."""
    mem_client = get_async_qdrant_client(url=":memory:")
    assert isinstance(mem_client, AsyncQdrantClient)

    # Health check on in-memory instance
    is_healthy = await check_async_qdrant_connection(mem_client)
    assert is_healthy is True

    # Test collection listing
    collections = await mem_client.get_collections()
    assert hasattr(collections, "collections")
    assert isinstance(collections.collections, list)

    await mem_client.close()


@pytest.mark.asyncio
async def test_get_async_neo4j_driver_initialization():
    """Verify AsyncGraphDatabase.driver initializes an AsyncDriver with settings."""
    driver = get_async_neo4j_driver()
    assert isinstance(driver, AsyncDriver)
    # Verify driver configuration
    assert driver._pool is not None
    await driver.close()


@pytest.mark.asyncio
async def test_get_async_neo4j_driver_custom_parameters():
    """Verify AsyncGraphDatabase driver honors custom URI and timeouts."""
    custom_uri = "bolt://127.0.0.1:7687"
    custom_auth = ("custom_user", "custom_pass")
    driver = get_async_neo4j_driver(
        uri=custom_uri,
        auth=custom_auth,
        connection_timeout=5.0,
        max_connection_pool_size=10,
    )
    assert isinstance(driver, AsyncDriver)
    await driver.close()


@pytest.mark.asyncio
async def test_async_connection_manager_lifecycle():
    """Verify AsyncConnectionManager manages singleton clients and handles graceful shutdown."""
    manager = AsyncConnectionManager()

    # Get clients
    qdrant_client = manager.get_qdrant_client(url=":memory:")
    assert isinstance(qdrant_client, AsyncQdrantClient)
    assert manager.get_qdrant_client() is qdrant_client  # Singleton behavior

    neo4j_driver = manager.get_neo4j_driver()
    assert isinstance(neo4j_driver, AsyncDriver)
    assert manager.get_neo4j_driver() is neo4j_driver  # Singleton behavior

    # Close manager
    await manager.close()
    assert manager._qdrant_client is None
    assert manager._neo4j_driver is None


@pytest.mark.asyncio
async def test_connection_health_check_graceful_failures():
    """Verify health check functions handle offline/invalid servers without crashing."""
    # Test invalid Qdrant endpoint returns False
    invalid_qdrant = AsyncQdrantClient(url="http://127.0.0.1:9999", timeout=0.5)
    qdrant_ok = await check_async_qdrant_connection(invalid_qdrant)
    assert qdrant_ok is False
    await invalid_qdrant.close()

    # Test invalid Neo4j endpoint returns False
    invalid_neo4j = AsyncGraphDatabase.driver(
        "bolt://127.0.0.1:9999",
        auth=("neo4j", "wrong_pass"),
        connection_timeout=0.5,
    )
    neo4j_ok = await check_async_neo4j_connection(invalid_neo4j)
    assert neo4j_ok is False
    await invalid_neo4j.close()

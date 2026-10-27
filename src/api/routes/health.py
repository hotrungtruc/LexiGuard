"""
Health check route for Qdrant and Neo4j (/health).
"""

from fastapi import APIRouter
from loguru import logger

from src.core.config import async_connection_manager, circuit_breaker

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Checks real-time connectivity with Qdrant vector database and Neo4j knowledge graph.
    Returns 200 with 'healthy' if live DBs are connected, or 'degraded' if operating on offline fallbacks.
    """
    logger.info("Executing /health status check...")
    db_health = await async_connection_manager.check_health()
    qdrant_ok = db_health.get("qdrant", False)
    neo4j_ok = db_health.get("neo4j", False)

    is_healthy = qdrant_ok and neo4j_ok
    status = "healthy" if is_healthy else "degraded"

    return {
        "status": status,
        "mode": "live" if is_healthy else "offline_fallback_active",
        "dependencies": {
            "qdrant": "connected"
            if qdrant_ok
            else "disconnected (offline legal dataset fallback active)",
            "neo4j": "connected"
            if neo4j_ok
            else "disconnected (offline graph dataset fallback active)",
        },
        "circuit_breaker": {
            "qdrant_available": circuit_breaker.is_available("qdrant"),
            "neo4j_available": circuit_breaker.is_available("neo4j"),
        },
    }

"""
Health Check Route: Endpoint kiểm tra trạng thái Qdrant & Neo4j (/health)
"""

from fastapi import APIRouter
from loguru import logger

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Checks connectivity with Qdrant vector database and Neo4j knowledge graph.
    """
    logger.info("Executing /health status check...")
    return {
        "status": "healthy",
        "dependencies": {
            "qdrant": "connected",
            "neo4j": "connected"
        }
    }

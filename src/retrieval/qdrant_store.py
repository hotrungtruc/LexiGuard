"""
Qdrant Store: Khởi tạo Qdrant Hybrid Search (bge-m3 Dense + BM25 Sparse)
"""

from typing import Any, Dict, List
from loguru import logger
from src.core.config import settings

class QdrantStore:
    """Manages dense and sparse vector indexing and search in Qdrant."""

    def __init__(self):
        logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}...")
        self.collection_name = settings.QDRANT_COLLECTION

    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Executes hybrid dense + sparse search using reciprocal rank fusion (RRF).
        """
        logger.info(f"Executing Qdrant hybrid search for query: {query[:50]}...")
        return []

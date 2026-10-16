"""
Integration Test: Kiểm tra truy vấn Qdrant + Neo4j và Hybrid Retrieval
"""

import pytest
from src.retrieval.hybrid_retriever import HybridRetriever

def test_hybrid_retriever_initialization():
    retriever = HybridRetriever()
    assert retriever is not None
    assert retriever.qdrant_store is not None
    assert retriever.neo4j_store is not None

"""
Unit & Integration Tests: LegalLifecycleManager & Data Lifecycle Synchronization (Phase 3)
===========================================================================================
Verifies:
1. Synchronized statute deactivation (is_active=False) across both Qdrant and Neo4j.
2. In-memory Qdrant payload filtering excluding inactive points when active_only=True.
3. Neo4j Cypher generation with `coalesce(target.is_active, true) = true`.
4. Fast-path and retriever-level automatic filtering of repealed statutes.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    SparseVector,
)

from src.core.schemas import LegalChunk
from src.retrieval.hybrid_retriever import HybridLegalRetriever, LegalCrossEncoderReranker
from src.retrieval.lifecycle_manager import LegalLifecycleManager
from src.retrieval.neo4j_store import Neo4jStore
from src.retrieval.qdrant_store import QdrantStore


@pytest.mark.asyncio
async def test_lifecycle_manager_deactivate_and_activate():
    """Verify LegalLifecycleManager coordinates updates across both stores."""
    mock_qdrant = MagicMock()
    mock_qdrant.update_statute_active_status_async = AsyncMock(return_value=True)

    mock_neo4j = MagicMock()
    mock_neo4j.update_statute_active_status_async = AsyncMock(return_value=128)

    manager = LegalLifecycleManager(
        qdrant_store=mock_qdrant,
        neo4j_store=mock_neo4j,
    )

    # 1. Deactivate statute (e.g. repealed Bo luat Dan su 2005)
    res_deact = await manager.deactivate_statute_async("bo-luat-dan-su-2005")
    assert res_deact["success"] is True
    assert res_deact["is_active"] is False
    assert res_deact["qdrant_updated"] is True
    assert res_deact["neo4j_nodes_updated"] == 128
    mock_qdrant.update_statute_active_status_async.assert_called_once_with(
        "bo-luat-dan-su-2005", False
    )
    mock_neo4j.update_statute_active_status_async.assert_called_once_with(
        "bo-luat-dan-su-2005", False
    )

    # 2. Activate statute
    res_act = await manager.activate_statute_async("bo-luat-dan-su-2015")
    assert res_act["success"] is True
    assert res_act["is_active"] is True


@pytest.mark.asyncio
async def test_qdrant_store_lifecycle_filter_and_payload_update():
    """
    Test real in-memory Qdrant client:
    - Upsert active chunks and older statute chunks
    - Deactivate old statute using update_statute_active_status_async
    - Verify hybrid_search_async with active_only=True filters out old statute
    - Verify hybrid_search_async with active_only=False includes old statute
    """
    client = AsyncQdrantClient(":memory:")
    col_name = "test_lifecycle_collection"

    store = QdrantStore(client=client, collection_name=col_name)
    await store.ensure_collection_exists(dense_dim=4)

    # Chunk from current statute (BLDS 2015)
    chunk_active = LegalChunk(
        chunk_id="VN_BLDS_2015_D357",
        law_id="bo-luat-dan-su-2015",
        law_title="Bộ luật Dân sự 2015",
        chapter_num="XVIII",
        chapter_title="Nghĩa vụ dân sự",
        article_num=357,
        article_title="Nghĩa vụ trả lãi do chậm thực hiện nghĩa vụ trả tiền",
        content="Nghĩa vụ trả lãi do chậm thực hiện nghĩa vụ trả tiền theo BLDS 2015.",
        canonical_citation="Điều 357 Bộ luật Dân sự 2015",
        context_enriched_text="[BLDS 2015] Điều 357: Nghĩa vụ trả lãi",
        cross_references=[],
        is_active=True,
    )

    # Chunk from old repealed statute (BLDS 2005)
    chunk_old = LegalChunk(
        chunk_id="VN_BLDS_2005_D305",
        law_id="bo-luat-dan-su-2005",
        law_title="Bộ luật Dân sự 2005",
        chapter_num="XVII",
        chapter_title="Nghĩa vụ dân sự cũ",
        article_num=305,
        article_title="Trách nhiệm do chậm thực hiện nghĩa vụ dân sự",
        content="Trách nhiệm do chậm thực hiện nghĩa vụ dân sự theo BLDS 2005 cũ.",
        canonical_citation="Điều 305 Bộ luật Dân sự 2005",
        context_enriched_text="[BLDS 2005] Điều 305: Trách nhiệm chậm thực hiện nghĩa vụ",
        cross_references=[],
        is_active=True,  # Initially active
    )

    dense_v = [0.1, 0.9, 0.8, 0.1]
    sparse_v = SparseVector(indices=[1, 2], values=[1.0, 2.0])

    await store.async_upsert_chunks(
        chunks=[chunk_active, chunk_old],
        dense_vectors=[dense_v, dense_v],
        sparse_vectors=[sparse_v, sparse_v],
    )

    # Deactivate the old statute via lifecycle method
    await store.update_statute_active_status_async("bo-luat-dan-su-2005", is_active=False)

    # Search with active_only=True (default) -> Old statute should be excluded
    results_active = await store.hybrid_search_async(
        query="chậm thực hiện nghĩa vụ",
        top_k=5,
        dense_vector=dense_v,
        sparse_vector=sparse_v,
        active_only=True,
    )
    returned_chunk_ids = [r["chunk_id"] for r in results_active]
    assert "VN_BLDS_2015_D357" in returned_chunk_ids
    assert "VN_BLDS_2005_D305" not in returned_chunk_ids

    # Search with active_only=False -> Old statute should be included (historical audit mode)
    results_all = await store.hybrid_search_async(
        query="chậm thực hiện nghĩa vụ",
        top_k=5,
        dense_vector=dense_v,
        sparse_vector=sparse_v,
        active_only=False,
    )
    returned_all_ids = [r["chunk_id"] for r in results_all]
    assert "VN_BLDS_2015_D357" in returned_all_ids
    assert "VN_BLDS_2005_D305" in returned_all_ids

    await store.close()


@pytest.mark.asyncio
async def test_neo4j_cypher_active_only_filtering():
    """Verify Neo4jStore includes `coalesce(target.is_active, true) = true` when active_only=True."""
    mock_session = AsyncMock()
    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=[])
    mock_session.run = AsyncMock(return_value=mock_result)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    store = Neo4jStore(driver=mock_driver)

    # 1. Traversal with active_only=True
    await store.expand_referenced_articles_async("VN_BLDS_2015_D357", active_only=True)
    call_query_active = mock_session.run.call_args[0][0]
    assert "coalesce(target.is_active, true) = true" in call_query_active

    # 2. Traversal with active_only=False
    mock_session.run.reset_mock()
    await store.expand_referenced_articles_async("VN_BLDS_2015_D357", active_only=False)
    call_query_all = mock_session.run.call_args[0][0]
    assert "coalesce(target.is_active, true) = true" not in call_query_all


@pytest.mark.asyncio
async def test_hybrid_retriever_active_only_filters_repealed_provisions():
    """Verify HybridLegalRetriever filters out inactive provisions during end-to-end retrieval."""
    mock_qdrant = MagicMock()
    mock_qdrant.hybrid_search_async = AsyncMock(
        return_value=[
            {
                "chunk_id": "VN_BLDS_2015_D357",
                "neo4j_node_id": "VN_BLDS_2015_D357",
                "canonical_citation": "Điều 357 BLDS 2015",
                "content": "Nội dung Điều 357",
                "is_active": True,
            }
        ]
    )

    mock_neo4j = MagicMock()
    mock_neo4j.expand_referenced_articles_async = AsyncMock(
        return_value={
            "root_id": "VN_BLDS_2015_D357",
            "root_ids": ["VN_BLDS_2015_D357"],
            "root_entities": {
                "VN_BLDS_2015_D357": {
                    "node_id": "VN_BLDS_2015_D357",
                    "name": "Điều 357",
                    "content": "Nội dung",
                    "is_active": True,
                }
            },
            "total_referenced": 1,
            "referenced_articles": [
                {
                    "node_id": "VN_BLDS_2015_D468",
                    "name": "Điều 468",
                    "title": "Lãi suất",
                    "content": "Lãi suất vay...",
                    "is_active": True,
                    "precedents": [],
                }
            ],
        }
    )

    reranker = LegalCrossEncoderReranker(custom_rerank_fn=lambda q, texts: [0.9] * len(texts))
    retriever = HybridLegalRetriever(
        qdrant_store=mock_qdrant,
        neo4j_store=mock_neo4j,
        reranker=reranker,
    )

    results = await retriever.retrieve_async("lãi suất Điều 357", active_only=True)
    assert len(results) == 2
    mock_qdrant.hybrid_search_async.assert_called_once()
    assert mock_qdrant.hybrid_search_async.call_args[1]["active_only"] is True
    assert mock_neo4j.expand_referenced_articles_async.call_args[1]["active_only"] is True

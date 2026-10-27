"""
Integration & Unit Test: HybridLegalRetriever (Qdrant + Neo4j + Cross-Encoder Reranker)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.retrieval.hybrid_retriever import (
    HybridLegalRetriever,
    HybridRetriever,
    LegalCrossEncoderReranker,
)


def test_hybrid_retriever_initialization():
    retriever = HybridLegalRetriever()
    assert retriever is not None
    assert retriever.qdrant_store is not None
    assert retriever.neo4j_store is not None
    assert retriever.reranker is not None
    # Alias check
    assert HybridRetriever is HybridLegalRetriever


def test_cross_encoder_custom_reranker_fn():
    """Verify LegalCrossEncoderReranker sorts candidates based on relevance scores."""
    # Custom scoring: prefers items mentioning 'Điều 468'
    def mock_scorer(query: str, texts: list) -> list:
        return [0.95 if "Điều 468" in t else 0.40 for t in texts]

    reranker = LegalCrossEncoderReranker(custom_rerank_fn=mock_scorer)
    candidates = [
        {"chunk_id": "item1", "content": "Điều 357 chậm trả tiền"},
        {"chunk_id": "item2", "content": "Điều 468 lãi suất tối đa 20%"},
    ]

    reranked = reranker.rerank(query="lãi suất tối đa", candidates=candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["chunk_id"] == "item2"
    assert reranked[0]["rerank_score"] == 0.95
    assert reranked[0]["final_rank"] == 1


@pytest.mark.asyncio
async def test_hybrid_legal_retriever_3_step_pipeline():
    """
    Simulate the complete 3-step retrieval pipeline:
    - Step 1: Qdrant returns candidate (Điều 357)
    - Step 2: Neo4j expands candidate to 1-hop cross-reference (Điều 468 + Án lệ)
    - Step 3: Cross-Encoder reranks and formats context
    """
    mock_qdrant = MagicMock()
    mock_qdrant.hybrid_search_async = AsyncMock(return_value=[
        {
            "chunk_id": "VN_BLDS_2015_D357",
            "neo4j_node_id": "VN_BLDS_2015_D357",
            "canonical_citation": "Điều 357 Bộ luật Dân sự 2015",
            "content": "Nghĩa vụ trả lãi do chậm thực hiện nghĩa vụ trả tiền.",
            "context_enriched_text": "[BLDS 2015] Điều 357: Nghĩa vụ trả lãi do chậm thực hiện nghĩa vụ trả tiền.",
            "cross_references": ["Điều 468"],
        }
    ])

    mock_neo4j = MagicMock()
    mock_neo4j.expand_referenced_articles_async = AsyncMock(return_value={
        "root_id": "VN_BLDS_2015_D357",
        "total_referenced": 1,
        "referenced_articles": [
            {
                "node_id": "VN_BLDS_2015_D468",
                "name": "Điều 468: Lãi suất",
                "title": "Lãi suất",
                "content": "Lãi suất vay do các bên thỏa thuận nhưng không được vượt quá 20%/năm.",
                "citation": "Điều 468 Bộ luật Dân sự 2015",
                "hop_distance": 1,
                "relationship_types": ["THAM_CHIEU_DEN"],
                "relationship_citations": ["quy định tại Điều 468 của Bộ luật này"],
                "precedents": [
                    {
                        "precedent_id": "VN_AL_01_2016",
                        "precedent_number": "Án lệ số 01/2016/AL",
                        "title": "Tranh chấp lãi suất chậm trả",
                        "legal_rule": "Áp dụng Điều 357 và Điều 468 để tính lãi suất quá hạn.",
                    }
                ],
            }
        ],
    })

    # Custom mock reranker
    def mock_rerank(query: str, texts: list) -> list:
        return [0.92 if "468" in t else 0.85 for t in texts]

    reranker = LegalCrossEncoderReranker(custom_rerank_fn=mock_rerank)

    retriever = HybridLegalRetriever(
        qdrant_store=mock_qdrant,
        neo4j_store=mock_neo4j,
        reranker=reranker,
    )

    results = await retriever.retrieve_async(
        query="lãi suất phạt vi phạm và chậm thanh toán",
        top_k=2,
        qdrant_limit=12,
        expansion_hops=2,
    )

    # Both Điều 357 (Qdrant) and Điều 468 (Neo4j Expansion) should be present
    assert len(results) == 2
    assert results[0]["chunk_id"] == "VN_BLDS_2015_D468"  # Higher rerank score
    assert results[0]["retrieval_source"] == "neo4j_expansion"
    assert len(results[0]["precedents"]) == 1

    assert results[1]["chunk_id"] == "VN_BLDS_2015_D357"
    assert results[1]["retrieval_source"] == "qdrant_rrf"

    # Test prompt formatting helper
    formatted_context = retriever.format_context_for_prompt(results)
    assert "CĂN CỨ PHÁP LUẬT VÀ ÁN LỆ THAM CHIẾU" in formatted_context
    assert "Điều 468 Bộ luật Dân sự 2015" in formatted_context
    assert "Án lệ số 01/2016/AL" in formatted_context


@pytest.mark.asyncio
async def test_hybrid_retriever_entity_fast_path():
    """
    Phase 2 Entity Fast-Path Test:
    When a query explicitly contains statutory citations (e.g. 'Điều 357 và Điều 468 BLDS'),
    the retriever parses them before Qdrant search and prepends them directly into
    Neo4j expansion seeds.
    """
    mock_qdrant = MagicMock()
    # Simulate Qdrant returning some generic candidate
    mock_qdrant.hybrid_search_async = AsyncMock(return_value=[
        {
            "chunk_id": "VN_BLDS_2015_D001",
            "neo4j_node_id": "VN_BLDS_2015_D001",
            "canonical_citation": "Điều 1 Bộ luật Dân sự 2015",
            "content": "Phạm vi điều chỉnh của Bộ luật Dân sự.",
            "context_enriched_text": "Điều 1: Phạm vi điều chỉnh",
            "cross_references": [],
        }
    ])

    mock_neo4j = MagicMock()
    mock_neo4j.expand_referenced_articles_async = AsyncMock(return_value={
        "root_id": "VN_BLDS_2015_D357",
        "root_ids": ["VN_BLDS_2015_D357", "VN_BLDS_2015_D468", "VN_BLDS_2015_D001"],
        "root_entities": {
            "VN_BLDS_2015_D357": {
                "node_id": "VN_BLDS_2015_D357",
                "name": "Điều 357: Nghĩa vụ trả lãi do chậm thực hiện nghĩa vụ trả tiền",
                "content": "Nghĩa vụ trả lãi do chậm trả...",
                "law_id": "bo-luat-dan-su-2015",
            },
            "VN_BLDS_2015_D468": {
                "node_id": "VN_BLDS_2015_D468",
                "name": "Điều 468: Lãi suất",
                "content": "Lãi suất vay do các bên thỏa thuận không quá 20%/năm...",
                "law_id": "bo-luat-dan-su-2015",
            },
        },
        "total_referenced": 0,
        "referenced_articles": [],
    })

    def mock_scorer(query: str, texts: list) -> list:
        return [0.98 if "357" in t else (0.95 if "468" in t else 0.3) for t in texts]

    reranker = LegalCrossEncoderReranker(custom_rerank_fn=mock_scorer)

    retriever = HybridLegalRetriever(
        qdrant_store=mock_qdrant,
        neo4j_store=mock_neo4j,
        reranker=reranker,
    )

    query = "Quy định về mức lãi suất theo Điều 357 và Điều 468 BLDS"
    results = await retriever.retrieve_async(
        query=query,
        top_k=2,
    )

    # 1. Verify that Neo4j expand received fast-path seeds as top priority
    assert mock_neo4j.expand_referenced_articles_async.call_count == 1
    call_args, call_kwargs = mock_neo4j.expand_referenced_articles_async.call_args
    passed_seeds = call_kwargs.get("article_or_clause_ids") or call_args[0]
    
    # Must contain both Điều 357 and Điều 468 at the front of seed list
    assert "VN_BLDS_2015_D357" in passed_seeds[:2]
    assert "VN_BLDS_2015_D468" in passed_seeds[:2]

    # 2. Verify results contain the fast-path articles
    assert len(results) == 2
    returned_ids = [r["chunk_id"] for r in results]
    assert "VN_BLDS_2015_D357" in returned_ids
    assert "VN_BLDS_2015_D468" in returned_ids

    # 3. Verify fast_path flag
    for res in results:
        assert res.get("is_fast_path") is True
        assert res.get("retrieval_source") == "entity_fast_path"


"""
Test Suite: Synchronized Schema between Qdrant Vector DB & Neo4j Knowledge Graph
================================================================================
Verifies:
- Deterministic 1:1 correlation: Qdrant payload chunk_id == Neo4j node_id
- Deterministic UUIDv5 generation: Qdrant PointStruct ID == node properties.qdrant_point_id
- SynchronizedLegalEntity bidirectional conversions
- Async QdrantStore collection creation and entity indexing
- Neo4j Cypher MERGE clause correctness
"""

import pytest

from src.core.config import get_async_qdrant_client
from src.core.schemas import (
    DENSE_VECTOR_DIM,
    LegalChunk,
    SynchronizedLegalEntity,
    generate_deterministic_uuid,
)
from src.retrieval.qdrant_store import QdrantStore


@pytest.fixture
def sample_legal_chunk() -> LegalChunk:
    return LegalChunk(
        chunk_id="VN_BLDS_2015_D004_K02",
        law_id="bo-luat-dan-su-2015",
        law_title="Bộ luật Dân sự 2015",
        chapter_num="I",
        chapter_title="Những quy định chung",
        article_num=4,
        article_title="Áp dụng Bộ luật dân sự",
        clause_num=2,
        point_id=None,
        content="Luật khác có liên quan điều chỉnh quan hệ dân sự trong các lĩnh vực cụ thể không được trái với các nguyên tắc cơ bản của pháp luật dân sự quy định tại Điều 3 của Bộ luật này.",
        canonical_citation="Khoản 2 Điều 4 Bộ luật Dân sự 2015",
        context_enriched_text="[Bộ luật Dân sự 2015] > [Chương I: Những quy định chung] > [Điều 4: Áp dụng Bộ luật dân sự]\nKhoản 2: Luật khác có liên quan điều chỉnh...",
        cross_references=["Điều 3"],
        page_start=1,
        page_end=1,
    )


def test_deterministic_uuid_stability():
    """Verify UUIDv5 generation is 100% stable, deterministic and reproducible."""
    uuid_1 = generate_deterministic_uuid("VN_BLDS_2015_D004_K02")
    uuid_2 = generate_deterministic_uuid("VN_BLDS_2015_D004_K02")
    uuid_diff = generate_deterministic_uuid("VN_BLDS_2015_D004_K03")

    assert uuid_1 == uuid_2
    assert uuid_1 != uuid_diff
    assert len(uuid_1) == 36  # Standard UUID string representation length


def test_legal_chunk_qdrant_and_neo4j_alignment(sample_legal_chunk: LegalChunk):
    """Verify LegalChunk generates matching IDs and schemas for both Qdrant and Neo4j."""
    assert sample_legal_chunk.chunk_id == "VN_BLDS_2015_D004_K02"
    assert sample_legal_chunk.neo4j_node_id == sample_legal_chunk.chunk_id
    assert sample_legal_chunk.neo4j_node_type == "Clause"
    assert sample_legal_chunk.qdrant_point_id == generate_deterministic_uuid(
        "VN_BLDS_2015_D004_K02"
    )

    # Qdrant Payload verification
    payload = sample_legal_chunk.to_qdrant_payload()
    assert payload["chunk_id"] == "VN_BLDS_2015_D004_K02"
    assert payload["neo4j_node_id"] == "VN_BLDS_2015_D004_K02"
    assert payload["neo4j_node_type"] == "Clause"
    assert payload["canonical_citation"] == sample_legal_chunk.canonical_citation
    assert payload["cross_references"] == ["Điều 3"]

    # Neo4j GraphNode conversion verification
    graph_node = sample_legal_chunk.to_neo4j_node()
    assert graph_node.node_id == sample_legal_chunk.chunk_id
    assert graph_node.node_type == "Clause"
    assert graph_node.name == sample_legal_chunk.canonical_citation
    assert graph_node.properties["qdrant_point_id"] == sample_legal_chunk.qdrant_point_id


def test_synchronized_legal_entity_bidirectional(sample_legal_chunk: LegalChunk):
    """Verify SynchronizedLegalEntity provides full 2-way conversion without data loss."""
    sync_entity = SynchronizedLegalEntity.from_legal_chunk(sample_legal_chunk)

    assert sync_entity.entity_id == sample_legal_chunk.chunk_id
    assert sync_entity.point_uuid == sample_legal_chunk.qdrant_point_id
    assert sync_entity.entity_type == "Clause"

    # Export to Qdrant Payload
    qdrant_payload = sync_entity.to_qdrant_payload()
    assert qdrant_payload["chunk_id"] == sync_entity.entity_id
    assert qdrant_payload["neo4j_node_id"] == sync_entity.entity_id

    # Export to Neo4j Node
    node = sync_entity.to_graph_node()
    assert node.node_id == sync_entity.entity_id
    assert node.properties["qdrant_point_id"] == sync_entity.point_uuid

    # Cypher generation
    cypher = node.to_cypher_merge()
    assert "MERGE (n:Clause {node_id: 'VN_BLDS_2015_D004_K02'})" in cypher
    assert "n.qdrant_point_id = " in cypher


@pytest.mark.asyncio
async def test_async_qdrant_store_schema_and_upsert(sample_legal_chunk: LegalChunk):
    """Verify QdrantStore initializes with AsyncQdrantClient and indexes synchronized entities."""
    mem_client = get_async_qdrant_client(url=":memory:")
    store = QdrantStore(client=mem_client, collection_name="test_sync_collection")

    # 1. Ensure collection is created with dense + sparse specs
    created = await store.ensure_collection_exists()
    assert created is True

    # 2. Convert chunk to synchronized entity and upsert
    entity = SynchronizedLegalEntity.from_legal_chunk(sample_legal_chunk)
    dummy_dense_vector = [0.05] * DENSE_VECTOR_DIM

    upserted_count = await store.async_upsert_entities(
        entities=[entity],
        dense_vectors=[dummy_dense_vector],
    )
    assert upserted_count == 1

    # 3. Verify retrieved point from memory
    points = await mem_client.retrieve(
        collection_name="test_sync_collection",
        ids=[entity.point_uuid],
        with_payload=True,
    )
    assert len(points) == 1
    assert points[0].id == entity.point_uuid
    assert points[0].payload["chunk_id"] == entity.entity_id
    assert points[0].payload["neo4j_node_id"] == "VN_BLDS_2015_D004_K02"

    await store.close()

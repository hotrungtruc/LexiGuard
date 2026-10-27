"""
Test Suite: Hybrid Qdrant Vector Store (FastEmbed + RRF) & Neo4j Legal Knowledge Graph
======================================================================================
Verifies:
1. FastEmbed ONNX Pipeline:
   - Sparse Vector (BM25): Catches exact keywords/citations ("Điều 127", "45/2019/QH14", "8%")
   - Dense Vector: Captures semantic similarity ("giam lương", "quỹ giữ chân" <-> "khấu trừ tiền lương")
2. Qdrant Hybrid Search:
   - Prefetch(dense) + Prefetch(sparse) + FusionQuery(Fusion.RRF) natively in Qdrant engine
3. Neo4j Legal Knowledge Graph:
   - Entities: (:Law), (:Article), (:Clause), (:Precedent)
   - Function: `expand_referenced_articles` (1..2 hops traversal & precedent linkage)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import SparseVector

from src.core.config import get_async_qdrant_client
from src.core.schemas import (
    GraphNode,
    LegalChunk,
    LegalPrecedent,
)
from src.retrieval.neo4j_store import Neo4jStore
from src.retrieval.qdrant_store import FastEmbedHybridPipeline, QdrantStore

# ==============================================================================
# FASTEMBED ONNX & HYBRID QDRANT TESTS
# ==============================================================================


def test_fastembed_bm25_exact_token_matching():
    """Verify FastEmbed SparseTextEmbedding (Qdrant/bm25) extracts statutory keywords."""
    pipeline = FastEmbedHybridPipeline(sparse_model_name="Qdrant/bm25")

    # Test query containing exact legal codes and numbers
    query = "Điều 127 Bộ luật Lao động số 45/2019/QH14 phạt 8%"
    sparse_vec = pipeline.embed_sparse(query)

    assert isinstance(sparse_vec, SparseVector)
    assert len(sparse_vec.indices) > 0
    assert len(sparse_vec.values) == len(sparse_vec.indices)
    # Check that indices and values are non-empty numeric lists
    assert all(isinstance(i, int) for i in sparse_vec.indices)
    assert all(isinstance(v, float) for v in sparse_vec.values)


@pytest.mark.asyncio
async def test_qdrant_hybrid_search_rrf_in_memory():
    """
    Verify core Qdrant Hybrid Search with Prefetch + FusionQuery(Fusion.RRF)
    using in-memory AsyncQdrantClient.
    """
    mem_client = get_async_qdrant_client(url=":memory:")
    col_name = "test_legal_hybrid_rrf"
    store = QdrantStore(client=mem_client, collection_name=col_name)

    # 1. Ensure collection exists with dual dense + sparse specs
    await store.ensure_collection_exists(dense_dim=4)

    # 2. Prepare sample legal chunks with distinct semantic and keyword properties:
    # Chunk A: Semantic focus (salary deductions and penalty amounts).
    chunk_a = LegalChunk(
        chunk_id="VN_BLLD_2019_D127_K01",
        law_id="bo-luat-lao-dong-2019",
        law_title="Bộ luật Lao động 2019",
        chapter_num="VIII",
        chapter_title="Kỷ luật lao động",
        article_num=127,
        article_title="Các hành vi bị nghiêm cấm khi xử lý kỷ luật lao động",
        clause_num=1,
        content="Xâm phạm sức khỏe, danh dự, tính mạng, uy tín, nhân phẩm của người lao động.",
        canonical_citation="Khoản 1 Điều 127 Bộ luật Lao động 2019",
        context_enriched_text="[BLLD 2019] Điều 127: Các hành vi bị nghiêm cấm. Khoản 1: Xâm phạm sức khỏe...",
        cross_references=[],
    )

    # Chunk B: Exact statutory keyword focus (salary deductions and monetary penalties).
    chunk_b = LegalChunk(
        chunk_id="VN_BLLD_2019_D127_K02",
        law_id="bo-luat-lao-dong-2019",
        law_title="Bộ luật Lao động 2019",
        chapter_num="VIII",
        chapter_title="Kỷ luật lao động",
        article_num=127,
        article_title="Các hành vi bị nghiêm cấm khi xử lý kỷ luật lao động",
        clause_num=2,
        content="Phạt tiền, cắt lương thay việc xử lý kỷ luật lao động.",
        canonical_citation="Khoản 2 Điều 127 Bộ luật Lao động 2019",
        context_enriched_text="[BLLD 2019] Điều 127: Khoản 2: Phạt tiền, cắt lương thay việc xử lý kỷ luật.",
        cross_references=[],
    )

    # Explicit mock dense and sparse vectors for controlled RRF verification
    dense_a = [0.8, 0.2, 0.1, 0.1]
    dense_b = [0.1, 0.9, 0.8, 0.1]
    sparse_a = SparseVector(indices=[101, 202], values=[1.5, 0.8])
    sparse_b = SparseVector(indices=[101, 303, 404], values=[1.2, 2.5, 1.0])

    await store.async_upsert_chunks(
        chunks=[chunk_a, chunk_b],
        dense_vectors=[dense_a, dense_b],
        sparse_vectors=[sparse_a, sparse_b],
    )

    # 3. Query with Prefetch + Fusion.RRF
    query_dense = [0.1, 0.85, 0.8, 0.1]  # Closest to chunk_b
    query_sparse = SparseVector(indices=[303, 404], values=[2.0, 1.0])

    results = await store.hybrid_search_async(
        query="giam lương cắt lương Điều 127",
        top_k=2,
        prefetch_limit=5,
        dense_vector=query_dense,
        sparse_vector=query_sparse,
    )

    assert len(results) == 2
    # Chunk B should rank #1 due to both dense match and sparse token overlap
    top_result = results[0]
    assert top_result["chunk_id"] == "VN_BLLD_2019_D127_K02"
    assert "Khoản 2 Điều 127" in top_result["canonical_citation"]
    assert top_result["neo4j_node_id"] == "VN_BLLD_2019_D127_K02"
    assert top_result["score"] is not None

    await store.close()


# ==============================================================================
# NEO4J LEGAL KNOWLEDGE GRAPH & MULTI-HOP EXPANSION TESTS
# ==============================================================================


def test_legal_precedent_schema():
    """Verify LegalPrecedent schema and conversion to :Precedent GraphNode."""
    precedent = LegalPrecedent(
        precedent_id="VN_AL_01_2016",
        precedent_number="Án lệ số 01/2016/AL",
        title="Vụ án tranh chấp hợp đồng mua bán tài sản",
        court="Hội đồng Thẩm phán Tòa án nhân dân tối cao",
        promulgation_date="2016-04-06",
        case_summary="Tranh chấp về hiệu lực thỏa thuận phạt cọc và bồi thường thiệt hại.",
        legal_rule="Nếu bên đặt cọc từ chối giao kết hợp đồng thì mất tiền đặt cọc theo Điều 328 BLDS.",
        applied_statutes=["VN_BLDS_2015_D328"],
    )

    node = precedent.to_graph_node()
    assert node.node_id == "VN_AL_01_2016"
    assert node.node_type == "Precedent"
    assert "Án lệ số 01/2016/AL" in node.name
    assert node.properties["applied_statutes"] == ["VN_BLDS_2015_D328"]
    assert "Điều 328" in node.properties["legal_rule"]


@pytest.mark.asyncio
async def test_neo4j_expand_referenced_articles_cypher_traversal():
    """
    Verify `expand_referenced_articles_async` traverses 1..2 hops
    and links both referenced articles and judicial precedents.
    """
    # Mock AsyncDriver and session
    mock_session = AsyncMock()
    mock_result = AsyncMock()

    # Simulated records returned from 1..2 hops Cypher traversal
    mock_records = [
        {
            "start_id": "VN_BLDS_2015_D004_K02",
            "start_name": "Khoản 2 Điều 4 Bộ luật Dân sự 2015",
            "start_content": "Luật khác có liên quan điều chỉnh quan hệ dân sự...",
            "start_law_id": "bo-luat-dan-su-2015",
            "referenced_id": "VN_BLDS_2015_D003",
            "referenced_name": "Điều 3: Các nguyên tắc cơ bản của pháp luật dân sự",
            "referenced_title": "Các nguyên tắc cơ bản của pháp luật dân sự",
            "referenced_content": "Mọi cá nhân, pháp nhân đều bình đẳng...",
            "referenced_citation": "Điều 3 Bộ luật Dân sự 2015",
            "referenced_law_id": "bo-luat-dan-su-2015",
            "target_labels": ["Article"],
            "hop_distance": 1,
            "relationship_types": ["THAM_CHIEU_DEN"],
            "relationship_citations": ["quy định tại Điều 3 của Bộ luật này"],
            "precedents": [
                {
                    "precedent_id": "VN_AL_43_2021",
                    "precedent_number": "Án lệ số 43/2021/AL",
                    "title": "Hiệu lực hợp đồng thế chấp quyền sử dụng đất",
                    "legal_rule": "Nguyên tắc thiện chí, trung thực theo Điều 3 BLDS",
                    "court": "TAND Tối cao",
                }
            ],
        }
    ]

    mock_result.data = AsyncMock(return_value=mock_records)
    mock_session.run = AsyncMock(return_value=mock_result)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    store = Neo4jStore(driver=mock_driver)

    # Execute expansion up to 2 hops
    expansion = await store.expand_referenced_articles_async(
        article_or_clause_id="VN_BLDS_2015_D004_K02",
        hops=2,
        include_precedents=True,
    )

    assert expansion["root_id"] == "VN_BLDS_2015_D004_K02"
    assert expansion["total_referenced"] == 1
    assert len(expansion["referenced_articles"]) == 1

    ref = expansion["referenced_articles"][0]
    assert ref["node_id"] == "VN_BLDS_2015_D003"
    assert ref["title"] == "Các nguyên tắc cơ bản của pháp luật dân sự"
    assert ref["hop_distance"] == 1
    assert ref["relationship_types"] == ["THAM_CHIEU_DEN"]

    # Verify precedent attachment
    assert len(ref["precedents"]) == 1
    prec = ref["precedents"][0]
    assert prec["precedent_id"] == "VN_AL_43_2021"
    assert prec["precedent_number"] == "Án lệ số 43/2021/AL"


@pytest.mark.asyncio
async def test_neo4j_expand_referenced_articles_batch_unwind():
    """
    Phase 1 Latency Optimization Test:
    Verify `expand_referenced_articles_async` accepts a List[str] of root_ids,
    generates a single Cypher query utilizing `UNWIND $root_ids AS root_id`,
    and populates `parent_referencing_node` and `root_entities`.
    """
    mock_session = AsyncMock()
    mock_result = AsyncMock()

    # Simulated multi-root traversal records returned from UNWIND
    mock_records = [
        {
            "start_id": "VN_BLDS_2015_D357",
            "start_name": "Điều 357: Nghĩa vụ trả lãi",
            "start_content": "Nghĩa vụ trả lãi do chậm trả...",
            "start_law_id": "bo-luat-dan-su-2015",
            "referenced_id": "VN_BLDS_2015_D468",
            "referenced_name": "Điều 468: Lãi suất",
            "referenced_title": "Lãi suất",
            "referenced_content": "Lãi suất vay do các bên thỏa thuận...",
            "referenced_citation": "Điều 468 Bộ luật Dân sự 2015",
            "referenced_law_id": "bo-luat-dan-su-2015",
            "target_labels": ["Article"],
            "hop_distance": 1,
            "relationship_types": ["THAM_CHIEU_DEN"],
            "relationship_citations": ["quy định tại Điều 468"],
            "precedents": [],
        },
        {
            "start_id": "VN_BLDS_2015_D004_K02",
            "start_name": "Khoản 2 Điều 4",
            "start_content": "Luật khác có liên quan...",
            "start_law_id": "bo-luat-dan-su-2015",
            "referenced_id": "VN_BLDS_2015_D003",
            "referenced_name": "Điều 3: Nguyên tắc cơ bản",
            "referenced_title": "Nguyên tắc cơ bản",
            "referenced_content": "Mọi cá nhân bình đẳng...",
            "referenced_citation": "Điều 3",
            "referenced_law_id": "bo-luat-dan-su-2015",
            "target_labels": ["Article"],
            "hop_distance": 1,
            "relationship_types": ["THAM_CHIEU_DEN"],
            "relationship_citations": ["quy định tại Điều 3"],
            "precedents": [
                {
                    "precedent_id": "VN_AL_43_2021",
                    "precedent_number": "Án lệ số 43/2021/AL",
                    "title": "Hiệu lực hợp đồng",
                    "legal_rule": "Nguyên tắc thiện chí",
                    "court": "TAND Tối cao",
                }
            ],
        },
    ]

    mock_result.data = AsyncMock(return_value=mock_records)
    mock_session.run = AsyncMock(return_value=mock_result)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    store = Neo4jStore(driver=mock_driver)

    seed_ids = ["VN_BLDS_2015_D357", "VN_BLDS_2015_D004_K02"]

    # 1. Execute batch expansion with list of seed IDs
    expansion = await store.expand_referenced_articles_async(
        article_or_clause_ids=seed_ids,
        hops=2,
        include_precedents=True,
    )

    # Verify single consolidated query executed with UNWIND
    assert mock_session.run.call_count == 1
    call_args, call_kwargs = mock_session.run.call_args
    executed_query = call_args[0]
    executed_params = call_args[1]

    assert "UNWIND $root_ids AS root_id" in executed_query
    assert executed_params["root_ids"] == seed_ids

    # Verify return structure
    assert expansion["root_ids"] == seed_ids
    assert len(expansion["root_entities"]) == 2
    assert "VN_BLDS_2015_D357" in expansion["root_entities"]
    assert "VN_BLDS_2015_D004_K02" in expansion["root_entities"]

    assert expansion["total_referenced"] == 2
    refs = expansion["referenced_articles"]
    assert len(refs) == 2

    # Check first referenced item provenance
    assert refs[0]["node_id"] == "VN_BLDS_2015_D468"
    assert refs[0]["parent_referencing_node"] == "VN_BLDS_2015_D357"

    # Check second referenced item provenance & precedent
    assert refs[1]["node_id"] == "VN_BLDS_2015_D003"
    assert refs[1]["parent_referencing_node"] == "VN_BLDS_2015_D004_K02"
    assert len(refs[1]["precedents"]) == 1
    assert refs[1]["precedents"][0]["precedent_id"] == "VN_AL_43_2021"


@pytest.mark.asyncio
async def test_neo4j_ingest_precedent():
    """Verify `async_ingest_precedent` issues expected MERGE and relationship queries."""
    mock_session = AsyncMock()
    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=[])
    mock_session.run = AsyncMock(return_value=mock_result)

    mock_driver = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_driver.session = MagicMock(return_value=mock_session)

    store = Neo4jStore(driver=mock_driver)

    precedent = LegalPrecedent(
        precedent_id="VN_AL_02_2016",
        precedent_number="Án lệ số 02/2016/AL",
        title="Tranh chấp đòi lại tài sản chuyển giao",
        case_summary="Tóm tắt án lệ số 02",
        legal_rule="Quy tắc hoàn trả tài sản",
        applied_statutes=["VN_BLDS_2015_D166"],
    )

    await store.async_ingest_precedent(precedent)
    # Ensure session.run was called for both node creation and edge linking
    assert mock_session.run.call_count >= 2


@pytest.mark.asyncio
async def test_neo4j_execute_write_and_read_managed_transactions():
    """Verify execute_write and execute_read properly route via managed transaction functions."""
    mock_tx = AsyncMock()
    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=[{"count": 42}])
    mock_tx.run = AsyncMock(return_value=mock_result)

    async def fake_execute_write(tx_func):
        return await tx_func(mock_tx)

    async def fake_execute_read(tx_func):
        return await tx_func(mock_tx)

    mock_session = MagicMock()
    mock_session.execute_write = AsyncMock(side_effect=fake_execute_write)
    mock_session.execute_read = AsyncMock(side_effect=fake_execute_read)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)

    store = Neo4jStore(driver=mock_driver)

    # Test execute_write
    write_res = await store.execute_write("MERGE (n:Article {node_id: 'TEST'})")
    assert write_res == [{"count": 42}]
    assert mock_session.execute_write.called
    assert mock_tx.run.called

    # Test execute_read
    read_res = await store.execute_read("MATCH (n:Article) RETURN count(n) AS count")
    assert read_res == [{"count": 42}]
    assert mock_session.execute_read.called


@pytest.mark.asyncio
async def test_qdrant_payload_indices_creation():
    """Verify ensure_payload_indices issues create_payload_index for all key fields."""
    mock_client = AsyncMock()
    mock_client.create_payload_index = AsyncMock()

    store = QdrantStore(client=mock_client, collection_name="test_col")
    await store.ensure_payload_indices()

    assert mock_client.create_payload_index.call_count == 7
    indexed_fields = [
        call[1]["field_name"] for call in mock_client.create_payload_index.call_args_list
    ]
    assert "law_id" in indexed_fields
    assert "article_num" in indexed_fields
    assert "chapter_num" in indexed_fields
    assert "clause_num" in indexed_fields
    assert "neo4j_node_id" in indexed_fields
    assert "neo4j_node_type" in indexed_fields
    assert "is_active" in indexed_fields


def test_phase_3_entity_extractor_multi_statute_scope_resolution():
    """Verify LegalEntityExtractor correctly parses distinct law codes in mixed queries."""
    from src.retrieval.entity_extractor import LegalEntityExtractor

    extractor = LegalEntityExtractor()

    query = (
        "Căn cứ Khoản 1 Điều 357 Bộ luật Dân sự 2015 và Điều 301 Luật Thương mại 2005 để giải quyết"
    )
    citations = extractor.extract_citations(query)

    assert len(citations) >= 2
    cit_map = {c.article_num: c for c in citations}

    assert 357 in cit_map
    assert cit_map[357].law_code == "VN_BLDS_2015"
    assert cit_map[357].canonical_node_id == "VN_BLDS_2015_D357_K01"

    assert 301 in cit_map
    assert cit_map[301].law_code == "VN_LTM_2005"
    assert cit_map[301].canonical_node_id == "VN_LTM_2005_D301"


def test_phase_3_graph_node_cypher_serialization_with_lists():
    """Verify GraphNode.to_cypher_merge properly serializes list attributes and params."""
    node = GraphNode(
        node_id="VN_AL_09_2016",
        node_type="Precedent",
        name="Án lệ 09",
        law_id="an-le",
        properties={
            "precedent_number": "09/2016/AL",
            "applied_statutes": ["VN_BLDS_2015_D305", "VN_LTM_2005_D301"],
            "is_binding": True,
        },
    )

    merge_cypher = node.to_cypher_merge()
    assert "MERGE (n:Precedent {node_id: 'VN_AL_09_2016'})" in merge_cypher
    assert "n.applied_statutes = ['VN_BLDS_2015_D305', 'VN_LTM_2005_D301']" in merge_cypher
    assert "n.is_binding = true" in merge_cypher

    params = node.to_cypher_params()
    assert params["node_id"] == "VN_AL_09_2016"
    assert params["props"]["applied_statutes"] == ["VN_BLDS_2015_D305", "VN_LTM_2005_D301"]


def test_phase_3_dynamic_law_code_resolution_no_collision():
    """Verify legal_chunker and graph_extractor resolve unique prefixes without generic VN_STATUTE collision."""
    from src.ingestion.graph_extractor import LegalGraphExtractor
    from src.ingestion.legal_chunker import LegalChunker

    chunker = LegalChunker
    extractor = LegalGraphExtractor

    code_dd = chunker._get_law_code("luat-dat-dai")
    code_no = chunker._get_law_code("luat-nha-o")
    code_tt = chunker._get_law_code("luat-trong-tai-thuong-mai")
    code_custom = chunker._get_law_code("luat-an-ninh-mang-2018")

    assert code_dd == "VN_LDD_2024"
    assert code_no == "VN_LNO_2023"
    assert code_tt == "VN_LTTTM_2010"
    assert code_custom == "VN_LANM_2018"
    assert len({code_dd, code_no, code_tt, code_custom}) == 4

    assert extractor._get_law_code("luat-dat-dai") == "VN_LDD_2024"
    assert extractor._get_law_code("luat-nha-o") == "VN_LNO_2023"

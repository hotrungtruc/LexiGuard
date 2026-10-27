"""
Unit Tests: Domain-Agnostic Legal Generalization
=================================================
Verifies that LexiGuard is completely domain-agnostic and functions dynamically
for ANY legal corpus (e.g., Cybersecurity Law, Personal Data Protection Decree,
Internal Corporate Financial SOPs, IP Law), without hardcoded statutory assumptions.
"""

import pytest
from src.core.legal_norm import canonical_law_slug, register_custom_law
from src.core.schemas import LegalArticle, LegalChapter, LegalDocument, LegalChunk
from src.ingestion.legal_chunker import LegalChunker
from src.ingestion.graph_extractor import LegalGraphExtractor
from src.retrieval.entity_extractor import LegalEntityExtractor
from src.mcp_server.legal_tools import FastMCPFilters
from src.agents.nodes import _extract_explicit_law_code, _is_citation_grounded


def test_universal_slug_generation_across_varied_domains():
    """Verify that any statute, decree, circular, or internal rulebook gets a canonical slug."""
    # 1. Cybersecurity Law
    assert canonical_law_slug("luat-an-ninh-mang-2018") == "VN_LANM_2018"
    assert canonical_law_slug("Luật An ninh mạng 2018") == "VN_LANM_2018"

    # 2. Decree on Personal Data Protection
    assert canonical_law_slug("nghi-dinh-13-2023-nd-cp") == "VN_ND13_2023"
    assert canonical_law_slug("Nghị định 13/2023/NĐ-CP") == "VN_ND13_2023"

    # 3. Intellectual Property Law
    assert canonical_law_slug("luat-so-huu-tri-tue-2022") == "VN_LSHTT_2022"

    # 4. Pharmaceutical Law
    slug_duoc = canonical_law_slug("Luật Dược 2016")
    assert "DUOC" in slug_duoc and "2016" in slug_duoc

    # 5. Internal Corporate Governance / Financial SOP
    slug_qc = canonical_law_slug("Quy chế Tài chính nội bộ 2024")
    assert slug_qc.startswith("VN_QC") and "2024" in slug_qc

    # 6. Circular
    slug_tt = canonical_law_slug("Thông tư 10/2020/TT-BLDTBXH")
    assert slug_tt.startswith("VN_TT") and "2020" in slug_tt

    # 7. Already canonical code
    assert canonical_law_slug("VN_CUSTOM_CORP_2025") == "VN_CUSTOM_CORP_2025"


def test_chunking_arbitrary_statute_produces_deterministic_ids():
    """Test chunking an arbitrary new legal document (Cybersecurity Law)."""
    chunker = LegalChunker()
    cyber_statute_text = """
CHƯƠNG I: NHỮNG QUY ĐỊNH CHUNG

Điều 8. Các hành vi bị nghiêm cấm về an ninh mạng
1. Sử dụng không gian mạng để thực hiện hành vi vi phạm pháp luật.
2. Phát tán thông tin sai sự thật xâm hại an ninh quốc gia.
"""
    chunks = chunker.chunk_statute(
        cyber_statute_text,
        law_id="luat-an-ninh-mang-2018",
        law_title="Luật An ninh mạng 2018",
    )
    assert len(chunks) == 2  # 2 clauses

    # Check deterministic chunk IDs
    assert chunks[0].chunk_id == "VN_LANM_2018_D008_K01"
    assert chunks[1].chunk_id == "VN_LANM_2018_D008_K02"
    assert chunks[0].article_num == 8
    assert "Luật An ninh mạng 2018" in chunks[0].context_enriched_text
    assert chunks[0].neo4j_node_id == "VN_LANM_2018_D008_K01"


def test_graph_extraction_on_arbitrary_statute_with_dynamic_cross_references():
    """Test graph extraction for arbitrary statutes with cross-statute references."""
    extractor = LegalGraphExtractor()
    chunk = LegalChunk(
        chunk_id="VN_LANM_2018_D008_K01",
        law_id="luat-an-ninh-mang-2018",
        law_title="Luật An ninh mạng 2018",
        chapter_num="I",
        chapter_title="Những quy định chung",
        article_num=8,
        article_title="Các hành vi bị nghiêm cấm",
        clause_num=1,
        content="Nghiêm cấm hành vi theo quy định của Luật An toàn thông tin mạng và Điều 12.",
        canonical_citation="Khoản 1 Điều 8 Luật An ninh mạng 2018",
        context_enriched_text="Context",
        cross_references=["Điều 12", "Luật An toàn thông tin mạng"]
    )

    graph_result = extractor.extract_from_chunks([chunk], law_id="luat-an-ninh-mang-2018", law_title="Luật An ninh mạng 2018")

    node_ids = [n.node_id for n in graph_result.nodes]
    # Verify root law node
    assert "VN_LANM_2018" in node_ids
    # Verify article node
    assert "VN_LANM_2018_D008" in node_ids
    # Verify clause node
    assert "VN_LANM_2018_D008_K01" in node_ids

    # Verify cross-statute node for Luật An toàn thông tin mạng was created dynamically
    law_nodes = [nid for nid in node_ids if "LATTTM" in nid or "AN_TOAN" in nid]
    assert len(law_nodes) >= 1, "Cross-statute node must be created dynamically"


def test_entity_extractor_dynamic_scope_resolution_on_new_statutes():
    """Verify that LegalEntityExtractor dynamically parses citations of any arbitrary law."""
    extractor = LegalEntityExtractor()

    # Query citing Cybersecurity Law
    query_cyber = "Căn cứ Khoản 1 Điều 8 Luật An ninh mạng 2018 để xử lý vi phạm"
    citations_cyber = extractor.extract_citations(query_cyber)
    assert len(citations_cyber) == 1
    assert citations_cyber[0].law_code == "VN_LANM_2018"
    assert citations_cyber[0].canonical_node_id == "VN_LANM_2018_D008_K01"
    assert citations_cyber[0].article_node_id == "VN_LANM_2018_D008"

    # Query citing Decree 13/2023 on Data Protection
    query_decree = "Theo Điều 4 Nghị định 13/2023/NĐ-CP về nguyên tắc bảo vệ dữ liệu"
    citations_decree = extractor.extract_citations(query_decree)
    assert len(citations_decree) == 1
    assert citations_decree[0].law_code == "VN_ND13_2023"
    assert citations_decree[0].canonical_node_id == "VN_ND13_2023_D004"

    # Query citing Corporate Internal Financial Bylaw
    query_bylaw = "Chiếu theo Điều 15 Quy chế Tài chính nội bộ 2024 của công ty"
    citations_bylaw = extractor.extract_citations(query_bylaw)
    assert len(citations_bylaw) == 1
    assert "VN_QC" in citations_bylaw[0].law_code
    assert citations_bylaw[0].article_num == 15


def test_fastmcp_filters_dynamic_extraction_on_arbitrary_laws():
    """Verify FastMCPFilters dynamically extracts filters for arbitrary legislation."""
    filters = FastMCPFilters.extract_filters("Căn cứ Điều 8 Luật An ninh mạng 2018")
    assert filters["law_id"] == "VN_LANM_2018"
    assert filters["article_num"] == 8

    filters_decree = FastMCPFilters.extract_filters("Khoản 2 Điều 13 Nghị định 13/2023/NĐ-CP")
    assert filters_decree["law_id"] == "VN_ND13_2023"
    assert filters_decree["article_num"] == 13
    assert filters_decree["clause_num"] == 2


def test_anti_hallucination_grading_with_dynamic_arbitrary_statutes():
    """Verify that hallucination verification works for arbitrary laws without false rejects."""
    retrieved_statutes = [
        {
            "chunk_id": "VN_LANM_2018_D008",
            "canonical_citation": "Điều 8 Luật An ninh mạng 2018",
            "law_title": "Luật An ninh mạng 2018",
            "law_id": "luat-an-ninh-mang-2018",
            "article_num": 8,
            "content": "Các hành vi bị nghiêm cấm về an ninh mạng.",
        },
        {
            "chunk_id": "VN_ND13_2023_D004",
            "canonical_citation": "Điều 4 Nghị định 13/2023/NĐ-CP",
            "law_title": "Nghị định 13/2023/NĐ-CP",
            "law_id": "nghi-dinh-13-2023-nd-cp",
            "article_num": 4,
            "content": "Nguyên tắc bảo vệ dữ liệu cá nhân.",
        }
    ]

    # Valid citation from newly ingested law (Luật An ninh mạng 2018)
    assert _is_citation_grounded(
        cit_text="Điều 8 Luật An ninh mạng 2018",
        article_id="VN_LANM_2018_D008",
        retrieved_statutes=retrieved_statutes,
    ) is True

    # Valid citation from newly ingested decree (Nghị định 13/2023/NĐ-CP)
    assert _is_citation_grounded(
        cit_text="Điều 4 Nghị định 13/2023/NĐ-CP",
        article_id="VN_ND13_2023_D004",
        retrieved_statutes=retrieved_statutes,
    ) is True

    # Hallucinated citation: Citing Điều 8 BLDS instead of Điều 8 LANM
    assert _is_citation_grounded(
        cit_text="Điều 8 Bộ luật Dân sự 2015",
        article_id=None,
        retrieved_statutes=retrieved_statutes,
    ) is False, "Cross-statute hallucination must be rejected"

    # Fabricated article number in valid law
    assert _is_citation_grounded(
        cit_text="Điều 99 Luật An ninh mạng 2018",
        article_id=None,
        retrieved_statutes=retrieved_statutes,
    ) is False, "Non-existent article in valid statute must be rejected"

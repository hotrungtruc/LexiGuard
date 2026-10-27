"""
Unit Test: Kiểm tra logic phân đoạn phân cấp Điều/Khoản và Bóc tách Đồ thị Pháp lý
"""

from src.core.schemas import (
    GraphRelationType,
    LegalChunk,
    LegalDocument,
    LegalGraphExtractionResult,
)
from src.ingestion.graph_extractor import LegalGraphExtractor
from src.ingestion.legal_chunker import LegalChunker

SAMPLE_STATUTE_TEXT = """
PHẦN THỨ NHẤT: QUY ĐỊNH CHUNG

CHƯƠNG I: NHỮNG QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
1. Bộ luật này quy định địa vị pháp lý, chuẩn mực pháp lý về cách ứng xử của cá nhân, pháp nhân.
2. Quyền dân sự được bảo vệ theo quy định của Hiến pháp và pháp luật.

Điều 2. Công nhận và bảo vệ quyền dân sự
1. Quyền dân sự được công nhận theo quy định tại Điều 1 của Bộ luật này.
2. Trừ trường hợp quy định tại Điều 3 của Bộ luật này, quyền dân sự chỉ bị hạn chế theo luật.

CHƯƠNG II: XÁC LẬP VÀ THỰC HIỆN QUYỀN

Điều 3. Các nguyên tắc cơ bản
1. Mọi cá nhân, pháp nhân đều bình đẳng theo quy định tại khoản 1 Điều 1.
a) Tự do cam kết, thỏa thuận.
b) Không vi phạm điều cấm của luật.
2. Việc áp dụng tập quán được thực hiện theo quy định của Bộ luật Dân sự.
"""


def test_legal_chunker_initialization():
    chunker = LegalChunker()
    assert chunker is not None


def test_chunk_statute_hierarchical_parsing():
    chunker = LegalChunker()
    doc = chunker.parse_statute(
        SAMPLE_STATUTE_TEXT, law_id="bo-luat-dan-su-2015", law_title="Bộ luật Dân sự 2015"
    )

    assert isinstance(doc, LegalDocument)
    assert doc.law_id == "bo-luat-dan-su-2015"
    assert doc.total_articles == 3
    assert len(doc.chapters) >= 2

    # Check Article 1
    chap1 = doc.chapters[0]
    art1 = chap1.articles[0]
    assert art1.article_num == 1
    assert "Phạm vi điều chỉnh" in art1.title
    assert len(art1.clauses) == 2
    assert art1.clauses[0].clause_num == 1
    assert "Khoản 1 Điều 1" in art1.clauses[0].canonical_citation


def test_chunk_statute_granular_chunks():
    chunker = LegalChunker()
    chunks = chunker.chunk_statute(
        SAMPLE_STATUTE_TEXT, law_id="bo-luat-dan-su-2015", law_title="Bộ luật Dân sự 2015"
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, LegalChunk)
        assert chunk.chunk_id.startswith("VN_BLDS_2015_")
        assert chunk.law_id == "bo-luat-dan-su-2015"
        assert chunk.chapter_num in ["I", "II"]
        # Context enrichment validation
        assert "Bộ luật Dân sự 2015" in chunk.context_enriched_text
        assert "Chương" in chunk.context_enriched_text
        assert "Điều" in chunk.context_enriched_text
        assert chunk.token_estimate > 0


def test_graph_extractor_triples_and_cypher():
    chunker = LegalChunker()
    extractor = LegalGraphExtractor()

    chunks = chunker.chunk_statute(
        SAMPLE_STATUTE_TEXT, law_id="bo-luat-dan-su-2015", law_title="Bộ luật Dân sự 2015"
    )
    result = extractor.extract_from_chunks(
        chunks, law_id="bo-luat-dan-su-2015", law_title="Bộ luật Dân sự 2015"
    )

    assert isinstance(result, LegalGraphExtractionResult)
    assert result.total_nodes > 0
    assert result.total_edges > 0

    # Verify structural CONTAINS edges
    contains_edges = [e for e in result.edges if e.relation == GraphRelationType.CONTAINS]
    assert len(contains_edges) > 0

    # Verify semantic citation: Article 2 references Article 1.
    tham_chieu_edges = [e for e in result.edges if e.relation == GraphRelationType.THAM_CHIEU_DEN]
    assert len(tham_chieu_edges) > 0

    # Verify exception citation: Article 2 is an exception to Article 3.
    ngoai_le_edges = [e for e in result.edges if e.relation == GraphRelationType.NGOAI_LE_CUA]
    assert len(ngoai_le_edges) > 0
    assert any("D003" in e.target_id for e in ngoai_le_edges)

    # Verify Cypher generation
    cypher_queries = extractor.to_cypher_statements(result)
    assert len(cypher_queries) > 0
    assert any("MERGE (n:Article" in q for q in cypher_queries)
    assert any("MERGE (s)-[r:" in q for q in cypher_queries)


def test_ocr_markdown_bullet_and_tight_clause_parsing():
    """Validates that clauses prefixed with markdown bullets or missing space after dot are parsed without loss."""
    raw_ocr_text = """
## Điều 4. Áp dụng Bộ luật dân sự

- 1.Bộ luật này là luật chung điều chỉnh các quan hệ dân sự.
2. Luật khác có liên quan điều chỉnh quan hệ dân sự trong các lĩnh vực cụ thể không được trái với các nguyên tắc cơ bản.
3. Trường hợp luật khác có liên quan không quy định thì quy định của Bộ luật này được áp dụng.
4. Trường hợp có sự khác nhau giữa quy định của Bộ luật này và điều ước quốc tế thì áp dụng điều ước quốc tế.
"""
    chunker = LegalChunker()
    doc = chunker.parse_statute(raw_ocr_text, law_id="blds-2015", law_title="Bộ luật Dân sự 2015")
    art4 = doc.chapters[0].articles[0]

    assert art4.article_num == 4
    assert len(art4.clauses) == 4, f"Expected 4 clauses, got {len(art4.clauses)}"
    assert art4.clauses[0].clause_num == 1
    assert "Bộ luật này là luật chung" in art4.clauses[0].content
    assert art4.clauses[1].clause_num == 2
    assert art4.clauses[2].clause_num == 3
    assert art4.clauses[3].clause_num == 4

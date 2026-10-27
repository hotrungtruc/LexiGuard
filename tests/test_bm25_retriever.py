"""
Unit Tests: BM25 Lexical Legal Retriever (Compound N-grams & Denoising)
======================================================================
Validates:
1. Vietnamese legal tokenization (unigram + bigram compound words).
2. Query denoising (stripping colloquial / competition tail noise).
3. BM25Index core ranking accuracy & IDF smoothing.
4. BM25LexicalRetriever metadata filtering (law_id, article_num, active_only).
5. Grounding prompt context formatting.
"""

from src.retrieval.bm25_retriever import (
    BM25Index,
    BM25LexicalRetriever,
    denoise_query,
    tokenize_vietnamese_legal,
)


def test_denoise_query():
    # Should strip competition suffixes
    q1 = "Hợp đồng đặt cọc có bắt buộc công chứng đúng hay sai?"
    assert denoise_query(q1) == "Hợp đồng đặt cọc có bắt buộc công chứng"

    q2 = "Người lao động nghỉ việc không báo trước như vậy đúng không?"
    assert denoise_query(q2) == "Người lao động nghỉ việc không báo trước"

    q3 = "Theo bạn đúng hay sai?"
    assert denoise_query(q3) == ""

    # Normal queries without tail noise should remain intact
    q4 = "Quy định về thời hiệu khởi kiện tranh chấp hợp đồng"
    assert denoise_query(q4) == "Quy định về thời hiệu khởi kiện tranh chấp hợp đồng"


def test_tokenize_vietnamese_legal():
    text = "Bồi thường thiệt hại hợp đồng"
    tokens = tokenize_vietnamese_legal(text)
    # Check unigrams
    assert "bồi" in tokens
    assert "thường" in tokens
    assert "thiệt" in tokens
    assert "hại" in tokens
    # Check compound bigrams
    assert "bồi_thường" in tokens
    assert "thiệt_hại" in tokens
    assert "hợp_đồng" in tokens


def test_bm25_index_ranking():
    docs = [
        {
            "id": "d1",
            "content": "Thời hiệu khởi kiện yêu cầu bồi thường thiệt hại do vi phạm hợp đồng là 03 năm.",
        },
        {
            "id": "d2",
            "content": "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động nhưng phải báo trước.",
        },
        {
            "id": "d3",
            "content": "Lãi suất vay do các bên thỏa thuận nhưng không được vượt quá 20%/năm của khoản tiền vay.",
        },
    ]

    index = BM25Index(
        documents=docs,
        text_accessor=lambda d: d["content"],
    )

    # Query 1: Damage compensation
    results1 = index.search("bồi thường thiệt hại hợp đồng", top_k=2)
    assert len(results1) == 2
    assert results1[0][1]["id"] == "d1"
    assert results1[0][0] > 0.0

    # Query 2: Interest rate cap
    results2 = index.search("trần lãi suất vay 20%", top_k=1)
    assert len(results2) == 1
    assert results2[0][1]["id"] == "d3"


def test_bm25_lexical_retriever_filtering():
    custom_chunks = [
        {
            "chunk_id": "VN_BLDS_D357",
            "canonical_citation": "Điều 357 Bộ luật Dân sự 2015",
            "law_id": "bo-luat-dan-su-2015",
            "article_num": 357,
            "clause_num": None,
            "content": "Nghĩa vụ trả lãi do chậm trả tiền.",
            "context_enriched_text": "[Bộ luật Dân sự 2015] Điều 357: Nghĩa vụ trả lãi do chậm trả tiền.",
            "is_active": True,
        },
        {
            "chunk_id": "VN_BLDS_D468",
            "canonical_citation": "Điều 468 Bộ luật Dân sự 2015",
            "law_id": "bo-luat-dan-su-2015",
            "article_num": 468,
            "clause_num": None,
            "content": "Lãi suất vay tối đa 20% mỗi năm.",
            "context_enriched_text": "[Bộ luật Dân sự 2015] Điều 468: Lãi suất vay tối đa 20% mỗi năm.",
            "is_active": True,
        },
        {
            "chunk_id": "VN_BLLD_D115",
            "canonical_citation": "Điều 115 Bộ luật Lao động 2019",
            "law_id": "bo-luat-lao-dong-2019",
            "article_num": 115,
            "clause_num": None,
            "content": "Nghỉ lễ tết hưởng nguyên lương.",
            "context_enriched_text": "[Bộ luật Lao động 2019] Điều 115: Nghỉ lễ tết hưởng nguyên lương.",
            "is_active": True,
        },
        {
            "chunk_id": "VN_REPEALED_D001",
            "canonical_citation": "Điều 1 Luật cũ hết hiệu lực",
            "law_id": "luat-cu",
            "article_num": 1,
            "clause_num": None,
            "content": "Quy định đã hết hiệu lực thi hành.",
            "context_enriched_text": "Quy định đã hết hiệu lực thi hành.",
            "is_active": False,
        },
    ]

    retriever = BM25LexicalRetriever(custom_chunks=custom_chunks)

    # 1. Search with law filter
    res_blds = retriever.search("lãi suất trả chậm", law_id="bo-luat-dan-su-2015", top_k=5)
    assert len(res_blds) == 2
    assert all(r["law_id"] == "bo-luat-dan-su-2015" for r in res_blds)

    # 2. Search with article filter
    res_art468 = retriever.search("lãi suất", article_num=468, top_k=2)
    assert len(res_art468) == 1
    assert res_art468[0]["article_num"] == 468

    # 3. Active filter: should skip repealed statute
    res_repealed = retriever.search("Quy định đã hết hiệu lực", law_id="luat-cu", active_only=True)
    assert len(res_repealed) == 0

    res_repealed_all = retriever.search(
        "Quy định đã hết hiệu lực", law_id="luat-cu", active_only=False
    )
    assert len(res_repealed_all) == 1
    assert res_repealed_all[0]["chunk_id"] == "VN_REPEALED_D001"

    # 4. Context formatting
    context_str = BM25LexicalRetriever.format_context_for_prompt(res_blds)
    assert "CĂN CỨ PHÁP LUẬT QUY CHUẨN (BM25 LEXICAL GROUNDING)" in context_str
    assert "Điều 357" in context_str or "Điều 468" in context_str

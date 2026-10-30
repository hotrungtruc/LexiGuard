"""
Tests for Phase 2: Context Grounding, Re-ranking & Anchor Highlighting
"""

import pytest
from src.retrieval.context_grounding import (
    extract_query_anchors,
    highlight_statutory_anchors,
    identify_focal_clause,
    reorder_provisions,
    format_grounded_statutory_context,
)


def test_extract_query_anchors_numeric_and_entities():
    query = "Theo Bộ luật Dân sự 2015, thời hạn chuyển giao quyền đại diện là bao nhiêu ngày? Có phải 15 ngày không?"
    anchors = extract_query_anchors(query)

    assert any("15 ngày" in a for a in anchors)
    assert any("quyền đại diện" in a.lower() for a in anchors)
    assert not any(a == "là" or a == "của" or a == "bao nhiêu" for a in anchors)


def test_extract_query_anchors_legal_authority_and_conditions():
    query = "Cơ quan nào có thẩm quyền quản lý đất đai: Bộ Tài nguyên và Môi trường hay UBND cấp tỉnh trừ trường hợp có thỏa thuận khác?"
    anchors = extract_query_anchors(query)

    assert any("bộ tài nguyên và môi trường" in a.lower() for a in anchors)
    assert any("trừ trường hợp" in a.lower() for a in anchors)


def test_extract_query_anchors_percentage():
    query = "Lãi suất vay tối đa theo Bộ luật Dân sự 2015 là 20%/năm đúng không?"
    anchors = extract_query_anchors(query)

    assert any("20%/năm" in a for a in anchors)


def test_highlight_statutory_anchors():
    text = "Thời hạn chuyển giao là 15 ngày kể từ ngày chấm dứt hợp đồng đại diện."
    anchors = ["15 ngày", "hợp đồng đại diện"]

    highlighted, matched = highlight_statutory_anchors(text, anchors)

    assert "**15 ngày**" in highlighted
    assert "**hợp đồng đại diện**" in highlighted
    assert "15 ngày" in matched
    # Ensure no double wrapping
    assert "****" not in highlighted


def test_highlight_statutory_anchors_case_preservation():
    text = "Vụ án thuộc thẩm quyền của Bộ Tài nguyên và Môi trường theo quy định."
    anchors = ["bộ tài nguyên và môi trường"]

    highlighted, matched = highlight_statutory_anchors(text, anchors)

    assert "**Bộ Tài nguyên và Môi trường**" in highlighted
    assert "****" not in highlighted


def test_identify_focal_clause():
    article_text = (
        "Điều 61. Chuyển giao quyền đại diện\n"
        "1. Người đại diện có thể ủy quyền lại cho người khác.\n"
        "2. Thời hạn chuyển giao quyền đại diện theo ủy quyền là 15 ngày kể từ thời điểm phát sinh.\n"
        "3. Việc chuyển giao phải lập thành văn bản công chứng."
    )
    anchors = ["15 ngày", "quyền đại diện", "thời hạn chuyển giao"]

    annotated, focal_idx = identify_focal_clause(article_text, anchors)

    assert focal_idx == 2  # Line index for Khoản 2
    assert "📌 [QUY ĐỊNH TRỌNG TÂM] 2. Thời hạn chuyển giao" in annotated


def test_reorder_provisions_proximity_and_recency():
    provisions = [
        {"citation": "Điều 1", "score": 0.5},
        {"citation": "Điều 2", "score": 0.9},  # Top rank (Rank 1)
        {"citation": "Điều 3", "score": 0.7},
    ]

    # Strategy: proximity_to_query (Rank 1 at index 0)
    reordered_prox = reorder_provisions(provisions, strategy="proximity_to_query")
    assert reordered_prox[0]["citation"] == "Điều 2"
    assert reordered_prox[0]["grounding_rank"] == 1
    assert reordered_prox[1]["citation"] == "Điều 3"

    # Strategy: recency_first (Rank 1 at the end, closest to prompt generation)
    reordered_recency = reorder_provisions(provisions, strategy="recency_first")
    assert reordered_recency[-1]["citation"] == "Điều 2"
    assert reordered_recency[-1]["grounding_rank"] == 1


def test_reorder_provisions_sandwich_u_shape():
    provisions = [
        {"citation": f"Điều {i}", "score": float(i)}
        for i in [1, 2, 3, 4, 5]
    ]
    # Scores: Điều 5 (5.0, Rank 1), Điều 4 (4.0, Rank 2), Điều 3 (3.0, Rank 3), Điều 2 (2.0, Rank 4), Điều 1 (1.0, Rank 5)

    u_shaped = reorder_provisions(provisions, strategy="sandwich_u_shape")
    # In Anthropic U-curve: Top and Bottom should contain the highest ranks (Rank 1 & Rank 2)
    outer_citations = {u_shaped[0]["citation"], u_shaped[-1]["citation"]}
    assert "Điều 5" in outer_citations  # Rank 1
    assert "Điều 4" in outer_citations  # Rank 2


def test_format_grounded_statutory_context_e2e():
    provisions = [
        {
            "canonical_citation": "BLDS 2015 - Điều 468",
            "content": "1. Lãi suất vay do các bên thỏa thuận.\n2. Trường hợp các bên có thỏa thuận về lãi suất thì lãi suất theo thỏa thuận không được vượt quá 20%/năm của khoản tiền vay.",
            "score": 0.95,
        },
        {
            "canonical_citation": "BLDS 2015 - Điều 466",
            "content": "Nghĩa vụ trả nợ của bên vay được thực hiện đúng thời hạn.",
            "score": 0.65,
        },
    ]
    query = "Lãi suất vay tối đa theo Bộ luật Dân sự 2015 là 20%/năm đúng không?"

    context = format_grounded_statutory_context(
        provisions=provisions,
        query=query,
        highlight_anchors=True,
        ordering_strategy="proximity_to_query",
    )

    assert "⭐ [CĂN CỨ CỐT LÕI - HẠNG 1]" in context
    assert "BLDS 2015 - Điều 468" in context
    assert "**20%/năm**" in context
    assert "📌 [QUY ĐỊNH TRỌNG TÂM]" in context

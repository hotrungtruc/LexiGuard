"""
Unit Tests for Dynamic Grounded Prompt System (src/agents/prompts.py)
=====================================================================
Validates:
- System Personas & Cognitive Scaffolding (TAND & VIAC dual-perspective)
- Domain-Agnostic Dynamic Prompt Assembly (zero hardcoded static doctrines)
- Prompt Builders sizing and token budget safety
- Anti-Hallucination Grader prompt rendering
"""

from src.agents.prompts import (
    CLAUSE_EXTRACTOR_SYSTEM_PROMPT,
    DUAL_PERSPECTIVE_AUDITOR_SYSTEM_PROMPT,
    HALLUCINATION_GRADER_SYSTEM_PROMPT,
    build_clause_extractor_prompt,
    build_hallucination_grader_prompt,
    build_loophole_auditor_prompt,
)


def test_system_personas_content_and_dual_perspective():
    """Verify that personas establish TAND judicial and VIAC arbitral perspectives."""
    assert "Chuyên gia Pháp lý AI cao cấp" in CLAUSE_EXTRACTOR_SYSTEM_PROMPT
    assert "Luật sư Tranh tụng Cấp cao" in DUAL_PERSPECTIVE_AUDITOR_SYSTEM_PROMPT
    assert "LĂNG KÍNH THẨM PHÁN TÒA ÁN NHÂN DÂN (TAND)" in DUAL_PERSPECTIVE_AUDITOR_SYSTEM_PROMPT
    assert "LĂNG KÍNH HỘI ĐỒNG TRỌNG TÀI VIAC" in DUAL_PERSPECTIVE_AUDITOR_SYSTEM_PROMPT
    assert "REDLINE THAY THẾ" in DUAL_PERSPECTIVE_AUDITOR_SYSTEM_PROMPT
    assert "LUẬN CỨ ĐÀM PHÁN" in DUAL_PERSPECTIVE_AUDITOR_SYSTEM_PROMPT
    assert "Chuyên gia Kiểm định Pháp lý Độc lập" in HALLUCINATION_GRADER_SYSTEM_PROMPT


def test_build_clause_extractor_prompt_rendering():
    """Test Node 1 prompt builder."""
    markdown = "# HỢP ĐỒNG KINH TẾ\nĐiều 1: Đối tượng..."
    prompt = build_clause_extractor_prompt(markdown, max_chars=100)
    assert "Chuyên gia Pháp lý AI cao cấp" in prompt
    assert "# HỢP ĐỒNG KINH TẾ" in prompt
    assert "ClauseExtractionList" in prompt


def test_build_loophole_auditor_prompt_dynamic_grounding():
    """Test Node 3 prompt builder with clauses, dynamic statutes, precedents, arithmetic findings, and feedback."""
    parsed_clauses = [
        {"clause_reference": "Điều 5", "verbatim_text": "Phạt 20% nếu vi phạm.", "is_table": False},
        {
            "clause_reference": "Điều 6",
            "verbatim_text": "Tranh chấp tại VIAC hoặc TAND.",
            "is_table": False,
        },
    ]
    retrieved_statutes = [
        {
            "canonical_citation": "Điều 301 Luật Thương mại 2005",
            "content": "Mức phạt tối đa 8%",
            "chunk_id": "VN_LTM_D301",
        },
        {
            "canonical_citation": "Nghị quyết 01/2014/NQ-HĐTP",
            "content": "Thỏa thuận trọng tài không thể thực hiện",
            "chunk_id": "NQ_01_2014",
        },
    ]
    retrieved_precedents = [
        {
            "precedent_number": "Án lệ số 09/2016/AL",
            "court": "TANDTC",
            "legal_rule": "Xác định lãi suất chậm trả",
        },
    ]
    arithmetic_findings = [
        {
            "clause_reference": "Phụ lục 1",
            "discrepancy_type": "Lệch tổng tiền",
            "actual_value": 10000000,
            "expected_value": 9000000,
            "legal_risk": "Tính thừa 1,000,000 VNĐ.",
        }
    ]
    feedback = "Cần loại bỏ trích dẫn Điều 9999 không tồn tại."

    prompt = build_loophole_auditor_prompt(
        parsed_clauses=parsed_clauses,
        retrieved_statutes=retrieved_statutes,
        retrieved_precedents=retrieved_precedents,
        arithmetic_findings=arithmetic_findings,
        critique_feedback=feedback,
    )

    # 1. Verify inclusion of essential components
    assert "LĂNG KÍNH THẨM PHÁN TÒA ÁN NHÂN DÂN (TAND)" in prompt
    assert "LĂNG KÍNH HỘI ĐỒNG TRỌNG TÀI VIAC" in prompt
    assert "Điều 301 Luật Thương mại 2005" in prompt
    assert "Án lệ số 09/2016/AL" in prompt
    assert "CẢNH BÁO TỰ SỬA LỖI TỪ VÒNG KIỂM TOÁN TRƯỚC" in prompt
    assert "Phụ lục 1" in prompt
    assert "AuditDraftReport" in prompt

    # 2. Verify token budget safety
    char_len = len(prompt)
    assert 1000 <= char_len <= 10000, f"Prompt character length {char_len} out of expected bounds"


def test_build_hallucination_grader_prompt_rendering():
    """Test Node 4 prompt builder."""
    findings = [
        {"finding_id": "f1", "statutory_citation": "Điều 301 Luật Thương mại 2005"},
    ]
    statutes = [
        {"canonical_citation": "Điều 301 Luật Thương mại 2005", "chunk_id": "c1"},
    ]
    prompt = build_hallucination_grader_prompt(findings, statutes)
    assert "Chuyên gia Kiểm định Pháp lý Độc lập" in prompt
    assert "Điều 301 Luật Thương mại 2005" in prompt
    assert "HallucinationGrade" in prompt

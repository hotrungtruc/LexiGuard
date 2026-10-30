"""
Unit Tests for Dynamic Grounded Prompt System (src/agents/prompts.py)
=====================================================================
Validates:
- System Personas & Cognitive Scaffolding (TAND & VIAC dual-perspective)
- Domain-Agnostic Dynamic Prompt Assembly (zero hardcoded static doctrines)
- Prompt Builders sizing and token budget safety
- Anti-Hallucination Grader prompt rendering
- Deliberative Reasoning Scratchpad (<thinking>) & Consultation prompt
"""

from src.agents.prompts import (
    CLAUSE_EXTRACTOR_SYSTEM_PROMPT,
    DUAL_PERSPECTIVE_AUDITOR_SYSTEM_PROMPT,
    HALLUCINATION_GRADER_SYSTEM_PROMPT,
    build_clause_extractor_prompt,
    build_hallucination_grader_prompt,
    build_legal_consultation_prompt,
    build_loophole_auditor_prompt,
    extract_thinking_and_answer,
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


def test_build_legal_consultation_prompt_and_scratchpad():
    """Test deliberative consultation prompt with <thinking> scratchpad."""
    query = "Lãi suất vay tối đa là bao nhiêu?"
    ctx = "### Điều 468 BLDS 2015\nLãi suất vay do các bên thỏa thuận không được vượt quá 20%/năm."
    
    prompt_vi = build_legal_consultation_prompt(query, ctx, lang="vi")
    assert "<thinking>" in prompt_vi
    assert "</thinking>" in prompt_vi
    assert "BÓC TÁCH VẤN ĐỀ CỐT LÕI" in prompt_vi
    assert "RÀ SOÁT ĐIỀU KIỆN & NGOẠI LỆ" in prompt_vi
    assert "TIỀN LỆ & THỰC TIỄN XÉT XỬ TAND/VIAC" in prompt_vi
    assert "Điều 468 BLDS 2015" in prompt_vi

    prompt_en = build_legal_consultation_prompt(query, ctx, lang="en")
    assert "<thinking>" in prompt_en
    assert "EXCEPTIONS & CARVE-OUTS" in prompt_en
    assert "DISPUTE FORUM ADJUDICATION" in prompt_en


def test_extract_thinking_and_answer():
    """Test parser separating internal thinking from visible response."""
    raw = "<thinking>\n1. Phân tích: Áp dụng Điều 468 BLDS.\n2. Ngoại lệ: Không có.\n</thinking>\n\nTheo quy định tại Điều 468 BLDS 2015, lãi suất tối đa là 20%/năm."
    thinking, answer = extract_thinking_and_answer(raw)
    assert thinking is not None
    assert "Áp dụng Điều 468 BLDS" in thinking
    assert answer.startswith("Theo quy định tại Điều 468")

    # Case without thinking tag
    plain = "Đây là câu trả lời trực tiếp không có thinking tag."
    t_none, ans_plain = extract_thinking_and_answer(plain)
    assert t_none is None
    assert ans_plain == plain

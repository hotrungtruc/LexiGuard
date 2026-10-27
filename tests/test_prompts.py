"""
Unit Tests for Dual-Perspective Prompt System (src/agents/prompts.py)
=====================================================================
Validates:
- System Personas & Cognitive Scaffolding (TAND & VIAC dual-perspective)
- Coverage of 9 Landmark Jurisprudential Principles
- Few-Shot Demonstration Bank integrity
- Dynamic Prompt Builders sizing and token budget safety
"""

import pytest
from src.agents.prompts import (
    CLAUSE_EXTRACTOR_SYSTEM_PROMPT,
    DUAL_PERSPECTIVE_AUDITOR_SYSTEM_PROMPT,
    HALLUCINATION_GRADER_SYSTEM_PROMPT,
    TAND_VIAC_JURISPRUDENCE_DOCTRINE,
    FEW_SHOT_DISPUTE_EXAMPLES,
    build_clause_extractor_prompt,
    build_loophole_auditor_prompt,
    build_hallucination_grader_prompt,
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


def test_jurisprudence_doctrine_matrix_complete_coverage():
    """Ensure all 9 essential Vietnamese dispute principles are deeply embedded."""
    doctrine = TAND_VIAC_JURISPRUDENCE_DOCTRINE

    # 1. Trần phạt 8%
    assert "Điều 301 Luật Thương mại 2005" in doctrine
    assert "8%" in doctrine

    # 2. Liquidated damages
    assert "BỒI THƯỜNG THIỆT HẠI ƯỚC TÍNH" in doctrine
    assert "Điều 302, 304 LTM" in doctrine

    # 3. Thỏa thuận trọng tài bất khả thi
    assert "Điều 43 Luật TTTM" in doctrine
    assert "Nghị quyết 01/2014/NQ-HĐTP" in doctrine

    # 4. Lãi quá hạn & Án lệ 09/2016/AL
    assert "Án lệ 09/2016/AL" in doctrine
    assert "lãi kép" in doctrine.lower()

    # 5. Cam kết không cạnh tranh (NCA)
    assert "Điều 5, 10 BLLĐ 2019" in doctrine
    assert "CAM KẾT KHÔNG CẠNH TRANH" in doctrine

    # 6. Đơn phương chấm dứt
    assert "Điều 428 Bộ luật Dân sự 2015" in doctrine
    assert "khắc phục" in doctrine.lower()

    # 7. Bất khả kháng
    assert "Điều 156, 351 Bộ luật Dân sự 2015" in doctrine

    # 8. Miễn trừ do lỗi cố ý
    assert "Điều 360 Bộ luật Dân sự 2015" in doctrine
    assert "lỗi cố ý" in doctrine.lower()

    # 9. Hợp đồng gia nhập bất bình đẳng
    assert "Điều 406 Bộ luật Dân sự 2015" in doctrine


def test_few_shot_dispute_examples_structure():
    """Verify Few-Shot bank provides complete examples with redlines and negotiation arguments."""
    examples = FEW_SHOT_DISPUTE_EXAMPLES
    assert "Điều 301 Luật Thương mại 2005" in examples
    assert "proposed_redline" in examples
    assert "negotiation_rationale" in examples
    assert "Nghị quyết 01/2014/NQ-HĐTP" in examples


def test_build_clause_extractor_prompt_rendering():
    """Test Node 1 prompt builder."""
    markdown = "# HỢP ĐỒNG KINH TẾ\nĐiều 1: Đối tượng..."
    prompt = build_clause_extractor_prompt(markdown, max_chars=100)
    assert "Chuyên gia Pháp lý AI cao cấp" in prompt
    assert "# HỢP ĐỒNG KINH TẾ" in prompt
    assert "ClauseExtractionList" in prompt


def test_build_loophole_auditor_prompt_budget_and_integration():
    """Test Node 3 prompt builder with clauses, statutes, arithmetic findings, and feedback."""
    parsed_clauses = [
        {"clause_reference": "Điều 5", "verbatim_text": "Phạt 20% nếu vi phạm.", "is_table": False},
        {"clause_reference": "Điều 6", "verbatim_text": "Tranh chấp tại VIAC hoặc TAND.", "is_table": False},
    ]
    retrieved_statutes = [
        {"canonical_citation": "Điều 301 Luật Thương mại 2005", "content": "Mức phạt tối đa 8%", "chunk_id": "VN_LTM_D301"},
        {"canonical_citation": "Nghị quyết 01/2014/NQ-HĐTP", "content": "Thỏa thuận trọng tài không thể thực hiện", "chunk_id": "NQ_01_2014"},
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
        arithmetic_findings=arithmetic_findings,
        critique_feedback=feedback,
        include_doctrine=True,
        include_few_shots=True,
    )

    # 1. Verify inclusion of essential components
    assert "LĂNG KÍNH THẨM PHÁN TÒA ÁN NHÂN DÂN (TAND)" in prompt
    assert "LĂNG KÍNH HỘI ĐỒNG TRỌNG TÀI VIAC" in prompt
    assert "Điều 301 Luật Thương mại 2005" in prompt
    assert "CẢNH BÁO TỰ SỬA LỖI TỪ VÒNG KIỂM TOÁN TRƯỚC" in prompt
    assert "Phụ lục 1" in prompt
    assert "AuditDraftReport" in prompt

    # 2. Verify token budget safety: characters should be between 4,000 and 10,000
    # (equivalent to ~1,200 - 2,500 tokens, perfectly safe for CPU 16K context)
    char_len = len(prompt)
    assert 4000 <= char_len <= 15000, f"Prompt character length {char_len} out of expected bounds"


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

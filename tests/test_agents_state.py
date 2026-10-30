"""
Unit Tests for Agent State & Pydantic v2 Structured Outputs (src/agents/state.py)
================================================================================
Validates:
- Pydantic v2 models serialization & schema validation
- Complete coverage of TAND & VIAC dispute vulnerability taxonomy
- TypedDict AgentState compatibility with LangGraph
- Structured LLM output formats across Nodes 1, 3, and 4
"""

from src.agents.state import (
    AgentState,
    AuditDraftReport,
    AuditFindingItem,
    ClauseExtractionList,
    ContractAuditReport,
    ContractProvenance,
    DisputeForum,
    DisputePrecedentInsight,
    ExtractedClauseItem,
    HallucinationGrade,
    LegalGrounding,
    Redline3Tier,
    VulnerabilityCategory,
    VulnerabilityItem,
)


def test_vulnerability_categories_tand_viac_coverage():
    """Verify all real-world dispute categories from TAND and VIAC are represented."""
    expected_categories = [
        "PENALTY_EXCEEDING_STATUTORY_CAP",
        "UNENFORCEABLE_LIQUIDATED_DAMAGES",
        "PATHOLOGICAL_ARBITRATION_CLAUSE",
        "UNLAWFUL_LIABILITY_EXCLUSION",
        "ABUSIVE_UNILATERAL_TERMINATION",
        "INVALID_FORCE_MAJEURE_EXPANSION",
        "NON_COMPETE_RESTRICTION_CONFLICT",
        "USURIOUS_OR_COMPOUND_LATE_INTEREST",
        "ADHESION_UNFAIR_TERMS",
        "ARITHMETIC_TABLE_DISCREPANCY",
        "STATUTORY_CONFLICT",
    ]
    for cat in expected_categories:
        assert hasattr(VulnerabilityCategory, cat)
        val = getattr(VulnerabilityCategory, cat).value
        assert val == cat


def test_dispute_precedent_insight_modeling():
    """Verify DisputePrecedentInsight captures judicial trends and precedent citations."""
    insight = DisputePrecedentInsight(
        target_forum=DisputeForum.VIAC,
        precedent_reference="Điều 301 Luật Thương mại 2005 & Thực tiễn Trọng tài VIAC",
        judicial_tendency="Hội đồng trọng tài kiên quyết giảm mức phạt vi phạm về đúng 8%.",
        adjudication_risk_probability="HIGH",
    )
    dumped = insight.model_dump()
    assert dumped["target_forum"] == "VIAC"
    assert "VIAC" in dumped["precedent_reference"]
    assert "8%" in dumped["judicial_tendency"]
    assert dumped["adjudication_risk_probability"] == "HIGH"


def test_node1_clause_extraction_schemas():
    """Verify Node 1: ClauseExtractionList and ExtractedClauseItem Pydantic v2 schemas."""
    item = ExtractedClauseItem(
        clause_id="clause_001",
        clause_reference="Điều 6: Phạt vi phạm",
        verbatim_text="Nếu Bên B giao hàng chậm, Bên B phải chịu phạt 20% giá trị hợp đồng.",
        risk_category="Phạt vi phạm",
        is_table=False,
    )
    extraction_list = ClauseExtractionList(
        clauses=[item],
        total_high_risk_clauses=1,
        contains_financial_tables=False,
    )

    dumped = extraction_list.model_dump()
    assert len(dumped["clauses"]) == 1
    assert dumped["total_high_risk_clauses"] == 1
    assert dumped["clauses"][0]["clause_id"] == "clause_001"

    # Roundtrip validation
    reloaded = ClauseExtractionList.model_validate(dumped)
    assert reloaded.clauses[0].verbatim_text == item.verbatim_text


def test_node3_audit_draft_schemas():
    """Verify Node 3: AuditDraftReport and AuditFindingItem Pydantic v2 schemas."""
    finding = AuditFindingItem(
        finding_id="finding_001",
        clause_reference="Điều 6",
        statutory_citation="Điều 301 Luật Thương mại 2005",
        statutory_article_id="VN_LTM_2005_D301",
        risk_tier="HIGH",
        vulnerability_classification=VulnerabilityCategory.PENALTY_EXCEEDING_STATUTORY_CAP.value,
        precedent_reference="Án lệ số 09/2016/AL",
        audit_verdict="Mức phạt 20% vượt quá trần tối đa 8% cho phép theo Điều 301 Luật Thương mại.",
        proposed_redline="Mức phạt vi phạm được giới hạn tối đa không quá 8% phần giá trị nghĩa vụ bị vi phạm.",
        negotiation_rationale="Tuân thủ trần pháp lý bắt buộc để tránh bị VIAC tuyên vô hiệu điều khoản.",
    )
    draft = AuditDraftReport(
        contract_name="HopDongMuaBan.pdf",
        findings=[finding],
        summary_notes="Phát hiện điều khoản phạt vi phạm trái luật.",
        dispute_forum_focus=DisputeForum.VIAC,
    )

    dumped = draft.model_dump()
    assert dumped["dispute_forum_focus"] == "VIAC"
    assert len(dumped["findings"]) == 1
    assert (
        dumped["findings"][0]["vulnerability_classification"] == "PENALTY_EXCEEDING_STATUTORY_CAP"
    )

    reloaded = AuditDraftReport.model_validate(dumped)
    assert reloaded.findings[0].statutory_citation == "Điều 301 Luật Thương mại 2005"


def test_node4_hallucination_grade_schemas():
    """Verify Node 4: HallucinationGrade Pydantic v2 schema."""
    grade_pass = HallucinationGrade(
        is_grounded=True,
        citation_accuracy_score=1.0,
        unverified_citations=[],
        critique_feedback=None,
    )
    assert grade_pass.is_grounded is True
    assert grade_pass.citation_accuracy_score == 1.0

    grade_fail = HallucinationGrade(
        is_grounded=False,
        citation_accuracy_score=0.5,
        unverified_citations=["Điều 9999 Bộ luật Vũ trụ"],
        critique_feedback="Trích dẫn không có trong cơ sở dữ liệu luật.",
    )
    assert grade_fail.is_grounded is False
    assert len(grade_fail.unverified_citations) == 1


def test_contract_audit_report_complete_structure():
    """Verify ContractAuditReport incorporating VulnerabilityItem, Provenance, Grounding, and Dispute Insight."""
    item = VulnerabilityItem(
        finding_id="finding_001",
        clause_reference="Điều 12",
        contract_provenance=ContractProvenance(
            source_file="contract.pdf",
            page_number=3,
            line_range="15-25",
            verbatim_text="Mọi tranh chấp được giải quyết tại VIAC hoặc Tòa án TP.HCM.",
        ),
        legal_grounding=LegalGrounding(
            statute_title="Luật Trọng tài thương mại 2010",
            article="Điều 43",
            clause="Khoản 1",
            statutory_article_id="VN_LTTTM_2010_D043",
        ),
        risk_tier="CRITICAL",
        vulnerability_classification=VulnerabilityCategory.PATHOLOGICAL_ARBITRATION_CLAUSE.value,
        dispute_insight=DisputePrecedentInsight(
            target_forum=DisputeForum.TAND,
            precedent_reference="Nghị quyết 01/2014/NQ-HĐTP",
            judicial_tendency="Thỏa thuận trọng tài không rõ ràng sẽ khiến thẩm quyền bị chuyển về Tòa án.",
            adjudication_risk_probability="CRITICAL",
        ),
        audit_verdict="Thỏa thuận lựa chọn cả Tòa án và Trọng tài là thỏa thuận không thể thực hiện được.",
        proposed_redline="Mọi tranh chấp sẽ được giải quyết chung thẩm tại Trung tâm Trọng tài Quốc tế Việt Nam (VIAC).",
        negotiation_rationale="Lựa chọn duy nhất một cơ quan tài phán để đảm bảo hiệu lực thi hành phán quyết.",
    )

    report = ContractAuditReport(
        contract_name="HopDongDichVu.pdf",
        overall_health_score=68,
        total_clauses_reviewed=15,
        findings=[item],
        tand_viac_summary="Cần sửa đổi thỏa thuận trọng tài để tránh tranh chấp thẩm quyền.",
    )

    dumped = report.model_dump()
    assert dumped["overall_health_score"] == 68
    assert len(dumped["findings"]) == 1
    assert dumped["findings"][0]["contract_provenance"]["page_number"] == 3
    assert dumped["findings"][0]["dispute_insight"]["target_forum"] == "TAND"


def test_agent_state_typeddict_compatibility():
    """Verify LangGraph AgentState TypedDict structure accommodates all workflow keys."""
    state: AgentState = {
        "raw_document_path": "contract.pdf",
        "contract_metadata": {"title": "Hợp đồng lao động", "governing_law": "Việt Nam"},
        "parsed_clauses": [{"clause_reference": "Điều 1", "verbatim_text": "Nội dung"}],
        "current_clause_index": 0,
        "retrieved_statutes": [{"canonical_citation": "Điều 102 BLLĐ 2019"}],
        "arithmetic_findings": [],
        "preliminary_findings": [],
        "verified_findings": [],
        "audit_draft": None,
        "hallucination_grade": None,
        "retry_count": 0,
        "final_report": None,
        "errors": [],
    }

    assert state["raw_document_path"] == "contract.pdf"
    assert state["contract_metadata"]["governing_law"] == "Việt Nam"
    assert state["retry_count"] == 0


def test_3tier_redline_structure_and_serialization():
    """Verify Redline3Tier accommodates Strict, Balanced, and Flexible tactical redlines."""
    r3 = Redline3Tier(
        strict_protective="Loại bỏ toàn bộ điều khoản phạt vi phạm đối với thân chủ.",
        balanced_standard="Điều chỉnh mức phạt vi phạm về đúng trần tối đa 8% theo Điều 301 Luật Thương mại.",
        flexible_commercial="Áp dụng mức phạt 8% kèm thời hạn khắc phục 30 ngày trước khi tính phạt.",
    )

    dumped = r3.model_dump()
    assert "strict_protective" in dumped
    assert "balanced_standard" in dumped
    assert "flexible_commercial" in dumped
    assert "8%" in dumped["balanced_standard"]

"""
Unit & Integration Tests: LangGraph Dual-LLM Workflow & Self-Correction Loops
=============================================================================
Tests:
1. Dual-LLM client initialization (Qwen2.5-3B and Qwen2.5-7B)
2. Node 1: ClauseExtractor (isolation of clauses and financial tables)
3. Node 2: LegalResearcher (FastMCP tools execution & table arithmetic audit)
4. Node 3: LoopholeAuditor (reasoning & redline synthesis)
5. Node 4: HallucinationGrader (citation verification & critique generation)
6. Self-Correction conditional routing (retry_audit vs. export_report)
7. End-to-end StateGraph compiled execution
"""

import pytest
from src.agents.state import (
    AgentState,
    AuditDraftReport,
    AuditFindingItem,
    HallucinationGrade,
)
from src.agents.nodes import (
    get_fast_cpu_llm,
    get_reasoning_cpu_llm,
    extract_clauses_node,
    retrieve_statutes_node,
    audit_vulnerabilities_node,
    grade_hallucination_node,
    decide_post_grading_branch,
)
from src.agents.workflow import create_lexiguard_graph, lexiguard_agent


# ============================================================================
# 1. TEST DUAL-LLM INITIALIZATION
# ============================================================================

def test_dual_llm_initialization():
    fast_llm = get_fast_cpu_llm()
    reasoning_llm = get_reasoning_cpu_llm()

    assert fast_llm.model == "qwen2.5:3b-instruct-q4_K_M"
    assert fast_llm.temperature == 0.0

    assert reasoning_llm.model == "qwen2.5:7b-instruct-q4_K_M"
    assert reasoning_llm.temperature == 0.1


# ============================================================================
# 2. TEST NODE 1: ClauseExtractor
# ============================================================================

def test_extract_clauses_node_parsing():
    sample_contract = """
# LEGAL SERVICES CONTRACT

Điều 1. Phạm vi công việc
Bên B cung cấp dịch vụ rà soát hợp đồng cho Bên A.

Điều 2. Biểu phí và thanh toán
| STT | Dịch vụ | Số lượng | Đơn giá | Thành tiền |
| 1 | Rà soát hợp đồng | 2 | 10.000.000 | 25.000.000 |
| Tổng cộng | | | | 25.000.000 |

Điều 3. Đơn phương chấm dứt hợp đồng
Bên A có quyền đơn phương chấm dứt hợp đồng bất kỳ lúc nào mà không cần báo trước và không phải bồi thường.
    """
    state: AgentState = {
        "raw_document_path": "",
        "parsed_clauses": [
            {
                "clause_id": "c1",
                "clause_reference": "Điều 2",
                "verbatim_text": "| STT | Dịch vụ | Số lượng | Đơn giá | Thành tiền |\n| 1 | Rà soát | 2 | 10.000.000 | 25.000.000 |\n| Tổng cộng | | | | 25.000.000 |",
                "risk_category": "Financial Table",
                "is_table": True,
                "table_markdown": "| STT | Dịch vụ | Số lượng | Đơn giá | Thành tiền |\n| 1 | Rà soát | 2 | 10.000.000 | 25.000.000 |\n| Tổng cộng | | | | 25.000.000 |",
            },
            {
                "clause_id": "c2",
                "clause_reference": "Điều 3",
                "verbatim_text": "Bên A có quyền đơn phương chấm dứt hợp đồng bất kỳ lúc nào mà không cần báo trước.",
                "risk_category": "Termination",
                "is_table": False,
                "table_markdown": None,
            },
        ],
        "current_clause_index": 0,
        "retrieved_statutes": [],
        "arithmetic_findings": [],
        "preliminary_findings": [],
        "verified_findings": [],
        "audit_draft": None,
        "hallucination_grade": None,
        "retry_count": 0,
        "final_report": None,
        "errors": [],
    }

    out = extract_clauses_node(state)
    assert len(out["parsed_clauses"]) >= 2
    assert out["retry_count"] == 0


# ============================================================================
# 3. TEST NODE 2: LegalResearcher (FastMCP Tools & Arithmetic Auditing)
# ============================================================================

def test_retrieve_statutes_node_execution():
    table_md = """
| Mặt hàng | Số lượng | Đơn giá | Thành tiền |
| Dịch vụ 1 | 2 | 10.000.000 | 25.000.000 |
| Tổng cộng | | | 25.000.000 |
    """
    state: AgentState = {
        "raw_document_path": "",
        "parsed_clauses": [
            {
                "clause_reference": "Phụ lục 1",
                "verbatim_text": table_md,
                "is_table": True,
                "table_markdown": table_md,
            },
            {
                "clause_reference": "Điều 4",
                "verbatim_text": "Quy định về áp dụng Bộ luật Dân sự trong quan hệ hợp đồng",
                "is_table": False,
                "table_markdown": None,
            },
        ],
        "current_clause_index": 0,
        "retrieved_statutes": [],
        "arithmetic_findings": [],
        "preliminary_findings": [],
        "verified_findings": [],
        "audit_draft": None,
        "hallucination_grade": None,
        "retry_count": 0,
        "final_report": None,
        "errors": [],
    }

    out = retrieve_statutes_node(state)
    assert "retrieved_statutes" in out
    assert "arithmetic_findings" in out
    # Arithmetic finding caught: 2 * 10,000,000 != 25,000,000
    assert len(out["arithmetic_findings"]) >= 1
    assert out["arithmetic_findings"][0]["type"] == "ROW_PRODUCT_MISMATCH"


# ============================================================================
# 4. TEST NODE 3: LoopholeAuditor (Reasoning & Redline Generation)
# ============================================================================

def test_audit_vulnerabilities_node_generation():
    state: AgentState = {
        "raw_document_path": "contract.pdf",
        "parsed_clauses": [
            {
                "clause_reference": "Điều 8",
                "verbatim_text": "Bên vi phạm phải chịu phạt 50% giá trị hợp đồng.",
            }
        ],
        "current_clause_index": 0,
        "retrieved_statutes": [
            {
                "canonical_citation": "Điều 301 Luật Thương mại 2005",
                "chunk_id": "VN_LTM_2005_D301",
                "content": "Mức phạt đối với vi phạm nghĩa vụ hợp đồng do các bên thoả thuận nhưng không quá 8% giá trị phần nghĩa vụ bị vi phạm.",
                "source_clause": "Điều 8",
            }
        ],
        "arithmetic_findings": [],
        "preliminary_findings": [],
        "verified_findings": [],
        "audit_draft": None,
        "hallucination_grade": None,
        "retry_count": 0,
        "final_report": None,
        "errors": [],
    }

    out = audit_vulnerabilities_node(state)
    assert "audit_draft" in out
    assert "preliminary_findings" in out
    draft = out["audit_draft"]
    assert len(draft.findings) >= 1
    assert draft.findings[0].proposed_redline is not None


# ============================================================================
# 5. TEST NODE 4 & CONDITIONAL ROUTING: HallucinationGrader
# ============================================================================

def test_hallucination_grader_pass():
    """Grounded citations pass with accuracy score 1.0."""
    draft = AuditDraftReport(
        contract_name="Test Contract",
        findings=[
            AuditFindingItem(
                finding_id="f1",
                clause_reference="Điều 4",
                statutory_citation="Điều 4 Bộ luật Dân sự 2015",
                audit_verdict="Valid",
                proposed_redline="Redline text",
            )
        ],
    )
    state: AgentState = {
        "raw_document_path": "contract.pdf",
        "parsed_clauses": [{"clause_reference": "Điều 4", "verbatim_text": "..."}],
        "current_clause_index": 0,
        "retrieved_statutes": [
            {
                "canonical_citation": "Điều 4 Bộ luật Dân sự 2015",
                "article_num": 4,
                "content": "Áp dụng Bộ luật dân sự",
            }
        ],
        "arithmetic_findings": [],
        "preliminary_findings": [],
        "verified_findings": [],
        "audit_draft": draft,
        "hallucination_grade": None,
        "retry_count": 0,
        "final_report": None,
        "errors": [],
    }

    out = grade_hallucination_node(state)
    grade: HallucinationGrade = out["hallucination_grade"]
    assert grade.is_grounded is True
    assert grade.citation_accuracy_score == 1.0
    assert out["final_report"] is not None

    # Test conditional routing
    branch = decide_post_grading_branch({**state, **out})
    assert branch == "export_report"


def test_hallucination_grader_fail_triggers_self_correction_retry():
    """Fictitious citations trigger retry loop back to auditor."""
    draft = AuditDraftReport(
        contract_name="Test Contract",
        findings=[
            AuditFindingItem(
                finding_id="f1",
                clause_reference="Điều 99",
                statutory_citation="Điều 9999 Bộ luật Vũ trụ 3000",
                audit_verdict="Hallucinated",
            )
        ],
    )
    state: AgentState = {
        "raw_document_path": "contract.pdf",
        "parsed_clauses": [{"clause_reference": "Điều 99", "verbatim_text": "..."}],
        "current_clause_index": 0,
        "retrieved_statutes": [
            {
                "canonical_citation": "Điều 4 Bộ luật Dân sự 2015",
                "article_num": 4,
                "content": "Áp dụng Bộ luật dân sự",
            }
        ],
        "arithmetic_findings": [],
        "preliminary_findings": [],
        "verified_findings": [],
        "audit_draft": draft,
        "hallucination_grade": None,
        "retry_count": 0,
        "final_report": None,
        "errors": [],
    }

    out = grade_hallucination_node(state)
    grade: HallucinationGrade = out["hallucination_grade"]
    assert grade.is_grounded is False
    assert grade.citation_accuracy_score < 1.0

    # Test conditional routing: retry_count=0 -> retry_audit
    branch = decide_post_grading_branch({**state, **out})
    assert branch == "retry_audit"

    # If retry_count >= 2: branch must be export_report
    max_state = {**state, **out, "retry_count": 2}
    branch_max = decide_post_grading_branch(max_state)
    assert branch_max == "export_report"


def test_hallucination_grader_rejects_cross_statute_mismatch():
    """Confirms cross-statute hallucination (matching article num but wrong law) is rejected."""
    draft = AuditDraftReport(
        contract_name="Test Cross-Statute",
        findings=[
            AuditFindingItem(
                finding_id="f_cross",
                clause_reference="Điều 8 Phạt vi phạm",
                statutory_citation="Điều 301 Bộ luật Dân sự 2015",
                audit_verdict="Mức phạt 8% theo Điều 301 BLDS",
            )
        ],
    )
    state: AgentState = {
        "raw_document_path": "contract.pdf",
        "parsed_clauses": [{"clause_reference": "Điều 8", "verbatim_text": "Phạt 20%"}],
        "current_clause_index": 0,
        "retrieved_statutes": [
            {
                "chunk_id": "VN_LTM_2005_D301",
                "canonical_citation": "Điều 301 Luật Thương mại 2005",
                "article_num": 301,
                "content": "Mức phạt đối với vi phạm nghĩa vụ hợp đồng do các bên thoả thuận nhưng không quá 8%",
            }
        ],
        "arithmetic_findings": [],
        "preliminary_findings": [],
        "verified_findings": [],
        "audit_draft": draft,
        "hallucination_grade": None,
        "retry_count": 0,
        "final_report": None,
        "errors": [],
    }

    out = grade_hallucination_node(state)
    grade: HallucinationGrade = out["hallucination_grade"]
    assert grade.is_grounded is False
    assert "Điều 301 Bộ luật Dân sự 2015" in grade.unverified_citations
    assert len(out["verified_findings"]) == 0



# ============================================================================
# 6. TEST FULL STATEGRAPH COMPILATION & EXECUTION
# ============================================================================

def test_full_stategraph_compilation():
    graph = create_lexiguard_graph()
    assert graph is not None


def test_full_stategraph_end_to_end_invocation():
    state: AgentState = {
        "raw_document_path": "sample.md",
        "parsed_clauses": [
            {
                "clause_id": "c1",
                "clause_reference": "Điều 5",
                "verbatim_text": "Bên A có quyền khấu trừ 50% tiền lương của Bên B khi vi phạm quy chế.",
                "risk_category": "Wage Deduction",
                "is_table": False,
                "table_markdown": None,
            }
        ],
        "current_clause_index": 0,
        "retrieved_statutes": [],
        "arithmetic_findings": [],
        "preliminary_findings": [],
        "verified_findings": [],
        "audit_draft": None,
        "hallucination_grade": None,
        "retry_count": 0,
        "final_report": None,
        "errors": [],
    }

    res = lexiguard_agent.invoke(state)
    assert res is not None
    assert "retrieved_statutes" in res
    assert "audit_draft" in res
    assert "hallucination_grade" in res
    assert res["final_report"] is not None
    assert res["final_report"].overall_health_score >= 0

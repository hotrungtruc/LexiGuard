"""
State & Schemas: AgentState (TypedDict) and Pydantic Schemas for Dual-LLM Architecture
=====================================================================================
Defines structured schemas for:
- ClauseExtractionList (Node 1: Qwen2.5-3B)
- AuditDraftReport (Node 3: Qwen2.5-7B)
- HallucinationGrade (Node 4: Qwen2.5-3B)
- ContractAuditReport & AgentState
"""

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class ContractProvenance(BaseModel):
    source_file: str
    page_number: int
    line_range: str
    bounding_box: Optional[Dict[str, float]] = None
    verbatim_text: str


class LegalGrounding(BaseModel):
    statute_title: str
    article: str
    clause: Optional[str] = None
    neo4j_node_id: Optional[str] = None
    qdrant_point_id: Optional[str] = None
    retrieval_similarity_score: Optional[float] = None


class VulnerabilityItem(BaseModel):
    finding_id: str
    clause_reference: str
    contract_provenance: ContractProvenance
    legal_grounding: LegalGrounding
    risk_tier: Literal["HIGH", "MEDIUM", "LOW"]
    vulnerability_classification: str
    audit_verdict: str
    proposed_redline: Optional[str] = None
    negotiation_rationale: Optional[str] = None


class ContractAuditReport(BaseModel):
    contract_name: str
    overall_health_score: int = Field(ge=0, le=100)
    total_clauses_reviewed: int
    findings: List[VulnerabilityItem] = []


# ============================================================================
# STRUCTURED OUTPUT SCHEMAS FOR DUAL-LLM CPU PIPELINE
# ============================================================================

class ExtractedClauseItem(BaseModel):
    """Atomic clause or financial table isolated from contract markdown."""
    clause_id: str
    clause_reference: str
    verbatim_text: str
    risk_category: str = Field(description="e.g., Termination, Liability, Damages, Salary, Table")
    is_table: bool = False
    table_markdown: Optional[str] = None


class ClauseExtractionList(BaseModel):
    """Structured Output for Node 1: ClauseExtractor (Qwen2.5-3B)."""
    clauses: List[ExtractedClauseItem] = Field(default_factory=list)
    total_high_risk_clauses: int = 0
    contains_financial_tables: bool = False


class AuditFindingItem(BaseModel):
    """Detailed audit finding comparing contractual clause vs. statutory rule."""
    finding_id: str
    clause_reference: str
    statutory_citation: str = Field(description="e.g., 'Khoản 2 Điều 102 Bộ luật Lao động 2019'")
    statutory_article_id: Optional[str] = Field(default=None, description="e.g., 'VN_BLLD_2019_D102_K02'")
    risk_tier: Literal["HIGH", "MEDIUM", "LOW"] = "HIGH"
    audit_verdict: str = Field(description="Conflict analysis with governing statute or precedent")
    proposed_redline: Optional[str] = Field(default=None, description="Enforceable replacement contract text")
    negotiation_rationale: Optional[str] = None


class AuditDraftReport(BaseModel):
    """Structured Output for Node 3: LoopholeAuditor (Qwen2.5-7B)."""
    contract_name: str = "Contract"
    findings: List[AuditFindingItem] = Field(default_factory=list)
    summary_notes: str = ""


class HallucinationGrade(BaseModel):
    """Structured Output for Node 4: HallucinationGrader (Qwen2.5-3B)."""
    is_grounded: bool = Field(description="True if all citations exist in retrieved context, False otherwise")
    citation_accuracy_score: float = Field(ge=0.0, le=1.0, description="1.0 for 100% grounded citations")
    unverified_citations: List[str] = Field(default_factory=list, description="Citations not supported by context")
    critique_feedback: Optional[str] = Field(default=None, description="Actionable critique for retry loop")


class AgentState(TypedDict):
    """LangGraph Agent State across nodes in Dual-LLM architecture."""
    raw_document_path: str
    parsed_clauses: List[Dict[str, Any]]
    current_clause_index: int
    retrieved_statutes: List[Dict[str, Any]]
    arithmetic_findings: List[Dict[str, Any]]
    preliminary_findings: List[Dict[str, Any]]
    verified_findings: List[VulnerabilityItem]
    audit_draft: Optional[AuditDraftReport]
    hallucination_grade: Optional[HallucinationGrade]
    retry_count: int
    final_report: Optional[ContractAuditReport]
    errors: List[str]

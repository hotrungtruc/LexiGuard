"""
State & Schemas: Định nghĩa AgentState (TypedDict) và Pydantic Schemas cho Báo cáo lỗi
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

class AgentState(TypedDict):
    """LangGraph Agent State across nodes"""
    raw_document_path: str
    parsed_clauses: List[Dict[str, Any]]
    current_clause_index: int
    retrieved_statutes: List[Dict[str, Any]]
    preliminary_findings: List[Dict[str, Any]]
    verified_findings: List[VulnerabilityItem]
    final_report: Optional[ContractAuditReport]
    errors: List[str]

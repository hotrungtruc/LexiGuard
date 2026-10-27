"""
State & Schemas: AgentState (TypedDict) and Pydantic v2 Models for Legal Multi-Agent Council
=============================================================================================
Defines typed state models and structured LLM outputs tailored to Vietnamese dispute
resolution reality, embedding vulnerability classifications and case law doctrine from:
- TAND (People's Court): TAND precedents, cassation decisions, and trial judgments.
- VIAC (Vietnam International Arbitration Centre): arbitral awards, arbitrability, and procedural rules.

Covers key dispute patterns:
1. Penalty exceeding the statutory cap (Article 301, Law on Commerce 2005).
2. Estimated or punitive liquidated damages.
3. Invalid or impracticable arbitration agreements (Article 43, Law on Commercial Arbitration).
4. Unlawful liability exclusions for intentional fault (Article 360, Civil Code 2015).
5. Abusive unilateral termination without a cure period (Article 428, Civil Code 2015).
6. Unlawful force majeure expansion (exchange-rate changes, inflation, or lack of funds).
7. Non-compete agreements conflicting with labor freedom (Articles 5 and 10, Labor Code 2019).
8. Excessive or compound late-payment interest.
9. Unfair standard-form contract terms (Article 406, Civil Code 2015).
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

# ============================================================================
# 1. LITIGATION & ARBITRATION DISPUTE PATTERNS (TAND & VIAC CASE LAW)
# ============================================================================


class VulnerabilityCategory(str, Enum):
    """
    Empirical contractual vulnerability taxonomy derived from Vietnamese
    court judgments and arbitral awards.
    """

    # 1. Penalties exceeding the statutory cap.
    PENALTY_EXCEEDING_STATUTORY_CAP = "PENALTY_EXCEEDING_STATUTORY_CAP"

    # 2. Unproven estimated or liquidated damages.
    UNENFORCEABLE_LIQUIDATED_DAMAGES = "UNENFORCEABLE_LIQUIDATED_DAMAGES"

    # 3. Impracticable arbitration agreements or jurisdictional conflicts.
    PATHOLOGICAL_ARBITRATION_CLAUSE = "PATHOLOGICAL_ARBITRATION_CLAUSE"

    # 4. Unlawful liability exclusions for intentional fault.
    UNLAWFUL_LIABILITY_EXCLUSION = "UNLAWFUL_LIABILITY_EXCLUSION"

    # 5. Abusive unilateral termination without a cure period.
    ABUSIVE_UNILATERAL_TERMINATION = "ABUSIVE_UNILATERAL_TERMINATION"

    # 6. Unlawful force majeure expansion.
    INVALID_FORCE_MAJEURE_EXPANSION = "INVALID_FORCE_MAJEURE_EXPANSION"

    # 7. Non-compete restrictions conflicting with freedom of employment.
    NON_COMPETE_RESTRICTION_CONFLICT = "NON_COMPETE_RESTRICTION_CONFLICT"

    # 8. Excessive or compound late-payment interest.
    USURIOUS_OR_COMPOUND_LATE_INTEREST = "USURIOUS_OR_COMPOUND_LATE_INTEREST"

    # 9. Unfair adhesion terms that remove the right to sue.
    ADHESION_UNFAIR_TERMS = "ADHESION_UNFAIR_TERMS"

    # 10. Arithmetic or tax-rate discrepancies in financial tables.
    ARITHMETIC_TABLE_DISCREPANCY = "ARITHMETIC_TABLE_DISCREPANCY"

    # 11. Conflicts with other specialized legislation.
    STATUTORY_CONFLICT = "STATUTORY_CONFLICT"


UNCLASSIFIED_VULNERABILITY = "UNCLASSIFIED"


class LegalRuleMetadata(BaseModel):
    """Metadata for a rule supplied by the legal knowledge base.

    The agent treats these values as opaque identifiers.  Adding a new rule
    therefore requires only ingesting its metadata, not changing Python code.
    """

    model_config = ConfigDict(extra="allow")

    category_id: str = Field(default=UNCLASSIFIED_VULNERABILITY)
    category_label: Optional[str] = None
    risk_tier: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    forum: Optional[str] = None
    precedent_reference: Optional[str] = None
    judicial_tendency: Optional[str] = None
    redline_template: Optional[str] = None


class DisputeForum(str, Enum):
    """Jurisdictional dispute resolution forum."""

    TAND = "TAND"  # People's Court
    VIAC = "VIAC"  # Vietnam International Arbitration Centre
    GENERAL = "GENERAL"  # Applies to all dispute-resolution forums


class DisputePrecedentInsight(BaseModel):
    """
    Practical adjudication insights referencing established TAND & VIAC jurisprudence.
    """

    model_config = ConfigDict(extra="ignore")

    target_forum: DisputeForum = Field(
        default=DisputeForum.GENERAL, description="Primary dispute-resolution forum"
    )
    precedent_reference: Optional[str] = Field(
        default=None,
        description="Relevant TAND precedent, judicial council resolution, or VIAC practice",
        examples=[
            "Án lệ số 09/2016/AL",
            "Nghị quyết 01/2014/NQ-HĐTP",
            "Thực tiễn VIAC về trần phạt 8%",
        ],
    )
    judicial_tendency: str = Field(
        default="",
        description="Observed tendency of the People's Court or VIAC tribunal",
        examples=[
            "Tòa án có xu hướng tuyên vô hiệu phần phạt vượt quá 8% theo Điều 301 Luật Thương mại."
        ],
    )
    adjudication_risk_probability: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(
        default="HIGH", description="Likelihood of invalidation or modification in a dispute"
    )


# ============================================================================
# 2. CORE PYDANTIC V2 MODELS FOR PROVENANCE & LEGAL GROUNDING
# ============================================================================


class ContractProvenance(BaseModel):
    """Spatial and structural provenance tracing back to original contract text."""

    model_config = ConfigDict(extra="ignore")

    source_file: str = Field(description="Name or path of the original contract file")
    page_number: int = Field(default=1, ge=1, description="Page containing the clause (1-indexed)")
    line_range: str = Field(default="1-10", description="Text line range or coordinates")
    bounding_box: Optional[Dict[str, float]] = Field(
        default=None, description="Spatial bounding-box coordinates [l, t, r, b]"
    )
    verbatim_text: str = Field(description="Verbatim contract wording")


class LegalGrounding(BaseModel):
    """Verifiable statutory grounding retrieved from Qdrant Hybrid & Neo4j 2-hops."""

    model_config = ConfigDict(extra="ignore")

    statute_title: str = Field(
        description="Statute title (e.g. 'Labor Code 2019', 'Law on Commerce 2005')"
    )
    article: str = Field(description="Cited article (e.g. 'Article 102', 'Article 301')")
    clause: Optional[str] = Field(default=None, description="Cited clause or point")
    statutory_article_id: Optional[str] = Field(
        default=None, description="Graph node identifier (e.g. 'VN_BLLD_2019_D102')"
    )
    neo4j_node_id: Optional[str] = Field(default=None, description="Neo4j knowledge-graph node ID")
    qdrant_point_id: Optional[str] = Field(default=None, description="Qdrant vector point ID")
    retrieval_similarity_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Semantic similarity score"
    )
    legal_hierarchy: str = Field(
        default="Law", description="Instrument level: law, code, decree, circular, or precedent"
    )


class VulnerabilityItem(BaseModel):
    """
    Standardized contract vulnerability report item incorporating TAND/VIAC dispute risk.
    """

    model_config = ConfigDict(extra="ignore")

    finding_id: str = Field(description="Finding identifier (e.g. 'finding_001')")
    clause_reference: str = Field(
        description="Contract clause location (e.g. 'Article 6, Clause 2')"
    )
    contract_provenance: ContractProvenance
    legal_grounding: LegalGrounding
    risk_tier: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(
        default="HIGH", description="Finding severity"
    )
    vulnerability_classification: str = Field(
        default=UNCLASSIFIED_VULNERABILITY,
        description="Classification ID supplied by knowledge-base metadata",
    )
    dispute_insight: Optional[DisputePrecedentInsight] = Field(
        default=None, description="Dispute risk assessment for court or arbitration"
    )
    audit_verdict: str = Field(description="Detailed analysis of the conflict with applicable law")
    proposed_redline: Optional[str] = Field(
        default=None, description="Legally compliant replacement wording"
    )
    negotiation_rationale: Optional[str] = Field(
        default=None, description="Negotiation rationale for the proposed change"
    )


class ContractAuditReport(BaseModel):
    """Final verified contract audit report output."""

    model_config = ConfigDict(extra="ignore")

    contract_name: str = Field(description="Name or title of the audited contract")
    overall_health_score: int = Field(
        ge=0, le=100, description="Overall legal health score (0-100)"
    )
    total_clauses_reviewed: int = Field(ge=0, description="Number of reviewed clauses")
    findings: List[VulnerabilityItem] = Field(
        default_factory=list, description="List of legal risk findings"
    )
    tand_viac_summary: Optional[str] = Field(
        default=None,
        description="Summary of litigation risk at TAND or VIAC if the contract remains unchanged",
    )


# ============================================================================
# 3. STRUCTURED OUTPUT SCHEMAS FOR DUAL-LLM CPU ARCHITECTURE
# ============================================================================


class ExtractedClauseItem(BaseModel):
    """Atomic clause or financial table isolated from contract markdown by Node 1."""

    model_config = ConfigDict(extra="ignore")

    clause_id: str = Field(description="Clause identifier (e.g. 'clause_001')")
    clause_reference: str = Field(
        description="Clause heading (e.g. 'Article 5: Contractual penalty')"
    )
    verbatim_text: str = Field(description="Verbatim text extracted from the contract")
    risk_category: str = Field(
        default="General",
        description="Preliminary risk group (e.g. penalty, damages, termination, confidentiality, or table)",
    )
    is_table: bool = Field(default=False, description="Whether the item is a financial table")
    table_markdown: Optional[str] = Field(
        default=None, description="Verbatim Markdown representation of the table"
    )


class ClauseExtractionList(BaseModel):
    """Structured Output for Node 1: ClauseExtractor (Qwen2.5-3B-Instruct)."""

    model_config = ConfigDict(extra="ignore")

    clauses: List[ExtractedClauseItem] = Field(
        default_factory=list, description="List of material clauses"
    )
    total_high_risk_clauses: int = Field(default=0, description="Number of high-risk clauses")
    contains_financial_tables: bool = Field(
        default=False, description="Whether financial tables require arithmetic verification"
    )


class AuditFindingItem(BaseModel):
    """
    Detailed audit draft finding from Node 3: LoopholeAuditor (Qwen2.5-7B-Instruct).
    Directly incorporates Vietnamese statutory rules and judicial precedent.
    """

    model_config = ConfigDict(extra="ignore")

    finding_id: str = Field(description="Audit finding identifier (e.g. 'finding_001')")
    clause_reference: str = Field(
        description="Location of the affected clause (e.g. 'Article 8, Clause 1')"
    )
    statutory_citation: str = Field(
        description="Accurate statutory citation (e.g. 'Article 301, Law on Commerce 2005')",
        examples=[
            "Điều 301 Luật Thương mại 2005",
            "Khoản 2 Điều 102 Bộ luật Lao động 2019",
            "Điều 43 Luật Trọng tài thương mại 2010",
        ],
    )
    statutory_article_id: Optional[str] = Field(
        default=None, description="Neo4j graph node ID (e.g. 'VN_LTM_2005_D301')"
    )
    risk_tier: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(
        default="HIGH", description="Risk severity"
    )
    vulnerability_classification: str = Field(
        default=UNCLASSIFIED_VULNERABILITY,
        description="Classification ID supplied by knowledge-base metadata",
    )
    precedent_reference: Optional[str] = Field(
        default=None, description="Related precedent or decision (e.g. 'Precedent 09/2016/AL')"
    )
    audit_verdict: str = Field(
        description="Analysis comparing the finding with statutory law or precedent"
    )
    proposed_redline: Optional[str] = Field(
        default=None, description="Legally compliant, balanced replacement wording"
    )
    negotiation_rationale: Optional[str] = Field(
        default=None, description="Basis and negotiation rationale for the counterparty"
    )


class AuditDraftReport(BaseModel):
    """Structured Output for Node 3: LoopholeAuditor (Qwen2.5-7B-Instruct)."""

    model_config = ConfigDict(extra="ignore")

    contract_name: str = Field(default="Contract", description="Name of the contract under review")
    findings: List[AuditFindingItem] = Field(
        default_factory=list, description="List of legal violation findings"
    )
    summary_notes: str = Field(
        default="", description="Preliminary assessment of the contract's dispute risk"
    )
    dispute_forum_focus: DisputeForum = Field(
        default=DisputeForum.GENERAL, description="Primary dispute-resolution forum (TAND or VIAC)"
    )


class HallucinationGrade(BaseModel):
    """
    Structured Output for Node 4: HallucinationGrader (Qwen2.5-3B-Instruct).
    Verifies that every legal citation generated by LoopholeAuditor is strictly grounded
    in the retrieved database context.
    """

    model_config = ConfigDict(extra="ignore")

    is_grounded: bool = Field(
        description="True when every cited provision exists in the retrieved context"
    )
    citation_accuracy_score: float = Field(
        ge=0.0, le=1.0, description="Citation accuracy ratio (1.0 means fully grounded)"
    )
    unverified_citations: List[str] = Field(
        default_factory=list,
        description="Citations that are fabricated or absent from the retrieved context",
    )
    critique_feedback: Optional[str] = Field(
        default=None,
        description="Detailed correction feedback sent to LoopholeAuditor for the next retry",
    )


# ============================================================================
# 4. LANGGRAPH AGENT STATE TYPEDDICT
# ============================================================================


class AgentState(TypedDict):
    """
    LangGraph Cyclic State Machine State.
    Thread-safe dictionary state passed across the 4 specialized agent nodes.
    Supports dynamic statutory and judicial precedent retrieval from Qdrant and Neo4j.
    """

    raw_document_path: str
    contract_metadata: Dict[str, Any]
    parsed_clauses: List[Dict[str, Any]]
    current_clause_index: int
    retrieved_statutes: List[Dict[str, Any]]
    retrieved_precedents: List[Dict[str, Any]]
    arithmetic_findings: List[Dict[str, Any]]
    preliminary_findings: List[Dict[str, Any]]
    verified_findings: List[VulnerabilityItem]
    audit_draft: Optional[AuditDraftReport]
    hallucination_grade: Optional[HallucinationGrade]
    retry_count: int
    final_report: Optional[ContractAuditReport]
    errors: List[str]

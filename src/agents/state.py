"""
State & Schemas: AgentState (TypedDict) and Pydantic v2 Models for Legal Multi-Agent Council
=============================================================================================
Defines typed state models and structured LLM outputs tailored to Vietnamese dispute
resolution reality, embedding vulnerability classifications and case law doctrine from:
- TAND (Tòa án Nhân dân): Precedents (Án lệ TANDTC), Cassation Decisions, Trial Judgments.
- VIAC (Trung tâm Trọng tài Quốc tế Việt Nam): Arbitral Awards, Arbitrability & Procedural Rules.

Covers key dispute patterns:
1. Phạt vi phạm vượt trần 8% (Điều 301 Luật Thương mại 2005).
2. Bồi thường thiệt hại ước tính / Phạt cọc kép (Liquidated Damages / Punitive Damages).
3. Thỏa thuận trọng tài vô hiệu / Bất khả thi (Điều 43 Luật TTTM, NQ 01/2014/NQ-HĐTP).
4. Miễn trừ trách nhiệm trái luật do lỗi cố ý (Điều 360 BLDS 2015).
5. Đơn phương chấm dứt hợp đồng tùy tiện không có thời gian khắc phục (Điều 428 BLDS 2015).
6. Mở rộng bất khả kháng trái luật: biến động tỷ giá, trượt giá, thiếu vốn (Điều 156, 351 BLDS 2015).
7. Thỏa thuận không cạnh tranh (NCA) xung đột quyền tự do lao động (Điều 5, 10 BLLĐ 2019).
8. Lãi chậm trả vượt trần hoặc lãi chồng lãi (Điều 306 LTM, Điều 468 BLDS, Án lệ 09/2016/AL).
9. Điều khoản hợp đồng theo mẫu bất bình đẳng (Điều 406 BLDS 2015).
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# 1. LITIGATION & ARBITRATION DISPUTE PATTERNS (TAND & VIAC CASE LAW)
# ============================================================================

class VulnerabilityCategory(str, Enum):
    """
    Empirical contractual vulnerability taxonomy derived from Vietnamese
    court judgments (Bản án TAND) and arbitral awards (Phán quyết VIAC).
    """
    # 1. Phạt vi phạm vượt trần luật định (Điều 301 LTM 2005: max 8%)
    PENALTY_EXCEEDING_STATUTORY_CAP = "PENALTY_EXCEEDING_STATUTORY_CAP"

    # 2. Bồi thường thiệt hại ước tính không chứng minh được thiệt hại thực tế (Liquidated damages)
    UNENFORCEABLE_LIQUIDATED_DAMAGES = "UNENFORCEABLE_LIQUIDATED_DAMAGES"

    # 3. Thỏa thuận trọng tài không thể thực hiện / Xung đột thẩm quyền (NQ 01/2014/NQ-HĐTP, Điều 43 Luật TTTM)
    PATHOLOGICAL_ARBITRATION_CLAUSE = "PATHOLOGICAL_ARBITRATION_CLAUSE"

    # 4. Miễn trừ trách nhiệm trái luật (Miễn trừ do lỗi cố ý - Điều 360 BLDS 2015)
    UNLAWFUL_LIABILITY_EXCLUSION = "UNLAWFUL_LIABILITY_EXCLUSION"

    # 5. Đơn phương chấm dứt hợp đồng bất hợp pháp không cho thời gian khắc phục (Điều 428 BLDS 2015)
    ABUSIVE_UNILATERAL_TERMINATION = "ABUSIVE_UNILATERAL_TERMINATION"

    # 6. Mở rộng điều kiện bất khả kháng trái luật (Biến động giá, tỷ giá, thiếu vốn)
    INVALID_FORCE_MAJEURE_EXPANSION = "INVALID_FORCE_MAJEURE_EXPANSION"

    # 7. Cam kết không cạnh tranh xâm phạm quyền tự do việc làm (Điều 5, 10 BLLĐ 2019, Điều 35 Hiến pháp 2013)
    NON_COMPETE_RESTRICTION_CONFLICT = "NON_COMPETE_RESTRICTION_CONFLICT"

    # 8. Lãi chậm trả vượt trần hoặc tính lãi kép trái luật (Điều 306 LTM, Điều 468 BLDS, Án lệ 09/2016/AL)
    USURIOUS_OR_COMPOUND_LATE_INTEREST = "USURIOUS_OR_COMPOUND_LATE_INTEREST"

    # 9. Điều khoản gia nhập bất bình đẳng, loại bỏ quyền khởi kiện (Điều 406 BLDS 2015)
    ADHESION_UNFAIR_TERMS = "ADHESION_UNFAIR_TERMS"

    # 10. Bảng biểu tài chính sai lệch số học hoặc thuế suất
    ARITHMETIC_TABLE_DISCREPANCY = "ARITHMETIC_TABLE_DISCREPANCY"

    # 11. Xung đột quy định pháp luật chuyên ngành khác
    STATUTORY_CONFLICT = "STATUTORY_CONFLICT"


class DisputeForum(str, Enum):
    """Jurisdictional dispute resolution forum."""
    TAND = "TAND"       # Tòa án Nhân dân (People's Court)
    VIAC = "VIAC"       # Trung tâm Trọng tài Quốc tế Việt Nam (Vietnam International Arbitration Centre)
    GENERAL = "GENERAL" # Chung cho mọi diễn đàn giải quyết tranh chấp


class DisputePrecedentInsight(BaseModel):
    """
    Practical adjudication insights referencing established TAND & VIAC jurisprudence.
    """
    model_config = ConfigDict(extra="ignore")

    target_forum: DisputeForum = Field(default=DisputeForum.GENERAL, description="Diễn đàn thụ lý tranh chấp trọng tâm")
    precedent_reference: Optional[str] = Field(
        default=None,
        description="Số hiệu Án lệ TANDTC, Nghị quyết HĐTP hoặc Thực tiễn Trọng tài VIAC liên quan",
        examples=["Án lệ số 09/2016/AL", "Nghị quyết 01/2014/NQ-HĐTP", "Thực tiễn VIAC về trần phạt 8%"]
    )
    judicial_tendency: str = Field(
        default="",
        description="Xu hướng xét xử thực tế của Thẩm phán TAND hoặc Hội đồng Trọng tài VIAC",
        examples=["Tòa án có xu hướng tuyên vô hiệu phần phạt vượt quá 8% theo Điều 301 Luật Thương mại."]
    )
    adjudication_risk_probability: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(
        default="HIGH",
        description="Xác suất bị tuyên vô hiệu hoặc sửa đổi khi xảy ra tranh chấp"
    )


# ============================================================================
# 2. CORE PYDANTIC V2 MODELS FOR PROVENANCE & LEGAL GROUNDING
# ============================================================================

class ContractProvenance(BaseModel):
    """Spatial and structural provenance tracing back to original contract text."""
    model_config = ConfigDict(extra="ignore")

    source_file: str = Field(description="Tên hoặc đường dẫn file hợp đồng gốc")
    page_number: int = Field(default=1, ge=1, description="Trang xuất hiện điều khoản (1-indexed)")
    line_range: str = Field(default="1-10", description="Dòng hoặc tọa độ văn bản")
    bounding_box: Optional[Dict[str, float]] = Field(default=None, description="Tọa độ không gian bbox [l, t, r, b]")
    verbatim_text: str = Field(description="Nguyên văn câu chữ trong hợp đồng")


class LegalGrounding(BaseModel):
    """Verifiable statutory grounding retrieved from Qdrant Hybrid & Neo4j 2-hops."""
    model_config = ConfigDict(extra="ignore")

    statute_title: str = Field(description="Tên văn bản luật (e.g. 'Bộ luật Lao động 2019', 'Luật Thương mại 2005')")
    article: str = Field(description="Điều luật viện dẫn (e.g. 'Điều 102', 'Điều 301')")
    clause: Optional[str] = Field(default=None, description="Khoản, điểm viện dẫn")
    statutory_article_id: Optional[str] = Field(default=None, description="ID định danh node đồ thị (e.g. 'VN_BLLD_2019_D102')")
    neo4j_node_id: Optional[str] = Field(default=None, description="ID node Neo4j phục vụ đồ thị tri thức")
    qdrant_point_id: Optional[str] = Field(default=None, description="ID điểm vector Qdrant")
    retrieval_similarity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Độ tương đồng ngữ nghĩa")
    legal_hierarchy: str = Field(default="Luật", description="Cấp bậc văn bản: Luật, Bộ luật, Nghị định, Thông tư, Án lệ")


class VulnerabilityItem(BaseModel):
    """
    Standardized contract vulnerability report item incorporating TAND/VIAC dispute risk.
    """
    model_config = ConfigDict(extra="ignore")

    finding_id: str = Field(description="Mã định danh lỗi (e.g. 'finding_001')")
    clause_reference: str = Field(description="Vị trí điều khoản hợp đồng (e.g. 'Điều 6 Khoản 2')")
    contract_provenance: ContractProvenance
    legal_grounding: LegalGrounding
    risk_tier: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(
        default="HIGH",
        description="Mức độ nghiêm trọng của lỗ hổng"
    )
    vulnerability_classification: str = Field(
        default=VulnerabilityCategory.STATUTORY_CONFLICT.value,
        description="Phân loại lỗ hổng pháp lý theo tiền lệ án TAND và VIAC"
    )
    dispute_insight: Optional[DisputePrecedentInsight] = Field(
        default=None,
        description="Đánh giá rủi ro tranh tụng tại Tòa án hoặc Trọng tài"
    )
    audit_verdict: str = Field(description="Phân tích chi tiết xung đột pháp lý với luật thực định")
    proposed_redline: Optional[str] = Field(default=None, description="Câu chữ thay thế (Redline) chuẩn pháp lý")
    negotiation_rationale: Optional[str] = Field(default=None, description="Luận điểm đàm phán giải thích lý do sửa đổi")


class ContractAuditReport(BaseModel):
    """Final verified contract audit report output."""
    model_config = ConfigDict(extra="ignore")

    contract_name: str = Field(description="Tên hoặc tiêu đề hợp đồng được kiểm toán")
    overall_health_score: int = Field(ge=0, le=100, description="Điểm sức khỏe pháp lý tổng thể (0 - 100)")
    total_clauses_reviewed: int = Field(ge=0, description="Tổng số điều khoản đã thẩm định")
    findings: List[VulnerabilityItem] = Field(default_factory=list, description="Danh sách các điểm rủi ro pháp lý")
    tand_viac_summary: Optional[str] = Field(
        default=None,
        description="Tổng quan rủi ro kiện tụng tại TAND hoặc VIAC nếu giữ nguyên hợp đồng"
    )


# ============================================================================
# 3. STRUCTURED OUTPUT SCHEMAS FOR DUAL-LLM CPU ARCHITECTURE
# ============================================================================

class ExtractedClauseItem(BaseModel):
    """Atomic clause or financial table isolated from contract markdown by Node 1."""
    model_config = ConfigDict(extra="ignore")

    clause_id: str = Field(description="Mã định danh điều khoản (e.g. 'clause_001')")
    clause_reference: str = Field(description="Tiêu đề điều khoản (e.g. 'Điều 5: Phạt vi phạm')")
    verbatim_text: str = Field(description="Nội dung câu chữ nguyên văn trích từ hợp đồng")
    risk_category: str = Field(
        default="General",
        description="Nhóm rủi ro sơ bộ (e.g. Phạt vi phạm, Bồi thường, Đơn phương chấm dứt, Bảo mật, Bảng biểu)"
    )
    is_table: bool = Field(default=False, description="Đánh dấu nếu phần tử là bảng số liệu tài chính")
    table_markdown: Optional[str] = Field(default=None, description="Nguyên văn Markdown của bảng biểu số liệu")


class ClauseExtractionList(BaseModel):
    """Structured Output for Node 1: ClauseExtractor (Qwen2.5-3B-Instruct)."""
    model_config = ConfigDict(extra="ignore")

    clauses: List[ExtractedClauseItem] = Field(default_factory=list, description="Danh sách các điều khoản trọng yếu")
    total_high_risk_clauses: int = Field(default=0, description="Số lượng điều khoản có rủi ro cao")
    contains_financial_tables: bool = Field(default=False, description="Có bảng biểu tài chính cần kiểm tra số học")


class AuditFindingItem(BaseModel):
    """
    Detailed audit draft finding from Node 3: LoopholeAuditor (Qwen2.5-7B-Instruct).
    Directly incorporates Vietnamese statutory rules and judicial precedent.
    """
    model_config = ConfigDict(extra="ignore")

    finding_id: str = Field(description="Mã định danh lỗi kiểm toán (e.g. 'finding_001')")
    clause_reference: str = Field(description="Vị trí điều khoản bị vi phạm (e.g. 'Điều 8 Khoản 1')")
    statutory_citation: str = Field(
        description="Điều luật viện dẫn chuẩn xác (e.g. 'Điều 301 Luật Thương mại 2005')",
        examples=["Điều 301 Luật Thương mại 2005", "Khoản 2 Điều 102 Bộ luật Lao động 2019", "Điều 43 Luật Trọng tài thương mại 2010"]
    )
    statutory_article_id: Optional[str] = Field(default=None, description="Mã node đồ thị Neo4j (e.g. 'VN_LTM_2005_D301')")
    risk_tier: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(
        default="HIGH",
        description="Mức độ nghiêm trọng của rủi ro"
    )
    vulnerability_classification: str = Field(
        default=VulnerabilityCategory.STATUTORY_CONFLICT.value,
        description="Phân loại lỗ hổng theo án lệ TAND / thực tiễn VIAC"
    )
    precedent_reference: Optional[str] = Field(
        default=None,
        description="Án lệ hoặc tiền lệ phán quyết liên quan (e.g. 'Án lệ số 09/2016/AL', 'NQ 01/2014/NQ-HĐTP')"
    )
    audit_verdict: str = Field(
        description="Phân tích đối chiếu lỗ hổng với quy định luật thực định hoặc án lệ"
    )
    proposed_redline: Optional[str] = Field(
        default=None,
        description="Đề xuất câu chữ thay thế (Redline) hợp pháp, cân bằng và bảo vệ thân chủ"
    )
    negotiation_rationale: Optional[str] = Field(
        default=None,
        description="Căn cứ và luận cứ đàm phán giải thích cho phía đối tác"
    )


class AuditDraftReport(BaseModel):
    """Structured Output for Node 3: LoopholeAuditor (Qwen2.5-7B-Instruct)."""
    model_config = ConfigDict(extra="ignore")

    contract_name: str = Field(default="Contract", description="Tên hợp đồng đang thẩm định")
    findings: List[AuditFindingItem] = Field(default_factory=list, description="Danh sách các phát hiện vi phạm pháp lý")
    summary_notes: str = Field(default="", description="Nhận định sơ bộ về mức độ rủi ro tranh tụng của hợp đồng")
    dispute_forum_focus: DisputeForum = Field(
        default=DisputeForum.GENERAL,
        description="Trọng tâm diễn đàn giải quyết tranh chấp (TAND hoặc VIAC)"
    )


class HallucinationGrade(BaseModel):
    """
    Structured Output for Node 4: HallucinationGrader (Qwen2.5-3B-Instruct).
    Verifies that every legal citation generated by LoopholeAuditor is strictly grounded
    in the retrieved database context.
    """
    model_config = ConfigDict(extra="ignore")

    is_grounded: bool = Field(
        description="True nếu 100% điều luật viện dẫn tồn tại trong Context tra cứu, False nếu phát hiện ảo giác"
    )
    citation_accuracy_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Tỷ lệ viện dẫn điều luật chuẩn xác (1.0 = hoàn toàn chuẩn xác, không có ảo giác)"
    )
    unverified_citations: List[str] = Field(
        default_factory=list,
        description="Danh sách các điều luật bị bịa đặt hoặc không có trong ngữ cảnh tra cứu"
    )
    critique_feedback: Optional[str] = Field(
        default=None,
        description="Góp ý sửa đổi chi tiết chuyển về LoopholeAuditor trong vòng lặp tự sửa lỗi"
    )


# ============================================================================
# 4. LANGGRAPH AGENT STATE TYPEDDICT
# ============================================================================

class AgentState(TypedDict):
    """
    LangGraph Cyclic State Machine State.
    Thread-safe dictionary state passed across the 4 specialized agent nodes.
    """
    raw_document_path: str
    contract_metadata: Dict[str, Any]
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

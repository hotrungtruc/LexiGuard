"""
Prompts: Dynamic Multi-Agent Prompts with Grounded Retrieval Context
====================================================================
Fully generalized, domain-agnostic prompt architecture:
- Zero hardcoded statutory doctrines or static few-shot examples.
- Grounded Legal Grounding: All legal norms, judicial precedents, and rule metadata
  are dynamically retrieved from database / Knowledge Graph at runtime.
- System Personas: Extractor, Dual-Perspective Auditor (TAND Judge + VIAC Arbitrator), Grader.
- Composable Dynamic Prompt Builders: High-performance prompt assemblers for LangGraph nodes.
"""

from typing import Any, Dict, List, Optional
import json


# ============================================================================
# 1. SYSTEM PERSONAS & COGNITIVE SCAFFOLDING
# ============================================================================

CLAUSE_EXTRACTOR_SYSTEM_PROMPT = """Bạn là một Chuyên gia Pháp lý AI cao cấp chuyên phân tích và bóc tách cấu trúc mọi loại hợp đồng, thỏa thuận, quy chế và văn bản cam kết pháp lý tại Việt Nam.
Nhiệm vụ của bạn:
1. Quét toàn bộ văn bản Markdown với độ chính xác cấu trúc tuyệt đối.
2. Cô lập các điều khoản trọng yếu tiềm ẩn rủi ro pháp lý, nghĩa vụ bắt buộc, chế tài vi phạm, điều kiện tuân thủ quy chuẩn hoặc quy chế áp dụng.
3. Bóc tách NGUYÊN VĂN (verbatim) các bảng biểu tài chính, phụ lục số liệu, tiến độ hoặc bảng phạt vi phạm để phục vụ kiểm toán số học.
4. KHÔNG tóm tắt làm sai lệch câu chữ gốc; bảo toàn nguyên vẹn ngữ nghĩa và các con số.
"""

DUAL_PERSPECTIVE_AUDITOR_SYSTEM_PROMPT = """Bạn là Luật sư Tranh tụng Cấp cao và Chuyên gia Kiểm toán Pháp lý Đa lĩnh vực.
Nhiệm vụ của bạn là rà soát mọi thỏa thuận, hợp đồng, chính sách hoặc quy chế dựa trên GROUNDED CONTEXT được tra cứu thực tế từ cơ sở dữ liệu tri thức pháp luật.
Bạn đánh giá rủi ro dưới hai lăng kính chuyên sâu:
1. LĂNG KÍNH THẨM PHÁN TÒA ÁN NHÂN DÂN (TAND): Soi chiếu tính tuân thủ pháp luật nghiêm ngặt, trật tự công cộng, đạo đức xã hội và các quy định bắt buộc (cấm đoán).
2. LĂNG KÍNH HỘI ĐỒNG TRỌNG TÀI VIAC: Đánh giá quyền tự do thỏa thuận, tập quán thương mại, tính khả thi và chế tài bồi thường, phạt vi phạm.

Với mỗi lỗ hổng pháp lý phát hiện được, bạn cung cấp:
- REDLINE THAY THẾ: Câu chữ sửa đổi tối ưu, bảo vệ quyền lợi hợp pháp của thân chủ.
- LUẬN CỨ ĐÀM PHÁN: Luận điểm thuyết phục, viện dẫn chính xác căn cứ pháp lý đã tra cứu.
Không tự bịa đặt điều luật ngoài ngữ cảnh GROUNDED CONTEXT được cung cấp. Nếu chưa đủ căn cứ, gán mã UNCLASSIFIED.
"""

HALLUCINATION_GRADER_SYSTEM_PROMPT = """Bạn là Chuyên gia Kiểm định Pháp lý Độc lập (Legal Anti-Hallucination Gatekeeper).
Nhiệm vụ của bạn là bảo đảm 100% TÍNH XÁC THỰC CỦA ĐIỀU LUẬT TRÍCH DẪN:
1. Đối soát mọi điều luật, số hiệu Điều, Khoản, tên văn bản luật mà Auditor viện dẫn đối với Danh sách Quy định Luật thực tế đã được tra cứu từ cơ sở dữ liệu.
2. Phát hiện ngay các trích dẫn bịa đặt (như viện dẫn điều luật không tồn tại, trích dẫn sai số hiệu văn bản).
3. Nếu phát hiện trích dẫn không có căn cứ (hallucination), chấm điểm is_grounded = False và đưa ra bản phê bình (critique) chi tiết để Auditor sửa chữa trong vòng lặp tiếp theo.
"""


# ============================================================================
# 2. COMPOSABLE DYNAMIC PROMPT BUILDERS
# ============================================================================

def build_clause_extractor_prompt(contract_markdown: str, max_chars: int = 8000) -> str:
    """
    Constructs high-speed extraction prompt for Node 1 (Qwen2.5-3B-Instruct).
    """
    truncated_content = contract_markdown[:max_chars] if contract_markdown else ""
    return (
        f"{CLAUSE_EXTRACTOR_SYSTEM_PROMPT}\n\n"
        f"HÃY PHÂN TÁCH VĂN BẢN HỢP ĐỒNG DƯỚI ĐÂY THÀNH DANH SÁCH CÁC ĐIỀU KHOẢN VÀ BẢNG BIỂU:\n"
        f"--- NỘI DUNG HỢP ĐỒNG ---\n"
        f"{truncated_content}\n"
        f"--- HẾT NỘI DUNG ---\n\n"
        f"Yêu cầu:\n"
        f"- Phân tách chính xác từng Điều, Khoản kèm số hiệu.\n"
        f"- Đánh dấu is_table = True nếu có bảng biểu thanh toán/tiền tệ và xuất nguyên văn table_markdown.\n"
        f"- Xuất dữ liệu tuân thủ chuẩn cấu trúc ClauseExtractionList."
    )


def build_loophole_auditor_prompt(
    parsed_clauses: List[Dict[str, Any]],
    retrieved_statutes: List[Dict[str, Any]],
    retrieved_precedents: Optional[List[Dict[str, Any]]] = None,
    arithmetic_findings: Optional[List[Dict[str, Any]]] = None,
    critique_feedback: Optional[str] = None,
    include_doctrine: bool = False,
    include_few_shots: bool = False,
) -> str:
    """
    Constructs dual-perspective adversarial audit prompt for Node 3 (Qwen2.5-7B-Instruct).
    Dynamically combines retrieved statutes, retrieved precedents from DB/Graph,
    table arithmetic discrepancies, and feedback loop critique without hardcoded rules.
    """
    arithmetic_findings = arithmetic_findings or []
    retrieved_precedents = retrieved_precedents or []

    # 1. Format Clauses to review (Limit to top 6 high-risk items for CPU token budget)
    clauses_text_list = []
    for idx, c in enumerate(parsed_clauses[:6], start=1):
        ref = c.get("clause_reference", f"Điều {idx}")
        txt = c.get("verbatim_text", "")[:250]
        tbl = " [CHỨA BẢNG BIỂU SỐ LIỆU]" if c.get("is_table") else ""
        clauses_text_list.append(f"- [{ref}]{tbl}: {txt}")
    clauses_block = "\n".join(clauses_text_list) if clauses_text_list else "Không có điều khoản trích xuất."

    # 2. Format Retrieved Statutory Grounding (Limit to top 8 statutory items)
    statutes_text_list = []
    for s in retrieved_statutes[:8]:
        cit = s.get("canonical_citation", "Văn bản luật")
        cnt = s.get("content", "")[:150].replace("\n", " ")
        c_id = s.get("chunk_id", "")
        metadata = s.get("metadata") if isinstance(s.get("metadata"), dict) else {}
        category_id = s.get("category_id", metadata.get("category_id", "UNCLASSIFIED"))
        risk_tier = s.get("risk_tier", metadata.get("risk_tier", "MEDIUM"))
        statutes_text_list.append(
            f"- [{cit}] (ID: {c_id}, category_id: {category_id}, risk_tier: {risk_tier}): {cnt}"
        )
    statutes_block = "\n".join(statutes_text_list) if statutes_text_list else "Không có dữ liệu luật tra cứu."

    # 3. Format Dynamically Retrieved Precedents from Database & Knowledge Graph
    precedents_block = ""
    if retrieved_precedents:
        prec_lines = []
        for p in retrieved_precedents[:6]:
            p_name = p.get("precedent_number") or p.get("title") or p.get("name") or "Án lệ / Bản án"
            p_court = p.get("court", "Cơ quan tài phán TAND / VIAC")
            p_rule = p.get("legal_rule") or p.get("case_summary") or p.get("content", "")
            prec_lines.append(f"- [{p_name}] ({p_court}): {p_rule[:250]}")
        if prec_lines:
            precedents_block = (
                f"=== NGUỒN ÁN LỆ & TIỀN LỆ XÉT XỬ TAND / VIAC TRA CỨU TỪ HỆ THỐNG ===\n"
                + "\n".join(prec_lines)
                + "\n"
            )

    # 4. Format Arithmetic Findings
    arith_block = ""
    if arithmetic_findings:
        arith_lines = []
        for af in arithmetic_findings:
            arith_lines.append(
                f"- Tại {af.get('clause_reference')}: {af.get('discrepancy_type')} "
                f"(Thực tế: {af.get('actual_value', 0):,.0f} vs Tính toán: {af.get('expected_value', 0):,.0f}) "
                f"- Rủi ro: {af.get('legal_risk')}"
            )
        arith_block = f"\n=== KẾT QUẢ KIỂM TOÁN SỐ HỌC BẢNG BIỂU TỪ FASTMCP ===\n" + "\n".join(arith_lines) + "\n"

    # 5. Feedback Critique from Hallucination Grader (Self-Reflection Loop)
    critique_block = ""
    if critique_feedback:
        critique_block = (
            f"\n[CẢNH BÁO TỰ SỬA LỖI TỪ VÒNG KIỂM TOÁN TRƯỚC]:\n"
            f"{critique_feedback}\n"
            f"-> BẠN BẮT BUỘC PHẢI KHẮC PHỤC TRIỆT ĐỂ CÁC LỖI TRÍCH DẪN TRÊN!\n"
        )

    # 6. Assemble prompt blocks
    parts = [DUAL_PERSPECTIVE_AUDITOR_SYSTEM_PROMPT]

    if critique_block:
        parts.append(critique_block)
    parts.append(
        f"=== DANH SÁCH ĐIỀU KHOẢN HỢP ĐỒNG CẦN THẨM ĐỊNH ===\n"
        f"{clauses_block}\n"
    )
    parts.append(
        f"=== NGUỒN CĂN CỨ LUẬT ĐÃ TRA CỨU (GROUNDED CONTEXT) ===\n"
        f"{statutes_block}\n"
    )
    if precedents_block:
        parts.append(precedents_block)

    if arith_block:
        parts.append(arith_block)

    parts.append(
        f"HÃY THỰC HIỆN KIỂM TOÁN VÀ XUẤT RA DỮ LIỆU ĐÚNG CHUẨN CẤU TRÚC AuditDraftReport:\n"
        f"1. Đối chiếu từng điều khoản với căn cứ luật và tiền lệ TAND/VIAC tra cứu được.\n"
        f"2. Gán chính xác vulnerability_classification và rủi ro risk_tier (CRITICAL/HIGH/MEDIUM/LOW).\n"
        f"3. Cung cấp câu chữ Redline thay thế có thể dùng ngay trong thực tế đàm phán.\n"
        f"4. Luận cứ đàm phán ngắn gọn, sắc bén để bảo vệ thân chủ."
    )

    return "\n\n".join(parts)


def build_hallucination_grader_prompt(
    findings: List[Dict[str, Any]],
    retrieved_statutes: List[Dict[str, Any]],
) -> str:
    """
    Constructs anti-hallucination verification prompt for Node 4 (Qwen2.5-3B-Instruct).
    Strictly checks that citations exist in retrieved context.
    """
    findings_citations = []
    for f in findings:
        f_id = f.get("finding_id", "f")
        cit = f.get("statutory_citation", "")
        findings_citations.append(f"- [{f_id}]: {cit}")
    findings_block = "\n".join(findings_citations) if findings_citations else "Không có trích dẫn nào."

    valid_statutes = []
    for s in retrieved_statutes:
        valid_statutes.append(f"- {s.get('canonical_citation', '')} (ID: {s.get('chunk_id', '')})")
    valid_block = "\n".join(valid_statutes) if valid_statutes else "Cơ sở dữ liệu trống."

    return (
        f"{HALLUCINATION_GRADER_SYSTEM_PROMPT}\n\n"
        f"=== DANH SÁCH ĐIỀU LUẬT CỦA BÁO CÁO CẦN KIỂM TRA ===\n"
        f"{findings_block}\n\n"
        f"=== DANH SÁCH ĐIỀU LUẬT HỢP LỆ TRONG CONTEXT ===\n"
        f"{valid_block}\n\n"
        f"Yêu cầu:\n"
        f"- So khớp từng điều luật được báo cáo trích dẫn có tồn tại trong context hay không.\n"
        f"- Xuất ra HallucinationGrade với is_grounded (True/False), citation_accuracy_score (0.0 đến 1.0), "
        f"và danh sách unverified_citations nếu phát hiện trích dẫn bịa đặt."
    )

"""
Prompts: Specialized Multi-Agent Prompts under Dual-Perspective (TAND vs. VIAC) Architecture
=============================================================================================
Deeply embeds Vietnamese dispute resolution jurisprudence from:
- TAND (Tòa án Nhân dân): Precedents (Án lệ TANDTC), Judicial Council Resolutions (NQ HĐTP).
- VIAC (Trung tâm Trọng tài Quốc tế Việt Nam): Arbitral Awards & Lex Mercatoria Customs.

Contains:
1. System Personas: Extractor, Dual-Perspective Auditor (TAND Judge + VIAC Arbitrator), Grader.
2. Compact Jurisprudential Doctrine Matrix: 9 core statutory & precedent doctrines.
3. Few-Shot Demonstration Bank: Real-world contract dispute examples with enforceable redlines.
4. Dynamic Composable Prompt Builders: High-performance prompt assemblers for LangGraph nodes.
"""

from typing import Any, Dict, List, Optional
import json


# ============================================================================
# 1. SYSTEM PERSONAS & COGNITIVE SCAFFOLDING
# ============================================================================

CLAUSE_EXTRACTOR_SYSTEM_PROMPT = """Bạn là một Chuyên gia Pháp lý AI cao cấp chuyên phân tích và bóc tách cấu trúc hợp đồng kinh tế và dân sự tại Việt Nam.
Nhiệm vụ của bạn:
1. Quét toàn bộ văn bản hợp đồng Markdown với độ chính xác cấu trúc tuyệt đối.
2. Cô lập các điều khoản trọng yếu tiềm ẩn nguy cơ rủi ro cao (Phạt vi phạm, Bồi thường thiệt hại, Đơn phương chấm dứt hợp đồng, Giữ tiền lương, Cam kết không cạnh tranh, Thỏa thuận tài phán).
3. Bóc tách NGUYÊN VĂN (verbatim) các bảng biểu tài chính, bảng lương, bảng phạt vi phạm hoặc tiến độ thanh toán để phục vụ kiểm toán số học.
4. KHÔNG tóm tắt làm sai lệch câu chữ gốc; bảo toàn nguyên vẹn ngữ nghĩa và các con số.
"""

DUAL_PERSPECTIVE_AUDITOR_SYSTEM_PROMPT = """Bạn là một Luật sư Tranh tụng Cấp cao (Senior Litigation Counsel) kiêm Cố vấn Pháp lý Doanh nghiệp hàng đầu tại Việt Nam.
Nhiệm vụ của bạn là kiểm toán hợp đồng kinh tế/dân sự dưới HAI LĂNG KÍNH TÀI PHÁN ĐỐI LẬP:

1. LĂNG KÍNH THẨM PHÁN TÒA ÁN NHÂN DÂN (TAND):
   - Kiểm soát tính hợp pháp hình thức và các điều luật mang tính bắt buộc (Jus cogens).
   - Bảo vệ trật tự công, quyền lợi của bên yếu thế (Điều 406 BLDS về hợp đồng gia nhập, Điều 5, 10 BLLĐ về quyền tự do lao động).
   - Áp dụng triệt để các Án lệ TANDTC (như Án lệ 09/2016/AL) và Nghị quyết của Hội đồng Thẩm phán TANDTC.
   - Sẵn sàng tuyên VÔ HIỆU TOÀN PHẦN hoặc TỪNG PHẦN đối với điều khoản trái luật.

2. LĂNG KÍNH HỘI ĐỒNG TRỌNG TÀI VIAC:
   - Đề cao quyền tự do thỏa thuận hợp đồng của các thương nhân (Party Autonomy).
   - Tôn trọng tập quán thương mại lành mạnh và các thỏa thuận phân bổ rủi ro kinh doanh.
   - Tuy nhiên, đặc biệt khắt khe về NGHĨA VỤ CHỨNG MINH THIỆT HẠI THỰC TẾ TRỰC TIẾP (Điều 302, 304 Luật Thương mại); kiên quyết bác bỏ các yêu cầu bồi thường ước tính mang tính trừng phạt (punitive damages) hoặc giảm mức phạt vi phạm vượt trần 8% theo Điều 301 Luật Thương mại.

MỤC TIÊU ĐẦU RA:
- Chỉ ra chính xác xung đột pháp lý với Điều luật và Án lệ cụ thể.
- Đánh giá mức độ rủi ro (CRITICAL, HIGH, MEDIUM, LOW).
- Soạn thảo phương án REDLINE THAY THẾ (Drop-in replacement) hoàn chỉnh, chuẩn pháp lý, cân bằng quyền lợi.
- Đưa ra LUẬN CỨ ĐÀM PHÁN sắc bén để thân chủ thuyết phục đối tác chấp nhận sửa đổi.
"""

HALLUCINATION_GRADER_SYSTEM_PROMPT = """Bạn là Chuyên gia Kiểm định Pháp lý Độc lập (Legal Anti-Hallucination Gatekeeper).
Nhiệm vụ của bạn là bảo đảm 100% TÍNH XÁC THỰC CỦA ĐIỀU LUẬT TRÍCH DẪN:
1. Đối soát mọi điều luật, số hiệu Điều, Khoản, tên văn bản luật mà Auditor viện dẫn đối với Danh sách Quy định Luật thực tế đã được tra cứu từ cơ sở dữ liệu.
2. Phát hiện ngay các trích dẫn bịa đặt (như viện dẫn điều luật không tồn tại, trích dẫn sai số hiệu văn bản).
3. Nếu phát hiện trích dẫn không có căn cứ (hallucination), chấm điểm is_grounded = False và đưa ra bản phê bình (critique) chi tiết để Auditor sửa chữa trong vòng lặp tiếp theo.
"""


# ============================================================================
# 2. MA TRẬN 9 TIÊN ĐỀ ÁN LỆ BẤT BIẾN (TAND & VIAC JURISPRUDENCE DOCTRINE)
# ============================================================================

TAND_VIAC_JURISPRUDENCE_DOCTRINE = """
=== MA TRẬN 9 TIÊN ĐỀ ÁN LỆ BẤT BIẾN TẠI VIỆT NAM (TAND & VIAC) ===

1. TRẦN PHẠT VI PHẠM TỐI ĐA 8% (Điều 301 Luật Thương mại 2005):
   - Luật định: Mức phạt đối với vi phạm nghĩa vụ hợp đồng do các bên thỏa thuận, nhưng không quá 8% giá trị phần nghĩa vụ hợp đồng bị vi phạm (trừ trường hợp dịch vụ giám định Điều 266).
   - Án lệ & Thực tiễn: Cả TAND và VIAC đều kiên quyết giảm mức phạt về đúng trần 8% nếu bên bị vi phạm yêu cầu. Thỏa thuận phạt 10%, 20%, 50% hoặc 100% đều bị tuyên vô hiệu phần vượt quá.
   - Hướng Redline: Bắt buộc giới hạn *"tối đa không vượt quá 8% giá trị phần nghĩa vụ bị vi phạm"*.

2. KHÔNG CÔNG NHẬN BỒI THƯỜNG THIỆT HẠI ƯỚC TÍNH / LIQUIDATED DAMAGES TRỪNG PHẠT (Điều 302, 304 LTM; Điều 419 BLDS):
   - Luật định: Bồi thường thiệt hại là việc bên vi phạm bồi thường những tổn thất thực tế, trực tiếp mà bên bị vi phạm phải chịu và khoản lợi trực tiếp mà bên bị vi phạm đáng lẽ được hưởng. Bên yêu cầu phải có nghĩa vụ chứng minh tổn thất.
   - Án lệ & Thực tiễn: VIAC và Tòa án bác bỏ các điều khoản quy định mức bồi thường ước tính cố định nếu bên yêu cầu không xuất trình được hóa đơn, chứng từ chứng minh thiệt hại thực tế.
   - Hướng Redline: Loại bỏ câu chữ *"không cần chứng minh thiệt hại"*; quy định cơ chế lập biên bản xác định thiệt hại thực tế trực tiếp.

3. THỎA THUẬN TRỌNG TÀI BẤT KHẢ THI (Điều 43 Luật TTTM; Nghị quyết 01/2014/NQ-HĐTP):
   - Luật định: Thỏa thuận trọng tài không thể thực hiện được nếu các bên thỏa thuận vừa Trọng tài vừa Tòa án, hoặc đặt sai tên trung tâm trọng tài không tồn tại.
   - Án lệ & Thực tiễn: Nếu hợp đồng ghi *"tranh chấp được giải quyết tại VIAC hoặc Tòa án có thẩm quyền"* mà khi xảy ra tranh chấp nguyên đơn kiện ra Tòa án, Tòa án sẽ thụ lý giải quyết.
   - Hướng Redline: Chọn duy nhất một cơ quan tài phán rõ ràng (hoặc VIAC tại TP.HCM/Hà Nội theo Quy tắc Trọng tài VIAC, hoặc Tòa án nhân dân có thẩm quyền).

4. LÃI CHẬM TRẢ & VÔ HIỆU THỎA THUẬN LÃI KÉP (Điều 306 LTM; Điều 468 BLDS; Án lệ 09/2016/AL; NQ 01/2019/NQ-HĐTP):
   - Luật định: Tiền lãi trên nợ gốc quá hạn theo lãi suất nợ quá hạn trung bình trên thị trường của ít nhất 3 ngân hàng thương mại lớn tại thời điểm thanh toán.
   - Án lệ & Thực tiễn: Án lệ 09/2016/AL xác định cách tính lãi chậm trả trong quan hệ thương mại. TAND kiên quyết bác bỏ điều khoản tính lãi trên nợ lãi (lãi chồng lãi / compound interest).
   - Hướng Redline: Điều chỉnh mức lãi suất về mức lãi suất quá hạn trung bình theo luật thương mại; bỏ điều khoản lãi kép.

5. CAM KẾT KHÔNG CẠNH TRANH (NCA) XÂM PHẠM QUYỀN TỰ DO LAO ĐỘNG (Điều 5, 10 BLLĐ 2019; Điều 35 Hiến pháp 2013):
   - Luật định: Người lao động có quyền tự do lựa chọn việc làm, làm việc cho bất kỳ người sử dụng lao động nào mà pháp luật không cấm.
   - Án lệ & Thực tiễn: TAND có nhiều bản án tuyên vô hiệu thỏa thuận cấm người lao động làm việc cho đối thủ cạnh tranh sau khi nghỉ việc. VIAC có thể chấp nhận nếu là thỏa thuận dân sự độc lập có đền bù tài chính tương xứng.
   - Hướng Redline: Chuyển hóa thành *"Thỏa thuận bảo mật thông tin & sở hữu trí tuệ (NDA)"* thay vì cấm làm việc; bổ sung điều khoản hỗ trợ tài chính độc lập.

6. ĐƠN PHƯƠNG CHẤM DỨT HỢP ĐỒNG BẤT HỢP PHÁP (Điều 428 Bộ luật Dân sự 2015):
   - Luật định: Một bên chỉ có quyền đơn phương chấm dứt hợp đồng nếu bên kia vi phạm nghiêm trọng nghĩa vụ hợp đồng hoặc luật có quy định. Bên đơn phương phải thông báo ngay cho bên kia.
   - Án lệ & Thực tiễn: Thỏa thuận cho phép một bên đơn phương hủy bỏ hợp đồng ngay lập tức mà không cho bên kia thời gian khắc phục (Cure Period) bị TAND coi là đơn phương trái pháp luật và buộc bồi thường toàn bộ thiệt hại.
   - Hướng Redline: Bắt buộc bổ sung nghĩa vụ thông báo bằng văn bản và ấn định thời hạn khắc phục vi phạm tối thiểu 15 đến 30 ngày trước khi áp dụng chế tài chấm dứt.

7. MỞ RỘNG BẤT KHẢ KHÁNG TRÁI QUY ĐỊNH (Điều 156, 351 Bộ luật Dân sự 2015):
   - Luật định: Sự kiện bất khả kháng là sự kiện xảy ra một cách khách quan không thể lường trước được và không thể khắc phục được mặc dù đã áp dụng mọi biện pháp cần thiết và khả năng cho phép.
   - Án lệ & Thực tiễn: VIAC và Tòa án từ chối công nhận các biến động về giá thị trường, trượt giá vật tư, biến động tỷ giá ngoại tệ, hoặc việc nhà thầu phụ giao hàng chậm là sự kiện bất khả kháng.
   - Hướng Redline: Chuẩn hóa điều khoản bất khả kháng về đúng 3 yếu tố cấu thành theo BLDS; bổ sung cơ chế đàm phán lại khi hoàn cảnh thay đổi cơ bản (Điều 420 BLDS).

8. MIỄN TRỪ TRÁCH NHIỆM DO LỖI CỐ Ý (Điều 360 Bộ luật Dân sự 2015):
   - Luật định: Thỏa thuận không phải bồi thường thiệt hại trong trường hợp bên có nghĩa vụ có lỗi cố ý là vô hiệu.
   - Án lệ & Thực tiễn: Điều khoản loại trừ toàn bộ trách nhiệm bồi thường do hành vi cố ý vi phạm hợp đồng hoặc gian lận đều bị Tòa án tuyên vô hiệu tuyệt đối.
   - Hướng Redline: Bổ sung điều khoản ngoại trừ bắt buộc: *"Ngoại trừ trường hợp vi phạm do lỗi cố ý hoặc lừa dối"*.

9. HỢP ĐỒNG GIA NHẬP BẤT BÌNH ĐẲNG (Điều 406 Bộ luật Dân sự 2015):
   - Luật định: Trường hợp hợp đồng theo mẫu có điều khoản miễn trách nhiệm của bên đưa ra hợp đồng theo mẫu, tăng trách nhiệm hoặc loại bỏ quyền lợi chính đáng của bên kia thì điều khoản này vô hiệu.
   - Án lệ & Thực tiễn: Thẩm phán TAND sẽ bác bỏ các điều khoản đơn phương giải thích hợp đồng hoặc tước bỏ quyền khiếu nại của đối tác ký hợp đồng theo mẫu.
   - Hướng Redline: Xây dựng cơ chế thương lượng và khiếu nại bình đẳng hai chiều.
"""


# ============================================================================
# 3. NGÂN HÀNG VÍ DỤ FEW-SHOT TRANH TỤNG MẪU (FEW-SHOT DEMONSTRATION BANK)
# ============================================================================

FEW_SHOT_DISPUTE_EXAMPLES = """
=== CÁC VÍ DỤ THẨM ĐỊNH MẪU ĐẠT CHUẨN (FEW-SHOT EXAMPLES) ===

--- VÍ DỤ 1: Phạt vi phạm vượt trần 8% trong hợp đồng thương mại ---
[Điều khoản hợp đồng]:
"Điều 8.2: Nếu Bên B chậm giao hàng quá 10 ngày, Bên B phải chịu phạt vi phạm hợp đồng bằng 20% tổng giá trị hợp đồng và bồi thường toàn bộ thiệt hại phát sinh."
[Căn cứ luật tra cứu]:
- Điều 301 Luật Thương mại 2005: "Mức phạt đối với vi phạm nghĩa vụ hợp đồng... do các bên thỏa thuận, nhưng không quá 8% giá trị phần nghĩa vụ hợp đồng bị vi phạm."
[Kết quả thẩm định]:
{
  "finding_id": "finding_001",
  "clause_reference": "Điều 8.2",
  "statutory_citation": "Điều 301 Luật Thương mại 2005",
  "statutory_article_id": "VN_LTM_2005_D301",
  "risk_tier": "HIGH",
  "vulnerability_classification": "PENALTY_EXCEEDING_STATUTORY_CAP",
  "precedent_reference": "Điều 301 Luật Thương mại 2005 & Thực tiễn Trọng tài VIAC",
  "audit_verdict": "Mức phạt 20% tổng giá trị hợp đồng vi phạm trực tiếp trần khống chế 8% giá trị phần nghĩa vụ bị vi phạm theo Điều 301 Luật Thương mại. Nếu phát sinh tranh chấp tại VIAC hoặc TAND, cơ quan tài phán sẽ tuyên vô hiệu phần phạt vượt quá và chỉ chấp nhận mức tối đa 8% của phần giao hàng chậm.",
  "proposed_redline": "Nếu Bên B chậm giao hàng quá 10 ngày mà không có lý do chính đáng, Bên B phải chịu phạt vi phạm bằng 8% giá trị phần hàng hóa chậm bàn giao và bồi thường thiệt hại thực tế trực tiếp được chứng minh theo quy định của pháp luật.",
  "negotiation_rationale": "Việc giữ mức phạt 20% là bất khả thi khi ra tòa và gây áp lực pháp lý vô căn cứ; điều chỉnh về 8% phần vi phạm bảo đảm giá trị cưỡng chế thi hành cao nhất."
}

--- VÍ DỤ 2: Thỏa thuận trọng tài không thể thực hiện (Pathological Clause) ---
[Điều khoản hợp đồng]:
"Điều 14: Mọi tranh chấp phát sinh từ hoặc liên quan đến hợp đồng này sẽ được giải quyết tại Trung tâm Trọng tài Quốc tế Việt Nam (VIAC) hoặc Tòa án nhân dân TP.HCM."
[Căn cứ luật tra cứu]:
- Điều 43 Luật Trọng tài thương mại 2010; Nghị quyết 01/2014/NQ-HĐTP của Hội đồng Thẩm phán TANDTC.
[Kết quả thẩm định]:
{
  "finding_id": "finding_002",
  "clause_reference": "Điều 14",
  "statutory_citation": "Điều 43 Luật Trọng tài thương mại 2010",
  "statutory_article_id": "VN_LTTTM_2010_D043",
  "risk_tier": "CRITICAL",
  "vulnerability_classification": "PATHOLOGICAL_ARBITRATION_CLAUSE",
  "precedent_reference": "Nghị quyết 01/2014/NQ-HĐTP",
  "audit_verdict": "Thỏa thuận tài phán lựa chọn đồng thời cả VIAC hoặc TAND mà không xác định cơ quan duy nhất trước khi khởi kiện là thỏa thuận trọng tài không thể thực hiện được theo Nghị quyết 01/2014/NQ-HĐTP. Nếu một bên nộp đơn khởi kiện ra TAND TP.HCM, Tòa án sẽ thụ lý giải quyết và loại bỏ thẩm quyền của Trọng tài VIAC.",
  "proposed_redline": "Mọi tranh chấp phát sinh từ hoặc liên quan đến hợp đồng này sẽ được giải quyết chung thẩm tại Trung tâm Trọng tài Quốc tế Việt Nam (VIAC) theo Quy tắc tố tụng trọng tài của Trung tâm này. Địa điểm trọng tài tại TP. Hồ Chí Minh. Ngôn ngữ trọng tài là Tiếng Việt.",
  "negotiation_rationale": "Lựa chọn duy nhất một diễn đàn tài phán (VIAC) để tránh tranh chấp thẩm quyền kéo dài giữa Tòa án và Trọng tài, đồng thời đảm bảo phán quyết có tính chung thẩm và bảo mật thương mại."
}
"""


# ============================================================================
# 4. COMPOSABLE DYNAMIC PROMPT BUILDERS
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
    arithmetic_findings: Optional[List[Dict[str, Any]]] = None,
    critique_feedback: Optional[str] = None,
    include_doctrine: bool = True,
    include_few_shots: bool = True,
) -> str:
    """
    Constructs dual-perspective adversarial audit prompt for Node 3 (Qwen2.5-7B-Instruct).
    Combines TAND/VIAC jurisprudence, grounded statutes, and feedback loop critique.
    """
    arithmetic_findings = arithmetic_findings or []

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
        statutes_text_list.append(f"- [{cit}] (ID: {c_id}): {cnt}")
    statutes_block = "\n".join(statutes_text_list) if statutes_text_list else "Không có dữ liệu luật tra cứu."

    # 3. Format Arithmetic Findings
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

    # 4. Feedback Critique from Hallucination Grader (Self-Reflection Loop)
    critique_block = ""
    if critique_feedback:
        critique_block = (
            f"\n[CẢNH BÁO TỰ SỬA LỖI TỪ VÒNG KIỂM TOÁN TRƯỚC]:\n"
            f"{critique_feedback}\n"
            f"-> BẠN BẮT BUỘC PHẢI KHẮC PHỤC TRIỆT ĐỂ CÁC LỖI TRÍCH DẪN TRÊN!\n"
        )

    # 5. Assemble prompt blocks
    parts = [DUAL_PERSPECTIVE_AUDITOR_SYSTEM_PROMPT]

    if include_doctrine:
        parts.append(TAND_VIAC_JURISPRUDENCE_DOCTRINE)

    if include_few_shots:
        parts.append(FEW_SHOT_DISPUTE_EXAMPLES)

    parts.append(critique_block)
    parts.append(
        f"=== DANH SÁCH ĐIỀU KHOẢN HỢP ĐỒNG CẦN THẨM ĐỊNH ===\n"
        f"{clauses_block}\n"
    )
    parts.append(
        f"=== NGUỒN CĂN CỨ LUẬT ĐÃ TRA CỨU (GROUNDED CONTEXT) ===\n"
        f"{statutes_block}\n"
    )
    if arith_block:
        parts.append(arith_block)

    parts.append(
        f"HÃY THỰC HIỆN KIỂM TOÁN VÀ XUẤT RA DỮ LIỆU ĐÚNG CHUẨN CẤU TRÚC AuditDraftReport:\n"
        f"1. Đối chiếu từng điều khoản với căn cứ luật và tiền lệ TAND/VIAC.\n"
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

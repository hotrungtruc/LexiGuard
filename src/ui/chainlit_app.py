"""
Chainlit UI for interactive contract uploads, multi-agent auditing, and risk reports.
======================================================================================
Connects directly to the compiled LangGraph StateGraph (lexiguard_agent) with real-time
step visualization for ClauseExtractor, LegalResearcher, LoopholeAuditor, and HallucinationGrader.
"""

import os
import tempfile
from typing import Optional

import chainlit as cl
from loguru import logger

from src.agents.state import AgentState, ContractAuditReport
from src.agents.workflow import lexiguard_agent


@cl.on_chat_start
async def start():
    """Initializes the Chainlit user session with guidance and file upload instructions."""
    logger.info("Initializing Chainlit user session...")
    welcome_message = """# 🛡️ LexiGuard — Hệ Thống Đa Tác Tử Thẩm Định Lỗ Hổng Hợp Đồng

Chào mừng bạn đến với **LexiGuard** — Hệ thống Đa tác tử AI kiểm toán & rà soát pháp lý đa lĩnh vực (hỗ trợ toàn diện cho mọi đạo luật, bộ luật, nghị định, thông tư hoặc quy chế nội bộ được nạp vào cơ sở tri thức).

### 🚀 Cách sử dụng:
1. **Tải lên tài liệu**: Nhấp vào biểu tượng đính kèm để tải lên tệp hợp đồng (`.pdf`, `.docx`, `.md`).
2. **Hoặc dán trực tiếp văn bản**: Dán nội dung các điều khoản hợp đồng cần kiểm tra vào khung chat.
3. Hệ thống sẽ tự động kích hoạt Hội đồng Đa tác tử (Multi-Agent Council) để trích xuất điều khoản, tra cứu pháp luật và đề xuất **Redline** phòng ngừa rủi ro tranh chấp!
"""
    await cl.Message(content=welcome_message).send()


@cl.on_message
async def main(message: cl.Message):
    """Processes user input (file attachment or pasted text) through LangGraph workflow."""
    logger.info(f"Received user message (elements={len(message.elements or [])})")

    temp_path: Optional[str] = None
    contract_name = "Hợp đồng thẩm định"

    # 1. Determine input source: file upload or pasted text
    if message.elements:
        file_el = message.elements[0]
        contract_name = file_el.name or "Uploaded_Contract.pdf"
        temp_path = file_el.path
        logger.info(f"User uploaded document via Chainlit: {contract_name} ({temp_path})")
    elif message.content and message.content.strip():
        # User pasted contract text
        contract_name = "Contract_Snippet.md"
        fd, temp_path = tempfile.mkstemp(suffix=".md", prefix="lexiguard_snippet_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(message.content)
        logger.info(f"User provided verbatim contract snippet saved to: {temp_path}")
    else:
        await cl.Message(
            content="⚠️ Vui lòng tải lên tệp hợp đồng (.pdf, .docx, .md) hoặc dán nội dung điều khoản cần thẩm định."
        ).send()
        return

    # Initial AgentState
    initial_state: AgentState = {
        "raw_document_path": temp_path,
        "parsed_clauses": [],
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

    final_report: Optional[ContractAuditReport] = None

    try:
        # Step 1: Clause Extractor
        async with cl.Step(name="📄 Bước 1: ClauseExtractor (Qwen2.5-3B)", type="tool") as step1:
            step1.output = (
                "Đang phân tích cấu trúc văn bản, bóc tách điều khoản và bảng biểu tài chính..."
            )

            # Run LangGraph streaming
            async for event in lexiguard_agent.astream(initial_state):
                for node_name, node_output in event.items():
                    if node_name == "extract_clauses":
                        clauses = node_output.get("parsed_clauses", [])
                        step1.output = f"✅ Đã trích xuất thành công {len(clauses)} điều khoản hợp đồng và bảng biểu."

                    elif node_name == "retrieve_statutes":
                        async with cl.Step(
                            name="⚖️ Bước 2: LegalResearcher (FastMCP & GraphRAG)", type="tool"
                        ) as step2:
                            statutes = node_output.get("retrieved_statutes", [])
                            precedents = node_output.get("retrieved_precedents", [])
                            arith = node_output.get("arithmetic_findings", [])
                            step2.output = (
                                f"✅ Tra cứu thành công {len(statutes)} căn cứ luật, "
                                f"{len(precedents)} án lệ TAND/VIAC và đối soát {len(arith)} bảng biểu số học."
                            )

                    elif node_name == "audit_vulnerabilities":
                        async with cl.Step(
                            name="🔍 Bước 3: LoopholeAuditor (Qwen2.5-7B)", type="tool"
                        ) as step3:
                            draft = node_output.get("audit_draft")
                            findings = draft.findings if draft else []
                            step3.output = f"✅ Phát hiện {len(findings)} điểm rủi ro pháp lý tiềm ẩn và soạn thảo Redline."

                    elif node_name == "grade_hallucination":
                        async with cl.Step(
                            name="🛡️ Bước 4: HallucinationGrader (Cổng Thẩm Định Chống Ảo Giác)",
                            type="tool",
                        ) as step4:
                            grade = node_output.get("hallucination_grade")
                            rep = node_output.get("final_report")
                            if rep:
                                final_report = rep
                            score = grade.citation_accuracy_score if grade else 1.0
                            is_grounded = grade.is_grounded if grade else True
                            step4.output = (
                                f"✅ Kiểm định trích dẫn hoàn tất (Độ tin cậy: {score * 100:.0f}%). "
                                f"Trạng thái: {'Hoàn toàn chuẩn xác' if is_grounded else 'Đã loại bỏ các trích dẫn ngoài context'}."
                            )

        # Render Final Formatted Report
        if final_report:
            score = final_report.overall_health_score
            badge_color = "🟢" if score >= 80 else ("🟡" if score >= 50 else "🔴")

            report_md = [
                f"# 📊 BÁO CÁO THẨM ĐỊNH HỢP ĐỒNG: {contract_name}",
                f"\n### {badge_color} Điểm sức khỏe hợp đồng: **{score}/100**",
                f"- **Tổng số điều khoản rà soát**: {final_report.total_clauses_reviewed}",
                f"- **Số rủi ro pháp lý phát hiện**: {len(final_report.findings)}",
                f"\n> **Đánh giá rủi ro tranh chấp TAND / VIAC**:\n> {final_report.tand_viac_summary}\n",
                "---\n### 🚨 Danh Sách Điểm Bất Cập & Đề Xuất Redline Sửa Đổi:\n",
            ]

            if not final_report.findings:
                report_md.append(
                    "🎉 **Hợp đồng đạt chuẩn an toàn cao, không phát hiện vi phạm luật định nghiêm trọng.**"
                )
            else:
                for idx, f in enumerate(final_report.findings, start=1):
                    risk_icon = (
                        "🔴"
                        if f.risk_tier == "CRITICAL"
                        else ("🟠" if f.risk_tier == "HIGH" else "🟡")
                    )
                    disp_text = (
                        f" ({f.dispute_insight.target_forum} - {f.dispute_insight.precedent_reference})"
                        if f.dispute_insight
                        else ""
                    )

                    finding_block = f"""
#### {idx}. [{risk_icon} {f.risk_tier}] {f.clause_reference}
- **Phân loại lỗ hổng**: `{f.vulnerability_classification}`{disp_text}
- **Căn cứ pháp lý**: **{f.legal_grounding.statute_title}** ({f.legal_grounding.article})
- **Phân tích rủi ro**: {f.audit_verdict}
- **Đề xuất Redline sửa đổi**:
```diff
{f.proposed_redline or "Chưa có đề xuất thay thế cụ thể."}
```
- **Luận cứ đàm phán**: *{f.negotiation_rationale or "N/A"}*
---
"""
                    report_md.append(finding_block)

            await cl.Message(content="\n".join(report_md)).send()
        else:
            await cl.Message(
                content="⚠️ Quá trình thẩm định đã hoàn thành nhưng chưa tạo được báo cáo tổng hợp. Vui lòng thử lại."
            ).send()

    except Exception as exc:
        logger.error(f"Error executing Chainlit audit workflow: {exc}")
        await cl.Message(content=f"❌ Đã xảy ra lỗi trong quá trình thẩm định: {exc}").send()
    finally:
        # Clean up temp file if created from pasted snippet
        if temp_path and "lexiguard_snippet_" in temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass

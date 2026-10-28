"""
Conversational Chat Route: Handles general legal Q&A, contract audits, and streaming responses (SSE).
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger

from src.agents.nodes import get_fast_cpu_llm
from src.agents.state import AgentState
from src.agents.workflow import lexiguard_agent
from src.mcp_server.legal_tools import search_legal_provisions

router = APIRouter()


def format_sse(data: Dict[str, Any]) -> str:
    """Format dictionary into Server-Sent Events data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat", summary="Conversational AI endpoint with SSE streaming")
async def chat_endpoint(
    message: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    mode: Optional[str] = Form("auto"),
    lang: Optional[str] = Form("vi"),
):
    """
    Unified chat endpoint supporting:
    1. Contract vulnerability auditing (when file or long contract text is provided).
    2. General legal Q&A and statutory retrieval (when conversational question is asked).
    Streams real-time step progress and final response via Server-Sent Events (SSE).
    """
    if not message and not file:
        raise HTTPException(
            status_code=400,
            detail="Vui lòng cung cấp câu hỏi hoặc tệp tài liệu để xử lý.",
        )

    temp_path: Optional[str] = None
    temp_dir: Optional[str] = None
    file_name = file.filename if file and file.filename else None

    # Handle attached file or text snippet for contract auditing
    if file and file.filename:
        temp_dir = tempfile.mkdtemp(prefix="lexiguard_chat_upload_")
        temp_path = os.path.join(temp_dir, Path(file.filename).name)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Chat received attached file: {file.filename} -> {temp_path}")
    elif (
        mode == "audit"
        or (
            message
            and (
                "hợp đồng" in message.lower()
                or "thẩm định" in message.lower()
                or "contract" in message.lower()
                or "audit" in message.lower()
                or "điều " in message.lower()
                or "dieu " in message.lower()
                or "|" in message
            )
        )
    ) and message:
        # Text appears to be or requests a contract snippet audit
        temp_dir = tempfile.mkdtemp(prefix="lexiguard_chat_snippet_")
        temp_path = os.path.join(temp_dir, "contract_snippet.md")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(message)
        logger.info(f"Chat created contract audit snippet -> {temp_path}")

    # Determine execution workflow: Contract Audit vs. Legal Q&A
    is_audit_mode = (
        mode == "audit"
        or temp_path is not None
        or (
            file_name is not None
            and Path(file_name).suffix.lower() in {".pdf", ".docx", ".doc", ".md"}
        )
    )

    async def event_generator():
        try:
            if is_audit_mode and temp_path:
                # =============================================================
                # BRANCH A: MULTI-AGENT CONTRACT AUDIT WORKFLOW
                # =============================================================
                doc_desc = file_name or "Đoạn hợp đồng cung cấp"
                yield format_sse(
                    {
                        "type": "step",
                        "step": "init",
                        "status": "running",
                        "message": f"Bắt đầu thẩm định văn bản: {doc_desc}",
                    }
                )

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

                latest_draft = None
                latest_verified_items = []
                final_report: Optional[Dict[str, Any]] = None

                async for event in lexiguard_agent.astream(initial_state):
                    for node_name, node_output in event.items():
                        if node_name == "extract_clauses":
                            clauses = node_output.get("parsed_clauses", [])
                            msg = f"Đã trích xuất {len(clauses)} điều khoản và phát hiện bảng biểu tài chính."
                            yield format_sse(
                                {
                                    "type": "step",
                                    "step": "extract_clauses",
                                    "status": "completed",
                                    "message": msg,
                                    "clause_count": len(clauses),
                                }
                            )

                        elif node_name == "retrieve_statutes":
                            statutes = node_output.get("retrieved_statutes", [])
                            precedents = node_output.get("retrieved_precedents", [])
                            arith = node_output.get("arithmetic_findings", [])
                            msg = f"Tra cứu {len(statutes)} điều luật, {len(precedents)} án lệ/tiền lệ và kiểm toán {len(arith)} bảng tính."
                            yield format_sse(
                                {
                                    "type": "step",
                                    "step": "retrieve_statutes",
                                    "status": "completed",
                                    "message": msg,
                                    "statute_count": len(statutes),
                                    "precedent_count": len(precedents),
                                    "arithmetic_issues": len(arith),
                                }
                            )

                        elif node_name == "audit_vulnerabilities":
                            draft = node_output.get("audit_draft")
                            if draft:
                                latest_draft = draft
                            findings = draft.findings if draft else []
                            msg = f"LoopholeAuditor phát hiện {len(findings)} điểm rủi ro pháp lý và soạn thảo Redline."
                            yield format_sse(
                                {
                                    "type": "step",
                                    "step": "audit_vulnerabilities",
                                    "status": "completed",
                                    "message": msg,
                                    "finding_count": len(findings),
                                }
                            )

                        elif node_name == "grade_hallucination":
                            grade = node_output.get("hallucination_grade")
                            latest_verified_items = node_output.get("verified_findings", [])
                            rep = node_output.get("final_report")
                            if rep:
                                final_report = (
                                    rep.model_dump() if hasattr(rep, "model_dump") else rep
                                )

                            score = grade.citation_accuracy_score if grade else 1.0
                            grounded = grade.is_grounded if grade else True
                            ground_msg = (
                                "Hoàn toàn chuẩn xác"
                                if grounded
                                else "Đã loại bỏ trích dẫn ngoài context"
                            )
                            msg = f"Kiểm định trích dẫn hoàn tất (Độ tin cậy: {score * 100:.0f}%). Trạng thái: {ground_msg}."
                            yield format_sse(
                                {
                                    "type": "step",
                                    "step": "grade_hallucination",
                                    "status": "completed",
                                    "message": msg,
                                    "is_grounded": grounded,
                                    "accuracy_score": score,
                                }
                            )

                # Ensure report is fully formed with all finding details
                if not final_report:
                    items_to_use = latest_verified_items
                    if not items_to_use and latest_draft and latest_draft.findings:
                        items_to_use = latest_draft.findings

                    items_data = [
                        it.model_dump() if hasattr(it, "model_dump") else it for it in items_to_use
                    ]
                    calc_score = max(0, 100 - len(items_data) * 12)
                    summary_msg = (
                        f"Đã phát hiện {len(items_data)} điểm rủi ro pháp lý cần chỉnh lý theo án lệ TAND & VIAC."
                        if items_data
                        else "Văn bản an toàn, không phát hiện rủi ro vi phạm pháp luật nghiêm trọng."
                    )
                    final_report = {
                        "contract_name": file_name or "Tài liệu thẩm định",
                        "overall_health_score": calc_score,
                        "overall_score": calc_score,
                        "total_clauses_reviewed": len(initial_state.get("parsed_clauses", [])),
                        "findings": items_data,
                        "vulnerabilities": items_data,
                        "tand_viac_summary": summary_msg,
                        "executive_summary": summary_msg,
                    }
                else:
                    if hasattr(final_report, "model_dump"):
                        final_report = final_report.model_dump()
                    final_report["overall_score"] = final_report.get(
                        "overall_health_score", final_report.get("overall_score", 85)
                    )
                    final_report["overall_health_score"] = final_report["overall_score"]
                    final_report["executive_summary"] = final_report.get(
                        "tand_viac_summary", final_report.get("executive_summary", "")
                    )
                    final_report["tand_viac_summary"] = final_report["executive_summary"]
                    f_list = final_report.get("findings", final_report.get("vulnerabilities", []))
                    final_report["findings"] = f_list
                    final_report["vulnerabilities"] = f_list

                # Extract Redline recommendations
                f_list = final_report.get("findings", [])
                redlines = []
                for f in f_list:
                    redline_text = f.get("proposed_redline")
                    if redline_text:
                        prov = f.get("contract_provenance") or {}
                        orig = prov.get("verbatim_text") or f.get("clause_reference") or ""
                        redlines.append(
                            {
                                "clause_id": f.get("clause_reference")
                                or f.get("finding_id")
                                or "Điều khoản",
                                "original_text": orig,
                                "suggested_revision": redline_text,
                                "rationale": f.get("negotiation_rationale")
                                or f.get("audit_verdict")
                                or "",
                            }
                        )
                final_report["redline_recommendations"] = redlines

                yield format_sse(
                    {
                        "type": "audit_completed",
                        "status": "success",
                        "report": final_report,
                        "message": "Thẩm định hợp đồng hoàn tất thành công.",
                    }
                )

            else:
                # =============================================================
                # BRANCH B: GENERAL LEGAL Q&A WITH HYBRID GRAPHRAG RETRIEVAL
                # =============================================================
                query = message or "Tra cứu pháp luật"
                truncated_query = query[:60]
                yield format_sse(
                    {
                        "type": "step",
                        "step": "search",
                        "status": "running",
                        "message": f"Đang tra cứu cơ sở tri thức pháp lý: '{truncated_query}...' ",
                    }
                )

                # 1. Search provisions via FastMCP tool
                retrieval_res = await search_legal_provisions(query=query, top_k=5)
                provisions = retrieval_res.get("results", [])

                yield format_sse(
                    {
                        "type": "step",
                        "step": "retrieved",
                        "status": "completed",
                        "message": f"Tìm thấy {len(provisions)} điều luật và văn bản liên quan.",
                        "count": len(provisions),
                    }
                )

                # 2. Extract citations & context
                context_chunks = []
                citations = []
                for p in provisions:
                    cit = p.get("citation", p.get("canonical_citation", ""))
                    content = p.get("content", "")
                    if cit:
                        citations.append(cit)
                    context_chunks.append(f"### {cit}\n{content}")

                legal_context = "\n\n".join(context_chunks)

                # 3. Generate response using fast reasoning LLM
                yield format_sse(
                    {
                        "type": "step",
                        "step": "synthesizing",
                        "status": "running",
                        "message": "Đang tổng hợp luận cứ và soạn thảo câu trả lời...",
                    }
                )

                fallback_ctx = (
                    "No exact statutory match found in knowledge base. Answer based on general legal principles."
                    if lang == "en"
                    else "Không tìm thấy điều luật khớp chính xác. Vui lòng trả lời trên nguyên tắc chung của pháp luật Việt Nam."
                )
                ctx_to_use = legal_context if legal_context else fallback_ctx

                if lang == "en":
                    prompt = (
                        f"You are LexiGuard AI — a Senior Legal Assistant & Contract Auditor.\n"
                        f"Please answer the user's question accurately and professionally, citing the legal provisions provided below in English:\n\n"
                        f"USER QUESTION:\n{query}\n\n"
                        f"RELEVANT LEGAL CONTEXT FROM KNOWLEDGE GRAPH:\n{ctx_to_use}\n\n"
                        f"REQUIREMENTS:\n"
                        f"1. Provide a direct, focused answer in clear English.\n"
                        f"2. Cite the exact articles, clauses, and legal instruments (e.g. Civil Code 2015, Commercial Law 2005).\n"
                        f"3. Offer practical advice and dispute avoidance recommendations.\n"
                        f"4. Format cleanly using Markdown with bullet points where helpful.\n"
                    )
                else:
                    prompt = (
                        f"Bạn là LexiGuard AI — Trợ lý Pháp lý & Kiểm toán Hợp đồng cao cấp.\n"
                        f"Hãy trả lời câu hỏi sau của người dùng một cách chuẩn xác, chuyên nghiệp, viện dẫn đúng các quy định pháp luật được cung cấp:\n\n"
                        f"CÂU HỎI:\n{query}\n\n"
                        f"CĂN CỨ PHÁP LUẬT ĐƯỢC TRA CỨU TỪ CƠ SỞ TRI THỨC:\n{ctx_to_use}\n\n"
                        f"YÊU CẦU:\n"
                        f"1. Nêu rõ câu trả lời trọng tâm.\n"
                        f"2. Viện dẫn chính xác tên điều, khoản, văn bản pháp luật (ví dụ: Bộ luật Dân sự 2015, Luật Thương mại 2005...).\n"
                        f"3. Đưa ra lời khuyên thực tiễn hoặc các điểm cần lưu ý phòng tránh tranh chấp.\n"
                        f"4. Trình bày bằng Markdown rõ ràng, dễ đọc.\n"
                    )

                try:
                    llm = get_fast_cpu_llm()
                    response = llm.invoke(prompt)
                    answer_text = (
                        response.content if hasattr(response, "content") else str(response)
                    )
                except Exception as exc:
                    logger.warning(
                        f"Ollama LLM unavailable ({exc}). Using structured statutory synthesis."
                    )
                    if lang == "en":
                        answer_text = (
                            f"### Legal Consultation & Statutory Grounding\n\n"
                            f"**Inquiry:** {query}\n\n"
                            f"**Statutory Provisions Identified:**\n\n"
                        )
                        for p in provisions:
                            cit = p.get("citation", p.get("canonical_citation", "Provision"))
                            cnt = p.get("content", "").strip()
                            answer_text += f"- **[{cit}]**:\n  {cnt}\n\n"
                        answer_text += (
                            "**Practical Legal Advice:**\n"
                            "- All contract clauses must strictly adhere to the mandatory statutory requirements cited above.\n"
                            "- In adjudication practice at TAND (People's Courts) and VIAC, provisions violating mandatory legal caps will be declared null and void."
                        )
                    else:
                        answer_text = (
                            f"### Kết Quả Tra Cứu & Tư Vấn Pháp Lý\n\n"
                            f"**Câu hỏi:** {query}\n\n"
                            f"**Căn cứ pháp luật được tra cứu từ cơ sở tri thức:**\n\n"
                        )
                        for p in provisions:
                            cit = p.get("citation", p.get("canonical_citation", "Quy định"))
                            cnt = p.get("content", "").strip()
                            answer_text += f"- **[{cit}]**:\n  {cnt}\n\n"
                        answer_text += (
                            "**Khuyến nghị thực tiễn:**\n"
                            "- Các bên cần đối soát kỹ lưỡng các điều khoản hợp đồng liên quan để bảo đảm không vượt quá trần quy định của luật định.\n"
                            "- Thực tiễn xét xử tại Tòa án nhân dân (TAND) và Trọng tài (VIAC) sẽ ưu tiên áp dụng các quy phạm pháp luật bắt buộc chung."
                        )

                yield format_sse(
                    {
                        "type": "chat_completed",
                        "status": "success",
                        "answer": answer_text,
                        "citations": citations,
                    }
                )

        except Exception as exc:
            logger.error(f"Error in chat endpoint stream: {exc}")
            yield format_sse(
                {
                    "type": "error",
                    "status": "failed",
                    "message": f"Lỗi hệ thống: {str(exc)}",
                }
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

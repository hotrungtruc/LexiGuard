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

    # Handle attached file
    if file and file.filename:
        temp_dir = tempfile.mkdtemp(prefix="lexiguard_chat_upload_")
        temp_path = os.path.join(temp_dir, Path(file.filename).name)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Chat received attached file: {file.filename} -> {temp_path}")
    elif (
        message
        and len(message.strip().splitlines()) >= 5
        and ("Điều " in message or "Khoản " in message or "|" in message)
    ):
        # Text appears to be a pasted contract snippet
        temp_dir = tempfile.mkdtemp(prefix="lexiguard_chat_snippet_")
        temp_path = os.path.join(temp_dir, "pasted_contract_snippet.md")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(message)
        logger.info(f"Chat received pasted contract snippet -> {temp_path}")

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

                fallback_ctx = "Không tìm thấy điều luật khớp chính xác. Vui lòng trả lời trên nguyên tắc chung của pháp luật Việt Nam."
                ctx_to_use = legal_context if legal_context else fallback_ctx

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

                llm = get_fast_cpu_llm()
                response = llm.invoke(prompt)
                answer_text = response.content if hasattr(response, "content") else str(response)

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

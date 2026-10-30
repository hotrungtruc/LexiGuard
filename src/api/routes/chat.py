"""
Conversational Chat Route: Handles general legal Q&A, contract audits, and streaming responses (SSE).
"""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from src.reporting.docx_exporter import generate_contract_audit_docx
from loguru import logger

from src.agents.nodes import get_fast_cpu_llm
from src.agents.state import AgentState
from src.agents.prompts import (
    build_legal_consultation_prompt,
    build_deliberative_qa_prompt,
    extract_thinking_and_answer,
)
from src.core.format_guardian import FormatGuardian
from src.agents.workflow import lexiguard_agent
from src.mcp_server.legal_tools import search_legal_provisions
from src.retrieval.context_grounding import format_grounded_statutory_context

router = APIRouter()


def format_sse(data: Dict[str, Any]) -> str:
    """Format dictionary into Server-Sent Events data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def is_explicit_contract_audit_request(message: str) -> bool:
    """
    Distinguishes whether a message is an explicit contract/clause snippet to be audited
    versus a conversational question about laws or contracts.
    """
    if not message:
        return False
    msg = message.strip()
    lower = msg.lower()

    # Explicit audit action keywords
    audit_keywords = [
        "thẩm định", "soát xét", "rà soát", "kiểm toán hợp đồng",
        "kiểm tra hợp đồng", "audit", "kiểm tra điều khoản",
        "đánh giá rủi ro điều khoản", "redline", "bắt lỗi hợp đồng",
        "kiểm tra hồ sơ", "check hồ sơ", "thẩm định hồ sơ", "rà soát hồ sơ", "soát hồ sơ"
    ]
    if any(k in lower for k in audit_keywords):
        return True

    # If it's an inquiry question (ends with '?' or inquiry phrases), treat as Legal Q&A
    question_starters = [
        "tại sao", "như thế nào", "là bao nhiêu", "cho tôi hỏi",
        "hỏi về", "giải thích", "xin hỏi", "có được", "có được phép", "có bị"
    ]
    is_question = msg.endswith("?") or any(qs in lower for qs in question_starters)
    if is_question and not any(k in lower for k in audit_keywords):
        return False

    # Formal multi-clause or contract text with parties or markdown tables
    has_clause_header = bool(re.search(r"\bđiều\s+\d+[:\.]", lower))
    has_parties = "bên a" in lower or "bên b" in lower or "party a" in lower or "party b" in lower
    has_table = "|" in msg and ("---" in msg or "-|-" in msg)

    if (has_clause_header and (has_parties or len(msg) > 300)) or has_table:
        return True

    return False


def classify_chat_intent(message: Optional[str], has_file: bool = False, mode: str = "auto") -> str:
    """
    Classifies user request into one of three distinct flows:
    1. 'audit': Contract auditing & vulnerability detection (Multi-Agent State Machine).
    2. 'legal_rag': Legal consultation & statutory retrieval (Hybrid GraphRAG search).
    3. 'general_chat': Standard chitchat, greetings, general questions (Direct LLM, no RAG, no thinking box).
    """
    # 1. Respect explicit mode from client if specified
    if mode == "audit":
        return "audit"
    if mode == "chat":
        return "general_chat"
    if mode == "legal":
        return "legal_rag"

    # 2. Check for file upload -> Always Contract Audit
    if has_file:
        return "audit"

    if not message or not message.strip():
        return "general_chat"

    msg = message.strip()
    lower = msg.lower()

    # 3. Check for explicit contract audit request or contract snippet
    if is_explicit_contract_audit_request(msg):
        return "audit"

    # 4. Check for common greetings and general chitchat
    clean_text = re.sub(r"[^\w\s]", "", lower).strip()

    greetings = {
        "chào", "chao", "xin chào", "xin chao", "chào bạn", "chao ban", "chào em", 
        "chào anh", "chào chị", "chào ad", "chào bot", "chào trợ lý", "hello", "hi", 
        "hey", "alo", "yo", "good morning", "good afternoon", "good evening", 
        "hi bot", "hello bot", "bắt đầu"
    }

    intro_queries = {
        "bạn là ai", "ban la ai", "bạn tên gì", "ban ten gi", "bạn làm được gì", 
        "ban lam duoc gi", "bạn có thể làm gì", "giới thiệu", "gioi thieu", 
        "giới thiệu về bạn", "hướng dẫn", "huong dan", "hướng dẫn sử dụng", 
        "who are you", "what can you do", "help"
    }

    courtesy_phrases = {
        "cảm ơn", "cam on", "cảm ơn bạn", "cảm ơn nhiều", "thanks", "thank you", 
        "tạm biệt", "tam biet", "bye", "bye bye", "hẹn gặp lại", "ok", "oke", 
        "được rồi", "rất tốt", "tuyệt vời", "good"
    }

    if clean_text in greetings or clean_text in intro_queries or clean_text in courtesy_phrases:
        return "general_chat"

    starts_with_greeting = any(lower.startswith(g) for g in ["chào ", "xin chào ", "hello ", "hi "])

    # 5. Check for substantive legal keywords
    legal_keywords = [
        "luật", "bộ luật", "nghị định", "thông tư", "quy chuẩn", "pháp lệnh", "văn bản quy phạm",
        "điều ", "khoản ", "điểm ", "quy định", "pháp luật", "pháp lý",
        "mức phạt", "phạt tiền", "xử phạt", "phạt nguội", "tước bằng",
        "vi phạm", "tội phạm", "truy cứu", "hình sự", "dân sự", "hành chính",
        "lao động", "bảo hiểm xã hội", "bhxh", "tiền lương", "hợp đồng lao động", "sa thải", "nghỉ việc",
        "thuế", "thuế thu nhập", "vat", "doanh nghiệp", "thành lập công ty", "giải thể", "phá sản",
        "đất đai", "sổ đỏ", "sổ hồng", "chuyển nhượng", "bất động sản", "quyền sử dụng đất",
        "thừa kế", "di chúc", "ly hôn", "hôn nhân", "chia tài sản", "nuôi con",
        "bồi thường", "thiệt hại", "khởi kiện", "tố tụng", "tòa án", "trọng tài", "viac", "tand", "kháng cáo",
        "thời hiệu", "lãi suất", "vay nợ", "quỵt nợ", "công chứng", "chứng thực",
        "sở hữu trí tuệ", "bản quyền", "nhãn hiệu", "sáng chế",
        "law", "statute", "decree", "circular", "penalty", "fine", "court", "arbitration"
    ]

    has_legal_intent = any(k in lower for k in legal_keywords)
    if has_legal_intent:
        return "legal_rag"

    if starts_with_greeting:
        return "general_chat"

    return "general_chat"


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
    3. Direct chitchat & natural conversation (no RAG, no agent thinking box).
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

    # Classify execution intent: 'audit', 'legal_rag', or 'general_chat'
    intent = classify_chat_intent(
        message=message,
        has_file=bool(file and file.filename),
        mode=mode or "auto",
    )

    # Handle attached file or text snippet for contract auditing
    if file and file.filename:
        temp_dir = tempfile.mkdtemp(prefix="lexiguard_chat_upload_")
        temp_path = os.path.join(temp_dir, Path(file.filename).name)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Chat received attached file: {file.filename} -> {temp_path}")
    elif intent == "audit" and message:
        temp_dir = tempfile.mkdtemp(prefix="lexiguard_chat_snippet_")
        temp_path = os.path.join(temp_dir, "contract_snippet.md")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(message)
        logger.info(f"Chat created contract audit snippet -> {temp_path}")

    async def event_generator():
        try:
            if intent == "audit" and temp_path:
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
                    dyn_summary = latest_draft.summary_notes if latest_draft and latest_draft.summary_notes else None
                    summary_msg = (
                        dyn_summary
                        if dyn_summary
                        else (
                            f"Đã phát hiện {len(items_data)} điểm rủi ro pháp lý cần chỉnh lý theo án lệ TAND & VIAC."
                            if items_data
                            else "Văn bản an toàn, không phát hiện rủi ro vi phạm pháp luật nghiêm trọng."
                        )
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

                # Extract 3-Tier Redline recommendations & judicial precedents
                f_list = final_report.get("findings", [])
                redlines = []
                for f in f_list:
                    redline_text = f.get("proposed_redline")
                    r_3t = f.get("redlines") or {}
                    if hasattr(r_3t, "model_dump"):
                        r_3t = r_3t.model_dump()
                    elif not isinstance(r_3t, dict):
                        r_3t = {}

                    if redline_text or r_3t:
                        prov = f.get("contract_provenance") or {}
                        orig = prov.get("verbatim_text") or f.get("clause_reference") or ""
                        strict_rev = r_3t.get("strict_protective") or f.get("strict_revision") or redline_text
                        balanced_rev = r_3t.get("balanced_standard") or f.get("balanced_revision") or redline_text
                        flexible_rev = r_3t.get("flexible_commercial") or f.get("flexible_revision") or redline_text
                        prec_ref = f.get("precedent_reference") or ""
                        if not prec_ref and f.get("dispute_insight"):
                            disp = f.get("dispute_insight")
                            if isinstance(disp, dict):
                                prec_ref = disp.get("precedent_reference") or ""

                        redlines.append(
                            {
                                "clause_id": f.get("clause_reference")
                                or f.get("finding_id")
                                or "Điều khoản",
                                "original_text": orig,
                                "suggested_revision": balanced_rev or redline_text,
                                "strict_revision": strict_rev,
                                "balanced_revision": balanced_rev or redline_text,
                                "flexible_revision": flexible_rev,
                                "precedent_reference": prec_ref,
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

            elif intent == "general_chat":
                # =============================================================
                # BRANCH B: GENERAL CHITCHAT / CASUAL RESPONSE (NO RAG, NO STEPS)
                # =============================================================
                user_msg = message or "Chào bạn"
                if lang == "en":
                    prompt = (
                        f"You are LexiGuard AI — an intelligent Vietnamese Legal Assistant & Contract Risk Auditor.\n"
                        f"User message: '{user_msg}'\n\n"
                        f"Instructions:\n"
                        f"- Respond naturally, warmly, and concisely in English (no rigid template or numbered headers).\n"
                        f"- Address their greeting or inquiry directly.\n"
                        f"- Briefly introduce your key capabilities: fast Vietnamese legal statute search, contract loophole risk auditing, redline suggestions, and financial schedule arithmetic checks.\n"
                        f"- Invite them to ask a legal question or upload a contract document to begin."
                    )
                else:
                    prompt = (
                        f"Bạn là LexiGuard AI — Trợ lý Pháp lý & Kiểm toán Hợp đồng thông minh.\n"
                        f"Tin nhắn của người dùng: '{user_msg}'\n\n"
                        f"Yêu cầu:\n"
                        f"- Trả lời một cách tự nhiên, lịch thiệp, thân thiện và súc tích (TUYỆT ĐỐI KHÔNG dùng barem hay liệt kê cứng nhắc).\n"
                        f"- Phản hồi trực diện câu chào hỏi hoặc nội dung trò chuyện của người dùng.\n"
                        f"- Giới thiệu ngắn gọn các năng lực cốt lõi: tra cứu văn bản pháp luật Việt Nam, thẩm định rủi ro hợp đồng, đề xuất câu chữ Redline sửa đổi, và kiểm toán số học bảng biểu.\n"
                        f"- Mời người dùng đặt câu hỏi pháp lý hoặc tải lên tệp hợp đồng để bắt đầu."
                    )

                try:
                    llm = get_fast_cpu_llm()
                    response = llm.invoke(prompt)
                    answer_text = (
                        response.content if hasattr(response, "content") else str(response)
                    )
                except Exception as exc:
                    logger.warning(
                        f"Ollama LLM unavailable ({exc}). Using friendly default greeting."
                    )
                    if lang == "en":
                        answer_text = (
                            "Hello! I am **LexiGuard AI** — your intelligent Legal Advisor and Contract Risk Auditor.\n\n"
                            "I can assist you with:\n"
                            "- 🔍 **Statutory Search**: Fast retrieval of Vietnamese Laws, Decrees, and Circulars.\n"
                            "- 🛡️ **Contract Auditing**: Detecting risky loophole clauses, penalties, and VIAC/TAND dispute risks.\n"
                            "- ✏️ **Redline Suggestions**: Generating legally balanced replacement wording.\n"
                            "- 🔢 **Arithmetic Audit**: Verifying payment schedules and penalty caps.\n\n"
                            "Feel free to ask a legal question or upload a contract document to begin!"
                        )
                    else:
                        answer_text = (
                            "Xin chào! Tôi là **LexiGuard AI** — Trợ lý Pháp lý & Kiểm toán Hợp đồng thông minh.\n\n"
                            "Tôi có thể đồng hành cùng bạn trong các tác vụ:\n"
                            "- 🔍 **Tra cứu pháp luật**: Tra cứu nhanh quy định trong các Bộ luật, Nghị định, Thông tư hiện hành.\n"
                            "- 🛡️ **Thẩm định hợp đồng**: Rà soát điều khoản bất lợi, bẫy pháp lý và rủi ro tranh chấp TAND / VIAC.\n"
                            "- ✏️ **Đề xuất Redline**: Soạn thảo câu chữ sửa đổi cân bằng quyền lợi cho các bên.\n"
                            "- 🔢 **Kiểm toán số học**: Tự động đối soát bảng tiến độ thanh toán và trần phạt vi phạm.\n\n"
                            "Bạn hãy nhập câu hỏi pháp lý cần giải đáp hoặc tải lên tệp hợp đồng để bắt đầu nhé!"
                        )

                yield format_sse(
                    {
                        "type": "chat_completed",
                        "status": "success",
                        "answer": answer_text,
                        "citations": [],
                    }
                )

            else:
                # =============================================================
                # BRANCH C: GENERAL LEGAL Q&A WITH HYBRID GRAPHRAG RETRIEVAL
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
                provisions = retrieval_res.get("provisions", retrieval_res.get("results", []))

                yield format_sse(
                    {
                        "type": "step",
                        "step": "retrieved",
                        "status": "completed",
                        "message": f"Tìm thấy {len(provisions)} điều luật và văn bản liên quan.",
                        "count": len(provisions),
                    }
                )

                # 2. Extract citations & Phase 2 Grounded Context (Re-ranking + Anchor Highlighting)
                citations = []
                for p in provisions:
                    cit = p.get("citation", p.get("canonical_citation", ""))
                    if cit:
                        citations.append(cit)

                # Phase 2: Proximity Re-ranking + Anchor Highlighting
                legal_context = format_grounded_statutory_context(
                    provisions=provisions,
                    query=query,
                    highlight_anchors=True,
                    ordering_strategy="proximity_to_query",
                )

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

                is_tf = FormatGuardian.is_true_false_query(query)
                is_mcq = FormatGuardian.is_multiple_choice_query(query)

                if is_tf:
                    prompt = build_deliberative_qa_prompt(
                        query=query,
                        legal_context=ctx_to_use,
                        question_type="Đúng/Sai",
                    )
                elif is_mcq:
                    prompt = build_deliberative_qa_prompt(
                        query=query,
                        legal_context=ctx_to_use,
                        question_type="Trắc nghiệm",
                    )
                else:
                    prompt = build_legal_consultation_prompt(
                        query=query,
                        legal_context=ctx_to_use,
                        lang=lang or "vi",
                    )

                try:
                    llm = get_fast_cpu_llm()
                    response = llm.invoke(prompt)
                    raw_text = (
                        response.content if hasattr(response, "content") else str(response)
                    )
                    thinking_content, raw_ans = extract_thinking_and_answer(raw_text)

                    # Sanitize output via FormatGuardian (strip leaked prefixes & enforce binary outputs)
                    if is_tf:
                        answer_text = FormatGuardian.enforce_true_false_format(raw_ans, query)
                    else:
                        answer_text = FormatGuardian.sanitize_conversational_text(raw_ans)

                    # Filter citations against hallucination (e.g. Điều 81 Hiến pháp on arrest)
                    citations = FormatGuardian.filter_hallucinated_citations(citations, provisions, query=query)

                    # If deliberative reasoning scratchpad was generated, emit it as a verified step
                    if thinking_content:
                        yield format_sse(
                            {
                                "type": "step",
                                "step": "reasoning",
                                "status": "completed",
                                "message": "Hoàn tất suy luận đa chiều (IRAC & rà soát ngoại lệ).",
                                "thought": thinking_content,
                            }
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


class ChatExportDocxRequest(BaseModel):
    report: Dict[str, Any] = Field(..., description="Dữ liệu JSON báo cáo thẩm định hợp đồng")
    contract_title: Optional[str] = Field("Báo cáo Thẩm định Hợp đồng", description="Tiêu đề tài liệu hợp đồng")


@router.post("/chat/export-docx")
@router.post("/export-docx")
async def chat_export_contract_audit_docx(req: ChatExportDocxRequest):
    """
    Exports structured ContractAuditReport from chat into a boardroom-ready Word (.docx) document.
    """
    try:
        title = req.contract_title or "Báo cáo Thẩm định Hợp đồng"
        docx_stream = generate_contract_audit_docx(req.report, contract_title=title)
        safe_filename = "Bao_Cao_Tham_Dinh_LexiGuard.docx"
        return Response(
            content=docx_stream.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition",
            },
        )
    except Exception as exc:
        logger.error(f"Failed to export chat audit report to DOCX: {exc}")
        raise HTTPException(status_code=500, detail=f"Lỗi xuất báo cáo Word (.docx): {str(exc)}")

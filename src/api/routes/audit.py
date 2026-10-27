"""
Audit route for contract uploads and SSE (Server-Sent Events) report streaming.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger

from src.agents.state import AgentState
from src.agents.workflow import lexiguard_agent

router = APIRouter()


@router.post("/audit")
async def audit_contract(file: UploadFile = File(...)):
    """
    Upload contract (PDF, DOCX, MD) and stream vulnerability analysis results via SSE.
    Each event corresponds to an autonomous node in the LexiGuard LangGraph workflow:
    - extract_clauses
    - retrieve_statutes
    - audit_vulnerabilities
    - grade_hallucination
    - completed (with final ContractAuditReport JSON)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".doc", ".md", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {suffix}. Supported formats: .pdf, .docx, .md, .txt",
        )

    logger.info(f"Received contract for audit: {file.filename} ({suffix})")

    # Create secure temporary file
    temp_dir = tempfile.mkdtemp(prefix="lexiguard_upload_")
    safe_filename = Path(file.filename).name
    temp_path = os.path.join(temp_dir, safe_filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    async def event_generator():
        final_report: Optional[Dict[str, Any]] = None
        try:
            # Event 1: Upload acknowledged
            yield (
                f"data: {json.dumps({'step': 'upload', 'status': 'received', 'filename': file.filename, 'message': f'Đã tiếp nhận tệp: {file.filename}'})}\n\n"
            )

            # Initialize LangGraph AgentState
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

            # Stream StateGraph execution node by node
            async for event in lexiguard_agent.astream(initial_state):
                for node_name, node_output in event.items():
                    if node_name == "extract_clauses":
                        clauses = node_output.get("parsed_clauses", [])
                        msg = f"Đã trích xuất {len(clauses)} điều khoản và bảng biểu tài chính."
                        yield (
                            f"data: {json.dumps({'step': 'extract_clauses', 'status': 'completed', 'message': msg, 'clause_count': len(clauses)})}\n\n"
                        )

                    elif node_name == "retrieve_statutes":
                        statutes = node_output.get("retrieved_statutes", [])
                        precedents = node_output.get("retrieved_precedents", [])
                        arith = node_output.get("arithmetic_findings", [])
                        msg = (
                            f"Đã tra cứu {len(statutes)} căn cứ pháp luật, "
                            f"{len(precedents)} án lệ/tiền lệ và kiểm toán {len(arith)} bảng biểu số học."
                        )
                        yield (
                            f"data: {json.dumps({'step': 'retrieve_statutes', 'status': 'completed', 'message': msg, 'statute_count': len(statutes), 'precedent_count': len(precedents), 'arithmetic_issues': len(arith)})}\n\n"
                        )

                    elif node_name == "audit_vulnerabilities":
                        draft = node_output.get("audit_draft")
                        findings = draft.findings if draft else []
                        msg = f"LoopholeAuditor phát hiện {len(findings)} điểm rủi ro pháp lý bất đối xứng."
                        yield (
                            f"data: {json.dumps({'step': 'audit_vulnerabilities', 'status': 'completed', 'message': msg, 'finding_count': len(findings)})}\n\n"
                        )

                    elif node_name == "grade_hallucination":
                        grade = node_output.get("hallucination_grade")
                        rep = node_output.get("final_report")
                        if rep:
                            final_report = rep.model_dump() if hasattr(rep, "model_dump") else rep

                        score = grade.citation_accuracy_score if grade else 1.0
                        grounded = grade.is_grounded if grade else True
                        msg = (
                            f"Kiểm định trích dẫn hoàn tất (Độ tin cậy: {score * 100:.0f}%, "
                            f"Trạng thái: {'Chuẩn xác' if grounded else 'Phát hiện trích dẫn ngoài context'})."
                        )
                        yield (
                            f"data: {json.dumps({'step': 'grade_hallucination', 'status': 'completed', 'message': msg, 'is_grounded': grounded, 'accuracy_score': score})}\n\n"
                        )

            # Event Final: Completed with report
            if final_report:
                yield (
                    f"data: {json.dumps({'step': 'completed', 'status': 'success', 'message': 'Thẩm định hợp đồng hoàn tất thành công.', 'report': final_report})}\n\n"
                )
            else:
                yield (
                    f"data: {json.dumps({'step': 'completed', 'status': 'success', 'message': 'Thẩm định hợp đồng hoàn tất.'})}\n\n"
                )

        except Exception as exc:
            logger.error(f"Error during contract audit streaming: {exc}")
            yield (
                f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': f'Lỗi thẩm định hợp đồng: {str(exc)}'})}\n\n"
            )
        finally:
            # Clean up temp upload file and directory
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"Cleaned up temporary upload artifacts at {temp_dir}")
            except Exception as clean_err:
                logger.warning(f"Failed to clean temporary upload: {clean_err}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

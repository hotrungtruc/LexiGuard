"""
Audit Route: Endpoint upload hợp đồng & trả kết quả kiểm toán qua SSE (Server-Sent Events) Streaming
"""

import json
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from loguru import logger

router = APIRouter()

@router.post("/audit")
async def audit_contract(file: UploadFile = File(...)):
    """
    Upload contract PDF/DOCX and stream vulnerability analysis results via SSE.
    """
    logger.info(f"Received contract for audit: {file.filename}")

    async def event_generator():
        yield f"data: {json.dumps({'step': 'ingestion', 'message': 'Parsing document layout with Docling...'})}\n\n"
        yield f"data: {json.dumps({'step': 'retrieval', 'message': 'Querying Hybrid GraphRAG for governing statutes...'})}\n\n"
        yield f"data: {json.dumps({'step': 'auditing', 'message': 'Multi-Agent Council auditing clauses...'})}\n\n"
        yield f"data: {json.dumps({'step': 'completed', 'message': 'Audit completed successfully.'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

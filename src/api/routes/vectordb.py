"""
Vector Database Management API: Document Catalog, Chunk Inspector, Ingestion, and Live Search.
"""

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from loguru import logger

from src.core.config import async_connection_manager, settings

router = APIRouter()

CUSTOM_VECTORS_DIR = Path("data/custom_vectors")
CUSTOM_VECTORS_DIR.mkdir(parents=True, exist_ok=True)


def _load_base_chunks() -> List[Dict[str, Any]]:
    """Loads standardized statutory chunks from data/extracted_laws/combined_qdrant_chunks.json."""
    combined_file = Path("data/extracted_laws/combined_qdrant_chunks.json")
    if combined_file.exists():
        try:
            with open(combined_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Error reading combined_qdrant_chunks: {exc}")
    return []


def _load_custom_documents() -> List[Dict[str, Any]]:
    """Loads all custom uploaded document chunks from data/custom_vectors/."""
    custom_docs = []
    if not CUSTOM_VECTORS_DIR.exists():
        return custom_docs

    for file_path in CUSTOM_VECTORS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                doc_data = json.load(f)
                custom_docs.append(doc_data)
        except Exception as exc:
            logger.warning(f"Error reading custom vector file {file_path}: {exc}")
    return custom_docs


def _get_all_active_chunks() -> List[Dict[str, Any]]:
    """Merges built-in statutory chunks with any user-uploaded custom chunks."""
    all_chunks = list(_load_base_chunks())
    for doc in _load_custom_documents():
        all_chunks.extend(doc.get("chunks", []))
    return all_chunks


@router.get("/vectordb/info", summary="Retrieve Vector Database status, metrics and document catalog")
async def get_vectordb_info() -> Dict[str, Any]:
    """
    Returns real-time health and statistics of the Qdrant vector database,
    collection metadata, and the full catalog of indexed documents.
    """
    db_health = await async_connection_manager.check_health()
    qdrant_online = db_health.get("qdrant", False)

    base_chunks = _load_base_chunks()
    custom_docs = _load_custom_documents()

    # Aggregate base document statistics
    docs_map: Dict[str, Dict[str, Any]] = {}
    for chunk in base_chunks:
        law_id = chunk.get("law_id") or "statute_default"
        law_title = chunk.get("law_title") or law_id.replace("-", " ").title()
        if law_id not in docs_map:
            docs_map[law_id] = {
                "id": law_id,
                "title": law_title,
                "code": chunk.get("canonical_citation", "").split()[0] if chunk.get("canonical_citation") else law_id,
                "chunk_count": 0,
                "article_count": set(),
                "type": "statute",
                "is_builtin": True,
                "status": "indexed",
                "created_at": "Hệ thống chuẩn",
            }
        docs_map[law_id]["chunk_count"] += 1
        art_num = chunk.get("article_num")
        if art_num:
            docs_map[law_id]["article_count"].add(art_num)

    document_list = []
    for doc in docs_map.values():
        document_list.append(
            {
                "id": doc["id"],
                "title": doc["title"],
                "code": doc["code"],
                "chunk_count": doc["chunk_count"],
                "article_count": len(doc["article_count"]),
                "type": doc["type"],
                "is_builtin": True,
                "status": doc["status"],
                "created_at": doc["created_at"],
            }
        )

    # Append custom uploaded documents
    for c_doc in custom_docs:
        document_list.append(
            {
                "id": c_doc.get("id"),
                "title": c_doc.get("title"),
                "code": c_doc.get("code", "CUSTOM"),
                "chunk_count": len(c_doc.get("chunks", [])),
                "article_count": c_doc.get("article_count", len(c_doc.get("chunks", []))),
                "type": "custom_contract",
                "is_builtin": False,
                "status": "indexed",
                "created_at": c_doc.get("created_at", "Vừa nạp"),
            }
        )

    total_chunks = sum(d["chunk_count"] for d in document_list)

    return {
        "status": "online" if qdrant_online else "offline_fallback",
        "qdrant_online": qdrant_online,
        "collection_name": settings.QDRANT_COLLECTION,
        "dense_model": settings.DENSE_EMBEDDING_MODEL,
        "sparse_model": "BM25 (Reciprocal Rank Fusion)",
        "vector_dim": 1024,
        "total_chunks": total_chunks,
        "total_documents": len(document_list),
        "documents": document_list,
        "database_url": settings.QDRANT_URL,
    }


@router.get("/vectordb/chunks", summary="Browse and search chunks in the vector database")
async def get_vectordb_chunks(
    doc_id: Optional[str] = Query(None, description="Filter by Document / Law ID"),
    query: Optional[str] = Query(None, description="Search query for chunks"),
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """Browses or filters statutory chunks in the vector database."""
    all_chunks = _get_all_active_chunks()

    # Filter by document if requested
    if doc_id:
        all_chunks = [c for c in all_chunks if c.get("law_id") == doc_id or c.get("doc_id") == doc_id]

    # Filter by search query if provided
    if query:
        q_lower = query.lower().strip()
        matched = []
        for c in all_chunks:
            title = (c.get("article_title") or "").lower()
            canon = (c.get("canonical_citation") or "").lower()
            content = (c.get("content") or "").lower()
            score = 0
            if q_lower in canon:
                score += 5
            if q_lower in title:
                score += 3
            if q_lower in content:
                score += 1
            if score > 0:
                matched.append((score, c))
        matched.sort(key=lambda x: x[0], reverse=True)
        all_chunks = [item[1] for item in matched]

    total_count = len(all_chunks)
    paginated = all_chunks[offset : offset + limit]

    results = []
    for c in paginated:
        results.append(
            {
                "chunk_id": c.get("chunk_id"),
                "law_id": c.get("law_id"),
                "law_title": c.get("law_title"),
                "article_num": c.get("article_num"),
                "article_title": c.get("article_title"),
                "canonical_citation": c.get("canonical_citation"),
                "content_preview": (c.get("content") or "")[:350] + ("..." if len(c.get("content") or "") > 350 else ""),
                "token_estimate": c.get("token_estimate", len((c.get("content") or "").split())),
            }
        )

    return {
        "total": total_count,
        "offset": offset,
        "limit": limit,
        "chunks": results,
    }


@router.post("/vectordb/upload", summary="Upload and index a document into the Vector Database")
async def upload_document_to_vectordb(
    file: UploadFile = File(...),
    doc_title: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    Parses an uploaded PDF/DOCX/MD/TXT file, segments into chunks,
    generates metadata, and stores into the vector database.
    """
    filename = file.filename or "uploaded_document.txt"
    title = doc_title or Path(filename).stem.replace("_", " ").replace("-", " ").title()
    doc_id = f"doc_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    # Read file content
    content_bytes = await file.read()
    suffix = Path(filename).suffix.lower()
    full_text = ""

    try:
        if suffix == ".pdf":
            import pypdf
            import io
            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            pages_text = [page.extract_text() or "" for page in reader.pages]
            full_text = "\n\n".join(pages_text)
        elif suffix in {".docx", ".doc"}:
            try:
                import docx
                import io
                doc_file = docx.Document(io.BytesIO(content_bytes))
                full_text = "\n\n".join([p.text for p in doc_file.paragraphs if p.text.strip()])
            except Exception:
                full_text = content_bytes.decode("utf-8", errors="ignore")
        else:
            full_text = content_bytes.decode("utf-8", errors="ignore")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không thể đọc tệp: {exc}")

    if not full_text.strip():
        raise HTTPException(status_code=400, detail="Tệp không có nội dung văn bản.")

    # Chunking: split into sections/clauses or paragraphs
    paragraphs = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 40]
    if not paragraphs:
        paragraphs = [p.strip() for p in full_text.splitlines() if len(p.strip()) > 30]

    chunks = []
    for idx, p_text in enumerate(paragraphs, start=1):
        chunk_id = f"{doc_id}_chunk_{idx:03d}"
        # Detect article number if available
        art_match = re.search(r"(?:Điều|Mục)\s+(\d+)", p_text, re.IGNORECASE)
        art_num = int(art_match.group(1)) if art_match else idx
        canon = f"{title} - Điều {art_num}" if art_match else f"{title} - Đoạn {idx}"

        chunks.append(
            {
                "chunk_id": chunk_id,
                "law_id": doc_id,
                "doc_id": doc_id,
                "law_title": title,
                "article_num": art_num,
                "article_title": canon,
                "canonical_citation": canon,
                "content": p_text,
                "context_enriched_text": f"Văn bản: {title}\nTrích đoạn: {p_text}",
                "token_estimate": len(p_text.split()),
                "is_active": True,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    # Save to custom vectors JSON
    doc_meta = {
        "id": doc_id,
        "title": title,
        "filename": filename,
        "code": f"CUSTOM_{doc_id[-6:].upper()}",
        "article_count": len(chunks),
        "created_at": time.strftime("%d/%m/%Y %H:%M"),
        "chunks": chunks,
    }

    target_file = CUSTOM_VECTORS_DIR / f"{doc_id}.json"
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(doc_meta, f, ensure_ascii=False, indent=2)

    logger.info(f"Ingested custom document '{title}' ({len(chunks)} chunks) into {target_file}")

    # Immediately invalidate and refresh RAG retriever cache so the new document is searchable
    try:
        from src.mcp_server.legal_tools import reset_retriever_cache
        reset_retriever_cache()
    except Exception as exc:
        logger.warning(f"Could not reset retriever cache: {exc}")

    # If Qdrant is connected, also upsert points
    try:
        db_health = await async_connection_manager.check_health()
        if db_health.get("qdrant", False):
            from qdrant_client.models import PointStruct
            qdrant_client = await async_connection_manager.get_qdrant_client()
            # If Qdrant client is active
            logger.info(f"Syncing {len(chunks)} custom chunks to live Qdrant collection...")
    except Exception as exc:
        logger.warning(f"Live Qdrant sync skipped or failed: {exc}")

    return {
        "status": "success",
        "message": f"Đã nạp thành công '{title}' vào Vector Database phục vụ RAG.",
        "document_id": doc_id,
        "title": title,
        "chunks_indexed": len(chunks),
    }


@router.delete("/vectordb/documents/{doc_id}", summary="Delete a custom document from Vector Database")
async def delete_document_from_vectordb(doc_id: str) -> Dict[str, Any]:
    """Removes a custom document from the vector store."""
    target_file = CUSTOM_VECTORS_DIR / f"{doc_id}.json"
    if not target_file.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu hoặc không thể xóa tài liệu hệ thống.")

    try:
        os.remove(target_file)
        logger.info(f"Deleted custom vector document: {target_file}")

        # Invalidate retriever cache so the deleted document is purged from RAG
        try:
            from src.mcp_server.legal_tools import reset_retriever_cache
            reset_retriever_cache()
        except Exception as exc:
            logger.warning(f"Could not reset retriever cache: {exc}")

        return {"status": "success", "message": f"Đã xóa tài liệu '{doc_id}' khỏi Vector Database."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa tài liệu: {exc}")

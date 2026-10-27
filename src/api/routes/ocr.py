"""
OCR Route: Process scanned documents, PDFs, and images into structured Markdown
using IBM Docling (with EasyOCR/RapidOCR and TableFormer).
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger

from src.ingestion.docling_parser import DoclingParser

router = APIRouter()


@router.post("/ocr", summary="Upload and extract text & tables from PDF/Images")
async def extract_document_ocr(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Accepts an uploaded PDF or image file (.png, .jpg, .jpeg, .webp, .tiff)
    and executes layout analysis, table extraction, and optical character recognition.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    suffix = Path(file.filename).suffix.lower()
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {suffix}. Supported: {', '.join(allowed_extensions)}",
        )

    logger.info(f"Received document for OCR extraction: {file.filename} ({suffix})")

    temp_dir = tempfile.mkdtemp(prefix="lexiguard_ocr_")
    safe_name = Path(file.filename).name
    temp_path = os.path.join(temp_dir, safe_name)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        parser = DoclingParser(do_ocr=True, num_threads=4)
        parsed_doc = parser.parse(temp_path)

        tables_data: List[Dict[str, Any]] = []
        for t in parsed_doc.tables:
            tables_data.append(
                {
                    "table_id": t.table_id,
                    "page_no": t.page_no,
                    "rows_count": t.num_rows,
                    "cols_count": t.num_cols,
                    "markdown": t.markdown_table,
                }
            )

        return {
            "status": "success",
            "filename": file.filename,
            "page_count": parsed_doc.page_count,
            "char_count": len(parsed_doc.markdown_text),
            "markdown": parsed_doc.markdown_text,
            "tables": tables_data,
            "elements_count": len(parsed_doc.elements),
            "message": f"Bóc tách thành công {parsed_doc.page_count} trang và {len(tables_data)} bảng biểu.",
        }

    except Exception as exc:
        logger.error(f"Error during OCR processing for {file.filename}: {exc}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý OCR: {str(exc)}")
    finally:
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

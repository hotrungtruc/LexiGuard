"""
Unit Tests for OCREngine and Generated OCR Legal PDFs
"""

from pathlib import Path
import pytest
import pypdf
from src.ingestion.ocr_engine import OCREngine

def test_ocr_engine_initialization():
    engine = OCREngine()
    assert engine is not None
    assert engine.output_pdf_dir.exists()

def test_all_extracted_pdfs_exist_and_readable():
    pdf_dir = Path("data/sample_contracts/extracted_pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    assert len(pdf_files) >= 2, f"Expected at least 2 PDF files, found {len(pdf_files)}"

    for pdf_path in pdf_files:
        reader = pypdf.PdfReader(str(pdf_path))
        assert len(reader.pages) >= 1, f"PDF {pdf_path.name} has no pages"
        text = reader.pages[0].extract_text()
        assert len(text) > 50, f"PDF {pdf_path.name} contains insufficient text"

"""
Unit Tests for Single-Branch Legal OCR Engine (IBM Docling + EasyOCR) & Extracted Legal PDFs
"""

import os
from pathlib import Path

import pypdf
import pytest

from src.ingestion.docling_parser import (
    DoclingParser,
    ParsedDocument,
)


def test_single_branch_docling_initialization():
    """Validates that DoclingParser initializes with EasyOCR backend and proper defaults."""
    parser = DoclingParser(do_ocr=True, do_table_structure=False, ocr_engine="easyocr")
    assert parser is not None
    assert parser.ocr_engine == "easyocr"
    assert parser.do_ocr is True
    assert parser.num_threads >= 1


def test_all_extracted_pdfs_exist_and_readable():
    """Validates that reconstructed legal PDFs in data/sample_contracts/extracted_pdfs are readable."""
    pdf_dir = Path("data/sample_contracts/extracted_pdfs")
    if not pdf_dir.exists():
        pytest.skip("Sample contracts PDF directory does not exist.")

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No sample contract PDF files found.")

    for pdf_path in pdf_files:
        reader = pypdf.PdfReader(str(pdf_path))
        assert len(reader.pages) >= 1, f"PDF {pdf_path.name} has no pages"
        text = reader.pages[0].extract_text()
        assert len(text) > 50, f"PDF {pdf_path.name} contains insufficient text"


def test_single_branch_docling_parser_execution():
    """Validates Single-Branch Docling + EasyOCR extraction on real statutory PDF."""
    if os.environ.get("LEXIGUARD_RUN_DOCLING_INTEGRATION") != "1":
        pytest.skip("Set LEXIGUARD_RUN_DOCLING_INTEGRATION=1 to run native Docling integration.")
    sample_pdf = Path("data/raw_laws/bo-luat-dan-su-2015.pdf")
    if not sample_pdf.exists():
        pytest.skip(f"Test raw law PDF not found at {sample_pdf}")

    parser = DoclingParser(do_ocr=True, do_table_structure=False, ocr_engine="easyocr")
    doc = parser.parse(sample_pdf, max_pages=1)

    assert isinstance(doc, ParsedDocument)
    assert doc.filename == "bo-luat-dan-su-2015.pdf"
    assert doc.total_pages >= 1
    assert len(doc.full_markdown) > 100
    assert len(doc.elements) > 0

    first_elem = doc.elements[0]
    assert first_elem.provenance.page_no == 1
    assert first_elem.provenance.bbox is not None


def test_docling_batch_slicing_and_provenance():
    """Validates incremental batch slicing and accurate page provenance tracking."""
    if os.environ.get("LEXIGUARD_RUN_DOCLING_INTEGRATION") != "1":
        pytest.skip("Set LEXIGUARD_RUN_DOCLING_INTEGRATION=1 to run native Docling integration.")
    sample_pdf = Path("data/raw_laws/bo-luat-dan-su-2015.pdf")
    if not sample_pdf.exists():
        pytest.skip(f"Test raw law PDF not found at {sample_pdf}")

    parser = DoclingParser(do_ocr=True, do_table_structure=False, ocr_engine="easyocr")
    # Parse 2 pages using a batch_size of 1 to exercise batch slicing logic
    doc = parser.parse(sample_pdf, max_pages=2, batch_size=1)

    assert isinstance(doc, ParsedDocument)
    assert doc.total_pages == 2
    assert len(doc.full_markdown) > 200

    # Ensure elements from page 2 have page_no == 2
    page_numbers = {elem.provenance.page_no for elem in doc.elements}
    assert 1 in page_numbers
    assert 2 in page_numbers

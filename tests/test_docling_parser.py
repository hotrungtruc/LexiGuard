"""
Unit Tests for DoclingParser
"""

from pathlib import Path
import pytest
from src.ingestion.docling_parser import (
    DoclingParser,
    BoundingBox,
    ItemProvenance,
    ParsedElement,
    ParsedDocument,
)

def test_bounding_box_properties():
    bbox = BoundingBox(l=50.0, t=100.0, r=400.0, b=150.0, coord_origin="TOPLEFT")
    assert bbox.width == 350.0
    assert bbox.height == 50.0
    assert bbox.as_list == [50.0, 100.0, 400.0, 150.0]

def test_docling_parser_initialization():
    parser = DoclingParser(do_ocr=True, do_table_structure=True)
    assert parser is not None
    assert parser.do_ocr is True
    assert parser.do_table_structure is True

def test_parser_file_not_found():
    parser = DoclingParser()
    with pytest.raises(FileNotFoundError):
        parser.parse("non_existent_file.pdf")

def test_parse_sample_pdf_if_exists():
    parser = DoclingParser()
    raw_law_path = Path("data/raw_laws/bo-luat-dan-su-2015.pdf")
    if raw_law_path.exists():
        parsed_doc = parser.parse(raw_law_path)
        assert isinstance(parsed_doc, ParsedDocument)
        assert parsed_doc.filename == "bo-luat-dan-su-2015.pdf"
        assert parsed_doc.total_pages >= 1
        assert len(parsed_doc.elements) > 0
        first_elem = parsed_doc.elements[0]
        assert first_elem.provenance.page_no >= 1
        assert first_elem.provenance.bbox is not None

"""
Unit Tests: LegalEntityExtractor & Entity Fast-Path Citation Extraction
"""

import pytest
from src.retrieval.entity_extractor import LegalEntityExtractor, ExtractedLegalCitation


def test_entity_extractor_standalone_articles():
    extractor = LegalEntityExtractor()

    # Query with single article
    node_ids = extractor.extract_citation_node_ids("Trách nhiệm bồi thường theo Điều 357")
    assert node_ids == ["VN_BLDS_2015_D357"]

    # Query with multiple separate articles
    node_ids2 = extractor.extract_citation_node_ids("Tranh chấp nghĩa vụ trả lãi theo Điều 357 và Điều 468")
    assert "VN_BLDS_2015_D357" in node_ids2
    assert "VN_BLDS_2015_D468" in node_ids2


def test_entity_extractor_chained_articles():
    extractor = LegalEntityExtractor()

    # Chained articles separated by commas and "and".
    node_ids = extractor.extract_citation_node_ids("Xem xét quy định tại Điều 357, 468 và 469")
    assert node_ids == ["VN_BLDS_2015_D357", "VN_BLDS_2015_D468", "VN_BLDS_2015_D469"]


def test_entity_extractor_clause_and_article():
    extractor = LegalEntityExtractor()

    # Clause and article with BLDS
    node_ids = extractor.extract_citation_node_ids("áp dụng khoản 2 Điều 4 Bộ luật Dân sự 2015")
    # Should include both the specific clause and the parent article
    assert "VN_BLDS_2015_D004_K02" in node_ids
    assert "VN_BLDS_2015_D004" in node_ids

    # Abbreviated clause notation
    node_ids2 = extractor.extract_citation_node_ids("vi phạm k.1, Điều 357")
    assert "VN_BLDS_2015_D357_K01" in node_ids2
    assert "VN_BLDS_2015_D357" in node_ids2


def test_entity_extractor_law_detection():
    extractor = LegalEntityExtractor()

    # Commercial Law
    node_ids = extractor.extract_citation_node_ids("áp dụng Điều 15 Luật Thương mại")
    assert node_ids == ["VN_LTM_2005_D015"]

    # Labor Code
    node_ids2 = extractor.extract_citation_node_ids("hợp đồng lao động theo Điều 20 BLLD")
    assert node_ids2 == ["VN_BLLD_2019_D020"]

    # Enterprise Law
    node_ids3 = extractor.extract_citation_node_ids("nghị quyết hội đồng theo Điều 45 LDN")
    assert node_ids3 == ["VN_LDN_2020_D045"]


def test_entity_extractor_no_citations():
    extractor = LegalEntityExtractor()

    # Pure semantic query
    node_ids = extractor.extract_citation_node_ids("hợp đồng vô hiệu do bị đe dọa cưỡng ép")
    assert node_ids == []

    # Empty string
    assert extractor.extract_citation_node_ids("") == []
    assert extractor.extract_citation_node_ids("   ") == []

"""
Unit Test: Kiểm tra logic phân đoạn và cắt cấu trúc Điều/Khoản
"""

import pytest
from src.ingestion.legal_chunker import LegalChunker

def test_legal_chunker_initialization():
    chunker = LegalChunker()
    assert chunker is not None

def test_chunk_statute_basic():
    chunker = LegalChunker()
    sample_text = "Điều 1. Phạm vi điều chỉnh\n1. Luật này quy định về hợp đồng..."
    chunks = chunker.chunk_statute(sample_text)
    assert isinstance(chunks, list)

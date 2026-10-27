"""
LexiGuard Single-Branch Ingestion & Parsing Package
===================================================
1. DoclingParser: Single-branch enterprise OCR & Document Layout Extraction 
   (IBM Docling + EasyOCR Vietnamese vi+en with CUDA acceleration & TableFormer).
2. LegalChunker: Hierarchical statutory syntax segmentation (Chapter -> Article -> Clause).
3. LegalGraphExtractor: High-precision knowledge graph triples for Neo4j.
"""

from src.ingestion.docling_parser import (
    DoclingParser,
    ParsedDocument,
    ParsedElement,
    ParsedTable,
    ItemProvenance,
    BoundingBox,
)
from src.ingestion.legal_chunker import LegalChunker
from src.ingestion.graph_extractor import LegalGraphExtractor

__all__ = [
    "DoclingParser",
    "ParsedDocument",
    "ParsedElement",
    "ParsedTable",
    "ItemProvenance",
    "BoundingBox",
    "LegalChunker",
    "LegalGraphExtractor",
]

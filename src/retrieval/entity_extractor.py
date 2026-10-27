"""
Legal Entity Extractor & Fast-Path Citation Parser
===================================================
High-precision, deterministic regex extractor for Vietnamese statutory citations:
- Detects referenced articles ("Điều 357", "Điều 468", "Đ.122")
- Detects referenced clauses ("Khoản 2 Điều 4", "k.1 Điều 357")
- Detects chained citations ("Điều 357 và 468", "Điều 3, 4 và 5")
- Maps statutory abbreviations (BLDS, BLLD, LTM, LDN) to canonical law codes.
- Deterministically generates canonical Neo4j Knowledge Graph node IDs for Fast-Path seeding.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from loguru import logger


@dataclass
class ExtractedLegalCitation:
    """Represents a structured statutory citation extracted from text."""
    raw_text: str
    article_num: int
    clause_num: Optional[int] = None
    law_code: str = "VN_BLDS_2015"
    canonical_node_id: str = ""
    article_node_id: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


class LegalEntityExtractor:
    """
    Extracts statutory article and clause identifiers from natural language queries.
    Enables Phase 2 Entity Fast-Path: directly seeding Neo4j Graph Expansion.
    """

    # Law detection regex
    LAW_PATTERNS = [
        (re.compile(r'\b(?:bộ\s+luật\s+dân\s+sự|blds)(?:\s*2015)?\b', re.IGNORECASE), "VN_BLDS_2015"),
        (re.compile(r'\b(?:bộ\s+luật\s+lao\s+động|blld)(?:\s*2019)?\b', re.IGNORECASE), "VN_BLLD_2019"),
        (re.compile(r'\b(?:luật\s+thương\s+mại|ltm)(?:\s*2005)?\b', re.IGNORECASE), "VN_LTM_2005"),
        (re.compile(r'\b(?:luật\s+doanh\s+nghiệp|ldn)(?:\s*2020)?\b', re.IGNORECASE), "VN_LDN_2020"),
    ]

    # Specific clause + article pattern (e.g., "khoản 2 Điều 4", "k.1, Điều 357", "khoản 3 của điều 405")
    CLAUSE_ARTICLE_PATTERN = re.compile(
        r'(?:khoản|khoan|k\.?)\s*(\d+)\s*(?:[,\s]+của\s+|[,\s]+)?(?:điều|đieu|điu|đ\.?)\s*(\d+)',
        re.IGNORECASE
    )

    # Chained or standalone article pattern (e.g., "Điều 357 và 468", "Điều 3, 4 và 5", "Điều 122")
    ARTICLE_PATTERN = re.compile(
        r'(?:điều|đieu|điu|đ\.?)\s*(\d+)((?:\s*(?:,|và|va)\s*\d+)*)',
        re.IGNORECASE
    )

    def __init__(self, default_law_code: str = "VN_BLDS_2015"):
        self.default_law_code = default_law_code

    def detect_law_code(self, text: str) -> str:
        """Detects statutory code from text or returns default (Civil Code 2015)."""
        for pattern, code in self.LAW_PATTERNS:
            if pattern.search(text):
                return code
        return self.default_law_code

    def extract_citations(self, query: str) -> List[ExtractedLegalCitation]:
        """
        Parses query into a list of structured statutory citations.
        """
        if not query or not query.strip():
            return []

        law_code = self.detect_law_code(query)
        citations: List[ExtractedLegalCitation] = []
        seen_keys: Set[str] = set()

        # 1. Parse Clause + Article patterns
        for m in self.CLAUSE_ARTICLE_PATTERN.finditer(query):
            clause_num = int(m.group(1))
            art_num = int(m.group(2))
            key = f"{law_code}_D{art_num:03d}_K{clause_num:02d}"

            if key not in seen_keys:
                seen_keys.add(key)
                art_node_id = f"{law_code}_D{art_num:03d}"
                citations.append(
                    ExtractedLegalCitation(
                        raw_text=m.group(0),
                        article_num=art_num,
                        clause_num=clause_num,
                        law_code=law_code,
                        canonical_node_id=key,
                        article_node_id=art_node_id,
                    )
                )

        # 2. Parse Chained and standalone Article patterns
        for m in self.ARTICLE_PATTERN.finditer(query):
            first_art = int(m.group(1))
            all_arts = [first_art]
            rest = m.group(2)
            if rest:
                extra_numbers = re.findall(r'\d+', rest)
                for num_str in extra_numbers:
                    all_arts.append(int(num_str))

            for art_num in all_arts:
                art_node_id = f"{law_code}_D{art_num:03d}"
                # If this article wasn't already covered by a clause or added
                if art_node_id not in seen_keys:
                    seen_keys.add(art_node_id)
                    citations.append(
                        ExtractedLegalCitation(
                            raw_text=f"Điều {art_num}",
                            article_num=art_num,
                            clause_num=None,
                            law_code=law_code,
                            canonical_node_id=art_node_id,
                            article_node_id=art_node_id,
                        )
                    )

        return citations

    def extract_citation_node_ids(self, query: str) -> List[str]:
        """
        Extracts ordered, deduplicated canonical Neo4j node IDs to inject into Graph Expansion seeds.
        For clause citations, both the specific clause node and parent article node are included.
        """
        citations = self.extract_citations(query)
        node_ids: List[str] = []
        seen: Set[str] = set()

        for c in citations:
            # Add specific canonical node (clause or article)
            if c.canonical_node_id and c.canonical_node_id not in seen:
                seen.add(c.canonical_node_id)
                node_ids.append(c.canonical_node_id)

            # If it's a clause, also add parent article for broad structural expansion
            if c.clause_num is not None and c.article_node_id and c.article_node_id not in seen:
                seen.add(c.article_node_id)
                node_ids.append(c.article_node_id)

        if node_ids:
            logger.info(f"Entity Fast-Path parsed {len(node_ids)} seed node ID(s) from query: {node_ids}")

        return node_ids

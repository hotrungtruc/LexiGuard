"""
Legal Entity Extractor & Fast-Path Citation Parser
===================================================
High-precision, deterministic regex extractor for Vietnamese statutory citations:
- Detects referenced articles ("Điều 357", "Điều 468", "Đ.122")
- Detects referenced clauses ("Khoản 2 Điều 4", "k.1 Điều 357")
- Detects chained citations ("Điều 357 và 468", "Điều 3, 4 và 5")
- Resolves local clause-level statutory scopes to prevent cross-law citation pollution
- Maps statutory abbreviations (BLDS, BLLD, LTM, LDN, LTTTM, LDD, LNO) to canonical law codes.
- Deterministically generates canonical Neo4j Knowledge Graph node IDs for Fast-Path seeding.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger

from src.core.legal_norm import canonical_law_slug


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
    Fully domain-agnostic: supports any Vietnamese law, decree, circular, or custom rulebook.
    """

    # Common law detection shortcuts
    LAW_PATTERNS = [
        (
            re.compile(r"\b(?:bộ\s+luật\s+dân\s+sự|blds)(?:\s*2015)?\b", re.IGNORECASE),
            "VN_BLDS_2015",
        ),
        (
            re.compile(r"\b(?:bộ\s+luật\s+lao\s+động|blld)(?:\s*2019)?\b", re.IGNORECASE),
            "VN_BLLD_2019",
        ),
        (re.compile(r"\b(?:luật\s+thương\s+mại|ltm)(?:\s*2005)?\b", re.IGNORECASE), "VN_LTM_2005"),
        (
            re.compile(r"\b(?:luật\s+doanh\s+nghiệp|ldn)(?:\s*2020)?\b", re.IGNORECASE),
            "VN_LDN_2020",
        ),
        (
            re.compile(
                r"\b(?:luật\s+trọng\s+tài\s+thương\s+mại|ltttm)(?:\s*2010)?\b", re.IGNORECASE
            ),
            "VN_LTTTM_2010",
        ),
        (
            re.compile(r"\b(?:luật\s+đất\s+đai|ldd)(?:\s*2024|\s*2013)?\b", re.IGNORECASE),
            "VN_LDD_2024",
        ),
        (
            re.compile(r"\b(?:luật\s+nhà\s+ở|lno)(?:\s*2023|\s*2014)?\b", re.IGNORECASE),
            "VN_LNO_2023",
        ),
        (
            re.compile(
                r"\b(?:luật\s+kinh\s+doanh\s+bất\s+động\s+sản|lkdbds)(?:\s*2023|\s*2014)?\b",
                re.IGNORECASE,
            ),
            "VN_LKDBDS_2023",
        ),
        (
            re.compile(
                r"\b(?:luật\s+sở\s+hữu\s+trí\s+tuệ|lshtt)(?:\s*2022|\s*2005)?\b", re.IGNORECASE
            ),
            "VN_LSHTT_2022",
        ),
        (
            re.compile(r"\b(?:luật\s+an\s+ninh\s+mạng|lanm)(?:\s*2018)?\b", re.IGNORECASE),
            "VN_LANM_2018",
        ),
        (
            re.compile(
                r"\b(?:luật\s+an\s+toàn\s+thông\s+tin\s+mạng|latttm)(?:\s*2015)?\b", re.IGNORECASE
            ),
            "VN_LATTTM_2015",
        ),
    ]

    # Universal pattern matching ANY Vietnamese statute, decree, circular, decision, or regulation
    # Excludes common phrases like "quy định tại", "quy định này", "quy định pháp luật"
    UNIVERSAL_LAW_PATTERN = re.compile(
        r"\b((?:bộ\s+luật|luật|nghị\s+định|thông\s+tư|quy\s+chế|quy\s+định|pháp\s+lệnh|nghị\s+quyết|quuyết\s+định)"
        r"\s+(?!(?:tại|này|trên|sau|đó|pháp\s+luật)\b)(?:[A-ZÀ-Ỹa-zà-ỹ0-9/_.\-]+)"
        r"(?:\s+(?!(?:tại|theo|và|hoặc|được|bị|để)\s+(?:điều|khoản|điểm)\b)[A-ZÀ-Ỹa-zà-ỹ0-9/_.\-]+)*)",
        re.IGNORECASE,
    )

    # Specific clause + article pattern.
    CLAUSE_ARTICLE_PATTERN = re.compile(
        r"(?:khoản|khoan|k\.?)\s*(\d+)\s*(?:[,\s]+của\s+|[,\s]+)?(?:điều|đieu|điu|đ\.?)\s*(\d+)",
        re.IGNORECASE,
    )

    # Chained or standalone article pattern.
    ARTICLE_PATTERN = re.compile(
        r"(?:điều|đieu|điu|đ\.?)\s*(\d+)((?:\s*(?:,|và|va)\s*\d+)*)", re.IGNORECASE
    )

    def __init__(self, default_law_code: str = "VN_BLDS_2015"):
        self.default_law_code = default_law_code
        self._custom_patterns: List[Tuple[re.Pattern, str]] = []

    def register_law(self, name_or_pattern: str, canonical_code: str) -> None:
        """Dynamically registers a new legal instrument into this extractor instance."""
        pat = re.compile(r"\b" + re.escape(name_or_pattern) + r"\b", re.IGNORECASE)
        self._custom_patterns.insert(0, (pat, canonical_code))

    def _find_law_spans(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Finds all explicit law mentions with start, end, and canonical law code.
        Combines registered shortcuts with universal dynamic Vietnamese law recognition.
        """
        spans: List[Tuple[int, int, str]] = []
        covered_ranges: List[Tuple[int, int]] = []

        # 1. Custom and registered shortcuts
        for pattern, code in self._custom_patterns + self.LAW_PATTERNS:
            for m in pattern.finditer(text):
                spans.append((m.start(), m.end(), code))
                covered_ranges.append((m.start(), m.end()))

        # 2. Universal Dynamic Statute Recognition for arbitrary legislation
        for m in self.UNIVERSAL_LAW_PATTERN.finditer(text):
            m_start, m_end = m.start(), m.end()
            # Skip if already matched by specific shortcut
            if any(c_start <= m_start and m_end <= c_end for c_start, c_end in covered_ranges):
                continue
            matched_text = m.group(1).strip()
            # Clean trailing punctuation
            matched_text = re.sub(r"[,;.\)]+$", "", matched_text).strip()
            # Disallow single-word generic prefix matches like just "Luật" or "Quy định" without title
            if len(matched_text.split()) < 2:
                continue

            slug = canonical_law_slug(matched_text)
            if slug and slug != "VN_STATUTE" and slug != "VN_LAW_UNKNOWN":
                spans.append((m_start, m_start + len(matched_text), slug))
                covered_ranges.append((m_start, m_start + len(matched_text)))

        spans.sort(key=lambda x: x[0])
        return spans

    def _resolve_local_law_code(
        self,
        match_start: int,
        match_end: int,
        law_spans: List[Tuple[int, int, str]],
        query: str,
    ) -> str:
        """
        Determines the precise statutory scope for a citation based on proximity.
        Prevents scope pollution in mixed queries citing multiple statutes.
        """
        if not law_spans:
            return self.default_law_code

        # 1. Proximity immediately following (e.g., 'Khoản 1 Điều 357 Bộ luật Dân sự 2015')
        for l_start, l_end, l_code in law_spans:
            if l_start >= match_end and (l_start - match_end) <= 60:
                between = query[match_end:l_start]
                if "." not in between and ";" not in between:
                    return l_code

        # 2. Proximity immediately preceding (e.g., 'Theo Luật Thương mại 2005, tại Điều 301')
        best_preceding = None
        min_dist = float("inf")
        for l_start, l_end, l_code in law_spans:
            if l_end <= match_start:
                dist = match_start - l_end
                if dist < min_dist and dist <= 120:
                    between = query[l_end:match_start]
                    if between.count(".") <= 1:
                        min_dist = dist
                        best_preceding = l_code

        if best_preceding:
            return best_preceding

        # 3. If query only names one unique statute, use it
        unique_laws = {code for _, _, code in law_spans}
        if len(unique_laws) == 1:
            return next(iter(unique_laws))

        return self.default_law_code

    def detect_law_code(self, text: str) -> Optional[str]:
        """Detects canonical statutory code from text or returns None if no law is explicitly cited."""
        spans = self._find_law_spans(text)
        if spans:
            return spans[0][2]
        return None

    def extract_citations(self, query: str) -> List[ExtractedLegalCitation]:
        """
        Parses query into structured statutory citations with localized statutory scope resolution.
        """
        if not query or not query.strip():
            return []

        law_spans = self._find_law_spans(query)
        citations: List[ExtractedLegalCitation] = []
        seen_keys: Set[str] = set()
        seen_clause_articles: Set[int] = set()

        # 1. Parse Clause + Article patterns
        for m in self.CLAUSE_ARTICLE_PATTERN.finditer(query):
            clause_num = int(m.group(1))
            art_num = int(m.group(2))
            seen_clause_articles.add(art_num)
            local_law_code = self._resolve_local_law_code(m.start(), m.end(), law_spans, query)
            key = f"{local_law_code}_D{art_num:03d}_K{clause_num:02d}"

            if key not in seen_keys:
                seen_keys.add(key)
                art_node_id = f"{local_law_code}_D{art_num:03d}"
                citations.append(
                    ExtractedLegalCitation(
                        raw_text=m.group(0),
                        article_num=art_num,
                        clause_num=clause_num,
                        law_code=local_law_code,
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
                extra_numbers = re.findall(r"\d+", rest)
                for num_str in extra_numbers:
                    all_arts.append(int(num_str))

            local_law_code = self._resolve_local_law_code(m.start(), m.end(), law_spans, query)

            for art_num in all_arts:
                art_node_id = f"{local_law_code}_D{art_num:03d}"
                # If this article wasn't already covered by a more specific clause or added
                if art_num not in seen_clause_articles and art_node_id not in seen_keys:
                    seen_keys.add(art_node_id)
                    citations.append(
                        ExtractedLegalCitation(
                            raw_text=f"Điều {art_num}",
                            article_num=art_num,
                            clause_num=None,
                            law_code=local_law_code,
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
            if c.canonical_node_id and c.canonical_node_id not in seen:
                seen.add(c.canonical_node_id)
                node_ids.append(c.canonical_node_id)

            if c.clause_num is not None and c.article_node_id and c.article_node_id not in seen:
                seen.add(c.article_node_id)
                node_ids.append(c.article_node_id)

        if node_ids:
            logger.info(
                f"Entity Fast-Path parsed {len(node_ids)} seed node ID(s) from query: {node_ids}"
            )

        return node_ids

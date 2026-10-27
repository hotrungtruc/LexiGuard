"""
Legal Chunker: Hierarchical Statutory & Contractual Syntax Segmentation
========================================================================
Segments Vietnamese legal statutes and contracts according to official drafting standards:
  Phần (Part) -> Chương (Chapter) -> Mục (Section) -> Điều (Article) -> Khoản (Clause) -> Điểm (Point)

Outputs:
  1. Complete structured LegalDocument AST (Pydantic v2)
  2. Granular, context-enriched LegalChunk objects for Qdrant Vector Database
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from src.core.legal_norm import canonical_law_slug
from src.core.schemas import (
    LegalArticle,
    LegalChapter,
    LegalClause,
    LegalDocument,
    LegalChunk,
    LegalPart,
    LegalPoint,
    LegalSection,
)


class LegalChunker:
    """
    Hierarchical chunker for Vietnamese legal codes (Civil Code, Labor Code, etc.)
    and commercial contracts.
    """

    ROMAN_UNICODE_MAP = {
        'Ⅰ': 'I', 'Ⅱ': 'II', 'Ⅲ': 'III', 'Ⅳ': 'IV', 'Ⅴ': 'V',
        'Ⅵ': 'VI', 'Ⅶ': 'VII', 'Ⅷ': 'VIII', 'Ⅸ': 'IX', 'Ⅹ': 'X',
        'Ⅺ': 'XI', 'Ⅻ': 'XII'
    }

    # Statutory Header Patterns (Supports Unicode diacritics, OCR distortions, and Unicode Roman numerals)
    PART_PATTERN = re.compile(
        r'^(?:#+\s*)?(?:PHẦN|PHÀN|PHAN)\s+(?:THỨ\s+([A-ZÀ-Ỹa-zà-ỹ]+)|([IVXLCDM\d]+|[Ⅰ-Ⅻ]+))(?:\s*[:.\-]\s*|\s+)?(.*)$',
        re.IGNORECASE | re.MULTILINE
    )
    CHAPTER_PATTERN = re.compile(
        r'^(?:#+\s*)?(?:CH[ƯUO][ƠO0]NG|CHR[ƠO0]NG|CH\s*[ƯUO][ƠO0]NG)\s+([IVXLCDM\d]+|[Ⅰ-Ⅻ]+)(?:\s*[:.\-]\s*|\s+)?(.*)$',
        re.IGNORECASE | re.MULTILINE
    )
    SECTION_PATTERN = re.compile(
        r'^(?:#+\s*)?(?:MỤC|MUC|MC)\s+(\d+|[IVXLCDM]+|[Ⅰ-Ⅻ]+)(?:\s*[:.\-]\s*|\s+)?(.*)$',
        re.IGNORECASE | re.MULTILINE
    )
    ARTICLE_PATTERN = re.compile(
        r'^(?:#+\s*)?(?:ĐIỀU|ĐIÈU|ĐIU|DIEU|Điều|Điu)\s+(\d+)\s*[:.]\s*(.*)$',
        re.MULTILINE
    )
    CLAUSE_PATTERN = re.compile(
        r'^(?:[#*\-\s]*?)(\d+)\s*[\.\)]\s*(.*)$',
        re.MULTILINE
    )
    POINT_PATTERN = re.compile(
        r'^(?:[#*\-\s]*?)([a-zđA-ZĐ])\)\s*(.*)$',
        re.MULTILINE
    )

    # Reference detection pattern inside clauses
    CITATION_PATTERN = re.compile(
        r'(?:căn cứ|theo quy định tại|quy định tại|áp dụng|chiểu theo|trừ trường hợp|ngoại trừ|khoản\s+\d+\s+)?(?:điều|điu)\s+(\d+)',
        re.IGNORECASE
    )

    def __init__(self):
        logger.info("Initializing Hierarchical LegalChunker (Vietnamese Drafting Conventions)...")

    def parse_statute(
        self,
        text: str,
        law_id: str = "bo-luat-viet-nam",
        law_title: str = "Bộ luật Việt Nam",
        source_file: str = "statute.pdf"
    ) -> LegalDocument:
        """
        Parses full statutory text into a complete nested LegalDocument model.
        """
        lines = [line.strip() for line in text.splitlines()]
        
        current_part_num = None
        current_part_title = None
        pending_part_title = False
        current_chapter_num = "I"
        current_chapter_title = "Những quy định chung"
        pending_chapter_title = False
        current_section_num = None
        current_section_title = None
        pending_section_title = False

        articles: List[LegalArticle] = []
        raw_article_blocks: List[Tuple[int, str, List[str], str, str, Optional[str], Optional[str], Optional[str], Optional[str]]] = []

        curr_art_num = None
        curr_art_title = ""
        curr_art_lines: List[str] = []

        for line in lines:
            if not line:
                continue

            # Check for Part header
            part_match = self.PART_PATTERN.match(line)
            if part_match:
                p_num = part_match.group(1) or part_match.group(2) or "I"
                p_title = part_match.group(3).strip() if part_match.group(3) else ""
                current_part_num = p_num.strip()
                if p_title:
                    current_part_title = p_title
                    pending_part_title = False
                else:
                    pending_part_title = True
                continue

            # Check for Chapter header
            chap_match = self.CHAPTER_PATTERN.match(line)
            if chap_match:
                c_num = chap_match.group(1).strip()
                for u_rom, a_rom in self.ROMAN_UNICODE_MAP.items():
                    c_num = c_num.replace(u_rom, a_rom)
                c_title = chap_match.group(2).strip() if chap_match.group(2) else ""

                # Correct OCR that scans Chapter III (employment contracts) as Chapter II after Chapter II.
                if c_num == "II" and current_chapter_num == "II":
                    c_num = "III"

                current_chapter_num = c_num
                if c_title:
                    current_chapter_title = c_title
                    pending_chapter_title = False
                else:
                    pending_chapter_title = True
                current_section_num = None
                current_section_title = None
                continue

            # Check for Section header
            sec_match = self.SECTION_PATTERN.match(line)
            if sec_match:
                s_num = sec_match.group(1).strip()
                s_title = sec_match.group(2).strip() if sec_match.group(2) else ""
                current_section_num = s_num
                if s_title:
                    current_section_title = s_title
                    pending_section_title = False
                else:
                    pending_section_title = True
                continue

            # Catch pending titles when title appears on immediate next line after header
            if pending_part_title:
                cleaned = re.sub(r'^[#*\s\-]+', '', line).strip()
                if cleaned and not (self.CHAPTER_PATTERN.match(line) or self.ARTICLE_PATTERN.match(line)):
                    current_part_title = cleaned
                    pending_part_title = False
                    continue
                pending_part_title = False

            if pending_chapter_title:
                cleaned = re.sub(r'^[#*\s\-]+', '', line).strip()
                if cleaned and not (self.SECTION_PATTERN.match(line) or self.ARTICLE_PATTERN.match(line) or self.CHAPTER_PATTERN.match(line)):
                    current_chapter_title = cleaned
                    pending_chapter_title = False
                    continue
                pending_chapter_title = False

            if pending_section_title:
                cleaned = re.sub(r'^[#*\s\-]+', '', line).strip()
                if cleaned and not (self.ARTICLE_PATTERN.match(line) or self.SECTION_PATTERN.match(line)):
                    current_section_title = cleaned
                    pending_section_title = False
                    continue
                pending_section_title = False

            # Check for Article header
            art_match = self.ARTICLE_PATTERN.match(line)
            if art_match:
                if curr_art_num is not None:
                    raw_article_blocks.append((
                        curr_art_num,
                        curr_art_title,
                        curr_art_lines,
                        current_chapter_num,
                        current_chapter_title,
                        current_part_num,
                        current_part_title,
                        current_section_num,
                        current_section_title
                    ))
                curr_art_num = int(art_match.group(1))
                curr_art_title = art_match.group(2).strip()
                curr_art_lines = []
                continue

            if curr_art_num is not None:
                curr_art_lines.append(line)

        # Flush final article
        if curr_art_num is not None:
            raw_article_blocks.append((
                curr_art_num,
                curr_art_title,
                curr_art_lines,
                current_chapter_num,
                current_chapter_title,
                current_part_num,
                current_part_title,
                current_section_num,
                current_section_title
            ))

        # Build structured LegalArticle and LegalClause objects
        structured_articles: List[LegalArticle] = []
        for block in raw_article_blocks:
            (art_num, art_title, art_lines, c_num, c_title, p_num, p_title, s_num, s_title) = block
            art_obj = self._build_article(art_num, art_title, art_lines, law_title)
            structured_articles.append(art_obj)

        # Organize into Chapters and Parts
        chapters_map: Dict[str, LegalChapter] = {}
        for block, art_obj in zip(raw_article_blocks, structured_articles):
            c_num = block[3]
            c_title = block[4]
            if c_num not in chapters_map:
                chapters_map[c_num] = LegalChapter(chapter_num=c_num, title=c_title, articles=[])
            chapters_map[c_num].articles.append(art_obj)

        document = LegalDocument(
            law_id=law_id,
            title=law_title,
            chapters=list(chapters_map.values()),
            total_articles=len(structured_articles),
            source_file=source_file
        )
        return document

    def _build_article(self, art_num: int, art_title: str, lines: List[str], law_title: str) -> LegalArticle:
        """Parses individual lines belonging to an article into clauses and points."""
        # If art_title is empty or just punctuation, extract from first line if appropriate
        if not art_title and lines:
            first_line = lines[0].strip()
            if not self.CLAUSE_PATTERN.match(first_line) and not self.POINT_PATTERN.match(first_line):
                art_title = re.sub(r'^[#*\s\-.:]+', '', first_line).strip()
                lines = lines[1:]

        full_text = "\n".join(lines).strip()
        clauses: List[LegalClause] = []

        curr_clause_num: Optional[int] = None
        curr_clause_lines: List[str] = []
        curr_points: List[LegalPoint] = []

        for line in lines:
            clause_match = self.CLAUSE_PATTERN.match(line)
            if clause_match:
                if curr_clause_num is not None:
                    clause_content = "\n".join(curr_clause_lines).strip()
                    clauses.append(LegalClause(
                        clause_num=curr_clause_num,
                        content=clause_content,
                        points=curr_points,
                        canonical_citation=f"Khoản {curr_clause_num} Điều {art_num} {law_title}"
                    ))
                elif curr_clause_lines:
                    # Preamble text preceding a numbered clause (e.g., Clause 1).
                    pre_content = "\n".join(curr_clause_lines).strip()
                    if pre_content:
                        clauses.append(LegalClause(
                            clause_num=1,
                            content=pre_content,
                            points=[],
                            canonical_citation=f"Khoản 1 Điều {art_num} {law_title}"
                        ))
                curr_clause_num = int(clause_match.group(1))
                curr_clause_lines = [clause_match.group(2).strip()]
                curr_points = []
                continue

            point_match = self.POINT_PATTERN.match(line)
            if point_match:
                p_id = point_match.group(1).lower()
                p_content = point_match.group(2).strip()
                citation = (
                    f"Điểm {p_id} Khoản {curr_clause_num or 1} Điều {art_num} {law_title}"
                )
                curr_points.append(LegalPoint(
                    point_id=p_id,
                    content=p_content,
                    canonical_citation=citation
                ))
                curr_clause_lines.append(line)
                continue

            if curr_clause_num is not None:
                curr_clause_lines.append(line)
            else:
                curr_clause_lines.append(line)

        # Flush final clause
        if curr_clause_num is not None:
            clause_content = "\n".join(curr_clause_lines).strip()
            clauses.append(LegalClause(
                clause_num=curr_clause_num,
                content=clause_content,
                points=curr_points,
                canonical_citation=f"Khoản {curr_clause_num} Điều {art_num} {law_title}"
            ))
        elif full_text:
            # Single-clause article without numbered "1."
            clauses.append(LegalClause(
                clause_num=1,
                content=full_text,
                points=curr_points,
                canonical_citation=f"Điều {art_num} {law_title}"
            ))

        return LegalArticle(
            article_num=art_num,
            title=art_title,
            full_text=full_text,
            clauses=clauses,
            canonical_citation=f"Điều {art_num} {law_title}"
        )

    def chunk_statute(
        self,
        text: str,
        law_id: str = "bo-luat-dan-su-2015",
        law_title: str = "Bộ luật Dân sự 2015"
    ) -> List[LegalChunk]:
        """
        Segments statute text into Qdrant-ready granular LegalChunk objects.
        Every chunk includes enriched hierarchical context (Văn bản > Phần > Chương > Điều).
        """
        doc = self.parse_statute(text, law_id=law_id, law_title=law_title)
        law_code = self._get_law_code(law_id)
        chunks: List[LegalChunk] = []

        for chapter in doc.chapters:
            for article in chapter.articles:
                # If article has multiple distinct clauses, chunk by clause
                if len(article.clauses) > 1:
                    for clause in article.clauses:
                        chunk_id = f"{law_code}_D{article.article_num:03d}_K{clause.clause_num:02d}"
                        canonical = clause.canonical_citation
                        
                        context_header = (
                            f"[{law_title}] > [Chương {chapter.chapter_num}: {chapter.title}] > "
                            f"[Điều {article.article_num}: {article.title}]"
                        )
                        enriched_text = f"{context_header}\nKhoản {clause.clause_num}: {clause.content}"
                        refs = self._extract_references(clause.content)

                        chunk = LegalChunk(
                            chunk_id=chunk_id,
                            law_id=law_id,
                            law_title=law_title,
                            chapter_num=chapter.chapter_num,
                            chapter_title=chapter.title,
                            article_num=article.article_num,
                            article_title=article.title,
                            clause_num=clause.clause_num,
                            content=clause.content,
                            canonical_citation=canonical,
                            context_enriched_text=enriched_text,
                            cross_references=refs
                        )
                        chunks.append(chunk)
                else:
                    # Single article chunk
                    chunk_id = f"{law_code}_D{article.article_num:03d}"
                    canonical = article.canonical_citation
                    context_header = (
                        f"[{law_title}] > [Chương {chapter.chapter_num}: {chapter.title}] > "
                        f"[Điều {article.article_num}: {article.title}]"
                    )
                    content = article.full_text if article.full_text else article.title
                    enriched_text = f"{context_header}\n{content}"
                    refs = self._extract_references(content)

                    chunk = LegalChunk(
                        chunk_id=chunk_id,
                        law_id=law_id,
                        law_title=law_title,
                        chapter_num=chapter.chapter_num,
                        chapter_title=chapter.title,
                        article_num=article.article_num,
                        article_title=article.title,
                        clause_num=1,
                        content=content,
                        canonical_citation=canonical,
                        context_enriched_text=enriched_text,
                        cross_references=refs
                    )
                    chunks.append(chunk)

        logger.info(f"Chunked '{law_title}': Generated {len(chunks)} contextual legal chunks.")
        return chunks

    def chunk_contract(self, text: str) -> List[Dict[str, Any]]:
        """
        Splits contract text by sections and operational clauses.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clauses: List[Dict[str, Any]] = []
        curr_clause_id = "general"
        curr_lines: List[str] = []

        for line in lines:
            if re.match(r'^(?:Điều|ĐIỀU|Clause|Article|ARTICLE)\s+(\d+)', line, re.IGNORECASE):
                if curr_lines:
                    clauses.append({
                        "clause_id": curr_clause_id,
                        "text": "\n".join(curr_lines).strip()
                    })
                curr_clause_id = line.split(".")[0].strip() if "." in line else line[:20].strip()
                curr_lines = [line]
            else:
                curr_lines.append(line)

        if curr_lines:
            clauses.append({
                "clause_id": curr_clause_id,
                "text": "\n".join(curr_lines).strip()
            })

        return clauses

    def _extract_references(self, text: str) -> List[str]:
        """Extracts article numbers referenced in the clause text."""
        matches = self.CITATION_PATTERN.findall(text)
        refs = []
        for m in matches:
            if isinstance(m, tuple):
                art = m[-1]
            else:
                art = m
            if art and art.isdigit():
                ref_str = f"Điều {int(art)}"
                if ref_str not in refs:
                    refs.append(ref_str)
        return refs

    @staticmethod
    def _get_law_code(law_id: str) -> str:
        """
        Dynamically maps any law ID into a unique canonical short code.
        Fully generalized across all Vietnamese legislation via canonical_law_slug.
        """
        if not law_id:
            return "VN_LAW_UNKNOWN"
        return canonical_law_slug(law_id)

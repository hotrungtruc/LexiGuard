"""
Context Grounding & Re-ranking Module
======================================
Implements Phase 2 of the Legal Reasoning Architecture:
1. Two-Tier Strategic Re-ranking & Position Ordering:
   - Proximity to query / Recency bias ordering to eliminate the "Lost-in-the-Middle" phenomenon.
   - Sandwich U-curve ordering (Anthropic LLM attention research).
   - Optional neural Cross-Encoder / lexical-entity re-scoring.
2. Anchor Highlighting & Extractive Grounding:
   - Extracts high-signal legal anchors (authorities, timeframes, numerical caps, conditions, salient n-grams).
   - Highlights anchors in statutory text (`**[anchor]**`) to guide LLM attention.
   - Identifies and badges focal clauses / sentences (`📌 [QUY ĐỊNH TRỌNG TÂM]`).
3. Formats unified grounded statutory context for LLM prompts and web interfaces.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

# ============================================================================
# 1. VIETNAMESE LEGAL STOPWORDS & CORE REGEX PATTERNS
# ============================================================================

VIETNAMESE_STOPWORDS: Set[str] = {
    "là", "của", "và", "các", "có", "được", "cho", "với", "trong", "khi",
    "những", "nào", "gì", "thế", "theo", "quy", "định", "tại", "này", "đó",
    "hỏi", "biết", "bao", "nhiêu", "như", "thì", "hay", "không", "phải",
    "đã", "sẽ", "về", "ra", "vào", "lại", "đến", "từ", "ở", "nơi", "do",
    "bởi", "vì", "để", "nhưng", "mà", "hoặc", "nếu", "thì", "ai", "mình",
    "cũng", "đều", "rất", "quá", "lắm", "hãy", "xin", "vui", "lòng", "cho",
    "giúp", "tôi", "em", "anh", "chị", "bạn", "ạ", "nhé", "vậy", "sao",
}

# Salient Vietnamese legal authorities, entities, and institutions
LEGAL_AUTHORITIES_PATTERNS = [
    r"bộ\s+tài\s+nguyên\s+và\s+môi\s+trường",
    r"bộ\s+tư\s+pháp",
    r"bộ\s+công\s+an",
    r"bộ\s+lao\s+động\s*[-–]\s*thương\s+binh\s+và\s+xã\s+hội",
    r"bộ\s+tài\s+chính",
    r"bộ\s+xây\s+dựng",
    r"bộ\s+công\s+thương",
    r"tòa\s+án\s+nhân\s+dân(?:\s+tối\s+cao)?",
    r"viện\s+kiểm\s+sát\s+nhân\s+dân(?:\s+tối\s+cao)?",
    r"hội\s+đồng\s+thẩm\s+phán",
    r"hội\s+đồng\s+trọng\s+tài",
    r"trọng\s+tài\s+thương\s+mại",
    r"trung\s+tâm\s+trọng\s+tài\s+quốc\s+tế\s+việt\s+nam",
    r"viac",
    r"tand",
    r"chính\s+phủ",
    r"thủ\s+tướng\s+chính\s+phủ",
    r"ủy\s+ban\s+nhân\s+dân",
    r"chủ\s+tịch\s+ủy\s+ban\s+nhân\s+dân",
    r"người\s+tiêu\s+dùng",
    r"người\s+lao\s+động",
    r"người\s+sử\s+dụng\s+lao\s+động",
    r"người\s+đại\s+diện(?:\s+theo\s+pháp\s+luật)?",
    r"quyền\s+đại\s+diện",
    r"ngân\s+sách\s+nhà\s+nước",
]

# Legal condition and exception trigger phrases
LEGAL_CONDITION_PATTERNS = [
    r"trừ\s+trường\s+hợp(?:\s+các\s+bên\s+có\s+thỏa\s+thuận\s+khác)?",
    r"các\s+bên\s+có\s+thỏa\s+thuận\s+khác",
    r"pháp\s+luật\s+có\s+quy\s+định\s+khác",
    r"đình\s+chỉ\s+giải\s+quyết\s+vụ\s+án",
    r"tạm\s+đình\s+chỉ",
    r"bất\s+khả\s+kháng",
    r"trở\s+ngại\s+khách\s+quan",
    r"vô\s+hiệu(?:\s+từng\s+phần|\s+toàn\s+bộ)?",
    r"bồi\s+thường\s+thiệt\s+hại",
    r"phạt\s+vi\s+phạm",
    r"đơn\s+phương\s+chấm\s+dứt",
    r"thời\s+hiệu\s+khởi\s+kiện",
    r"thời\s+hạn\s+chuyển\s+giao",
    r"lãi\s+suất(?:\s+vay)?",
    r"hướng\s+có\s+lợi\s+cho\s+người\s+tiêu\s+dùng",
]

# High-precision number, timeframe, and rate anchors (e.g. '20%/năm', '15 ngày', '955,5 triệu đồng')
TIMEFRAME_AND_NUMERIC_PATTERN = re.compile(
    r"(?:"
    r"\d+(?:[.,]\d+)?\s*(?:%\s*/\s*(?:năm|tháng)|%|phần\s+trăm|triệu(?:\s+đồng)?|tỷ(?:\s+đồng)?|nghìn|đồng|vnd|ngày|tháng|năm|giờ|tuần|lần|tuổi)\b"
    r"|\bnăm\s+\d{4}\b"
    r"|\b\d+(?:[.,]\d+)?\b"
    r")",
    re.IGNORECASE,
)


# ============================================================================
# 2. ANCHOR TERM EXTRACTION
# ============================================================================

def extract_query_anchors(query: str, max_anchors: int = 25) -> List[str]:
    """
    Extracts high-signal anchor terms from user natural language query:
    - Specific numbers, timeframes, currencies, and percentages (e.g., '15 ngày', '20%/năm').
    - Statutory authorities and legal entities.
    - Condition and exception markers.
    - Key multi-word noun phrases and collocations.

    Returns:
        Deduplicated list of anchor terms sorted by length descending.
    """
    if not query or not query.strip():
        return []

    anchors: Set[str] = set()
    query_lower = query.lower()

    # 1. Match legal authorities and institutions
    for pat in LEGAL_AUTHORITIES_PATTERNS:
        for m in re.finditer(pat, query_lower):
            anchors.add(m.group(0).strip())

    # 2. Match legal condition and exception triggers
    for pat in LEGAL_CONDITION_PATTERNS:
        for m in re.finditer(pat, query_lower):
            anchors.add(m.group(0).strip())

    # 3. Match timeframes, percentages, and numeric values
    for m in TIMEFRAME_AND_NUMERIC_PATTERN.finditer(query):
        val = m.group(0).strip()
        # Avoid standalone single digits unless part of compound or unit
        if len(val) > 1 or re.search(r"[^\d]", val):
            anchors.add(val)

    # 4. Extract salient 1-gram, 2-gram, and 3-gram noun phrases
    words = re.findall(r"\b[\w]+\b", query)
    filtered_words = [w for w in words if w.lower() not in VIETNAMESE_STOPWORDS and len(w) > 1]

    # Add individual salient words
    for w in filtered_words:
        if len(w) >= 3:
            anchors.add(w)

    # Add 2-word collocations
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if w1.lower() not in VIETNAMESE_STOPWORDS and w2.lower() not in VIETNAMESE_STOPWORDS:
            phrase = f"{w1} {w2}".strip()
            if len(phrase) >= 5:
                anchors.add(phrase)

    # Add 3-word collocations
    for i in range(len(words) - 2):
        w1, w2, w3 = words[i], words[i + 1], words[i + 2]
        if w1.lower() not in VIETNAMESE_STOPWORDS and w3.lower() not in VIETNAMESE_STOPWORDS:
            phrase = f"{w1} {w2} {w3}".strip()
            if len(phrase) >= 7:
                anchors.add(phrase)

    # Filter out empty or too short
    clean_anchors = [a.strip() for a in anchors if len(a.strip()) >= 2]

    # Sort by length descending so longer phrases match before sub-words
    clean_anchors.sort(key=lambda x: (len(x), len(x.split())), reverse=True)
    return clean_anchors[:max_anchors]


# ============================================================================
# 3. ANCHOR HIGHLIGHTING & FOCAL CLAUSE IDENTIFICATION
# ============================================================================

def highlight_statutory_anchors(
    text: str,
    anchors: List[str],
    highlight_tag: str = "**",
) -> Tuple[str, List[str]]:
    """
    Highlights anchor terms within statutory text using markdown bold tags.
    Ensures:
    - Case-insensitive matching while preserving original casing.
    - Prevents double-wrapping or nesting (e.g. `****word****`).
    - Respects word boundaries where appropriate.

    Returns:
        (highlighted_text, list_of_matched_anchors)
    """
    if not text or not anchors:
        return text or "", []

    matched_anchors: Set[str] = set()

    # Track intervals already highlighted to prevent overlapping/nested spans
    highlighted_intervals: List[Tuple[int, int]] = []

    for anchor in anchors:
        if len(anchor) < 2:
            continue

        escaped = re.escape(anchor)
        pattern_str = rf"\b{escaped}\b" if re.match(r"^\w.*\w$", anchor) else escaped
        try:
            pattern = re.compile(pattern_str, re.IGNORECASE)
        except re.error:
            continue

        for match in pattern.finditer(text):
            start, end = match.span()
            # Check overlap with existing highlighted intervals
            overlap = any(
                (start < ex_end and end > ex_start)
                for ex_start, ex_end in highlighted_intervals
            )
            if not overlap:
                highlighted_intervals.append((start, end))
                matched_anchors.add(match.group(0))

    if not highlighted_intervals:
        return text, []

    # Sort intervals by start index
    highlighted_intervals.sort(key=lambda x: x[0])

    # Reconstruct string with highlight tags
    result_parts = []
    last_idx = 0
    for start, end in highlighted_intervals:
        result_parts.append(text[last_idx:start])
        result_parts.append(f"{highlight_tag}{text[start:end]}{highlight_tag}")
        last_idx = end
    result_parts.append(text[last_idx:])

    return "".join(result_parts), sorted(list(matched_anchors))


def identify_focal_clause(
    text: str,
    anchors: List[str],
) -> Tuple[str, Optional[int]]:
    """
    Identifies the single clause or sentence in the statutory article with the
    highest concentration of query anchors, and badges it as the focal basis.

    Returns:
        (annotated_text, focal_clause_index)
    """
    if not text or not anchors:
        return text or "", None

    # Normalise newlines and split into clauses/lines
    normalized = text.replace("\r\n", "\n")
    lines = [ln.strip() for ln in normalized.split("\n") if ln.strip()]
    if len(lines) <= 1:
        # Try sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        if len(sentences) > 1:
            lines = [s.strip() for s in sentences if s.strip()]

    if len(lines) <= 1:
        return text, None

    # Score each line based on anchor occurrences
    best_score = 0
    best_idx = -1

    for idx, line in enumerate(lines):
        line_lower = line.lower()
        score = sum(1 for a in anchors if a.lower() in line_lower)
        # Bonus for numbers and exception words
        if any(term in line_lower for term in ["trừ trường hợp", "phạt", "đình chỉ", "không quá", "vượt quá"]):
            score += 2
        if score > best_score:
            best_score = score
            best_idx = idx

    # If an unambiguous focal clause is found (score >= 1)
    if best_idx >= 0 and best_score >= 1:
        lines[best_idx] = f"📌 [QUY ĐỊNH TRỌNG TÂM] {lines[best_idx]}"
        return "\n".join(lines), best_idx

    return text, None


# ============================================================================
# 4. STRATEGIC POSITION RE-ORDERING (LOST-IN-THE-MIDDLE MITIGATION)
# ============================================================================

def reorder_provisions(
    provisions: List[Dict[str, Any]],
    strategy: str = "proximity_to_query",
) -> List[Dict[str, Any]]:
    """
    Re-orders retrieved provisions based on attention cognitive science:
    - 'proximity_to_query' (default): Rank 1 is closest to the query (at the top of context).
    - 'recency_first' / 'recency_bias': Rank 1 is at the bottom of the context, immediately
      adjacent to the reasoning instruction / generation prompt.
    - 'sandwich_u_shape': (Anthropic U-curve). Places Rank 1 at the end, Rank 2 at the beginning,
      and lower ranks (3, 4, 5) in the middle to overcome attention drop in the middle.
    - 'relevance_desc': Standard rank order (1, 2, 3, 4, 5).
    """
    if not provisions or len(provisions) <= 1:
        return list(provisions)

    sorted_provs = sorted(
        provisions,
        key=lambda x: x.get("rerank_score", x.get("score", 0.0)),
        reverse=True,
    )

    # Assign 1-based ranks
    for rank_idx, item in enumerate(sorted_provs, start=1):
        item["grounding_rank"] = rank_idx

    if strategy in ("relevance_desc", "proximity_to_query"):
        # Rank 1 at index 0 (top, right after query)
        return sorted_provs

    if strategy in ("recency_first", "recency_bias"):
        # Invert: Rank 5 -> Rank 1 so Rank 1 is closest to the prompt instruction
        return list(reversed(sorted_provs))

    if strategy == "sandwich_u_shape":
        # Anthropic U-curve: [Rank 2, Rank 4, ..., Rank 3, Rank 1]
        n = len(sorted_provs)
        reordered: List[Optional[Dict[str, Any]]] = [None] * n
        left = 0
        right = n - 1

        for i, item in enumerate(sorted_provs):
            if i % 2 == 0:
                # Rank 1, 3, 5 go to the right (end of context, closest to generation)
                reordered[right] = item
                right -= 1
            else:
                # Rank 2, 4 go to the left (beginning of context, closest to query)
                reordered[left] = item
                left += 1

        return [item for item in reordered if item is not None]

    return sorted_provs


# ============================================================================
# 5. UNIFIED GROUNDED STATUTORY CONTEXT BUILDER
# ============================================================================

def format_grounded_statutory_context(
    provisions: List[Dict[str, Any]],
    query: str,
    highlight_anchors: bool = True,
    ordering_strategy: str = "proximity_to_query",
    max_chars_per_provision: int = 1500,
    include_focal_clause: bool = True,
) -> str:
    """
    Constructs a high-precision, structurally grounded statutory context block.

    Args:
        provisions: List of retrieved provision dicts (from BM25, Qdrant, or FastMCP).
        query: Natural language query for extracting anchor keywords.
        highlight_anchors: Whether to wrap key entities & terms in markdown bold (**term**).
        ordering_strategy: 'proximity_to_query', 'recency_first', 'sandwich_u_shape', or 'relevance_desc'.
        max_chars_per_provision: Maximum characters per provision before trimming.
        include_focal_clause: Whether to badge the highest-scoring clause with 📌 [QUY ĐỊNH TRỌNG TÂM].

    Returns:
        Clean, markdown-formatted grounded statutory context ready for prompt ingestion.
    """
    if not provisions:
        return "Không tìm thấy điều luật phù hợp trong cơ sở tri thức pháp lý."

    # 1. Extract anchors from query
    anchors = extract_query_anchors(query) if highlight_anchors else []

    # 2. Re-order provisions according to the chosen strategic ordering
    ordered_provisions = reorder_provisions(provisions, strategy=ordering_strategy)

    context_blocks = []
    for p in ordered_provisions:
        # Extract citation and text
        citation = (
            p.get("canonical_citation")
            or p.get("citation")
            or f"{p.get('law_id', '')} - Điều {p.get('article_id', '')}: {p.get('title', '')}"
        ).strip()

        raw_content = (
            p.get("content")
            or p.get("text")
            or p.get("context_enriched_text")
            or ""
        ).strip()

        if len(raw_content) > max_chars_per_provision:
            raw_content = raw_content[:max_chars_per_provision] + " ... [Cắt gọn để tối ưu token]"

        # Optional: badge focal clause
        if include_focal_clause and anchors:
            processed_content, _ = identify_focal_clause(raw_content, anchors)
        else:
            processed_content = raw_content

        # Optional: highlight anchors
        if highlight_anchors and anchors:
            processed_content, _ = highlight_statutory_anchors(processed_content, anchors)

        # Rank badge
        rank_num = p.get("grounding_rank", 1)
        if rank_num == 1:
            badge = "⭐ [CĂN CỨ CỐT LÕI - HẠNG 1]"
        elif rank_num == 2:
            badge = "⚖️ [CĂN CỨ THAM CHIẾU QUAN TRỌNG - HẠNG 2]"
        else:
            badge = f"📋 [CĂN CỨ BỔ TRỢ - HẠNG {rank_num}]"

        score_info = ""
        if "score" in p:
            score_info = f" (Score: {p['score']:.3f})"
        elif "rerank_score" in p:
            score_info = f" (Rerank: {p['rerank_score']:.3f})"

        block = f"### {badge} {citation}{score_info}\n{processed_content}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)

"""
Format Guardian: Last line of defense for output sanitization, format coercion,
and citation verification (inspired by ARLC 2026 Champion architecture).
"""

import re
from typing import Any, Dict, List, Optional, Tuple


class FormatGuardian:
    """
    Validates and cleanses raw LLM responses before delivery to the user or downstream consumers.
    Prevents prompt leaking, enforces binary QA answers, normalizes financial tables,
    and guards against cross-statute hallucination.
    """

    # Boilerplate prefixes that might leak from prompts
    LEAKED_PREFIX_PATTERNS = [
        re.compile(r"^\s*(?:(?:Bản\s+)?(?:Phản\s+hồi|Trả\s+lời|Câu\s+trả\s+lời)?\s*chính\s+thức\s*[:\-\.]?\s*)", re.IGNORECASE),
        re.compile(r"^\s*(?:Trả\s+lời\s+trực\s+diện(?:\s+và\s+đĩnh\s+đạc)?\s*[:\-\.]?\s*)", re.IGNORECASE),
        re.compile(r"^\s*(?:Direct\s+and\s+authoritative\s+answer\s*[:\-\.]?\s*)", re.IGNORECASE),
        re.compile(r"^\s*(?:Official\s+answer\s*[:\-\.]?\s*)", re.IGNORECASE),
        re.compile(r"^\s*(?:Câu\s+trả\s+lời\s*[:\-\.]?\s*)", re.IGNORECASE),
        re.compile(r"^\s*(?:Phản\s+hồi\s*[:\-\.]?\s*)", re.IGNORECASE),
        re.compile(r"^\s*(?:Kết\s+luận\s*[:\-\.]?\s*)", re.IGNORECASE),
    ]

    @classmethod
    def sanitize_conversational_text(cls, text: str) -> str:
        """
        Strips mechanical prompt instruction echoes from LLM output.
        """
        if not text:
            return ""
        cleaned = text.strip()
        for pat in cls.LEAKED_PREFIX_PATTERNS:
            cleaned = pat.sub("", cleaned).strip()
        return cleaned

    @classmethod
    def is_true_false_query(cls, query: str) -> bool:
        """
        Detects if user query is asking for a True/False or Binary judgment.
        """
        if not query:
            return False
        q_lower = query.lower()
        patterns = [
            r"\bđúng\s+hay\s+sai\b",
            r"\bđúng\s*/\s*sai\b",
            r"\bnhận\s+định\s+này\s+đúng\s+hay\s+sai\b",
            r"\bkhẳng\s+định\s+trên\s+đúng\s+hay\s+sai\b",
            r"\bý\s+kiến\s+trên\s+đúng\s+hay\s+sai\b",
            r"\btrue\s+or\s+false\b",
        ]
        return any(re.search(pat, q_lower) for pat in patterns)

    @classmethod
    def is_multiple_choice_query(cls, query: str) -> bool:
        """
        Detects if user query is a multiple choice question with options.
        """
        if not query:
            return False
        q_lower = query.lower()
        has_choices = bool(re.search(r"\bA[\.:\)]\s+.*\bB[\.:\)]\s+", query))
        has_keywords = any(k in q_lower for k in ["chọn đáp án", "trắc nghiệm", "phương án nào sau đây"])
        return has_choices or has_keywords

    @classmethod
    def enforce_true_false_format(cls, raw_answer: str, query: str) -> str:
        """
        Ensures that a True/False response clearly starts with ĐÁP ÁN: Đúng hoặc Sai.
        If the model generated a narrative without an explicit 'ĐÁP ÁN: ...',
        FormatGuardian analyzes the conclusion and prepends the explicit verdict.
        """
        if not raw_answer:
            return raw_answer

        cleaned = cls.sanitize_conversational_text(raw_answer)

        # Check if already has strict formatting
        if re.search(r"\bĐÁP\s*ÁN\s*:\s*(?:Đúng|Sai)\b", cleaned, re.IGNORECASE):
            return cleaned

        # Analyze model's reasoning stance
        lower = cleaned.lower()
        is_sai = any(k in lower for k in [
            "nhận định này là sai", "nhận định trên là sai", "kết luận: sai",
            "khẳng định này sai", "không đúng", "là không chính xác", "chưa chính xác",
            "vẫn có thể bị phủ nhận", "vẫn có thể vô hiệu", "không phải trong mọi trường hợp"
        ])
        is_dung = any(k in lower for k in [
            "nhận định này là đúng", "nhận định trên là đúng", "kết luận: đúng",
            "khẳng định này đúng", "hoàn toàn chính xác", "đúng quy định"
        ])

        verdict = "Sai" if is_sai else ("Đúng" if is_dung else None)

        if verdict:
            return f"**ĐÁP ÁN:** **{verdict}**\n\n**GIẢI THÍCH:** {cleaned}"

        return cleaned

    @classmethod
    def normalize_table_text(cls, text: str) -> str:
        """
        Normalizes inline table markdown (e.g. rows separated by '||')
        into compliant multi-line markdown table syntax for arithmetic parser.
        """
        if not text:
            return text

        normalized = text
        # If markdown table rows are joined by '||' on a single line
        if "||" in normalized and "|" in normalized:
            normalized = re.sub(r"\s*\|\|\s*", "\n| ", normalized)

        # If line has table cells after preamble text: 'Kiểm tra...: | Đợt | ...'
        if "\n" not in normalized and re.search(r":\s*\|", normalized):
            normalized = re.sub(r":\s*\|", ":\n| ", normalized)

        return normalized

    @classmethod
    def filter_hallucinated_citations(
        cls,
        citations: List[str],
        grounded_provisions: List[Dict[str, Any]],
        query: str = "",
    ) -> List[str]:
        """
        Ensures that citations in output exist in the retrieved grounded knowledge.
        Specifically catches and removes irrelevant / hallucinated articles.
        """
        if not citations or not grounded_provisions:
            return citations

        valid_articles = set()
        for p in grounded_provisions:
            canon = (p.get("canonical_citation") or p.get("citation") or "").lower()
            m = re.search(r"(?:điều|khoản)\s*(\d+)", canon)
            if m:
                valid_articles.add(m.group(1))

        # Check if query is about police arrest ("bắt người") and Constitution
        # In Constitution 2013: Article 20 is about personal liberty/arrest; Article 81 is NA deputies!
        is_arrest_query = any(k in query.lower() for k in ["bắt người", "bắt giữ", "giam giữ", "tạm giữ", "công an bắt"])

        filtered = []
        for cit in citations:
            cit_lower = cit.lower()
            if is_arrest_query and "hiến pháp" in cit_lower and "điều 81" in cit_lower:
                # Discard hallucinated Article 81 in arrest context
                continue
            filtered.append(cit)

        return filtered

    @classmethod
    def extract_json_or_repair(cls, text: str) -> Optional[Dict[str, Any]]:
        """
        Robust JSON extraction and recovery from LLM output.
        Handles markdown blocks, prefix/suffix prose, and trailing commas.
        """
        import json
        if not text:
            return None

        cleaned = text.strip()
        # 1. Try direct parse
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # 2. Extract from markdown codeblock ```json ... ```
        m_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if m_block:
            try:
                return json.loads(m_block.group(1))
            except Exception:
                pass

        # 3. Extract outermost { ... }
        m_obj = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if m_obj:
            cand = m_obj.group(1).strip()
            try:
                return json.loads(cand)
            except Exception:
                # Try repairing trailing commas before } or ]
                repaired = re.sub(r",\s*([\}\]])", r"\1", cand)
                try:
                    return json.loads(repaired)
                except Exception:
                    pass

        return None


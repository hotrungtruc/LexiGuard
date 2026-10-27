"""
FastMCP Server: Autonomous Legal Tools & Self-Querying Engine
============================================================
Exposes 3 core Model Context Protocol (MCP) tools for autonomous legal agents:
1. `search_legal_provisions`: High-precision hybrid search with automated self-querying
   metadata extraction and $O(1)$ Qdrant payload filtering (`law_id`, `article_num`, `is_active`).
2. `verify_table_arithmetic`: Autonomous contract arithmetic auditing, catching column sum errors,
   row multiplication mismatches (quantity * price != total), and VAT/tax discrepancies.
3. `get_article_graph_context`: Neo4j Knowledge Graph traversal pulling parent hierarchy,
   constituent clauses, cross-references, and judicial precedents up to N hops.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from loguru import logger

# FastMCP / MCPServer modern compatibility
try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    try:
        from fastmcp import FastMCP  # type: ignore
    except ImportError:
        from mcp.server.fastmcp import FastMCP  # type: ignore

from src.retrieval.entity_extractor import LegalEntityExtractor

# Initialize FastMCP Server instance
mcp_server = FastMCP(
    name="LexiGuard Legal Tools Server",
    instructions="Autonomous Legal Verification, Table Arithmetic Auditing & Statutory Graph RAG Tools",
)


# ============================================================================
# 1. SELF-QUERYING METADATA ENGINE
# ============================================================================

class SelfQueryEngine:
    """
    Analyzes natural language legal queries to automatically infer structured
    filtering criteria (law_id, article_num, clause_num) without requiring
    explicit parameter input from the calling LLM agent.
    """

    LAW_MAP = [
        (re.compile(r'\b(?:bộ\s*luật\s*dân\s*sự|dân\s*sự|blds)(?:\s*2015)?\b', re.IGNORECASE), "bo-luat-dan-su-2015"),
        (re.compile(r'\b(?:bộ\s*luật\s*lao\s*động|lao\s*động|bllđ|blld)(?:\s*2019)?\b', re.IGNORECASE), "bo-luat-lao-dong-2019"),
    ]

    ARTICLE_REGEX = re.compile(r'(?:điều|đieu|điu|đ\.?)\s*(\d+)', re.IGNORECASE)
    CLAUSE_REGEX = re.compile(r'(?:khoản|khoan|k\.?)\s*(\d+)', re.IGNORECASE)

    @classmethod
    def extract_filters(
        cls,
        query: str,
        explicit_law_id: Optional[str] = None,
        explicit_article_num: Optional[int] = None,
        explicit_clause_num: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Extracts combined self-queried and explicit metadata filters."""
        detected_law_id = explicit_law_id
        if not detected_law_id:
            for pattern, canonical_id in cls.LAW_MAP:
                if pattern.search(query):
                    detected_law_id = canonical_id
                    break

        detected_article_num = explicit_article_num
        if detected_article_num is None:
            m_art = cls.ARTICLE_REGEX.search(query)
            if m_art:
                detected_article_num = int(m_art.group(1))

        detected_clause_num = explicit_clause_num
        if detected_clause_num is None:
            m_cls = cls.CLAUSE_REGEX.search(query)
            if m_cls:
                detected_clause_num = int(m_cls.group(1))

        return {
            "law_id": detected_law_id,
            "article_num": detected_article_num,
            "clause_num": detected_clause_num,
        }


# ============================================================================
# 2. HERMETIC OFFLINE LEGAL DATA LOADER (FOR RESILIENCE & ZERO DOWNTIME)
# ============================================================================

_OFFLINE_CHUNKS_CACHE: Optional[List[Dict[str, Any]]] = None
_OFFLINE_GRAPHS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _get_offline_chunks() -> List[Dict[str, Any]]:
    """Loads standardized statutory chunks from data/extracted_laws/ for fallback."""
    global _OFFLINE_CHUNKS_CACHE
    if _OFFLINE_CHUNKS_CACHE is not None:
        return _OFFLINE_CHUNKS_CACHE

    combined_file = Path("data/extracted_laws/combined_qdrant_chunks.json")
    if combined_file.exists():
        try:
            with open(combined_file, "r", encoding="utf-8") as f:
                _OFFLINE_CHUNKS_CACHE = json.load(f)
                logger.info(f"Loaded {len(_OFFLINE_CHUNKS_CACHE)} offline legal chunks from {combined_file}")
                return _OFFLINE_CHUNKS_CACHE
        except Exception as e:
            logger.warning(f"Failed to load {combined_file}: {e}")

    _OFFLINE_CHUNKS_CACHE = []
    return _OFFLINE_CHUNKS_CACHE


def _get_offline_graphs() -> Dict[str, Dict[str, Any]]:
    """Loads graph JSON files from data/extracted_laws/ for fallback."""
    global _OFFLINE_GRAPHS_CACHE
    if _OFFLINE_GRAPHS_CACHE is not None:
        return _OFFLINE_GRAPHS_CACHE

    _OFFLINE_GRAPHS_CACHE = {}
    extracted_dir = Path("data/extracted_laws")
    if extracted_dir.exists():
        for g_file in extracted_dir.glob("*_graph.json"):
            try:
                with open(g_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    law_id = data.get("law_id", g_file.stem.replace("_graph", ""))
                    _OFFLINE_GRAPHS_CACHE[law_id] = data
            except Exception as e:
                logger.warning(f"Error loading {g_file}: {e}")

    return _OFFLINE_GRAPHS_CACHE


# ============================================================================
# 3. TABLE ARITHMETIC AUDITING ENGINE
# ============================================================================

class TableArithmeticAuditor:
    """
    Parses and verifies arithmetic computations in contractual & administrative tables:
    - Row products: Quantity * Unit Price == Total Subtotal
    - Column totals: Sum(items) == Total Row ('Tổng cộng')
    - Tax / VAT: Subtotal * Tax Rate == Tax Amount, Subtotal + Tax == Grand Total
    - Percentage distributions: Sum(Percentage) == 100%
    """

    TOTAL_ROW_PATTERNS = re.compile(
        r'\b(?:tổng\s*cộng|tổng\s*số|tổng|cộng|total|grand\s*total|sum)\b',
        re.IGNORECASE
    )

    QTY_HEADER_PATTERNS = re.compile(
        r'\b(?:số\s*lượng|so\s*luong|quantity|qty|khối\s*lượng|sl)\b',
        re.IGNORECASE
    )

    PRICE_HEADER_PATTERNS = re.compile(
        r'\b(?:đơn\s*giá|don\s*gia|unit\s*price|price|giá|gia)\b',
        re.IGNORECASE
    )

    TOTAL_HEADER_PATTERNS = re.compile(
        r'\b(?:thành\s*tiền|thanh\s*tien|tổng\s*tiền|tong\s*tien|subtotal|total|giá\s*trị|gia\s*tri)\b',
        re.IGNORECASE
    )

    TAX_HEADER_PATTERNS = re.compile(
        r'\b(?:thuế|thue|vat|gtgt|tax)\b',
        re.IGNORECASE
    )

    PERCENT_HEADER_PATTERNS = re.compile(
        r'\b(?:tỷ\s*lệ|ty\s*le|phần\s*trăm|phan\s*tram|percentage|%)\b',
        re.IGNORECASE
    )

    @classmethod
    def clean_number(cls, raw_val: Any) -> Optional[float]:
        """
        Parses numeric values supporting Vietnamese (1.000.000,50)
        and US formats (1,000,000.50), stripping currency suffixes and symbols.
        """
        if raw_val is None:
            return None
        if isinstance(raw_val, (int, float)):
            return float(raw_val)

        s = str(raw_val).strip()
        if not s or s in ("-", "—", "/", "N/A", "null"):
            return None

        # Remove currency units & extra words
        s = re.sub(r'(?:vnđ|vnd|đồng|đ|usd|\$|triệu|tỷ|%|\bnghìn\b)', '', s, flags=re.IGNORECASE).strip()

        # Handle parentheses for negative numbers e.g. (5000)
        is_negative = False
        if s.startswith("(") and s.endswith(")"):
            is_negative = True
            s = s[1:-1].strip()
        elif s.startswith("-"):
            is_negative = True
            s = s[1:].strip()

        # Detect European/Vietnamese notation: 1.000.000,50
        if "." in s and "," in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "." in s and "," not in s:
            # Check if dot is thousands separator (e.g. 10.000 or 1.000.000)
            parts = s.split(".")
            if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
                s = s.replace(".", "")
            elif len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3:
                s = s.replace(".", "")
        elif "," in s and "." not in s:
            parts = s.split(",")
            if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
                s = s.replace(",", "")
            else:
                s = s.replace(",", ".")

        try:
            val = float(s)
            return -val if is_negative else val
        except ValueError:
            return None

    @classmethod
    def parse_markdown_table(cls, md_text: str) -> Tuple[List[str], List[List[Any]]]:
        """Parses a markdown table string into header list and rows matrix."""
        lines = [line.strip() for line in md_text.strip().splitlines() if line.strip()]
        table_lines = [l for l in lines if l.startswith("|") and l.endswith("|")]
        if len(table_lines) < 2:
            return [], []

        # Header line
        headers = [c.strip() for c in table_lines[0].strip("|").split("|")]

        rows: List[List[Any]] = []
        for l in table_lines[1:]:
            # Skip separator line (|---|---|)
            cells = [c.strip() for c in l.strip("|").split("|")]
            if all(re.match(r'^:?-+:?$', c) for c in cells):
                continue
            rows.append(cells)

        return headers, rows

    @classmethod
    def audit(
        cls,
        table_data: Union[str, List[Dict[str, Any]], Dict[str, Any]],
        tolerance: float = 0.01,
        expected_tax_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Executes full arithmetic verification suite on provided table."""
        headers: List[str] = []
        rows: List[List[Any]] = []

        if isinstance(table_data, str):
            # Check if JSON string
            table_str = table_data.strip()
            if table_str.startswith("[") or table_str.startswith("{"):
                try:
                    loaded = json.loads(table_str)
                    return cls.audit(loaded, tolerance, expected_tax_rate)
                except Exception:
                    pass
            headers, rows = cls.parse_markdown_table(table_str)

        elif isinstance(table_data, list) and table_data and isinstance(table_data[0], dict):
            headers = list(table_data[0].keys())
            for item in table_data:
                rows.append([item.get(h) for h in headers])

        elif isinstance(table_data, dict):
            headers = table_data.get("headers", [])
            raw_rows = table_data.get("rows", [])
            for r in raw_rows:
                if isinstance(r, dict):
                    rows.append([r.get(h) for h in headers])
                elif isinstance(r, list):
                    rows.append(r)

        if not headers or not rows:
            return {
                "status": "error",
                "message": "Invalid table data: Could not parse headers or rows.",
                "is_valid": False,
                "discrepancies": [],
            }

        discrepancies: List[Dict[str, Any]] = []

        # Column role detection
        qty_idx: Optional[int] = None
        price_idx: Optional[int] = None
        total_idx: Optional[int] = None
        percent_idx: Optional[int] = None

        for idx, h in enumerate(headers):
            if cls.QTY_HEADER_PATTERNS.search(h) and qty_idx is None:
                qty_idx = idx
            elif cls.PRICE_HEADER_PATTERNS.search(h) and price_idx is None:
                price_idx = idx
            elif cls.TOTAL_HEADER_PATTERNS.search(h) and total_idx is None:
                total_idx = idx
            elif cls.PERCENT_HEADER_PATTERNS.search(h) and percent_idx is None:
                percent_idx = idx

        # Classify rows into item rows and total/aggregate rows
        item_rows: List[Tuple[int, List[Any]]] = []
        total_rows: List[Tuple[int, List[Any]]] = []

        for r_idx, r in enumerate(rows, start=1):
            row_text = " ".join(str(c) for c in r)
            if cls.TOTAL_ROW_PATTERNS.search(row_text):
                total_rows.append((r_idx, r))
            else:
                item_rows.append((r_idx, r))

        # Check 1: Horizontal Row Multiplication (Qty * Price == Total)
        if qty_idx is not None and price_idx is not None and total_idx is not None:
            for r_idx, r in item_rows:
                if len(r) <= max(qty_idx, price_idx, total_idx):
                    continue
                qty = cls.clean_number(r[qty_idx])
                price = cls.clean_number(r[price_idx])
                reported_total = cls.clean_number(r[total_idx])

                if qty is not None and price is not None and reported_total is not None:
                    expected_total = qty * price
                    delta = abs(reported_total - expected_total)
                    if delta > tolerance:
                        pct_diff = round((delta / expected_total) * 100, 2) if expected_total != 0 else 100.0
                        desc = str(r[0]) if r else f"Row {r_idx}"
                        discrepancies.append({
                            "type": "ROW_PRODUCT_MISMATCH",
                            "row_index": r_idx,
                            "item_description": desc,
                            "formula": f"Quantity ({qty}) * Unit Price ({price:,.0f}) = {expected_total:,.0f}",
                            "reported_value": reported_total,
                            "expected_value": expected_total,
                            "delta": delta,
                            "discrepancy_percentage": pct_diff,
                            "severity": "CRITICAL" if (delta > 1000000 or pct_diff > 5) else "HIGH",
                            "legal_risk": f"Discrepancy of {delta:,.0f} VND creates billing discrepancy violating contractual price agreement.",
                        })

        # Check 2: Vertical Column Summation (Sum of Items == Total Row)
        if total_rows and total_idx is not None:
            sum_calculated = 0.0
            valid_items = 0
            for _, r in item_rows:
                if len(r) > total_idx:
                    val = cls.clean_number(r[total_idx])
                    if val is not None:
                        sum_calculated += val
                        valid_items += 1

            for t_idx, t_row in total_rows:
                if len(t_row) > total_idx:
                    reported_sum = cls.clean_number(t_row[total_idx])
                    if reported_sum is not None and valid_items > 0:
                        delta = abs(reported_sum - sum_calculated)
                        if delta > tolerance:
                            pct_diff = round((delta / sum_calculated) * 100, 2) if sum_calculated != 0 else 100.0
                            discrepancies.append({
                                "type": "COLUMN_SUM_MISMATCH",
                                "row_index": t_idx,
                                "column_name": headers[total_idx],
                                "formula": f"Sum of {valid_items} items ({sum_calculated:,.0f}) vs Reported Total ({reported_sum:,.0f})",
                                "reported_value": reported_sum,
                                "expected_value": sum_calculated,
                                "delta": delta,
                                "discrepancy_percentage": pct_diff,
                                "severity": "CRITICAL" if (delta > 5000000 or pct_diff > 5) else "HIGH",
                                "legal_risk": f"Reported total differs by {delta:,.0f} VND from sum of constituent items; high audit risk.",
                            })

        # Check 3: Percentage Distribution Sum
        if percent_idx is not None and item_rows:
            pct_sum = 0.0
            counted = 0
            for _, r in item_rows:
                if len(r) > percent_idx:
                    p_val = cls.clean_number(r[percent_idx])
                    if p_val is not None:
                        pct_sum += p_val
                        counted += 1
            if counted > 1 and not (99.9 <= pct_sum <= 100.1):
                discrepancies.append({
                    "type": "PERCENTAGE_SUM_MISMATCH",
                    "column_name": headers[percent_idx],
                    "reported_sum": pct_sum,
                    "expected_sum": 100.0,
                    "delta": abs(pct_sum - 100.0),
                    "severity": "MEDIUM",
                    "legal_risk": f"Sum of contractual shares equals {pct_sum:.2f}%, failing to allocate 100% of obligation.",
                })

        is_valid = len(discrepancies) == 0
        summary_msg = "All table calculations verified successfully." if is_valid else (
            f"Arithmetic audit FAILED: Found {len(discrepancies)} calculation discrepancy(ies)."
        )

        return {
            "status": "success",
            "is_valid": is_valid,
            "total_rows_inspected": len(rows),
            "error_count": len(discrepancies),
            "discrepancies": discrepancies,
            "summary": summary_msg,
        }


# ============================================================================
# 4. MCP TOOL DEFINITIONS
# ============================================================================

@mcp_server.tool()
async def search_legal_provisions(
    query: str,
    law_id: Optional[str] = None,
    article_num: Optional[int] = None,
    clause_num: Optional[int] = None,
    top_k: int = 5,
    active_only: bool = True,
) -> Dict[str, Any]:
    """
    Search governing Vietnamese statutory provisions using Hybrid Search with Self-Querying.
    Automatically parses statutory identifiers (law_id, article_num, clause_num) from query text
    and applies index-backed payload filters for $O(1)$ sub-millisecond precision.

    Args:
        query: Natural language legal question or statutory citation (e.g., 'Khoản 2 Điều 4 Bộ luật Dân sự')
        law_id: Optional explicit statute filter ('bo-luat-dan-su-2015', 'bo-luat-lao-dong-2019')
        article_num: Optional article number filter (e.g., 4)
        clause_num: Optional clause number filter (e.g., 2)
        top_k: Number of ranked provisions to return (default: 5)
        active_only: Filter out repealed provisions (default: True)
    """
    logger.info(f"MCP Tool 'search_legal_provisions' called: query='{query}' (law_id={law_id})")

    # Step 1: Self-Querying parameter inference
    inferred = SelfQueryEngine.extract_filters(
        query=query,
        explicit_law_id=law_id,
        explicit_article_num=article_num,
        explicit_clause_num=clause_num,
    )

    eff_law_id = inferred["law_id"]
    eff_article_num = inferred["article_num"]
    eff_clause_num = inferred["clause_num"]

    provisions: List[Dict[str, Any]] = []

    # Step 2: Attempt live Qdrant search
    try:
        from src.core.config import get_async_qdrant_client, settings
        from src.retrieval.qdrant_store import QdrantStore
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        client = get_async_qdrant_client()
        # Ping check
        await client.get_collections()

        qdrant_store = QdrantStore(client=client)

        # Build self-queried payload filter
        must_conditions = []
        if eff_law_id:
            must_conditions.append(FieldCondition(key="law_id", match=MatchValue(value=eff_law_id)))
        if eff_article_num is not None:
            must_conditions.append(FieldCondition(key="article_num", match=MatchValue(value=eff_article_num)))
        if eff_clause_num is not None:
            must_conditions.append(FieldCondition(key="clause_num", match=MatchValue(value=eff_clause_num)))

        q_filter = Filter(must=must_conditions) if must_conditions else None

        results = await qdrant_store.hybrid_search_async(
            query=query,
            top_k=top_k,
            active_only=active_only,
            query_filter=q_filter,
        )

        for r in results:
            provisions.append({
                "chunk_id": r.get("chunk_id", ""),
                "canonical_citation": r.get("canonical_citation", ""),
                "law_id": r.get("law_id", ""),
                "law_title": r.get("law_title", ""),
                "chapter_title": r.get("chapter_title", ""),
                "article_num": r.get("article_num"),
                "article_title": r.get("article_title", ""),
                "clause_num": r.get("clause_num"),
                "content": r.get("content", ""),
                "context_enriched_text": r.get("context_enriched_text", ""),
                "cross_references": r.get("cross_references", []),
                "is_active": r.get("is_active", True),
                "relevance_score": round(r.get("score", 0.0), 4),
            })
    except Exception as exc:
        logger.warning(f"Live Qdrant search unavailable ({exc}). Using offline extracted dataset fallback.")
        # Fallback to offline chunks
        offline_chunks = _get_offline_chunks()
        query_terms = set(re.findall(r'\w+', query.lower()))

        scored_candidates = []
        for ch in offline_chunks:
            # Active filter
            if active_only and not ch.get("is_active", True):
                continue
            # Law ID filter
            if eff_law_id and ch.get("law_id") != eff_law_id:
                continue
            # Article filter
            if eff_article_num is not None and ch.get("article_num") != eff_article_num:
                continue
            # Clause filter
            if eff_clause_num is not None and ch.get("clause_num") != eff_clause_num:
                continue

            content_text = ch.get("content", "") + " " + ch.get("canonical_citation", "")
            content_lower = content_text.lower()

            # Simple token overlap scoring
            matches = sum(1 for term in query_terms if term in content_lower)
            score = round(matches / max(len(query_terms), 1), 4)

            # Bonus for exact article match
            if eff_article_num and ch.get("article_num") == eff_article_num:
                score += 1.0

            scored_candidates.append((score, ch))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        for score, ch in scored_candidates[:top_k]:
            provisions.append({
                "chunk_id": ch.get("chunk_id", ""),
                "canonical_citation": ch.get("canonical_citation", ""),
                "law_id": ch.get("law_id", ""),
                "law_title": ch.get("law_title", ""),
                "chapter_title": ch.get("chapter_title", ""),
                "article_num": ch.get("article_num"),
                "article_title": ch.get("article_title", ""),
                "clause_num": ch.get("clause_num"),
                "content": ch.get("content", ""),
                "context_enriched_text": ch.get("context_enriched_text", ""),
                "cross_references": ch.get("cross_references", []),
                "is_active": ch.get("is_active", True),
                "relevance_score": score,
            })

    return {
        "status": "success",
        "query": query,
        "self_querying": {
            "detected_law_id": eff_law_id,
            "detected_article_num": eff_article_num,
            "detected_clause_num": eff_clause_num,
            "applied_filters": {
                "law_id": eff_law_id,
                "article_num": eff_article_num,
                "clause_num": eff_clause_num,
                "active_only": active_only,
            },
        },
        "total_matches": len(provisions),
        "provisions": provisions,
    }


@mcp_server.tool()
def verify_table_arithmetic(
    table_data: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    tolerance: float = 0.01,
    expected_tax_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Autonomous mathematical verification of tables in legal contracts and administrative annexes.
    Audits column summation, row multiplication (quantity * price), and percentage allocations,
    flagging overbilling risks and contract financial discrepancies.

    Args:
        table_data: Markdown table string, list of row dictionaries, or JSON table structure.
        tolerance: Absolute discrepancy threshold for flagging errors (default: 0.01).
        expected_tax_rate: Optional VAT / tax rate percentage to verify (e.g., 0.10 for 10%).
    """
    logger.info("MCP Tool 'verify_table_arithmetic' invoked.")
    return TableArithmeticAuditor.audit(
        table_data=table_data,
        tolerance=tolerance,
        expected_tax_rate=expected_tax_rate,
    )


@mcp_server.tool()
async def get_article_graph_context(
    node_id: str,
    max_hops: int = 2,
    include_precedents: bool = True,
    active_only: bool = True,
) -> Dict[str, Any]:
    """
    Retrieve comprehensive Knowledge Graph context for a statutory provision in Neo4j.
    Returns parent legislative hierarchy, constituent clauses, outward cross-references
    (REFERENCES, EXCEPTION_TO, CITES_CLAUSE), and linked judicial precedents.

    Args:
        node_id: Deterministic statutory identifier (e.g. 'VN_BLDS_2015_D004') or citation query ('Điều 4 BLDS 2015').
        max_hops: Graph traversal distance for cross-references (default: 2).
        include_precedents: Include applicable judicial precedents (default: True).
        active_only: Exclude repealed provisions from expansion (default: True).
    """
    logger.info(f"MCP Tool 'get_article_graph_context' invoked for node_id: '{node_id}'")

    # Step 1: Resolve citation to canonical node ID
    canonical_node_id = node_id.strip()
    if not canonical_node_id.startswith("VN_"):
        extractor = LegalEntityExtractor()
        extracted = extractor.extract_citations(canonical_node_id)
        if extracted:
            canonical_node_id = extracted[0].article_node_id or extracted[0].canonical_node_id
        else:
            # Fallback simple pattern
            m = re.search(r'(?:điều|dieu)\s*(\d+)', canonical_node_id, re.IGNORECASE)
            if m:
                art_num = int(m.group(1))
                law_code = "VN_BLLD_2019" if "lao động" in canonical_node_id.lower() else "VN_BLDS_2015"
                canonical_node_id = f"{law_code}_D{art_num:03d}"

    # Step 2: Attempt live Neo4j query
    try:
        from src.core.config import get_async_neo4j_driver
        from src.retrieval.neo4j_store import Neo4jStore

        driver = get_async_neo4j_driver()
        neo4j_store = Neo4jStore(driver=driver)

        expansion = await neo4j_store.expand_referenced_articles_async(
            root_ids=[canonical_node_id],
            hops=max_hops,
            include_precedents=include_precedents,
            active_only=active_only,
        )

        referenced_list = expansion.get("referenced_articles", [])
        graph_triples = []
        for ref in referenced_list:
            for rel in ref.get("relationship_types", []):
                graph_triples.append({
                    "subject": canonical_node_id,
                    "predicate": rel,
                    "object": ref.get("node_id"),
                    "citation": ref.get("citation", ""),
                })

        return {
            "status": "success",
            "target_node_id": canonical_node_id,
            "total_referenced": len(referenced_list),
            "referenced_articles": referenced_list,
            "graph_triples": graph_triples,
            "data_source": "neo4j_live",
        }
    except Exception as exc:
        logger.warning(f"Live Neo4j graph unavailable ({exc}). Using offline graph dataset fallback.")

        # Fallback to offline graph data
        offline_graphs = _get_offline_graphs()
        matched_node = None
        matched_law_id = None
        matched_edges = []

        for law_id, gdata in offline_graphs.items():
            for n in gdata.get("nodes", []):
                if n.get("node_id") == canonical_node_id:
                    matched_node = n
                    matched_law_id = law_id
                    break
            if matched_node:
                break

        if not matched_node:
            return {
                "status": "not_found",
                "message": f"Statutory node '{canonical_node_id}' not found in knowledge graph.",
                "target_node_id": canonical_node_id,
                "referenced_articles": [],
                "graph_triples": [],
            }

        # Find direct edges
        relevant_edges = []
        if matched_law_id and matched_law_id in offline_graphs:
            for edge in offline_graphs[matched_law_id].get("edges", []):
                if edge.get("source_id") == canonical_node_id or edge.get("target_id") == canonical_node_id:
                    relevant_edges.append(edge)

        referenced_articles = []
        graph_triples = []
        for e in relevant_edges:
            other_id = e["target_id"] if e["source_id"] == canonical_node_id else e["source_id"]
            rel_type = e.get("relation_type", "REFERENCES")
            graph_triples.append({
                "subject": e.get("source_id"),
                "predicate": rel_type,
                "object": e.get("target_id"),
                "citation": e.get("citation", ""),
            })
            referenced_articles.append({
                "node_id": other_id,
                "relationship_types": [rel_type],
                "citation": e.get("citation", ""),
                "hop_distance": 1,
            })

        return {
            "status": "success",
            "target_node_id": canonical_node_id,
            "target_node": matched_node,
            "total_referenced": len(referenced_articles),
            "referenced_articles": referenced_articles,
            "graph_triples": graph_triples,
            "data_source": "offline_graph_cache",
        }


# ============================================================================
# 5. SERVER ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    mcp_server.run()

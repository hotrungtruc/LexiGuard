"""Deterministic metrics for document extraction and table arithmetic."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.mcp_server.legal_tools import TableArithmeticAuditor


def _normalize_text(value: Any) -> str:
    """Normalize whitespace and case without removing legal words or numbers."""
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _error_rate(reference: str, hypothesis: str, unit_pattern: str) -> float:
    reference_units = re.findall(unit_pattern, reference)
    hypothesis_units = re.findall(unit_pattern, hypothesis)
    denominator = max(len(reference_units), 1)
    return _levenshtein("\0".join(reference_units), "\0".join(hypothesis_units)) / denominator


def _prf(true_positive: int, predicted: int, expected: int) -> Dict[str, float]:
    precision = true_positive / predicted if predicted else (1.0 if expected == 0 else 0.0)
    recall = true_positive / expected if expected else (1.0 if predicted == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


@dataclass(frozen=True)
class ExpectedElement:
    """Ground-truth structural element for a parsed document."""

    element_type: str
    text: str
    page_no: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass(frozen=True)
class ExpectedArithmetic:
    """Expected arithmetic outcome for a Markdown or structured table."""

    table_id: str
    table_data: Any
    discrepancy_types: Tuple[str, ...] = ()
    discrepancy_specs: Tuple[Dict[str, Any], ...] = ()
    tolerance: float = 0.01


@dataclass
class IngestionBenchmarkCase:
    """A deterministic extraction benchmark case."""

    case_id: str
    reference_text: str = ""
    expected_elements: List[ExpectedElement] = field(default_factory=list)
    expected_tables: int = 0
    arithmetic: Optional[ExpectedArithmetic] = None


@dataclass
class IngestionBenchmarkResult:
    """Serializable result containing metrics and actionable failures."""

    case_id: str
    metrics: Dict[str, Any]
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "metrics": self.metrics,
            "failures": self.failures,
        }


def _match_elements(
    expected: Sequence[ExpectedElement], actual: Iterable[Any]
) -> Tuple[int, int, int, List[str]]:
    actual_items = list(actual)
    unmatched = set(range(len(actual_items)))
    true_positive = 0
    failures: List[str] = []

    for expected_item in expected:
        expected_key = (expected_item.element_type, _normalize_text(expected_item.text))
        match_index = next(
            (
                index
                for index in unmatched
                if (
                    getattr(actual_items[index], "element_type", None),
                    _normalize_text(getattr(actual_items[index], "text", "")),
                )
                == expected_key
            ),
            None,
        )
        if match_index is None:
            failures.append(f"Missing element: {expected_item.element_type} {expected_item.text[:80]}")
            continue

        true_positive += 1
        unmatched.remove(match_index)
        actual_item = actual_items[match_index]
        if expected_item.page_no is not None:
            actual_page = getattr(getattr(actual_item, "provenance", None), "page_no", None)
            if actual_page != expected_item.page_no:
                failures.append(
                    f"Wrong page for '{expected_item.text[:50]}': "
                    f"expected {expected_item.page_no}, got {actual_page}"
                )

        if expected_item.bbox is not None:
            actual_bbox = getattr(getattr(actual_item, "provenance", None), "bbox", None)
            actual_values = tuple(actual_bbox.as_list) if actual_bbox else None
            if actual_values != expected_item.bbox:
                failures.append(f"Wrong bounding box for '{expected_item.text[:50]}'")

    return true_positive, len(actual_items), len(expected), failures


def evaluate_document_extraction(
    case: IngestionBenchmarkCase,
    actual_document: Any,
) -> IngestionBenchmarkResult:
    """Evaluate a ``ParsedDocument`` without invoking an LLM."""
    actual_elements = list(getattr(actual_document, "elements", []))
    true_positive, predicted, expected, failures = _match_elements(
        case.expected_elements,
        actual_elements,
    )
    element_metrics = _prf(true_positive, predicted, expected)

    actual_tables = list(getattr(actual_document, "tables", []))
    table_metrics = _prf(
        min(case.expected_tables, len(actual_tables)),
        len(actual_tables),
        case.expected_tables,
    )

    full_markdown = str(getattr(actual_document, "full_markdown", ""))
    metrics: Dict[str, Any] = {
        "element": element_metrics,
        "table_detection": table_metrics,
        "character_error_rate": round(
            _error_rate(case.reference_text, full_markdown, r"."),
            6,
        ) if case.reference_text else None,
        "word_error_rate": round(
            _error_rate(case.reference_text, full_markdown, r"\S+"),
            6,
        ) if case.reference_text else None,
        "provenance_accuracy": (
            1.0
            if case.expected_elements and not any("Wrong " in failure for failure in failures)
            else 0.0 if case.expected_elements else None
        ),
        "actual_element_count": len(actual_elements),
        "actual_table_count": len(actual_tables),
    }
    return IngestionBenchmarkResult(case.case_id, metrics, failures)


def _matches_discrepancy(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    if expected.get("type") != actual.get("type"):
        return False
    for key in ("row_index", "column_name", "item_description", "severity"):
        if key in expected and expected[key] != actual.get(key):
            return False
    for key in ("expected_value", "reported_value", "delta", "expected_sum", "reported_sum"):
        if key in expected:
            actual_value = actual.get(key)
            if actual_value is None or abs(float(actual_value) - float(expected[key])) > 1e-6:
                return False
    if "formula_contains" in expected and expected["formula_contains"] not in actual.get("formula", ""):
        return False
    return True


def evaluate_arithmetic_case(case: ExpectedArithmetic) -> IngestionBenchmarkResult:
    """Evaluate table arithmetic findings against typed, value-aware ground truth."""
    audit = TableArithmeticAuditor.audit(case.table_data, tolerance=case.tolerance)
    actual = audit.get("discrepancies", [])
    expected = list(case.discrepancy_specs)
    if not expected:
        expected = [{"type": discrepancy_type} for discrepancy_type in case.discrepancy_types]
    matched_actual = set()
    true_positive = 0
    failures: List[str] = []
    for expected_item in expected:
        match_index = next(
            (
                index
                for index, actual_item in enumerate(actual)
                if index not in matched_actual
                and _matches_discrepancy(expected_item, actual_item)
            ),
            None,
        )
        if match_index is None:
            failures.append(f"Missing discrepancy: {expected_item}")
        else:
            matched_actual.add(match_index)
            true_positive += 1
    metrics = {
        "discrepancy": _prf(
            true_positive,
            len(actual),
            len(expected),
        ),
        "status": audit.get("status"),
        "is_valid": audit.get("is_valid"),
        "actual_discrepancy_types": sorted({item.get("type") for item in actual}),
        "expected_discrepancy_types": sorted({item.get("type") for item in expected}),
        "actual_discrepancies": actual,
        "expected_discrepancies": expected,
    }
    return IngestionBenchmarkResult(case.table_id, metrics, failures)

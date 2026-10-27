"""Deterministic agent classification and citation guardrail metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


@dataclass(frozen=True)
class AgentBenchmarkCase:
    """Ground truth for one classification and citation-verification case."""

    case_id: str
    case_type: str
    expected_category: Optional[str]
    expected_risk: Optional[str]
    critical: bool = False
    supported_citations: Set[str] = field(default_factory=set)
    expected_abstention: bool = False


@dataclass
class AgentBenchmarkResult:
    case_id: str
    metrics: Dict[str, Any]
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "metrics": self.metrics,
            "failures": self.failures,
        }


def _f1(precision: float, recall: float) -> float:
    return round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0


def _classification_metrics(
    cases: Sequence[AgentBenchmarkCase],
    predictions: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    labels = sorted(
        {
            case.expected_category
            for case in cases
            if case.expected_category is not None
        }
        | {
            predictions.get(case.case_id, {}).get("category")
            for case in cases
            if predictions.get(case.case_id, {}).get("category")
        }
    )
    per_label: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(
            predictions.get(case.case_id, {}).get("category") == label
            and case.expected_category == label
            for case in cases
        )
        predicted = sum(predictions.get(case.case_id, {}).get("category") == label for case in cases)
        expected = sum(case.expected_category == label for case in cases)
        precision = tp / predicted if predicted else 0.0
        recall = tp / expected if expected else 0.0
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": _f1(precision, recall),
        }

    critical_cases = [case for case in cases if case.critical]
    critical_hits = sum(
        predictions.get(case.case_id, {}).get("category") == case.expected_category
        and predictions.get(case.case_id, {}).get("risk") == case.expected_risk
        for case in critical_cases
    )
    abstention_hits = sum(
        bool(predictions.get(case.case_id, {}).get("abstain")) == case.expected_abstention
        for case in cases
    )
    return {
        "macro_f1": round(
            sum(item["f1"] for item in per_label.values()) / len(per_label),
            6,
        ) if per_label else 0.0,
        "critical_recall": round(critical_hits / len(critical_cases), 6) if critical_cases else 1.0,
        "abstention_accuracy": round(abstention_hits / len(cases), 6) if cases else 1.0,
        "per_category": per_label,
    }


def evaluate_citation_verification(
    cases: Sequence[AgentBenchmarkCase],
    predictions: Mapping[str, Mapping[str, Any]],
) -> Dict[str, float]:
    """Score supported citations, unsupported citations, and hallucination rate."""
    expected_total = 0
    verified_hits = 0
    unsupported_total = 0
    hallucinated = 0
    for case in cases:
        expected = {citation.casefold() for citation in case.supported_citations}
        predicted = {
            str(citation).casefold()
            for citation in predictions.get(case.case_id, {}).get("verified_citations", [])
        }
        unsupported = predictions.get(case.case_id, {}).get("unsupported_citations", [])
        expected_total += len(expected)
        verified_hits += len(expected & predicted)
        unsupported_total += len(unsupported)
        hallucinated += sum(str(item).casefold() not in expected for item in unsupported)
    return {
        "citation_recall": round(verified_hits / expected_total, 6) if expected_total else 1.0,
        "unsupported_citation_count": float(unsupported_total),
        "hallucination_count": float(hallucinated),
        "hallucination_rate": round(
            hallucinated / unsupported_total, 6
        ) if unsupported_total else 0.0,
    }


def evaluate_agent_predictions(
    cases: Sequence[AgentBenchmarkCase],
    predictions: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Evaluate classification and citation verification as separate tasks."""
    classification = _classification_metrics(cases, predictions)
    citation = evaluate_citation_verification(cases, predictions)
    failures: List[str] = []
    for case in cases:
        prediction = predictions.get(case.case_id, {})
        if case.expected_abstention != bool(prediction.get("abstain")):
            failures.append(f"{case.case_id}: abstention mismatch")
        if (
            case.expected_category is not None
            and prediction.get("category") != case.expected_category
        ):
            failures.append(f"{case.case_id}: classification mismatch")
    return {"classification": classification, "citation_verification": citation, "failures": failures}

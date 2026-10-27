"""Deterministic retrieval benchmark metrics and invariant checks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RetrievalQuery:
    """A query with chunk and graph-path ground truth."""

    query_id: str
    query: str
    query_type: str
    relevant_chunks: Mapping[str, float] = field(default_factory=dict)
    relevant_graph_paths: Tuple[Tuple[str, ...], ...] = ()
    required_law_id: Optional[str] = None
    expected_dynamic_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalBenchmarkResult:
    query_id: str
    metrics: Dict[str, float]
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"query_id": self.query_id, "metrics": self.metrics, "failures": self.failures}


def _item_id(item: Any) -> str:
    if isinstance(item, str):
        return item
    return str(item.get("chunk_id") or item.get("node_id") or item.get("id") or "")


def _dcg(relevances: Sequence[float]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevances))


def _ranked_relevances(ranked: Sequence[Any], relevant: Mapping[str, float], k: int) -> List[float]:
    return [float(relevant.get(_item_id(item), 0.0)) for item in list(ranked)[:k]]


def evaluate_ranked_results(
    query: RetrievalQuery,
    ranked_results: Sequence[Any],
    k: int = 10,
) -> RetrievalBenchmarkResult:
    """Evaluate Recall@k, MRR, and graded nDCG@k for one query."""
    ranked_ids = [_item_id(item) for item in ranked_results]
    relevant_ids = set(query.relevant_chunks)
    top_ids = ranked_ids[:k]
    hits = [item_id for item_id in top_ids if item_id in relevant_ids]
    first_rank = next(
        (index + 1 for index, item_id in enumerate(ranked_ids) if item_id in relevant_ids),
        None,
    )
    ideal = sorted(query.relevant_chunks.values(), reverse=True)[:k]
    actual_dcg = _dcg(_ranked_relevances(ranked_results, query.relevant_chunks, k))
    ideal_dcg = _dcg(ideal)
    metrics = {
        "recall_at_k": round(len(set(hits)) / len(relevant_ids), 6) if relevant_ids else 1.0,
        "mrr": round(1.0 / first_rank, 6) if first_rank else 0.0,
        "ndcg_at_k": round(actual_dcg / ideal_dcg, 6) if ideal_dcg else 1.0,
        "retrieved_count": float(len(ranked_ids)),
        "relevant_count": float(len(relevant_ids)),
    }
    failures = []
    if query.required_law_id:
        wrong_laws = [
            _item_id(item)
            for item in ranked_results[:k]
            if isinstance(item, dict)
            and item.get("law_id") not in (None, query.required_law_id)
        ]
        if wrong_laws:
            failures.append(f"Active-law/law filter mismatch: {wrong_laws}")
    for key, expected_value in query.expected_dynamic_metadata.items():
        if any(
            isinstance(item, dict) and item.get(key) == expected_value
            for item in ranked_results[:k]
        ):
            continue
        failures.append(f"Missing dynamic metadata propagation: {key}={expected_value!r}")
    return RetrievalBenchmarkResult(query.query_id, metrics, failures)


def evaluate_graph_paths(
    query: RetrievalQuery,
    graph_paths: Iterable[Sequence[str]],
) -> Dict[str, float]:
    """Measure graph-path recall against annotated node sequences."""
    expected = {tuple(path) for path in query.relevant_graph_paths}
    actual = {tuple(path) for path in graph_paths}
    matched = len(expected & actual)
    return {
        "graph_path_recall": round(matched / len(expected), 6) if expected else 1.0,
        "graph_path_count": float(len(actual)),
        "expected_graph_path_count": float(len(expected)),
    }


def aggregate_results(results: Iterable[RetrievalBenchmarkResult]) -> Dict[str, float]:
    rows = list(results)
    if not rows:
        return {"query_count": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}
    return {
        "query_count": float(len(rows)),
        "recall_at_k": round(sum(row.metrics["recall_at_k"] for row in rows) / len(rows), 6),
        "mrr": round(sum(row.metrics["mrr"] for row in rows) / len(rows), 6),
        "ndcg_at_k": round(sum(row.metrics["ndcg_at_k"] for row in rows) / len(rows), 6),
    }

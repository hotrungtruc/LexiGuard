"""Run Phase 3 retrieval metrics over saved ablation predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.retrieval_benchmark import (
    RetrievalBenchmarkResult,
    RetrievalQuery,
    aggregate_results,
    evaluate_graph_paths,
    evaluate_ranked_results,
)


def _load_queries(path: Path) -> List[RetrievalQuery]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return [
        RetrievalQuery(
            query_id=item["query_id"],
            query=item["query"],
            query_type=item["query_type"],
            relevant_chunks=item.get("relevant_chunks", {}),
            relevant_graph_paths=tuple(
                tuple(path) for path in item.get("relevant_graph_paths", [])
            ),
            required_law_id=item.get("required_law_id"),
            expected_dynamic_metadata=item.get("expected_dynamic_metadata", {}),
        )
        for item in payload.get("queries", [])
    ]


def _evaluate_method(
    queries: List[RetrievalQuery],
    predictions: Dict[str, Any],
    k: int,
) -> Dict[str, Any]:
    results: List[RetrievalBenchmarkResult] = []
    graph_metrics = []
    for query in queries:
        prediction = predictions.get(query.query_id, {})
        ranked = prediction if isinstance(prediction, list) else prediction.get("results", [])
        result = evaluate_ranked_results(query, ranked, k=k)
        if not isinstance(prediction, list):
            graph_result = evaluate_graph_paths(query, prediction.get("graph_paths", []))
            result.metrics.update(graph_result)
            graph_metrics.append(graph_result["graph_path_recall"])
        results.append(result)
    aggregate = aggregate_results(results)
    if graph_metrics:
        aggregate["graph_path_recall"] = round(sum(graph_metrics) / len(graph_metrics), 6)
    return {
        "aggregate": aggregate,
        "queries": [result.to_dict() for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval_dataset/retrieval_benchmark.json"),
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        help="Directory containing dense.json, sparse.json, rrf.json, graph.json, reranker.json.",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/retrieval_benchmark"))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    if not args.predictions_dir:
        parser.error("--predictions-dir is required; the benchmark does not fabricate retrieval results")
    queries = _load_queries(args.dataset)
    output: Dict[str, Any] = {}
    for prediction_file in sorted(args.predictions_dir.glob("*.json")):
        predictions = json.loads(prediction_file.read_text(encoding="utf-8-sig"))
        output[prediction_file.stem] = _evaluate_method(queries, predictions, args.k)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for method, result in output.items():
        aggregate = result["aggregate"]
        print(
            f"{method}: Recall@{args.k}={aggregate['recall_at_k']:.4f} "
            f"MRR={aggregate['mrr']:.4f} nDCG@{args.k}={aggregate['ndcg_at_k']:.4f}"
        )


if __name__ == "__main__":
    main()

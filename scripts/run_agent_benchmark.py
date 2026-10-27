"""Run Phase 4 agent classification and citation guardrail benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.agent_benchmark import AgentBenchmarkCase, evaluate_agent_predictions


def _load_cases(path: Path) -> List[AgentBenchmarkCase]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return [
        AgentBenchmarkCase(
            case_id=item["case_id"],
            case_type=item["case_type"],
            expected_category=item.get("expected_category"),
            expected_risk=item.get("expected_risk"),
            critical=bool(item.get("critical", False)),
            supported_citations=set(item.get("supported_citations", [])),
            expected_abstention=bool(item.get("expected_abstention", False)),
        )
        for item in payload.get("cases", [])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval_dataset/agent_guardrail_benchmark.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="JSON mapping case_id to classification and citation-verification outputs.",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/agent_benchmark"))
    args = parser.parse_args()

    cases = _load_cases(args.dataset)
    predictions: Dict[str, Any] = json.loads(
        args.predictions.read_text(encoding="utf-8-sig")
    )
    result = evaluate_agent_predictions(cases, predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    classification = result["classification"]
    citation = result["citation_verification"]
    print(
        f"macro-F1={classification['macro_f1']:.4f} "
        f"critical-recall={classification['critical_recall']:.4f} "
        f"abstention-accuracy={classification['abstention_accuracy']:.4f} "
        f"hallucination-rate={citation['hallucination_rate']:.4f}"
    )
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

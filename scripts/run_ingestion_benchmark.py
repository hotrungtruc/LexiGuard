"""Run the deterministic LexiGuard ingestion and table arithmetic benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.ingestion_benchmark import (
    ExpectedArithmetic,
    ExpectedElement,
    IngestionBenchmarkCase,
    evaluate_arithmetic_case,
    evaluate_document_extraction,
)


def _load_cases(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("cases", [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval_dataset/ingestion_table_benchmark.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/ingestion_benchmark"))
    parser.add_argument("--min-recall", type=float, default=0.98)
    parser.add_argument(
        "--document",
        type=Path,
        help="Optional PDF/image path for an explicit Docling integration case.",
    )
    parser.add_argument(
        "--document-manifest",
        type=Path,
        help="JSON manifest containing expected_elements and expected_tables for --document.",
    )
    args = parser.parse_args()

    results = []
    for case in _load_cases(args.dataset):
        result = evaluate_arithmetic_case(
            ExpectedArithmetic(
                table_id=case["case_id"],
                table_data=case["table_data"],
                discrepancy_types=tuple(case.get("expected_discrepancy_types", [])),
                discrepancy_specs=tuple(case.get("expected_discrepancies", [])),
                tolerance=float(case.get("tolerance", 0.01)),
            )
        )
        results.append(result.to_dict())

    if args.document:
        if not args.document_manifest:
            parser.error("--document-manifest is required when --document is provided")
        from src.ingestion.docling_parser import DoclingParser

        with args.document_manifest.open("r", encoding="utf-8") as handle:
            document_case = json.load(handle)
        expected_elements = [
            ExpectedElement(
                element_type=item["element_type"],
                text=item["text"],
                page_no=item.get("page_no"),
                bbox=tuple(item["bbox"]) if item.get("bbox") else None,
            )
            for item in document_case.get("expected_elements", [])
        ]
        parsed_document = DoclingParser().parse(args.document)
        document_result = evaluate_document_extraction(
            IngestionBenchmarkCase(
                case_id=document_case.get("case_id", args.document.stem),
                reference_text=document_case.get("reference_text", ""),
                expected_elements=expected_elements,
                expected_tables=int(document_case.get("expected_tables", 0)),
            ),
            parsed_document,
        )
        results.append(document_result.to_dict())

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "f1", "precision", "recall", "failures"])
        writer.writeheader()
        for result in results:
            discrepancy = result["metrics"].get("discrepancy", {})
            writer.writerow({
                "case_id": result["case_id"],
                "f1": discrepancy.get("f1", ""),
                "precision": discrepancy.get("precision", ""),
                "recall": discrepancy.get("recall", ""),
                "failures": " | ".join(result["failures"]),
            })

    arithmetic_results = [
        result for result in results if "discrepancy" in result["metrics"]
    ]
    recalls = [result["metrics"]["discrepancy"]["recall"] for result in arithmetic_results]
    aggregate_recall = sum(recalls) / len(recalls) if recalls else 0.0
    summary = {
        "dataset": str(args.dataset),
        "case_count": len(results),
        "arithmetic_recall": round(aggregate_recall, 6),
        "minimum_recall": args.min_recall,
        "passed": aggregate_recall >= args.min_recall,
        "failed_cases": [
            result["case_id"]
            for result in arithmetic_results
            if result["metrics"]["discrepancy"]["recall"] < args.min_recall
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Arithmetic recall: {summary['arithmetic_recall']:.4f} "
        f"(threshold {args.min_recall:.4f})"
    )
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

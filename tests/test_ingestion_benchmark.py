"""Tests for deterministic ingestion and table arithmetic benchmark metrics."""

from types import SimpleNamespace

from src.evaluation.ingestion_benchmark import (
    ExpectedArithmetic,
    ExpectedElement,
    IngestionBenchmarkCase,
    evaluate_arithmetic_case,
    evaluate_document_extraction,
)


def test_document_extraction_metrics_match_element_and_provenance():
    document = SimpleNamespace(
        full_markdown="Article 1. Payment",
        elements=[
            SimpleNamespace(
                element_type="heading",
                text="Article 1. Payment",
                provenance=SimpleNamespace(page_no=1, bbox=None),
            )
        ],
        tables=[],
    )
    result = evaluate_document_extraction(
        IngestionBenchmarkCase(
            case_id="DOC-001",
            reference_text="Article 1. Payment",
            expected_elements=[ExpectedElement("heading", "Article 1. Payment", page_no=1)],
        ),
        document,
    )

    assert result.metrics["element"]["f1"] == 1.0
    assert result.metrics["provenance_accuracy"] == 1.0
    assert result.failures == []


def test_arithmetic_benchmark_detects_row_product_mismatch():
    result = evaluate_arithmetic_case(
        ExpectedArithmetic(
            table_id="TABLE-001",
            table_data=(
                "| Item | Quantity | Unit Price | Total |\n"
                "|---|---:|---:|---:|\n"
                "| Service | 2 | 10.000.000 | 25.000.000 |"
            ),
            discrepancy_types=("ROW_PRODUCT_MISMATCH",),
        )
    )

    assert result.metrics["discrepancy"]["recall"] == 1.0
    assert result.metrics["actual_discrepancy_types"] == ["ROW_PRODUCT_MISMATCH"]
    assert result.failures == []


def test_arithmetic_benchmark_reports_missing_expected_discrepancy():
    result = evaluate_arithmetic_case(
        ExpectedArithmetic(
            table_id="TABLE-VALID-001",
            table_data=(
                "| Item | Quantity | Unit Price | Total |\n"
                "|---|---:|---:|---:|\n"
                "| Service | 2 | 10.000.000 | 20.000.000 |"
            ),
            discrepancy_types=("COLUMN_SUM_MISMATCH",),
        )
    )

    assert result.metrics["discrepancy"]["recall"] == 0.0
    assert result.failures == ["Missing discrepancy: {'type': 'COLUMN_SUM_MISMATCH'}"]


def test_arithmetic_benchmark_matches_row_and_numeric_values():
    result = evaluate_arithmetic_case(
        ExpectedArithmetic(
            table_id="TABLE-DETAIL-001",
            table_data=(
                "| Item | Quantity | Unit Price | Total |\n"
                "|---|---:|---:|---:|\n"
                "| Service | 2 | 10.000.000 | 25.000.000 |"
            ),
            discrepancy_specs=(
                {
                    "type": "ROW_PRODUCT_MISMATCH",
                    "row_index": 1,
                    "expected_value": 20_000_000,
                    "reported_value": 25_000_000,
                    "formula_contains": "Quantity (2.0)",
                },
            ),
        )
    )

    assert result.metrics["discrepancy"]["f1"] == 1.0
    assert result.failures == []

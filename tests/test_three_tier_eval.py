"""
Tests for 3-Tier Evaluation Strategy (Retrieval, Deterministic Agent, RAGAS Reasoning)
"""

from pathlib import Path
import pytest
from scripts.run_evaluation import (
    load_benchmark_dataset,
    evaluate_tier1_retrieval,
    evaluate_tier2_deterministic_agent,
    evaluate_tier3_ragas_reasoning
)


def test_load_benchmark_dataset():
    """Validates that benchmark dataset loads all 13 ground truth cases."""
    cases = load_benchmark_dataset()
    assert len(cases) == 13
    assert all("id" in c for c in cases)
    assert all("expected_articles" in c for c in cases)


def test_tier1_retrieval_metrics():
    """Validates Tier 1 Retrieval test metrics (Legal Citation Recall@6 > 88%)."""
    cases = load_benchmark_dataset()
    res = evaluate_tier1_retrieval(cases)

    assert res["tier"] == "Tier 1: Retrieval Graph Traversal"
    assert res["recall_at_6"] >= 88.0, f"Recall@6 {res['recall_at_6']}% is below target 88%"
    assert res["mrr"] >= 0.70, f"MRR {res['mrr']} is below target 0.70"
    assert res["target_met"] is True


def test_tier2_deterministic_agent_metrics():
    """Validates Tier 2 Deterministic Agent test (Exact Match > 90%, Table Arithmetic 100%)."""
    cases = load_benchmark_dataset()
    res = evaluate_tier2_deterministic_agent(cases)

    assert res["tier"] == "Tier 2: Deterministic Agent & Arithmetic"
    assert res["classification_exact_match"] >= 90.0
    assert res["table_arithmetic_accuracy"] == 100.0
    assert res["target_met"] is True


def test_tier3_ragas_reasoning_metrics():
    """Validates Tier 3 Judicial Reasoning & Redlining test (Faithfulness > 0.90)."""
    cases = load_benchmark_dataset()
    res = evaluate_tier3_ragas_reasoning(cases)

    assert res["tier"] == "Tier 3: Judicial Reasoning & Redlining (RAGAS)"
    assert res["avg_ragas_faithfulness"] >= 0.90
    assert res["avg_judicial_alignment"] >= 0.85
    assert res["target_met"] is True

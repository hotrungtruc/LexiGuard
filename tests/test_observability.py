from src.core.observability import (
    Observation,
    estimate_cost,
    summarize_observations,
    trace_metadata,
)


def test_latency_percentiles_and_throughput_summary():
    observations = [
        Observation("cold", "cold", 100, input_tokens=10, output_tokens=5),
        Observation("warm", "warm", 200, input_tokens=20, output_tokens=10, retries=1),
        Observation("error", "warm", 300, error="Timeout", cost_usd=0.01),
    ]
    summary = summarize_observations(observations)
    assert summary["latency_ms"]["p50"] == 200
    assert summary["latency_ms"]["p95"] == 290
    assert summary["latency_ms"]["p99"] == 298
    assert summary["retries"] == 1
    assert summary["errors"] == 1
    assert summary["throughput_cases_per_second"] > 0


def test_trace_metadata_and_cost_are_stable():
    metadata = trace_metadata("CASE-1", session_id="SESSION-1", category_id="NEW_LAW_RULE")
    assert metadata["case_id"] == "CASE-1"
    assert metadata["session_id"] == "SESSION-1"
    assert metadata["category_id"] == "NEW_LAW_RULE"
    assert estimate_cost(1_000_000, 500_000, 1.0, 2.0) == 2.0

"""Optional Langfuse tracing and deterministic latency/usage aggregation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from loguru import logger


@dataclass
class Observation:
    """One measured workflow or benchmark case execution."""

    case_id: str
    cache_mode: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    error: Optional[str] = None
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def percentile(values: Iterable[float], p: float) -> float:
    """Return an interpolated percentile without requiring NumPy."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return round(ordered[0], 6)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 6)


def summarize_observations(observations: Iterable[Observation]) -> Dict[str, Any]:
    rows = list(observations)
    latencies = [row.latency_ms for row in rows]
    duration_s = sum(latencies) / 1000
    return {
        "case_count": len(rows),
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "mean": round(mean(latencies), 6) if latencies else 0.0,
        },
        "input_tokens": sum(row.input_tokens for row in rows),
        "output_tokens": sum(row.output_tokens for row in rows),
        "retries": sum(row.retries for row in rows),
        "errors": sum(row.error is not None for row in rows),
        "cost_usd": round(sum(row.cost_usd for row in rows), 8),
        "throughput_cases_per_second": round(len(rows) / duration_s, 6) if duration_s else 0.0,
    }


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float = 0.0,
    output_price_per_million: float = 0.0,
) -> float:
    """Estimate cost from configurable provider prices."""
    return (
        input_tokens * input_price_per_million / 1_000_000
        + output_tokens * output_price_per_million / 1_000_000
    )


def trace_metadata(
    case_id: str,
    *,
    session_id: Optional[str] = None,
    cache_mode: str = "unknown",
    **metadata: Any,
) -> Dict[str, Any]:
    """Build the stable metadata contract shared by traces and reports."""
    result = {
        "case_id": case_id,
        "session_id": session_id or case_id,
        "cache_mode": cache_mode,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "model_fast": os.getenv("FAST_LLM_MODEL", "qwen2.5:3b-instruct-q4_K_M"),
        "model_reasoning": os.getenv("REASONING_LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M"),
    }
    result.update(metadata)
    return result


def get_langfuse_client() -> Any:
    """Return an optional Langfuse client, never failing a legal audit."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        )
    except (ImportError, TypeError, OSError) as exc:
        logger.warning("Langfuse tracing disabled: {}", exc)
        return None


def observe(
    operation: Callable[[], Any],
    *,
    case_id: str,
    cache_mode: str = "unknown",
    session_id: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    retries: int = 0,
    cost_usd: float = 0.0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[Any, Observation]:
    """Measure a synchronous operation and optionally emit one Langfuse trace."""
    tags = trace_metadata(
        case_id, session_id=session_id, cache_mode=cache_mode, **dict(metadata or {})
    )
    client = get_langfuse_client()
    trace = (
        client.trace(name="lexiguard.workflow", session_id=tags["session_id"], metadata=tags)
        if client
        else None
    )
    started = time.perf_counter()
    error: Optional[str] = None
    try:
        value = operation()
    except Exception as exc:
        error = type(exc).__name__
        if trace:
            trace.update(output={"error": error})
        raise
    finally:
        latency_ms = (time.perf_counter() - started) * 1000
        if trace:
            trace.update(
                output={
                    "latency_ms": round(latency_ms, 3),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "retries": retries,
                    "error": error,
                }
            )
            # Langfuse SDK flushes automatically in background worker batches.
            # Avoid blocking synchronous HTTP flush on the critical path.
    return value, Observation(
        case_id=case_id,
        cache_mode=cache_mode,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        retries=retries,
        error=error,
        cost_usd=cost_usd,
        metadata=tags,
    )

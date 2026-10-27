"""Run latency and usage aggregation over recorded LexiGuard observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.core.observability import Observation, summarize_observations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="JSON list with case_id, cache_mode, latency_ms, tokens, retries, and errors.",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/observability_benchmark"))
    args = parser.parse_args()
    payload = json.loads(args.observations.read_text(encoding="utf-8-sig"))
    observations = [Observation(**item) for item in payload]
    summary = summarize_observations(observations)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

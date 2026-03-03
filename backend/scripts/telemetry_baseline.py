#!/usr/bin/env python3
"""Capture telemetry baselines and guardrail compliance for optimizer pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

if TYPE_CHECKING:
    from services.optimizer.guardrails import GuardrailResult

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))


@dataclass(frozen=True)
class BaselineSummary:
    window_days: int
    cutoff_iso: str
    run_count: int
    avg_latency_ms: float
    max_latency_ms: float
    avg_tokens_saved: float
    avg_similarity: Optional[float]
    min_similarity: Optional[float]
    similarity_samples: int
    guard_results: List[GuardrailResult]


def _cutoff_timestamp(window_days: int) -> str:
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    return cutoff.isoformat()


def _collect_duration_by_run(cutoff_iso: str) -> Dict[str, float]:
    from database import get_db

    durations: Dict[str, float] = defaultdict(float)
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT optimization_id, duration_ms
            FROM performance_telemetry
            WHERE created_at >= ?
            """,
            (cutoff_iso,),
        ).fetchall()
        for row in rows:
            if not row["optimization_id"]:
                continue
            durations[row["optimization_id"]] += row["duration_ms"] or 0.0
    return durations


def _collect_history_rows(cutoff_iso: str) -> List[Dict[str, Optional[float]]]:
    from database import get_db

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT raw_tokens, optimized_tokens, semantic_similarity
            FROM optimization_history
            WHERE created_at >= ?
            """,
            (cutoff_iso,),
        ).fetchall()
    return [dict(row) for row in rows]


def _safe_average(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def compute_baseline(window_days: int) -> BaselineSummary:
    from services.optimizer.guardrails import (evaluate_latency_guard,
                                               evaluate_similarity_guard,
                                               evaluate_token_savings_guard)

    cutoff_iso = _cutoff_timestamp(window_days)
    durations = _collect_duration_by_run(cutoff_iso)
    history_rows = _collect_history_rows(cutoff_iso)

    latency_values = list(durations.values())
    avg_latency = _safe_average(latency_values)
    max_latency = max(latency_values) if latency_values else 0.0

    tokens_saved: List[float] = []
    similarities: List[float] = []

    for record in history_rows:
        raw_tokens = float(record.get("raw_tokens") or 0.0)
        optimized_tokens = float(record.get("optimized_tokens") or 0.0)
        tokens_saved.append(raw_tokens - optimized_tokens)
        similarity_value = record.get("semantic_similarity")
        if similarity_value is not None:
            similarities.append(float(similarity_value))

    avg_tokens_saved = _safe_average(tokens_saved)
    avg_similarity = _safe_average(similarities) if similarities else None
    min_similarity = min(similarities) if similarities else None

    guard_results = [
        evaluate_latency_guard(max_latency),
        evaluate_similarity_guard(min_similarity),
        evaluate_token_savings_guard(avg_tokens_saved),
    ]

    return BaselineSummary(
        window_days=window_days,
        cutoff_iso=cutoff_iso,
        run_count=len(durations),
        avg_latency_ms=avg_latency,
        max_latency_ms=max_latency,
        avg_tokens_saved=avg_tokens_saved,
        avg_similarity=avg_similarity,
        min_similarity=min_similarity,
        similarity_samples=len(similarities),
        guard_results=guard_results,
    )


def _format_guard_result(result: GuardrailResult) -> str:
    status = "PASS" if result.passed else "FAIL"
    return (
        f"{status}: {result.name} (value={result.value:.3f}, "
        f"threshold={result.threshold:.3f}) — {result.detail}"
    )


def display_summary(summary: BaselineSummary, *, json_output: bool = False) -> None:
    if json_output:
        payload = {
            "window_days": summary.window_days,
            "cutoff_iso": summary.cutoff_iso,
            "run_count": summary.run_count,
            "avg_latency_ms": summary.avg_latency_ms,
            "max_latency_ms": summary.max_latency_ms,
            "avg_tokens_saved": summary.avg_tokens_saved,
            "avg_similarity": summary.avg_similarity,
            "min_similarity": summary.min_similarity,
            "similarity_samples": summary.similarity_samples,
            "guard_results": [
                {
                    "name": result.name,
                    "passed": result.passed,
                    "threshold": result.threshold,
                    "value": result.value,
                    "detail": result.detail,
                }
                for result in summary.guard_results
            ],
        }
        print(json.dumps(payload, indent=2))
        return

    print(f"Telemetry baseline (window: last {summary.window_days} days)")
    print(f"Cutoff timestamp (UTC): {summary.cutoff_iso}")
    print(f"Optimization runs analyzed (telemetry table): {summary.run_count}")
    print(
        f"Average latency: {summary.avg_latency_ms:.2f} ms | "
        f"Max latency: {summary.max_latency_ms:.2f} ms"
    )
    print(f"Average tokens saved per run: {summary.avg_tokens_saved:.2f}")

    if summary.avg_similarity is not None:
        print(
            f"Semantic similarity (avg/min): "
            f"{summary.avg_similarity:.3f} / {summary.min_similarity:.3f} "
            f"from {summary.similarity_samples} samples"
        )
    else:
        print("Semantic similarity not recorded in the selected window.")

    print("\nGuardrail evaluation:")
    for result in summary.guard_results:
        print(f"  {_format_guard_result(result)}")


def parse_args() -> argparse.Namespace:
    from services.optimizer import config

    parser = argparse.ArgumentParser(
        description="Summarize telemetry baselines and guardrail compliance."
    )
    parser.add_argument(
        "--window-days",
        "-w",
        type=int,
        default=config.TELEMETRY_BASELINE_WINDOW_DAYS,
        help="Number of trailing days to consider when summarizing telemetry.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the summary in machine-readable JSON.",
    )
    return parser.parse_args()


def main() -> None:
    from database import init_db

    args = parse_args()
    init_db()
    summary = compute_baseline(args.window_days)
    display_summary(summary, json_output=args.json)


if __name__ == "__main__":
    main()

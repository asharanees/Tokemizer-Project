"""Performance telemetry integration for the optimizer pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from database import TelemetryRecord

logger = logging.getLogger(__name__)

_PRIORS_CACHE_TTL_SECONDS = 60.0
_PRIORS_CACHE_LOCK = Lock()
_PRIORS_CACHE: Dict[Tuple[str, str, str], Tuple[float, Dict[str, "PassUtilityPrior"]]] = {}


@dataclass(frozen=True)
class PassUtilityPrior:
    pass_name: str
    expected_utility: float
    expected_duration_ms: float
    sample_count: int


def token_bin_for_count(tokens: int) -> str:
    if tokens < 512:
        return "xs"
    if tokens < 2048:
        return "sm"
    if tokens < 8192:
        return "md"
    if tokens < 16384:
        return "lg"
    return "xl"


def get_pass_utility_priors(
    *,
    content_profile: str,
    optimization_mode: str,
    token_bin: str,
) -> Dict[str, PassUtilityPrior]:
    cache_key = (content_profile, optimization_mode, token_bin)
    now = monotonic()
    with _PRIORS_CACHE_LOCK:
        cached = _PRIORS_CACHE.get(cache_key)
        if cached and cached[0] > now:
            return dict(cached[1])

    priors: Dict[str, PassUtilityPrior] = {}
    try:
        from database import get_db, init_db

        init_db()
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT
                    pass_name,
                    AVG(COALESCE(actual_utility, CASE
                        WHEN duration_ms > 0 THEN CAST(tokens_saved AS REAL) / duration_ms
                        ELSE 0.0
                    END)) AS expected_utility,
                    AVG(duration_ms) AS expected_duration_ms,
                    COUNT(*) AS sample_count
                FROM performance_telemetry
                WHERE content_profile = ?
                    AND optimization_mode = ?
                    AND token_bin = ?
                    AND pass_skipped_reason IS NULL
                GROUP BY pass_name
                """,
                (content_profile, optimization_mode, token_bin),
            ).fetchall()

        for row in rows:
            name = str(row["pass_name"])
            priors[name] = PassUtilityPrior(
                pass_name=name,
                expected_utility=float(row["expected_utility"] or 0.0),
                expected_duration_ms=max(0.0, float(row["expected_duration_ms"] or 0.0)),
                sample_count=int(row["sample_count"] or 0),
            )
    except Exception as exc:
        logger.debug("Failed to load utility priors: %s", exc)

    # Cold-start priors: when telemetry is empty, enforce a deterministic ordering
    # for the heavy passes (otherwise sorting falls back to alphabetical).
    default_expected_utility = {
        "prune_low_entropy": 0.7,
        "compress_examples": 0.5,
        "summarize_history": 0.3,
    }
    for pass_name, expected_utility in default_expected_utility.items():
        priors.setdefault(
            pass_name,
            PassUtilityPrior(
                pass_name=pass_name,
                expected_utility=float(expected_utility),
                expected_duration_ms=0.0,
                sample_count=0,
            ),
        )

    with _PRIORS_CACHE_LOCK:
        _PRIORS_CACHE[cache_key] = (now + _PRIORS_CACHE_TTL_SECONDS, dict(priors))
    return priors


class OptimizationTelemetryCollector:
    """
    Collects per-pass performance metrics during optimization.
    Designed to have minimal performance impact on the optimization pipeline.
    """

    __slots__ = ("_optimization_id", "_passes", "_pass_order", "_enabled", "_metadata")

    def __init__(self, optimization_id: str, enabled: bool = True):
        self._optimization_id = optimization_id
        self._passes: List[Tuple[str, int, float, int, int, Dict[str, Any]]] = []
        self._pass_order = 0
        self._enabled = enabled
        self._metadata: Dict[str, Any] = {}

    def record_pass(
        self,
        pass_name: str,
        duration_ms: float,
        tokens_before: int,
        tokens_after: int,
        *,
        estimated_tokens_after: Optional[int] = None,
        exact_tokens_after: Optional[int] = None,
        expected_utility: Optional[float] = None,
        actual_utility: Optional[float] = None,
        pass_skipped_reason: Optional[str] = None,
        content_profile: Optional[str] = None,
        optimization_mode: Optional[str] = None,
        token_bin: Optional[str] = None,
    ) -> None:
        """Record metrics for a single optimization pass."""
        if not self._enabled:
            return

        self._pass_order += 1
        metadata = {
            "estimated_tokens_after": estimated_tokens_after,
            "exact_tokens_after": exact_tokens_after,
            "expected_utility": expected_utility,
            "actual_utility": actual_utility,
            "pass_skipped_reason": pass_skipped_reason,
            "content_profile": content_profile,
            "optimization_mode": optimization_mode,
            "token_bin": token_bin,
        }
        self._passes.append(
            (pass_name, self._pass_order, duration_ms, tokens_before, tokens_after, metadata)
        )

    def record_flag(self, name: str, value: Any) -> None:
        """Record a lightweight metadata flag for downstream analysis."""
        if not self._enabled:
            return
        self._metadata[name] = value

    def get_telemetry_record(self) -> Optional["TelemetryRecord"]:
        """
        Convert collected metrics into a TelemetryRecord for async persistence.
        Returns None if telemetry is disabled or no passes were recorded.
        """
        if not self._enabled or not self._passes:
            return None

        try:
            from database import TelemetryPassMetric, TelemetryRecord

            pass_metrics: List[TelemetryPassMetric] = []
            for entry in self._passes:
                (
                    pass_name,
                    pass_order,
                    duration_ms,
                    tokens_before,
                    tokens_after,
                    metadata,
                ) = entry
                tokens_saved = max(0, tokens_before - tokens_after)
                reduction_percent = (
                    (tokens_saved / tokens_before * 100.0) if tokens_before > 0 else 0.0
                )

                pass_metrics.append(
                    TelemetryPassMetric(
                        pass_name=pass_name,
                        pass_order=pass_order,
                        duration_ms=duration_ms,
                        tokens_before=tokens_before,
                        tokens_after=tokens_after,
                        tokens_saved=tokens_saved,
                        reduction_percent=reduction_percent,
                        expected_utility=metadata.get("expected_utility"),
                        actual_utility=metadata.get("actual_utility"),
                        pass_skipped_reason=metadata.get("pass_skipped_reason"),
                        content_profile=metadata.get("content_profile"),
                        optimization_mode=metadata.get("optimization_mode"),
                        token_bin=metadata.get("token_bin"),
                    )
                )

            return TelemetryRecord(
                optimization_id=self._optimization_id,
                passes=pass_metrics,
                metadata=dict(self._metadata),
            )

        except Exception as e:
            logger.warning(f"Failed to create telemetry record: {e}")
            return None

    def is_enabled(self) -> bool:
        """Check if telemetry collection is enabled."""
        return self._enabled


def submit_optimization_telemetry(collector: OptimizationTelemetryCollector) -> None:
    """
    Submit collected telemetry to the async batch writer.
    This is a fire-and-forget operation that won't block the response.
    """
    if not collector.is_enabled():
        return

    try:
        record = collector.get_telemetry_record()
        if record is None:
            return

        from database import submit_telemetry

        submit_telemetry(record)
        logger.debug(
            f"Submitted telemetry for optimization {collector._optimization_id}"
        )

    except Exception as e:
        # Never let telemetry submission failures affect the optimization response
        logger.warning(f"Failed to submit telemetry: {e}")
